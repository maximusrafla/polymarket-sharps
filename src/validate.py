"""Out-of-sample validation — CLAUDE.md "Validation": fit/select on each
wallet's first-half bet history, then check whether edge persists in the
held-out second half. Non-negotiable per CLAUDE.md; rank.py must not run on
unvalidated data. See DECISIONS.md for the exact split rule and thresholds.

Every wallet still gets a row here (nothing is dropped, per CLAUDE.md's
"never remove a wallet from the dataset or ranking" rule) — wallets with too
little history simply get NaN out-of-sample edge and `edge_persisted=False`,
which rank.py down-weights via the sample-size term rather than excluding.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy import stats

from src.common import (
    WALLET_FEATURES_PATH,
    WALLET_VALIDATED_PATH,
    ensure_dirs,
    load_config,
    load_ledger,
)
from src.features import (
    FEATURE_COLUMNS,
    compute_wallet_features,
    fit_price_baseline,
    only_buys,
    residual_edge_per_bet,
)

# Ledger columns compute_oos_validation actually reads (a subset of
# FEATURE_COLUMNS). In the nightly pipeline features are already cached, so
# validate loads only these — keeping it off the ~1.5 GB of unused string columns
# (market_id/token_id/tx_hash/outcome/question/slug) at 4.7M rows.
VALIDATE_LEDGER_COLUMNS = [
    "wallet", "side", "resolved", "entry_price", "resolved_value", "timestamp",
    # market_id is read dictionary-encoded (categorical) so the held-out
    # distinct-market count costs almost nothing: it is the effective sample size
    # the cluster-count floor below gates on. Legacy ledgers without the column
    # degrade gracefully (the floor goes inert — see compute_oos_validation).
    "market_id",
]

# Defaults if not overridden in config.scoring. See DECISIONS.md "Out-of-sample
# validation" and HANDOFF.md for why the old sign-only test on 2-bet halves was
# replaced: it measured the favorite-longshot base rate, not skill.
DEFAULT_MIN_BETS_PER_HALF = 10      # resolved bets required in EACH half
DEFAULT_OOS_ALPHA = 0.05            # one-sided significance for held-out skill edge
DEFAULT_PRICE_BASELINE_BINS = 20    # quantile bins for the favorite-longshot curve
DEFAULT_MIN_SKILL_EDGE = 0.02       # economic-magnitude floor for held-out skill edge
# Distinct held-out MARKETS required to certify persistence. Bets that share a
# market share one resolution event, so the effective sample size is the number
# of markets, not bets — `scripts/audit_persistence_cluster.py` measured a median
# design effect of 1.60 (max 230) on the persisted set, and cluster-robust
# inference is itself unreliable below ~30 clusters. This gate (parallel to
# min_bets_per_half, which counts bets) is what keeps a wallet that piled many
# bets into a handful of markets out of the certified set. See HANDOFF.md /
# docs/persistence_cluster_recheck.md.
DEFAULT_MIN_OOS_MARKETS = 30
# Market-block bootstrap resamples for the cluster-robust significance test
# (`_cluster_bootstrap_p`). 2000 gives a p-floor of 1/2001 ~ 5e-4, ample for a
# 0.05 gate. The audit used 2000-4000 and the count was stable across that range.
DEFAULT_OOS_BOOTSTRAP = 2000
# Fixed base seed so the bootstrap is reproducible (the pipeline carries no other
# randomness; audit_persistence.py uses the same value for the same reason). Each
# wallet draws from default_rng([SEED, hash(wallet)]) so its p-value is a function
# of that wallet's own bets alone — independent of iteration order and of which
# OTHER wallets are in the run, so a growing universe never perturbs a prior
# wallet's verdict.
CLUSTER_BOOTSTRAP_SEED = 12345
# --- metric D (non-stationary / regime-change) defaults (see DECISIONS.md) ---
DEFAULT_REGIME_MIN_BETS = 50        # min recent-half bets to attempt the D3 sub-split
DEFAULT_REGIME_TREND_ALPHA = 0.05   # significance for the D1 half-difference + Spearman trend

# Project 1's own scoped certification threshold. Read from
# `scoring.project1.oos_significance_alpha`, falling back to the global
# `scoring.oos_significance_alpha` when the block is absent (legacy configs
# validate exactly as before). 0.005 is the shipped value — see
# `certification_alpha` and config/config.example.yaml.
P1_CFG_SECTION = "project1"


def certification_alpha(scoring: dict) -> float:
    """Project 1's one-sided significance level for `edge_significant`.

    SCOPED ON PURPOSE. `scoring.oos_significance_alpha` is a GLOBAL key that the
    slow-forecaster and sports arms also read (`src/slow_validate.py`,
    `src/slow_forward.py`, `src/sports_forward.py`, `src/forecaster_metrics.py`),
    and those arms have live, pre-registered forward tests frozen against it
    (`slow_freeze_manifest.json`, `sports_freeze_manifest.json`,
    `sports_sig2c_freeze_manifest.json`). Moving the global would silently re-tune
    three running experiments. So Project 1 reads `scoring.project1.
    oos_significance_alpha` first and the global is left at 0.05 for everyone
    else — the same nesting discipline `scoring.slow.*` / `scoring.sports.*`
    already use. Asserted inert for the other arms by
    tests/test_scoped_alpha.py.

    Why it is tighter than the global: `docs/persistence_fdr_hardened.md`
    (2026-07-26) measured the certified set's false-discovery rate for the first
    time under the hardened gate. α=0.05 applied per wallet across a 691-candidate
    search is the dominant source of false discoveries (~35 expected from
    multiplicity alone), and the measured FDR curve (strict cell-permutation null,
    real and null at identical settings) is:

        α=0.05 -> 41 wallets @ 55% FDR   (the old operating point)
        α=0.005 -> 26 wallets @ 20% FDR  (shipped)
        α=0.001 -> 17 wallets @ 10% FDR

    Nothing is dropped from the dataset or the ranking by this — only the
    `edge_persisted` flag moves (CLAUDE.md's never-remove-a-wallet rule)."""
    p1 = scoring.get(P1_CFG_SECTION) or {}
    return float(
        p1.get("oos_significance_alpha",
               scoring.get("oos_significance_alpha", DEFAULT_OOS_ALPHA))
    )


VALIDATED_COLUMNS = [
    "in_sample_edge", "out_of_sample_edge",
    "in_sample_residual_edge", "out_of_sample_residual_edge", "out_of_sample_residual_p",
    "out_of_sample_cluster_p",
    "in_sample_n", "out_of_sample_n", "out_of_sample_markets",
    "edge_significant", "edge_magnitude_ok", "edge_markets_ok", "edge_persisted",
    "regime_flag", "regime_watch", "persisted_recent",
]


def split_in_sample_out_of_sample(wallet_bets: pd.DataFrame, oos_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split of one wallet's resolved bets: first `oos_split`
    fraction (by count) is in-sample (the "fit/select" half), the rest is the
    held-out out-of-sample half.

    The sort is STABLE (`kind="mergesort"`) on purpose. With the default
    quicksort, a wallet whose timestamp at the split boundary is tied with its
    neighbour has its two halves decided by sort internals rather than by the
    data — `scripts/audit_persistence_cluster.py` found 400 such wallets on the
    live ledger, one of which flipped `edge_persisted` between orderings. A
    stable sort makes the split a deterministic function of the input row order
    (which is itself persisted), so the certified set is reproducible."""
    ordered = wallet_bets.sort_values("timestamp", kind="mergesort")
    n = len(ordered)
    split_idx = math.floor(n * oos_split)
    return ordered.iloc[:split_idx], ordered.iloc[split_idx:]


def _gated_mean(values, min_n: int) -> tuple[float, int]:
    """Mean of `values`, but NaN when there are fewer than `min_n` observations.
    NaN (not 0) marks "insufficient data" so it can be down-weighted rather than
    read as a measured absence of edge. Returns the true count regardless."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < min_n:
        return np.nan, n
    return float(arr.mean()), n


def _oos_significance(residuals, min_n: int) -> float:
    """One-sided p-value for H1: mean held-out residual (skill) edge > 0, via a
    t-test. NaN when there are too few bets or no variance to test against (e.g.
    degenerate / replicated bets on one shared outcome) — the caller treats NaN
    as 'not significant', which is the conservative choice.

    Kept and REPORTED (`out_of_sample_residual_p`) for continuity and contrast,
    but it is NO LONGER the arbiter of `edge_significant`: it assumes bets are
    independent, which they are not (bets in one market share one resolution
    event), so on clustered wallets its p-value is too small by ~sqrt(design
    effect) — median 1.60, up to 230 on the live persisted set. The cluster-robust
    `_cluster_bootstrap_p` below is the gate. See DECISIONS.md 'Cluster-robust
    significance'."""
    resid = np.asarray(residuals, dtype=float)
    if resid.size < max(min_n, 2) or np.allclose(resid.std(), 0.0):
        return np.nan
    return float(stats.ttest_1samp(resid, 0.0, alternative="greater").pvalue)


def _wallet_seed(wallet) -> int:
    """Deterministic per-wallet seed from a stable hash of the address, so each
    wallet's bootstrap draws depend only on itself. Python's built-in hash() is
    salted per process, so it cannot be used for reproducibility — blake2b is."""
    digest = hashlib.blake2b(str(wallet).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _cluster_bootstrap_p(residuals, market_labels, rng, n_boot: int) -> float:
    """One-sided p-value for H1: mean held-out skill edge > 0, cluster-robust to
    the fact that bets sharing a market share one resolution event.

    Market-block bootstrap: aggregate the held-out residuals to per-market
    (sum, count), then resample WHOLE markets with replacement and recompute the
    bet-weighted mean, so a market's bets move together and one resolution event
    is one draw. p = (#resamples with mean <= 0 + 1) / (n_boot + 1) — the add-one
    gives a proper, never-zero one-sided p (floor 1/(n_boot+1)).

    NaN with fewer than 2 markets: one resolution event cannot establish
    significance no matter how many bets rode on it, and the caller reads NaN as
    'not significant' — the conservative choice, and the mechanism by which a
    many-bets/few-markets wallet the t-test would certify is now rejected."""
    resid = np.asarray(residuals, dtype=float)
    uniq, inv = np.unique(np.asarray(market_labels), return_inverse=True)
    k = uniq.size
    if k < 2:
        return np.nan
    vsum = np.bincount(inv, weights=resid, minlength=k)
    vcnt = np.bincount(inv, minlength=k).astype(float)
    # Chunk the resamples so peak memory stays ~tens of MB even for a wallet with
    # thousands of held-out markets (index array is the driver); int32 halves it.
    chunk = max(1, min(n_boot, 4_000_000 // k))
    n_le = 0
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, k, size=(b, k), dtype=np.int32)
        means = vsum[idx].sum(axis=1) / vcnt[idx].sum(axis=1)
        n_le += int(np.count_nonzero(means <= 0.0))
        done += b
    return (n_le + 1.0) / (n_boot + 1.0)


# --- metric D: non-stationary / regime-change detection --------------------
# The chronological candidate gate (in_sample_residual_edge > 0) is anti-leakage
# and MUST stay, but it silently buries wallets whose skill edge is NON-STATIONARY
# — most importantly the ones that *became* sharp (early half not sharp, recent
# half strongly sharp). Metric D is DETECTION + honest routing, never a gate
# relaxation: every column below is additive, descriptive metadata that never
# feeds the score or `edge_persisted`. See DECISIONS.md "Metric D" and HANDOFF.md.


def _two_sample_diff_p(early_resid, recent_resid) -> float:
    """Two-sided Welch (unequal-variance) t-test p-value that the recent-half and
    early-half mean residual (skill) edge DIFFER. NaN when either side has no
    variance to test (e.g. degenerate replicated bets), treated as 'no detectable
    difference' by the caller."""
    a = np.asarray(early_resid, dtype=float)
    b = np.asarray(recent_resid, dtype=float)
    if a.size < 2 or b.size < 2 or (np.allclose(a.std(), 0.0) and np.allclose(b.std(), 0.0)):
        return np.nan
    return float(stats.ttest_ind(b, a, equal_var=False).pvalue)


def _trend_spearman(residuals) -> tuple[float, float]:
    """Spearman rank correlation of the per-bet residual (skill) edge against
    chronological order, over one wallet's full ordered series. Returns (rho, p);
    (NaN, NaN) when there are too few points or no variance. Corroborates the
    half-difference test so a regime label needs BOTH a shifted mean and a
    monotonic drift, not one alone."""
    r = np.asarray(residuals, dtype=float)
    if r.size < 3 or np.allclose(r.std(), 0.0):
        return np.nan, np.nan
    rho, p = stats.spearmanr(np.arange(r.size), r)
    return float(rho), float(p)


def _regime_flag(early_resid, recent_resid, min_per_half: int, regime_alpha: float) -> str:
    """D1 classification of one wallet's regime from its two chronological
    residual-edge halves: `insufficient` (a half below `min_per_half`, so
    untestable) / `improving` / `decaying` / `stable`. A directional label
    (improving/decaying) requires BOTH the Welch half-difference AND the Spearman
    time-trend to be significant at `regime_alpha` AND agree in sign; anything
    else is `stable`. `improving_confirmed` is layered on later by D3 only when the
    recent regime itself survives a leakage-free sub-split (see classify_regime)."""
    a = np.asarray(early_resid, dtype=float)
    b = np.asarray(recent_resid, dtype=float)
    if a.size < min_per_half or b.size < min_per_half:
        return "insufficient"
    delta = float(b.mean() - a.mean())
    diff_p = _two_sample_diff_p(a, b)
    rho, trend_p = _trend_spearman(np.concatenate([a, b]))
    diff_sig = (not np.isnan(diff_p)) and diff_p < regime_alpha
    trend_sig = (not np.isnan(trend_p)) and trend_p < regime_alpha
    if delta > 0 and diff_sig and trend_sig and rho > 0:
        return "improving"
    if delta < 0 and diff_sig and trend_sig and rho < 0:
        return "decaying"
    return "stable"


def classify_regime(
    early_resid,
    recent_resid,
    recent_mean: float,
    recent_p: float,
    min_per_half: int,
    alpha: float,
    min_skill_edge: float,
    regime_alpha: float,
    regime_min_bets: int,
) -> tuple[str, bool, bool]:
    """Full metric-D routing for one wallet. Returns
    `(regime_flag, regime_watch, persisted_recent)`.

    - D1 `regime_flag`: improving / decaying / stable / insufficient (from
      `_regime_flag`), promoted to `improving_confirmed` when D3 confirms.
    - D2 `regime_watch`: the "became sharp" watchlist — early residual <= 0 while
      the recent half is significant (`recent_p < alpha`, positive) AND clears the
      economic floor (`recent_mean >= min_skill_edge`). These are exactly the
      copy targets the candidate gate buries; a SORTABLE watchlist column, never
      `edge_persisted` (retro-validating them off a single split would leak).
    - D3 `persisted_recent`: a leakage-free confirmation CONDITIONAL on the
      detected regime change — only for became-sharp wallets whose recent half is
      itself deep (>= `regime_min_bets`); sub-split the RECENT half 50/50 and
      require BOTH recent sub-halves significant (`alpha`) AND material
      (`min_skill_edge`). The early/recent boundary is used only to detect the
      regime; the two recent sub-halves do the honest select/confirm. Such wallets
      get `persisted_recent=True` and `regime_flag='improving_confirmed'`, while
      `edge_persisted` stays False so the canonical certified set is untouched.
    """
    early = np.asarray(early_resid, dtype=float)
    recent = np.asarray(recent_resid, dtype=float)
    flag = _regime_flag(early, recent, min_per_half, regime_alpha)

    testable = early.size >= min_per_half and recent.size >= min_per_half
    early_mean = float(early.mean()) if early.size else np.nan
    recent_significant = (
        (not np.isnan(recent_mean)) and recent_mean > 0
        and (not np.isnan(recent_p)) and recent_p < alpha
    )
    recent_material = (not np.isnan(recent_mean)) and recent_mean >= min_skill_edge
    became_sharp = bool(
        testable and (not np.isnan(early_mean)) and early_mean <= 0
        and recent_significant and recent_material
    )

    persisted_recent = False
    if became_sharp and recent.size >= regime_min_bets:
        h = recent.size // 2
        sub_a, sub_b = recent[:h], recent[h:]
        ma, _ = _gated_mean(sub_a, min_per_half)
        mb, _ = _gated_mean(sub_b, min_per_half)
        pa = _oos_significance(sub_a, min_per_half)
        pb = _oos_significance(sub_b, min_per_half)
        persisted_recent = bool(
            (not np.isnan(ma)) and ma >= min_skill_edge and (not np.isnan(pa)) and pa < alpha
            and (not np.isnan(mb)) and mb >= min_skill_edge and (not np.isnan(pb)) and pb < alpha
        )
    if persisted_recent:
        flag = "improving_confirmed"
    return flag, became_sharp, persisted_recent


def compute_oos_validation(ledger: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Out-of-sample validation on favorite-longshot-neutralized (residual) edge.

    A wallet is a *candidate* if its in-sample residual edge is positive over at
    least `min_bets_per_half` bets. Persistence is gated on THREE independent
    conditions, reported separately:

    - `edge_significant`: held-out residual edge is positive and statistically
      greater than 0 by the CLUSTER-ROBUST market-block bootstrap
      (`out_of_sample_cluster_p < certification_alpha(scoring)`, i.e. the scoped
      `scoring.project1.oos_significance_alpha`). The per-bet t-test
      (`out_of_sample_residual_p`) is still reported for contrast but no longer
      gates: it treats bets as independent, so on a wallet that piled many bets
      into few markets it certifies a mean that rests on a handful of resolution
      events. The bootstrap resamples whole markets, charging one event once.
    - `edge_magnitude_ok`: held-out residual edge clears the economic-magnitude
      floor `min_skill_edge`. Significance is not economic meaning — a bootstrap
      can certify a ~1c edge that is real but below fees. The floor gates those.
    - `edge_markets_ok`: at least `min_oos_markets` distinct held-out markets —
      the effective sample size, so cluster-robust inference has enough clusters
      to be reliable in the first place.

    `edge_persisted = edge_significant AND edge_magnitude_ok AND edge_markets_ok`.
    Raw and residual, in- and out-of-sample edges plus BOTH p-values are reported
    so luck-decay, the favorite-longshot adjustment, and the clustering correction
    are all visible. Every wallet keeps a row (nothing is dropped, per CLAUDE.md —
    this gates the flag/score, never membership)."""
    scoring = cfg["scoring"]
    oos_split = scoring["oos_split"]
    min_per_half = scoring.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF)
    # CERTIFICATION threshold — Project 1 scoped (scoring.project1.
    # oos_significance_alpha, default = the global key). See certification_alpha:
    # tightened to 0.005 because α=0.05 over 691 candidates was the measured
    # dominant source of false discoveries (docs/persistence_fdr_hardened.md).
    alpha = certification_alpha(scoring)
    # metric D deliberately stays on the GLOBAL alpha. D is additive descriptive
    # metadata, never a certification (it never sets edge_persisted), and it is not
    # part of the 691-wide certification search the multiplicity correction above
    # is aimed at — so tightening the gate must not silently move regime_flag /
    # regime_watch / persisted_recent. Asserted by tests/test_scoped_alpha.py.
    regime_sig_alpha = scoring.get("oos_significance_alpha", DEFAULT_OOS_ALPHA)
    n_bins = scoring.get("price_baseline_bins", DEFAULT_PRICE_BASELINE_BINS)
    min_skill_edge = scoring.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE)
    min_oos_markets = scoring.get("min_oos_markets", DEFAULT_MIN_OOS_MARKETS)
    n_boot = scoring.get("oos_bootstrap_resamples", DEFAULT_OOS_BOOTSTRAP)
    regime_min_bets = scoring.get("regime_min_bets", DEFAULT_REGIME_MIN_BETS)
    regime_alpha = scoring.get("regime_trend_alpha", DEFAULT_REGIME_TREND_ALPHA)

    # The cluster-count floor needs market_id; legacy ledgers without it degrade
    # gracefully — the floor goes inert (out_of_sample_markets=NaN, gate True) so
    # old data validates exactly as before rather than silently rejecting.
    has_market = "market_id" in ledger.columns

    bets = only_buys(ledger)
    resolved = bets.loc[bets["resolved"]].copy()
    baseline = fit_price_baseline(resolved, n_bins)
    resolved["residual_edge"] = (
        residual_edge_per_bet(resolved, baseline) if not resolved.empty else pd.Series(dtype=float)
    )

    all_wallets = pd.Index(ledger["wallet"].unique(), name="wallet")
    rows = []
    for wallet, group in resolved.groupby("wallet"):
        ins, out = split_in_sample_out_of_sample(group, oos_split)
        in_raw = ins["resolved_value"] - ins["entry_price"]
        out_raw = out["resolved_value"] - out["entry_price"]

        in_edge, in_n = _gated_mean(in_raw, min_per_half)
        out_edge, out_n = _gated_mean(out_raw, min_per_half)
        in_resid, _ = _gated_mean(ins["residual_edge"], min_per_half)
        out_resid, _ = _gated_mean(out["residual_edge"], min_per_half)
        out_pval = _oos_significance(out["residual_edge"], min_per_half)

        # Distinct held-out markets = the effective (cluster) sample size the
        # persistence gate should rest on. NaN when market_id is unavailable, in
        # which case edge_markets_ok is left True so the floor never silently
        # rejects on legacy data.
        out_markets = (
            int(out["market_id"].nunique()) if has_market and len(out) else
            (0 if has_market else np.nan)
        )

        is_candidate = (not np.isnan(in_resid)) and in_resid > 0

        # Cluster-robust significance (the arbiter of edge_significant). Only worth
        # the bootstrap where it can matter — a candidate whose held-out edge is
        # positive; otherwise it is not significant regardless, so leave the p NaN
        # and skip the work. With no market_id we cannot cluster, so fall back to
        # the t-test to preserve legacy behaviour exactly.
        cluster_p = np.nan
        if is_candidate and (not np.isnan(out_resid)) and out_resid > 0 and len(out) >= min_per_half:
            if has_market:
                rng = np.random.default_rng([CLUSTER_BOOTSTRAP_SEED, _wallet_seed(wallet)])
                cluster_p = _cluster_bootstrap_p(
                    out["residual_edge"].to_numpy(dtype=float),
                    out["market_id"].to_numpy(), rng, n_boot,
                )
            else:
                cluster_p = out_pval  # legacy ledger: no clustering available

        significant = bool(
            is_candidate
            and (not np.isnan(out_resid)) and out_resid > 0
            and (not np.isnan(cluster_p)) and cluster_p < alpha
        )
        magnitude_ok = bool((not np.isnan(out_resid)) and out_resid >= min_skill_edge)
        markets_ok = bool((not has_market) or out_markets >= min_oos_markets)
        persisted = bool(significant and magnitude_ok and markets_ok)

        # metric D — additive regime metadata; never feeds `persisted`/score above.
        # It intentionally keeps the t-test (`out_pval`): its sub-splits are small
        # (regime_min_bets ~ 50, halved again) and often span few markets, where a
        # market-block bootstrap would be mostly undefined — and since D never gates
        # certification, the extra conservatism buys nothing. See DECISIONS.md.
        regime_flag, regime_watch, persisted_recent = classify_regime(
            ins["residual_edge"].to_numpy(dtype=float),
            out["residual_edge"].to_numpy(dtype=float),
            out_resid,
            out_pval,
            min_per_half,
            regime_sig_alpha,
            min_skill_edge,
            regime_alpha,
            regime_min_bets,
        )
        rows.append(
            {
                "wallet": wallet,
                "in_sample_edge": in_edge,
                "out_of_sample_edge": out_edge,
                "in_sample_residual_edge": in_resid,
                "out_of_sample_residual_edge": out_resid,
                "out_of_sample_residual_p": out_pval,
                "out_of_sample_cluster_p": cluster_p,
                "in_sample_n": in_n,
                "out_of_sample_n": out_n,
                "out_of_sample_markets": out_markets,
                "edge_significant": significant,
                "edge_magnitude_ok": magnitude_ok,
                "edge_markets_ok": markets_ok,
                "edge_persisted": persisted,
                "regime_flag": regime_flag,
                "regime_watch": regime_watch,
                "persisted_recent": persisted_recent,
            }
        )

    validated = (
        pd.DataFrame(rows).set_index("wallet") if rows else pd.DataFrame(columns=VALIDATED_COLUMNS)
    )
    validated = validated.reindex(all_wallets)
    validated["in_sample_n"] = validated["in_sample_n"].fillna(0).astype(int)
    validated["out_of_sample_n"] = validated["out_of_sample_n"].fillna(0).astype(int)
    # out_of_sample_markets stays nullable (NaN = untestable / no market_id), so a
    # genuine 0 is distinguishable from "not applicable"; leave dtype as-is.
    for col in ("edge_significant", "edge_magnitude_ok", "edge_markets_ok",
                "edge_persisted", "regime_watch", "persisted_recent"):
        validated[col] = validated[col].fillna(False)
    # wallets with no resolved bets never enter the loop — mark them untestable.
    validated["regime_flag"] = validated["regime_flag"].fillna("insufficient")
    return validated.reset_index()


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    # Reuse the features src.features already computed and cached this run
    # (run.sh runs features before validate) instead of recomputing the whole
    # feature set — which re-ran the two forward-drift passes redundantly and was
    # a large part of the ~30-min re-rank. When cached, validate needs only the
    # slim VALIDATE_LEDGER_COLUMNS; standalone (no cache) it must load the full
    # FEATURE_COLUMNS to compute features as a fallback. Behaviour is unchanged.
    cache_present = WALLET_FEATURES_PATH.exists()
    # market_id read dictionary-encoded — the distinct held-out market count is
    # the cluster-count floor's input and object-materializing it would spike RSS.
    ledger = load_ledger(
        columns=VALIDATE_LEDGER_COLUMNS if cache_present else FEATURE_COLUMNS,
        categorical=["market_id"],
    )
    if ledger.empty:
        print("[validate] ledger is empty — run src.ingest first.")
        return
    if cache_present:
        features = pd.read_parquet(WALLET_FEATURES_PATH)
        print(f"[validate] loaded cached features for {len(features)} wallets")
    else:
        print("[validate] no cached features found — computing (run src.features first to skip this)")
        features = compute_wallet_features(ledger, cfg)
    validated = compute_oos_validation(ledger, cfg)
    merged = features.merge(validated, on="wallet", how="left")

    WALLET_VALIDATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(WALLET_VALIDATED_PATH, compression="gzip")

    n_candidates = int((merged["in_sample_residual_edge"] > 0).sum())
    n_significant = int(merged["edge_significant"].sum())
    n_magnitude = int(merged["edge_magnitude_ok"].sum())
    n_markets_ok = int(merged["edge_markets_ok"].sum())
    n_persisted = int(merged["edge_persisted"].sum())
    rate = f"{n_persisted / n_candidates:.1%}" if n_candidates else "n/a"
    min_skill_edge = cfg["scoring"].get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE)
    min_oos_markets = cfg["scoring"].get("min_oos_markets", DEFAULT_MIN_OOS_MARKETS)
    alpha = certification_alpha(cfg["scoring"])
    print(
        f"[validate] {len(merged)} wallets validated: {n_candidates} candidates "
        f"(positive in-sample skill edge); {n_significant} clear cluster-robust "
        f"significance at alpha={alpha:g}, "
        f"{n_magnitude} clear the {min_skill_edge:+.3f} magnitude floor, "
        f"{n_markets_ok} clear the {min_oos_markets}-held-out-market floor, "
        f"{n_persisted} persisted (all gates, rate {rate}) -> {WALLET_VALIDATED_PATH}"
    )
    # metric D (additive metadata; does not change persisted/rank above)
    n_watch = int(merged["regime_watch"].sum())
    n_recent = int(merged["persisted_recent"].sum())
    flag_counts = merged["regime_flag"].value_counts()
    improving = int(flag_counts.get("improving", 0) + flag_counts.get("improving_confirmed", 0))
    decaying = int(flag_counts.get("decaying", 0))
    print(
        f"[validate] metric D: regime_flag improving={improving} decaying={decaying} "
        f"stable={int(flag_counts.get('stable', 0))} insufficient={int(flag_counts.get('insufficient', 0))}; "
        f"regime_watch (became-sharp)={n_watch}; persisted_recent (D3-confirmed)={n_recent}"
    )


if __name__ == "__main__":
    main()
