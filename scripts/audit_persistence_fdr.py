"""True false-discovery rate of Project 1's certified set, under the HARDENED gate.

THE QUESTION. `docs/persistence_cluster_recheck.md` (2026-07-23) reported an
implied **~50% FDR** on the persisted set: real 118 persisted of 691 candidates
vs a cluster-preserving null mean of 59.0 over 200 shuffles. That number has been
carried on the headline deliverable ever since — but it was measured against a
gate that **no longer exists**. Immediately after that audit three fixes landed in
`src/validate.py`:

  1. stable split (`kind="mergesort"`),
  2. a cluster-count floor (`out_of_sample_markets >= scoring.min_oos_markets`,
     default 30)  — took the set 118 -> 70,
  3. cluster-robust significance (`edge_significant` gated by
     `validate._cluster_bootstrap_p`, not the per-bet t-test) — took it 70 -> 41.

The null was never re-run through that gate, so "~50%" is an upper bound carried
over from a weaker test and the true FDR of the live 41 is unmeasured. This script
measures it.

WHY A SIBLING SCRIPT rather than a mode on `scripts/audit_persistence_cluster.py`:
that script's vectorized replica IS the pre-hardening gate (per-bet t-test, no
market floor). It is the reproduction recipe printed at the top of
`docs/persistence_cluster_recheck.md`, and rewriting its gate would silently
invalidate the published 118/59 numbers. It stays frozen as the historical
record; this file reuses its permutation (via `audit_speed_gradient`) and adds a
replica of TODAY's gate.

WHAT IT RUNS.

  REAL ARM — a vectorized replica of `src.validate.compute_oos_validation`'s four
  gates (candidate / cluster-robust significance / magnitude floor / market floor)
  that calls `src.validate._cluster_bootstrap_p` itself, with `src.validate`'s own
  per-wallet seeding, so its p-values are bit-identical to the shipped pipeline.
  It is asserted equal to the shipped `data/interim/wallet_validated.parquet`
  (the artifact `python -m src.validate` produced from this ledger) before any
  null is scored. If it does not reproduce the live count, the run aborts.

  NULL ARM B (cluster-preserving; the 2026-07-23 audit's bracket) — permute WHICH
  WALLET made each bet, within each market, leaving every market's bets, prices,
  outcomes and timestamps exactly intact. Implied FDR = null_mean / real_count.

  NULL ARM B2 (cluster-preserving, market-mix-randomizing) — B is a CONDITIONAL
  null. It holds each wallet's market footprint (which markets, how many bets in
  each) exactly fixed and scrambles only who made which bet inside a market, so it
  asks "with this footprint but no within-market edge, how often does the gate
  fire?". Whatever a wallet earns purely by being exposed to a set of markets whose
  buyers collectively beat E[outcome|price] therefore survives into B's null. B2
  removes that conditioning: it treats each (wallet, market) CELL as the unit and
  permutes cell OWNERSHIP between wallets within cell-size strata. Every wallet
  keeps its exact bet count and cell-size profile (and its distinct-market count
  bar a rare same-market collision, which points conservatively); every cell keeps
  its bets, prices, outcomes and shared resolution intact; but WHICH markets it
  held is randomized, so the wallet<->outcome link is destroyed outright. B2 is
  therefore the unconditional "pure chance" null and B the stricter conditional one.
  Report both: B2 answers "how many certifications does chance manufacture", B
  answers "...and how many survive holding the market footprint fixed".

  DIAGNOSTICS THAT SAY WHICH NULL IS READABLE — measured, not assumed:
    * label retention: share of bets whose wallet label the permutation leaves
      unchanged (overall, and restricted to the certified set). A permutation that
      hands a wallet its own record back cannot manufacture a false discovery, it
      can only reproduce a true one — B is the identity in any market a single
      wallet trades alone, so this has to be checked, not assumed.
    * structural retention: the same quantity in closed form from the per-market
      wallet mix, available without shuffling.
    * overlap: how many of a replicate's null "persisters" ARE the real certified
      wallets. High overlap would mean the null is re-certifying the real signal
      rather than manufacturing new ones.

  DESIGN EFFECT — D = Var_null(held-out mean residual) / mean_null(s^2/m), per
  wallet, i.e. the same Var(null)/Var(independence-model) ratio
  `docs/blackswan_cluster_null.md` demands before any null is trusted. That doc
  found D ~ 0.17 for the within-market permutation in a SPARSE stratum, which made
  it unusable for per-wallet arbitration. D is measured for both brackets here so
  the reader knows which claims each null can carry.

LIKE-FOR-LIKE. Both arms use identical `oos_split`, `min_bets_per_half`,
`oos_significance_alpha`, `min_skill_edge`, `min_oos_markets` and — critically —
identical `--boot` (bootstrap resamples). `--boot` overrides
`scoring.oos_bootstrap_resamples` for BOTH arms or neither.

EXACTNESS NOTE (a speed optimization that changes nothing). `edge_persisted` is a
conjunction, so in the null arm the expensive bootstrap is evaluated LAST, only
for wallets that already clear the candidate, magnitude and market gates. Wallets
that fail a cheap gate cannot persist whatever their p-value is, so the persisted
count is exact; the null's `edge_significant` count is simply not produced (it is
not what FDR is computed from). The real arm evaluates every gate for every
eligible wallet so it can be checked against the shipped artifact.

READ-ONLY. Reads `data/interim/bet_ledger.parquet` and
`data/interim/wallet_validated.parquet`; writes nothing.

    PYTHONPATH=. .venv/bin/python scripts/audit_persistence_fdr.py \
        --shuffles 200 --boot 2000

Take `data/interim/.analysis.lock` around it (it loads the full ledger) and
detach it — a full run is tens of minutes:

    setsid nohup flock data/interim/.analysis.lock \
        .venv/bin/python scripts/audit_persistence_fdr.py > ~/.pmrun/fdr.log 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.common import WALLET_VALIDATED_PATH, load_config, load_ledger  # noqa: E402
from src.features import fit_price_baseline, residual_edge_per_bet  # noqa: E402
from src.validate import (  # noqa: E402
    CLUSTER_BOOTSTRAP_SEED,
    DEFAULT_MIN_BETS_PER_HALF,
    DEFAULT_MIN_OOS_MARKETS,
    DEFAULT_MIN_SKILL_EDGE,
    DEFAULT_OOS_BOOTSTRAP,
    DEFAULT_PRICE_BASELINE_BINS,
    _cluster_bootstrap_p,
    _wallet_seed,
    certification_alpha,
)
from audit_speed_gradient import permute_wallets_within_market  # noqa: E402

SEED = 20260726  # null-arm base seed (the real arm uses validate's own)
LEDGER_COLS = ["wallet", "market_id", "side", "entry_price", "resolved_value",
               "resolved", "timestamp"]


# --------------------------------------------------------------------------- #
# Data prep                                                                    #
# --------------------------------------------------------------------------- #
def _lexicographic_codes(cat: pd.Series) -> np.ndarray:
    """Integer codes for a categorical column, renumbered so that sorting the
    codes is the same as sorting the underlying strings.

    Load-bearing for exactness, not cosmetics: `_cluster_bootstrap_p` calls
    `np.unique(market_labels)`, so the ORDER of the per-market (sum, count)
    vectors it resamples is the sorted order of whatever labels it is handed.
    `src.validate` hands it market_id STRINGS. Handing it integer codes in
    dictionary (first-appearance) order would permute those vectors and draw a
    different — equally valid but not identical — bootstrap sample, so the
    replica's p-values would only match the pipeline's up to Monte-Carlo noise.
    Ranking the codes lexicographically makes them match exactly."""
    cats = cat.cat.categories.to_numpy()
    rank = np.empty(cats.size, dtype=np.int64)
    rank[np.argsort(cats, kind="stable")] = np.arange(cats.size)
    codes = cat.cat.codes.to_numpy()
    return rank[codes].astype(np.int64)


def prepare_realworld(cfg):
    """The same arrays, but for the deep REAL-WORLD sample (src/realworld_deepen.py).

    Built by calling `realworld_validate.assemble_population` rather than
    re-implementing it, so the tape this null is measured against is bit-identical
    to the one `data/interim/realworld/validated.parquet` was produced from —
    including the micro-crypto restriction being applied BEFORE the price baseline
    is fitted. Anything less and the null would be calibrated against a population
    the real arm never saw, which is the failure mode `check_against_pipeline`
    exists to catch."""
    from src.realworld_validate import assemble_population

    bets, _report = assemble_population(real_world=True)
    bets = bets.loc[bets["resolved"].astype(bool)]
    for col in ("wallet", "market_id"):
        if str(bets[col].dtype) != "category":
            bets[col] = bets[col].astype("category")

    wallets = bets["wallet"].cat.categories.to_numpy()
    wcode = bets["wallet"].cat.codes.to_numpy().astype(np.int64)
    mcode = _lexicographic_codes(bets["market_id"])
    ts = bets["timestamp"].to_numpy(dtype=float)
    frame = pd.DataFrame({"entry_price": bets["entry_price"].to_numpy(dtype=float),
                          "resolved_value": bets["resolved_value"].to_numpy(dtype=float)})
    del bets

    n_bins = cfg["scoring"].get("price_baseline_bins", DEFAULT_PRICE_BASELINE_BINS)
    baseline = fit_price_baseline(frame, n_bins)
    resid = residual_edge_per_bet(frame, baseline)
    del frame
    return {"wcode": wcode, "mcode": mcode, "ts": ts, "resid": resid,
            "wallets": wallets, "n_wallets": wallets.size,
            "n_markets": int(mcode.max()) + 1 if mcode.size else 1}


def prepare(cfg, tape: str = "ledger"):
    if tape == "realworld":
        return prepare_realworld(cfg)
    return _prepare_ledger(cfg)


def _prepare_ledger(cfg):
    """Resolved BUY bets as flat arrays: wallet codes, lexicographic market codes,
    timestamps, and the per-bet residual (skill) edge the validator scores,
    computed from the same price baseline it uses.

    Everything stays as numpy arrays (and the two big string columns are read
    dictionary-encoded) because this box has 3.8 GB of RAM and materializing 4.7M
    Python strings has OOM'd it before."""
    led = load_ledger(columns=LEDGER_COLS,
                      categorical=["market_id", "wallet", "side"])
    keep = (led["side"] == "BUY").to_numpy() & led["resolved"].to_numpy(dtype=bool)
    wallets = led["wallet"].cat.categories.to_numpy()
    wcode = led["wallet"].cat.codes.to_numpy().astype(np.int64)[keep]
    mcode = _lexicographic_codes(led["market_id"])[keep]
    ts = led["timestamp"].to_numpy(dtype=float)[keep]
    entry = led["entry_price"].to_numpy(dtype=float)[keep]
    value = led["resolved_value"].to_numpy(dtype=float)[keep]
    del led, keep

    frame = pd.DataFrame({"entry_price": entry, "resolved_value": value})
    n_bins = cfg["scoring"].get("price_baseline_bins", DEFAULT_PRICE_BASELINE_BINS)
    baseline = fit_price_baseline(frame, n_bins)
    resid = residual_edge_per_bet(frame, baseline)
    del frame, entry, value
    return {"wcode": wcode, "mcode": mcode, "ts": ts, "resid": resid,
            "wallets": wallets, "n_wallets": wallets.size,
            "n_markets": int(mcode.max()) + 1 if mcode.size else 1}


# --------------------------------------------------------------------------- #
# Null B2 — cell-permuting, cluster-preserving, un-frozen                      #
# --------------------------------------------------------------------------- #
def build_cells(wcode, mcode):
    """Decompose the tape into (wallet, market) CELLS.

    Returns `(order, cell_owner, cell_size)` where `order` sorts the bets into
    contiguous cells and `cell_owner`/`cell_size` describe each cell. A cell is the
    natural cluster unit: all of its bets settle on one resolution event, so moving
    a cell moves that event's whole contribution together."""
    order = np.lexsort((mcode, wcode))
    w, m = wcode[order], mcode[order]
    new = np.empty(w.size, dtype=bool)
    new[0] = True
    np.not_equal(w[1:], w[:-1], out=new[1:])
    new[1:] |= m[1:] != m[:-1]
    starts = np.flatnonzero(new)
    cell_size = np.diff(np.append(starts, w.size))
    return order, w[starts], cell_size


def permute_cells_within_size(order, cell_owner, cell_size, rng):
    """CLUSTER-PRESERVING, UN-FROZEN null: permute which wallet owns each
    (wallet, market) cell, among cells of the SAME size.

    Each wallet therefore keeps its exact bet count and its exact cell-size profile
    — what the hardened gate's bet and magnitude floors key on — while the markets
    (and hence the outcomes) it held are randomized. Bets inside a cell never
    separate, so the correlation from a shared resolution event survives exactly as
    in bracket B.

    Its distinct-market count is preserved up to one rare collision: two cells of
    the SAME market and the same size can land on one wallet, which costs that
    wallet a distinct market. With 348k markets this is negligible, and it points
    the conservative way (fewer held-out markets = harder to clear the cluster-count
    floor = a SMALLER null count, so it cannot inflate the FDR this measures). The
    run prints the null's `markets_ok` population against the real one so the size
    of the effect is visible rather than assumed.

    Limitation, stated plainly and symmetric with B's: this destroys any skill that
    comes from CHOOSING which markets to bet, so it cannot see market selection as
    anything but chance. Unlike B, it does not also hand a monopolist wallet its
    own record back.

    Returns the permuted per-bet wallet codes in the ORIGINAL bet order."""
    by_size = np.argsort(cell_size, kind="stable")
    perm = np.lexsort((rng.random(cell_size.size), cell_size))
    new_owner = np.empty_like(cell_owner)
    new_owner[by_size] = cell_owner[perm]
    out = np.empty_like(cell_owner, shape=order.size)
    out[order] = np.repeat(new_owner, cell_size)
    return out


def structural_retention(wcode, mcode, n_wallets):
    """Closed-form share of each wallet's bets that bracket B leaves ON THAT WALLET.

    Bracket B rearranges a market's label multiset uniformly at random, so a bet in
    a market with `n` bets of which the wallet holds `c` keeps its own label with
    probability c/n — the wallet's share of that market's tape (and exactly 1 when
    it is the market's only trader). Averaging over a wallet's bets gives the
    fraction of its record the permutation cannot touch. A value near 1 means
    bracket B is close to the IDENTITY for that wallet, and its "null" verdict is
    not a null at all."""
    order, cell_owner, cell_size = build_cells(wcode, mcode)
    mkt = mcode[order][np.append(0, np.cumsum(cell_size))[:-1]]
    mkt_total = np.bincount(mkt, weights=cell_size, minlength=int(mcode.max()) + 1)
    n = mkt_total[mkt]
    c = cell_size.astype(float)
    keep = c / n
    num = np.bincount(cell_owner, weights=c * keep, minlength=n_wallets)
    den = np.bincount(cell_owner, weights=c, minlength=n_wallets)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# --------------------------------------------------------------------------- #
# The hardened gate, vectorized                                                #
# --------------------------------------------------------------------------- #
def split_stats(wcode, ts, resid, n_wallets, oos_split, min_per_half):
    """Per-wallet chronological split statistics, vectorized over a
    (wallet, time)-sorted view.

    Reproduces `src.validate.split_in_sample_out_of_sample` + `_gated_mean`:
    a STABLE sort by timestamp within each wallet (validate uses
    `kind="mergesort"`; `np.lexsort` is likewise stable, so ties keep input row
    order in both), split at `floor(n * oos_split)`, and in/out means that are
    NaN below `min_per_half` observations.

    Returns `(order, starts, k, m, in_mean, out_mean, out_var)` where `order` is
    the sort permutation (so the caller can slice held-out rows), `starts` the
    per-wallet block starts, `k`/`m` the in/held-out counts, and `out_var` the
    ddof=1 variance of each wallet's held-out residuals (NaN below 2), used only
    to report the design effect."""
    order = np.lexsort((ts, wcode))
    w = wcode[order]
    x = resid[order]
    idx = np.arange(n_wallets)
    starts = np.searchsorted(w, idx, side="left")
    ends = np.searchsorted(w, idx, side="right")
    n = ends - starts
    k = np.floor(n * oos_split).astype(np.int64)
    m = n - k

    csum = np.concatenate([[0.0], np.cumsum(x)])
    with np.errstate(invalid="ignore", divide="ignore"):
        in_mean = np.where(k >= min_per_half,
                           (csum[starts + k] - csum[starts]) / np.maximum(k, 1), np.nan)
        raw_out = np.where(m > 0, (csum[ends] - csum[starts + k]) / np.maximum(m, 1), np.nan)
        out_mean = np.where(m >= min_per_half, raw_out, np.nan)

    # Held-out spread, TWO-PASS. A one-pass sum-of-squares loses all precision
    # here: many wallets replicate one bet hundreds of times, so E[x^2] and
    # mean^2 agree to ~15 digits and their difference is pure rounding noise.
    pos = np.arange(x.size) - np.repeat(starts, n)
    held = pos >= np.repeat(k, n)
    xh = x[held]
    dev = xh - np.repeat(np.where(np.isfinite(raw_out), raw_out, 0.0), m)
    off = np.concatenate([[0], np.cumsum(m)])[:-1]
    valid = m > 0
    ssq = np.zeros(n_wallets)
    if xh.size:
        ssq[valid] = np.add.reduceat(dev * dev, off[valid])
    with np.errstate(invalid="ignore", divide="ignore"):
        out_var = np.where(m >= 2, ssq / np.maximum(m - 1, 1), np.nan)
    return order, starts, k, m, in_mean, out_mean, out_var


def held_out_market_counts(order, starts, k, m, wcode, mcode, n_wallets, n_markets):
    """Distinct held-out markets per wallet — `out["market_id"].nunique()` in
    `src.validate`, vectorized for every wallet at once."""
    w = wcode[order]
    mm = mcode[order]
    n = k + m
    pos = np.arange(w.size) - np.repeat(starts, n)
    held = pos >= np.repeat(k, n)
    key = w[held] * np.int64(n_markets) + mm[held]
    if key.size == 0:
        return np.zeros(n_wallets, dtype=np.int64)
    uniq = np.unique(key)
    return np.bincount(uniq // np.int64(n_markets), minlength=n_wallets).astype(np.int64)


def hardened_gate(data, params, boot_seed, full=False, mkt_counts=None):
    """`src.validate.compute_oos_validation`'s persistence verdict, replicated.

    `boot_seed(wallet_code) -> np.random.Generator` supplies the bootstrap RNG, so
    the real arm can use `src.validate`'s exact per-wallet seeding while the null
    arm varies its stream per shuffle.

    With `full=False` the expensive bootstrap runs only for wallets that already
    clear the candidate, magnitude and market gates. `edge_persisted` is a
    conjunction, so the persisted set is identical either way; `full=True`
    additionally evaluates significance for every wallet `src.validate` would
    bootstrap, which is what the real-arm equality check needs.

    Returns a dict of boolean arrays plus the diagnostics the caller reports."""
    wcode, mcode, ts, resid = data["wcode"], data["mcode"], data["ts"], data["resid"]
    n_wallets = data["n_wallets"]
    order, starts, k, m, in_mean, out_mean, out_var = split_stats(
        wcode, ts, resid, n_wallets, params["oos_split"], params["min_per_half"])

    candidate = np.isfinite(in_mean) & (in_mean > 0)
    positive = candidate & np.isfinite(out_mean) & (out_mean > 0)
    magnitude_ok = np.isfinite(out_mean) & (out_mean >= params["min_skill_edge"])

    if mkt_counts is None:
        mkt_counts = held_out_market_counts(order, starts, k, m, wcode, mcode,
                                            n_wallets, data["n_markets"])
    markets_ok = mkt_counts >= params["min_oos_markets"]

    to_test = positive if full else (positive & magnitude_ok & markets_ok)
    xs = resid[order]
    ms = mcode[order]
    cluster_p = np.full(n_wallets, np.nan)
    for wid in np.flatnonzero(to_test):
        lo = starts[wid] + k[wid]
        hi = lo + m[wid]
        cluster_p[wid] = _cluster_bootstrap_p(
            xs[lo:hi], ms[lo:hi], boot_seed(wid), params["n_boot"])

    significant = positive & np.isfinite(cluster_p) & (cluster_p < params["alpha"])
    persisted = significant & magnitude_ok & markets_ok
    # The same gate at stricter significance thresholds. Free: tightening alpha can
    # only remove wallets from `to_test`, so no extra bootstrap is needed, and it is
    # what turns a single FDR number into an FDR *curve* the reader can operate on.
    base = positive & magnitude_ok & markets_ok & np.isfinite(cluster_p)
    at_alpha = {a: int((base & (cluster_p < a)).sum()) for a in params.get("alphas", ())}
    return {"candidate": candidate, "edge_significant": significant, "at_alpha": at_alpha,
            "edge_magnitude_ok": magnitude_ok, "edge_markets_ok": markets_ok,
            "edge_persisted": persisted, "out_mean": out_mean, "out_var": out_var,
            "out_markets": mkt_counts, "cluster_p": cluster_p, "m": m,
            "n_tested": int(to_test.sum())}


# --------------------------------------------------------------------------- #
# Design effect                                                                #
# --------------------------------------------------------------------------- #
def design_effect(mean_sq_sum, mean_sum, indep_var_sum, n_shuffles):
    """D = Var_null(held-out mean) / mean_null(s^2 / m), per wallet.

    Var(null) over the permutation replicates vs the variance the validator's own
    per-bet test assumes. D > 1 means the null is wider than independence (the
    correction a clustering-aware null should produce). D < 1 means the
    permutation is MORE constrained than independence — over-restricted, and per
    `docs/blackswan_cluster_null.md` unusable for per-wallet inference, though its
    count-level contrast still stands because the same constraint binds both arms.

    Takes running sums so the caller never has to hold an (n_shuffles x n_wallets)
    matrix on a 3.8 GB box. Var(null) uses the UNBIASED (ddof=1) estimator: the
    replicates are a finite sample from the permutation distribution, and the
    population divisor would shrink D by (S-1)/S — a 33% understatement at S=3,
    which is exactly the regime the smoke runs use."""
    if n_shuffles < 2:
        return np.full_like(mean_sum, np.nan, dtype=float)
    mean = mean_sum / n_shuffles
    var_null = np.maximum(mean_sq_sum - n_shuffles * mean * mean, 0.0) / (n_shuffles - 1)
    indep = indep_var_sum / n_shuffles
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(indep > 0, var_null / indep, np.nan)


def benjamini_hochberg(pvals, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up rejections at level `q` over a family of
    p-values. Returns a boolean mask of rejections.

    The count-level nulls in this script cannot name wallets (their design effect
    is below 1, see `design_effect`). BH over the per-wallet CLUSTER-ROBUST
    bootstrap p — which IS the shipped gate's own statistic — can: it controls the
    expected false-discovery proportion of the named set directly, rather than
    estimating it for the count. NaN p-values are treated as 1.0 (never rejected),
    matching how `src.validate` reads a NaN cluster p."""
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresh = q * np.arange(1, m + 1) / m
    below = np.flatnonzero(ranked <= thresh)
    out = np.zeros(m, dtype=bool)
    if below.size:
        out[order[: below[-1] + 1]] = True
    return out


def mc_error(counts) -> float:
    """Monte-Carlo standard error of the null mean over `counts` replicates."""
    counts = np.asarray(counts, dtype=float)
    if counts.size < 2:
        return float("nan")
    return float(counts.std(ddof=1) / np.sqrt(counts.size))


# --------------------------------------------------------------------------- #
def check_against_pipeline(flags, data, tape: str = "ledger") -> int:
    """Assert the replica reproduces the SHIPPED artifact.

    `data/interim/wallet_validated.parquet` is what `python -m src.validate`
    wrote from this ledger, so comparing against it is the strongest available
    check — it validates the replica against the number the repo actually
    certifies, not against a re-derivation. Returns the live persisted count."""
    truth_path = WALLET_VALIDATED_PATH
    if tape == "realworld":
        from src.realworld_validate import VALIDATED_PATH as truth_path  # noqa: N813
    truth = pd.read_parquet(truth_path,
                            columns=["wallet", "edge_significant", "edge_magnitude_ok",
                                     "edge_markets_ok", "edge_persisted"])
    mine = pd.DataFrame({
        "wallet": data["wallets"],
        "edge_significant": flags["edge_significant"],
        "edge_magnitude_ok": flags["edge_magnitude_ok"],
        "edge_markets_ok": flags["edge_markets_ok"],
        "edge_persisted": flags["edge_persisted"],
    })
    j = mine.merge(truth, on="wallet", suffixes=("_r", "_t"), how="inner")
    print(f"  [check] compared {len(j):,} wallets against {truth_path.name}")
    ok = True
    for col in ("edge_significant", "edge_magnitude_ok", "edge_markets_ok", "edge_persisted"):
        diff = j[f"{col}_r"].astype(bool) != j[f"{col}_t"].astype(bool)
        print(f"  [check] {col}: {int(diff.sum())} mismatches "
              f"(replica {int(j[f'{col}_r'].sum())} / shipped {int(j[f'{col}_t'].sum())})")
        ok &= int(diff.sum()) == 0
    if not ok:
        raise SystemExit(
            "[abort] the replica does not reproduce the shipped gate. A null measured "
            "against a mis-wired real number is worthless — debug before reporting.")
    print("  [check] hardened replica == src.validate's shipped output ✓")
    return int(j["edge_persisted_t"].sum())


def run_bh(data, params, real, n_cand, bh_boot, wallets) -> None:
    """Per-wallet multiplicity control over the candidate family.

    The gate is a per-wallet test at alpha over ~691 candidates, so ~35 rejections
    are expected from multiplicity alone before any clustering is considered. BH
    prices that in directly, and unlike the count-level nulls it NAMES the set it
    controls.

    It is run at higher resolution than the shipped 2000 resamples for two
    reasons: that setting pins the strongest wallets to its p-floor of 1/2001 as
    exact ties (BH's step-up cannot order tied p-values, so the rejected set is
    decided by an artefact of the resample count), and re-running the same wallets
    at a different resolution doubles as a stability check on the shipped p."""
    print(f"\n--- 2b. Per-wallet multiplicity: BH over the cluster-robust p "
          f"({bh_boot} resamples) ---", flush=True)
    t0 = time.time()

    def seed(wid):
        return np.random.default_rng([CLUSTER_BOOTSTRAP_SEED, _wallet_seed(wallets[wid])])

    hi = hardened_gate(data, dict(params, n_boot=bh_boot), seed, full=True,
                       mkt_counts=real["out_markets"])
    print(f"    re-bootstrapped {hi['n_tested']} wallets at {bh_boot} resamples "
          f"[{time.time()-t0:.0f}s]; p-floor {1/(bh_boot+1):.2e}")
    fam = real["candidate"]
    p = np.where(fam, np.where(np.isfinite(hi["cluster_p"]), hi["cluster_p"], 1.0), np.nan)
    pv = p[fam]
    print(f"    family = {pv.size} candidates; BH first threshold q/m = "
          f"{0.10/max(pv.size,1):.2e}")
    cert = real["edge_persisted"]
    for q in (0.05, 0.10, 0.20):
        rej = np.zeros(data["n_wallets"], dtype=bool)
        rej[fam] = benjamini_hochberg(pv, q)
        both = rej & real["edge_magnitude_ok"] & real["edge_markets_ok"]
        print(f"    q={q:<5} BH rejections: {int(rej.sum()):3d}   "
              f"...also clearing the magnitude + market floors: {int(both.sum()):3d}   "
              f"of which in the certified set: {int((both & cert).sum()):3d} / {int(cert.sum())}")
    # how far the certified set's own p-values move at higher resolution
    c = np.flatnonzero(cert)
    print(f"    certified set's cluster p at {bh_boot} resamples: median "
          f"{np.median(hi['cluster_p'][c]):.5f}  max {np.nanmax(hi['cluster_p'][c]):.5f}  "
          f"(at {params['n_boot']}: median {np.median(real['cluster_p'][c]):.5f}  "
          f"max {np.nanmax(real['cluster_p'][c]):.5f})")
    still = int((hi["cluster_p"][c] < params["alpha"]).sum())
    print(f"    of the {c.size} certified, {still} still clear alpha={params['alpha']} "
          f"at the higher resolution (a stability check on the shipped 2000)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--boot", type=int, default=None,
                    help="bootstrap resamples; applied to BOTH arms "
                         "(default: scoring.oos_bootstrap_resamples)")
    ap.add_argument("--bh-boot", type=int, default=0,
                    help="re-bootstrap every tested wallet at this many resamples and "
                         "apply Benjamini-Hochberg over the candidate family. Higher than "
                         "the shipped 2000 because that setting piles the strongest "
                         "wallets onto its 1/2001 p-floor as exact ties, which BH cannot "
                         "order. 0 = skip.")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the equality check against the shipped artifact "
                         "(for parameter sweeps where the gate is deliberately different)")
    ap.add_argument("--tape", choices=("ledger", "realworld"), default="ledger",
                    help="population: the shared bet ledger (default, checked against "
                         "wallet_validated.parquet) or the deep real-world sample "
                         "(checked against realworld/validated.parquet). The gate and "
                         "both nulls are identical either way.")
    args = ap.parse_args()

    cfg = load_config()
    sc = cfg["scoring"]
    n_boot = args.boot if args.boot is not None else sc.get(
        "oos_bootstrap_resamples", DEFAULT_OOS_BOOTSTRAP)
    params = {
        "oos_split": sc["oos_split"],
        "min_per_half": sc.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF),
        # Project 1's SCOPED certification threshold (scoring.project1.
        # oos_significance_alpha, default = the global key). Must go through the
        # same resolver src.validate uses or the bit-identity check below fails.
        "alpha": certification_alpha(sc),
        "min_skill_edge": sc.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE),
        "min_oos_markets": sc.get("min_oos_markets", DEFAULT_MIN_OOS_MARKETS),
        "n_boot": n_boot,
        # The gate's own alpha plus tighter operating points. 0.0171 and 0.0073 are
        # the largest p BH rejects at q=0.10 and q=0.05 over this 691-wide family
        # (see --bh-boot); reporting the null count there is what converts BH's
        # named set into a measured false-discovery share.
        "alphas": (0.05, 0.0171, 0.0073, 0.005, 0.001),
    }

    print("=" * 84)
    print("PROJECT 1 — TRUE FDR OF THE CERTIFIED SET UNDER THE HARDENED GATE (read-only)")
    print("=" * 84)
    t0 = time.time()
    data = prepare(cfg, args.tape)
    print(f"resolved BUY bets: {data['resid'].size:,}   wallets in ledger: "
          f"{data['n_wallets']:,}   distinct markets: {data['n_markets']:,}   "
          f"[{time.time()-t0:.0f}s]")
    print(f"gates (IDENTICAL in both arms): oos_split={params['oos_split']} "
          f"min_bets_per_half={params['min_per_half']} alpha={params['alpha']} "
          f"min_skill_edge={params['min_skill_edge']} "
          f"min_oos_markets={params['min_oos_markets']} bootstrap_resamples={n_boot}")

    # -- 1. Real arm ---------------------------------------------------------
    print("\n--- 1. Real arm: reproduce the live certified set ---", flush=True)
    t0 = time.time()
    wallets = data["wallets"]

    def real_seed(wid):
        return np.random.default_rng([CLUSTER_BOOTSTRAP_SEED, _wallet_seed(wallets[wid])])

    real = hardened_gate(data, params, real_seed, full=True)
    n_real = int(real["edge_persisted"].sum())
    n_cand = int(real["candidate"].sum())
    print(f"  candidates (in-sample skill edge > 0): {n_cand}")
    print(f"  bootstrapped: {real['n_tested']}   cluster-significant: "
          f"{int(real['edge_significant'].sum())}   magnitude_ok: "
          f"{int(real['edge_magnitude_ok'].sum())}   markets_ok: "
          f"{int(real['edge_markets_ok'].sum())}")
    print(f"  edge_persisted: {n_real}   (rate {n_real/max(n_cand,1):.1%} of candidates)"
          f"   [{time.time()-t0:.0f}s]")
    if not args.no_check:
        n_real = check_against_pipeline(real, data, args.tape)

    # -- 2. Where the certified set sits on a stricter reading ---------------
    idx = np.flatnonzero(real["edge_persisted"])
    if idx.size:
        p = real["cluster_p"][idx]
        mk = real["out_markets"][idx]
        om = real["out_mean"][idx]
        print(f"\n--- 2. The certified {idx.size} on a stricter reading ---")
        for lab, thr in (("cluster p < 0.01", (p < 0.01)), ("cluster p < 0.005", (p < 0.005)),
                         ("held-out markets >= 100", (mk >= 100)),
                         ("held-out skill edge >= 0.05", (om >= 0.05)),
                         ("all three (p<0.01, mkts>=100, edge>=0.05)",
                          (p < 0.01) & (mk >= 100) & (om >= 0.05))):
            print(f"    {lab:<44} {int(np.sum(thr)):3d} of {idx.size}")
        print(f"    cluster p: median {np.median(p):.4f}  max {p.max():.4f}")
        print(f"    held-out markets: min {mk.min()}  median {np.median(mk):.0f}  max {mk.max()}")

    if args.bh_boot:
        run_bh(data, params, real, n_cand, args.bh_boot, wallets)
    if args.shuffles <= 0:
        print("\n[--shuffles 0] count-level null arms skipped.")
        return

    # -- 3. How much can each null actually move? ----------------------------
    struct = structural_retention(data["wcode"], data["mcode"], data["n_wallets"])
    print("\n--- 3. Frozenness: what share of a wallet's record does null B hand back? ---")
    testable = np.isfinite(struct) & (real["m"] >= params["min_per_half"])
    print(f"  structural retention under B (weighted mean market share):")
    print(f"    all {int(testable.sum()):,} testable wallets: median "
          f"{np.median(struct[testable]):.3f}  90th {np.percentile(struct[testable],90):.3f}")
    cset = real["edge_persisted"]
    print(f"    the {int(cset.sum())} certified wallets:  median {np.median(struct[cset]):.3f}  "
          f"min {struct[cset].min():.3f}  max {struct[cset].max():.3f}  "
          f"share >0.9: {100*(struct[cset] > 0.9).mean():.0f}%")
    print("  Retention is the share of a wallet's bets that bracket B leaves on that wallet,")
    print("  outcome and all. Near 1 the permutation is the identity and B tests nothing;")
    print("  low retention means B really does scramble the record and its count is readable.")

    # -- 4. Null arms --------------------------------------------------------
    order_cells, cell_owner, cell_size = build_cells(data["wcode"], data["mcode"])
    # Fixed per-bracket stream ids: Python's hash() is salted per process, so it
    # can never appear in a seed that has to be reproducible across runs.
    brackets = {
        "B": (1, "within-market label permutation (2026-07-23 bracket)",
              lambda rng: permute_wallets_within_market(data["wcode"], data["mcode"], rng)),
        "B2": (2, "cell permutation within size strata (un-frozen)",
               lambda rng: permute_cells_within_size(order_cells, cell_owner, cell_size, rng)),
    }
    results = {}
    n_w = data["n_wallets"]
    for key, (stream, label, permute) in brackets.items():
        print(f"\n--- 4{key}. Null {key} — {label}; {args.shuffles} shuffles ---", flush=True)
        rng = np.random.default_rng([SEED, stream])
        per_counts, cand_counts, overlap_counts, retention, mkt_ok = [], [], [], [], []
        alpha_counts = {}
        s_sum = np.zeros(n_w); s_sq = np.zeros(n_w)
        s_indep = np.zeros(n_w); s_n = np.zeros(n_w)
        t0 = time.time()
        for s in range(args.shuffles):
            wp = permute(rng)
            retention.append(float(np.mean(wp == data["wcode"])))

            def null_seed(wid, _s=s, _stream=stream):
                return np.random.default_rng(
                    [SEED, _stream, _s, _wallet_seed(wallets[wid])])

            flags = hardened_gate(dict(data, wcode=wp), params, null_seed, full=False)
            per_counts.append(int(flags["edge_persisted"].sum()))
            for a, v in flags["at_alpha"].items():
                alpha_counts.setdefault(a, []).append(v)
            cand_counts.append(int(flags["candidate"].sum()))
            overlap_counts.append(int((flags["edge_persisted"] & cset).sum()))
            mkt_ok.append(int(flags["edge_markets_ok"].sum()))

            om, ov, mm = flags["out_mean"], flags["out_var"], flags["m"]
            good = np.isfinite(om) & np.isfinite(ov) & (mm > 0)
            s_sum[good] += om[good]; s_sq[good] += om[good] ** 2
            s_indep[good] += ov[good] / mm[good]; s_n[good] += 1

            if s == 0:
                dt = time.time() - t0
                print(f"    [{dt:.1f}s/shuffle; ~{dt*args.shuffles/60:.0f} min for this bracket]",
                      flush=True)
            elif (s + 1) % 25 == 0:
                print(f"    [{s+1}/{args.shuffles}] persisted mean {np.mean(per_counts):.2f}  "
                      f"max {max(per_counts)}  [{(time.time()-t0)/60:.0f} min]", flush=True)

        per = np.array(per_counts, dtype=float)
        ovl = np.array(overlap_counts, dtype=float)
        D = design_effect(s_sq, s_sum, s_indep, args.shuffles)
        results[key] = {"label": label, "per": per, "cand": np.array(cand_counts, dtype=float),
                        "overlap": ovl, "retention": float(np.mean(retention)), "D": D,
                        "mkt_ok": float(np.mean(mkt_ok)), "seen": s_n == args.shuffles,
                        "alpha_counts": {a: np.array(v, dtype=float)
                                         for a, v in alpha_counts.items()}}

    # -- 5. Verdict ----------------------------------------------------------
    print(f"\n--- 5. Result: implied FDR of the certified {n_real} ---")
    print(f"  real edge_persisted (hardened gate): {n_real} of {n_cand} candidates "
          f"({n_real/max(n_cand,1):.1%})")
    for key, r in results.items():
        per, ovl = r["per"], r["overlap"]
        se = mc_error(per)
        novel = per - ovl
        print(f"\n  null {key} — {r['label']}")
        print(f"    label retention (share of bets keeping their own wallet): "
              f"{r['retention']:.3f}")
        print(f"    persisted: mean {per.mean():.2f} +/- {se:.2f} (MC s.e.)  "
              f"sd {per.std(ddof=1):.2f}  median {np.median(per):.0f}  "
              f"95th {np.percentile(per,95):.0f}  max {int(per.max())}")
        print(f"    candidates: mean {r['cand'].mean():.1f} (real {n_cand});  "
              f"markets_ok: mean {r['mkt_ok']:.1f} (real "
              f"{int(real['edge_markets_ok'].sum())})")
        print(f"    of those persisters, {ovl.mean():.2f} are ON AVERAGE members of the real "
              f"certified set")
        print(f"      (so {novel.mean():.2f} are wallets the real gate does NOT certify)")
        print(f"    P(null >= real) = {float((per >= n_real).mean()):.3f}")
        print(f"    IMPLIED FDR = {per.mean()/max(n_real,1):.1%} "
              f"(+/- {se/max(n_real,1):.1%} MC)")
        d = r["D"][r["seen"] & np.isfinite(r["D"])]
        dc = r["D"][r["seen"] & cset & np.isfinite(r["D"])]
        print(f"    design effect D over {d.size:,} wallets: median {np.median(d):.3f}  "
              f"10th {np.percentile(d,10):.3f}  90th {np.percentile(d,90):.3f}  "
              f"share D>1 {100*(d>1).mean():.1f}%")
        if dc.size:
            print(f"      over the certified {dc.size}: median {np.median(dc):.3f}  "
                  f"share D>1 {100*(dc>1).mean():.1f}%")
        print(f"    FDR curve — the same gate at tighter significance thresholds:")
        print(f"      {'alpha':>8}  {'real':>5}  {'null mean':>9}  {'FDR':>6}")
        floors = real["edge_magnitude_ok"] & real["edge_markets_ok"]
        with np.errstate(invalid="ignore"):
            for a, v in sorted(r["alpha_counts"].items(), reverse=True):
                rc = int((floors & (real["cluster_p"] < a)).sum())
                print(f"      {a:>8.4f}  {rc:>5d}  {v.mean():>9.2f}  "
                      f"{v.mean()/max(rc,1):>5.0%}")
        if np.median(d) < 1:
            print(f"    => D<1: null {key} is MORE constrained than independence. Per")
            print(f"       docs/blackswan_cluster_null.md it cannot arbitrate individual")
            print(f"       wallets; read its count only, and only with the retention above.")
        else:
            print(f"    => D>=1: null {key} is at least as wide as independence.")
    print("\n" + "=" * 84)
    print("B2 is the unconditional FDR (chance alone). B is the conditional one (chance,")
    print("holding each wallet's market footprint fixed) and is therefore the stricter of")
    print("the two. Read both, and read neither at the wallet level unless its D >= 1.")
    print("Check retention first: a null that hands a wallet most of its own record back")
    print("cannot manufacture a false discovery, and its ratio would measure freeze only.")
    print("=" * 84)


if __name__ == "__main__":
    main()
