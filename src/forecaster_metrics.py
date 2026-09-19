"""Per-(wallet, category) forecaster scoring for Project 2 — the metric stack
that turns the market-first discovery tape (`src/discover.py`) into a ranked
list of *copyable real-world forecasters*.

Read the design in `docs/project2_forecaster_discovery.md` §2 first. This module
implements three of the four metrics in that stack, per (wallet, category) cell:

  A — is the edge REAL?  Favorite-longshot-neutralized ("skill") edge, selected
      in-sample and validated out-of-sample with the same chronological split +
      one-sided significance test the Project-1 pipeline uses (`src/validate.py`).
      The one correctness change §2.0 demands is a **per-category price baseline**
      (a 0.75 NFL favorite is not a 0.75 crypto coin-flip), so residualization
      uses the right E[outcome | entry_price] curve for each bet's category.

  B — COPYABILITY (the metric that separates Project 2 from Project 1). Reusing
      `features._forward_price_arrays` with the exact leakage discipline (bounded
      window, resolution guard, NO resolved_value fallback), for each resolved BUY
      bet we measure the price a follower entering Δ seconds late would pay, and
      the favorite-longshot-neutral edge they'd still keep. Two components, kept
      separate: reachability (can a follower even get a fill?) and retained edge.

  C — economic MAGNITUDE. Significance ("is it real?") is not magnitude ("is it
      big enough to clear fees + slippage?"). On large samples the t-test passes
      trivial edges, so magnitude is an independent floor on the copyable edge,
      in residual-skill cents — NEVER $ profit / return %, the trap this whole
      engine exists to avoid.

(Metric D — recency / regime-change — is deferred; the winner record carries
`bets_per_month` and a last-N skill column as lightweight form signals, but the
change-point detector is a follow-on.)

Everything is a sortable column and additive gate flag — NOTHING is ever dropped
(CLAUDE.md). READ-ONLY: reads only `data/interim/discovery/`, writes only there.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import load_config
from src.discover import DISCOVERY_DIR, load_discovery_trades, load_tape_stats, market_tape_complete
from src.features import (
    PriceBaseline,
    _forward_price_arrays,
    expected_outcome,
    fit_price_baseline,
)
from src.validate import (
    DEFAULT_MIN_BETS_PER_HALF,
    DEFAULT_MIN_SKILL_EDGE,
    DEFAULT_OOS_ALPHA,
    DEFAULT_PRICE_BASELINE_BINS,
    _gated_mean,
    _oos_significance,
    split_in_sample_out_of_sample,
)

# Output (discovery dir only — Project 1's data/processed/ is another chat's).
FORECASTER_TABLE_PATH = DISCOVERY_DIR / "forecasters.parquet"

# Special baseline keys and the pooled all-categories cell label.
GLOBAL_KEY = "__global__"
REAL_WORLD_KEY = "__real_world__"
ALL_CELL = "all_real_world"

# Copyability latencies to report (seconds). 0 == own-edge ceiling (skill_edge_all).
COPY_DELTAS = (30, 60, 300)

# Defaults (overridable via config; see the copyability/magnitude blocks in the
# spec's §2.y config sketch). Kept as module constants so this runs even before
# another chat adds the config keys.
DEFAULT_FILL_BANDWIDTH_SEC = 60      # follower fill window (matches watch.py poll)
DEFAULT_MIN_REACHABILITY = 0.5       # gate: fraction of positions with followable liquidity
DEFAULT_MIN_COPYABLE_EDGE_CENTS = 2.0  # economic floor on copyable edge (cover trading cost)
DEFAULT_SLIPPAGE_BUFFER_CENTS = 1.0    # conservative impact+spread haircut (true value = live only)
# A copyable *forecaster* needs a multi-market track record — a single-market cell
# (breadth 1-2) is a concentrated one-off, not a repeatable skill, and is exactly
# the "mirage" HANDOFF flagged. Winners must clear this breadth floor.
DEFAULT_MIN_WINNER_BREADTH = 3
# Fraction of a cell's bets that must be on FULL (untruncated) market tapes for its
# skill edge to be trusted (§1.2/§1.4). The top-volume markets' tapes are truncated
# at the ~10.5k offset cap to a near-resolution slice, which inflates metric-A skill
# 10-50x (endgame entries scored as forecasts). `full_tape` is an additive data-
# quality gate: like the spec's §3.2 default filters it is a *view*, never row
# removal — every cell keeps its `frac_full_tape` column and the user can relax it.
DEFAULT_MIN_FRAC_FULL_TAPE = 0.8
# A sub-category needs at least this many resolved bets to fit its OWN FL baseline;
# below it, bets fall back to the real-world-wide curve, then global (§2.0).
DEFAULT_MIN_BETS_FOR_CATEGORY_BASELINE = 500
# Columns scoring actually reads. The on-disk tape also carries heavy text columns
# (question/slug/tx_hash/outcome) that scoring never uses but that dominate RAM at
# millions of rows — dropping them up front is what keeps the low-end box (3.7 GB)
# from OOMing once the /events full-tape pull grows the tape. timestamp is kept at
# full int precision (float32 can't represent a ~1.78e9 unix ts exactly, which would
# corrupt the searchsorted follower-price windows).
_SCORING_COLUMNS = ["wallet", "market_id", "token_id", "side", "entry_price", "size",
                    "timestamp", "resolved", "resolved_value", "category"]


def lean_tape(tape: pd.DataFrame) -> pd.DataFrame:
    """Project the tape to the columns scoring needs and downcast price/size to
    float32 — roughly halves peak RAM on the real (multi-million-row) tape without
    changing any result. No-op-safe on fixtures that already lack the text columns."""
    out = tape[[c for c in _SCORING_COLUMNS if c in tape.columns]].copy()
    for col in ("entry_price", "size", "resolved_value"):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    return out


# Cells below this many resolved BUY bets get only the cheap vectorized base
# metrics (skill_edge_all, sample_size, breadth) — the expensive per-cell OOS
# t-test + copyability follower-price computation is skipped for them. They are
# NEVER dropped (they stay in the table, sortable, with NaN gates) — a 1-4 bet
# cell simply cannot be validated (min_bets_per_half needs ~20 total) or copy-
# measured meaningfully, and scoring ~256k such cells is pure waste. Same spirit
# as validate.py gating thin samples to NaN while keeping the row.
DEFAULT_MIN_SCORE_BETS = 5


# ---------------------------------------------------------------------------
# Metric A foundation: per-category favorite-longshot baseline (§2.0)
# ---------------------------------------------------------------------------

def fit_category_baselines(
    resolved_bets: pd.DataFrame,
    n_bins: int = DEFAULT_PRICE_BASELINE_BINS,
    min_bets_for_own: int = DEFAULT_MIN_BETS_FOR_CATEGORY_BASELINE,
) -> dict[str, PriceBaseline]:
    """Fit E[resolved_value | entry_price] per category, with a fallback hierarchy.

    The market-wide curve is dominated by whatever category has the most bets, so
    residualizing (say) sports bets against it mis-measures skill (§2.0). Fit a
    curve per category where sample allows; a category too thin for its own curve
    falls back to the **real-world-wide** curve (all non-micro bets pooled), then
    to the **global** curve. Fallbacks are resolved here, so the returned map has
    an entry for every category present plus GLOBAL_KEY and REAL_WORLD_KEY."""
    global_bl = fit_price_baseline(resolved_bets, n_bins)
    if resolved_bets.empty:
        return {GLOBAL_KEY: global_bl, REAL_WORLD_KEY: global_bl}

    rw = resolved_bets[resolved_bets["category"] != "micro_crypto"]
    rw_bl = fit_price_baseline(rw, n_bins) if len(rw) else global_bl

    out: dict[str, PriceBaseline] = {GLOBAL_KEY: global_bl, REAL_WORLD_KEY: rw_bl}
    for cat, group in resolved_bets.groupby("category"):
        fallback = rw_bl if cat != "micro_crypto" else global_bl
        if len(group) >= min_bets_for_own:
            own = fit_price_baseline(group, n_bins)
            # fit_price_baseline degrades to empty edges on thin/degenerate data;
            # only keep an own curve that actually fitted bins.
            out[cat] = own if own.edges.size > 0 else fallback
        else:
            out[cat] = fallback
    return out


def category_baseline(baselines: dict[str, PriceBaseline], category: str) -> PriceBaseline:
    """The resolved baseline for a category (fallbacks already baked in at fit
    time). Unknown categories map to real-world-wide (non-micro) / global."""
    if category in baselines:
        return baselines[category]
    if category == "micro_crypto":
        return baselines[GLOBAL_KEY]
    return baselines.get(REAL_WORLD_KEY, baselines[GLOBAL_KEY])


def expected_outcome_by_category(
    baselines: dict[str, PriceBaseline], categories, prices
) -> np.ndarray:
    """Per-row structural E[outcome | entry_price], each row using its category's
    baseline. NaN prices stay NaN (so follower-skill on un-reachable bets is NaN,
    not a clamped guess)."""
    categories = np.asarray(categories)
    prices = np.asarray(prices, dtype=float)
    out = np.full(prices.shape, np.nan)
    for cat in pd.unique(categories):
        m = (categories == cat) & ~np.isnan(prices)
        if m.any():
            out[m] = expected_outcome(category_baseline(baselines, cat), prices[m])
    return out


# ---------------------------------------------------------------------------
# Metric B foundation: leakage-guarded follower price at latency Δ
# ---------------------------------------------------------------------------

def build_token_arrays(tape: pd.DataFrame, guard: float) -> dict:
    """Per token_id: timestamp-sorted (ts, price, size, wallet) arrays and the
    resolution-guard cutoff (drop the final `guard` fraction of the token's
    observed lifespan). Built once so the Δ sweep never re-parses the frame —
    same structure as scripts/audit_edge_decay.build_token_arrays."""
    tok = {}
    for token_id, g in tape.groupby("token_id"):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].to_numpy(dtype=float)
        t_min, t_max = ts.min(), ts.max()
        cutoff = t_max - guard * (t_max - t_min)
        tok[token_id] = (
            ts,
            g["entry_price"].to_numpy(dtype=float),
            g["size"].to_numpy(dtype=float),
            g["wallet"].to_numpy(),
            cutoff,
        )
    return tok


def follower_prices(bets: pd.DataFrame, tok: dict, delta: float, bandwidth: float) -> np.ndarray:
    """price_at(entry_ts + delta) per bet via the leakage-guarded forward proxy
    (reuses features._forward_price_arrays, so semantics are byte-identical to the
    production copy_window path). NaN where no *other-wallet* trade qualifies in
    (entry_ts+delta, entry_ts+delta+bandwidth] before the guard cutoff — i.e. the
    follower could not have entered at that latency.

    Each token's arrays are timestamp-sorted, so the window is located with two
    `np.searchsorted` calls and `_forward_price_arrays` is called on that tiny
    slice — O(log T) per bet instead of O(T). On mega-market tokens (~10k trades)
    the full-array mask per bet would dominate; the slice makes it identical but
    fast. (The delegated function re-applies its exact bound on the slice, so the
    result is unchanged — the tests pin that.)"""
    out = np.full(len(bets), np.nan)
    tids = bets["token_id"].to_numpy()
    wallets = bets["wallet"].to_numpy()
    entries = bets["timestamp"].to_numpy(dtype=float)
    for i in range(len(bets)):
        arrs = tok.get(tids[i])
        if arrs is None:
            continue
        ts, price, size, wal, cutoff = arrs
        lo_t = entries[i] + delta
        hi_t = min(lo_t + bandwidth, cutoff)
        if hi_t <= lo_t:
            continue  # window empties out past the resolution guard
        lo = int(np.searchsorted(ts, lo_t, side="right"))   # first ts > entry+Δ
        hi = int(np.searchsorted(ts, hi_t, side="right"))   # first ts > min(window end, cutoff)
        if hi <= lo:
            continue
        fv = _forward_price_arrays(
            ts[lo:hi], price[lo:hi], size[lo:hi], wal[lo:hi],
            lo_t, wallets[i], bandwidth, cutoff,
        )
        if fv is not None:
            out[i] = fv
    return out


# ---------------------------------------------------------------------------
# Per-(wallet, category) cell metrics
# ---------------------------------------------------------------------------

def _bets_per_month(timestamps: np.ndarray, n: int) -> float:
    """Frequency, shown but NEVER used as a quality proxy (CLAUDE.md / §2-A:
    frequency != sample size). NaN when the span is degenerate."""
    if n < 2:
        return float("nan")
    span_days = (float(np.max(timestamps)) - float(np.min(timestamps))) / 86400.0
    if span_days <= 0:
        return float("nan")
    return n / (span_days / 30.0)


def cell_metrics(cell: pd.DataFrame, category: str, cfg_vals: dict) -> dict:
    """All A/B/C metrics for one (wallet, category) cell of resolved BUY bets.

    `cell` must carry the precomputed per-bet columns `residual_edge` (skill edge
    vs. the bet's category baseline), `fskill_<Δ>` (follower skill edge at Δ) and
    `price_at_<Δ>`. Metric A uses the chronological in/out split from validate.py;
    B/C use the copyability columns. Gates are additive booleans — nothing here
    drops or reweights a row."""
    min_per_half = cfg_vals["min_per_half"]
    alpha = cfg_vals["alpha"]
    min_skill_edge = cfg_vals["min_skill_edge"]
    oos_split = cfg_vals["oos_split"]
    min_reach = cfg_vals["min_reachability"]
    min_copy_cents = cfg_vals["min_copyable_edge_cents"]
    slippage = cfg_vals["slippage_buffer_cents"]

    min_breadth = cfg_vals["min_winner_breadth"]

    ordered = cell.sort_values("timestamp")
    n = len(ordered)
    resid = ordered["residual_edge"].to_numpy(dtype=float)
    raw = (ordered["resolved_value"] - ordered["entry_price"]).to_numpy(dtype=float)
    breadth = int(ordered["market_id"].nunique())
    # Win rate = share of bets whose backed token resolved YES. A rate near 1.0
    # paired with a large skill edge is the near-resolution-slice artifact (endgame
    # entries on the winning side), NOT forecasting skill — surfaced as a diagnostic.
    win_rate = float((ordered["resolved_value"].to_numpy(dtype=float) == 1.0).mean()) if n else np.nan
    # Fraction of this cell's bets on FULL (untruncated) tapes — the near-resolution
    # bias only afflicts bets on cap-truncated tapes (§1.2). Defaults to 1.0 when the
    # tape-completeness column is absent (hand-built fixtures / legacy callers).
    if "tape_complete" in ordered.columns:
        tc = ordered["tape_complete"].to_numpy()
        frac_full_tape = float(np.mean(tc.astype(float))) if len(tc) else np.nan
    else:
        frac_full_tape = 1.0

    # --- A: skill edge, in/out-of-sample split + significance ---------------
    ins, out = split_in_sample_out_of_sample(ordered, oos_split)
    in_resid, in_n = _gated_mean(ins["residual_edge"], min_per_half)
    out_resid, out_n = _gated_mean(out["residual_edge"], min_per_half)
    out_p = _oos_significance(out["residual_edge"], min_per_half)

    is_candidate = (not np.isnan(in_resid)) and in_resid > 0
    significant = bool(
        is_candidate
        and (not np.isnan(out_resid)) and out_resid > 0
        and (not np.isnan(out_p)) and out_p < alpha
    )
    magnitude_ok = bool((not np.isnan(out_resid)) and out_resid >= min_skill_edge)
    persisted = bool(significant and magnitude_ok)

    # --- B: copyability (reachability + retained edge) at each Δ -------------
    # If the follower-price columns are absent (copyability skipped — the token-array
    # step is memory-heavy on the low-end box), B is left NaN and the cell simply
    # can't be a winner (a winner requires reachable copyable edge). Metric A above
    # is unaffected, so the de-biasing segmentation still holds.
    row: dict = {}
    reach60 = np.nan
    has_copy = f"price_at_{COPY_DELTAS[0]}" in ordered.columns
    for d in COPY_DELTAS:
        if not has_copy:
            row[f"reachability_{d}"] = np.nan
            row[f"n_reachable_{d}"] = 0
            row[f"copyable_skill_edge_{d}"] = np.nan
            continue
        pa = ordered[f"price_at_{d}"].to_numpy(dtype=float)
        fs = ordered[f"fskill_{d}"].to_numpy(dtype=float)
        reachable = ~np.isnan(pa)
        n_reach = int(reachable.sum())
        reach = n_reach / n if n else np.nan
        retained = float(np.nanmean(fs)) if n_reach > 0 else np.nan
        row[f"reachability_{d}"] = reach
        row[f"n_reachable_{d}"] = n_reach
        row[f"copyable_skill_edge_{d}"] = retained
        if d == 60:
            reach60 = reach

    # --- C: economic magnitude on the copyable (Δ=60) edge ------------------
    copy60 = row["copyable_skill_edge_60"]
    copy60_cents = copy60 * 100 if not np.isnan(copy60) else np.nan
    net_cents = copy60_cents - slippage if not np.isnan(copy60_cents) else np.nan
    clears_mag = bool((not np.isnan(net_cents)) and net_cents >= min_copy_cents)

    expected_per_signal = (
        reach60 * copy60 if (not np.isnan(reach60) and not np.isnan(copy60)) else np.nan
    )

    is_winner = bool(
        persisted
        and (not np.isnan(reach60)) and reach60 >= min_reach
        and clears_mag
        and breadth >= min_breadth
    )

    row.update(
        {
            "category": category,
            "sample_size": n,
            "breadth": breadth,
            "win_rate": win_rate,
            "frac_full_tape": frac_full_tape,
            "bets_per_month": _bets_per_month(ordered["timestamp"].to_numpy(dtype=float), n),
            "raw_edge_all": float(np.mean(raw)) if n else np.nan,
            "skill_edge_all": float(np.mean(resid)) if n else np.nan,
            "skill_edge_last20": float(np.mean(resid[-20:])) if n >= 5 else np.nan,
            "in_sample_skill_edge": in_resid,
            "out_of_sample_skill_edge": out_resid,
            "out_of_sample_skill_p": out_p,
            "in_sample_n": in_n,
            "out_of_sample_n": out_n,
            "edge_significant": significant,
            "edge_magnitude_ok": magnitude_ok,
            "edge_persisted": persisted,
            "own_skill_edge": float(np.mean(resid)) if n else np.nan,  # Δ=0 ceiling
            "copyable_skill_edge_60_cents": copy60_cents,
            "net_copyable_edge_cents": net_cents,
            "clears_magnitude_floor": clears_mag,
            "expected_copyable_edge_per_signal": expected_per_signal,
            "is_winner": is_winner,
        }
    )
    return row


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _config_values(cfg: dict) -> dict:
    scoring = cfg.get("scoring", {})
    copy = cfg.get("copyability", {})
    mag = cfg.get("magnitude", {})
    return {
        "n_bins": scoring.get("price_baseline_bins", DEFAULT_PRICE_BASELINE_BINS),
        "min_per_half": scoring.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF),
        "alpha": scoring.get("oos_significance_alpha", DEFAULT_OOS_ALPHA),
        "min_skill_edge": scoring.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE),
        "oos_split": scoring.get("oos_split", 0.5),
        "guard": scoring.get("fair_value_resolution_guard", 0.2),
        "bandwidth": copy.get("fill_bandwidth_sec", DEFAULT_FILL_BANDWIDTH_SEC),
        "min_reachability": copy.get("min_reachability", DEFAULT_MIN_REACHABILITY),
        "min_copyable_edge_cents": mag.get("min_copyable_edge_cents", DEFAULT_MIN_COPYABLE_EDGE_CENTS),
        "slippage_buffer_cents": mag.get("slippage_buffer_cents", DEFAULT_SLIPPAGE_BUFFER_CENTS),
        "min_winner_breadth": cfg.get("copyability", {}).get("min_winner_breadth", DEFAULT_MIN_WINNER_BREADTH),
        "min_frac_full_tape": scoring.get("min_frac_full_tape", DEFAULT_MIN_FRAC_FULL_TAPE),
        "min_bets_for_category_baseline": scoring.get(
            "min_bets_for_category_baseline", DEFAULT_MIN_BETS_FOR_CATEGORY_BASELINE
        ),
        "min_score_bets": scoring.get("min_forecaster_score_bets", DEFAULT_MIN_SCORE_BETS),
    }


# Columns cell_metrics fills that the cheap base table cannot — defaulted so base
# rows share the full schema (NaN numeric / False gate) and can be overwritten by
# full rows on the (wallet, category) key.
_EXPENSIVE_DEFAULTS = {
    "skill_edge_last20": float("nan"),
    "in_sample_skill_edge": float("nan"), "out_of_sample_skill_edge": float("nan"),
    "out_of_sample_skill_p": float("nan"), "in_sample_n": 0, "out_of_sample_n": 0,
    "edge_significant": False, "edge_magnitude_ok": False, "edge_persisted": False,
    "own_skill_edge": float("nan"), "copyable_skill_edge_60_cents": float("nan"),
    "net_copyable_edge_cents": float("nan"), "clears_magnitude_floor": False,
    "expected_copyable_edge_per_signal": float("nan"), "is_winner": False,
    "scored": False,
}
for _d in COPY_DELTAS:
    _EXPENSIVE_DEFAULTS[f"reachability_{_d}"] = float("nan")
    _EXPENSIVE_DEFAULTS[f"n_reachable_{_d}"] = 0
    _EXPENSIVE_DEFAULTS[f"copyable_skill_edge_{_d}"] = float("nan")


def _base_cell_table(bets: pd.DataFrame) -> pd.DataFrame:
    """Cheap, fully-vectorized base metrics for EVERY (wallet, category) cell plus
    each multi-category wallet's `all_real_world` aggregate. This is what keeps the
    'nothing is dropped' invariant affordable at ~300k cells: skill_edge_all,
    sample_size, breadth, bets_per_month via groupby.agg (no Python per-cell loop).
    The expensive OOS/copyability columns are defaulted here and filled later only
    for cells worth the per-cell computation."""
    # NB: `bets` is already a fresh copy owned by compute_forecaster_table (from
    # tape.loc[...].copy()), so we add columns in place rather than copying the
    # ~1.5M-row frame a second time — that redundant copy was ~500MB of peak RAM and
    # pushed the low-end box over its limit. The extra columns are harmless to the
    # caller (they ride along into sbets).
    bets["raw_edge"] = bets["resolved_value"].to_numpy(dtype=float) - bets["entry_price"].to_numpy(dtype=float)
    bets["is_win"] = (bets["resolved_value"].to_numpy(dtype=float) == 1.0).astype(float)
    if "tape_complete" not in bets.columns:
        bets["tape_complete"] = True
    bets["_tc"] = bets["tape_complete"].astype(float)

    def _agg(keys, cat_value=None):
        g = bets.groupby(keys)
        span_days = (g["timestamp"].max() - g["timestamp"].min()) / 86400.0
        n = g.size()
        out = pd.DataFrame({
            "sample_size": n,
            "breadth": g["market_id"].nunique(),
            "win_rate": g["is_win"].mean(),
            "frac_full_tape": g["_tc"].mean(),
            "raw_edge_all": g["raw_edge"].mean(),
            "skill_edge_all": g["residual_edge"].mean(),
            "bets_per_month": n / (span_days / 30.0).where(span_days > 0),
        })
        out = out.reset_index()
        if cat_value is not None:  # all-cell: keys was ['wallet'] -> attach category
            out["category"] = cat_value
        return out

    sub = _agg(["wallet", "category"])
    multi = bets.groupby("wallet")["category"].nunique()
    multi_wallets = multi[multi >= 2].index
    if len(multi_wallets):
        allcells = _agg(["wallet"], cat_value=ALL_CELL)
        allcells = allcells[allcells["wallet"].isin(multi_wallets)]
        base = pd.concat([sub, allcells], ignore_index=True)
    else:
        base = sub
    for col, default in _EXPENSIVE_DEFAULTS.items():
        base[col] = default
    return base


def compute_forecaster_table(
    tape: pd.DataFrame, cfg: dict, tape_stats: pd.DataFrame | None = None,
    compute_copyability: bool = True,
) -> pd.DataFrame:
    """The Project-2 winner-record table: one row per (wallet, category) cell plus
    a per-wallet `all_real_world` aggregate. Reads the discovery tape and returns a
    sortable frame. No row is ever dropped — but the expensive per-cell validation
    (OOS t-test) and copyability (follower-price) computation runs ONLY for cells
    with >= `min_forecaster_score_bets` resolved bets. Thinner cells keep their row
    with cheap base metrics and NaN gates (`scored=False`): a 1-4 bet cell can't be
    validated or copy-measured, and scoring hundreds of thousands of them is waste.

    `tape_stats` (per-market hit_cap record) segments trustworthy full-tape skill
    from near-resolution-inflated skill (§1.2): each cell gets a `frac_full_tape`
    column and a `full_tape` gate. When None, completeness falls back to the tape-
    length heuristic alone (fine for fixtures / when no stats file exists yet)."""
    cfg_vals = _config_values(cfg)
    if tape.empty:
        return pd.DataFrame()
    print(f"[forecaster] scoring {len(tape):,} tape rows ({tape['market_id'].nunique()} markets, "
          f"copyability={'on' if compute_copyability else 'off'})...", flush=True)

    bets = tape.loc[(tape["side"] == "BUY") & tape["resolved"].fillna(False)].copy()
    bets = bets[bets["resolved_value"].notna()]
    if bets.empty:
        return pd.DataFrame()

    # Per-market tape completeness -> per-bet `tape_complete` (§1.2/§1.4). Computed
    # from the FULL tape (all sides) so a market's true length/cap-hit is seen.
    comp = market_tape_complete(tape, tape_stats if tape_stats is not None else pd.DataFrame())
    complete_by_mkt = dict(zip(comp["market_id"], comp["tape_complete"]))
    bets["tape_complete"] = bets["market_id"].map(complete_by_mkt).fillna(True).astype(bool)

    # Per-category FL baselines (§2.0), then per-bet residual (skill) edge (cheap,
    # needed by every cell for skill_edge_all).
    baselines = fit_category_baselines(
        bets, cfg_vals["n_bins"], cfg_vals["min_bets_for_category_baseline"]
    )
    e_entry = expected_outcome_by_category(baselines, bets["category"], bets["entry_price"])
    bets["residual_edge"] = bets["resolved_value"].to_numpy(dtype=float) - e_entry

    if not compute_copyability:
        # metric-A-only: the full tape is unused past here (copyability, its only
        # remaining consumer, is skipped) — free ~700MB before the memory-heavy base
        # groupby so the low-end box doesn't OOM. Effective only when the caller holds
        # no other reference (main passes the tape inline for exactly this reason).
        del tape

    base = _base_cell_table(bets)

    # Full scoring only for the active subset: wallets with enough total resolved
    # bets that any of their cells (incl. the all-cell) could be validated/copyable.
    min_score = cfg_vals["min_score_bets"]
    wallet_counts = bets.groupby("wallet").size()
    score_wallets = wallet_counts[wallet_counts >= min_score].index
    sbets = bets[bets["wallet"].isin(score_wallets)].copy()

    full_rows = []
    if not sbets.empty:
        # Copyability follower prices at each Δ. The liquidity a follower enters
        # against is OTHER wallets' trades on the SAME token, so we need every trade
        # (all wallets, all sides) on the tokens the active wallets bet — but NOT
        # tokens no active wallet touched. Scoping build_token_arrays to those tokens
        # is exact (follower_prices only looks up sbets' tokens) and bounds peak
        # memory: on the small box the full-tape token dict OOMs once the tape grows
        # (the mid-volume /events pull ~doubled it), while active tokens are a subset.
        # `compute_copyability=False` skips the token arrays entirely (metric B ->
        # NaN, no winners) for a metric-A-only run when even the scoped dict won't fit
        # — the de-biasing segmentation (metric A + full_tape) is unaffected.
        if compute_copyability:
            active_tokens = set(sbets["token_id"].unique())
            tok = build_token_arrays(tape[tape["token_id"].isin(active_tokens)], cfg_vals["guard"])
            del tape  # full tape no longer needed; free ~700MB before the per-cell loop
            for d in COPY_DELTAS:
                pa = follower_prices(sbets, tok, d, cfg_vals["bandwidth"])
                sbets[f"price_at_{d}"] = pa
                e_at = expected_outcome_by_category(baselines, sbets["category"], pa)
                sbets[f"fskill_{d}"] = sbets["resolved_value"].to_numpy(dtype=float) - e_at
            del tok

        for (wallet, category), cell in sbets.groupby(["wallet", "category"]):
            rec = cell_metrics(cell, category, cfg_vals)
            rec["wallet"] = wallet
            rec["scored"] = True
            full_rows.append(rec)
        for wallet, cell in sbets.groupby("wallet"):
            if cell["category"].nunique() < 2:
                continue  # identical to the single sub-category cell; don't duplicate
            rec = cell_metrics(cell, ALL_CELL, cfg_vals)
            rec["wallet"] = wallet
            rec["scored"] = True
            full_rows.append(rec)

    if full_rows:
        full = pd.DataFrame(full_rows)
        # Full rows override base rows on the (wallet, category) key (keep last).
        table = pd.concat([base, full], ignore_index=True).drop_duplicates(
            subset=["wallet", "category"], keep="last"
        )
    else:
        table = base

    # Derive the full-tape data-quality gate once, from frac_full_tape (single source
    # of truth for base + scored rows). A `winner_full_tape` convenience flag marks
    # the DE-BIASED winner (§1.4): a winner whose skill is measured predominantly on
    # untruncated tapes, not the near-resolution slice. is_winner stays unchanged
    # (§3.2: the full-tape filter is a toggleable *view*, not a redefinition).
    min_frac = cfg_vals["min_frac_full_tape"]
    table["full_tape"] = table["frac_full_tape"].fillna(0.0) >= min_frac
    table["winner_full_tape"] = table["is_winner"].fillna(False).astype(bool) & table["full_tape"]

    front = ["wallet", "category", "scored", "sample_size", "breadth", "win_rate",
             "frac_full_tape", "full_tape", "bets_per_month", "skill_edge_all",
             "raw_edge_all", "out_of_sample_skill_edge", "edge_persisted",
             "reachability_60", "copyable_skill_edge_60", "copyable_skill_edge_60_cents",
             "clears_magnitude_floor", "is_winner", "winner_full_tape"]
    cols = front + [c for c in table.columns if c not in front]
    return table[cols].sort_values(
        ["winner_full_tape", "is_winner", "copyable_skill_edge_60", "skill_edge_all"],
        ascending=False, na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-(wallet, category) forecaster scoring (Project 2).")
    parser.add_argument("--min-cell-bets", type=int, default=1,
                        help="only keep cells with at least this many resolved bets in the table")
    parser.add_argument("--min-score-bets", type=int, default=None,
                        help="run the expensive OOS+copyability scoring only for cells with at "
                             "least this many bets (default 5); thinner cells keep cheap base "
                             "metrics only. Higher = faster/leaner (fewer cells fully scored).")
    parser.add_argument("--no-copyability", action="store_true",
                        help="skip metric B (copyability follower-prices) — the token-array step is "
                             "the memory peak; skipping it lets the metric-A de-biasing (skill + "
                             "full_tape) run on the low-end box. Metric B stays NaN (no winners).")
    args = parser.parse_args()

    cfg = load_config()
    if args.min_score_bets is not None:
        cfg.setdefault("scoring", {})["min_forecaster_score_bets"] = args.min_score_bets
    # Projected read: load ONLY the columns scoring needs so the heavy unused text
    # columns are never materialized (they peak RAM at the read step, before any
    # in-memory drop could help), and downcast price/size. Pass the tape INLINE (no
    # local reference held) so compute_forecaster_table can `del` it partway and free
    # ~700MB — keeps the low-end box from OOMing on the grown /events tape.
    table = compute_forecaster_table(
        lean_tape(load_discovery_trades(columns=_SCORING_COLUMNS)),
        cfg, tape_stats=load_tape_stats(),
        compute_copyability=not args.no_copyability,
    )
    if table.empty:
        print("[forecaster] no resolved discovery bets to score yet (run --resolve first).")
        return
    if args.min_cell_bets > 1:
        table = table[table["sample_size"] >= args.min_cell_bets].reset_index(drop=True)

    FORECASTER_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    from src.common import atomic_to_parquet
    atomic_to_parquet(table, FORECASTER_TABLE_PATH, compression="gzip")

    n_cells = len(table)
    n_persisted = int(table["edge_persisted"].sum())
    n_winners = int(table["is_winner"].sum())
    n_winners_ft = int(table["winner_full_tape"].sum())
    print(
        f"[forecaster] scored {n_cells} (wallet, category) cells "
        f"({table['wallet'].nunique()} wallets); {n_persisted} persist (skill A+C), "
        f"{n_winners} clear all gates (A+B+C); {n_winners_ft} of those are FULL-TAPE "
        f"(de-biased, §1.4) -> {FORECASTER_TABLE_PATH}"
    )


if __name__ == "__main__":
    main()
