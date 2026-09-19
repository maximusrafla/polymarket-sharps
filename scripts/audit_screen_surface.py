"""The SCREENING SURFACE: E[out-of-sample ROI | (success_rate, roi, n_bets)].

Every gate in this repo asks *"am I certain this wallet is skilled?"* — p<0.005,
cluster-robust bootstraps, BH-FDR. That is the right objective for identification
and it produced 26 certified wallets. It is NOT the objective for making money: a
portfolio does not need per-wallet certainty, it needs positive expected value in
aggregate. This script asks the money question instead — *which screening rule,
over the WHOLE population, maximizes realized out-of-sample dollars?* — and prices
the answer against three baselines and three nulls that could fake it.

METHOD (pre-registered in this docstring before any number was read):

  * Population = every wallet in `data/interim/bet_ledger.parquet` with resolved
    BUY bets. No certification filter, no rank filter, nothing dropped.
  * Each wallet's resolved BUY bets are sorted chronologically (STABLE sort, as in
    `src.validate.split_in_sample_out_of_sample`) and cut into THREE disjoint
    windows at floor(n/3) and floor(2n/3):
        W1 = the SCREEN window   (success_rate, roi, n_bets, mean entry price)
        W2 = the TARGET window   (realized ROI; the surface is fitted here)
        W3 = the HONEST window   (the chosen rule is priced here, once)
    The screen never sees its own target.
  * The surface is a 2-D grid of (W1 success_rate decile) x (W1 roi decile),
    stratified by a W1 depth threshold. The rule family searched is every
    axis-aligned rectangle on that grid at every depth threshold, subject to a
    minimum wallet count, and the objective is the region's pooled
    dollar-weighted W2 ROI. Reported raw AND residualized against
    E[outcome | entry_price] (`src.features.fit_price_baseline`).
  * Costs: gross, plus net at the paper trader's stressed taker-fee constant and
    a one-tick adverse-fill charge (`src/paper_trader.py`).
  * CIs resample CLUSTERS, never bets. The cluster unit is the resolution EVENT
    (`src.sports_events.resolution_event`), which collapses a championship's whole
    outright field to one draw and falls back to the market for everything else.

THE THREE NULLS (a 2-D surface fitted on noise ALWAYS shows a sweet spot):

  * null P — permute the wallet->(W2,W3) record assignment within W1-depth strata.
    This is H0 exactly: screen and target marginals are preserved bit-for-bit and
    only their LINKAGE is destroyed. Every target record stays a real wallet's
    real, intact, clustered record. Headline null.
  * null O — permute `resolved_value` within entry-price bins across the whole
    tape. Destroys skill and clustering; preserves the favorite-longshot curve.
    The permissive bracket (see docs/blackswan_cluster_null.md).
  * null C — `permute_cells_within_size` from `scripts/audit_persistence_fdr.py`:
    the strict cluster-preserving cell permutation. Each synthetic wallet keeps
    its bet count and cell-size profile; the markets it held are randomized.
  Each null re-runs the ENTIRE rectangle search, so what is reported is the null's
  BEST region — the max statistic, which is what the real optimum must beat.

THE CONTROLS THE WINNING SCREEN MUST THEN SURVIVE (section 11; see
docs/screening_surface.md for the written-up verdict):

  * PRICE — scored on the residual target E[outcome|entry_price], inside fixed
    entry-price bands, and against a "price-mimicking index" (the pool's own return
    reweighted to the screen's exact price mix). In this repo price level has
    explained apparent skill more often than skill has.
  * COST — the REAL per-category taker fee from docs/polymarket_mechanics.md
    (`rate x (1-p)` of stake, taker-only, verified on-chain), not the k=0.10 stress
    constant, plus one adverse tick. Reported gross AND net, per category.
  * COPYABILITY — where the dollars actually live. An edge inside the 5-minute
    micro-crypto tape is real and unreachable.
  * WALLET-DRAW RISK — a screen is a wallet-selection rule, so every headline gets a
    wallet-resampled CI beside its event-resampled one.
  * ITS OWN NULLS — null P on the screen (not just on the rectangle search) and a
    full rebuild of the certification gate on a null-O shuffled tape.

--real-world (added 2026-07-26, supersedes the unrestricted headline)

  The unrestricted run above answers a question nobody can trade. 91% of the
  ledger's resolved BUY bets are `micro_crypto` — the 5-minute BTC/ETH up-down
  tape — so an unrestricted population analysis IS a crypto analysis, and its
  winning screen turned out to be 95% crypto stake by construction: the gate's
  30-held-out-event floor can only be cleared by very-high-frequency wallets, and
  on this venue high-frequency means the fast tape. Filtering that screen's
  leftovers afterwards does NOT give a real-world answer; it gives the residue of
  a crypto-selected screen, which is a different object.

  `--real-world` therefore re-runs the whole thing inside the population
  `src.discover.is_real_world` admits — the same definition the sports arm, the
  slow-forecaster arm and the edge-decay arm use (`category != "micro_crypto"`) —
  with EVERY stage recalibrated rather than inherited:

    * the wallet universe is wallets with real-world resolved BUY bets;
    * `fit_price_baseline` is refitted on real-world bets, so the residual target
      is the real-world favorite-longshot curve, not the crypto one;
    * the depth ladder drops to `REALWORLD_DEPTH_THRESHOLDS` — a real-world wallet
      with 30 bets is a serious trader, a crypto bot does 30 bets in an hour, so
      reusing the unrestricted ladder would ask about strata that do not exist;
    * the certification gate's distinct-event floor is SWEPT over
      `REALWORLD_EVENT_FLOORS` instead of assumed at the production 30, and which
      floors are attainable at all is itself reported;
    * decile edges, depth strata, nulls, baselines and the price-mimicking index
      are all computed within the restricted population.

  Every control the unrestricted arm used is kept verbatim: screen on W1 / price
  on W3, the residual target, the price bands, the price-mimicking index, null P
  charging for the SELECTION step, equal-weight beside stake-weight, the real
  per-category taker fee plus a tick, and effEv/topW next to every number.

REPRODUCE:
  flock data/interim/.analysis.lock -c \
    'PYTHONPATH=. .venv/bin/python scripts/audit_screen_surface.py --real-world'
  # drop --real-world to reproduce the (superseded) unrestricted run verbatim
  # smoke: --shuffles-p 20 --shuffles-o 3 --shuffles-c 3 --boot 200

Read-only: opens the ledger and the ranked table, writes nothing but stdout.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_persistence_fdr import (  # noqa: E402
    _lexicographic_codes,
    build_cells,
    permute_cells_within_size,
)
from src.common import RANKED_WALLETS_PATH, load_config, load_ledger  # noqa: E402
from src.discover import classify_market, is_real_world  # noqa: E402
from src.features import expected_outcome, fit_price_baseline  # noqa: E402
from src.paper_trader import DEFAULT_TAKER_FEE_K_STRESS  # noqa: E402
from src.sports_events import resolution_event  # noqa: E402
from src.validate import _cluster_bootstrap_p  # noqa: E402

LEDGER_COLS = ["wallet", "market_id", "side", "entry_price", "resolved_value",
               "resolved", "size", "timestamp", "slug", "question"]

# --- pre-registered grid ---------------------------------------------------- #
DEPTH_THRESHOLDS = (5, 10, 20, 30, 50, 100, 200, 500, 1000)
DETAIL_DEPTHS = (5, 30, 100, 500)     # the four depths the per-depth tables print
# --- the REAL-WORLD ladder -------------------------------------------------- #
# 91% of the ledger's resolved BUY bets are `micro_crypto`, the 5-minute BTC/ETH
# up-down tape, and a wallet there places more bets in an hour than a real-world
# forecaster places in a year. Reusing the unrestricted ladder inside the
# real-world population would therefore ask a question with no answer: a >=1000-bet
# depth stratum simply does not exist there. These are the same shape of ladder
# (roughly log-spaced, floor at the smallest depth that can still be split into
# three windows and screened) recalibrated to real-world trading frequency.
REALWORLD_DEPTH_THRESHOLDS = (3, 5, 10, 20, 30, 50, 100, 200, 500)
REALWORLD_DETAIL_DEPTHS = (3, 10, 30, 100)
# The gate's distinct-held-out-event floor. Production is 30, which on this venue
# only high-frequency wallets can clear (see docs/screening_surface.md §10) — so in
# real-world mode it is swept rather than assumed.
REALWORLD_EVENT_FLOORS = (3, 5, 10, 20, 30)

# Rebound by `main()` under --real-world; every depth loop reads these, so the
# whole analysis recalibrates to the population being analysed rather than
# inheriting a frequency scale set by the crypto tape.
DEPTHS = DEPTH_THRESHOLDS
DETAILS = DETAIL_DEPTHS

N_BINS = 10           # deciles on each screen axis
MIN_WALLETS = 15      # a "region" must be a portfolio, not a lucky single wallet
MIN_TARGET_BETS = 5   # a wallet needs this many W2 bets to contribute a target
MIN_SPEND_SHARE = 0.01  # a region must carry >=1% of its pool's target-window
                        # stake. Without it the argmax of "maximize dollars" is
                        # won by a $7.5k corner of a $50M tape — technically the
                        # highest ROI, economically nothing, and pure noise.
SEARCH_SETTINGS = ((15, 0.0), (15, MIN_SPEND_SHARE), (30, MIN_SPEND_SHARE),
                   (60, MIN_SPEND_SHARE))
# The category `src.discover.is_real_world` excludes, and the reason it does:
# these markets resolve ~5 minutes after entry, so a copier has no window to act.
UNCOPYABLE_CATEGORY = "micro_crypto"


def set_population(real_world: bool) -> None:
    """Point the module's depth ladders at the population being analysed.

    A depth threshold is a statement about trading FREQUENCY, and frequency means
    something completely different on the two sides of `is_real_world`. Calling
    this before anything else in `main()` is what makes the real-world arm a
    recalibrated analysis rather than the unrestricted analysis with rows deleted."""
    global DEPTHS, DETAILS
    DEPTHS = REALWORLD_DEPTH_THRESHOLDS if real_world else DEPTH_THRESHOLDS
    DETAILS = REALWORLD_DETAIL_DEPTHS if real_world else DETAIL_DEPTHS

# --- the REAL taker fee, per docs/polymarket_mechanics.md §5 (committed 363f6df) --
# `fee = shares x rate x p x (1-p)`, TAKER ONLY, verified on-chain to 7 s.f. As a
# fraction of the stake (`shares x p`) that is exactly `rate x (1-p)` — biggest at
# LOW prices, which is the opposite of the intuition the repo carried for months.
# Rates are the live `feeSchedule.rate` cross-tab, mapped onto this repo's own
# `discover.classify_market` labels. Geopolitics (rate 0, `feesEnabled: false`) has
# no separate label here, so nothing is zero-rated: that makes every net number
# below a conservative (slightly pessimistic) bound, never an optimistic one.
FEE_RATE_BY_CATEGORY = {
    "micro_crypto": 0.07,   # crypto_fees_v2
    "crypto_event": 0.07,   # crypto_fees_v2
    "politics": 0.04,       # politics_fees / tech_fees
    "econ_macro": 0.05,     # economics_fees
    "culture": 0.05,        # culture_fees
    "other": 0.05,          # general_fees
}
DEFAULT_FEE_RATE = 0.05     # sports_fees_v2 and general_fees are both 0.05
# Entry-price bands for the bet-level price control. Fixed and interpretable, not
# quantiles, so the "fee is worst at low prices" effect is readable off the rows.
PRICE_BANDS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in tests/test_screen_surface.py)                    #
# --------------------------------------------------------------------------- #
def three_way_bounds(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cut each wallet's contiguous, time-sorted block into three windows.

    Returns `(b0, b1, b2, b3)`, absolute row offsets, with the cuts at
    `floor(n/3)` and `floor(2n/3)` — the same floor convention `src.validate`
    uses for its single 50/50 cut, extended to thirds. Windows are disjoint and
    jointly exhaustive; short wallets get empty windows rather than a special
    case."""
    counts = np.asarray(counts, dtype=np.int64)
    b0 = np.asarray(starts, dtype=np.int64)
    b1 = b0 + counts // 3
    b2 = b0 + (2 * counts) // 3
    b3 = b0 + counts
    return b0, b1, b2, b3


def segment_sums(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum `values[lo[i]:hi[i]]` for every i; empty segments give 0.0.

    Implemented with `np.add.reduceat` over a zero-padded copy — padding is what
    makes a segment ending at the array end (`hi == len(values)`) a legal index —
    and deliberately NOT with a global cumsum: the ledger's stakes span eight
    orders of magnitude, so a 4M-element cumulative sum bleeds real precision to
    cancellation. Odd (gap) slots of the reduceat result are discarded."""
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    out = np.zeros(lo.size, dtype=float)
    ok = hi > lo
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.any(ok):
        return out
    padded = np.append(values, 0.0)
    idx = np.empty(2 * int(ok.sum()), dtype=np.int64)
    idx[0::2] = lo[ok]
    idx[1::2] = hi[ok]
    out[ok] = np.add.reduceat(padded, idx)[0::2]
    return out


def bin_by_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Half-open bin index for each value, clamped into `[0, len(edges)-2]`."""
    if edges.size < 2:
        return np.zeros(np.asarray(values).size, dtype=np.int64)
    return np.clip(np.searchsorted(edges, values, side="right") - 1,
                   0, edges.size - 2).astype(np.int64)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bin edges, de-duplicated and opened at both ends so no value
    can fall outside. Degenerate inputs collapse to a single bin."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.array([-np.inf, np.inf])
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def grid_accumulate(row_bin, col_bin, weights, shape):
    """Sum `weights` into a `shape` grid at `(row_bin, col_bin)`."""
    out = np.zeros(shape, dtype=float)
    np.add.at(out, (np.asarray(row_bin), np.asarray(col_bin)), weights)
    return out


def rect_sums(grid: np.ndarray) -> np.ndarray:
    """2-D inclusive prefix sums padded with a zero row/column, so the sum over
    `[r0:r1, c0:c1]` is `P[r1,c1]-P[r0,c1]-P[r1,c0]+P[r0,c0]`."""
    p = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=float)
    p[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
    return p


def all_rectangles(n_rows: int, n_cols: int) -> tuple[np.ndarray, ...]:
    """Every axis-aligned rectangle `(r0, r1, c0, c1)`, r0<r1 and c0<c1 — the full
    rule family `success_rate in [a,b) AND roi in [c,d)`."""
    r0, r1 = np.triu_indices(n_rows + 1, k=1)
    c0, c1 = np.triu_indices(n_cols + 1, k=1)
    return (np.repeat(r0, c0.size), np.repeat(r1, c0.size),
            np.tile(c0, r0.size), np.tile(c1, r0.size))


def rect_window(prefix, r0, r1, c0, c1):
    return prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]


def best_rectangle(profit_grid, spend_grid, count_grid, rects, min_wallets,
                   min_spend=0.0):
    """Argmax pooled ROI over every rectangle clearing `min_wallets` wallets and
    `min_spend` of target-window stake.

    The spend floor is what makes this an answer to "maximize dollars" rather
    than "maximize a ratio": without it the winner is reliably a handful of
    micro-stake wallets whose ROI is a rounding error on a $50M tape.

    Returns `(best_roi, (r0, r1, c0, c1), n_wallets, profit, spend)`, or
    `(nan, None, 0, 0.0, 0.0)` when no rectangle qualifies."""
    r0, r1, c0, c1 = rects
    pp, ps, pc = rect_sums(profit_grid), rect_sums(spend_grid), rect_sums(count_grid)
    n = rect_window(pc, r0, r1, c0, c1)
    prof = rect_window(pp, r0, r1, c0, c1)
    spend = rect_window(ps, r0, r1, c0, c1)
    ok = (n >= min_wallets) & (spend > 0) & (spend >= min_spend)
    if not np.any(ok):
        return float("nan"), None, 0, 0.0, 0.0
    roi = np.full(n.shape, -np.inf)
    roi[ok] = prof[ok] / spend[ok]
    j = int(np.argmax(roi))
    return (float(roi[j]), (int(r0[j]), int(r1[j]), int(c0[j]), int(c1[j])),
            int(round(n[j])), float(prof[j]), float(spend[j]))


def pooled_roi(profit, spend) -> float:
    s = float(np.sum(spend))
    return float(np.sum(profit)) / s if s > 0 else float("nan")


def cluster_bootstrap_ci(profit_by_cluster, spend_by_cluster, rng, n_boot, alpha=0.05):
    """Percentile CI for pooled ROI, resampling CLUSTERS (resolution events) with
    replacement. Bets inside a cluster move together, so one resolution event is
    one draw — the repo's standard and the only honest unit here."""
    prof = np.asarray(profit_by_cluster, dtype=float)
    spend = np.asarray(spend_by_cluster, dtype=float)
    m = prof.size
    if m < 2:
        return float("nan"), float("nan")
    # CHUNKED, and int32. A population region here spans ~156k resolution events;
    # a single (n_boot x m) int64 index array would be 2.5 GB and OOM-kills this
    # 3.8 GB box (it did, twice, before this was chunked) — same defence as
    # src.validate._cluster_bootstrap_p.
    chunk = max(1, min(n_boot, 4_000_000 // m))
    parts, done = [], 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, m, size=(b, m), dtype=np.int32)
        num = prof[idx].sum(axis=1)
        den = spend[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            parts.append(np.where(den > 0, num / den, np.nan))
        done += b
        del idx, num, den
    draws = np.concatenate(parts)
    draws = draws[np.isfinite(draws)]
    if draws.size < 10:
        return float("nan"), float("nan")
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def fee_rate_for_categories(categories) -> np.ndarray:
    """Live taker-fee `rate` for each `discover.classify_market` label.

    Sports leagues all share `sports_fees_v2` = 0.05, which is also `general_fees`,
    so the default covers both and only the explicitly-listed labels differ."""
    return np.array([FEE_RATE_BY_CATEGORY.get(c, DEFAULT_FEE_RATE) for c in categories],
                    dtype=float)


def fee_fraction_of_stake(rate, price) -> np.ndarray:
    """`fee / stake` for a taker entry: `rate x (1 - p)`.

    Derived, not asserted: fee = shares x rate x p x (1-p) and stake = shares x p,
    so the shares cancel and the price cancels once. This is why the fee is a
    *larger* tax on cheap longshots than on expensive favourites, even though the
    absolute USDC fee peaks at p = 0.5."""
    return np.asarray(rate, dtype=float) * (1.0 - np.asarray(price, dtype=float))


def wallet_bootstrap_ci(profit_by_wallet, spend_by_wallet, rng, n_boot, alpha=0.05):
    """Percentile CI for pooled ROI, resampling WALLETS with replacement.

    The event-cluster CI answers "given these K wallets, how lucky were their
    markets?". It does NOT answer "how lucky was the draw of K wallets?" — and a
    screen is a wallet-selection rule, so wallet-draw risk is the risk the operator
    actually bears. With K=17 and a top wallet carrying a third of the stake the two
    intervals differ by an order of magnitude, so both are reported."""
    return cluster_bootstrap_ci(profit_by_wallet, spend_by_wallet, rng, n_boot, alpha)


def tick_for_price(price):
    """Polymarket's per-token tick grid, approximated from price alone.

    `src/paper_trader.py` is emphatic that the tick is a per-token property read
    off `/book`, NOT a 0.001 constant. This retrospective pass has no book, so it
    uses the venue's observed convention — 1c on the body, 0.1c on the tails where
    the fine grid is enabled — and every cost number derived from it is labelled
    an approximation."""
    price = np.asarray(price, dtype=float)
    return np.where((price >= 0.10) & (price <= 0.90), 0.01, 0.001)


# --------------------------------------------------------------------------- #
# Data preparation                                                             #
# --------------------------------------------------------------------------- #
def _market_meta(led, mcode_all, n_markets):
    """Per-market resolution-event code AND fee-category code, from the market's
    first-seen slug/question.

    Both are one Python pass over the ~349k distinct markets rather than the 4.07M
    bets, and they share that pass because the expensive part (recovering the slug
    and question strings behind the categorical codes) is identical."""
    first = np.zeros(n_markets, dtype=np.int64)
    uniq, first_idx = np.unique(mcode_all, return_index=True)
    first[uniq] = first_idx
    slug_cats = led["slug"].cat.categories.to_numpy()
    q_cats = led["question"].cat.categories.to_numpy()
    mkt_cats = led["market_id"].cat.categories.to_numpy()
    inv = np.argsort(mkt_cats, kind="stable")      # new code -> position in cats
    sc = led["slug"].cat.codes.to_numpy()[first]
    qc = led["question"].cat.codes.to_numpy()[first]
    keys, cats = [], []
    for i, (s, q) in enumerate(zip(sc, qc)):
        slug = slug_cats[s] if s >= 0 else None
        question = q_cats[q] if q >= 0 else None
        keys.append(resolution_event(slug, question, mkt_cats[inv[i]])[0])
        cats.append(classify_market(slug, question))
    ecodes = pd.factorize(pd.Index(keys))[0].astype(np.int64)
    cat_names, cat_codes = np.unique(np.array(cats, dtype=object), return_inverse=True)
    return ecodes, cat_codes.astype(np.int16), cat_names


def load_tape(cols, categorical, tape: str = "ledger"):
    """The analysis' population source. `ledger` is the shared bet ledger (the
    original, and the default — every prior run of this script used it). `realworld`
    is the isolated deep tape from `src/realworld_deepen.py`: a stratified,
    PERFORMANCE-BLIND probability sample of 2,500 discovery wallets, deepened to
    near-complete `/trades?user=` histories.

    The switch exists because the screens and the population are separate claims.
    Every prior real-world verdict here was measured on the ledger's real-world
    residue — 3,192 wallets, 98 of them with 1000+ bets — which is what a global
    firehose that is 82% 5-minute crypto leaves behind, not a sample of real-world
    traders. Nothing about the rules below changes; only which wallets they see."""
    if tape == "realworld":
        from src.realworld_deepen import load_deep_tape
        return load_deep_tape(columns=cols, categorical=categorical)
    return load_ledger(columns=cols, categorical=categorical)


def prepare(cfg, real_world: bool = False, tape: str = "ledger"):
    """Resolved BUY bets as flat numpy arrays, time-sorted within wallet, with
    per-wallet three-window boundaries and per-bet economics attached.

    `real_world=True` keeps only the bets `src.discover.is_real_world` admits —
    i.e. drops `micro_crypto`, the 5-minute up-down tape — and does so BEFORE
    anything downstream is computed. That ordering is the whole point: the price
    baseline, the wallet universe, the decile edges, the depth strata, the nulls
    and the baselines are then all fitted inside the real-world population rather
    than inherited from a tape that is 91% of the rows. The leftovers of a
    crypto-fitted screen and a real-world screen are different objects."""
    cols = list(LEDGER_COLS) + (["speed_bucket"] if real_world else [])
    led = load_tape(cols, ["market_id", "wallet", "side", "slug", "question"], tape)
    keep = (led["side"] == "BUY").to_numpy() & led["resolved"].to_numpy(dtype=bool)
    wallets = led["wallet"].cat.categories.to_numpy()
    wcode = led["wallet"].cat.codes.to_numpy().astype(np.int64)[keep]
    mcode_all = _lexicographic_codes(led["market_id"])
    n_markets = int(mcode_all.max()) + 1 if mcode_all.size else 1
    event_of_market, cat_of_market, cat_names = _market_meta(led, mcode_all, n_markets)
    mcode = mcode_all[keep]
    ecode = event_of_market[mcode]
    ccode = cat_of_market[mcode]
    ts = led["timestamp"].to_numpy(dtype=np.int64)[keep]
    price = led["entry_price"].to_numpy(dtype=float)[keep]
    value = led["resolved_value"].to_numpy(dtype=float)[keep]
    size = led["size"].to_numpy(dtype=float)[keep]
    speed = (led["speed_bucket"].to_numpy().astype("U8")[keep]
             if real_world else None)
    del led, mcode_all, keep, event_of_market, cat_of_market

    drop_report = None
    if real_world:
        rw_cat = np.array([is_real_world(str(c)) for c in cat_names], dtype=bool)
        rw = rw_cat[ccode]
        drop_report = {
            "bets_before": int(rw.size), "bets_after": int(rw.sum()),
            "stake_before": float(np.sum(size * price)),
            "stake_after": float(np.sum((size * price)[rw])),
            "markets_before": int(np.unique(mcode).size),
            "markets_after": int(np.unique(mcode[rw]).size),
            "wallets_before": int(np.unique(wcode).size),
            "wallets_after": int(np.unique(wcode[rw]).size),
            "dropped_categories": sorted(
                {str(cat_names[c]) for c in np.unique(ccode[~rw])}),
            "kept_categories": sorted(
                {str(cat_names[c]) for c in np.unique(ccode[rw])}),
            # copyability sanity check on what SURVIVED: `speed_bucket` is the
            # measured entry->resolution gap, independent of the slug classifier.
            "kept_speed": (dict(zip(*[list(x) for x in
                                      np.unique(speed[rw], return_counts=True)]))
                           if speed is not None else {}),
        }
        wcode, mcode, ecode, ccode = wcode[rw], mcode[rw], ecode[rw], ccode[rw]
        ts, price, value, size = ts[rw], price[rw], value[rw], size[rw]
        del rw, rw_cat, speed

    baseline = fit_price_baseline(
        pd.DataFrame({"entry_price": price, "resolved_value": value}),
        cfg["scoring"].get("price_baseline_bins", 20))
    expected = expected_outcome(baseline, price)

    order = np.lexsort((ts, wcode))
    d = {"wallets": wallets, "n_wallets": wallets.size,
         # counted as DISTINCT-PRESENT, not as a code ceiling: under --real-world
         # the codes still span the full ledger's range but most are now absent.
         "n_markets": int(np.unique(mcode).size),
         "n_events": int(np.unique(ecode).size),
         "cat_names": cat_names, "real_world": bool(real_world),
         "drop_report": drop_report,
         "fee_rate_by_code": fee_rate_for_categories(cat_names)}
    # int32 codes and an in-place reorder-then-drop keep peak RSS down: this box
    # has 3.8 GB and the ledger has OOM-killed analyses here before.
    for name, arr in (("wcode", wcode.astype(np.int32)), ("mcode", mcode.astype(np.int32)),
                      ("ecode", ecode.astype(np.int32)), ("ccode", ccode),
                      ("ts", ts), ("price", price), ("value", value), ("size", size),
                      ("expected", expected)):
        d[name] = arr[order]
    del wcode, mcode, ecode, ccode, ts, price, value, size, expected, order
    attach_economics(d, costs=True)
    # `size` and `expected` only ever fed the economics above; the per-bet skill
    # residual is what the certification gate needs, so keep that instead.
    d["resid_bet"] = d["value"] - d["expected"]
    del d["size"], d["expected"]
    gc.collect()
    attach_blocks(d)
    return d


def attach_economics(d, costs: bool):
    """Per-bet spend / profit, plus (when `costs`) the residualized profit and the
    two cost legs. Nulls skip the cost legs to keep peak RSS down."""
    d["spend"] = d["size"] * d["price"]
    d["profit"] = d["size"] * (d["value"] - d["price"])
    if costs:
        # Price-residualized: profit earned OVER the market's own calibration
        # curve at the price paid. Strips the favorite-longshot base rate that
        # makes a 0.70-favorite better look 70% skilled.
        d["resid_profit"] = d["size"] * (d["value"] - d["expected"])
        k = DEFAULT_TAKER_FEE_K_STRESS
        d["fee_stress"] = d["size"] * k * d["price"] * (1.0 - d["price"])
        d["tick_cost"] = d["size"] * tick_for_price(d["price"])
        # The REAL fee, per category, from docs/polymarket_mechanics.md. Note that
        # `fee_stress` above uses k=0.10 — above every live rate — so the pre-existing
        # net@stress columns bracket this from the pessimistic side.
        if "ccode" in d:
            rate = d["fee_rate_by_code"][d["ccode"]]
            d["fee_real"] = d["spend"] * fee_fraction_of_stake(rate, d["price"])


def attach_blocks(d):
    """Per-wallet block starts and the three window boundaries."""
    idx = np.arange(d["n_wallets"])
    starts = np.searchsorted(d["wcode"], idx, side="left")
    ends = np.searchsorted(d["wcode"], idx, side="right")
    counts = ends - starts
    b0, b1, b2, b3 = three_way_bounds(starts, counts)
    d.update(starts=starts, counts=counts, b0=b0, b1=b1, b2=b2, b3=b3)


def window_frame(d, lo, hi, costs: bool = True):
    """Per-wallet aggregates over each wallet's row range `[lo, hi)`."""
    n = (np.asarray(hi) - np.asarray(lo)).astype(np.int64)
    spend = segment_sums(d["spend"], lo, hi)
    profit = segment_sums(d["profit"], lo, hi)
    wins = segment_sums(d["value"], lo, hi)
    px = segment_sums(d["price"], lo, hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "n": n, "spend": spend, "profit": profit,
            "success_rate": np.where(n > 0, wins / np.maximum(n, 1), np.nan),
            "mean_price": np.where(n > 0, px / np.maximum(n, 1), np.nan),
            "roi": np.where(spend > 0, profit / np.maximum(spend, 1e-12), np.nan),
        }
    if costs:
        out["resid_profit"] = segment_sums(d["resid_profit"], lo, hi)
        out["fee_stress"] = segment_sums(d["fee_stress"], lo, hi)
        out["tick_cost"] = segment_sums(d["tick_cost"], lo, hi)
        if "fee_real" in d:
            # profit after the REAL per-category taker fee and one adverse tick,
            # carried per wallet so a PER-WALLET statistic (median, equal weight)
            # can be charged costs too. Charging costs only to the pooled number
            # would flatter exactly the equal-weight view a copier actually runs.
            out["fee_real"] = segment_sums(d["fee_real"], lo, hi)
            out["net_profit"] = out["profit"] - out["fee_real"] - out["tick_cost"]
    return out


# --------------------------------------------------------------------------- #
# The surface                                                                  #
# --------------------------------------------------------------------------- #
def search(w1, target, sr_bin, roi_bin, rects, depth_thresholds, min_wallets,
           eligible, profit_key="profit", n_bins=N_BINS, min_spend_share=0.0):
    """Rectangle search at every depth threshold, keyed on pooled target ROI.
    Returns `(best, per_depth)`."""
    per_depth, best = [], None
    for thr in depth_thresholds:
        sel = eligible & (w1["n"] >= thr)
        k_pool = int(sel.sum())
        if k_pool < min_wallets:
            per_depth.append({"depth": thr, "n_pool": k_pool, "roi": float("nan"),
                              "rect": None, "k": 0, "profit": 0.0, "spend": 0.0})
            continue
        rb, cb = sr_bin[sel], roi_bin[sel]
        pg = grid_accumulate(rb, cb, target[profit_key][sel], (n_bins, n_bins))
        sg = grid_accumulate(rb, cb, target["spend"][sel], (n_bins, n_bins))
        cg = grid_accumulate(rb, cb, np.ones(k_pool), (n_bins, n_bins))
        roi, rect, k, prof, spend = best_rectangle(
            pg, sg, cg, rects, min_wallets, min_spend_share * float(sg.sum()))
        rec = {"depth": thr, "n_pool": k_pool, "roi": roi, "rect": rect,
               "k": k, "profit": prof, "spend": spend}
        per_depth.append(rec)
        if rect is not None and (best is None or roi > best["roi"]):
            best = rec
    return best, per_depth


def rule_mask(w1, sr_bin, roi_bin, eligible, depth, rect):
    r0, r1, c0, c1 = rect
    return (eligible & (w1["n"] >= depth)
            & (sr_bin >= r0) & (sr_bin < r1) & (roi_bin >= c0) & (roi_bin < c1))


def surface_pass(d, costs: bool = True, n_bins: int = N_BINS):
    """W1/W2/W3 frames, screen bins and eligibility for one (real or null)
    dataset. Bins are always refit on the dataset being analysed, so a null gets
    exactly the procedure the real arm got."""
    w1 = window_frame(d, d["b0"], d["b1"], costs=costs)
    w2 = window_frame(d, d["b1"], d["b2"], costs=costs)
    w3 = window_frame(d, d["b2"], d["b3"], costs=costs)
    screenable = (w1["n"] >= DEPTHS[0]) & (w1["spend"] > 0)
    eligible = screenable & (w2["n"] >= MIN_TARGET_BETS)
    sr_edges = quantile_edges(w1["success_rate"][eligible], n_bins)
    roi_edges = quantile_edges(w1["roi"][eligible], n_bins)
    sr_bin = bin_by_edges(np.nan_to_num(w1["success_rate"], nan=0.0), sr_edges)
    roi_bin = bin_by_edges(np.nan_to_num(w1["roi"], nan=0.0), roi_edges)
    return dict(w1=w1, w2=w2, w3=w3, screenable=screenable, eligible=eligible,
                sr_bin=sr_bin, roi_bin=roi_bin, sr_edges=sr_edges, roi_edges=roi_edges)


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def region_report(d, mask, lo, hi, rng, n_boot, label):
    """Every dollar statistic for a selected set of wallets over one window."""
    wl = np.flatnonzero(mask)
    rows = (np.concatenate([np.arange(lo[i], hi[i]) for i in wl])
            if wl.size else np.zeros(0, dtype=np.int64))
    spend, profit = d["spend"][rows], d["profit"][rows]
    tot = float(spend.sum())
    out = {"label": label, "wallets": int(wl.size), "bets": int(rows.size),
           "spend": tot, "profit": float(profit.sum()),
           "roi": pooled_roi(profit, spend),
           "roi_resid": pooled_roi(d["resid_profit"][rows], spend),
           # net@tick = today's venue (the taker fee constant k defaults to 0)
           # minus one adverse tick; net@stress adds the unpinned fee at its
           # stressed k. The truth is bracketed, not asserted.
           "roi_net_tick": pooled_roi(profit - d["tick_cost"][rows], spend),
           "roi_net": pooled_roi(profit - d["fee_stress"][rows] - d["tick_cost"][rows], spend),
           # net of the REAL per-category taker fee plus one adverse tick
           "roi_net_real": pooled_roi(profit - d["fee_real"][rows] - d["tick_cost"][rows],
                                      spend) if "fee_real" in d else float("nan"),
           "events": int(np.unique(d["ecode"][rows]).size) if rows.size else 0}
    if rows.size == 0 or tot <= 0:
        out.update(roi_wallet_eq=np.nan, roi_flat_stake=np.nan, top_wallet_spend_share=np.nan,
                   ci_lo=np.nan, ci_hi=np.nan, eff_events=np.nan, roi_net_tick=np.nan,
                   roi_net_real=np.nan)
        return out
    w_spend = segment_sums(d["spend"], lo[wl], hi[wl])
    w_prof = segment_sums(d["profit"], lo[wl], hi[wl])
    ok = w_spend > 0
    out["roi_wallet_eq"] = float(np.mean(w_prof[ok] / w_spend[ok])) if ok.any() else np.nan
    out["roi_flat_stake"] = float(np.mean((d["value"][rows] - d["price"][rows])
                                          / d["price"][rows]))
    out["top_wallet_spend_share"] = float(w_spend.max() / tot)
    uniq, invx = np.unique(d["ecode"][rows], return_inverse=True)
    pc = np.bincount(invx, weights=profit, minlength=uniq.size)
    sc = np.bincount(invx, weights=spend, minlength=uniq.size)
    out["ci_lo"], out["ci_hi"] = cluster_bootstrap_ci(pc, sc, rng, n_boot)
    share = sc / sc.sum()
    out["eff_events"] = float(1.0 / np.sum(share ** 2))
    return out


def rows_for(mask, lo, hi):
    """Every ledger row belonging to the masked wallets inside one window."""
    wl = np.flatnonzero(mask)
    if wl.size == 0:
        return wl, np.zeros(0, dtype=np.int64)
    return wl, np.concatenate([np.arange(lo[i], hi[i]) for i in wl])


def priced_screen(d, mask, lo, hi, rng, n_boot, label=""):
    """The three numbers a screen actually has to survive, each with a CI.

    * GROSS pooled ROI — what the previous pass reported.
    * RESIDUAL pooled ROI — the same dollars scored against E[outcome|entry_price],
      i.e. with the favorite-longshot base rate removed. In this repo price level has
      explained apparent skill more often than skill has, so a screen that only
      survives gross has not survived.
    * NET pooled ROI — gross minus the real per-category taker fee (rate x (1-p) of
      stake) and one adverse tick.

    Two CIs, because a screen carries two independent risks: the markets its wallets
    happened to land in (resample EVENTS) and the wallets the rule happened to pick
    (resample WALLETS)."""
    return priced_rows(d, rows_for(mask, lo, hi)[1], rng, n_boot, label)


def priced_rows(d, rows, rng, n_boot, label=""):
    """`priced_screen` over an explicit set of ledger rows.

    Taking rows rather than a wallet mask is what lets the same statistics be run on
    a SLICE of a screen's dollars — one category, one price band — which is how the
    copyability question below gets asked without inventing a second code path."""
    rows = np.asarray(rows, dtype=np.int64)
    wcodes = np.unique(d["wcode"][rows]) if rows.size else np.zeros(0)
    out = {"label": label, "wallets": int(wcodes.size), "bets": int(rows.size)}
    if rows.size == 0:
        return {**out, "spend": 0.0, "events": 0, **{k: np.nan for k in
                ("roi", "ci_lo", "ci_hi", "wci_lo", "wci_hi", "resid", "rci_lo",
                 "rci_hi", "net", "nci_lo", "nci_hi", "resid_net", "fee_share",
                 "tick_share", "eff_events", "eqw", "eqw_resid", "eqw_net",
                 "eqw_ci_lo", "eqw_ci_hi", "top_wallet", "loo_lo", "loo_hi")}}
    spend = d["spend"][rows]
    profit = d["profit"][rows]
    resid = d["resid_profit"][rows]
    cost = d["fee_real"][rows] + d["tick_cost"][rows]
    tot = float(spend.sum())
    out["spend"] = tot
    out["roi"] = pooled_roi(profit, spend)
    out["resid"] = pooled_roi(resid, spend)
    out["net"] = pooled_roi(profit - cost, spend)
    out["resid_net"] = pooled_roi(resid - cost, spend)
    out["fee_share"] = float(d["fee_real"][rows].sum() / tot) if tot > 0 else np.nan
    out["tick_share"] = float(d["tick_cost"][rows].sum() / tot) if tot > 0 else np.nan
    uniq, invx = np.unique(d["ecode"][rows], return_inverse=True)
    ev_s = np.bincount(invx, weights=spend, minlength=uniq.size)
    out["events"] = int(uniq.size)
    out["eff_events"] = float(1.0 / np.sum((ev_s / ev_s.sum()) ** 2))
    for key, vals, tag in (("roi", profit, ""), ("resid", resid, "r"),
                           ("net", profit - cost, "n")):
        ev_p = np.bincount(invx, weights=vals, minlength=uniq.size)
        cl, ch = cluster_bootstrap_ci(ev_p, ev_s, rng, n_boot)
        out[f"{tag}ci_lo"], out[f"{tag}ci_hi"] = cl, ch
    _, winv = np.unique(d["wcode"][rows], return_inverse=True)
    nw = int(wcodes.size)
    w_spend = np.bincount(winv, weights=spend, minlength=nw)
    w_prof = np.bincount(winv, weights=profit, minlength=nw)
    w_res = np.bincount(winv, weights=resid, minlength=nw)
    w_cost = np.bincount(winv, weights=cost, minlength=nw)
    out["wci_lo"], out["wci_hi"] = wallet_bootstrap_ci(w_prof, w_spend, rng, n_boot)
    ok = w_spend > 0
    out["eqw"] = float(np.mean(w_prof[ok] / w_spend[ok])) if ok.any() else np.nan
    out["eqw_resid"] = float(np.mean(w_res[ok] / w_spend[ok])) if ok.any() else np.nan
    out["eqw_net"] = (float(np.mean((w_prof[ok] - w_cost[ok]) / w_spend[ok]))
                      if ok.any() else np.nan)
    # equal weight is a MEAN OF WALLETS, so its only honest CI resamples wallets
    if int(ok.sum()) >= 2:
        r = w_prof[ok] / w_spend[ok]
        idx = rng.integers(0, r.size, size=(n_boot, r.size), dtype=np.int32)
        draws = r[idx].mean(axis=1)
        out["eqw_ci_lo"] = float(np.quantile(draws, 0.025))
        out["eqw_ci_hi"] = float(np.quantile(draws, 0.975))
    else:
        out["eqw_ci_lo"] = out["eqw_ci_hi"] = np.nan
    out["top_wallet"] = float(w_spend.max() / tot)
    # leave-one-wallet-out: the crudest possible check that K wallets are a portfolio
    # and not one wallet with 16 passengers.
    if nw > 1:
        loo = (w_prof.sum() - w_prof) / np.maximum(w_spend.sum() - w_spend, 1e-12)
        out["loo_lo"], out["loo_hi"] = float(loo.min()), float(loo.max())
    else:
        out["loo_lo"] = out["loo_hi"] = np.nan
    return out


def fmt_priced(r):
    return (f"{r['label']:<34} K={r['wallets']:>4} $={r['spend']:>12,.0f} "
            f"gross={r['roi']:>+7.4f}[{r['ci_lo']:>+.4f},{r['ci_hi']:>+.4f}] "
            f"resid={r['resid']:>+7.4f}[{r['rci_lo']:>+.4f},{r['rci_hi']:>+.4f}] "
            f"net={r['net']:>+7.4f}[{r['nci_lo']:>+.4f},{r['nci_hi']:>+.4f}] "
            f"walletCI=[{r['wci_lo']:>+.4f},{r['wci_hi']:>+.4f}] "
            f"effEv={r['eff_events']:>7.1f} topW={r['top_wallet']:>4.2f}")


def price_band_table(d, mask, lo, hi, pool_mask, rng, n_boot, bands=PRICE_BANDS):
    """The screen's dollars sliced by ENTRY PRICE, against the same slice of the
    whole screenable pool.

    This is the price control done at the bet level rather than by residualizing:
    if the screen is really just a price habit, then inside a price band it should
    earn what everyone else earns there, and the whole gap should live in the
    band MIX. The 'matched' column reprices the screen's own band weights at the
    pool's band returns — i.e. what a price-mimicking index fund would have made."""
    _, rows = rows_for(mask, lo, hi)
    _, prows = rows_for(pool_mask, lo, hi)
    edges = np.asarray(bands, dtype=float)
    b = bin_by_edges(d["price"][rows], edges)
    pb = bin_by_edges(d["price"][prows], edges)
    out, matched_num, matched_den = [], 0.0, 0.0
    for k in range(edges.size - 1):
        m, pm = b == k, pb == k
        s = float(d["spend"][rows][m].sum())
        ps = float(d["spend"][prows][pm].sum())
        pool_roi = (float(d["profit"][prows][pm].sum()) / ps) if ps > 0 else np.nan
        if s > 0 and np.isfinite(pool_roi):
            matched_num += s * pool_roi
            matched_den += s
        rec = {"lo": edges[k], "hi": edges[k + 1], "spend": s, "pool_spend": ps,
               "bets": int(m.sum()), "pool_roi": pool_roi}
        if s > 0:
            sp, pr = d["spend"][rows][m], d["profit"][rows][m]
            cost = d["fee_real"][rows][m] + d["tick_cost"][rows][m]
            rec["roi"] = pooled_roi(pr, sp)
            rec["resid"] = pooled_roi(d["resid_profit"][rows][m], sp)
            rec["net"] = pooled_roi(pr - cost, sp)
            rec["fee_share"] = float(d["fee_real"][rows][m].sum() / s)
            u, ix = np.unique(d["ecode"][rows][m], return_inverse=True)
            rec["ci_lo"], rec["ci_hi"] = cluster_bootstrap_ci(
                np.bincount(ix, weights=pr, minlength=u.size),
                np.bincount(ix, weights=sp, minlength=u.size), rng, n_boot)
            rec["events"] = int(u.size)
        else:
            rec.update(roi=np.nan, resid=np.nan, net=np.nan, fee_share=np.nan,
                       ci_lo=np.nan, ci_hi=np.nan, events=0)
        out.append(rec)
    matched = matched_num / matched_den if matched_den > 0 else np.nan
    return out, matched


def category_table(d, mask, lo, hi):
    """Per-category stake, gross ROI and the fee it would actually pay today."""
    _, rows = rows_for(mask, lo, hi)
    out = []
    if rows.size == 0:
        return out
    cc = d["ccode"][rows]
    for c in np.unique(cc):
        m = cc == c
        s = float(d["spend"][rows][m].sum())
        if s <= 0:
            continue
        pr = d["profit"][rows][m]
        cost = d["fee_real"][rows][m] + d["tick_cost"][rows][m]
        out.append({"cat": str(d["cat_names"][c]), "rate": float(d["fee_rate_by_code"][c]),
                    "spend": s, "bets": int(m.sum()),
                    "roi": pooled_roi(pr, d["spend"][rows][m]),
                    "resid": pooled_roi(d["resid_profit"][rows][m], d["spend"][rows][m]),
                    "net": pooled_roi(pr - cost, d["spend"][rows][m]),
                    "fee_share": float(d["fee_real"][rows][m].sum() / s),
                    "mean_price": float(np.average(d["price"][rows][m],
                                                   weights=d["spend"][rows][m]))})
    return sorted(out, key=lambda r: -r["spend"])


def fmt_region(r):
    return (f"{r['label']:<38} K={r['wallets']:>5} bets={r['bets']:>8} "
            f"ev={r['events']:>6}(eff{r['eff_events']:>8.1f}) "
            f"spend=${r['spend']:>13,.0f} profit=${r['profit']:>12,.0f} "
            f"ROI={r['roi']:>+8.4f}[{r['ci_lo']:>+.4f},{r['ci_hi']:>+.4f}] "
            f"resid={r['roi_resid']:>+8.4f} net@tick={r['roi_net_tick']:>+8.4f} "
            f"net@stress={r['roi_net']:>+8.4f} "
            f"eqw={r['roi_wallet_eq']:>+8.4f} flat={r['roi_flat_stake']:>+9.4f} "
            f"topW={r['top_wallet_spend_share']:>5.2f}")


def report_null(name, draws, real_best):
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        print(f"  null {name}: no usable draws")
        return
    p = (float((draws >= real_best).sum()) + 1.0) / (draws.size + 1.0)
    print(f"  null {name}")
    print(f"    best-region ROI over {draws.size} shuffles: median {np.median(draws):+.4f}, "
          f"p90 {np.quantile(draws, 0.90):+.4f}, max {draws.max():+.4f}   |   "
          f"REAL {real_best:+.4f}  ->  p = {p:.4f}")


# --------------------------------------------------------------------------- #
# Nulls                                                                        #
# --------------------------------------------------------------------------- #
def depth_strata_codes(n_bets):
    """Log-spaced W1-depth strata — the blocks null P permutes within, so a deep
    wallet's target can only be swapped for another deep wallet's."""
    return bin_by_edges(np.log10(np.maximum(np.asarray(n_bets, dtype=float), 1)),
                        np.array([-np.inf, 1, 1.3, 1.7, 2, 2.3, 2.7, 3, np.inf]))


def null_p_permutations(sp, rng, n_shuffles):
    """Yield index permutations that re-attach W2/W3 records to W1 screens within
    depth strata, over the eligible population only — so eligibility, and hence
    the searched pool, is bit-identical to the real arm. Marginals of BOTH the
    screen and the target are preserved exactly; only the linkage dies."""
    strata = depth_strata_codes(sp["w1"]["n"])
    pool = np.flatnonzero(sp["eligible"])
    groups = [pool[strata[pool] == s] for s in np.unique(strata[pool])]
    base = np.arange(sp["w1"]["n"].size)
    for _ in range(n_shuffles):
        perm = base.copy()
        for g in groups:
            perm[g] = g[rng.permutation(g.size)]
        yield perm


def null_p_draws(sp, rects, rng, n_shuffles, min_wallets=MIN_WALLETS,
                 min_spend_share=0.0):
    """Null P best-region ROI, one draw per shuffle, under the SAME search."""
    w1, w2, elig = sp["w1"], sp["w2"], sp["eligible"]
    out = np.empty(n_shuffles)
    for s, perm in enumerate(null_p_permutations(sp, rng, n_shuffles)):
        shuffled = {"profit": w2["profit"][perm], "spend": w2["spend"][perm]}
        best, _ = search(w1, shuffled, sp["sr_bin"], sp["roi_bin"], rects,
                         DEPTHS, min_wallets, elig,
                         min_spend_share=min_spend_share)
        out[s] = best["roi"] if best else np.nan
    return out


# --------------------------------------------------------------------------- #
# Pre-specified 1-D screens (no argmax, so no search burden to correct for)     #
# --------------------------------------------------------------------------- #
def certify_on_window(d, lo, hi, pool, alpha_list, min_edge, min_events, n_boot, seed):
    """Run this repo's own certification gate INSIDE an arbitrary window.

    This is what makes the certainty-vs-dollars comparison fair. The certified 26
    in `ranked_wallets.parquet` were selected using a 50/50 split of each wallet's
    FULL history — which includes W2 and W3 — so their W2/W3 dollars are not
    out-of-sample for them and cannot be compared with a screen fitted on W1. Here
    the identical gate (chronological candidacy, held-out residual skill edge,
    magnitude floor, distinct-event floor, market/event-block bootstrap
    significance) is run on W1 alone, so both screens see exactly the same
    information and W2/W3 are honestly held out for both.

    Returns `(passes, diagnostics)` where `passes[alpha]` is a boolean mask."""
    resid = d["resid_bet"]
    idx = np.flatnonzero(pool)
    n_w = d["n_wallets"]
    held_edge = np.full(n_w, np.nan)
    cand = np.zeros(n_w, dtype=bool)
    n_ev = np.zeros(n_w, dtype=np.int64)
    pval = np.full(n_w, np.nan)
    for i in idx:
        a, b = int(lo[i]), int(hi[i])
        n = b - a
        if n < 4:
            continue
        cut = a + n // 2
        early, late = resid[a:cut], resid[cut:b]
        if early.size == 0 or late.size == 0:
            continue
        cand[i] = early.mean() > 0
        held_edge[i] = late.mean()
        ev = d["ecode"][cut:b]
        n_ev[i] = np.unique(ev).size
        # The bootstrap runs ONLY for wallets that already clear the magnitude and
        # event floors. `passes` is `base & (p < alpha)` and `base` contains both
        # floors, so this cannot change any result — it just stops spending 2000
        # resamples on wallets whose verdict is already decided, which is what makes
        # re-running this whole gate inside a null affordable.
        if (cand[i] and held_edge[i] >= min_edge and n_ev[i] >= max(min_events, 2)):
            pval[i] = _cluster_bootstrap_p(late, ev, np.random.default_rng(seed + int(i)),
                                           n_boot)
    diag = {"cand": cand, "held_edge": held_edge, "n_ev": n_ev, "p": pval,
            "min_events_run": min_events}
    return {a: gate_passes(diag, min_edge, min_events, a) for a in alpha_list}, diag


def gate_passes(diag, min_edge, min_events, alpha):
    """The gate's verdict recovered from `certify_on_window`'s diagnostics.

    Splitting this out is what makes the EVENT FLOOR sweepable. The floor is not a
    statistical choice, it is a statement about how often the population trades:
    30 held-out resolution events inside half of a screening window is routine for
    a 5-minute-tape scalper and near-impossible for a real-world forecaster. The
    bootstrap p is computed once, at the lowest floor in the ladder, and every
    higher floor is a strictly tighter subset of it — so this is exactly what
    `certify_on_window` would return, without paying for the resamples again.

    Requires `diag` to come from a run whose `min_events` was <= this one, since a
    higher-floored run never computed `p` for the wallets a lower floor admits."""
    if min_events < diag.get("min_events_run", min_events):
        raise ValueError("diag was produced at a higher event floor; p is missing "
                         "for the wallets this floor would admit")
    base = (diag["cand"] & np.isfinite(diag["held_edge"]) & (diag["held_edge"] > 0)
            & (diag["held_edge"] >= min_edge) & (diag["n_ev"] >= min_events))
    if alpha >= 1.0:
        return base
    return base & np.isfinite(diag["p"]) & (diag["p"] < alpha)


def decile_profile(w1, target, pool, axis_key, n_bins=N_BINS):
    """Pooled target ROI per within-pool decile of a single screen axis.

    Deciles are refit inside `pool`, so this is the rule an operator would
    actually write ("take the top tenth by ROI among wallets with >=N bets") and
    not a rectangle chosen after seeing the answer."""
    idx = np.flatnonzero(pool)
    edges = quantile_edges(w1[axis_key][idx], n_bins)
    b = bin_by_edges(np.nan_to_num(w1[axis_key][idx], nan=0.0), edges)
    nb = edges.size - 1
    prof = np.bincount(b, weights=target["profit"][idx], minlength=nb)
    spend = np.bincount(b, weights=target["spend"][idx], minlength=nb)
    cnt = np.bincount(b, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        roi = np.where(spend > 0, prof / spend, np.nan)
    return {"edges": edges, "roi": roi, "count": cnt, "spend": spend, "profit": prof,
            "pool_roi": pooled_roi(target["profit"][idx], target["spend"][idx]),
            "top_minus_pool": float(roi[-1] - pooled_roi(target["profit"][idx],
                                                         target["spend"][idx])),
            "top_minus_bottom": float(roi[-1] - roi[0])}


def screen_w3_stats(sel_mask, w3, min_target=MIN_TARGET_BETS, profit_key="profit"):
    """`(pooled, equal-weight mean, per-wallet median)` W3 ROI for a wallet mask.

    All three, always, because they fail in different directions: pooled follows
    the biggest stake, the equal-weight mean is unbounded above (a $10 wallet
    returning +300% counts as much as a $1M one) and only the median is robust to
    both. A screen that is only positive on one of the three has not worked.

    `profit_key` selects the numerator: `profit` (gross), `resid_profit` (scored
    over `E[outcome|entry_price]`, i.e. the price control) or `net_profit` (after
    the real per-category taker fee and one adverse tick). The same three
    weightings on all three numerators is what stops a result from surviving only
    on the one combination that happens to flatter it."""
    m = sel_mask & (w3["n"] >= min_target)
    p_, s_ = w3[profit_key][m], w3["spend"][m]
    ok = s_ > 0
    r = p_[ok] / s_[ok] if np.any(ok) else np.zeros(0)
    return (pooled_roi(p_, s_),
            float(r.mean()) if r.size else float("nan"),
            float(np.median(r)) if r.size else float("nan"),
            int(m.sum()), float(s_.sum()))


def size_matched_median(sel_mask, pool_mask, w3, min_target=MIN_TARGET_BETS,
                        profit_key="profit", n_bins=10):
    """The pool's own median per-wallet ROI, reweighted to the SCREEN's wallet-size
    mix — the wallet-level analogue of the price-mimicking index in §11c.

    A top-ROI screen systematically selects SMALLER wallets (a $200 book swings a
    ratio that a $2M book cannot), so "the screen's median beats a random wallet's"
    could be nothing but "small wallets have a higher median than big ones". This
    reprices that: wallets are decile-binned on W3 stake inside the pool, each bin
    contributes the pool's own median in that bin, and the bins are mixed at the
    screen's own weights. Subtract it from the screen's median and what is left is
    the part that is not a size habit."""
    p = pool_mask & (w3["n"] >= min_target) & (w3["spend"] > 0)
    s = sel_mask & p
    if not np.any(s) or int(p.sum()) < n_bins:
        return float("nan")
    stake = np.log10(np.maximum(w3["spend"], 1e-9))
    edges = quantile_edges(stake[p], n_bins)
    b = bin_by_edges(stake, edges)
    r = np.where(w3["spend"] > 0, w3[profit_key] / np.maximum(w3["spend"], 1e-12), np.nan)
    num = den = 0.0
    for k in range(edges.size - 1):
        w = int((s & (b == k)).sum())
        vals = r[p & (b == k)]
        vals = vals[np.isfinite(vals)]
        if w and vals.size:
            num += w * float(np.median(vals))
            den += w
    return num / den if den > 0 else float("nan")


def screen_w3_null_p(sel_mask, w3, perms, min_target=MIN_TARGET_BETS,
                     profit_key="profit"):
    """Null-P p-values for `screen_w3_stats`, on the UNTOUCHED third window.

    The screen itself is held bit-for-bit; only the wallet -> (W2,W3) record
    linkage is redrawn within depth strata. That is what makes this a test of the
    SELECTION rather than a CI conditioned on the wallets the rule already picked —
    the distinction the unrestricted run found decisive (its conditional CI said
    p<0.005 where the honest permutation said p=0.052). `w3["n"]` is permuted along
    with the records, so the null's own sample-size filter is applied to the same
    records it is filtering."""
    real = screen_w3_stats(sel_mask, w3, min_target, profit_key)[:3]
    draws = np.array([
        screen_w3_stats(sel_mask, {profit_key: w3[profit_key][p],
                                   "spend": w3["spend"][p], "n": w3["n"][p]},
                        min_target, profit_key)[:3]
        for p in perms], dtype=float) if perms else np.zeros((0, 3))
    out = []
    for j in range(3):
        col = draws[:, j][np.isfinite(draws[:, j])] if draws.size else np.zeros(0)
        out.append((float((col >= real[j]).sum()) + 1.0) / (col.size + 1.0)
                   if col.size else float("nan"))
    return real, out


def null_o_dataset(d, rng, n_price_bins=20):
    """Null O: permute `resolved_value` within entry-price quantile bins. Returns
    a slim dataset sharing the parent's untouched arrays."""
    edges = quantile_edges(d["price"], n_price_bins)
    b = bin_by_edges(d["price"], edges)
    value = d["value"].copy()
    for k in range(int(b.max()) + 1):
        m = np.flatnonzero(b == k)
        if m.size > 1:
            value[m] = value[m[rng.permutation(m.size)]]
    e = {k: d[k] for k in ("wcode", "ts", "price", "n_wallets",
                           "starts", "counts", "b0", "b1", "b2", "b3")}
    e["value"] = value
    # Stake is untouched by an outcome shuffle, so spend carries over verbatim and
    # profit is recovered from it without needing the (dropped) share count:
    # size = spend / price, so profit = spend * (value - price) / price.
    e["spend"] = d["spend"]
    e["profit"] = d["spend"] * (value - d["price"]) / d["price"]
    return e


def null_o_gate_dataset(d, rng, n_price_bins=20):
    """Null O, but complete enough to re-run the CERTAINTY GATE, not just the
    rectangle search.

    Same shuffle — `resolved_value` permuted within entry-price quantile bins — so
    the favorite-longshot curve survives bit-for-bit and every trace of skill and of
    market clustering dies. It additionally carries `ecode` (the gate's cluster unit)
    and a rebuilt per-bet residual. The calibration curve itself is NOT refit: it is
    a function of price, price is untouched, and permuting outcomes *within* its own
    bins leaves E[outcome|price] unchanged by construction — so reusing it applies
    literally the same yardstick to the real and null arms."""
    edges = quantile_edges(d["price"], n_price_bins)
    b = bin_by_edges(d["price"], edges)
    value = d["value"].copy()
    for k in range(int(b.max()) + 1):
        m = np.flatnonzero(b == k)
        if m.size > 1:
            value[m] = value[m[rng.permutation(m.size)]]
    expected = d["value"] - d["resid_bet"]          # == E[outcome|entry_price]
    shares = d["spend"] / np.maximum(d["price"], 1e-12)
    e = {k: d[k] for k in ("wcode", "ts", "price", "ecode", "ccode", "n_wallets",
                           "starts", "counts", "b0", "b1", "b2", "b3",
                           "cat_names", "fee_rate_by_code", "spend")}
    e["value"] = value
    e["resid_bet"] = value - expected
    e["profit"] = shares * (value - d["price"])
    e["resid_profit"] = shares * (value - expected)
    e["fee_real"] = d["fee_real"]                   # stake and price are untouched
    e["tick_cost"] = d["tick_cost"]
    return e


def null_c_dataset(d, order_cells, cell_owner, cell_size, rng):
    """Null C: cell permutation. Reassigns each (wallet, market) cell to a wallet
    with the same cell size, then re-sorts into the canonical (wallet, time)
    order. Per-bet economics are carried along unchanged — only ownership moves."""
    new_w = permute_cells_within_size(order_cells, cell_owner, cell_size, rng)
    o = np.lexsort((d["ts"], new_w))
    e = {"n_wallets": d["n_wallets"], "wcode": new_w[o]}
    for k in ("ts", "price", "value", "spend", "profit"):
        e[k] = d[k][o]
    attach_blocks(e)
    return e


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shuffles-p", type=int, default=500)
    ap.add_argument("--shuffles-o", type=int, default=25)
    ap.add_argument("--shuffles-c", type=int, default=25)
    ap.add_argument("--shuffles-og", type=int, default=25,
                    help="null-O draws for the CERTAINTY-GATE rebuild (section 11e)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--real-world", action="store_true",
                    help="restrict the ENTIRE analysis to markets "
                         "`src.discover.is_real_world` admits (i.e. drop the "
                         "5-minute micro_crypto tape) and recalibrate the depth "
                         "and event-floor ladders to that population. Without it "
                         "the run reproduces the unrestricted analysis verbatim.")
    ap.add_argument("--tape", choices=("ledger", "realworld"), default="ledger",
                    help="population source: the shared bet ledger (default, what "
                         "every prior run used) or the isolated deep real-world "
                         "sample from src/realworld_deepen.py. The screening rules "
                         "are identical either way; only the wallets change.")
    args = ap.parse_args()

    set_population(args.real_world)
    cfg = load_config()
    print("=" * 118)
    print("SCREENING SURFACE — E[out-of-sample ROI | (success_rate, roi, n_bets)]"
          + ("   [REAL-WORLD ONLY]" if args.real_world else "")
          + (f"   [TAPE: {args.tape}]" if args.tape != "ledger" else ""))
    print("=" * 118)

    d = prepare(cfg, real_world=args.real_world, tape=args.tape)
    if d["drop_report"] is not None:
        dr = d["drop_report"]
        print("\n--- 0a. REAL-WORLD RESTRICTION (src.discover.is_real_world) ---")
        print(f"  kept   : {dr['bets_after']:,} of {dr['bets_before']:,} resolved BUY "
              f"bets ({100 * dr['bets_after'] / max(dr['bets_before'], 1):.1f}%), "
              f"${dr['stake_after']:,.0f} of ${dr['stake_before']:,.0f} stake "
              f"({100 * dr['stake_after'] / max(dr['stake_before'], 1e-9):.1f}%)")
        print(f"  markets: {dr['markets_after']:,} of {dr['markets_before']:,}   "
              f"wallets with >=1 kept bet: {dr['wallets_after']:,} of "
              f"{dr['wallets_before']:,}")
        print(f"  DROPPED categories: {', '.join(dr['dropped_categories'])}")
        print(f"  KEPT    categories: {', '.join(dr['kept_categories'])}")
        print(f"  measured speed_bucket of the KEPT bets (independent of the slug "
              f"classifier): {dr['kept_speed']}")
    rects = all_rectangles(N_BINS, N_BINS)
    sp = surface_pass(d)
    w1, w2, w3 = sp["w1"], sp["w2"], sp["w3"]
    eligible, sr_bin, roi_bin = sp["eligible"], sp["sr_bin"], sp["roi_bin"]
    sr_edges, roi_edges = sp["sr_edges"], sp["roi_edges"]
    rng_b = np.random.default_rng(args.seed + 1)

    # ---- 0. population and survivorship ----
    has_any = d["counts"] > 0
    print("\n--- 0. Population & survivorship ---")
    print(f"resolved BUY bets            : {d['wcode'].size:,}")
    print(f"wallets with any resolved BUY: {int(has_any.sum()):,} "
          f"(of {d['n_wallets']:,} addresses in the ledger)")
    print(f"distinct markets {d['n_markets']:,} -> resolution events {d['n_events']:,}")
    print(f"  {'W1 depth':>10} {'pool':>7} {'W2>=5':>8} {'%':>7} {'+W3>=5':>8} {'%':>7}")
    for thr in DEPTHS:
        pool = has_any & (w1["n"] >= thr)
        surv = pool & (w2["n"] >= MIN_TARGET_BETS)
        surv3 = surv & (w3["n"] >= MIN_TARGET_BETS)
        print(f"  {'>=' + str(thr):>10} {int(pool.sum()):>7} {int(surv.sum()):>8} "
              f"{100 * surv.sum() / max(pool.sum(), 1):>6.1f}% {int(surv3.sum()):>8} "
              f"{100 * surv3.sum() / max(pool.sum(), 1):>6.1f}%")
    # SURVIVORSHIP / SELECTION. Windows are thirds of one record, so "present in
    # W2" is mechanical (W1>=5 implies n>=15 implies W2>=5). The selection that
    # actually bites is upstream: which wallets the backfill ever deepened.
    try:
        import json
        seeds = set(json.loads(
            (ROOT / "data/interim/backfill_cursors.json").read_text()).keys())
        deep = np.isin(d["wallets"], np.array(sorted(seeds)))
        for thr in DETAILS:
            pool = has_any & (w1["n"] >= thr)
            print(f"  SELECTION: of {int(pool.sum()):>5} wallets with W1>={thr}, "
                  f"{int((pool & deep).sum()):>5} "
                  f"({100 * (pool & deep).sum() / max(pool.sum(), 1):>5.1f}%) were "
                  f"deepened by the backfill, which seeds from the RANKED table")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  SELECTION: backfill cursor read failed ({exc})")
    print(f"screen bin edges success_rate: "
          f"{np.array2string(sr_edges[1:-1], precision=4, floatmode='fixed')}")
    print(f"screen bin edges roi         : "
          f"{np.array2string(roi_edges[1:-1], precision=4, floatmode='fixed')}")

    # ---- 1. the surface ----
    print("\n--- 1. The surface: W2 pooled ROI by (W1 success_rate decile, W1 roi decile) ---")
    for thr in DETAILS[:3]:
        sel = eligible & (w1["n"] >= thr)
        if sel.sum() == 0:
            continue
        rb, cb = sr_bin[sel], roi_bin[sel]
        pg = grid_accumulate(rb, cb, w2["profit"][sel], (N_BINS, N_BINS))
        rg = grid_accumulate(rb, cb, w2["resid_profit"][sel], (N_BINS, N_BINS))
        sg = grid_accumulate(rb, cb, w2["spend"][sel], (N_BINS, N_BINS))
        cg = grid_accumulate(rb, cb, np.ones(int(sel.sum())), (N_BINS, N_BINS))
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(sg > 0, pg / sg, np.nan)
            res = np.where(sg > 0, rg / sg, np.nan)
        for name, surf in (("RAW", raw), ("PRICE-RESIDUALIZED", res)):
            print(f"\n  [{name}] depth W1>={thr} ({int(sel.sum())} wallets). "
                  f"rows = success_rate decile, cols = W1 roi decile, cell = W2 pooled ROI (K)")
            print("        " + "".join(f"{c:>15}" for c in range(N_BINS)))
            for r in range(N_BINS):
                cells = "".join(
                    f"{surf[r, c]:>+10.3f}({int(cg[r, c]):>3})" if cg[r, c] > 0
                    else f"{'.':>15}" for c in range(N_BINS))
                print(f"  sr{r:<3}" + cells)

    # ---- 2. baselines on W2 ----
    print("\n--- 2. Baselines (window W2) ---")
    ranked = pd.read_parquet(RANKED_WALLETS_PATH, columns=["wallet", "edge_persisted"])
    certified = ranked.loc[ranked["edge_persisted"].fillna(False), "wallet"].to_numpy()
    cert_mask = np.isin(d["wallets"], certified)
    masks = {
        "everyone": has_any & (w2["n"] > 0),
        "all": eligible,
        "certified": cert_mask & (w2["n"] > 0),
    }
    labels = {"everyone": "EVERY wallet with W2 bets",
              "all": f"ALL screenable (W1>={DEPTHS[0]}, W2>={MIN_TARGET_BETS})",
              "certified": f"CERTIFIED ({int(cert_mask.sum())} edge_persisted)"}
    for key in ("everyone", "all", "certified"):
        print("  " + fmt_region(region_report(d, masks[key], d["b1"], d["b2"],
                                              rng_b, args.boot, labels[key])))

    # ---- 2b. pre-specified 1-D screens, and the minimum viable screening depth --
    print("\n--- 2b. PRE-SPECIFIED 1-D screens (no argmax): top decile vs its pool ---")
    rng_1d = np.random.default_rng(args.seed + 7)
    perms = list(null_p_permutations(sp, rng_1d, args.shuffles_p))
    for axis in ("roi", "success_rate"):
        print(f"\n  screen axis = W1 {axis}")
        print(f"  {'depth':>7} {'pool':>6} {'pool ROI':>10} {'top-dec ROI':>12} {'K':>5} "
              f"{'top-pool':>10} {'nullP med':>10} {'nullP p95':>10} {'p':>7} "
              f"{'W3 top-dec':>11} {'W3 pool':>9}")
        for thr in DEPTHS:
            pool = eligible & (w1["n"] >= thr)
            if pool.sum() < 4 * N_BINS:
                continue
            prof = decile_profile(w1, w2, pool, axis)
            stat = prof["top_minus_pool"]
            null_stats = np.array([
                decile_profile(w1, {"profit": w2["profit"][p], "spend": w2["spend"][p]},
                               pool, axis)["top_minus_pool"] for p in perms])
            null_stats = null_stats[np.isfinite(null_stats)]
            p = (float((null_stats >= stat).sum()) + 1.0) / (null_stats.size + 1.0)
            # the same decile rule, priced on the untouched third window
            idx = np.flatnonzero(pool)
            edges = quantile_edges(w1[axis][idx], N_BINS)
            b = bin_by_edges(np.nan_to_num(w1[axis][idx], nan=0.0), edges)
            top = idx[b == edges.size - 2]
            m3 = np.zeros(d["n_wallets"], dtype=bool)
            m3[top] = True
            m3 &= w3["n"] >= MIN_TARGET_BETS
            p3 = eligible & (w1["n"] >= thr) & (w3["n"] >= MIN_TARGET_BETS)
            print(f"  {'>=' + str(thr):>7} {int(pool.sum()):>6} {prof['pool_roi']:>+10.4f} "
                  f"{prof['roi'][-1]:>+12.4f} {int(prof['count'][-1]):>5} {stat:>+10.4f} "
                  f"{np.median(null_stats):>+10.4f} {np.quantile(null_stats, 0.95):>+10.4f} "
                  f"{p:>7.4f} "
                  f"{pooled_roi(w3['profit'][m3], w3['spend'][m3]):>+11.4f} "
                  f"{pooled_roi(w3['profit'][p3], w3['spend'][p3]):>+9.4f}")
            if axis == "roi" and thr == 30:
                mt = np.zeros(d["n_wallets"], dtype=bool)
                mt[top] = True
                print("  " + fmt_region(region_report(
                    d, mt, d["b1"], d["b2"], rng_b, args.boot,
                    "  ^ that top decile, W2 detail")))
                print("  " + fmt_region(region_report(
                    d, mt & (w3["n"] >= MIN_TARGET_BETS), d["b2"], d["b3"], rng_b,
                    args.boot, "  ^ that top decile, W3 detail")))

    # ---- 3. the search ----
    print(f"\n--- 3. Rectangle search (objective: W2 pooled ROI, min {MIN_WALLETS} wallets, "
          f"min {100 * MIN_SPEND_SHARE:.0f}% of pool stake) ---")
    best, per_depth = search(w1, w2, sr_bin, roi_bin, rects, DEPTHS,
                             MIN_WALLETS, eligible, min_spend_share=MIN_SPEND_SHARE)
    print(f"  {'depth':>7} {'pool':>6} {'best W2 ROI':>13} {'K':>5} "
          f"{'success_rate band':>24} {'roi band':>26} {'W2 $profit':>14}")
    for rec in per_depth:
        if rec["rect"] is None:
            print(f"  {'>=' + str(rec['depth']):>7} {rec['n_pool']:>6} {'--':>13}")
            continue
        r0, r1, c0, c1 = rec["rect"]
        srb = f"[{sr_edges[r0]:.3f},{sr_edges[r1]:.3f})"
        roib = f"[{roi_edges[c0]:.4f},{roi_edges[c1]:.4f})"
        print(f"  {'>=' + str(rec['depth']):>7} {rec['n_pool']:>6} {rec['roi']:>+13.4f} "
              f"{rec['k']:>5} {srb:>24} {roib:>26} {rec['profit']:>14,.0f}")
    if best is None:
        print("  no qualifying rectangle; nothing further to report.")
        return
    r0, r1, c0, c1 = best["rect"]
    print(f"\n  OPTIMUM (fitted on W2): depth W1>={best['depth']}, success_rate in "
          f"[{sr_edges[r0]:.4f},{sr_edges[r1]:.4f}), roi in "
          f"[{roi_edges[c0]:.4f},{roi_edges[c1]:.4f})  ->  K={best['k']}")
    opt_mask = rule_mask(w1, sr_bin, roi_bin, eligible, best["depth"], best["rect"])
    opt_w2 = region_report(d, opt_mask, d["b1"], d["b2"], rng_b, args.boot,
                           "OPTIMUM  (W2, in-sample)")
    opt_w3 = region_report(d, opt_mask & (w3["n"] >= MIN_TARGET_BETS), d["b2"], d["b3"],
                           rng_b, args.boot, "OPTIMUM  (W3, HONEST)")
    print("  " + fmt_region(opt_w2))
    print("  " + fmt_region(opt_w3))

    # the residual-objective optimum, for the price-control question
    best_r, _ = search(w1, w2, sr_bin, roi_bin, rects, DEPTHS,
                       MIN_WALLETS, eligible, profit_key="resid_profit",
                       min_spend_share=MIN_SPEND_SHARE)
    if best_r is not None:
        r0, r1, c0, c1 = best_r["rect"]
        print(f"  residual-objective optimum: depth>={best_r['depth']}, "
              f"sr [{sr_edges[r0]:.4f},{sr_edges[r1]:.4f}), roi "
              f"[{roi_edges[c0]:.4f},{roi_edges[c1]:.4f}), K={best_r['k']}, "
              f"W2 residual ROI {best_r['roi']:+.4f}")

    # ---- 4. W3 comparisons ----
    print("\n--- 4. W3 (honest window) comparisons ---")
    for key in ("everyone", "all", "certified"):
        m3 = masks[key] & (w3["n"] >= MIN_TARGET_BETS)
        print("  " + fmt_region(region_report(d, m3, d["b2"], d["b3"], rng_b, args.boot,
                                              labels[key] + " [W3]")))

    print(f"\n--- 4b. Random selections of the same size (K={best['k']}) ---")
    pool = np.flatnonzero(eligible & (w1["n"] >= best["depth"]))
    rng_r = np.random.default_rng(args.seed + 2)
    for win, lo, hi, tag, got in ((w2, d["b1"], d["b2"], "W2", opt_w2["roi"]),
                                  (w3, d["b2"], d["b3"], "W3", opt_w3["roi"])):
        draws = []
        for _ in range(500):
            pick = rng_r.choice(pool, size=min(best["k"], pool.size), replace=False)
            m = np.zeros(d["n_wallets"], dtype=bool)
            m[pick] = True
            m &= win["n"] >= MIN_TARGET_BETS
            draws.append(pooled_roi(win["profit"][m], win["spend"][m]))
        draws = np.array([x for x in draws if np.isfinite(x)])
        pct = float((draws < got).mean()) if draws.size else np.nan
        print(f"  random-K {tag}: median {np.median(draws):+.4f} "
              f"[p5 {np.quantile(draws, 0.05):+.4f}, p95 {np.quantile(draws, 0.95):+.4f}] "
              f"| optimum {got:+.4f} = percentile {100 * pct:.1f}")

    # ---- 5. nulls ----
    print("\n--- 5. Nulls: what does the SAME search find when there is nothing there? ---")
    real_best = best["roi"]
    print("\n  5a. null P across search settings (the calibrated null: identical "
          "screen and target marginals)")
    print(f"  {'minK':>5} {'minSpend':>9} {'REAL best':>11} {'K':>5} {'$profit':>13} "
          f"{'null med':>10} {'null p90':>10} {'null max':>10} {'p':>7}")
    for mk, msh in SEARCH_SETTINGS:
        rb, _ = search(w1, w2, sr_bin, roi_bin, rects, DEPTHS, mk,
                       eligible, min_spend_share=msh)
        draws = null_p_draws(sp, rects, np.random.default_rng(args.seed + 3),
                             args.shuffles_p, min_wallets=mk, min_spend_share=msh)
        draws = draws[np.isfinite(draws)]
        if rb is None or draws.size == 0:
            print(f"  {mk:>5} {msh:>9.2f} {'--':>11}")
            continue
        p = (float((draws >= rb["roi"]).sum()) + 1.0) / (draws.size + 1.0)
        print(f"  {mk:>5} {msh:>9.2f} {rb['roi']:>+11.4f} {rb['k']:>5} "
              f"{rb['profit']:>13,.0f} {np.median(draws):>+10.4f} "
              f"{np.quantile(draws, 0.90):>+10.4f} {draws.max():>+10.4f} {p:>7.4f}")

    print("\n  5b. the other two nulls, at the headline setting")
    o_draws = np.empty(args.shuffles_o)
    rng_o = np.random.default_rng(args.seed + 4)
    for s in range(args.shuffles_o):
        e = null_o_dataset(d, rng_o)
        nsp = surface_pass(e, costs=False)
        nb, _ = search(nsp["w1"], nsp["w2"], nsp["sr_bin"], nsp["roi_bin"], rects,
                       DEPTHS, MIN_WALLETS, nsp["eligible"],
                       min_spend_share=MIN_SPEND_SHARE)
        o_draws[s] = nb["roi"] if nb else np.nan
        del e, nsp
    report_null("O (outcome shuffle within price bin)", o_draws, real_best)

    order_cells, cell_owner, cell_size = build_cells(d["wcode"], d["mcode"])
    c_draws = np.empty(args.shuffles_c)
    rng_c = np.random.default_rng(args.seed + 5)
    for s in range(args.shuffles_c):
        e = null_c_dataset(d, order_cells, cell_owner, cell_size, rng_c)
        nsp = surface_pass(e, costs=False)
        nb, _ = search(nsp["w1"], nsp["w2"], nsp["sr_bin"], nsp["roi_bin"], rects,
                       DEPTHS, MIN_WALLETS, nsp["eligible"],
                       min_spend_share=MIN_SPEND_SHARE)
        c_draws[s] = nb["roi"] if nb else np.nan
        del e, nsp
    report_null("C (cell permutation within size, cluster-preserving)", c_draws, real_best)

    # ---- 6. cross-validation of the ARGMAX ----
    print("\n--- 6. Cross-validation: how much of the optimum is region-selection overfit? ---")
    rng_cv = np.random.default_rng(args.seed + 6)
    ins, oos = [], []
    for _ in range(100):
        coin = rng_cv.random(d["n_wallets"]) < 0.5
        ba, _ = search(w1, w2, sr_bin, roi_bin, rects, DEPTHS,
                       MIN_WALLETS, eligible & coin, min_spend_share=MIN_SPEND_SHARE)
        if ba is None:
            continue
        mb = rule_mask(w1, sr_bin, roi_bin, eligible & ~coin, ba["depth"], ba["rect"])
        ins.append(ba["roi"])
        oos.append(pooled_roi(w2["profit"][mb], w2["spend"][mb]))
    ins, oos = np.array(ins), np.array(oos)
    ok = np.isfinite(oos)
    print(f"  100 random wallet half-splits: fit-half best ROI median {np.median(ins):+.4f}; "
          f"held-half realized median {np.median(oos[ok]):+.4f} over {int(ok.sum())} folds; "
          f"shrinkage {np.median(ins) - np.median(oos[ok]):+.4f}; "
          f"held-half > 0 in {100 * (oos[ok] > 0).mean():.0f}% of folds")

    # ---- 7. edge vs breadth ----
    print("\n--- 7. Edge vs breadth: the highest-ROI region is not the highest-dollar region ---")
    print(f"  {'depth':>7} {'K':>5} {'W2 ROI':>10} {'W2 $profit':>16} {'W2 $spend':>16}")
    for rec in per_depth:
        if rec["rect"] is not None:
            print(f"  {'>=' + str(rec['depth']):>7} {rec['k']:>5} {rec['roi']:>+10.4f} "
                  f"{rec['profit']:>16,.0f} {rec['spend']:>16,.0f}")

    # ---- 8b. CERTAINTY vs DOLLARS, both fitted on W1 only ----
    print("\n--- 8b. THE HEAD-TO-HEAD: certainty screen vs dollars screen, both fitted "
          "on W1 only ---")
    print("  The repo's certified 26 were selected on a 50/50 split of FULL history, so")
    print("  W2/W3 are in-sample for them. Here the SAME gate runs inside W1 alone, so")
    print("  both screens see the same information and W2/W3 are honestly held out.")
    alpha_ladder = (0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
    min_edge = float(cfg["scoring"].get("min_skill_edge", 0.02))
    prod_ev = int(cfg["scoring"].get("min_oos_markets", 30))
    # The gate is run at the LOWEST floor on the ladder so `gate_passes` can
    # reconstruct every higher floor from the same diagnostics (see its docstring).
    ev_ladder = REALWORLD_EVENT_FLOORS if args.real_world else (prod_ev,)
    min_ev = min(ev_ladder)
    passes, diag = certify_on_window(
        d, d["b0"], d["b1"], eligible, alpha_ladder, min_edge, min_ev,
        int(cfg["scoring"].get("oos_bootstrap_resamples", 2000)), args.seed + 8)

    if args.real_world:
        # --- 8b-RW. The event floor is a FREQUENCY threshold, so sweep it. ---
        # The unrestricted run's headline screen cleared a 30-held-out-event floor
        # inside half of W1; that is one hour of the 5-minute tape and a career in
        # real-world markets. Which floors are even ATTAINABLE here is a finding in
        # its own right, and it is reported before any performance number.
        print(f"\n  8b-RW. EVENT-FLOOR LADDER (min_edge {min_edge}); the production "
              f"floor is {prod_ev}")
        print(f"  {'minEv':>6} {'alpha':>7} {'K':>5} {'W3 ROI':>9} {'W3 eqw':>9} "
              f"{'W3 net':>9} {'W3 resid':>9} {'W3 $stake':>12} {'W3 effEv':>9} "
              f"{'W3 topW':>8}")
        for ev in ev_ladder:
            for a in (0.005, 0.05, 1.0):
                m = gate_passes(diag, min_edge, ev, a) & (w3["n"] >= MIN_TARGET_BETS)
                if m.sum() == 0:
                    print(f"  {ev:>6} {a:>7} {0:>5}   (floor unattainable at this alpha)")
                    continue
                pr = priced_screen(d, m, d["b2"], d["b3"], rng_b, args.boot, "")
                print(f"  {ev:>6} {a:>7} {pr['wallets']:>5} {pr['roi']:>+9.4f} "
                      f"{pr['eqw']:>+9.4f} {pr['net']:>+9.4f} {pr['resid']:>+9.4f} "
                      f"{pr['spend']:>12,.0f} {pr['eff_events']:>9.1f} "
                      f"{pr['top_wallet']:>8.2f}")
        # Pre-registered headline: the production alpha, at the SHALLOWEST floor on
        # the ladder that still yields a portfolio (>= MIN_WALLETS wallets). Chosen
        # on K alone — never on a return — so it cannot be a performance argmax.
        head_ev = next((ev for ev in sorted(ev_ladder, reverse=True)
                        if int(gate_passes(diag, min_edge, ev, 0.005).sum())
                        >= MIN_WALLETS), min(ev_ladder))
        print(f"  headline event floor = {head_ev} (deepest floor on the ladder that "
              f"still certifies >= {MIN_WALLETS} wallets at alpha 0.005; selected on "
              f"K only, never on a return)")
        min_ev = head_ev
        passes = {a: gate_passes(diag, min_edge, head_ev, a) for a in alpha_ladder}

    print(f"  gate: candidate & held-W1 skill edge >= {min_edge} & >= {min_ev} events "
          f"& cluster-p < alpha")
    print(f"  {'alpha':>7} {'K':>5} {'W2 ROI':>9} {'W2 eqw':>9} {'W2 $profit':>13} "
          f"{'W3 ROI':>9} {'W3 95% CI':>19} {'W3 eqw':>9} {'W3 $profit':>13} "
          f"{'W3 effEv':>9} {'W3 topW':>8}")
    for a in alpha_ladder:
        m = passes[a]
        if m.sum() == 0:
            print(f"  {a:>7} {0:>5}")
            continue
        r2 = region_report(d, m & (w2["n"] > 0), d["b1"], d["b2"], rng_b, args.boot, "")
        r3 = region_report(d, m & (w3["n"] > 0), d["b2"], d["b3"], rng_b, args.boot, "")
        print(f"  {a:>7} {int(m.sum()):>5} {r2['roi']:>+9.4f} {r2['roi_wallet_eq']:>+9.4f} "
              f"{r2['profit']:>13,.0f} {r3['roi']:>+9.4f} "
              f"[{r3['ci_lo']:>+.4f},{r3['ci_hi']:>+.4f}] {r3['roi_wallet_eq']:>+9.4f} "
              f"{r3['profit']:>13,.0f} {r3['eff_events']:>9.1f} "
              f"{r3['top_wallet_spend_share']:>8.2f}")
    no_sig = passes[1.0]
    print(f"  (no-significance-test row = alpha 1.0: the pure magnitude+breadth screen, "
          f"K={int(no_sig.sum())})")

    # ---- 9. equal-weight portfolio view (the diversification thesis) ----
    print("\n--- 9. Equal-weight portfolio view: the brief's diversification thesis, "
          "tested directly ---")
    print("  Pooled ROI mirrors stake size, so a single whale can carry a whole region.")
    print("  Equal weight per wallet is what 'a portfolio of K wallets' actually means.")
    print(f"  {'depth':>7} {'axis':>13} {'pool eqw':>10} {'top-dec eqw':>12} {'K':>5} "
          f"{'nullP med':>10} {'nullP p95':>10} {'p':>7} {'W3 top eqw':>11}")
    for thr in DETAILS:
        pool = eligible & (w1["n"] >= thr)
        if pool.sum() < 4 * N_BINS:
            continue
        for axis in ("roi", "success_rate"):
            idx = np.flatnonzero(pool)
            edges = quantile_edges(w1[axis][idx], N_BINS)
            b = bin_by_edges(np.nan_to_num(w1[axis][idx], nan=0.0), edges)
            top = idx[b == edges.size - 2]

            def eqw(sel, win):
                s, p_ = win["spend"][sel], win["profit"][sel]
                ok = s > 0
                return float(np.mean(p_[ok] / s[ok])) if ok.any() else np.nan

            stat = eqw(top, w2)
            nulls = np.array([eqw(top, {"spend": w2["spend"][p], "profit": w2["profit"][p]})
                              for p in perms])
            nulls = nulls[np.isfinite(nulls)]
            pv = (float((nulls >= stat).sum()) + 1.0) / (nulls.size + 1.0)
            top3 = top[w3["n"][top] >= MIN_TARGET_BETS]
            print(f"  {'>=' + str(thr):>7} {axis:>13} {eqw(idx, w2):>+10.4f} "
                  f"{stat:>+12.4f} {top.size:>5} {np.median(nulls):>+10.4f} "
                  f"{np.quantile(nulls, 0.95):>+10.4f} {pv:>7.4f} "
                  f"{eqw(top3, w3):>+11.4f}")

    # ---- 9b. is the equal-weight result a tail artifact? ----
    print("\n--- 9b. Robustness of the equal-weight screen: mean vs MEDIAN vs winsorized ---")
    print("  A mean of per-wallet ROIs is unbounded above (a $10 wallet that returns")
    print("  +300% counts as much as a $1M one), so the mean alone cannot distinguish")
    print("  'the portfolio works' from 'one micro-stake wallet got lucky'. The median")
    print("  and the 5/95-winsorized mean can, and each gets its own null P p-value.")
    def _stats(sel, win, tick=None, fee=None):
        s = win["spend"][sel]
        ok = s > 0
        if not np.any(ok):
            return dict(mean=np.nan, med=np.nan, wins=np.nan, pos=np.nan,
                        med_spend=np.nan, med_tick=np.nan, med_stress=np.nan,
                        spend=0.0)
        s = s[ok]
        r = win["profit"][sel][ok] / s
        lo, hi = np.quantile(r, [0.05, 0.95])
        out = dict(mean=float(r.mean()), med=float(np.median(r)),
                   wins=float(np.clip(r, lo, hi).mean()), pos=float((r > 0).mean()),
                   med_spend=float(np.median(s)), spend=float(s.sum()))
        if tick is not None:
            out["med_tick"] = float(np.median((win["profit"][sel][ok] - tick[sel][ok]) / s))
            out["med_stress"] = float(np.median(
                (win["profit"][sel][ok] - tick[sel][ok] - fee[sel][ok]) / s))
        else:
            out["med_tick"] = out["med_stress"] = np.nan
        return out

    print(f"  {'depth':>7} {'set':>10} {'K':>5} {'W2 mean':>9} {'p':>7} {'W2 med':>9} "
          f"{'p':>7} {'W2 wins':>9} {'p':>7} {'W2 pos%':>8} {'W3 mean':>9} {'W3 med':>9} "
          f"{'W3 pos%':>8} {'medSpend':>10}")
    for thr in DETAILS:
        pool = eligible & (w1["n"] >= thr)
        if pool.sum() < 4 * N_BINS:
            continue
        idx = np.flatnonzero(pool)
        edges = quantile_edges(w1["roi"][idx], N_BINS)
        b = bin_by_edges(np.nan_to_num(w1["roi"][idx], nan=0.0), edges)
        top = idx[b == edges.size - 2]
        for name, sel in (("pool", idx), ("top-decile", top)):
            s2 = _stats(sel, w2)
            s3 = _stats(sel[w3["n"][sel] >= MIN_TARGET_BETS], w3)
            ps = {}
            for key in ("mean", "med", "wins"):
                nulls = np.array([
                    _stats(sel, {"spend": w2["spend"][p], "profit": w2["profit"][p]})[key]
                    for p in perms])
                nulls = nulls[np.isfinite(nulls)]
                ps[key] = ((float((nulls >= s2[key]).sum()) + 1.0) / (nulls.size + 1.0)
                           if nulls.size else np.nan)
            print(f"  {'>=' + str(thr):>7} {name:>10} {sel.size:>5} {s2['mean']:>+9.4f} "
                  f"{ps['mean']:>7.4f} {s2['med']:>+9.4f} {ps['med']:>7.4f} "
                  f"{s2['wins']:>+9.4f} {ps['wins']:>7.4f} {100 * s2['pos']:>7.1f}% "
                  f"{s3['mean']:>+9.4f} {s3['med']:>+9.4f} {100 * s3['pos']:>7.1f}% "
                  f"{s2['med_spend']:>10,.0f}")

    # ---- 9c. the operating-point menu: how wide should the screen be? ----
    print("\n--- 9c. OPERATING POINTS: equal-dollar-per-wallet, screened on W1 roi, "
          "priced on W3 ---")
    print("  With equal dollars per wallet the region's ROI does not scale with K, so")
    print("  breadth buys VARIANCE REDUCTION and CAPACITY, not a higher rate. Both are")
    print("  shown: W3 median per-wallet ROI (the rate), W3 % of wallets positive (the")
    print("  variance), and the region's total W3 stake (the capacity ceiling).")
    print(f"  {'depth':>7} {'keep':>6} {'W1 roi >=':>10} {'K':>5} {'W2 med':>8} {'p':>7} "
          f"{'W3 med':>8} {'W3 mean':>9} {'W3 pos%':>8} {'W3 med@tick':>12} "
          f"{'W3 med@stress':>14} {'W3 stake $':>13}")
    for thr in DETAILS:
        pool = eligible & (w1["n"] >= thr)
        if pool.sum() < 4 * N_BINS:
            continue
        idx = np.flatnonzero(pool)
        vals = np.nan_to_num(w1["roi"][idx], nan=0.0)
        for keep in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
            cut = float(np.quantile(vals, 1.0 - keep)) if keep < 1.0 else -np.inf
            sel = idx[vals >= cut]
            if sel.size < 10:
                continue
            s2 = _stats(sel, w2)
            nulls = np.array([
                _stats(sel, {"spend": w2["spend"][p], "profit": w2["profit"][p]})["med"]
                for p in perms])
            nulls = nulls[np.isfinite(nulls)]
            pv = ((float((nulls >= s2["med"]).sum()) + 1.0) / (nulls.size + 1.0)
                  if nulls.size else np.nan)
            sel3 = sel[w3["n"][sel] >= MIN_TARGET_BETS]
            s3 = _stats(sel3, w3, w3["tick_cost"], w3["fee_stress"])
            print(f"  {'>=' + str(thr):>7} {keep:>6.2f} {cut:>+10.4f} {sel.size:>5} "
                  f"{s2['med']:>+8.4f} {pv:>7.4f} {s3['med']:>+8.4f} {s3['mean']:>+9.4f} "
                  f"{100 * s3['pos']:>7.1f}% {s3['med_tick']:>+12.4f} "
                  f"{s3['med_stress']:>+14.4f} {s3['spend']:>13,.0f}")

    # ---- 9d. price control ON THE FINDING ----
    print("\n--- 9d. Price control on the equal-weight screen (keep top decile by W1 roi) ---")
    print("  ROI is mechanically noisier at low entry prices, so the screen could be")
    print("  selecting a price habit rather than skill. Two controls: the same screen")
    print("  scored on the PRICE-RESIDUAL target (profit over E[outcome|entry_price]),")
    print("  and the same screen run separately inside each mean-W1-price tercile.")
    cuts_p = np.nanquantile(w1["mean_price"][eligible], [1 / 3, 2 / 3])
    for thr in DETAILS[1:]:
        pool = eligible & (w1["n"] >= thr)
        idx = np.flatnonzero(pool)
        vals = np.nan_to_num(w1["roi"][idx], nan=0.0)
        cut = float(np.quantile(vals, 0.9))
        top = idx[vals >= cut]

        def _med(sel, win, key="profit"):
            s = win["spend"][sel]
            ok = s > 0
            return float(np.median(win[key][sel][ok] / s[ok])) if ok.any() else np.nan

        t3 = top[w3["n"][top] >= MIN_TARGET_BETS]
        i3 = idx[w3["n"][idx] >= MIN_TARGET_BETS]
        print(f"  depth>={thr}: RESIDUAL target — W2 top-decile median "
              f"{_med(top, w2, 'resid_profit'):+.4f} vs pool {_med(idx, w2, 'resid_profit'):+.4f}"
              f" | W3 top-decile median {_med(t3, w3, 'resid_profit'):+.4f} vs pool "
              f"{_med(i3, w3, 'resid_profit'):+.4f}")
        for lo_p, hi_p in zip([-np.inf, *cuts_p], [*cuts_p, np.inf]):
            band = pool & (w1["mean_price"] >= lo_p) & (w1["mean_price"] < hi_p)
            bidx = np.flatnonzero(band)
            if bidx.size < 20:
                continue
            bvals = np.nan_to_num(w1["roi"][bidx], nan=0.0)
            btop = bidx[bvals >= float(np.quantile(bvals, 0.9))]
            b3 = btop[w3["n"][btop] >= MIN_TARGET_BETS]
            bp3 = bidx[w3["n"][bidx] >= MIN_TARGET_BETS]
            print(f"    price band [{lo_p:>6.3f},{hi_p:>6.3f}): pool {bidx.size:>4} "
                  f"K={btop.size:>3}  W3 top-decile median {_med(b3, w3):>+8.4f} "
                  f"vs band pool {_med(bp3, w3):>+8.4f}")

    # ---- 9e. the owner's ROI screen, on the UNTOUCHED window, null-tested ----
    if args.real_world:
        print("\n--- 9e. THE OWNER'S ROI SCREEN (top X% by W1 money-won/money-spent), "
              "priced on the UNTOUCHED W3 window ---")
        print("  Sections 9/9b/9c null-test this screen on W2. W2 is legitimately")
        print("  held out for a 1-D rule with no argmax, but W3 is the window nothing")
        print("  in this script has ever looked at, and the null below charges for the")
        print("  SELECTION step (the wallet->record linkage is redrawn, the rule is not).")
        print("  Three statistics because they fail differently: pooled follows the")
        print("  biggest stake, the equal-weight mean is unbounded above, only the")
        print("  median is robust to both. And each of the three is charged the two")
        print("  controls that have killed everything else in this repo: GROSS, then")
        print("  RESID (scored over E[outcome|entry_price], because a wallet's ROI is")
        print("  mostly a persistent PRICE HABIT — see CLAUDE.md), then NET (the real")
        print("  per-category taker fee plus one adverse tick, charged PER WALLET so")
        print("  the equal-weight and median views pay costs too).")
        print("  sizeIdx = the pool's OWN median reweighted to the screen's wallet-size")
        print("            mix, and excess = median - sizeIdx. A top-ROI screen picks")
        print("            small books (a $200 book swings a ratio a $2M book cannot),")
        print("            so this is the wallet-level twin of the price-mimicking index.")
        print(f"  {'depth':>7} {'keep':>6} {'K':>5} {'W3 $stake':>12} "
              f"{'pooled':>9} {'p':>7} {'eqw':>9} {'p':>7} {'median':>9} {'p':>7} "
              f"| {'RESID med':>9} {'p':>7} | {'NET med':>9} {'p':>7} {'NET eqw':>9} "
              f"| {'sizeIdx':>8} {'excess':>8} {'effEv':>7} {'topW':>5}")
        rw_roi_screen = []
        for thr in DEPTHS:
            pool = eligible & (w1["n"] >= thr)
            idx = np.flatnonzero(pool)
            if idx.size < 4 * N_BINS:
                continue
            vals = np.nan_to_num(w1["roi"][idx], nan=0.0)
            for keep in (0.05, 0.10, 0.25):
                cut = float(np.quantile(vals, 1.0 - keep))
                sel = np.zeros(d["n_wallets"], dtype=bool)
                sel[idx[vals >= cut]] = True
                (pl, eq, md), ps = screen_w3_null_p(sel, w3, perms)
                (_, _, rmd), rps = screen_w3_null_p(sel, w3, perms,
                                                    profit_key="resid_profit")
                (_, neq, nmd), nps = screen_w3_null_p(sel, w3, perms,
                                                      profit_key="net_profit")
                pr = priced_screen(d, sel & (w3["n"] >= MIN_TARGET_BETS),
                                   d["b2"], d["b3"], rng_b, 200, "")
                sidx = size_matched_median(sel, pool, w3)
                rw_roi_screen.append((thr, keep, md, ps[2], rmd, rps[2], nmd, nps[2],
                                      md - sidx))
                print(f"  {'>=' + str(thr):>7} {keep:>6.2f} {pr['wallets']:>5} "
                      f"{pr['spend']:>12,.0f} {pl:>+9.4f} {ps[0]:>7.4f} {eq:>+9.4f} "
                      f"{ps[1]:>7.4f} {md:>+9.4f} {ps[2]:>7.4f} "
                      f"| {rmd:>+9.4f} {rps[2]:>7.4f} | {nmd:>+9.4f} {nps[2]:>7.4f} "
                      f"{neq:>+9.4f} | {sidx:>+8.4f} {md - sidx:>+8.4f} "
                      f"{pr['eff_events']:>7.1f} {pr['top_wallet']:>5.2f}")
        # A 27-cell grid x 3 numerators is 81 looks, so count the survivors rather
        # than letting the reader pick the best cell.
        n_cells = len(rw_roi_screen)
        for tag, j in (("gross median", 3), ("resid median", 5), ("net median", 7)):
            hit = sum(1 for r in rw_roi_screen if np.isfinite(r[j]) and r[j] < 0.05)
            pos = sum(1 for r in rw_roi_screen if np.isfinite(r[j - 1]) and r[j - 1] > 0)
            print(f"  MULTIPLICITY: {tag}: {hit}/{n_cells} cells at p<0.05, "
                  f"{pos}/{n_cells} cells positive (expected under the null at "
                  f"p<0.05: {0.05 * n_cells:.1f} cells, but the cells are nested and "
                  f"share {len(perms)} permutations, so they are NOT independent tests)")
        exc = [r[8] for r in rw_roi_screen if np.isfinite(r[8])]
        if exc:
            print(f"  SIZE CONTROL: median excess over the size-matched index is "
                  f"positive in {sum(1 for x in exc if x > 0)}/{len(exc)} cells, "
                  f"median excess {np.median(exc):+.4f} (raw medians ran "
                  f"{np.median([r[2] for r in rw_roi_screen]):+.4f}), so "
                  f"{100 * (1 - np.median(exc) / max(np.median([r[2] for r in rw_roi_screen]), 1e-9)):.0f}% "
                  f"of the raw median is wallet-size mix")

    # ---- 8. price control ----
    print("\n--- 8. Does the screen add anything beyond the PRICE LEVEL it trades at? ---")
    mp = w1["mean_price"]
    cuts = np.nanquantile(mp[eligible], [1 / 3, 2 / 3])
    for lo_p, hi_p in zip([-np.inf, *cuts], [*cuts, np.inf]):
        band = eligible & (mp >= lo_p) & (mp < hi_p)
        if band.sum() < MIN_WALLETS:
            continue
        bb, _ = search(w1, w2, sr_bin, roi_bin, rects, DEPTHS,
                       MIN_WALLETS, band, min_spend_share=MIN_SPEND_SHARE)
        msg = "--" if bb is None else f"{bb['roi']:+.4f} (K={bb['k']}, depth>={bb['depth']})"
        print(f"  mean W1 price [{lo_p:>6.3f},{hi_p:>6.3f}): pool {int(band.sum()):>5}  "
              f"band W2 ROI {pooled_roi(w2['profit'][band], w2['spend'][band]):+.4f}  "
              f"band W2 residual ROI "
              f"{pooled_roi(w2['resid_profit'][band], w2['spend'][band]):+.4f}  "
              f"best region {msg}")

    # ---- 10. the summary table ----
    print("\n--- 10. SUMMARY: every candidate screen, priced on the untouched third window ---")
    top_masks = {}
    for thr in (DETAILS[0], DETAILS[2], DETAILS[3]):
        pool = eligible & (w1["n"] >= thr)
        idx = np.flatnonzero(pool)
        edges = quantile_edges(w1["roi"][idx], N_BINS)
        b = bin_by_edges(np.nan_to_num(w1["roi"][idx], nan=0.0), edges)
        m = np.zeros(d["n_wallets"], dtype=bool)
        m[idx[b == edges.size - 2]] = True
        top_masks[f"top-decile W1 roi, depth>={thr}"] = m
    rows = {
        "BASELINE everyone": masks["everyone"],
        "BASELINE all screenable": eligible,
        "BASELINE certified 26 (IN-SAMPLE)": masks["certified"],
        "certainty screen, W1-only, a=0.005": passes[0.005],
        "certainty screen, W1-only, a=0.05": passes[0.05],
        "certainty screen, W1-only, NO sig test": passes[1.0],
        "2-D argmax region (W2-fitted)": opt_mask,
        **top_masks,
    }
    print(f"  {'screen':>38} {'K':>5} {'W3 pooled ROI':>14} {'W3 95% CI':>20} "
          f"{'W3 eqw':>9} {'W3 $profit':>13} {'W3 effEv':>9}")
    summary = {}
    for name, m in rows.items():
        r3 = region_report(d, m & (w3["n"] >= MIN_TARGET_BETS), d["b2"], d["b3"],
                           rng_b, args.boot, "")
        summary[name] = r3
        print(f"  {name:>38} {r3['wallets']:>5} {r3['roi']:>+14.4f} "
              f"[{r3['ci_lo']:>+.4f},{r3['ci_hi']:>+.4f}] {r3['roi_wallet_eq']:>+9.4f} "
              f"{r3['profit']:>13,.0f} {r3['eff_events']:>9.1f}")

    # ---- 10b. the same table, but each screen made to survive PRICE and COST ----
    print("\n--- 10b. The same screens, price-controlled and cost-charged (W3) ---")
    print("  gross    = the section-10 number.")
    print("  resid    = the same dollars scored on profit over E[outcome|entry_price],")
    print("             i.e. with the favorite-longshot base rate removed.")
    print("  net      = gross minus the REAL per-category taker fee (rate x (1-p) of")
    print("             stake; docs/polymarket_mechanics.md) and one adverse tick.")
    print("  walletCI = the CI when WALLETS are resampled instead of events — the")
    print("             risk that the RULE picked lucky wallets, which the event CI")
    print("             cannot see and which is the risk an operator actually runs.")
    priced = {}
    for name, m in rows.items():
        pr = priced_screen(d, m & (w3["n"] >= MIN_TARGET_BETS), d["b2"], d["b3"],
                           rng_b, args.boot, name)
        priced[name] = pr
        print("  " + fmt_priced(pr))

    # ---- 11. the headline screen, under every control that could kill it ----
    head = "certainty screen, W1-only, a=0.005"
    head_mask = passes[0.005] & (w3["n"] >= MIN_TARGET_BETS)
    print("\n" + "=" * 118)
    print(f"--- 11. THE HEADLINE SCREEN UNDER CONTROL: {head} ---")
    print("=" * 118)

    print("\n  11a. Is it the production certified 26 under another name?")
    prod = set(np.flatnonzero(cert_mask).tolist())
    w1only = set(np.flatnonzero(passes[0.005]).tolist())
    print(f"    production `edge_persisted` set (selected on FULL history, so its W2 "
          f"and W3 are IN-SAMPLE): K={len(prod)}")
    print(f"    W1-only gate at the same alpha (W2/W3 genuinely held out): K={len(w1only)}")
    print(f"    overlap {len(prod & w1only)}; W1-only-not-production {len(w1only - prod)}; "
          f"production-not-W1-only {len(prod - w1only)}")
    print("    ONLY the W1-only number is uncontaminated. The production 26's +0.0385")
    print("    is not evidence of anything: it was chosen with the data it is priced on.")
    both = np.zeros(d["n_wallets"], dtype=bool)
    both[sorted(prod & w1only)] = True
    if both.any():
        print("  " + fmt_priced(priced_screen(d, both & (w3["n"] >= MIN_TARGET_BETS),
                                              d["b2"], d["b3"], rng_b, args.boot,
                                              "  overlap wallets only")))
    onlyw1 = np.zeros(d["n_wallets"], dtype=bool)
    onlyw1[sorted(w1only - prod)] = True
    if onlyw1.any():
        print("  " + fmt_priced(priced_screen(d, onlyw1 & (w3["n"] >= MIN_TARGET_BETS),
                                              d["b2"], d["b3"], rng_b, args.boot,
                                              "  W1-only, NOT production")))

    print("\n  11b. Alpha-ladder fragility. The sets are strictly NESTED (passes[a] =")
    print("       base & p<a), so each row only ADDS wallets to the row above. If the")
    print("       headline is a property of 'certainty' rather than of 17 particular")
    print("       wallets, the number must not swing when the next-most-certain")
    print("       wallets are let in.")
    print(f"    {'alpha':>7} {'K':>4} {'added':>5} {'W3 pooled':>10} {'W3 eqw':>9} "
          f"{'W3 resid':>9} {'W3 net':>9} {'added $stake':>13} {'added ROI':>10} "
          f"{'effEv':>8}")
    prev = np.zeros(d["n_wallets"], dtype=bool)
    for a in alpha_ladder:
        m = passes[a] & (w3["n"] >= MIN_TARGET_BETS)
        if m.sum() == 0:
            continue
        add = m & ~prev
        pr = priced_screen(d, m, d["b2"], d["b3"], rng_b, 200, "")
        _, arows = rows_for(add, d["b2"], d["b3"])
        astake = float(d["spend"][arows].sum()) if arows.size else 0.0
        aroi = (pooled_roi(d["profit"][arows], d["spend"][arows])
                if arows.size else float("nan"))
        eqw = pr["eqw"]
        print(f"    {a:>7} {int(m.sum()):>4} {int(add.sum()):>5} {pr['roi']:>+10.4f} "
              f"{eqw:>+9.4f} {pr['resid']:>+9.4f} {pr['net']:>+9.4f} "
              f"{astake:>13,.0f} {aroi:>+10.4f} {pr['eff_events']:>8.1f}")
        prev = m
    print("    (alpha=0.005 is NOT a tuned choice: it is scoring.project1."
          "oos_significance_alpha,")
    print("     the repo's pre-registered production value. The instability across "
          "neighbouring")
    print("     alphas is therefore a property of the estimate, not evidence of "
          "alpha-mining.)")

    print("\n  11c. PRICE CONTROL — does +gross survive stripping the entry-price effect?")
    hp = priced[head]
    print("  " + fmt_priced(hp))
    print(f"    leave-one-wallet-out W3 pooled ROI spans "
          f"[{hp['loo_lo']:+.4f}, {hp['loo_hi']:+.4f}] over K={hp['wallets']} wallets")
    bands, matched = price_band_table(d, head_mask, d["b2"], d["b3"], eligible,
                                      rng_b, args.boot)
    print("    W3 dollars by ENTRY PRICE, screen vs the same band in the whole "
          "screenable pool:")
    print(f"      {'band':>14} {'bets':>7} {'$stake':>12} {'share':>7} {'gross':>8} "
          f"{'95% CI':>20} {'resid':>8} {'net':>8} {'fee/stake':>10} {'pool same band':>15}")
    tot_s = sum(b["spend"] for b in bands) or 1.0
    for bd in bands:
        if bd["spend"] <= 0:
            continue
        print(f"      [{bd['lo']:.2f},{bd['hi']:.2f}){'':>3} {bd['bets']:>7,} "
              f"{bd['spend']:>12,.0f} {100 * bd['spend'] / tot_s:>6.1f}% "
              f"{bd['roi']:>+8.4f} [{bd['ci_lo']:>+.4f},{bd['ci_hi']:>+.4f}] "
              f"{bd['resid']:>+8.4f} {bd['net']:>+8.4f} {bd['fee_share']:>9.2%} "
              f"{bd['pool_roi']:>+15.4f}")
    print(f"    PRICE-MIMICKING INDEX: the pool's own return, reweighted to the "
          f"screen's price mix = {matched:+.4f}")
    print(f"    screen gross {hp['roi']:+.4f} - price-matched index {matched:+.4f} "
          f"= {hp['roi'] - matched:+.4f} of genuinely non-price excess")

    print("\n  11d. COST CONTROL — the real fee, by category "
          "(fee/stake = rate x (1-p), worst at LOW prices)")
    print(f"      {'category':>14} {'rate':>6} {'bets':>7} {'$stake':>12} {'wt price':>9} "
          f"{'gross':>8} {'resid':>8} {'net':>8} {'fee/stake':>10}")
    for c in category_table(d, head_mask, d["b2"], d["b3"]):
        print(f"      {c['cat']:>14} {c['rate']:>6.2f} {c['bets']:>7,} "
              f"{c['spend']:>12,.0f} {c['mean_price']:>9.3f} {c['roi']:>+8.4f} "
              f"{c['resid']:>+8.4f} {c['net']:>+8.4f} {c['fee_share']:>9.2%}")
    print(f"    blended fee {hp['fee_share']:.2%} of stake + tick {hp['tick_share']:.2%} "
          f"=> gross {hp['roi']:+.4f} becomes net {hp['net']:+.4f} "
          f"({100 * (1 - hp['net'] / hp['roi']) if hp['roi'] else float('nan'):.0f}% of "
          f"the gross edge eaten)")
    print("    CAVEAT: Polymarket was genuinely zero-fee before 2026-01-05, so the")
    print("    historical gross number is what these wallets really earned. The net")
    print("    number is what COPYING THEM TODAY would earn, which is the only")
    print("    number a copy-trading decision may use.")

    print("\n  11d2. COPYABILITY — where the screen's dollars actually live.")
    print("    `src.discover.is_real_world` already defines micro_crypto (the 5-minute")
    print("    BTC/ETH up-down tape) as the UN-COPYABLE slice, and every other arm of")
    print("    this repo excludes it on that ground. If the screen's edge lives there,")
    print("    the edge is real and unreachable — a copier cannot act on a market that")
    print("    resolves in five minutes (docs: edge-decay measured at 1h).")
    _, hrows = rows_for(head_mask, d["b2"], d["b3"])
    _, prows = rows_for(eligible, d["b2"], d["b3"])
    if d["real_world"]:
        # micro_crypto is gone by construction, so the question is no longer
        # "copyable or not" but "is the surviving edge spread across the copyable
        # categories or is it one of them?" — the same concentration test, one
        # level down.
        print("    (under --real-world micro_crypto is excluded by construction, so")
        print("     this splits the screen's dollars by CATEGORY against the same")
        print("     slice of the pool: a screen whose edge is one category is one bet.)")
        cats = np.unique(np.concatenate([d["ccode"][hrows], d["ccode"][prows]])) \
            if hrows.size or prows.size else np.zeros(0, dtype=int)
        for c in cats:
            name = str(d["cat_names"][c])
            hs = hrows[d["ccode"][hrows] == c]
            ps_ = prows[d["ccode"][prows] == c]
            if hs.size == 0:
                continue
            print("  " + fmt_priced(priced_rows(d, hs, rng_b, args.boot,
                                                f"  SCREEN {name}")))
            print("  " + fmt_priced(priced_rows(d, ps_, rng_b, args.boot,
                                                f"  POOL   {name}")))
    else:
        is_micro = d["cat_names"][d["ccode"][hrows]] == UNCOPYABLE_CATEGORY
        p_micro = d["cat_names"][d["ccode"][prows]] == UNCOPYABLE_CATEGORY
        for tag, sub in (("  SCREEN micro_crypto (un-copyable)", hrows[is_micro]),
                         ("  POOL   micro_crypto (same slice)", prows[p_micro]),
                         ("  SCREEN real-world (copyable)", hrows[~is_micro]),
                         ("  POOL   real-world (same slice)", prows[~p_micro])):
            print("  " + fmt_priced(priced_rows(d, sub, rng_b, args.boot, tag)))
    # a CATEGORY-matched index, exactly parallel to the price-matched one above:
    # the pool's own return reweighted to the screen's category mix.
    num = den = 0.0
    for c in np.unique(d["ccode"][hrows]):
        s = float(d["spend"][hrows][d["ccode"][hrows] == c].sum())
        pm = d["ccode"][prows] == c
        ps = float(d["spend"][prows][pm].sum())
        if s > 0 and ps > 0:
            num += s * float(d["profit"][prows][pm].sum()) / ps
            den += s
    cmatch = num / den if den > 0 else np.nan
    print(f"    CATEGORY-MIMICKING INDEX: the pool's return reweighted to the screen's "
          f"category mix = {cmatch:+.4f}")
    print(f"    screen gross {hp['roi']:+.4f} - category-matched index {cmatch:+.4f} "
          f"= {hp['roi'] - cmatch:+.4f}")
    if hrows.size:
        for tag, lo_, hi_ in (("W1 screen", d["b0"], d["b1"]),
                              ("W2 target", d["b1"], d["b2"]),
                              ("W3 honest", d["b2"], d["b3"])):
            t = d["ts"][rows_for(head_mask, lo_, hi_)[1]]
            if t.size:
                print(f"    {tag} window spans "
                      f"{pd.to_datetime(t.min(), unit='s').date()} .. "
                      f"{pd.to_datetime(t.max(), unit='s').date()}")
        print("    Taker fees went live 2026-01-05 and every ledger row is a TAKER leg")
        print("    (docs/polymarket_mechanics.md: one /trades row = one taker's")
        print("    aggregated match). Where W3 sits after that date the NET column is")
        print("    not a projection — it is the correct accounting of what was actually")
        print("    charged, and the gross column overstates what the wallet kept.")

    print("\n  11e. NULLS for this screen (not for the rectangle search).")
    nd_roi, nd_eq, nd_res = [], [], []
    for perm in perms:
        nn = w3["n"][perm]
        mm = passes[0.005] & (nn >= MIN_TARGET_BETS)
        pr_, sp_ = w3["profit"][perm][mm], w3["spend"][perm][mm]
        nd_roi.append(pooled_roi(pr_, sp_))
        nd_res.append(pooled_roi(w3["resid_profit"][perm][mm], sp_))
        ok = sp_ > 0
        nd_eq.append(float(np.mean(pr_[ok] / sp_[ok])) if ok.any() else np.nan)
    for tag, draws, real in (("pooled", nd_roi, hp["roi"]), ("eqw", nd_eq, hp["eqw"]),
                             ("resid", nd_res, hp["resid"])):
        a = np.array(draws, dtype=float)
        a = a[np.isfinite(a)]
        p = (float((a >= real).sum()) + 1.0) / (a.size + 1.0)
        print(f"    null P ({tag}): W1 gate held fixed, W2/W3 records re-attached "
              f"within depth strata")
        print(f"      {a.size} draws: median {np.median(a):+.4f} "
              f"[p5 {np.quantile(a, 0.05):+.4f}, p95 {np.quantile(a, 0.95):+.4f}] "
              f"| REAL {real:+.4f} -> p = {p:.4f}")
    if args.real_world:
        # The headline cell is one cell of an alpha x event-floor grid, and in this
        # population the alpha ladder is where the money moves (§8b-RW). Null-testing
        # only the headline would leave the reader free to point at a neighbouring
        # cell that was never charged for its selection, so every alpha gets the
        # same permutation test. The MULTIPLICITY is the point: 7 alphas x 3
        # statistics is 21 looks, so a single p just under 0.05 here is expected
        # under the null, not evidence.
        print("\n    null P across the WHOLE alpha ladder (same permutations), because")
        print("    in this population the alpha ladder is where the money moves. Note")
        print(f"    the multiplicity: {len(alpha_ladder)} alphas x 3 statistics = "
              f"{3 * len(alpha_ladder)} looks at the same {len(perms)} permutations.")
        print(f"    {'alpha':>7} {'K':>4} {'W3 pooled':>10} {'p':>7} {'W3 eqw':>9} "
              f"{'p':>7} {'W3 median':>10} {'p':>7}")
        for a in alpha_ladder:
            (pl, eq, md), ps = screen_w3_null_p(passes[a], w3, perms)
            k = int((passes[a] & (w3["n"] >= MIN_TARGET_BETS)).sum())
            print(f"    {a:>7} {k:>4} {pl:>+10.4f} {ps[0]:>7.4f} {eq:>+9.4f} "
                  f"{ps[1]:>7.4f} {md:>+10.4f} {ps[2]:>7.4f}")
    if args.shuffles_og > 0:
        print(f"    null O ({args.shuffles_og} shuffles): outcomes permuted within "
              f"entry-price bins and the ENTIRE gate rebuilt on the shuffled tape —")
        print("      the brief's 'surface rebuilt on shuffled outcomes', applied to "
              "the screen that won.")
        og_roi, og_k = [], []
        rng_og = np.random.default_rng(args.seed + 9)
        for s in range(args.shuffles_og):
            e = null_o_gate_dataset(d, rng_og)
            nw1 = window_frame(e, e["b0"], e["b1"], costs=False)
            nw2 = window_frame(e, e["b1"], e["b2"], costs=False)
            nw3 = window_frame(e, e["b2"], e["b3"], costs=False)
            nelig = ((nw1["n"] >= DEPTHS[0]) & (nw1["spend"] > 0)
                     & (nw2["n"] >= MIN_TARGET_BETS))
            npass, _ = certify_on_window(
                e, e["b0"], e["b1"], nelig, (0.005,), min_edge, min_ev,
                int(cfg["scoring"].get("oos_bootstrap_resamples", 2000)),
                args.seed + 100 + s)
            nm = npass[0.005] & (nw3["n"] >= MIN_TARGET_BETS)
            og_k.append(int(nm.sum()))
            og_roi.append(pooled_roi(nw3["profit"][nm], nw3["spend"][nm]))
            del e, nw1, nw2, nw3, npass
            gc.collect()
        a = np.array(og_roi, dtype=float)
        a = a[np.isfinite(a)]
        kk = np.array(og_k, dtype=float)
        pk = (float((kk >= hp["wallets"]).sum()) + 1.0) / (kk.size + 1.0)
        print(f"      HOW MANY it certifies: REAL {hp['wallets']} vs null median "
              f"{np.median(kk):.0f} [{kk.min():.0f},{kk.max():.0f}] -> p = {pk:.4f}")
        if a.size:
            p = (float((a >= hp["roi"]).sum()) + 1.0) / (a.size + 1.0)
            print(f"      HOW WELL they then do: null W3 pooled ROI median "
                  f"{np.median(a):+.4f} [p5 {np.quantile(a, 0.05):+.4f}, "
                  f"p95 {np.quantile(a, 0.95):+.4f}], max {a.max():+.4f} | "
                  f"REAL {hp['roi']:+.4f} -> p = {p:.4f}")
            print("      Read these two together: the null's ROI draw is a handful of")
            print("      wallets, so its ROI is wildly dispersed and comparing point")
            print("      estimates across different K is not like-for-like. The count")
            print("      is the statistic with power here.")
        else:
            print("      null O produced no usable draws (the gate certified nobody).")

    print("\n  11f. What the two CIs disagree about, and the equal-weight alternative.")
    print(f"    event-cluster CI [{hp['ci_lo']:+.4f},{hp['ci_hi']:+.4f}] "
          f"(effEv {hp['eff_events']:.1f}) — 'were its markets lucky?'")
    print(f"    wallet-cluster CI [{hp['wci_lo']:+.4f},{hp['wci_hi']:+.4f}] "
          f"(K={hp['wallets']}, top wallet {100 * hp['top_wallet']:.0f}% of stake) "
          f"— 'was its WALLET DRAW lucky?'")
    print(f"    EQUAL-WEIGHT (equal dollars per wallet, which is what a copier with a")
    print(f"    position cap actually runs): gross {hp['eqw']:+.4f} "
          f"[{hp['eqw_ci_lo']:+.4f},{hp['eqw_ci_hi']:+.4f}] "
          f"resid {hp['eqw_resid']:+.4f} net {hp['eqw_net']:+.4f}")
    print("    A screen is a wallet-selection rule. If the wallet CI covers zero, the")
    print("    rule has not been shown to work no matter how tight the event CI is.")

    print("\n" + "=" * 118)
    print("done.")


if __name__ == "__main__":
    main()
