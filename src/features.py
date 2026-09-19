"""Per-wallet metrics computed from the bet ledger. See CLAUDE.md "Scoring" and
DECISIONS.md for the exact formulas and the reasoning behind each modeling
choice (fair value proxy, earliness vs. copy-window, the two descriptive
flags). Every wallet present in the ledger gets a row here — nothing is
filtered out; `min_sample_size` only controls down-weighting later in
rank.py, never inclusion.
"""

from __future__ import annotations

import ctypes
import gc
import math
from collections import namedtuple

import numpy as np
import pandas as pd

from src.common import WALLET_FEATURES_PATH, ensure_dirs, load_config, load_ledger


def only_buys(ledger: pd.DataFrame) -> pd.DataFrame:
    """Bets = BUY-side fills only (see DECISIONS.md: SELL rows are position
    closes, not fresh entries, and are excluded from entry-skill metrics but
    kept in the ledger untouched)."""
    return ledger.loc[ledger["side"] == "BUY"].copy()


def compute_edge(bets: pd.DataFrame) -> pd.Series:
    """Mean (resolved_value - entry_price) per wallet, over resolved bets only.

    This is CLAUDE.md's literal "edge over entry price". Kept and reported, but
    the audit (see HANDOFF.md / scripts/audit_persistence.py) showed raw edge is
    dominated by the market's favorite-longshot base rate rather than skill, so
    ranking and validation now key on `skill_edge`/residual edge below instead.
    """
    resolved = bets.loc[bets["resolved"]]
    if resolved.empty:
        return pd.Series(dtype=float, name="edge")
    per_bet = resolved["resolved_value"] - resolved["entry_price"]
    return per_bet.groupby(resolved["wallet"]).mean().rename("edge")


# --- favorite-longshot-neutralized ("skill") edge --------------------------
# The market has a structural buyer's edge that varies strongly with entry price
# (favorites at ~0.7 resolve YES more often than 0.7 of the time here; longshots
# at ~0.3 less often). A wallet's *raw* edge therefore mostly reflects which
# price band it habitually buys, not forecasting skill, and that price
# preference is a stable per-wallet trait that trivially "persists" out of
# sample. Subtracting the market-wide calibration curve E[outcome | entry_price]
# leaves the residual: how much the wallet beat the price it actually paid. See
# DECISIONS.md "Favorite-longshot residualization" and HANDOFF.md.

PriceBaseline = namedtuple("PriceBaseline", ["edges", "means", "global_mean"])


def fit_price_baseline(resolved_bets: pd.DataFrame, n_bins: int = 20) -> PriceBaseline:
    """Fit the market-wide calibration curve E[resolved_value | entry_price] over
    equal-count (quantile) entry-price bins across ALL resolved BUY bets. This is
    the structural favorite-longshot edge available to *any* buyer at a price; a
    single wallet contributes negligibly to a market-wide bin, so the full sample
    is used (leave-one-wallet-out would not meaningfully change a bin mean). With
    too little or degenerate data, returns empty bins and `expected_outcome`
    falls back to the global base rate."""
    if resolved_bets.empty:
        return PriceBaseline(np.array([]), np.array([]), 0.0)
    prices = resolved_bets["entry_price"].to_numpy(dtype=float)
    outcomes = resolved_bets["resolved_value"].to_numpy(dtype=float)
    global_mean = float(outcomes.mean())
    if prices.size < n_bins or np.unique(prices).size < 2:
        return PriceBaseline(np.array([]), np.array([]), global_mean)
    edges = np.unique(np.quantile(prices, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return PriceBaseline(np.array([]), np.array([]), global_mean)
    bin_idx = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, edges.size - 2)
    means = np.array([
        outcomes[bin_idx == b].mean() if np.any(bin_idx == b) else global_mean
        for b in range(edges.size - 1)
    ])
    return PriceBaseline(edges, means, global_mean)


def expected_outcome(baseline: PriceBaseline, prices) -> np.ndarray:
    """Structural E[resolved_value | entry_price] per price from a fitted
    baseline. Prices outside the fitted range clamp to the nearest end bin."""
    prices = np.asarray(prices, dtype=float)
    if baseline.edges.size == 0:
        return np.full(prices.shape, baseline.global_mean, dtype=float)
    idx = np.clip(np.searchsorted(baseline.edges, prices, side="right") - 1, 0, baseline.means.size - 1)
    return baseline.means[idx]


def residual_edge_per_bet(bets: pd.DataFrame, baseline: PriceBaseline) -> np.ndarray:
    """Per-bet skill edge = resolved_value - E[resolved_value | entry_price].
    Positive means the wallet beat the market's own price calibration at the
    price it paid — genuine edge rather than the favorite-longshot base rate."""
    return bets["resolved_value"].to_numpy(dtype=float) - expected_outcome(baseline, bets["entry_price"])


def compute_skill_edge(bets: pd.DataFrame, baseline: PriceBaseline) -> pd.Series:
    """Mean residual (favorite-longshot-neutralized) edge per wallet, over
    resolved bets only — the skill-isolating counterpart to `compute_edge`."""
    resolved = bets.loc[bets["resolved"]]
    if resolved.empty:
        return pd.Series(dtype=float, name="skill_edge")
    per_bet = residual_edge_per_bet(resolved, baseline)
    return (
        pd.Series(per_bet, index=resolved["wallet"].to_numpy())
        .groupby(level=0)
        .mean()
        .rename("skill_edge")
    )


def compute_sample_size(bets: pd.DataFrame) -> pd.Series:
    """Count of resolved bets per wallet — the primary sample-size signal used
    to down-weight small samples in rank.py."""
    resolved = bets.loc[bets["resolved"]]
    return resolved.groupby("wallet").size().rename("sample_size")


def compute_breadth(bets: pd.DataFrame) -> pd.Series:
    """Distinct markets a wallet has bet in (any resolution status)."""
    return bets.groupby("wallet")["market_id"].nunique().rename("breadth")


# --- forward drift / fair-value proxy (copy_window, earliness) --------------
# LEAKAGE NOTE (audit 2026-07-18, see DECISIONS.md "Forward-price leakage" and
# HANDOFF.md): "fair value" is proxied by *other wallets'* later trades on the
# same token. As those trades approach resolution their price approaches the 0/1
# outcome, so an unbounded forward proxy is look-ahead — it re-encodes the
# answer. The audit found copy_window correlated ~0.87 with the realized outcome
# and the old resolved_value fallback injected the outcome directly. To stay a
# genuine "how far did price still have to move" signal, the proxy now:
#   1. never falls back to resolved_value (that IS the outcome),
#   2. is bounded strictly to (entry_ts, entry_ts + window] — no unbounded
#      "nearest future trade at any horizon" reach,
#   3. excludes trades in the final `resolution_guard` fraction of the token's
#      observed lifespan, so near-resolution prices can't stand in for fair value.
# No qualifying trade -> NaN (unknown), which contributes nothing and is
# down-weighted, never the outcome.


def _forward_price_arrays(
    ts: np.ndarray,
    price: np.ndarray,
    size: np.ndarray,
    wallet: np.ndarray,
    entry_ts: float,
    exclude_wallet: str,
    window_seconds: float,
    cutoff_ts: float,
) -> float | None:
    """Numpy-array core of `forward_price` (see below) — takes plain arrays so
    callers can extract each token's trades to numpy once and reuse them across
    many bets. Size-weighted mean price of *other* wallets' trades strictly
    inside (entry_ts, entry_ts + window_seconds] and at or before `cutoff_ts`
    (the resolution guard). Returns None when no trade qualifies."""
    mask = (
        (wallet != exclude_wallet)
        & (ts > entry_ts)
        & (ts <= entry_ts + window_seconds)
        & (ts <= cutoff_ts)
    )
    if not mask.any():
        return None
    cprice, csize = price[mask], size[mask]
    weight = csize.sum()
    if weight <= 0:
        return float(cprice.mean())
    return float((cprice * csize).sum() / weight)


def forward_price(
    token_trades: pd.DataFrame,
    entry_ts: float,
    exclude_wallet: str,
    window_seconds: float,
    cutoff_ts: float = np.inf,
) -> float | None:
    """Size-weighted average price of *other* wallets' trades on this token in
    (entry_ts, entry_ts + window_seconds] and at or before `cutoff_ts` (a
    resolution guard; default no guard). Returns None when nothing qualifies —
    there is deliberately no resolved_value or beyond-window fallback, so the
    outcome can never leak in. `token_trades` must have columns [timestamp,
    entry_price, wallet, size] and may include the bet's own and others' trades
    on the same token, any side."""
    return _forward_price_arrays(
        token_trades["timestamp"].to_numpy(),
        token_trades["entry_price"].to_numpy(),
        token_trades["size"].to_numpy(),
        token_trades["wallet"].to_numpy(),
        entry_ts,
        exclude_wallet,
        window_seconds,
        cutoff_ts,
    )


def _factorize_pair(ledger_col: pd.Series, bets_col: pd.Series):
    """Dense integer codes for `ledger_col` and `bets_col` under ONE shared
    value→code mapping (bets is a subset of the ledger), plus the code count.

    `pd.factorize` assigns codes in first-appearance order for object and category
    dtype alike (it operates on the values, not the dtype, and does not materialize
    category strings). The bet side must be mapped through the SAME appearance-order
    uniques — but for a categorical ledger `uniq` comes back as a Categorical whose
    own order is *sorted*, so it is flattened with `np.asarray` before being used as
    `categories=` (otherwise bets would be coded in sorted order while the ledger is
    coded in appearance order — inconsistent). With the flatten, category-typed and
    object-typed ledgers give byte-identical codes → byte-identical drift (the
    prefix-sum accumulation order is preserved too)."""
    codes, uniq = pd.factorize(ledger_col)
    uniq = np.asarray(uniq)
    bc = pd.Categorical(bets_col, categories=uniq).codes.astype(np.int64)
    return codes.astype(np.int64), bc, uniq


def _range_sum(prefix: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Vectorized sum over each half-open slice [lo, hi) given a prefix-sum array
    `prefix` (length N+1, prefix[0]=0, prefix[k]=sum of the first k elements).
    All (lo, hi) are guaranteed by the composite-key construction below to lie
    within a single token/group, so the plain prefix difference is the exact
    within-token range sum."""
    return prefix[hi] - prefix[lo]


def compute_forward_drift_multi(
    bets: pd.DataFrame,
    ledger: pd.DataFrame,
    window_hours_list,
    resolution_guard: float = 0.0,
) -> list:
    """`compute_forward_drift` for SEVERAL windows in one pass, sharing the
    window-independent trade-side machinery (the two argsorts + prefix sums over
    the whole ledger and the per-token guard cutoff) across all windows. Returns
    one Series per entry in `window_hours_list`, each byte-identical to a separate
    `compute_forward_drift` call.

    Only the per-bet upper bound `min(entry + window, guard_cutoff)` and the
    searchsorted `hi`/`ohi` it drives depend on the window; the sort order, prefix
    sums, and lower bounds `lo`/`olo` (`ts > entry_ts`) are built once. `BIG` is
    sized for the LARGEST window — it only scales the composite keys without
    changing their (token, ts) ordering, so each window's result is identical to
    computing it with its own `BIG`. features asks for copy_window + earliness, so
    this halves the per-recompute drift cost and, on a RAM-tight host, collapses
    two swap-thrashing peak phases into one."""
    if ledger.empty or bets.empty:
        return [pd.Series(dtype=float) for _ in window_hours_list]
    return _forward_drift_from_arrays(
        _drift_inputs(bets, ledger), window_hours_list, resolution_guard
    )


def _drift_inputs(bets: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    """Extract exactly the numpy arrays the drift needs from the two frames.

    Split out so a memory-tight caller can free the (far larger) DataFrames before
    the heavy computation: the frames cost ~1.3 GB at 4.7M rows while these arrays
    are ~0.4 GB, and `_forward_drift_from_arrays` never touches the frames again."""
    tok_codes, b_tok, tok_uniq = _factorize_pair(ledger["token_id"], bets["token_id"])
    wal_codes, b_wal, wal_uniq = _factorize_pair(ledger["wallet"], bets["wallet"])
    return {
        "t_ts": ledger["timestamp"].to_numpy().astype(np.int64),
        "t_price": ledger["entry_price"].to_numpy().astype(float),
        "t_size": ledger["size"].to_numpy().astype(float),
        "t_tok": tok_codes,
        "t_wal": wal_codes,
        "n_tok": len(tok_uniq),
        "n_wal": len(wal_uniq),
        "b_ts": bets["timestamp"].to_numpy().astype(np.int64),
        "b_entry": bets["entry_price"].to_numpy().astype(float),
        "b_tok": b_tok,
        "b_wal": b_wal,
        # group by wallet CODE and relabel at the end: avoids carrying 4.4M
        # wallet strings just to be a groupby key (same groups, same values).
        "wallet_values": wal_uniq,
    }


def _arrow_col(path, column):
    """One parquet column as a single (non-chunked) Arrow array."""
    import pyarrow.parquet as pq

    arr = pq.read_table(path, columns=[column]).column(column).combine_chunks()
    if hasattr(arr, "num_chunks"):
        arr = arr.chunk(0) if arr.num_chunks else arr
    return arr


def _arrow_codes(path, column):
    """Appearance-order integer codes + unique values for a string column, read
    straight from parquet without materializing Python strings.

    Arrow dictionary-encodes the column (values stay in Arrow's buffer), then
    pandas factorizes the integer CODES. Because those codes are a 1:1 relabeling
    of the strings, factorizing them in appearance order yields exactly what
    `pd.factorize` would return on the strings — so the drift's sort order, prefix
    sums and results stay byte-identical."""
    enc = _arrow_col(path, column).dictionary_encode()
    idx = enc.indices.to_numpy(zero_copy_only=False)
    vals = np.asarray(enc.dictionary)
    del enc
    codes, uniq_pos = pd.factorize(idx)
    del idx
    return codes.astype(np.int64), vals[np.asarray(uniq_pos)]


def _drift_inputs_arrow() -> dict:
    """`_drift_inputs` without ever building a pandas DataFrame.

    Reads ONE column at a time, so peak memory is the largest single column
    (~0.4 GB for token_id) plus the compact arrays already extracted — instead of
    the ~1.3 GB of ledger+bets frames the pandas path materializes just to pull
    these same arrays out of. Byte-identical (see `_arrow_codes`)."""
    from src.common import BET_LEDGER_PATH as P

    side_codes, side_vals = _arrow_codes(P, "side")
    buy_code = np.flatnonzero(np.asarray(side_vals) == "BUY")
    buy = (side_codes == buy_code[0]) if buy_code.size else np.zeros(side_codes.shape, bool)
    del side_codes, side_vals
    _release_memory()

    t_ts = _arrow_col(P, "timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
    t_price = _arrow_col(P, "entry_price").to_numpy(zero_copy_only=False).astype(float)
    t_size = _arrow_col(P, "size").to_numpy(zero_copy_only=False).astype(float)
    _release_memory()
    tok_codes, tok_uniq = _arrow_codes(P, "token_id")
    _release_memory()
    wal_codes, wal_uniq = _arrow_codes(P, "wallet")
    _release_memory()

    return {
        "t_ts": t_ts,
        "t_price": t_price,
        "t_size": t_size,
        "t_tok": tok_codes,
        "t_wal": wal_codes,
        "n_tok": len(tok_uniq),
        "n_wal": len(wal_uniq),
        "b_ts": t_ts[buy],
        "b_entry": t_price[buy],
        "b_tok": tok_codes[buy],
        "b_wal": wal_codes[buy],
        "wallet_values": wal_uniq,
    }


def _forward_drift_from_arrays(
    inp: dict, window_hours_list, resolution_guard: float = 0.0, batch: int = 1_000_000
) -> list:
    """Array-only core of the forward drift (see compute_forward_drift_multi).

    The trade-side sort + prefix sums are built once and shared across windows.
    The BET side is processed in batches of `batch` rows: every bet's fair value
    is an independent searchsorted + prefix-difference, so batching is exactly
    byte-identical while shrinking the ~20 per-window bet-side intermediates from
    ~700 MB (4.4M rows each) to ~50 MB. Trade-side temporaries are freed as soon
    as they are consumed. Together these keep peak RSS to a fraction of the
    all-at-once version, which is what lets the stage run in ~1 GB RAM."""
    n_bets = int(inp["b_ts"].size)
    if int(inp["t_ts"].size) == 0 or n_bets == 0:
        return [pd.Series(dtype=float) for _ in window_hours_list]

    w_floors = [math.floor(wh * 3600.0) for wh in window_hours_list]

    ts = inp["t_ts"]
    ts_min = ts.min()
    ts0 = ts - ts_min  # shift to 0 so composite keys stay small
    price = inp["t_price"]
    size = inp["t_size"]
    sp = size * price
    tok_codes = inp["t_tok"]
    wal_codes = inp["t_wal"]
    n_tok = inp["n_tok"]
    n_wal = inp["n_wal"]

    span = int(ts0.max())
    BIG = span + max(w_floors) + 1  # sized for the largest window; ordering identical

    # per-token resolution-guard cutoff, floored to an int threshold on ts0
    tok_ts = pd.DataFrame({"tok": tok_codes, "ts0": ts0}).groupby("tok")["ts0"]
    t_min = tok_ts.min().reindex(range(n_tok)).to_numpy()
    t_max = tok_ts.max().reindex(range(n_tok)).to_numpy()
    cutoff_floor = np.floor(t_max - resolution_guard * (t_max - t_min)).astype(np.int64)
    del tok_ts, t_min, t_max

    # window sums: sort trades by (token, ts); prefix sums with a leading 0
    key = tok_codes * BIG + ts0
    o = np.argsort(key, kind="stable")
    K1s = key[o]
    del key
    P1_size = np.concatenate(([0.0], np.cumsum(size[o])))
    P1_sp = np.concatenate(([0.0], np.cumsum(sp[o])))
    P1_price = np.concatenate(([0.0], np.cumsum(price[o])))
    del o

    # own-wallet sums: sort trades by ((token, wallet), ts)
    key = (tok_codes * n_wal + wal_codes) * BIG + ts0
    o = np.argsort(key, kind="stable")
    K2s = key[o]
    del key
    P2_size = np.concatenate(([0.0], np.cumsum(size[o])))
    P2_sp = np.concatenate(([0.0], np.cumsum(sp[o])))
    P2_price = np.concatenate(([0.0], np.cumsum(price[o])))
    del o, price, size, sp, ts0, ts, tok_codes, wal_codes
    for k in ("t_ts", "t_price", "t_size", "t_tok", "t_wal"):
        inp.pop(k, None)
    gc.collect()

    b_ts0 = inp["b_ts"] - ts_min
    b_entry = inp["b_entry"]
    b_tok = inp["b_tok"]
    b_wal = inp["b_wal"]
    b_gwt = b_tok * n_wal + b_wal
    wallet_values = inp["wallet_values"]

    results = []
    for w_floor in w_floors:
        fv = np.full(n_bets, np.nan)
        for start in range(0, n_bets, batch):
            stop = min(start + batch, n_bets)
            bt = b_tok[start:stop]
            bg = b_gwt[start:stop]
            bts = b_ts0[start:stop]
            upper = np.minimum(bts + w_floor, cutoff_floor[bt])

            lo = np.searchsorted(K1s, bt * BIG + bts, side="right")   # ts > entry_ts
            hi = np.searchsorted(K1s, bt * BIG + upper, side="right")  # ts <= upper
            n_win = hi - lo
            win_size = _range_sum(P1_size, lo, hi)
            win_sp = _range_sum(P1_sp, lo, hi)
            win_price = _range_sum(P1_price, lo, hi)

            olo = np.searchsorted(K2s, bg * BIG + bts, side="right")
            ohi = np.searchsorted(K2s, bg * BIG + upper, side="right")
            own_cnt = ohi - olo
            own_size = _range_sum(P2_size, olo, ohi)
            own_sp = _range_sum(P2_sp, olo, ohi)
            own_price = _range_sum(P2_price, olo, ohi)

            # fair value = size-weighted mean of OTHER wallets' in-window trades
            n_other = n_win - own_cnt
            oth_size = win_size - own_size
            oth_sp = win_sp - own_sp
            oth_price = win_price - own_price

            # n_win <= 0 when the window is empty or the bet sits past the guard
            # cutoff; own_cnt <= n_win always, so n_other >= 0 whenever n_win > 0.
            valid = (n_win > 0) & (n_other > 0)
            chunk = np.full(stop - start, np.nan)
            weighted = valid & (oth_size > 0)
            chunk[weighted] = oth_sp[weighted] / oth_size[weighted]
            # degenerate non-positive weight (real ledger sizes are always > 0):
            # fall back to the unweighted mean, matching `_forward_price_arrays`.
            plain = valid & ~(oth_size > 0)
            chunk[plain] = oth_price[plain] / n_other[plain]
            fv[start:stop] = chunk

        # fv is finite exactly where the bet was valid above, so this reproduces
        # the all-at-once `valid` mask (and its element order) exactly.
        keep = ~np.isnan(fv)
        if not keep.any():
            results.append(pd.Series(dtype=float))
            continue
        drift = fv[keep] - b_entry[keep]
        per_wallet = pd.Series(drift, index=b_wal[keep]).groupby(level=0).mean()
        per_wallet.index = pd.Index(wallet_values[per_wallet.index.to_numpy()])
        results.append(per_wallet)
    return results

def compute_forward_drift(
    bets: pd.DataFrame, ledger: pd.DataFrame, window_hours: float, resolution_guard: float = 0.0
) -> pd.Series:
    """Per-bet (fair_value - entry_price), averaged per wallet, where fair value
    is the leakage-guarded forward price (see the note above and `forward_price`).
    `resolution_guard` (0-1) drops trades in the final fraction of each token's
    observed lifespan so near-resolution prices can't proxy fair value. Used for
    both copy_window (long window) and earliness (short window); with the guard
    they only coincide when a market has no trades between the two horizons.

    Fully vectorized — no Python per-token/per-bet loop. The old implementation
    rebuilt an O(trades) boolean mask for every bet (O(bets*trades) per token),
    run twice per recompute over 800k+ bets, which made a re-rank take ~30 min;
    the loop over 57k tiny tokens alone dominated. Here every bet's forward
    window `(entry_ts, entry_ts + window] ∩ (-inf, cutoff]` is located with a
    single vectorized `np.searchsorted` over **integer composite keys**:

      * window (fair-value) sums use trades sorted by `token * BIG + ts`;
      * own-wallet sums (excluded from fair value) use trades sorted by
        `(token, wallet) * BIG + ts`.

    `BIG` exceeds any (shifted ts + window), so a token's/group's key block never
    overlaps its neighbours and each searchsorted range stays within one
    token/group. Timestamps are integer seconds, so `ts <= cutoff_float` is
    exactly `ts <= floor(cutoff)` and `ts <= entry + window` is exactly
    `ts <= entry + floor(window)` — no float boundary drift versus the mask.

    NOTE — numerically equivalent, not bit-identical, to the old mask path. The
    size-weighted sums are formed by differencing prefix sums (accumulated in
    timestamp order) rather than summing each masked subset in original row
    order, so results differ from the mask at the ~1e-8 level (float non-
    associativity). That is far below any decision threshold: copy_window is a
    cents-scale quantity, ranking scores differ by orders of magnitude more, and
    the copyable gate uses a real magnitude floor — verified on the full ledger
    to change no wallet's rank and to flip the copy_window sign only for wallets
    whose value is exactly 0 (~1e-16). `tests/test_features.py` pins agreement
    with the mask reference to a tight tolerance."""
    return compute_forward_drift_multi(
        bets, ledger, [window_hours], resolution_guard
    )[0]


def compute_time_consistency(bets: pd.DataFrame, buckets: int) -> pd.Series:
    """Fraction of chronological equal-count buckets with positive mean edge,
    per wallet. NaN if a wallet has fewer resolved bets than `buckets`."""
    resolved = bets.loc[bets["resolved"]].copy()
    if resolved.empty:
        return pd.Series(dtype=float, name="time_consistency")
    resolved["edge"] = resolved["resolved_value"] - resolved["entry_price"]

    results = {}
    for wallet, group in resolved.groupby("wallet"):
        if len(group) < buckets:
            results[wallet] = np.nan
            continue
        ordered = group.sort_values("timestamp")
        bucket_id = pd.qcut(range(len(ordered)), buckets, labels=False)
        bucket_means = ordered["edge"].groupby(bucket_id).mean()
        results[wallet] = float((bucket_means > 0).mean())
    return pd.Series(results, name="time_consistency")


def compute_market_windows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Observed first/last trade timestamp per market, across all wallets."""
    return ledger.groupby("market_id")["timestamp"].agg(["min", "max"]).rename(
        columns={"min": "market_min_ts", "max": "market_max_ts"}
    )


def compute_manufactured_record_flag(
    ledger: pd.DataFrame, counterparty_share_threshold: float, min_markets: int
) -> pd.Series:
    """Descriptive, non-exclusionary flag: True when a wallet's matched notional
    is concentrated against one recurring counterparty across several markets.
    See DECISIONS.md — this is a concentration heuristic, not evidence of
    common control or intent."""
    # tx_hash is ~unique per row, so a plain groupby("tx_hash") would iterate
    # millions of singleton groups — every one skipped by the len(wallets)!=2
    # guard (a 1-row group has 1 wallet). Only tx_hashes appearing on >=2 rows can
    # pair two wallets, so restrict to those first. Exact: it drops only groups
    # that contribute nothing.
    dup = ledger["tx_hash"].duplicated(keep=False)
    edges = _manufactured_edges(ledger.loc[dup])
    return _manufactured_flags(
        edges, ledger["wallet"].unique(), counterparty_share_threshold, min_markets
    )


def _manufactured_edges(dup_rows: pd.DataFrame) -> dict:
    """Accumulate per-counterparty-pair matched notional + market set over rows
    that share a tx_hash with at least one other row. Split out of
    `compute_manufactured_record_flag` so the streaming path can feed it the same
    rows found a cheaper way, with identical semantics."""
    edges: dict[tuple[str, str], dict] = {}
    for _tx_hash, group in dup_rows.groupby("tx_hash"):
        wallets = group["wallet"].unique()
        if len(wallets) != 2:
            continue
        sides = group.set_index("wallet")["side"]
        if sides.iloc[0] == sides.iloc[1]:
            continue
        w1, w2 = wallets
        notional = float((group["entry_price"] * group["size"]).sum())
        market_id = group["market_id"].iloc[0]
        key = tuple(sorted((w1, w2)))
        entry = edges.setdefault(key, {"notional": 0.0, "markets": set()})
        entry["notional"] += notional
        entry["markets"].add(market_id)
    return edges


def _manufactured_flags(
    edges: dict, wallets, counterparty_share_threshold: float, min_markets: int
) -> pd.Series:
    """Turn the counterparty edges into the per-wallet boolean flag. `wallets` is
    every wallet in the ledger — wallets with no matched notional get False."""
    wallet_totals: dict[str, float] = {}
    wallet_top: dict[str, tuple[float, int]] = {}
    for (w1, w2), info in edges.items():
        for wallet, counterparty in ((w1, w2), (w2, w1)):
            wallet_totals[wallet] = wallet_totals.get(wallet, 0.0) + info["notional"]
            n_markets = len(info["markets"])
            if info["notional"] > wallet_top.get(wallet, (0.0, 0))[0]:
                wallet_top[wallet] = (info["notional"], n_markets)

    flags = {}
    for wallet in wallets:
        total = wallet_totals.get(wallet, 0.0)
        if total <= 0:
            flags[wallet] = False
            continue
        top_notional, top_markets = wallet_top.get(wallet, (0.0, 0))
        share = top_notional / total
        flags[wallet] = bool(share >= counterparty_share_threshold and top_markets >= min_markets)
    return pd.Series(flags, name="manufactured_record_flag")


def compute_pattern_flag(
    bets: pd.DataFrame,
    market_windows: pd.DataFrame,
    late_entry_percentile: float,
    min_bets_for_pattern: int = 5,
    micro_market_seconds: float = 1800,
) -> pd.Series:
    """One descriptive enum per wallet (or None): "late_concentrated_entry" or
    "high_frequency_micro_market". Purely descriptive metadata — see
    DECISIONS.md. Never affects score or rank."""
    joined = bets.merge(market_windows, on="market_id", how="left")
    lifespan = (joined["market_max_ts"] - joined["market_min_ts"]).clip(lower=1e-9)
    joined["relative_entry_time"] = (joined["timestamp"] - joined["market_min_ts"]) / lifespan
    joined["market_lifespan"] = joined["market_max_ts"] - joined["market_min_ts"]

    flags = {}
    for wallet, group in joined.groupby("wallet"):
        flags[wallet] = None
        if len(group) < min_bets_for_pattern:
            continue

        micro_share = (group["market_lifespan"] < micro_market_seconds).mean()
        if micro_share > 0.5:
            flags[wallet] = "high_frequency_micro_market"
            continue

        late_cutoff = 1 - late_entry_percentile
        late = group.loc[group["relative_entry_time"] >= late_cutoff]
        if len(late) >= 3 and late["size"].median() > group["size"].median():
            flags[wallet] = "late_concentrated_entry"

    return pd.Series(flags, name="pattern_flag")


def compute_wallet_features(ledger: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    scoring = cfg["scoring"]
    bets = only_buys(ledger)
    all_wallets = pd.Index(ledger["wallet"].unique(), name="wallet")

    edge = compute_edge(bets)
    baseline = fit_price_baseline(bets.loc[bets["resolved"]], scoring.get("price_baseline_bins", 20))
    skill_edge = compute_skill_edge(bets, baseline)
    sample_size = compute_sample_size(bets)
    breadth = compute_breadth(bets)
    guard = scoring.get("fair_value_resolution_guard", 0.2)
    # One shared pass over the ledger for both windows (see
    # compute_forward_drift_multi): halves the drift cost and, on RAM-tight hosts,
    # avoids a second swap-thrashing peak phase.
    copy_window, earliness = compute_forward_drift_multi(
        bets, ledger,
        [scoring["copy_window_hours"], scoring["earliness_window_hours"]], guard,
    )
    copy_window = copy_window.rename("copy_window")
    earliness = earliness.rename("earliness")
    time_consistency = compute_time_consistency(bets, scoring["time_consistency_buckets"])
    market_windows = compute_market_windows(ledger)
    manufactured_flag = compute_manufactured_record_flag(
        ledger,
        scoring["manufactured_flag_counterparty_share"],
        scoring["manufactured_flag_min_markets"],
    )
    pattern_flag = compute_pattern_flag(
        bets, market_windows, scoring["late_entry_percentile"]
    )

    features = pd.DataFrame(index=all_wallets)
    for series in (edge, skill_edge, sample_size, breadth, copy_window, earliness, time_consistency,
                   manufactured_flag, pattern_flag):
        features = features.join(series)

    features["sample_size"] = features["sample_size"].fillna(0).astype(int)
    features["breadth"] = features["breadth"].fillna(0).astype(int)
    features["manufactured_record_flag"] = features["manufactured_record_flag"].fillna(False)
    features = features.reset_index()
    # `wallet` may arrive as a category (see main()'s memory optimization); the
    # persisted schema must stay plain str for validate.py/rank.py, and the values
    # are identical either way.
    features["wallet"] = features["wallet"].astype(str)
    return features


# Columns features.py actually reads. `outcome`, `question`, `slug` are carried
# in the ledger as metadata but never used here (~480 MB at 4.7M rows), so we
# push a projection down to the parquet reader and never load them.
FEATURE_COLUMNS = [
    "wallet", "market_id", "token_id", "side", "entry_price", "size",
    "timestamp", "resolved", "resolved_value", "tx_hash",
]

# Repetitive string columns collapsed to category dtype before the heavy per-
# wallet computation. At 4.7M rows these dominate the footprint (market_id ~347MB,
# token_id ~399MB, side ~52MB as objects); category is ~5x smaller with identical
# values, so downstream groupby/comparison/factorize/merge are unchanged
# (pandas>=2.1 groupby on a categorical key defaults observed=True == object
# behaviour; _factorize_pair handles category token_id in compute_forward_drift).
# `wallet` is deliberately left as str: it is the output index and a groupby key
# in nearly every metric, and keeping it str avoids categorical-index alignment
# subtleties on the join back to all_wallets for zero real memory cost at output.
# tx_hash is left as str too (unique per row — categorizing gains nothing).
CATEGORICAL_COLS = ["market_id", "token_id", "side"]


def _release_memory() -> None:
    """Return freed memory to the OS between streaming passes.

    `del` only hands a block back to glibc's arena free-lists; the next pass then
    allocates *differently sized* blocks that often don't fit those chunks, so
    malloc requests fresh pages and RSS climbs pass-over-pass (measured: the
    passes stacked to ~3 GB even though no single pass needs more than ~1 GB).
    `malloc_trim(0)` releases the arena's free tail back to the kernel so the next
    pass starts from a low RSS and the peak becomes the largest single pass.
    Best-effort: a no-op on non-glibc platforms."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# Column sets for each streaming pass (see compute_wallet_features_streaming).
# Each pass loads ONLY what its metrics read, so no pass materializes the whole
# ledger. `side` is needed everywhere (only_buys); `wallet` keys every metric.
_PASS_FLAG_COLS = ["tx_hash", "wallet", "side", "entry_price", "size", "market_id"]
_PASS_EDGE_COLS = ["wallet", "side", "resolved", "entry_price", "resolved_value", "timestamp"]
_PASS_MARKET_COLS = ["wallet", "side", "market_id", "timestamp", "size"]
_PASS_DRIFT_COLS = ["wallet", "side", "token_id", "timestamp", "entry_price", "size"]


def _manufactured_flag_streaming(all_wallets, counterparty_share_threshold, min_markets):
    """`manufactured_record_flag` without materializing tx_hash as a pandas frame.

    Only rows that SHARE a tx_hash can contribute a counterparty edge, so first ask
    Arrow whether any tx_hash repeats at all (distinct-count vs row-count) — a
    single scan over one column. If every tx_hash is unique (true on this ledger:
    4.7M distinct over 4.7M rows) there are no edges and every wallet's flag is
    False, so nothing else needs loading.

    This matters because the naive path set the whole stage's peak: loading the six
    flag columns cost ~1.5 GB and pandas' `duplicated()` over 4.7M unique strings
    then spiked a further ~880 MB. Result is identical to
    `compute_manufactured_record_flag`."""
    from src.common import BET_LEDGER_PATH

    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc

        tbl = pq.read_table(BET_LEDGER_PATH, columns=["tx_hash"])
        n_rows = tbl.num_rows
        n_distinct = pc.count_distinct(tbl.column("tx_hash")).as_py()
        del tbl
        _release_memory()
        if n_distinct == n_rows:
            return _manufactured_flags(
                {}, all_wallets, counterparty_share_threshold, min_markets
            )
    except (ImportError, OSError, ValueError):
        pass  # fall through to the general path

    # Rare path: some tx_hashes repeat, so the edge rows are genuinely needed.
    led = load_ledger(columns=_PASS_FLAG_COLS, categorical=["market_id", "side"])
    edges = _manufactured_edges(led.loc[led["tx_hash"].duplicated(keep=False)])
    del led
    _release_memory()
    return _manufactured_flags(
        edges, all_wallets, counterparty_share_threshold, min_markets
    )


def compute_wallet_features_streaming(cfg: dict) -> pd.DataFrame:
    """Memory-bounded equivalent of `compute_wallet_features`, used by main().

    Produces the same result — it calls exactly the same metric functions on
    exactly the same data — but reads the ledger in several column-projected
    passes instead of holding every column (plus the ~93% `only_buys` copy, plus
    the forward-drift arrays) in memory at once.

    Why this shape: glibc does not return freed memory to the OS, so peak RSS is
    set by the largest *momentary* allocation and no later free can lower it.
    Loading all columns together peaked ~2.9 GB at 4.7M rows. Splitting into
    passes makes the peak the largest SINGLE pass (~0.5-0.8 GB) rather than the
    sum, because each pass's freed arena is reused by the next. That is what lets
    the pipeline run in the free e2-micro's 1 GB RAM without swapping.

    `compute_wallet_features` is kept as the in-memory reference implementation
    (tests, validate's standalone fallback, small ledgers). See DECISIONS.md.
    """
    scoring = cfg["scoring"]

    # --- pass 1: per-wallet edge metrics (no market/token/tx_hash needed) ---
    led = load_ledger(columns=_PASS_EDGE_COLS, categorical=["side"])
    if led.empty:
        return pd.DataFrame()
    all_wallets = pd.Index(led["wallet"].unique(), name="wallet")
    bets = only_buys(led)
    del led
    _release_memory()
    edge = compute_edge(bets)
    baseline = fit_price_baseline(
        bets.loc[bets["resolved"]], scoring.get("price_baseline_bins", 20)
    )
    skill_edge = compute_skill_edge(bets, baseline)
    sample_size = compute_sample_size(bets)
    time_consistency = compute_time_consistency(bets, scoring["time_consistency_buckets"])
    del bets
    _release_memory()

    # --- pass 2: manufactured_record_flag (short-circuits when tx_hash is unique) ---
    manufactured_flag = _manufactured_flag_streaming(
        all_wallets,
        scoring["manufactured_flag_counterparty_share"],
        scoring["manufactured_flag_min_markets"],
    )
    _release_memory()

    # --- pass 3: market-scoped metrics (market_windows spans ALL rows) ---
    led = load_ledger(columns=_PASS_MARKET_COLS, categorical=["market_id", "side"])
    market_windows = compute_market_windows(led)
    bets = only_buys(led)
    del led
    _release_memory()
    breadth = compute_breadth(bets)
    pattern_flag = compute_pattern_flag(bets, market_windows, scoring["late_entry_percentile"])
    del bets, market_windows
    _release_memory()

    # --- pass 4: forward drift; both windows share one trade-side pass ---
    # Read the drift columns straight from parquet into numpy, one column at a
    # time — no ledger/bets DataFrames are ever built (they cost ~1.3 GB purely to
    # be unpacked into these arrays). Falls back to the pandas path if Arrow is
    # unavailable or the layout is unexpected.
    try:
        drift_inputs = _drift_inputs_arrow()
    except Exception:
        led = load_ledger(columns=_PASS_DRIFT_COLS, categorical=["token_id", "side"])
        bets = only_buys(led)
        drift_inputs = _drift_inputs(bets, led)
        del led, bets
    _release_memory()
    copy_window, earliness = _forward_drift_from_arrays(
        drift_inputs,
        [scoring["copy_window_hours"], scoring["earliness_window_hours"]],
        scoring.get("fair_value_resolution_guard", 0.2),
    )
    del drift_inputs
    _release_memory()
    copy_window = copy_window.rename("copy_window")
    earliness = earliness.rename("earliness")

    features = pd.DataFrame(index=all_wallets)
    for series in (edge, skill_edge, sample_size, breadth, copy_window, earliness,
                   time_consistency, manufactured_flag, pattern_flag):
        features = features.join(series)

    features["sample_size"] = features["sample_size"].fillna(0).astype(int)
    features["breadth"] = features["breadth"].fillna(0).astype(int)
    features["manufactured_record_flag"] = features["manufactured_record_flag"].fillna(False)
    features = features.reset_index()
    features["wallet"] = features["wallet"].astype(str)
    return features


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    # Streaming (column-projected, pass-at-a-time) so peak RSS is the largest
    # single pass rather than the whole ledger at once — see
    # compute_wallet_features_streaming and DECISIONS.md.
    features = compute_wallet_features_streaming(cfg)
    if features.empty:
        print("[features] ledger is empty — run src.ingest first.")
        return
    WALLET_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(WALLET_FEATURES_PATH, compression="gzip")
    print(f"[features] computed features for {len(features)} wallets -> {WALLET_FEATURES_PATH}")


if __name__ == "__main__":
    main()
