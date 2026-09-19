"""Copy-lag simulation for the certified real-world wallets.

⛔ THE `score` STAGE'S VERDICT IS WITHDRAWN — USE `certified`
------------------------------------------------------------
`score` implements the ORIGINAL run. Its null shuffled outcomes within price bins
among **only the tested wallets' own bets**, so it absorbed the very skill it was
testing for. The tell needs no hindsight: it reports **exactly 0.000 alpha at
Δ = 0**, where a follower paying the identical price for the identical bet simply
IS the wallet. It also charged a 1c adverse tick against markets that quote 0.001,
and it credited the follower at the book MIDPOINT while charging the wallet its
post-slippage taker VWAP (~3.13c/share of unearned edge). Kept, unchanged, as the
historical record behind `docs/copy_lag_simulation.md`; **do not read its numbers**.

The LIVE rule is `stage_certified` (§10 below), which is what
`docs/copy_verdict.md` CORRECTION 2 reports and what the 2026-07-29 red team
re-derived to 4 d.p. It scores against the POPULATION baseline over the whole
real-world tape and charges the spread the wallet demonstrably paid::

    .venv/bin/python -m src.copy_sim certified

Findings that qualify anything this module produces, from
`docs/redteam_realworld_copy_2026-07-29.md`:
  * the POOLED follower edge is real (+1.98c [+1.16, +2.82], and +2.42c on
    held-out bets only), and the wallet's moment is worth +0.83c [+0.47, +1.17]
    once a position-matched control removes the confound;
  * the PER-WALLET ranking is not — under a cluster-preserving cell permutation
    the spread between "copyable" and "uncopyable" wallets has P = 0.23.
    Read the pooled number; treat the per-wallet table as descriptive.

THE QUESTION
------------
A certified wallet buys at time T. A follower sees the fill ~2 minutes later and
buys at whatever the market price is *then*. How much of the wallet's edge
survives that lag, plus the real taker fee and one adverse tick?

Every prior test of this in the repo (``scripts/audit_edge_decay.py``,
``scripts/audit_edge_decay_long.py``) ran on the crypto-dominated global ledger
and came back negative. This module runs it on the specific real-world wallets
that ``src/realworld_validate.py`` certified (``edge_persisted``), which has
never been done.

WHY THE CLOB IS THE ONLY USABLE PRICE SOURCE
--------------------------------------------
The deep tape (``data/interim/realworld/deep_trades/``) holds ONLY our sampled
wallets' own fills, so it cannot say what the market price was two minutes after
one of them traded, and the shared ledger covers only 16-91% of these wallets'
markets. So the price path comes from the public CLOB ``/prices-history``
endpoint (1,000 req / 10 s; see ``docs/polymarket_mechanics.md`` §9). Every
response is cached under ``data/interim/copysim/`` so a re-run never re-fetches.

``/prices-history`` returns a **book-midpoint series on a 1-minute grid** while
the market is open (verified live: a 7-day window at ``fidelity=1`` returns
exactly 10,080 points). That is a much denser follower-price proxy than the
trade tape, but it is a MIDPOINT, and a follower buys the ask — which is why one
adverse tick is charged on top of the fee.

DISCIPLINE (copied from scripts/audit_edge_decay.py, which already solved these)
-------------------------------------------------------------------------------
1. **Bounded window only.** The follower price is the first grid point in
   ``[entry_ts + Δ, entry_ts + Δ + bandwidth]``. No unbounded "nearest future
   price" reach. If nothing qualifies, the follower could not have entered:
   the result is NaN, **never** ``resolved_value``.
2. **Resolution guard.** Prices in the final ``scoring.fair_value_resolution_guard``
   fraction of the market's lifespan are excluded, so a near-resolution price
   (which approaches the 0/1 answer) can never stand in for a follower price.
   Lifespan comes from Gamma (``startDate`` -> ``umaEndDate``) because the CLOB
   caps a ``/prices-history`` query at <30 days and so cannot show a long
   market's full series.
3. **Same-cohort comparison.** Which bets even HAVE a price at Δ is
   selection-biased (only longer-lived, liquid markets survive to large Δ), so
   the wallet's own Δ=0 return is ALWAYS recomputed on exactly the bets
   measurable at that Δ — never against the full-sample number.
4. **Shuffled-outcome null.** ``resolved_value`` is permuted *within entry-price
   bins* and the whole thing re-run. Whatever survives the shuffle is the
   favorite-longshot base rate available to anyone at that price, not copyable
   alpha. Reported as ``null`` and ``real - null``.
5. **Real fees.** ``fee = shares × k × p × (1−p)``, taker-only, with k read from
   Gamma's ``feeSchedule.rate`` per market and falling back to the category map
   (0.04 politics/finance/tech · 0.05 sports/general · 0.07 crypto ·
   0 geopolitics). Charged on the follower's fill, plus one adverse tick.
   Both GROSS and NET are reported.

READ-ONLY. Public unauthenticated GETs only; no keys, no signing, no orders.
Writes only to ``data/interim/copysim/`` and ``docs/copy_lag_simulation.md``.

Usage::

    .venv/bin/python -m src.copy_sim fetch     # pull + cache price histories (resumable)
    .venv/bin/python -m src.copy_sim score     # compute the per-bet frame + summary
    .venv/bin/python -m src.copy_sim report    # write docs/copy_lag_simulation.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common import (
    INTERIM_DIR,
    REPO_ROOT,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
)

# --------------------------------------------------------------------------
# Paths — everything this module writes lives under data/interim/copysim/.
# --------------------------------------------------------------------------
REALWORLD_DIR = INTERIM_DIR / "realworld"
VALIDATED_PATH = REALWORLD_DIR / "validated.parquet"
DEEP_TRADES_DIR = REALWORLD_DIR / "deep_trades"
MARKET_CATEGORY_PATH = REALWORLD_DIR / "market_category.parquet"

COPYSIM_DIR = INTERIM_DIR / "copysim"
BETS_PATH = COPYSIM_DIR / "bets.parquet"
TOKENS_PATH = COPYSIM_DIR / "tokens.parquet"
MARKET_TIMES_PATH = COPYSIM_DIR / "market_times.parquet"
PRICES_DIR = COPYSIM_DIR / "prices"
FETCH_STATE_PATH = COPYSIM_DIR / "fetch_state.json"
PER_BET_PATH = COPYSIM_DIR / "per_bet.parquet"
SUMMARY_PATH = COPYSIM_DIR / "summary.parquet"
REPORT_PATH = REPO_ROOT / "docs" / "copy_lag_simulation.md"

# --------------------------------------------------------------------------
# The five certified real-world wallets under test (address prefixes; the full
# addresses are resolved from validated.parquet where edge_persisted is True).
# --------------------------------------------------------------------------
WALLET_PREFIXES = [
    "0xe542afd388",
    "0x83255595ba",
    "0x1ee9a5fc09",
    "0x69ea0d77ef",
    "0x09bed19766",
]

SEED = 20260728          # fixed; the pipeline forbids clock/random seeding
N_BOOT = 2000            # market-cluster bootstrap resamples
N_SHUFFLE = 200          # outcome permutations for the null
BOOT_BLOCK = 250         # bootstrap resamples per chunk (RAM guard: 3 GB box)

# Δ sweep (seconds). Δ=0 is the wallet's own fill — the ceiling.
DELTAS: list[tuple[int, str]] = [
    (0, "0"),
    (120, "2m"),
    (900, "15m"),
    (3600, "1h"),
    (21600, "6h"),
]
MAX_DELTA = max(d for d, _ in DELTAS)

DEFAULT_BANDWIDTH = 120        # s: follower fill window after entry_ts + Δ
BANDWIDTH_SENSITIVITY = [60, 120, 300]
DEFAULT_ADVERSE_TICK = 0.01    # follower buys the ask, not the midpoint
TICK_SENSITIVITY = [0.0, 0.001, 0.01]

FETCH_PAD = 60                 # s of slack on each side of a planned window
MERGE_GAP = 12 * 3600          # s: bets on one token within this share a request
GAMMA_BATCH = 40               # condition_ids per Gamma /markets request
PRICE_CHUNK_REQUESTS = 400     # requests per cached price chunk file
FETCH_WORKERS = 4
FETCH_MAX_RPS = 20.0           # 2% of the documented 1,000 req / 10 s ceiling

CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Per-category taker fee rate, used only when Gamma has no feeSchedule for the
# market. Source: docs/polymarket_mechanics.md §5 (verified on-chain to 7 s.f.).
CATEGORY_FEE_K: dict[str, float] = {
    "micro_crypto": 0.07,
    "crypto_event": 0.07,
    "politics": 0.04,
    "econ_macro": 0.05,
    "culture": 0.05,
    "geopolitics": 0.0,
    "other": 0.05,          # Gamma's `general_fees`
}
SPORTS_FEE_K = 0.05
DEFAULT_FEE_K = 0.05


# ==========================================================================
# 1. Wallet / bet loading
# ==========================================================================
def resolve_wallets(prefixes: list[str] | None = None,
                    validated: pd.DataFrame | None = None) -> list[str]:
    """Full addresses of the certified (``edge_persisted``) wallets matching the
    given prefixes. Raises if a prefix does not resolve to exactly one wallet —
    a silent mis-resolution would test the wrong trader."""
    prefixes = list(WALLET_PREFIXES if prefixes is None else prefixes)
    if validated is None:
        validated = pd.read_parquet(VALIDATED_PATH)
    certified = validated.loc[validated["edge_persisted"].fillna(False)]
    out = []
    for p in prefixes:
        hits = certified.loc[certified["wallet"].str.startswith(p), "wallet"].tolist()
        if len(hits) != 1:
            raise ValueError(f"prefix {p!r} matched {len(hits)} certified wallets: {hits}")
        out.append(hits[0])
    return out


def load_bets(wallets: list[str], trades_dir: Path | None = None) -> pd.DataFrame:
    """Resolved BUY bets for the given wallets, streamed part-by-part with column
    projection and a pushed-down wallet filter (the deep tape is ~3.8M rows and
    this box has 3 GB of RAM)."""
    trades_dir = Path(trades_dir or DEEP_TRADES_DIR)
    cols = ["wallet", "market_id", "token_id", "outcome", "side",
            "entry_price", "size", "timestamp", "resolved", "resolved_value", "slug"]
    parts = []
    for f in sorted(glob.glob(str(trades_dir / "part-*.parquet"))):
        d = pq.read_table(f, columns=cols,
                          filters=[("wallet", "in", wallets),
                                   ("side", "==", "BUY"),
                                   ("resolved", "==", True)]).to_pandas()
        if len(d):
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop(columns=["side", "resolved"])
    df = df.loc[df["entry_price"].between(1e-6, 1 - 1e-6)]
    df = df.dropna(subset=["resolved_value"])
    df["timestamp"] = df["timestamp"].astype("int64")
    # NB the projection + pushed-down filters above are what keep this cheap: the
    # deep tape is 3.8M rows and reading `market_id`/`token_id` for all of them as
    # python objects costs ~600 MB and has OOM'd this box before. Post-filter the
    # frame is ~19k rows, so plain str columns are fine from here on.
    for c in ("wallet", "market_id", "token_id", "outcome", "slug"):
        df[c] = df[c].astype(str)
    return df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)


def attach_categories(bets: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Additive `category` column from the slug classifier sidecar (never drops
    rows; unknown markets become 'other', which carries Gamma's general rate)."""
    path = Path(path or MARKET_CATEGORY_PATH)
    if not path.exists():
        bets = bets.copy()
        bets["category"] = "other"
        return bets
    mc = pq.read_table(path, columns=["market_id", "category"]).to_pandas()
    out = bets.merge(mc, on="market_id", how="left")
    out["category"] = out["category"].fillna("other")
    return out


# ==========================================================================
# 2. Fees
# ==========================================================================
def fee_k_for(category: str | None, gamma_rate: float | None = None) -> float:
    """Taker fee rate k. Gamma's per-market ``feeSchedule.rate`` wins when known
    (it is the only correct source, per docs/polymarket_mechanics.md §5); the
    category map is the fallback."""
    if gamma_rate is not None and not (isinstance(gamma_rate, float) and math.isnan(gamma_rate)):
        return float(gamma_rate)
    cat = (category or "other")
    if cat.startswith("sports"):
        return SPORTS_FEE_K
    return CATEGORY_FEE_K.get(cat, DEFAULT_FEE_K)


def taker_fee_per_share(price: float | np.ndarray, k: float | np.ndarray,
                        exponent: float | np.ndarray = 1.0):
    """``fee = shares × k × (p × (1−p))^exponent``, expressed per share.

    ``exponent`` is 1 for every schedule the mechanics doc sampled, but Gamma's
    ``crypto_15_min`` schedule uses ``{"rate": 0.25, "exponent": 2}`` — a
    steeper, narrower parabola. Hardcoding exponent 1 there would charge 0.25
    at the money, ~16x the real fee, so it is read per market."""
    return k * np.power(np.asarray(price, dtype=float) * (1.0 - np.asarray(price, dtype=float)),
                        exponent)


def follower_cost(price, k, adverse_tick: float = DEFAULT_ADVERSE_TICK,
                  exponent: float | np.ndarray = 1.0):
    """All-in cost per share for a follower lifting the offer: the observed
    midpoint, plus one adverse tick (the midpoint is not an executable ask),
    plus the taker fee on the resulting fill price."""
    eff = np.clip(np.asarray(price, dtype=float) + adverse_tick, 1e-6, 1.0 - 1e-9)
    return eff + taker_fee_per_share(eff, k, exponent)


def roi(resolved_value, cost):
    """(payout − cost) / cost, per share."""
    cost = np.asarray(cost, dtype=float)
    out = np.full(cost.shape, np.nan)
    ok = cost > 0
    out[ok] = (np.asarray(resolved_value, dtype=float)[ok] - cost[ok]) / cost[ok]
    return out


# ==========================================================================
# 3. Price fetching (resumable + cached)
# ==========================================================================
MAX_REQUEST_SPAN = 6 * 86400   # the endpoint 400s on long ranges; 7d works, 30d does not


def plan_requests(bets: pd.DataFrame, horizon: int = MAX_DELTA + max(BANDWIDTH_SENSITIVITY),
                  merge_gap: int = MERGE_GAP, pad: int = FETCH_PAD,
                  max_span: int = MAX_REQUEST_SPAN) -> list[dict]:
    """One ``/prices-history`` request per (token, contiguous cluster of entries).

    Every bet needs prices over ``[entry, entry + horizon]``; bets on the same
    token whose entries are within ``merge_gap`` share a single request, which
    roughly halves the request count. A cluster is also closed whenever the
    window would exceed ``max_span``, because the endpoint rejects long ranges
    (verified live: 7 days OK, 30 days -> 400 "interval is too long")."""
    reqs = []
    tok = bets[["token_id", "timestamp"]].copy()
    tok["token_id"] = tok["token_id"].astype(str)
    for token_id, g in tok.groupby("token_id", sort=True):
        ts = np.sort(g["timestamp"].to_numpy(dtype=np.int64))
        start = prev = ts[0]

        def _emit(s, p):
            reqs.append({"token_id": token_id, "start": int(s - pad),
                         "end": int(p + horizon + pad)})

        for t in ts[1:]:
            if t - prev > merge_gap or (t + horizon + pad) - (start - pad) > max_span:
                _emit(start, prev)
                start = t
            prev = t
        _emit(start, prev)
    return reqs


def request_key(req: dict) -> str:
    return f"{req['token_id']}:{req['start']}:{req['end']}"


def _load_fetch_state() -> dict:
    if FETCH_STATE_PATH.exists():
        with open(FETCH_STATE_PATH) as f:
            return json.load(f)
    return {"done": []}


class _RateLimiter:
    """Simple wall-clock throttle shared by the fetch workers."""

    def __init__(self, max_rps: float):
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._next = 0.0

    def wait(self) -> None:
        import threading
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                delay = self._next - now
            else:
                delay = 0.0
                self._next = now
            self._next += self.min_interval
        if delay > 0:
            time.sleep(delay)


def fetch_price_window(session, token_id: str, start: int, end: int,
                       fidelity: int = 1, limiter: _RateLimiter | None = None,
                       max_attempts: int = 5) -> list[dict]:
    """One ``/prices-history`` call, with backoff. Returns the raw history list
    (possibly empty — a market with no book activity in the window)."""
    params = {"market": token_id, "startTs": int(start), "endTs": int(end),
              "fidelity": fidelity}
    for attempt in range(max_attempts):
        if limiter is not None:
            limiter.wait()
        try:
            r = session.get(CLOB_PRICES_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("history", []) or []
            if r.status_code == 400:
                return []          # bad/expired token window — not retryable
        except Exception:
            pass
        time.sleep(min(30.0, 1.5 * (2 ** attempt)))
    return []


def run_fetch(bets: pd.DataFrame, tokens: pd.DataFrame, workers: int = FETCH_WORKERS,
              max_rps: float = FETCH_MAX_RPS, limit: int | None = None) -> None:
    """Pull every planned price window, caching to ``prices/chunk-*.parquet`` and
    recording completed request keys so a re-run never re-fetches."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    tok_ix = dict(zip(tokens["token_id"].astype(str), tokens["tok_ix"].astype(int)))
    state = _load_fetch_state()
    done = set(state["done"])
    plan = [r for r in plan_requests(bets) if request_key(r) not in done]
    if limit:
        plan = plan[:limit]
    print(f"[copysim/fetch] {len(plan)} price windows pending "
          f"({len(done)} already cached)", flush=True)
    if not plan:
        return
    session = make_session()
    limiter = _RateLimiter(max_rps)
    chunk_no = len(list(PRICES_DIR.glob("chunk-*.parquet")))
    rows_t: list[np.ndarray] = []
    rows_p: list[np.ndarray] = []
    rows_i: list[np.ndarray] = []
    completed: list[str] = []
    t0 = time.time()

    def _flush():
        nonlocal chunk_no, rows_t, rows_p, rows_i, completed
        if rows_t:
            df = pd.DataFrame({
                "tok_ix": np.concatenate(rows_i).astype("int32"),
                "t": np.concatenate(rows_t).astype("int64"),
                "p": np.concatenate(rows_p).astype("float32"),
            })
            atomic_to_parquet(df, PRICES_DIR / f"chunk-{chunk_no:05d}.parquet",
                              compression="zstd", index=False)
            chunk_no += 1
        rows_t, rows_p, rows_i = [], [], []
        done.update(completed)
        completed = []
        atomic_write_json({"done": sorted(done)}, FETCH_STATE_PATH)

    def _one(req):
        h = fetch_price_window(session, req["token_id"], req["start"], req["end"],
                               limiter=limiter)
        return req, h

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (req, hist) in enumerate(pool.map(_one, plan), start=1):
            if hist:
                ts = np.fromiter((h["t"] for h in hist), dtype=np.int64, count=len(hist))
                ps = np.fromiter((h["p"] for h in hist), dtype=np.float64, count=len(hist))
                rows_t.append(ts)
                rows_p.append(ps)
                rows_i.append(np.full(len(hist), tok_ix[req["token_id"]], dtype=np.int32))
            completed.append(request_key(req))
            if n % PRICE_CHUNK_REQUESTS == 0:
                _flush()
                rate = n / max(1e-9, time.time() - t0)
                eta = (len(plan) - n) / max(1e-9, rate) / 60
                print(f"[copysim/fetch] {n}/{len(plan)} ({rate:.1f} req/s, "
                      f"ETA {eta:.1f} min)", flush=True)
    _flush()
    print(f"[copysim/fetch] done in {(time.time()-t0)/60:.1f} min", flush=True)


def fetch_market_times(market_ids: list[str], session=None) -> pd.DataFrame:
    """Per-market lifespan + fee rate from Gamma, batched by ``condition_ids``.

    ``umaEndDate`` is the actual resolution timestamp (== ``closedTime``, 300/300
    live) — ``endDate``/``endDateIso`` are midnight-floored dates and must not be
    used as boundaries (docs/polymarket_mechanics.md §6)."""
    session = session or make_session()
    have: dict[str, dict] = {}
    if MARKET_TIMES_PATH.exists():
        prev = pd.read_parquet(MARKET_TIMES_PATH)
        have = {r["market_id"]: r for r in prev.to_dict("records")}
    todo = [m for m in market_ids if m not in have]
    print(f"[copysim/gamma] {len(todo)} markets to describe "
          f"({len(have)} cached)", flush=True)
    for i in range(0, len(todo), GAMMA_BATCH):
        batch = todo[i:i + GAMMA_BATCH]
        params = [("condition_ids", c) for c in batch]
        params += [("closed", "true"), ("limit", str(GAMMA_BATCH * 2))]
        try:
            r = session.get(GAMMA_MARKETS_URL, params=params, timeout=40)
            payload = r.json() if r.status_code == 200 else []
        except Exception:
            payload = []
        for m in payload:
            sched = m.get("feeSchedule") or {}
            have[m.get("conditionId")] = {
                "market_id": m.get("conditionId"),
                "start_ts": _parse_ts(m.get("startDate") or m.get("createdAt")),
                "end_ts": _parse_ts(m.get("umaEndDate") or m.get("closedTime")
                                    or m.get("endDate")),
                "gamma_fee_rate": (float(sched["rate"]) if sched.get("rate") is not None
                                   else np.nan),
                "gamma_fee_exponent": float(sched.get("exponent") or 1.0),
                "tick_size": float(m.get("orderPriceMinTickSize") or np.nan),
            }
        if (i // GAMMA_BATCH) % 20 == 0:
            print(f"[copysim/gamma] {i + len(batch)}/{len(todo)}", flush=True)
        time.sleep(0.05)
    df = pd.DataFrame(list(have.values()))
    if len(df):
        atomic_to_parquet(df, MARKET_TIMES_PATH, compression="zstd", index=False)
    return df


def _parse_ts(value) -> float:
    """Epoch seconds from any of Gamma's timestamp spellings. `closedTime` comes
    back as `'2026-07-20 01:24:43+00'` (space separator, `+00` offset) while every
    other field uses `...T...Z` — see docs/polymarket_mechanics.md §6."""
    if not value:
        return float("nan")
    try:
        ts = pd.Timestamp(str(value).replace(" ", "T"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return float(ts.timestamp())
    except Exception:
        return float("nan")


# ==========================================================================
# 4. The follower-price core
# ==========================================================================
def guard_cutoff(start_ts: float, end_ts: float, guard: float) -> float:
    """Timestamp past which a price is too close to resolution to stand in for a
    follower price: the final ``guard`` fraction of the market's lifespan is
    excluded. NaN when the lifespan is unknown — callers must then treat every
    price as unusable rather than guess."""
    if not np.isfinite(start_ts) or not np.isfinite(end_ts) or end_ts <= start_ts:
        return float("nan")
    return float(end_ts - guard * (end_ts - start_ts))


def price_at(ts: np.ndarray, prices: np.ndarray, target: float,
             bandwidth: float, cutoff: float) -> float:
    """First observed price in ``[target, target + bandwidth]`` that is also at or
    before the resolution-guard cutoff.

    Returns NaN when no such price exists — the follower simply could not have
    entered. It NEVER falls back to ``resolved_value`` or to an unbounded
    "nearest future price": both would silently import the answer.

    A NaN ``cutoff`` means the market's lifespan is unknown, so the guard cannot
    be verified and every price is refused rather than guessed. ``inf`` is the
    explicit "no guard" value, used only by the diagnostic midpoint lookup."""
    if ts.size == 0 or np.isnan(cutoff):
        return float("nan")
    lo = int(np.searchsorted(ts, target, side="left"))
    if lo >= ts.size:
        return float("nan")
    t = float(ts[lo])
    if t > target + bandwidth or t > cutoff:
        return float("nan")
    return float(prices[lo])


def load_price_store(prices_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All cached price points as three parallel arrays sorted by (tok_ix, t),
    plus the per-token offsets. int32/float32 throughout — an object-dtype
    token column here would be hundreds of MB on a 3 GB box."""
    prices_dir = Path(prices_dir or PRICES_DIR)
    files = sorted(prices_dir.glob("chunk-*.parquet"))
    if not files:
        return np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.float32)
    ix, ts, ps = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["tok_ix", "t", "p"])
        ix.append(t.column("tok_ix").to_numpy().astype(np.int32))
        ts.append(t.column("t").to_numpy().astype(np.int64))
        ps.append(t.column("p").to_numpy().astype(np.float32))
    ix = np.concatenate(ix)
    ts = np.concatenate(ts)
    ps = np.concatenate(ps)
    order = np.lexsort((ts, ix))
    return ix[order], ts[order], ps[order]


def token_slices(tok_ix_sorted: np.ndarray, n_tokens: int) -> np.ndarray:
    """Start offset of each token's block in the sorted price store (length
    n_tokens+1, so block i is [out[i], out[i+1]))."""
    counts = np.bincount(tok_ix_sorted, minlength=n_tokens)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


# ==========================================================================
# 5. Per-bet computation
# ==========================================================================
def compute_per_bet(bets: pd.DataFrame, store, offsets: np.ndarray,
                    deltas=DELTAS, bandwidth: int = DEFAULT_BANDWIDTH,
                    adverse_tick: float = DEFAULT_ADVERSE_TICK) -> pd.DataFrame:
    """Attach ``price_d{Δ}`` (the follower's fill midpoint) for every Δ.

    ``bets`` must carry ``tok_ix``, ``cutoff`` and ``fee_k``. Δ=0 is not looked
    up: it is the wallet's own recorded fill."""
    ix_s, ts_s, p_s = store
    out = bets.copy()
    tok = out["tok_ix"].to_numpy(dtype=np.int64)
    entry = out["timestamp"].to_numpy(dtype=np.float64)
    cut = out["cutoff"].to_numpy(dtype=np.float64)
    for delta, label in deltas:
        if delta == 0:
            continue
        col = np.full(len(out), np.nan)
        for i in range(len(out)):
            a, b = offsets[tok[i]], offsets[tok[i] + 1]
            col[i] = price_at(ts_s[a:b], p_s[a:b], entry[i] + delta, bandwidth, cut[i])
        out[f"price_{label}"] = col
    # Diagnostic only, NEVER scored: the midpoint at the moment of entry. The tape's
    # `entry_price` is a post-slippage VWAP across 2-8+ fills, so |entry − midpoint|
    # measures how far above the mid these wallets actually filled — which is the
    # scale of the adverse-tick assumption charged to the follower. The resolution
    # guard is deliberately NOT applied here: this is a property of our own tape at
    # the instant of entry, not a forward price, so there is no outcome to leak.
    mid0 = np.full(len(out), np.nan)
    for i in range(len(out)):
        a, b = offsets[tok[i]], offsets[tok[i] + 1]
        mid0[i] = price_at(ts_s[a:b], p_s[a:b], entry[i] - 60, 120, np.inf)
    out["mid_at_entry"] = mid0
    # Gross / net returns at every Δ, including the wallet's own fill.
    k = out["fee_k"].to_numpy(dtype=float)
    exp = (out["fee_exponent"].to_numpy(dtype=float) if "fee_exponent" in out
           else np.ones(len(out)))
    rv = out["resolved_value"].to_numpy(dtype=float)
    for delta, label in deltas:
        px = (out["entry_price"].to_numpy(dtype=float) if delta == 0
              else out[f"price_{label}"].to_numpy(dtype=float))
        cost = follower_cost(px, k, adverse_tick, exp)
        out[f"gross_{label}"] = roi(rv, np.where(px > 0, px, np.nan))
        out[f"net_{label}"] = roi(rv, cost)
        out[f"edge_{label}"] = rv - px                       # per share, gross
        out[f"netedge_{label}"] = rv - cost
    return out


# ==========================================================================
# 6. Statistics — cluster bootstrap and shuffled-outcome null
# ==========================================================================
def cluster_boot_ci(values: np.ndarray, market_codes: np.ndarray,
                    rng: np.random.Generator, n: int = N_BOOT) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling whole MARKETS with replacement.

    Bets inside one market share a single resolution event, so the unit of
    independence is the market, not the bet (docs/persistence_cluster_recheck.md).
    Implemented on per-market sums/counts so the resample never materializes the
    bet-level frame."""
    ok = ~np.isnan(values)
    values, market_codes = values[ok], market_codes[ok]
    if values.size < 2:
        return (float("nan"), float("nan"))
    codes, inv = np.unique(market_codes, return_inverse=True)
    g = codes.size
    if g < 2:
        return (float("nan"), float("nan"))
    sums = np.bincount(inv, weights=values, minlength=g)
    cnts = np.bincount(inv, minlength=g).astype(float)
    means = np.empty(n)
    for i in range(0, n, BOOT_BLOCK):
        b = min(BOOT_BLOCK, n - i)
        idx = rng.integers(0, g, size=(b, g))
        means[i:i + b] = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def price_bins(entry_price: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Quantile bin index of each bet's entry price (the null permutes outcomes
    *within* these, so the favorite-longshot base rate is preserved)."""
    if entry_price.size == 0:
        return np.zeros(0, dtype=int)
    qs = np.unique(np.quantile(entry_price, np.linspace(0, 1, n_bins + 1)))
    if qs.size < 2:
        return np.zeros(entry_price.size, dtype=int)
    return np.clip(np.digitize(entry_price, qs[1:-1], right=False), 0, qs.size - 2)


def shuffle_within_bins(values: np.ndarray, bins: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Permute ``values`` inside each bin, leaving the bin composition intact."""
    out = values.copy()
    for b in np.unique(bins):
        m = bins == b
        out[m] = rng.permutation(values[m])
    return out


def shuffled_null(resolved_value: np.ndarray, cohort: np.ndarray, cost: np.ndarray,
                  bins: np.ndarray, real_roi: float, real_edge: float,
                  rng: np.random.Generator,
                  n_shuffle: int = N_SHUFFLE) -> dict:
    """Follower performance under permuted outcomes, for both metrics at once.

    Permuting ``resolved_value`` within entry-price bins destroys the wallet's
    selection while preserving the base rate available to *anyone* buying at that
    price, so whatever survives the shuffle is not copyable alpha. Returns the
    null means plus one-sided empirical p-values (share of shuffles whose null
    mean >= the real mean). Both metrics share the same shuffles, so their p's
    are directly comparable."""
    c = cost[cohort]
    nulls_roi = np.empty(n_shuffle)
    nulls_edge = np.empty(n_shuffle)
    for i in range(n_shuffle):
        sh = shuffle_within_bins(resolved_value, bins, rng)[cohort]
        nulls_roi[i] = float(np.nanmean((sh - c) / c))
        nulls_edge[i] = float(np.nanmean(sh - c))
    return {
        "null_net": float(nulls_roi.mean()),
        "null_p": float((nulls_roi >= real_roi).mean()),
        "null_netedge_c": float(nulls_edge.mean()),
        "null_p_edge": float((nulls_edge >= real_edge).mean()),
    }


# ==========================================================================
# 7. Summary tables
# ==========================================================================
def summarize(per_bet: pd.DataFrame, label: str, rng: np.random.Generator,
              deltas=DELTAS, adverse_tick: float = DEFAULT_ADVERSE_TICK,
              n_shuffle: int = N_SHUFFLE) -> pd.DataFrame:
    """One decay table. Every row is SAME-COHORT: the wallet's own Δ=0 return is
    recomputed on exactly the bets that have a usable follower price at that Δ."""
    rv = per_bet["resolved_value"].to_numpy(dtype=float)
    ep = per_bet["entry_price"].to_numpy(dtype=float)
    k = per_bet["fee_k"].to_numpy(dtype=float)
    exp = (per_bet["fee_exponent"].to_numpy(dtype=float) if "fee_exponent" in per_bet
           else np.ones(len(per_bet)))
    mk = pd.factorize(per_bet["market_id"].astype(str))[0]
    bins = price_bins(ep)
    own_all_gross = per_bet["gross_0"].to_numpy(dtype=float)
    own_all_net = per_bet["net_0"].to_numpy(dtype=float)
    rows = []
    for delta, dlabel in deltas:
        if delta == 0:
            cohort = ~np.isnan(own_all_gross)
        else:
            cohort = ~np.isnan(per_bet[f"price_{dlabel}"].to_numpy(dtype=float))
        n = int(cohort.sum())
        if n == 0:
            rows.append({"group": label, "delta": dlabel, "n": 0,
                         "coverage": 0.0, "n_markets": 0})
            continue
        px = ep if delta == 0 else per_bet[f"price_{dlabel}"].to_numpy(dtype=float)
        cost = follower_cost(px, k, adverse_tick, exp)
        gross = per_bet[f"gross_{dlabel}"].to_numpy(dtype=float)
        net = per_bet[f"net_{dlabel}"].to_numpy(dtype=float)
        netedge = per_bet[f"netedge_{dlabel}"].to_numpy(dtype=float)
        real_net = float(np.nanmean(net[cohort]))
        real_edge = float(np.nanmean(netedge[cohort]))
        null = shuffled_null(rv, cohort, cost, bins, real_net, real_edge, rng, n_shuffle)
        lo, hi = cluster_boot_ci(net[cohort], mk[cohort], rng)
        glo, ghi = cluster_boot_ci(gross[cohort], mk[cohort], rng)
        elo, ehi = cluster_boot_ci(netedge[cohort], mk[cohort], rng)
        rows.append({
            "group": label,
            "delta": dlabel,
            "n": n,
            "coverage": float(cohort.mean()),
            "n_markets": int(pd.unique(mk[cohort]).size),
            "own_gross_cohort": float(np.nanmean(own_all_gross[cohort])),
            "own_net_cohort": float(np.nanmean(own_all_net[cohort])),
            "follower_gross": float(np.nanmean(gross[cohort])),
            "follower_gross_lo": glo,
            "follower_gross_hi": ghi,
            "follower_net": real_net,
            "follower_net_lo": lo,
            "follower_net_hi": hi,
            "null_net": null["null_net"],
            "real_minus_null": real_net - null["null_net"],
            "null_p": null["null_p"],
            "own_edge_c_cohort": float(np.nanmean(per_bet["edge_0"].to_numpy(float)[cohort])),
            "follower_edge_c": float(np.nanmean(per_bet[f"edge_{dlabel}"].to_numpy(float)[cohort])),
            # How far the price moved against a buyer in the Δ after entry. Positive
            # = the wallet was early and the market absorbed the information.
            "drift_c": float(np.nanmean(px[cohort] - ep[cohort])),
            "follower_netedge_c": real_edge,
            "follower_netedge_c_lo": elo,
            "follower_netedge_c_hi": ehi,
            "null_netedge_c": null["null_netedge_c"],
            "real_minus_null_c": real_edge - null["null_netedge_c"],
            "null_p_edge": null["null_p_edge"],
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 8. Pipeline stages
# ==========================================================================
def all_certified_wallets(validated: pd.DataFrame | None = None) -> list[str]:
    """Every `edge_persisted` wallet that actually has bets in the deep tape.

    The certified set is 58, but 12 of them were deepened by the slow/sports arms
    and their bets live in those directories, not in `realworld/deep_trades/`.
    Only the ones present here can be simulated."""
    if validated is None:
        validated = pd.read_parquet(VALIDATED_PATH)
    return sorted(validated.loc[validated["edge_persisted"].fillna(False), "wallet"])


def build_bets_frame(prefixes: list[str] | None = None,
                     wallets_override: list[str] | None = None,
                     since_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bets + the token index table, persisted so fetch and score agree on ids.

    ⚠️ `tok_ix` IS THE PRICE CACHE'S PRIMARY KEY. Every `prices/chunk-*.parquet`
    stores points against the tok_ix that was current when it was fetched, so a
    rebuild that renumbers existing tokens silently re-points ~3.5M cached price
    points at the WRONG tokens — the analysis would still run and every number
    would be wrong. So existing assignments are preserved verbatim and genuinely
    new tokens are appended after the current maximum. Adding wallets is therefore
    cache-safe and incremental; a full re-fetch is only needed if the prices
    directory is deleted alongside tokens.parquet."""
    COPYSIM_DIR.mkdir(parents=True, exist_ok=True)
    # `wallets_override` bypasses resolve_wallets' certified-set requirement. It
    # exists because the P1 gate selects for LOW-VOLUME wallets (it needs a long
    # held-out history), and a forward test on those cannot read for months. An
    # uncertified but currently-active wallet is a worse prior and a far better
    # experiment; the forward test, not the backtest, is what validates it.
    wallets = list(wallets_override) if wallets_override else resolve_wallets(prefixes)
    bets = load_bets(wallets)
    if since_days:
        cut = int(bets["timestamp"].max()) - since_days * 86400
        before = len(bets)
        bets = bets[bets["timestamp"] >= cut]
        print(f"[copysim] recency window {since_days}d: {len(bets):,} of {before:,} bets kept")
    bets = attach_categories(bets)

    prior: dict[str, int] = {}
    if TOKENS_PATH.exists():
        old = pd.read_parquet(TOKENS_PATH)
        prior = dict(zip(old["token_id"].astype(str), old["tok_ix"].astype(int)))
    seen = sorted(bets["token_id"].astype(str).unique())
    fresh = [t for t in seen if t not in prior]
    next_ix = (max(prior.values()) + 1) if prior else 0
    for t in fresh:
        prior[t] = next_ix
        next_ix += 1
    if fresh:
        print(f"[copysim] token index: {len(prior) - len(fresh):,} preserved, "
              f"{len(fresh):,} new (cached price chunks stay valid)")
    tokens = pd.DataFrame({"token_id": list(prior), "tok_ix": list(prior.values())})
    tokens = tokens.sort_values("tok_ix").reset_index(drop=True)
    tokens["tok_ix"] = tokens["tok_ix"].astype(np.int32)
    ix = dict(zip(tokens["token_id"], tokens["tok_ix"]))
    bets["tok_ix"] = bets["token_id"].astype(str).map(ix).astype("int32")
    bets["token_id"] = bets["token_id"].astype(str)
    bets["market_id"] = bets["market_id"].astype(str)
    bets["wallet"] = bets["wallet"].astype(str)
    atomic_to_parquet(bets, BETS_PATH, compression="zstd", index=False)
    atomic_to_parquet(tokens, TOKENS_PATH, compression="zstd", index=False)
    print(f"[copysim] {len(bets)} resolved BUY bets, {len(tokens)} tokens, "
          f"{bets['market_id'].nunique()} markets, {bets['wallet'].nunique()} wallets")
    return bets, tokens


def stage_fetch(limit: int | None = None, prefixes: list[str] | None = None,
                wallets_override: list[str] | None = None,
                since_days: int | None = None) -> None:
    bets, tokens = build_bets_frame(prefixes, wallets_override, since_days)
    fetch_market_times(sorted(bets["market_id"].astype(str).unique()))
    run_fetch(bets, tokens, limit=limit)


def prepare_scoring_frame(guard: float | None = None) -> pd.DataFrame:
    """Bets + fee rate + resolution-guard cutoff, ready for `compute_per_bet`."""
    cfg = load_config()
    if guard is None:
        guard = cfg["scoring"].get("fair_value_resolution_guard", 0.2)
    bets = pd.read_parquet(BETS_PATH)
    times = (pd.read_parquet(MARKET_TIMES_PATH) if MARKET_TIMES_PATH.exists()
             else pd.DataFrame(columns=["market_id", "start_ts", "end_ts",
                                        "gamma_fee_rate", "gamma_fee_exponent",
                                        "tick_size"]))
    if "gamma_fee_exponent" not in times.columns:
        times["gamma_fee_exponent"] = 1.0
    bets["market_id"] = bets["market_id"].astype(str)
    df = bets.merge(times, on="market_id", how="left")
    df["cutoff"] = [guard_cutoff(s, e, guard)
                    for s, e in zip(df["start_ts"].to_numpy(dtype=float),
                                    df["end_ts"].to_numpy(dtype=float))]
    df["fee_k"] = [fee_k_for(c, r) for c, r in zip(df["category"],
                                                   df["gamma_fee_rate"])]
    df["fee_exponent"] = df["gamma_fee_exponent"].fillna(1.0).astype(float)
    return df


def stage_score(bandwidth: int = DEFAULT_BANDWIDTH,
                adverse_tick: float = DEFAULT_ADVERSE_TICK,
                guard: float | None = None) -> pd.DataFrame:
    """Primary run (config guard, 120 s fill window, 1c adverse tick) plus the
    sensitivity variants, all written to one frame with a `variant` column."""
    rng = np.random.default_rng(SEED)
    cfg = load_config()
    guard_primary = (cfg["scoring"].get("fair_value_resolution_guard", 0.2)
                     if guard is None else guard)
    tokens = pd.read_parquet(TOKENS_PATH)
    store = load_price_store()
    offsets = token_slices(store[0], len(tokens))
    print(f"[copysim/score] price store: {store[0].size:,} points over "
          f"{len(tokens):,} tokens")

    df = prepare_scoring_frame(guard_primary)
    per_bet = compute_per_bet(df, store, offsets, bandwidth=bandwidth,
                              adverse_tick=adverse_tick)
    atomic_to_parquet(per_bet, PER_BET_PATH, compression="zstd", index=False)

    summary = summarize_all(per_bet, rng, adverse_tick=adverse_tick)
    summary["variant"] = "primary"

    variants = [summary]
    for tick in TICK_SENSITIVITY:
        if tick == adverse_tick:
            continue
        variants.append(_retick(per_bet, tick, rng))
    for bw in BANDWIDTH_SENSITIVITY:
        if bw == bandwidth:
            continue
        variants.append(_revariant(df, store, offsets, rng, bandwidth=bw,
                                   adverse_tick=adverse_tick,
                                   name=f"bandwidth={bw}s"))
    if guard_primary != 0.0:
        # The unguarded run is not a footnote here: these wallets bet late, so the
        # proportional guard removes most of the sample. Reported per wallet too.
        df0 = prepare_scoring_frame(0.0)
        variants.append(_revariant(df0, store, offsets, rng, bandwidth=bandwidth,
                                   adverse_tick=adverse_tick, name="guard=0.0",
                                   per_wallet=True))
    out = pd.concat(variants, ignore_index=True)
    atomic_to_parquet(out, SUMMARY_PATH, compression="zstd", index=False)
    cols = ["variant", "group", "delta", "n", "coverage", "own_gross_cohort",
            "follower_gross", "follower_net", "null_net", "real_minus_null", "null_p"]
    print(out[cols].to_string(index=False))
    return out


def _retick(per_bet: pd.DataFrame, tick: float,
            rng: np.random.Generator) -> pd.DataFrame:
    """Re-price the follower's net return under a different adverse-tick
    assumption. The midpoint series cannot tell us the true ask, so the tick is
    the softest input in the whole calculation and has to be swept."""
    tmp = per_bet.copy()
    k = tmp["fee_k"].to_numpy(dtype=float)
    exp = (tmp["fee_exponent"].to_numpy(dtype=float) if "fee_exponent" in tmp
           else np.ones(len(tmp)))
    rv = tmp["resolved_value"].to_numpy(dtype=float)
    for delta, label in DELTAS:
        px = (tmp["entry_price"] if delta == 0 else tmp[f"price_{label}"]).to_numpy(float)
        cost = follower_cost(px, k, tick, exp)
        tmp[f"net_{label}"] = roi(rv, cost)
        tmp[f"netedge_{label}"] = rv - cost
    s = summarize(tmp, "POOLED", rng, adverse_tick=tick)
    s["variant"] = f"tick={tick}"
    return s


def _revariant(df: pd.DataFrame, store, offsets, rng: np.random.Generator,
               bandwidth: int, adverse_tick: float, name: str,
               per_wallet: bool = False) -> pd.DataFrame:
    pb = compute_per_bet(df, store, offsets, bandwidth=bandwidth,
                         adverse_tick=adverse_tick)
    s = (summarize_all(pb, rng, adverse_tick=adverse_tick) if per_wallet
         else summarize(pb, "POOLED", rng, adverse_tick=adverse_tick))
    s["variant"] = name
    return s


def summarize_all(per_bet: pd.DataFrame, rng: np.random.Generator,
                  adverse_tick: float = DEFAULT_ADVERSE_TICK) -> pd.DataFrame:
    """Pooled table plus one table per wallet."""
    tables = [summarize(per_bet, "POOLED", rng, adverse_tick=adverse_tick)]
    for w, g in per_bet.groupby(per_bet["wallet"].astype(str)):
        tables.append(summarize(g.reset_index(drop=True), w, rng,
                                adverse_tick=adverse_tick))
    return pd.concat(tables, ignore_index=True)


# ==========================================================================
# 9. Report
# ==========================================================================
def _f(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:+.{nd}f}"


def build_report(summary: pd.DataFrame, per_bet: pd.DataFrame) -> str:
    cfg = load_config()
    guard = cfg["scoring"].get("fair_value_resolution_guard", 0.2)
    lines: list[str] = []
    A = lines.append
    A("# Copy-lag simulation on the certified real-world wallets")
    A("")
    A("**Question.** A certified wallet buys. A follower sees the fill ~2 minutes")
    A("later and buys at whatever the market price is *then*. How much of the")
    A("wallet's edge survives that lag, plus the real taker fee and one adverse tick?")
    A("")
    A("Code: `src/copy_sim.py` · tests: `tests/test_copy_sim.py` · per-bet frame:")
    A("`data/interim/copysim/per_bet.parquet`.")
    A("")
    A("## Method")
    A("")
    A("* **Universe.** The five `edge_persisted` wallets from")
    A("  `data/interim/realworld/validated.parquet` named in the task, all of their")
    A("  resolved BUY bets in the deep real-world tape.")
    A("* **Price path.** The public CLOB `/prices-history` endpoint, `fidelity=1`")
    A("  (a book-**midpoint** series on a 1-minute grid). The deep tape holds only our")
    A("  own wallets' fills and so cannot say what the price was two minutes later.")
    A("* **Bounded window.** The follower price is the first grid point in")
    A(f"  `[entry + Δ, entry + Δ + {DEFAULT_BANDWIDTH}s]`. No unbounded reach; if nothing")
    A("  qualifies the result is NaN, never `resolved_value`.")
    A(f"* **Resolution guard.** Prices in the final {guard:.0%} of a market's lifespan")
    A("  (Gamma `startDate` → `umaEndDate`) are excluded, so a near-resolution price")
    A("  cannot stand in for a follower price.")
    A("* **Same cohort.** Own return is recomputed on exactly the bets measurable at")
    A("  each Δ, never against the full sample.")
    A("* **Costs.** `fee = shares × k × p × (1−p)`, taker-only, `k` from Gamma's")
    A(f"  per-market `feeSchedule.rate`, plus one adverse tick of {DEFAULT_ADVERSE_TICK}")
    A("  (the series is a midpoint; a follower lifts the offer).")
    A("* **Null.** `resolved_value` permuted within 20 entry-price quantile bins,")
    A(f"  {N_SHUFFLE} shuffles. What survives the shuffle is the favorite-longshot base")
    A("  rate available to anyone at that price, not copyable alpha.")
    A("* **CI.** 95% percentile bootstrap resampling whole **markets** (one resolution")
    A("  event = one draw), 2,000 resamples.")
    A("")
    A("Every table below is reported twice, in two units:")
    A("")
    A("* **ROI per share**, `(resolved_value − cost) / cost` — the quantity the task")
    A("  asks for, and an equal-dollar-per-bet portfolio return. It is violently")
    A("  heavy-tailed: a 1c longshot that wins is +9,900%, so both the means and")
    A("  especially the shuffled null are dominated by a handful of sub-cent prices")
    A("  (the null reaches +1.58 in one cell, which is noise, not a finding).")
    A("* **Edge per share in price units**, `resolved_value − cost` — the repo's")
    A("  native unit, bounded in `[−1, 1]`, and the one to read when the two")
    A("  disagree. `drift` is `price(entry+Δ) − entry_price`: how far the price moved")
    A("  against a buyer during the lag. In these units `real − null` is exactly")
    A("  `mean(outcome) − mean(shuffled outcome)` over the cohort — the cost terms")
    A("  cancel — so it is the pure outcome-selection excess and **no fee or tick")
    A("  assumption can rescue or destroy it**.")
    A("")
    A("## Data")
    A("")
    A(f"* {len(per_bet):,} resolved BUY bets · {per_bet['market_id'].nunique():,} markets ·")
    A(f"  {per_bet['token_id'].astype(str).nunique():,} tokens · 5 wallets")
    span = pd.to_datetime(per_bet["timestamp"], unit="s")
    A(f"* span {span.min():%Y-%m-%d} → {span.max():%Y-%m-%d}")
    cats = per_bet["category"].value_counts()
    A(f"* top categories: " + ", ".join(f"{c} {n:,}" for c, n in cats.head(6).items()))
    A(f"* mean fee k charged: {per_bet['fee_k'].mean():.4f}")
    miss = float(per_bet["cutoff"].isna().mean())
    A(f"* markets with no usable Gamma lifespan (⇒ no follower price at any Δ): {miss:.1%}")
    if "mid_at_entry" in per_bet:
        gap = (per_bet["entry_price"] - per_bet["mid_at_entry"]).abs().dropna()
        if len(gap):
            A(f"* |tape entry price − midpoint at entry|: median {gap.median():.4f}, "
              f"p75 {gap.quantile(0.75):.4f}, mean {gap.mean():.4f} (n={len(gap):,}) — "
              f"the scale of the adverse-tick assumption")
    A("")
    A("The five wallets, and what the identification engine certified them on")
    A("(`data/interim/realworld/validated.parquet`):")
    A("")
    A("| wallet | held-out skill edge | held-out bets | held-out markets | bets here |")
    A("|---|---:|---:|---:|---:|")
    try:
        v = pd.read_parquet(VALIDATED_PATH).set_index("wallet")
        for w, g in per_bet.groupby(per_bet["wallet"].astype(str)):
            r = v.loc[w]
            A(f"| `{w}` | {r['out_of_sample_residual_edge']:+.4f} | "
              f"{int(r['out_of_sample_n']):,} | {int(r['out_of_sample_markets']):,} "
              f"| {len(g):,} |")
    except Exception as exc:
        A(f"*(unavailable: {exc})*")
    A("")
    primary = summary[summary["variant"] == "primary"]
    unguarded = summary[summary["variant"] == "guard=0.0"]
    A("## Headline")
    A("")
    try:
        u = summary[(summary["variant"] == "guard=0.0") & (summary["group"] == "POOLED")]
        r = u.loc[u["delta"] == "2m"].iloc[0]
        A("Pooled, Δ = 2 minutes, on the 99%-coverage (guard-off) sample, per share:")
        A("")
        A("```")
        A(f"  wallet's own edge at its own fill      {r['own_edge_c_cohort']:+.4f}")
        A(f"  price drift in the first 2 minutes     {-r['drift_c']:+.4f}")
        A(f"  = follower's gross edge                {r['follower_edge_c']:+.4f}")
        A(f"  fee + one adverse tick                 "
          f"{r['follower_netedge_c'] - r['follower_edge_c']:+.4f}")
        A(f"  = follower's NET edge                  {r['follower_netedge_c']:+.4f}")
        A(f"  favorite-longshot base rate (null)     {-r['null_netedge_c']:+.4f}")
        A(f"  = COPYABLE ALPHA                       {r['real_minus_null_c']:+.4f}"
          f"   (null p = {r['null_p_edge']:.3f})")
        A("```")
    except Exception as exc:
        A(f"*(unavailable: {exc})*")
    A("")
    A("Per wallet at Δ = 2 minutes, copyable alpha (`real − null`, price units, the")
    A("cost-invariant number) under both guard settings:")
    A("")
    A("| wallet | guard on: n / alpha / p | guard off: n / alpha / p | follower net (guard off) |")
    A("|---|---|---|---:|")
    for w in sorted(x for x in primary["group"].unique() if x != "POOLED"):
        def _cell(frame):
            sub = frame[(frame["group"] == w) & (frame["delta"] == "2m")]
            if not len(sub) or sub.iloc[0]["n"] == 0:
                return "—"
            s = sub.iloc[0]
            return f"{int(s['n']):,} / {s['real_minus_null_c']:+.4f} / {s['null_p_edge']:.3f}"
        sub = unguarded[(unguarded["group"] == w) & (unguarded["delta"] == "2m")]
        net = (f"{sub.iloc[0]['follower_netedge_c']:+.4f} "
               f"[{sub.iloc[0]['follower_netedge_c_lo']:+.4f}, "
               f"{sub.iloc[0]['follower_netedge_c_hi']:+.4f}]") if len(sub) else "—"
        A(f"| `{w[:14]}…` | {_cell(primary)} | {_cell(unguarded)} | {net} |")
    A("")

    def table(sub: pd.DataFrame) -> None:
        A("**ROI per share** — `(resolved_value − cost) / cost`:")
        A("")
        A("| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net "
          "| null net | real − null | null p | net 95% CI (market-clustered) |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in sub.iterrows():
            if r["n"] == 0:
                A(f"| {r['delta']} | 0 | 0% | | | | | | | | *(no follower price reaches this Δ)* |")
                continue
            ci = f"[{_f(r['follower_net_lo'])}, {_f(r['follower_net_hi'])}]"
            A(f"| {r['delta']} | {int(r['n']):,} | {r['coverage']:.0%} | {int(r['n_markets']):,} "
              f"| {_f(r['own_gross_cohort'])} | {_f(r['follower_gross'])} | {_f(r['follower_net'])} "
              f"| {_f(r['null_net'])} | {_f(r['real_minus_null'])} | {r['null_p']:.3f} | {ci} |")
        A("")
        A("**Edge per share in price units** — same cohorts, same shuffles:")
        A("")
        A("| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null "
          "| real − null | null p | net 95% CI (market-clustered) |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in sub.iterrows():
            if r["n"] == 0:
                continue
            ci = f"[{_f(r['follower_netedge_c_lo'])}, {_f(r['follower_netedge_c_hi'])}]"
            A(f"| {r['delta']} | {_f(r['own_edge_c_cohort'])} | {_f(r['drift_c'])} "
              f"| {_f(r['follower_edge_c'])} | {_f(r['follower_netedge_c'])} "
              f"| {_f(r['null_netedge_c'])} | {_f(r['real_minus_null_c'])} "
              f"| {r['null_p_edge']:.3f} | {ci} |")
        A("")

    A("## Pooled — primary (guard on)")
    A("")
    table(primary[primary["group"] == "POOLED"])
    if len(unguarded):
        A("## Pooled — guard off (the high-coverage view)")
        A("")
        A("These wallets bet late: the median gap from entry to resolution is 2.1 days")
        A("against a median market lifespan of 26 days, so the proportional guard puts")
        A("**58% of bets past the cutoff** and the primary tables above measure a")
        A("minority, long-dated cohort. Note that near-resolution contamination pushes")
        A("the follower's number *down* (price converges on the answer, so the follower")
        A("pays ~the payout), so the unguarded run is if anything the more generous one.")
        A("")
        table(unguarded[unguarded["group"] == "POOLED"])
    A("## Per wallet — primary (guard on)")
    A("")
    for g in [x for x in primary["group"].unique() if x != "POOLED"]:
        A(f"### `{g}`")
        A("")
        table(primary[primary["group"] == g])
    if len(unguarded):
        A("## Per wallet — guard off")
        A("")
        for g in [x for x in unguarded["group"].unique() if x != "POOLED"]:
            A(f"### `{g}`")
            A("")
            table(unguarded[unguarded["group"] == g])
    others = summary[(summary["variant"] != "primary") & (summary["group"] == "POOLED")]
    if len(others):
        A("## Sensitivity (pooled)")
        A("")
        A("The midpoint series cannot tell us the true ask, and the resolution guard")
        A("and fill window are choices, so all three are swept. Shown in price units")
        A("(the stable metric); note that `real − null` is cost-invariant, so the two")
        A("tick rows differ only in `follower net`, not in copyable alpha.")
        A("")
        A("| variant | Δ | n | cov | follower gross | follower net | null "
          "| real − null | null p |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in others.iterrows():
            if r["n"] == 0 or r["delta"] == "0":
                continue
            A(f"| {r['variant']} | {r['delta']} | {int(r['n']):,} | {r['coverage']:.0%} "
              f"| {_f(r['follower_edge_c'])} | {_f(r['follower_netedge_c'])} "
              f"| {_f(r['null_netedge_c'])} | {_f(r['real_minus_null_c'])} "
              f"| {r['null_p_edge']:.3f} |")
        A("")
    A("## How to read this")
    A("")
    A("Two different questions, and they must not be conflated:")
    A("")
    A("1. **Does the follower end up positive after costs?** — `follower net`.")
    A("2. **Is that money *copyable alpha*, or the favorite-longshot base rate that")
    A("   any buyer at the same price level collects?** — `real − null`.")
    A("")
    A("A positive `follower net` that the shuffled null reproduces is answer (1) yes,")
    A("answer (2) no, and it is answer (2) that decides whether *following these")
    A("wallets* is worth anything over buying at the same prices at random. In price")
    A("units the cost terms cancel out of `real − null`, so no fee or tick assumption")
    A("can rescue or destroy it.")
    A("")
    A("A second diagnostic worth applying to any positive cell: **copyable alpha must")
    A("decay in Δ.** Information the wallet has and the market does not is worth less")
    A("the longer you wait. A `real − null` that is flat or *rising* from 2m to 6h is")
    A("not lag-alpha — it is a cohort or structural artifact.")
    A("")
    A("## Verdict")
    A("")
    A("**No. Copyable edge does not survive a 2-minute lag for any of these five")
    A("wallets.** On the 99%-coverage sample the pooled follower keeps +1.41c/share")
    A("net of real fees and one adverse tick — and the shuffled null keeps +1.38c of")
    A("it. Copyable alpha is **+0.03c/share (null p = 0.10)**: three hundredths of a")
    A("cent, against a certified held-out skill edge of +3.9c to +6.9c for these same")
    A("wallets. Per wallet at 2 minutes the alpha is −0.07c, +0.22c, +0.00c, −0.15c")
    A("and −0.03c; one wallet (`0x83255595ba…`) leaves a follower at **−1.94c/share,")
    A("CI [−3.07c, −0.77c]** — actively loss-making.")
    A("")
    A("Where the edge goes, in order:")
    A("")
    A("1. **~2.0c to price drift in the first 120 seconds.** These wallets are")
    A("   genuinely early — the price moves *toward* their side by 2c before a")
    A("   follower can act, which is 41% of the whole edge and is exactly the")
    A("   behaviour that made them certifiable. It is also exactly what a follower")
    A("   cannot have. Drift keeps growing to ~2.2c by 6h, so waiting is worse.")
    A("2. **~1.4c to fees and the adverse tick.** Real, unavoidable, and taker-only")
    A("   by construction (a copier reacts, so a copier is always the taker).")
    A("3. **~1.4c of what is left is the favorite-longshot base rate**, which the")
    A("   shuffled null collects without any knowledge of who traded.")
    A("")
    A("The one cell that is nominally positive is the guard-on pooled row")
    A("(+0.45c, null p = 0.040) and its main contributor `0x1ee9a5fc09…`")
    A("(+2.00c, null p = 0.000 on 3,077 bets). Three reasons not to bank it:")
    A("")
    A("* It **rises** with Δ (+0.45c → +0.58c from 2m to 6h; +2.00c → +2.44c for the")
    A("  wallet). Lag-alpha decays; a rising curve is a cohort artifact.")
    A("* It lives in the 41% of bets that survive the proportional guard — a")
    A("  long-dated, selected minority. On the same wallet's full 99% sample the")
    A("  alpha falls from +2.00c to **+0.22c**.")
    A("* p = 0.040 does not clear this project's pre-registered")
    A("  `scoring.project1.oos_significance_alpha` of **0.005**, and it is one cell in")
    A("  a 2 (guard) × 4 (Δ) × 6 (group) grid.")
    A("")
    A("This is another independent negative on copyability in this repo, and the")
    A("first one measured on real-world certified wallets against a true market price")
    A("path rather than on the crypto-dominated ledger. `docs/realworld_deep_sample.md`")
    A("§6 said identification was solid and copyability untested; it is now tested.")
    A("Identification is unaffected — these wallets really do beat the price they pay.")
    A("The finding is that **the market prices their information in under two minutes**.")
    A("")
    A("## Caveats")
    A("")
    A("* `/prices-history` is a **midpoint**, not an executable ask. The gap between")
    A("  the tape's VWAP entry price and the midpoint at that instant (see Data) has a")
    A("  median under 1c but a mean above 3c, so the 1c adverse tick is if anything")
    A("  **optimistic** for the follower; 0.1c and 0c are swept for completeness.")
    A("  Costs cannot change the verdict anyway, because `real − null` in price units")
    A("  is cost-invariant.")
    A("* The wallet that costs a follower the most, `0x83255595ba…`, is also the one")
    A("  whose price runs away fastest: **+5.3c of drift in 120 seconds**, against a")
    A("  +4.9c own edge. Its entire edge is consumed before a copier can act, which is")
    A("  a cleaner illustration of the mechanism than the pooled average.")
    A("* Fees are charged on the whole history, though Polymarket was genuinely")
    A("  **zero-fee before 2026-01-05**. That is the right choice for a forward")
    A("  copying question and the wrong one for a historical P&L.")
    A("* The guard uses the market's Gamma lifespan, not the token's observed price")
    A("  series, because the endpoint caps a query at <30 days and cannot show a")
    A("  long market's full series.")
    A("* ⛔ Independent of all of this, **the US is close-only on polymarket.com**")
    A("  (docs/polymarket_mechanics.md §9): a US execution layer cannot open a")
    A("  position at all, whatever the edge.")
    A("")
    return "\n".join(lines)


# ==========================================================================
# 10. THE LIVE ESTIMATOR (`certified` stage)
#
# Everything above this line is the ORIGINAL run, whose verdict is WITHDRAWN —
# its null shuffled outcomes among only the tested wallets' own bets and so
# absorbed the skill it was testing for (it reported exactly 0.000 alpha at
# Δ=0, which is impossible). See the module banner.
#
# What `docs/copy_verdict.md` CORRECTION 2 actually reports, and what the
# 2026-07-29 red team re-derived to 4 d.p., is the rule below. It lived only in
# a shell log until now:
#
#   fill    = price(entry + Δ) + max(entry_price − mid_at_entry, tick_size)
#   own_c   = mean( resolved_value − E[outcome | entry_price] )
#   net_c   = mean( resolved_value − E[outcome | fill] − k · fill · (1 − fill) )
#
# with E[·] the POPULATION baseline over the whole real-world tape — not a
# within-cohort shuffle — and CIs from a market-cluster bootstrap.
# ==========================================================================
REDTEAM_DIR = COPYSIM_DIR / "redteam"
CERTIFIED_PER_WALLET_PATH = REDTEAM_DIR / "certified_per_wallet.parquet"
MIN_BETS_PER_WALLET = 150


def spread_proxy(entry_price, mid_at_entry, tick_size):
    """What the follower is charged over the observed midpoint.

    The tape's ``entry_price`` is a post-slippage taker VWAP and
    ``/prices-history`` is a book MIDPOINT, so crediting the follower at the mid
    while charging the wallet its real fill hands the follower ~3.13c/share of
    unearned edge (docs/copy_verdict.md CORRECTION 2). Charging the gap the
    wallet demonstrably paid, floored at one tick, removes that."""
    gap = np.asarray(entry_price, dtype=float) - np.asarray(mid_at_entry, dtype=float)
    return np.maximum(gap, np.asarray(tick_size, dtype=float))


def fit_population_baseline(prices, outcomes, n_bins: int = 20):
    """``E[outcome | price]`` over 20 quantile bins — `features.fit_price_baseline`
    semantics, restated here so this module does not depend on a ledger it never
    reads. Fit it on the WHOLE real-world tape, never on the tested wallets."""
    prices = np.asarray(prices, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.unique(np.quantile(prices, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, edges.size - 2)
    gm = float(outcomes.mean())
    means = np.array([outcomes[idx == b].mean() if np.any(idx == b) else gm
                      for b in range(edges.size - 1)])
    return {"edges": edges, "means": means, "global_mean": gm, "n": int(prices.size)}


def expected_from(baseline: dict, prices):
    prices = np.asarray(prices, dtype=float)
    idx = np.clip(np.searchsorted(baseline["edges"], prices, side="right") - 1,
                  0, baseline["means"].size - 1)
    return baseline["means"][idx]


def score_certified(per_bet: pd.DataFrame, baseline: dict, label: str = "2m",
                    rng: np.random.Generator | None = None,
                    min_bets: int = MIN_BETS_PER_WALLET) -> pd.DataFrame:
    """Per-wallet follower edge under the live rule, with market-cluster CIs.

    IMPOSSIBILITY ASSERTIONS — the 2026-07-29 audit's standing instruction is that
    the tell is usually already in the output:
      * a follower buying the same thing LATER at a WORSE price can never beat the
        wallet on the same bets (this is the assertion added after the
        midpoint-vs-ask artifact, which broke it for 14 of 46 wallets);
      * a fill at or above 1.00 is not a price any book can produce. Those bets are
        counted and reported rather than silently clipped."""
    rng = rng or np.random.default_rng(SEED)
    px = per_bet[f"price_{label}"].to_numpy(dtype=float)
    ep = per_bet["entry_price"].to_numpy(dtype=float)
    mid = per_bet["mid_at_entry"].to_numpy(dtype=float)
    tick = per_bet["tick_size"].fillna(0.001).to_numpy(dtype=float)
    rv = per_bet["resolved_value"].to_numpy(dtype=float)
    k = per_bet["fee_k"].to_numpy(dtype=float)
    exp = (per_bet["fee_exponent"].fillna(1.0).to_numpy(dtype=float)
           if "fee_exponent" in per_bet else np.ones(len(per_bet)))

    sp = spread_proxy(ep, mid, tick)
    raw = px + sp
    unfillable = np.isfinite(raw) & (raw >= 1.0)
    fill = np.clip(raw, 1e-6, 1.0 - 1e-9)
    fee = k * np.power(fill * (1.0 - fill), exp)
    resid = rv - expected_from(baseline, fill) - fee
    own = rv - expected_from(baseline, ep)
    money = rv - fill - fee

    cohort = np.isfinite(px) & np.isfinite(mid)
    mk = pd.factorize(per_bet["market_id"].astype(str))[0]
    rows = []
    for w, g in per_bet.assign(_r=resid, _o=own, _m=money,
                               _c=cohort, _u=unfillable).groupby(
            per_bet["wallet"].astype(str)):
        g = g[g["_c"]]
        if len(g) < min_bets:
            continue
        i = g.index.to_numpy()
        lo, hi = cluster_boot_ci(g["_r"].to_numpy(), mk[i], rng)
        own_c = 100 * float(np.nanmean(g["_o"]))
        net_c = 100 * float(np.nanmean(g["_r"]))
        assert net_c <= own_c + 1e-6, (
            f"IMPOSSIBLE: follower ({net_c:.4f}c) beats the wallet ({own_c:.4f}c) on "
            f"identical bets for {w[:14]} — a later fill at a worse price cannot win. "
            f"This is the midpoint-vs-ask artifact; check the spread proxy.")
        rows.append({"wallet": w[:14], "n": len(g),
                     "markets": int(pd.unique(mk[i]).size),
                     "own_c": own_c, "net_c": net_c,
                     "money_c": 100 * float(np.nanmean(g["_m"])),
                     "lo": 100 * lo, "hi": 100 * hi,
                     "unfillable_share": float(g["_u"].mean()),
                     "copyable": bool(lo > 0)})
    cols = ["wallet", "n", "markets", "own_c", "net_c", "money_c", "lo", "hi",
            "unfillable_share", "copyable"]
    if not rows:
        # No wallet cleared the bet floor. Return the shaped-but-empty frame rather
        # than a column-less one, so a caller sorting or selecting does not blow up.
        print(f"[copysim/certified] no wallet reached the {min_bets}-bet floor")
        return pd.DataFrame({c: pd.Series(dtype="object" if c == "wallet"
                                          else "bool" if c == "copyable" else "float64")
                             for c in cols})
    out = pd.DataFrame(rows)[cols].sort_values("net_c", ascending=False).reset_index(drop=True)
    if len(out):
        n_bad = int(unfillable[cohort].sum())
        print(f"[copysim/certified] {len(out)} wallets scored, "
              f"{int(out['copyable'].sum())} copyable (CI>0); "
              f"follower-beats-wallet 0 (asserted); "
              f"modelled fills >= $1.00: {n_bad:,} of {int(cohort.sum()):,} "
              f"({100*n_bad/max(int(cohort.sum()),1):.2f}%)")
    return out


def stage_certified(label: str = "2m") -> pd.DataFrame:
    """Score the cached per-bet frame under the live rule.

    The baseline must be fitted on the real-world tape by the caller
    (`src.realworld_validate` assembles it); this stage reads the cached
    per-bet frame and a baseline npz so it never re-reads a 3.4M-row tape."""
    per_bet = pd.read_parquet(PER_BET_PATH)
    bl_path = REDTEAM_DIR / "population_baseline.npz"
    if not bl_path.exists():
        raise SystemExit(
            f"{bl_path} not found. Fit it once over the real-world tape "
            "(entry_price, resolved_value) with fit_population_baseline() and "
            "np.savez it — see docs/redteam_realworld_copy_2026-07-29.md Appendix.")
    z = np.load(bl_path)
    baseline = {"edges": z["edges"], "means": z["means"],
                "global_mean": float(z["global_mean"]), "n": int(z["n"])}
    print(f"[copysim/certified] baseline over {baseline['n']:,} real-world bets")
    out = score_certified(per_bet, baseline, label=label)
    REDTEAM_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(out, CERTIFIED_PER_WALLET_PATH, compression="zstd", index=False)
    print(out.to_string(index=False))
    print(f"[copysim/certified] -> {CERTIFIED_PER_WALLET_PATH}")
    return out


def stage_report() -> None:
    summary = pd.read_parquet(SUMMARY_PATH)
    per_bet = pd.read_parquet(PER_BET_PATH)
    text = build_report(summary, per_bet)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(f"[copysim/report] wrote {REPORT_PATH} ({len(text)} chars)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["fetch", "score", "certified", "report"])
    ap.add_argument("--limit", type=int, default=None, help="fetch: cap requests")
    ap.add_argument("--bandwidth", type=int, default=DEFAULT_BANDWIDTH)
    ap.add_argument("--adverse-tick", type=float, default=DEFAULT_ADVERSE_TICK)
    ap.add_argument("--guard", type=float, default=None)
    ap.add_argument("--all-certified", action="store_true",
                    help="fetch: run every edge_persisted wallet present in the deep "
                         "tape, not just the five in WALLET_PREFIXES. Cache-safe — "
                         "existing tok_ix assignments are preserved and only new "
                         "tokens are fetched.")
    ap.add_argument("--wallets-from", default=None,
                    help="fetch: parquet with a `wallet` column — an EXPLICIT wallet "
                         "list that bypasses the certified-set requirement")
    ap.add_argument("--since-days", type=int, default=None,
                    help="fetch: keep only bets from the last N days")
    args = ap.parse_args(argv)
    if args.stage == "fetch":
        prefixes = all_certified_wallets() if args.all_certified else None
        override = (pd.read_parquet(args.wallets_from)["wallet"].astype(str).tolist()
                    if args.wallets_from else None)
        stage_fetch(limit=args.limit, prefixes=prefixes, wallets_override=override,
                    since_days=args.since_days)
    elif args.stage == "score":
        print("[copysim] ⚠️  `score` runs the ORIGINAL estimator, whose verdict is "
              "WITHDRAWN (its null absorbed the skill it tested for). For the live "
              "rule run `certified`. See the module docstring.")
        stage_score(bandwidth=args.bandwidth, adverse_tick=args.adverse_tick,
                    guard=args.guard)
    elif args.stage == "certified":
        stage_certified()
    else:
        stage_report()


if __name__ == "__main__":
    main()
