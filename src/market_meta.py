"""Per-market speed metadata sidecar — Project 3 build step 1 (see
`docs/project3_slow_markets.md` §1.2/§1.3, §2.2, §7.1).

Builds `data/interim/market_meta.parquet`: one row per market carrying the
timestamp-derived speed axes the slow-market pivot rests on — lifespan (L),
the raw fields horizon (H) is anchored on later, and (partial) velocity (V).

READ-ONLY / ANALYSIS-ONLY, same discipline as `ingest.py` / `discover.py`:
public unauthenticated GETs, no keys, nothing placed on-chain. Writes ONLY the
new sidecar (never the ledger, resolutions cache, or cursors), and NEVER deletes
a row (upsert, like `discover.merge_universe`) — micro-crypto keeps its row and
stays fully addressable (§8, the no-drop invariant).

The two metadata sources have complementary failure modes and are combined as
*lower bounds*, never as a single authority (§1.2 soundness rule):

  - **CLOB per-market GET is the PRIMARY path** — path-keyed, 100% coverage of
    the markets we hold bets in, ~20 req/s. It carries `accepting_order_timestamp`,
    `end_date_iso`, `game_start_time`, `tags`, `closed`. Two of those timestamps
    are individually POISONED and must never define lifespan on their own:
      * `accepting_order_timestamp` is a series-creation artifact = resolution −24h
        for micro-crypto, carrying zero info about the real 5-minute trading window;
      * `end_date_iso` is the midnight floor of the scheduled end day (it put 87%
        of bets at a negative horizon on the stage-1 run).
    A naive "accepting_order → end_date ≥ 6h" lifespan re-admits ~75% of the
    micro-crypto slice as "slow" (§1.1b). So lifespan is NEVER derived from that
    window — see `combine_lifespan`. These fields are persisted only as inputs to
    the H-anchor `max(end_ts_clob, tape_t_max)` that later steps use.

  - **Gamma date-sliced enumeration** gives the PRECISE `closedTime`/`startDate`
    (true lifespan) plus `volumeNum`/`liquidityNum` (velocity) — but only ~41.7%
    coverage of the ledger's real-world markets, and widening the window makes it
    worse (§6). So it is OPTIONAL enrichment here, never the universe authority:
    where present it upgrades a market to a precise lifespan; where absent the
    market falls back to the observed-tape lower bound. The §6 Gamma coverage fix
    is deliberately out of scope for step 1 (L+H from CLOB+tape are the spine).

  - **Observed tape span** (`t_max − t_min` from the ledger, all rows) is the
    complete-coverage LOWER BOUND: it proves a market lived at least that long,
    never that it is fast. This is what keeps micro out of "slow" — a 5-minute
    coin-flip has a ~2-minute observed span, so it can never clear a slow gate on
    the tape branch, and Gamma excludes it from the precise branch.

`speed_source` records provenance honestly: `gamma` (precise) / `clob_tape`
(observed-tape lower bound) / `unknown` (neither — parked, retried next build,
NOT dropped). A slow classification resting on a lower bound is sound for
*inclusion* but must never read as certainty (§1.2).
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import (
    BET_LEDGER_PATH,
    INTERIM_DIR,
    atomic_to_parquet,
    load_config,
    make_session,
    print_disk_usage_summary,
)
from src.ingest import fetch_market_resolution

# --- Constants (kept local, NOT in config/config.yaml, which has pending edits
#     from other live Project 3 workstreams — same choice as discover.py). ------
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GAMMA_PAGE_SIZE = 100              # /markets silently caps limit at 100
GAMMA_MAX_PAGES_PER_DAY = 60       # a 1-day end-date slice is ~21 pages (§1.3); headroom
REQUEST_TIMEOUT_SEC = 25
SLEEP_BETWEEN_REQUESTS_SEC = 0.15  # polite: shared API, ~20-22 req/s aggregate cap
CLOB_FETCH_WORKERS = 5             # CLOB rate-limits aggregate to ~20-22 req/s regardless
CHECKPOINT_EVERY = 2000            # persist the sidecar every N CLOB fetches (resumable)

# A tape span below this (seconds) is not a usable lower bound — a single observed
# trade gives t_min == t_max (span 0). Such markets are parked as `unknown` rather
# than mislabelled fast on a zero-width span.
MIN_TAPE_SPAN_S = 1.0

MARKET_META_PATH = INTERIM_DIR / "market_meta.parquet"

MARKET_META_COLUMNS = [
    "market_id",
    # CLOB fields — persisted as H-anchor inputs ONLY, never as a lifespan source.
    "open_ts_clob",     # accepting_order_timestamp (−24h artifact for micro)
    "end_ts_clob",      # end_date_iso (midnight-floored)
    "game_start_ts",    # game_start_time (event anchor; ~93% of real-world markets)
    # Gamma fields — precise, but partial coverage (~42%).
    "start_ts_gamma",   # startDate (precise open)
    "closed_ts_gamma",  # closedTime (precise resolution)
    "volume",           # volumeNum
    "liquidity",        # liquidityNum
    # Observed tape span — complete-coverage lower bound.
    "tape_t_min",
    "tape_t_max",
    # Derived (§1.2 soundness rule).
    "lifespan_s",       # gamma precise, else observed-tape lower bound, else NaN
    "velocity",         # volume / lifespan_s — ONLY when precise (speed_source==gamma)
    "speed_source",     # gamma | clob_tape | unknown
    "tags",
    "closed",
    "first_seen",
    "last_seen",
]

# Mutable fields refreshed on re-seeing a market; first_seen is preserved.
_MUTABLE_COLUMNS = [c for c in MARKET_META_COLUMNS if c not in ("market_id", "first_seen")]


# ---------------------------------------------------------------------------
# Pure logic (unit-tested; no network)
# ---------------------------------------------------------------------------

def _iso_to_ts(x) -> float:
    """Parse a timestamp to a UTC epoch (float seconds). Accepts ISO-8601 strings
    (with 'Z', a space separator, or a bare '+00' offset — all shapes CLOB/Gamma
    emit) and numeric epochs. Anything unparseable/empty -> NaN, never an error."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float("nan") if (isinstance(x, float) and isnan(x)) else float(x)
    s = str(x).strip()
    if not s:
        return float("nan")
    try:
        ts = pd.Timestamp(s)
    except Exception:  # noqa: BLE001 - a malformed timestamp is data, not a crash
        return float("nan")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.timestamp()


def _to_float(x) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def parse_clob_market_meta(market_id: str, raw: dict | None) -> dict | None:
    """Extract the speed-relevant fields from a CLOB `/markets/{cid}` response.
    Returns None if `raw` isn't a dict (a 404/garbage response). The two poisoned
    timestamps (`open_ts_clob`, `end_ts_clob`) are carried verbatim for H-anchoring;
    they are NOT lifespan sources (see `combine_lifespan`)."""
    if not isinstance(raw, dict):
        return None
    return {
        "market_id": market_id,
        "open_ts_clob": _iso_to_ts(raw.get("accepting_order_timestamp")),
        "end_ts_clob": _iso_to_ts(raw.get("end_date_iso")),
        "game_start_ts": _iso_to_ts(raw.get("game_start_time")),
        "tags": "|".join(raw.get("tags") or []),
        "closed": bool(raw.get("closed", False)),
    }


def parse_gamma_market_row(raw: dict) -> dict | None:
    """Extract precise lifespan + volume from a Gamma `/markets` listing row.
    Returns None without a `conditionId` (nothing to key on)."""
    cid = raw.get("conditionId")
    if not cid:
        return None
    return {
        "market_id": cid,
        "start_ts_gamma": _iso_to_ts(raw.get("startDate")),
        "closed_ts_gamma": _iso_to_ts(raw.get("closedTime")),
        "volume": _to_float(raw.get("volumeNum")),
        "liquidity": _to_float(raw.get("liquidityNum")),
    }


def _notna(x) -> bool:
    return x is not None and not (isinstance(x, float) and isnan(x))


def combine_lifespan(
    clob_row: dict | None,
    gamma_row: dict | None,
    tape_span: tuple[float, float],
    min_span_s: float = MIN_TAPE_SPAN_S,
) -> tuple[float, float, str]:
    """The §1.2 lower-bound soundness rule. Returns (lifespan_s, velocity, speed_source).

    Order of preference:
      1. **gamma** — a precise `closedTime − startDate` (both present, positive).
         Velocity `volume / lifespan_s` is computed ONLY here, because dividing by
         a lower-bound (too-small) lifespan would OVER-estimate competition density
         and could wrongly exclude an uncontested slow market — an exclusion error,
         which the soundness rule forbids.
      2. **clob_tape** — the observed tape span `t_max − t_min`, a hard lower bound
         (complete coverage). Velocity is NaN (no sound denominator / no volume).
      3. **unknown** — neither available (no Gamma record and ≤1 observed trade).
         lifespan NaN; the market is parked and retried on the next build.

    CRITICAL: lifespan is NEVER derived from the CLOB accepting_order→end_date
    window. That window is a −24h / midnight-floor artifact that reports a median
    ~13.5h "lifespan" for markets that trade for 5 minutes, and using it would
    re-admit ~75% of micro-crypto as slow (§1.1b). The observed tape span is what
    holds micro out: a 5-minute market's span is ~2 minutes, so it can never clear
    a slow gate on branch 2, and Gamma excludes it from branch 1.
    """
    if gamma_row:
        start = gamma_row.get("start_ts_gamma")
        close = gamma_row.get("closed_ts_gamma")
        if _notna(start) and _notna(close) and (close - start) > 0:
            lifespan = close - start
            vol = gamma_row.get("volume")
            velocity = (vol / lifespan) if (_notna(vol) and lifespan > 0) else float("nan")
            return lifespan, velocity, "gamma"

    t_min, t_max = tape_span
    if _notna(t_min) and _notna(t_max) and (t_max - t_min) >= min_span_s:
        return t_max - t_min, float("nan"), "clob_tape"

    return float("nan"), float("nan"), "unknown"


def build_meta_row(
    market_id: str,
    clob_raw: dict | None,
    gamma_row: dict | None,
    tape_span: tuple[float, float],
) -> dict:
    """Assemble one full sidecar row from the three sources. `first_seen`/`last_seen`
    are left unset here (stamped by `merge_meta`, which owns provenance timing)."""
    clob = parse_clob_market_meta(market_id, clob_raw) or {}
    lifespan, velocity, source = combine_lifespan(clob, gamma_row, tape_span)
    t_min, t_max = tape_span
    row = {
        "market_id": market_id,
        "open_ts_clob": clob.get("open_ts_clob", float("nan")),
        "end_ts_clob": clob.get("end_ts_clob", float("nan")),
        "game_start_ts": clob.get("game_start_ts", float("nan")),
        "start_ts_gamma": (gamma_row or {}).get("start_ts_gamma", float("nan")),
        "closed_ts_gamma": (gamma_row or {}).get("closed_ts_gamma", float("nan")),
        "volume": (gamma_row or {}).get("volume", float("nan")),
        "liquidity": (gamma_row or {}).get("liquidity", float("nan")),
        "tape_t_min": t_min,
        "tape_t_max": t_max,
        "lifespan_s": lifespan,
        "velocity": velocity,
        "speed_source": source,
        "tags": clob.get("tags", ""),
        "closed": clob.get("closed", False),
    }
    return row


def merge_meta(existing: pd.DataFrame, fresh_rows: list[dict], now_ts: int) -> pd.DataFrame:
    """Upsert freshly-built rows into the sidecar. New markets get
    first_seen=last_seen=now; re-seen markets KEEP first_seen, refresh last_seen and
    their mutable fields. Idempotent, and NEVER deletes a row (the no-drop invariant
    — a market that drops out of a later enumeration keeps its row). Mirrors
    `discover.merge_universe`."""
    fresh = pd.DataFrame(fresh_rows)
    if not fresh.empty:
        fresh["first_seen"] = now_ts
        fresh["last_seen"] = now_ts
    if existing.empty:
        combined = fresh
    elif fresh.empty:
        combined = existing
    else:
        prev_first = dict(zip(existing["market_id"], existing["first_seen"]))
        fresh["first_seen"] = fresh["market_id"].map(prev_first).fillna(fresh["first_seen"])
        # fresh rows win for mutable fields; keep='last' after concat.
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["market_id"], keep="last")
    if combined.empty:
        return pd.DataFrame(columns=MARKET_META_COLUMNS)
    for col in MARKET_META_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[MARKET_META_COLUMNS].reset_index(drop=True)


# CLOB-sourced columns — the only ones the free ingest-time capture refreshes.
# Everything else (tape span + the §1.2-derived lifespan/velocity/speed_source, and
# any Gamma fields) is owned by the full build over the ledger and must survive a
# cheap CLOB refresh untouched.
_CLOB_FIELDS = ["open_ts_clob", "end_ts_clob", "game_start_ts", "tags", "closed"]


def refresh_clob_fields(captured: dict, now_ts: int, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Zero-extra-request 'free capture': refresh ONLY the CLOB-sourced columns of
    the sidecar from raw market dicts collected during ingest's resolution GETs
    (`update_resolutions(meta_sink=...)`, §1.3).

    CRITICAL: this must not clobber the ledger-derived fields. For a market already
    in the sidecar it updates only `_CLOB_FIELDS` + `last_seen`, preserving its
    tape span, lifespan, velocity, speed_source, Gamma fields and first_seen. A
    market NOT yet in the sidecar is inserted with its CLOB fields populated and
    tape/derived left NaN/`unknown` (parked) — the next full `build_market_meta`
    over the ledger fills those in. Never deletes a row."""
    if existing is None:
        existing = load_market_meta()
    parsed = [r for r in (parse_clob_market_meta(c, raw) for c, raw in captured.items()) if r]
    if not parsed:
        return existing if not existing.empty else pd.DataFrame(columns=MARKET_META_COLUMNS)

    upd = pd.DataFrame(parsed)
    upd["last_seen"] = now_ts
    have = set(existing["market_id"]) if not existing.empty else set()

    # 1. In-place refresh of CLOB fields for markets already present (DataFrame.update
    #    aligns on the market_id index and only overwrites the columns it carries,
    #    so tape/lifespan/velocity/speed_source/first_seen are left exactly as-is).
    if not existing.empty:
        e = existing.set_index("market_id")
        u = upd[upd["market_id"].isin(have)].set_index("market_id")
        if not u.empty:
            e.update(u[_CLOB_FIELDS + ["last_seen"]])
        existing = e.reset_index()

    # 2. Insert markets not yet in the sidecar (CLOB-only; tape/derived parked).
    new_cids = [c for c in upd["market_id"] if c not in have]
    if new_cids:
        rows = [
            build_meta_row(c, captured[c], None, (float("nan"), float("nan")))
            for c in new_cids
        ]
        existing = merge_meta(existing, rows, now_ts)

    for col in MARKET_META_COLUMNS:
        if col not in existing.columns:
            existing[col] = None
    return existing[MARKET_META_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# speed_bucket classification (Project 3 step 2, §1.4/§2.2)
# ---------------------------------------------------------------------------

# Default thresholds (§1.4 working defaults). Config-tunable via
# scoring.speed.{slow_lifespan_hours, deep_slow_lifespan_days}; NOT load-bearing.
DEFAULT_SLOW_LIFESPAN_HOURS = 24.0
DEFAULT_DEEP_SLOW_LIFESPAN_DAYS = 7.0

SPEED_BUCKETS = ("fast", "slow", "deep_slow", "unknown")


def speed_thresholds(cfg: dict | None = None) -> tuple[float, float]:
    """Return (slow_lifespan_s, deep_slow_lifespan_s) from config, defaulting to the
    §1.4 working values when the keys are absent (the live gitignored config predates
    this knob)."""
    speed = ((cfg or {}).get("scoring", {}) or {}).get("speed", {}) or {}
    slow_h = float(speed.get("slow_lifespan_hours", DEFAULT_SLOW_LIFESPAN_HOURS))
    deep_d = float(speed.get("deep_slow_lifespan_days", DEFAULT_DEEP_SLOW_LIFESPAN_DAYS))
    return slow_h * 3600.0, deep_d * 86400.0


def classify_speed_bucket(lifespan_s, slow_s: float, deep_slow_s: float) -> str:
    """Map a market lifespan to a speed bucket. NaN/None lifespan -> 'unknown'
    (parked, never 'fast' — absence of a lifespan never proves a market is fast,
    §1.2). Ordered so deep_slow ⊂ slow: L>=deep_slow_s -> deep_slow, else
    L>=slow_s -> slow, else fast."""
    if lifespan_s is None or (isinstance(lifespan_s, float) and isnan(lifespan_s)):
        return "unknown"
    L = float(lifespan_s)
    if L >= deep_slow_s:
        return "deep_slow"
    if L >= slow_s:
        return "slow"
    return "fast"


def market_bucket_map(meta: pd.DataFrame, slow_s: float, deep_slow_s: float) -> dict:
    """{market_id -> speed_bucket} from the sidecar's lifespan_s. Markets absent from
    this map (not in the sidecar) are treated as 'unknown' by the caller."""
    if meta is None or meta.empty:
        return {}
    return {
        mid: classify_speed_bucket(L, slow_s, deep_slow_s)
        for mid, L in zip(meta["market_id"], meta["lifespan_s"])
    }


def assign_speed_bucket(market_ids, bucket_map: dict) -> pd.Series:
    """Vectorized market_id -> bucket lookup; unmapped ids -> 'unknown'. Returned as
    a plain object Series aligned to `market_ids`."""
    s = pd.Series(list(market_ids), dtype="object")
    return s.map(bucket_map).fillna("unknown")


def populate_speed_bucket(cfg: dict | None = None, batch_size: int = 200_000,
                          ledger_path=None) -> dict:
    """One-shot / nightly: (re)derive the ledger's additive `speed_bucket` column
    from the current market_meta sidecar and rewrite the ledger IN PLACE.

    STREAMING (batched read -> add column -> write), like `stream_merge_ledger`, so
    peak memory is one batch rather than the whole ~2.8 GB ledger — the pattern this
    codebase adopted to fit the 1 GB VM, and what keeps populate safe to run in the
    nightly recompute wherever it lands.

    Row order is PRESERVED exactly (batches stream in file order; a per-row `.map`,
    never a sort). This is load-bearing: validate.py's chronological split is a
    stable-sort function of the persisted row order, so any reordering would change
    the certified set — Project 1 must stay bit-identical (§5.2). Only the additive
    column changes; every existing column's values, types, and order are untouched
    (byte-identity harness: test_populate_*). Idempotent; safe to re-run. Written to a
    temp file + os.replace, so a kill mid-write leaves the prior ledger intact."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.common import BET_LEDGER_PATH, LEDGER_COLUMNS, LEDGER_COMPRESSION

    cfg = cfg or load_config()
    ledger_path = ledger_path or BET_LEDGER_PATH
    slow_s, deep_slow_s = speed_thresholds(cfg)
    bucket_map = market_bucket_map(load_market_meta(columns=["market_id", "lifespan_s"]),
                                   slow_s, deep_slow_s)

    if not pd.io.common.file_exists(str(ledger_path)):
        print("[market_meta] ledger empty — nothing to populate")
        return {}

    pf = pq.ParquetFile(ledger_path)
    # Output schema = the source's existing columns (byte-identical types) + a string
    # speed_bucket, arranged in LEDGER_COLUMNS order. Handles both a legacy 13-col
    # ledger and a re-run over an already-populated one.
    name_to_field = {f.name: f for f in pf.schema_arrow if not f.name.startswith("__index_level_")}
    name_to_field["speed_bucket"] = pa.field("speed_bucket", pa.string())
    schema = pa.schema([name_to_field[c] for c in LEDGER_COLUMNS])

    counts: dict = {}
    total = 0
    tmp = Path(ledger_path).with_name(f"{Path(ledger_path).name}.tmp{os.getpid()}")
    writer = pq.ParquetWriter(tmp, schema, compression=LEDGER_COMPRESSION)
    try:
        for rb in pf.iter_batches(batch_size=batch_size):
            df = rb.to_pandas()
            df["speed_bucket"] = assign_speed_bucket(df["market_id"], bucket_map).values
            writer.write_table(
                pa.Table.from_pandas(df[LEDGER_COLUMNS], schema=schema, preserve_index=False)
            )
            total += len(df)
            for b, c in df["speed_bucket"].value_counts().items():
                counts[b] = counts.get(b, 0) + int(c)
            del rb, df
        writer.close()
        os.replace(tmp, ledger_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    print(f"[market_meta] speed_bucket populated over {total:,} ledger rows: {counts}")
    return counts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_market_meta(columns: list[str] | None = None) -> pd.DataFrame:
    if MARKET_META_PATH.exists():
        return pd.read_parquet(MARKET_META_PATH, columns=columns)
    return pd.DataFrame(columns=columns or MARKET_META_COLUMNS)


def save_market_meta(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, MARKET_META_PATH, compression="gzip")


# ---------------------------------------------------------------------------
# Ledger tape span (complete-coverage lower bound; batched, low-memory)
# ---------------------------------------------------------------------------

def compute_tape_span(ledger_path=BET_LEDGER_PATH) -> pd.DataFrame:
    """Per-market observed [t_min, t_max] over ALL ledger rows (BUY+SELL, resolved
    or not) — the widest honest lower bound on how long the market was trading.
    Batched via pyarrow `iter_batches` so it stays well under the memory ceiling on
    the low-end box (only market_id+timestamp are read). Returns a frame indexed by
    market_id with columns [tape_t_min, tape_t_max]; empty if there is no ledger."""
    import pyarrow.parquet as pq

    if not pd.io.common.file_exists(str(ledger_path)):
        return pd.DataFrame(columns=["market_id", "tape_t_min", "tape_t_max"])

    parts = []
    pf = pq.ParquetFile(ledger_path)
    for batch in pf.iter_batches(batch_size=500_000, columns=["market_id", "timestamp"]):
        d = batch.to_pandas()
        parts.append(
            d.groupby("market_id", sort=False)["timestamp"].agg(["min", "max"]).reset_index()
        )
        del batch, d
    if not parts:
        return pd.DataFrame(columns=["market_id", "tape_t_min", "tape_t_max"])
    tape = (
        pd.concat(parts, ignore_index=True)
        .groupby("market_id", sort=False)
        .agg(tape_t_min=("min", "min"), tape_t_max=("max", "max"))
        .reset_index()
    )
    return tape


# ---------------------------------------------------------------------------
# Network I/O (exercised via FakeSession in tests, not the live API)
# ---------------------------------------------------------------------------

def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def enumerate_gamma_day(
    session, day_start_iso: str, day_end_iso: str, max_pages: int = GAMMA_MAX_PAGES_PER_DAY,
) -> list[dict]:
    """Date-sliced Gamma enumeration for one end-date day (§1.3): page
    `/markets?closed=true&end_date_min=..&end_date_max=..` (range filters bind, and
    a 1-day slice fits under the offset cap). Returns parsed Gamma rows. OPTIONAL
    enrichment — its ~42% coverage (§6) means it upgrades markets to a precise
    lifespan where present but never defines the universe."""
    rows: list[dict] = []
    offset = 0
    for _page in range(max_pages):
        resp = session.get(
            f"{GAMMA_API_BASE}/markets",
            params={
                "closed": "true",
                "end_date_min": day_start_iso,
                "end_date_max": day_end_iso,
                "limit": GAMMA_PAGE_SIZE,
                "offset": offset,
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        for m in page:
            parsed = parse_gamma_market_row(m)
            if parsed is not None:
                rows.append(parsed)
        if len(page) < GAMMA_PAGE_SIZE:
            break
        offset += GAMMA_PAGE_SIZE
        if SLEEP_BETWEEN_REQUESTS_SEC:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    return rows


def enumerate_gamma_days(session, days_back: int, end_ts: int | None = None) -> dict[str, dict]:
    """Enumerate `days_back` end-date day-slices ending at `end_ts` (default now),
    returning {market_id: gamma_row}. Best-effort: a failed day is logged and
    skipped so a transient error never aborts the whole sidecar build."""
    end_ts = end_ts or _now_ts()
    out: dict[str, dict] = {}
    for i in range(days_back):
        day_end = end_ts - i * 86400
        day_start = day_end - 86400
        start_iso = datetime.fromtimestamp(day_start, tz=timezone.utc).strftime("%Y-%m-%d")
        end_iso = datetime.fromtimestamp(day_end, tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            rows = enumerate_gamma_day(session, start_iso, end_iso)
        except Exception as exc:  # noqa: BLE001 - one bad day shouldn't abort the build
            print(f"[market_meta] WARNING: Gamma enumeration failed for {start_iso}: {exc}")
            continue
        for r in rows:
            out[r["market_id"]] = r
        print(f"[market_meta]   Gamma {start_iso}: {len(rows)} markets "
              f"(cumulative {len(out)})", flush=True)
    return out


def fetch_clob_meta_batch(session, cfg, market_ids: list[str], workers: int) -> dict[str, dict]:
    """Fetch CLOB `/markets/{cid}` for each id in a small thread pool (reusing the
    tested `ingest.fetch_market_resolution` GET, which returns the raw dict or None
    on 404). Returns {market_id: raw_dict}; per-market errors are swallowed so a
    single failure never aborts the batch (the market simply stays as-is)."""

    def _one(cid):
        try:
            return cid, fetch_market_resolution(session, cfg, cid)
        except Exception as exc:  # noqa: BLE001 - keep fetching other markets
            print(f"[market_meta] WARNING: CLOB meta fetch failed for {cid}: {exc}")
            return cid, None

    out: dict[str, dict] = {}
    if workers <= 1 or len(market_ids) <= 1:
        for cid in market_ids:
            _, raw = _one(cid)
            if raw is not None:
                out[cid] = raw
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for cid, raw in pool.map(_one, market_ids):
                if raw is not None:
                    out[cid] = raw
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_market_meta(
    gamma_days: int = 0,
    clob_max_fetches: int = 0,
    refresh_clob: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
    clob_workers: int = CLOB_FETCH_WORKERS,
) -> pd.DataFrame:
    """Build/refresh the market-metadata sidecar over EVERY market in the ledger.

    TAPE-FIRST, because `lifespan_s` (hence `speed_bucket`) needs only the observed
    tape span, which is free from the ledger — the CLOB per-market GET is required
    only for the H-anchor fields, so it is capped, optional, and non-blocking.

    Phase 1 (instant, no per-market network): observed tape span for all markets +
      optional Gamma precise lifespan/volume (`gamma_days` end-date day-slices) ->
      lifespan_s + speed_source for 100% of ledger markets. Existing CLOB fields are
      CARRIED FORWARD so a re-run never clobbers earlier CLOB enrichment.
    Phase 2 (optional, capped): CLOB `/markets/{cid}` for up to `clob_max_fetches`
      markets still missing CLOB fields (or all, if `refresh_clob`), overlaid via
      `refresh_clob_fields` — adds H-anchor fields WITHOUT touching tape/derived.
      Resumable + checkpointed. `clob_max_fetches=0` (default) skips this phase.
    Never deletes a row; persists atomically.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    session = make_session()
    now = _now_ts()

    tape = compute_tape_span()
    tape_by_market = {
        mid: (tmin, tmax)
        for mid, tmin, tmax in zip(tape["market_id"], tape["tape_t_min"], tape["tape_t_max"])
    }
    all_market_ids = list(tape_by_market.keys())
    print(f"[market_meta] {len(all_market_ids):,} markets with observed tape span in the ledger")

    gamma_by_market: dict[str, dict] = {}
    if gamma_days > 0:
        print(f"[market_meta] enumerating {gamma_days} Gamma end-date day-slice(s) "
              f"(optional precise-lifespan enrichment; ~42% coverage)")
        gamma_by_market = enumerate_gamma_days(session, gamma_days, now)

    existing = load_market_meta()
    # Carry forward CLOB fields already captured (from a prior CLOB enrichment / the
    # ingest free-capture) so this tape rebuild does not wipe them.
    existing_clob = (
        existing.set_index("market_id")[_CLOB_FIELDS].to_dict("index")
        if not existing.empty else {}
    )

    # Phase 1: tape (+ Gamma) rows for ALL markets.
    fresh_rows = []
    for mid in all_market_ids:
        row = build_meta_row(mid, None, gamma_by_market.get(mid), tape_by_market[mid])
        prior = existing_clob.get(mid)
        if prior:  # keep any CLOB fields we already have
            for col in _CLOB_FIELDS:
                row[col] = prior[col]
        fresh_rows.append(row)
    sidecar = merge_meta(existing, fresh_rows, now)
    save_market_meta(sidecar)
    src_counts = sidecar["speed_source"].value_counts().to_dict()
    print(f"[market_meta] phase 1: sidecar now {len(sidecar):,} markets from tape"
          f"{'+gamma' if gamma_days else ''}; speed_source: {src_counts}")

    # Phase 2 (optional): CLOB H-anchor enrichment, capped + non-clobbering.
    if clob_max_fetches > 0 or refresh_clob:
        if refresh_clob:
            todo = all_market_ids
        else:
            need_clob = sidecar["open_ts_clob"].isna()
            todo = [m for m in sidecar.loc[need_clob, "market_id"] if m in tape_by_market]
        if clob_max_fetches > 0:
            todo = todo[:clob_max_fetches]
        print(f"[market_meta] phase 2: CLOB-enriching {len(todo):,} markets "
              f"(H-anchor fields; non-clobbering)")
        t0 = time.time()
        for start in range(0, len(todo), checkpoint_every):
            chunk = todo[start:start + checkpoint_every]
            captured = fetch_clob_meta_batch(session, cfg, chunk, clob_workers)
            save_market_meta(refresh_clob_fields(captured, now, load_market_meta()))
            done = min(start + checkpoint_every, len(todo))
            el = time.time() - t0
            rate = done / el if el > 0 else 0.0
            print(f"[market_meta]   {done:,}/{len(todo):,} enriched "
                  f"({rate:.1f} req/s) — checkpointed", flush=True)

    final = load_market_meta()
    if final.empty:
        print("[market_meta] no markets — is the ledger present?")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the per-market speed-metadata sidecar (Project 3 step 1).")
    parser.add_argument("--gamma-days", type=int, default=0,
                        help="enrich with N Gamma end-date day-slices for precise "
                             "lifespan/volume (optional; ~42%% coverage, §6). Default 0 = "
                             "tape-only lifespan.")
    parser.add_argument("--clob-max-fetches", type=int, default=0,
                        help="phase 2: CLOB-enrich up to N markets with the H-anchor fields "
                             "(non-clobbering). Default 0 = tape-only, no CLOB fetch.")
    parser.add_argument("--refresh-clob", action="store_true",
                        help="phase 2: CLOB-enrich ALL markets (re-fetch even if present)")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help="persist the sidecar every N CLOB fetches (resumable)")
    parser.add_argument("--populate", action="store_true",
                        help="skip fetching; (re)derive the ledger's additive speed_bucket "
                             "column from the current sidecar and rewrite the ledger in place "
                             "(row order preserved -> Project 1 bit-identical). Run after a "
                             "sidecar build, and nightly after --fold-delta once cron resumes.")
    args = parser.parse_args()
    if args.populate:
        populate_speed_bucket()
    else:
        build_market_meta(
            gamma_days=args.gamma_days,
            clob_max_fetches=args.clob_max_fetches,
            refresh_clob=args.refresh_clob,
            checkpoint_every=args.checkpoint_every,
        )
    print_disk_usage_summary()


if __name__ == "__main__":
    main()
