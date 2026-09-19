"""Pull Polymarket trade history and persist it as a compact, resumable bet ledger.

Data source (see DECISIONS.md for the full reasoning): two public, unauthenticated
Polymarket REST APIs.
  - https://data-api.polymarket.com/trades   — per-fill trade history (wallet,
    market, outcome token, side, price, size, timestamp).
  - https://clob.polymarket.com/markets/{condition_id} — ground-truth market
    resolution (which outcome token settled at 1 vs 0).

The /trades feed has no working server-side time filter (`after`/`before` params
are accepted but silently ignored — verified empirically), so incremental
ingestion is done client-side: sweep the whole served window from offset 0 and
keep everything newer than the on-disk cursor. The sweep must cover every page —
the feed is not time-ordered across pages, so a stale row is no evidence that
deeper offsets are stale too (see `fetch_new_trades`).

Writes are incremental: each run appends its new rows to a small delta parquet
part under `data/interim/ledger_delta/` rather than rewriting the whole
multi-million-row bet ledger, which is what a 5-minute poll cadence needs to fit.
The nightly recompute folds the parts in with `python -m src.ingest --fold-delta`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.common import (
    BET_LEDGER_PATH,
    CURSOR_PATH,
    LEDGER_COMPRESSION,
    LEDGER_COLUMNS,
    LEDGER_DELTA_DIR,
    RAW_TRADES_DIR,
    RESOLUTIONS_PATH,
    atomic_to_parquet,
    atomic_write_json,
    delta_part_paths,
    delta_pending_rows,
    ensure_dirs,
    load_config,
    load_ledger,
    make_session,
    print_disk_usage_summary,
    save_ledger,
)

# Known, independently-verifiable resolved market used for the one-time build
# sanity gate: "Will Donald Trump win the 2024 US Presidential Election?"
KNOWN_MARKET_CONDITION_ID = (
    "0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917"
)


def trade_key(trade: dict) -> str:
    return f"{trade['transactionHash']}:{trade['proxyWallet']}:{trade['asset']}:{trade['side']}"


def load_cursor() -> dict:
    if CURSOR_PATH.exists():
        with open(CURSOR_PATH) as f:
            return json.load(f)
    cfg = load_config()
    lookback_days = cfg.get("ingest", {}).get("backfill_lookback_days", 3)
    floor_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
    return {"max_timestamp": floor_ts - 1, "keys_at_max": []}


def save_cursor(cursor: dict) -> None:
    atomic_write_json(cursor, CURSOR_PATH)


def fetch_trades_page(session, cfg: dict, offset: int, limit: int) -> list[dict]:
    base = cfg["data_source"]["trades_api_base"]
    resp = session.get(
        f"{base}/trades",
        params={"limit": limit, "offset": offset},
        timeout=cfg["ingest"]["request_timeout_sec"],
    )
    resp.raise_for_status()
    return resp.json()


def fetch_new_trades(session, cfg: dict, cursor: dict) -> list[dict]:
    """Sweep the whole global /trades window and keep everything newer than the cursor.

    The feed is NOT globally time-ordered: measured 2026-07-23, 12 of 19 page
    boundaries are inversions with excursions up to ~4 minutes (rows are tightly
    clustered *within* a 500-row page, but the pages themselves arrive out of
    order — presumably a sharded backend). So a trade older than the cursor floor
    is NOT evidence that everything deeper is older too: an earlier version of
    this function stopped on the first stale row and could abandon pages holding
    newer, never-seen trades.

    Therefore: never break early on a stale row. Sweep every page the API will
    serve (the 10,000-row offset ceiling, or a short/empty page = end of feed) and
    filter purely on timestamp. Keys are de-duplicated within the sweep because
    the feed shifts underneath us between page requests, so one trade can surface
    on two consecutive pages. See DECISIONS.md "`/trades` pages are NOT
    time-ordered"."""
    page_size = cfg["ingest"]["page_size"]
    max_pages = cfg["ingest"]["max_pages_per_run"]
    seen_at_max = set(cursor["keys_at_max"])
    floor_ts = cursor["max_timestamp"]

    new_trades: list[dict] = []
    seen_keys: set[str] = set()
    pages_swept = 0
    offset = 0
    for _page in range(max_pages):
        try:
            page = fetch_trades_page(session, cfg, offset, page_size)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                # The API hard-caps offset+limit (observed: 10,000) — this is the
                # end of what the global feed will serve, not a real error.
                break
            raise
        if not page:
            break
        pages_swept += 1
        for trade in page:
            ts = trade["timestamp"]
            if ts < floor_ts:
                continue
            key = trade_key(trade)
            if ts == floor_ts and key in seen_at_max:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_trades.append(trade)
        if len(page) < page_size:
            break
        offset += page_size
    print(f"[ingest] swept {pages_swept} page(s) of the global feed")
    return new_trades


def write_raw_batch(trades: list[dict], batch_ts: int) -> None:
    if not trades:
        return
    RAW_TRADES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trades)
    df.to_parquet(RAW_TRADES_DIR / f"batch_{batch_ts}.parquet", compression="gzip")


def prune_raw_batches(keep_last_n: int) -> None:
    if not RAW_TRADES_DIR.exists():
        return
    batches = sorted(RAW_TRADES_DIR.glob("batch_*.parquet"))
    for stale in batches[:-keep_last_n] if keep_last_n > 0 else batches:
        stale.unlink()


def load_resolutions_cache() -> pd.DataFrame:
    if RESOLUTIONS_PATH.exists():
        return pd.read_parquet(RESOLUTIONS_PATH)
    return pd.DataFrame(
        columns=["market_id", "token_id", "outcome", "resolved", "resolved_value", "closed"]
    )


def save_resolutions_cache(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, RESOLUTIONS_PATH, compression="gzip")


def fetch_market_resolution(session, cfg: dict, condition_id: str) -> dict | None:
    base = cfg["data_source"]["clob_api_base"]
    resp = session.get(
        f"{base}/markets/{condition_id}",
        timeout=cfg["ingest"]["request_timeout_sec"],
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def update_resolutions(
    session, cfg: dict, condition_ids: set[str], existing: pd.DataFrame,
    max_fetches: int | None = None,
    meta_sink: dict | None = None,
) -> pd.DataFrame:
    """Fetch resolutions for unseen markets, and re-check any still-unresolved ones.

    `max_fetches` caps the number of resolution GETs this run (default unlimited,
    preserving ingest behavior). A deep per-wallet backfill can introduce
    thousands of new markets at once; capping keeps any single run bounded, and
    the rest are picked up on later runs via `resolution_recheck`. Unseen markets
    (from `condition_ids`) are prioritized over re-checks of known-unresolved ones.

    `meta_sink` (optional): when a dict is passed, each fetched market's RAW CLOB
    response is stored as `meta_sink[condition_id] = raw_dict`. This is the
    zero-extra-request 'free capture' of the speed-metadata fields
    (accepting_order_timestamp / end_date_iso / game_start_time / tags) that the
    resolution GET already returns and this function otherwise discards
    (docs/project3_slow_markets.md §1.3). It is purely additive: the resolution
    rows this function computes and returns are byte-identical whether or not a
    sink is supplied (default None -> exact prior behavior)."""
    recheck = cfg["ingest"].get("resolution_recheck", True)
    known_resolved = set(existing.loc[existing["closed"], "market_id"]) if len(existing) else set()
    seen = set(existing["market_id"]) if len(existing) else set()

    unseen = (set(condition_ids) - seen) - known_resolved
    rechecks = set()
    if recheck:
        rechecks = (set(existing.loc[~existing["closed"], "market_id"]) if len(existing) else set())
        rechecks -= known_resolved
    # New markets first, then re-checks; cap the total fetched this run.
    ordered = list(unseen) + [m for m in rechecks if m not in unseen]
    if max_fetches is not None:
        ordered = ordered[:max_fetches]

    # Resolution GETs are independent, read-only, and I/O-bound; the CLOB endpoint
    # tolerates ~8 concurrent requests (verified: 0 rate-limits at 8, heavy 429s at
    # 12+), so fetch the batch in a small thread pool rather than serially — the
    # per-market work below is otherwise a single blocking GET each. `session` has
    # a Retry (429/5xx + backoff) so an occasional straggler self-heals; per-market
    # errors are swallowed so the market simply stays unresolved for a later run.
    # workers<=1 falls back to the original serial behavior.
    workers = int(cfg.get("ingest", {}).get("resolution_fetch_workers", 8))

    def _rows_for(condition_id):
        try:
            market = fetch_market_resolution(session, cfg, condition_id)
        except Exception as exc:  # noqa: BLE001 - keep ingesting other markets
            print(f"[ingest] WARNING: resolution fetch failed for {condition_id}: {exc}")
            return []
        if market is None:
            return []
        # Free capture: stash the raw response for the speed-metadata sidecar. A
        # distinct key per thread (condition_id) -> GIL-atomic, no lock needed. This
        # does not touch the resolution rows below, so the returned frame is
        # unaffected (byte-identity harness: test_update_resolutions_meta_sink_*).
        if meta_sink is not None:
            meta_sink[condition_id] = market
        closed = bool(market.get("closed", False))
        return [
            {
                "market_id": condition_id,
                "token_id": token["token_id"],
                "outcome": token["outcome"],
                "resolved": closed,
                "resolved_value": float(token["price"]) if closed else None,
                "closed": closed,
            }
            for token in market.get("tokens", [])
        ]

    rows = []
    if workers <= 1 or len(ordered) <= 1:
        for condition_id in ordered:
            rows.extend(_rows_for(condition_id))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for market_rows in pool.map(_rows_for, ordered):
                rows.extend(market_rows)

    fresh = pd.DataFrame(rows)
    if existing.empty:
        combined = fresh
    elif fresh.empty:
        combined = existing
    else:
        combined = pd.concat([existing, fresh], ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(subset=["market_id", "token_id"], keep="last")
    return combined.reset_index(drop=True)


def fold_trades_to_ledger(trades: list[dict], resolutions: pd.DataFrame) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    df = pd.DataFrame(trades)
    df = df.rename(
        columns={
            "proxyWallet": "wallet",
            "conditionId": "market_id",
            "asset": "token_id",
            "price": "entry_price",
            "transactionHash": "tx_hash",
        }
    )
    df = df[
        ["wallet", "market_id", "token_id", "outcome", "side", "entry_price", "size",
         "timestamp", "title", "slug", "tx_hash"]
    ].rename(columns={"title": "question"})

    if resolutions.empty:
        df["resolved"] = False
        df["resolved_value"] = None
    else:
        res = resolutions[["market_id", "token_id", "resolved", "resolved_value"]]
        df = df.merge(res, on=["market_id", "token_id"], how="left")
        df["resolved"] = df["resolved"].fillna(False)

    # New trades are speed-unclassified until market_meta.populate_speed_bucket runs
    # (nightly, from the sidecar). "unknown" is the additive default — never a filter-out.
    df["speed_bucket"] = "unknown"
    return df[LEDGER_COLUMNS]


LEDGER_DEDUP_KEY = ["tx_hash", "wallet", "token_id", "side"]


def _ensure_speed_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add the additive `speed_bucket` column (default 'unknown') if a frame lacks
    it. Makes the fold/merge paths backward-compatible with a pre-Project-3 (13-col)
    ledger or delta part: the column is filled by market_meta.populate_speed_bucket,
    not here. No-op once the column is present."""
    if "speed_bucket" not in df.columns:
        df = df.copy()
        df["speed_bucket"] = "unknown"
    return df


def stream_merge_ledger(
    src_path,
    dst_path,
    new_rows: pd.DataFrame,
    resolutions: pd.DataFrame,
    batch_size: int = 200_000,
) -> tuple[int, int, int]:
    """Streaming equivalent of the in-memory ledger merge, for RAM-tight hosts.

    Reproduces exactly:

        combined = pd.concat(
            [refresh_ledger_resolutions(load_ledger(), resolutions), new_rows],
            ignore_index=True,
        ).drop_duplicates(subset=LEDGER_DEDUP_KEY, keep="last")

    but reads `src_path` a batch at a time and writes `dst_path` as it goes, so
    peak memory is one batch rather than the whole ledger (~2 GB at 4.7M rows,
    which a 1 GB host cannot hold). Returns (rows, resolved_rows, distinct_wallets),
    accumulated during the stream so the caller never needs the merged frame.

    Equivalence rests on one invariant: the existing ledger is already unique on
    LEDGER_DEDUP_KEY (every prior run ended with the same drop_duplicates). Given
    that, `keep="last"` only ever means "a new row supersedes an identical-key old
    row", so it is enough to drop superseded old rows and append the (self-deduped)
    new rows last — which preserves the exact row order of the in-memory path.
    `refresh_ledger_resolutions` is row-wise, so applying it per batch is identical.

    The caller writes to a temp path and os.replace()s it into position, so an
    interrupted run leaves the previous ledger intact (see common.atomic_to_parquet
    — a mid-write kill corrupted the ledger once, 2026-07-19)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src_path)
    # The stored ledger carries a stale pandas index column (`__index_level_0__`,
    # a leftover of drop_duplicates before the save). It is never read — every
    # consumer resets or ignores the ledger's index — so the rewritten file keeps
    # only the real data columns, which also lets us write with preserve_index=False.
    schema = pa.schema(
        [f for f in pf.schema_arrow if not f.name.startswith("__index_level_")]
    )
    # Backward-compat: a pre-Project-3 (13-col) source lacks speed_bucket. Always
    # emit it (as 'unknown', filled later by populate_speed_bucket) so the output
    # schema is the current LEDGER_COLUMNS regardless of the input's vintage.
    if "speed_bucket" not in schema.names:
        schema = schema.append(pa.field("speed_bucket", pa.string()))

    if new_rows is not None and not new_rows.empty:
        new_rows = _ensure_speed_bucket(new_rows)
        new_rows = new_rows.drop_duplicates(subset=LEDGER_DEDUP_KEY, keep="last")
        new_idx = pd.MultiIndex.from_frame(new_rows[LEDGER_DEDUP_KEY])
    else:
        new_rows, new_idx = None, None

    written = 0
    resolved = 0
    wallets: set = set()
    writer = pq.ParquetWriter(dst_path, schema, compression=LEDGER_COMPRESSION)
    try:
        for rb in pf.iter_batches(batch_size=batch_size):
            df = _ensure_speed_bucket(rb.to_pandas())
            df = refresh_ledger_resolutions(df, resolutions)
            if new_idx is not None and not df.empty:
                superseded = pd.MultiIndex.from_frame(df[LEDGER_DEDUP_KEY]).isin(new_idx)
                df = df.loc[~superseded]
            if df.empty:
                continue
            writer.write_table(
                pa.Table.from_pandas(df[LEDGER_COLUMNS], schema=schema, preserve_index=False)
            )
            written += len(df)
            resolved += int(df["resolved"].sum())
            wallets.update(df["wallet"].unique())
            del df
        if new_rows is not None and not new_rows.empty:
            writer.write_table(
                pa.Table.from_pandas(
                    new_rows[LEDGER_COLUMNS], schema=schema, preserve_index=False
                )
            )
            written += len(new_rows)
            resolved += int(new_rows["resolved"].sum())
            wallets.update(new_rows["wallet"].unique())
    finally:
        writer.close()
    return written, resolved, len(wallets)


def refresh_ledger_resolutions(ledger: pd.DataFrame, resolutions: pd.DataFrame) -> pd.DataFrame:
    """Backfill resolved_value for ledger rows whose market has since closed."""
    if ledger.empty or resolutions.empty:
        return ledger
    res = resolutions[["market_id", "token_id", "resolved", "resolved_value"]].rename(
        columns={"resolved": "resolved_new", "resolved_value": "resolved_value_new"}
    )
    merged = ledger.merge(res, on=["market_id", "token_id"], how="left")
    needs_update = merged["resolved_new"].fillna(False) & ~merged["resolved"]
    merged.loc[needs_update, "resolved"] = True
    merged.loc[needs_update, "resolved_value"] = merged.loc[needs_update, "resolved_value_new"]
    return merged[LEDGER_COLUMNS]


LEDGER_DTYPES = {
    "wallet": "object",
    "market_id": "object",
    "token_id": "object",
    "outcome": "object",
    "side": "object",
    "entry_price": "float64",
    "size": "float64",
    "timestamp": "int64",
    "resolved": "bool",
    "resolved_value": "float64",
    "question": "object",
    "slug": "object",
    "tx_hash": "object",
    "speed_bucket": "object",
}


def normalize_ledger_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ledger rows to the ledger's stored types.

    Needed because delta rows now make a parquet round-trip before they reach
    `stream_merge_ledger`, which casts them to the existing ledger's arrow schema.
    An all-null `resolved_value` (a batch where nothing has resolved yet) would
    otherwise be written as a parquet `null`-typed column and come back as object
    dtype, which does not cast cleanly."""
    df = df.copy()
    for col, dtype in LEDGER_DTYPES.items():
        if col not in df.columns:
            continue
        if dtype == "object":
            continue
        if dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        elif dtype == "bool":
            df[col] = df[col].fillna(False).astype("bool")
    return df


def append_ledger_delta(new_rows: pd.DataFrame, batch_ts: int):
    """Append this run's new ledger rows as one small delta part.

    This is the "v1 incremental ledger write": a 5-min ingest tick must not pay
    for a full read+rewrite of the multi-million-row ledger. Measured on the VM,
    that rewrite (per-batch resolutions merge + object-dtype MultiIndex.isin ×24
    batches) took 5-8 min — longer than the poll interval, so every other cron
    tick SKIPped on the writer lock and consecutive polls stopped overlapping,
    losing ~20% of platform trades. Writing a part is O(new rows), ~a second.

    Parts are named by the batch's newest trade timestamp (zero-padded so
    lexicographic order is chronological) plus the pid, so two writers can never
    collide on a filename. Written atomically, like every other ledger write."""
    if new_rows is None or new_rows.empty:
        return None
    LEDGER_DELTA_DIR.mkdir(parents=True, exist_ok=True)
    path = LEDGER_DELTA_DIR / f"part_{batch_ts:012d}_{os.getpid()}.parquet"
    atomic_to_parquet(
        normalize_ledger_dtypes(new_rows[LEDGER_COLUMNS]),
        path,
        compression=LEDGER_COMPRESSION,
        index=False,
    )
    return path


def _group_delta_parts(parts: list, max_rows: int) -> list[list]:
    """Chunk parts into fold groups of at most `max_rows` rows (>=1 part each).

    A fold pass holds one group in memory while streaming the ledger past it, so
    the group size — not the number of pending parts — bounds peak RSS. A day of
    parts can be ~1M rows; folding that in one shot would not fit the 1 GB host."""
    import pyarrow.parquet as pq

    groups: list[list] = []
    current: list = []
    current_rows = 0
    for part in parts:
        try:
            rows = pq.ParquetFile(part).metadata.num_rows
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest] WARNING: unreadable delta part {part.name} ({exc}) — skipping")
            continue
        if current and current_rows + rows > max_rows:
            groups.append(current)
            current, current_rows = [], 0
        current.append(part)
        current_rows += rows
    if current:
        groups.append(current)
    return groups


def fold_delta_into_ledger(resolutions: pd.DataFrame | None = None,
                           max_rows_per_pass: int | None = None) -> int:
    """Fold pending delta parts into the bet ledger, then delete them.

    Run by the nightly recompute (before anything reads the ledger). Equivalent to
    what ingest used to do inline every run, just batched: for each group of parts,
    stream the ledger through `stream_merge_ledger` with the group as `new_rows`
    and atomically replace. Also refreshes ledger resolutions even when there is
    nothing pending — that refresh used to ride along on every ingest.

    Idempotent: parts are unlinked only after the replaced ledger is in place, so a
    crash in between merely re-folds rows that dedup to the same result. Returns
    the number of delta rows folded."""
    cfg = load_config()
    if max_rows_per_pass is None:
        max_rows_per_pass = int(cfg.get("ingest", {}).get("delta_fold_rows_per_pass", 300_000))
    if resolutions is None:
        resolutions = load_resolutions_cache()

    parts = delta_part_paths()
    groups = _group_delta_parts(parts, max_rows_per_pass) if parts else [[]]
    folded = 0
    n_rows = n_resolved = n_wallets = 0

    for group in groups:
        if group:
            new_rows = pd.concat(
                [pd.read_parquet(p) for p in group], ignore_index=True
            )
            new_rows = normalize_ledger_dtypes(new_rows[LEDGER_COLUMNS])
        else:
            new_rows = None

        if BET_LEDGER_PATH.exists():
            tmp = BET_LEDGER_PATH.with_name(f"{BET_LEDGER_PATH.name}.tmp{os.getpid()}")
            try:
                n_rows, n_resolved, n_wallets = stream_merge_ledger(
                    BET_LEDGER_PATH, tmp, new_rows, resolutions
                )
                os.replace(tmp, BET_LEDGER_PATH)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        elif new_rows is not None and not new_rows.empty:
            combined = new_rows.drop_duplicates(subset=LEDGER_DEDUP_KEY, keep="last")
            save_ledger(combined)
            n_rows = len(combined)
            n_resolved = int(combined["resolved"].sum())
            n_wallets = combined["wallet"].nunique()

        if new_rows is not None:
            folded += len(new_rows)
        del new_rows
        for part in group:
            try:
                part.unlink()
            except OSError:
                pass

    print(
        f"[ingest] folded {folded} delta rows from {len(parts)} part(s) in "
        f"{len(groups)} pass(es); ledger now has {n_rows} bet rows "
        f"({n_resolved} resolved) across {n_wallets} wallets"
    )
    return folded


def _ledger_row_count() -> int:
    if not BET_LEDGER_PATH.exists():
        return 0
    import pyarrow.parquet as pq

    return pq.ParquetFile(BET_LEDGER_PATH).metadata.num_rows


def run_ingest() -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    session = make_session(
        max_retries=cfg["ingest"]["max_retries"], backoff=cfg["ingest"]["retry_backoff_sec"]
    )

    cursor = load_cursor()
    new_trades = fetch_new_trades(session, cfg, cursor)
    print(f"[ingest] fetched {len(new_trades)} new trade rows")

    resolutions = load_resolutions_cache()

    condition_ids = {t["conditionId"] for t in new_trades}
    # Cap resolution GETs per run so ingest stays inside its ~5-min cron cadence.
    # update_resolutions rechecks EVERY still-open market each run; the deep
    # backfill leaves a large, growing tail of genuinely-open real-world markets
    # (6k+ and climbing), and re-checking all of them at the CLOB's ~20 req/s
    # ceiling was pushing a single ingest to ~9 min — longer than the poll
    # interval, so cron ticks would just skip on the writer lock. The cap keeps
    # the time-critical *trade fetch* every-run while the non-time-critical
    # *recheck* drains across runs; unseen (freshly-traded) markets are always
    # prioritized ahead of rechecks, so new bets still get resolved promptly.
    # Null/absent -> uncapped (the pre-cron behavior).
    max_res = cfg.get("ingest", {}).get("max_resolution_fetches_per_run")
    # Free capture: collect the raw CLOB market dicts the resolution GETs already
    # return, and refresh the speed-metadata sidecar's CLOB fields from them at zero
    # extra requests (docs/project3_slow_markets.md §1.3). Additive: `resolutions`
    # is unaffected by the sink. Best-effort — a sidecar hiccup must never break the
    # time-critical ingest path, so it is wrapped and swallowed.
    meta_sink: dict = {}
    resolutions = update_resolutions(
        session, cfg, condition_ids, resolutions, max_fetches=max_res, meta_sink=meta_sink)
    save_resolutions_cache(resolutions)
    if meta_sink:
        try:
            from src.market_meta import (  # lazy: market_meta imports ingest
                load_market_meta,
                refresh_clob_fields,
                save_market_meta,
                _now_ts,
            )

            save_market_meta(refresh_clob_fields(meta_sink, _now_ts(), load_market_meta()))
            print(f"[ingest] refreshed market_meta CLOB fields for {len(meta_sink)} markets")
        except Exception as exc:  # noqa: BLE001 - never let the sidecar break ingest
            print(f"[ingest] WARNING: market_meta refresh skipped: {exc}")

    new_ledger_rows = fold_trades_to_ledger(new_trades, resolutions)
    batch_ts = 0
    if new_trades:
        batch_ts = max(t["timestamp"] for t in new_trades)
        write_raw_batch(new_trades, batch_ts)
        prune_raw_batches(cfg["ingest"]["raw_batches_to_keep"])

        max_ts = batch_ts
        keys_at_max = [trade_key(t) for t in new_trades if t["timestamp"] == max_ts]
        save_cursor({"max_timestamp": max_ts, "keys_at_max": keys_at_max})

    # INCREMENTAL WRITE: append the new rows as a small delta part instead of
    # streaming the whole ledger through a rewrite. The nightly recompute folds
    # the parts in (`--fold-delta`) before anything reads the ledger. This is what
    # keeps a tick to ~a minute, so the 5-min cron cadence actually holds and
    # consecutive polls OVERLAP inside the feed's ~8.3-min window instead of
    # gapping (see DECISIONS.md "incremental ledger write").
    if not new_ledger_rows.empty:
        append_ledger_delta(new_ledger_rows, batch_ts)

    n_rows = _ledger_row_count()
    pending = delta_pending_rows()
    print(
        f"[ingest] ledger has {n_rows} folded bet rows; "
        f"{pending} row(s) pending in {len(delta_part_paths())} delta part(s)"
    )
    # A fold only happens nightly, so a stuck recompute would let the delta grow
    # without bound. Say so loudly rather than quietly accumulating parts.
    warn_rows = cfg.get("ingest", {}).get("delta_warn_rows", 5_000_000)
    if warn_rows and pending > warn_rows:
        print(
            f"[ingest] WARNING: {pending} delta rows pending (> {warn_rows}). "
            "The nightly fold may not be running — check scripts/recompute.sh / cron.log."
        )
    return n_rows


def sanity_check_known_market() -> bool:
    """Hard gate: verify the pipeline's real data against a known resolved market.

    Checks the 2024 US Presidential Election ("Trump") market: the winning
    outcome, a plausible settled price, and a plausible trade timestamp.
    """
    cfg = load_config()
    session = make_session(
        max_retries=cfg["ingest"]["max_retries"], backoff=cfg["ingest"]["retry_backoff_sec"]
    )

    print(f"[sanity-check] fetching known market {KNOWN_MARKET_CONDITION_ID} ...")
    market = fetch_market_resolution(session, cfg, KNOWN_MARKET_CONDITION_ID)
    if not market:
        print("[sanity-check] FAILED: market resolution lookup returned nothing.")
        return False
    if not market.get("closed"):
        print("[sanity-check] FAILED: expected market to be closed/resolved.")
        return False

    tokens = {t["outcome"]: t for t in market.get("tokens", [])}
    yes = tokens.get("Yes")
    if yes is None or not yes.get("winner") or abs(float(yes["price"]) - 1.0) > 1e-6:
        print(f"[sanity-check] FAILED: unexpected resolution shape: {tokens}")
        return False
    print("[sanity-check] resolution OK: 'Yes' (Trump) won, settled at price 1.0 — matches reality.")

    base = cfg["data_source"]["trades_api_base"]
    resp = session.get(
        f"{base}/trades",
        params={"market": KNOWN_MARKET_CONDITION_ID, "limit": 5},
        timeout=cfg["ingest"]["request_timeout_sec"],
    )
    resp.raise_for_status()
    trades = resp.json()
    if not trades:
        print("[sanity-check] FAILED: no trades returned for a known high-volume market.")
        return False

    election_day = datetime(2024, 11, 5, tzinfo=timezone.utc).timestamp()
    window_end = datetime(2024, 12, 15, tzinfo=timezone.utc).timestamp()
    for t in trades:
        if not (0.0 <= t["price"] <= 1.0):
            print(f"[sanity-check] FAILED: implausible price {t['price']}")
            return False
        if not (election_day - 30 * 86400 <= t["timestamp"] <= window_end):
            print(f"[sanity-check] FAILED: implausible timestamp {t['timestamp']}")
            return False
    print(
        f"[sanity-check] {len(trades)} sample trades OK: prices in [0,1] and "
        "timestamps fall within the 2024 election window."
    )
    print("[sanity-check] PASSED.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="run the hard-gate check only")
    parser.add_argument(
        "--fold-delta",
        action="store_true",
        help="fold pending delta parts into the bet ledger (nightly recompute step)",
    )
    args = parser.parse_args()

    if args.sanity_check:
        ok = sanity_check_known_market()
        sys.exit(0 if ok else 1)

    if args.fold_delta:
        ensure_dirs()
        fold_delta_into_ledger()
        print_disk_usage_summary()
        return

    run_ingest()
    print_disk_usage_summary()


if __name__ == "__main__":
    main()
