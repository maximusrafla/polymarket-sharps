"""Shared config, paths, and small utilities used across the pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CURSOR_PATH = RAW_DIR / "cursor.json"
RAW_TRADES_DIR = RAW_DIR / "trades"
RESOLUTIONS_PATH = RAW_DIR / "resolutions.parquet"
BET_LEDGER_PATH = INTERIM_DIR / "bet_ledger.parquet"
# Incremental ("delta") ledger parts. The 5-min ingest appends each poll's new
# rows here as one small parquet file instead of rewriting the whole multi-
# million-row ledger; the nightly recompute folds the parts in and deletes them
# (src.ingest --fold-delta). See DECISIONS.md "incremental ledger write".
LEDGER_DELTA_DIR = INTERIM_DIR / "ledger_delta"
WALLET_FEATURES_PATH = INTERIM_DIR / "wallet_features.parquet"
WALLET_VALIDATED_PATH = INTERIM_DIR / "wallet_validated.parquet"
RANKED_WALLETS_PATH = PROCESSED_DIR / "ranked_wallets.parquet"
REPORT_PATH = PROCESSED_DIR / "report.md"
WATCH_STATE_PATH = INTERIM_DIR / "watch_state.json"
ALERTS_LOG_PATH = DATA_DIR / "alerts.log"

LEDGER_COLUMNS = [
    "wallet",
    "market_id",
    "token_id",
    "outcome",
    "side",
    "entry_price",
    "size",
    "timestamp",
    "resolved",
    "resolved_value",
    "question",
    "slug",
    "tx_hash",
    # ADDITIVE, backward-compatible (Project 3 step 2, docs/project3_slow_markets.md
    # §2.2): per-market speed classification derived from the market_meta sidecar's
    # lifespan. Pre-existing rows / markets with no usable lifespan -> "unknown",
    # filled on the next populate. NEVER feeds the score, rank order, or
    # edge_persisted, and NEVER removes a row — micro stays fully scored/addressable
    # (§8). Scoring code (features/validate) projects the columns it needs and never
    # reads this one, so the certified set is unchanged by its presence. Legacy
    # ledgers without the column load fine (load_ledger falls back to a full read).
    "speed_bucket",
]


# Ledger parquet codec. zstd over gzip: measured on the 4.7M-row ledger, a full
# rewrite takes 20.8s vs 192.7s (~9x faster) AND the file is smaller (354 MB vs
# 395 MB). ingest rewrites the whole ledger every run, so the write cost is what
# decides whether a 5-minute poll fits its cadence on a small host; the smaller
# file also helps the ~32 GB disk budget. Readers are codec-agnostic (pyarrow
# handles gzip and zstd transparently), so old gzip ledgers still load fine.
LEDGER_COMPRESSION = "zstd"


def delta_part_paths() -> list:
    """Pending delta parts, oldest first (filenames are zero-padded timestamps, so
    lexicographic order is chronological). Later parts supersede earlier ones on a
    duplicate key, which is why fold order matters."""
    if not LEDGER_DELTA_DIR.exists():
        return []
    return sorted(LEDGER_DELTA_DIR.glob("part_*.parquet"))


def delta_pending_rows() -> int:
    """Row count across pending delta parts, read from parquet footers only."""
    import pyarrow.parquet as pq

    total = 0
    for p in delta_part_paths():
        try:
            total += pq.ParquetFile(p).metadata.num_rows
        except Exception:  # noqa: BLE001 - a torn part shouldn't break a status print
            pass
    return total


def load_ledger(columns=None, categorical=None):
    """Load the bet ledger. `columns` (optional) pushes a column projection down
    to the parquet reader so callers that only need a subset (e.g. features.py)
    never materialize the unused string columns — falls back to a full read if a
    requested column is absent (older ledgers).

    `categorical` (optional list) reads those string columns dictionary-encoded in
    Arrow, so pandas receives `category` dtype without ever materializing Python
    objects. This matters because glibc does not return freed memory to the OS:
    peak RSS is set by the largest momentary allocation, so materializing a big
    repetitive column as objects (then converting) leaves a high-water mark that
    later frees cannot lower. Encoding at read time avoids that spike."""
    import pandas as pd

    if not BET_LEDGER_PATH.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    # The ledger on disk excludes any not-yet-folded delta parts (see
    # LEDGER_DELTA_DIR). scripts/recompute.sh folds them in before anything reads
    # the ledger; an ad-hoc read in between simply misses the newest trades, so
    # say so rather than silently under-reporting.
    n_pending = len(delta_part_paths())
    if n_pending:
        print(
            f"[ledger] NOTE: {n_pending} unfolded delta part(s) pending "
            f"({LEDGER_DELTA_DIR}); this read excludes them. "
            "Run `python -m src.ingest --fold-delta` for an up-to-date ledger."
        )

    if categorical:
        try:
            import pyarrow.parquet as pq
            import pyarrow.compute as pc

            table = pq.read_table(BET_LEDGER_PATH, columns=columns)
            for col in categorical:
                if col in table.column_names:
                    i = table.schema.get_field_index(col)
                    table = table.set_column(i, col, pc.dictionary_encode(table.column(col)))
            df = table.to_pandas()
            del table
            return df
        except (ValueError, KeyError, ImportError):
            pass  # fall through to a plain read

    if columns is not None:
        try:
            return pd.read_parquet(BET_LEDGER_PATH, columns=columns)
        except (ValueError, KeyError):
            return pd.read_parquet(BET_LEDGER_PATH)
    return pd.read_parquet(BET_LEDGER_PATH)


def atomic_to_parquet(df, path, **kwargs) -> None:
    """Write a parquet file atomically: serialize to a temp file in the same
    directory, then os.replace() it into place (an atomic rename on the same
    filesystem). A crash/kill mid-write leaves the previous good file intact and
    only a stray .tmp — never a half-written, footer-less (unreadable) target.
    Plain df.to_parquet() writes in place, so an interrupted write destroys the
    only copy; that corrupted the bet ledger once (2026-07-19)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_json(obj, path) -> None:
    """Write JSON atomically (temp file + os.replace), same rationale as
    atomic_to_parquet."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_ledger(df) -> None:
    atomic_to_parquet(df, BET_LEDGER_PATH, compression=LEDGER_COMPRESSION)


@lru_cache(maxsize=1)
def load_config() -> dict:
    config_path = REPO_ROOT / "config" / "config.yaml"
    example_path = REPO_ROOT / "config" / "config.example.yaml"
    path = config_path if config_path.exists() else example_path
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (RAW_TRADES_DIR, INTERIM_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def make_session(max_retries: int = 4, backoff: float = 1.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def print_disk_usage_summary() -> None:
    cfg = load_config()
    cap_gb = cfg.get("disk", {}).get("cap_gb", 5)
    total_bytes = _dir_size_bytes(DATA_DIR)
    total_gb = total_bytes / (1024**3)
    print(f"\n[disk] data/ usage: {total_gb:.3f} GB (cap: {cap_gb} GB)")
    for label, path in (
        ("raw", RAW_DIR),
        ("interim", INTERIM_DIR),
        ("processed", PROCESSED_DIR),
    ):
        gb = _dir_size_bytes(path) / (1024**3)
        print(f"  - {label}: {gb:.3f} GB ({path})")
    if total_gb > cap_gb:
        print(
            f"[disk] WARNING: data/ usage ({total_gb:.3f} GB) exceeds configured "
            f"cap ({cap_gb} GB). Consider lowering ingest.raw_batches_to_keep or "
            "pruning data/interim/."
        )
