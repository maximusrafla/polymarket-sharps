"""Resolve ONLY the slow markets in the deepened dataset (Project 3 step 5b→5c bridge).

The deepening (src/slow_deepen.py) pulls each shortlisted wallet's FULL history, which
includes hundreds of thousands of fast micro-crypto markets 5c never scores. Resolving
all of them is ~3.4h of CLOB GETs; resolving only the SLOW markets (the ones 5c
validates) is ~40min. This script does that, in RESUMABLE 5k-market chunks (saves
data/interim/slow_deepening/deep_resolutions.parquet after each chunk, skips
already-resolved on a re-run), then refreshes deep_trades' resolved/resolved_value.

Read-only CLOB GETs, isolated to the deepening dir (shared cache read-only). Idempotent.

    PYTHONPATH=. .venv/bin/python scripts/resolve_slow_deep.py [--chunk 5000]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.common import RESOLUTIONS_PATH, load_config, make_session
from src.ingest import refresh_ledger_resolutions, update_resolutions
from src.market_meta import classify_speed_bucket, load_market_meta, speed_thresholds
from src.slow_deepen import (
    RESOLUTION_COLUMNS,
    load_deep_resolutions,
    load_deep_trades,
    save_deep_resolutions,
    save_deep_trades,
)
from src.slow_market import SLOW_BUCKETS


def _resolved_ids(df: pd.DataFrame) -> set:
    return set(df.loc[df["closed"].fillna(False), "market_id"]) if not df.empty else set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve only the slow deep markets (resumable).")
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_config()
    sess = make_session()
    slow_s, deep_s = speed_thresholds(cfg)

    deep = load_deep_trades()
    if deep.empty:
        print("[resolve_slow] no deep dataset — run slow_deepen first.")
        return

    # Classify each deep market's speed: sidecar lifespan preferred, else deep tape span.
    span = deep.groupby("market_id")["timestamp"].agg(["min", "max"])
    deep_life = (span["max"] - span["min"]).to_dict()
    meta = load_market_meta(columns=["market_id", "lifespan_s"])
    sidecar = dict(zip(meta["market_id"], meta["lifespan_s"])) if not meta.empty else {}

    def bucket(m):
        L = sidecar.get(m)
        if L is None or (isinstance(L, float) and np.isnan(L)):
            L = deep_life.get(m, np.nan)
        return classify_speed_bucket(L, slow_s, deep_s)

    slow_markets = {m for m in deep_life if bucket(m) in SLOW_BUCKETS}

    shared = pd.read_parquet(RESOLUTIONS_PATH) if RESOLUTIONS_PATH.exists() else pd.DataFrame(columns=RESOLUTION_COLUMNS)
    deep_res = load_deep_resolutions()
    todo = sorted(slow_markets - _resolved_ids(shared) - _resolved_ids(deep_res))
    print(f"[resolve_slow] {len(slow_markets):,} slow deep markets; {len(todo):,} need resolution", flush=True)

    t0 = time.time()
    for i in range(0, len(todo), args.chunk):
        chunk = set(todo[i:i + args.chunk])
        deep_res = update_resolutions(sess, cfg, chunk, deep_res, max_fetches=None)
        save_deep_resolutions(deep_res)
        done = min(i + args.chunk, len(todo))
        el = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta = (len(todo) - done) / rate / 60 if rate > 0 else 0.0
        print(f"[resolve_slow]   {done:,}/{len(todo):,} ({rate:.0f}/s) eta {eta:.0f}min", flush=True)

    combined = pd.concat(
        [shared[RESOLUTION_COLUMNS] if not shared.empty else shared, deep_res], ignore_index=True)
    combined = combined.drop_duplicates(subset=["market_id", "token_id"], keep="last")
    enriched = refresh_ledger_resolutions(deep, combined)
    save_deep_trades(enriched)
    n_res = int(enriched["resolved"].fillna(False).sum())
    print(f"[resolve_slow] DONE: deep dataset {n_res:,}/{len(enriched):,} resolved", flush=True)


if __name__ == "__main__":
    main()
