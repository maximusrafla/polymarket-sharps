"""Build the drill-down index for the local forensics server.

WHY AN INDEX AND NOT A NEW TABLE. Full per-bet detail for the 1,628 scored wallets
is 3.27M rows / ~129 MB as JSON — far too much to inline in the dashboard, and
duplicating it as its own parquet would cost ~60-90 MB on a box that is 87% full.
But `deep_trades` shards are append-only fetch checkpoints, so a given wallet's
trades landed in only one or two of the 88 shards. Mapping wallet -> shard ids lets
a request read one ~6 MB shard instead of scanning the whole set, with no data
duplication at all.

Read-only. Writes data/interim/forensics/shard_index.json (wallet -> [shard ids],
plus the shard path list).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "data" / "interim" / "realworld"
OUT = ROOT / "data" / "interim" / "forensics"


def main() -> None:
    scored = set(json.loads(
        (ROOT / "data" / "processed" / "forensics_dashboard.json").read_text()
    )["wallets"]["wallet"])
    shards = sorted(glob.glob(str(RW / "deep_trades" / "*.parquet")))
    idx: dict[str, list[int]] = {}
    for i, p in enumerate(shards):
        w = pd.read_parquet(p, columns=["wallet"])["wallet"].unique()
        for x in w:
            if x in scored:
                idx.setdefault(x, []).append(i)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(shards)}]", flush=True)
    spread = np.array([len(v) for v in idx.values()])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shard_index.json").write_text(json.dumps({
        "shards": [str(Path(s).relative_to(ROOT)) for s in shards],
        "wallets": idx,
    }, separators=(",", ":")))
    print(f"indexed {len(idx):,} wallets over {len(shards)} shards")
    print(f"shards per wallet: median {np.median(spread):.0f}, "
          f"mean {spread.mean():.1f}, max {spread.max()}")
    print(f"wrote {OUT/'shard_index.json'} "
          f"({(OUT/'shard_index.json').stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
