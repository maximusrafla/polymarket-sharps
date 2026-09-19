"""CAPACITY: at what order size does slippage eat the copyable edge?

THE QUESTION. `docs/copy_verdict.md` measured that a follower of the two
copy-validated wallets keeps **+2.02c/share** at a 2-minute lag, net of the real
taker fee and the market's actual quoted tick. But one tick prices a *trivially
small* order resting at the top of the book. The paper-trade smoke test filled a
$100 market order at an average 9.6c where the wallet paid 0.8c on a thin book.
So the honest statement is that the edge is size-dependent and nobody has measured
the curve. This does.

METHOD. For each market these wallets actually trade, fetch the LIVE CLOB order
book and walk it (`paper_trader.walk_book`, the same code the paper arm fills
against) at a ladder of dollar sizes. Slippage is `avg_fill_price - best_ask`, in
cents per share. Capacity is the largest size whose slippage plus fee still leaves
the measured edge positive.

WHAT THIS CAN AND CANNOT SAY.
  * It measures the book as it is NOW, not as it was at the moment each wallet
    bet. Historical books are not retrievable, so this is a forward-looking proxy
    — which is the right object anyway, since the question is what a copier faces
    from here.
  * Only markets still OPEN have a book. Resolved markets are silently absent, so
    the sample skews toward longer-lived markets. Reported, not hidden.
  * It walks the ask side only (a copier buys). It says nothing about exiting.

READ-ONLY: public unauthenticated GETs, no keys, no signing, nothing on-chain.
Writes only `data/interim/capacity/`.
"""

from __future__ import annotations

import argparse
import gc
import glob
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.common import INTERIM_DIR, atomic_to_parquet, load_config, make_session  # noqa: E402
from src.paper_trader import fetch_book, parse_book, walk_book  # noqa: E402

CAP_DIR = INTERIM_DIR / "capacity"
BOOKS_PATH = CAP_DIR / "books.parquet"
CURVE_PATH = CAP_DIR / "curve.parquet"

# The two wallets whose edge survived the copy lag (docs/copy_verdict.md), plus
# the three that did not — kept so the comparison is visible rather than assumed.
COPY_VALIDATED = ["0x1ee9a5fc09", "0x69ea0d77ef"]
OTHERS = ["0x09bed19766", "0xe542afd388", "0x83255595ba"]

SIZES_USD = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
MEASURED_EDGE_C = 2.02          # docs/copy_verdict.md, follower net at a 2-min lag
RECENT_DAYS = 45                # how far back to look for markets they trade


def recent_markets(prefixes: list[str], days: int = RECENT_DAYS) -> pd.DataFrame:
    """(wallet, market_id, token_id) the wallet bought recently, newest first."""
    keep = []
    for p in sorted(glob.glob(str(INTERIM_DIR / "realworld/deep_trades/part-*.parquet"))):
        d = pd.read_parquet(p, columns=["wallet", "market_id", "token_id", "side",
                                        "timestamp", "entry_price", "size"])
        d = d[d["wallet"].str[:12].isin(prefixes) & (d["side"] == "BUY")]
        if len(d):
            keep.append(d)
        del d
        gc.collect()
    if not keep:
        return pd.DataFrame(columns=["wallet", "market_id", "token_id"])
    t = pd.concat(keep, ignore_index=True)
    cut = t["timestamp"].max() - days * 86400
    t = t[t["timestamp"] >= cut]
    t["stake"] = t["size"] * t["entry_price"]
    return t.sort_values("timestamp", ascending=False)


def probe_books(tokens: pd.DataFrame, session, cfg, max_tokens: int,
                sleep_s: float = 0.05) -> pd.DataFrame:
    """Walk each live book at every ladder size. Absent book -> market closed."""
    rows = []
    seen = 0
    for _, r in tokens.iterrows():
        if seen >= max_tokens:
            break
        try:
            book = parse_book(fetch_book(session, cfg, r["token_id"]))
        except Exception as exc:  # noqa: BLE001 — one dead token must not end the probe
            print(f"[capacity] warn {r['token_id'][:12]}: {exc}", flush=True)
            continue
        asks = (book or {}).get("asks") or []
        if not asks:
            continue                       # resolved/closed: no book to walk
        seen += 1
        best = float(asks[0][0])
        depth = sum(p * s for p, s in asks)
        for usd in SIZES_USD:
            w = walk_book(asks, usd)
            if not w["shares"]:
                continue
            rows.append({
                "wallet": r["wallet"], "market_id": r["market_id"],
                "token_id": r["token_id"], "size_usd": usd,
                "best_ask": best, "avg_price": w["avg_price"],
                "filled_usd": w["cost_usd"], "shares": w["shares"],
                "levels": w["levels_used"], "book_depth_usd": depth,
                "slippage_c": 100.0 * (w["avg_price"] - best),
                "filled_frac": w["cost_usd"] / usd,
            })
        if seen % 25 == 0:
            print(f"[capacity] {seen}/{max_tokens} live books walked", flush=True)
        time.sleep(sleep_s)
    return pd.DataFrame(rows)


def curve(books: pd.DataFrame, edge_c: float = MEASURED_EDGE_C) -> pd.DataFrame:
    """Median slippage by size, and what is left of the measured edge."""
    out = []
    for (grp, usd), g in books.groupby(["group", "size_usd"]):
        out.append({
            "group": grp, "size_usd": usd, "n_books": g["token_id"].nunique(),
            "median_slip_c": g["slippage_c"].median(),
            "mean_slip_c": g["slippage_c"].mean(),
            "p75_slip_c": g["slippage_c"].quantile(0.75),
            "median_filled_frac": g["filled_frac"].median(),
            "frac_fully_filled": float((g["filled_frac"] > 0.99).mean()),
            "edge_left_c": edge_c - g["slippage_c"].median(),
        })
    return pd.DataFrame(out).sort_values(["group", "size_usd"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tokens", type=int, default=150,
                    help="live books to walk per group (each costs one GET)")
    ap.add_argument("--days", type=int, default=RECENT_DAYS)
    args = ap.parse_args()

    CAP_DIR.mkdir(parents=True, exist_ok=True)
    cfg, session = load_config(), make_session()
    frames = []
    for label, prefixes in (("copy_validated", COPY_VALIDATED), ("others", OTHERS)):
        t = recent_markets(prefixes, args.days).drop_duplicates("token_id")
        print(f"\n[capacity] {label}: {len(t):,} distinct tokens traded in the last "
              f"{args.days}d; walking up to {args.max_tokens} live books", flush=True)
        if not len(t):
            continue
        b = probe_books(t, session, cfg, args.max_tokens)
        if len(b):
            b["group"] = label
            frames.append(b)
        print(f"[capacity] {label}: {b['token_id'].nunique() if len(b) else 0} live books",
              flush=True)

    if not frames:
        print("[capacity] no live books found — every sampled market has closed.")
        return
    books = pd.concat(frames, ignore_index=True)
    atomic_to_parquet(books, BOOKS_PATH, compression="zstd", index=False)
    c = curve(books)
    atomic_to_parquet(c, CURVE_PATH, compression="zstd", index=False)

    print(f"\n{'='*100}\nSLIPPAGE vs ORDER SIZE — measured edge is {MEASURED_EDGE_C:.2f}c/share")
    print(f"{'='*100}")
    for grp, g in c.groupby("group"):
        print(f"\n{grp}  (books walked: {books[books.group==grp]['token_id'].nunique()})")
        print(f"{'size':>8}{'books':>7}{'median slip':>13}{'p75 slip':>10}"
              f"{'edge left':>11}{'% fully filled':>16}")
        for _, r in g.iterrows():
            flag = "  <-- edge gone" if r["edge_left_c"] <= 0 else ""
            print(f"${r.size_usd:>7,.0f}{r.n_books:>7}{r.median_slip_c:>12.2f}c"
                  f"{r.p75_slip_c:>9.2f}c{r.edge_left_c:>10.2f}c"
                  f"{100*r.frac_fully_filled:>15.0f}%{flag}")
    print(f"\n[capacity] -> {BOOKS_PATH}\n[capacity] -> {CURVE_PATH}")


if __name__ == "__main__":
    main()
