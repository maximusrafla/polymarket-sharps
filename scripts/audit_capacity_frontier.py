"""CAPACITY, done properly: the fill-rate / cost frontier under a slippage cap.

WHY THIS REPLACES scripts/audit_capacity.py
-------------------------------------------
The 2026-07-29 red-team (docs/redteam_realworld_copy_2026-07-29.md §2.5) found
three defects in the original probe, all pushing the same way:

  1. It reports MEDIAN slippage. For a portfolio of equal-dollar bets the total
     cost is n x MEAN, so the mean is the right aggregate and it is 5-8x larger
     ($100: median 0.48c, mean 3.94c). Both are reported here, and the frontier
     is built on the mean.
  2. It compares against a hardcoded `MEASURED_EDGE_C = 2.02`, which is the
     SUPERSEDED pre-Correction-2 figure, applied to a group defined by the
     superseded verdict (one of whose two members is no longer copyable). Edges
     here are read PER WALLET from the re-derived table and never hardcoded.
  3. It walks an UNCAPPED market order. `paper_rw`'s own known_limits says that
     is a LOWER bound and that the obvious refinement is a max-slippage cap. No
     real copier pays whatever is resting. Under a cap the pathological books
     simply do not trade, which turns a bad average into a design parameter.

It also never reported its own denominator. Survivorship is the largest single
uncertainty in any live-book probe — resolved markets have no book — so this
script counts every token it TRIES and reports the share that still had one.

WHAT IT COMPUTES
  * DEPTH: dollars resting within `cap` cents of the best ask, per token. This is
    the capacity number in its rawest form and needs no size ladder.
  * FRONTIER: for each (order size, slippage cap), the fill rate, the mean/median
    realised slippage and the expected dollars per signal
        $/signal = shares_filled x (edge_per_share - slippage_per_share)
    with unfilled budget earning exactly zero. Then $/month using each wallet's
    MEASURED recent signal rate from the deep tape.

WHAT IT STILL CANNOT SAY
  * The book is TODAY's, not the one standing when each wallet traded. That is
    the right object for a forward decision but it is not a backtest.
  * Only markets still OPEN have a book, so the walked sample skews long-lived
    and liquid. The rate is reported rather than hidden.
  * Ask side only: this says nothing about exiting.
  * Market impact is not modelled -- displayed size is consumed without moving.

READ-ONLY: public unauthenticated GETs, no keys, no signing, nothing on-chain.
Writes only data/interim/capacity/.

    PYTHONPATH=. .venv/bin/python scripts/audit_capacity_frontier.py
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
BOOKS_PATH = CAP_DIR / "frontier_books.parquet"
DEPTH_PATH = CAP_DIR / "frontier_depth.parquet"
CURVE_PATH = CAP_DIR / "frontier_curve.parquet"
PER_WALLET = INTERIM_DIR / "copysim" / "redteam" / "per_wallet.parquet"
DEEP = INTERIM_DIR / "realworld" / "deep_trades"

# The five wallets in the LIVE forward arm's headline stratum (rw5_copyable_v2).
# 14 chars, matching the key used everywhere else in the copy analysis — the
# manifest spells them with 12, which silently matches nothing.
LIVE5 = ["0x253da8157571", "0x9fc043287797", "0x69ea0d77ef34",
         "0xd06f0f7719df", "0x05c5aab002fa"]

# Slippage measured on a 0.2c book does not transfer to a wallet trading at 0.80,
# and converting a per-share edge into dollars at an arbitrary price explodes
# (at p=0.001 a $10 stake buys 10,000 shares, and 10,000 x 2.3c reads as $230).
# The frontier is therefore computed only where the cohort actually trades.
PRICE_BAND = (0.10, 0.95)

SIZES_USD = [10, 25, 50, 100, 250, 500, 1000, 2500]
# Slippage caps in cents over the best ask. `None` = uncapped market order, the
# arm the original probe measured, kept so the comparison is visible.
CAPS_C = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, None]
SEED = 20260729


def recent_tokens(days: int) -> pd.DataFrame:
    """(wallet, token_id) each scored wallet BOUGHT inside the window, with the
    last trade timestamp so token age can be reported."""
    per = pd.read_parquet(PER_WALLET)
    prefixes = set(per["wallet"])
    keep = []
    for p in sorted(glob.glob(str(DEEP / "part-*.parquet"))):
        d = pd.read_parquet(p, columns=["wallet", "market_id", "token_id", "side",
                                        "timestamp", "entry_price", "size"])
        d = d[(d["side"] == "BUY") & d["wallet"].str[:14].isin(prefixes)]
        if len(d):
            keep.append(d)
        del d
        gc.collect()
    t = pd.concat(keep, ignore_index=True)
    cut = t["timestamp"].max() - days * 86400
    t = t[t["timestamp"] >= cut].copy()
    t["w"] = t["wallet"].str[:14]
    g = (t.groupby(["w", "token_id"])
           .agg(last_ts=("timestamp", "max"), n_fills=("timestamp", "size"),
                mean_px=("entry_price", "mean"),
                stake=("size", lambda s: float((s).sum())))
           .reset_index())
    g["age_days"] = (t["timestamp"].max() - g["last_ts"]) / 86400.0
    return g, t


def signals_per_month(trades: pd.DataFrame, days: int) -> pd.Series:
    """Each wallet's MEASURED new-position rate: distinct (token) first-buys per
    30 days over the window. A repeat fill in a token already held is not a new
    signal for a copier who sizes once per position."""
    first = trades.groupby(["w", "token_id"])["timestamp"].min().reset_index()
    n = first.groupby("w").size()
    return (n * 30.0 / days).rename("signals_per_month")


def depth_within(asks, cap_c: float) -> float:
    """Dollars resting at or below best_ask + cap."""
    if not asks:
        return 0.0
    lim = asks[0][0] + cap_c / 100.0
    return float(sum(p * s for p, s in asks if p <= lim + 1e-12))


def probe(tokens: pd.DataFrame, session, cfg, max_tokens: int, rng,
          sleep_s: float = 0.06) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk every live book at every (size, cap). Counts tokens TRIED so the
    survivorship rate is a reported number, not an unknown."""
    if len(tokens) > max_tokens:
        tokens = tokens.sample(max_tokens, random_state=rng)   # random, not newest-first
    rows, depths = [], []
    tried = live = 0
    for _, r in tokens.iterrows():
        tried += 1
        try:
            book = parse_book(fetch_book(session, cfg, r["token_id"]))
        except Exception as exc:                      # noqa: BLE001
            print(f"[cap] warn {str(r['token_id'])[:12]}: {exc}", flush=True)
            continue
        asks = (book or {}).get("asks") or []
        if not asks:
            continue                                  # resolved/closed: no book
        live += 1
        best = float(asks[0][0])
        depths.append({"w": r["w"], "token_id": r["token_id"], "best_ask": best,
                       "age_days": r["age_days"], "wallet_px": r["mean_px"],
                       "total_depth_usd": float(sum(p * s for p, s in asks)),
                       **{f"depth_{c}c": depth_within(asks, c)
                          for c in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)}})
        for usd in SIZES_USD:
            for cap in CAPS_C:
                lim = None if cap is None else best + cap / 100.0
                wres = walk_book(asks, usd, limit_price=lim)
                rows.append({
                    "w": r["w"], "token_id": r["token_id"], "size_usd": usd,
                    "cap_c": -1.0 if cap is None else cap,
                    "best_ask": best, "shares": wres["shares"],
                    "filled_usd": wres["cost_usd"],
                    "fill_frac": wres["cost_usd"] / usd,
                    "avg_price": wres["avg_price"],
                    "slippage_c": (100.0 * (wres["avg_price"] - best)
                                   if wres["shares"] > 0 else np.nan),
                })
        if live % 25 == 0:
            print(f"[cap] {live} live books / {tried} tried", flush=True)
        time.sleep(sleep_s)
    return (pd.DataFrame(rows), pd.DataFrame(depths),
            {"tried": tried, "live": live,
             "live_rate": live / tried if tried else float("nan")})


def frontier(books: pd.DataFrame, edge_c: pd.Series) -> pd.DataFrame:
    """The cost frontier at each (order size, slippage cap).

    Deliberately PRICE-FREE. The quantity that decides the thesis is *what
    fraction of the measured edge does size cost*, and that is a ratio of two
    per-share numbers — no conversion to dollars, and therefore no dependence on
    the probed token's price level. `edge_kept` is the honest headline:
    (edge - slippage) / edge, weighted by how much of the budget actually filled,
    so a book that only takes a third of the order is credited a third of the
    edge and no more."""
    b = books[(books["best_ask"] >= PRICE_BAND[0])
              & (books["best_ask"] <= PRICE_BAND[1])].copy()
    b["edge_c"] = b["w"].map(edge_c)
    b = b.dropna(subset=["edge_c"])
    b = b[b["edge_c"] > 0]                       # erosion is undefined against a
    out = []                                     # non-positive edge
    for (usd, cap), g in b.groupby(["size_usd", "cap_c"]):
        filled = g[g["shares"] > 0]
        slip = g["slippage_c"].fillna(0.0)
        kept = (g["fill_frac"].clip(0, 1) * (g["edge_c"] - slip) / g["edge_c"])
        out.append({
            "size_usd": usd, "cap_c": cap, "n_books": g["token_id"].nunique(),
            "full_fill_rate": float((g["fill_frac"] > 0.99).mean()),
            "mean_fill_frac": float(g["fill_frac"].mean()),
            "median_slip_c": float(filled["slippage_c"].median()) if len(filled) else np.nan,
            "mean_slip_c": float(filled["slippage_c"].mean()) if len(filled) else np.nan,
            "mean_edge_kept": float(kept.mean()),
            "median_edge_kept": float(kept.median()),
        })
    return pd.DataFrame(out).sort_values(["cap_c", "size_usd"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="tokens to try per group (each costs one GET)")
    ap.add_argument("--reuse", action="store_true",
                    help="re-analyse the cached books instead of re-probing the "
                         "live API (the analysis is pure; the probe is not)")
    args = ap.parse_args()

    CAP_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    cfg, session = load_config(), make_session()

    per = pd.read_parquet(PER_WALLET)
    edge_c = per.set_index("wallet")["net_c"]
    pooled_edge = float((per["net_c"] * per["n"]).sum() / per["n"].sum())
    print(f"[cap] per-wallet edges from the re-derived table: median "
          f"{edge_c.median():.2f}c, bet-weighted pooled {pooled_edge:.2f}c "
          f"(NOT a hardcoded constant)")

    tokens, trades = recent_tokens(args.days)
    spm = signals_per_month(trades, args.days)
    print(f"[cap] {len(tokens):,} distinct (wallet, token) positions opened in the "
          f"last {args.days}d by {tokens['w'].nunique()} wallets")

    groups = {"cohort37": sorted(per["wallet"]), "live5": LIVE5}
    missing = [w for w in LIVE5 if w not in set(per["wallet"])]
    if missing:
        raise SystemExit(f"LIVE5 does not resolve against the scored table: {missing}")

    if args.reuse and BOOKS_PATH.exists():
        books = pd.read_parquet(BOOKS_PATH)
        depth = pd.read_parquet(DEPTH_PATH)
        stats = {g: {"live": int(d["token_id"].nunique()), "tried": int(args.max_tokens),
                     "live_rate": d["token_id"].nunique() / args.max_tokens}
                 for g, d in depth.groupby("group")}
        print(f"[cap] REUSING cached books ({len(depth)} live books) — no network")
    else:
        all_books, all_depth, stats = [], [], {}
        for name, members in groups.items():
            sub = tokens[tokens["w"].isin(members)]
            print(f"\n[cap] {name}: {len(sub):,} distinct tokens in window; trying up "
                  f"to {args.max_tokens}", flush=True)
            if not len(sub):
                continue
            b, d, st = probe(sub, session, cfg, args.max_tokens, rng)
            stats[name] = st
            print(f"[cap] {name}: {st['live']} live books of {st['tried']} tried "
                  f"({100*st['live_rate']:.1f}% still open)")
            if len(b):
                b["group"] = name; d["group"] = name
                all_books.append(b); all_depth.append(d)
        if not all_books:
            print("[cap] no live books found.")
            return
        books = pd.concat(all_books, ignore_index=True)
        depth = pd.concat(all_depth, ignore_index=True)
        atomic_to_parquet(books, BOOKS_PATH, compression="zstd", index=False)
        atomic_to_parquet(depth, DEPTH_PATH, compression="zstd", index=False)

    print(f"\n{'='*94}\nDEPTH — dollars resting within N cents of the best ask\n{'='*94}")
    for grp, g in depth.groupby("group"):
        print(f"\n{grp}  ({len(g)} live books, median token age "
              f"{g['age_days'].median():.1f}d)")
        print(f"{'cap':>6}{'p25 $':>12}{'median $':>12}{'mean $':>12}{'p75 $':>12}")
        for c in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            col = g[f"depth_{c}c"]
            print(f"{c:>5.1f}c{col.quantile(.25):>12,.0f}{col.median():>12,.0f}"
                  f"{col.mean():>12,.0f}{col.quantile(.75):>12,.0f}")

    curves = []
    for grp, g in books.groupby("group"):
        c = frontier(g, edge_c)
        c["group"] = grp
        curves.append(c)
        members = groups[grp]
        sig = float(spm.reindex(members).dropna().sum())
        inband = g[(g["best_ask"] >= PRICE_BAND[0]) & (g["best_ask"] <= PRICE_BAND[1])]
        print(f"\n{'='*94}\nFRONTIER — {grp}   ({inband['token_id'].nunique()} of "
              f"{g['token_id'].nunique()} live books inside the "
              f"{PRICE_BAND[0]:.2f}-{PRICE_BAND[1]:.2f} price band; "
              f"{sig:,.0f} new positions/month across {len(members)} wallets)\n{'='*94}")
        print("  edge kept = (edge - slippage)/edge, scaled by the share of the order "
              "that actually filled")
        print(f"{'cap':>7}{'size':>8}{'filled%':>9}{'full fill':>11}"
              f"{'med slip':>10}{'mean slip':>11}{'edge kept':>11}")
        for _, r in c.iterrows():
            cap = "none" if r.cap_c < 0 else f"{r.cap_c:.1f}c"
            print(f"{cap:>7}${r.size_usd:>7,.0f}{100*r.mean_fill_frac:>8.0f}%"
                  f"{100*r.full_fill_rate:>10.0f}%{r.median_slip_c:>9.2f}c"
                  f"{r.mean_slip_c:>10.2f}c{100*r.mean_edge_kept:>10.0f}%")
        best = c.loc[c["mean_edge_kept"].idxmax()]
        capl = "uncapped" if best.cap_c < 0 else f"{best.cap_c:.1f}c cap"
        print(f"  BEST EDGE RETENTION: ${best.size_usd:,.0f}/bet at {capl} -> "
              f"{100*best.mean_edge_kept:.0f}% of the edge kept "
              f"({100*best.mean_fill_frac:.0f}% of the order filled, mean slip "
              f"{best.mean_slip_c:.2f}c)")
        for target in (0.90, 0.75, 0.50):
            ok = c[c["mean_edge_kept"] >= target]
            if len(ok):
                r = ok.loc[ok["size_usd"].idxmax()]
                capl = "uncapped" if r.cap_c < 0 else f"{r.cap_c:.1f}c cap"
                print(f"  largest size keeping >= {100*target:.0f}% of the edge: "
                      f"${r.size_usd:,.0f} at {capl}")
        print(f"  SURVIVORSHIP: measured on {stats[grp]['live']} of "
              f"{stats[grp]['tried']} tokens tried "
              f"({100*stats[grp]['live_rate']:.1f}% still open) — the rest had "
              f"resolved and cannot be measured. Read every number above as "
              f"conditional on that.")
    atomic_to_parquet(pd.concat(curves, ignore_index=True), CURVE_PATH,
                      compression="zstd", index=False)
    print(f"\n[cap] -> {BOOKS_PATH}\n[cap] -> {DEPTH_PATH}\n[cap] -> {CURVE_PATH}")


if __name__ == "__main__":
    main()
