"""Does the bid-ask spread kill the fade? — 2026-07-30

THE FADE, AND WHY SPREAD IS ITS SPECIFIC PROBLEM. Fading a wallet means: when it
buys outcome X, you buy NOT-X. The gross estimate credits the fader at exactly
(1 − p), the complement of the wallet's fill. That is wrong in a knowable
direction. If the wallet paid the ask on X, then

    ask(X)  = p                 (what the wallet paid)
    bid(X)  = p − s             (s = the bid-ask spread)
    ask(¬X) = 1 − bid(X) = 1 − p + s

so the fader pays **1 − p + s**, i.e. the FULL spread on top of the fee — not half
of it, and not none of it. Crossing to the other side of the book is the whole
mechanic of the trade. The fade's headline (+0.33¢ to +0.49¢ net of fee at the
bottom decile) does not include this at all, and the repo's own live probe has
previously reported a ~1.0¢ median spread, which would be enough to erase it.

WHAT THIS MEASURES. The bottom-decile wallets' own markets have resolved, so their
books cannot be re-read. Instead: measure the live spread on currently-open markets,
stratified to match the cohort's own category and entry-price mix, and apply the
measured distribution to the fade estimate. Reported as a distribution, never a
single number, because the median hides that spread is strongly price-dependent.

Read-only public unauthenticated GETs. No keys, no orders, nothing on-chain.
Writes data/interim/forensics/fade_spread.json.
Run:  PYTHONPATH=. .venv/bin/python scripts/fade_spread_cost.py [--max-books N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config                      # noqa: E402
from src.paper_trader import fetch_book, parse_book     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data" / "interim" / "forensics"
GAMMA = "https://gamma-api.polymarket.com/markets"
FEE_K = 0.05
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def log(m): print(m, flush=True)


def cohort_price_mix() -> tuple[list[str], pd.Series]:
    """Bottom-decile wallets by event-disjoint H1, and their entry-price histogram."""
    P = pd.read_parquet(F / "persistence.parquet")
    prof = pd.read_parquet(F / "wallet_profile.parquet")
    d = P.merge(prof[["wallet", "n_events", "mean_price"]], on="wallet")
    k = d[(d.n_bets >= 100) & (d.n_events >= 30)
          & (d.n1_disjoint >= 20) & (d.n2_disjoint >= 20)]
    cut = k.h1_disjoint.quantile(0.10)
    bot = k[k.h1_disjoint <= cut]
    bands = pd.read_parquet(F / "wallet_bands.parquet")
    mix = bands[bands.wallet.isin(set(bot.wallet))].groupby("band").n_bets.sum()
    mix = (mix / mix.sum()).sort_index(key=lambda s: [int(x.split("-")[0]) for x in s])
    return list(bot.wallet), mix


def open_markets(session, cfg, want: int) -> list[dict]:
    """Currently-open, tradeable markets from Gamma, highest volume first."""
    # Gamma refuses deep offsets (422 past ~2.5k rows), so paginate several orderings
    # and union them rather than trying to walk one list to the end.
    out, seen_ids = [], set()
    for order in ("volumeNum", "liquidityNum", "startDate"):
        offset = 0
        while len(out) < want and offset < 2400:
            try:
                r = session.get(GAMMA, params={"closed": "false", "active": "true",
                                               "limit": 200, "offset": offset,
                                               "order": order, "ascending": "false"},
                                timeout=cfg["ingest"]["request_timeout_sec"])
                if r.status_code == 422:
                    break
                r.raise_for_status()
                page = r.json()
            except requests.RequestException:
                break
            if not page:
                break
            for m in page:
                mid = m.get("id")
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    out.append(m)
            offset += 200
            time.sleep(0.15)
    return out


def gamma_price(m: dict) -> float | None:
    """Best available price hint from Gamma, so books are only fetched for markets
    already in a band that still needs samples. Without this pre-filter the sample
    is ~87% extreme-priced (the venue's open book is dominated by near-resolved
    markets), which is what made the first pass thin exactly where it mattered."""
    for key in ("lastTradePrice", "bestAsk"):
        v = m.get(key)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    op = m.get("outcomePrices")
    try:
        op = json.loads(op) if isinstance(op, str) else op
        if op:
            return float(op[0])
    except (TypeError, ValueError):
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-books", type=int, default=400)
    ap.add_argument("--per-band", type=int, default=60,
                    help="books required per price band before stopping")
    a = ap.parse_args()
    cfg = load_config()
    session = requests.Session()

    wallets, mix = cohort_price_mix()
    log(f"[cohort] bottom-decile wallets: {len(wallets)}")
    log("[cohort] their entry-price mix (share of bets):")
    for b, v in mix.items():
        log(f"    {b:>10}  {100*v:5.1f}%")

    log(f"\n[books] pulling open markets from Gamma…")
    mkts = open_markets(session, cfg, 8000)
    log(f"  {len(mkts):,} open markets")

    # PER-BAND QUOTAS. The venue's open book is dominated by near-resolved markets,
    # so an unstratified sample lands ~87% in 0-20c and leaves the mid bands — which
    # carry the cohort's weight — on a handful of books. Quota each band instead.
    def band_of_price(p):
        for lo, hi in BANDS:
            if lo <= p < hi:
                return f"{int(lo*100)}-{int(hi*100)}c"
        return None

    quota = a.per_band
    got = {f"{int(lo*100)}-{int(hi*100)}c": 0 for lo, hi in BANDS}
    rows, seen, skipped = [], 0, 0
    for m in mkts:
        if all(v >= quota for v in got.values()):
            break
        gp = gamma_price(m)
        if gp is None:
            continue
        want_band = band_of_price(gp)
        if want_band is None or got[want_band] >= quota:
            skipped += 1
            continue
        try:
            toks = m.get("clobTokenIds")
            toks = json.loads(toks) if isinstance(toks, str) else toks
            if not toks:
                continue
        except (ValueError, TypeError):
            continue
        try:
            bk = parse_book(fetch_book(session, cfg, str(toks[0])))
        except Exception:
            continue
        if not bk["asks"] or not bk["bids"]:
            continue
        ask, bid = bk["asks"][0][0], bk["bids"][0][0]
        if not (0 < bid < ask < 1):
            continue
        mid = (ask + bid) / 2
        b = band_of_price(mid)                     # bucket on the BOOK, not the hint
        if b is None:
            continue
        got[b] += 1
        rows.append({"mid": mid, "spread_c": 100 * (ask - bid),
                     "ask_sz": bk["asks"][0][1], "vol": float(m.get("volumeNum") or 0)})
        seen += 1
        if seen % 100 == 0:
            log(f"  {seen} books… {got}")
        time.sleep(0.05)
    log(f"  pre-filtered past {skipped:,} markets whose band was already full")
    B = pd.DataFrame(rows)
    log(f"\n[books] usable two-sided books: {len(B):,}")
    if B.empty:
        log("no books — aborting"); return

    log("\n[spread] by price band (cents, full bid-ask):")
    log(f"  {'band':>10}{'n':>6}{'median':>9}{'p25':>8}{'p75':>8}{'mean ask $':>12}")
    per_band = {}
    for lo, hi in BANDS:
        s = B[(B["mid"] >= lo) & (B["mid"] < hi)]
        if not len(s):
            continue
        per_band[f"{int(lo*100)}-{int(hi*100)}c"] = float(s.spread_c.median())
        log(f"  {f'{int(lo*100)}-{int(hi*100)}c':>10}{len(s):>6}"
            f"{s.spread_c.median():>8.2f}¢{s.spread_c.quantile(.25):>7.2f}¢"
            f"{s.spread_c.quantile(.75):>7.2f}¢{s.ask_sz.mean()*s.mid.mean():>11.0f}")

    # cohort-weighted spread: match the bottom decile's own price mix
    band_key = {f"{a0}-{a0+10}c": None for a0 in range(0, 100, 10)}
    def band_of(p):
        for lo, hi in BANDS:
            if lo <= p < hi:
                return f"{int(lo*100)}-{int(hi*100)}c"
        return "80-100c"
    w, tot = 0.0, 0.0
    for b, share in mix.items():
        mid = (int(b.split("-")[0]) + 5) / 100.0
        k = band_of(mid)
        if k in per_band:
            w += share * per_band[k]; tot += share
    cohort_spread = w / tot if tot else float(B.spread_c.median())

    log(f"\n[spread] cohort-weighted median spread: {cohort_spread:.2f}¢")
    log(f"[spread] unweighted median:              {B.spread_c.median():.2f}¢")

    dw = json.loads((F / "dollar_weighted.json").read_text())
    log("\n" + "=" * 74)
    log("THE FADE, WITH SPREAD CHARGED IN FULL")
    log("=" * 74)
    log(f"  {'weighting/cut':<24}{'gross':>9}{'fee':>7}{'spread':>9}{'NET':>9}")
    out = {}
    for k_, v in dw["fade"].items():
        gross = -v["their_h2"]; fee = v["fee"]
        net = gross - fee - cohort_spread
        out[k_] = {"gross": gross, "fee": fee, "spread": cohort_spread, "net": net}
        log(f"  {k_:<24}{gross:>+8.2f}¢{fee:>6.2f}¢{cohort_spread:>8.2f}¢{net:>+8.2f}¢")

    best = max(out.values(), key=lambda x: x["net"])
    log("")
    if best["net"] > 0:
        log(f"  VERDICT: survives — best cell nets {best['net']:+.2f}¢ after fee AND spread.")
    else:
        log(f"  VERDICT: DEAD ON SPREAD — best cell nets {best['net']:+.2f}¢. "
            f"The spread ({cohort_spread:.2f}¢) exceeds what the fee left.")
    log(f"  Break-even spread (bottom decile, bet-equal): "
        f"{-dw['fade']['bet-equal/decile']['their_h2'] - dw['fade']['bet-equal/decile']['fee']:.2f}¢")

    (F / "fade_spread.json").write_text(json.dumps(
        {"n_books": len(B), "cohort_spread_c": cohort_spread,
         "median_spread_c": float(B.spread_c.median()),
         "per_band": per_band, "fade": out,
         "cohort_wallets": len(wallets)}, indent=2))
    log("\nwrote fade_spread.json\nDONE")


if __name__ == "__main__":
    main()
