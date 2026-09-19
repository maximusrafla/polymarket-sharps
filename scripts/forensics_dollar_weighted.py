"""Re-weight the load-bearing numbers by money actually staked — 2026-07-30.

WHY. `scripts/wallet_forensics.py` found that wallets bet BIGGER on their WORSE
bets (notional- minus equal-weighted residual: median −0.112¢, 57.4% of wallets
negative, Wilcoxon p=8.8e-5, same sign in all three price terciles). Every headline
in this repo counts each bet once regardless of stake, so every one of them
flatters its subject by an unknown amount. This measures the amount.

THREE WEIGHTINGS, and what each answers:

  bet-equal    each bet counts once. The repo's current convention. Answers
               "was the wallet right more often than the price implied".
  share-wtd    each bet weighted by shares bought. Answers "per share I actually
               held, what did I earn" — the right unit for a cents/share statistic
               and the one a copier sizing like them would experience.
  notional-wtd each bet weighted by dollars staked (shares x price). Answers
               "return on capital deployed".

Event clustering is preserved in all three: the weight applies WITHIN a resolution
event, then events are averaged equally. Weighting across events too would let one
big event dominate, which is the pathology the event unit exists to prevent — that
variant is reported separately and labelled as such.

Halves are the EVENT-DISJOINT ones (events appearing on both sides of the split are
dropped from both), same construction as scripts/forensics_persistence.py.

Read-only. Writes data/interim/forensics/dollar_weighted.json.
Run:  flock data/interim/.analysis.lock -c \
        'PYTHONPATH=. .venv/bin/python scripts/forensics_dollar_weighted.py'
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.forensics_checks import RW, OUT, curve  # noqa: E402
from src.sports_events import resolution_event       # noqa: E402

MIN_BETS, MIN_EVENTS, MIN_HALF = 100, 30, 20
FEE_K = 0.05          # sports/other taker rate; the frozen paper_rw map


def log(m): print(m, flush=True)


def load_with_size():
    """Same universe as forensics_checks.load, carrying `size` as well."""
    cat = pd.read_parquet(RW / "market_category.parquet")
    cat_of = dict(zip(cat["market_id"], cat["category"]))
    wcode, mcode, slug_by = {}, {}, {}
    parts = []
    for p in sorted(glob.glob(str(RW / "deep_trades" / "*.parquet"))):
        d = pd.read_parquet(p, columns=["wallet", "market_id", "side", "entry_price",
                                        "size", "timestamp", "resolved",
                                        "resolved_value", "slug"])
        d = d[(d["side"] == "BUY") & d["resolved"].fillna(False) & d["resolved_value"].notna()]
        if not len(d):
            continue
        keep = np.array([cat_of.get(x, "other") != "micro_crypto" for x in d["market_id"]])
        d = d.loc[keep]
        if not len(d):
            continue
        w = np.fromiter((wcode.setdefault(x, len(wcode)) for x in d["wallet"].to_numpy()),
                        np.int32, len(d))
        m = np.fromiter((mcode.setdefault(x, len(mcode)) for x in d["market_id"].to_numpy()),
                        np.int32, len(d))
        for mc, s in zip(m, d["slug"].to_numpy()):
            if mc not in slug_by and isinstance(s, str):
                slug_by[int(mc)] = s
        parts.append(pd.DataFrame({
            "w": w, "m": m,
            "price": d["entry_price"].to_numpy(np.float32),
            "size": d["size"].to_numpy(np.float32),
            "ts": d["timestamp"].to_numpy(np.int64),
            "rv": d["resolved_value"].to_numpy(np.float32)}))
    b = pd.concat(parts, ignore_index=True)
    del parts
    markets = np.empty(len(mcode), object)
    for s, c in mcode.items():
        markets[c] = s
    ev = {int(c): resolution_event(slug_by.get(int(c)), market_id=markets[c])[0]
          for c in b["m"].unique()}
    ecode: dict[str, int] = {}
    b["e"] = np.fromiter((ecode.setdefault(ev[int(c)], len(ecode))
                          for c in b["m"].to_numpy()), np.int32, len(b))
    wallets = np.empty(len(wcode), object)
    for s, c in wcode.items():
        wallets[c] = s
    log(f"  bets {len(b):,} | wallets {len(wcode):,} | events {len(ecode):,}")
    return b, wallets


def half_stat(ev, r, wt):
    """Event-weighted mean of the WITHIN-event weighted mean. Returns (mean, n_events)."""
    if not len(ev):
        return np.nan, 0
    u, inv = np.unique(ev, return_inverse=True)
    num = np.bincount(inv, weights=r * wt)
    den = np.bincount(inv, weights=wt)
    ok = den > 0
    if not ok.any():
        return np.nan, 0
    return float((num[ok] / den[ok]).mean()), int(ok.sum())


def main() -> None:
    log("[load]")
    b, wallets = load_with_size()
    price = b["price"].to_numpy(float); rv = b["rv"].to_numpy(float)
    w = b["w"].to_numpy(); e = b["e"].to_numpy(); ts = b["ts"].to_numpy()
    size = np.nan_to_num(b["size"].to_numpy(float), nan=0.0)
    r = 100.0 * (rv - curve(price, rv, 200))
    notional = size * price

    WEIGHTS = {"bet-equal": np.ones_like(size),
               "share-wtd": size,
               "notional-wtd": notional}

    o = np.lexsort((ts, w))
    ws, es, rs, ps = w[o], e[o], r[o], price[o]
    WS = {k: v[o] for k, v in WEIGHTS.items()}
    bnd = np.r_[0, np.where(np.diff(ws) != 0)[0] + 1, len(ws)]

    rows = []
    for a, z in zip(bnd[:-1], bnd[1:]):
        n = z - a; k = int(np.ceil(n / 2))
        e1, e2 = es[a:a + k], es[a + k:z]
        shared = np.intersect1d(np.unique(e1), np.unique(e2))
        k1 = ~np.isin(e1, shared); k2 = ~np.isin(e2, shared)
        rec = {"wallet": wallets[int(ws[a])], "n_bets": n,
               "mean_price": float(ps[a:z].mean()),
               "n_events": len(np.unique(es[a:z]))}
        for name, wv in WS.items():
            w1, w2 = wv[a:a + k], wv[a + k:z]
            m1, n1 = half_stat(e1[k1], rs[a:a + k][k1], w1[k1])
            m2, n2 = half_stat(e2[k2], rs[a + k:z][k2], w2[k2])
            rec[f"h1_{name}"] = m1; rec[f"h2_{name}"] = m2
            rec[f"n1_{name}"] = n1; rec[f"n2_{name}"] = n2
        rows.append(rec)
    P = pd.DataFrame(rows)
    P.to_parquet(OUT / "dollar_weighted_halves.parquet", index=False)

    out = {}
    log("\n" + "=" * 78)
    log("SPLIT-HALF PERSISTENCE AND THE QUINTILE TABLE, UNDER EACH WEIGHTING")
    log("=" * 78)
    for name in WEIGHTS:
        k = P[(P.n_bets >= MIN_BETS) & (P.n_events >= MIN_EVENTS)
              & (P[f"n1_{name}"] >= MIN_HALF) & (P[f"n2_{name}"] >= MIN_HALF)].dropna(
                  subset=[f"h1_{name}", f"h2_{name}"]).copy()
        rho = float(k[f"h1_{name}"].corr(k[f"h2_{name}"], method="spearman"))
        k["q"] = pd.qcut(k[f"h1_{name}"], 5, labels=False)
        g = k.groupby("q").agg(n=("wallet", "size"), h1=(f"h1_{name}", "mean"),
                               h2=(f"h2_{name}", "mean"), price=("mean_price", "mean"))
        lo = k[k.q == 0][f"h2_{name}"]; hi = k[k.q == 4][f"h2_{name}"]
        tl, pl = stats.ttest_1samp(lo, 0); th, ph = stats.ttest_1samp(hi, 0)
        log(f"\n--- {name}  (n={len(k):,}, spearman H1~H2 = {rho:+.3f}) ---")
        log(f"  {'quintile':<10}{'n':>5}{'H1':>9}{'H2':>9}{'price':>8}")
        for q in range(5):
            log(f"  {['worst','2','3','4','best'][q]:<10}{int(g.n[q]):>5}"
                f"{g.h1[q]:>+8.2f}¢{g.h2[q]:>+8.2f}¢{g.price[q]:>8.2f}")
        log(f"  worst quintile H2 {lo.mean():+.2f}¢  t={tl:.2f} p={pl:.2g}")
        log(f"  best  quintile H2 {hi.mean():+.2f}¢  t={th:.2f} p={ph:.2g}")
        out[name] = {"n": len(k), "spearman": rho,
                     "q_h1": [float(x) for x in g.h1], "q_h2": [float(x) for x in g.h2],
                     "worst_h2": float(lo.mean()), "worst_t": float(tl), "worst_p": float(pl),
                     "best_h2": float(hi.mean()), "best_t": float(th), "best_p": float(ph)}

    log("\n" + "=" * 78)
    log("THE FADE, UNDER EACH WEIGHTING (fader buys the complement; same fee)")
    log("=" * 78)
    log(f"  {'weighting':<14}{'cut':<10}{'n':>5}{'their H2':>11}{'gross':>9}{'fee':>7}{'NET':>9}")
    out["fade"] = {}
    for name in WEIGHTS:
        k = P[(P.n_bets >= MIN_BETS) & (P.n_events >= MIN_EVENTS)
              & (P[f"n1_{name}"] >= MIN_HALF) & (P[f"n2_{name}"] >= MIN_HALF)].dropna(
                  subset=[f"h1_{name}", f"h2_{name}"]).copy()
        for lab, frac in [("quintile", 0.2), ("decile", 0.1)]:
            s = k[k[f"h1_{name}"] <= k[f"h1_{name}"].quantile(frac)]
            their = float(s[f"h2_{name}"].mean())
            fee = float(100 * FEE_K * (s.mean_price * (1 - s.mean_price)).mean())
            log(f"  {name:<14}{lab:<10}{len(s):>5}{their:>+10.2f}¢{-their:>+8.2f}¢"
                f"{fee:>6.2f}¢{-their-fee:>+8.2f}¢")
            out["fade"][f"{name}/{lab}"] = {"n": len(s), "their_h2": their,
                                            "fee": fee, "net": -their - fee}

    (OUT / "dollar_weighted.json").write_text(json.dumps(out, indent=2))
    log("\nwrote dollar_weighted.json")
    log("DONE")


if __name__ == "__main__":
    main()
