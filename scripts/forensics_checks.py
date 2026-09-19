"""Three adversarial checks on the forensics findings — 2026-07-30.

Run after scripts/wallet_forensics.py. Each check exists because a first-pass
number looked interesting and the obvious way for it to be fake had not been ruled
out. Read-only; writes data/interim/forensics/checks.json.

  CHECK 1 — BASELINE RESOLUTION. features.fit_price_baseline uses 20 quantile bins.
    Entry prices here are near-continuous (467k distinct values over 3.4M bets), so
    within-bin price slope can manufacture per-wallet residual for wallets that sit
    systematically high or low inside their bins. Rescore every wallet at 20 / 200 /
    1000 bins and report how much of the top cohort's edge is bin-resolution.
    NOTE an exact-price baseline is NOT the control: ~7 bets per distinct price
    means it absorbs genuine skill along with the artifact.

  CHECK 2 — PERSISTENCE WITH DISJOINT EVENTS. Chronological split-half persistence
    reads spearman ~0.26 over 1,371 wallets. A wallet whose bets on ONE event
    straddle the split contributes the same outcome to both halves, which
    manufactures correlation from nothing — the exact trap the streak analysis fell
    into (65-80% same-event leakage). Recompute with every event that appears in
    both halves dropped from both.

  CHECK 3 — CLUSTER-PRESERVING NULL, DONE PROPERLY. The first null permuted wallet
    labels across (wallet, event) cells freely. That destroys each wallet's
    bets-per-event structure, which ranges 1.05 to 713 across this population, and
    inflates the null's variance — it returned real dispersion NARROWER than null
    (P=1.000), which is a broken null, not a result. Fix: permute labels WITHIN
    strata of cell size, so each wallet keeps its own multiset of event weights.
    Applied to both dispersion and the split-half persistence statistic.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sports_events import resolution_event

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "data" / "interim" / "realworld"
OUT = ROOT / "data" / "interim" / "forensics"
SEED = 20260730
N_NULL = 300
MIN_BETS, MIN_EVENTS, MIN_HALF_EVENTS = 100, 30, 20


def log(m): print(m, flush=True)


def load():
    cat = pd.read_parquet(RW / "market_category.parquet")
    cat_of = dict(zip(cat["market_id"], cat["category"]))
    wcode, mcode, slug_by = {}, {}, {}
    parts = []
    for p in sorted(glob.glob(str(RW / "deep_trades" / "*.parquet"))):
        d = pd.read_parquet(p, columns=["wallet", "market_id", "side", "entry_price",
                                        "timestamp", "resolved", "resolved_value", "slug"])
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
        parts.append(pd.DataFrame({"w": w, "m": m,
                                   "price": d["entry_price"].to_numpy(np.float32),
                                   "ts": d["timestamp"].to_numpy(np.int64),
                                   "rv": d["resolved_value"].to_numpy(np.float32)}))
    b = pd.concat(parts, ignore_index=True)
    del parts
    markets = np.empty(len(mcode), object)
    for s, c in mcode.items():
        markets[c] = s
    ev_key = {int(c): resolution_event(slug_by.get(int(c)), market_id=markets[c])[0]
              for c in b["m"].unique()}
    ecode: dict[str, int] = {}
    b["e"] = np.fromiter((ecode.setdefault(ev_key[int(c)], len(ecode))
                          for c in b["m"].to_numpy()), np.int32, len(b))
    log(f"  bets {len(b):,} | wallets {len(wcode):,} | events {len(ecode):,}")
    wallets = np.empty(len(wcode), object)
    for s, c in wcode.items():
        wallets[c] = s
    return b, wallets


def curve(price, rv, nb):
    e = np.unique(np.quantile(price, np.linspace(0, 1, nb + 1)))
    i = np.clip(np.searchsorted(e, price, side="right") - 1, 0, e.size - 2)
    s = np.bincount(i, weights=rv, minlength=e.size - 1)
    c = np.bincount(i, minlength=e.size - 1)
    return np.where(c > 0, s / np.maximum(c, 1), rv.mean())[i]


def event_wtd(w, e, r, nW):
    """Event-weighted mean residual per wallet: mean over the wallet's events of
    the within-event mean. Returns (mean, n_events)."""
    key = w.astype(np.int64) * (e.max() + 1) + e
    _, cell = np.unique(key, return_inverse=True)
    nC = cell.max() + 1
    cs = np.bincount(cell, weights=r, minlength=nC)
    cn = np.bincount(cell, minlength=nC).astype(float)
    cm = cs / cn
    cw = np.zeros(nC, np.int64)
    cw[cell] = w
    n = np.bincount(cw, minlength=nW).astype(float)
    s = np.bincount(cw, weights=cm, minlength=nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, n, cw, cm, cn


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    log("[load]")
    b, wallet_addr = load()
    nW = len(wallet_addr)
    price = b["price"].to_numpy(float); rv = b["rv"].to_numpy(float)
    w = b["w"].to_numpy(); e = b["e"].to_numpy(); ts = b["ts"].to_numpy()

    # ---------------------------------------------------------------- CHECK 1
    log("\n[CHECK 1] baseline resolution")
    scores = {}
    for nb in (20, 200, 1000):
        r = 100.0 * (rv - curve(price, rv, nb))
        m, n, *_ = event_wtd(w, e, r, nW)
        scores[nb] = pd.DataFrame({"m": m, "n": n})
    nb_ct = pd.Series(np.bincount(w, minlength=nW))
    ok = (nb_ct >= MIN_BETS).to_numpy() & (scores[20]["n"] >= MIN_EVENTS).to_numpy()
    log(f"  wallets scored: {ok.sum():,}")
    top = scores[20]["m"].where(ok).nlargest(100).index
    row = {}
    for nb in (20, 200, 1000):
        row[nb] = float(scores[nb]["m"].reindex(top).mean())
    log(f"  top-100 by the 20-bin metric, rescored:")
    for nb in (20, 200, 1000):
        log(f"    {nb:>5} bins  mean edge {row[nb]:+.3f}c   "
            f"({100*row[nb]/row[20]:.1f}% of the 20-bin figure)")
    ov = {nb: len(set(top) & set(scores[nb]["m"].where(ok).nlargest(100).index))
          for nb in (200, 1000)}
    log(f"  top-100 membership overlap vs 20-bin: 200 bins {ov[200]}/100, "
        f"1000 bins {ov[1000]}/100")
    pr = pd.DataFrame({"p": price}).groupby(w)["p"].mean()
    for nb in (20, 200, 1000):
        c = float(pd.Series(scores[nb]["m"]).where(ok).corr(pr.where(ok)))
        log(f"  corr(wallet mean price, residual) at {nb:>5} bins: {c:+.3f}")
        out[f"corr_price_resid_{nb}"] = c
    out["top100_edge_by_bins"] = {str(k): v for k, v in row.items()}
    out["top100_overlap"] = {str(k): v for k, v in ov.items()}

    # ---------------------------------------------------------------- CHECK 2
    log("\n[CHECK 2] split-half persistence with disjoint events")
    r = 100.0 * (rv - curve(price, rv, 200))          # finer baseline, post-check-1
    o = np.lexsort((ts, w))
    ws, es, rs = w[o], e[o], r[o]
    bnd = np.r_[0, np.where(np.diff(ws) != 0)[0] + 1, len(ws)]
    rows = []
    for a, z in zip(bnd[:-1], bnd[1:]):
        wi = int(ws[a]); n = z - a
        k = int(np.ceil(n / 2))
        e1, e2 = es[a:a + k], es[a + k:z]
        r1, r2 = rs[a:a + k], rs[a + k:z]
        shared = np.intersect1d(np.unique(e1), np.unique(e2))
        def ew(ev, rr):
            if not len(ev): return np.nan, 0
            u, inv = np.unique(ev, return_inverse=True)
            return float((np.bincount(inv, weights=rr) / np.bincount(inv)).mean()), len(u)
        m1, n1 = ew(e1, r1); m2, n2 = ew(e2, r2)
        k1 = ~np.isin(e1, shared); k2 = ~np.isin(e2, shared)
        d1, dn1 = ew(e1[k1], r1[k1]); d2, dn2 = ew(e2[k2], r2[k2])
        rows.append((wi, n, m1, m2, n1, n2, d1, d2, dn1, dn2, len(shared)))
    P = pd.DataFrame(rows, columns=["w", "n_bets", "h1", "h2", "n1", "n2",
                                    "d1", "d2", "dn1", "dn2", "n_shared"])
    base = P[(P.n_bets >= MIN_BETS) & (P.n1 >= MIN_HALF_EVENTS) & (P.n2 >= MIN_HALF_EVENTS)]
    dis = base[(base.dn1 >= MIN_HALF_EVENTS) & (base.dn2 >= MIN_HALF_EVENTS)]
    sp_all = float(base.h1.corr(base.h2, method="spearman"))
    sp_dis = float(dis.d1.corr(dis.d2, method="spearman"))
    log(f"  wallets: {len(base):,} | with enough disjoint events: {len(dis):,}")
    log(f"  median events shared across the split: {base.n_shared.median():.0f} "
        f"({100*(base.n_shared/((base.n1+base.n2)/2)).median():.1f}% of a half)")
    log(f"  spearman(h1, h2)  raw            {sp_all:+.3f}")
    log(f"  spearman(h1, h2)  events DISJOINT {sp_dis:+.3f}")
    pos = dis[dis.d1 > 0]
    hit = float((pos.d2 > 0).mean()); chance = float((dis.d2 > 0).mean())
    log(f"  positive in H1 -> positive in H2: {100*hit:.1f}% (base rate {100*chance:.1f}%)")
    out.update(n_persist=len(base), n_persist_disjoint=len(dis),
               spearman_raw=sp_all, spearman_disjoint=sp_dis,
               h1pos_to_h2pos=hit, h2pos_base=chance)

    # ---------------------------------------------------------------- CHECK 3
    log("\n[CHECK 3] cluster-preserving null, stratified by cell size")
    m_obs, n_obs, cw, cm, cn = event_wtd(w, e, r, nW)
    keep = (np.bincount(w, minlength=nW) >= MIN_BETS) & (n_obs >= MIN_EVENTS)
    # strata: cell bet-count buckets, so a wallet keeps its own event-weight profile
    strata = np.digitize(cn, [2, 3, 4, 6, 11, 21, 51])
    real_disp = float(np.nanstd(m_obs[keep]))
    nd = np.empty(N_NULL)
    for i in range(N_NULL):
        perm = cw.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            perm[idx] = cw[rng.permutation(idx)]
        n_ = np.bincount(perm, minlength=nW).astype(float)
        s_ = np.bincount(perm, weights=cm, minlength=nW)
        with np.errstate(invalid="ignore", divide="ignore"):
            nd[i] = np.nanstd((s_ / n_)[keep])
    p_disp = float((nd >= real_disp).mean())
    log(f"  wallets: {int(keep.sum()):,}")
    log(f"  per-wallet edge dispersion: real {real_disp:.3f}c vs null "
        f"{nd.mean():.3f}c [{np.percentile(nd,2.5):.3f}, {np.percentile(nd,97.5):.3f}]"
        f"  P={p_disp:.4f}")
    # null for persistence: shuffle whole wallets' H2 against H1
    dd = dis.dropna(subset=["d1", "d2"])
    npx = np.empty(N_NULL)
    for i in range(N_NULL):
        npx[i] = pd.Series(dd.d1.to_numpy()).corr(
            pd.Series(rng.permutation(dd.d2.to_numpy())), method="spearman")
    p_pers = float((npx >= sp_dis).mean())
    log(f"  persistence null (H2 shuffled across wallets): {npx.mean():+.4f} "
        f"[{np.percentile(npx,2.5):+.3f}, {np.percentile(npx,97.5):+.3f}]  "
        f"real {sp_dis:+.3f}  P={p_pers:.4f}")
    out.update(disp_real=real_disp, disp_null=float(nd.mean()), disp_p=p_disp,
               pers_null=float(npx.mean()), pers_p=p_pers,
               n_null=N_NULL, seed=SEED)
    (OUT / "checks.json").write_text(json.dumps(out, indent=2))
    log("\nDONE")


if __name__ == "__main__":
    main()
