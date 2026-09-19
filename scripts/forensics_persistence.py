"""Persist the per-wallet split-half table with EVENT-DISJOINT halves.

scripts/forensics_checks.py computes this internally to report one correlation;
the dashboard needs it per wallet, because the quintile panel and every number
derived from it must use the disjoint halves rather than the raw ones. A wallet
whose bets on one event straddle the split otherwise contributes the same outcome
to both halves and manufactures agreement out of nothing.

Same construction as CHECK 2 in forensics_checks.py (200-bin baseline, event-
weighted halves, shared events removed from BOTH sides). Read-only; writes
data/interim/forensics/persistence.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.forensics_checks import RW, OUT, curve, load  # noqa: E402


def main() -> None:
    print("[load]", flush=True)
    b, addr = load()
    price = b["price"].to_numpy(float)
    rv = b["rv"].to_numpy(float)
    w = b["w"].to_numpy(); e = b["e"].to_numpy(); ts = b["ts"].to_numpy()
    r = 100.0 * (rv - curve(price, rv, 200))

    o = np.lexsort((ts, w))
    ws, es, rs = w[o], e[o], r[o]
    bnd = np.r_[0, np.where(np.diff(ws) != 0)[0] + 1, len(ws)]

    def ew(ev, rr):
        if not len(ev):
            return np.nan, 0
        u, inv = np.unique(ev, return_inverse=True)
        return float((np.bincount(inv, weights=rr) / np.bincount(inv)).mean()), len(u)

    rows = []
    for a, z in zip(bnd[:-1], bnd[1:]):
        wi = int(ws[a]); n = z - a; k = int(np.ceil(n / 2))
        e1, e2 = es[a:a + k], es[a + k:z]
        r1, r2 = rs[a:a + k], rs[a + k:z]
        shared = np.intersect1d(np.unique(e1), np.unique(e2))
        k1 = ~np.isin(e1, shared); k2 = ~np.isin(e2, shared)
        d1, dn1 = ew(e1[k1], r1[k1])
        d2, dn2 = ew(e2[k2], r2[k2])
        m1, n1 = ew(e1, r1); m2, n2 = ew(e2, r2)
        rows.append((addr[wi], n, m1, m2, n1, n2, d1, d2, dn1, dn2, len(shared)))

    P = pd.DataFrame(rows, columns=["wallet", "n_bets", "h1_raw", "h2_raw", "n1_raw",
                                    "n2_raw", "h1_disjoint", "h2_disjoint",
                                    "n1_disjoint", "n2_disjoint", "n_shared_events"])
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_parquet(OUT / "persistence.parquet", index=False)
    ok = P[(P.n_bets >= 100) & (P.n1_disjoint >= 20) & (P.n2_disjoint >= 20)]
    print(f"  wrote persistence.parquet — {len(P):,} wallets, {len(ok):,} scorable")
    print(f"  spearman raw {P[(P.n_bets>=100)&(P.n1_raw>=20)&(P.n2_raw>=20)].h1_raw.corr(P[(P.n_bets>=100)&(P.n1_raw>=20)&(P.n2_raw>=20)].h2_raw, method='spearman'):+.3f}"
          f" | disjoint {ok.h1_disjoint.corr(ok.h2_disjoint, method='spearman'):+.3f}")
    q = ok.assign(q=pd.qcut(ok.h1_disjoint, 5, labels=False)).groupby("q").agg(
        n=("wallet", "size"), h1=("h1_disjoint", "mean"), h2=("h2_disjoint", "mean"))
    print(q.round(3).to_string())
    print("DONE")


if __name__ == "__main__":
    main()
