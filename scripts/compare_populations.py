"""Compare clustering-adjusted skill dispersion BETWEEN two populations, with a
wallet-level bootstrap interval on the difference.

Motivation: `audit_speed_gradient.py` reports, per population, an adjusted ratio
    varR_observed / varR_cluster_null
and an empirical p that it exceeds 1. That answers "is there real skill dispersion
here", but NOT "does population A have more of it than population B" — which is
the whole question a cohort pilot asks. Two ratios printed side by side are not a
comparison without an interval on their difference.

Two confounds this controls for, both of which make raw numbers misleading:
  * DEPTH. rho, and to a lesser degree varR, rise with per-wallet bet depth. A
    cohort with 1,100 bets/wallet will out-score one with 50 at identical true
    skill. `--depth-cap` truncates each wallet to its first N bets so both
    populations are read at matched depth.
  * CLUSTERING. Bets sharing a market share a resolution event, and the amount of
    that differs hugely between cohorts. Dividing by each population's OWN
    cluster-preserving null is what makes the two comparable at all.

Bootstrap design. varR is a pure function of the per-wallet triples
(mean residual, residual variance, n):
    varR = Var_w(mean) / E_w(var / n)      over wallets with n >= MIN_CELL_N
so a wallet bootstrap resamples those triples directly — no frame rebuilding, and
no risk of the self-correlation artifact that makes resampling invalid for a
split-half statistic. The cluster null is held FIXED at its full-sample value
(it is already an average over many permutations, so its own sampling error is
small next to the numerator's); the reported interval is therefore an interval on
the numerator, and is approximate in the denominator. Stated so it is not
mistaken for exact.

READ-ONLY. Writes nothing unless --out is given.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.features import fit_price_baseline, expected_outcome  # noqa: E402
from scripts.audit_speed_gradient import (  # noqa: E402
    _wallet_stats, permute_wallets_within_market, MIN_CELL_N,
)


def _prep(df, depth_cap):
    if depth_cap:
        df = (df.sort_values(["wallet", "timestamp"], kind="stable")
                .groupby("wallet", sort=False, observed=True).head(depth_cap))
    df = df.sort_values(["wallet", "timestamp"], kind="stable")
    codes = pd.factorize(df["wallet"], sort=False)[0]
    w_len = np.bincount(codes)
    w_start = np.concatenate([[0], np.cumsum(w_len)[:-1]])
    price = df["entry_price"].to_numpy(float)
    resid = df["resolved_value"].to_numpy(float) - expected_outcome(
        fit_price_baseline(df, n_bins=20), price)
    return df, codes, w_len, w_start, resid


def _var_ratio_from_triples(mean, var, n):
    big = n >= MIN_CELL_N
    if big.sum() < 8:
        return np.nan
    obs = float(np.var(mean[big], ddof=1))
    exp = float(np.mean(var[big] / n[big]))
    return obs / exp if exp > 0 else np.nan


def analyse(df, rng, n_null=100, depth_cap=None, n_boot=2000):
    df, codes, w_len, w_start, resid = _prep(df, depth_cap)
    mean, var = _wallet_stats(resid, w_start, w_len)
    observed = _var_ratio_from_triples(mean, var, w_len)

    # cluster-preserving null, full sample
    mcodes = pd.factorize(df["market_id"], sort=False)[0]
    ts = df["timestamp"].to_numpy()
    nulls = []
    for _ in range(n_null):
        wc = permute_wallets_within_market(codes, mcodes, rng)
        o = np.lexsort((ts, wc))
        w2, r2 = wc[o], resid[o]
        bnd = np.flatnonzero(np.diff(w2)) + 1
        s2 = np.concatenate([[0], bnd])
        l2 = np.diff(np.concatenate([s2, [len(w2)]]))
        m2, v2 = _wallet_stats(r2, s2, l2)
        x = _var_ratio_from_triples(m2, v2, l2)
        if np.isfinite(x):
            nulls.append(x)
    null_mean = float(np.mean(nulls)) if nulls else np.nan

    # wallet bootstrap on the numerator (resample the triples)
    k = len(w_len)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        boots[i] = _var_ratio_from_triples(mean[pick], var[pick], w_len[pick])
    ratios = boots / null_mean
    return {"bets": len(df), "wallets": k, "var_ratio": observed, "cluster_null": null_mean,
            "adjusted_ratio": observed / null_mean if null_mean else np.nan,
            "median_depth": float(np.median(w_len)), "boot": ratios}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--depth-cap", type=int, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(20260723)
    res = {}
    for path, label in ((args.a, args.label_a), (args.b, args.label_b)):
        df = pd.read_parquet(path, columns=["wallet", "market_id", "entry_price",
                                            "resolved_value", "timestamp"])
        r = analyse(df, rng, depth_cap=args.depth_cap, n_boot=args.boot)
        res[label] = r
        lo, hi = np.percentile(r["boot"], [2.5, 97.5])
        print(f"\n=== {label} ===")
        print(f"  bets={r['bets']:,}  wallets={r['wallets']:,}  median_depth={r['median_depth']:.0f}"
              f"  (depth_cap={args.depth_cap})")
        print(f"  varR={r['var_ratio']:.2f}  cluster_null={r['cluster_null']:.2f}")
        print(f"  ADJUSTED RATIO = {r['adjusted_ratio']:.3f}   bootstrap 95% CI [{lo:.3f}, {hi:.3f}]",
              flush=True)

    a, b = res[args.label_a], res[args.label_b]
    diff = b["boot"] - a["boot"]
    dlo, dhi = np.percentile(diff, [2.5, 97.5])
    print(f"\n=== DIFFERENCE ({args.label_b} − {args.label_a}) ===")
    print(f"  point {b['adjusted_ratio'] - a['adjusted_ratio']:+.3f}   "
          f"95% CI [{dlo:+.3f}, {dhi:+.3f}]")
    print(f"  => {'EXCLUDES 0 — a real difference' if (dlo > 0 or dhi < 0) else 'INCLUDES 0 — not distinguishable'}")
    print(f"  P({args.label_b} > {args.label_a}) = {(diff > 0).mean():.3f}")

    if args.out:
        pd.DataFrame([{k: v for k, v in r.items() if k != "boot"} | {"population": lbl}
                      for lbl, r in res.items()]).to_parquet(args.out)
        print(f"[out] {args.out}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
