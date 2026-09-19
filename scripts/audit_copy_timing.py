"""The clean timing experiment: does the wallet's MOMENT carry information?

THE PROBLEM THIS SOLVES
-----------------------
`scripts/audit_copy_null.py` compared copying (entry + 2 min) against three
wallet-agnostic anchors and found copying ahead of two of them. But all three sit
on the same monotone gradient — a market's price converges on its answer, so
**earlier is better, everywhere, for everyone**:

    pre-entry (-2m)                 +4.27c   earlier than entry  -> biased UP
    random moment in whole life     +1.40c   mostly earlier      -> biased UP
    real (+2m)                      +1.98c
    random moment AFTER the signal  +0.31c   always later        -> biased DOWN

So `real - random_POST = +1.67c` may be measuring nothing but "entry + 2 min is
earlier than a random later time." The comparison brackets the answer; it does not
isolate it.

THE CONTROL
-----------
Hold the position on the gradient FIXED and vary only whether the moment is the
one the wallet chose *for this market*.

For bet i in token T, let `u_i` be where the follower's fill sits in that market's
usable life:

    u_i = (t_i + delta - start_i) / (cutoff_i - start_i)

The DONOR anchor enters bet i's own token at relative position `u_j`, where j is
another bet **by the same wallet**. `u_j` is drawn from that wallet's own
distribution of relative entry positions, so the anchor lands at a statistically
identical point on the convergence curve — same market, same side, same wallet,
same typical earliness — but at a moment chosen for a *different* market.

  * If these wallets simply enter at a good STAGE of a market's life, real and
    donor agree and the +1.67c was the gradient.
  * If the wallet's moment carries information about THIS market, real wins.

K donors are drawn per bet and averaged, so the donor leg is a Monte-Carlo
estimate of E_u[edge at that wallet's typical relative position] rather than one
noisy draw.

Costs are identical on every leg (the same per-bet spread proxy, the fee at that
leg's own fill price), so nothing but the anchor moves.

READ-ONLY. No network. Writes only data/interim/copysim/redteam/.

    PYTHONPATH=. .venv/bin/python scripts/audit_copy_timing.py \
        --store <store>.npz --baseline <baseline>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, ".")

from src.common import INTERIM_DIR  # noqa: E402
from scripts.audit_copy_null import (  # noqa: E402
    Baseline, build_store, cluster_boot, first_price_in, offsets_for,
)

OUT_DIR = INTERIM_DIR / "copysim" / "redteam"
PER_BET = INTERIM_DIR / "copysim" / "per_bet.parquet"

SEED = 20260729
DELTA = 120
BANDWIDTH = 120
MIN_BETS = 150
N_DONORS = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--donors", type=int, default=N_DONORS)
    ap.add_argument("--wide", type=int, default=900,
                    help="sensitivity: a wider donor fill window (s)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    z = np.load(args.baseline)
    bl = Baseline(z["edges"], z["means"], float(z["global_mean"]))

    pb = pd.read_parquet(PER_BET)
    pb["w"] = pb["wallet"].astype(str)
    tok = pb["tok_ix"].to_numpy(np.int32)
    store = build_store(np.unique(tok), args.store)
    offs = offsets_for(store[0], int(tok.max()) + 1)
    ts_s, p_s = store[1], store[2]
    print(f"[timing] {len(pb):,} bets / {pb['w'].nunique()} wallets; "
          f"{store[0].size:,} price points", flush=True)

    ep = pb["entry_price"].to_numpy(float)
    mid = pb["mid_at_entry"].to_numpy(float)
    p2 = pb["price_2m"].to_numpy(float)
    cut = pb["cutoff"].to_numpy(float)
    start = pb["start_ts"].to_numpy(float)
    entry = pb["timestamp"].to_numpy(float)
    rv = pb["resolved_value"].to_numpy(float)
    k = pb["fee_k"].to_numpy(float)
    ex = pb["fee_exponent"].fillna(1.0).to_numpy(float)
    tick = pb["tick_size"].fillna(0.001).to_numpy(float)
    spread = np.maximum(ep - mid, tick)
    codes = pd.factorize(pb["market_id"].astype(str))[0]

    # where the follower's fill sits in the market's usable life
    span = cut - start
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (entry + DELTA - start) / span
    u = np.where(np.isfinite(u) & (span > 0), np.clip(u, 0.0, 1.0), np.nan)
    pb["u"] = u

    def score(px, i):
        fill = np.clip(px + spread[i], 1e-6, 1.0 - 1e-9)
        fee = k[i] * np.power(fill * (1.0 - fill), ex[i])
        return rv[i] - bl(fill) - fee

    base_ok = np.isfinite(p2) & np.isfinite(mid) & np.isfinite(u)
    print(f"[timing] bets with a real fill AND a defined position: "
          f"{int(base_ok.sum()):,}", flush=True)

    # ---- the gradient, made visible -------------------------------------
    print("\n=== 0. THE GRADIENT (why the earlier anchors were confounded) ===")
    print("  follower edge by where the fill sits in the market's usable life")
    idx = np.where(base_ok)[0]
    band = pd.cut(u[idx], np.linspace(0, 1, 11))
    ge = score(p2[idx], idx)
    grad = pd.DataFrame({"band": band, "edge_c": 100 * ge}).groupby(
        "band", observed=True)["edge_c"].agg(["size", "mean"])
    print(grad.to_string(float_format=lambda x: f"{x:9.3f}"))

    # ---- donor anchors ---------------------------------------------------
    donor_edges = np.full((args.donors, len(pb)), np.nan)
    donor_edges_wide = np.full((args.donors, len(pb)), np.nan)
    donor_u = np.full((args.donors, len(pb)), np.nan)
    for d in range(args.donors):
        tgt = np.full(len(pb), np.nan)
        uu = np.full(len(pb), np.nan)
        for w, g in pb[base_ok].groupby("w"):
            i = g.index.to_numpy()
            if i.size < 2:
                continue
            # draw a donor j != i from the same wallet, then re-use ITS relative
            # position on bet i's own market clock
            j = rng.integers(0, i.size, size=i.size)
            clash = j == np.arange(i.size)
            j[clash] = (j[clash] + 1) % i.size
            uj = u[i[j]]
            uu[i] = uj
            tgt[i] = start[i] + uj * span[i]
        ok = np.isfinite(tgt)
        px = np.full(len(pb), np.nan)
        pxw = np.full(len(pb), np.nan)
        ii = np.where(ok)[0]
        px[ii] = first_price_in(ts_s, p_s, offs, tok[ii], tgt[ii], BANDWIDTH, cut[ii])
        pxw[ii] = first_price_in(ts_s, p_s, offs, tok[ii], tgt[ii], args.wide, cut[ii])
        donor_edges[d, ii] = score(px[ii], ii)
        donor_edges_wide[d, ii] = score(pxw[ii], ii)
        donor_u[d] = uu
        print(f"[timing] donor {d+1}/{args.donors}: "
              f"{int(np.isfinite(donor_edges[d]).sum()):,} filled "
              f"({int(np.isfinite(donor_edges_wide[d]).sum()):,} at the wide window)",
              flush=True)

    donor = np.nanmean(donor_edges, axis=0)
    donor_w = np.nanmean(donor_edges_wide, axis=0)
    du = np.nanmean(donor_u, axis=0)
    real = np.full(len(pb), np.nan)
    real[base_ok] = score(p2[base_ok], np.where(base_ok)[0])

    for label, dn in (("narrow (120 s, symmetric with the real leg)", donor),
                      (f"wide ({args.wide} s)", donor_w)):
        pair = np.isfinite(real) & np.isfinite(dn)
        n = int(pair.sum())
        print(f"\n=== 1. PAIRED vs the DONOR anchor — {label} ===")
        print(f"  paired bets {n:,} of {int(base_ok.sum()):,}  "
              f"({100*n/max(int(base_ok.sum()),1):.1f}% coverage)")
        print(f"  mean relative position: real {np.nanmean(u[pair]):.3f}   "
              f"donor {np.nanmean(du[pair]):.3f}   "
              f"(matched by construction — this is the check)")
        mr, lr, hr, _ = cluster_boot(real[pair], codes[pair], rng)
        md, ld, hd, _ = cluster_boot(dn[pair], codes[pair], rng)
        mdiff, ldiff, hdiff, p = cluster_boot(real[pair] - dn[pair], codes[pair], rng)
        print(f"  real  {100*mr:+.4f}c [{100*lr:+.3f}, {100*hr:+.3f}]")
        print(f"  donor {100*md:+.4f}c [{100*ld:+.3f}, {100*hd:+.3f}]")
        print(f"  REAL - DONOR  {100*mdiff:+.4f}c  95% CI "
              f"[{100*ldiff:+.4f}, {100*hdiff:+.4f}]   one-sided p={p:.4f}")
        if label.startswith("narrow"):
            json.dump({"n_paired": n, "real_c": 100 * mr, "donor_c": 100 * md,
                       "diff_c": 100 * mdiff, "lo": 100 * ldiff, "hi": 100 * hdiff,
                       "p": p, "mean_u_real": float(np.nanmean(u[pair])),
                       "mean_u_donor": float(np.nanmean(du[pair]))},
                      open(OUT_DIR / "timing_donor.json", "w"), indent=2)
            rows = []
            for w, g in pb[pair].groupby("w"):
                if len(g) < MIN_BETS:
                    continue
                i = g.index.to_numpy()
                m, lo, hi, pp = cluster_boot(real[i] - dn[i], codes[i], rng)
                rows.append(dict(wallet=w[:14], n=len(g),
                                 real_c=100 * np.nanmean(real[i]),
                                 donor_c=100 * np.nanmean(dn[i]),
                                 diff_c=100 * m, lo=100 * lo, hi=100 * hi, p=pp))
            pw = pd.DataFrame(rows).sort_values("diff_c", ascending=False)
            print("\n--- per wallet ---")
            print(pw.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
            print(f"  CI above zero: {int((pw['lo']>0).sum())} of {len(pw)}   "
                  f"CI below zero: {int((pw['hi']<0).sum())}")
            pw.to_parquet(OUT_DIR / "timing_per_wallet.parquet", index=False)

    print(f"\n[timing] artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
