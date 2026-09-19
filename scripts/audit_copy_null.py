"""RED TEAM 2026-07-29: the null the copy analysis never had.

WHY THIS EXISTS
---------------
`docs/copy_verdict.md` CORRECTION 2 reports that 11 of 37 certified real-world
wallets are "copyable" at a 2-minute lag. That number rests on a **baseline**
(`E[outcome | price]` fitted over the deep real-world tape) and on **per-wallet
market-cluster bootstrap CIs**. It rests on **no null at all** — nothing in the
pipeline asks how many wallets would look copyable if the wallet identity, or the
wallet's chosen moment, carried no information. The repo's own history says that
is exactly where copy theses die: every earlier one was killed by a
wallet-agnostic placebo anchor (`scripts/audit_edge_decay_long.py`), not by a
cost assumption.

This script supplies the three missing controls, on the SAME per-bet frame the
published number came from (`data/interim/copysim/per_bet.parquet`), scored with
the SAME rule (verified to reproduce `own_c` to 4 d.p. and the copyable set
exactly):

  fill      = price(entry + 120s) + max(entry_price - mid_at_entry, tick)
  net edge  = resolved_value - E[outcome | fill] - fee(fill)

CONTROLS
  1. RANDOM ANCHOR  — same bet, same token, same side, same costs, but entered at
     a uniformly random grid point in the token's observed pre-guard life. Tests
     whether the wallet's *moment* is worth anything. ("buy this market whenever")
  2. PRE-ENTRY ANCHOR — entered 120 s BEFORE the wallet, so it cannot be copying.
  3. WALLET-LABEL PERMUTATION — bet -> wallet labels permuted within entry-price
     bins, preserving every wallet's bet count and price mix, destroying only the
     wallet<->bet link. Counts how many "copyable" wallets the procedure invents
     on data with no wallet signal in it. This is the direct calibration of
     "11 of 37".

Plus two robustness axes the published run fixed rather than swept:
  * SPREAD PROXY swept (tick only / half the observed gap / observed gap / 1.5x).
  * BASELINE refit EXCLUDING the 46 simulated wallets, to size the circularity.

READ-ONLY. No network. Writes only under data/interim/copysim/redteam/.

Reproduce::

    PYTHONPATH=. .venv/bin/python scripts/audit_copy_null.py --store <store.npz>

`--store` is an optional cache of the price points restricted to these tokens
(built on first run if absent).
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")

from src.common import INTERIM_DIR  # noqa: E402

COPYSIM = INTERIM_DIR / "copysim"
OUT_DIR = COPYSIM / "redteam"
PER_BET = COPYSIM / "per_bet.parquet"
PRICES = COPYSIM / "prices"

SEED = 20260729
N_BOOT = 2000
BOOT_BLOCK = 250
N_PERM = 200
DELTA = 120          # the published lag
BANDWIDTH = 120
MIN_BETS = 150       # the published per-wallet floor


# ---------------------------------------------------------------------------
# price store
# ---------------------------------------------------------------------------
def build_store(tok_needed: np.ndarray, cache: Path | None) -> tuple:
    if cache is not None and cache.exists():
        z = np.load(cache)
        return z["ix"], z["ts"], z["ps"]
    ix, ts, ps = [], [], []
    for f in sorted(glob.glob(str(PRICES / "chunk-*.parquet"))):
        t = pq.read_table(f, columns=["tok_ix", "t", "p"])
        a = t.column("tok_ix").to_numpy().astype(np.int32)
        m = np.isin(a, tok_needed)
        if m.any():
            ix.append(a[m])
            ts.append(t.column("t").to_numpy().astype(np.int64)[m])
            ps.append(t.column("p").to_numpy().astype(np.float32)[m])
        del t, a, m
        gc.collect()
    ix = np.concatenate(ix); ts = np.concatenate(ts); ps = np.concatenate(ps)
    order = np.lexsort((ts, ix))
    ix, ts, ps = ix[order], ts[order], ps[order]
    if cache is not None:
        np.savez(cache, ix=ix, ts=ts, ps=ps)
    return ix, ts, ps


def offsets_for(ix: np.ndarray, n_tokens: int) -> np.ndarray:
    counts = np.bincount(ix, minlength=n_tokens)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def first_price_in(ts_s, p_s, offs, tok, target, bandwidth, cutoff):
    """Vectorised `copy_sim.price_at`: first grid point in [target, target+bw]
    that is also at or before the guard cutoff. NaN if none."""
    out = np.full(tok.size, np.nan)
    for i in range(tok.size):
        a, b = offs[tok[i]], offs[tok[i] + 1]
        if b <= a or not np.isfinite(cutoff[i]):
            continue
        seg = ts_s[a:b]
        lo = int(np.searchsorted(seg, target[i], side="left"))
        if lo >= seg.size:
            continue
        t = float(seg[lo])
        if t > target[i] + bandwidth or t > cutoff[i]:
            continue
        out[i] = p_s[a + lo]
    return out


def random_anchor_price(ts_s, p_s, offs, tok, cutoff, rng, floor=None):
    """Price at a uniformly random grid point in the token's observed life, at or
    before the guard cutoff — a wallet-agnostic 'buy this market whenever'.

    ``floor`` restricts the draw to grid points at or after that timestamp. With
    ``floor = entry + Δ`` the anchor becomes EXECUTABLE by a copier: it only ever
    looks at moments after the wallet's fill was visible. Without it the draw is
    mostly EARLIER than the wallet's entry (these wallets bet late), and earlier
    prices are systematically better because a market's price converges on its
    answer — so the unfloored anchor flatters the placebo."""
    out = np.full(tok.size, np.nan)
    for i in range(tok.size):
        a, b = offs[tok[i]], offs[tok[i] + 1]
        if b <= a or not np.isfinite(cutoff[i]):
            continue
        seg = ts_s[a:b]
        hi = int(np.searchsorted(seg, cutoff[i], side="right"))
        lo = 0 if floor is None else int(np.searchsorted(seg, floor[i], side="left"))
        if hi <= lo:
            continue
        out[i] = p_s[a + int(rng.integers(lo, hi))]
    return out


def certification_splits(wallets: list[str]) -> pd.DataFrame:
    """Each certified wallet's chronological in-sample/held-out boundary, as
    `src.validate` drew it: real-world resolved BUY bets, deduped, stable-sorted
    by timestamp, cut at `in_sample_n`. Verified against validate's stored
    `in_sample_n + out_of_sample_n` for every wallet."""
    v = pd.read_parquet(INTERIM_DIR / "realworld" / "validated.parquet")
    cert = v[v["edge_persisted"].fillna(False) & (v["prior_arm"] == "")]
    cert = cert[cert["wallet"].isin(wallets)]
    mc = pq.read_table(INTERIM_DIR / "realworld" / "market_category.parquet",
                       columns=["market_id", "category"]).to_pandas()
    micro = set(mc.loc[mc["category"] == "micro_crypto", "market_id"])
    del mc
    parts = []
    for f in sorted(glob.glob(str(INTERIM_DIR / "realworld/deep_trades/part-*.parquet"))):
        d = pq.read_table(f, columns=["wallet", "market_id", "token_id", "side",
                                      "tx_hash", "timestamp"],
                          filters=[("side", "==", "BUY"), ("resolved", "==", True),
                                   ("wallet", "in", list(cert["wallet"]))]).to_pandas()
        if len(d):
            parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t = t.drop_duplicates(subset=["tx_hash", "wallet", "token_id", "side"], keep="last")
    t = t[~t["market_id"].isin(micro)]
    rows = []
    for w, g in t.groupby("wallet"):
        ts = np.sort(g["timestamp"].to_numpy(), kind="mergesort")
        r = cert.loc[cert["wallet"] == w].iloc[0]
        n_in, n_out = int(r["in_sample_n"]), int(r["out_of_sample_n"])
        rows.append(dict(wallet=w, n_here=len(g), n_in=n_in, n_out=n_out,
                         split_ts=int(ts[n_in]) if 0 < n_in < len(ts) else np.nan,
                         reconstructed=(len(g) == n_in + n_out)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
class Baseline:
    def __init__(self, edges, means, global_mean):
        self.edges, self.means, self.gm = edges, means, float(global_mean)

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if self.edges.size == 0:
            return np.full(p.shape, self.gm)
        idx = np.clip(np.searchsorted(self.edges, p, side="right") - 1,
                      0, self.means.size - 1)
        return self.means[idx]


def fit_baseline(prices, outcomes, n_bins: int = 20) -> Baseline:
    edges = np.unique(np.quantile(prices, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, edges.size - 2)
    gm = float(outcomes.mean())
    means = np.array([outcomes[idx == b].mean() if np.any(idx == b) else gm
                      for b in range(edges.size - 1)])
    return Baseline(edges, means, gm)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def cluster_boot(values, codes, rng, n=N_BOOT):
    """Percentile CI + one-sided p (share of resamples <= 0), resampling whole
    markets. Same estimator as `copy_sim.cluster_boot_ci`, with the p added."""
    ok = np.isfinite(values)
    values, codes = values[ok], codes[ok]
    if values.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    u, inv = np.unique(codes, return_inverse=True)
    g = u.size
    if g < 2:
        return float(values.mean()), np.nan, np.nan, np.nan
    sums = np.bincount(inv, weights=values, minlength=g)
    cnts = np.bincount(inv, minlength=g).astype(float)
    means = np.empty(n)
    for i in range(0, n, BOOT_BLOCK):
        b = min(BOOT_BLOCK, n - i)
        idx = rng.integers(0, g, size=(b, g))
        means[i:i + b] = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(values.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), float((means <= 0).mean()))


def cluster_z(values, codes):
    """Cluster-robust z of the mean (markets are the clusters). Used where a
    2,000-resample bootstrap per wallet per permutation would not finish."""
    ok = np.isfinite(values)
    v, c = values[ok], codes[ok]
    if v.size < 2:
        return np.nan, np.nan
    u, inv = np.unique(c, return_inverse=True)
    g = u.size
    if g < 2:
        return np.nan, np.nan
    n = v.size
    xbar = v.mean()
    sums = np.bincount(inv, weights=v, minlength=g)
    cnts = np.bincount(inv, minlength=g).astype(float)
    resid = sums - cnts * xbar
    var = (resid ** 2).sum() * g / max(g - 1, 1) / (n ** 2)
    se = float(np.sqrt(var))
    return float(xbar), (float(xbar / se) if se > 0 else np.nan)


def bh_reject(pvals, q=0.10):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(p.size, dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    if passed.any():
        k = np.max(np.where(passed)[0])
        out[order[:k + 1]] = True
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=None,
                    help="npz cache of the restricted price store")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="npz with edges/means/global_mean of the published baseline")
    ap.add_argument("--perms", type=int, default=N_PERM)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pb = pd.read_parquet(PER_BET)
    pb["w"] = pb["wallet"].astype(str)
    print(f"[null] per-bet frame: {len(pb):,} bets / {pb['w'].nunique()} wallets", flush=True)

    tok = pb["tok_ix"].to_numpy(np.int32)
    store = build_store(np.unique(tok), args.store)
    offs = offsets_for(store[0], int(tok.max()) + 1)
    ts_s, p_s = store[1], store[2]
    print(f"[null] price store: {store[0].size:,} points", flush=True)

    ep = pb["entry_price"].to_numpy(float)
    mid = pb["mid_at_entry"].to_numpy(float)
    p2 = pb["price_2m"].to_numpy(float)
    cut = pb["cutoff"].to_numpy(float)
    entry = pb["timestamp"].to_numpy(float)
    rv = pb["resolved_value"].to_numpy(float)
    k = pb["fee_k"].to_numpy(float)
    ex = pb["fee_exponent"].fillna(1.0).to_numpy(float)
    tick = pb["tick_size"].fillna(0.001).to_numpy(float)

    # placebo anchors, on exactly the same bets
    pre = first_price_in(ts_s, p_s, offs, tok, entry - DELTA, BANDWIDTH, cut)
    rnd = random_anchor_price(ts_s, p_s, offs, tok, cut, rng)
    rndp = random_anchor_price(ts_s, p_s, offs, tok, cut, rng, floor=entry + DELTA)
    pb["price_pre"], pb["price_rnd"], pb["price_rndpost"] = pre, rnd, rndp

    gap = ep - mid
    spread = {"tick_only": tick,
              "half_gap": np.maximum(0.5 * gap, tick),
              "published": np.maximum(gap, tick),
              "one_and_half": np.maximum(1.5 * gap, tick)}

    bl_pub = None
    if args.baseline is not None and args.baseline.exists():
        z = np.load(args.baseline)
        bl_pub = Baseline(z["edges"], z["means"], float(z["global_mean"]))
    if bl_pub is None:
        raise SystemExit("--baseline is required (npz with edges/means/global_mean)")

    def score(px, sp, bl, i=None):
        kk, xx, rr = (k, ex, rv) if i is None else (k[i], ex[i], rv[i])
        fill = np.clip(px + sp, 1e-6, 1.0 - 1e-9)
        fee = kk * np.power(fill * (1.0 - fill), xx)
        return rr - bl(fill) - fee, rr - fill - fee   # residual, money

    codes = pd.factorize(pb["market_id"].astype(str))[0]

    # ------------------------------------------------------------------
    # 1. per-wallet, published rule, with p-values and BH
    # ------------------------------------------------------------------
    sp_pub = spread["published"]
    resid, money = score(p2, sp_pub, bl_pub)
    own = rv - bl_pub(ep)
    pb["resid"], pb["money"], pb["own"] = resid, money, own

    valid = np.isfinite(p2) & np.isfinite(mid)
    rows = []
    for w, g in pb[valid].groupby("w"):
        if len(g) < MIN_BETS:
            continue
        c = codes[g.index.to_numpy()]
        m, lo, hi, p = cluster_boot(g["resid"].to_numpy(), c, rng)
        mm, mlo, mhi, mp = cluster_boot(g["money"].to_numpy(), c, rng)
        _, z = cluster_z(g["resid"].to_numpy(), c)
        rows.append(dict(wallet=w[:14], n=len(g), mkts=int(pd.unique(c).size),
                         own_c=100 * g["own"].mean(), net_c=100 * m,
                         lo=100 * lo, hi=100 * hi, boot_p=p, z=z,
                         money_c=100 * mm, money_lo=100 * mlo, money_hi=100 * mhi,
                         avg_fill=float(np.nanmean(np.clip(g["price_2m"] + sp_pub[g.index.to_numpy()], 0, 1)))))
    per = pd.DataFrame(rows)
    per["copyable"] = per["lo"] > 0
    per["bh_q10"] = bh_reject(per["boot_p"], 0.10)
    per["bh_q05"] = bh_reject(per["boot_p"], 0.05)
    per = per.sort_values("net_c", ascending=False).reset_index(drop=True)
    print("\n=== 1. PUBLISHED RULE, re-derived, with multiplicity ===")
    print(per.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"copyable (CI>0): {int(per['copyable'].sum())} of {len(per)}   "
          f"BH q=0.10: {int(per['bh_q10'].sum())}   BH q=0.05: {int(per['bh_q05'].sum())}")
    per.to_parquet(OUT_DIR / "per_wallet.parquet", index=False)

    # ------------------------------------------------------------------
    # 2. placebo anchors, paired on bets where all three exist
    # ------------------------------------------------------------------
    both = valid & np.isfinite(pre) & np.isfinite(rnd) & np.isfinite(rndp)
    print(f"\n=== 2. PLACEBO ANCHORS (paired n={int(both.sum()):,} of {int(valid.sum()):,}) ===")
    anchors = {"real(+2m)": p2, "pre-entry(-2m)": pre, "random": rnd,
               "random_POST(exec)": rndp}
    out2 = []
    for name, px in anchors.items():
        r, mny = score(px, sp_pub, bl_pub)
        m, lo, hi, p = cluster_boot(r[both], codes[both], rng)
        mm, *_ = cluster_boot(mny[both], codes[both], rng)
        out2.append(dict(anchor=name, resid_c=100 * m, lo=100 * lo, hi=100 * hi,
                         money_c=100 * mm))
    o2 = pd.DataFrame(out2)
    print(o2.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    for other in ("pre-entry(-2m)", "random", "random_POST(exec)"):
        d = (score(p2, sp_pub, bl_pub)[0] - score(anchors[other], sp_pub, bl_pub)[0])
        m, lo, hi, p = cluster_boot(d[both], codes[both], rng)
        print(f"  real - {other:<18}: {100*m:+.4f}c  95% CI [{100*lo:+.4f}, {100*hi:+.4f}]")
    o2.to_parquet(OUT_DIR / "placebo_pooled.parquet", index=False)

    # per wallet
    rows = []
    for w, g in pb[both].groupby("w"):
        if len(g) < MIN_BETS:
            continue
        i = g.index.to_numpy(); c = codes[i]
        r_real = score(p2[i], sp_pub[i], bl_pub, i)[0]
        r_rnd = score(rnd[i], sp_pub[i], bl_pub, i)[0]
        r_pre = score(pre[i], sp_pub[i], bl_pub, i)[0]
        r_rp = score(rndp[i], sp_pub[i], bl_pub, i)[0]
        m1, l1, h1, _ = cluster_boot(r_real - r_rnd, c, rng)
        m2, l2, h2, _ = cluster_boot(r_real - r_pre, c, rng)
        m3, l3, h3, _ = cluster_boot(r_real - r_rp, c, rng)
        rows.append(dict(wallet=w[:14], n=len(g),
                         real_c=100 * np.nanmean(r_real), rand_c=100 * np.nanmean(r_rnd),
                         rpost_c=100 * np.nanmean(r_rp), pre_c=100 * np.nanmean(r_pre),
                         d_rand=100 * m1, d_rand_lo=100 * l1, d_rand_hi=100 * h1,
                         d_rpost=100 * m3, d_rpost_lo=100 * l3, d_rpost_hi=100 * h3,
                         d_pre=100 * m2))
    pw = pd.DataFrame(rows).sort_values("d_rpost", ascending=False)
    print("\n--- per wallet, paired placebo differences ---")
    print(pw.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    pw.to_parquet(OUT_DIR / "placebo_per_wallet.parquet", index=False)

    # ------------------------------------------------------------------
    # 3. wallet-label permutation null
    # ------------------------------------------------------------------
    print(f"\n=== 3. WALLET-LABEL PERMUTATION NULL ({args.perms} permutations) ===")
    sub = pb[valid].copy()
    sub["c"] = codes[sub.index.to_numpy()]
    sizes = sub.groupby("w").size()
    keep = sizes[sizes >= MIN_BETS].index
    sub = sub[sub["w"].isin(keep)].reset_index(drop=True)
    labels = sub["w"].to_numpy()
    resid_v = sub["resid"].to_numpy()
    cvec = sub["c"].to_numpy()
    epv = sub["entry_price"].to_numpy()
    qs = np.unique(np.quantile(epv, np.linspace(0, 1, 21)))
    binid = np.clip(np.digitize(epv, qs[1:-1], right=False), 0, qs.size - 2)

    def count_sig(lab, thresh):
        n = 0
        for w in keep:
            m = lab == w
            _, z = cluster_z(resid_v[m], cvec[m])
            if np.isfinite(z) and z > thresh:
                n += 1
        return n

    def dispersion(lab):
        """SD of per-wallet mean follower edge. THIS is the statistic with power:
        the count test has none, because permuting labels leaves the pool's mean
        intact and hands every pseudo-wallet the pool mean."""
        ms = [np.nanmean(resid_v[lab == w]) for w in keep]
        return float(np.nanstd(ms, ddof=1))

    # calibrate the z threshold to reproduce the bootstrap "CI>0" call on real data
    real_calls = set(per.loc[per["copyable"], "wallet"])
    best_t, best_err = 1.96, 1e9
    for t in np.arange(1.2, 3.2, 0.05):
        calls = set()
        for w in keep:
            m = labels == w
            _, z = cluster_z(resid_v[m], cvec[m])
            if np.isfinite(z) and z > t:
                calls.add(w[:14])
        err = len(calls ^ real_calls)
        if err < best_err:
            best_err, best_t = err, t
    print(f"  z-threshold matched to the bootstrap rule: z>{best_t:.2f} "
          f"(disagrees on {best_err} of {len(keep)} wallets)")
    real_n = count_sig(labels, best_t)
    real_disp = dispersion(labels)
    null_counts, null_disp = [], []
    for _ in range(args.perms):
        lab = labels.copy()
        for b in np.unique(binid):
            m = binid == b
            lab[m] = rng.permutation(lab[m])
        null_counts.append(count_sig(lab, best_t))
        null_disp.append(dispersion(lab))
    null_counts = np.array(null_counts); null_disp = np.array(null_disp)
    p_count = float((null_counts >= real_n).mean())
    p_disp = float((null_disp >= real_disp).mean())
    print(f"  real copyable count (z rule): {real_n} of {len(keep)}")
    print(f"  null mean {null_counts.mean():.2f}  sd {null_counts.std():.2f}  "
          f"p95 {np.percentile(null_counts,95):.0f}  max {null_counts.max()}  "
          f"P(null >= real) = {p_count:.3f}")
    print("  NB a label permutation preserves the POOL's mean edge, so every "
          "pseudo-wallet\n     inherits it and the count test has no power in the "
          "claimed direction.\n     The statistic that does have power is the "
          "DISPERSION of per-wallet edge:")
    print(f"  real SD of per-wallet follower edge: {100*real_disp:.3f}c   "
          f"null mean {100*null_disp.mean():.3f}c  sd {100*null_disp.std():.3f}  "
          f"P(null >= real) = {p_disp:.3f}")
    est_fdr = float(np.clip(null_counts.mean() / max(real_n, 1), 0, 1))
    print(f"  count-based estimated FDR of the copyable set: {100*est_fdr:.1f}%")
    json.dump({"real_n": int(real_n), "null_mean": float(null_counts.mean()),
               "null_sd": float(null_counts.std()), "p": p_count,
               "est_fdr": est_fdr, "z_threshold": float(best_t),
               "n_wallets": int(len(keep)),
               "real_dispersion_c": 100 * real_disp,
               "null_dispersion_mean_c": 100 * float(null_disp.mean()),
               "p_dispersion": p_disp},
              open(OUT_DIR / "permutation_null.json", "w"), indent=2)

    # ------------------------------------------------------------------
    # 4. spread sweep
    # ------------------------------------------------------------------
    print("\n=== 4. SPREAD PROXY SWEEP ===")
    sweep = []
    for name, sp in spread.items():
        r, _ = score(p2, sp, bl_pub)
        tmp = pb.assign(r=r)
        nc = 0
        pooled = []
        for w, g in tmp[valid].groupby("w"):
            if len(g) < MIN_BETS:
                continue
            c = codes[g.index.to_numpy()]
            m, lo, hi, _ = cluster_boot(g["r"].to_numpy(), c, rng, n=1000)
            nc += int(lo > 0)
            pooled.append(m)
        m, lo, hi, _ = cluster_boot(r[valid], codes[valid], rng, n=1000)
        sweep.append(dict(spread=name, mean_charge_c=100 * float(np.nanmean(sp[valid])),
                          pooled_c=100 * m, pooled_lo=100 * lo, pooled_hi=100 * hi,
                          n_copyable=nc, n_wallets=len(pooled)))
    sw = pd.DataFrame(sweep)
    print(sw.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    sw.to_parquet(OUT_DIR / "spread_sweep.parquet", index=False)

    # ------------------------------------------------------------------
    # 5. IN-SAMPLE CONTAMINATION — the copy sim scores every bet, including the
    #    chronological first half that SELECTED these wallets.
    # ------------------------------------------------------------------
    print("\n=== 5. HELD-OUT ONLY (the half the wallets were NOT selected on) ===")
    sp_tab = certification_splits(sorted(pb["w"].unique()))
    print(f"  split reconstructed for {int(sp_tab['reconstructed'].sum())} of "
          f"{len(sp_tab)} wallets (checked against validate's in_sample_n+out_of_sample_n)")
    sp_tab.to_parquet(OUT_DIR / "wallet_splits.parquet", index=False)
    split = dict(zip(sp_tab["wallet"], sp_tab["split_ts"]))
    is_out = np.array([pb["timestamp"].iloc[i] >= split.get(pb["w"].iloc[i], np.inf)
                       for i in range(len(pb))])
    print(f"  scored cohort: {int((valid & ~is_out).sum()):,} in-sample bets, "
          f"{int((valid & is_out).sum()):,} held-out bets")
    rows = []
    for w, g in pb[valid].groupby("w"):
        for half, mask in (("in_sample", ~is_out[g.index.to_numpy()]),
                           ("held_out", is_out[g.index.to_numpy()])):
            gg = g[mask]
            if len(gg) < MIN_BETS:
                rows.append(dict(wallet=w[:14], half=half, n=len(gg), net_c=np.nan,
                                 lo=np.nan, hi=np.nan, copyable=False))
                continue
            c = codes[gg.index.to_numpy()]
            m, lo, hi, p = cluster_boot(gg["resid"].to_numpy(), c, rng, n=1000)
            rows.append(dict(wallet=w[:14], half=half, n=len(gg), net_c=100 * m,
                             lo=100 * lo, hi=100 * hi, copyable=bool(lo > 0)))
    hs = pd.DataFrame(rows)
    piv = hs.pivot(index="wallet", columns="half",
                   values=["n", "net_c", "lo", "copyable"])
    print(piv.to_string(float_format=lambda x: f"{x:8.3f}"))
    for half in ("in_sample", "held_out"):
        s = hs[hs["half"] == half]
        scored = s[s["n"] >= MIN_BETS]
        m, lo, hi, _ = cluster_boot(pb.loc[valid & ((~is_out) if half == "in_sample" else is_out),
                                           "resid"].to_numpy(),
                                    codes[valid & ((~is_out) if half == "in_sample" else is_out)],
                                    rng, n=1000)
        print(f"  {half:<10} pooled {100*m:+.3f}c [{100*lo:+.3f}, {100*hi:+.3f}]   "
              f"copyable {int(scored['copyable'].sum())} of {len(scored)} scored")
    hs.to_parquet(OUT_DIR / "halves.parquet", index=False)

    # ------------------------------------------------------------------
    # 6. BASELINE CIRCULARITY — refit E[outcome|price] with these wallets removed
    # ------------------------------------------------------------------
    print("\n=== 6. BASELINE CIRCULARITY ===")
    share = len(pb) / float(bl_pub.n_fit) if hasattr(bl_pub, "n_fit") else np.nan
    ex_path = OUT_DIR / "baseline_excl.npz"
    if ex_path.exists():
        z = np.load(ex_path)
        bl_ex = Baseline(z["edges"], z["means"], float(z["global_mean"]))
        r2, _ = score(p2, sp_pub, bl_ex)
        m, lo, hi, _ = cluster_boot(r2[valid], codes[valid], rng, n=1000)
        nc = 0
        for w, g in pb.assign(r2=r2)[valid].groupby("w"):
            if len(g) < MIN_BETS:
                continue
            c = codes[g.index.to_numpy()]
            _, l, _, _ = cluster_boot(g["r2"].to_numpy(), c, rng, n=1000)
            nc += int(l > 0)
        print(f"  baseline refit EXCLUDING these wallets: pooled {100*m:+.4f}c "
              f"[{100*lo:+.4f}, {100*hi:+.4f}]   copyable {nc} of 37")
    else:
        print(f"  (skipped: build {ex_path} first — see the audit doc)")

    # ------------------------------------------------------------------
    # 7. CELL PERMUTATION — the cluster-preserving version of §3's dispersion
    #    test. Permuting bet labels within price bins destroys market
    #    clustering and so understates the null's spread. This permutes whole
    #    (wallet, market) CELLS within size strata instead — every market keeps
    #    its bets, prices, outcomes and shared resolution, only WHICH wallet
    #    held it is randomized. Same bracket as null B2 in
    #    scripts/audit_persistence_fdr.py.
    # ------------------------------------------------------------------
    print(f"\n=== 7. CELL-PERMUTATION DISPERSION NULL ({args.perms} permutations) ===")
    cell = sub.groupby(["w", "c"], observed=True)["resid"].agg(["sum", "size"]).reset_index()
    csz = cell["size"].to_numpy()
    strata = np.clip(np.searchsorted(np.unique(np.quantile(csz, [0.2, 0.4, 0.6, 0.8])),
                                     csz, side="right"), 0, 4)
    owner = cell["w"].to_numpy()
    csum = cell["sum"].to_numpy()

    def disp_from(own):
        s = pd.DataFrame({"w": own, "s": csum, "n": csz}).groupby("w").sum()
        return float((s["s"] / s["n"]).std(ddof=1))

    real_cd = disp_from(owner)
    null_cd = []
    for _ in range(args.perms):
        o = owner.copy()
        for st in np.unique(strata):
            m = strata == st
            o[m] = rng.permutation(o[m])
        null_cd.append(disp_from(o))
    null_cd = np.array(null_cd)
    p_cd = float((null_cd >= real_cd).mean())
    print(f"  cells: {len(cell):,}   real SD of per-wallet follower edge "
          f"{100*real_cd:.3f}c   null mean {100*null_cd.mean():.3f}c "
          f"sd {100*null_cd.std():.3f}   P(null >= real) = {p_cd:.3f}")
    json.dump({"cells": int(len(cell)), "real_dispersion_c": 100 * real_cd,
               "null_dispersion_mean_c": 100 * float(null_cd.mean()),
               "p": p_cd}, open(OUT_DIR / "cell_permutation.json", "w"), indent=2)

    print(f"\n[null] artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
