"""Speed-gradient audit — Project 3 stage 1 (docs/project3_slow_markets.md §4).

Question: does forecasting skill get *stronger* as markets get slower? The
slow-market thesis predicts per-wallet skill edge and its out-of-sample
persistence rise with market lifespan L and fall with competition density.

READ-ONLY. Writes nothing outside --out (a scratch parquet of per-cell results).

    PYTHONPATH=. .venv/bin/python scripts/audit_speed_gradient.py --meta <clob_meta.parquet>

⚠️ ONE-WAY EVIDENCE, NOT A GATE. The deep real-world bets in this ledger belong
to the top-500 wallets, ranked under the old 82%-micro-crypto regime (28 of 36
persisters are `high_frequency_micro_market`). So this test's power comes almost
entirely from micro-specialists trading real-world markets on the side; the
slow-market specialists the thesis is about are largely absent by construction.
A RISING gradient is strong evidence (edge despite adverse selection). A FLAT
gradient is ambiguous — it cannot distinguish "no slow-market edge" from "the
wallets with it aren't in this sample" — and does not falsify the thesis.

Statistics per (speed bucket) cell, all favorite-longshot-neutral:
  1. mean skill edge vs a GLOBAL real-world price baseline (descriptive: does the
     cell beat the market-wide curve — conflates cell calibration with skill);
  2. PRIMARY — split-half persistence: residualize against a CELL-LOCAL baseline
     (so within-cell favorite-longshot cannot masquerade as skill), then Spearman
     between each wallet's first-half and second-half mean residual;
  3. excess cross-wallet variance: Var(wallet means) / E[within-wallet var / n].
     >1 means wallets genuinely differ in skill beyond sampling noise;
  4. shuffled-outcome null for (2) and (3): outcomes permuted within
     (cell, 1c price bin), preserving each cell's favorite-longshot base rate and
     each wallet's price mix, destroying only the wallet<->outcome link.
  5. within-wallet paired slow-vs-fast test — controls wallet identity, which is
     the direct answer to the adverse-selection caveat above.
  6. micro-crypto as the null-control cell.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from src.common import BET_LEDGER_PATH  # noqa: E402
from src.features import fit_price_baseline, expected_outcome  # noqa: E402

HOUR = 3600.0
# Market-lifespan buckets (L = end_date_iso - accepting_order_timestamp).
L_EDGES = [0, 6 * HOUR, 24 * HOUR, 3 * 86400, 7 * 86400, 30 * 86400, np.inf]
L_NAMES = ["<6h", "6-24h", "1-3d", "3-7d", "7-30d", ">30d"]
# Per-bet horizon buckets (H = resolution anchor - entry timestamp).
H_EDGES = [0, HOUR, 6 * HOUR, 24 * HOUR, 3 * 86400, 7 * 86400, np.inf]
H_NAMES = ["<1h", "1-6h", "6-24h", "1-3d", "3-7d", ">7d"]

MIN_HALF = 5        # bets required in EACH chronological half to enter the Spearman
MIN_CELL_N = 20     # bets required in a cell to enter the variance decomposition
MIN_PAIR = 20       # bets required in BOTH buckets for the within-wallet paired test
N_SHUFFLES = 200
PRICE_BIN = 0.01


# --------------------------------------------------------------------------
# core statistics (numpy; the shuffle loop re-runs these ~200x per cell)
# --------------------------------------------------------------------------
def _wallet_half_means(resid, w_start, w_len):
    """Per-wallet (first-half mean, second-half mean) over a wallet-major,
    chronologically-sorted residual array. Wallets with <2*MIN_HALF bets -> NaN."""
    n = len(w_start)
    h1 = np.full(n, np.nan)
    h2 = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(resid)])
    half = w_len // 2
    ok = (half >= MIN_HALF) & ((w_len - half) >= MIN_HALF)
    s, m, e = w_start[ok], (w_start + half)[ok], (w_start + w_len)[ok]
    h1[ok] = (csum[m] - csum[s]) / (m - s)
    h2[ok] = (csum[e] - csum[m]) / (e - m)
    return h1, h2


def _wallet_stats(resid, w_start, w_len):
    """Per-wallet mean and unbiased variance of the residual."""
    csum = np.concatenate([[0.0], np.cumsum(resid)])
    csq = np.concatenate([[0.0], np.cumsum(resid ** 2)])
    s, e = w_start, w_start + w_len
    tot = csum[e] - csum[s]
    sq = csq[e] - csq[s]
    mean = tot / w_len
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sq - w_len * mean ** 2) / np.maximum(w_len - 1, 1)
    return mean, np.maximum(var, 0.0)


def cell_statistics(resid, w_start, w_len, want_depth=False):
    """(spearman rho, n wallets in rho, variance ratio, n wallets in ratio).

    NOTE for cross-cell reading: rho rises mechanically with per-wallet DEPTH
    (deeper wallets have less noisy half-means at identical true skill), so a
    cell whose wallets are deeper shows a higher rho for free. `want_depth`
    returns the median depth of the wallets entering rho so that confound is
    visible. `var_ratio` explicitly divides out sampling noise (sigma^2/n) and is
    therefore the sounder statistic to compare ACROSS cells."""
    h1, h2 = _wallet_half_means(resid, w_start, w_len)
    ok = np.isfinite(h1) & np.isfinite(h2)
    rho = np.nan
    if ok.sum() >= 8:
        rho = stats.spearmanr(h1[ok], h2[ok]).statistic
    if want_depth:
        depth = float(np.median(w_len[ok])) if ok.sum() else np.nan
    mean, var = _wallet_stats(resid, w_start, w_len)
    big = w_len >= MIN_CELL_N
    ratio = np.nan
    if big.sum() >= 8:
        observed = float(np.var(mean[big], ddof=1))
        expected = float(np.mean(var[big] / w_len[big]))
        ratio = observed / expected if expected > 0 else np.nan
    if want_depth:
        return rho, int(ok.sum()), ratio, int(big.sum()), depth
    return rho, int(ok.sum()), ratio, int(big.sum())


def shuffle_within_bins(y, bin_codes, rng):
    """Permute outcomes within price bins (preserves the favorite-longshot base
    rate and each wallet's price mix; destroys the wallet<->outcome link)."""
    by_bin = np.argsort(bin_codes, kind="stable")
    perm = np.lexsort((rng.random(y.size), bin_codes))
    out = np.empty_like(y)
    out[by_bin] = y[perm]
    return out


def permute_wallets_within_market(wallet_codes, mcodes, rng):
    """CLUSTER-PRESERVING null: permute which wallet made each bet, WITHIN each
    market. Every market keeps its exact bets, prices and outcomes — only
    attribution changes — so the correlation induced by bets sharing a single
    resolution event survives intact.

    This is the null the bet-level price-bin shuffle cannot provide: that one
    destroys market clustering, so its spread is too narrow and any
    cluster-driven excess reads as significance. That is precisely how Project 2
    §1.5 manufactured two false positives from single-event artifacts.

    Limitation, stated plainly: holding markets fixed also removes any skill that
    comes from CHOOSING which markets to bet — this null can only see
    side/timing selection within a market. It is therefore the conservative
    bracket; the price-bin shuffle is the permissive one. Report both."""
    perm = np.lexsort((rng.random(wallet_codes.size), mcodes))
    by_mkt = np.argsort(mcodes, kind="stable")
    out = np.empty_like(wallet_codes)
    out[by_mkt] = wallet_codes[perm]
    return out


def stats_for_wallet_assignment(wallet_codes, ts, resid):
    """Re-sort into wallet-major/chronological order for an arbitrary wallet
    assignment and return the cell statistics."""
    o = np.lexsort((ts, wallet_codes))
    w, r = wallet_codes[o], resid[o]
    bnd = np.flatnonzero(np.diff(w)) + 1
    w_start = np.concatenate([[0], bnd])
    w_len = np.diff(np.concatenate([w_start, [len(w)]]))
    return cell_statistics(r, w_start, w_len)


# --------------------------------------------------------------------------
def market_influence(df, base_local, top_k=20, max_rows=1_500_000):
    """Leave-one-market-out influence on the split-half Spearman.

    The bet-level shuffle null breaks the correlation induced by bets sharing a
    market (and therefore a single resolution event), so it runs too narrow —
    exactly the failure that manufactured Project 2 §1.5's two false positives.
    This is the market-level robustness check, in the same form
    `audit_edge_decay_long.py` used (leave-one-family-out: +0.123 -> +0.092).

    `df` MUST be the frame `base_local` was fitted on, in the same row order
    (`prepare_cell` sorts internally and returns it as `sorted_df`). Pairing an
    unsorted frame with a sorted baseline gives every wallet a spurious constant
    residual offset, which is present in both chronological halves and therefore
    inflates the split-half correlation dramatically — this bug produced
    "leave-one-out" values of +0.5..+0.8 against point estimates near +0.16.

    Returns (rho_min, worst_market_share): the lowest rho over the `top_k`
    largest markets each removed in turn, and the largest market's share of the
    cell's bets."""
    n_mkt = df["market_id"].nunique()
    if len(df) > max_rows or n_mkt < 10:
        return np.nan, np.nan
    resid_all = df["resolved_value"].to_numpy(float) - base_local
    wallet_codes = pd.factorize(df["wallet"], sort=False)[0]
    ts = df["timestamp"].to_numpy()
    mcodes = pd.factorize(df["market_id"], sort=False)[0]
    counts = np.bincount(mcodes)
    biggest = np.argsort(counts)[::-1][:top_k]

    rhos = []
    for m in biggest:
        keep = mcodes != m
        w, t, r = wallet_codes[keep], ts[keep], resid_all[keep]
        o = np.lexsort((t, w))
        w, r = w[o], r[o]
        bnd = np.flatnonzero(np.diff(w)) + 1
        w_start = np.concatenate([[0], bnd])
        w_len = np.diff(np.concatenate([w_start, [len(w)]]))
        rho, _, _, _ = cell_statistics(r, w_start, w_len)
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return np.nan, np.nan
    return float(min(rhos)), float(counts.max() / counts.sum())


def effective_breadth(df):
    """Kish effective number of markets: (sum n)^2 / sum n^2. Collapses toward 1
    when one market dominates the cell."""
    n = df.groupby("market_id", sort=False).size().to_numpy(float)
    return float(n.sum() ** 2 / (n ** 2).sum())


def prepare_cell(df, global_baseline):
    """Sort wallet-major/chronological and return the arrays the stats need."""
    df = df.sort_values(["wallet", "timestamp"], kind="stable")
    codes, _ = pd.factorize(df["wallet"], sort=False)
    # factorize after sorting => codes are already grouped and monotone
    w_len = np.bincount(codes)
    w_start = np.concatenate([[0], np.cumsum(w_len)[:-1]])
    price = df["entry_price"].to_numpy(float)
    y = df["resolved_value"].to_numpy(float)
    local = fit_price_baseline(df, n_bins=20)
    base_local = expected_outcome(local, price)
    base_global = expected_outcome(global_baseline, price)
    bin_codes = np.round(price / PRICE_BIN).astype(np.int32)
    return dict(y=y, price=price, base_local=base_local, base_global=base_global,
                bin_codes=bin_codes, w_start=w_start, w_len=w_len, sorted_df=df,
                n_wallets=len(w_len), n_markets=df["market_id"].nunique())


def analyse_cell(name, df, global_baseline, rng, n_shuffles=N_SHUFFLES):
    c = prepare_cell(df, global_baseline)
    resid = c["y"] - c["base_local"]
    rho, n_rho, ratio, n_ratio, depth = cell_statistics(
        resid, c["w_start"], c["w_len"], want_depth=True)

    null_rho, null_ratio = [], []
    for _ in range(n_shuffles):
        y_s = shuffle_within_bins(c["y"], c["bin_codes"], rng)
        r_s, _, ra_s, _ = cell_statistics(y_s - c["base_local"], c["w_start"], c["w_len"])
        if np.isfinite(r_s):
            null_rho.append(r_s)
        if np.isfinite(ra_s):
            null_ratio.append(ra_s)
    null_rho = np.array(null_rho)
    null_ratio = np.array(null_ratio)

    # Cluster-preserving null: permute wallet labels within each market.
    sdf = c["sorted_df"]
    wcodes = pd.factorize(sdf["wallet"], sort=False)[0]
    mcodes = pd.factorize(sdf["market_id"], sort=False)[0]
    ts_arr = sdf["timestamp"].to_numpy()
    cl_rho, cl_ratio = [], []
    n_cluster = min(n_shuffles, 100 if len(sdf) > 500_000 else n_shuffles)
    for _ in range(n_cluster):
        wc = permute_wallets_within_market(wcodes, mcodes, rng)
        r_s, _, ra_s, _ = stats_for_wallet_assignment(wc, ts_arr, resid)
        if np.isfinite(r_s):
            cl_rho.append(r_s)
        if np.isfinite(ra_s):
            cl_ratio.append(ra_s)
    cl_rho = np.array(cl_rho)
    cl_ratio = np.array(cl_ratio)

    def emp_p(obs, null):
        if not np.isfinite(obs) or null.size == 0:
            return np.nan
        return float((np.sum(null >= obs) + 1) / (null.size + 1))

    rho_lomo, top_share = market_influence(c["sorted_df"], c["base_local"])

    return {
        "cell": name,
        "n_bets": len(df),
        "n_wallets": c["n_wallets"],
        "n_markets": c["n_markets"],
        "eff_breadth": effective_breadth(df),
        "spearman_lomo_min": rho_lomo,
        "top_market_share": top_share,
        "mean_price": float(c["price"].mean()),
        "raw_edge": float((c["y"] - c["price"]).mean()),
        "skill_edge_global": float((c["y"] - c["base_global"]).mean()),
        "spearman": rho,
        "spearman_n": n_rho,
        "median_depth": depth,
        "spearman_null_mean": float(null_rho.mean()) if null_rho.size else np.nan,
        "spearman_p": emp_p(rho, null_rho),
        "var_ratio": ratio,
        "var_ratio_n": n_ratio,
        "var_ratio_null_mean": float(null_ratio.mean()) if null_ratio.size else np.nan,
        "var_ratio_p": emp_p(ratio, null_ratio),
        "spearman_clnull_mean": float(cl_rho.mean()) if cl_rho.size else np.nan,
        "spearman_clnull_p": emp_p(rho, cl_rho),
        "var_ratio_clnull_mean": float(cl_ratio.mean()) if cl_ratio.size else np.nan,
        "var_ratio_clnull_p": emp_p(ratio, cl_ratio),
    }


def within_wallet_paired(df, fast_mask, slow_mask, label):
    """Same wallet, fast bucket vs slow bucket. Each side residualized against
    its OWN bucket baseline, so this asks 'did they beat the price more in slow
    markets', not 'is one bucket better calibrated'."""
    out = []
    for mask, side in ((fast_mask, "fast"), (slow_mask, "slow")):
        sub = df[mask]
        if sub.empty:
            return None
        base = fit_price_baseline(sub, n_bins=20)
        r = sub["resolved_value"].to_numpy(float) - expected_outcome(base, sub["entry_price"].to_numpy(float))
        g = pd.DataFrame({"wallet": sub["wallet"].to_numpy(), "r": r}).groupby("wallet")["r"]
        out.append(g.agg(["mean", "size"]).rename(columns={"mean": side, "size": f"n_{side}"}))
    j = out[0].join(out[1], how="inner")
    j = j[(j.n_fast >= MIN_PAIR) & (j.n_slow >= MIN_PAIR)]
    if len(j) < 8:
        print(f"  {label}: only {len(j)} wallets qualify (need >=8) — underpowered, skipped")
        return None
    d = j["slow"] - j["fast"]
    t = stats.ttest_1samp(d, 0.0)
    w = stats.wilcoxon(d) if len(d) >= 10 else None
    print(f"  {label}: n_wallets={len(j)}  mean(slow-fast)={d.mean():+.4f}  "
          f"median={d.median():+.4f}  t p={t.pvalue:.4f}"
          + (f"  wilcoxon p={w.pvalue:.4f}" if w else ""))
    print(f"     mean skill edge  fast={j['fast'].mean():+.4f}  slow={j['slow'].mean():+.4f}  "
          f"| wallets improving in slow: {(d > 0).sum()}/{len(j)}")
    return j


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="parquet: market_id, open_ts, end_ts, game_ts")
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bets", default=None,
                    help="parquet of resolved bets to analyse instead of the shared ledger "
                         "(same columns). Used for the slow-specialist pilot cohort.")
    ap.add_argument("--label", default="ledger", help="name for this population in the output")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only the aggregate real-world cell (the cross-population comparison); "
                         "skip the L/H buckets, micro control and paired tests")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(20260723)

    meta = pd.read_parquet(args.meta)
    meta = meta[meta.open_ts.notna() & meta.end_ts.notna()].copy()
    print(f"[meta] markets with open+end timestamps: {len(meta):,}")

    cols = ["wallet", "market_id", "side", "resolved", "entry_price", "resolved_value",
            "timestamp", "slug", "question"]
    import pyarrow.parquet as pq
    keep, micro_keep, tape_parts = [], [], []
    wanted = set(meta["market_id"])
    from src.discover import is_micro_crypto
    src = args.bets or BET_LEDGER_PATH
    print(f"[data] population: {args.label}  source: {src}")
    pf = pq.ParquetFile(src)
    for b in pf.iter_batches(batch_size=500_000, columns=cols):
        d = b.to_pandas()
        # Per-market observed tape bounds over ALL rows (BUY+SELL, resolved or not):
        # t_max is a hard LOWER bound on when the market was still trading.
        tape_parts.append(d.groupby("market_id", sort=False)["timestamp"].agg(["min", "max"]).reset_index())
        d = d[(d.side == "BUY") & d.resolved]
        micro = np.fromiter((is_micro_crypto(s, q) for s, q in zip(d.slug, d.question)),
                            bool, len(d))
        mk = d[micro]
        if len(mk):
            micro_keep.append(mk[["wallet", "market_id", "entry_price", "resolved_value", "timestamp"]])
        d = d[~micro & d.market_id.isin(wanted)]
        if len(d):
            keep.append(d[["wallet", "market_id", "entry_price", "resolved_value", "timestamp"]])
        del b, d
    rw = pd.concat(keep, ignore_index=True)
    del keep
    micro_df = pd.concat(micro_keep, ignore_index=True)
    del micro_keep
    tape = (pd.concat(tape_parts, ignore_index=True)
              .groupby("market_id", sort=False).agg(t_min=("min", "min"), t_max=("max", "max")))
    del tape_parts

    # `end_date_iso` is the MIDNIGHT FLOOR of the scheduled end day (verified), and
    # actual resolution can be later still, so it alone puts 87% of bets at a
    # negative horizon. The observed tape's last trade is a hard lower bound on
    # when the market was still live, so anchor on max(end_ts, t_max): H is then
    # non-negative by construction and a conservative UNDER-estimate of the true
    # entry->resolution horizon.
    meta = meta.join(tape, on="market_id")
    meta["anchor"] = np.fmax(meta["end_ts"], meta["t_max"])
    meta["L"] = meta["anchor"] - meta["open_ts"]
    meta = meta[meta["L"] > 0]
    meta_idx = meta.set_index("market_id")[["L", "anchor"]]

    before = len(rw)
    rw = rw.join(meta_idx, on="market_id")
    rw = rw[rw["L"].notna()]
    rw["H"] = rw["anchor"] - rw["timestamp"]
    print(f"[data] real-world bets with usable metadata: {len(rw):,} of {before:,} joined   "
          f"micro control: {len(micro_df):,}")
    print(f"[data] negative horizons after tape anchoring: {(rw['H'] < 0).sum():,} "
          f"(should be 0)   load {time.time()-t0:.0f}s")

    global_baseline = fit_price_baseline(rw, n_bins=20)

    rw["L_bucket"] = pd.cut(rw["L"], L_EDGES, labels=L_NAMES, right=False)
    rw["H_bucket"] = pd.cut(rw["H"], H_EDGES, labels=H_NAMES, right=False)

    rows = []

    # Aggregate real-world cell: the directly comparable number ACROSS populations
    # (per-bucket cells are not, because bucket composition differs by cohort).
    print(f"\n{'='*100}\n=== AGGREGATE — all real-world bets in this population\n{'='*100}")
    r = analyse_cell("ALL:real_world", rw, global_baseline, rng, args.shuffles)
    r["axis"] = "aggregate"
    rows.append(r)
    print(f"  ALL   : n={r['n_bets']:>9,} w={r['n_wallets']:>6,} mkts={r['n_markets']:>7,} "
          f"effB={r['eff_breadth']:>8.1f}\n"
          f"          rho={r['spearman']:+.3f} (n={r['spearman_n']}, depth={r['median_depth']:.0f}, "
          f"binNull={r['spearman_null_mean']:+.3f} p={r['spearman_p']:.3f}, "
          f"clNull={r['spearman_clnull_mean']:+.3f} p={r['spearman_clnull_p']:.3f})\n"
          f"          varR={r['var_ratio']:.2f} (binNull={r['var_ratio_null_mean']:.2f} "
          f"p={r['var_ratio_p']:.3f}, clNull={r['var_ratio_clnull_mean']:.2f} "
          f"p={r['var_ratio_clnull_p']:.3f})  ==> ADJUSTED RATIO = "
          f"{r['var_ratio']/r['var_ratio_clnull_mean']:.2f}", flush=True)

    for axis, col, names in ((("L", "L_bucket", L_NAMES), ("H", "H_bucket", H_NAMES))
                             if not args.aggregate_only else ()):
        print(f"\n{'='*100}\n=== AXIS {axis} — "
              f"{'market lifespan (open -> close)' if axis=='L' else 'bet horizon (entry -> resolution)'}\n{'='*100}")
        for name in names:
            sub = rw[rw[col] == name]
            if len(sub) < 200:
                print(f"  {name:>6}: n={len(sub):,} — too small, skipped")
                continue
            r = analyse_cell(f"{axis}:{name}", sub, global_baseline, rng, args.shuffles)
            r["axis"] = axis
            rows.append(r)
            print(f"  {name:>6}: n={r['n_bets']:>7,} w={r['n_wallets']:>5,} mkts={r['n_markets']:>6,} "
                  f"effB={r['eff_breadth']:>7.1f} | skill(global)={r['skill_edge_global']:+.4f} | "
                  f"rho={r['spearman']:+.3f} (n={r['spearman_n']:>4}, depth={r['median_depth']:>6.0f}, "
                  f"null={r['spearman_null_mean']:+.3f}, "
                  f"p={r['spearman_p']:.3f}, lomo>={r['spearman_lomo_min']:+.3f}, "
                  f"top_mkt={100*r['top_market_share']:.1f}%)\n"
                  f"          clusterNull: rho_null={r['spearman_clnull_mean']:+.3f} p={r['spearman_clnull_p']:.3f}"
                  f" | varR={r['var_ratio']:.2f} (binNull={r['var_ratio_null_mean']:.2f} p={r['var_ratio_p']:.3f},"
                  f" clNull={r['var_ratio_clnull_mean']:.2f} p={r['var_ratio_clnull_p']:.3f})", flush=True)

    if args.aggregate_only:
        res = pd.DataFrame(rows)
        if args.out:
            res["population"] = args.label
            res.to_parquet(args.out)
            print(f"\n[out] {args.out}")
        print(f"[done] {time.time()-t0:.0f}s")
        return

    print(f"\n{'='*100}\n=== NULL CONTROL — micro-crypto (the un-copyable slice)\n{'='*100}")
    r = analyse_cell("micro", micro_df, global_baseline, rng, args.shuffles)
    r["axis"] = "control"
    rows.append(r)
    print(f"  micro : n={r['n_bets']:>9,} w={r['n_wallets']:>5,} | "
          f"rho={r['spearman']:+.3f} (n={r['spearman_n']}, null={r['spearman_null_mean']:+.3f}, "
          f"p={r['spearman_p']:.3f}) | varR={r['var_ratio']:.2f} (null={r['var_ratio_null_mean']:.2f})")

    print(f"\n{'='*100}\n=== WITHIN-WALLET PAIRED (controls wallet identity)\n{'='*100}")
    within_wallet_paired(rw, rw["L"] < 24 * HOUR, rw["L"] >= 7 * 86400, "L: <24h  vs  >=7d")
    within_wallet_paired(rw, rw["L"] < 3 * 86400, rw["L"] >= 3 * 86400, "L: <3d   vs  >=3d")
    within_wallet_paired(rw, rw["H"] < 6 * HOUR, rw["H"] >= 3 * 86400, "H: <6h   vs  >=3d")

    res = pd.DataFrame(rows)
    if args.out:
        res["population"] = args.label
        res.to_parquet(args.out)
        print(f"\n[out] {args.out}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
