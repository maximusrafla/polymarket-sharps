"""Project 1's persisted set, re-arbitrated against a cluster-aware null.

WHY. `src/validate.py` certifies a wallet as `edge_persisted` when its held-out
(second-half) residual skill edge is positive, significant by a one-sided
**t-test over bets**, and clears the 0.02 economic floor. The persistence rate
was reported as real ~27.3% vs a shuffled-outcomes null of ~5% — the last live
positive result in this repo validated against a **bet-level** null.

Both of those lean on the same assumption: that a wallet's bets are independent
observations. They are not. Bets in one market share one resolution event, so
the effective sample size is closer to the number of markets than the number of
bets, and a t-test over bets understates the variance. That is exactly what
overturned the black-swan tail finding (`docs/blackswan_cluster_null.md`).

THE BRACKETS.

  A. BET-LEVEL null (the repo's existing arbiter, reproduced for contrast) —
     permute per-bet residual edges within 1c entry-price bins, preserving the
     favorite-longshot baseline and each wallet's price mix while destroying the
     wallet<->outcome link. Then re-run the full validator and count persisters.

  B. CLUSTER-PRESERVING null (aggregate/count-level only) — permute WHICH WALLET
     made each bet, within each market, leaving every market's bets, prices,
     outcomes and timestamps exactly intact. Re-run the full validator and count
     persisters. Per `docs/blackswan_cluster_null.md` this permutation can be
     over-constrained in sparse strata and must NOT be used to arbitrate
     individual wallets — but its count-level comparison is a like-for-like
     contrast of the same procedure on real vs permuted data, so it stands.

  C. MARKET-BLOCK BOOTSTRAP (the per-wallet arbiter) — for each persisted
     wallet, resample its HELD-OUT markets with replacement, bets within a market
     kept together, so a shared resolution event is charged once instead of once
     per bet. This replaces the t-test's independence assumption while keeping
     everything the wallet actually chose. Reported with the DESIGN EFFECT
     D = Var_bootstrap(mean) / (s^2/n), i.e. how much the t-test understates the
     true sampling variance; a wallet's t-test p-value is too small by ~sqrt(D).

Both the significance gate and the 0.02 magnitude floor are re-tested under C,
since a cluster-robust interval also widens the magnitude question.

READ-ONLY. Reads data/interim/bet_ledger.parquet, calls only pure functions from
src.validate; writes nothing.
    PYTHONPATH=. .venv/bin/python scripts/audit_persistence_cluster.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.common import load_config, load_ledger  # noqa: E402
from src.features import fit_price_baseline, residual_edge_per_bet  # noqa: E402
from src.validate import (  # noqa: E402
    DEFAULT_MIN_BETS_PER_HALF,
    DEFAULT_MIN_SKILL_EDGE,
    DEFAULT_PRICE_BASELINE_BINS,
    certification_alpha,
    compute_oos_validation,
)
from audit_speed_gradient import permute_wallets_within_market  # noqa: E402

SEED = 12345
LEDGER_COLS = ["wallet", "market_id", "side", "entry_price", "resolved_value",
               "resolved", "timestamp"]
CENT_BIN = 0.01
N_BOOT = 4000        # market-block bootstrap resamples per wallet
N_SHUFFLES = 200     # aggregate null replications per bracket
# Cluster-robust inference needs enough independent clusters; below ~30 the
# block bootstrap is itself unreliable (and anti-conservative). Wallets under
# this many held-out markets cannot be certified by EITHER test and are reported
# separately rather than counted as survivors.
MIN_CLUSTERS = 30


# --------------------------------------------------------------------------- #
# Data prep                                                                    #
# --------------------------------------------------------------------------- #
def prepare(cfg):
    """Resolved BUY bets with the per-bet residual (skill) edge the validator
    scores, computed from the same price baseline it uses."""
    led = load_ledger(columns=LEDGER_COLS, categorical=["market_id"])
    led["wallet"] = led["wallet"].astype(str)
    r = led.loc[(led["side"] == "BUY") & led["resolved"]].copy()
    del led
    n_bins = cfg["scoring"].get("price_baseline_bins", DEFAULT_PRICE_BASELINE_BINS)
    baseline = fit_price_baseline(r, n_bins)
    r["resid"] = residual_edge_per_bet(r, baseline)
    r["cent"] = np.floor(r["entry_price"] / CENT_BIN).astype(int)
    return r


# --------------------------------------------------------------------------- #
# Vectorized replica of validate.compute_oos_validation's gates                #
# --------------------------------------------------------------------------- #
def validator_flags(wcode, ts, resid, n_wallets, oos_split, min_per_half,
                    alpha, min_skill_edge):
    """The validator's four gates, vectorized over a (wallet, time)-sorted view.

    Reproduces `compute_oos_validation` exactly: chronological split at
    floor(n*oos_split); in/out means NaN below `min_per_half`; a one-sided
    t-test on the held-out residuals (NaN when there is no variance, treated as
    not-significant); magnitude floor applied independently of candidacy.

    Returns (candidate, significant, magnitude_ok, persisted, out_mean, out_p,
    out_n) as arrays indexed by wallet code. Exactness is asserted against the
    real implementation in `check_replica`."""
    order = np.lexsort((ts, wcode))
    w, x = wcode[order], resid[order]
    starts = np.searchsorted(w, np.arange(n_wallets), side="left")
    ends = np.searchsorted(w, np.arange(n_wallets), side="right")
    n = ends - starts
    k = np.floor(n * oos_split).astype(int)          # in-sample count
    m = n - k                                        # held-out count

    c1 = np.concatenate([[0.0], np.cumsum(x)])
    s, e = starts, ends
    with np.errstate(invalid="ignore", divide="ignore"):
        in_mean = np.where(k >= min_per_half, (c1[s + k] - c1[s]) / np.maximum(k, 1), np.nan)
        out_sum = c1[e] - c1[s + k]
        raw_out_mean = np.where(m > 0, out_sum / np.maximum(m, 1), np.nan)
        out_mean = np.where(m >= min_per_half, raw_out_mean, np.nan)

    # Held-out spread, TWO-PASS. A one-pass sum-of-squares loses all precision on
    # this data: many wallets replicate one bet hundreds of times, so E[x^2] and
    # mean^2 are equal to ~15 digits and their difference is pure rounding noise —
    # which manufactured t~1e8 and false significance for degenerate wallets.
    pos = np.arange(x.size) - np.repeat(starts, n)
    held = pos >= np.repeat(k, n)
    xh = x[held]
    dev = xh - np.repeat(np.where(np.isfinite(raw_out_mean), raw_out_mean, 0.0), m)
    off = np.concatenate([[0], np.cumsum(m)])[:-1]
    valid = m > 0
    ssq = np.zeros(n_wallets)
    if xh.size:
        red = np.add.reduceat(dev * dev, off[valid])
        ssq[valid] = red
    with np.errstate(invalid="ignore", divide="ignore"):
        sd_pop = np.sqrt(np.where(m > 0, ssq / np.maximum(m, 1), np.nan))
        sd = np.sqrt(np.where(m >= 2, ssq / np.maximum(m - 1, 1), np.nan))
        # validate._oos_significance returns NaN when np.allclose(resid.std(), 0)
        # — population std, atol 1e-8 — and the caller reads NaN as not-significant.
        degenerate = ~(sd_pop > 1e-8)
        tstat = np.where((m >= max(min_per_half, 2)) & ~degenerate,
                         out_mean / (sd / np.sqrt(np.maximum(m, 1))), np.nan)
        out_p = np.where(np.isfinite(tstat), stats.t.sf(tstat, np.maximum(m - 1, 1)), np.nan)

    candidate = np.isfinite(in_mean) & (in_mean > 0)
    significant = candidate & np.isfinite(out_mean) & (out_mean > 0) \
        & np.isfinite(out_p) & (out_p < alpha)
    magnitude_ok = np.isfinite(out_mean) & (out_mean >= min_skill_edge)
    persisted = significant & magnitude_ok
    return candidate, significant, magnitude_ok, persisted, out_mean, out_p, m


def tie_straddlers(r: pd.DataFrame, oos_split: float) -> set:
    """Wallets whose in/out split is not determined by the data: the timestamp at
    the split boundary is tied with its neighbour, so an unstable sort can put
    either bet in either half. Small in practice, but it must be named rather
    than silently absorbed into a tolerance."""
    out = set()
    for w, g in r.groupby("wallet", observed=True, sort=False):
        ts = np.sort(g["timestamp"].to_numpy(float))
        k = int(np.floor(ts.size * oos_split))
        if 0 < k < ts.size and ts[k - 1] == ts[k]:
            out.add(w)
    return out


def check_replica(r, cfg, wcode, n_wallets, codes) -> pd.DataFrame:
    """Assert the vectorized replica reproduces src.validate exactly on real
    data. Without this the nulls would be scoring a lookalike, not the gate the
    repo actually emits."""
    print("[check] running src.validate.compute_oos_validation on the real ledger…",
          flush=True)
    t0 = time.time()
    truth = compute_oos_validation(r, cfg)
    print(f"[check] done in {time.time()-t0:.0f}s", flush=True)
    sc = cfg["scoring"]
    cand, sig, mag, per, out_mean, out_p, out_n = validator_flags(
        wcode, r["timestamp"].to_numpy(float), r["resid"].to_numpy(float), n_wallets,
        sc["oos_split"], sc.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF),
        certification_alpha(sc),  # scoped Project 1 alpha, matching src.validate
        sc.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE))
    mine = pd.DataFrame({"wallet": codes, "candidate": cand, "edge_significant": sig,
                         "edge_magnitude_ok": mag, "edge_persisted": per,
                         "out_mean": out_mean, "out_p": out_p, "out_n": out_n})
    t = truth.loc[truth["wallet"].isin(set(codes))].copy()
    j = mine.merge(t, on="wallet", suffixes=("_r", "_t"))

    # KNOWN, DOCUMENTED AMBIGUITY (not a replica bug): validate.py splits with
    # `sort_values("timestamp")`, whose default quicksort is UNSTABLE, so when a
    # wallet has tied timestamps straddling the split boundary, which bets land in
    # which half is not determined by the data. The replica uses a stable lexsort.
    # Such wallets are excluded from the equality assert and counted out loud.
    tied = tie_straddlers(r, cfg["scoring"]["oos_split"])
    amb = j["wallet"].isin(tied)
    print(f"  [check] wallets whose split is ambiguous (tied timestamps straddle the "
          f"boundary): {int(amb.sum())}")

    for col in ("edge_significant", "edge_magnitude_ok", "edge_persisted"):
        diff = j[f"{col}_r"].astype(bool) != j[f"{col}_t"].astype(bool)
        d_all, d_det = int(diff.sum()), int((diff & ~amb).sum())
        print(f"  [check] {col}: {d_det} mismatches of {int((~amb).sum())} "
              f"deterministic wallets ({d_all} incl. ambiguous)")
        assert d_det == 0, f"replica disagrees with src.validate on {col}"
    cand_t = (j["in_sample_residual_edge"] > 0).fillna(False)
    print(f"  [check] candidate: {int((j['candidate'] != cand_t).sum())} mismatches")
    print("  [check] vectorized replica == src.validate ✓")
    return mine


# --------------------------------------------------------------------------- #
# C — market-block bootstrap, the per-wallet arbiter                           #
# --------------------------------------------------------------------------- #
def market_block_bootstrap(held: pd.DataFrame, rng, n_boot, min_skill_edge):
    """Cluster-robust re-test of ONE wallet's held-out half.

    The validator asks: is the mean held-out residual > 0 (t-test over bets) and
    >= min_skill_edge? Both questions are re-asked with the wallet's MARKETS as
    the resampling unit, so bets sharing a resolution event cannot each count as
    independent evidence.

    Returns p_zero (P(bootstrap mean <= 0)), p_floor (P(mean < min_skill_edge)),
    the 5th-percentile lower bound, and the design effect D."""
    blk = held.groupby("market_id", observed=True)["resid"].agg(["sum", "size"])
    v = blk["sum"].to_numpy(float)
    c = blk["size"].to_numpy(float)
    k = v.size
    idx = rng.integers(0, k, size=(n_boot, k))
    means = v[idx].sum(axis=1) / c[idx].sum(axis=1)
    x = held["resid"].to_numpy(float)
    n = x.size
    t_var = x.var(ddof=1) / n                    # the t-test's variance of the mean
    boot_var = means.var()
    return {
        "p_zero": float((means <= 0).mean()),
        "p_floor": float((means < min_skill_edge).mean()),
        "lo5": float(np.percentile(means, 5)),
        "D": float(boot_var / t_var) if t_var > 0 else np.nan,
        "n_markets": int(k),
        "n_bets": int(n),
        "bets_per_market": float(n / k),
    }


# --------------------------------------------------------------------------- #
# Aggregate nulls                                                              #
# --------------------------------------------------------------------------- #
def null_bet_level(resid, cent, rng):
    """Null A: permute per-bet residuals within 1c entry-price bins."""
    out = resid.copy()
    for c in np.unique(cent):
        g = np.where(cent == c)[0]
        out[g] = resid[rng.permutation(g)]
    return out


def aggregate_nulls(r, wcode, n_wallets, cfg, rng, n_shuffles):
    """Persisted/candidate counts under both aggregate nulls."""
    sc = cfg["scoring"]
    args = (sc["oos_split"], sc.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF),
            certification_alpha(sc),  # scoped Project 1 alpha, matching src.validate
            sc.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE))
    ts = r["timestamp"].to_numpy(float)
    resid = r["resid"].to_numpy(float)
    cent = r["cent"].to_numpy()
    mcode = pd.factorize(r["market_id"], sort=False)[0].astype(np.int64)

    res = {"A": {"cand": [], "per": []}, "B": {"cand": [], "per": []}}
    t0 = time.time()
    for s in range(n_shuffles):
        shuf = null_bet_level(resid, cent, rng)
        c, _, _, p, *_ = validator_flags(wcode, ts, shuf, n_wallets, *args)
        res["A"]["cand"].append(int(c.sum())); res["A"]["per"].append(int(p.sum()))

        wp = permute_wallets_within_market(wcode, mcode, rng)
        c, _, _, p, *_ = validator_flags(wp, ts, resid, n_wallets, *args)
        res["B"]["cand"].append(int(c.sum())); res["B"]["per"].append(int(p.sum()))
        if s == 4:
            per = (time.time() - t0) / 5
            print(f"    [{per:.1f}s/shuffle; ~{per*n_shuffles/60:.0f} min total]", flush=True)
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in res.items()}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLES)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--skip-check", action="store_true",
                    help="skip the (slow) equality assert against src.validate")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    cfg = load_config()
    sc = cfg["scoring"]
    alpha = certification_alpha(sc)
    min_skill_edge = sc.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE)

    r = prepare(cfg)
    codes, wcode = np.unique(r["wallet"].to_numpy(), return_inverse=True)
    n_wallets = codes.size

    print("=" * 84)
    print("PROJECT 1 PERSISTED SET — CLUSTER-AWARE RE-CHECK  (read-only)")
    print("=" * 84)
    print(f"resolved BUY bets: {len(r):,}   wallets with resolved bets: {n_wallets:,}")
    print(f"gates: oos_split={sc['oos_split']}  min_bets_per_half="
          f"{sc.get('min_bets_per_half', DEFAULT_MIN_BETS_PER_HALF)}  alpha={alpha}  "
          f"min_skill_edge={min_skill_edge}")

    if args.skip_check:
        c, sig, mag, per, out_mean, out_p, out_n = validator_flags(
            wcode, r["timestamp"].to_numpy(float), r["resid"].to_numpy(float), n_wallets,
            sc["oos_split"], sc.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF),
            alpha, min_skill_edge)
        mine = pd.DataFrame({"wallet": codes, "candidate": c, "edge_significant": sig,
                             "edge_magnitude_ok": mag, "edge_persisted": per,
                             "out_mean": out_mean, "out_p": out_p, "out_n": out_n})
    else:
        mine = check_replica(r, cfg, wcode, n_wallets, codes)

    n_cand = int(mine["candidate"].sum())
    n_per = int(mine["edge_persisted"].sum())
    print(f"\n--- 1. The live funnel on today's ledger ---")
    print(f"  candidates (in-sample skill edge > 0): {n_cand}")
    print(f"  edge_significant: {int(mine['edge_significant'].sum())}   "
          f"edge_magnitude_ok: {int(mine['edge_magnitude_ok'].sum())}")
    print(f"  edge_persisted:   {n_per}   (rate {n_per/n_cand:.1%} of candidates)")

    # -- 2. Clustering exposure of the persisted set --------------------------
    persisted = mine.loc[mine["edge_persisted"], "wallet"].tolist()
    held_rows = []
    oos_split = sc["oos_split"]
    for w in persisted:
        g = r.loc[r["wallet"] == w].sort_values("timestamp")
        k = int(np.floor(len(g) * oos_split))
        held_rows.append(g.iloc[k:].assign(_w=w))
    held = pd.concat(held_rows) if held_rows else pd.DataFrame()

    print(f"\n--- 2. How clustered is the held-out evidence? ---")
    dens = held.groupby(["_w", "market_id"], observed=True).size()
    per_w = dens.groupby("_w").agg(["size", "mean"])
    per_w.columns = ["n_markets", "bets_per_market"]
    print(f"  persisted wallets: {len(persisted)}")
    print(f"  held-out bets per market — median {per_w['bets_per_market'].median():.2f}, "
          f"p90 {per_w['bets_per_market'].quantile(0.9):.2f}, "
          f"max {per_w['bets_per_market'].max():.2f}")
    print(f"  held-out markets per wallet — median {per_w['n_markets'].median():.0f}, "
          f"min {per_w['n_markets'].min():.0f}, "
          f"share with <{MIN_CLUSTERS} markets: "
          f"{100*(per_w['n_markets'] < MIN_CLUSTERS).mean():.0f}%")
    print("  Bets per market well above 1 means the t-test's independence assumption is")
    print("  materially wrong here: these wallets place many bets into the same market,")
    print("  and one resolution settles all of them at once.")

    # -- 3. Per-wallet cluster-robust re-test --------------------------------
    print(f"\n--- 3. Per-wallet re-test: market-block bootstrap ({args.boot} resamples) ---")
    recs = []
    for w in persisted:
        g = held.loc[held["_w"] == w]
        b = market_block_bootstrap(g, rng, args.boot, min_skill_edge)
        row = mine.loc[mine["wallet"] == w].iloc[0]
        b.update({"wallet": w, "out_mean": row["out_mean"], "t_p": row["out_p"]})
        recs.append(b)
    B = pd.DataFrame(recs)
    B["sig_boot"] = B["p_zero"] < alpha
    B["mag_boot"] = B["p_floor"] < 0.5          # point estimate clears the floor
    B["strict"] = B["lo5"] >= min_skill_edge    # 5th-pct lower bound clears the floor
    B["survives"] = B["sig_boot"] & B["mag_boot"]
    B["few_clusters"] = B["n_markets"] < MIN_CLUSTERS

    print(f"  design effect D = Var(bootstrap mean) / (s^2/n) — the factor by which the")
    print(f"  validator's t-test understates the variance of the held-out mean:")
    print(f"    median {B['D'].median():.2f}   10th {B['D'].quantile(.1):.2f}   "
          f"90th {B['D'].quantile(.9):.2f}   share D>1: {100*(B['D']>1).mean():.0f}%   "
          f"D>2: {100*(B['D']>2).mean():.0f}%")
    print(f"  => t-test p-values are too small by ~sqrt(D) (median "
          f"{np.sqrt(B['D'].median()):.2f}x) where D>1.")
    print(f"\n  of {len(B)} persisted wallets:")
    print(f"    still significant (cluster-robust p<{alpha}):     {int(B['sig_boot'].sum())}")
    print(f"    still clear the {min_skill_edge} floor (point est): {int(B['mag_boot'].sum())}")
    print(f"    SURVIVE both (cluster-robust edge_persisted):    {int(B['survives'].sum())}")
    print(f"    also clear the floor at the 5th-pct lower bound: {int(B['strict'].sum())}"
          "   <- the strict reading")
    ok = B["survives"] & ~B["few_clusters"]
    print(f"\n  Cluster-count adequacy (>= {MIN_CLUSTERS} held-out markets):")
    print(f"    persisted wallets with too few clusters to certify either way: "
          f"{int(B['few_clusters'].sum())}")
    print(f"    SURVIVORS ON ADEQUATE CLUSTER COUNTS: {int(ok.sum())}   "
          f"<- the defensible set")
    print(f"    of those, strict (5th-pct >= floor): "
          f"{int((ok & B['strict']).sum())}")
    print(f"\n  NOTE on multiplicity: the gate is a per-wallet test at alpha={alpha} over "
          f"{n_cand} candidates,")
    print(f"  so ~{alpha*n_cand:.0f} false positives are expected from the gate alone. "
          "That is what bracket B measures.")

    lost = B.loc[~B["survives"]].sort_values("t_p")
    print(f"\n  Wallets the t-test certified but the cluster-robust test does not: {len(lost)}")
    if not lost.empty:
        s = lost.head(25).copy()
        s["wallet"] = s["wallet"].str[:12] + "…"
        for c in ("out_mean", "lo5"):
            s[c] = s[c].round(4)
        s["D"] = s["D"].round(2)
        s["t_p"] = s["t_p"].map(lambda x: f"{x:.1e}")
        s["p_zero"] = s["p_zero"].round(3)
        print(s[["wallet", "n_bets", "n_markets", "bets_per_market", "out_mean",
                 "t_p", "p_zero", "lo5", "D"]].to_string(index=False))

    surv = B.loc[B["survives"]].sort_values("p_zero")
    print(f"\n  Survivors ({len(surv)}):")
    if not surv.empty:
        s = surv.copy()
        s["wallet"] = s["wallet"].str[:12] + "…"
        for c in ("out_mean", "lo5"):
            s[c] = s[c].round(4)
        s["D"] = s["D"].round(2)
        s["t_p"] = s["t_p"].map(lambda x: f"{x:.1e}")
        s["p_zero"] = s["p_zero"].round(4)
        s["few_cl"] = s["few_clusters"]
        print(s[["wallet", "n_bets", "n_markets", "bets_per_market", "out_mean",
                 "t_p", "p_zero", "lo5", "D", "strict", "few_cl"]].to_string(index=False))

    # -- 4. Aggregate: real persisted count vs both nulls ---------------------
    print(f"\n--- 4. Aggregate: is the persisted COUNT above chance? "
          f"({args.shuffles} shuffles/bracket) ---")
    nulls = aggregate_nulls(r, wcode, n_wallets, cfg, rng, args.shuffles)
    for key, label in (("A", "bet-level (repo's existing null)"),
                       ("B", "cluster-preserving (wallet within market)")):
        per = nulls[key]["per"]
        cand = nulls[key]["cand"]
        rate = per / np.maximum(cand, 1)
        p = float((per >= n_per).mean())
        print(f"  null {key} — {label}:")
        print(f"    persisted: mean {per.mean():.1f}, 95th {np.percentile(per,95):.0f}, "
              f"max {per.max()}   (real {n_per})")
        print(f"    rate:      mean {rate.mean():.1%}   (real {n_per/n_cand:.1%})")
        print(f"    P(null >= real) = {p:.3f}")

    print("\n" + "=" * 84)
    print("Read the per-wallet verdict from the market-block bootstrap (bracket C) and")
    print("the count-level verdict from bracket B. The t-test in src.validate treats")
    print("bets as independent; where D>1 its p-values are too small by ~sqrt(D).")
    print("=" * 84)


if __name__ == "__main__":
    main()
