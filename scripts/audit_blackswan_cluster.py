"""Black-swan / tail-edge detector, re-run against a CLUSTER-PRESERVING null.

WHY. `scripts/audit_blackswan.py` arbitrated its tail-edge winners with a
bet-level shuffled-outcome null (permute resolved outcomes within 1c price
bins). Project 3 stage 1 (docs/project3_slow_markets.md §10.1) measured that
this null is **too permissive for wallet-level statistics**: it destroys market
clustering, so bets sharing a single resolution event read as skill. On the
speed-gradient cells it cut significance from 8/12 to 3/12 and moved the
variance reference from ~1.0 to 1.75-5.78.

The black-swan result (19 wallets clear, 11 survive BH-FDR, null mean 3.1) is
the repo's only live positive claim, and it was certified by exactly the null
now known to be too permissive. This script re-runs it against the
cluster-preserving null and reports what survives.

THE THREE BRACKETS (all re-run here on today's ledger, so the comparison is
apples-to-apples; the historic 19/11 was measured on a smaller ledger):

  A. BET-LEVEL (the permissive bracket, as in audit_blackswan.py) — permute
     resolved outcomes within 1c price bins. Preserves the market-wide base rate
     and each wallet's price mix; destroys only wallet<->outcome. Also destroys
     the fact that many bets share one resolution event.

  B. CLUSTER-PRESERVING (the conservative bracket, as in
     audit_speed_gradient.permute_wallets_within_market) — permute WHICH WALLET
     made each tail bet, within each market. Every market keeps its exact bets,
     prices and outcomes; only attribution changes, so correlation through a
     shared resolution event survives intact. Each wallet keeps its exact tail
     bet count (the label multiset is preserved), but its price mix and markets
     are resampled from what its markets actually contained.

     Limitation, stated plainly (same as the speed-gradient audit): holding
     markets fixed also removes any skill that comes from CHOOSING which markets
     to bet in — this null can only see side/timing selection within a market.

     MEASURED LIMITATION, found by this run and NOT anticipated by §10.1: in the
     sparse tail (median 2 tail bets per market) this permutation is so
     constrained that its per-wallet reference is NARROWER than the independence
     model — median design effect 0.17, i.e. 6x LESS variance than a
     Poisson-binomial. It therefore manufactures its own per-wallet false
     positives (wallets with pb_p=0.4 land at p_cluster=5e-4) and must not be
     used as a wallet-level arbiter here. Its COUNT-level comparison, which is
     a like-for-like contrast of the same procedure on real vs permuted data,
     remains informative. §6/§10.1's "make it standard for every wallet-level
     statistic" needs this qualification: check the design effect first.

  C. MARKET-BLOCK BOOTSTRAP (added here; the per-wallet arbiter) — resample each
     wallet's MARKETS with replacement, bets within a market kept together, so a
     shared resolution event is charged once instead of once per bet. Unlike B it
     keeps the wallet's own bets and its market choices, so it tests tail skill
     INCLUDING market selection while still correcting the clustering. Design
     effect median 1.45 (40% of wallets >2) — it inflates variance where
     clustering exists, which is what a cluster-robust instrument should do.

STATISTICS. Real per-wallet tail metrics are computed exactly as in
audit_blackswan.py (hit rate vs. market base rate E[outcome|entry_price], exact
Poisson-binomial p, ROI, Kelly growth). Two null-based verdicts:

  1. WINNER COUNT — how many wallets clear the audit's winner gate
     (resid_vs_base>0 AND Poisson-binomial p<alpha) in the real data vs. under
     each null. Directly comparable to the audit's "19 real vs null mean 3.1".
  2. PER-WALLET EMPIRICAL p — under the cluster null, P(null resid_vs_base >=
     observed) for each tested wallet, then Benjamini-Hochberg. This is what
     answers "which individual wallets survive", the audit's BH-FDR claim.

A concentration guard (Project 2 section 1.5's lesson) is reported for
survivors: effective breadth 1/HHI over per-market tail bets and distinct
decision-days, so a single-event artifact cannot be read as tail skill.

READ-ONLY. Reads data/interim/bet_ledger.parquet; writes nothing.
    PYTHONPATH=. .venv/bin/python scripts/audit_blackswan_cluster.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.common import RANKED_WALLETS_PATH, load_ledger  # noqa: E402
from audit_speed_gradient import permute_wallets_within_market  # noqa: E402
from audit_blackswan import (  # noqa: E402  (reuse the audit's own primitives)
    ALPHA,
    CENT_BIN,
    MIN_TAIL_BETS,
    PRIMARY_THRESHOLD,
    SEED,
    TAIL_THRESHOLDS,
    bh_reject,
    poisson_binomial_sf,
)

N_SHUFFLES_COUNT = 300      # winner-count replications per null (matches the audit)
N_SHUFFLES_WALLET = 2000    # cluster replications for per-wallet empirical p / BH
LEDGER_COLS = ["wallet", "market_id", "side", "entry_price", "size",
               "timestamp", "resolved", "resolved_value"]


# --------------------------------------------------------------------------- #
# Data prep (mirrors audit_blackswan.prepare, but with a projected read so this #
# fits the 3.8 GB Chromebook)                                                   #
# --------------------------------------------------------------------------- #
def prepare() -> pd.DataFrame:
    bets = load_ledger(columns=LEDGER_COLS, categorical=["wallet", "market_id"])
    r = bets.loc[(bets["side"] == "BUY") & bets["resolved"]].copy()
    del bets
    n_before = len(r)
    r = r.loc[r["resolved_value"].isin((0.0, 1.0))].copy()
    r.attrs["dropped_non_binary"] = n_before - len(r)
    r["outcome"] = r["resolved_value"].to_numpy(dtype=float)
    r["cent"] = np.floor(r["entry_price"] / CENT_BIN).astype(int)
    bin_rate = r.groupby("cent")["outcome"].mean()
    r["base_rate"] = r["cent"].map(bin_rate).to_numpy(dtype=float)
    return r


def tail_arrays(r: pd.DataFrame, threshold: float, min_bets: int):
    """Compact arrays for the tail universe at `threshold`, with an integer
    wallet id for tested wallets (>= min_bets tail bets) and -1 for the rest,
    plus integer market codes for the cluster permutation."""
    tail = r.loc[r["entry_price"] <= threshold,
                 ["wallet", "market_id", "entry_price", "outcome", "base_rate",
                  "cent", "size", "timestamp"]].copy()
    counts = tail.groupby("wallet", observed=True).size()
    tested = counts.index[counts >= min_bets]
    wid_map = {w: i for i, w in enumerate(tested)}
    a = {
        "wid": tail["wallet"].map(wid_map).fillna(-1).astype(np.int64).to_numpy(),
        "mcode": pd.factorize(tail["market_id"], sort=False)[0].astype(np.int64),
        "outcome": tail["outcome"].to_numpy(dtype=float),
        "base": tail["base_rate"].to_numpy(dtype=float),
        "cent": tail["cent"].to_numpy(),
        "tested": list(tested),
        "frame": tail,
    }
    a["n_tested"] = len(tested)
    a["n_list"] = np.bincount(a["wid"][a["wid"] >= 0], minlength=a["n_tested"])
    return a


# --------------------------------------------------------------------------- #
# Winner scoring shared by both nulls                                          #
# --------------------------------------------------------------------------- #
def pb_sf_from_probs(probs: np.ndarray) -> np.ndarray:
    """Poisson-binomial survival function, computed by convolving one BINOMIAL
    pmf per distinct probability instead of one Bernoulli per bet. Identical
    result, but the tail's prices are cent-binned so a wallet with thousands of
    bets has only ~15 distinct base rates — this is what makes re-deriving the
    exact PB null inside a shuffle loop affordable."""
    vals, cnts = np.unique(probs, return_counts=True)
    pmf = np.array([1.0])
    for p, k in zip(vals, cnts):
        j = np.arange(k + 1)
        # binomial pmf via logs (k reaches thousands; direct products underflow)
        logp = (gammaln(k + 1) - gammaln(j + 1) - gammaln(k - j + 1)
                + j * np.log(max(p, 1e-300)) + (k - j) * np.log(max(1 - p, 1e-300)))
        pmf = np.convolve(pmf, np.exp(logp))
    return np.cumsum(pmf[::-1])[::-1]


def real_wallet_stats(r: pd.DataFrame, threshold: float, min_bets: int) -> pd.DataFrame:
    """Per-wallet real tail metrics — identical to audit_blackswan.tail_wallet_stats
    except the exact Poisson-binomial uses the binomial-per-distinct-price
    convolution above (same distribution, affordable at n in the thousands; the
    audit's Bernoulli-per-bet version is O(n^2) and this ledger's biggest tail
    wallet now has 5,792 tail bets). Equality is asserted in check_pb_equivalence."""
    tail = r.loc[r["entry_price"] <= threshold]
    rows = []
    for wallet, g in tail.groupby("wallet", observed=True):
        n = len(g)
        if n < min_bets:
            continue
        outcome = g["outcome"].to_numpy(dtype=float)
        base = g["base_rate"].to_numpy(dtype=float)
        price = g["entry_price"].to_numpy(dtype=float)
        size = g["size"].to_numpy(dtype=float)
        hits = int(outcome.sum())
        cost = float((size * price).sum())
        pnl = float((size * (outcome - price)).sum())
        sf = pb_sf_from_probs(base)
        rows.append({
            "wallet": wallet,
            "n_tail": n,
            "hits": hits,
            "hit_rate": hits / n,
            "mean_price": float(price.mean()),
            "base_rate": float(base.mean()),
            "resid_vs_price": hits / n - float(price.mean()),
            "resid_vs_base": hits / n - float(base.mean()),
            "pnl_usdc": pnl,
            "roi": pnl / cost if cost > 0 else np.nan,
            "pb_p": float(sf[hits]) if hits < sf.size else 0.0,
            "breadth": int(g["market_id"].nunique()),
        })
    return pd.DataFrame(rows)


def check_pb_equivalence(rng) -> None:
    """The fast PB must equal the audit's Bernoulli-convolution PB exactly."""
    for n in (20, 57, 143):
        probs = np.round(rng.uniform(0.005, 0.2, size=n), 4)
        a = poisson_binomial_sf(probs)
        b = pb_sf_from_probs(probs)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-10), (n, a[:5], b[:5])
    print("  [check] fast Poisson-binomial == audit's Bernoulli convolution "
          "(n=20/57/143, atol 1e-10) ✓")


def count_winners(wid: np.ndarray, outcome: np.ndarray, base: np.ndarray,
                  n_tested: int, alpha: float = ALPHA) -> int:
    """Number of wallets clearing the audit's winner gate (resid_vs_base>0 AND
    exact Poisson-binomial p<alpha) for an arbitrary (wid, outcome, base)
    assignment. `base` is passed explicitly because the cluster null resamples a
    wallet's prices, so its PB null is NOT a fixed precomputable constant."""
    m = wid >= 0
    w, o, b = wid[m], outcome[m], base[m]
    order = np.argsort(w, kind="stable")
    w, o, b = w[order], o[order], b[order]
    starts = np.searchsorted(w, np.arange(n_tested), side="left")
    ends = np.searchsorted(w, np.arange(n_tested), side="right")
    winners = 0
    for i in range(n_tested):
        s, e = starts[i], ends[i]
        if e <= s:
            continue
        bi = b[s:e]
        hits = int(o[s:e].sum())
        if hits / (e - s) <= bi.mean():
            continue
        sf = pb_sf_from_probs(bi)
        p = sf[hits] if hits < sf.size else 0.0
        if p < alpha:
            winners += 1
    return winners


def wallet_resid(wid: np.ndarray, outcome: np.ndarray, base: np.ndarray,
                 n_tested: int, n_list: np.ndarray) -> np.ndarray:
    """Per-wallet resid_vs_base = mean(outcome) - mean(base), vectorized."""
    m = wid >= 0
    w = wid[m]
    hits = np.bincount(w, weights=outcome[m], minlength=n_tested)
    bsum = np.bincount(w, weights=base[m], minlength=n_tested)
    n = np.bincount(w, minlength=n_tested).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (hits - bsum) / n


# --------------------------------------------------------------------------- #
# The two nulls                                                                #
# --------------------------------------------------------------------------- #
def shuffle_outcomes_in_price_bins(outcome, cent, rng):
    """Null A: permute outcomes within each 1c price bin (audit_blackswan's)."""
    shuf = outcome.copy()
    for c in np.unique(cent):
        grp = np.where(cent == c)[0]
        shuf[grp] = outcome[rng.permutation(grp)]
    return shuf


def null_A_counts(a, rng, n_shuffles):
    counts = np.empty(n_shuffles, dtype=int)
    for s in range(n_shuffles):
        shuf = shuffle_outcomes_in_price_bins(a["outcome"], a["cent"], rng)
        counts[s] = count_winners(a["wid"], shuf, a["base"], a["n_tested"])
    return counts


def null_B_counts_and_resid(a, rng, n_count, n_wallet):
    """Null B (cluster-preserving). Returns (winner counts over the first
    n_count shuffles, per-wallet null resid matrix over n_wallet shuffles).
    Both come from the same permutation stream; the PB winner gate is only
    scored on the first n_count because it is the expensive part."""
    counts = np.empty(n_count, dtype=int)
    resid_null = np.empty((n_wallet, a["n_tested"]), dtype=float)
    n_total = max(n_count, n_wallet)
    t0 = time.time()
    for s in range(n_total):
        wid_p = permute_wallets_within_market(a["wid"], a["mcode"], rng)
        # NOTE: prices/outcomes stay welded to their bets; only `wid` moves, so
        # each permuted wallet gets the base rates of the bets it now owns.
        if s < n_wallet:
            resid_null[s] = wallet_resid(wid_p, a["outcome"], a["base"],
                                         a["n_tested"], a["n_list"])
        if s < n_count:
            counts[s] = count_winners(wid_p, a["outcome"], a["base"], a["n_tested"])
        if s == 9:
            print(f"    [{(time.time()-t0)/10:.2f}s/shuffle; "
                  f"~{(time.time()-t0)/10*n_total/60:.1f} min for {n_total}]",
                  flush=True)
    return counts, resid_null


def market_cluster_bootstrap(a, stats, rng, n_boot=2000):
    """NULL C — the market-CLUSTERED bootstrap, and the bracket that actually
    matches the tail-edge claim.

    Null B is conservative for a specific structural reason: permuting wallet
    labels within a market preserves each wallet's per-market bet COUNT, so a
    wallet stays in exactly the markets it chose and only the bets it holds
    inside them can move. Every bit of "I picked the longshot markets that came
    in" — which is precisely what black-swan skill is supposed to be — is handed
    to the null. And a wallet that is the only tail bettor in its markets is
    frozen outright: its null equals its observation and the test is vacuous.

    So this bracket keeps the clustering correction but not the choice penalty:
    resample the wallet's MARKETS with replacement (whole market blocks, bets
    inside a market kept together, so one resolution event counts once, not once
    per bet), and ask whether resid_vs_base stays > 0. This is the standard
    cluster-robust answer to "bets sharing a resolution event inflate n".

    Returns (p, n_markets, sd) per tested wallet, where p = P(bootstrap resid
    <= 0) and sd is the bootstrap SD used for the design-effect diagnostic."""
    f = a["frame"]
    f = f.assign(_wid=a["wid"])
    tested = f.loc[f["_wid"] >= 0]
    out = np.full(a["n_tested"], np.nan)
    sd = np.full(a["n_tested"], np.nan)
    n_mkts = np.zeros(a["n_tested"], dtype=int)
    for wid, g in tested.groupby("_wid", observed=True, sort=False):
        blk = g.groupby("market_id", observed=True).agg(
            o=("outcome", "sum"), b=("base_rate", "sum"), n=("outcome", "size"))
        o = blk["o"].to_numpy(float)
        b = blk["b"].to_numpy(float)
        n = blk["n"].to_numpy(float)
        k = o.size
        n_mkts[wid] = k
        idx = rng.integers(0, k, size=(n_boot, k))
        num = o[idx].sum(axis=1) - b[idx].sum(axis=1)
        den = n[idx].sum(axis=1)
        stat = np.where(den > 0, num / den, np.nan)
        out[wid] = (np.sum(stat <= 0) + 1.0) / (n_boot + 1.0)
        sd[wid] = np.nanstd(stat)
    return out, n_mkts, sd


def pb_sd(a) -> np.ndarray:
    """Per-wallet SD of resid_vs_base under the INDEPENDENCE (Poisson-binomial)
    model — the reference the audit's p-values are built on. Comparing each
    null's SD against this gives the design effect: how much the correlation of
    bets sharing a resolution event inflates (or, for an over-constrained
    permutation, deflates) the true sampling variance."""
    f = a["frame"].assign(_wid=a["wid"])
    t = f.loc[f["_wid"] >= 0]
    out = np.full(a["n_tested"], np.nan)
    for wid, g in t.groupby("_wid", observed=True, sort=False):
        p = g["base_rate"].to_numpy(float)
        out[wid] = np.sqrt((p * (1 - p)).sum()) / p.size
    return out


def design_effects(sd_cluster, sd_boot, sd_indep) -> None:
    """Var(null) / Var(independence), per wallet, for both cluster brackets.

    D > 1 means the independence model understates sampling variance, so the
    audit's Poisson-binomial p-values are too small by ~sqrt(D). D < 1 means the
    null is MORE constrained than independence — an over-restricted permutation,
    whose p-values are too small for the opposite reason and must not be trusted
    at the wallet level."""
    for name, sd in (("B  cluster-preserving permutation", sd_cluster),
                     ("C  market-block bootstrap", sd_boot)):
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (sd / sd_indep) ** 2
        d = d[np.isfinite(d)]
        print(f"  {name}: median D={np.median(d):.2f}  "
              f"(10th {np.percentile(d,10):.2f} / 90th {np.percentile(d,90):.2f})  "
              f"share D>1: {100*(d>1).mean():.0f}%  D>2: {100*(d>2).mean():.0f}%")


def split_half_rho(wid, ts, outcome, base, n_tested, min_half):
    """Split-half tail persistence: Spearman between each wallet's first- and
    second-half mean resid_vs_base, chronologically split. This is the audit's
    OTHER positive claim (+0.218, cited in docs/project3_slow_markets.md §5.3 as
    beating the mean-edge pipeline's +0.105), so it needs the same re-arbitration.

    Vectorized via cumulative sums over a (wallet, time)-sorted view so it can be
    re-scored inside a shuffle loop."""
    m = wid >= 0
    w, t, o, b = wid[m], ts[m], outcome[m], base[m]
    order = np.lexsort((t, w))
    w, o, b = w[order], o[order], b[order]
    starts = np.searchsorted(w, np.arange(n_tested), side="left")
    ends = np.searchsorted(w, np.arange(n_tested), side="right")
    n = ends - starts
    k = n // 2                       # first half = the older k bets
    ok = (k >= min_half) & (n - k >= min_half)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    co = np.concatenate([[0.0], np.cumsum(o)])
    cb = np.concatenate([[0.0], np.cumsum(b)])
    s, e, kk = starts[ok], ends[ok], k[ok]
    h1 = ((co[s + kk] - co[s]) - (cb[s + kk] - cb[s])) / kk
    h2 = ((co[e] - co[s + kk]) - (cb[e] - cb[s + kk])) / (e - s - kk)
    r1 = pd.Series(h1).rank().to_numpy()
    r2 = pd.Series(h2).rank().to_numpy()
    rho = np.corrcoef(r1, r2)[0, 1]
    return float(rho), int(ok.sum())


def persistence_vs_nulls(a, rng, n_shuffles, min_half):
    """Real split-half rho, plus its distribution under both nulls."""
    ts = a["frame"]["timestamp"].to_numpy(dtype=float)
    real, n_qual = split_half_rho(a["wid"], ts, a["outcome"], a["base"],
                                  a["n_tested"], min_half)
    nulls = {"A": [], "B": []}
    for _ in range(n_shuffles):
        shuf = shuffle_outcomes_in_price_bins(a["outcome"], a["cent"], rng)
        rA, _ = split_half_rho(a["wid"], ts, shuf, a["base"], a["n_tested"], min_half)
        wid_p = permute_wallets_within_market(a["wid"], a["mcode"], rng)
        rB, _ = split_half_rho(wid_p, ts, a["outcome"], a["base"],
                               a["n_tested"], min_half)
        if np.isfinite(rA):
            nulls["A"].append(rA)
        if np.isfinite(rB):
            nulls["B"].append(rB)
    return real, n_qual, np.array(nulls["A"]), np.array(nulls["B"])


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #
def wallet_freedom(a) -> np.ndarray:
    """Per tested wallet: the share of its tail bets sitting in markets where at
    least one OTHER wallet also has a tail bet. Bets outside those markets cannot
    move under null B, so a wallet at ~0 here is untestable by that null rather
    than cleared or rejected by it — the difference has to be visible."""
    f = a["frame"].assign(_wid=a["wid"])
    nw = f.groupby("market_id", observed=True)["wallet"].nunique()
    f["_shared"] = f["market_id"].map(nw).to_numpy() >= 2
    t = f.loc[f["_wid"] >= 0]
    num = np.bincount(t["_wid"].to_numpy(), weights=t["_shared"].to_numpy(float),
                      minlength=a["n_tested"])
    den = np.bincount(t["_wid"].to_numpy(), minlength=a["n_tested"]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def permutation_freedom(a) -> None:
    """How much can the cluster null actually move? Bets in single-bet markets,
    or in markets where one wallet holds every tail bet, are frozen — a null
    with little freedom is conservative for a degenerate reason, and that has to
    be visible rather than assumed away."""
    f = a["frame"]
    g = f.groupby("market_id", observed=True)
    sz = g.size()
    nw = g["wallet"].nunique()
    n = len(f)
    movable = sz[nw >= 2].sum()
    print(f"  tail bets: {n:,} in {sz.size:,} markets "
          f"(median {sz.median():.0f}/market)")
    print(f"  in markets with >=2 tail bets:            {sz[sz>=2].sum()/n:6.1%}")
    print(f"  in markets with >=2 DISTINCT wallets:     {movable/n:6.1%}  "
          "<- the permutable fraction")


def concentration(frame: pd.DataFrame, wallets) -> pd.DataFrame:
    """Project 2 section 1.5's guard: effective breadth 1/HHI over per-market
    tail bets, plus distinct UTC decision-days."""
    rows = []
    sub = frame.loc[frame["wallet"].isin(wallets)]
    for w, g in sub.groupby("wallet", observed=True):
        share = g.groupby("market_id", observed=True).size().to_numpy(float)
        share /= share.sum()
        days = pd.to_datetime(g["timestamp"], unit="s", utc=True).dt.date.nunique()
        rows.append({"wallet": w, "eff_breadth": 1.0 / np.square(share).sum(),
                     "decision_days": int(days)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-shuffles", type=int, default=N_SHUFFLES_COUNT)
    ap.add_argument("--wallet-shuffles", type=int, default=N_SHUFFLES_WALLET)
    ap.add_argument("--persist-shuffles", type=int, default=200)
    ap.add_argument("--thresholds", type=float, nargs="*", default=list(TAIL_THRESHOLDS))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    check_pb_equivalence(np.random.default_rng(7))
    r = prepare()

    print("=" * 84)
    print("BLACK-SWAN TAIL-EDGE — CLUSTER-PRESERVING NULL RE-RUN  (read-only)")
    print("=" * 84)
    print(f"resolved BUY bets (clean 0/1): {len(r):,}  "
          f"(dropped {r.attrs['dropped_non_binary']:,} non-binary)")
    print(f"distinct wallets: {r['wallet'].nunique():,}")
    print("NOTE: today's ledger is larger than the 2026-07-19 run, so BOTH nulls are")
    print("      re-measured here; the historic '19 clear / 11 BH / null 3.1' is not")
    print("      directly comparable to the real counts below.")

    # -- 1. Winner funnel: real vs BOTH nulls, across thresholds --------------
    print(f"\n--- 1. Tail-winner funnel vs both nulls "
          f"(min {MIN_TAIL_BETS} tail bets, alpha={ALPHA}, "
          f"{args.count_shuffles} shuffles) ---")
    primary = None
    rows = []
    for thr in args.thresholds:
        stats = real_wallet_stats(r, thr, MIN_TAIL_BETS)
        if stats.empty:
            continue
        a = tail_arrays(r, thr, MIN_TAIL_BETS)
        sig = int(((stats["resid_vs_base"] > 0) & (stats["pb_p"] < ALPHA)).sum())
        bh_an = int(((stats["resid_vs_base"] > 0).to_numpy()
                     & bh_reject(stats["pb_p"].to_numpy(), ALPHA)).sum())
        print(f"  thr={thr:.2f}: {a['n_tested']} tested wallets, "
              f"{len(a['frame']):,} tail bets — scoring nulls…", flush=True)
        cA = null_A_counts(a, rng, args.count_shuffles)
        nW = args.wallet_shuffles if thr == PRIMARY_THRESHOLD else 0
        cB, residB = null_B_counts_and_resid(a, rng, args.count_shuffles, nW)
        rows.append({
            "thr": thr, "tested": a["n_tested"], "real_sig": sig, "real_BH": bh_an,
            "A_mean": cA.mean(), "A_p95": np.percentile(cA, 95), "A_max": cA.max(),
            "A_pval": float((cA >= sig).mean()),
            "B_mean": cB.mean(), "B_p95": np.percentile(cB, 95), "B_max": cB.max(),
            "B_pval": float((cB >= sig).mean()),
        })
        if thr == PRIMARY_THRESHOLD:
            primary = (stats, a, residB, cA, cB)

    tab = pd.DataFrame(rows)
    print()
    print(tab.round(3).to_string(index=False))
    print("  real_sig = wallets with resid_vs_base>0 & Poisson-binomial p<alpha")
    print("  A_* = bet-level price-bin null (permissive) | B_* = cluster-preserving "
          "null (conservative)")
    print("  *_pval = P(null winner count >= real). >=0.05 means the winner COUNT is "
          "indistinguishable from that null.")

    if primary is None:
        print("\nNo primary-threshold stats. Done.")
        return
    stats, a, resid_null, cA, cB = primary

    print(f"\n--- 2. Cluster-null permutation freedom at thr={PRIMARY_THRESHOLD} ---")
    permutation_freedom(a)

    # -- 3. Per-wallet empirical p under the cluster null, then BH ------------
    print(f"\n--- 3. Per-wallet verdict at thr={PRIMARY_THRESHOLD} "
          f"({args.wallet_shuffles} cluster shuffles) ---")
    obs = np.array([stats.set_index("wallet").loc[w, "resid_vs_base"]
                    for w in a["tested"]], dtype=float)
    p_emp = (np.sum(resid_null >= obs[None, :], axis=0) + 1.0) / (resid_null.shape[0] + 1.0)
    res = pd.DataFrame({"wallet": a["tested"], "resid_vs_base": obs,
                        "p_cluster": p_emp})
    res = res.merge(stats[["wallet", "n_tail", "hits", "hit_rate", "base_rate",
                           "roi", "pnl_usdc", "pb_p", "breadth"]], on="wallet")
    # null spread per wallet + how much of its record the permutation can move:
    # a wallet with ~zero freedom is UNTESTED by null B, not cleared by it.
    res["null_sd"] = resid_null.std(axis=0)
    res["null_mean"] = resid_null.mean(axis=0)
    res["movable"] = wallet_freedom(a)
    p_boot, n_mkts, sd_boot = market_cluster_bootstrap(a, stats, rng)
    res["p_mktboot"] = p_boot
    res["n_markets"] = n_mkts
    res["bh_cluster"] = bh_reject(res["p_cluster"].to_numpy(), ALPHA) & (res["resid_vs_base"] > 0)
    res["bh_pb"] = bh_reject(res["pb_p"].to_numpy(), ALPHA) & (res["resid_vs_base"] > 0)
    res["bh_mktboot"] = bh_reject(res["p_mktboot"].to_numpy(), ALPHA) & (res["resid_vs_base"] > 0)

    n_sig_pb = int(((res["resid_vs_base"] > 0) & (res["pb_p"] < ALPHA)).sum())
    n_sig_cl = int(((res["resid_vs_base"] > 0) & (res["p_cluster"] < ALPHA)).sum())
    n_sig_mb = int(((res["resid_vs_base"] > 0) & (res["p_mktboot"] < ALPHA)).sum())
    print(f"  tested wallets:                                     {len(res)}")
    print(f"  A  clear at p<{ALPHA}, bet-level Poisson-binomial:   {n_sig_pb}")
    print(f"  B  clear at p<{ALPHA}, cluster-preserving perm:      {n_sig_cl}")
    print(f"  C  clear at p<{ALPHA}, market-clustered bootstrap:   {n_sig_mb}")
    print(f"  A  BH-FDR q={ALPHA}:  {int(res['bh_pb'].sum()):>3d}   <- the audit's bracket "
          "(11 historically, on a smaller ledger)")
    print(f"  B  BH-FDR q={ALPHA}:  {int(res['bh_cluster'].sum()):>3d}   <- cluster-preserving "
          "re-run (no credit for market choice)")
    print(f"  C  BH-FDR q={ALPHA}:  {int(res['bh_mktboot'].sum()):>3d}   <- clustering "
          "corrected, market choice still credited")
    print(f"  p floor: null B {1/(args.wallet_shuffles+1):.2e} / null C 5.00e-04; "
          f"BH needs <= {ALPHA/len(res):.2e} for the smallest of {len(res)}")

    print("\n  Design effect vs the independence (Poisson-binomial) model:")
    design_effects(res["null_sd"].to_numpy(), sd_boot, pb_sd(a))
    print("    D<1 => the null is more constrained than independence: its p-values are")
    print("    too small and it CANNOT arbitrate individual wallets. D>1 => it correctly")
    print("    charges for bets that share a resolution event.")

    print("\n  Power of null B (can it move the wallet at all?):")
    frozen = res["movable"] < 0.05
    lowsd = res["null_sd"] < 0.2 * res["resid_vs_base"].abs().clip(lower=1e-9)
    print(f"    wallets with <5% of tail bets in shared markets (frozen): "
          f"{int(frozen.sum())} of {len(res)}")
    print(f"    median movable share: {res['movable'].median():.1%}; "
          f"median null sd of resid: {res['null_sd'].median():.4f}")
    print(f"    wallets whose null sd < 20% of |observed resid| (near-vacuous test): "
          f"{int(lowsd.sum())}")
    print(f"    mean null resid across wallets: {res['null_mean'].mean():+.4f} vs "
          f"mean observed {res['resid_vs_base'].mean():+.4f}  <- if these are close, "
          "null B\n      is absorbing the wallet's record into its own markets, which is "
          "what it is designed to do")

    surv = res.loc[res["bh_cluster"]].sort_values("p_cluster")
    print("\n  Cluster-null BH-FDR survivors:")
    if surv.empty:
        print("    NONE.")
    else:
        conc = concentration(a["frame"], set(surv["wallet"]))
        surv = surv.merge(conc, on="wallet")
        show = surv.copy()
        show["wallet"] = show["wallet"].astype(str).str[:12] + "…"
        for c in ("resid_vs_base", "hit_rate", "base_rate", "roi", "eff_breadth"):
            show[c] = show[c].round(3)
        show["pb_p"] = show["pb_p"].map(lambda x: f"{x:.1e}")
        show["p_cluster"] = show["p_cluster"].map(lambda x: f"{x:.1e}")
        print(show[["wallet", "n_tail", "hits", "hit_rate", "base_rate",
                    "resid_vs_base", "roi", "pb_p", "p_cluster", "breadth",
                    "eff_breadth", "decision_days"]].to_string(index=False))
        print("  concentration guard (Project 2 §1.5): credible needs eff_breadth>=3 "
              "AND decision_days>=3")

    # -- 3b. Null C survivors, with the concentration guard -------------------
    survC = res.loc[res["bh_mktboot"]].sort_values("p_mktboot")
    print(f"\n  Null-C (market-clustered bootstrap) BH survivors: {len(survC)}")
    if not survC.empty:
        conc = concentration(a["frame"], set(survC["wallet"]))
        s = survC.merge(conc, on="wallet").head(30).copy()
        s["credible"] = np.where((s["eff_breadth"] >= 3) & (s["decision_days"] >= 3),
                                 "", "CONCENTRATED")
        s["wallet"] = s["wallet"].astype(str).str[:12] + "…"
        for c in ("resid_vs_base", "hit_rate", "base_rate", "roi", "eff_breadth",
                  "movable"):
            s[c] = s[c].round(3)
        for c in ("pb_p", "p_mktboot", "p_cluster"):
            s[c] = s[c].map(lambda x: f"{x:.1e}")
        print(s[["wallet", "n_tail", "hits", "hit_rate", "base_rate", "resid_vs_base",
                 "roi", "n_markets", "eff_breadth", "decision_days", "movable",
                 "pb_p", "p_mktboot", "p_cluster", "credible"]].to_string(index=False))
        n_cred = int(((survC.merge(conc, on="wallet")["eff_breadth"] >= 3)
                      & (survC.merge(conc, on="wallet")["decision_days"] >= 3)).sum())
        print(f"  passing the §1.5 concentration guard too (eff_breadth>=3 & "
              f"decision_days>=3): {n_cred}")

    print("\n  Three-way agreement (BH survivors):")
    print(f"    A only:      {int((res['bh_pb'] & ~res['bh_cluster'] & ~res['bh_mktboot']).sum())}")
    print(f"    A and C:     {int((res['bh_pb'] & res['bh_mktboot'] & ~res['bh_cluster']).sum())}")
    print(f"    A, B and C:  {int((res['bh_pb'] & res['bh_cluster'] & res['bh_mktboot']).sum())}")
    print(f"    B (any):     {int(res['bh_cluster'].sum())}")

    # -- 4. What the permissive null certified but the conservative one does not
    lost = res.loc[res["bh_pb"] & ~res["bh_cluster"]].sort_values("pb_p")
    print(f"\n--- 4. Certified by the bet-level null, NOT by the cluster null: "
          f"{len(lost)} ---")
    if not lost.empty:
        show = lost.head(25).copy()
        show["wallet"] = show["wallet"].astype(str).str[:12] + "…"
        for c in ("resid_vs_base", "hit_rate", "base_rate", "roi"):
            show[c] = show[c].round(3)
        show["pb_p"] = show["pb_p"].map(lambda x: f"{x:.1e}")
        show["p_cluster"] = show["p_cluster"].map(lambda x: f"{x:.2f}")
        show["p_mktboot"] = show["p_mktboot"].map(lambda x: f"{x:.1e}")
        for c in ("movable",):
            show[c] = show[c].round(2)
        print(show[["wallet", "n_tail", "hits", "hit_rate", "base_rate",
                    "resid_vs_base", "roi", "pb_p", "p_cluster", "p_mktboot",
                    "movable", "breadth"]].to_string(index=False))

    # -- 4b. Split-half tail persistence vs both nulls ------------------------
    print(f"\n--- 4b. Split-half tail persistence (the audit's +0.218 claim) ---")
    mh = MIN_TAIL_BETS // 2
    real_rho, n_qual, nA, nB = persistence_vs_nulls(a, rng, args.persist_shuffles, mh)
    def _p(obs, null):
        return np.nan if not null.size else (np.sum(null >= obs) + 1) / (null.size + 1)
    print(f"  qualifying wallets (>= {mh} tail bets per half): {n_qual}")
    print(f"  observed Spearman(first-half resid, second-half resid): {real_rho:+.3f}")
    print(f"  null A (bet-level):        mean {nA.mean():+.3f}  95th {np.percentile(nA,95):+.3f}"
          f"  p={_p(real_rho, nA):.3f}")
    print(f"  null B (cluster-preserving): mean {nB.mean():+.3f}  95th {np.percentile(nB,95):+.3f}"
          f"  p={_p(real_rho, nB):.3f}")
    print("  (p = P(null rho >= observed); a null centred well above 0 means most of the")
    print("   persistence is wallets repeatedly trading the same markets, not tail skill.)")

    # -- 5. Cross-reference against the mean-edge pipeline --------------------
    print("\n--- 5. Cross-reference vs the mean-edge ranking ---")
    if not RANKED_WALLETS_PATH.exists():
        print("  (ranked_wallets.parquet not found)")
    else:
        rk = pd.read_parquet(RANKED_WALLETS_PATH)
        if "rank" not in rk.columns:
            rk = rk.reset_index(drop=True)
            rk["rank"] = np.arange(1, len(rk) + 1)
        for label, sel in (("A bet-level BH survivors", res["bh_pb"]),
                           ("B cluster-perm BH survivors", res["bh_cluster"]),
                           ("C market-bootstrap BH survivors", res["bh_mktboot"])):
            s = res.loc[sel, ["wallet", "n_tail", "roi"]].copy()
            s["wallet"] = s["wallet"].astype(str)
            m = s.merge(rk[["wallet", "rank", "edge_persisted"]], on="wallet", how="left")
            m["edge_persisted"] = m["edge_persisted"].fillna(False)
            print(f"  {label}: {len(m)} | already edge_persisted: "
                  f"{int(m['edge_persisted'].sum())} | missed by the pipeline: "
                  f"{int((~m['edge_persisted']).sum())}")

    print("\n" + "=" * 84)
    print("HOW TO READ THIS (measured, not assumed — see the design-effect note below).")
    print("Null A (bet-level) assumes bets are independent; they are not, so it is too")
    print("permissive. Null B (permute wallet within market) is the right idea but is")
    print("OVER-CONSTRAINED in this sparse tail: tail markets hold a median of 2 tail")
    print("bets, so most bets are frozen and its per-wallet reference is NARROWER than")
    print("the independence model (median design effect 0.17) — it manufactures its own")
    print("per-wallet false positives and must not be used as a wallet-level arbiter")
    print("here. Its COUNT-level comparison is still informative.")
    print("Null C (market-block bootstrap) is the per-wallet arbiter: it keeps the")
    print("wallet's own bets and choices and resamples whole markets, so one resolution")
    print("event counts once (median design effect 1.45, 40% of wallets >2).")
    print("=" * 84)


if __name__ == "__main__":
    main()
