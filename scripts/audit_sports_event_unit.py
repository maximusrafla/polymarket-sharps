"""Re-run the sports gate at the RESOLUTION-EVENT unit — Project 3 sports step 2.

THE QUESTION
------------
The gate opened the sports arm on two numbers, both computed with `market_id` as
the cluster: ~8.4x excess cross-wallet skill dispersion and rho=+0.48 split-half
persistence, each clearing a cluster-preserving null at the p-floor. `market_id`
is not the resolution unit in that corpus (see src/sports_events.py: 284 markets
fold to ~25 events), so this re-runs both statistics with the cluster unit
corrected and reports them side by side.

WHAT IS MEASURED, AT BOTH UNITS
-------------------------------
  1. DESIGN EFFECT — the repo rule (HANDOFF, "measure the design effect before
     trusting any null"): var of a wallet's mean under cluster resampling over
     var under independence. A unit whose DE is ~1 is not doing any work; a
     permutation null in a stratum where DE < 1 is over-constrained and
     manufactures its own false positives.
  2. DISPERSION — variance of per-wallet mean residual skill across the findable
     set, as a ratio to the same quantity under a cluster-preserving null. This
     is the gate's 8.4x.
  3. SPLIT-HALF PERSISTENCE — Spearman between a wallet's mean residual in its
     chronological first and second half. This is the gate's +0.48.
  4. BETWEEN-EVENT PERSISTENCE — the honest forward question, and the one the
     split-half statistic cannot answer in a corpus where 79% of wallets span
     <=2 events: does a wallet's skill in ONE event predict its skill in a
     DIFFERENT one? Split each wallet's events (not its bets) in two.

THE NULL
--------
`permute_wallets_within_market` from audit_speed_gradient, applied at the chosen
cluster unit: every cluster keeps its exact bets, prices and outcomes; only
attribution changes, so the correlation from sharing one resolution survives and
all wallet skill is destroyed. Its stated limitation applies here with force at
the event unit — holding the event fixed also removes the skill of CHOOSING
which team in the field to back. That makes it the conservative bracket for
"is there wallet skill", which is the right direction for a gate re-check, and
the design effect above is what says whether it is TOO conservative.

Residuals use a population (non-LOWO) hierarchical baseline at league and
league|form so that a bet's residual is a fixed property of its price, league
and outcome — unchanged under wallet permutation, which is what the null needs.
A wallet heavy enough to move its own baseline has its residual shrunk toward
zero, which is conservative for finding skill. The LOWO variant is reported
alongside as a sensitivity.

READ-ONLY: no network, no writes to the shared ledger. Reads the isolated gate
corpus copy under data/interim/sports/gate/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_speed_gradient import permute_wallets_within_market  # noqa: E402

from src.sports_deepen import GATE_DIR, load_gate_corpus  # noqa: E402

OUT_PATH = GATE_DIR / "event_unit_audit.parquet"

SEED = 12345
MIN_BETS = 20          # the gate's "findable" rule
MIN_PER_HALF = 10
N_SHUFFLES = 200


# --------------------------------------------------------------------------- #
# Corpus                                                                       #
# --------------------------------------------------------------------------- #
def load_corpus(lowo: bool = False, refresh: bool = False) -> pd.DataFrame:
    """The gate corpus with `event` and per-bet residual skill — the SAME builder
    the shortlist screens on, imported rather than re-implemented so the audit and
    the screen can never drift apart."""
    return load_gate_corpus(lowo=lowo, refresh=refresh)


def describe_corpus(r: pd.DataFrame) -> None:
    n_mkt = r["market_id"].nunique()
    n_ev = r["event"].nunique()
    kinds = r.drop_duplicates("market_id")["event_kind"].value_counts()
    print(f"[corpus] {len(r):,} bets  {r.wallet.nunique():,} wallets  "
          f"{n_mkt:,} markets  ->  {n_ev:,} EVENTS  ({n_mkt / max(n_ev, 1):.1f}x fold)")
    print(f"[corpus] market kinds: {kinds.to_dict()}  "
          f"(unmatched share of markets: {kinds.get('market', 0) / max(n_mkt, 1):.1%})")
    top = (r.groupby("event").size().sort_values(ascending=False) / len(r)).head(5)
    print("[corpus] top events by bet share: "
          + ", ".join(f"{k}={v:.1%}" for k, v in top.items()))

    f = r.groupby("wallet").agg(n=("event", "size"), ev=("event", "nunique"),
                                mk=("market_id", "nunique"))
    f = f[f.n >= MIN_BETS]
    print(f"[corpus] findable wallets (>={MIN_BETS} bets): {len(f):,}  "
          f"median markets={f.mk.median():.0f}  median events={f.ev.median():.0f}  "
          f"share spanning 1 event={np.mean(f.ev == 1):.1%}  <=2 events="
          f"{np.mean(f.ev <= 2):.1%}")


# --------------------------------------------------------------------------- #
# Design effect                                                                #
# --------------------------------------------------------------------------- #
def design_effect(resid: np.ndarray, blocks: np.ndarray) -> float:
    """var(mean) under cluster structure / var(mean) under independence.

    Computed analytically from the block sums: for equal-probability resampling
    of whole blocks, var(block-mean estimator) = var over blocks of (block sum -
    blockmean*count) terms. DE > 1 means bets within a cluster move together and
    the independence t-test is too generous; DE < 1 means the cluster unit is
    MORE constrained than independence, the over-constraint trap."""
    n = resid.size
    if n < 2:
        return np.nan
    uniq, inv = np.unique(blocks, return_inverse=True)
    k = uniq.size
    if k < 2:
        return np.inf
    s = np.bincount(inv, weights=resid, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    mu = resid.mean()
    # cluster-robust variance of the bet-weighted mean:
    #   Var(ybar) = k/(k-1) * sum_j (s_j - ybar*c_j)^2 / n^2
    var_cluster = np.sum((s - mu * c) ** 2) / (k - 1) * (k / n ** 2)
    var_indep = resid.var(ddof=1) / n
    if var_indep <= 0:
        return np.nan
    return float(var_cluster / var_indep)


def design_effects(r: pd.DataFrame, unit: str) -> pd.Series:
    """Per-findable-wallet design effect, vectorized over the whole corpus.

    Same quantity as `design_effect` above (which stays as the readable
    reference and is what the tests pin), computed with bincounts so 233k
    wallets cost one pass instead of 233k Python-level groups."""
    wcode, wuniq = pd.factorize(r["wallet"].to_numpy())
    bcode = pd.factorize(r[unit].to_numpy())[0]
    resid = r["residual_skill"].to_numpy(dtype=float)
    n_w = wuniq.size

    n = np.bincount(wcode, minlength=n_w).astype(float)
    s = np.bincount(wcode, weights=resid, minlength=n_w)
    ss = np.bincount(wcode, weights=resid ** 2, minlength=n_w)
    mu = np.divide(s, n, out=np.zeros_like(s), where=n > 0)

    pair = pd.factorize(wcode.astype(np.int64) * (bcode.max() + 1) + bcode)[0]
    p_sum = np.bincount(pair, weights=resid)
    p_cnt = np.bincount(pair).astype(float)
    p_wallet = np.zeros(p_sum.size, dtype=np.int64)
    p_wallet[pair] = wcode                                  # each pair's wallet
    k = np.bincount(p_wallet, minlength=n_w).astype(float)  # blocks per wallet

    dev2 = (p_sum - mu[p_wallet] * p_cnt) ** 2
    block_ss = np.bincount(p_wallet, weights=dev2, minlength=n_w)

    with np.errstate(divide="ignore", invalid="ignore"):
        var_cluster = block_ss / (k - 1) * (k / n ** 2)
        var_indep = (ss - n * mu ** 2) / (n - 1) / n
        de = np.where((k >= 2) & (var_indep > 0), var_cluster / var_indep, np.nan)
    return pd.Series(de[n >= MIN_BETS], index=wuniq[n >= MIN_BETS], dtype=float)


# --------------------------------------------------------------------------- #
# The two gate statistics, vectorized over an arbitrary wallet assignment       #
# --------------------------------------------------------------------------- #
def gate_statistics(wcode: np.ndarray, ts: np.ndarray, resid: np.ndarray,
                    n_wallets: int) -> tuple[float, float, int]:
    """(dispersion, split-half Spearman, n findable) for one wallet assignment.

    Dispersion = variance across findable wallets of their mean residual.
    Split-half = Spearman between first- and second-half means for wallets with
    at least MIN_PER_HALF bets in each half."""
    o = np.lexsort((ts, wcode))
    w, rr = wcode[o], resid[o]
    counts = np.bincount(w, minlength=n_wallets)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    sums = np.bincount(w, weights=rr, minlength=n_wallets)
    findable = counts >= MIN_BETS
    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    disp = float(np.var(means[findable], ddof=1)) if findable.sum() > 1 else np.nan

    # chronological halves (floor split, matching validate.split_in_sample_out_of_sample)
    half = counts // 2
    ok = findable & (half >= MIN_PER_HALF) & (counts - half >= MIN_PER_HALF)
    idx = np.flatnonzero(ok)
    if idx.size < 3:
        return disp, np.nan, int(findable.sum())
    csum = np.concatenate([[0.0], np.cumsum(rr)])
    a_end = starts[idx] + half[idx]
    b_end = starts[idx] + counts[idx]
    first = (csum[a_end] - csum[starts[idx]]) / half[idx]
    second = (csum[b_end] - csum[a_end]) / (counts[idx] - half[idx])
    rho = float(stats.spearmanr(first, second).statistic)
    return disp, rho, int(findable.sum())


def between_event_persistence(r: pd.DataFrame, min_per_side: int = MIN_PER_HALF
                              ) -> tuple[float, int]:
    """Spearman between a wallet's mean residual in its EARLY events and its LATE
    events — events split, not bets, so the two sides never share a resolution.

    The split-half statistic cannot separate skill from one lucky event when a
    wallet's whole record is that event; this can. Wallets spanning a single
    event are excluded BY RULE (they have no second side), and how many that
    removes is itself the finding."""
    ev = (r.groupby(["wallet", "event"], sort=False)
          .agg(s=("residual_skill", "sum"), n=("residual_skill", "size"),
               t0=("timestamp", "min")).reset_index())
    tot = ev.groupby("wallet")["n"].transform("sum")
    n_ev = ev.groupby("wallet")["event"].transform("size")
    ev = ev[(tot >= MIN_BETS) & (n_ev >= 2)]
    if ev.empty:
        return np.nan, 0

    ev = ev.sort_values(["wallet", "t0"], kind="mergesort")
    rank = ev.groupby("wallet").cumcount().to_numpy()
    k = ev.groupby("wallet")["event"].transform("size").to_numpy()
    early = rank < np.maximum(k // 2, 1)

    wcode, wuniq = pd.factorize(ev["wallet"].to_numpy())
    nw = wuniq.size
    s_arr, n_arr = ev["s"].to_numpy(dtype=float), ev["n"].to_numpy(dtype=float)
    a_s = np.bincount(wcode[early], weights=s_arr[early], minlength=nw)
    a_n = np.bincount(wcode[early], weights=n_arr[early], minlength=nw)
    b_s = np.bincount(wcode[~early], weights=s_arr[~early], minlength=nw)
    b_n = np.bincount(wcode[~early], weights=n_arr[~early], minlength=nw)
    ok = (a_n >= min_per_side) & (b_n >= min_per_side)
    if int(ok.sum()) < 3:
        return np.nan, int(ok.sum())
    return (float(stats.spearmanr(a_s[ok] / a_n[ok], b_s[ok] / b_n[ok]).statistic),
            int(ok.sum()))


# --------------------------------------------------------------------------- #
def run_unit(r: pd.DataFrame, unit: str, n_shuffles: int, rng) -> dict:
    print(f"\n{'=' * 74}\nCLUSTER UNIT: {unit}\n{'=' * 74}")
    de = design_effects(r, unit)
    print(f"[design effect] median={de.median():.2f}  mean={de.mean():.2f}  "
          f"share<1={np.mean(de < 1):.1%}  (DE>1: clustering matters; "
          f"DE<1: unit is over-constrained)")

    wcodes, wcode = np.unique(r["wallet"].to_numpy(), return_inverse=True)
    mcode = pd.factorize(r[unit].to_numpy())[0]
    ts = r["timestamp"].to_numpy()
    resid = r["residual_skill"].to_numpy(dtype=float)
    n_w = wcodes.size

    obs_disp, obs_rho, n_find = gate_statistics(wcode, ts, resid, n_w)
    print(f"[observed] findable={n_find:,}  dispersion(var)={obs_disp:.6f}  "
          f"split-half rho={obs_rho:+.3f}")

    disps, rhos = [], []
    t0 = time.time()
    for i in range(n_shuffles):
        wp = permute_wallets_within_market(wcode, mcode, rng)
        d, rh, _ = gate_statistics(wp, ts, resid, n_w)
        disps.append(d)
        rhos.append(rh)
        if i == 4:
            per = (time.time() - t0) / 5
            print(f"    [{per:.2f}s/shuffle; ~{per * n_shuffles / 60:.1f} min total]",
                  flush=True)
    disps, rhos = np.array(disps), np.array(rhos)

    ratio = obs_disp / np.nanmean(disps)
    p_disp = (np.sum(disps >= obs_disp) + 1) / (n_shuffles + 1)
    p_rho = (np.sum(rhos >= obs_rho) + 1) / (n_shuffles + 1)
    print(f"[null x{n_shuffles}] dispersion mean={np.nanmean(disps):.6f}  "
          f"-> EXCESS RATIO {ratio:.2f}x   P(null>=obs)={p_disp:.4f}")
    print(f"[null x{n_shuffles}] split-half rho mean={np.nanmean(rhos):+.3f} "
          f"(sd {np.nanstd(rhos):.3f})  P(null>=obs)={p_rho:.4f}")

    return {"unit": unit, "n_findable": n_find, "design_effect_median": float(de.median()),
            "design_effect_share_lt1": float(np.mean(de < 1)),
            "dispersion": obs_disp, "dispersion_null_mean": float(np.nanmean(disps)),
            "dispersion_ratio": float(ratio), "dispersion_p": float(p_disp),
            "split_half_rho": obs_rho, "split_half_null_mean": float(np.nanmean(rhos)),
            "split_half_null_sd": float(np.nanstd(rhos)), "split_half_p": float(p_rho),
            "n_shuffles": n_shuffles}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLES)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the cached residualized corpus")
    ap.add_argument("--lowo", action="store_true",
                    help="residualize leave-one-wallet-out (sensitivity; the null "
                         "needs permutation-invariant residuals, so the default is "
                         "the population baseline)")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    r = load_corpus(lowo=args.lowo, refresh=args.refresh)
    r = r[~r["residual_skill"].isna()].reset_index(drop=True)
    describe_corpus(r)

    rows = [run_unit(r, "market_id", args.shuffles, rng),
            run_unit(r, "event", args.shuffles, rng)]

    print(f"\n{'=' * 74}\nBETWEEN-EVENT PERSISTENCE (events split, not bets)\n{'=' * 74}")
    rho_be, n_be = between_event_persistence(r)
    print(f"[observed] wallets with >=2 events and both sides populated: {n_be:,}  "
          f"rho={rho_be:+.3f}")
    be_null = []
    wcodes, wcode = np.unique(r["wallet"].to_numpy(), return_inverse=True)
    ecode = pd.factorize(r["event"].to_numpy())[0]
    for _ in range(max(20, args.shuffles // 4)):
        wp = permute_wallets_within_market(wcode, ecode, rng)
        rr = r.assign(wallet=wcodes[wp])
        rho_n, _ = between_event_persistence(rr)
        be_null.append(rho_n)
    be_null = np.array(be_null, dtype=float)
    p_be = (np.sum(be_null >= rho_be) + 1) / (be_null.size + 1)
    print(f"[null x{be_null.size}] mean rho={np.nanmean(be_null):+.3f} "
          f"(sd {np.nanstd(be_null):.3f})  P(null>=obs)={p_be:.4f}")
    rows.append({"unit": "between_event", "n_findable": n_be,
                 "split_half_rho": rho_be, "split_half_null_mean": float(np.nanmean(be_null)),
                 "split_half_null_sd": float(np.nanstd(be_null)),
                 "split_half_p": float(p_be), "n_shuffles": int(be_null.size)})

    out = pd.DataFrame(rows)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, compression="gzip", index=False)
    print(f"\n[audit] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
