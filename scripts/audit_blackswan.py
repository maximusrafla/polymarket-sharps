"""Black-swan / tail-edge detector — a DIFFERENT lens on the ledger from the
mean-edge + t-test machinery in features/validate/rank.

WHY A DIFFERENT LENS (the premise). The pipeline ranks on mean skill (residual)
edge and validates it with a one-sided t-test. That machinery *structurally
misses* wallets whose skill is buying underpriced longshots: their +EV lives in
fat-tail variance (a few 30x hits among many total losses), so their per-bet
mean edge reads ~0 or negative and a t-test — whose standard error a couple of
huge outcomes dominate — cannot separate genuine tail-skill from luck without an
enormous sample. This script asks the tail question directly, with tools built
for rare events instead of means.

THE LENS (four pieces):
  1. Tail CALIBRATION, not mean edge. For low-priced bets (entry <= threshold),
     does a wallet's realized hit-rate beat (a) the price it paid — the implied
     probability — and (b) the market-wide base rate E[outcome | entry_price]?
     Beating the base rate is the strong signal: it is favorite-longshot-neutral
     tail skill (buying longshots the market itself underprices), the residual-
     edge idea from features.py restricted to the tail.
  2. GROWTH, not mean. Total realized return (dollar P&L) and ROI — which a few
     30x hits rightly dominate — plus a Kelly-style log-growth rate (the KL
     divergence D(q||p), the log-optimal bankroll growth of a bettor who forecasts
     at the wallet's demonstrated tail accuracy q against price p).
  3. VARIANCE-AWARE significance. Hit-rate vs. a heterogeneous base rate is tested
     with an exact Poisson-binomial tail probability (the correct model for a sum
     of Bernoullis with per-bet probabilities — not a t-test on a mean). Growth/ROI,
     where the fat tail really does dominate the SE, is tested by BOOTSTRAP.
  4. NULL-FIRST posture. With only a few hundred wallets holding tail bets and a
     handful of hits each, chance alone manufactures "winners." So every claimed
     winner is assumed to be luck until a shuffled-outcome null says otherwise:
     we shuffle resolved outcomes *within fine price buckets* (preserving the base
     rate and each wallet's price mix, destroying only the wallet<->outcome link)
     and re-run the whole detector K times, calibrating how many winners the
     procedure invents on pure noise. Benjamini-Hochberg FDR is reported alongside.

LEAKAGE. This lens is inherently look-ahead-free: it uses only `entry_price`
(known at entry) and `resolved_value` (known at resolution) over *resolved* bets.
Unlike copy_window/earliness it never touches a forward price, so none of the
forward-price leakage traps (see DECISIONS.md) apply. Non-{0,1} resolved outcomes
(a handful of 0.5 splits) are dropped for a clean binary hit-rate.

CAVEAT ON THE PRODUCT. Tail edge is the LEAST copyable signal of all — its payoff
is a rare event you cannot reliably enter alongside — so any wallet this surfaces
is a WATCHLIST candidate, not an auto-copy target. See HANDOFF.md black-swan item.

Read-only against data/interim/bet_ledger.parquet. Writes nothing to data/.
Usage:  PYTHONPATH=. python scripts/audit_blackswan.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import RANKED_WALLETS_PATH, load_ledger
from src.features import only_buys

# --- knobs (fixed; no Date/random in the pipeline, mirror audit_persistence) ---
SEED = 12345
TAIL_THRESHOLDS = (0.10, 0.15, 0.20)  # "longshot" entry-price ceilings to scan
PRIMARY_THRESHOLD = 0.15              # the headline tail band (task's e.g.)
MIN_TAIL_BETS = 20                    # per-wallet floor to have any power at all
ALPHA = 0.05                          # one-sided significance / FDR level
N_SHUFFLES = 300                      # null replications
N_BOOTSTRAP = 2000                    # bootstrap resamples for growth/ROI CIs
CENT_BIN = 0.01                       # market base-rate / shuffle price resolution
SHORTLIST_MAX = 25                    # cap on printed per-wallet detail


# --------------------------------------------------------------------------- #
# Data prep + market base rate                                                 #
# --------------------------------------------------------------------------- #
def prepare() -> pd.DataFrame:
    """Resolved BUY bets with clean 0/1 outcomes, plus a per-bet market-wide base
    rate E[outcome | entry_price] estimated over fine (1-cent) price bins across
    ALL resolved BUY bets. A single wallet contributes negligibly to a market-wide
    cent-bin (thousands of bets each in the tail), so no leave-one-out is needed —
    same reasoning as features.fit_price_baseline."""
    bets = only_buys(load_ledger())
    r = bets.loc[bets["resolved"]].copy()
    n_before = len(r)
    r = r.loc[r["resolved_value"].isin((0.0, 1.0))].copy()
    dropped = n_before - len(r)
    r["outcome"] = r["resolved_value"].to_numpy(dtype=float)
    r["cent"] = np.floor(r["entry_price"] / CENT_BIN).astype(int)

    # market-wide hit rate per cent-bin -> per-bet base rate
    bin_rate = r.groupby("cent")["outcome"].mean()
    r["base_rate"] = r["cent"].map(bin_rate).to_numpy(dtype=float)
    r.attrs["dropped_non_binary"] = dropped
    r.attrs["bin_rate"] = bin_rate
    return r


# --------------------------------------------------------------------------- #
# Variance-aware significance primitives                                       #
# --------------------------------------------------------------------------- #
def poisson_binomial_sf(probs: np.ndarray) -> np.ndarray:
    """Survival function of a Poisson-binomial (sum of independent Bernoullis with
    per-bet success probabilities `probs`): returns sf where sf[k] = P(Hits >= k),
    for k = 0..len(probs). Exact, via iterative convolution of the per-bet pmfs.

    This is the correct null model for "#hits among tail bets, each hitting at its
    market base rate" — variance-aware by construction (each bet carries its own
    probability), and, unlike a t-test on a mean, undominated by the couple of
    big-payoff hits (a hit is a hit here; magnitude lives in the ROI lens)."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    tail = np.cumsum(pmf[::-1])[::-1]  # tail[k] = P(Hits >= k)
    return tail


def bootstrap_ci(outcome, base, price, size, rng, n_boot=N_BOOTSTRAP):
    """Bootstrap (resample tail bets with replacement) one-sided lower bounds for
    the two fat-tailed quantities where a few 30x hits dominate the SE:
      - resid_base = mean(outcome) - mean(base_rate)   (tail skill vs the market)
      - roi        = sum(size*(outcome-price)) / sum(size*price)   (dollar growth)
    Returns (resid_lo, resid_hi, roi_lo, roi_hi) at the 5th/95th percentiles."""
    n = len(outcome)
    idx = rng.integers(0, n, size=(n_boot, n))
    o, b = outcome[idx], base[idx]
    resid = o.mean(axis=1) - b.mean(axis=1)
    s, p = size[idx], price[idx]
    cost = (s * p).sum(axis=1)
    pnl = (s * (o - p)).sum(axis=1)
    roi = np.where(cost > 0, pnl / cost, np.nan)
    return (
        float(np.percentile(resid, 5)), float(np.percentile(resid, 95)),
        float(np.nanpercentile(roi, 5)), float(np.nanpercentile(roi, 95)),
    )


def kelly_growth_nats(q: float, p: float) -> float:
    """Directional Kelly log-growth per bet: the KL divergence D(q||p), which is
    exactly the log-optimal bankroll growth rate of a full-Kelly bettor who backs
    the wallet's side at price p and is right with probability q. Zero when q<=p
    (no edge on the side they bet -> optimal not to bet -> no growth)."""
    if not (0.0 < p < 1.0) or not (0.0 < q < 1.0) or q <= p:
        return 0.0
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR                                                       #
# --------------------------------------------------------------------------- #
def bh_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of hypotheses rejected at FDR level q."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    passed = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        reject[order[: kmax + 1]] = True
    return reject


# --------------------------------------------------------------------------- #
# Per-wallet tail statistics                                                   #
# --------------------------------------------------------------------------- #
def tail_wallet_stats(r: pd.DataFrame, threshold: float, min_bets: int) -> pd.DataFrame:
    """Per-wallet tail metrics for wallets with >= min_bets bets at entry<=threshold.
    Returns one row per qualifying wallet with hit-rate, base rate, both residuals,
    realized P&L/ROI, Kelly growth, and the exact Poisson-binomial one-sided p-value
    for H1: the wallet hits more than its bets' market base rate predicts."""
    tail = r.loc[r["entry_price"] <= threshold]
    rows = []
    for wallet, g in tail.groupby("wallet"):
        n = len(g)
        if n < min_bets:
            continue
        outcome = g["outcome"].to_numpy(dtype=float)
        base = g["base_rate"].to_numpy(dtype=float)
        price = g["entry_price"].to_numpy(dtype=float)
        size = g["size"].to_numpy(dtype=float)

        hits = int(outcome.sum())
        qhat = hits / n
        pbar = float(price.mean())
        base_mean = float(base.mean())
        cost = float((size * price).sum())
        pnl = float((size * (outcome - price)).sum())

        sf = poisson_binomial_sf(base)          # sf[k] = P(Hits >= k) under base rate
        pb_p = float(sf[hits]) if hits < sf.size else 0.0

        rows.append({
            "wallet": wallet,
            "n_tail": n,
            "hits": hits,
            "hit_rate": qhat,
            "mean_price": pbar,
            "base_rate": base_mean,
            "resid_vs_price": qhat - pbar,
            "resid_vs_base": qhat - base_mean,
            "pnl_usdc": pnl,
            "roi": pnl / cost if cost > 0 else np.nan,
            "kelly_nats_per_bet": kelly_growth_nats(qhat, pbar),
            "pb_p": pb_p,
            "breadth": int(g["market_id"].nunique()),
        })
    return pd.DataFrame(rows)


def precompute_null_lookup(r: pd.DataFrame, threshold: float, min_bets: int):
    """Everything needed to score the shuffle-null cheaply. Because a wallet's tail
    PRICES are fixed under the null (only outcomes shuffle), its Poisson-binomial
    survival function and base-rate mean are constants — precompute them once, then
    each shuffle only needs the wallet's shuffled hit-count to read off a p-value.

    Returns:
      tail_df   : the tail bets (entry<=threshold), with 'cent' and a compact
                  integer wallet id for tested wallets (-1 for untested).
      sf_list   : per tested-wallet Poisson-binomial survival function.
      base_mean : per tested-wallet mean base rate.
      n_list    : per tested-wallet tail-bet count.
    """
    tail = r.loc[r["entry_price"] <= threshold].copy()
    counts = tail.groupby("wallet").size()
    tested = counts.index[counts >= min_bets]
    wid = {w: i for i, w in enumerate(tested)}
    tail["wid"] = tail["wallet"].map(wid).fillna(-1).astype(int)

    sf_list, base_mean, n_list = [], [], []
    for w in tested:
        base = tail.loc[tail["wallet"] == w, "base_rate"].to_numpy(dtype=float)
        sf_list.append(poisson_binomial_sf(base))
        base_mean.append(float(base.mean()))
        n_list.append(int(base.size))
    return tail, sf_list, np.array(base_mean), np.array(n_list, dtype=int)


def null_winner_counts(tail: pd.DataFrame, sf_list, base_mean, n_list, rng,
                       n_shuffles=N_SHUFFLES, alpha=ALPHA) -> np.ndarray:
    """Shuffle resolved outcomes WITHIN each cent price-bin (preserving the market
    base rate and every wallet's price mix, destroying only which wallet holds which
    outcome), then re-score each tested wallet's winner condition (resid_vs_base>0
    AND Poisson-binomial p<alpha). Returns the winner count per shuffle."""
    outcome = tail["outcome"].to_numpy(dtype=float)
    wid = tail["wid"].to_numpy()
    cent = tail["cent"].to_numpy()
    # index groups per cent-bin once (shuffle is a within-group permutation)
    bin_groups = [np.where(cent == c)[0] for c in np.unique(cent)]
    n_tested = len(sf_list)
    tested_mask = wid >= 0

    counts = np.empty(n_shuffles, dtype=int)
    for s in range(n_shuffles):
        shuf = outcome.copy()
        for grp in bin_groups:
            shuf[grp] = outcome[rng.permutation(grp)]
        # hits per tested wallet
        hits = np.bincount(wid[tested_mask], weights=shuf[tested_mask],
                           minlength=n_tested).astype(int)
        winners = 0
        for i in range(n_tested):
            h = hits[i]
            qhat = h / n_list[i]
            if qhat <= base_mean[i]:
                continue
            p = sf_list[i][h] if h < sf_list[i].size else 0.0
            if p < alpha:
                winners += 1
        counts[s] = winners
    return counts


def oos_tail_persistence(r: pd.DataFrame, threshold: float, min_half: int) -> None:
    """Secondary lens: does a wallet's tail skill PERSIST out-of-sample? Split each
    wallet's tail bets chronologically 50/50; among wallets with >= min_half tail
    bets in BOTH halves, does first-half resid_vs_base predict the sign/level of the
    held-out second half? Reported honestly — tail bets are so sparse that very few
    wallets even qualify."""
    tail = r.loc[r["entry_price"] <= threshold]
    a1, a2 = [], []
    n_qual = 0
    for _w, g in tail.groupby("wallet"):
        g = g.sort_values("timestamp")
        k = len(g) // 2
        h1, h2 = g.iloc[:k], g.iloc[k:]
        if len(h1) < min_half or len(h2) < min_half:
            continue
        n_qual += 1
        a1.append(h1["outcome"].mean() - h1["base_rate"].mean())
        a2.append(h2["outcome"].mean() - h2["base_rate"].mean())
    print(f"\n--- Out-of-sample tail persistence (split-half, >= {min_half} tail "
          f"bets/half) ---")
    if n_qual < 5:
        print(f"  only {n_qual} wallets qualify — too few to say anything; tail bets "
              "are too sparse to split. (Expected: the tail is rare by definition.)")
        return
    a1, a2 = np.array(a1), np.array(a2)
    rho = pd.Series(a1).corr(pd.Series(a2), method="spearman")
    sign_agree = float(((a1 > 0) == (a2 > 0)).mean())
    pos_then_pos = float((a2[a1 > 0] > 0).mean()) if (a1 > 0).any() else float("nan")
    print(f"  qualifying wallets: {n_qual}")
    print(f"  Spearman(first-half resid, second-half resid): {rho:+.3f}")
    print(f"  sign agreement across halves: {sign_agree:.1%}")
    print(f"  P(second-half resid>0 | first-half resid>0): {pos_then_pos:.1%}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(SEED)
    r = prepare()

    print("=" * 78)
    print("BLACK-SWAN / TAIL-EDGE DETECTOR  (read-only; writes nothing)")
    print("=" * 78)
    print(f"resolved BUY bets (clean 0/1): {len(r):,}   "
          f"(dropped {r.attrs['dropped_non_binary']} non-binary resolved outcomes)")
    print(f"distinct wallets: {r['wallet'].nunique():,}")

    # -- 0. Is there even a favorite-longshot bias in the tail of THIS data? --
    print("\n--- 0. Market-wide tail calibration (favorite-longshot in the tail) ---")
    print("  hit_rate vs mean price by band; resid = hit - price "
          "(>0 => longshots UNDERpriced here)")
    bands = [0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    tb = r.loc[r["entry_price"] <= 0.20].copy()
    tb["band"] = pd.cut(tb["entry_price"], bands)
    cal = tb.groupby("band", observed=True).agg(
        n=("outcome", "size"), hit=("outcome", "mean"), price=("entry_price", "mean"))
    cal["resid"] = cal["hit"] - cal["price"]
    print(cal.round(4).to_string())
    print("  NOTE: in this micro-crypto-dominated ledger the tail is ~calibrated "
          "(small, mixed-sign residuals),")
    print("        NOT the classic 'longshots overpriced' book — so beating price "
          "~= beating base rate here.")

    # -- 1. Winner funnel across thresholds (headline: real vs null) ----------
    print("\n--- 1. Tail-winner funnel, real vs. shuffled-outcome null "
          f"(min {MIN_TAIL_BETS} tail bets, alpha={ALPHA}) ---")
    print(f"{'thr':>5} {'tested':>7} {'beat_price':>10} {'beat_base':>9} "
          f"{'p<a(sig)':>8} {'BH-FDR':>6} | {'null_mean':>9} {'null_p95':>8} "
          f"{'null_max':>8}")
    primary_stats = None
    for thr in TAIL_THRESHOLDS:
        stats = tail_wallet_stats(r, thr, MIN_TAIL_BETS)
        if stats.empty:
            print(f"{thr:>5.2f}  (no wallets with >= {MIN_TAIL_BETS} tail bets)")
            continue
        beat_price = int((stats["resid_vs_price"] > 0).sum())
        beat_base = int((stats["resid_vs_base"] > 0).sum())
        sig = int(((stats["resid_vs_base"] > 0) & (stats["pb_p"] < ALPHA)).sum())
        bh = int(((stats["resid_vs_base"] > 0).to_numpy()
                  & bh_reject(stats["pb_p"].to_numpy(), ALPHA)).sum())

        tail, sf_list, base_mean, n_list = precompute_null_lookup(r, thr, MIN_TAIL_BETS)
        null_counts = null_winner_counts(tail, sf_list, base_mean, n_list, rng)
        print(f"{thr:>5.2f} {len(stats):>7d} {beat_price:>10d} {beat_base:>9d} "
              f"{sig:>8d} {bh:>6d} | {null_counts.mean():>9.1f} "
              f"{np.percentile(null_counts, 95):>8.0f} {null_counts.max():>8d}")
        if thr == PRIMARY_THRESHOLD:
            primary_stats = (stats, sig, bh, null_counts)

    if primary_stats is None:
        print("\nNo primary-threshold stats to detail. Done.")
        return
    stats, sig, bh, null_counts = primary_stats

    # -- 2. Interpret the null: is the real signal above chance? --------------
    print(f"\n--- 2. Verdict at threshold {PRIMARY_THRESHOLD} "
          f"(the null is the arbiter) ---")
    real_sig = sig
    null_mean = null_counts.mean()
    null_p95 = np.percentile(null_counts, 95)
    exceed = float((null_counts >= real_sig).mean())
    print(f"  real 'winners' (resid_vs_base>0 & PB p<{ALPHA}): {real_sig}")
    print(f"  null winners: mean {null_mean:.1f}, 95th pct {null_p95:.0f}, "
          f"max {null_counts.max()} over {N_SHUFFLES} shuffles")
    print(f"  P(null >= real) = {exceed:.3f}  "
          f"(<0.05 => the COUNT of winners exceeds chance)")
    print(f"  BH-FDR survivors at q={ALPHA}: {bh}  "
          "(these are the individually-defensible wallets, if any)")

    # -- 3. Bootstrap the shortlist (growth/ROI, where the fat tail dominates) --
    shortlist = stats.loc[(stats["resid_vs_base"] > 0) & (stats["pb_p"] < ALPHA)] \
        .sort_values("pb_p")
    bh_mask = (stats["resid_vs_base"] > 0).to_numpy() & bh_reject(stats["pb_p"].to_numpy(), ALPHA)
    bh_wallets = set(stats.loc[bh_mask, "wallet"])
    print(f"\n--- 3. Shortlist detail (top {min(len(shortlist), SHORTLIST_MAX)} by "
          "PB p-value) with bootstrap CIs ---")
    if shortlist.empty:
        print("  none — no wallet clears resid_vs_base>0 & significance. "
              "The tail lens finds nothing here.")
    else:
        tail_primary = r.loc[r["entry_price"] <= PRIMARY_THRESHOLD]
        recs = []
        for _, row in shortlist.head(SHORTLIST_MAX).iterrows():
            g = tail_primary.loc[tail_primary["wallet"] == row["wallet"]]
            rlo, rhi, roilo, roihi = bootstrap_ci(
                g["outcome"].to_numpy(float), g["base_rate"].to_numpy(float),
                g["entry_price"].to_numpy(float), g["size"].to_numpy(float), rng)
            recs.append({
                "wallet": row["wallet"][:12] + "…",
                "n": int(row["n_tail"]), "hits": int(row["hits"]),
                "brd": int(row["breadth"]),
                "hit": round(row["hit_rate"], 3),
                "base": round(row["base_rate"], 3),
                "res_base": round(row["resid_vs_base"], 3),
                "res_lo5": round(rlo, 3),          # bootstrap one-sided lower bound
                "roi": round(row["roi"], 2),
                "roi_lo5": round(roilo, 2),
                "kelly": round(row["kelly_nats_per_bet"], 4),
                "pnl": round(row["pnl_usdc"], 1),
                "pb_p": f"{row['pb_p']:.2e}",
                "BH": "Y" if row["wallet"] in bh_wallets else "",
            })
        detail = pd.DataFrame(recs)
        print(detail.to_string(index=False))
        print("\n  res_lo5 / roi_lo5 = bootstrap 5th-pct lower bounds (variance-aware; "
              "a few 30x hits dominate these SEs).")
        print("  A wallet is only credible if res_lo5>0 (skill survives the fat tail) "
              "AND BH='Y' (survives multiple testing).")
        surv = detail[(detail["res_lo5"] > 0)]
        print(f"  wallets with bootstrap res_lo5>0: {len(surv)} of "
              f"{len(detail)} shown; with BH='Y' too: "
              f"{int((surv['BH'] == 'Y').sum())}")

    # -- 4. Growth-only view (pure tail return, independent of hit-rate sig) ---
    print("\n--- 4. Pure-growth view: top tail wallets by realized ROI ---")
    print("  (growth can be real even when hit-rate significance is weak — and "
          "vice-versa; cross-check against §3)")
    growth = stats.sort_values("roi", ascending=False).head(10)
    gv = growth[["wallet", "n_tail", "hits", "hit_rate", "base_rate",
                 "roi", "pnl_usdc", "kelly_nats_per_bet", "pb_p"]].copy()
    gv["wallet"] = gv["wallet"].str[:12] + "…"
    gv["hit_rate"] = gv["hit_rate"].round(3)
    gv["base_rate"] = gv["base_rate"].round(3)
    gv["roi"] = gv["roi"].round(2)
    gv["pnl_usdc"] = gv["pnl_usdc"].round(1)
    gv["kelly_nats_per_bet"] = gv["kelly_nats_per_bet"].round(4)
    gv["pb_p"] = gv["pb_p"].map(lambda x: f"{x:.2e}")
    print(gv.to_string(index=False))

    # -- 5. Cross-reference vs. the mean-edge pipeline (the whole point) -------
    # The premise is that the mean-edge + t-test ranking MISSES tail wallets.
    # Test it: how many FDR-surviving tail wallets does the main pipeline fail to
    # flag as persisted, and how deep in the ranking are they buried?
    print("\n--- 5. Cross-reference: which tail survivors does the mean-edge "
          "pipeline miss? ---")
    ranked_path = RANKED_WALLETS_PATH
    if not ranked_path.exists():
        print("  (ranked_wallets.parquet not found — run the pipeline to compare)")
    else:
        rk = pd.read_parquet(ranked_path)
        if "rank" not in rk.columns:
            rk = rk.reset_index(drop=True)
            rk["rank"] = np.arange(1, len(rk) + 1)
        surv = stats.loc[bh_mask].merge(
            rk[["wallet", "rank", "edge_persisted"]], on="wallet", how="left")
        surv["edge_persisted"] = surv["edge_persisted"].fillna(False)
        surv = surv.sort_values("pb_p")
        missed = surv.loc[~surv["edge_persisted"]]
        print(f"  BH-FDR tail survivors: {len(surv)}  |  already edge_persisted by "
              f"the pipeline: {int(surv['edge_persisted'].sum())}  |  MISSED by it: "
              f"{len(missed)}")
        if not missed.empty:
            show = missed[["wallet", "n_tail", "hits", "hit_rate", "base_rate",
                           "resid_vs_base", "roi", "pb_p", "rank"]].copy()
            show["wallet"] = show["wallet"].str[:12] + "…"
            for c in ("hit_rate", "base_rate", "resid_vs_base", "roi"):
                show[c] = show[c].round(3)
            show["pb_p"] = show["pb_p"].map(lambda x: f"{x:.2e}")
            show["rank"] = show["rank"].astype("Int64")
            print("  Tail-skilled wallets the pipeline buries (mean edge too small "
                  "for its t-test to certify):")
            print(show.to_string(index=False))
            print("  => the premise holds: these earn via fat-tail variance, so their "
                  "per-bet MEAN edge is\n     tiny/negative and the ranking sinks them "
                  "(ranks up to "
                  f"{int(missed['rank'].max()) if missed['rank'].notna().any() else 'NA'}"
                  "), yet their tail hit-rate is significant.")

    # -- 6. Out-of-sample tail persistence -------------------------------------
    oos_tail_persistence(r, PRIMARY_THRESHOLD, min_half=MIN_TAIL_BETS // 2)

    # -- 6. Honest closing note ------------------------------------------------
    print("\n" + "=" * 78)
    print("READ THE NULL FIRST. If real winners <= null 95th pct (or P(null>=real) "
          ">= 0.05),")
    print("the tail 'winners' are indistinguishable from luck given how few tail "
          "events exist —")
    print("trust only BH-FDR survivors with a positive bootstrap lower bound, and "
          "even those are")
    print("WATCHLIST-only: tail edge is the least copyable signal (its payoff is a "
          "rare event you")
    print("cannot reliably enter alongside). See HANDOFF.md black-swan backlog item.")
    print("=" * 78)


if __name__ == "__main__":
    main()
