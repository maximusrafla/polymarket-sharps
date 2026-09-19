"""Edge-left-at-detection, LONG-LATENCY re-run (Δ = 1h primary, 6h secondary).

Companion to `scripts/audit_edge_decay.py` (2026-07-18), which answered the
sub-15-minute question on a 1.4 h ledger. That ledger could not reach Δ ≥ 1h at
all (0% coverage — unmeasurable by construction). The deep per-wallet backfills
since then pulled full `/trades?user=` histories, so the ledger now spans
~2,740 dense hours and **Δ = 1h is measurable**: a 54k-bet, 6.7k-token sample
that is 98.4% real-world.

This script re-runs the SAME analysis at those long latencies. It differs from
the original in three ways, all forced by what changed:

  1. **Vectorized.** The original looped `_forward_price_arrays` per bet — fine
     for 21k bets, hopeless for the 4.12M resolved BUYs in today's ledger. The
     follower price here is computed with the same composite-key + prefix-sum
     machinery `features._forward_drift_from_arrays` uses, which is the vectorized
     form of the identical window semantics (verified against
     `_forward_price_arrays` by the self-check in `--verify`).
  2. **Scoped to real-world markets.** The micro-crypto horizon is CLOSED, not
     re-litigated: its coverage decay is structural (5-minute BTC/ETH "up or down"
     markets are resolved an hour after entry — there is no tape left to price
     against), so only 877 micro bets have any 1h follow-on print versus 53,897
     real-world. Ingesting longer cannot change that. The micro cohort is reported
     as a one-line count, not analyzed.
  3. **Concentration guard.** The 24h slice's top tokens are all 2026 World Cup
     football markets. Correlated markets sharing one resolution event are exactly
     what manufactured Project 2 §1.5's false survivors, so every cohort here also
     reports effective breadth (Kish), distinct decision-days, top-cluster share,
     a **cluster bootstrap** CI (resampling markets / slug-families, not bets), and
     a **leave-one-family-out** jackknife. A result that only survives the iid
     bootstrap is not believed.

Everything else is deliberately unchanged from the original: same-cohort own-edge
baseline, shuffled-outcome null, skill-neutralized (favorite-longshot-removed)
follower edge, bootstrap 95% CIs, entry-price-band stratification, and the exact
copy_window leakage discipline (bounded window, resolution guard, no
`resolved_value` fallback).

Read-only. Writes NOTHING.

Usage:  PYTHONPATH=. .venv/bin/python scripts/audit_edge_decay_long.py [--verify]
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import pandas as pd

from src.common import load_config, load_ledger
from src.features import _forward_price_arrays, expected_outcome, fit_price_baseline

SEED = 12345          # fixed — the pipeline forbids Date/random for reproducibility
N_BOOT = 2000         # bootstrap resamples for the follower-edge CIs
N_SHUFFLE = 200       # outcome permutations for the null follower edge
BOOT_CHUNK = 100      # resamples per batch (bounds the bootstrap index matrix)

# Δ sweep (seconds). Δ=0 == the wallet's own edge (the ceiling). The short end is
# kept so this run is directly comparable to the 2026-07-18 numbers.
DELTAS = [
    (0, "0"),
    (30, "30s"),
    (60, "1m"),
    (300, "5m"),
    (900, "15m"),
    (3600, "1h"),
    (21600, "6h"),
    (86400, "24h"),
]
PRIMARY_DELTA = 3600            # the run this session exists to do
PRIMARY_BANDWIDTH = 60          # follower fill window (s); matches watch.py's poll
SENSITIVITY_BANDWIDTHS = [30, 60, 300]
PRICE_BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
DAY = 86400


def micro_slug_values(slugs: np.ndarray) -> np.ndarray:
    """`audit_edge_decay.classify_horizon`'s slug half, applied to category VALUES
    (not rows): micro = the 5m/15m/hourly crypto 'up or down' coinflips that
    dominate the raw feed; everything else is real_world. Keyed on slug/question,
    never on observed lifespan (a 15-minute crypto market outlives an in-progress
    game inside a short poll window, so lifespan is a bad horizon proxy)."""
    s = pd.Series(slugs).fillna("").str.lower()
    return (s.str.contains("updown", regex=False)
            | s.str.contains("up-or-down", regex=False)).to_numpy()


def micro_question_values(questions: np.ndarray) -> np.ndarray:
    """The question half of the same classifier."""
    q = pd.Series(questions).fillna("").str.lower()
    return q.str.contains("up or down", regex=False).to_numpy()


def slug_family(slugs: np.ndarray) -> np.ndarray:
    """Coarse event family = first three hyphen-segments of the slug.

    A deliberately blunt heuristic for 'markets that resolve off the same event'
    (`fifa-world-cup-brazil-argentina`, `fifa-world-cup-france-spain`, … collapse
    to `fifa-world-cup`). It over-merges rather than under-merges, which is the
    conservative direction for a concentration guard: an over-merged family makes
    the cluster bootstrap and the leave-one-out jackknife HARSHER, never softer.
    """
    s = pd.Series(slugs).fillna("").str.lower()
    return s.str.split("-").str[:3].str.join("-").to_numpy()


# --------------------------------------------------------------------------
# Vectorized follower price (same window semantics as _forward_price_arrays)
# --------------------------------------------------------------------------
class FollowerPricer:
    """Size-weighted price of *other* wallets' trades in the bounded window
    (entry + Δ, entry + Δ + bandwidth], at or before the per-token resolution
    guard cutoff. NaN when nothing qualifies — never a `resolved_value` fallback.

    Trades are sorted once by (token, ts) and by ((token, wallet), ts); each
    query is then a pair of `searchsorted`s and a prefix-sum difference, so the
    whole Δ sweep costs a handful of vector ops instead of 4M Python calls. The
    own-wallet contribution is subtracted (window minus own) rather than masked,
    exactly as `features._forward_drift_from_arrays` does.
    """

    def __init__(self, ts, price, size, tok_codes, wal_codes, n_tok, n_wal,
                 guard: float, max_offset: int):
        ts = ts.astype(np.int64)
        self.ts_min = int(ts.min())
        ts0 = ts - self.ts_min
        span = int(ts0.max())
        self.BIG = span + int(max_offset) + 1

        g = pd.DataFrame({"tok": tok_codes, "ts0": ts0}).groupby("tok")["ts0"]
        t_min = g.min().reindex(range(n_tok)).to_numpy()
        t_max = g.max().reindex(range(n_tok)).to_numpy()
        self.cutoff = np.floor(t_max - guard * (t_max - t_min)).astype(np.int64)
        self.tok_first = t_min.astype(np.int64)  # for the random-anchor placebo
        del g, t_min, t_max

        sp = size * price
        key = tok_codes * self.BIG + ts0
        o = np.argsort(key, kind="stable")
        self.K1 = key[o]
        del key
        self.P1_size = np.concatenate(([0.0], np.cumsum(size[o])))
        self.P1_sp = np.concatenate(([0.0], np.cumsum(sp[o])))
        self.P1_price = np.concatenate(([0.0], np.cumsum(price[o])))
        del o

        key = (tok_codes * n_wal + wal_codes) * self.BIG + ts0
        o = np.argsort(key, kind="stable")
        self.K2 = key[o]
        del key
        self.P2_size = np.concatenate(([0.0], np.cumsum(size[o])))
        self.P2_sp = np.concatenate(([0.0], np.cumsum(sp[o])))
        self.P2_price = np.concatenate(([0.0], np.cumsum(price[o])))
        del o, sp
        self.n_wal = n_wal
        gc.collect()

    def _windows(self, b_tok, b_wal, b_ts0, lo_off, hi_off):
        """(n_other, oth_size, oth_sp, oth_price) for the window
        (entry+lo_off, min(entry+hi_off, guard_cutoff)]. hi_off=None => cutoff.
        Offsets may be scalars or per-bet arrays (the placebo anchors use arrays)."""
        lo_t = b_ts0 + np.asarray(lo_off, dtype=np.int64)
        cap = self.cutoff[b_tok]
        upper = cap if hi_off is None else np.minimum(b_ts0 + np.asarray(hi_off, dtype=np.int64), cap)
        bg = b_tok * self.n_wal + b_wal

        lo = np.searchsorted(self.K1, b_tok * self.BIG + lo_t, side="right")
        hi = np.searchsorted(self.K1, b_tok * self.BIG + upper, side="right")
        olo = np.searchsorted(self.K2, bg * self.BIG + lo_t, side="right")
        ohi = np.searchsorted(self.K2, bg * self.BIG + upper, side="right")

        n_other = (hi - lo) - (ohi - olo)
        oth_size = (self.P1_size[hi] - self.P1_size[lo]) - (self.P2_size[ohi] - self.P2_size[olo])
        oth_sp = (self.P1_sp[hi] - self.P1_sp[lo]) - (self.P2_sp[ohi] - self.P2_sp[olo])
        oth_price = (self.P1_price[hi] - self.P1_price[lo]) - (self.P2_price[ohi] - self.P2_price[olo])
        # (hi-lo) <= 0 when the window is empty or sits past the guard cutoff
        empty = (hi - lo) <= 0
        n_other = np.where(empty, 0, n_other)
        return n_other, oth_size, oth_sp, oth_price

    def price_at(self, b_tok, b_wal, b_ts0, delta, bandwidth: int) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.int64)
        n_other, oth_size, oth_sp, oth_price = self._windows(
            b_tok, b_wal, b_ts0, delta, delta + bandwidth
        )
        out = np.full(b_tok.size, np.nan)
        valid = n_other > 0
        w = valid & (oth_size > 0)
        out[w] = oth_sp[w] / oth_size[w]
        # degenerate non-positive weight (real sizes are always > 0): unweighted
        # mean, matching `_forward_price_arrays`.
        p = valid & ~(oth_size > 0)
        out[p] = oth_price[p] / n_other[p]
        return out

    def has_followable_liquidity(self, b_tok, b_wal, b_ts0) -> np.ndarray:
        """Copyability ceiling: does ANY other-wallet trade happen after entry and
        before the guard? If not the position is un-copyable at any Δ."""
        n_other, *_ = self._windows(b_tok, b_wal, b_ts0, 0, None)
        return n_other > 0


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def boot_ci(values: np.ndarray, rng, n: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean, batched so the index matrix never
    exceeds BOOT_CHUNK x len(values)."""
    values = values[~np.isnan(values)]
    if values.size < 2:
        return (np.nan, np.nan)
    means = []
    done = 0
    while done < n:
        k = min(BOOT_CHUNK, n - done)
        idx = rng.integers(0, values.size, size=(k, values.size))
        means.append(values[idx].mean(axis=1))
        done += k
    means = np.concatenate(means)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cluster_boot_ci(values: np.ndarray, cluster_codes: np.ndarray, rng,
                    n: int = N_BOOT) -> tuple[float, float]:
    """Cluster (block) bootstrap 95% CI: resample whole CLUSTERS with replacement.

    The iid bootstrap assumes bets are independent. They are not when many bets
    ride the same resolution event (all World Cup group-stage markets settle off
    one tournament), so an iid CI is far too narrow there. Resampling clusters
    prices that dependence in: with G effective clusters the CI widens toward what
    G independent observations would support."""
    ok = ~np.isnan(values)
    values, cluster_codes = values[ok], cluster_codes[ok]
    g = np.unique(cluster_codes, return_inverse=True)[1]
    G = int(g.max()) + 1 if g.size else 0
    if G < 2:
        return (np.nan, np.nan)
    sums = np.bincount(g, weights=values, minlength=G)
    cnts = np.bincount(g, minlength=G).astype(float)
    means = []
    done = 0
    while done < n:
        k = min(BOOT_CHUNK, n - done)
        idx = rng.integers(0, G, size=(k, G))
        means.append(sums[idx].sum(axis=1) / cnts[idx].sum(axis=1))
        done += k
    means = np.concatenate(means)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def null_follower_edge(resolved_value, price_at, rng) -> tuple[float, float]:
    """Shuffled-outcome null: mean follower edge if outcomes were random. Permuting
    `resolved_value` destroys every outcome<->price relationship, so whatever
    survives is pure price-level base rate (the favorite-longshot edge available to
    anyone at that price), not copyable timing. Returns (null mean, one-sided
    empirical p = share of shuffles at least as good as the real edge)."""
    cohort = ~np.isnan(price_at)
    rv, pv = resolved_value[cohort], price_at[cohort]
    real = float((rv - pv).mean())
    nulls = np.empty(N_SHUFFLE)
    for k in range(N_SHUFFLE):
        shuffled = rng.permutation(resolved_value)
        nulls[k] = float((shuffled[cohort] - pv).mean())
    return float(nulls.mean()), float((nulls >= real).mean())


def effective_breadth(codes: np.ndarray) -> float:
    """Kish effective number of clusters: (Σn)² / Σn². Equals the true count when
    bets are spread evenly and collapses toward 1 when one cluster dominates."""
    cnt = np.bincount(np.unique(codes, return_inverse=True)[1]).astype(float)
    return float(cnt.sum() ** 2 / (cnt**2).sum()) if cnt.size else 0.0


def concentration_report(skill: np.ndarray, mkt: np.ndarray, fam: np.ndarray,
                         days: np.ndarray, rng, indent="    ") -> None:
    """The World-Cup guard. Prints breadth, cluster-bootstrap CIs and the
    leave-one-family-out jackknife for one cohort's skill-edge vector."""
    n = skill.size
    top_fam, top_n = None, 0
    fams, counts = np.unique(fam, return_counts=True)
    if fams.size:
        j = int(counts.argmax())
        top_fam, top_n = fams[j], int(counts[j])
    print(f"{indent}breadth: {np.unique(mkt).size} markets, {fams.size} slug-families, "
          f"{np.unique(days).size} decision-days")
    print(f"{indent}effective breadth (Kish): {effective_breadth(mkt):.1f} markets, "
          f"{effective_breadth(fam):.1f} families, {effective_breadth(days):.1f} days")
    if top_fam is not None:
        print(f"{indent}largest family: {top_fam!r} = {top_n} bets ({top_n / n:.0%} of cohort)")
    m_lo, m_hi = cluster_boot_ci(skill, mkt, rng)
    f_lo, f_hi = cluster_boot_ci(skill, fam, rng)
    d_lo, d_hi = cluster_boot_ci(skill, days, rng)
    print(f"{indent}skill_edge 95% CI  — cluster by market:  [{fmt(m_lo)},{fmt(m_hi)}]")
    print(f"{indent}                     cluster by family:  [{fmt(f_lo)},{fmt(f_hi)}]")
    print(f"{indent}                     cluster by day:     [{fmt(d_lo)},{fmt(d_hi)}]")
    # leave-one-family-out: is the whole result one event?
    order = np.argsort(-counts)[:5]
    worst = None
    for j in order:
        keep = fam != fams[j]
        if keep.sum() < 2:
            continue
        m = float(skill[keep].mean())
        if worst is None or m < worst[1]:
            worst = (fams[j], m, int(keep.sum()))
    if worst is not None:
        print(f"{indent}leave-one-family-out (worst of top-5): drop {worst[0]!r} -> "
              f"skill {worst[1]:+.3f} on n={worst[2]}  (full: {skill.mean():+.3f})")


def cohort_selection_controls(pricer, b, baseline, delta: int, bandwidth: int, rng) -> None:
    """Is a large follower edge at Δ a property of the WALLET'S ENTRY, or just of
    the tiny, weird subset of bets that still has tape Δ later?

    At long Δ only ~1% of bets are measurable, and that 1% is selected on
    "somebody else traded this token exactly Δ later" — a condition that has
    nothing to do with the wallet. Three controls separate the two stories, all
    computed on the IDENTICAL cohort so only the anchor changes:

      * random-anchor placebo — price the same token at a uniformly random time in
        its observed pre-guard life instead of at entry+Δ. This is "buy this market
        at some arbitrary moment"; it is not conditioned on the wallet at all.
      * pre-entry placebo — price at entry−Δ, i.e. a follower who acted an HOUR
        BEFORE the wallet did and therefore could not have been copying it.
      * cohort own-edge vs. full-sample own-edge — how unrepresentative the
        measurable subset is in the first place.

    If the placebos land near the real follower edge, the number is the cohort, not
    the copy. Only the gap real − placebo is attributable to following the wallet.
    """
    pv = pricer.price_at(b["tok"], b["wal"], b["ts0"], delta, bandwidth)
    cohort = ~np.isnan(pv)
    if not cohort.any():
        print("    (empty cohort)")
        return
    tok_c = b["tok"][cohort]
    ts_c = b["ts0"][cohort]
    wal_c = b["wal"][cohort]
    rv_c = b["rv"][cohort]
    real = rv_c - expected_outcome(baseline, pv[cohort])

    # random anchor uniformly in [token first trade, guard cutoff − bandwidth]
    lo_a = pricer.tok_first[tok_c]
    hi_a = pricer.cutoff[tok_c] - bandwidth
    span = np.maximum(hi_a - lo_a, 0)
    anchor = lo_a + (rng.random(span.size) * span).astype(np.int64)
    rand_off = anchor - ts_c
    pv_rand = pricer.price_at(tok_c, wal_c, ts_c, rand_off, bandwidth)
    m_rand = ~np.isnan(pv_rand)
    skill_rand = rv_c[m_rand] - expected_outcome(baseline, pv_rand[m_rand])

    pv_pre = pricer.price_at(tok_c, wal_c, ts_c, -delta, bandwidth)
    m_pre = ~np.isnan(pv_pre)
    skill_pre = rv_c[m_pre] - expected_outcome(baseline, pv_pre[m_pre])

    fam_c, mkt_c = b["fam"][cohort], b["mkt"][cohort]
    r_lo, r_hi = cluster_boot_ci(real, fam_c, rng)
    print(f"    real  (entry+Δ)      : skill {real.mean():+.3f} on n={real.size:,} "
          f"[family CI {fmt(r_lo)},{fmt(r_hi)}]")
    if skill_rand.size:
        a_lo, a_hi = cluster_boot_ci(skill_rand, fam_c[m_rand], rng)
        print(f"    placebo random-anchor: skill {skill_rand.mean():+.3f} on n={skill_rand.size:,} "
              f"[family CI {fmt(a_lo)},{fmt(a_hi)}]   <-- not conditioned on the wallet")
    if skill_pre.size:
        p_lo, p_hi = cluster_boot_ci(skill_pre, fam_c[m_pre], rng)
        print(f"    placebo pre-entry −Δ : skill {skill_pre.mean():+.3f} on n={skill_pre.size:,} "
              f"[family CI {fmt(p_lo)},{fmt(p_hi)}]   <-- could not have been copying")
    print(f"    cohort own-edge {float((rv_c - b['ep'][cohort]).mean()):+.3f} vs full-sample "
          f"own-edge {float((b['rv'] - b['ep']).mean()):+.3f}  "
          f"(cohort is {cohort.mean():.2%} of bets, "
          f"{np.unique(wal_c).size:,} wallets, Kish {effective_breadth(wal_c):.0f})")
    print(f"    entry-price mix: cohort mean {float(b['ep'][cohort].mean()):.3f} vs "
          f"full-sample {float(b['ep'].mean()):.3f} | follower price mean "
          f"{float(pv[cohort].mean()):.3f}")

    # Paired test — the three anchors above are measurable on different subsets, so
    # their means are not directly comparable. Restrict to bets where ALL THREE
    # prices exist and difference them per bet: same bets, same outcomes, only the
    # entry anchor differs, so the paired mean is exactly "what following the
    # wallet at +Δ bought you over the placebo".
    both = m_rand & m_pre
    if both.sum() >= 10:
        r = rv_c[both] - expected_outcome(baseline, pv[cohort][both])
        a = rv_c[both] - expected_outcome(baseline, pv_rand[both])
        p_ = rv_c[both] - expected_outcome(baseline, pv_pre[both])
        f_b = fam_c[both]
        d1_lo, d1_hi = cluster_boot_ci(r - a, f_b, rng)
        d2_lo, d2_hi = cluster_boot_ci(r - p_, f_b, rng)
        print(f"    PAIRED on the {int(both.sum()):,} bets where all three anchors exist:")
        print(f"      real {r.mean():+.3f} | random-anchor {a.mean():+.3f} | pre-entry {p_.mean():+.3f}")
        print(f"      real − random-anchor = {r.mean() - a.mean():+.3f} "
              f"[family CI {fmt(d1_lo)},{fmt(d1_hi)}]")
        print(f"      real − pre-entry     = {r.mean() - p_.mean():+.3f} "
              f"[family CI {fmt(d2_lo)},{fmt(d2_hi)}]")
    else:
        print(f"    PAIRED: only {int(both.sum())} bets have all three anchors — not testable")


def fmt(x, w: int = 6) -> str:
    return "   nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}".rjust(w)


# --------------------------------------------------------------------------
def decay_table(pricer, b, baseline, bandwidth: int, rng) -> dict:
    """One decay curve at a fixed bandwidth. Everything is same-cohort: own edge is
    recomputed on exactly the bets measurable at each Δ, so cohort selection can
    never masquerade as decay. Returns the per-Δ cohort masks + skill vectors."""
    rv, ep = b["rv"], b["ep"]
    out = {}
    print(f"  {'Δ':>4} | {'cov':>5} {'n':>7} | {'own(coh)':>8} {'follow_raw':>10} "
          f"{'skill_edge [95% CI iid]':>24} | {'null_raw':>8} {'excess':>7} {'p':>5} {'retain':>6}")
    for delta, label in DELTAS:
        if delta == 0:
            own = float((rv - ep).mean())
            lo, hi = boot_ci(rv - ep, rng)
            print(f"  {label:>4} | {'100%':>5} {rv.size:>7} | {fmt(own,8)} {fmt(own,10)} "
                  f"{fmt(own):>8} [{fmt(lo)},{fmt(hi)}] | {'--':>8} {'--':>7} {'--':>5} {'100%':>6}")
            continue
        pv = pricer.price_at(b["tok"], b["wal"], b["ts0"], delta, bandwidth)
        cohort = ~np.isnan(pv)
        n = int(cohort.sum())
        if n == 0:
            print(f"  {label:>4} | {'0%':>5} {0:>7} | (no follower liquidity reaches this Δ)")
            continue
        own_c = float((rv[cohort] - ep[cohort]).mean())
        fol_raw = float((rv[cohort] - pv[cohort]).mean())
        skill = rv[cohort] - expected_outcome(baseline, pv[cohort])
        s_lo, s_hi = boot_ci(skill, rng)
        null_m, p = null_follower_edge(rv, pv, rng)
        retain = fol_raw / own_c if own_c != 0 else np.nan
        print(f"  {label:>4} | {cohort.mean():>5.2%} {n:>7} | {fmt(own_c,8)} {fmt(fol_raw,10)} "
              f"{fmt(float(skill.mean())):>8} [{fmt(s_lo)},{fmt(s_hi)}] | {fmt(null_m,8)} "
              f"{fmt(fol_raw - null_m,7)} {p:>5.2f} "
              f"{('nan' if np.isnan(retain) else f'{retain:.0%}'):>6}")
        out[delta] = (cohort, skill, pv)
    return out


def price_band_table(pricer, b, baseline, delta: int, bandwidth: int) -> None:
    rv, ep = b["rv"], b["ep"]
    pv = pricer.price_at(b["tok"], b["wal"], b["ts0"], delta, bandwidth)
    print(f"  {'band':>10} | {'n':>7} {'cov':>5} | {'own':>6} {'follow_raw':>10} {'skill_edge':>10}")
    for lo, hi in PRICE_BANDS:
        m = (ep >= lo) & (ep < hi)
        if not m.any():
            continue
        c = m & ~np.isnan(pv)
        own = float((rv[m] - ep[m]).mean())
        if not c.any():
            print(f"  [{lo:.1f},{hi:.1f}) | {int(m.sum()):>7} {'0%':>5} | {fmt(own)} {'--':>10} {'--':>10}")
            continue
        fol = float((rv[c] - pv[c]).mean())
        skill = float((rv[c] - expected_outcome(baseline, pv[c])).mean())
        print(f"  [{lo:.1f},{hi:.1f}) | {int(m.sum()):>7} {c.sum() / m.sum():>5.2%} | "
              f"{fmt(own)} {fmt(fol,10)} {fmt(skill,10)}")


def verify_against_reference(pricer, b, ledger_arrays, rng) -> None:
    """Self-check: the vectorized follower price must equal the original scalar
    `features._forward_price_arrays` (the production copy_window core) on a random
    sample of bets, at the primary Δ. If this fails, nothing else here is valid."""
    ts, price, size, tok, wal, wal_vals = ledger_arrays
    idx = rng.choice(b["tok"].size, size=min(300, b["tok"].size), replace=False)
    fast = pricer.price_at(b["tok"][idx], b["wal"][idx], b["ts0"][idx],
                           PRIMARY_DELTA, PRIMARY_BANDWIDTH)
    order = np.argsort(tok, kind="stable")
    tok_s, ts_s = tok[order], ts[order]
    bad = 0
    for k, i in enumerate(idx):
        t = b["tok"][i]
        lo = np.searchsorted(tok_s, t, side="left")
        hi = np.searchsorted(tok_s, t, side="right")
        sel = order[lo:hi]
        sel = sel[np.argsort(ts[sel], kind="stable")]
        cutoff = pricer.cutoff[t] + pricer.ts_min
        ref = _forward_price_arrays(
            ts[sel].astype(float), price[sel], size[sel], wal[sel],
            float(b["ts0"][i] + pricer.ts_min + PRIMARY_DELTA), b["wal"][i],
            float(PRIMARY_BANDWIDTH), float(cutoff),
        )
        ref = np.nan if ref is None else ref
        f = fast[k]
        if not ((np.isnan(ref) and np.isnan(f)) or abs(ref - f) < 1e-9):
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH bet {i}: scalar={ref} vectorized={f}")
    print(f"  [verify] {len(idx)} sampled bets, {bad} mismatches vs "
          f"features._forward_price_arrays  -> {'OK' if bad == 0 else 'FAILED'}")
    del tok_s, ts_s, order


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg = load_config()
    guard = cfg["scoring"].get("fair_value_resolution_guard", 0.2)
    n_bins = cfg["scoring"].get("price_baseline_bins", 20)
    do_verify = "--verify" in sys.argv

    print("=" * 88)
    print("EDGE LEFT AT DETECTION — LONG LATENCY RE-RUN (Δ=1h primary, 6h secondary)")
    print("=" * 88)

    cols = ["wallet", "market_id", "token_id", "side", "entry_price", "size",
            "timestamp", "resolved", "resolved_value", "slug", "question"]
    ledger = load_ledger(columns=cols,
                         categorical=["wallet", "market_id", "token_id", "side", "slug", "question"])
    if ledger.empty:
        print("[edge-decay] ledger is empty — run src.ingest first.")
        return

    # --- horizon per row, classified on category VALUES (cheap) -------------
    # slug and question have separate category tables; classify each table's
    # VALUES (thousands of strings, not 4.7M rows) and OR the two row maps.
    slug_cat = ledger["slug"].astype("category")
    q_cat = ledger["question"].astype("category")
    micro_by_slug = micro_slug_values(np.asarray(slug_cat.cat.categories))
    micro_by_q = micro_question_values(np.asarray(q_cat.cat.categories))
    sc = slug_cat.cat.codes.to_numpy()
    qc = q_cat.cat.codes.to_numpy()
    is_micro = np.where(sc >= 0, micro_by_slug[np.maximum(sc, 0)], False) | \
               np.where(qc >= 0, micro_by_q[np.maximum(qc, 0)], False)

    fam_vals = slug_family(np.asarray(slug_cat.cat.categories))
    fam_row = np.where(sc >= 0, fam_vals[np.maximum(sc, 0)], "")
    del q_cat, qc, micro_by_q
    gc.collect()

    # --- trade-side arrays (ALL trades: the price path) ---------------------
    tok_codes, tok_uniq = pd.factorize(ledger["token_id"])
    wal_codes, wal_uniq = pd.factorize(ledger["wallet"])
    tok_codes = tok_codes.astype(np.int64)
    wal_codes = wal_codes.astype(np.int64)
    t_ts = ledger["timestamp"].to_numpy().astype(np.int64)
    t_price = ledger["entry_price"].to_numpy().astype(float)
    t_size = ledger["size"].to_numpy().astype(float)

    is_buy = (ledger["side"].astype(str) == "BUY").to_numpy()
    resolved = ledger["resolved"].to_numpy(dtype=bool)
    rv_all = ledger["resolved_value"].to_numpy(dtype=float)
    mkt_codes = pd.factorize(ledger["market_id"])[0].astype(np.int64)

    span_h = (t_ts.max() - t_ts.min()) / 3600.0
    n_rows = len(ledger)
    del ledger, slug_cat, sc, micro_by_slug
    gc.collect()

    bet_mask = is_buy & resolved & ~np.isnan(rv_all)
    print(f"\nledger: {n_rows:,} trades | {int(bet_mask.sum()):,} resolved BUY bets across "
          f"{np.unique(tok_codes[bet_mask]).size:,} tokens / "
          f"{np.unique(mkt_codes[bet_mask]).size:,} markets")
    print(f"observed wall-clock span: {span_h:,.0f} h ({span_h / 24:,.0f} days) "
          "— the 1.4h ceiling of the 2026-07-18 run is gone")
    print(f"resolved BUY bets by horizon: micro_crypto {int((bet_mask & is_micro).sum()):,} | "
          f"real_world {int((bet_mask & ~is_micro).sum()):,}")

    max_offset = max(d for d, _ in DELTAS) + max(SENSITIVITY_BANDWIDTHS)
    print("\nbuilding vectorized price path (sorts + prefix sums over all trades)...")
    pricer = FollowerPricer(t_ts, t_price, t_size, tok_codes, wal_codes,
                            len(tok_uniq), len(wal_uniq), guard, max_offset)

    # --- baseline: market calibration curve over ALL resolved BUY bets ------
    baseline = fit_price_baseline(
        pd.DataFrame({"entry_price": t_price[bet_mask], "resolved_value": rv_all[bet_mask]}),
        n_bins,
    )
    mean_entry = float(t_price[bet_mask].mean())

    # --- scope: real-world only (micro is CLOSED, see module docstring) -----
    rw = bet_mask & ~is_micro
    b = {
        "tok": tok_codes[rw],
        "wal": wal_codes[rw],
        "ts0": t_ts[rw] - pricer.ts_min,
        "ep": t_price[rw],
        "rv": rv_all[rw],
        "mkt": mkt_codes[rw],
        "fam": fam_row[rw],
        "day": (t_ts[rw] // DAY),
    }
    micro_n = int((bet_mask & is_micro).sum())
    micro_tok = np.unique(tok_codes[bet_mask & is_micro]).size

    del fam_row, is_micro, is_buy, resolved, mkt_codes
    gc.collect()

    if do_verify:
        print("\n" + "-" * 88)
        print("SELF-CHECK: vectorized follower price vs. features._forward_price_arrays")
        print("-" * 88)
        verify_against_reference(pricer, b, (t_ts, t_price, t_size, tok_codes, wal_codes, wal_uniq), rng)
    del t_ts, t_price, t_size, tok_codes, wal_codes, rv_all
    gc.collect()

    # --- Section A: copyability ceiling -------------------------------------
    print("\n" + "-" * 88)
    print("SCOPE + COPYABILITY CEILING")
    print("-" * 88)
    liq = pricer.has_followable_liquidity(b["tok"], b["wal"], b["ts0"])
    print(f"  real_world : {b['rv'].size:,} resolved BUY bets | any followable other-wallet "
          f"trade before the guard: {liq.mean():.0%}")
    print(f"  micro_crypto: {micro_n:,} bets / {micro_tok:,} tokens — NOT ANALYSED. The "
          "horizon is closed:")
    print("    5-minute BTC/ETH/HYPE 'up or down' markets are RESOLVED an hour after entry, so")
    print("    there is no tape to price a 1h-latency follower against. That is structural, not")
    print("    a data limitation — longer ingest cannot fix it. See HANDOFF.md.")
    print(f"  favorite-longshot base rate over all resolved BUYs: mean resolved_value "
          f"{baseline.global_mean:.3f} vs mean entry_price {mean_entry:.3f} "
          f"(=> {baseline.global_mean - mean_entry:+.3f}/bet free to any buyer)")

    # --- Section B: decay curve ---------------------------------------------
    print("\n" + "-" * 88)
    print(f"DECAY CURVE — real_world (bandwidth = {PRIMARY_BANDWIDTH}s fill window; same-cohort)")
    print("  own(coh)=wallet edge on the SAME bets | follow_raw=resolved−price_at (follower P&L)")
    print("  skill_edge=resolved−E[outcome|price_at] (copyable alpha net of favorite-longshot)")
    print("  null_raw=shuffled-outcome floor | excess=follow_raw−null_raw | p=share of shuffles")
    print("  at least as good as the real edge | retain=follow_raw/own")
    print("  CI here is the IID bootstrap — see the concentration guard below before believing it")
    print("-" * 88)
    cohorts = decay_table(pricer, b, baseline, PRIMARY_BANDWIDTH, rng)

    # --- Section C: concentration guard (the World Cup trap) ----------------
    print("\n" + "-" * 88)
    print("CONCENTRATION GUARD — correlated markets sharing one resolution event")
    print("  (the failure mode that produced Project 2 §1.5's false survivors)")
    print("-" * 88)
    for delta, label in DELTAS:
        if delta == 0 or delta not in cohorts:
            continue
        if delta < PRIMARY_DELTA:
            continue
        cohort, skill, _pv = cohorts[delta]
        print(f"\n  [Δ={label}]  n={skill.size:,}  skill_edge={skill.mean():+.3f}")
        concentration_report(skill, b["mkt"][cohort], b["fam"][cohort], b["day"][cohort], rng)

    # --- Section C2: cohort-selection controls ------------------------------
    print("\n" + "-" * 88)
    print("COHORT-SELECTION CONTROLS — is the edge the WALLET, or the 1% subset that")
    print("  still has tape Δ later? (placebos share the identical cohort; only the anchor moves)")
    print("-" * 88)
    for delta, label in [(30, "30s"), (900, "15m"), (PRIMARY_DELTA, "1h"), (21600, "6h")]:
        print(f"\n  [Δ={label}]")
        cohort_selection_controls(pricer, b, baseline, delta, PRIMARY_BANDWIDTH, rng)

    # --- Section D: bandwidth sensitivity at the primary Δ ------------------
    print("\n" + "-" * 88)
    print(f"BANDWIDTH SENSITIVITY — real_world follower_raw edge at Δ=1h (coverage in parens)")
    print("-" * 88)
    cells = []
    for bw in SENSITIVITY_BANDWIDTHS:
        pv = pricer.price_at(b["tok"], b["wal"], b["ts0"], PRIMARY_DELTA, bw)
        c = ~np.isnan(pv)
        fol = float((b["rv"][c] - pv[c]).mean()) if c.any() else np.nan
        sk = float((b["rv"][c] - expected_outcome(baseline, pv[c])).mean()) if c.any() else np.nan
        cells.append(f"h={bw}s: raw {fmt(fol)} skill {fmt(sk)} ({c.mean():.2%})")
    print("  real_world | " + "\n             | ".join(cells))

    # --- Section E: price-band stratification -------------------------------
    print("\n" + "-" * 88)
    print(f"PRICE-BAND STRATIFICATION — real_world at Δ=1h, bandwidth={PRIMARY_BANDWIDTH}s")
    print("-" * 88)
    price_band_table(pricer, b, baseline, PRIMARY_DELTA, PRIMARY_BANDWIDTH)

    print("\n" + "=" * 88)
    print("Read the verdict in HANDOFF.md 'Edge-left-at-detection'. This script writes nothing.")
    print("=" * 88)


if __name__ == "__main__":
    main()
