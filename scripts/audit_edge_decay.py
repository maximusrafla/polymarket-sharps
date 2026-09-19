"""Edge-left-at-detection: how much of a wallet's edge survives the latency
between when it takes a position and when a follower could copy it?

Read-only against data/interim/bet_ledger.parquet. Writes NOTHING to data/.
This answers the roadmap question in HANDOFF.md ("Edge-left-at-detection
analysis"): it decides whether auto-replication is viable, and for which market
segment (micro sub-6h crypto vs. slower real-world markets).

WHAT IT COMPUTES
----------------
For each resolved BUY bet by wallet w on token T (entry_ts, entry_price):
  follower_edge(Δ) = resolved_value − price_at(entry_ts + Δ)
where price_at(t) is the size-weighted price of *other* wallets' trades on the
same token in the bounded window (t, t + bandwidth], excluding w's own trades.
Δ=0 is the wallet's own edge (resolved_value − entry_price) — the ceiling.

LEAKAGE DISCIPLINE (identical to the copy_window fix, see DECISIONS.md
"Forward-price leakage fix"):
  * bounded window only — never an unbounded "nearest future trade" reach;
  * resolution guard — trades in the final `fair_value_resolution_guard`
    fraction of a token's observed lifespan are excluded, so a near-resolution
    price (which approaches the 0/1 answer) can never stand in for the follower
    price;
  * no resolved_value fallback — if no trade qualifies, price_at is NaN (the
    follower simply could not have entered), never the outcome.
The whole analysis reuses `src.features._forward_price_arrays` (the numpy core
of `forward_price`) so the follower-price semantics are byte-identical to the
production copy_window path.

WHY THE STATS ARE SHAPED THIS WAY
---------------------------------
  * Same-cohort comparison. Which bets even HAVE a measurable follower price at
    Δ is selection-biased (only positions on tokens that keep trading survive to
    larger Δ, and those are the longer-lived, more-liquid markets). So decay is
    always reported as follower_edge(Δ) vs. the SAME bets' own edge (Δ=0) — never
    against the full-sample own edge.
  * Shuffled-outcome null. "Assume any too-good result is leakage until a null
    test says otherwise." Permuting resolved_value destroys every outcome↔price
    relationship; whatever follower edge survives the shuffle is pure price-level
    base rate (the favorite-longshot edge available to anyone at that price), not
    copyable timing/selection. real − null = the outcome-correlated part.
  * Skill-neutralized follower edge. resolved_value − E[outcome | price_at],
    using the market calibration curve (features.fit_price_baseline). This is
    "did the follower still beat the price they paid" net of favorite-longshot —
    the cleanest copyable-alpha measure, consistent with the rest of the engine.
  * Bootstrap 95% CIs on the follower edges, because the actionable cohorts are
    small (esp. real-world) and a point estimate alone would over-claim.

Usage:  PYTHONPATH=. python scripts/audit_edge_decay.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import load_config, load_ledger
from src.features import (
    _forward_price_arrays,
    expected_outcome,
    fit_price_baseline,
    only_buys,
)

SEED = 12345          # fixed — the pipeline forbids Date/random for reproducibility
N_BOOT = 2000         # bootstrap resamples for the follower-edge CIs
N_SHUFFLE = 200       # outcome permutations for the null follower edge

# Δ sweep from HANDOFF (seconds). Δ=0 == the wallet's own edge (the ceiling).
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
PRIMARY_BANDWIDTH = 60          # follower fill window (s); matches watch.py's 60s poll
SENSITIVITY_BANDWIDTHS = [30, 60, 300]
PRICE_BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


# --- market horizon classifier --------------------------------------------
# Micro = the 5m/15m/hourly crypto "up or down" coinflip markets that dominate
# the raw feed (see DECISIONS.md ingest notes); everything else (sports, events)
# is "real_world". We key on the slug/question, NOT the observed lifespan — a
# 15-minute crypto market has a longer *observed* lifespan than an in-progress
# sports game inside our short poll window, so lifespan is a bad horizon proxy.
def classify_horizon(slug: str | None, question: str | None) -> str:
    s = (slug or "").lower()
    q = (question or "").lower()
    if "updown" in s or "up-or-down" in s or "up or down" in q:
        return "micro_crypto"
    return "real_world"


def build_token_arrays(ledger: pd.DataFrame, guard: float) -> dict:
    """Per token: sorted (ts, price, size, wallet) numpy arrays and the
    resolution-guard cutoff (drop the final `guard` fraction of observed
    lifespan). Built once so the Δ sweep never re-parses the frame."""
    tok = {}
    for token_id, g in ledger.groupby("token_id"):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].to_numpy(dtype=float)
        t_min, t_max = ts.min(), ts.max()
        cutoff = t_max - guard * (t_max - t_min)
        tok[token_id] = (
            ts,
            g["entry_price"].to_numpy(dtype=float),
            g["size"].to_numpy(dtype=float),
            g["wallet"].to_numpy(),
            cutoff,
        )
    return tok


def follower_prices(bets: pd.DataFrame, tok: dict, delta: float, bandwidth: float) -> np.ndarray:
    """price_at(entry_ts + delta) per bet via the leakage-guarded forward proxy
    (reuses features._forward_price_arrays). NaN where no other-wallet trade
    qualifies in (entry_ts+delta, entry_ts+delta+bandwidth] before the guard."""
    out = np.full(len(bets), np.nan)
    tids = bets["token_id"].to_numpy()
    wallets = bets["wallet"].to_numpy()
    entries = bets["timestamp"].to_numpy(dtype=float)
    for i in range(len(bets)):
        ts, price, size, wal, cutoff = tok[tids[i]]
        fv = _forward_price_arrays(
            ts, price, size, wal, entries[i] + delta, wallets[i], bandwidth, cutoff
        )
        if fv is not None:
            out[i] = fv
    return out


def has_followable_liquidity(bets: pd.DataFrame, tok: dict) -> np.ndarray:
    """Ceiling on copyability: does ANY other-wallet trade occur after entry and
    before the resolution guard? If not, the position is un-copyable at any Δ —
    there is simply no later print to enter against."""
    out = np.zeros(len(bets), dtype=bool)
    tids = bets["token_id"].to_numpy()
    wallets = bets["wallet"].to_numpy()
    entries = bets["timestamp"].to_numpy(dtype=float)
    for i in range(len(bets)):
        ts, _price, _size, wal, cutoff = tok[tids[i]]
        lo = np.searchsorted(ts, entries[i], side="right")
        if lo >= ts.size:
            continue
        out[i] = bool(((ts[lo:] <= cutoff) & (wal[lo:] != wallets[i])).any())
    return out


def boot_ci(values: np.ndarray, rng: np.random.Generator, n: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    values = values[~np.isnan(values)]
    if values.size < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def null_follower_edge(
    resolved_value: np.ndarray, price_at: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Shuffled-outcome null: mean follower edge if outcomes were random. Returns
    (mean, one-sided empirical p) where p = share of shuffles whose null edge
    >= the real follower edge — i.e. 'the real edge is no better than base rate'."""
    cohort = ~np.isnan(price_at)
    rv, pv = resolved_value[cohort], price_at[cohort]
    real = float((rv - pv).mean())
    nulls = np.empty(N_SHUFFLE)
    for k in range(N_SHUFFLE):
        shuffled = rng.permutation(resolved_value)  # permute the whole horizon's outcomes
        nulls[k] = float((shuffled[cohort] - pv).mean())
    return float(nulls.mean()), float((nulls >= real).mean())


def fmt(x: float, w: int = 6) -> str:
    return "   nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}".rjust(w)


def decay_table(sub: pd.DataFrame, tok: dict, baseline, bandwidth: int, rng: np.random.Generator):
    """One horizon's decay curve at a fixed bandwidth. Everything is same-cohort:
    own edge is recomputed on exactly the bets measurable at each Δ."""
    rv = sub["resolved_value"].to_numpy(dtype=float)
    ep = sub["entry_price"].to_numpy(dtype=float)
    print(
        f"  {'Δ':>4} | {'cov':>4} {'n':>5} | {'own(coh)':>8} {'follow_raw':>10} "
        f"{'skill_edge [95% CI]':>22} | {'null_raw':>8} {'excess':>7} {'retain':>6}"
    )
    for delta, label in DELTAS:
        if delta == 0:
            own = float((rv - ep).mean())
            lo, hi = boot_ci(rv - ep, rng)
            print(
                f"  {label:>4} | {'100%':>4} {len(sub):>5} | {fmt(own,8)} {fmt(own,10)} "
                f"{fmt(own):>8} [{fmt(lo)},{fmt(hi)}] | {'--':>8} {'--':>7} {'100%':>6}"
            )
            continue
        pv = follower_prices(sub, tok, delta, bandwidth)
        cohort = ~np.isnan(pv)
        n = int(cohort.sum())
        if n == 0:
            print(f"  {label:>4} | {'0%':>4} {0:>5} | {'(no follower liquidity reaches this Δ)':>40}")
            continue
        own_c = float((rv[cohort] - ep[cohort]).mean())
        fol_raw = float((rv[cohort] - pv[cohort]).mean())
        skill = rv[cohort] - expected_outcome(baseline, pv[cohort])
        skill_m = float(skill.mean())
        s_lo, s_hi = boot_ci(skill, rng)
        null_m, _p = null_follower_edge(rv, pv, rng)
        excess = fol_raw - null_m
        retain = fol_raw / own_c if own_c != 0 else np.nan
        print(
            f"  {label:>4} | {cohort.mean():>4.0%} {n:>5} | {fmt(own_c,8)} {fmt(fol_raw,10)} "
            f"{fmt(skill_m):>8} [{fmt(s_lo)},{fmt(s_hi)}] | {fmt(null_m,8)} {fmt(excess,7)} "
            f"{('nan' if np.isnan(retain) else f'{retain:.0%}'):>6}"
        )


def price_band_table(sub: pd.DataFrame, tok: dict, baseline, delta: int, bandwidth: int):
    """Follower edge by entry-price band at a single latency — where does the
    surviving edge live (longshots / mid / favorites)?"""
    rv = sub["resolved_value"].to_numpy(dtype=float)
    ep = sub["entry_price"].to_numpy(dtype=float)
    pv = follower_prices(sub, tok, delta, bandwidth)
    print(f"  {'band':>10} | {'n':>5} {'cov':>4} | {'own':>6} {'follow_raw':>10} {'skill_edge':>10}")
    for lo, hi in PRICE_BANDS:
        m = (ep >= lo) & (ep < hi)
        if m.sum() == 0:
            continue
        c = m & ~np.isnan(pv)
        own = float((rv[m] - ep[m]).mean())
        if c.sum() == 0:
            print(f"  [{lo:.1f},{hi:.1f}) | {int(m.sum()):>5} {'0%':>4} | {fmt(own)} {'--':>10} {'--':>10}")
            continue
        fol = float((rv[c] - pv[c]).mean())
        skill = float((rv[c] - expected_outcome(baseline, pv[c])).mean())
        print(
            f"  [{lo:.1f},{hi:.1f}) | {int(m.sum()):>5} {c.sum()/m.sum():>4.0%} | "
            f"{fmt(own)} {fmt(fol,10)} {fmt(skill,10)}"
        )


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg = load_config()
    guard = cfg["scoring"].get("fair_value_resolution_guard", 0.2)
    n_bins = cfg["scoring"].get("price_baseline_bins", 20)

    ledger = load_ledger()
    if ledger.empty:
        print("[edge-decay] ledger is empty — run src.ingest first.")
        return
    ledger = ledger.copy()
    ledger["horizon"] = [
        classify_horizon(s, q) for s, q in zip(ledger["slug"], ledger["question"])
    ]

    bets = only_buys(ledger)
    resolved = bets.loc[bets["resolved"]].copy()
    baseline = fit_price_baseline(resolved, n_bins)  # market calibration curve E[outcome|price]
    tok = build_token_arrays(ledger, guard)          # price path from ALL trades on each token

    # ---- Section A: dataset + the hard temporal constraint -----------------
    span_h = (ledger["timestamp"].max() - ledger["timestamp"].min()) / 3600
    print("=" * 78)
    print("EDGE LEFT AT DETECTION — price-decay of follower edge vs. latency Δ")
    print("=" * 78)
    print(f"\nledger: {len(resolved)} resolved BUY bets across {resolved['token_id'].nunique()} "
          f"tokens / {resolved['market_id'].nunique()} markets")
    print(f"observed wall-clock span: {span_h:.1f} h  <-- HARD CEILING on measurable Δ")
    print("  The ingest is a rolling-window poll of a fast feed (DECISIONS.md); the ledger")
    print("  holds ~1.4h of history, so NO token has trades an hour+ past any entry. Δ>=1h")
    print("  is therefore unmeasurable here (0% coverage), by construction, not by finding.")
    print("\nresolved BUY bets by horizon:")
    for horiz, n in resolved["horizon"].value_counts().items():
        sub = resolved[resolved["horizon"] == horiz]
        liq = has_followable_liquidity(sub, tok).mean()
        print(f"  {horiz:13s}: {n:6d} bets | any followable other-wallet liquidity "
              f"before guard: {liq:.0%}")
    print("\nbaseline market edge (favorite-longshot): mean resolved_value "
          f"{resolved['resolved_value'].mean():.3f} vs mean entry_price "
          f"{resolved['entry_price'].mean():.3f}  (=> +{resolved['resolved_value'].mean()-resolved['entry_price'].mean():.3f}/bet free to any buyer)")

    # ---- Section B: decay curve per horizon (primary bandwidth) ------------
    print("\n" + "-" * 78)
    print(f"DECAY CURVE  (bandwidth = {PRIMARY_BANDWIDTH}s follower fill window; same-cohort)")
    print("  own(coh)=wallet edge on the SAME bets | follow_raw=resolved-price_at (P&L)")
    print("  skill_edge=resolved-E[outcome|price_at] (copyable alpha net of favorite-longshot)")
    print("  null_raw=shuffled-outcome floor | excess=follow_raw-null_raw | retain=follow_raw/own")
    print("-" * 78)
    for horiz in ["micro_crypto", "real_world"]:
        sub = resolved[resolved["horizon"] == horiz].reset_index(drop=True)
        print(f"\n[{horiz}]  n={len(sub)}")
        decay_table(sub, tok, baseline, PRIMARY_BANDWIDTH, rng)

    # ---- Section C: bandwidth sensitivity ----------------------------------
    print("\n" + "-" * 78)
    print("BANDWIDTH SENSITIVITY  (follower_raw edge; coverage in parens) at Δ=1m")
    print("-" * 78)
    for horiz in ["micro_crypto", "real_world"]:
        sub = resolved[resolved["horizon"] == horiz].reset_index(drop=True)
        rv = sub["resolved_value"].to_numpy(dtype=float)
        cells = []
        for bw in SENSITIVITY_BANDWIDTHS:
            pv = follower_prices(sub, tok, 60, bw)
            c = ~np.isnan(pv)
            fol = float((rv[c] - pv[c]).mean()) if c.any() else np.nan
            cells.append(f"h={bw}s: {fmt(fol)} ({c.mean():.0%})")
        print(f"  {horiz:13s} | " + "   ".join(cells))

    # ---- Section D: price-band stratification at realistic latency ---------
    print("\n" + "-" * 78)
    print(f"PRICE-BAND STRATIFICATION  at Δ=1m, bandwidth={PRIMARY_BANDWIDTH}s")
    print("-" * 78)
    for horiz in ["micro_crypto", "real_world"]:
        sub = resolved[resolved["horizon"] == horiz].reset_index(drop=True)
        print(f"\n[{horiz}]")
        price_band_table(sub, tok, baseline, 60, PRIMARY_BANDWIDTH)

    print("\n" + "=" * 78)
    print("Read the verdict in HANDOFF.md 'Edge-left-at-detection'. Re-run this after a")
    print("multi-day ingest/backfill to measure the hour-to-day latencies this 1.4h")
    print("ledger cannot reach, and restrict to validated wallets once samples are deep.")
    print("=" * 78)


if __name__ == "__main__":
    main()
