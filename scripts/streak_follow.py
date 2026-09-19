"""Streak-following backtest — the owner's rule, run as a pooled strategy.

THE RULE UNDER TEST (owner, 2026-07-30): "if a wallet wins 3-4 substantial
bets in a row (>=15% ROI if it hits), follow it — copy its next bet(s)."
Run systematically: EVERY time ANY wallet in the deep real-world sample
completes such a streak, copy its next bet. Pooling across all wallets gives
the strategy the sample size no single watched wallet can have.

PRE-REGISTERED DESIGN, fixed before results were computed:

  Universe     All resolved real-world BUY bets of the 2,094 deepened wallets
               (performance-blind sample), micro-crypto excluded, NO
               time-to-close filter (docs/funnel_conditional_2026-07-30.md —
               tape-derived close filters are forward-looking).
  Substantial  entry price <= 1/1.15 ~= 0.8696 (>=15% ROI if the bet wins).
  Trigger      at bet t when the wallet's last K bets (t-K+1..t) were all
               substantial AND all won, K in {3,4,5,6}. Ride-the-streak: if
               the followed bet wins substantially, the next trigger fires.
  Follow       the wallet's next BUY (t+1). Scored at THE WALLET'S OWN PRICE —
               an optimistic bound (a real copier does worse). Also reported:
               the subset where the followed bet is itself substantial
               (a copier sees the price before copying).
  Costs        frozen per-category taker fee k*p*(1-p) reported beside gross.
  Controls     (a) the same wallets' unconditional bets (their base rate);
               (b) the ANTI-streak arm: next bet after K substantial LOSSES —
                   if both arms move the same way, it is selection, not skill;
               (c) the Xu & Harvey check: price of post-streak bets vs base
                   (streak winners choosing safer odds fakes a hot hand);
               (d) the band-matched excess: each followed bet minus the mean
                   edge of ALL universe bets in its 10c price band — kills
                   favorite-longshot composition;
               (e) event-clustered bootstrap CIs; wallet counts and top-wallet
                   share beside every number.
  KNOWABILITY  Sequence-streaks ignore whether the K wins had RESOLVED before
               bet t+1 was placed (resolution timestamps are not collected).
               With the own-price scoring this makes the whole test an UPPER
               BOUND on the signal: if it reads ~0 here, the implementable
               version is dead a fortiori. A positive would need resolution
               timestamps before being believed.

Read-only; writes only data/interim/funnel/streak_follow.parquet.
Run: flock data/interim/.analysis.lock -c \
       'PYTHONPATH=. .venv/bin/python scripts/streak_follow.py'
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.funnel_conditional import (FUNNEL, build_universe, cell_stats,
                                        fee_cents, log)

SEED = 20260730
SUBSTANTIAL = 1.0 / 1.15          # price ceiling for ">=15% ROI if it wins"
KS = (3, 4, 5, 6)


def run_lengths(ok: np.ndarray, wallet_start: np.ndarray) -> np.ndarray:
    """Consecutive-True run length ENDING at each position, reset at wallet
    boundaries. ok False at t -> 0."""
    n = len(ok)
    pos = np.arange(n)
    lb = np.where(~ok, pos, -1)
    lb = np.maximum.accumulate(lb)
    last_break = np.maximum(lb, wallet_start - 1)
    return pos - last_break


def arm_report(name: str, follow: pd.DataFrame, universe: pd.DataFrame,
               band_mean: pd.Series, rng: np.random.Generator) -> dict:
    s = cell_stats(follow, rng)
    win_rate = float((follow["rv"] > 0.5).mean()) if len(follow) else np.nan
    # band-matched excess: followed bet edge minus its band's universe mean
    if len(follow):
        exc = follow["edge_c"].to_numpy() - band_mean.reindex(
            follow["band"].astype(str)).to_numpy()
        fx = follow.assign(edge_c=exc)
        sx = cell_stats(fx, rng)
    else:
        sx = {k: np.nan for k in ("event_wtd", "ci_low", "ci_high")}
    top_w = (follow.groupby("w", observed=True).size().max() / len(follow)
             if len(follow) else np.nan)
    fee = fee_cents("other", s["mean_price"])
    log(f"  {name:26} n {s['n_bets']:6,} | wallets {s['n_wallets']:5,} "
        f"(top {top_w:5.1%}) | events {s['n_events']:6,} "
        f"(eff {s['eff_events']:7,.0f})\n"
        f"  {'':26} price {s['mean_price']:.3f} | win {win_rate:5.1%} | "
        f"edge bet-wtd {s['bet_wtd']:+.2f}c, event-wtd {s['event_wtd']:+.2f}c "
        f"[{s['ci_low']:+.2f},{s['ci_high']:+.2f}] | fee ~{fee:.2f}c\n"
        f"  {'':26} band-matched excess {sx['event_wtd']:+.2f}c "
        f"[{sx['ci_low']:+.2f},{sx['ci_high']:+.2f}]")
    return dict(arm=name, **{k: v for k, v in s.items()},
                win_rate=win_rate, top_wallet_share=top_w,
                excess_event_wtd=sx["event_wtd"], excess_ci_low=sx["ci_low"],
                excess_ci_high=sx["ci_high"], fee_c=fee)


def main() -> None:
    rng = np.random.default_rng(SEED)
    log("building universe (no time-to-close filter) …")
    bets, wallets = build_universe("none")
    bets = bets.sort_values(["w", "ts"], kind="mergesort").reset_index(drop=True)

    w = bets["w"].to_numpy()
    n = len(bets)
    starts = np.r_[0, np.where(w[1:] != w[:-1])[0] + 1]
    counts = np.diff(np.r_[starts, n])
    wallet_start = np.repeat(starts, counts)

    sub = (bets["price"].to_numpy() <= SUBSTANTIAL)
    win = (bets["rv"].to_numpy() > 0.5)
    s_win = run_lengths(sub & win, wallet_start)
    s_loss = run_lengths(sub & ~win, wallet_start)

    same_next = np.r_[(w[1:] == w[:-1]), False]   # t+1 exists in same wallet

    band_mean = bets.groupby(bets["band"].astype(str), observed=True)["edge_c"].mean()

    log("\n=============== BASE RATE (all universe bets) ===============")
    base = arm_report("base: all bets", bets, bets, band_mean, rng)

    ts_arr = bets["ts"].to_numpy()
    e_arr = bets["e"].to_numpy()

    def clean_follow(trig: np.ndarray, K: int) -> tuple[np.ndarray, float]:
        """Follow indices where the next bet is STRICTLY LATER and in a
        DIFFERENT resolution event than every one of the K streak bets —
        removes the same-event outcome leakage (streak and 'next bet'
        resolving together)."""
        T = np.where(trig)[0]
        F = T + 1
        ok = ts_arr[F] > ts_arr[T]
        for j in range(K):
            ok &= e_arr[T - j] != e_arr[F]
        return F[ok], (1.0 - float(ok.mean())) if len(T) else np.nan

    out = [dict(base, k=0)]
    for K in KS:
        log(f"\n=============== K = {K} consecutive substantial wins ===============")
        trig = (s_win >= K) & same_next
        follow_idx = np.where(trig)[0] + 1
        follow = bets.iloc[follow_idx]
        out.append(dict(arm_report(f"RAW next after {K} wins", follow,
                                   bets, band_mean, rng), k=K, variant="raw"))
        cidx, excl = clean_follow(trig, K)
        log(f"  [same-event / same-moment 'next bets' excluded: {excl:.1%}]")
        cfollow = bets.iloc[cidx]
        out.append(dict(arm_report(f"CLEAN (diff-event, later)", cfollow,
                                   bets, band_mean, rng), k=K, variant="clean"))
        csub = cfollow[cfollow["price"] <= SUBSTANTIAL]
        out.append(dict(arm_report(f"  … clean + next subst.", csub,
                                   bets, band_mean, rng), k=K,
                        variant="clean_sub"))
        atrig = (s_loss >= K) & same_next
        aidx, aexcl = clean_follow(atrig, K)
        out.append(dict(arm_report(f"ANTI clean after {K} losses",
                                   bets.iloc[aidx], bets, band_mean, rng),
                        k=K, variant="anti_clean"))

    res = pd.DataFrame(out)
    res.to_parquet(FUNNEL / "streak_follow.parquet", index=False)
    log(f"\nwrote {FUNNEL / 'streak_follow.parquet'}")
    log("\nRead the band-matched excess and the win-vs-loss symmetry before "
        "reading any raw row. Own-price scoring + sequence streaks make every "
        "positive an UPPER bound.")


if __name__ == "__main__":
    main()
