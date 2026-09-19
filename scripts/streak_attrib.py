"""Attribution checks for the CLEAN streak-follow residual (+1.8c at K=3).

Three questions, pre-registered before running:
  1. TRAIT vs STATE — is the post-streak excess just "streaky wallets are
     better wallets"? Contrast: follow-bet edge minus the SAME WALLET's own
     overall mean edge (event-weighted, event bootstrap). ~0 -> all trait
     (the rule is a noisy wallet-quality screen, and the per-wallet
     instability results apply); ~ +1.8c -> genuine state dependence.
  2. COMPOSITION — does it survive matching on (category x coarse price band)
     instead of price band alone?
  3. NARRATIVE LADDERS — the event unit splits rolling-deadline ladders
     ("...-by-march-15" vs "...-by-june-30") and weather buckets into
     different "events" that co-resolve on one story. Guard: follow bet's
     event FAMILY (event key minus trailing date/threshold tokens) must
     differ from all K streak bets' families. Does the excess survive?

Read-only; prints only. Run under data/interim/.analysis.lock.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.funnel_conditional import build_universe, cell_stats, log
from scripts.streak_follow import SUBSTANTIAL, run_lengths

SEED = 20260730
K = 3

# strip trailing date-ish / threshold-ish tail from an event key:
#   us-strikes-iran-by-march-15  -> us-strikes-iran
#   highest-temperature-in-denver-july-27-88-89f -> highest-temperature-in-denver
_TAIL = re.compile(
    r"(-(by|on|before|in)-.*$)"
    r"|(-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-\d.*$)"
    r"|(-\d{4}-\d{2}-\d{2}.*$)|(-\d+(-\d+)?f?$)")


def family_of(key: str) -> str:
    return _TAIL.sub("", key)


def main() -> None:
    rng = np.random.default_rng(SEED)
    bets, wallets = build_universe("none")
    event_names = build_universe.last_event_names
    bets = bets.sort_values(["w", "ts"], kind="mergesort").reset_index(drop=True)

    w = bets["w"].to_numpy()
    n = len(bets)
    starts = np.r_[0, np.where(w[1:] != w[:-1])[0] + 1]
    wallet_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    sub = bets["price"].to_numpy() <= SUBSTANTIAL
    win = bets["rv"].to_numpy() > 0.5
    s_win = run_lengths(sub & win, wallet_start)
    same_next = np.r_[(w[1:] == w[:-1]), False]
    ts_arr = bets["ts"].to_numpy()
    e_arr = bets["e"].to_numpy()

    fam_names = np.array([family_of(k) for k in event_names], dtype=object)
    fcode = {f: i for i, f in enumerate(pd.unique(fam_names))}
    f_of_e = np.array([fcode[f] for f in fam_names], dtype=np.int32)
    f_arr = f_of_e[e_arr]
    log(f"{len(event_names):,} events collapse to {len(fcode):,} families")

    trig = (s_win >= K) & same_next
    T = np.where(trig)[0]
    F = T + 1
    ok = ts_arr[F] > ts_arr[T]
    for j in range(K):
        ok &= e_arr[T - j] != e_arr[F]
    okfam = ok.copy()
    for j in range(K):
        okfam &= f_arr[T - j] != f_arr[F]
    follow = bets.iloc[F[ok]]
    follow_fam = bets.iloc[F[okfam]]
    log(f"clean follows {len(follow):,}; after family guard {len(follow_fam):,} "
        f"({1 - okfam.sum() / max(ok.sum(), 1):.1%} of clean removed)")

    band_mean = bets.groupby(bets["band"].astype(str), observed=True)["edge_c"].mean()
    catband_mean = bets.groupby(
        [bets["category"].astype(str), bets["coarse"].astype(str)],
        observed=True)["edge_c"].mean()
    wallet_mean = bets.groupby("w", observed=True)["edge_c"].mean()

    def report(name, fr, ref):
        fx = fr.assign(edge_c=fr["edge_c"].to_numpy() - ref)
        s = cell_stats(fx, rng)
        log(f"  {name:44} {s['event_wtd']:+.2f}c "
            f"[{s['ci_low']:+.2f},{s['ci_high']:+.2f}]  "
            f"(n {s['n_bets']:,}, events {s['n_events']:,})")

    log("\n---- attribution of the K=3 CLEAN excess ----")
    report("band-matched (replication)", follow,
           band_mean.reindex(follow["band"].astype(str)).to_numpy())
    report("(category x band)-matched", follow,
           catband_mean.reindex(pd.MultiIndex.from_arrays(
               [follow["category"].astype(str),
                follow["coarse"].astype(str)])).to_numpy())
    report("within-WALLET (trait removed)", follow,
           wallet_mean.reindex(follow["w"]).to_numpy())
    report("family guard + band-matched", follow_fam,
           band_mean.reindex(follow_fam["band"].astype(str)).to_numpy())
    report("family guard + within-wallet", follow_fam,
           wallet_mean.reindex(follow_fam["w"]).to_numpy())

    log("\n---- calendar stability (band-matched excess, clean K=3) ----")
    years = pd.to_datetime(follow["ts"], unit="s").dt.year
    for yr in sorted(years.unique()):
        fy = follow[years.to_numpy() == yr]
        report(f"  {yr}", fy, band_mean.reindex(fy["band"].astype(str)).to_numpy())


if __name__ == "__main__":
    main()
