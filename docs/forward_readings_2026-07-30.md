# Forward readings — the frozen identification cohorts, 2026-07-30

**Why this doc exists.** Every score job in this repo writes its scoreboard and
then `git checkout`s it away so the tree stays clean; the committed scoreboards
therefore permanently show their freeze-time "AWAITING DATA" state, and the live
numbers appear only in `data/cron.log`. The "SCOREBOARD HAS DATA" banner had
fired **nine times with nobody reading it**. This session re-ran every scorer by
hand (under the same locks the cron uses), committed the snapshots beside this
doc, and read them in each cohort's pre-registered reading order.

Data-as-of: `s_sig2c`/`sports_forward` re-fetched at read time (2026-07-30
~02:00Z); `slow_forward` scored from its 2026-07-29T13:00Z fetch (slow markets;
the staleness is immaterial).

---

## 1. `s2_core` — the re-opened sports cohort reads ZERO so far

The strongest retrospective identification claim in the repo (10 wallets, ≥2¢
held-out edge, p<0.05 under all five baseline variants). Forward, 3.5 days:
11,599 bets from 5 of 12 wallets, 868 events, **176.7 effective events**:

> event-weighted **−0.4¢**, 95% CI **[−2.2¢, +1.6¢]**, p = 0.62
> (bet-weighted +3.6¢; the pre-registered statistic is event-weighted)

The CI's upper edge already sits *below* the ≥2¢ retrospective claim. Not final
— 7 of 12 haven't traded, and `s2_slow` (the only copy-relevant tier) has 0
forward bets — but the cohort's forward start is a null, and it is essentially
all fast in-play volume, the uncopyable kind.

## 2. The parent sports tiers — the unvetted pool leans positive

`s_all` (174 candidates, pre-registered as "expected to be mostly noise"):
67,342 bets, 1,338 events, 188.5 effective: **+1.6¢ [+0.5, +2.6], p=0.002**,
all of it in the ≤2-day in-play stratum (the >2d stratum is −3.6¢, n.s.).
The 2026-07-26 reading was +3.1¢ on 18.8 effective events; with 10× the
evidence the point halved and stabilized positive.

Read with care: the *vetted* selection (`s2_core`) reads 0 while the *unvetted
population* reads +1.6¢ — the fourth independent instance of this repo's
recurring result: **the faint edge lives in the population/markets, and
wallet-level selection out of it is noise.** Do not mine `s_all`'s forward
winners into a new cohort; that is the winner's-curse machine that produced
the last three dead shortlists. `s_core`/`s_wide`: still 0 events.

## 3. Slow forecasters — headline unreadable, wide tier flat

t12 (headline): 22 bets, **2.2 effective events**, +0.33 [−0.12, +0.76] — no
reading. t30/t52: +0.25/+0.14 on 3.2/3.8 effective events — too thin to mean
anything. t699 (15,210 bets, 167 wallets, 40 events, 8.2 effective): **−0.05
[−0.11, +0.01], p=0.95.** Months from a verdict, as expected at freeze time.

## 4. The real-world copy cohort — genuinely awaiting data

56 signals from 3 of 9 wallets, 7 scored, 5.4 effective events. Nothing here
yet; the pooled +2¢ result (`rw5_pooled`, per the 2026-07-29 audit's reading
correction) reads in weeks-to-months. It is the one forward reading that could
still move the copy conclusion.

---

## What changed in the repo's live questions

| question | status after this read |
|---|---|
| Copy the P1-certified 26 (micro) | **CLOSED, forward** — all arms deeply negative |
| Copy/track `s2_twelve` in-play sports | Negative forward at every arm; `s2_slow` unread |
| `s2_core` identification (sports) | Null so far on 177 eff events; watch, don't extend |
| Slow-forecaster tiers | Unreadable headline; wide tier flat-negative |
| Pooled +2¢ real-world copy (`rw5_pooled`) | **OPEN — the one unread forward test, weeks out** |
| Wallet-level selection anywhere | Failed every forward/null test to date |

**Process note.** The scoreboard-revert design nearly buried all of this. The
convention stands (snapshots are committed by hand), but the reading duty is
now explicit: when a banner fires, someone must run the score and commit the
snapshot. This doc and the snapshots beside it are the first such commit.
