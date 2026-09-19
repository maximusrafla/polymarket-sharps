# Streak-following ("hot hand") — the owner's rule, tested at scale

**The rule** (owner, 2026-07-30): *"if a wallet wins 3–4 substantial bets in a
row (≥15% ROI if it hits), follow it."* Tested as a pooled systematic strategy
over the deep real-world tape (3.39M resolved BUYs, 2,094 performance-blind
wallets): every time ANY wallet completes such a streak, copy its next bet.
Scripts: `scripts/streak_follow.py` (pre-registered design in docstring),
`scripts/streak_attrib.py` (attribution + stability).

**Context from the literature** (fetched 2026-07-30): Xu & Harvey 2014
(565,915 bets) — streak winners keep winning *because they switch to safer
odds*, a win-rate illusion with no ROI content; the copy-trading literature —
leaderboard-picked leaders don't deliver for copiers; the 13F-cloning
literature — copying works when the leader's holding horizon vastly exceeds
the copy lag (cloning Buffett at a 1-month lag worked for 30 years). Our
repo's own results reproduce those boundary conditions exactly.

## 1. The raw rule is an illusion — and now we know which one

| K wins | follow-bet win rate | apparent excess edge |
|---|---:|---:|
| 3 | 79.7% | +8.1¢ |
| 6 | 87.6% | +12.0¢ |

Looks spectacular. It isn't: **65–80% of "next bets" are placed on the SAME
resolution event as the streak bets, before anything has resolved** (buying
Team A across three related markets, then a fourth). The streak and the "next
bet" are one outcome counted several times. This is precisely what watching a
hot wallet by eye feels like — 4 wins that are really 1. (Not Xu & Harvey's
mechanism, incidentally: post-win bets here got *cheaper* not safer — the
leakage dwarfs the odds-selection effect.)

## 2. Cleaned (next bet strictly later AND in a different event), something real remains

Band-matched excess (vs all universe bets at the same price level),
event-bootstrap CIs:

| K | n follows | excess | 95% CI |
|---|---:|---:|---|
| 3 | 91,775 | **+1.80¢** | [+1.41, +2.16] |
| 4 | 54,330 | +2.23¢ | [+1.74, +2.69] |
| 5 | 34,736 | +2.57¢ | [+2.01, +3.14] |
| 6 | 23,520 | +2.90¢ | [+2.23, +3.54] |

Anti-streak arm (next bet after K substantial losses): **−1.1¢** — slump
wallets underperform their price band. Attribution of the K=3 excess:

- **(category × band)-matched: +1.92¢** — not category composition.
- **Within-wallet (trait removed): +1.44¢ [+1.09, +1.79]** — mostly STATE,
  not "streaky wallets are better wallets". Genuine post-win elevation.
- Narrative-family guard: −0.02¢ change — but the guard is weak (regex
  collapses only 166k events → 137k families, removes 0.7% of follows), so
  correlated-narrative leakage is *not* excluded by this check.
- **Calendar: positive every year, and DECAYING — 2024 +3.80¢, 2025 +2.34¢,
  2026 +1.29¢ [+0.85, +1.75].**

## 3. Verdict

The phenomenon is real: recently-winning real-world wallets' next picks run
~1.5–2¢ better than their price band, and it is state-dependent, not just
wallet quality. It is almost certainly the same animal as the pooled copy
edge in `docs/redteam_realworld_copy_2026-07-29.md` (+2¢) — two doors into
one room: *recently-successful real-world wallets' selections carry a small,
pooled, market-level edge.*

**As a standalone selection rule the margin is thin**, for three reasons
recorded before anyone gets excited:

1. **Every number above is an upper bound**: fills at the wallet's own price
   (a real copier does worse — the audit's executable-anchor haircut applies),
   and sequence-streaks ignore whether the K wins had *resolved* before the
   next bet (resolution timestamps are not collected; a live implementation
   fires on resolved streaks, a different and rarer condition).
2. **The 2026 margin (+1.29¢ gross) ≈ the taker fee (~1.24¢)** before spread.
   The decay 2024→2026 is what an edge being arbitraged away looks like.
3. The K=5–6 rows are larger (+2.6–2.9¢) but thinner, and the per-year split
   was only run at K=3.

**If pursued**: the honest instrument is a fresh pre-registration triggered by
*resolved* streaks, scored forward. The question it would answer, and the only
one still open here, is whether streak-triggering (recency) beats certification
(long history) as the selection signal.
