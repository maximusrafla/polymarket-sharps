# The copy verdict — corrected

**2026-07-28.** This supersedes the conclusion in `docs/copy_lag_simulation.md`.
That document's *method* stands; its **headline verdict does not**. Two errors,
one in the null and one in the cost, both pushed the answer the same way.

---

## What the original run said, and why it was wrong

It reported **copyable alpha = +0.0003 (p = 0.10)** and concluded no copyable
edge survives a 2-minute lag. Two problems:

### 1. The null could not see the effect it was testing for

The null shuffled outcomes **within entry-price bins among only these five
wallets' own bets**. All five are certified skilled, so their pooled outcomes at
any given price are already elevated *by that skill*, and a within-pool shuffle
preserves the elevation. The null therefore absorbed the very thing it was
supposed to detect.

**The tell is mechanical and needs no hindsight: at Δ = 0 the reported alpha was
exactly 0.000.** At zero lag, paying the identical price for the identical bet, a
follower is the wallet. Any null that scores that as zero edge is broken on its
face, whichever answer you were hoping for.

That null answers *"can you tell which of these wallets' bets to copy?"* (no). It
does not answer *"does copying them beat the market?"* — which is the question.
The right comparison is the **population baseline**, `E[outcome | price]` fitted
over all 3,385,372 real-world bets: the same baseline the certification gate uses.

### 2. The adverse tick was over-charged ~10×

The run charged a **1¢** adverse tick. The markets actually quote:

| tick size | share of bets |
|---|---|
| **0.001** | **97.0%** |
| 0.0025 | 1.7% |
| 0.01 | 1.3% |

⚠️ **Honesty note.** The null was re-specified *after* seeing a null result, which
is exactly the move that manufactures findings. The defence is that the Δ = 0 tell
is visible without reference to the answer — but discount accordingly, and treat
everything below as a hypothesis for the forward test, not a result.

---

## The corrected numbers

Same-cohort (both arms on exactly the bets that have a price at that Δ),
guard-off 99%-coverage sample, real per-market fees, actual quoted tick,
CIs resampling whole markets.

### Where the edge goes, at a 2-minute lag

| | cents/share |
|---|---|
| wallet's gross skill edge | **+4.54¢** |
| **lost to price drift in 120 s** | **−1.90¢** |
| = follower's gross skill | +2.64¢ |
| taker fee (identical for both) | −0.50¢ |
| spread / adverse tick | −0.10¢ |
| **= follower net** | **+2.02¢** [+1.07, +2.99] |

**Almost the entire loss is drift.** Fees and spread together are ~0.6¢ — barely a
quarter of what the price movement takes.

### Drift is front-loaded, so a faster bot does not rescue it

| lag | drift against you | % of edge gone |
|---|---|---|
| **1 min** | **1.80¢** | **37%** |
| 2 min | 1.99¢ | 41% |
| 5 min | 2.16¢ | 45% |
| 15 min | 2.47¢ | 51% |
| 1 h | 2.33¢ | 49% |

**72% of all the drift that ever happens occurs inside the first 60 seconds**, then
the curve flattens. Halving the lag from 2 min to 1 min buys back **0.19¢**. This
closes the "run a faster bot" hypothesis: by the time a trade is visible on-chain,
the price has already moved.

### Follower return vs wallet return at a 2-minute lag

Return on amount staked, equal size per bet so no single bet dominates, 2-minute
lag, all costs charged:

| wallet | avg price | their ROI | **your ROI** | 95% CI |
|---|---|---|---|---|
| `0x1ee9a5fc0966` | 64¢ | 35.8% | **+31.9%** | [+9.7%, +62.9%] |
| `0x69ea0d77ef34` | 89¢ | 17.0% | **+13.8%** | [+5.6%, +26.4%] |
| `0x09bed1976600` | 91¢ | 4.4% | +4.5% | [−3.9%, +16.1%] |
| `0xe542afd3881c` | 27¢ | **60.0%** | **+2.4%** | [−29.4%, +43.5%] |
| `0x83255595ba1f` | 53¢ | 13.3% | **−16.6%** | [−25.0%, −8.1%] |
| **pooled** | 54¢ | 36.2% | **+9.3%** | [−4.7%, +25.7%] |

---

## The finding, in one line

**The highest-ROI wallets are the least copyable, and the modest ones are the most
copyable.**

`0xe542afd3881c` earns **60%** and hands a follower **2.4%**. It buys at 27¢, and a
longshot price moves violently the moment size touches it — by the time the trade
is visible the cheap price is gone. Its edge is real and essentially unreachable.

The two that survive buy at **64¢ and 89¢**: boring, liquid, slow-moving prices
where two minutes barely moves the book.

`0x83255595ba1f` is worse than useless to a follower at **−16.6%**, and a second,
completely independent pass reached the same verdict by classifying it: it is an
**in-play football trader** (~85% of its signals from ~10 live over/under markets),
which is the shape least reachable by anyone arriving after the print.

---

## What this does NOT establish

- **The pooled CI includes zero** (−4.7% to +25.7%). Only two wallets individually
  clear it.
- **Those two were named after seeing the results.** Winner's curse applies, and
  five wallets is a small enough set that two survivors is not a surprising draw.
- **One tick is a floor, not the truth.** It prices a trivially small order at the
  top of the book, so every number here is the most generous reading available to
  a follower.
- **Their pooled ROI is +36% equal-weighted but +8% stake-weighted**, so their big
  bets are much worse than their small ones, and their headline lifetime return
  overstates what the record actually contains.

## Next

The honest instrument is a pre-registered forward test: declare the tiers in
advance, including the ones expected to fail, so "they will fail forward" is a
falsifiable prediction rather than a quiet exclusion after the fact.

---

# CORRECTION 2 (2026-07-28, later): the midpoint-vs-ask artifact

Everything above understated the follower's cost. **The trade tape's `entry_price`
is a post-slippage taker VWAP** — what the wallet actually paid, at or above the
ask — while **CLOB `/prices-history` returns a book MIDPOINT**. So the wallet was
charged its real fill and the follower was credited at the mid.

Measured over 133,003 bets:

| | |
|---|---|
| wallet paid above the mid | **92.8% of bets** |
| mean `entry_price − mid_at_entry` | **+3.23¢** |
| what the follower was charged (quoted tick) | 0.10¢ |
| **unearned edge handed to the follower** | **≈ +3.13¢/share** |

That is larger than most of the edges reported above. **The tell was in the output
and I missed it: 14 of 46 wallets showed the follower BEATING the wallet on
identical bets** — impossible when you buy the same thing two minutes later at a
worse price. That impossibility should have been checked before any number was
written down, and it is now an assertion.

## Corrected: charge the follower the spread the wallet demonstrably paid

`max(entry_price − mid, one tick)` per bet, plus real per-category fees (the
earlier pass also zeroed fees across the whole `other` bucket — 48.8% of the tape
— by extrapolating from geopolitics, far beyond what was verified; reverted).

**Result across the certified set, guard-on (conservative) frame:**

| | |
|---|---|
| wallets scored | 37 |
| **copyable (CI > 0)** | **11** (was 38 before the fix) |
| **follower-beats-wallet** | **0** ✓ |
| median edge retained after lag | **78%** of the wallet's own |

| wallet | markets | own | follower net | 95% CI | category |
|---|---|---|---|---|---|
| `0x5ac8f582c98b` | 107 | 18.13¢ | +16.43¢ | [+0.13, +30.21] | other |
| `0xa8638d8d7a00` | 86 | 12.00¢ | +10.58¢ | [+3.82, +17.26] | **micro_crypto** |
| `0xdceecdb83fcb` | 159 | 11.41¢ | +8.71¢ | [+1.87, +13.97] | other |
| `0x253da8157571` | 221 | 7.46¢ | +6.56¢ | [+2.62, +10.10] | other |
| `0xc9b6227a2959` | 428 | 9.87¢ | +5.97¢ | [+1.47, +10.53] | other |
| `0x9fc043287797` | 114 | 6.14¢ | +5.61¢ | [+2.93, +7.82] | other |
| `0x69ea0d77ef34` | 447 | 6.71¢ | +5.23¢ | [+3.07, +7.51] | other |
| `0x92d0cb81e6c8` | 812 | 8.37¢ | +4.19¢ | [+0.77, +7.47] | other |
| `0xd06f0f7719df` | 816 | 5.72¢ | +3.52¢ | [+1.24, +5.49] | other |
| `0x05c5aab002fa` | 165 | 4.31¢ | +3.35¢ | [+2.02, +4.84] | politics |
| `0x433723566347` | 1611 | 5.79¢ | +2.32¢ | [+0.44, +4.10] | other |

## Two conclusions above are now WRONG

1. **`0x1ee9a5fc09` is NOT copyable** — `+1.61¢ [−1.05, +4.22]`. It was the
   headline pick at "+31.9%"; that was largely the midpoint artifact.
   `0x69ea0d77ef` survives at `+5.23¢ [+3.07, +7.51]`.
2. **The geopolitics signature dissolved.** At n=2 it looked like a screen. Across
   37 the copyable set is 9 `other` / 1 politics / 1 micro-crypto — the sample's
   base rate. **There is no category shortcut for finding copyable wallets.**

## On the sequence of corrections

Four now. The first three (broken null, 10× tick, fee) all moved the answer toward
"copying works". This one moves hard the other way and is **larger than all three
combined**. Read that as evidence of drift toward a preferred conclusion, not as a
wash — and as the reason the pre-registered forward test, which none of these
corrections can touch, is the only real arbiter.

⚠️ Scope: this correction ran on the **guard-on** frame (~51% coverage, the
conservative one), which also dropped two wallets below the 150-bet floor. It is
not directly comparable to the guard-off +4.415¢ figure above.
