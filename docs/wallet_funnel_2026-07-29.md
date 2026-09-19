# Wallet funnel — structural filters, 2026-07-29

> ### ⛔ 2026-07-30 — the bet universe under F1–F3 and the whole PERFORMANCE
> ### stage carry a forward-looking artifact. Read `docs/funnel_conditional_2026-07-30.md` first.
>
> "Copyable bet" was defined as `tape_t_max − entry ≥ 24 h`, and `tape_t_max`
> is the **last observed trade** — so the rule conditions each bet on
> post-entry market activity. Removing it *inverts* the price-band structure
> (longshots +4.5¢ → −0.0¢; favorites −4.8¢ → +2.3¢). The performance stage's
> numbers (the 65% base rate, the winner's-curse table, the six provisional
> wallets) were computed inside that universe and should not be quoted until
> re-derived on a scheduled (Gamma) close time. The structural logic and power
> arithmetic below stand; the per-bet universe does not.

**Why the split matters, and it is the whole design.** Structural filters
(speed, sample size, activity, tradeability) are safe: they cannot manufacture a
winner, because they never look at performance. Performance filters (profit rate,
consistency) are dangerous: applied to a large pool they *guarantee* apparent
winners even when none exist. So every structural filter is applied first and in
full, and performance is only touched afterwards — selected on one half of a
wallet's history and **verified on the other half it has never seen**.

That last step is not another filter. It is the exam, and it is what
`docs/redteam_realworld_copy_2026-07-29.md` showed was missing: 11 wallets read
"copyable", and re-running with a different random seed produced a different 11.

**Process.** Each filter is proposed with its measurement, its calibration
against a known case, its counterfactual at several thresholds, and its failure
modes — and is approved by the owner before the next is designed. Rejected
attempts are recorded here too, because a filter that failed calibration is
evidence about the data.

**Nothing is deleted.** Wallets that fail a filter, and wallets that cannot be
measured by it, are flagged rather than dropped (CLAUDE.md's standing invariant).

---

## The funnel so far

| step | wallets | note |
|---|---:|---|
| deep real-world wallets | **2,094** | `realworld_deepen`, performance-blind sample |
| — scorable for speed | 1,344 | 750 carried as `speed_unknown` |
| **F1** median bet has ≥24 h to market close | **1,046** | 298 cut |
| **F2** ≥100 distinct copyable markets | **199** | 847 cut |
| **F3** ≥44 *effective* independent events | **79** | 120 cut — and see the F2 correction |
| **F4** traded within 30 days | **74** | 5 cut |
| ~~F5~~ tradeable spreads | — | **rejected as a filter**, kept as a sensitivity |

Survivor profile: median **712 bets**, **188 markets**, **68 effective events**,
last traded **1.2 days** ago.

---

## FILTER 1 — the wallet must trade markets slow enough to copy

**Rule.** Keep a wallet if the **median time from its bet to the market's close is
≥ 24 hours**.

**Why.** A follower acts ~2 minutes behind. If the market resolves in under a day,
whatever the wallet reacted to has already played out. This is the only filter that
is about whether the *opportunity* exists at all — skill is irrelevant if the
window is closed.

**Measurement.** `tape_t_max` (last observed CLOB trade in that market, from
`data/interim/market_meta.parquet`) minus the bet timestamp. `tape_t_max` is a
*lower* bound on true close, so time-to-close is if anything understated — the
filter errs toward calling wallets fast, which is the conservative direction.

**Calibration.** `0x83255595ba1f` is the one wallet independently confirmed
uncopyable, by two separate routes (an in-play football trader; measured at
−1.44 ¢/share to a follower). Under this rule:

> median **0.8 hours** to close · **0.2%** of its bets have 24 h+ · **1st percentile**

**Counterfactual.**

| threshold | kept | % |
|---|---:|---:|
| ≥6 h | 1,199 | 89% |
| ≥12 h | 1,114 | 83% |
| **≥24 h** | **1,046** | **78%** |
| ≥48 h | 949 | 71% |
| ≥72 h | 837 | 62% |

**Two rejected measurements, recorded because they are evidence.**

1. *Median gap between repeat bets in the same market* (proposed by the analyst).
   Failed: the known in-play trader scored 19.7 %, **below the 90th percentile** —
   it bets across a 90-minute match, not in 10-minute bursts.
2. *Share of recent bets in the wallet's top 5 markets* (concentration). Failed
   worse: the known in-play trader is at the **18th percentile** — 473 markets in
   30 days, 3.6 bets each. Its "~10 markets" reputation came from a 7-day probe of
   detected *signals*, not from how it trades.

Both were rejected **before** any performance was looked at. The lesson for a
future auditor: a plausible structural proxy is not a validated one, and the
calibration case is what separates them.

**Failure modes.** (a) Coverage is 30 % of bets — 750 of 2,094 wallets have too
little timing data to score, and are flagged `speed_unknown` rather than dropped;
their disposition is deferred. (b) A wallet that is mostly slow but occasionally
trades same-day markets passes, which is intended — the bet-level rule handles
those individually rather than discarding the wallet.

---

## FILTER 2 — the wallet must have enough independent markets to prove an edge

**Rule.** Keep a wallet with **≥100 distinct markets** among its copyable
(24 h+) bets.

**Why, from first principles.** A bet pays 1 or 0, so a single bet carries ~0.5 of
noise. An edge is only distinguishable from luck when it exceeds ~2 standard
errors, and the independent unit is the **market** (bets in one market share one
resolution), so the requirement is `n ≥ (2 × 0.5 / edge)²`:

| to detect | markets needed | wallets that have it |
|---|---:|---:|
| 2 ¢/share | 2,500 | **0** of 1,046 |
| 3 ¢/share | 1,111 | **0** |
| 5 ¢/share | 400 | 15 |
| **10 ¢/share** | **100** | **199** |
| 15 ¢/share | 44 | 666 |

**This table is the single most important result in the funnel.** No wallet in this
dataset has the history to prove a 2–3 ¢ edge — which is precisely the effect size
the 2026-07-29 audit was measuring, and precisely why its per-wallet rankings
failed their null. That was never a methodology failure; the data cannot carry a
per-wallet claim at that size.

It also defines what this search *can* honestly find: wallets earning roughly
**10 ¢/share or more (~20 % ROI at typical prices)** — which is the stated target.

**Choosing 100.** ≥400 leaves 15 wallets, too few to work with, and would only
license claims about ~10 % ROI wallets, below the target. ≥44 keeps 666 but a
wallet "proving" a 20 % edge on 44 markets is exactly the artifact that dissolved
under testing. 100 is the point where the claim you can make matches the claim you
want to make.

**Checked: does F2 quietly undo F1?** Requiring many markets could select
high-volume wallets, which tend toward faster markets. Measured:
**Spearman(distinct markets, median hours-to-close) = −0.011, p = 0.71.** No
relationship. Median time-to-close moves 399 h → 262 h, still ~11 days. F2 does not
re-admit fast traders.

**Failure modes.** (a) The power calculation assumes markets are independent;
correlated markets (same event, same league, same day) inflate the effective count,
so 100 markets may be worth fewer than 100 in practice — a later concentration
filter should address this. (b) It selects for wallets with long histories, so a
genuinely sharp recent entrant is excluded. That is accepted: you cannot validate
what you cannot measure.

---

## FILTER 3 — enough *effective* events, and a correction to Filter 2

> ### ⚠️ F2's power table was WRONG. This filter is the repair.
>
> F2 counted distinct **markets** as independent observations. They are not. Bets
> cluster inside events, and clustered bets carry less information than their count
> suggests. Measured on the 199 F2 survivors:
>
> | | median |
> |---|---:|
> | distinct markets | 161 |
> | distinct resolution events | 133 |
> | **effective independent events (Kish)** | **39** |
>
> Markets barely collapse into events (1.13 markets/event). The collapse is **bet
> concentration** — a wallet touches 133 events and pours most of its bets into a
> few. So the effective sample is 39, not 161, and the detectable edge is
> `2 × 0.5 / √39` = **16 ¢/share, not the 10 ¢ F2 claimed.**
>
> Corrected table:
>
> | to detect | effective events needed | wallets that have it |
> |---|---:|---:|
> | 5 ¢/share | 400 | **0** of 199 |
> | 10 ¢/share (~20 % ROI) | 100 | **12** |
> | 15 ¢/share (~30 % ROI) | 44 | **79** |
>
> This is the same error as the median-vs-mean mistake in
> `docs/redteam_realworld_copy_2026-07-29.md` §2.5, in different clothing: taking a
> summary statistic at face value without asking what distribution sits under it.
> Recorded rather than quietly patched.

**Rule.** Keep a wallet with **≥44 effective independent events** (Kish, over its
copyable bets) — superseding F2's raw market count, which is retained above only
because the funnel was built in that order.

**Why 44.** It is the sample that supports an honest claim about a **15 ¢/share
(~30 % ROI)** edge — the target profile. Requiring 100 would license 10 ¢ claims but
leaves **12 wallets**, too thin to then split in half for the held-out exam.

**Measured bet-weighted, on purpose.** Equal dollars per bet is how a copier would
actually experience these returns, so a wallet's concentrated events should
dominate the statistic exactly as they would dominate the P&L.

**Failure modes.** (a) `sports_events.resolution_event` does not group weather
buckets — `...denver-july-27-88-89f` and its sibling thresholds each become their
own event — so true independence is **lower** than measured and 44 is a floor, not a
ceiling. (b) Kish is a summary of the clustering, not a substitute for
cluster-robust inference; the later performance step must still bootstrap over
events.

---

## FILTER 4 — still trading

**Rule.** Keep a wallet that has traded within **30 days**. `79 → 74`.

**Why.** You cannot follow a wallet that stopped. This is the only filter about
actionability rather than evidence.

**What it actually showed: almost nothing, and that is the result.** Median survivor
last traded **1.2 days** ago; 94 % within a month. F1–F3 had already selected for
active wallets, because 44 effective events cannot be accumulated while idle. 30
days cuts 5. A 7-day rule would cut 13, but a forecaster betting month-long markets
can idle a week without meaning anything, so tightening would select against the
target profile.

**Failure mode.** Measured against the tape's collection date (2026-07-27). These
numbers age; re-measure before acting on them later.

---

## FILTER 5 — tradeable spreads — **PROPOSED AND REJECTED**

**Rule considered.** Keep a wallet whose markets carry a tight bid-ask spread.

**Measured** on 693 live CLOB books across 67 of the 74 survivors (39 % of recently
traded tokens still open):

| per-wallet median half-spread | |
|---|---:|
| median | **0.50 ¢** |
| p75 | 1.00 ¢ |
| p90 | 1.30 ¢ |
| max | 9.75 ¢ |

**Rejected, because the trade is bad.** A 2 ¢ threshold cuts **4 wallets** while
leaving **25 of 74 unscorable** for want of live books. Shrinking the pool by 5 %
and going blind on 34 % of it, to remove wallets the earlier filters had already
made rare, is not worth it.

**Kept as a measurement instead:** conditioning on bets whose spread is under 2 ¢
raises the follower's measured edge by **+36 %** (1.98 ¢ → 2.69 ¢, retaining 74 %
of bets; `docs/redteam_realworld_copy_2026-07-29.md`). That is the cleanest
quantification in the repo of how much of the copy edge is spread-driven, and it is
a statement about the estimator, not a rule to apply.

---

## The structural funnel is closed

**2,094 → 74.** Median survivor: **712 bets, 188 markets, 68 effective events, last
traded 1.2 days ago.**

No further structural filter is proposed. The remaining candidates (category
concentration, sizing patterns, wallet age) either duplicate F3 or shade into
performance, which must be handled by the split-half design below rather than by
another cut.

> ### ⛔ THE BOUNDARY ON EVERY PERFORMANCE FILTER FROM HERE
>
> The 74 survivors have a median of **68 effective events**. Split in half for the
> exam, that is **~34 events per half**, where the smallest edge distinguishable
> from luck is `2 × 0.5 / √34` ≈ **17 ¢/share, ~30 % ROI**.
>
> A performance filter specified below that — "profitable", "wins over 55 %",
> "positive edge" — **will** return survivors, and they **will** be noise. That is
> not a caution, it is arithmetic, and it is precisely what produced the 11
> disappearing wallets in the red-team audit. Filters must ask *how large* the edge
> is, not *whether* it is positive.

---

## PERFORMANCE STAGE — run, and it does not produce a defensible shortlist

Owner's spec: a 15 % ROI floor, "consistent, not one huge bet", on the first half
only, verified on the second. Three things came out of it, in order of importance.

**1. ROI % is price-confounded and must not be the filter.**
`Spearman(first-half ROI, average price paid) = −0.57`. It ranks *how cheap the
tickets are*, not skill. Of the 38 wallets clearing a 15 % ROI floor, **16 have a
median bet of −100 %** — they lose on most trades and are carried by rare longshot
hits. That is exactly the pattern the owner ruled out. Cents-per-share is better but
**not clean either** (`ρ = −0.47`); an earlier version of this note called that
"near zero", which was wrong.

**2. The filter barely beats the base rate.**

| | share with a positive held-out edge |
|---|---:|
| **all 74 survivors (base rate)** | **65 %** |
| after a ≥15 % first-half ROI filter | 79 % (n=38) |
| after a ≥7 ¢/share first-half filter | 81 % (n=16) |

At the 7 ¢ bar, 13 of 16 held up where chance predicts ~10. **13 vs 10 is not a
result.** Adding the owner's consistency requirement makes persistence *worse*
(79 % → 73 %).

**3. Persistence peaks at 7 ¢/share and DECLINES above it** — the winner's curse,
the same shape the 2026-07-26 audit diagnosed in the sports 10 ¢ floor:

| first-half edge ≥ | wallets | held-out edge >0 | kept ≥ half |
|---|---:|---:|---:|
| 2 ¢ | 36 | 78 % | 50 % |
| 5 ¢ | 24 | 79 % | 54 % |
| **7 ¢** | **16** | **81 %** | **69 %** |
| 10 ¢ | 5 | 60 % | 60 % |
| 15 ¢ | 2 | 50 % | 50 % |

A stricter magnitude floor makes the set worse, not better.

**The six wallets clearing every criterion** (≥7 ¢, profitable median bet, edge
survived into the held-out half). Provisional, not proven:

| wallet | edge 1st → 2nd | median bet | avg price |
|---|---|---:|---:|
| `0x933ca00f565b` | 11.9 ¢ → 9.7 ¢ | +7.5 % | 0.6 |
| `0x630f096a6333` | 10.6 ¢ → 6.9 ¢ | +40.9 % | 0.4 |
| `0x4478d7bd8a29` | 8.0 ¢ → 5.9 ¢ | +3.3 % | 0.4 |
| `0xd501dd1c7240` | 7.2 ¢ → 4.5 ¢ | +13.4 % | 0.5 |
| `0x44c1dfe43260` | 7.1 ¢ → 4.2 ¢ | +14.8 % | 0.6 |
| `0x1ee9a5fc0966` | 7.6 ¢ → 1.0 ¢ | +4.6 % | 0.5 |

---

## BRIEF FOR THE NEXT SESSION — the conditional analysis

**The question.** Not *is this wallet good* but **what is it good at**. Across the 74
survivors, does edge concentrate by category, entry-price band, or time-to-
resolution? Nothing in this repo has ever asked this.

**Why it is the right next move.** Every per-wallet ranking in this project has
dissolved under testing, and the sample-size arithmetic above says it must: ~34
effective events per half cannot separate a 15 ¢ edge from luck. A **cohort-level**
conditional pattern has ~74× the data behind it, and if one exists it is a *rule*
applicable to any wallet rather than a list of six names to babysit. It is also the
most plausible explanation for why flat rankings keep failing — a real edge in one
condition averaged with noise in another reads as noise overall.

**Run it on all 74, not the six.** Splitting 321 bets by category *and* price band
leaves single-digit cells. If a cohort pattern turns up, then check whether the six
sit inside it.

**Durable artifacts** (`data/interim/funnel/`, gitignored but on disk):

| file | what |
|---|---|
| `structural_survivors_74.parquet` | the 74, with bets / markets / events / eff_events / days_since |
| `performance_halves.parquet` | per-wallet split-half ROI and edge |
| `event_concentration_199.parquet` | the F2→F3 event work |
| `live_spreads.parquet` | per-wallet live half-spreads |
| `rejected_*.parquet` | the two F1 measures that failed calibration |

**Traps this session paid for — do not re-learn them.**

1. **ROI % ranks cheap tickets** (ρ = −0.57 with price). Cents-per-share is better
   but still ρ = −0.47. Report both and expect them to disagree.
2. **A stricter magnitude floor makes persistence worse** past ~7 ¢.
3. **Always compute the base rate first.** 65 % of these wallets have a positive
   held-out edge by default; any filter must be judged against that, not against zero.
4. **Markets are not independent observations.** Use effective events (Kish) and
   bootstrap over events, never over bets.
5. **A plausible structural proxy is not a validated one.** Two of them failed
   against the one wallet known to be uncopyable before a third worked.


## Still to come

- Further structural filters (activity/recency, tradeable spreads, event
  concentration), each proposed and approved the same way.
- Then performance filters, specified by the owner, applied to the **first half**
  of each surviving wallet's history only.
- Then the exam: does the edge hold in the **second half**?
- Then per-wallet conditional analysis — not *is this wallet good* but *what is it
  good at* (category, price band, time-to-resolution). This is unexplored in the
  repo and is the most promising idea in the plan, because a real edge averaged
  together with noise is one explanation for why flat per-wallet rankings keep
  dissolving.
- Then an independent red-team of the whole funnel.
