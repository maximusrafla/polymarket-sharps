# The screening surface — E[out-of-sample ROI | (success_rate, roi, n_bets)]

**Written 2026-07-26.** Read-only. Script: `scripts/audit_screen_surface.py`.
Tests: `tests/test_screen_surface.py`. Reproduce:

```bash
flock data/interim/.analysis.lock -c \
  'PYTHONPATH=. .venv/bin/python scripts/audit_screen_surface.py --real-world'
# drop --real-world to reproduce the (SUPERSEDED) unrestricted run below
# ~7 min unrestricted, ~10 min real-world (more per-cell nulls); ~1.6 GB peak.
# Smoke: --shuffles-p 20 --shuffles-o 2 --shuffles-c 2 --shuffles-og 3 --boot 200
```

> ### ⚠ READ PART II FIRST
>
> **Sections 0–11 below analyse the WHOLE ledger, and the whole ledger is 82% of
> bets / 45% of stake in `micro_crypto` — the 5-minute up-down tape this repo has
> already ruled un-copyable. That makes the unrestricted headline a crypto result
> that happens to be phrased as a general one, and §10 says so itself.**
>
> **[Part II (§12–§20)](#part-ii--the-real-world-only-rerun-supersedes-the-headline)
> re-runs the entire analysis inside real-world markets from the ground up — new
> wallet universe, new price baseline, recalibrated depth and event-floor ladders,
> new nulls and baselines. It supersedes the headline. Part I is kept because its
> machinery, its controls and its self-criticism are the ones Part II reuses, and
> because "the only screen that worked was a crypto-frequency artifact" is only
> demonstrable by having both halves side by side.**

## 0. The question, and the one-line answer

The brief: map `E[out-of-sample ROI | screening-window (success_rate, roi, n_bets)]`
over every wallet in the ledger and find the selection rule that maximizes realized
dollars. Screen on an early window, price the answer on a later untouched window.

**Answer: the 2-D "sweet spot" rule the brief was commissioned to find does not
work. A pre-registered significance screen does — but only in the 5-minute
crypto tape, where nothing can be copied.** Details below; verdict in §9.

## 1. Method

Every wallet's resolved BUY bets are time-sorted (stable, as in
`src.validate.split_in_sample_out_of_sample`) and cut into three disjoint windows at
`floor(n/3)` and `floor(2n/3)`:

| | window | role |
|---|---|---|
| **W1** | 2026-02-11 → 2026-07-20 | the SCREEN — `success_rate`, `roi`, `n_bets`, mean entry price |
| **W2** | 2026-05-02 → 2026-07-20 | the TARGET — the surface is fitted here |
| **W3** | 2026-06-16 → 2026-07-21 | the HONEST window — the chosen rule is priced here, once |

(Dates are the span of the headline screen's own bets; windows are per wallet, so
they overlap across wallets.) Population: **4,070,091 resolved BUY bets**, 10,821
wallets with any resolved BUY, of which **1,718** clear `W1>=5 & W2>=5` and are
screenable. Nothing is dropped, per CLAUDE.md — every wallet is scored and the
flags stay metadata.

CIs resample **clusters, never bets**. Two cluster units are reported and they
answer different questions:

* **event** (`src.sports_events.resolution_event`) — "were its markets lucky?"
* **wallet** — "was its WALLET DRAW lucky?" A screen is a wallet-selection rule, so
  this is the risk an operator actually bears, and the event CI cannot see it.

`effEv` next to every dollar figure is the inverse-Herfindahl effective event count
(`1 / Σ shareᵢ²` over per-event stake). Treat any row with `effEv` in single digits
as anecdote regardless of its point estimate.

## 2. The surface

W2 pooled ROI by (W1 success-rate decile × W1 roi decile), depth `W1>=5`, 1718
wallets; cell = pooled ROI (wallet count).

```
              0            1            2            3            4            5            6            7            8            9
sr0  -0.191(124)  +0.030( 22)  -0.029(  7)  -0.627(  1)  +0.070(  1)  -0.015(  4)  -0.604(  3)  -0.172(  2)  +0.502(  4)  +0.411(  4)
sr1  -0.217( 36)  +0.061( 61)  +0.003( 21)  +0.064(  7)  +0.003(  2)  -0.050(  8)  +0.363(  5)  +0.055( 11)  -0.187( 12)  +0.234(  9)
sr2  +0.122(  3)  +0.073( 30)  -0.020( 35)  +0.009( 17)  +0.009(  8)  +0.014( 18)  -0.044( 15)  +0.084( 20)  -0.074( 14)  -0.159( 12)
sr3  +0.243(  2)  -0.058( 17)  -0.022( 22)  +0.012( 28)  -0.001( 11)  +0.066( 22)  +0.051( 24)  +0.107( 22)  -0.396( 14)  -0.581(  9)
sr4  +0.001(  1)  +0.126( 10)  -0.036( 16)  +0.009( 14)  +0.039( 12)  +0.023( 18)  +0.039( 31)  +0.035( 24)  +0.115( 15)  -0.025(  4)
sr5  -0.114(  2)  -0.012( 14)  -0.019( 13)  +0.031( 19)  +0.012( 12)  +0.019( 21)  +0.013( 17)  +0.051( 19)  +0.015( 23)  -0.364( 12)
sr6  -0.148(  2)  -0.051( 13)  +0.008( 29)  +0.001( 30)  +0.002( 18)  +0.013( 23)  +0.001( 17)  +0.249( 22)  +0.082( 29)  +0.074( 35)
sr7  -0.807(  2)  -0.267(  4)  -0.014( 27)  -0.002( 23)  +0.005( 22)  +0.012( 15)  +0.017( 13)  -0.182( 12)  -0.065( 14)  -0.122( 38)
sr8            .  +0.080(  1)  -0.105(  2)  -0.004( 32)  -0.001( 86)  +0.009( 43)  -0.032( 46)  -0.014( 40)  +0.106( 47)  +0.222( 49)
```

There is no ridge. The interior is a field of ±1–3% cells and the eye-catching
corners are 1–4-wallet cells. The price-residualized surface (in the script output)
looks the same with every cell shifted down ~1.5 points. **Screening-window
success-rate and ROI carry almost no monotone information about the next window.**

## 3. The 2-D optimum — and why it is not real

The rule family searched is every axis-aligned rectangle on that grid at every depth
threshold, subject to ≥15 wallets and ≥1% of pool stake, maximizing pooled W2 ROI.

**Optimum (fitted on W2):** `W1>=10`, `success_rate ∈ [0.667, 0.799)`,
`roi ∈ [0.0638, 0.1366)` → K=18, **W2 ROI +0.2490**.

Priced on W3, the untouched window, it collapses:

| | K | pooled ROI | 95% CI (event) | effEv | topW |
|---|---:|---:|---|---:|---:|
| optimum, W2 (in-sample) | 18 | **+0.2490** | [−0.0741, +0.2564] | 1.1 | 0.99 |
| optimum, W3 (honest) | 18 | **+0.0045** | [−0.0712, +0.0615] | **6.1** | **0.90** |

Four independent things say the +0.2490 is a search artifact:

* **Random-K comparison.** 500 random draws of 18 wallets from the same pool:
  W2 median +0.0141 — the optimum is the 100th percentile. W3 median +0.0030 — the
  optimum is the **52nd percentile**. It is a coin flip out of sample.
* **Cross-validation.** 100 random wallet half-splits: fit-half best +0.2496,
  held-half realized **−0.0050**; shrinkage +0.2545; the held half is positive in
  only **44%** of folds.
* **Null P** (permute the wallet→(W2,W3) linkage within W1-depth strata, preserving
  both marginals bit-for-bit, re-running the entire search): the null's *best region*
  reaches +0.2268 median / +0.3269 max. **p = 0.17.** The strict cluster-preserving
  null C is worse: **p = 0.35**.
* **Concentration.** `effEv 6.1` and `topW 0.90` — one wallet is 90% of the stake.
  Whatever this row's point estimate is, it is one wallet's quarter.

Only null O (outcome shuffle within price bin, which destroys clustering as well as
skill) gives the optimum a p of 0.0385, and that null is the known-permissive
bracket (`docs/blackswan_cluster_null.md`).

**Verdict on the 2-D screen: it does not work, at any depth, on any objective.**
The residual-objective search finds the identical rectangle, so this is not a
price artifact — there is simply no rectangle there.

## 4. The screens, priced on W3

Every candidate on the untouched window. `resid` = the same dollars scored on
profit over `E[outcome | entry_price]`; `net` = gross minus the real per-category
taker fee and one adverse tick (§6).

| screen | K | W3 gross | 95% CI (event) | 95% CI (wallet) | resid | net | effEv |
|---|---:|---:|---|---|---:|---:|---:|
| BASELINE everyone | 1934 | −0.0039 | [−0.0212,+0.0100] | [−0.0330,+0.0229] | −0.0193 | −0.0285 | 119.5 |
| BASELINE all screenable | 1718 | −0.0038 | [−0.0195,+0.0106] | [−0.0328,+0.0230] | −0.0192 | −0.0284 | 119.4 |
| certified 26 *(IN-SAMPLE — see §7)* | 26 | +0.0385 | [−0.0083,+0.0717] | [+0.0200,+0.0708] | +0.0224 | +0.0128 | 22.9 |
| **certainty screen, W1-only, α=0.005** | **17** | **+0.0446** | **[+0.0288,+0.0611]** | **[+0.0251,+0.0654]** | **+0.0295** | **+0.0170** | 424.2 |
| certainty screen, W1-only, α=0.05 | 36 | +0.0041 | [−0.0881,+0.0654] | [−0.0710,+0.0593] | −0.0128 | −0.0278 | 333.2 |
| certainty screen, W1-only, no sig test | 73 | +0.0823 | [−0.0818,+0.2470] | [−0.0375,+0.1806] | +0.0636 | +0.0490 | 37.6 |
| 2-D argmax region (W2-fitted) | 18 | +0.0045 | [−0.0712,+0.0615] | [−0.0093,+0.1762] | −0.0134 | −0.0150 | **6.1** |
| top-decile W1 roi, depth≥5 | 172 | +0.0463 | [−0.0852,+0.1874] | [−0.1158,+0.2788] | +0.0138 | +0.0026 | 75.1 |
| top-decile W1 roi, depth≥100 | 70 | +0.0020 | [−0.1016,+0.0653] | [−0.2992,+0.2057] | −0.0223 | −0.0350 | **8.3** |
| top-decile W1 roi, depth≥500 | 53 | −0.0190 | [−0.1032,+0.0396] | [−0.2506,+0.1544] | −0.0441 | −0.0582 | 11.8 |

This reproduces the inherited summary table to the digit. The α=0.005 row is the
only screen whose W3 CI excludes zero — **on both cluster units** — and the only one
that is still positive after costs.

The screen is the repo's own certification gate run **inside W1 alone**: candidate
on the first half of W1, held-out residual skill edge ≥ 0.02 over ≥ 30 distinct
resolution events in the second half of W1, cluster-bootstrap p < α. α = 0.005 is
not tuned — it is `scoring.project1.oos_significance_alpha`, the pre-registered
production value.

## 5. Price control — the finding SURVIVES it

This repo's history is that price level explains apparent skill more often than
skill does, so the headline was scored three more ways.

**(a) Residual target.** Profit over `E[outcome|entry_price]` (`fit_price_baseline`,
20 quantile bins): **+0.0295, CI [+0.0122, +0.0454]** — still excludes zero, and
still above the pool's −0.0192.

**(b) Inside price bands.** The screen's W3 dollars sliced by entry price, against
the *same band* in the whole screenable pool:

| entry price | bets | $ stake | share | gross | 95% CI | resid | net | fee/stake | pool, same band |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| [0.00,0.10) | 2,246 | 7,647 | 0.4% | +0.4640 | [+0.1088,+0.8524] | −0.9102 | +0.3646 | 6.59% | −0.2462 |
| [0.10,0.25) | 4,105 | 32,430 | 1.6% | +0.3169 | [+0.1655,+0.4749] | +0.3299 | +0.2033 | 5.69% | +0.0251 |
| [0.25,0.50) | 15,110 | 323,295 | 16.1% | +0.1148 | [+0.0358,+0.2135] | +0.0967 | +0.0498 | 4.03% | −0.0792 |
| [0.50,0.75) | 16,485 | 503,890 | 25.1% | +0.0337 | [−0.0707,+0.1078] | +0.0096 | −0.0105 | 2.74% | +0.0236 |
| [0.75,0.90) | 5,608 | 218,839 | 10.9% | +0.0337 | [−0.0019,+0.0673] | +0.0212 | +0.0089 | 1.26% | +0.0129 |
| [0.90,1.00) | 5,025 | 922,998 | 45.9% | +0.0156 | [+0.0127,+0.0187] | +0.0160 | +0.0130 | 0.15% | +0.0014 |

It beats the pool in **every** band. The screen's price mix is insurance-heavy
(46% of stake at ≥ 90¢, the Project-4 shape), but that mix is not what pays: the
**price-mimicking index** — the pool's own return reweighted to the screen's exact
price mix — is **−0.0053**, so **+0.0499 of the +0.0446 is non-price excess.**

**(c) Category-matched.** The same construction on category weights gives a
**category-mimicking index of +0.0060**, leaving **+0.0386** of excess. Within the
crypto tape itself the screen earns +0.0494 where the pool earns +0.0073.

**The price control does not kill it.** Stated plainly because it is the opposite
of what this repo's track record predicted.

## 6. Cost control — the real fee, and it takes 62%

`docs/polymarket_mechanics.md` (363f6df) pins the taker fee on-chain to 7 s.f.:
`fee = shares × rate × p × (1−p)`, taker-only, rate = **0.07 crypto / 0.05 sports,
economics, culture, general / 0.04 politics, tech / 0 geopolitics**. As a fraction
of stake that is exactly `rate × (1−p)` — **largest at low prices**. Geopolitics has
no separate label in `discover.classify_market`, so nothing here is zero-rated and
every net number is a conservative bound.

The previous pass's `net@stress` used k = 0.10, above every live rate. Charged
correctly, per category:

| category | rate | bets | $ stake | wt. price | gross | resid | net | fee/stake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro_crypto | 0.07 | 48,311 | 1,907,661 | 0.756 | +0.0494 | +0.0337 | +0.0211 | 1.71% |
| other | 0.05 | 262 | 96,024 | 0.875 | −0.1063 | −0.1079 | −0.1191 | 0.62% |
| sports_soccer | 0.05 | 5 | 5,413 | 0.486 | +1.0331 | +0.9663 | +0.9865 | 2.57% |
| crypto_event | 0.07 | 1 | 3 | 0.020 | −1.0000 | −3.5375 | −1.1186 | 6.86% |

Blended: **fee 1.66% of stake, adverse tick 1.11% of stake.**

| | pooled | equal-weight |
|---|---:|---:|
| gross | +4.46% | +6.46% |
| less taker fee | **+2.80%** | — |
| less fee + one adverse tick | **+1.70%** (CI [+0.0000, +0.0327]) | **+2.66%** |

**62% of the gross edge is eaten**, and the fee-and-tick CI's lower bound lands on
zero. Two caveats in opposite directions:

* the tick charge is an **approximation** (`tick_for_price`: 1¢ on the body, 0.1¢ on
  the tails; the true tick is a per-market config value this retrospective pass
  cannot read). On a 0.1¢ grid the net would be ≈ +2.7%, not +1.7%. The fee-only
  row +2.80% brackets this from above.
* W3 runs 2026-06-16 → 2026-07-21, entirely **after** fees went live on 2026-01-05,
  and `docs/polymarket_mechanics.md` establishes that one `/trades` row is one
  taker's aggregated match. So the fee is not a projection: **it was already
  charged, and the gross column overstates what these wallets kept.**

## 7. Is it just the certified 26 renamed? No — and only the W1-only number counts

| | K |
|---|---:|
| production `edge_persisted` (selected on FULL history → W2/W3 are in-sample) | 26 |
| W1-only gate at the same α (W2/W3 genuinely held out) | 17 |
| **overlap** | **7** |
| W1-only, not production | 10 |
| production, not W1-only | 19 |

The two sets are mostly different. **The production 26's +0.0385 is not evidence of
anything** — it was chosen with the data it is priced on. Only the W1-only 17 is
uncontaminated.

But splitting the 17 is uncomfortable:

| | K | $ stake | gross | resid | net |
|---|---:|---:|---:|---:|---:|
| the 7 that are also production-certified | 7 | 1,364,788 | +0.0567 | +0.0432 | **+0.0390** |
| the 10 that are not | 10 | 644,312 | +0.0190 | +0.0005 | **−0.0298** |

Net of costs, **all of the money is in the 7**, and the 7 are exactly the wallets a
full-history gate also likes. The W1-only *rule* is honest; the fact that its
profitable core coincides with the contaminated set means the 10 genuinely-new
wallets are, on their own, a cost-negative selection.

A separate and unfixable contamination sits underneath everything: the backfill that
deepened wallet histories **seeds from the ranked table**, which was built on full
history. 96.8% of wallets with `W1>=100` were deepened. The population of deep
wallets is itself selected on full-history performance. The W3 population baseline is
−0.4%, so this is not visibly inflating W3, but it is not provably absent either.

## 8. Nulls for the headline screen

The brief required the surface rebuilt on shuffled outcomes. Applied to the screen
that actually won, not just to the rectangle search:

| null | statistic | real | null distribution | p |
|---|---|---:|---|---:|
| **P** — W1 gate fixed, W2/W3 records re-attached within depth strata (500 draws) | pooled W3 ROI | +0.0446 | median +0.0020 [p5 −0.0797, p95 +0.0428] | **0.052** |
| P | **equal-weight** W3 ROI | +0.0646 | median +0.0039 [p5 −0.0282, p95 +0.0302] | **0.002** |
| P | residual W3 ROI | +0.0295 | median −0.0146 [p5 −0.1007, p95 +0.0228] | **0.046** |
| **O** — outcomes shuffled within entry-price bins, **entire gate rebuilt** (25 draws) | **how many it certifies** | **17** | median 2 [0, 8] | **0.039** |
| O | how well they then do | +0.0446 | median +0.0280 [p5 −0.0639, p95 +0.1016] | 0.250 |

Read the two null-O rows together. The null's ROI draw is 0–8 wallets, so its ROI is
wildly dispersed and comparing point estimates across different K is not
like-for-like; **the count is the statistic with power**, and on the count the gate
finds 17 where pure price structure finds 2.

Note the disagreement between "the CI excludes zero" and "p = 0.052". The CI
conditions on the 17 selected wallets; the permutation test also charges for the
selection step. **The permutation test is the better instrument, and it puts the
pooled result at the 5% line, not at the 0.5% its own α advertises.** Only the
equal-weight version is convincing (p = 0.002).

## 9. Fragility

**Across α.** The sets are strictly nested (`passes[α] = base & p<α`), so each row
only adds wallets:

| α | K | added | W3 pooled | W3 eqw | W3 resid | W3 net | added $ stake | added ROI | effEv |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 17 | 17 | **+0.0446** | +0.0646 | +0.0295 | +0.0170 | 2,009,100 | +0.0446 | 424.2 |
| 0.01 | 20 | 3 | **−0.0074** | +0.0573 | −0.0238 | −0.0380 | 611,516 | **−0.1785** | 228.0 |
| 0.05 | 36 | 16 | +0.0041 | +0.0514 | −0.0128 | −0.0278 | 612,149 | +0.0537 | 333.2 |
| 0.10 | 47 | 11 | +0.0887 | +0.0554 | +0.0718 | +0.0554 | 1,747,457 | +0.2450 | 30.8 |
| 0.25 | 66 | 19 | +0.0826 | +0.0508 | +0.0638 | +0.0492 | 418,653 | +0.0102 | 36.0 |
| 0.50 | 73 | 7 | +0.0823 | +0.0483 | +0.0636 | +0.0490 | 126,592 | +0.0691 | 37.6 |
| 1.00 (no sig test) | 73 | 0 | +0.0823 | +0.0483 | +0.0636 | +0.0490 | 0 | — | 37.6 |

Three more wallets — the next-most-certain three — carry $611k at **−17.85%** and
flip the pooled headline from +4.46% to −0.74%. **The pooled point estimate is not a
property of "certainty"; it is a property of exactly 17 wallets.** Nor is the ladder
monotone: α = 0.10/0.25/0.50 all show a *higher* pooled W3 ROI than α = 0.005, on 30-odd
effective events.

**The equal-weight column is the stable one**: +0.048 → +0.065 across the entire
ladder. If anything here survives, it is the equal-weight statement, not the
pooled one.

**Within the 17.** Leave-one-wallet-out spans only [+0.0369, +0.0516], so the result
is not one wallet carrying sixteen passengers (top wallet is 33% of stake). The
fragility is at the *boundary* of the rule, not inside the selected set.

## 10. The control that actually decides it: copyability

95% of the screen's W3 stake, and 16 of its 17 wallets, are `micro_crypto` — the
5-minute BTC/ETH up-down tape.

| | K | $ stake | gross | 95% CI | resid | net | effEv | topW |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| SCREEN, micro_crypto | 16 | 1,907,661 | **+0.0494** | [+0.0388,+0.0614] | +0.0337 | +0.0211 | 650.0 | 0.30 |
| POOL, micro_crypto | 1557 | 22,431,814 | +0.0073 | [+0.0035,+0.0111] | −0.0066 | −0.0157 | 4392.0 | 0.06 |
| SCREEN, real-world | **3** | **101,439** | **−0.0455** | [−0.2738,+0.6151] | −0.0506 | −0.0601 | **2.6** | **0.98** |
| POOL, real-world | 346 | 24,587,946 | −0.0139 | [−0.0455,+0.0129] | −0.0306 | −0.0400 | 32.8 | 0.15 |

`src.discover.is_real_world` already defines `micro_crypto` as the **un-copyable**
slice, and every other arm of this repo excludes it on that ground. The
classification is sound here: over the ledger's 229,092 micro-crypto slugs, 96%
contain `updown` and the *measured* `speed_bucket` is `fast` for 3.67M of the 3.75M
micro-crypto bets (515 are slow).

So the mechanism behind the whole result is mundane. The gate demands ≥ 30 distinct
held-out resolution events **inside half of W1** plus a cluster-bootstrap
p < 0.005. On this venue only very-high-frequency wallets can clear that, and
very-high-frequency on Polymarket means the 5-minute crypto tape. **The "certainty"
screen is a frequency screen wearing a significance costume** — which is exactly why
its `effEv` is 424 against the certified 26's 22.9.

And the copyable residue is nothing: 3 wallets, $101k, `effEv 2.6`, one wallet at 98%
of the stake, point estimate negative, CI 90 points wide.

## 11. Verdict

1. **The 2-D screen the owner asked for does not beat a significance screen. It
   does not beat a coin flip.** W3 +0.0045 at the 52nd percentile of random K=18
   draws, held-half positive in 44% of CV folds, null P p = 0.17, `effEv 6.1`,
   one wallet at 90% of stake. Dead on every axis.

2. **The significance screen is the only rule with out-of-sample content**, and
   that content is real: +4.46% gross, **surviving** the price residual (+2.95%),
   surviving within every price band, surviving the price-matched index (+4.99%
   of non-price excess) and the category-matched index (+3.86%). This repo's usual
   killer did not kill it.

3. **But it does not survive costs.** Net of the real taker fee and one adverse
   tick it is **+1.70%**, CI lower bound on zero, 62% of the edge eaten. Its
   permutation p is 0.052, not 0.005. It flips sign when the next three
   most-certain wallets are admitted. And 95% of it lives in 5-minute markets that
   this repo has already ruled un-copyable, with the copyable residue at 3 wallets
   and `effEv 2.6`.

4. **The one claim worth keeping** is narrower and equal-weighted: *a
   pre-registered significance gate run on an early window selects wallets whose
   later equal-weighted ROI is +6.5% gross / +2.7% net, beating a linkage-destroying
   null at p = 0.002 and stable across the whole α ladder.* That is a genuine
   identification result. It is not an execution result, because the venue where it
   happens resolves every five minutes.

5. **The framing "optimizing for statistical certainty IS the better dollar
   strategy" is too strong.** Certainty is not monotone in dollars here (α = 0.10
   pooled beats α = 0.005 pooled), and the mechanism is not certainty but
   *frequency*: the gate's event floor selects scalpers, and scalpers happen to be
   the only wallets on this venue with enough independent resolutions to prove
   anything about themselves. What the exercise really shows is that **sample size
   is the only screening variable on this surface that carries signal** — and it
   buys signal in the one market class a copier cannot reach.

Nothing here changes the standing copy-thesis position; it adds one more front on
which the copyable universe came back empty.

---
---

# PART II — the real-world-only rerun (supersedes the headline)

**Added 2026-07-26, same day.** Log: `~/.pmrun/rw_full3.log`. Reproduce with
`--real-world`.

## 12. Why Part I had to be redone, not filtered

Part I was commissioned over the whole ledger with an explicit instruction not to
restrict the population. That was a briefing error. On this ledger:

| | resolved BUY bets | stake |
|---|---:|---:|
| `micro_crypto` (dropped) | 3,335,883 (82.0%) | $72.96M (45.3%) |
| everything else (kept) | **734,208 (18.0%)** | **$88.05M (54.7%)** |

An unrestricted population analysis is therefore mostly a crypto analysis, and
Part I's winning screen was 95% crypto stake **by construction**: its gate demands
≥30 distinct held-out resolution events inside half of a screening window, and on
this venue only the 5-minute tape trades often enough to clear that.

Part I did report a real-world "residue" — 3 wallets, `effEv 2.6`, negative point
estimate. **That number answers a different question and is now withdrawn.** It is
the leftovers of a crypto-selected screen, not a real-world screen. Every stage
downstream of the population — the wallet universe, the price baseline, the decile
edges, the depth strata, the nulls, the baselines — was fitted on the crypto tape,
so filtering its output afterwards cannot recover the real-world answer.

**The one-line answer for Part II: nothing works. No screen — significance, the
owner's ROI screen, or the owner's 2-D screen — produces a positive, permutation-
significant, cost-surviving return in real-world markets.** One narrow statistical
effect does survive everything (§17), and it is explicitly not a dollar strategy.

## 13. What was kept and what was dropped

Filter: `src.discover.is_real_world(classify_market(slug, question))`, i.e.
`category != "micro_crypto"` — bit-identical to what the sports arm, the
slow-forecaster arm and the edge-decay arm exclude, so Part II is comparable to
them.

* **Dropped:** `micro_crypto` only — 3,335,883 bets, 212,020 markets.
* **Kept:** 734,208 bets over 101,591 markets → **56,017 resolution events**,
  in 13 categories: `other`, `sports_{mlb,nba,nfl,nhl,soccer,tennis,ufc,cricket}`,
  `politics`, `econ_macro`, `crypto_event`, `culture`.
* **Wallets with ≥1 kept bet: 3,192** (of 10,821 with any resolved BUY).

Stake mix of the screenable pool's W3 dollars ($26.48M): `other` **54.0%**, sports
**39.4%** (tennis 13.3, mlb 13.1, nba 7.6, soccer 3.9, cricket 0.7, ufc 0.5, nhl
0.3, nfl 0.01), `crypto_event` 5.4%, `politics` 0.9%, `culture` 0.2%, `econ_macro`
0.02%. The `other` share is large because `classify_market` is a conservative
keyword classifier and its catch-all is the residual; `other` is rated at 0.05,
the same as sports and general, so nothing is under-charged by it. Note how thin
politics and macro are — the categories a "forecasting skill" thesis would want
are ~1% of the money here.

**Caveat that matters and is not in the filter's name:** `is_real_world` is a
*category* filter, not a *speed* filter. The measured `speed_bucket` of the kept
bets is **fast 429,089 / slow 226,934 / deep_slow 22,085 / unknown 56,100** — so
58% of surviving bets are still in markets with a <24 h lifespan. "Real-world"
here means "not the 5-minute tape", not "slow enough to copy comfortably". Nothing
below turns on this, because nothing below is positive, but a future arm that
finds something here must re-check it against `speed_bucket`.

## 14. The recalibrated ladders, and which depths exist at all

A depth threshold is a statement about trading *frequency*, and frequency means
something different on the two sides of the filter — a real-world wallet with 30
bets is a serious trader, a 5-minute-tape bot does 30 bets in an hour. So the
ladder is recalibrated (`REALWORLD_DEPTH_THRESHOLDS`), not inherited.

| W1 depth | wallets in pool | screenable (W2≥5) | + W3≥5 |
|---:|---:|---:|---:|
| ≥3 | 452 | **353** | 353 |
| ≥5 | 346 | 346 | 346 |
| ≥10 | 259 | 259 | 259 |
| ≥20 | 204 | 204 | 204 |
| ≥30 | 182 | 182 | 182 |
| ≥50 | 157 | 157 | 157 |
| ≥100 | 136 | 136 | 136 |
| ≥200 | 109 | 109 | 109 |
| ≥500 | 88 | 88 | 88 |

**Every depth from ≥3 to ≥500 is attainable**; ≥1000 is not populated enough to
screen, which is why the ladder stops at ≥500. The whole screenable universe is
**353 wallets**, against 1,718 unrestricted — that is the real size of the problem.

Windows are thirds of each wallet's OWN record, so they overlap across wallets;
the headline screen's bets span W1 2025-03-02 → 2026-07-20, W2 2025-12-19 →
2026-07-20, W3 2026-01-01 → 2026-07-21. W3 therefore sits almost entirely after
taker fees went live on 2026-01-05, so its net column is accounting, not a
projection.

**Upstream selection, unchanged and unfixable:** the backfill that deepened wallet
histories seeds from the ranked table, which was built on full history. 61% of
wallets with W1≥3 were deepened, rising to **97% at W1≥30 and 99% at W1≥100**. The
deep end of this population is itself selected on full-history performance. The W3
population baseline is −2.1%, so it is not visibly inflating W3, but it is not
provably absent either.

## 15. Baselines on the untouched window (W3)

| | K | $ stake | pooled | 95% CI (event) | equal-weight | resid | net | effEv |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| everyone | 365 | 26,482,552 | **−0.0213** | [−0.0521,+0.0052] | −0.0891 | −0.0481 | −0.0468 | 37.3 |
| all screenable (W1≥3) | 353 | 26,480,769 | **−0.0213** | [−0.0514,+0.0063] | −0.0832 | −0.0481 | −0.0469 | 37.3 |
| production certified 26 *(in-sample, meaningless)* | 4 | 3,070,430 | +0.0233 | [−0.0576,+0.0695] | +0.0320 | +0.0139 | −0.0010 | 10.9 |

Real-world traders lose 2.1% of stake gross and 4.7% net. Note the certified 26
barely exist here: only **4** of them have ≥5 real-world W3 bets. The certified set
is a crypto set.

Costs, computed per category rather than inherited: the pool's blended charge is
**2.56% of stake** (gross −2.13% → net −4.69%). It is *higher* than Part I's 1.66%
fee even though the rates are the same or lower, because the real-world price mix
is cheaper — and `fee/stake = rate × (1−p)` is largest at low prices.

## 16. The two screens the owner proposed

### 16a. The 2-D (success_rate × ROI) argmax — dead

Optimum fitted on W2: depth W1≥5, `success_rate ∈ [0.628, 0.778)`,
`roi ∈ [0.0586, ∞)` → K=19, **W2 pooled ROI +0.2502**.

| | K | W3 pooled | 95% CI | eqw | net | effEv | topW |
|---|---:|---:|---|---:|---:|---:|---:|
| optimum, W2 (in-sample) | 19 | +0.2502 | [−0.1186,+0.2571] | −0.0411 | +0.2202 | **1.1** | **0.99** |
| optimum, W3 (honest) | 19 | **+0.0123** | [−0.0593,+0.0866] | +0.0032 | **−0.0074** | **6.0** | **0.91** |

* **Random-K:** 500 random draws of 19 wallets — W2 median +0.0188 (optimum at the
  99.8th percentile), **W3 median −0.0113, optimum at the 75.2nd percentile.**
* **Cross-validation:** 100 wallet half-splits, fit-half best +0.2441 → held-half
  **+0.0374**, shrinkage +0.2067, held-half positive in 76% of folds.
* **Null P** (permute wallet→(W2,W3) linkage within depth strata, re-run the whole
  search, 500 draws): null best-region median +0.2431, max +0.3269 →
  **p = 0.339**. Loosening to minK=30/60 gives p = 0.264 / 0.309.
* **Null C** (cluster-preserving cell permutation, 25 draws): **p = 0.423.**
  Null O: p = 0.077.
* **Concentration:** `effEv 6.0`, one wallet at 91% of the W3 stake.

The residual-objective search returns the identical rectangle, so this is not a
price artifact — there is simply no rectangle. It does slightly better here than
in the crypto pool (75th percentile vs 52nd; 76% of CV folds positive vs 44%), but
the honest permutation is unambiguous: **the 2-D screen does not work.**

### 16b. Top-decile by ROI — dead net of cost

Pre-specified, no argmax, so no search burden. Top decile of W1 `roi` against its
own pool, at every depth (null P, 500 draws):

| depth | pool | pool W2 ROI | top-dec W2 ROI | K | top − pool | p | **W3 top-dec** | W3 pool |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ≥3 | 353 | +0.0678 | +0.0680 | 36 | +0.0003 | 0.178 | **−0.1556** | −0.0213 |
| ≥5 | 346 | +0.0678 | +0.0696 | 35 | +0.0018 | 0.176 | **−0.1573** | −0.0213 |
| ≥10 | 259 | +0.0677 | −0.0738 | 26 | −0.1416 | 0.964 | −0.0096 | −0.0211 |
| ≥20 | 204 | +0.0679 | −0.0442 | 21 | −0.1120 | 0.942 | −0.0237 | −0.0209 |
| ≥30 | 182 | +0.0681 | −0.0470 | 19 | −0.1150 | 0.958 | −0.0227 | −0.0210 |
| ≥100 | 136 | +0.0688 | −0.0466 | 14 | −0.1154 | 0.964 | −0.0227 | −0.0209 |
| ≥500 | 88 | +0.0766 | −0.0365 | 9 | −0.1131 | 0.932 | −0.0135 | −0.0128 |

**At every depth ≥10 the top-ROI decile does WORSE than its own pool**, in both
windows, and its permutation p is on the wrong side of 0.93. Priced with costs on
W3 the top decile is −2.3% gross / −6.3% net (depth≥30, `effEv 5.3`, topW 0.50).

**The owner's ROI screen does not work in real-world markets.** It is
not merely insignificant — at every depth where a wallet has a track record worth
the name, it is actively worse than not screening.

## 17. The one thing that survives everything — and why it still is not tradeable

There is a real, robust effect hiding underneath §16b, and it deserves to be
stated as carefully as the negatives.

Score the same ROI screen on the **median per-wallet W3 return** instead of the
pooled or mean one, over a 27-cell grid (9 depths × keep 5%/10%/25%), each cell
null-tested with the same 500 permutations that charge for the selection step:

| statistic | cells positive | cells at p<0.05 |
|---|---:|---:|
| gross median per-wallet W3 ROI | **27/27** | **25/27** |
| the same, on the price-residual target | **27/27** | **23/27** |
| the same, after the real taker fee + one adverse tick | 16/27 | 15/27 |
| median excess over a **wallet-size-matched** index | **26/27** | — |

Representative cells (`p` is the permutation p, not a CI):

| depth | keep | K | W3 median | p | resid median | p | net median | p | size-matched excess |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ≥10 | 0.10 | 26 | +0.1093 | 0.002 | +0.0676 | 0.002 | +0.0528 | 0.002 | +0.1275 |
| ≥30 | 0.10 | 19 | +0.0400 | 0.008 | +0.0277 | 0.006 | +0.0084 | 0.010 | +0.0388 |
| ≥500 | 0.25 | 22 | +0.0372 | 0.002 | +0.0317 | 0.002 | +0.0049 | 0.004 | +0.0314 |

This survives the price residual, survives real costs at 15 of 27 cells, survives
a wallet-size-matched index (a top-ROI screen picks small books, and that is not
what is paying), and holds from depth ≥3 to depth ≥500. **A wallet's ROI rank does
carry genuine, out-of-sample, non-price, non-size information about where the
*middle* of its next-window return distribution sits.** That is a real
identification finding and it is the first thing in this document that the repo's
usual killers did not kill.

**It is nonetheless not a dollar strategy, for one decisive reason: a copier
realizes the MEAN, not the median.**

| statistic | how a copier gets it | cells positive | cells at p<0.05 |
|---|---|---:|---:|
| per-wallet median | you cannot trade a median | 27/27 | 25/27 |
| equal-weight mean, gross | equal dollars per wallet | 14/27 | 3/27 |
| **equal-weight mean, net** | **equal dollars per wallet, after costs** | **2/27** | — |
| pooled | size-weighted, as the wallets actually traded | **2/27** | 1/27 |

The gap between a reliably positive median and a reliably negative mean *is* the
finding: **the ROI screen selects wallets with a better typical outcome and a
fatter left tail.** Costs then take the median from +4.0% to +0.8% at the deep
cells, which is inside the tick-cost approximation's own error bar.

Do not restate this as "the ROI screen works." It works on a statistic nobody can
hold.

## 18. The significance screen in real-world markets

Part I's winner was the repo's own certification gate run inside W1 alone. Its
binding constraint is the **≥30 distinct held-out resolution events** floor, which
is a frequency threshold, so here it is swept rather than assumed:

| event floor | K at α=0.005 | K at α=0.05 | K with no sig test |
|---:|---:|---:|---:|
| 30 *(production)* | **2** | 10 | 16 |
| 20 | 2 | 11 | 18 |
| 10 | 3 | 12 | 25 |
| 5 | 7 | 18 | 37 |
| 3 | 15 | 29 | 61 |

**The production gate is effectively unattainable in real-world markets: at the
pre-registered α it certifies two wallets.** That is the mechanism from Part I §10
seen from the other side.

Headline cell, pre-registered on **K alone** (deepest floor still certifying ≥15
wallets at the production α → floor 3, α=0.005, K=15):

| | value |
|---|---|
| W3 stake | $52,301 |
| W3 pooled gross | **+0.0017** |
| W3 residual | −0.0006 |
| W3 net (real fee + tick) | **−0.0130** |
| W3 equal-weight | **−0.0550** |
| effEv / topW | **3.3** / 0.63 |
| event CI / wallet CI | [−0.113,+0.114] / [−0.467,+0.115] |
| **null-P permutation p (pooled / eqw / resid)** | **0.415 / 0.303 / 0.389** |
| price-mimicking index | −0.0143 (so +0.0159 of non-price excess — on $52k) |
| overlap with the production certified 26 | **0** |

Every α on the ladder gets the same permutation test rather than only the headline
(7 alphas × 3 statistics = 21 looks at the same 500 permutations — the multiplicity
is the point):

| α | K | W3 pooled | p | W3 eqw | p | W3 median | p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 15 | +0.0017 | 0.415 | −0.0550 | 0.303 | +0.0140 | 0.250 |
| 0.01 | 17 | −0.1996 | 0.966 | −0.0731 | 0.381 | +0.0061 | 0.279 |
| 0.05 | 29 | +0.0637 | 0.088 | −0.0344 | 0.264 | +0.0265 | 0.014 |
| 0.10 | 31 | +0.0638 | 0.088 | −0.0187 | 0.168 | +0.0400 | 0.008 |
| 0.25 | 47 | +0.0712 | 0.064 | −0.0087 | 0.122 | +0.0159 | 0.036 |
| 0.50 / 1.00 | 61 | +0.0543 | 0.076 | −0.0487 | 0.248 | +0.0096 | 0.066 |

**The equal-weight column — the one Part I concluded was the only defensible
statement — is negative at every single α and never comes close to significance.**
The pooled column never clears 0.05 either. Only the median column is significant,
and that is the §17 effect showing up again, not an independent result.

Null O (outcomes shuffled within entry-price bins, the entire gate rebuilt on the
shuffled tape, 25 draws): the gate certifies **15** where pure price structure
certifies a median of 10 → p = 0.0385 on the count, and p = 0.577 on how well they
then do. The gate is finding *something* beyond the favorite-longshot curve, and
whatever it is, it is trivially small on `effEv 3.3`.

## 19. Direct comparison: what changed when the population was fixed

| | unrestricted (Part I) | real-world (Part II) |
|---|---:|---:|
| screenable wallets | 1,718 | **353** |
| W3 pool ROI | −0.0038 | **−0.0213** |
| blended cost / stake | 2.77% | **2.56%** |
| certainty screen K at α=0.005, floor 30 | 17 | **2** |
| its W3 pooled gross | +0.0446 | +0.0017 *(at floor 3, K=15)* |
| its W3 net | +0.0170 | **−0.0130** |
| its W3 equal-weight | +0.0646 | **−0.0550** |
| its permutation p (pooled / eqw) | 0.052 / **0.002** | **0.415 / 0.303** |
| its effEv | 424.2 | **3.3** |
| 2-D argmax W3, percentile of random-K | +0.0045, 52nd | +0.0123, 75th |
| 2-D argmax null-P p | 0.17 | **0.339** |
| top-decile-by-ROI W3 vs pool | +0.0463 vs −0.0038 | **−0.1556 vs −0.0213** |

Part I §11.4 kept one claim: *"a pre-registered significance gate run on an early
window selects wallets whose later equal-weighted ROI is +6.5% gross / +2.7% net,
beating a linkage-destroying null at p = 0.002."* **In real-world markets the same
sentence reads −5.5% gross / −8.5% net at p = 0.303.** The claim was a property of
the 5-minute tape, exactly as Part I §11.5 suspected but could not demonstrate
without this half.

## 20. Verdict

1. **No screen works in real-world markets.** Significance screens, the owner's
   ROI screen and the owner's 2-D screen all fail on the untouched window, on the
   permutation test that charges for selection, and after real costs. The
   real-world population's own baseline is −2.1% gross / −4.7% net, and no
   selection rule tested here beats it in a way a permutation cannot reproduce.

2. **The one screen that ever worked was a crypto-frequency artifact.** Part I
   could show that its winner *lived* in micro-crypto; Part II shows that the same
   rule, recalibrated and re-run where a copier can actually act, produces a
   negative equal-weight return at p = 0.30. The gate's event floor was selecting
   scalpers, and outside the scalping venue it certifies two wallets.

3. **The owner's ROI screen is the clearest negative of the three.** At every
   depth ≥10 the top-ROI decile underperforms its own pool in both held-out
   windows. It is not "unproven"; it is backwards.

4. **One real effect survives and is worth recording:** a wallet's ROI rank
   predicts the *median* of its next-window per-wallet return, out of sample,
   after the price residual, after real costs (15/27 cells), and after a
   wallet-size-matched index (26/27 cells), from depth ≥3 to ≥500. It is not
   tradeable because the same wallets' mean and pooled returns are negative — the
   screen buys a better middle and a fatter left tail, and a copier eats the mean.
   If anything here is ever revisited, revisit *that* shape, not the threshold.

5. **Minimum viable screening depth: there isn't one that helps.** Depths ≥3
   through ≥500 are all populated and all were tested; no depth turns any screen
   positive-and-significant on a tradeable statistic. Depth buys precision on the
   §17 median effect (its permutation p stays ≤0.02 out to ≥500) and buys nothing
   on the net statistic.

Part II closes the screening front on the copyable universe the same way the category sweep, the edge-decay
arm, the sports arm and the value-betting gate closed theirs.
