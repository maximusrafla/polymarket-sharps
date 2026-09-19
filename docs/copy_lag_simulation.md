# Copy-lag simulation on the certified real-world wallets

> ### ⛔⛔ EVERY TABLE BELOW IS SUPERSEDED. Read `docs/redteam_realworld_copy_2026-07-29.md`.
>
> This file is the **historical record of the original run**, kept so the corrections
> can be checked against it. Do not quote a number from it.
>
> * The live estimator is `python -m src.copy_sim **certified**` (not `score`, which
>   still runs the withdrawn shuffled null and now says so on stdout).
> * Current numbers: pooled follower edge **+1.98¢ [+1.16, +2.82]**; **+2.42¢** on
>   held-out bets only; the wallet's own moment worth **+0.83¢ [+0.47, +1.17]** once a
>   position-matched control removes the confound; **11 of 37** wallets read copyable
>   but *which* 11 is not distinguishable from chance (cluster-preserving null,
>   P = 0.23) and churns on the bootstrap seed alone.
>
> ---
>
> ### ⛔ THE VERDICT IN THIS DOCUMENT HAS BEEN OVERTURNED — read `docs/copy_verdict.md`
>
> The **method** below stands and is reused. The **conclusion** ("no copyable edge
> survives 2 minutes") does not. Two errors, both pushing the same way:
>
> 1. **The null could not see the effect it tested for.** It shuffled outcomes
>    within price bins among *only these five certified-skilled wallets' own bets*,
>    so it absorbed their skill. The tell needs no hindsight: it reports **exactly
>    0.000 alpha at Δ = 0** — at zero lag, paying the identical price, where a
>    follower simply *is* the wallet.
> 2. **The adverse tick was over-charged ~10×** — 1¢ charged against a 0.001 tick
>    quoted by 97% of these markets.
>
> Corrected against the population baseline: the follower nets **+2.02¢/share**
> at a 2-minute lag (CI [+1.07, +2.99]), ≈ **+9.3%** equal-weighted ROI pooled,
> concentrated in two wallets. Drift — not cost — is what takes the rest.

**Question.** A certified wallet buys. A follower sees the fill ~2 minutes
later and buys at whatever the market price is *then*. How much of the
wallet's edge survives that lag, plus the real taker fee and one adverse tick?

Code: `src/copy_sim.py` · tests: `tests/test_copy_sim.py` · per-bet frame:
`data/interim/copysim/per_bet.parquet`.

## Method

* **Universe.** The five `edge_persisted` wallets from
  `data/interim/realworld/validated.parquet` named in the task, all of their
  resolved BUY bets in the deep real-world tape.
* **Price path.** The public CLOB `/prices-history` endpoint, `fidelity=1`
  (a book-**midpoint** series on a 1-minute grid). The deep tape holds only our
  own wallets' fills and so cannot say what the price was two minutes later.
* **Bounded window.** The follower price is the first grid point in
  `[entry + Δ, entry + Δ + 120s]`. No unbounded reach; if nothing
  qualifies the result is NaN, never `resolved_value`.
* **Resolution guard.** Prices in the final 20% of a market's lifespan
  (Gamma `startDate` → `umaEndDate`) are excluded, so a near-resolution price
  cannot stand in for a follower price.
* **Same cohort.** Own return is recomputed on exactly the bets measurable at
  each Δ, never against the full sample.
* **Costs.** `fee = shares × k × p × (1−p)`, taker-only, `k` from Gamma's
  per-market `feeSchedule.rate`, plus one adverse tick of 0.01
  (the series is a midpoint; a follower lifts the offer).
* **Null.** `resolved_value` permuted within 20 entry-price quantile bins,
  200 shuffles. What survives the shuffle is the favorite-longshot base
  rate available to anyone at that price, not copyable alpha.
* **CI.** 95% percentile bootstrap resampling whole **markets** (one resolution
  event = one draw), 2,000 resamples.

Every table below is reported twice, in two units:

* **ROI per share**, `(resolved_value − cost) / cost` — the quantity the task
  asks for, and an equal-dollar-per-bet portfolio return. It is violently
  heavy-tailed: a 1c longshot that wins is +9,900%, so both the means and
  especially the shuffled null are dominated by a handful of sub-cent prices
  (the null reaches +1.58 in one cell, which is noise, not a finding).
* **Edge per share in price units**, `resolved_value − cost` — the repo's
  native unit, bounded in `[−1, 1]`, and the one to read when the two
  disagree. `drift` is `price(entry+Δ) − entry_price`: how far the price moved
  against a buyer during the lag. In these units `real − null` is exactly
  `mean(outcome) − mean(shuffled outcome)` over the cohort — the cost terms
  cancel — so it is the pure outcome-selection excess and **no fee or tick
  assumption can rescue or destroy it**.

## Data

* 18,703 resolved BUY bets · 6,771 markets ·
  7,805 tokens · 5 wallets
* span 2025-06-29 → 2026-07-26
* top categories: other 11,906, sports_soccer 1,792, sports_nba 1,235, politics 1,000, sports_nfl 779, sports_tennis 519
* mean fee k charged: 0.0504
* markets with no usable Gamma lifespan (⇒ no follower price at any Δ): 0.6%
* |tape entry price − midpoint at entry|: median 0.0075, p75 0.0276, mean 0.0328 (n=18,693) — the scale of the adverse-tick assumption

The five wallets, and what the identification engine certified them on
(`data/interim/realworld/validated.parquet`):

| wallet | held-out skill edge | held-out bets | held-out markets | bets here |
|---|---:|---:|---:|---:|
| `0x09bed1976600971fd9f9db454a9c302c7a559c32` | +0.0452 | 157 | 39 | 314 |
| `0x1ee9a5fc09665909c0cce297c581703bfbb9197f` | +0.0693 | 2,535 | 1,175 | 5,271 |
| `0x69ea0d77ef34f1acb03aaed901df7620fc4215cd` | +0.0388 | 1,654 | 428 | 3,308 |
| `0x83255595ba1fadd2e734cb30a0fb8110301a19cc` | +0.0431 | 1,695 | 470 | 3,389 |
| `0xe542afd3881c4c330ba0ebbb603bb470b2ba0a37` | +0.0539 | 3,210 | 1,099 | 6,421 |

## Headline

Pooled, Δ = 2 minutes, on the 99%-coverage (guard-off) sample, per share:

```
  wallet's own edge at its own fill      +0.0485
  price drift in the first 2 minutes     -0.0200
  = follower's gross edge                +0.0285
  fee + one adverse tick                 -0.0144
  = follower's NET edge                  +0.0141
  favorite-longshot base rate (null)     -0.0138
  = COPYABLE ALPHA                       +0.0003   (null p = 0.100)
```

Per wallet at Δ = 2 minutes, copyable alpha (`real − null`, price units, the
cost-invariant number) under both guard settings:

| wallet | guard on: n / alpha / p | guard off: n / alpha / p | follower net (guard off) |
|---|---|---|---:|
| `0x09bed1976600…` | 15 / -0.0327 / 0.945 | 311 / -0.0007 / 1.000 | +0.0272 [-0.0046, +0.0652] |
| `0x1ee9a5fc0966…` | 3,077 / +0.0200 / 0.000 | 5,216 / +0.0022 / 0.000 | +0.0239 [+0.0050, +0.0426] |
| `0x69ea0d77ef34…` | 1,947 / +0.0016 / 0.230 | 3,305 / +0.0000 / 0.975 | +0.0512 [+0.0363, +0.0674] |
| `0x83255595ba1f…` | 10 / -0.1010 / 0.875 | 3,309 / -0.0015 / 0.975 | -0.0194 [-0.0307, -0.0077] |
| `0xe542afd3881c…` | 2,625 / -0.0170 / 1.000 | 6,418 / -0.0003 / 0.980 | +0.0038 [-0.0155, +0.0275] |

## Pooled — primary (guard on)

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 18,703 | 100% | 6,771 | +0.5170 | +0.5170 | +0.1430 | +0.1498 | -0.0069 | 0.995 | [+0.0336, +0.2766] |
| 2m | 7,674 | 41% | 3,082 | +0.4470 | +0.2957 | +0.1150 | +0.1294 | -0.0143 | 0.715 | [-0.0189, +0.2763] |
| 15m | 7,646 | 41% | 3,070 | +0.4500 | +0.3113 | +0.1088 | +0.1319 | -0.0231 | 0.815 | [-0.0235, +0.2623] |
| 1h | 7,548 | 40% | 3,031 | +0.4440 | +0.2400 | +0.0727 | +0.2055 | -0.1328 | 1.000 | [-0.0579, +0.2388] |
| 6h | 7,108 | 38% | 2,803 | +0.4735 | +0.5084 | +0.1087 | +0.2696 | -0.1608 | 1.000 | [-0.0569, +0.3086] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0483 | +0.0000 | +0.0483 | +0.0339 | +0.0339 | +0.0000 | 1.000 | [+0.0243, +0.0442] |
| 2m | +0.0569 | +0.0115 | +0.0455 | +0.0302 | +0.0257 | +0.0045 | 0.040 | [+0.0152, +0.0449] |
| 15m | +0.0574 | +0.0178 | +0.0396 | +0.0244 | +0.0199 | +0.0046 | 0.035 | [+0.0103, +0.0389] |
| 1h | +0.0579 | +0.0193 | +0.0385 | +0.0236 | +0.0183 | +0.0053 | 0.030 | [+0.0095, +0.0373] |
| 6h | +0.0597 | +0.0266 | +0.0331 | +0.0185 | +0.0128 | +0.0058 | 0.030 | [+0.0038, +0.0330] |

## Pooled — guard off (the high-coverage view)

These wallets bet late: the median gap from entry to resolution is 2.1 days
against a median market lifespan of 26 days, so the proportional guard puts
**58% of bets past the cutoff** and the primary tables above measure a
minority, long-dated cohort. Note that near-resolution contamination pushes
the follower's number *down* (price converges on the answer, so the follower
pays ~the payout), so the unguarded run is if anything the more generous one.

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 18,703 | 100% | 6,771 | +0.5170 | +0.5170 | +0.1430 | +0.1494 | -0.0064 | 0.980 | [+0.0328, +0.2768] |
| 2m | 18,559 | 99% | 6,729 | +0.5198 | +0.1423 | +0.0017 | +0.1862 | -0.1845 | 1.000 | [-0.1020, +0.1302] |
| 15m | 18,386 | 98% | 6,672 | +0.5258 | +0.1644 | -0.0273 | +0.6202 | -0.6475 | 1.000 | [-0.1351, +0.1047] |
| 1h | 17,382 | 93% | 6,519 | +0.5176 | +0.0560 | -0.0622 | +1.5838 | -1.6460 | 1.000 | [-0.1696, +0.0700] |
| 6h | 11,985 | 64% | 4,152 | +0.4239 | +0.3553 | +0.0562 | +0.6040 | -0.5478 | 1.000 | [-0.1124, +0.2533] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0483 | +0.0000 | +0.0483 | +0.0339 | +0.0339 | +0.0000 | 1.000 | [+0.0240, +0.0433] |
| 2m | +0.0485 | +0.0200 | +0.0285 | +0.0141 | +0.0138 | +0.0003 | 0.100 | [+0.0045, +0.0249] |
| 15m | +0.0486 | +0.0246 | +0.0240 | +0.0104 | +0.0100 | +0.0004 | 0.140 | [+0.0013, +0.0203] |
| 1h | +0.0473 | +0.0232 | +0.0241 | +0.0116 | +0.0124 | -0.0009 | 0.910 | [+0.0024, +0.0212] |
| 6h | +0.0498 | +0.0224 | +0.0275 | +0.0142 | +0.0125 | +0.0017 | 0.160 | [+0.0018, +0.0284] |

## Per wallet — primary (guard on)

### `0x09bed1976600971fd9f9db454a9c302c7a559c32`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 314 | 100% | 100 | +0.0515 | +0.0515 | +0.0345 | +0.0875 | -0.0529 | 0.810 | [-0.0414, +0.1472] |
| 2m | 15 | 5% | 9 | -0.1177 | -0.1174 | -0.1238 | -0.0318 | -0.0920 | 0.980 | [-0.4188, +0.0208] |
| 15m | 15 | 5% | 9 | -0.1177 | -0.1175 | -0.1239 | -0.0419 | -0.0820 | 0.940 | [-0.4167, +0.0206] |
| 1h | 15 | 5% | 9 | -0.1177 | -0.1158 | -0.1223 | -0.0235 | -0.0988 | 0.990 | [-0.4139, +0.0235] |
| 6h | 14 | 4% | 8 | -0.1314 | -0.1290 | -0.1348 | -0.0310 | -0.1037 | 1.000 | [-0.4514, +0.0194] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0376 | +0.0000 | +0.0376 | +0.0262 | +0.0262 | -0.0000 | 0.745 | [-0.0079, +0.0579] |
| 2m | -0.0325 | +0.0009 | -0.0334 | -0.0421 | -0.0094 | -0.0327 | 0.945 | [-0.1488, +0.0191] |
| 15m | -0.0325 | +0.0009 | -0.0335 | -0.0422 | -0.0145 | -0.0277 | 0.925 | [-0.1475, +0.0192] |
| 1h | -0.0325 | -0.0005 | -0.0320 | -0.0407 | -0.0041 | -0.0367 | 0.990 | [-0.1519, +0.0225] |
| 6h | -0.0398 | -0.0013 | -0.0385 | -0.0468 | -0.0071 | -0.0396 | 1.000 | [-0.1592, +0.0187] |

### `0x1ee9a5fc09665909c0cce297c581703bfbb9197f`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 5,271 | 100% | 2,616 | +0.4179 | +0.4179 | +0.2333 | +0.3119 | -0.0786 | 1.000 | [+0.0702, +0.4419] |
| 2m | 3,077 | 58% | 1,546 | +0.7062 | +0.6423 | +0.3743 | +0.3400 | +0.0342 | 0.255 | [+0.1184, +0.7342] |
| 15m | 3,062 | 58% | 1,539 | +0.7104 | +0.7181 | +0.3861 | +0.3416 | +0.0445 | 0.200 | [+0.1172, +0.7415] |
| 1h | 3,033 | 58% | 1,528 | +0.7203 | +0.5348 | +0.2973 | +0.4172 | -0.1199 | 0.980 | [+0.0487, +0.6394] |
| 6h | 2,806 | 53% | 1,371 | +0.7794 | +1.2423 | +0.4023 | +0.5393 | -0.1370 | 0.980 | [+0.0640, +0.8373] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0417 | +0.0000 | +0.0417 | +0.0281 | +0.0281 | -0.0000 | 1.000 | [+0.0105, +0.0472] |
| 2m | +0.0696 | +0.0121 | +0.0575 | +0.0413 | +0.0213 | +0.0200 | 0.000 | [+0.0148, +0.0683] |
| 15m | +0.0701 | +0.0212 | +0.0489 | +0.0330 | +0.0123 | +0.0207 | 0.000 | [+0.0065, +0.0594] |
| 1h | +0.0721 | +0.0260 | +0.0461 | +0.0305 | +0.0083 | +0.0222 | 0.000 | [+0.0040, +0.0559] |
| 6h | +0.0763 | +0.0437 | +0.0326 | +0.0176 | -0.0069 | +0.0244 | 0.000 | [-0.0104, +0.0443] |

### `0x69ea0d77ef34f1acb03aaed901df7620fc4215cd`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3,308 | 100% | 748 | +0.2050 | +0.2050 | +0.1017 | +0.1476 | -0.0458 | 0.990 | [+0.0423, +0.1792] |
| 2m | 1,947 | 59% | 447 | +0.2615 | +0.1926 | +0.1036 | +0.1540 | -0.0504 | 0.970 | [+0.0378, +0.1956] |
| 15m | 1,942 | 59% | 447 | +0.2621 | +0.1772 | +0.0969 | +0.1562 | -0.0593 | 0.970 | [+0.0337, +0.1818] |
| 1h | 1,905 | 58% | 443 | +0.2668 | +0.1754 | +0.0940 | +0.1893 | -0.0953 | 1.000 | [+0.0314, +0.1779] |
| 6h | 1,811 | 55% | 420 | +0.2784 | +0.0880 | +0.0693 | +0.2111 | -0.1418 | 1.000 | [+0.0271, +0.1149] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0598 | +0.0000 | +0.0598 | +0.0477 | +0.0477 | +0.0000 | 0.535 | [+0.0325, +0.0638] |
| 2m | +0.0714 | -0.0012 | +0.0726 | +0.0594 | +0.0577 | +0.0016 | 0.230 | [+0.0377, +0.0801] |
| 15m | +0.0715 | +0.0009 | +0.0707 | +0.0575 | +0.0562 | +0.0013 | 0.330 | [+0.0355, +0.0800] |
| 1h | +0.0726 | +0.0024 | +0.0702 | +0.0572 | +0.0557 | +0.0015 | 0.290 | [+0.0357, +0.0761] |
| 6h | +0.0741 | -0.0002 | +0.0743 | +0.0613 | +0.0596 | +0.0017 | 0.250 | [+0.0401, +0.0835] |

### `0x83255595ba1fadd2e734cb30a0fb8110301a19cc`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3,389 | 100% | 987 | +0.1973 | +0.1973 | +0.0536 | +0.0684 | -0.0148 | 0.840 | [-0.0795, +0.1997] |
| 2m | 10 | 0% | 7 | -0.2200 | -0.2443 | -0.2678 | +0.0819 | -0.3497 | 0.900 | [-0.5820, +0.0536] |
| 15m | 10 | 0% | 7 | -0.2200 | -0.2917 | -0.3116 | +0.0253 | -0.3369 | 0.870 | [-0.5728, -0.0701] |
| 1h | 7 | 0% | 4 | -0.3421 | -0.3406 | -0.3615 | +0.1083 | -0.4698 | 0.875 | [-0.6953, -0.2139] |
| 6h | 5 | 0% | 3 | -0.3964 | -0.3925 | -0.4114 | +0.0735 | -0.4849 | 0.905 | [-1.0000, -0.1912] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0502 | +0.0000 | +0.0502 | +0.0334 | +0.0334 | -0.0000 | 1.000 | [+0.0208, +0.0468] |
| 2m | -0.0242 | +0.0163 | -0.0405 | -0.0614 | +0.0396 | -0.1010 | 0.875 | [-0.2454, +0.0981] |
| 15m | -0.0242 | +0.0453 | -0.0695 | -0.0898 | +0.0057 | -0.0955 | 0.825 | [-0.3002, +0.0511] |
| 1h | -0.0657 | -0.0057 | -0.0600 | -0.0813 | +0.0501 | -0.1314 | 0.845 | [-0.2747, -0.0202] |
| 6h | -0.0900 | -0.0052 | -0.0848 | -0.1059 | +0.0371 | -0.1430 | 0.905 | [-0.4460, -0.0197] |

### `0xe542afd3881c4c330ba0ebbb603bb470b2ba0a37`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 6,421 | 100% | 2,550 | +0.9506 | +0.9506 | +0.1425 | +0.1468 | -0.0043 | 0.900 | [-0.1242, +0.4788] |
| 2m | 2,625 | 41% | 1,135 | +0.2866 | -0.0298 | -0.1776 | +0.0794 | -0.2569 | 1.000 | [-0.3668, +0.0297] |
| 15m | 2,617 | 41% | 1,129 | +0.2905 | -0.0602 | -0.2038 | +0.0791 | -0.2829 | 1.000 | [-0.3873, +0.0073] |
| 1h | 2,588 | 40% | 1,108 | +0.2559 | -0.0544 | -0.2039 | +0.1954 | -0.3993 | 1.000 | [-0.3960, +0.0170] |
| 6h | 2,472 | 38% | 1,046 | +0.2744 | -0.0113 | -0.1932 | +0.1869 | -0.3801 | 1.000 | [-0.3964, +0.0504] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0473 | +0.0000 | +0.0473 | +0.0321 | +0.0321 | -0.0000 | 0.930 | [+0.0119, +0.0557] |
| 2m | +0.0322 | +0.0201 | +0.0121 | -0.0038 | +0.0132 | -0.0170 | 1.000 | [-0.0273, +0.0204] |
| 15m | +0.0328 | +0.0263 | +0.0064 | -0.0093 | +0.0069 | -0.0162 | 1.000 | [-0.0309, +0.0139] |
| 1h | +0.0312 | +0.0242 | +0.0070 | -0.0085 | +0.0085 | -0.0170 | 1.000 | [-0.0315, +0.0150] |
| 6h | +0.0310 | +0.0269 | +0.0041 | -0.0112 | +0.0060 | -0.0172 | 1.000 | [-0.0331, +0.0125] |

## Per wallet — guard off

### `0x09bed1976600971fd9f9db454a9c302c7a559c32`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 314 | 100% | 100 | +0.0515 | +0.0515 | +0.0345 | +0.0845 | -0.0500 | 0.730 | [-0.0459, +0.1395] |
| 2m | 311 | 99% | 98 | +0.0498 | +0.0515 | +0.0345 | +0.0944 | -0.0599 | 0.810 | [-0.0463, +0.1427] |
| 15m | 311 | 99% | 98 | +0.0498 | +0.0456 | +0.0294 | +0.1698 | -0.1405 | 0.975 | [-0.0459, +0.1284] |
| 1h | 309 | 98% | 97 | +0.0499 | +0.0375 | +0.0226 | +0.2966 | -0.2739 | 1.000 | [-0.0470, +0.1106] |
| 6h | 264 | 84% | 73 | +0.0621 | +0.0318 | +0.0182 | +0.6524 | -0.6342 | 1.000 | [-0.0453, +0.1022] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0376 | +0.0000 | +0.0376 | +0.0262 | +0.0262 | -0.0000 | 0.735 | [-0.0060, +0.0625] |
| 2m | +0.0362 | -0.0023 | +0.0385 | +0.0272 | +0.0279 | -0.0007 | 1.000 | [-0.0046, +0.0652] |
| 15m | +0.0362 | -0.0002 | +0.0364 | +0.0257 | +0.0266 | -0.0009 | 0.880 | [-0.0038, +0.0575] |
| 1h | +0.0362 | -0.0009 | +0.0370 | +0.0270 | +0.0281 | -0.0011 | 0.945 | [+0.0020, +0.0570] |
| 6h | +0.0384 | +0.0048 | +0.0336 | +0.0242 | +0.0220 | +0.0023 | 0.470 | [+0.0038, +0.0494] |

### `0x1ee9a5fc09665909c0cce297c581703bfbb9197f`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 5,271 | 100% | 2,616 | +0.4179 | +0.4179 | +0.2333 | +0.3116 | -0.0783 | 1.000 | [+0.0595, +0.4487] |
| 2m | 5,216 | 99% | 2,598 | +0.4295 | +0.3900 | +0.2065 | +0.3472 | -0.1407 | 1.000 | [+0.0448, +0.4209] |
| 15m | 5,097 | 97% | 2,545 | +0.4425 | +0.4412 | +0.2181 | +0.3917 | -0.1736 | 1.000 | [+0.0398, +0.4329] |
| 1h | 5,052 | 96% | 2,522 | +0.4468 | +0.3090 | +0.1513 | +0.6717 | -0.5205 | 1.000 | [-0.0057, +0.3569] |
| 6h | 4,210 | 80% | 2,030 | +0.5445 | +0.7915 | +0.2182 | +0.8923 | -0.6742 | 1.000 | [-0.0212, +0.5103] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0417 | +0.0000 | +0.0417 | +0.0281 | +0.0281 | -0.0000 | 1.000 | [+0.0095, +0.0459] |
| 2m | +0.0439 | +0.0057 | +0.0382 | +0.0239 | +0.0217 | +0.0022 | 0.000 | [+0.0050, +0.0426] |
| 15m | +0.0443 | +0.0114 | +0.0329 | +0.0192 | +0.0161 | +0.0030 | 0.000 | [+0.0013, +0.0372] |
| 1h | +0.0446 | +0.0161 | +0.0285 | +0.0154 | +0.0122 | +0.0032 | 0.005 | [-0.0012, +0.0343] |
| 6h | +0.0530 | +0.0334 | +0.0197 | +0.0064 | -0.0010 | +0.0074 | 0.000 | [-0.0120, +0.0255] |

### `0x69ea0d77ef34f1acb03aaed901df7620fc4215cd`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3,308 | 100% | 748 | +0.2050 | +0.2050 | +0.1017 | +0.1453 | -0.0435 | 0.980 | [+0.0430, +0.1800] |
| 2m | 3,305 | 100% | 748 | +0.2052 | +0.1580 | +0.0964 | +0.1487 | -0.0523 | 0.995 | [+0.0466, +0.1575] |
| 15m | 3,307 | 100% | 748 | +0.2051 | +0.1212 | +0.0688 | +0.1514 | -0.0827 | 1.000 | [+0.0289, +0.1205] |
| 1h | 3,297 | 100% | 747 | +0.2055 | +0.1086 | +0.0575 | +0.2270 | -0.1695 | 1.000 | [+0.0184, +0.1089] |
| 6h | 2,851 | 86% | 616 | +0.2245 | +0.0581 | +0.0427 | +0.3584 | -0.3157 | 1.000 | [+0.0122, +0.0731] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0598 | +0.0000 | +0.0598 | +0.0477 | +0.0477 | +0.0000 | 0.605 | [+0.0321, +0.0644] |
| 2m | +0.0599 | -0.0036 | +0.0635 | +0.0512 | +0.0512 | +0.0000 | 0.975 | [+0.0363, +0.0674] |
| 15m | +0.0598 | +0.0040 | +0.0559 | +0.0440 | +0.0440 | +0.0000 | 1.000 | [+0.0289, +0.0595] |
| 1h | +0.0599 | +0.0091 | +0.0508 | +0.0397 | +0.0397 | -0.0001 | 1.000 | [+0.0256, +0.0541] |
| 6h | +0.0611 | +0.0056 | +0.0554 | +0.0440 | +0.0450 | -0.0010 | 0.910 | [+0.0276, +0.0592] |

### `0x83255595ba1fadd2e734cb30a0fb8110301a19cc`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3,389 | 100% | 987 | +0.1973 | +0.1973 | +0.0536 | +0.0687 | -0.0151 | 0.880 | [-0.0765, +0.2022] |
| 2m | 3,309 | 98% | 966 | +0.1864 | -0.1476 | -0.1886 | +0.1065 | -0.2951 | 1.000 | [-0.2705, -0.1062] |
| 15m | 3,254 | 96% | 962 | +0.1915 | -0.2059 | -0.2430 | +1.1574 | -1.4004 | 1.000 | [-0.3218, -0.1672] |
| 1h | 2,366 | 70% | 859 | +0.0862 | -0.3203 | -0.3360 | +5.0642 | -5.4002 | 1.000 | [-0.3784, -0.2932] |
| 6h | 96 | 3% | 39 | -0.1839 | -0.3649 | -0.3822 | +5.5380 | -5.9202 | 1.000 | [-0.5268, -0.2346] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0502 | +0.0000 | +0.0502 | +0.0334 | +0.0334 | -0.0000 | 1.000 | [+0.0217, +0.0466] |
| 2m | +0.0485 | +0.0526 | -0.0041 | -0.0194 | -0.0178 | -0.0015 | 0.975 | [-0.0307, -0.0077] |
| 15m | +0.0484 | +0.0548 | -0.0064 | -0.0199 | -0.0176 | -0.0022 | 0.985 | [-0.0306, -0.0094] |
| 1h | +0.0410 | +0.0431 | -0.0021 | -0.0132 | -0.0003 | -0.0129 | 1.000 | [-0.0236, -0.0025] |
| 6h | +0.0053 | +0.0157 | -0.0104 | -0.0255 | +0.0355 | -0.0610 | 0.910 | [-0.0679, +0.0200] |

### `0xe542afd3881c4c330ba0ebbb603bb470b2ba0a37`

**ROI per share** — `(resolved_value − cost) / cost`:

| Δ | n | cov | markets | own gross (cohort) | follower gross | follower net | null net | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 6,421 | 100% | 2,550 | +0.9506 | +0.9506 | +0.1425 | +0.1466 | -0.0041 | 0.870 | [-0.1322, +0.4647] |
| 2m | 6,418 | 100% | 2,548 | +0.9497 | +0.0869 | -0.1169 | +0.1988 | -0.3157 | 1.000 | [-0.3645, +0.2026] |
| 15m | 6,417 | 100% | 2,547 | +0.9499 | +0.1604 | -0.1650 | +0.8245 | -0.9896 | 1.000 | [-0.4055, +0.1772] |
| 1h | 6,358 | 99% | 2,510 | +0.9191 | -0.0315 | -0.1962 | +1.8736 | -2.0698 | 1.000 | [-0.4377, +0.1285] |
| 6h | 4,564 | 71% | 1,554 | +0.4708 | +0.1725 | -0.0734 | +0.4934 | -0.5668 | 1.000 | [-0.4180, +0.3978] |

**Edge per share in price units** — same cohorts, same shuffles:

| Δ | own edge | drift | follower edge (gross) | follower edge (net) | null | real − null | null p | net 95% CI (market-clustered) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | +0.0473 | +0.0000 | +0.0473 | +0.0321 | +0.0321 | -0.0000 | 0.960 | [+0.0118, +0.0545] |
| 2m | +0.0470 | +0.0280 | +0.0190 | +0.0038 | +0.0041 | -0.0003 | 0.980 | [-0.0155, +0.0275] |
| 15m | +0.0470 | +0.0317 | +0.0153 | +0.0008 | +0.0011 | -0.0003 | 0.995 | [-0.0180, +0.0233] |
| 1h | +0.0459 | +0.0299 | +0.0160 | +0.0024 | +0.0038 | -0.0014 | 1.000 | [-0.0170, +0.0253] |
| 6h | +0.0414 | +0.0238 | +0.0176 | +0.0030 | +0.0071 | -0.0041 | 0.940 | [-0.0236, +0.0330] |

## Sensitivity (pooled)

The midpoint series cannot tell us the true ask, and the resolution guard
and fill window are choices, so all three are swept. Shown in price units
(the stable metric); note that `real − null` is cost-invariant, so the two
tick rows differ only in `follower net`, not in copyable alpha.

| variant | Δ | n | cov | follower gross | follower net | null | real − null | null p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tick=0.0 | 2m | 7,674 | 41% | +0.0455 | +0.0399 | +0.0352 | +0.0047 | 0.025 |
| tick=0.0 | 15m | 7,646 | 41% | +0.0396 | +0.0341 | +0.0294 | +0.0047 | 0.040 |
| tick=0.0 | 1h | 7,548 | 40% | +0.0385 | +0.0332 | +0.0284 | +0.0048 | 0.065 |
| tick=0.0 | 6h | 7,108 | 38% | +0.0331 | +0.0280 | +0.0216 | +0.0064 | 0.010 |
| tick=0.001 | 2m | 7,674 | 41% | +0.0455 | +0.0389 | +0.0344 | +0.0045 | 0.050 |
| tick=0.001 | 15m | 7,646 | 41% | +0.0396 | +0.0331 | +0.0282 | +0.0049 | 0.030 |
| tick=0.001 | 1h | 7,548 | 40% | +0.0385 | +0.0322 | +0.0268 | +0.0054 | 0.050 |
| tick=0.001 | 6h | 7,108 | 38% | +0.0331 | +0.0270 | +0.0209 | +0.0061 | 0.035 |
| bandwidth=60s | 2m | 7,513 | 40% | +0.0455 | +0.0302 | +0.0257 | +0.0045 | 0.035 |
| bandwidth=60s | 15m | 7,499 | 40% | +0.0397 | +0.0246 | +0.0196 | +0.0049 | 0.055 |
| bandwidth=60s | 1h | 7,397 | 40% | +0.0383 | +0.0234 | +0.0185 | +0.0049 | 0.045 |
| bandwidth=60s | 6h | 6,969 | 37% | +0.0328 | +0.0182 | +0.0124 | +0.0059 | 0.010 |
| bandwidth=300s | 2m | 7,675 | 41% | +0.0455 | +0.0302 | +0.0256 | +0.0045 | 0.020 |
| bandwidth=300s | 15m | 7,648 | 41% | +0.0396 | +0.0245 | +0.0196 | +0.0048 | 0.020 |
| bandwidth=300s | 1h | 7,553 | 40% | +0.0386 | +0.0237 | +0.0186 | +0.0051 | 0.040 |
| bandwidth=300s | 6h | 7,112 | 38% | +0.0331 | +0.0185 | +0.0125 | +0.0059 | 0.020 |
| guard=0.0 | 2m | 18,559 | 99% | +0.0285 | +0.0141 | +0.0138 | +0.0003 | 0.100 |
| guard=0.0 | 15m | 18,386 | 98% | +0.0240 | +0.0104 | +0.0100 | +0.0004 | 0.140 |
| guard=0.0 | 1h | 17,382 | 93% | +0.0241 | +0.0116 | +0.0124 | -0.0009 | 0.910 |
| guard=0.0 | 6h | 11,985 | 64% | +0.0275 | +0.0142 | +0.0125 | +0.0017 | 0.160 |

## How to read this

Two different questions, and they must not be conflated:

1. **Does the follower end up positive after costs?** — `follower net`.
2. **Is that money *copyable alpha*, or the favorite-longshot base rate that
   any buyer at the same price level collects?** — `real − null`.

A positive `follower net` that the shuffled null reproduces is answer (1) yes,
answer (2) no, and it is answer (2) that decides whether *following these
wallets* is worth anything over buying at the same prices at random. In price
units the cost terms cancel out of `real − null`, so no fee or tick assumption
can rescue or destroy it.

A second diagnostic worth applying to any positive cell: **copyable alpha must
decay in Δ.** Information the wallet has and the market does not is worth less
the longer you wait. A `real − null` that is flat or *rising* from 2m to 6h is
not lag-alpha — it is a cohort or structural artifact.

## Verdict

**No. Copyable edge does not survive a 2-minute lag for any of these five
wallets.** On the 99%-coverage sample the pooled follower keeps +1.41c/share
net of real fees and one adverse tick — and the shuffled null keeps +1.38c of
it. Copyable alpha is **+0.03c/share (null p = 0.10)**: three hundredths of a
cent, against a certified held-out skill edge of +3.9c to +6.9c for these same
wallets. Per wallet at 2 minutes the alpha is −0.07c, +0.22c, +0.00c, −0.15c
and −0.03c; one wallet (`0x83255595ba…`) leaves a follower at **−1.94c/share,
CI [−3.07c, −0.77c]** — actively loss-making.

Where the edge goes, in order:

1. **~2.0c to price drift in the first 120 seconds.** These wallets are
   genuinely early — the price moves *toward* their side by 2c before a
   follower can act, which is 41% of the whole edge and is exactly the
   behaviour that made them certifiable. It is also exactly what a follower
   cannot have. Drift keeps growing to ~2.2c by 6h, so waiting is worse.
2. **~1.4c to fees and the adverse tick.** Real, unavoidable, and taker-only
   by construction (a copier reacts, so a copier is always the taker).
3. **~1.4c of what is left is the favorite-longshot base rate**, which the
   shuffled null collects without any knowledge of who traded.

The one cell that is nominally positive is the guard-on pooled row
(+0.45c, null p = 0.040) and its main contributor `0x1ee9a5fc09…`
(+2.00c, null p = 0.000 on 3,077 bets). Three reasons not to bank it:

* It **rises** with Δ (+0.45c → +0.58c from 2m to 6h; +2.00c → +2.44c for the
  wallet). Lag-alpha decays; a rising curve is a cohort artifact.
* It lives in the 41% of bets that survive the proportional guard — a
  long-dated, selected minority. On the same wallet's full 99% sample the
  alpha falls from +2.00c to **+0.22c**.
* p = 0.040 does not clear this project's pre-registered
  `scoring.project1.oos_significance_alpha` of **0.005**, and it is one cell in
  a 2 (guard) × 4 (Δ) × 6 (group) grid.

This is another independent negative on copyability in this repo, and the
first one measured on real-world certified wallets against a true market price
path rather than on the crypto-dominated ledger. `docs/realworld_deep_sample.md`
§6 said identification was solid and copyability untested; it is now tested.
Identification is unaffected — these wallets really do beat the price they pay.
The finding is that **the market prices their information in under two minutes**.

## Caveats

* `/prices-history` is a **midpoint**, not an executable ask. The gap between
  the tape's VWAP entry price and the midpoint at that instant (see Data) has a
  median under 1c but a mean above 3c, so the 1c adverse tick is if anything
  **optimistic** for the follower; 0.1c and 0c are swept for completeness.
  Costs cannot change the verdict anyway, because `real − null` in price units
  is cost-invariant.
* The wallet that costs a follower the most, `0x83255595ba…`, is also the one
  whose price runs away fastest: **+5.3c of drift in 120 seconds**, against a
  +4.9c own edge. Its entire edge is consumed before a copier can act, which is
  a cleaner illustration of the mechanism than the pooled average.
* Fees are charged on the whole history, though Polymarket was genuinely
  **zero-fee before 2026-01-05**. That is the right choice for a forward
  copying question and the wrong one for a historical P&L.
* The guard uses the market's Gamma lifespan, not the token's observed price
  series, because the endpoint caps a query at <30 days and cannot show a
  long market's full series.
