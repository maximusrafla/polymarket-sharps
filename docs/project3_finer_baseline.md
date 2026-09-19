# Project 3 step 6 — the finer-baseline re-check: the coarse-baseline confound is REAL but SMALL, and a bigger one was found underneath

**Measured 2026-07-25. Commits c5b7d3c → 0301985.**
Reproduce:

```bash
python -m src.slow_validate --fdr-q 0.10                                  # 5c reference
python -m src.slow_validate --baseline niche_form --fdr-q 0.10            # finer baseline
python -m src.slow_validate --baseline niche_form --cluster niche_l1      # + event-complex null
python scripts/audit_slow_niche.py --level niche_l1 --save                # the copier diagnostic
```

## The question this closes

The 5c positive (52 `edge_persisted`, held-out +0.14…+0.54) residualizes against
`E[outcome | price, category]`, and 59% of slow bets are `category == "other"`.
The coordinator elevated the resulting worry above the verdict doc's footnote:
if a sub-niche inside `other` is structurally mispriced, **every** buyer in it
earns a large positive residual whether skilled or not — the favorite-longshot
problem one level up. It survives disjoint data, the cluster null, the
concentration guards and the horizon check, and the forward test cannot separate
it either, because a persistently mispriced niche pays wallet and copier alike.
Only a finer baseline separates them.

**Answer: the confound is real in kind, but roughly an order of magnitude too
small to account for the result. The re-check passes.**
**But measuring it surfaced a larger confound that does most of the damage.**

## Headline grid

Deep slow universe: 1,397,864 resolved BUY bets / 1,295 wallets / 55,160 markets.

| baseline | significance clusters | persisted | BH-FDR @ 0.10 |
|---|---|---:|---:|
| `category` (5c) | `market_id` | **52** | 55 |
| `niche_form` | `market_id` | **46** | 52 |
| `category` (5c) | `niche_l1` (event complex) | **34** | 26 |
| `niche_form` | `niche_l1` | **30** | 27 |

- Finer baseline alone: 43 of the original 52 hold, 3 new join → 46.
- Event-complex clustering alone: 28 of 52 hold, 6 new → 34.
- Both: 23 of 52 hold, 7 new → **30**.

## 1. The baseline result — the confound is small

Per-wallet edge shift over the 52, coarse → finest baseline with
leave-one-wallet-out: **median −0.0103** (about one cent), mean −0.0107, range
−0.0945 … +0.0298. The `+0.40…+0.54` band survives **8/8**. Wallets with
`edge_5c ≥ 0.20` survive 16/17; the losses are concentrated in the marginal
`edge_5c < 0.15` tail (16 of 24 hold), and they fail on cluster-p just above
0.05 rather than on magnitude.

This is **not** a null strip. The estimated shrinkage constants are small —
`k(niche_l1) = 3.97`, `k(niche_l2) = 2.88` — meaning the method-of-moments
estimator found genuine between-niche variance and let the fine cells keep their
own data (mean realized pooling weight 0.995, with 96.4% of bets in cells that
cleared the 50-bet floor). The niches really do differ from their parent
category; stripping that difference simply costs the survivors about a cent.

### The copier diagnostic — the direct version

`scripts/audit_slow_niche.py` decomposes each survivor's held-out edge as

```
wallet_vs_parent  ≈  niche_population_residual  +  wallet_vs_niche
```

where `niche_population_residual` is the mean residual of **all** participants in
the niche — literally "what a copier collects by trading the niche blind" — read
off the **discovery corpus** (market-first, 542k wallets, unselected), not the
screen-selected deep tape.

Every niche's unselected population residual lands within ±3.5¢:

| niche | pop. residual | bets | wallets |
|---|---:|---:|---:|
| `lg_nba` | +0.035 | 9,128 | 1,854 |
| `geo_israel` | +0.019 | 12,057 | 4,377 |
| **`geo_iran`** | **+0.010** | **128,433** | **46,783** |
| `us_politics` | +0.002 | 70,698 | 31,091 |
| `sports_soccer` | −0.000 | 250,402 | 87,469 |
| `lg_fifwc` | −0.004 | 54,563 | 12,945 |
| `intl_politics` | −0.005 | 135,245 | 49,351 |
| `geo_ukraine` | −0.020 | 22,044 | 10,303 |
| `geo_asia` | −0.031 | 14,706 | 7,513 |

Against survivor edges of 10–54¢ that leaves:

- **median niche-explained fraction +6.6%**
- **1 of 52** survivors is >50% niche-explained (`0xdc26d77769`, whose top-3-niche
  coarse edge is only +0.021 — a marginal name that fails the finer gates anyway)
- **45 of 52** still beat their own niche's fair rate by ≥10¢
- discovery covers **87.6%** of survivor niche volume

For the niches discovery does not reach, the deep-tape number is an *upper bound*
(the deep wallets were screened for positive residuals, so it is biased up), and
those come in at ≤0 anyway: `commodity_crude` +0.0003, `weather_temp` −0.0105,
`hormuz_shipping` −0.0114, `ai_tech` −0.0367.

## 2. The bigger confound found underneath: rolling-deadline ladders

The slow `other` bucket is not amorphous. It is dominated by mechanically
generated market families, and the largest single one is the 2026 US–Iran
complex: **`geo_iran` is 11.5% of the entire slow universe** (160,396 bets), and
the single niche `geo_iran|deadline` is 9.9%. It takes the form of a *rolling
deadline ladder*:

> "US strikes Iran by January 31 / February 27 / February 28 / March 1 /
> March 15 / March 31 / June 30 …"

Dozens of distinct markets, distinct resolution days, nearly all resolving NO,
**one underlying event**. Every existing guard counts them as independent:
`eff_breadth` (up to 60), `entry_days`, the §3.3 resolution-day clustering guard,
and the market-block bootstrap all see many separate markets.

Resampling **event complexes** instead of markets costs **18 of the 52** — three
times the damage the baseline question did.

The structure is stark at the top of the table: **5 of the top 8 by 5c edge —
including the +0.54 name — have their entire held-out record inside a single
complex**, so the complex bootstrap returns NaN. One event, one observation,
nothing to replicate against. The highest edges in the 5c set are precisely the
most concentrated ones.

This is the same lesson as `cluster-preserving-null-required`, one level up: the
cluster unit has to be the unit of *independent resolution*, and for a deadline
ladder that is the narrative, not the market.

## Method notes

- **Partition is frozen and syntax-only** (`src/slow_niche.py`): family (event
  complex / generator) × form (moneyline, spread, total, deadline, outright,
  threshold, …), derived from slug/question text alone — never from outcomes,
  residuals, prices or wallet identity, so it cannot be tuned toward a verdict.
  9.3% of the universe falls to `misc` and pools to its parent.
- **The brief assumed Gamma `tags` were available.** They are not: the sidecar
  column is populated for **16 of 348,657 markets** — written but never filled.
  Slug syntax is also strictly more informative here, since it encodes *form*,
  which tags do not.
- **Hierarchical partial pooling** (`src/slow_baseline.py`):
  `m[level,bin] = (n·ȳ + k·m[parent,bin])/(n+k)` along
  `niche_l2 → niche_l1 → category → global`, `k` estimated per level by method of
  moments, plus a hard 50-bet floor below which a cell takes its parent's value
  exactly. Bin edges are shared across levels (5c gave each category its own),
  which is a prerequisite for pooling and is verified near-inert on its own:
  per-wallet residual correlation 0.9986, 51 of 52 survivors retained.
- **Leave-one-wallet-out**, recursed up the whole parent chain. Not in the brief,
  but required: `features.fit_price_baseline` documents (correctly) that LOWO is
  unnecessary market-wide, and that reasoning **breaks** at niche granularity,
  where a wallet can be a large share of one niche's volume and would
  residualize against itself — collapsing to zero mechanically and producing a
  false negative indistinguishable from "no skill". In the event it changed the
  count by ≤1, but without it a collapse verdict would have been uninterpretable.
- **Anti-trap reporting**: bin populations and realized pooling weights are
  printed per level, so "these niches were not fit on noise" is checkable.

## Honest limits

- **Selection bias in the fitted baseline.** Fit over the deep tape, the baseline
  inherits the screen's selection (deepened wallets were chosen for positive
  residuals), so `E[outcome|price]` sits above the true population curve and
  residuals are biased *down*. That is conservative for "survivors persist" and
  anti-conservative for "survivors collapse" — which is why the copier
  diagnostic reads its population numbers off discovery instead.
- **`misc` is 9.3%** of the universe and pools to its parent category; a
  mispriced niche hiding inside `misc` would still be un-stripped. The copier
  diagnostic covers it separately (`misc` population residual −0.0037).
- **Still retrospective.** Nothing here is a forward test.
- **~50% aggregate FDR context** applies as before; individual small-n names
  remain provisional.

## What this means for the next step

The baseline question the coordinator raised is **answered, and answered
negatively**: niche miscalibration does not explain the 5c positive. On that
axis the result is real.

But **metric B and the forward freeze are not yet earned**, and the reason is
new rather than the one we set out to test. The surviving set under both
controls is **30, not 52**, and the highest-edge names are exactly those whose
record is one event complex. Before a prospective freeze it is worth:

1. adopting the **event-complex block** as the standard cluster unit for the slow
   path (it is strictly harsher and the ladder structure is pervasive), and
2. re-reading the concentration gates in complex units — `eff_breadth ≥ 3` over
   markets is not the same guarantee over complexes.

Freezing the 30 rather than the 52 is the defensible version of the forward test.
