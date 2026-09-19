# Project 3 step 8 — multi-tier aggregate forward test

**2026-07-25. Commits 0a17a4f → 5abd782.** Extends the step-7 freeze without moving it.

```bash
python -m src.slow_forward score            # weekly; updates the scoreboard
python -m src.slow_forward score --no-placebo   # faster, skips market-tape fetching
```

Live artifacts (all git-tracked): `data/processed/slow_frozen_set.parquet`,
`slow_freeze_manifest.json`, `slow_forward_scoreboard.md`, `slow_forward_scoreboard.parquet`.

## Why widen

The 12-only forward test is thin and slow. Testing a **crowd** rather than a wallet
is what makes forward evidence usable: three post-freeze bets cannot grade a
wallet, but 3×N can grade a cohort, because luck-selected wallets wash to zero
forward while a real crowd edge survives.

## The five pre-registered tiers

A tier is a **selection rule only**. Forward scoring is identical across all of
them, so a tier says how heavily vetted its members are and nothing else. All five
were vetted on the same pre-freeze data and share the **same freeze cutoff**
(2026-07-25T06:32:34Z).

| tier | rule | n | median edge | **eff. narratives** | median top-narr. share | single-narr. | pre-freeze bets |
|---|---|---:|---:|---:|---:|---:|---:|
| **t12** | niche baseline, complex clusters, complex gates — **HEADLINE** | 12 | +0.1285 | **3.13** | 0.51 | 1 | 3,514 |
| t30 | niche baseline, complex clusters, no complex gates | 30 | +0.1381 | 2.96 | 0.73 | 13 | 14,027 |
| t52 | 5c rule: coarse baseline, market clusters | 52 | +0.1514 | 2.24 | 0.81 | 28 | 15,542 |
| t87 | clears 10¢ + markets + concentration, **no significance test** | 87 | +0.1400 | 2.68 | 0.77 | 40 | 26,215 |
| t699 | in-sample skill > 0 and nothing else — **least vetted, volume only** | 699 | +0.0036 | 3.47 | 0.72 | 300 | 686,940 |

`t699`'s median edge of +0.0036 is a useful sanity check: an unvetted candidate
pool is retrospectively ~flat, which is what a meaningful vetting ladder should
look like at its bottom rung.

### Two things the measurement said, both recorded before any forward data existed

**1. The tiers are not a nested ladder.** There are two overlapping ones, because
they use different baselines: legacy `t699 ⊃ t87 ⊃ t52` (coarse category baseline,
market clusters) and standard `t30 ⊃ t12` (niche baseline, event-complex clusters).
`t30` is **not** a subset of `t52` (7 wallets outside), and neither is `t12` (2
outside). The manifest carries `tiers_are_nested: false` and a note saying so.

**2. Widening buys volume, not narrative breadth.** `t699` has **58× the wallets**
and **195× the pre-freeze bets** of `t12` but only **3.47 vs 3.13** effective
independent narratives — and `t87`/`t52` are actually *narrower* than `t12` (2.68,
2.24), because the wide crowd is disproportionately `mideast_escalation` (47 of 87,
33 of 52).

So forward **bet counts will grow far faster than forward evidence**. Every bet
count in the scoreboard is printed next to `effective_events` for exactly this
reason. A tier with 50,000 bets across 3 complexes carries the weight of ~3
observations and must never read as 50,000.

## The aggregate statistic

Headline: **event-weighted mean** — the mean over event complexes of the
within-complex mean residual, so one hyperactive rolling-deadline ladder cannot
become the answer. The bet-weighted mean is reported beside it; a large gap
between the two is itself a signal that one event is dominating.

Uncertainty resamples **event complexes**, not bets, and is **undefined below two
events** (NaN by rule, not a silent zero). Three properties are pinned by tests:

- an event with 1,000 bets and an event with 2 bets weigh the same
- **100× the bets in the same events does not narrow the CI**
- more *events* does narrow it

Piling bets into the same complexes cannot manufacture confidence. That is the
whole design.

Also reported: a **resolution-speed split** (≤14 days entry→resolution vs longer,
keyed on the market's own speed so a bet's stratum can't change between runs), and
a secondary `(complex × resolution-week)` event unit — shown so a reader can see
how much of the uncertainty is the unit choice, never so significance can be
harvested from the looser definition.

## The placebo — the control that outranks the baseline

Beating the frozen baseline is the **weak** test. The baseline is a calibration
curve fit on pre-freeze data, so a tier can clear it merely because the markets it
traded after the freeze were mispriced relative to that curve — anything lifting
*every* participant. That is the steps 6–7 "edge lives in the niche, not the
wallet" confound arriving in forward data, and the absolute number cannot see it.

For the same resolving markets, two relative readouts:

- `placebo_edge` — event-weighted edge of **all non-cohort** BUY bets in those markets
- `placebo_percentile` — the tier's edge against a null of **count-matched** random
  wallet crowds from those same markets

Crowd-size matching matters: a 12-wallet and a 699-wallet crowd have very
different sampling variance. Cohort wallets are excluded from their own placebo,
and both sides use the identical event-weighted statistic.

Market-tape fetching is capped per run (default 300) and **any truncation is
reported** in the scoreboard's coverage section — a silently capped placebo would
understate the control and read as a pass.

## Discipline

- **`t12` is the headline and stays the headline.** Wider tiers are pre-registered
  secondary readings; none is promoted on the basis of forward results. `t87` and
  `t699` carry their under-vetted status in their own manifest descriptions so a
  good number there cannot be read as if it came from `t12`.
- **The freeze cutoff did not move.** `--amend` preserves `freeze_ts` and logs the
  change with the forward-observation count at the time. This amendment: **0
  observations, `legitimate_pre_registration: true`**. The superseded scoring rule
  is retained in the log.
- **`--amend` now refuses**, because 647 forward observations have since been
  captured. The tier widening is therefore permanently on record as having
  happened before any outcome was observable.
- Baseline is the persisted pre-freeze curve; **no forward refit**.
- **Forward margins are expected to be smaller than the retrospective ones.** That
  is the expected consequence of removing selection, not a failure.

## Status

Harness verified end-to-end against live data: **699/699 wallets fetched, 647
post-freeze trades captured, 0 resolved slow bets so far.** The scoreboard is live
and correctly shows "awaiting forward data". The placebo path was separately
smoke-tested against a real market tape (159 rows fetched, 154 scored, percentile
computed against 46 other wallets).

Slow markets resolve over days to weeks, so the first scoreable sample is expected
in weeks and a decisive one in months. **Out of scope and deliberately not done:
deepening new wallets.** Scaling the crowd is a separate track, gated on whether
the early aggregate read looks alive.
