# Project 3 sports steps 3–5 — deepen, validate, freeze

**Measured 2026-07-25. Commits 7b16f8a → ce868d6.** Read-only; the shared ledger
was never opened for writing. Reproduce:

```bash
./scripts/run_sports_arm.sh                      # the whole chain, resumable
python -m src.sports_validate --finer-recheck    # just the mandatory re-check
```

Artifacts: `data/processed/sports_frozen_set.parquet`,
`sports_freeze_manifest.json`, `sports_forward_scoreboard.md` (+ `.parquet`),
all git-tracked. The isolated deep tape lives in `data/interim/sports/`.

## 1. Deepening — 293 wallets, three arms

The screen ran on the gate corpus, whose 284 markets fold to ~25 resolution
events with 81% of findable wallets spanning ≤2 of them
(`docs/project3_sports_event_unit.md`). It is a **selection device only**. Two of
the three arms reproduce the gate's own shortlist almost exactly, which confirms
the screen is the same object it used:

| arm | rule | n | vs the gate |
|---|---|---:|---|
| core | shrunk residual skill ≥ 10¢ | 26 | its 29 |
| default | shrunk residual skill ≥ 2¢ | 136 | its 135 |
| **breadth** | ≥3 events **and** event-weighted skill > 0 | 212 | new |

Union **293 wallets**. The breadth arm exists because the first two select almost
entirely on within-event luck; it is the only arm whose members cannot be one
lucky tournament. Median shortlist event-breadth 3; only 7.5% span a single
event, down from 39% in the findable population.

Deepening pulled **2,021,449 trades across 212,826 markets**. Resolution is
restricted to the 52% of markets that are sports — the rest are the wallets'
crypto/politics history, which this arm never scores, and the CLOB rate-limits to
~20 req/s, so that halves an 80-minute step.

## 2. The deep sports universe — this is what the arm was opened for

| | screen corpus | deep data |
|---|---:|---:|
| resolved BUY bets | 765,485 | **1,135,069** |
| markets | 284 | **105,331** |
| **resolution events** | **~25** | **40,256** |
| leagues | 8 | 212 |

96,986 of the 105,331 markets are single games; only 3.0% fall back to
"own event" for want of a parse. Top leagues by bet share: `lg_fifwc` 28.2%,
`lg_nba` 7.9%, `lg_mlb` 7.0%, `lg_atp` 6.0%, `lg_cs2` 4.5%, `lg_wta` 3.5%.

This is the disjoint-data firewall working as designed: the screen's ~25 events
could never have supported per-wallet inference, and the deep tape has 40,256.

## 3. Validation — 174 candidates, ONE survivor

Event clusters, event-unit concentration gates, hierarchical LOWO league|form
baseline, scoped 10¢ floor (`scoring.sports.*`; the global 2¢ is untouched).

| stage | n |
|---|---:|
| wallets with enough resolved sports bets | 291 |
| candidates (in-sample skill > 0) | 174 |
| clearing magnitude + markets + concentration | 4 |
| **`edge_persisted` (all gates incl. cluster-robust significance)** | **1** |
| BH-FDR survivors @ q=0.10 | 1 |

Held-out skill across candidates: **median +0.0051**, p90 +0.0324, p99 +0.1097,
max +0.1876. Magnitude sensitivity (diagnostic; the pre-registered floor does not
move): 12 wallets clear everything else at a 2¢ floor, 4 at 5¢, 1 at 10¢, 1 at 15¢.

**The survivor — `0x8ade1d07fd3fa1b1821e54a142f2fb3b6c01e692`**

| | |
|---|---|
| held-out skill edge | **+0.1876** |
| held-out bets / markets / **events** | 3,962 / 254 / **182** |
| effective event breadth (1/HHI) | 25.19 |
| top event share | 0.1095 (`lol-los-png1-2026-01-31`) |
| cluster-robust p (event blocks) | 0.0205 |
| pre-freeze bets / events / leagues | 7,923 / 335 / 22 |

It is not a one-tournament artifact: 182 independent held-out events, no event
above 11% of its record, and the held-out edge is spread across leagues —
`lg_lol` +0.248 (58 events), `lg_nba` +0.229 (36), `lg_dota2` +0.163 (12),
`lg_nhl` +0.075 (45).

## 4. The mandatory finer-baseline re-check — PASSES

The confound that cut the forecasters 52 → 12: a mispriced sub-niche hands every
buyer in it a positive residual. Here the finer level adds the line **side**
(home/away/draw/over/under — where home-favourite bias would live), and price
bins are separately doubled.

| baseline | bins | candidates | persisted | median edge shift vs standard |
|---|---:|---:|---:|---:|
| `league` (coarse) | 20 | 173 | **0** | +0.00035 |
| `league_form` (**standard**) | 20 | 174 | 1 | — |
| `sub_form` (+ side) | 20 | 174 | **1** | −0.0000002 |
| `league_form` | 40 | 172 | 1 | +0.00018 |
| `sub_form` (+ side) | 40 | 171 | **1** | +0.00021 |

Refining the baseline moves the survivor's edge by **less than 0.0003** — three
orders of magnitude below its +0.1876. For contrast, the same check cost the
forecaster survivors ~1¢ each and killed 6 of 52. And the cells are not being fit
on noise: at `niche_l3`/40 bins the estimated shrinkage k is 2.58 with 85.8% of
bets in cells that cleared the 50-bet floor.

**The coarse baseline gives ZERO survivors**, i.e. the edge is if anything
*masked* by a coarse fit rather than manufactured by one. Both directions
checked; both clean.

## 5. Freeze — a new pre-registered forward tier

**Frozen 2026-07-25T23:25:48Z at commit `8ce1f0b`, event partition v1.0.0, with
`forward_observations_at_freeze = 0`.**

| tier | wallets | pre-freeze events | leagues | median held-out edge | median held-out events |
|---|---:|---:|---:|---:|---:|
| **s_core** (headline) | 1 | 335 | 22 | +0.1876 | 182 |
| s_wide (no significance test) | 4 | 614 | 43 | +0.1113 | 87 |
| s_all (volume only) | 174 | 31,651 | 212 | +0.0051 | 134.5 |

A **separate artifact** from the forecaster freeze, deliberately: that one already
has 647 forward observations against it and correctly refuses amendment, so
bolting a sports tier on would either break its rule or force a shared cutoff
that means nothing for a population frozen months later.

The breadth contrast is the whole argument for this arm: the 699-wallet slow tier
carries ~3.5 effective narratives; `s_all` spans **31,651 distinct events across
212 leagues**.

**Sports resolves in hours-to-days, so this scoreboard should populate in days —
not the forecasters' months.** `python -m src.sports_forward score`.

## 6. What to be suspicious of — read this before believing the survivor

1. **Its edge is entirely in the held-out half, and that half is 13 days long.**

   | half | bets | events | leagues | span | mean residual |
   |---|---:|---:|---:|---|---:|
   | in-sample | 3,961 | 155 | 18 | 2025-12-21 → 2026-01-29 | **+0.0011** |
   | held-out | 3,962 | 182 | 11 | 2026-01-29 → 2026-02-10 | **+0.1876** |

   The chronological split makes this "persistence" mean *held up over two
   weeks*, not over months. A wallet at ~0 for six weeks and +18.8¢ for two is
   either a genuine regime change (metric D's became-sharp shape) or a hot
   streak that the event-block bootstrap could not rule out.

2. **The FDR correction is over 4 tests, not 174.** Per repo convention BH-FDR is
   applied to candidates that clear the economic + concentration gates. At
   m=4 the q=0.10 threshold is 0.025 and p=0.0205 passes; across all 174
   candidates the rank-1 threshold would be 0.00057 and **it would not**. The
   selection burden is genuinely 174-wide, and the survivor sits close to the
   line.

3. **The profile looks like high-frequency in-play trading.** ~300 bets/day for
   13 days, mid-price entries (mean 0.433, realizing 0.608), across 11 leagues
   simultaneously. If the edge comes from reacting to live game state faster than
   the book, it is real skill and **the least copyable kind there is** — a
   copier arrives after the price has already moved. Nothing here measures that.

4. **This is identification, not copyability.** Metric B is untested and is the
   priority next step; sports is the one population where it can be answered in
   days.

## 7. Project 1 is bit-identical — verified, not asserted

The sports arm added a `scoring.sports` block and two optional parameters to
`slow_validate.validate_slow`. Both were checked three ways:

1. **Unit-level:** Project 1's `compute_oos_validation` run on the same ledger
   with and without an extreme sports block returns frames that compare equal
   (`test_project1_output_is_bit_identical_with_and_without_the_sports_block`).
2. **Refactor-level:** `validate_slow` called the old way (defaults) and the new
   way (explicitly passing what used to be hardcoded) returns equal frames under
   both cluster settings — so the frozen forecaster cohort cannot have moved.
3. **End-to-end:** `python -m src.validate` re-run against the live 4.7M-row
   ledger after all changes. `wallet_validated.parquet` is **identical across all
   26 columns and all 23,956 rows**, **41 persisted** before and after, and
   `ranked_wallets.parquet` is byte-unchanged (sha256 `ad011e7215c1bce6a72c5f80`).
   The magnitude floor printed by that run is `+0.020` — the global 2¢, untouched.

## 8. Status

Forward scoreboard is live and correctly shows "awaiting forward data" — 174/174
frozen wallets fetched, 0 post-freeze trades at freeze time (the freeze is one
minute old). Re-run `python -m src.sports_forward score` in a few days; the first
readable sample should arrive within days, and the tiers are never re-frozen in
light of what it says.
