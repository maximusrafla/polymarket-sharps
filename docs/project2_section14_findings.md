# Project 2 — §1.4: De-biasing the near-resolution tape

**Date:** 2026-07-21. **Status:** the §1.4 follow-on to
`docs/project2_first_ranked_output.md`. That first run was a *correct pipeline on
biased data*: the top-120-by-volume market tapes return only the most-recent
~10.5k trades (the near-resolution slice), which inflated metric-A skill edges
10–50× (winner win-rate median 0.80; 16 of 83 undefeated). This document records
the fix — both prongs of spec §1.4 — and the de-biased re-score.

Read `docs/project2_forecaster_discovery.md` §1.2/§1.4 (the design) and
`docs/project2_first_ranked_output.md` (the biased first result) first.

**Isolation unchanged.** Everything writes ONLY to `data/interim/discovery/`
(gitignored/local). The shared `bet_ledger.parquet`, `backfill_cursors.json`, and
`data/raw/resolutions.parquet` were read read-only or untouched. No Project-1 code
(`validate.py`/`features.py`/`rank.py`) was modified.

---

## The bias, measured

The `/trades?market=` endpoint hard-caps at the most-recent ~10,500 trades per
market (offset ≤ 10,000). On the first run's top-120 markets:

- **116 of 120 hit the cap.** Only 4 returned a full tape.
- Where market lifespan is known, the retrievable tape covered a **median ~8% of
  the market's life** — a near-resolution sliver (Trump 2024: 10,500 trades
  spanning 0.35 of 305 days = 0.1% of its life).
- The consequence: nearly every *observed* bet on those markets is an endgame
  entry on the eventually-winning side, scored as if it were a forecast → the
  10–50× skill inflation.

The fix is **data-side**: get tapes whose early entries survive.

## Why offset paging can't reach the full-tape tail (probed live 2026-07-20/21)

The full tapes live in the **mid/low-volume** markets (fewer than ~10.5k trades),
but the volume-ranked `/markets` offset sweep cannot reach them:

- Every market in the offset-reachable head (`/markets?order=volumeNum`, offset
  0…~2000, all ≥ ~$4.5M volume) **hits the cap** — verified down to $5.6M.
- `ascending=true` is all **$0 dead markets** (bottom ~2000), then the 422 offset
  ceiling. The full-tape band ($0 < volume < ~$4.5M) is unreachable by offset
  paging in **either** direction.
- **`/events?tag_slug=` is the only path to it.** It surfaces hundreds of
  mid-volume markets per category with full tapes (probe: nfl 435, nba 707,
  politics 567, soccer 787 in the $100k–$4M closed band on one pass) — and gives
  category breadth directly.
- **Cap-hit is set by trade _count_, not $ volume** — two $3.8M markets, one full,
  one capped (retail-heavy markets accrue more trades). So volume is only a noisy
  proxy; the reliable signal is the recorded `hit_cap` per market.

---

## What was built (this session)

All in `src/discover.py`, with the scoring change in `src/forecaster_metrics.py`
and the dashboard in `src/rank_forecasters.py`. 176 tests green.

### 1. Live open-market early-tape capture (§1.4 prong 1) — the core new capability

`run_discover(... )`'s sibling **`run_live_capture()`** (CLI: `--live`). A
long-lived, checkpointed poll over OPEN real-world markets. Because the per-market
cursor is overlap-safe, each tick fetches only trades newer than the last, so as
long as fewer than ~10.5k arrive between ticks the **full tape — including the
early entries where skill lives — is recorded before it can scroll past the cap**.
Same rolling-overlap discipline `ingest.py` uses for the global feed, scoped
per-market (far slower per market, so trivially kept overlapping).

- Re-enumerates the open set every N ticks (markets open/close), backoff-guarded so
  a network blip never crash-loops, bounded by `--max-iterations`/`--max-minutes`.
- Checkpointed + resumable: cursors + tape + per-market stats all persisted; a
  restart re-enumerates and resumes from the cursors. Meant to run detached
  (setsid/nohup) — the model is not in the loop.
- **Demonstrated:** a 2-iteration run captured **101,601 trades across 12 open
  real-world markets** on tick 0; tick 1 (5 s later) returned **+0 new** — the
  overlap-safe cursor caught up exactly, no re-fetch, no gap. Its payoff is
  forward: it accrues clean early-entry tapes on high-liquidity markets over days,
  which is what eventually gives BOTH liquidity (metric B) AND an honest early tape
  (metric A) on the same market.

### 2. Mid-volume full-tape tail (§1.4 prong 2)

`run_discover(enum_mode="events")` (CLI: `--events --tags … --vol-min/--vol-max`).
Enumerates the full-tape band via `/events?tag_slug=`, pulls each market's tape
into the same isolated discovery dataset. Same cursor/checkpoint/dedup discipline
as the volume sweep. This is the lever that de-biases metric-A **now** (unlike live
capture, whose markets must first resolve).

### 3. Tape-completeness signal (the correctness lever)

`fetch_market_trades` now reports `hit_cap`; it is persisted per market in
`market_tape_stats.parquet` (sticky-OR: once truncated, always truncated) on the
same checkpoint cadence as the tape, so a kill mid-run can't lose it. Scoring joins
it as a per-bet `tape_complete`, and `market_tape_complete()` falls back to a
tape-length heuristic (≥ 10,450 rows ⇒ truncated) for markets pulled before the
signal existed. This is what lets the re-score **segment honest full-tape skill
from near-resolution-inflated skill** instead of trusting volume as a proxy.

### 4. Scoring + dashboard: the `full_tape` gate

`forecaster_metrics` adds two additive columns per (wallet, category) cell:
- **`frac_full_tape`** — fraction of the cell's bets on untruncated tapes.
- **`full_tape`** — `frac_full_tape ≥ 0.8` (config `scoring.min_frac_full_tape`).

`is_winner` (A∩B∩C∩breadth) is **unchanged**; a new **`winner_full_tape` =
is_winner ∧ full_tape** is the de-biased headline, and the dashboard's default view
+ sort now lead with it. Per the project invariant (§3.2), this is a toggleable
*view* — nothing is dropped, every cell keeps `frac_full_tape`, and the user can
relax the filter.

---

## The de-biased result — and the surprise it exposed

**Data:** the /events full-tape pull added ~465 mid-volume markets to the tape →
**585 markets, 2.73M trades, 2.63M resolved bets**, of which **438 markets are
recorded full-tape vs 27 capped** (plus the 120 legacy top-volume markets, all
cap-truncated). Re-scored to **657,223 (wallet, category) cells over 463,561
wallets** (metric A + the full_tape segmentation; metric B deferred, see caveats).

**The headline is not a winner list — it is that the near-resolution tape
truncation was NOT the primary driver of the apparent skill inflation.** Two
findings, together, reframe the first run:

1. **Pooled full-tape skill ≈ 0 at every entry time.** Binning the 835k full-tape
   resolved BUY bets by position in each market's life:

   | entry window | n | mean skill edge | win% |
   |---|---|---|---|
   | early [0–25%] | 85,577 | **+0.0019** | 0.74 |
   | mid [25–50%] | 186,726 | −0.0001 | 0.82 |
   | mid [50–75%] | 207,393 | −0.0009 | 0.72 |
   | late [75–100%] | 355,198 | **+0.0044** | 0.73 |

   Mean entry price ~0.73, win rate ~0.73 — a **calibrated, favorite-heavy** book
   with essentially **zero average residual skill, even for late entries.** The
   per-category favorite-longshot baseline is doing its job: on honest full tapes
   there is no systematic near-resolution inflation to find. (This is exactly what
   the first run *couldn't* see — its tapes were all endgame, so it had no early
   entries to contrast against.)

2. **Yet the persistence gate still fires far above the null.** Of **4,822** cells
   deep enough to run the chronological OOS split, **1,204 (25%)** cleared the
   one-sided OOS significance test (α=0.05) — **5× the 5% chance rate** — and
   **593 (12.3%)** also cleared the 2¢ magnitude floor. The full-tape persisted,
   breadth≥3 survivors (**317** cells) have **median skill +0.117 (~12¢)** and a
   **win% median 0.90 (29% undefeated)** — an order of magnitude above Project-1's
   validated 1–7¢, and the win-rate diagnostic did **not** normalize.

**Reconciling the two:** the population mean is ~0, but 25% "persist" positive at
~12¢. That is the signature of **selection + residual per-wallet baseline structure
surviving the chronological split** — the "favorite-longshot survives out-of-sample"
failure mode CLAUDE.md/HANDOFF flagged — searched across hundreds of thousands of
cells, **not** confirmed copyable skill and **not** near-resolution tape bias. The
"16 undefeated winners" of the first run were undefeated because they are the
selected tail of a massive multiple-comparison search; full tapes prove it by
showing the pooled distribution they were drawn from is centered on zero.

**So §1.4's data fix was necessary and works mechanically, but the honest
copyable-forecaster signal on this data is unconfirmed.** The binding problem is now
**statistical, not data-coverage**: the persistence gate needs a per-cell
shuffled-outcome null and/or a false-discovery-rate correction before any cell can be
called a forecaster, and the forward test (spec §3.3) remains the only real
arbiter. The mechanisms built here (live capture, the full-tape tail, the
tape-completeness signal) are the durable deliverable and the prerequisite that made
this measurable; the winner list is deliberately withheld rather than published as a
mirage.

---

## §1.5 — The shuffled-outcome null + FDR verdict (2026-07-21): SELECTION NOISE

The statistical gate §1.4 said was needed is now built and run
(`scripts/audit_forecaster_null.py`, lifting `audit_blackswan.py`'s shuffle-null
posture + Benjamini-Hochberg machinery). **It is decisive: no credible copyable
forecaster survives on this data.** The apparent winners are a multiple-testing
mirage plus a residue of concentrated single-event artifacts.

### The null
Permute resolved outcomes **within each (category, 1¢-price) bucket** over all
1.69M resolved BUY bets — preserving the per-category favorite-longshot base rate
the residual is measured against AND every wallet's price mix, destroying only the
wallet↔outcome link (i.e. forecasting skill). Re-score metric A on each of 500
shuffles against a FIXED baseline (only outcomes move). The vectorized scorer
reproduces the production gate exactly (1,204 significant / 595 persisted, matching
`forecasters.parquet` to the cell). Two readouts:

1. **Count-null (aggregate, assumption-light).** How many cells pass the gate under
   the null vs. for real:

   | gate | real | null mean | null p95 | P(null≥real) | **est. FDR** |
   |---|---|---|---|---|---|
   | significant (OOS p<0.05) | 1,204 | 742 | 767 | 0.000 | **0.62** |
   | persisted (A+C) | 595 | 235 | 257 | 0.000 | **0.40** |
   | winnerA (A+C+breadth+FT+RW) | 318 | 127 | 143 | 0.000 | **0.40** |

   The count *exceeds* chance (P(null≥real)=0 — there is *some* aggregate signal),
   but **~40% of the "winners" are manufactured by the structural null itself.** A
   list that is 40% false, with no way to say which 40%, is not publishable.

2. **Pooled-null empirical-p + BH-FDR (per-cell).** The OOS t-statistic is pivotal,
   so pool all null t's (2.29M of them; min empirical p 4.4e-7 — ample BH
   resolution) and score each real cell's one-sided empirical p against it, then BH.
   - **q=0.05 → 6 cells survive; q=0.10 → 8.** Of the 6, **2** also clear
     magnitude + breadth≥3 + full-tape + real-world.
   - **For contrast, BH on the *analytic* t-test p (H0: mean=0) would pass 932** —
     the anti-conservative count the structural null corrects (the analytic null
     wrongly trusts H0: per-wallet residual mean = 0, which a persistent finer-than-
     baseline price preference violates with no skill).

### The 2 survivors are concentrated-event artifacts, not forecasters
Both have a ~50¢ OOS "skill edge" (t≈85–130) — an order of magnitude past anything
real (Project-1's validated copyable edge is 1–7¢). Inspection shows why:
- `0x4686…` (sports_nfl, "n=60", breadth 5): the 60 fills are **one Sunday's 5 NFL
  games** (0.99-day span; 27 fills on a single Seahawks spread). ~5 correlated
  same-day positions, fill-count-inflated to n=60.
- `0xc0e9…` (all_real_world, "n=27", breadth 3): **3 props of one June Fed press
  conference** ("Will Powell say 'Tariff'/'Inflation' N+ times"), all 27 resolved
  YES. Breadth 3 but **one** independent event.

The exchangeable within-bucket shuffle treats each fill as independent, so it
*cannot* see that these n's are a handful of correlated events replicated — the one
failure mode the null structurally misses. A **concentration / independence guard**
folded into the winner definition catches them: `eff_breadth` (1/HHI of per-market
fill shares — effective independent-market count, robust to fill inflation) **≥ 3**
AND `entry_days` (distinct decision-days) **≥ 3**. `0x4686…` fails on entry_days=2
(one slate); `0xc0e9…` fails on eff_breadth=1.25 (one event). **After the guard: 0
credible winners.**

### Verdict
**The §1.4 "winners" are statistical selection (est. gate FDR ~40%) plus a residue
of concentrated single-event artifacts.** The winner list stays **withheld**, and
**metric B (copyability) is NOT run** — it would multiply onto a skill signal that
is a mirage. The durable deliverables remain the §1.4 discovery mechanisms (live
capture, full-tape tail, tape-completeness signal) and now this null+FDR+
concentration gate; the forward test (spec §3.3) is still the only real
arbiter. Per-cell flags → `data/interim/discovery/forecaster_null_fdr.parquet`;
headline scalars → `forecaster_null_summary.json`; the dashboard leads with the
verdict banner and a `WIN·NULL` column (0 rows).

**Residual limitation (next guard).** The concentration guard catches fill-inflation
over few markets/days, but the fully general case — correlated markets sharing one
*resolution event* (e.g. same-day slates) — needs **resolution-time clustering**,
which the discovery resolutions table lacks a timestamp for today. That, and the
forward test, are the follow-ons.

---

## Honest caveats / what's next

- **DONE — the statistical gate is built and run (§1.5 above).** The shuffled-outcome
  null + BH-FDR + concentration guard is the fix this doc called for; its verdict is
  **selection noise → no credible winner**. The forward test (spec §3.3) stays
  the real arbiter.
- **Metric B (copyability) was deferred on this run for a hard infra reason:** the
  box is 3.7 GB and the copyability token-array step peaks ~2.9 GB on the now-2.7M-row
  tape — it OOM'd the machine. Scoring was made lean (projected parquet read, dropped
  unused text columns, freed the tape before the base groupby, an RSS-watchdog runner)
  so **metric A runs at ~1.9 GB peak**, but B is left for a larger box or a targeted
  pass over just the A-passing cells. It doesn't change the story: copyability
  multiplies onto a skill signal that is ~0 in the pooled data.
- **Copyability is also thinner on mid-volume markets** even when computed: deep
  liquidity and a full tape are in tension at a fixed market size (mega markets =
  liquidity + truncated tape; mid-volume tail = honest tape + thin follow-on).
  **Live capture (prong 1) is what resolves the tension** — clean early tapes on
  high-liquidity markets — but only as those markets resolve over the coming
  days/weeks.
- **The live-capture markets are still open** ⇒ unresolved ⇒ not yet scoreable.
  Their value is realized on resolution; the mechanism is running/available.
- **Sample depth stays the binding constraint** for low-frequency real-world
  forecasters (spec §2.z) — real bet frequency caps n regardless of ingest.
- **Deferred:** `/markets/keyset` deep pagination for the sub-$4.5M tail beyond the
  offset ceiling (events reaches it category-first, which is what we use);
  promotion of the module-constant knobs to `config.yaml`; metric D wiring.

## How to run

```
# de-biasing data (detached; survives teardown; resumable via cursors):
python -m src.discover --events --tags politics,nba,nfl,soccer,economy \
       --markets-per-run 100 --max-markets 100 --closed-only

# continuous live early-tape capture on open markets (run detached):
python -m src.discover --live --poll-seconds 300 --reenumerate-every 12

# then resolve + re-score + null/FDR gate + dashboard:
python -m src.discover --resolve
python -m src.forecaster_metrics
python scripts/audit_forecaster_null.py            # shuffled null + BH-FDR + concentration gate (§1.5)
python -m src.rank_forecasters                      # dashboard joins the null flags -> WIN·NULL column
```

The null gate is statistically light (permutation + BH on already-scored cells, no
token arrays) — ~1.1 GB peak, safe on the 3.7 GB box even with the scorer's caveats.
`--shuffles` (default 500) sets the pooled-null resolution.
