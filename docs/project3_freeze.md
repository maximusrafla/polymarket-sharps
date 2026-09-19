# Project 3 step 7 — complex-unit gates, narrative honesty, and the pre-registered freeze

**2026-07-25. Commits 499b6d2 → 3123d9c. Retrospective work on Project 3 ends here.**

```bash
python -m src.slow_validate --fdr-q 0.10                 # the STANDARD pipeline -> 12
python scripts/audit_narrative_concentration.py --save   # narrative spread
python -m src.slow_forward freeze                        # (already run; refuses to re-run)
python -m src.slow_forward score                         # run periodically from now on
```

## 1. The event complex is now the pipeline's cluster unit

Step 6 found that the market is the wrong unit for the slow path: a rolling-deadline
ladder ("US strikes Iran by Feb 28 / Mar 1 / Mar 15 / Mar 31 …") is dozens of
distinct markets with distinct resolution days but **one event**. This makes the
correction the default rather than a post-hoc flag.

`STANDARD_BASELINE = niche_form`, `STANDARD_CLUSTER = niche_l1`, complex gates ON.
The concentration floors are re-read in complex units **alongside** the market-unit
ones — no gate removed, nothing dropped, complex columns reported as additive
metadata even when they do not gate:

| gate | value |
|---|---|
| `min_oos_complexes` | ≥ 3 |
| `min_eff_breadth_complex` | ≥ 3.0 |
| `min_entry_days_complex` | ≥ 3 (distinct (complex, entry-day) pairs) |
| `min_resolution_days_complex` | ≥ 3 (distinct (complex, resolution-day) pairs) |

**The thresholds are unchanged from their market-unit values by design.** This
corrects the unit; it does not re-tune the number. Relaxing to ≥2 would have
returned 19 instead of 12 — reverse-engineering the gate to a target, which is the
"relax until the table fills" failure this repo exists to avoid.

### The clean count

| configuration | persisted |
|---|---:|
| category baseline, market clusters (5c) | 52 |
| niche_form baseline, market clusters (6d) | 46 |
| niche_form baseline, complex clusters (6d) | 30 |
| **+ complex-unit concentration gates (STANDARD)** | **12** |

Median `top_complex_share` falls from **0.61** across the 30 to **0.37** across the 12.

Single-ladder exclusion happens **by rule**: one complex → the bootstrap has a
single cluster → NaN p → `significant` is False. The +0.54 name (`0xe89c4c9e0e`,
100% `geo_iran`) falls out mechanically, not by hand. Pinned by
`test_single_complex_wallet_is_not_persisted_by_rule`.

## 2. Narrative concentration — how broad is the set really?

The complex unit catches one ladder. It does **not** catch a wallet betting
correlated complexes: "Iran closes Hormuz" + "Israel strikes Iran" + "Hormuz
shipping falls" are three complexes but one geopolitical bet. Same confound, one
level up.

A frozen, versioned narrative map sits above the complex. Because `commodity_crude`
spikes on Hormuz risk but also trades on OPEC/demand/inventories, the map is
applied **both ways** rather than decided by fiat — `strict` (crude standalone) and
`broad` (crude folds into `mideast_escalation`).

| cohort | dominant narratives | **effective independent** | median top-narrative share | single-narrative (≥80%) |
|---|---:|---:|---:|---:|
| **PRIMARY (12)** | 4 | **3.13** (strict = broad) | 0.51 | 1 of 12 |
| SECONDARY (30) | 8 strict / 7 broad | 3.24 / **2.96** | 0.70 / 0.73 | 13 of 30 |

Primary breakdown: `mideast_escalation` 5, `sports` 4, `intl_politics` 2, `other` 1.

**The finding that matters: both cohorts carry roughly the same ~3 effective
independent narratives.** The secondary's extra 18 wallets add almost no
independent evidence — they are largely replicates of the same narratives, and 13
of them are single-narrative. That independently vindicates headlining the primary
rather than the wider set: the wider set is bigger without being broader.

Also worth noting: strict and broad are **identical for the primary cohort**, since
no primary wallet is crude-dominated. The judgement call that could have been
accused of steering the result has zero effect on the headline.

This is reported as metadata and a flag. It **never** gates `edge_persisted` and
never removes a row — the regress does not terminate retrospectively, and chasing
it further would just relocate the problem.

## 3. The freeze

**Frozen 2026-07-25T06:32:34Z at commit `dcae30d5`, partition v1.0.0.**

- `data/processed/slow_frozen_set.parquet` — 30 wallets, full gate row + frozen
  narrative metadata, `cohort ∈ {primary, secondary}`.
- `data/processed/slow_freeze_manifest.json` — freeze timestamp, git commit,
  partition/narrative versions, **every** gate parameter, the verbatim scoring
  rule, the fitted baseline, and the known limits.

Pre-registration properties:

- **The baseline is fit on pre-freeze data only and travels inside the manifest.**
  A re-score months from now needs no refit and no access to the original tape, so
  the forward number cannot drift because the baseline moved.
- **`primary` is the headline**, named as such in the manifest. `secondary` is
  frozen so its forward data is not wasted, but it re-admits the single-complex
  concentration the gates removed, and is never the headline.
- **Cohorts are not re-frozen in light of forward outcomes.** Picking the cohort
  after seeing which one worked would reintroduce exactly the selection this chain
  exists to eliminate. `freeze` refuses to overwrite without `--force`.
- **The known limits are on the record before any forward data exists** — they are
  a pre-commitment, not an explanation offered afterwards.

### Known limits, recorded at freeze time

1. The cohorts carry only **~3 effective independent narratives** despite their
   wallet counts. They are far fewer independent pieces of evidence than *n*
   suggests.
2. The primary cohort is **mideast-weighted** (5 of 12 dominant). If that narrative
   goes quiet, the primary forward test may have **no bets to score for months**.
   Accepted going in; it is why the secondary is frozen too.
3. Slow markets resolve over days to weeks, so a scoreable sample takes months.
   This is not a real-time signal and needs no `watch.py`.
4. ~50% aggregate FDR context still applies to individual names; the cohort
   aggregate is the unit of inference.
5. **Identification only.** Copyability (metric B) is untested — an edge that is
   real is not thereby fillable.

## Operating from here

`python -m src.slow_forward score` — run periodically (weekly is ample). It pulls
the frozen wallets' post-freeze bets via the public `/trades?user=` endpoint,
resolves them, scores them against the frozen baseline, and reports per-cohort
forward edge with a complex-block bootstrap p. Read-only, no keys.

Verified end-to-end at freeze time: all 30 wallets fetched, 0 post-freeze trades,
correct empty-case handling.

**The forward clock is running. Retrospective analysis on Project 3 is done, and
the next real information arrives on calendar time, not from more re-analysis.**
