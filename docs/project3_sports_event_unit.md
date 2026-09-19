# Project 3 sports step 2 — the gate re-run at the resolution-event unit

**Measured 2026-07-25. Commits 496f1f9 → this one.** Read-only; no ledger contact.

```bash
python scripts/audit_sports_event_unit.py --shuffles 200      # ~12 min, corpus cached after the first run
```

Artifacts: `data/interim/sports/gate/event_unit_audit.parquet` (+ the isolated
copy of the gate corpus and its residualized cache, all gitignored).

## Why re-run a gate that already passed

The sports arm was opened on two numbers computed with `market_id` as the
cluster: ~8.4× excess cross-wallet skill dispersion and ρ=+0.48 split-half
persistence, both clearing a cluster-preserving null at the p-floor and described
as robust "at GAME level, 284 independent games".

**The 284 markets are not 284 games.** Joining them to their slugs:

| what they actually are | markets |
|---|---:|
| season outrights — `will-the-<team>-win-the-2025-nba-finals` (54 NBA), `-win-the-2025-world-series` (40 MLB), `-win-the-2025-stanley-cup` (20 NHL), World Cup / UCL fields (58 soccer), Super Bowl (17 NFL) | 117 |
| season props — `will-<player>-lead-the-nba-in-scoring` (26) / `-in-assists` (16) | 42 |
| derivative lines on **three** matches — `fifwc-esp-arg-2026-07-19-*` (28), `fifwc-fra-esp-2026-07-14-*` (27), `nfl-sea-ne-2026-02-08-*` (14), `nba-sas-nyk-2026-06-10-*` (14) | 83 |

Folded to what resolves (`src/sports_events.py`), **284 markets → 33 event keys**
(~25 genuinely distinct events; a couple of competitions split across
year-labelled and unlabelled slugs, which errs high). Concentration is severe:
`win-2026-fifa-world-cup` alone is 22.9% of corpus bets, the top five events 55%.
Per wallet it is worse — of the 2,081 findable (≥20-bet) wallets, **39.0% sit in
a single event and 81.3% in ≤2**, with a median of 6 "distinct markets" that are
typically six teams in the same tournament.

So a market-block bootstrap treats one wallet's six World Cup team-outrights as
six independent draws when they are one bet on one tournament whose outcomes are
mutually exclusive. That is the concentrated single-event artifact that killed
Project 2 §1.5 and cut the forecasters 52 → 12, one level up.

`niche_l1` (the league) is wrong in the other direction: the main ledger holds
**25,390 distinct coded games across 193 league codes** (`atp`, `cs2`, `mlb`,
`wnba`, `nwsl`, …), and collapsing every NBA game into one cluster would destroy
exactly the breadth that makes sports worth testing. The unit in between — the
resolution event — is what this measures.

## Result: the gate's population claim SURVIVES the correction

200 cluster-preserving shuffles per unit, `permute_wallets_within_market` applied
at the stated unit. The observed statistics do not depend on the unit; only the
null does.

| cluster unit | design effect (median) | dispersion excess | P(null≥obs) | split-half ρ | null ρ | P(null≥obs) |
|---|---:|---:|---:|---:|---:|---:|
| `market_id` (the gate's) | 2.46 | **7.29×** | 0.005 | +0.493 | +0.072 | 0.005 |
| `event` (corrected) | 1.80 | **8.32×** | 0.005 | +0.493 | −0.004 | 0.005 |

Both p-values are at the floor (1/201). The dispersion excess is **larger**, not
smaller, at the corrected unit, and the split-half null centres at **−0.004**
instead of +0.072 — i.e. correcting the unit *removes* a small upward bias in the
null rather than the signal. This reproduces the gate (7.29× vs its 8.4×, +0.493
vs its +0.48; the residual gap is baseline/shrinkage choice, not unit).

**The honest reading: the unit error does not invalidate the population claim.**
What it invalidates is *per-wallet* inference — a bootstrap over "20 markets"
that are one championship, and concentration gates counting those 20 as breadth.
That is precisely where validation happens, so the correction is load-bearing for
step 4 and not for the GO decision.

### Design effects say the unit is doing real work and is not over-constrained

Median DE is 2.46 at market level and 1.80 at event level, both >1, so clustering
genuinely matters. The repo's over-constraint trap (`cluster-preserving-null-required`:
permute-within-market measured DE 0.17 in the sparse black-swan tail and
manufactured its own false positives) does **not** reproduce here: only 5.9%
(market) and 14.8% (event) of findable wallets sit below DE 1.

## The new statistic: between-event persistence

Split-half persistence cannot separate skill from one lucky event in a corpus
where 81% of wallets span ≤2 events — both halves of a wallet's record are often
the *same* tournament, so ρ measures "the event resolved the way it resolved".
The honest question is whether skill in one event predicts skill in a **different**
one. Splitting each wallet's *events* (not its bets) in two:

| statistic | value |
|---|---|
| wallets with ≥2 events and both sides populated | **548** of 2,081 |
| observed ρ (early events → late events) | **+0.180** |
| cluster-preserving null (50 reps) | −0.044 ± 0.048 |
| P(null ≥ observed) | **0.0196** (the floor) |

Positive and null-decisive, but **much weaker than the +0.493 headline** — most
of that headline is within-event. This is the number to carry forward: it is the
one that survives being asked the copyable question.

## Limits, stated plainly

- **The event-level null destroys team-picking skill.** Holding the event fixed
  and permuting attribution reshuffles which team in the field a wallet held, so
  the null erases exactly the skill under test. That makes it the conservative
  bracket for "is there wallet skill" — the right direction for a gate re-check,
  and the reason the design effects above are reported rather than assumed.
- **The fallback is permissive.** A market matching neither rule becomes its own
  event: 3 of 284 markets (1.1%). Two competitions also split across labelled and
  unlabelled slugs (`win-world-series` vs `win-2025-world-series`), which
  over-counts independence slightly. Both errors run toward *more* clusters, so
  the corrected numbers are if anything generous.
- **This is still the screen's corpus.** Everything above is a population
  statement about 233k wallets, not evidence about any wallet. The shortlist
  remains selection noise until it survives step 4 on disjoint deep data.

## What this changes downstream

1. The cluster unit for sports validation is `event`, and the concentration gates
   are read in event units (`--cluster event`, complex gates on `event`). A wallet
   whose held-out record is one championship fails **by rule**, not by hand: the
   bootstrap has one cluster and returns NaN. Pinned by
   `test_one_event_wallet_is_rejected_at_the_event_unit`, alongside its contrast
   `test_one_event_wallet_is_certified_under_the_gates_setting`.
2. The deepening shortlist gains a third **breadth arm** (≥3 events,
   event-weighted skill > 0), because the gate's own two arms select almost
   entirely on within-event luck.
3. The GO stands. Deepening proceeds.
