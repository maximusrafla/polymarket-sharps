# Project 3 — Stage 5c verdict: slow-market forecasters DO exist (identification)

**Measured 2026-07-25 via the discovery→backfill→validate chain (steps 1→2→4→5).**
Reproduce: `python scripts/resolve_slow_deep.py && python -m src.slow_validate --fdr-q 0.10`.

## The result

Running the crude wide screen on the **shallow** combined corpus (5a), deepening the
1,518-wallet shortlist via `/trades?user=` into an **isolated** dataset (5b), and
validating on that **freshly-fetched deep data the screen never saw** (5c):

- deep slow universe: **1,397,864 resolved BUY bets, 1,295 wallets, 55,160 markets**
- 699 candidates (in-sample residual skill > 0)
- 87 clear the **10¢ magnitude** + held-out-markets + concentration gates
- **52 `edge_persisted`** — all gates *including* cluster-robust significance
- **55 BH-FDR survivors @ q=0.10**

Held-out skill edges of the persisted set: **+0.14 to +0.54** (median ≈ +0.12 across
their bets), cluster-bootstrap p 0.0005–0.043, eff_breadth up to 60, distinct
resolution-days up to 45.

**This is the first Project-3 thesis that survives.** It is exactly what §4 stage 1
said a flat gradient could not rule out: the slow-market forecasters exist; they were
absent from the adversely-selected micro-specialist ledger, so the chain had to
**deepen them out of the broad discovery population** — which it did, and they hold up.

## Why this is not the usual artifact

Every prior positive in this repo collapsed under one specific control. This one was
checked against all of them:

1. **Disjoint selection / validation data (§5.3 #1).** Selection ran on shallow data
   (~1 bet/wallet in discovery); validation on the deep histories the screen never
   saw. Selection noise cannot survive into a fresh sample. This is strictly stronger
   than Project 1's within-history half-split.
2. **Cluster-preserving null.** Significance is the market-block bootstrap
   (`validate._cluster_bootstrap_p`), not the bet-level shuffled null — so bets
   sharing a resolution event do not read as skill (`cluster-preserving-null-required`).
3. **Concentration guards.** eff_breadth ≥ 3, entry-days ≥ 3, and a resolution-time
   clustering guard (distinct resolution-days ≥ 3, §3.3) — the §1.5 single-event
   mirage is rejected. The survivors are genuinely broad.
4. **10¢ magnitude floor (scoped).** `scoring.slow.min_skill_edge = 0.10`; the global
   2¢ is untouched and Project 1 is bit-identical.
5. **Not near-resolution scalping (§1.2).** The edge is horizon-robust: mean skill edge
   is +0.121 over all bets, **+0.127 restricted to H ≥ 6h, +0.124 at H ≥ 24h** — it
   does NOT come from entries near resolution. Median entry horizon is **5.5 days**;
   77.9% of the persisted set's bets enter ≥ 24h before resolution. These are forecasts.

## What this does NOT establish (honest limits)

- **Identification, not copyability.** These wallets earn edge; whether it is
  *actionable* (metric B, §7.7) is untested and out of scope here. Slow markets are
  thin-tape — the resolution, if any, is that you receive a signal you have days to
  act on (the 5.5-day median horizon helps), not a fill to copy. **Unmeasured.**
- **Retrospective, not forward.** Per HANDOFF the forward test (§7.8) is the only
  true arbiter. This is a strong retrospective result; the prospective freeze is next.
- **Coarse baseline.** 67% of slow bets classify as `"other"`, so the per-category
  baseline is broad. Some residual could be per-wallet *niche* baseline miscalibration
  rather than pure skill — the forward test is what separates skill from persistent
  miscalibration. A finer baseline (sub-category / category×price) would tighten this.
- **Partial screen↔deep overlap.** The screen's original bets are a (tiny, <5%) subset
  of each wallet's deep data, so the disjointness is strong but not perfect; the
  chronological OOS split within the deep data is the second, independent layer.
- **~50% aggregate FDR context** (like Project 1). The 55 BH-FDR survivors include
  expected false positives; individual small-n names are where they hide. The
  aggregate signal is decisive; treat any single wallet as provisional.
- **86% of the shortlist** deepened (1,295 of 1,518; the missing 14% are lowest-scored).
  Does not affect the 52.

## Next (out of current scope, gated on this result)

The result earns steps 7–8: (a) add the H ≥ 6h gate into `slow_validate` for
completeness (post-hoc it doesn't move the number); (b) metric B / copyability at slow
latencies with the placebo battery (§7.7); (c) the **prospective freeze + forward
test** (§7.8) — freeze these 52 with a timestamp and score only bets resolving after.
That is the experiment that turns "identified" into "real and followable."
