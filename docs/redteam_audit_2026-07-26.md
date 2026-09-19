# RED TEAM AUDIT — polymarket-sharps (2026-07-26)

**Auditor:** independent adversarial red-team session (fresh chat, no prior context, max effort).
**Mandate:** assume the project's conclusions are wrong in both directions — that at least one real
edge was killed by an over-conservative control, and that the surviving positives are overstated.
Verify everything from primary sources; trust no agent summary.
**Method:** read HANDOFF.md, DECISIONS.md, all 17 docs, all 14 memory notes, the full git log
(107 commits), and the actual source of `validate/features/rank/slow_validate/sports_validate/
slow_baseline` plus the `scripts/audit_*.py`. Every load-bearing number was re-derived from primary
data with an independent implementation (ledger 4,695,081 rows; the saved sports/slow validation
tables; the discovery census). All work was read-only; the 390-test suite passes on the audited tree.
Reproduction recipes are in the Appendix.

> **Status note:** this file is the audit deliverable, written by the red-team session. It was left
> uncommitted (the audit's mandate was to touch no git); commit it if it should persist in history.

---

## 1. The single most important finding

**A real, certifiable sports cohort was killed by a mis-scoped pre-registered parameter, and the
entire sports category was then closed based on the one artifact that parameter let through.**

The sports arm's magnitude floor is `scoring.sports.min_skill_edge = 0.10` (10¢). It was copied from
the slow-forecaster arm, whose 10¢ rationale the repo itself states is for a "small-n, large-edge"
universe — explicitly contrasted with "large-n, small-edge" universes like Project 1, which uses 2¢
(DECISIONS.md, "Scoped `scoring.slow.min_skill_edge`"). The deep sports universe is unambiguously
large-n (1,135,069 bets, 40,256 events; median candidate held-out n in the hundreds-to-thousands).
By the repo's own scoping logic, the floor is wrong by 5×. In sports, a 10% per-bet ROI essentially
cannot exist in a contested market; the floor was set where only anomalies can pass.

Re-derived from `data/interim/sports/sports_validated.parquet` (the arm's own saved output):

- **12 wallets clear every other gate** — in-sample candidacy, event-clustered bootstrap
  significance (p ≤ 0.045), ≥5 markets, event-unit concentration floors — **with held-out skill
  edge ≥ 2¢. 11 of the 12 survive BH-FDR at q=0.10** under the repo's own convention (BH over the
  29 gated candidates). At a 5¢ floor (comfortably above any cost estimate in
  `docs/project4_value_gate.md`): 4 clear, 3 survive BH.
- The signal is not marginal at the population level: **37 of 174 candidates have event-clustered
  p < 0.05 (vs ~8.7 expected under a global null), 26 have p < 0.01 (vs ~1.7 expected)**. BH across
  *all 174 candidates* — the severest multiplicity reading — yields **27 survivors at q=0.10**
  (26 at q=0.05). The validation doc's caveat that the survivor "would not survive" FDR across 174
  (`docs/project3_sports_validation.md` §6.2) mis-applies BH: it checks only the rank-1 threshold
  (0.00057), but BH is a step-up procedure and the mass of small p-values carries the set.
  Ironically, under that correct all-174 BH the **certified 10¢ survivor fails (p=0.0205 > cutoff
  0.0120) while 6 of the floor-killed 12 pass**.
- Splitting the 12 by profile (re-derived from the deep tape): ~7 are in-play HFT bots whose whole
  deep history spans days — the 10k `/trades?user=` cap truncates high-frequency wallets, so the
  certified survivor's "13-day held-out window" caveat is a *data artifact shared by this whole
  profile*, not a property of one wallet. **But at least 4 are slow, stable, months-long sports
  bettors:**

  | wallet | held-out skill | in-sample skill | held-out span | profile |
  |---|---:|---:|---:|---|
  | `0x7e3a1f95c558…` | +5.7¢ | +4.8¢ | 107 days (Mar→Jun) | 667 bets / 159 d, 171 events, eff. breadth 83 |
  | `0x97df146fda53…` | +2.8¢ | +1.5¢ | 130 days | 1,893 bets / 576-day record, 460 mkts, top event 7% |
  | `0x90448cec34e3…` | +2.5¢ | +2.1¢ | 186 days | 2,169 bets / 228 d, CL outrights |
  | `0x09fe78c8b9f1…` | +2.4¢ | +0.3¢ | 43 days | 4,592 bets / 50 d |

  These have the stationary in≈out shape the certified survivor conspicuously lacks (its in-sample
  edge is **+0.1¢**, held-out +18.8¢). A 10¢ floor on *held-out* edge is a winner's-curse gate: at
  fixed true skill it selects the most upward-noisy record — which is exactly what it certified.
- One of the killed 12, `0xdbdd45150249e229eb4ca8aa48a30dca21faa5de`, is **independently in
  Project 1's certified 41** (rank 23) — cross-confirmation across two separate pipelines and
  datasets.

The validation doc's own diagnostic printed all of this ("12 wallets clear everything else at a 2¢
floor") and the narrative still concluded "the shortlist was overwhelmingly selection noise." The
next session then closed the category ("sports = uncopyable in-play HFT", memory note
`project3-category-space-exhausted`) based on the *one survivor's* profile, and the named "priority
next step" — metric B / copyability for sports — was never run. The freeze contains no tier for this
cohort (`s_wide` is 10¢-gated; `s_all` dilutes them among 162 noise wallets), so the
currently-running forward test **cannot read them**.

To be precise about what is and is not claimed here: this is *not* a recommendation to retroactively
relax a pre-registered floor to fill a table — that is the failure mode the repo rightly guards
against. The floor was mis-derived at design time from a rationale that does not apply to this
universe; the cost is quantified by the repo's own diagnostic; and the correct remedy is
forward-looking: pre-register the 2¢-significant cohort as its own new tier (its own artifact, its
own freeze timestamp, exactly as the sports freeze was legitimately created beside the forecaster
freeze). Sports resolves in days; the cron scorer is already running and already sees post-freeze
trades. This is the cheapest, fastest test of the most plausible surviving edge in the project.

---

## 2. Per-claim verdicts

### A. The negative verdicts

**A1. The confound chain (forecasters 52→46→30→12; sports →1) — slow chain SOUND; sports chain
WRONG at the magnitude step.**
The slow chain reproduces exactly from the saved variant tables: 52 (category/market) → 46 (niche
baseline, −1.0¢ median shift) → 34 (complex clusters alone) → 30 (both) → 12 (complex-unit gates).
Each step has a verified mechanism: the finer baseline is a real-but-small confound (and the copier
diagnostic against the *unselected* discovery corpus is the right instrument); rolling-deadline
ladders genuinely are one event, so complex clustering is a unit correction, not a screw-tightening.
The 30→12 step is the most conservative — all 18 cuts fail *only* `eff_breadth_complex ≥ 3` (each
checked; none fail significance, which is already complex-clustered) — but it is principled (a
wallet with 1.2 effective complexes has ~one independent piece of evidence), thresholds were
transposed rather than tuned, and crucially **the cut cohorts were retained as pre-registered
forward tiers (t30/t52)**, so nothing was destroyed — it awaits forward data. This is not an
unfalsifiable standard; the forward test is the falsifier and it is running. The sports chain fails
this test: see §1 — the cut cohort was *not* retained as a tier, and the category was closed.

**A2. Event-unit clustering — SOUND, verified in both directions.**
The gate re-run at the corrected unit (`docs/project3_sports_event_unit.md`) *strengthened* the
population claim (dispersion 8.32× vs 7.29×; split-half null re-centred −0.004), and the doc records
that the opposite was expected. The unit deliberately avoids the over-coarse league trap, design
effects were measured (1.80/2.46 — not the 0.17 over-constraint pathology), and the permissive
fallback errs toward more clusters. Re-derivation shows the alternative-unit question was not what
killed the sports survivor set — the magnitude floor was. For forecasters, family-level complexes
are the defensible middle (market-level clustering = 46–52 survivors, preserved in the saved
variants and the t52 tier). No survivor set reappears from re-choosing the unit; it reappears from
fixing the floor.

**A3. Weather NO-GO — QUESTIONABLE, but better-supported than suspected.**
Re-derived from the ledger: the slice numbers check out (128,412 resolved BUY weather bets, raw edge
+0.36¢, 68 findable wallets). The premise "the actual weather specialists were never ingested" is
half-wrong: **25 of the 68 findable wallets are >50% weather-concentrated (p90 share 94%)** —
genuine specialists were in the tested slice, and dispersion still sat at the universal floor with
persistence failing the event null. The copyability half (26.5% follow-on prints, placebos dominate)
is solid for the *copy* thesis. What remains open: (a) the full specialist population is larger than
the 25 the firehose happened to catch — a one-night market-first ingest was declined; (b) the
*value-betting* closure never touched weather at all — **the discovery census contains zero weather
rows** (verified; Gamma enumeration misses recurring series). Premature closure: mild for copying,
real for value-betting. Also a process gap: no committed script/artifact exists for the weather gate.

**A4. Copyability / edge-decay — methodology SOUND for timing; interpretation OVERREACHES on
selection.**
The placebo design (`audit_edge_decay_long.py::cohort_selection_controls`) holds the *bets fixed*
and moves only the entry anchor — so it attributes *timing*, and "the wallet's entry time carries no
exploitable information at ≥30s" is supported. But the headline "following a validated wallet is
measurably WORSE than entering the same market at an arbitrary moment" quietly extends this to
*selection*, which the placebo cannot test: the "arbitrary moment" strategy still requires the
wallet to tell you which market and side. The pre-entry anchor beating the post-entry price is
exactly what informative, price-moving entries look like. The untested strategy is "take the
selection, refuse the chase" (limit orders at pre-entry levels). Two systematic understatements, one
already flagged by the repo: the "tape" is 98% deepened-tracked-wallet prints (not the book), and
prints-are-not-quotes makes the 25% follow-on coverage a *lower bound* on real fill opportunities.
The audit's own bottom line — only the forward test can answer this — is correct and was acted on.

**A5. Value-betting NO-GO — core SOUND (verified exactly); edges OVERSTATED.**
G0 reproduces to four decimals: census aggregate −0.16¢, census 0.6–0.8 band −3.94¢, ledger positive
in all 14 bands (+1.35¢ at 0.6–0.8). The selection diagnosis is right, and the Project-1 baseline
experiment (B6 below) independently corroborates it. The insurance-band kill is methodologically
correct (split-half genuinely cannot test unfired tails; the fair-price null is the right
instrument) and the doc words it honestly ("not statistically distinguishable"). Two overstatements
in the summaries: (i) p=0.33–0.47 on 179 events with 4 tail firings is a *no-power* test — "cannot
distinguish from fair" hardened into "fair-priced / fully closed"; (ii) the census markets have
**median $7.1M volume (min $51k)** — "no structural mispricing" is demonstrated only on the venue's
most liquid core, and hidden series are absent entirely. The arb NO-GO rests on one point-in-time
snapshot of 22 outcome sets — a reasonable prior-confirming spot check, honestly caveated in the
memory note, but "TESTED live = NO-GO, FULLY closed" is stronger than one snapshot supports. No
committed script exists for the arb probe either.

### B. The positive claims

**B6. Skill-edge residualization — SOUND, and conservative rather than generous.**
An independent reimplementation reproduces every certified wallet's held-out residual edge to 4
decimals. The one real issue cuts the *other* way: `fit_price_baseline` is fit on the full ledger,
and **98.1% of resolved BUY bets come from the 721 deepened (sharpness-selected) wallets** — so
E[outcome|price] is elevated above true market calibration. Refit on the un-deepened organic tail
only, the 41's mean residual rises from **+3.19¢ to +4.28¢**. The residualization is not stripping
real edge along with the base rate; it is *understating* certified skill by ~1¢. (The slow/sports
arms use hierarchical leave-one-wallet-out baselines that handle the analogous problem correctly.)

**B7. The 41-wallet certified set — SOUND as stated, with one stale number.**
Funnel verified (23,956 wallets → 691 candidates → 146 significant → 41 persisted). Every wallet's
edge and cluster-p reproduces; **0 of 41 flip significance across 5 alternative bootstrap seeds**;
the set is parameter-stable (markets floor 20/30/50/100 → 43/41/40/36; magnitude 1¢/2¢/3¢/5¢ →
63/41/23/16 — a smooth gradient, no cliff). The insurance caveat does not bite it (held-out entries:
45% mid-book, 7% above 0.98). One correction: the "~50% FDR" attached to it was measured against the
*pre-hardening* gate (118 persisted vs null 59). The hardened gate (cluster bootstrap + floors) has
never had the null re-run through it; ~50% is a stale upper bound and the true FDR of the 41 is
unmeasured (probably better). Honest caveats that stand: 36/41 are micro-crypto HFT (uncopyable per
the repo's own analysis), and the 10k user-cap means their "persistence" horizon is weeks, not
months.

**B8. Insurance-blindness caveat — SOUND and correctly scoped.**
A genuine hole in split-half validation (demonstrated concretely by [0.98,1.00) passing G1 and
failing the fair-price null), worded conditionally, and it does not undermine the 41 (verified) or
the slow 12 (mid/low-price forecast buys). Not an overreach.

### C. Foundations

**C9. Data foundation — the binding constraint on every generalization; mostly acknowledged, but
the bottom line outruns it.**
The engine holds *any* data on 7,142 of the 542,397 wallets seen in just the 573 census markets
(**1.3%**); deep data on ~721 rank-selected wallets plus the arm screens; the census itself is
volume-selected top markets; Gamma enumeration misses ~58% of markets including whole recurring
series. Robust to this: micro-crypto conclusions (the firehose's home turf) and population-level
claims within screened categories. Not robust: "the retrospective search across Polymarket is
COMPLETE", "no un-swept ground". What was exhausted is the category space *as visible through a
volume-selected keyhole*. A representative random-market census was demonstrably feasible (the CLOB
metadata sweep did 39,941 markets in 30 minutes) and never built.

**C10. The nulls — SOUND, with unusual self-correction discipline.**
The bet-level → cluster-preserving → design-effect progression is textbook: the over-constraint
pathology (D=0.17) was *discovered by this project*, documented, and turned into a standing rule.
The market-block bootstrap is correctly implemented (verified line-by-line and reproduced
numerically; deterministic per-wallet seeding; +1 smoothing; NaN-below-2-clusters is conservative).
Two blemishes: the stale FDR figure (B7), and the BH mis-application in the sports doc (§1) — which,
notably, made the evidence look *weaker* than it is.

### D. Workflow & disposition

**D11. Process — strong overall; one systematic failure at the end.**
Atomic writes after the corruption incident, byte-identity verification culture, pre-registration
with refuse-to-amend, honest incident write-ups (the 55%-coverage gradient pilot carries its own
provenance warnings). The failure: **the final three verdicts (weather gate, category sweep, live
arb probe) have no committed scripts or artifacts anywhere** — searched the repo, git history, and
`~/.pmrun`. In a repo whose every other verdict says "reproduce with the scripts named below," the
decisive last-mile closures are unreproducible coordinator-chat one-shots. No earlier conclusion
appears corrupted by process failures.

**D12. Negativity bias — the kills were individually earned; the *closures* show
premature-abandonment disposition in three places.**
Every major negative re-derived checked out mechanically (FLB census exact; black-swan withdrawal
well-reasoned; §1.5 artifacts concrete). The real pattern: (i) sports — "metric B is the priority
next step, answerable in days" (freeze doc) → next session opened Project 4 instead, and the
category was closed citing the survivor's uncopyable profile, with the 12-at-2¢ diagnostic sitting
unacted-on in the arm's own output; (ii) "not distinguishable from fair" → "fully closed";
(iii) one arb snapshot → "arb dead." A structural amplifier: each NO-GO memory note instructs future
sessions "do NOT re-attempt" — prudent against thrashing, but it fossilizes any error in the verdict
it protects (the sports note fossilizes the floor error). Counter-evidence also on record: the team
withdrew its *own* positive (black-swan) under a stricter null, elevated confounds against its *own*
favorite result (slow 5c), and pre-registered forward tests instead of quietly dropping threads.
Not a rationalization machine — a conservative one with a fast trigger finger at the very end.

### E. What was never tried / prematurely closed — ranked by expected value

| # | Item | Cost | EV assessment |
|---|---|---|---|
| 1 | **Pre-register a sports `s_sig2c` tier** (the 12; separately track the 4 slow-profile names) as its own frozen artifact; the cron scorer already runs | ~an hour | **High.** Reads in days; tests the most plausible surviving edge |
| 2 | **Metric B / copyability for the sports 2¢ cohort** (the named-then-dropped priority) — copy-window + follow-on liquidity on the 4 slow-profile wallets' markets | ~a day | **High.** Decides whether #1's signal is actionable |
| 3 | **Re-run the cluster null through the hardened P1 gate** → true FDR of the 41 | ~a day | Medium-high; sharpens the flagship deliverable, probably favorably |
| 4 | **Representative (random condition-id) census ingest** → venue-wide FLB/value read + discovery beyond the volume keyhole | 1–2 nights | Medium; converts "closed on top-volume markets" into "closed" or finds the tail |
| 5 | News-lag with a free external clock (GDELT/RSS) — "blocked" was a choice, not a hard block | days | Unknown; genuinely unexplored |
| 6 | Weather specialist market-first ingest | a night | Low for copying (liquidity wall stands); mild for value |
| 7 | SELL-side / exit-copying analysis (SELLs ingested but never scored — acknowledged scope cut) | days | Low-medium |
| 8 | Standing arb monitor for transients; the 2 tail-lens wallets (ranks 497/320); common-price-band H-axis check | small | Low each; listed for completeness |

---

## 3. What is genuinely sound and well-done

The negative results are only as good as the machinery, so credit where due: the leakage discipline
(forward-price guard, no-resolved-value fallback, chronological splits, stable sort); the
selection-vs-validation firewalls (disjoint screen/deep data — stronger than half-splits); the
cluster-inference progression ending in measured design effects; the pre-registration culture
(freeze manifests carrying the fitted baseline so forward numbers cannot drift; refuse-to-amend with
the widening honestly logged at 0 observations); the no-drop invariant actually holding in code
(verified in `rank.py` — flags never touch the score); byte-identity verification of every
optimization; and the willingness to withdraw its own positives. The identification engine's core
result — 41 seed- and parameter-stable wallets whose residual edge is if anything *understated* —
is real. The FLB selection diagnosis (Project 4 G0) is exactly right and is the single best analysis
in the repo.

## 4. Bottom-line verdict

**"There is no actionable edge in this data" is about 80% justified and 20% premature — and the
premature 20% is specific, cheap to test, and currently testable.** The original thesis (copy trades
at execution latency) is dead on strong evidence: micro-crypto is uncopyable, retrospective
copy-windows fail placebo controls, and the forecaster universe is structurally narrow. The
methodology did not evolve into a confound-manufacturing machine — under adversarial re-derivation
the controls were correct, and twice the pipeline was found to be *under*-claiming (the conservative
baseline; the BH mis-statement against its own sports evidence). But the final week's
generalization — *no* edge, search *complete*, sports closed as "uncopyable in-play HFT" — is
contradicted in one concrete place by the project's own saved output: **a 12-wallet sports cohort
(11 surviving FDR, four with months-long stable records in slow-resolving sports markets, one
independently certified by Project 1) was excluded by a magnitude floor mis-transposed from a
different statistical universe, has no forward tier, and was never tested for copyability.** That is
the wrongly-killed live thread this audit was asked to find. Re-open it now, while the sports
forward window is fresh — the fix is a new pre-registered tier plus the metric-B run the freeze doc
already named as the priority, and the venue will grade it in days.

---

## Appendix — reproduction recipes (all read-only)

The audit's scratch scripts lived in a session scratchpad; everything below reconstructs the same
numbers from durable repo data.

### A. Sports funnel + counterfactual floors (finding §1)

```python
import pandas as pd, numpy as np
s = pd.read_parquet("data/interim/sports/sports_validated.parquet")
# published funnel: 289 rows, 174 candidates, 42 significant, 1 persisted
base = s[s.candidate & s.significant & s.markets_ok
         & s.concentration_ok & s.complex_concentration_ok]     # 19 wallets
for f in (0.02, 0.05, 0.10, 0.15):
    print(f, (base.out_sample_skill >= f).sum())                # 12 / 4 / 1 / 1
# BH-FDR (copy bh_reject from src/slow_validate.py):
cand = s[s.candidate]
gated = cand[(cand.out_sample_skill >= 0.02) & cand.markets_ok
             & cand.concentration_ok & cand.complex_concentration_ok]  # 29
# bh_reject(gated.cluster_p, 0.10).sum() -> 11
# bh_reject(cand.cluster_p, 0.10).sum() -> 27 ; at q=0.05 -> 26
# cluster_p among 174 candidates: <0.01: 26, <0.05: 37, <0.10: 47
```

Held-out spans/profiles of the 12: filter `data/interim/sports/deep_trades.parquet` to those
wallets (`side==BUY & resolved`), sort by timestamp per wallet (mergesort), split 50/50, measure the
out-half span. Key rows: `0x8ade1d…` out-span 11.8 d @ 337 bets/day; `0xdbdd45…` 4.1 d @ 1,185/day
(in-play HFT); vs `0x7e3a1f95…` 106.6 d @ 3.1/day, `0x97df146f…` 130.4 d @ 7.3/day (slow bettors).

### B. Slow chain 52→46→34→30→12 (verdict A1)

All saved variants under `data/interim/slow_deepening/`:
`slow_validated_category_market_id_nocxgates.parquet` (52 persisted / 55 FDR),
`_niche_form_market_id` (46), `_category_niche_l1` (34), `_niche_form_niche_l1` (30),
`slow_validated.parquet` (STANDARD, 12). Diffing the 30-set against the 12-set: all 18 cuts fail
`eff_breadth_complex < 3.0` (values 1.12–2.93); none fail complex-clustered significance.

### C. Project 1 verification + baseline-selection experiment (verdicts B6/B7)

```python
# funnel: data/interim/wallet_validated.parquet -> 691 candidates, 146 significant, 41 persisted
# ledger: data/interim/bet_ledger.parquet, 4,695,081 rows, 4,070,091 resolved BUY bets
# 1) refit the 20-quantile-bin baseline exactly as features.fit_price_baseline;
#    per-wallet mergesort split; recompute held-out residual means
#    -> matches every stored out_of_sample_residual_edge to 4 decimals.
# 2) cluster bootstrap (validate._cluster_bootstrap_p semantics) re-run with 5
#    alternative seeds per wallet -> 0 of 41 flip significance at 0.05.
# 3) deepened set = keys of data/interim/backfill_cursors.json (721 wallets)
#    -> 98.1% of resolved BUY bets. Refit bin means EXCLUDING them:
#    certified-41 mean residual +0.0319 (production) -> +0.0428 (un-deepened baseline).
# 4) certified-41 held-out entry prices: 44.7% in (0.3,0.7], 7.0% in (0.98,1.0].
# 5) gate sensitivity: min_oos_markets 20/30/40/50/100 -> 43/41/41/40/36;
#    magnitude 0.01/0.02/0.03/0.05 -> 63/41/23/16.
```

### D. FLB census verification (verdict A5)

`data/interim/discovery/discovery_trades.parquet`: 1,685,995 resolved BUY bets, 573 markets,
463,561 wallets; mean raw edge −0.0016; 0.6–0.8 band −0.0394. Ledger: +0.0087 aggregate, positive
in all 14 bands, 0.6–0.8 = +0.0135. Census market selection:
`data/interim/discovery/market_universe.parquet` — volume min $50.8k, median $7.1M, max $1.53B.
Weather coverage: 0 census rows match slug `temperature`.

### E. Weather slice (verdict A3)

Ledger rows with slug containing `temperature`: 128,412 resolved BUY bets, 24,498 markets,
324 wallets; raw edge +0.0036; 68 wallets ≥20 bets; **25 of 68 have >50% of their total resolved
BUYs in weather** (p90 share 94%). No committed script exists for the weather gate itself.

### F. Identification reach (verdict C9)

Census wallets 542,397 ∩ ledger wallets 23,956 = 7,142 (1.3%).

### G. Integrity

`python -m pytest tests/ -q` → 390 passed on the audited tree. `data/cron.log` (2026-07-26 01:27)
shows the sports forward scorer already fetching post-freeze trades — the forward window is live.
