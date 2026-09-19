# RED TEAM AUDIT — the real-world copy arm (2026-07-29)

**Auditor:** independent adversarial session, max effort, mandated by
`docs/REDTEAM_BRIEF_2026-07-29.md` to assume every conclusion is wrong **in both
directions** — that the copy edge is fake, *and* that a real edge was destroyed by
an over-correction.

**Method.** Read `CLAUDE.md`, `HANDOFF.md`, `docs/redteam_audit_2026-07-26.md`
(the standard), `docs/copy_verdict.md`, `docs/copy_lag_simulation.md`,
`src/copy_sim.py`, `src/realworld_validate.py`, `scripts/audit_persistence_fdr.py`,
and the frozen cohort definitions. Every
load-bearing number was re-derived from primary data with an independent
implementation: the 3,385,240-bet real-world price baseline rebuilt from the 88
deep-tape shards; the 133,093-bet per-bet frame re-scored; the 38.1M-point CLOB
price store re-indexed. All work read-only.
`pytest tests/ -q` → **688 passed** on the audited tree.

**New code committed with this report:** `scripts/audit_copy_null.py` — the null the
copy analysis never had, plus the spread sweep, the held-out split and the baseline
refit. Reproduction recipes in the Appendix.

---

## 0. The published rule, stated — because nothing in the repo states it

The headline "11 of 37" has **no committed script**. It was produced by an ad-hoc
heredoc whose only surviving trace is `~/.pmrun/corrected46.log` and two
unversioned parquets. `src/copy_sim.py`, the committed code, implements a
*different* estimator (shuffled-outcome null, ROI units, 1¢ tick) whose verdict the
docs themselves say is overturned — a reader who runs `python -m src.copy_sim score`
today gets the **withdrawn** analysis.

I reverse-engineered the rule from the artifacts and confirmed it by reproducing
`own_c` to **4 decimal places on all 37 wallets** and the copyable set exactly:

```
fill    = price(entry + 120 s)  +  max(entry_price − mid_at_entry, tick_size)
own_c   = mean( resolved_value − E[outcome | entry_price] )
net_c   = mean( resolved_value − E[outcome | fill] − k · fill · (1 − fill) )
baseline E[·]  = 20 quantile bins fitted on 3,385,240 deepened real-world bets
cohort  = bets with a finite guard-on price at Δ = 2 m AND a finite mid_at_entry
floor   = 150 bets;  CI = 2,000-resample market-cluster bootstrap
```

Two things follow that the docs do not say. First, **`net_c` is a residual (skill)
edge, not money** — it is measured against the baseline evaluated at the
*follower's own fill price*. (This turns out not to matter: pooled money edge is
**+2.29¢** against pooled residual **+1.98¢**, correlation across wallets 0.998,
mean absolute difference 0.33¢. The unit choice is immaterial and I report both.)
Second, **`copy_verdict.md`'s two tables use two different estimators.** The
decomposition table ("gross +4.54 → drift −1.90 → fee −0.50 → spread −0.10 → net
+2.02") is `own − drift − costs`; the per-wallet table is the residual-at-fill rule
above. On this cohort the first is ~0.66¢/share *below* the second. The document
presents them as one analysis.

**The mechanism behind that 0.66¢ gap, found while writing the estimator into
`src/copy_sim.py`:** `E[outcome | price]` is a *step function* over 20 quantile
bins, so a price move that stays **inside a bin** changes the follower's cost but
not the baseline it is scored against — and the residual does not see it. Near 0.6
the production bin is ~8¢ wide, so a typical 2¢ drift is often invisible to
`net_c` while being fully visible in the money column. This is why both are now
reported side by side, and it is pinned by
`tests/test_copy_sim.py::test_residual_metric_is_blind_to_drift_inside_one_baseline_bin`
so it cannot regress silently. It does not change any headline (money and residual
agree to 0.33¢ mean absolute difference across the 37) but it is the reason to
prefer the money column when the two disagree.

> **A live demonstration of §1's instability, from the committed code.** Running
> `python -m src.copy_sim certified` reproduces `own_c` to 4 d.p. and again returns
> **11 of 37 copyable** — but a *different* 11. `0x5ac8f582c98b`, the published
> table's top entry, comes back at CI [−0.30, +29.72] and drops out, while
> `0xb6cb0e2c37c6` enters at [+0.0099, +9.25]. Same rule, same data, different
> bootstrap stream. Three runs in this audit produced three different memberships at
> a stable count.

---

## 1. The single most important finding

**The pooled copy edge survives every control I could build. The per-wallet
ranking does not — and the live forward arm's headline stratum is a per-wallet
ranking.**

The decisive test is the one the repo already knows to run and did not:
a **cell permutation** that holds market clustering fixed (each market keeps its
bets, prices, outcomes and shared resolution) and randomizes only *which wallet
held which market* — the same bracket as null B2 in `scripts/audit_persistence_fdr.py`.

| dispersion of per-wallet follower edge | real | null mean | P(null ≥ real) |
|---|---:|---:|---:|
| bet-level permutation (within price bins) | 3.65¢ | 1.28¢ | **0.000** |
| **cell permutation (market clustering preserved)** | **3.65¢** | **3.22¢ ± 0.61** | **0.230** |

Under the naive bracket the wallets look strongly differentiated. Under the
cluster-preserving bracket, **the observed spread between "copyable" and
"uncopyable" wallets is what you get by dealing each wallet a random hand of
markets.** This is the same reversal that withdrew the black-swan finding on
2026-07-23, and it converges with null B in the FDR run (which, holding each
wallet's footprint fixed, still certifies 40 of 58): the per-wallet differences
*are* footprint differences.

Three independent stability checks agree:

* **Split-half.** Reconstructing each wallet's certification split exactly (verified
  against `validate`'s stored `in_sample_n + out_of_sample_n` for **46 of 46**
  wallets): **5 of 33** scored wallets are copyable on the in-sample half, **10 of 32**
  on the held-out half, and only **3** on both.
* **Seed.** Changing nothing but the bootstrap RNG state moves the count 11 ↔ 12
  (`0xb6cb0e2c37c6`, CI lower bound −0.07 vs +0.24). Dropping from 2,000 to 1,000
  resamples moves it 11 → 10.
* **Multiplicity.** BH-FDR over the 37 boot-p values: **12** survive at q=0.10, **10**
  at q=0.05. This is the one soft spot the analyst feared that turns out to be *mild*.

The consequence is concrete. The pre-registration makes `rw5_copyable_v2` — five
individually-named wallets — the **headline** stratum, and that is precisely the
object the cluster-preserving null cannot distinguish from chance. The *secondary*
`rw5_pooled` stratum, and the pre-registered reading order ("per-wallet rows
last"), are what carry the readable hypothesis. No amendment is needed and none
should be made; this is a reading correction, not a rule change.

---

## 2. Per-claim verdicts

### 2.1 "11 of 37 certified wallets are copyable at a 2-minute lag" — **CONFIRMED as arithmetic, NOT ESTABLISHED as a per-wallet fact**

Re-derived independently: **11 of 37**, identical membership. But see §1 — the
identity of the 11 does not survive a cluster-preserving null, a split-half test, or
a change of RNG seed. Report it as "the pooled cohort of 37 shows a positive
follower edge"; do not report it as "these eleven wallets are copyable."

### 2.2 "Median 78% of the wallet's own edge retained" — **OVERSTATED, exactly as suspected**

77.6% is the median **over the survivors**. The unconditional figures:

| | edge retained |
|---|---:|
| median over the 11 copyable (published) | **77.6%** |
| median over all 37 scored | **49.1%** |
| bet-weighted pooled (Σ net / Σ own) | **39.1%** |

The honest headline is that a 2-minute follower keeps **roughly 40–50%** of these
wallets' edge, not 78%.

### 2.3 "Best ~+5–6¢/share" — **CONFIRMED for the live five; the table tops out higher**

The five live headline wallets re-derive at +6.56 / +5.61 / +5.22 / +3.51 / +3.35¢,
matching the manifest to 0.01¢. The full table's top entries are +16.4¢ and +10.6¢,
both on wide CIs.

### 2.4 "4 of 72 high-volume *uncertified* wallets are copyable — certification does real work" — **WRONG**

Three independent defects:

1. **The pool is not uncertified.** **9 of the 72** are `edge_persisted` wallets
   (`0x1ee9a5fc09`, `0xe542afd388`, `0x83255595ba`, `0x5415c298dde8`,
   `0x2bcd792138a4`, `0x92d0cb81e6c8`, `0x433723566347`, `0x44c1dfe43260`,
   `0x09bed19766`).
2. **Two of the four "copyable" ones are certified wallets** (`0x5415c298dde8`,
   `0x2bcd792138a4`). The uncertified hit rate is 2 of 63, not 4 of 72.
3. **Like-for-like, the contrast reverses.** Inside the *same* high-volume run — same
   recency window, same scoring, same power — mean follower edge is **+1.35¢ for the
   certified 9** and **+1.47¢ for the uncertified 63**. Fisher exact on the
   copyable counts (2/9 vs 2/63) gives p = 0.074, and the difference is driven by CI
   width (median 6.73¢ vs 8.52¢), not by edge.

The published "30% vs 5.6%" compares two runs on different bet windows with
different statistical power. It is not evidence that certification does work for
copyability.

### 2.5 The tradeable-size claim — **out of scope, and the probe behind it was mis-sampled**

The original claim was about tradeable size, which this repo does not measure and
which is not reproduced here. One methodological finding from re-deriving it is
worth keeping, because it recurs:

* **The probe sampled live books only.** For the `others` group it found books for
  **88 of 1,912** tokens traded in the last 45 days — **4.6%** — which reads as
  severe survivorship toward long-lived markets. Re-measured properly the figure is
  **104 of 400 (26.0%)**: the 4.6% was itself an artifact of exhausting one group's
  whole 45-day history rather than sampling it. So the first correction was as wrong
  as the thing it corrected.
* **The wrong-central-statistic trap, twice.** I criticised the original for reading
  only the **median** and then read only the **mean**, which on a distribution whose
  mean is ~20× its median is the same error with the opposite sign. Neither statistic
  settles anything; the distribution is the answer. This is the general lesson and it
  applies well beyond the probe.
* **Stale comparator.** The constant it compared against was the *superseded*
  pre-Correction-2 pooled figure, applied to a group defined by the *superseded*
  verdict. It happened to land within 0.04¢ of the re-derived pooled +1.98¢, which
  is luck, not correctness.

### 2.6 "High-volume wallets are structurally uncopyable because they trade liquid fast markets" — **NOT ESTABLISHED**

| test | ρ | p |
|---|---:|---:|
| certified 37: bets/day vs follower edge | −0.157 | 0.35 |
| high-volume 72: bets in last 30 d vs follower edge | −0.146 | 0.22 |
| high-volume 72: total bets vs follower edge | −0.184 | 0.12 |

Consistent sign in three tests, significant in none. A tendency, not a structure.

### 2.7 "FDR 17.3% → ~48 of the 58 are real" — **the number is right, the reporting is selective**

* **The real arm's self-check is genuinely excellent** and I confirm it: the
  vectorised replica reproduces the shipped `validated.parquet` with **0 mismatches
  on all four gates** (427 / 434 / 2026 / 58) *before* any null is scored.
* **The D ≥ 1 criterion is mis-scoped.** Its source, `docs/blackswan_cluster_null.md`,
  scopes the design effect to *per-wallet arbitration*. Here it is used to pick which
  *count-level* FDR to headline. That is not what the criterion licenses.
* **B2 is nevertheless the better null for this claim — on a criterion the docs
  compute but under-use.** Label retention: null B leaves **25.1%** of bets on their
  own wallet and re-certifies **9.7 of the real 58 per replicate**; B2 leaves 4.1% and
  re-certifies 0.27. A null that hands a wallet a quarter of its own record back
  cannot cleanly count manufactured discoveries. Net of the self-recertifications
  B's implied rate is ~52%, not 69.2%.
* **So the brief's own suspicion is correct in substance.** The honest statement is
  *"unconditional FDR 17.3%; conditional on the wallet's market footprint, 40 of 58
  certifications survive"* — and **the second half was never carried into the copy
  conclusion**, even though it is the same "the edge is the footprint" result three
  earlier runs converged on. Quoting 17.3% alone is selective.

### 2.8 "The population baseline may be circular — the single biggest hole" — **WRONG, and conservative in the other direction**

The 46 simulated wallets are **3.87%** of the 3,385,240-bet baseline tape. Refitting
the baseline with all of them removed:

| baseline | pooled follower edge | copyable |
|---|---:|---:|
| as published (includes them) | +1.98¢ [+1.21, +2.81] | 11 |
| **refit excluding all 46** | **+2.25¢ [+1.47, +3.09]** | 11 |

Removing the contamination **raises** the measured edge by 0.27¢. Same direction as
the 2026-07-26 audit's finding B6 on Project 1: the baseline is *understating*
these wallets, not manufacturing them. The listed "biggest hole" is not a hole.

*(One real discrepancy found en route: the copy analysis's baseline is fitted on the
3.39M deepened-only tape, while the certification gate's baseline is fitted on the
4.88M-bet deep + prior-arm population. `copy_verdict.md` says it is "the same
baseline the certification gate uses". It is not. Numerically it does not matter —
`own_c` reproduces to 4 d.p. either way — but the sentence is false.)*

### 2.9 "There is currently NO null in the copy analysis at all" — **CONFIRMED; now supplied, and the result is the most favourable thing in this audit**

Three wallet-agnostic anchors on exactly the same bets, same costs, paired
(n = 67,988):

| anchor | follower residual edge | 95% CI |
|---|---:|---|
| **real (+2 m — copying)** | **+1.98¢** | [+1.16, +2.82] |
| pre-entry (−2 m — cannot be copying) | +4.27¢ | [+3.49, +5.10] |
| random moment in the token's whole pre-guard life | +1.40¢ | [+0.66, +2.18] |
| **random moment AFTER the signal (the executable one)** | **+0.31¢** | **[−0.44, +1.16]** |

| paired difference | value | 95% CI |
|---|---:|---|
| real − pre-entry | −2.28¢ | [−2.39, −2.18] |
| real − random (whole life) | **+0.58¢** | [+0.40, +0.76] |
| **real − random_POST (executable)** | **+1.67¢** | **[+1.45, +1.91]** |

**This is the first copy thesis in this repo to survive a wallet-agnostic placebo.**
Every earlier one — `audit_edge_decay_long.py` 2026-07-22, sports metric B
2026-07-26 — died because `real − random` came out negative. Here it is positive
under both random anchors, and the executable one has the larger margin.

It also **kills the repo's standing "selection, not timing" hypothesis for this
cohort**: taking the wallet's market and side at an arbitrary later moment earns
**+0.31¢, CI spanning zero**. The tradeable object is the market/side *at the moment
they take it* — not the footprint, and not the timing alone.

⚠️ **The honest caveat, which the analyst should hold against this result as hard as
against the negatives.** All three anchors sit on a monotone "earlier is better"
price gradient (that is exactly what the front-loaded drift *is*). `random` mostly
samples times *before* the wallet entered, so it is biased **up**; `random_POST` is
always *later*, so it is biased **down**. The wallet's specific instant cannot be
cleanly separated from where on the convergence curve you happen to land. The
comparison brackets the answer; it does not isolate it.

Per wallet: `real − random_POST` has a CI above zero for **24 of 37** and below zero
for **0**. On the whole-life anchor it is above zero for 11 and **below zero for 6** —
including two of the published copyable eleven (`0x9fc0432877` at −2.42¢
[−3.98, −0.82] and `0x69ea0d77ef` at −1.04¢ [−1.86, −0.28]), both of which are in the
**live forward arm's headline stratum**.

> ### ✅ AMENDMENT B — 2026-07-29, same session: the clean experiment was run
>
> `scripts/audit_copy_timing.py` (committed) removes the gradient confound by
> **matching on position**. For bet *i*, the control enters *i*'s own token at the
> relative position in that market's usable life taken from a **different bet by the
> same wallet** — same market, same side, same wallet, same typical earliness, but a
> moment chosen for a different market. Five donors per bet, averaged; identical
> costs on both legs; paired, market-clustered CIs.
>
> | | narrow window (120 s) | wide window (900 s) |
> |---|---:|---:|
> | paired bets | 28,226 (41.2%) | 28,748 (42.0%) |
> | mean relative position, real vs donor | 0.542 / 0.544 | 0.544 / 0.544 |
> | real | +3.27¢ [+1.72, +5.09] | +3.24¢ [+1.78, +4.99] |
> | donor | +2.44¢ [+1.03, +3.99] | +2.51¢ [+1.14, +4.02] |
> | **REAL − DONOR** | **+0.83¢ [+0.47, +1.17]**, p<0.0001 | **+0.72¢ [+0.37, +1.08]**, p<0.0001 |
>
> **VERDICT: the wallet's specific moment carries genuine market-specific
> information — but it is worth ~0.7–0.8¢, not the +1.67¢ the confounded anchor
> implied.** Roughly half of that +1.67¢ was position on the curve; the other half is
> real. The effect survives both window choices and its CI excludes zero decisively.
>
> **The decomposition this finally licenses**, on the paired subset (real +3.27¢):
> * **~0.8¢ is moment-specific** — it requires acting near their instant.
> * **~2.4¢ is market/side + typical stage of the market's life** — available at a
>   statistically similar moment, not at their exact one.
>
> That second component is *not* the "footprint alone" the earlier random anchor
> tested and found worthless (+0.31¢). Uniformly-later entry is much worse than
> entry at these wallets' *typical stage*, and since you cannot travel backwards to
> reach that stage, **acting promptly is how you capture it in practice** even though
> only ~a quarter of it is strictly moment-specific.
>
> **A caveat of mine to withdraw.** §2.9 warned that all the anchors sit on a
> monotone "earlier is better" gradient. In *residual* units they do not: follower
> edge by decile of relative position runs 2.33, 1.77, 1.81, 1.24, 2.34, 2.30, 3.82,
> 2.42, 0.23, 1.72 — noisy and non-monotone, because the baseline already absorbs
> the price level. The pre-entry and post-entry differences are **local** drift
> effects around the wallet's trade, not a lifecycle gradient. The caveat was
> directionally right about confounding (matching on position does cut +1.67¢ to
> +0.83¢) but wrong about the mechanism.
>
> **Limits.** Coverage is 41%: a donor price must exist within the window, which
> selects markets with dense price series. The paired subset's own edge is +3.27¢
> against the full cohort's +1.98¢, so it is a more liquid, higher-edge slice — read
> +0.83¢ as *within that slice*, not as a cohort-wide constant. Per wallet the result
> is noise again: CI above zero for **8 of 31**, below for 2. The donor anchor is a
> **scientific control, not a strategy** — about half its draws land before the
> wallet traded and are not executable.

### 2.10 In-sample contamination — **NEW FINDING, and it goes the analyst's way**

Nobody flagged this: `copy_sim.load_bets` scores **every** resolved BUY bet,
including the chronological first half on which `validate.py` *selected* these
wallets. I reconstructed each wallet's split exactly (46 of 46 verified).

| half | pooled follower edge | copyable |
|---|---:|---:|
| in-sample (the selection half) | +1.60¢ [+0.28, +3.15] | 5 of 33 |
| **held-out (never used to select)** | **+2.42¢ [+1.53, +3.35]** | 10 of 32 |

The held-out half is **better**. The pooled copy edge is not a selection artifact.
(The per-wallet churn between halves — only 3 wallets copyable in both — is the
finding in §1.)

### 2.11 "`0xa8638d8d7a00` is micro_crypto, so the real-world filter leaks" — **WRONG; the label leaks, the data does not**

`0xa8638d8d7a00` is 48.6% micro-crypto across its whole tape, but **0 of the 624
bets it was scored on** are micro-crypto. The published `cat` column is the wallet's
*modal category over all its bets*, not over the scored ones. Across the whole
scored cohort micro-crypto is **153 of 68,491 bets (0.22%)**; in the copyable set,
**10 of 15,857 (0.06%)**.

The underlying observation is still correct and worth recording: `copy_sim` genuinely
does **not** apply `discover.is_real_world` (its frame is 133,093 bets against the
certification's 131,096 real-world ones — the difference is exactly the 1,996 micro
bets). The filter works **by accident**: a 5-minute market almost never has a usable
price two minutes later inside the resolution guard. Incidental, not designed, and
worth making explicit before someone changes Δ.

### 2.12 The spread proxy — **swept; the pooled result is robust, the count is not**

| spread charged | mean charge | pooled edge | 95% CI | copyable |
|---|---:|---:|---|---:|
| tick only | 0.10¢ | +3.97¢ | [+3.13, +4.78] | 21 |
| half the observed gap | 1.25¢ | +3.03¢ | [+2.28, +3.90] | 20 |
| **published (full gap)** | **2.48¢** | **+1.98¢** | [+1.21, +2.81] | 11 |
| 1.5× the observed gap | 3.71¢ | +1.13¢ | [+0.30, +2.01] | 8 |

The pooled edge stays positive with a CI above zero even at 1.5× the proxy. The
per-wallet count halves. **The published choice is the conservative middle of the
range, not a cherry-pick** — a point in the analyst's favour that the doc does not
make.

### 2.13 API truncation — **CONFIRMED and quantified**

**344 of 2,117** deepened wallets (16.2%) sit exactly at the 10,000-row `/trades?user=`
cap. Of the 46 simulated certified wallets, **7 (15.2%)**; of the published copyable
eleven, **1** (`0x433723566347`). Recency bias touches about one in seven of the
scored set.

### 2.14 Fees applied retroactively — **CORRECT for the decision, and immaterial to it**

Polymarket was genuinely zero-fee before 2026-01-05 and most of this tape predates
that, so these wallets never optimised against fees. The doc already says this. What
it does not say is the magnitude: at the cohort's mean fill the fee is ~0.5¢/share
against a spread charge of 2.48¢ and a drift loss of ~2.0¢. **Fees are the smallest
of the three costs and cannot decide anything here.**

### 2.15 AMENDMENT 1 — **legitimate in substance, defective in form, and one risk went undisclosed**

* **The strongest version of the analyst's defence is true, and stronger than argued.**
  All **16** forward observations at the re-freeze came from wallets the amendment
  **demoted** — 11 from `0x1ee9a5fc09`, 4 from `0x09bed19766`, 1 from `0xe542afd388`.
  **Zero** came from the five promoted wallets, so their membership could not have
  been conditioned on any forward record, resolved or not.
* **But that is also the configuration where "unresolved ≠ uninformative" bites.** The
  demoted wallet was the *only* one with a record, and its 11 open positions had live
  prices visible at the moment of demotion. The manifest records the count (16) but
  not the composition, and not whether those marks were inspected.
* **The reason for the demotion is independently verifiable and purely retrospective**:
  `0x1ee9a5fc09`'s copy edge falls to +1.61¢ [−1.05, +4.22] under the midpoint fix;
  I re-derive **+1.58¢ [−1.12, +4.24]**. Nothing forward is needed to explain it.
* **Form defects in a pre-registration document:** `amendments: []` is empty although
  an amendment occurred (it survives only as a `_superseded` string).
  `hypothesis.H4` still names `rw2_copyable = {0x1ee9a5fc09, 0x69ea0d77ef}` with the
  superseded "+31.9% / +13.8%" as THE HEADLINE, while `strata.headline` is
  `rw5_copyable_v2` with five different wallets. `strata.note` still says "the two
  wallets". `scripts/crontab.example` still says "Headline rw2_copyable". **The
  pre-registration contradicts itself about what is being tested.** Fix the strings;
  do not touch the membership or the freeze timestamp.
* **`evidence_for_these_five.set_level_measured_fdr: 0.173` and
  `expected_genuine_of_five: 4.13` are a misuse.** A set-level FDR does not transfer
  to a subset selected on a second criterion. `known_limits` #6 repeats it.
* **Undisclosed risk:** on the whole-life placebo, two of the five live headline
  wallets (`0x9fc0432877`, `0x69ea0d77ef`) enter *significantly worse* than a random
  moment in the market they chose; on the executable placebo `0x69ea0d77ef` has no
  measurable timing edge (+0.40¢ [−0.44, +1.21]). Four of the five clear the
  executable placebo; one does not.

  | live headline wallet | net_c | real − random_POST | in-half | held-out half |
  |---|---:|---|---|---|
  | `0x253da81575` | +6.56 [+3.04, +10.33] | +2.51 [+1.15, +3.84] | copyable | copyable |
  | `0x9fc0432877` | +5.61 [+2.89, +7.62] | +1.43 [+0.21, +2.43] | — | copyable |
  | `0x69ea0d77ef` | +5.22 [+3.06, +7.41] | **+0.40 [−0.44, +1.21]** | copyable | copyable |
  | `0xd06f0f7719` | +3.51 [+1.47, +5.58] | +2.00 [+1.49, +2.45] | copyable | — |
  | `0x05c5aab002` | +3.35 [+2.02, +4.76] | +0.61 [+0.01, +1.21] | copyable | copyable |

  Notably these five are *more* stable than the copyable set at large (3 of 5 pass in
  both halves, against 3 of ~11 overall).

### 2.16 "Drift is front-loaded" — **CONFIRMED**

From `data/interim/copysim/drift_curve.parquet`: pooled drift **1.80¢ at 1 m**,
1.99¢ at 2 m, 2.16¢ at 5 m, **2.47¢ at 15 m** → 73% of the 15-minute total lands
inside the first minute; halving the lag from 2 m to 1 m buys back 0.19¢. It is a
price measurement, independent of every cost assumption, and the "faster bot"
closure stands. If anything it is understated: the anchor is the wallet's
post-slippage VWAP, so a midpoint-to-midpoint measure would be *more* front-loaded.

### 2.17 The unasserted impossibilities — one found, quantified, and it is conservative

The brief asked for the other impossibilities nobody asserted. The one I found:
**the spread proxy produces fill prices the market cannot produce.**
`price_2m + max(entry_price − mid, tick)` lands at or above **$1.00 on 1,671 of
68,491 scored bets (2.44%)**, exceeding it by a mean of **13.7¢**, and the code
silently clips to `1 − 1e-9`. For `0x9b979a065641` **39.3%** of modelled fills are
clipped; `0xa486f1e84d08` 18.1%; `0x5415c298dde8` 12.7%.

Direction, measured rather than assumed: dropping those bets as unfillable moves the
pooled edge **+1.98¢ → +2.10¢** and the count 10 → 11. **The truncation is working
*against* the follower.** Reported because it is a real modelling defect and because
the audit's job is to report the check, not only the checks that bite.

Two more checks that came back clean and are worth recording as such:

* **The guard-on cohort is not a flattering selection.** At 41% coverage one would
  expect the surviving bets to be the wallet's best. They are not: median cohort
  `own_c` **5.07¢** against median certified held-out edge **5.11¢**, ratio 0.94×,
  Spearman(cohort own edge, certified held-out edge) = **+0.744, p = 0.000**.
* **`mid_at_entry` cannot be a post-impact price.** On a 60-second grid the first
  point in `[entry − 60, entry + 60]` is always at or before `entry`, so the spread
  proxy is never measured after the wallet's own fill moved the book.

---

## 3. Process

The 2026-07-26 audit's §D11 found that "the decisive last-mile closures are
unreproducible coordinator-chat one-shots, in a repo whose every other verdict says
*reproduce with the scripts named below*." **That finding has repeated, one week
later, on the most consequential number in the project.** The copyable set, the
high-volume comparison and the corrected per-wallet table all live in
`~/.pmrun/*.log` and unversioned parquets. `data/interim/copysim/` is gitignored.
Nothing in `src/` or `scripts/` computes them.

`docs/copy_lag_simulation.md` also still carries the full superseded tables, and
`src/copy_sim.py`'s own docstring and report generator still describe the broken
shuffled null as the method.

Against that: the **honesty discipline is exemplary and rare.** Five errors
self-reported in the brief before I looked; an impossibility assertion added after
error 4; the observation that three of four corrections moved toward the preferred
conclusion, recorded by the person who made them; a re-specified null flagged as
post-hoc in the document that uses it; `forward_observations_at_freeze` recorded
honestly; the mark-to-market shortcut rejected before it ran. Nothing I re-derived
was misreported in a way that survived the analyst's own disclosure.

---

## 4. Bottom-line verdict

### Is there a copyable edge here?

**Yes, probably, at the cohort level — about +2¢/share, or 40–50% of these wallets'
own edge, at a 2-minute lag with real fees and a demonstrated spread. It is the
first copy result in this repo to survive a wallet-agnostic placebo, an
out-of-sample split, and a de-contaminated baseline. It is not a per-wallet
result.**

What survives, with the number I re-derived:

| | |
|---|---:|
| pooled follower edge, published rule | **+1.98¢ [+1.16, +2.82]** |
| … as raw money rather than residual | +2.29¢ |
| … on held-out bets only | **+2.42¢ [+1.53, +3.35]** |
| … with the baseline refit to exclude these wallets | **+2.25¢ [+1.47, +3.09]** |
| … at 1.5× the spread proxy | +1.13¢ [+0.30, +2.01] |
| … minus an executable post-signal placebo | **+1.67¢ [+1.45, +1.91]** |

Five independent robustness axes, all positive, all with CIs excluding zero. That is
a materially stronger result than the document claims for itself.

What does not survive: **which wallets**. Cluster-preserving dispersion p = 0.23;
3 of ~11 stable across the certification split; 11 ↔ 12 on RNG state alone.

### Would I act on it?

**Not on the shape currently pre-registered.** Two reasons, in order of size
*(reason 2 was revised by AMENDMENT B, which resolved it)*:

1. **The per-wallet selection is noise.** Anything that concentrates on named
   wallets rests on the one part of this analysis that failed its null. The cohort
   is the only unit the evidence supports.
2. ~~**The placebo is bracketed, not clean.**~~ **RESOLVED by AMENDMENT B, and it
   holds.** With position matched, the wallet's moment is worth **+0.83¢
   [+0.47, +1.17], p<0.0001** — half the +1.67¢ the confounded anchor implied, but
   decisively non-zero. This is no longer a reason to withhold.

**A forward reading remains the right arbiter** — with the reading correction in
§1: read the pooled stratum first, and treat `rw5_copyable_v2` as a per-wallet
curiosity rather than the headline.

### The three things I would do next, in order

1. **Commit the scoring rule.** Turn the reconstruction in `scripts/audit_copy_null.py`
   §1 into `src/copy_sim.py`'s actual estimator, delete or clearly quarantine the
   superseded shuffled-null path, and regenerate `docs/copy_lag_simulation.md`. The
   repo's central number should not be reproducible only from a log file in `~/.pmrun`.
2. ~~**Re-derive §2.5's probe properly.**~~ ✅ **DONE — AMENDMENT A**. It found my
   own §2.5 half wrong, for the same reason §2.5 found the original wrong.
3. ~~**Run the one clean timing experiment.**~~ ✅ **DONE — AMENDMENT B**,
   `scripts/audit_copy_timing.py`. It survives at +0.83¢ [+0.47, +1.17].

**What is left after A and B.** The timing edge is clean and the pooled effect is
real. What stands against it is unchanged: **the per-wallet selection is noise**.
The open analytical question is no longer whether the effect exists but whether it
survives forward on data the selection never saw.

**A note on the mandate.** I was asked to assume the conclusion was wrong in both
directions. It is wrong in both: the per-wallet claims are overstated to the point
of not being supported, and the pooled claim is *understated* — the analyst's own
"discount accordingly, treat everything as a hypothesis" framing is more pessimistic
than four independent robustness checks warrant. The 2026-07-26 audit found this
repo twice *under*-claiming against its own evidence. That pattern has repeated.

---

## Appendix — reproduction (all read-only)

```bash
# 0. the price baseline (20 quantile bins, 3,385,240 deepened real-world bets)
#    stream the 88 shards, dedup on (tx_hash, wallet, token_id, side), drop
#    market_ids whose category in market_category.parquet is micro_crypto, then
#    features.fit_price_baseline semantics. Reproduces own_c to 4 d.p.

# 1-7. the nulls, the spread sweep, the split-half and the baseline refit
PYTHONPATH=. .venv/bin/python scripts/audit_copy_null.py \
    --store <cache>.npz --baseline <baseline>.npz --perms 200
#   artifacts -> data/interim/copysim/redteam/
#     per_wallet.parquet          §2.1  (11 of 37; BH q=0.10 -> 12, q=0.05 -> 10)
#     placebo_pooled.parquet      §2.9  (four anchors, paired n=67,988)
#     placebo_per_wallet.parquet  §2.9  (24/37 clear the executable placebo)
#     permutation_null.json       §1    (bet-level dispersion, p=0.000)
#     cell_permutation.json       §1    (cell-level dispersion, p=0.230)  <- the finding
#     spread_sweep.parquet        §2.12
#     halves.parquet              §2.10 (in 5/33, out 10/32, both 3)
#     wallet_splits.parquet       §2.10 (46/46 reconstructed against validate)

# 2.4 the high-volume comparison
python -c "
import pandas as pd
hv=pd.read_parquet('data/interim/copysim/hv_scored.parquet')
v=pd.read_parquet('data/interim/realworld/validated.parquet')
c=set(v[v.edge_persisted.fillna(False)].wallet.str[:14]); hv['cert']=hv.wallet.isin(c)
print(hv.groupby('cert')[['net_c','copyable']].agg(['mean','size']))"
#   -> certified 9: mean +1.35c, 2 copyable;  uncertified 63: mean +1.47c, 2 copyable

# 2.7 the FDR nulls, retention and design effects
#   ~/.pmrun/fdr_rw_full.log  (real arm: 0 mismatches on all four gates)
#   B  retention 0.251, persisted 40.12, of which 9.73 are the real certified 58
#   B2 retention 0.041, persisted 10.03, of which 0.27 are the real certified 58

# 2.13 truncation
#   344 of 2,117 deepened wallets at exactly 10,000 rows; 7 of the 46 simulated

```

`pytest tests/ -q` → 688 passed```

`pytest tests/ -q` → 688 passed on the audited tree.
