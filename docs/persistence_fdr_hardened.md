# The true FDR of the certified 41, under the HARDENED gate

**Date:** 2026-07-26 · **Script:** `scripts/audit_persistence_fdr.py` (read-only, writes
nothing) · **Data:** `data/interim/bet_ledger.parquet` (2026-07-24 snapshot), 4,070,091
resolved BUY bets, 23,956 wallets, 348,656 distinct markets.

```
# count-level nulls (~45 min, both brackets)
flock data/interim/.analysis.lock .venv/bin/python scripts/audit_persistence_fdr.py \
    --shuffles 200 --boot 2000
# per-wallet multiplicity control (~5 min)
flock data/interim/.analysis.lock .venv/bin/python scripts/audit_persistence_fdr.py \
    --shuffles 0 --bh-boot 50000
```

> ✅ **ITS RECOMMENDATION WAS IMPLEMENTED (2026-07-26, same day).** Recommendation 3 below
> — tighten the significance threshold — shipped as a **scoped** key,
> `scoring.project1.oos_significance_alpha = 0.005` (`src/validate.py`,
> `validate.certification_alpha`), with the global `scoring.oos_significance_alpha` left at
> 0.05 so the slow/sports arms' frozen forward tests are untouched. The pipeline was re-run:
> **691 candidates → 98 cluster-significant → 26 persisted** (24 copyable), and this script
> was re-run against the new artifact — the bit-identity check reproduces the shipped 26
> exactly. Everything below describes the measurement, which was made at the *old* α=0.05
> and is unchanged as a record. See DECISIONS.md "Tightened significance threshold" and the
> certified-set section of HANDOFF.md.

This supersedes the FDR number in `docs/persistence_cluster_recheck.md` (2026-07-23).
That doc's measurements are **not withdrawn** — they were correct for the gate that
existed on 2026-07-23. They just no longer describe the gate the repo ships.

## Why this was re-run

`docs/persistence_cluster_recheck.md` measured **real 118 persisted / 691 candidates vs a
cluster-preserving null mean of 59.0 → implied ~50% FDR**. Immediately afterwards three
fixes landed in `src/validate.py` — a stable split, a **≥30 held-out markets** floor
(118→70), and **cluster-robust significance** via the market-block bootstrap (70→41). The
null was never re-run through the hardened gate, so "~50%" was an upper bound inherited
from a weaker test, flagged as the one stale number in `docs/redteam_audit_2026-07-26.md`
(verdict B7). The expectation on record was that the hardened gate would look *better*.

**It does not.** Measured against the same bracket, the FDR of the certified set is
**80.6%**, not 50%. Against a stricter, better-specified null it is **55%**. The
consolation is that the audit also finds the fix: the false-discovery share is dominated
by the gate's own α=0.05 over a 691-wide search, and tightening it collapses the FDR to
10–20% for a set of 17–26 wallets.

## Verdict up front

| question | answer |
|---|---|
| Implied FDR of the 41, like-for-like with the 2026-07-23 bracket (null B) | **80.6% ± 0.7%** (null mean 33.06 ± 0.29 vs real 41) — **worse than the stale 50%** |
| …under a null that also randomizes market footprint (null B2) | **55.0% ± 0.8%** (null mean 22.55 ± 0.32) |
| Is the count still above chance? | **Under B2 yes, decisively** (P(null≥real)=0.000). **Under B only marginally: P=0.045**, and the null's max over 200 shuffles is **48 > 41** |
| Can either null arbitrate individual wallets? | **B: no** (D=0.89 on the 41). **B2: yes** (D=1.93, 88% of the 41 above 1) |
| Per-wallet, BH over the cluster-robust p | names **36 of 41** at q=0.10, **28** at q=0.05 — but the permutation says BH's nominal q **understates** the true rate ~3× |
| Operating point that actually delivers ~10% measured FDR | **α=0.001 → 17 wallets** (null 1.77) |
| Is the shipped 2000-resample p stable? | **Yes** — at 50,000 resamples 41/41 still clear α, median p 0.0025→0.0022 |

The two answers are not in conflict, and the FDR *curve* below is the operational
takeaway: **α=0.05 over 691 candidates is simply too loose, and tightening it is free.**

## The harness reproduces the shipped 41 exactly

Before any null is scored, the vectorized replica of the hardened gate is asserted equal
to `data/interim/wallet_validated.parquet` — the artifact `python -m src.validate` wrote
from this ledger — across all 23,956 wallets:

```
candidates 691 → bootstrapped 380 → cluster-significant 146 → magnitude_ok 365
→ markets_ok 737 → edge_persisted 41   (5.9% of candidates)
edge_significant: 0 mismatches   edge_magnitude_ok: 0   edge_markets_ok: 0   edge_persisted: 0
```

The replica calls `src.validate._cluster_bootstrap_p` itself with `src.validate`'s own
per-wallet seeding, and renumbers market ids to lexicographic rank so `np.unique` inside
the bootstrap orders the per-market blocks identically. Its p-values are therefore
bit-identical to the pipeline's, not merely distributionally equal. The run **aborts** if
this check fails.

## The two nulls

| | what it permutes | what it holds fixed | what it can see |
|---|---|---|---|
| **B** within-market | which wallet made each bet, inside a market | every market's bets/prices/outcomes/timestamps; **each wallet's exact market footprint** | only within-market side/timing edge |
| **B2** cell | which wallet owns each (wallet, market) **cell**, within cell-size strata | every cell's bets/prices/outcomes intact; each wallet's bet count and cell-size profile | nothing — the wallet↔outcome link is destroyed outright |

B is the bracket the 2026-07-23 number was measured with, so it is the like-for-like
comparison. It is a **conditional** null: a wallet keeps the exact set of markets it
traded and how many bets it put in each, so any edge that comes from *being exposed to
those markets* survives into the null. B2 removes that conditioning — it moves whole
cells between wallets, so a wallet keeps its shape but gets other people's markets.

B2 is new here. It exists because B's conditioning is not neutral for this set: 36 of the
41 are `high_frequency_micro_market` specialists, and holding their footprint fixed hands
the null a large structural component of their edge.

### Both nulls were checked for the failure mode that killed the black-swan finding

`docs/blackswan_cluster_null.md` withdrew a result after finding `permute_wallets_within_market`
had design effect **0.17** in a sparse stratum — six times *less* variance than
independence, so it certified anything. That doc's rule is: measure D before trusting a
null. Measured here over the 1,468 wallets with a testable held-out half:

| bracket | median D | 10th | 90th | share D>1 | median D on the 41 | share D>1 on the 41 |
|---|---|---|---|---|---|---|
| **B** | **0.933** | 0.751 | 1.090 | 28.3% | **0.891** | 19.5% |
| **B2** | **1.508** | 0.961 | 5.281 | 85.8% | **1.931** | 87.8% |

**B is not over-constrained on this stratum** (0.93, essentially at independence) — the
black-swan pathology was specific to the sparse tail, and it does not recur here. But it
is still *below* 1, so **B licenses its count and not per-wallet inference**. B2 is above
1 as a clustering-aware null should be, and on the certified set it is ~1.9× wider than
the per-bet t-test — close to the 1.60 the 2026-07-23 audit measured for the market-block
bootstrap, which is the reassuring cross-check.

A second check, because a permutation that hands a wallet its own record back cannot
manufacture a false discovery: **label retention**. Under B, 19.5% of all bets keep their
own wallet, and the certified 41 keep a *median of 8.6%* of their record (max 86%, none
above 90%). B genuinely scrambles these wallets — the 80.6% is not a freeze artifact.
Under B2 retention is 1.8%.

And the null's persisters are **not** the real ones: under B a replicate's ~33 persisters
include on average **3.62** members of the certified 41 (so ~29 are wallets the real gate
rejects); under B2, **1.12**. Both nulls manufacture new names rather than re-certifying
the set.

## Result 1 — the count-level FDR got worse, not better

200 shuffles per bracket, `--boot 2000` (the shipped `scoring.oos_bootstrap_resamples`) in
**both** arms and in the real arm.

| null | persisted mean ± MC s.e. | sd | median | 95th | max | candidates | P(null ≥ 41) | implied FDR |
|---|---|---|---|---|---|---|---|---|
| **B** within-market | **33.06 ± 0.29** | 4.14 | 33 | 40 | **48** | 728.2 | **0.045** | **80.6%** |
| **B2** cell | **22.55 ± 0.32** | 4.53 | 22 | 30 | 38 | 754.4 | 0.000 | **55.0%** |

Both nulls generate a slightly *larger* candidate pool than the real data (728 / 754 vs
691), so `null_mean / real_count` is mildly conservative. Rate-adjusted — apply the null's
persistence *rate* to the real 691 candidates — B gives 31.4 expected false discoveries
(FDR 76.5%) and B2 gives 20.7 (FDR 50.4%). The headline numbers are the unadjusted ones.

Comparing like with like against 2026-07-23 (both bracket B):

| | candidates | real persisted | null B mean | null B max | P(null≥real) | implied FDR |
|---|---|---|---|---|---|---|
| pre-hardening (2026-07-23) | 691 | 118 | 59.0 | 81 | 0.000 | ~50% |
| **hardened (today)** | 691 | **41** | **33.06** | **48** | **0.045** | **80.6%** |

**The hardening removed true discoveries faster than false ones.** It cut the real count
by 65% (118→41) and the bracket-B null count by only 44% (59.0→33.1). The margin over
chance went from "the null never came within 37 of the real count" to "the null exceeded
it in 4.5% of shuffles". The cluster-count floor and the cluster-robust bootstrap removed
a genuinely untestable tier — 48 of the old 118 had fewer than 30 held-out markets and
could not be certified either way — but they did **not** make the aggregate claim
stronger, and the "~50%" that was being carried as an upper bound was in fact an
under-statement of the bracket-B figure.

Under B2 the aggregate claim is comfortable: 41 vs 22.55 ± 4.53, never reached in 200
shuffles (max 38), P=0.000.

## Result 2 — the FDR is a knob, and the gate is set on the wrong notch

The same gate at tighter significance thresholds, real and null measured identically at
2,000 resamples (this costs nothing extra — tightening α can only *remove* wallets, so no
additional bootstrap is needed):

| α | real | null B mean | FDR (B) | null B2 mean | **FDR (B2)** |
|---|---|---|---|---|---|
| **0.05** (shipped) | **41** | 33.06 | 81% | 22.55 | **55%** |
| 0.0171 (BH q=0.10) | 36 | 24.25 | 67% | 11.51 | **32%** |
| 0.0073 (BH q=0.05) | 28 | 19.54 | 70% | 6.47 | **23%** |
| 0.005 | 26 | 18.27 | 70% | 5.21 | **20%** |
| 0.001 | 17 | 14.17 | 83% | 1.77 | **10%** |

Under B2 the FDR falls monotonically from 55% to 10% as α tightens — the gate is throwing
away far more noise than signal at every step, which is what a real effect under a loose
threshold looks like. **Under B it does not fall at all** (81% → 67% → 70% → 83%):
tightening removes real and null discoveries at the same rate, the signature of a null
that still contains a real component. That component is the market footprint B holds
fixed, and it is the cleanest evidence in this run that a large part of the certified
set's edge is *which markets it was in* rather than *how it traded inside them*.

## Result 3 — per wallet: BH names the set, and says the shipped p is anti-conservative

Neither count-level null names wallets (B's D<1 forbids it; B2's count is an aggregate).
The instrument that does is the per-wallet cluster-robust bootstrap p — which is already
the shipped gate's own statistic — put through a Benjamini–Hochberg step-up over the
**691-candidate family**, the family the gate actually searched.

Run at **50,000** resamples, because the shipped 2,000 pins the strongest wallets to its
1/2001 p-floor as exact ties and BH cannot order ties:

| q | BH rejections (of 691) | …also clearing the magnitude + market floors | of which certified |
|---|---|---|---|
| 0.05 | 101 | **28** | 28 of 41 |
| 0.10 | 118 | **36** | 36 of 41 |
| 0.20 | 142 | **41** | 41 of 41 |

Every wallet that clears BH *and* the economic floors is already in the certified 41 — BH
adds no names, it only removes them. **BH names 36 of the 41 at q=0.10 and 28 at q=0.05.**

**But do not read q as the achieved FDR.** BH's guarantee assumes the p-values are
calibrated, and the permutation null says these are not: at BH's q=0.10 threshold
(α=0.0171) the B2 null still manufactures 11.51 of the 36, a measured FDR of **32%**, and
at q=0.05 it manufactures 6.47 of 28, **23%**. The market-block bootstrap p is
anti-conservative by roughly 3× on this set — consistent with its design effect (B2 is
~1.9× wider at the wallet level than the per-bet variance the bootstrap is compared
against). The right way to use BH here is as the *naming* instrument, with the permutation
curve as its calibration.

The higher resolution also serves as a stability check on the shipped setting: at 50,000
resamples **41 of 41 still clear α=0.05**, with median cluster p moving 0.00250 → 0.00220
and max 0.03698 → 0.03222. The shipped 2,000 is not flattering anyone.

Where the certified set sits on other tightenings (shipped p, 2,000 resamples):

| tightening | count |
|---|---|
| `edge_persisted` (the certified set) | **41** |
| cluster p < 0.01 | 31 |
| cluster p < 0.005 | 26 |
| ≥100 held-out markets | 36 |
| held-out skill edge ≥ 0.05 | 16 |
| all three (p<0.01, ≥100 markets, ≥0.05 edge) | **8** |

Held-out markets across the 41 run min 44 / median 806 / max 4,750, and cluster p max
0.037 — the 2026-07-23 funnel (118 → 83 → 42 → 16) has been absorbed into the gate, which
is why there is no longer a large "survives the bootstrap but has too few clusters" tier.

## Reconciling the two answers

Count-level: ~55% of certifications at α=0.05 are what chance manufactures (B2).
Per-wallet: BH names 36 of 41 at q=0.10. Both are true, and Result 2 is the bridge.

The 55% is the false-discovery share **at the gate's operating point** — α=0.05 per wallet
over 691 candidates, a deliberately permissive threshold under which ~35 false positives
are expected from multiplicity alone before clustering is even considered. That is
essentially the 22.55 B2 measures once the magnitude and market floors have thinned it.
BH does not dispute it, it corrects it, by tightening the threshold; and the permutation
curve then says how much correction actually landed (32% at q=0.10 rather than the nominal
10%).

**Practical reading: `scoring.oos_significance_alpha = 0.05` is the single largest source
of false discoveries in this pipeline, and tightening it costs nothing but names.** The
measured trade, in B2 terms: 41 wallets at 55% FDR (today), 26 at 20%, or 17 at 10%.

## Re-measured at the shipped α = 0.005 (2026-07-26, after the change landed)

The same script, same ledger, same 200 shuffles per bracket, re-run against the **new**
artifact (`edge_persisted` = 26). The harness's bit-identity check passed first —
`candidates 691 → cluster-significant 98 → magnitude_ok 365 → markets_ok 737 →
edge_persisted 26`, 0 mismatches on all four flags — so the scoped α propagates through
the replica exactly as it does through `src/validate.py`.

| bracket | null persisted mean ± MC s.e. | sd | 95th | max | P(null ≥ 26) | implied FDR |
|---|---|---|---|---|---|---|
| **B** within-market | 18.27 ± 0.19 | 2.74 | 23 | **25** | **0.000** | **70.3% ± 0.7%** |
| **B2** cell | **5.21 ± 0.16** | 2.27 | 9 | 12 | **0.000** | **20.1% ± 0.6%** |

Both numbers reproduce the Result-2 curve to the decimal (B predicted 18.27, B2 predicted
5.21), which is expected — the nulls are seeded and the ledger did not move — so this is a
confirmation of the operating point, not a new measurement.

Three things worth reading off it:

1. **The strict-null FDR is 20.1%**, down from 55.0%. That is the number to quote for the
   certified 26: *~5 of the 26 are what chance manufactures.*
2. **The aggregate claim got stronger under the bracket that was marginal.** At α=0.05,
   `P(null_B ≥ real)` was **0.045** with a null max of 48 > 41. At α=0.005 the B null never
   reaches the real count in 200 shuffles (max 25 < 26), so **P = 0.000 under both
   brackets**. The tightening did not improve B's FDR *ratio* (70.3%, as predicted), but it
   moved the count from "the null exceeded it 4.5% of the time" to "the null never reached
   it". Those are different claims and both matter.
3. **The null's persisters are still not the real ones.** A B replicate's ~18 persisters
   contain on average **2.22** of the certified 26 (was 3.62 of 41); under B2, **0.20**.
4. Retention and design effects are unchanged in character: B retention 0.195 with median
   D **0.860** on the certified 26 (still < 1 → count-only, no per-wallet verdicts); B2
   retention 0.018 with median D **1.776** (≥ 1 → per-wallet capable).

The gap between 70.3% (B) and 20.1% (B2) is the same gap Result 2 identified and it has not
closed: B holds each wallet's market footprint fixed, so the share of this set's edge that
comes from *which markets it was in* survives into that null. Tightening α cannot touch
that, and no threshold will — only a test that can credit or debit market selection would.

## Coverage and settings — everything that bounds this result

- **Identical in both arms:** `oos_split=0.5`, `min_bets_per_half=10`,
  `oos_significance_alpha=0.05`, `min_skill_edge=0.02`, `min_oos_markets=30`,
  `oos_bootstrap_resamples=2000`. No parameter was cheapened for the null.
- **200 shuffles per bracket.** MC s.e. on the null mean is ±0.29 (B) and ±0.32 (B2), i.e.
  ±0.7 and ±0.8 percentage points of FDR. The uncertainty in these numbers is not the
  shuffle count.
- **No candidate-pool restriction, no truncation.** Every wallet in the ledger is scored in
  every replicate.
- **One exact speed optimization:** `edge_persisted` is a conjunction, so in the null arm
  the bootstrap is evaluated last, only for wallets that already clear the candidate,
  magnitude and market gates. The persisted count is unaffected; the null's
  `edge_significant` count is simply not produced.
- **The 50,000-resample BH pass is the real arm only.** Re-running 200 shuffles at that
  resolution is ~25× the cost and was not attempted; the like-for-like null comparison at
  BH's thresholds is the FDR curve in Result 2, computed at 2,000 in both arms. The two
  BH threshold values (0.0171, 0.0073) are the largest p BH rejects at q=0.10/0.05 in the
  50,000-resample run and are used as fixed constants in the curve.
- **What both nulls destroy:** market-*selection* skill in B2's case (it hands wallets other
  people's markets), and within-market side/timing skill in both. Neither null can credit a
  wallet for choosing which markets to enter, so if that is a real and copyable skill,
  both FDRs are over-estimates. This is the same limitation `permute_wallets_within_market`
  has always carried, stated in `scripts/audit_speed_gradient.py`.
- **`null_mean / real_count` assumes π₀ = 1** (no wallet has skill), which makes every FDR
  here an **upper bound**. Same convention as 2026-07-23.
- This says nothing about copyability. The forward test remains the only arbiter of
  whether any of these wallets can be followed profitably.

## What to do with this

1. **Replace "~50% FDR" with the measured numbers.** The honest form of the headline is
   "41 certified, of which ~23 are what chance manufactures at this α (55%, B2); 36 survive
   BH at q=0.10." Not "41 persisted", and not "~50%".
2. **The aggregate claim is weaker than advertised under bracket B** — P(null≥real)=0.045
   with a null max of 48. It is *not* weak under B2 (P=0.000). Quote both; the set is
   above chance, but "decisively" is only true of the unconditional null.
3. **Tighten `scoring.oos_significance_alpha`.** This is the concrete, cheap action the
   run recommends: α=0.001 gives **17 wallets at a measured 10% FDR**, α=0.005 gives
   **26 at 20%**, against today's 41 at 55%. Not changed in this commit — this audit is
   read-only and the change belongs in a pipeline commit with its own test and a re-rank.
   ✅ **DONE later the same day at α=0.005** (scoped to Project 1; see the note at the top).
   0.001 was rejected on resolution grounds: at `oos_bootstrap_resamples=2000` the p-floor
   is 1/2001, so below α=0.001 only two attainable p-values exist and the gate would be
   quantized by the resample count rather than the evidence.
   Note that a BH step (rather than a fixed α) would adapt as the ledger grows, but its
   nominal q must be read as ~3× optimistic on this data until the bootstrap's calibration
   is fixed.
4. **Do not read per-wallet verdicts off bracket B** (D=0.89). B2 (D=1.93) and the BH pass
   are the per-wallet instruments.
5. The 2026-07-23 doc's numbers stay on the record as the pre-hardening measurement.
