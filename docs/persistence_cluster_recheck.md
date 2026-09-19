# Project 1's persisted set, re-checked against a cluster-aware null

> ⚠️ **SUPERSEDED FOR THE FDR NUMBER (2026-07-26) — see `docs/persistence_fdr_hardened.md`.**
> Everything below is the **pre-hardening** measurement and stands as the historical
> record for the gate that existed on 2026-07-23 (per-bet t-test, no cluster-count floor).
> The three fixes listed in "What to do with this" landed immediately afterwards and took
> the set 118 → 70 → 41, and the null was re-run through that hardened gate on 2026-07-26.
> Under the SAME bracket B the implied FDR of the certified 41 is **80.6% ± 0.7%** (null
> mean 33.06 vs real 41, P(null≥real)=**0.045**), not ~50%; under a stricter null that
> also randomizes market footprint it is **55.0%**. **The "~50%" figure below must not be
> quoted for the current set** — it describes the 118, not the 41.

**Date:** 2026-07-23 · **Script:** `scripts/audit_persistence_cluster.py` (read-only, writes
nothing) · **Data:** `data/interim/bet_ledger.parquet`, 4,070,091 resolved BUY bets, 10,821
wallets with resolved bets.

```
PYTHONPATH=. .venv/bin/python scripts/audit_persistence_cluster.py --shuffles 200 --boot 4000
```

## Verdict up front

**The persistence result survives.** Unlike the black-swan tail finding
(`docs/blackswan_cluster_null.md`), the persisted set is still decisively above a
cluster-preserving null. But it is *smaller and noisier* than the flag count suggests: about
half of it is chance, and only a minority of individual wallets can be certified.

| question | answer |
|---|---|
| Is the persisted **count** above chance under a cluster-preserving null? | **Yes** — real 118 vs null mean 59.0, max 81 over 200 shuffles, P(null≥real)=0.000 |
| What share of the set is manufactured? | **~50%** (null B produces 59 of the 118 by chance) — *pre-hardening; the hardened gate reads 80.6% under this same bracket, see `docs/persistence_fdr_hardened.md`* |
| How many **individual wallets** survive a cluster-robust re-test? | **83 of 118**, but only **42** have enough clusters to certify — and **16** under a strict reading |
| Is `validate.py`'s t-test correctly calibrated? | **No** — median design effect **1.60**, 43% of persisters above 2, max 230 |

## The brackets

| | what it permutes | role |
|---|---|---|
| **A** bet-level | per-bet residual edges within 1¢ price bins | the repo's existing null, reproduced for contrast |
| **B** cluster-preserving | which wallet made each bet, within each market | aggregate/count verdict |
| **C** market-block bootstrap | resamples a wallet's held-out markets with replacement | per-wallet verdict |

Per `docs/blackswan_cluster_null.md`, B can be over-constrained and must not arbitrate
individual wallets; its count-level comparison is a like-for-like contrast of the same
procedure on real vs permuted data and stands. C is the per-wallet instrument.

**The vectorized validator replica used inside the null loops is asserted equal to
`src.validate.compute_oos_validation` on all 10,421 deterministic wallets** (0 mismatches on
`edge_significant`, `edge_magnitude_ok`, `edge_persisted`) before any null is scored. Without
that, the nulls would be arbitrating a lookalike rather than the gate the repo emits.

## Result 1 — the aggregate claim holds, with a ~50% false-discovery rate

Live funnel on today's ledger: **691 candidates → 179 significant → 118 persisted (17.1%)**.

| null | persisted (mean / 95th / max) | rate | P(null ≥ real) |
|---|---|---|---|
| **A** bet-level (repo's existing) | 39.2 / 49 / 57 | 5.4% | **0.000** |
| **B** cluster-preserving | 59.0 / 69 / 81 | 8.1% | **0.000** |

Null A reproduces the historically documented "~5% null" almost exactly. The cluster-preserving
null **raises the chance floor by half again, 5.4% → 8.1%**, which is the correction the
black-swan re-run predicted — but the real 17.1% clears it decisively, and the null's maximum
over 200 shuffles (81) never approaches the real 118.

The honest reading of the gap: null B manufactures **59** persisters from pure noise against the
real **118**, so **roughly half the persisted set is expected to be false**. This is the same
shape as Project 2 §1.5's estimated ~40% gate FDR — the aggregate beats chance while a large
fraction of the individual names do not. The gate is a per-wallet test at α=0.05 over 691
candidates, so ~35 false positives are expected from multiplicity alone before clustering is
even considered.

## Result 2 — the t-test is miscalibrated, and by how much

Design effect D = Var(market-block bootstrap mean) / (s²/n), over the 118 persisted wallets'
held-out halves:

| median | 10th | 90th | share D>1 | share D>2 | max |
|---|---|---|---|---|---|
| **1.60** | 0.75 | 11.47 | 66% | 43% | 230 |

D>1 means `validate.py`'s one-sided t-test understates the variance of the held-out mean, so its
p-values are too small by ~√D — a median of 1.27×, and far more in the tail. **This contradicts
the assumption that these wallets are sparse per market.** Held-out bets per market: median
**2.63**, p90 **9.94**, max **70.4**. The `high_frequency_micro_market` profile means many bets
into the *same* market, all settled by one resolution — which is precisely the clustering the
t-test ignores.

The wallets that fail the cluster-robust test are exactly the high-D ones:

| wallet | held-out bets | markets | bets/market | held-out edge | t-test p | cluster-robust p | D |
|---|---|---|---|---|---|---|---|
| `0xcb016f2b41…` | 3,098 | 44 | 70.4 | +0.060 | 2e-15 | **0.31** | 230 |
| `0x5386c28311…` | 3,474 | 134 | 25.9 | +0.054 | 2e-13 | **0.18** | 64 |
| `0x84cfffc3f1…` | 193 | 10 | 19.3 | +0.160 | 3e-07 | **0.22** | 37 |
| `0x4a19d30608…` | 3,850 | 425 | 9.1 | +0.030 | 7e-06 | **0.11** | 13 |
| `0xf3a6ef82d0…` | 4,242 | 566 | 7.5 | +0.035 | 8e-08 | **0.08** | 15 |

A t-test p-value of 2e-15 becoming 0.31 is the entire finding in one row: 3,098 bets spread over
44 markets is not 3,098 independent observations, it is closer to 44.

## Result 3 — per-wallet: 118 → 83 → 42

| step | count |
|---|---|
| `edge_persisted` per `validate.py` | **118** |
| still significant under the market-block bootstrap (p<0.05) | 83 |
| still clears the 0.02 floor (point estimate) | 118 |
| **survives both** | **83** |
| survives *and* has ≥30 held-out markets | **42** ← the defensible set |
| …and clears the floor at the 5th-percentile lower bound | **16** ← the strict reading |

**48 of the 118 persisted wallets have fewer than 30 held-out markets**, some as few as 1–3.
Cluster-robust inference over 3 blocks certifies nothing, and neither does a t-test over bets
that share those 3 resolutions. Those wallets are not refuted — they are *untestable*, and
counting them as survivors would repeat the mistake this audit exists to catch. Of the 42
defensible wallets, 10 of the 12 checked against the ranked table carry
`pattern_flag=high_frequency_micro_market`; ranks run from 1 to 497.

The magnitude floor is doing almost no work as implemented: all 118 clear 0.02 on the point
estimate, but only 52 clear it at the bootstrap's 5th percentile. A floor applied to a point
estimate is not an economic guarantee.

## Incidental finding — `validate.py`'s split is non-deterministic for 400 wallets

`split_in_sample_out_of_sample` orders with `sort_values("timestamp")`, whose default quicksort
is **unstable**. For **400 wallets** the timestamp at the split boundary is tied with its
neighbour, so which bet lands in which half is decided by sort internals rather than by the
data. One of those wallets flips `edge_persisted` between orderings, and two flip
`edge_magnitude_ok`.

This is not what overturns anything here, but it means the pipeline's headline count is not
reproducible bit-for-bit across pandas versions. **Fix:** pass `kind="mergesort"` (stable) in
`split_in_sample_out_of_sample`. Not applied here — this audit is read-only and the change
belongs in a pipeline commit with its own test.

## Also worth knowing — the persisted set has drifted from what HANDOFF.md documents

HANDOFF.md's current baseline records **37 persisted of 247 candidates (15.0%)** with "0
persisters below 50 held-out bets, min out-of-sample n = 103". Today's ledger gives **118
persisted of 691 candidates (17.1%)**, and **41% of persisters have fewer than 30 held-out
markets**. The cached `data/interim/wallet_validated.parquet` is a day older than the ledger, so
the documented numbers are stale.

The shallow-tail re-accrual that HANDOFF.md anticipated ("re-check that thin wallets stay out of
the persisted set each cycle") **has not held** — thin wallets are now entering the persisted
set. Backlog #4's proposed guards (deepen-on-reached-persisted-set, or an explicit breadth floor
on the flag) are no longer hypothetical; a **cluster-count floor on `edge_persisted`** would be
the most direct version, since held-out *markets*, not bets, are what the evidence actually
rests on.

## What to do with this

1. **Keep the persistence result** — it is real at the aggregate level under the stricter null,
   which is now the only positive claim in this repo that has survived one.
2. **Report the persisted set with its FDR.** "118 persisted" overstates it; "~59 real of 118, of
   which 42 are individually defensible" is the honest form.
3. ✅ **DONE — cluster-count floor added** (≥30 held-out markets, `scoring.min_oos_markets`) to
   `edge_persisted`, plus **stable split** (`kind="mergesort"`). Committed 2026-07-23; on the
   current ledger the persisted set goes **118 → 70** (the 48 with <30 held-out markets are
   removed). See DECISIONS.md "Cluster-count gate".
4. ✅ **DONE — the significance test is now cluster-robust.** `edge_significant` is gated by a
   market-block bootstrap (`validate._cluster_bootstrap_p`, `scoring.oos_bootstrap_resamples`),
   not the t-test, closing the 70→42 gap: on the live ledger the persisted set is now **41**
   (matching this audit's independently-derived defensible set). The t-test p is still reported
   as `out_of_sample_residual_p` for contrast; the gate is `out_of_sample_cluster_p`. Committed
   2026-07-23; see DECISIONS.md "Cluster-robust significance".
5. The forward test remains the only arbiter of copyability; none of this speaks to whether
   these wallets can be followed profitably.
6. ✅ **DONE 2026-07-26 — the null was re-run through the hardened gate.**
   `scripts/audit_persistence_fdr.py`, write-up in `docs/persistence_fdr_hardened.md`. The
   "~50%" in item 2 above does **not** carry over: like-for-like under bracket B the
   certified 41 sit at **80.6% implied FDR** (null 33.06 ± 0.29, P(null≥real)=0.045), and
   under a market-footprint-randomizing null at **55.0%** (null 22.55 ± 0.32, P=0.000).
   The hardening cut true discoveries faster than false ones. The actionable finding is
   that `scoring.oos_significance_alpha=0.05` over 691 candidates is the dominant source:
   at α=0.005 the set is 26 wallets at 20% FDR, at α=0.001 it is 17 at 10%.
