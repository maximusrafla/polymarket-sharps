# Black-swan / tail-edge, re-arbitrated against a cluster-preserving null

**Date:** 2026-07-23 · **Script:** `scripts/audit_blackswan_cluster.py` (read-only, writes
nothing) · **Data:** `data/interim/bet_ledger.parquet`, 4,067,653 resolved BUY bets with clean
0/1 outcomes, 10,746 wallets.

```
PYTHONPATH=. .venv/bin/python scripts/audit_blackswan_cluster.py \
    --count-shuffles 300 --wallet-shuffles 2000 --persist-shuffles 200
```

## Why this re-run exists

`scripts/audit_blackswan.py` (2026-07-19) reported the repo's only live positive claim: at entry
price ≤ 0.15, tail-edge wallets clear a hit-rate test decisively above a shuffled-outcome null (19
clear, 11 survive BH-FDR, null mean 3.1), with split-half persistence +0.218 — better than the
mean-edge pipeline's +0.105.

It arbitrated that with a **bet-level** shuffled-outcome null. Project 3 stage 1
(`docs/project3_slow_markets.md` §10.1) then measured that this null is too permissive for
wallet-level statistics: it destroys market clustering, so bets sharing one resolution event read
as skill. On the speed-gradient cells it cut significance from 8/12 to 3/12. That put the
black-swan finding on notice, and this is the re-check.

Today's ledger is larger than the 2026-07-19 run (498 tested tail wallets at thr=0.15 vs 85 then),
so **both** nulls are re-measured here. The historic "19 clear / 11 BH / null 3.1" is not directly
comparable to the real counts below; the comparison that matters is real-vs-null within this run.

## The three brackets

| | what it permutes | what it preserves | what it destroys |
|---|---|---|---|
| **A** bet-level (the audit's) | resolved outcomes within 1¢ price bins | market-wide base rate, each wallet's price mix | wallet↔outcome link **and** market clustering |
| **B** cluster-preserving | which wallet made each tail bet, within each market | every market's exact bets, prices, outcomes; each wallet's per-market bet count | wallet↔bet attribution within a market |
| **C** market-block bootstrap | resamples the wallet's own markets with replacement | the wallet's own bets, prices and market choices | nothing — it re-weights whole market blocks |

B is `permute_wallets_within_market` from `audit_speed_gradient.py`. C was added here because B
turned out to be the wrong instrument for this statistic (see "The methodological finding").

## Result 1 — the winner-count claim fails

"Winners" = wallets with `resid_vs_base > 0` **and** exact Poisson-binomial p < 0.05, exactly the
audit's gate. 300 shuffles per null.

| thr | tested | real winners | null A mean / p95 | P(A ≥ real) | null B mean / p95 | P(B ≥ real) |
|---|---|---|---|---|---|---|
| 0.10 | 445 | 58 | 15.2 / 21 | **0.000** | 63.5 / 68 | **0.983** |
| 0.15 | 498 | 78 | 18.4 / 25 | **0.000** | 86.6 / 92 | **0.993** |
| 0.20 | 536 | 82 | 20.5 / 27 | **0.000** | 108.8 / 114 | **1.000** |

Against the bet-level null the result looks as decisive as the audit reported. Against the
cluster-preserving null **the procedure invents more winners on permuted data than the real data
contains, at every threshold.** The aggregate "tail-edge wallets exist here" claim does not
survive.

## Result 2 — the split-half persistence claim fails

| | value |
|---|---|
| qualifying wallets (≥10 tail bets per half) | 498 |
| **observed** Spearman(first-half resid, second-half resid) | **+0.214** (reproduces the historic +0.218) |
| null A (bet-level) mean / 95th | −0.004 / +0.083 → p = **0.005** |
| null B (cluster-preserving) mean / 95th | **+0.244** / +0.280 → p = **0.915** |

The cluster-preserving null **centres above the observed value**. Permuting wallet labels within
markets leaves each null "wallet" holding the same portfolio of markets in both halves, and that
alone reproduces +0.24 of split-half correlation. So the persistence is a property of *which
markets a wallet is in*, persisting across both halves — not evidence of tail hit-rate skill.
(Note what this does not settle: persistent market *choice* could itself be skill; this null
cannot separate that from persistent market fixed effects. What dies is the specific claim that
+0.218 evidences tail calibration skill.)

## Result 3 — what survives per wallet: 46 → 8

At thr=0.15, BH-FDR q=0.05 across 498 tested wallets:

| bracket | clears p<0.05 | BH-FDR survivors |
|---|---|---|
| A bet-level Poisson-binomial | 78 | **46** |
| B cluster-preserving permutation | 55 | 35 — **not trustworthy, see below** |
| C market-block bootstrap | 39 | **8** |

The 8 null-C survivors, all of which also pass the Project 2 §1.5 concentration guard
(eff_breadth ≥ 3 and decision_days ≥ 3):

| wallet | n_tail | hits | hit rate | base | resid | ROI | markets | eff_breadth | days | rank | already `edge_persisted` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0xec47cb4e0a…` | 466 | 59 | 0.127 | 0.057 | +0.070 | 0.27 | 466 | 466.0 | 72 | 1 | yes |
| `0x1b2e5156f2…` | 1750 | 291 | 0.166 | 0.095 | +0.071 | 0.99 | 1204 | 690.7 | 109 | 2 | yes |
| `0x0e7bcc59ec…` | 2265 | 243 | 0.107 | 0.074 | +0.033 | 0.39 | 1599 | 1180.2 | 83 | 12 | yes |
| `0x361528e242…` | 2149 | 222 | 0.103 | 0.067 | +0.037 | 0.96 | 1901 | 1691.0 | 33 | 13 | yes |
| `0xd189664c53…` | 1340 | 114 | 0.085 | 0.050 | +0.035 | 0.73 | 1206 | 1085.6 | 125 | 17 | yes |
| `0xba559219ef…` | 837 | 119 | 0.142 | 0.102 | +0.040 | 0.45 | 774 | 717.1 | 72 | 20 | yes |
| `0x16c91ea1c8…` | 1114 | 265 | 0.238 | 0.097 | +0.141 | 1.87 | 931 | 778.5 | 70 | 497 | **no** |
| `0xee65685de4…` | 1495 | 45 | 0.030 | 0.014 | +0.016 | 0.64 | 970 | 619.6 | 13 | 320 | **no** |

**The distinctive payoff of the tail lens shrinks from 6 wallets to 2.** The audit's headline was
that 6 of its 11 survivors were invisible to the mean-edge pipeline. Under the cluster-robust
bracket, 6 of the 8 survivors are wallets the pipeline already flags (ranks 1, 2, 12, 13, 17, 20)
— the tail lens is re-confirming known persisters from a variance-aware angle, which is worth
something but is not new information. Only `0x16c91ea1c8…` (rank 497) and `0xee65685de4…`
(rank 320) are genuinely surfaced by the tail lens alone. Both survive with large n (1,114 and
1,495 tail bets), wide breadth, and no concentration flag, so they are the honest remainder of
the finding.

Note also that every wallet whose "edge" was huge — hit rates of 0.42–0.65 against 6–9¢ base
rates, ROI 3–9× — is gone (§4 of the run output). Those are exactly the small-n, few-markets
records that a per-bet independence test rewards and a per-market one does not.

## The methodological finding — the cluster null is not a universal drop-in

`docs/project3_slow_markets.md` §6 and §10.1 recommend making `permute_wallets_within_market`
standard for *every* wallet-level statistic in this repo. **This run shows that needs a
qualification.**

Design effect D = Var(null) / Var(independence model), per wallet, measured at thr=0.15:

| bracket | median D | 10th | 90th | share D>1 | share D>2 |
|---|---|---|---|---|---|
| **B** cluster-preserving permutation | **0.17** | 0.03 | 0.40 | 1% | 0% |
| **C** market-block bootstrap | **1.45** | 0.34 | 6.83 | 67% | 40% |

D > 1 is what a clustering correction should produce: the independence model understates sampling
variance, so the Poisson-binomial p-values are too small by ~√D. **Null B produces D ≈ 0.17 — six
times *less* variance than independence.** The tail is sparse (median 2 tail bets per market), so
holding each wallet's per-market count fixed freezes most of its record; the permutation can
barely move, its reference distribution collapses, and any small positive residual reads as
p = 5e-4. That is why B's 35 "survivors" include wallets with `pb_p = 0.40` and a +0.005 residual.
Only 6 of 498 wallets are fully frozen (median movable share 86%), so this is not a coverage
problem — it is over-constraint, and it points the wrong way at the wallet level.

Two things follow:

1. **B's count-level comparison stands** — it contrasts the same procedure on real vs permuted
   data, so the constraint applies equally to both sides. That is why Results 1 and 2 are read
   from B.
2. **B's per-wallet p-values do not** — for individual-wallet arbitration in a sparse regime, use
   the market-block bootstrap (or any cluster-robust method whose design effect is verified > 1).
   **Measure the design effect before trusting any null.** The rule to adopt repo-wide is that,
   not one specific permutation.

Why the tail differs from the speed-gradient cells, where B worked: those cells have far denser
per-market bet counts, so the within-market permutation has real freedom. The pathology is
specific to sparse strata.

## Verdict

- The black-swan / tail-edge finding as stated in HANDOFF.md is **withdrawn**: its winner-count
  claim reverses (P(null ≥ real) = 0.99), and its persistence claim (+0.218) is fully reproduced
  by a null with no wallet↔outcome link at all.
- What survives is much smaller and should be described as such: **8 wallets clear a cluster-robust
  per-wallet test, 6 of which the mean-edge pipeline already ranks in its top 20.** The two the
  pipeline misses (`0x16c91ea1c8…`, `0xee65685de4…`) are the real, narrow payoff of the tail lens.
- The caveats from the original audit still apply to those 8 and now matter more: tail edge is the
  least copyable signal (its payoff is a rare event you cannot reliably enter alongside), so this
  is a **watchlist, never an auto-copy target**, and the forward test remains the only
  arbiter of copyability.
- Repo-wide: adopt "verify the design effect of your null" as the standard, and treat
  `permute_wallets_within_market` as one instrument among several rather than the default.
