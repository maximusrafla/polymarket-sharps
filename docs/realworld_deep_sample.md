# The deep real-world sample, and what the screens say on it

**Built 2026-07-27/28.** Code: `src/realworld_deepen.py`, `src/realworld_validate.py`,
`scripts/run_realworld_deepen.sh`, `scripts/run_realworld_screens.sh`.
Commits: `8601e09` (build), `382c7bb` (fetch complete + profile), `f56fca3` (screens
+ P1 gate wiring), `0954022` (OOM fix).

---

## 1. Why this exists: the thin real-world sample was a collection artifact

Every "the real-world sample is too thin" conclusion in this repo traced to one
number — the shared ledger holds **3,192 real-world wallets, 98 with 1000+ bets**.
That number never described the population. It described the collector.

`src/ingest.py` is a global-firehose poller and **~82% of the firehose is 5-minute
crypto**, so real-world traders barely register no matter how long it runs.
Meanwhile `data/interim/discovery/discovery_trades.parquet` — built **market-first**
over just **585 markets, 0.6% of the 101,591 real-world markets in the ledger** —
already held **542,397 wallets, 10,791 of them with ≥20 bets**, sitting unused on
disk. The missing step was never more discovery. It was **deepening**.

## 2. What makes this sample different from the slow/sports arms

Those arms deepened a *screen shortlist* — wallets picked for measured edge, behind
a disjoint-data firewall. That is the right shape when chasing a named hypothesis.
This arm inverts it, because the question here is about a *population*, not a
hypothesis.

**Selection is performance-blind.** The sampler is handed two columns — discovery
bet count and distinct-market count. It never sees edge, profit, win rate or
residual skill. A test asserts that adding a performance column which would flip
any performance-based ranking moves **zero** wallets.

**It is a probability sample with known weights**, not a top-N slice:

| stratum | frame | drawn | inclusion p |
|---|---|---|---|
| ≥200 discovery bets | 397 | 397 | 1.0000 |
| 100–199 | 342 | 342 | 1.0000 |
| 50–99 | 1,172 | 1,172 | 1.0000 |
| 20–49 | 4,891 | 589 | **0.1204** |
| **total** | **6,802** | **2,500** | |

Frame = discovery wallets with ≥20 bets **and** ≥5 distinct markets. Seed 20260727,
frozen. The whole frame is persisted to `pool_census.parquet` with every wallet's
stratum, inclusion probability and `selected` flag — nothing dropped, per CLAUDE.md
— so results reweight to the frame by Horvitz–Thompson.

> ⚠️ **The frame is not "real-world Polymarket traders."** It is *wallets with ≥20
> bets and ≥5 markets inside the 585-market discovery corpus*. Generalizing past
> that is unsupported by this design, and is exactly the error the volume-selected
> census made (red-team audit §3).

**Reuse without bias.** 383 drawn wallets were already deepened by the slow/sports
arms and were reused rather than re-fetched. Ordering is deliberate: **the draw
happens first, the on-disk check second**, so inclusion probability is untouched.
Checking first — "sample from the not-yet-deepened wallets" — would have
conditioned the frame on a performance-correlated variable, since those arms
selected on edge.

## 3. What was collected

Fetch 2 h 22 m, resolve 4 h 23 m, finished 2026-07-27T13:29:10Z. No warnings, no
fetch errors. **0.52 GB against a 1.6 GB cap** — the disk guard never fired.

| | ledger's real-world side | **this sample** |
|---|---|---|
| wallets | 3,192 | **2,117 deepened (+383 reused)** |
| wallets with 1000+ real-world bets | **98** | **758** |
| resolved BUY bets | — | **3,842,708** (3,385,372 real-world, 88.1%) |
| markets / resolution events | — | 410,523 / 166,230 |
| span | — | 2023-07-27 → 2026-07-27 |

Survivorship is now ~100% at every depth threshold (1,384 wallets with ≥100
screen-window bets), and only **0.5%** of the population was ever touched by the
ranked-table-seeded backfill — so the screens are not being shown their own
selection.

**Measuring the real-world share needs the slug classifier, not `market_meta`.**
The sidecar covers 116k of these 410k markets (294,529 absent, 32,476 NaN
lifespan), which would park 66% of the tape as "unknown". Deriving a lifespan from
our own tape would be *actively wrong*: we hold only our wallets' fills, so a
market touched once looks zero-second and reads as micro. `discover.classify_market`
on slug/question has no such failure mode. **Caveat: 48.8% of the tape lands in the
classifier's `other` fallback** — sampling shows geopolitics, golf, esports and
league codes the keyword rules don't know (`fl1`, `itc`, `fifwc`, `lol`). That is a
*granularity* gap, not a micro leak: the 88.1% holds, finer splits below `other`
do not.

## 4. What the screens say

Same rules as every prior run — `audit_screen_surface.py` gained a `--tape` switch
and nothing else. Only the population changed.

### The 2-D screen: dead, and cleanly

| | |
|---|---|
| real best region | +0.0595 |
| null P median | **+0.0771** — the real optimum is *worse* than the median null (**p = 0.85**) |
| null O / null C | p = 0.77 / **p = 1.00** |
| untouched W3 window | **47.6th percentile** of random selections of the same size |

A coin flip. This repeats the prior negative on a population that is not crypto
residue, which makes it a considerably stronger negative than the one it repeats.

### The owner's ROI screen: dead at every normal depth; one cell survives

Top-decile W1 ROI at depth ≥3 / ≥30 / ≥100 → net **−0.0027 / −0.0020 / +0.0007**,
wallet CIs all spanning zero. Note it is **no longer backwards** — on the ledger it
lost to its own pool at permutation p ≥ 0.93.

The exception is the deepest stratum: **depth ≥500, keep top 10% (63 wallets)** →
net median **+0.0153** (p = 0.002), resid median +0.0293 (p = 0.002), **+0.0394
excess over the size-matched index**, so not a small-book artifact, with a coherent
monotone depth gradient behind it.

**Read it as a hypothesis, not a result.** It is 1 cell in a 27-cell nested search;
3 of 27 cleared p<0.05 against 1.4 expected, which is chance-level on count. Only
the direction (1 positive, 2 negative) and the gradient argue for it. p = 0.002 is
the 500-permutation floor — "no permutation beat it" — which is the strongest that
test can say and still one cell of 27.

### The certainty screen: survives every control, fails its own null

Every CI excludes zero — gross +0.0322 [+0.0118, +0.0538], **net +0.0217 [+0.0025,
+0.0432]**, wallet-resampled [+0.0124, +0.0699], equal-weight net +0.0322. The
price-mimicking index explains only +0.0075, leaving **+0.0248 of non-price
excess**. Real per-category fees eat 33% of the gross edge, not all of it.

**But null P — the null that charges for the selection step — puts it at p = 0.218**
at the pre-registered α = 0.005. Looser alphas reach p = 0.006–0.002, but that is
21 looks at the same 500 permutations, and 0.005 is `scoring.project1.
oos_significance_alpha`, the production value, not a tuned one. Null O is the one
supporting signal: it certifies **32 wallets against a null median of 9, p = 0.039**.

## 5. The Project 1 gate on this population

Same gate, unchanged: candidacy → cluster-robust market-block bootstrap at the
scoped α = 0.005 → magnitude floor 2¢ → ≥30 held-out markets.
`python -m src.realworld_validate`, 4,880,446 resolved BUY bets, 2,475 wallets.

| gate | sample | frame estimate (of 6,762) |
|---|---|---|
| candidates (in-sample residual edge > 0) | 1,489 / 2,475 (60.2%) | — |
| `edge_significant` | 427 | 1,077 (15.9%) |
| `edge_magnitude_ok` | 434 | 1,077 (15.9%) |
| `edge_markets_ok` | 2,026 | 5,729 (84.7%) |
| **`edge_persisted`** | **58** | **146 (2.15%)**, 95% CI [93, 198] |

Certified wallets: **median held-out skill edge +5.0¢ over a median 314 held-out
markets**.

Three structural checks that make the count more credible than its face value:

- **The gates are nearly disjoint.** 427 significant, 434 magnitude-ok, but only
  **67 in both**. Significant wallets mostly have small edges; big-edge wallets
  mostly are not significant. The 58 are the intersection — which is the gate
  doing its job, not a coincidence.
- **Not driven by pre-selected wallets.** The 381 prior-arm wallets (chosen by the
  slow/sports *edge* screens) certify at **3.15%**; the 2,094 never-screened at
  **2.20%**. The clean, selection-free number is **46 fresh certifications**.
- **Uniform across strata** (2.04 / 2.65 / 2.06 / 2.11%), so the Horvitz–Thompson
  reweighting is not leaning on a single band.

### The FDR, measured: 17.3% ± 0.4% — about 48 of the 58 are real

`scripts/audit_persistence_fdr.py --tape realworld --shuffles 200 --boot 2000`,
completed 2026-07-28T02:13:14Z. The real arm reproduced the shipped artifact
**bit-for-bit before any null was scored** — 0 mismatches on all four gates
(427/434/2026/58) — so the null is calibrated against the real number, and the 58
is now reproduced by two independent implementations over two different code paths.

| | null B2 (unconditional) | null B (conditional) |
|---|---|---|
| null persisted, mean | **10.03** ± 0.21 (sd 2.94, max 16) | 40.12 ± 0.37 (sd 5.27, max 53) |
| **implied FDR** | **17.3% ± 0.4%** | 69.2% ± 0.6% |
| P(null ≥ real) | 0.000 | 0.000 |
| label retention | **0.041** | 0.251 |
| null persisters that ARE real certified members | **0.27** | **9.73** |
| design effect D (certified 58) | median **4.91**, **100% > 1** | median 0.81, 5.2% > 1 |

**B2 is the readable null and B is not**, on the script's own pre-registered
criteria rather than on preference:

- **D ≥ 1 for 100% of the certified 58 under B2** (median 4.91), so it is at least
  as wide as independence — it can arbitrate at the wallet level, and being wider
  makes its FDR *conservative*. Under B, D is 0.81 with only 5.2% above 1: more
  constrained than independence, which `docs/blackswan_cluster_null.md` established
  disqualifies a null from per-wallet use.
- **B hands wallets back 25% of their own record**, and it shows: **9.73 of B's
  40.12 null "persisters" are actual members of the real certified set**. B is
  partly re-certifying true signal and scoring it as a false discovery, which
  inflates its ratio. B2's equivalent is **0.27** — essentially zero, so its null
  wallets are genuinely manufactured, not recycled.

**FDR curve under B2** — the gate at tighter significance:

| α | real | null mean | FDR |
|---|---|---|---|
| 0.05 | 136 | 67.11 | 49% |
| 0.0171 | 98 | 27.49 | 28% |
| 0.0073 | 73 | 13.27 | 18% |
| **0.005 (production)** | **58** | **10.03** | **17%** |
| 0.001 | 34 | 2.68 | **8%** |

So there are two defensible sets: **58 at 17% FDR (~48 real)** and a tighter
**34 at 8% FDR (~31 real)**. For comparison the shared ledger measured **20% FDR at
the same α, yielding 26**. This population gives **more than twice the certified
wallets at a slightly better false-discovery rate** — and the two calibrations
landing this close on completely different tapes is independent support for the
gate itself.

### What this does and does not establish

**IDENTIFICATION, not actionability.** ~48 wallets carry real, out-of-sample,
cluster-robust, cost-relevant skill edge. That is a statement about *whose bets
beat the price they paid*, and this repo has spent months establishing that the
step from there to *capturing any of that edge by following them* is where every
attempt has died.
The screens in §4 say exactly that on this same population: the certainty screen
(this gate applied W1-only) shows net +2.17% with both CIs excluding zero, yet
fails null P at the pre-registered α with p = 0.218. Identification is solid here;
copyability is untested and is the next question, not a corollary.

**The frame is still the frame.** ~48 real sharp wallets among *wallets with ≥20
bets and ≥5 markets in the 585-market discovery corpus*. The Horvitz–Thompson
figure (2.15% of 6,762) generalizes to that frame and no further.

**A discarded diagnostic, recorded so it is not repeated.** A Storey π₀ estimate
on the `out_of_sample_cluster_p` distribution returned "FDR ≈ 0%". It is invalid:
`validate.py:435` computes the bootstrap p **only when held-out edge is positive**,
so the distribution is conditioned and truncated at 0.5 by construction, while
Storey's method assumes full support on [0,1]. The method was inapplicable — the
answer was not favourable.

## 6. Bottom line


**The identification engine works on this population. The copy question is untouched.**

**~48 genuinely sharp real-world wallets exist and are named** — 58 certified at a
measured **17.3% FDR**, or a tighter **34 at 8%**. That is the deepening's payoff:
the largest certified real-world set this project has produced, against 26 from the
entire shared ledger, and it exists because the sample was collected properly rather
than because a new method was invented.

**No screen clears its pre-registered test**, and that is a separate finding, not a
contradiction. What changed is the *shape* of the screen failure. Previously the ROI
screen ran backwards and the one screen that "worked" was a crypto-frequency
artifact (16 of its 17 wallets were the 5-minute tape). On a deep, real-world-only,
performance-blind sample the certainty screen is directionally positive, survives
price control, cost control and both CIs, and fails only on significance against
the selection null — an **underpowered** result rather than a refuted one.

The two results are consistent and both are worth carrying: *which wallets are
sharp* is now answered with a measured error rate; *whether a rule that picks them
forward captures anything* is not.

Two figures worth keeping in view: the pool baseline is **−0.0063 net**, so the
average wallet in this population is net-negative after fees; and the deep-stratum ROI
cell and the certainty screen are *both* concentrated in high-activity wallets,
which is where copyability is hardest.
