# Project 3 — Slow-specialist cross-population gradient re-run

**Status:** completed analysis, results durable here. **One-way evidence** (see
`docs/project3_slow_markets.md` §4 / §10 for why the gradient test is asymmetric).
Read-only throughout; no keys, no order placement.

**Date:** 2026-07-23/24. **Prereqs:** `docs/project3_slow_markets.md` (esp. §10, the
stage-1 flat gradient) and `docs/project3_slow_verdict.md` (the sibling 5c verdict).

> ⚠️ **Provenance / honesty flags — read first.**
> - **The full-resolution re-run did NOT complete.** The pilot's assembled bet
>   table reached only **~55% resolution coverage** before the working scratchpad
>   was wiped (resolution fetch was at ~16k of 130k markets). The numbers below are
>   the **stage-1 baseline** and the **55%-coverage preliminary** cross-population
>   comparison. **No full-resolution figures exist** — none are stated here.
> - The assembled `pilot_bets.parquet` (3.23M backfilled trades) was in the wiped
>   scratchpad and is **not recovered**. What is durable: the **cohort definition**
>   (`data/interim/project3_pilot_cohort.parquet`, regenerated deterministically
>   from the surviving discovery corpus) and the scripts (committed bf728b4).
>   §6 gives the exact steps to reproduce the assembled table.

---

## 0. What this pilot asked, and why it is not redundant with stage 1

Stage 1 (§10) found the speed gradient **flat**, but on a population selected under
the old 82%-micro regime — so it could not speak to whether *slow-market
specialists* exist, because they are largely absent from that ledger by
construction. This pilot builds a population that is **not** micro-selected: the
top 400 wallets by bet count in `discover.py`'s market-first corpus (463,561
real-world wallets across 573 real-world markets), backfilled via `/trades?user=`,
then run through the identical §10 machinery.

**Cohort provenance (durable, reproducible):**
- Source: `data/interim/discovery/discovery_trades.parquet` (market-first corpus).
- Selection: resolved BUY bets → per-wallet count + distinct-market breadth →
  **top 400 by bet count with breadth ≥ 3** (a capacity screen, NOT a skill
  screen; breadth floor guards single-event artifacts).
- Persisted: `data/interim/project3_pilot_cohort.parquet` (400 rows:
  wallet, n, breadth, ncat, t0, t1, span_d). Regenerated deterministically 2026-07-24.
- **Not micro-selected:** only **13 of 400** overlap the micro-ranked top-500;
  **254 of 400** are entirely new to the main ledger. After backfill the cohort's
  resolved BUY bets are **93.8% real-world** (vs 18% for the main ledger).

This is the right population for the question. Its one structural bias: selecting
by bet count is a capacity screen, so it favors high-frequency real-world traders
and **cannot represent the twice-a-month forecaster** (the §5.4 un-validatable
population). It answers "do frequent real-world specialists show more skill
dispersion than micro specialists," not "do rare forecasters exist."

---

## 1. Headline result: the population-level advantage is NULL

The comparison statistic is **clustering-adjusted skill dispersion** =
`varR_observed / varR_cluster_null`, per `scripts/compare_populations.py`, with a
wallet-level bootstrap CI on the difference (the between-population test §10
lacked). `varR` = between-wallet variance of mean residual edge ÷ the
sampling-noise floor; dividing by each population's **own** cluster-preserving null
(permute wallet labels within market) is what makes two cohorts of very different
clustering comparable.

| population | bets | wallets | median depth | varR | cluster null | **adjusted ratio** | bootstrap 95% CI |
|---|---|---|---|---|---|---|---|
| micro-selected ledger | 734,208 | 3,192 | 2 | 4.08 | 1.53 | **2.664** | [1.693, 4.005] |
| slow-specialist pilot (55%) | 1,066,819 | 390 | 1,810 | 39.00 | 13.34 | **2.924** | [1.612, 4.796] |

**Difference (pilot − micro-selected): +0.259, 95% CI [−1.526, +2.432] — includes 0.
P(pilot > micro-selected) = 0.590.**

The advantage is **not distinguishable from zero.** Both populations carry real,
significant skill dispersion (both ratios sit well above 1) — but the slow cohort
does **not** carry measurably *more* of it than the micro cohort.

**The within-wallet paired tests agree — all three null** (same wallet, slow bucket
vs fast bucket, controls identity):

| comparison | wallets | mean(slow − fast) | t p | Wilcoxon p | improving in slow |
|---|---|---|---|---|---|
| L: <24h vs ≥7d | 71 | −0.0028 | 0.887 | 0.574 | 31/71 (44%) |
| L: <3d vs ≥3d | 161 | +0.0039 | 0.698 | 0.551 | 86/161 (53%) |
| H: <6h vs ≥3d | 237 | −0.0047 | 0.516 | 0.385 | 109/237 (46%) |

Effect sizes ≤0.5¢/bet, inconsistent signs, coin-flip improving-counts. Same
result as stage 1, now on a population selected for real-world (not micro) trading.

> **Do not over-read the raw ρ / varR.** Pilot median depth is **1,810 bets/wallet
> vs 2** in the ledger, and both ρ and varR rise mechanically with depth. That is
> why raw ρ (+0.34 vs +0.16) and raw varR (39.0 vs 4.1) look dramatic and mean
> little — only the cluster-adjusted ratio is comparable, and it came back null.

---

## 2. The one suggestive signal — and why it is NOT yet skill evidence

Per-bucket adjusted ratios from the pilot audit (`audit_speed_gradient.py`,
`--bets pilot`, 55% coverage). Stage-1 (micro) ratios shown for contrast:

| bucket | pilot ratio | stage-1 ratio |
|---|---|---|
| **L > 30d** | **4.46** | 1.70 |
| L 7–30d | 2.13 | 2.59 |
| L 3–7d | 2.09 | 1.67 |
| L 1–3d | 1.96 | 2.03 |
| L 6–24h | 2.65 | 2.52 |
| L <6h | 1.64 | 1.78 |
| **H > 7d** | **2.61** | 1.54 |
| H 3–7d | 2.61 | 2.14 |
| H 1–3d | 1.56 | 1.49 |
| H 6–24h | 1.63 | 2.41 |
| H 1–6h | 2.08 | 1.92 |
| **H < 1h** | **1.18** | 2.09 |
| micro control | **1.39** (lowest) | 1.66 |

Two things stage 1 did not show: **L>30d = 4.46 is the highest cell in either
population** (on 311,426 bets / 358 wallets), and the **H-axis endpoints are
ordered the thesis-predicted way** — near-resolution lowest (1.18), longest horizon
highest (2.61). Micro is now the *lowest* cell rather than mid-pack.

**This is a hypothesis, not a finding, for three independent reasons:**

1. **No per-bucket interval.** The *aggregate* CI was already [1.6, 4.8]; a single
   bucket's is wider still, and 12 buckets give 12 chances at a high outlier. The
   difference test that killed the aggregate (§1) was not run per bucket.
2. **55% coverage, and the missing markets skew OLD** (median last trade 2026-03-04
   vs 2026-07-03 for resolved). Long-lifespan markets live exactly in that older
   tail, so **L>30d is the cell most likely to move at full resolution.** It is
   built on the least-complete data.
3. **The H-axis is confounded with price mix — verdict below.**

### 2.1 VERDICT on the ordered H-axis: price-mix confound is NOT excluded

**Plainly: the ordered H-axis is real in the data, but it is NOT established as
horizon skill. A price-geometry confound is present, only partially controlled, and
the decisive check was not run. Treat the H-axis ordering as unproven.**

The mechanism (from §5 of the stage-1 doc, measured): near-resolution (H<1h) bets
cluster at **extreme prices** (→0 / →1), where residual SD is **0.165–0.227**;
long-horizon (H>7d) bets span **mid-book**, where residual SD is **0.489**. At an
extreme entry price the residual is *mechanically range-bounded* (you cannot beat a
0.97 price by more than +0.03), so between-wallet skill dispersion is compressed
there **independent of any skill** — which alone predicts the low end of the
H-axis.

What the cluster-adjusted ratio **does** control: the null permutes wallet labels
*within each market*, holding each bet's price and outcome fixed. So pure
residual-range compression and within-market price mix appear in **both** numerator
and null and largely divide out — this is exactly why the adjusted ratio is the
right statistic and why I trust the *aggregate* null in §1.

What it does **NOT** control: **cross-market, cross-wallet price-regime
concentration.** A longshot-specialist wallet (only 0.9+ markets) and a mid-book
wallet trade *different* markets, so the within-market permutation can never swap
them; any residual baseline mis-calibration that a wallet inherits from its price
concentration then reads as between-wallet dispersion that is not skill. Long-horizon
markets, spanning the full price range, give more room for this to vary across
wallets than near-resolution markets do.

**Evidence it is at least partly artifact, not pure skill:**
- The H-axis is **not monotone** — 1.18, 2.08, 1.63, 1.56, 2.61, 2.61. Only the
  endpoints order cleanly; the middle is noise-shaped.
- The lowest bucket (H<1h, 1.18) coincides exactly with the most extreme-price,
  most range-compressed regime — the geometry story predicts this without invoking
  skill.

**Evidence it is not *pure* artifact:** the highest H buckets (3–7d, >7d) reach 2.61
against a cluster null already ~4–6, i.e. dispersion beyond within-market
reshuffling. But that surplus is exactly where the uncontrolled cross-wallet
price-concentration confound lives, so it cannot be attributed to horizon skill on
this evidence.

**The decisive check that was NOT run (top follow-up):** restrict every H (and L)
bucket to a **common price band** (e.g. 0.25–0.75, where residual SD is ~flat at
0.48–0.49) and re-measure the adjusted ratio. If the ordering survives inside a
fixed price band, it is horizon skill; if it collapses, it was price geometry. The
scratchpad was wiped before this could run. It needs the reassembled pilot bets
(§6) — cheap once those exist.

---

## 3. Relation to the 5c verdict (`docs/project3_slow_verdict.md`)

The sibling 5c verdict: **52 slow-market forecasters persist out-of-sample,
horizon-robust — identification, not copyability.** This gradient re-run is the
complementary population-level lens, and the two **corroborate, with a
qualification**:

- **Corroborates:** both pilot and micro populations show adjusted skill dispersion
  significantly above 1 (2.66 / 2.92). Real, identifiable skill heterogeneity in
  real-world markets **exists** — fully consistent with 5c isolating 52 specific
  persisters. A skilled identifiable minority is exactly what elevated-but-not-huge
  dispersion looks like.
- **Qualifies:** the *slow-vs-fast population contrast* is **null** (§1), and the
  within-wallet slow-vs-fast test is null. So slowness is **not a population-level
  amplifier** of skill on this evidence. 5c's forecasters are findable *as specific
  wallets*, but "slow markets are categorically where the edge lives" is **not**
  supported at the population level — the edge is concentrated in individuals, not
  conferred by the regime. This is squarely consistent with 5c's own framing
  (identification, not a broad copyable/regime effect).

**Net:** the two results do not conflict. 5c says *these specific wallets are sharp
and it survives OOS*; the gradient says *slowness itself doesn't make a population
sharper*. Both point to identification-of-individuals as the viable path, and away
from "trade the slow regime" as a blanket thesis.

---

## 4. Caveats (consolidated)

1. **55% resolution coverage; missing markets skew old** → L>30d least reliable
   (§2, reason 2). Full-resolution re-run not completed.
2. **Depth is not matched across populations** (1,810 vs 2 bets/wallet); only the
   cluster-adjusted ratio is comparable (§1). `compare_populations.py --depth-cap`
   exists to match depth but was not run at full resolution.
3. **H-axis price-mix confound unresolved** (§2.1) — the common-price-band check is
   the top follow-up.
4. **Capacity-screened cohort** — cannot speak to low-frequency (twice-a-month)
   forecasters (§0, and §5.4 of the stage-1 doc).
5. **1 of 400 wallets lost** to repeated HTTP 500s at offset 9500 during backfill
   (0.25%). The backfill wrapper discarded that wallet on the exception rather than
   keeping its already-fetched ~9,500 trades — the same failure mode HANDOFF
   documents for the 408 case. If the pilot backfill is reused, keep partials.

---

## 5. Reproducing the pilot bet table (what was wiped)

The cohort is durable; the assembled bets are not. To rebuild `pilot_bets.parquet`:

1. **Cohort:** `data/interim/project3_pilot_cohort.parquet` (400 wallets) — already
   persisted; or regenerate from `discovery_trades.parquet` (top 400 by resolved-BUY
   count, breadth ≥ 3).
2. **Backfill** each wallet via `/trades?user=` (`src.backfill.fetch_user_trades`),
   into an **isolated** parquet — do NOT merge into the shared ledger (that merge is
   what broke the VM). ~20 min for the 400 at ~4 workers; ~3.23M raw trades.
3. **Resolutions:** join the two existing read-only caches
   (`data/raw/resolutions.parquet`, `discovery/discovery_resolutions.parquet`) first;
   then CLOB-fetch the remainder (~130k markets, ~100 min at ~22 req/s). *This is the
   step that did not finish — 55% coverage was reached.* At full coverage the older
   long-lifespan markets fill in.
4. **Metadata:** `data/interim/market_meta.parquet` already covers open/end/game
   timestamps (durable, from the step-1 build).
5. **Audit:** `PYTHONPATH=. python scripts/audit_speed_gradient.py --bets <pilot_bets>
   --meta data/interim/market_meta.parquet --label pilot`; cross-compare with
   `scripts/compare_populations.py --a <ledger_rw> --b <pilot_bets> --boot 2000`
   (add `--depth-cap` to match depth). **Then run the §2.1 common-price-band check.**

---

## 6. Bottom line

On the question the pilot was built to answer — *do slow-market specialists, in a
population not selected by the micro regime, show more skill than micro
specialists?* — the answer at the population level is **no, not detectably**
(difference +0.26, CI includes 0; paired tests null). The stage-1 flat gradient
**replicates** on a de-biased population. The one suggestive residue (L>30d = 4.46,
ordered H-axis) is **confounded, under-covered, and unproven**, with a named next
check. This does not overturn 5c's 52-forecaster positive — it complements it:
skill is real and identifiable in **specific wallets**, but **slowness is not a
population-level amplifier** of it.
