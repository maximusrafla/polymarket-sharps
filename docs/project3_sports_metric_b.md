# Project 3 sports — Metric B / copyability of the three slow-cadence wallets

**Measured 2026-07-26. Read-only; nothing was written, no keys, no orders.**

```bash
# the whole audit (network section 5 included)
flock data/interim/.analysis.lock -c 'PYTHONPATH=. .venv/bin/python \
    scripts/audit_sports_copyability.py'
# offline only
PYTHONPATH=. .venv/bin/python scripts/audit_sports_copyability.py --no-ledger --no-live
```

---

## VERDICT

> **NO-GO for latency copying** — "see one of these wallets enter, then enter
> yourself." There is no measurable advantage to acting on their entries at any
> latency from 30 seconds to 3 days. Every positive follower number in this
> analysis is matched or beaten by a wallet-agnostic placebo.
>
> **INCONCLUSIVE for selection copying** — "take their market and side but
> refuse the chase." That strategy is not tested here and *cannot* be tested on
> this data: the placebo that kills the chase is itself a selection strategy, and
> the entire testable cohort carries only ~5 effective resolution events. It is
> open, not answered, and the forward scoreboard is the only instrument left.
>
> **n = 3 wallets. This is DIRECTIONAL, not decisive.** One of the three
> (`0x90448cec…`) dominates every pooled statistic and is the concentrated,
> baseline-fragile member; read the per-wallet tables, not the pooled row.

The five numbers that drive it:

| | |
|---|---|
| followability ceiling (any later other-wallet print before the guard) | **30.4% pooled** — 9.5% / 65.6% / 12.9% per wallet |
| copy window (guarded 24h forward fair value − entry) | **+1.2¢ pooled, median −0.05¢, positive on 25.3%**, defined on only 15.4% of bets |
| follower skill on the most generous anchor, Δ=1h | **+2.8¢ bet-weighted / +4.4¢ event-weighted**, CI [+1.6¢,+4.1¢] — i.e. positive |
| **the same number vs. its placebos** | **real − random = −0.42¢ [−1.6¢,+0.2¢]; real − pre-entry = −0.34¢ [−1.3¢,−0.0¢]** — copying is never better |
| live sports game books (24 probed) | median spread **1.0¢**, median **4,126 shares** at best ask; **0 open sports positions** across the three wallets |

---

## 0. What was asked, and against what

Three wallets, frozen 2026-07-26T06:03:11Z as tier `s2_slow` in
`data/processed/sports_sig2c_freeze_manifest.json` — the only members of the
sports cohort whose held-out cadence (3.1 / 5.8 / 7.3 bets/day) is slow enough
that a human or a bot could act on them. Everything else in that cohort is
in-play HFT.

The question is **not** "do they have skill" — that is established
(`docs/project3_sports_validation.md`, `docs/redteam_audit_2026-07-26.md` §1). It
is: *if you had seen each entry N seconds/minutes/hours later and entered then, at
the price actually available, would you have captured any of their edge?*

Method, all of it inherited rather than invented:

- **Bets**: their 3,597 resolved BUY bets in the deep sports universe, across
  1,593 markets → **816 resolution events**
  (`src/sports_events.py::assign_events`; a game is one event, a championship's
  whole outright field is also one event). Spans 2024-12-20 → 2026-07-19.
- **Baseline**: the hierarchical `niche_l1` / `niche_l2` (league, league|form)
  baseline **copied verbatim from the freeze manifest and never refit**, applied
  with `src.slow_baseline.expected_outcome_hier`. Skill edge =
  `resolved_value − E[outcome | price, league, league|form]`.
- **Leakage discipline**: `src/features.py`'s. Bounded forward window, the
  `scoring.fair_value_resolution_guard = 0.2` resolution guard, and **never** a
  `resolved_value` fallback. The follower price is computed by calling the
  production core `features._forward_price_arrays` itself — at 3.6k bets the
  vectorized `audit_edge_decay_long.FollowerPricer` buys nothing, and calling the
  production function removes any reimplementation risk (it is the function that
  script's `--verify` mode checks itself against). The bootstrap machinery
  (`cluster_boot_ci`, `effective_breadth`) is imported from that script directly.
- **Clustering**: bootstraps resample **EVENTS**, never bets, never markets. The
  headline aggregate is the **event-weighted** mean (the freeze manifest's own
  `scoring_rule`); bet-weighted is reported alongside. Design effects are printed
  for every null.

**Sanity check that the frozen baseline reproduces the identification claim:** at
Δ=0 over all 3,597 bets the subjects' own skill edge is **+2.66¢ bet-weighted /
+2.09¢ event-weighted, event CI [+1.5¢, +3.8¢], design effect 1.5–1.9**. The
manifest's median held-out skill for this tier is +2.80¢. Consistent — the
copyability analysis below is measured against the same object that was frozen.

### The systematic limit, stated once and true of every retrospective number

The deep sports tape is a backfill of **selected** wallets (`/trades?user=` for
293 sports-screened wallets). The "market tape" a follower is priced against is
therefore *other tracked wallets' prints*, not the order book. Concretely: on the
1,638 tokens these three wallets bought, the visible tape is 51,790 prints from
241 wallets, plus 4,611 more from `bet_ledger.parquet` (116 wallets) after
deduplication — **56,401 prints total, from at most a few hundred traders.**

Every coverage number below is a **lower bound**, and every follower price is a
biased sample of what was actually quotable. Section 5 exists because the live
order book is the only unbiased liquidity evidence available — and it says the
bias is large (see "What contradicts the framing", below).

---

## 1. Coverage / followability ceiling

Share of resolved BUY bets that have **any** other-wallet print after entry and
before the resolution guard. If there is no later print there is no price to
enter against, at any latency.

| wallet | bets | events | eff. events | **covered** | no-guard | delay to 1st other print (p25/p50/p75, s) |
|---|---:|---:|---:|---:|---:|---|
| `0x7e3a1f95c5…` | 666 | 382 | 206.7 | **9.5%** | 48.8% | 93 / 702 / 1,317 |
| `0x90448cec34…` | 1,243 | 61 | 4.9 | **65.6%** | 96.7% | 16,003 / 73,754 / 326,067 |
| `0x97df146fda…` | 1,688 | 387 | 81.1 | **12.9%** | 28.4% | 97,502 / 443,287 / 1,355,786 |
| **POOLED** | **3,597** | **816** | **36.2** | **30.4%** | **55.8%** | 13,759 / 82,728 / 426,110 |

Reference: the deep real-world ledger's ceiling in the 2026-07-22 run was 25%.
This cohort sits in the same place.

Two things matter more than the headline 30.4%:

1. **The pooled number is one wallet.** `0x90448cec…` supplies 815 of the 1,095
   covered bets — and it is the wallet with 61 events and **4.9 effective
   events**, i.e. the least independent evidence in the tier. The two broad
   wallets are at 9.5% and 12.9%.
2. **The wait is enormous.** Median time to the next visible other-wallet print
   is 702 s for `0x7e3a…`, 20.5 h for `0x90448…` and **5.1 days** for
   `0x97df…`. On this tape a "follower" is not reacting to an entry; it is
   waiting days for someone else to trade.

---

## 2. Copy window

`copy_window = guarded 24h forward fair value − entry price`, exactly
`features.compute_forward_drift`'s quantity.

| wallet | defined on | mean entry | mean fwd | copy_window | evt-weighted | median | share > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0x7e3a1f95c5…` | 9.5% | 0.502 | 0.563 | **+0.1051** | +0.0766 | +0.0984 | 74.6% |
| `0x90448cec34…` | 35.3% | 0.959 | 0.975 | **−0.0014** | −0.0105 | −0.0008 | 14.8% |
| `0x97df146fda…` | 3.1% | 0.824 | 0.603 | **+0.0147** | +0.0114 | +0.0009 | 53.8% |
| **POOLED** | **15.4%** | 0.811 | 0.893 | **+0.0122** | +0.0378 | −0.0005 | **25.3%** |

Read this with the entry prices in view. `0x90448cec…` enters at a mean price of
**0.959** — this is an insurance-shaped book, the exact profile
`docs/project4_value_gate.md` §4 showed split-half validation cannot test, and
there is no room left after it acts (copy window negative, positive on 14.8% of
bets). `0x97df146fda…` enters at 0.824 with a +1.5¢ window on 3.1% of its bets.
Only `0x7e3a1f95c5…` (mean entry 0.502) has a genuine +10.5¢ window — and it is
measurable on **9.5%** of its record.

Pooled median copy window is **−0.05¢**. Room is the exception, not the rule.

---

## 3. Latency decay

### 3a. At realistic bounded fill windows — unmeasurable

Coverage at each Δ, at three fill bandwidths (`follower buys inside
(entry+Δ, entry+Δ+bandwidth]`):

| Δ | cov @60s | cov @300s | cov @3600s | skill(evt) @3600s | event CI @3600s |
|---|---:|---:|---:|---:|---|
| 0 | 100% | 100% | 100% | +0.0209 | [+0.016,+0.038] |
| 30s | 0.4% | 1.2% | 4.2% | +0.0760 | [−0.037,+0.177] |
| 1m | 0.3% | 1.1% | 4.1% | +0.0890 | [−0.031,+0.177] |
| 5m | 0.2% | 0.8% | 3.8% | +0.0990 | [−0.029,+0.193] |
| 15m | 0.4% | 1.0% | 3.4% | +0.0360 | [−0.080,+0.162] |
| 1h | 0.0% | 0.3% | 2.1% | +0.1002 | [+0.010,+0.203] |
| 6h | 0.0% | 0.1% | 1.2% | +0.0335 | [+0.023,+0.058] |
| 24h | 0.0% | 0.1% | 1.7% | +0.0248 | [+0.013,+0.062] |
| 3d | 0.0% | 0.2% | 1.1% | +0.0300 | [−0.009,+0.062] |

Peak coverage anywhere in this grid is **4.2%** (Δ=30s, one-hour fill window),
and it falls to 1–2% past 15 minutes. The n behind the "positive" cells is 3 to
150 bets over 1 to 56 events, with **effective event counts of 1.0–7.5**. There
is no decay curve here to read; there is a handful of bets. Design effects are
1.1–3.8 at the usable rows (correctly > 1) and blow up to 8.6 / 29.5 at Δ=6h /
24h — which is the cluster bootstrap saying, correctly, that those rows are one
or two events wearing a sample size.

### 3b. On the most generous anchor — flat, which is the tell

To rule out "the fill window was drawn too narrow", the **next-print anchor**
takes the first other-wallet print at or after entry+Δ, however long the wait, up
to the guard. `own_*` is recomputed on the same cohort at every Δ.

| Δ | cov | n | wait p50 (s) | own_sk(bet) | own_sk(evt) | fol_sk(bet) | fol_sk(evt) | event CI | retain |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 30.4% | 1,095 | 82,728 | +0.0332 | +0.0734 | +0.0301 | +0.0636 | [+0.016,+0.049] | 87% |
| 30s | 30.4% | 1,093 | 83,382 | +0.0328 | +0.0694 | +0.0295 | +0.0594 | [+0.014,+0.049] | 86% |
| 1m | 30.3% | 1,090 | 83,617 | +0.0332 | +0.0762 | +0.0298 | +0.0655 | [+0.014,+0.051] | 86% |
| 5m | 30.2% | 1,085 | 84,916 | +0.0339 | +0.0813 | +0.0300 | +0.0682 | [+0.015,+0.050] | 84% |
| 15m | 29.9% | 1,075 | 87,413 | +0.0318 | +0.0712 | +0.0281 | +0.0580 | [+0.012,+0.049] | 81% |
| 1h | 28.6% | 1,029 | 103,288 | +0.0300 | +0.0568 | +0.0277 | +0.0439 | [+0.016,+0.041] | 77% |
| 6h | 27.9% | 1,003 | 111,636 | +0.0250 | +0.0261 | +0.0264 | +0.0404 | [+0.016,+0.038] | 155% |
| 24h | 27.4% | 987 | 119,634 | +0.0254 | +0.0298 | +0.0270 | +0.0486 | [+0.015,+0.038] | 163% |
| 3d | 26.4% | 951 | 124,243 | +0.0258 | +0.0545 | +0.0271 | +0.0695 | [+0.017,+0.038] | 128% |

**Taken alone this table reads like a GO.** A follower retains 77–90% of the
wallet's own skill, the CI excludes zero at every Δ, and the edge survives a
3-day delay.

It is not a GO, and the shape says why: the curve is **flat**. A copyable edge
decays — the whole premise of latency analysis is that information gets priced
in. An edge that is identical at Δ=30s and Δ=3 days is not being conferred by
the wallet's entry time; it is a property of the bets in the cohort. The
2026-07-22 run learned this the expensive way: there, the curve went *up* with
latency and the number was still an artifact. Here it is flat. Either way the
placebo is what adjudicates.

---

## 4. The placebos — the decisive test

Same bets, same outcomes, only the entry anchor moves.

* **random anchor** — the price available at a uniformly random moment in the
  token's pre-guard life. "Buy this market whenever." Not conditioned on the
  wallet's timing at all.
* **pre-entry anchor** — the price available at entry **minus** Δ: a "follower"
  who acted *before* the wallet and therefore cannot have been copying it.

### 4a. At bounded fill windows: NOT TESTABLE, and that is itself a result

| Δ | marginal cov (real / random / pre-entry) | paired n |
|---|---|---:|
| 30s | 1.2% / 1.2% / 1.4% | **9** |
| 5m | 0.8% / 1.4% / 1.6% | **4** |
| 1h | 0.3% / 0.8% / 0.9% | **0** |
| 24h | 0.1% / 1.0% / 0.1% | **0** |

The test that killed the last copy thesis **cannot be run** at a realistic fill
window on this tape. Reported, not hidden.

### 4b. On the next-print anchor — pooled, the only testable form

The next-print anchor is both the highest-coverage and the most generous-to-
copying anchor, so this is the strongest available form of the test.

| Δ | paired n | events (eff.) | real | random | pre-entry | **real − random** | **real − pre-entry** |
|---|---:|---:|---:|---:|---:|---|---|
| 30s | 992 | 96 (5.6) | +0.0269 | +0.0292 | +0.0279 | **−0.0022** [−0.012,+0.004] | **−0.0010** [−0.004,−0.000] |
| 5m | 988 | 86 (5.5) | +0.0302 | +0.0328 | +0.0329 | **−0.0025** [−0.012,+0.003] | **−0.0026** [−0.011,+0.000] |
| 1h | 963 | 65 (5.2) | +0.0281 | +0.0323 | +0.0315 | **−0.0042** [−0.016,+0.002] | **−0.0034** [−0.013,−0.000] |
| 24h | 926 | 49 (5.0) | +0.0263 | +0.0259 | +0.0263 | **+0.0003** [−0.006,+0.010] | **−0.0000** [−0.001,+0.002] |

Event-weighted, the same picture: real − random = +0.005 / +0.009 / −0.017 /
+0.015 — noise around zero on ~5 effective events. Design effects on the
difference are 2.7–4.5 (correctly > 1).

**Following the wallet never beats buying the same market at an arbitrary
moment.** Three of four Δs are negative; the fourth is +0.03¢.

A necessary caveat on the pre-entry column: on a tape this sparse the pre-entry
anchor usually resolves to the **same print** as the real one — identical fill in
98.5% / 94.1% / 83.6% / 41.1% of pairs at 30s / 5m / 1h / 24h. Where the anchors
genuinely differ, copying is *much* worse: on the discordant pairs only,
real − pre-entry = **−0.0664** (n=15), **−0.0447** (n=58), **−0.0206** (n=158),
−0.0000 (n=545). The direction is unambiguous and matches the 2026-07-22 finding:
a price-moving entry means the post-entry price is worse than the pre-entry price.

### 4c. Per wallet (n=3 — directional)

`0x7e3a1f95c5…` (the broadest and highest-edge wallet):

| Δ | paired n | events (eff.) | real | random | pre-entry | real − random | real − pre-entry |
|---|---:|---:|---:|---:|---:|---|---|
| 30s | 59 | 36 (21.9) | +0.1036 | +0.1276 | +0.1203 | −0.0240 [−0.064,+0.019] | −0.0167 [−0.034,−0.000] |
| 5m | 50 | 29 (17.1) | +0.0775 | +0.1040 | +0.1307 | −0.0264 [−0.073,+0.023] | −0.0532 [−0.107,−0.003] |
| 1h | 14 | 8 (5.2) | +0.1649 | +0.2622 | +0.3534 | **−0.0973** [−0.149,−0.001] | **−0.1885** [−0.301,−0.050] |
| 24h | 0 | — | — | — | — | NOT TESTABLE | NOT TESTABLE |

Copying it is *strictly worse* than both placebos at every measurable Δ, with the
CI excluding zero at Δ=1h. This is the wallet with the most real skill and the
clearest negative copyability result.

`0x90448cec34…` (65.6% coverage, but 3.5 effective events throughout): every
difference is within ±0.4¢ of zero and its own **event-weighted** skill on the
covered cohort is *negative* at 30s/5m (−0.044 / −0.026). Its bet counts are
large and its evidence is not.

`0x97df146fda…`: real − random = +0.0107 / −0.0066 / +0.0050 / **+0.0081** at
30s / 5m / 1h / 24h. The Δ=24h cell is the **one positive cell with a CI
excluding zero** in the entire analysis ([+0.001,+0.017], evt-weighted +0.0094).
It should not be believed, for three stated reasons: its **design effect is
0.17** — the exact over-constraint signature this repo documented in
`docs/blackswan_cluster_null.md`, which makes that CI unusable for arbitration;
it is 1 of 12 per-wallet × Δ cells with no multiplicity correction; and it does
not replicate at the same wallet's other three latencies. It is logged here so
that a future session can find it, not as evidence.

---

## 5. Live order book (read-only public GETs)

**There are no open sports positions to price.** Across the three wallets:
219 / 37 / 11 positions on record, of which **0 / 0 / 9 are still open**, and
none of the nine is a sports market (they are `will-gpt-6-be-released`, a
Weinstein-sentencing field, and two 2026-House markets). `0x7e3a1f95c5…` and
`0x90448cec34…` currently hold nothing at all. Their open books, for the record,
show 0.2–2.7¢ spreads with 5 to ~81,600 shares at the ask.

Substitute probe — 24 currently-open **sports game** markets (`event_kind ==
"game"`: the match lines these wallets actually trade, not the long-dated season
outrights that dominate a volume-sorted sweep):

- median spread **0.0100 (1.00¢)**; median size at best ask **4,126 shares**;
  median best ask 0.387.
- The spread distribution is wide: 0.1¢ on liquid mid-priced lines
  (`f1-…-norris`), 1.0¢ typical, and 5–24¢ on totals and thin lines
  (`mlb-bos-nyy-…-total-8pt5` at 21¢).
- Gamma reports `fee: None` on all 24. The taker fee model in
  `docs/project4_value_gate.md` is `shares × k × price × (1−price)`; `k` is not
  pinned anywhere in this repo, so only the curve factor is quoted:
  p(1−p) = **0.237** at the median ask, i.e. the fee curve is near its **maximum**
  for these mid-priced game lines — the opposite of the 0.99-band case where it
  vanished.

**Interpretation: liquidity is not the binding constraint.** There is real,
takeable size at a price a follower could hit. A taker crossing half of a 1.0¢
spread pays ~0.5¢ per bet against an event-weighted edge of 2–6¢ — survivable, if
the edge were capturable. It has not been shown to be.

---

## 6. Limits — read before quoting any number above

1. **n = 3 wallets.** Nothing here is decisive. The pooled rows are dominated by
   `0x90448cec…`, which contributes 75% of the covered bets and has 3.5–5 effective
   events.
2. **Effective events, not bets.** The pooled paired placebo test looks like
   n≈960 and is worth ~**5 independent observations**. Every CI in section 4b
   should be read at that width, and the repo's own manifest says exactly this:
   *"read effective_events, never n_bets."*
3. **The tape is selected.** Coverage is measured against ≤~350 tracked wallets'
   prints, not the book. It is a lower bound, and section 5 shows the bound is
   loose (see below).
4. **The placebo tests timing, not selection.** As `docs/redteam_audit_2026-07-26.md`
   §2 A4 correctly points out, the random anchor still requires the wallet to tell
   you *which market and side*. "Real ≈ random" therefore kills the chase and says
   nothing about whether the selection is worth having. That distinction is the
   whole reason this verdict is split.
5. **Two of three wallets are high-price bettors** (mean entry 0.959 and 0.824).
   The insurance-blindness caveat (`docs/project4_value_gate.md` §4) applies to
   them: split-half persistence cannot test a tail that has not fired.
6. **No forward data.** The `s2_slow` tier was frozen the same day this was run
   (`forward_observations_at_freeze = 0`). Nothing here is a forward test.

---

## 7. What this changes

- The `s2_slow` tier stays exactly as frozen. This audit is **identification-side
  neutral**: it does not touch the tier, the baseline, or the forward scoreboard,
  and it produces no reason to re-freeze anything. The retrospective skill claim
  reproduces (§0).
- Re-open item #2 in HANDOFF ("metric B for the slow sports wallets") is
  **answered for the latency question and explicitly left open for the selection
  question.** It should not be recorded as "sports copyability closed".
- The untested strategy — *take the selection, refuse the chase*: enter at or
  below the wallet's own price with a resting limit order rather than crossing
  the spread after it — is now the single most specific open copy hypothesis in
  the project. It is not answerable retrospectively on this tape (5 effective
  events, and a print tape that cannot tell you whether a limit would have
  filled). The sports forward scoreboard resolves in days, and it is the right
  instrument.
- **Do not open another retrospective copyability arm on this data.** Between
  this run and the 2026-07-22 long-latency run, the retrospective print tape has
  now been asked this question twice at every latency from 30 s to 3 days and has
  returned "no timing advantage" both times, for two different cohorts.

---

## 8. What contradicts the brief this audit was given

Recorded deliberately, because the framing was explicit and one part of it did
not survive contact with the data.

**"Coverage is the single most important number — if coverage is near zero,
nothing else matters."** It is the most important number *about our tape*, and it
is not the most important number about the world. The live book (§5) shows
continuous two-sided liquidity with thousands of shares at the ask in exactly the
kind of market these wallets trade. A real follower is not restricted to moments
when another *tracked* wallet happened to print; they can hit a resting order at
any time. The 30.4% figure is therefore a measurement artifact of a
selected-wallet backfill, and it is a **weak** basis for a NO-GO.

What actually carries this verdict is the placebo comparison, which is
tape-limited in the same way for the real anchor and the placebo anchors alike
and so is not confounded by the sparsity. The coverage number belongs in the
write-up as a limit on what could be measured, not as evidence that copying is
impossible.

Two smaller notes:

- The brief expected a decaying edge curve to be readable at long Δ. It is not —
  at any realistic fill window coverage never exceeds 4.2%. The long-Δ question
  is answerable only under the unbounded next-print anchor, where the curve is
  flat rather than decaying.
- The pre-entry placebo is much weaker on a slow, sparse tape than it was on the
  HFT tape, because in 41–99% of pairs it resolves to the *identical fill*. The
  discordant-pairs restriction (§4b) is the fix and it is reported separately.
