# Project 4 — is the structural mispricing real? The cheap gate

**Verdict: no.** Dated 2026-07-26. Read-only throughout; no keys, no orders.
Reproduce with the three scripts named below.

The copy thesis is exhausted (`docs/project3_category_space.md` and the HANDOFF
sections above it). This gate asked a prior question: forget the wallets — is the
favorite-longshot mispricing this repo had been quoting a property of the *market*
at all? The figure under test was the one already measured here and quoted in the
brief as *~+1.8¢ average buyer's edge, +14¢ in the 0.6–0.8 band*.

The answer is no, and the reason is **not** the expected one. Costs do not eat this
edge — they are small exactly where the candidate edge lives, which is why §3 is
kept below as a control rather than a verdict. It fails one step earlier: on
unselected data the edge is **not statistically distinguishable from fair pricing
at all**.

---

## 0. The result in one table

| gate | question | outcome |
|---|---|---|
| **G0** | Is the measured favorite-longshot edge a property of the market or of our wallet selection? | **Mostly selection.** The brief's numbers do not reproduce on unselected data. A real but different FLB tilt survives. |
| **G1** | Does it persist out-of-sample? | **One band only.** 0.90–0.95 and 0.95–0.98 decay to CIs spanning zero; mid bands flip sign; 0.98–1.00 persists (+0.39¢ → +0.43¢). |
| **G2a** | Net of fees + spread? | **Survives.** Tick is 0.1¢; effective spread ~0.16¢; the fee curve vanishes near 100¢. Costs take ~25% of the edge, not all of it. |
| **G2b** | Is the surviving edge distinguishable from fair pricing? | **No — p = 0.33 to 0.47.** This is what kills it. |
| **#2** | Cross-market arbitrage? | **Not tractable from existing data** — needs a live book capture. |
| **#3** | Stale-price / news-lag? | Not run; blocked in kind (needs an external news clock). |

---

## 1. G0 — the edge source is largely a selection artifact

`scripts/audit_flb_selection.py`

Two datasets answer "what is E[outcome | entry_price]?" and they are not
interchangeable:

- `data/interim/bet_ledger.parquet` — 500 wallets **chosen by this repo's own
  sharpness ranking**, then deep-backfilled. 4.07M resolved BUY bets.
- `data/interim/discovery/discovery_trades.parquet` — `/trades?market=` tapes,
  i.e. **every wallet that traded those markets**. 1.69M resolved BUY bets, 573
  markets, 463,561 distinct wallets. Unselected with respect to wallet skill.

| cohort | bets | markets | wallets | mean raw edge | 95% CI (market-block) | 0.6–0.8 band |
|---|---:|---:|---:|---:|---|---:|
| ledger (selected wallets) | 4,070,091 | 313,611 | 10,821 | **+0.0087** | [+0.0074, +0.0101] | +0.0135 |
| census (market-centric) | 1,685,995 | 573 | 463,561 | **−0.0016** | [−0.0052, +0.0019] | **−0.0394** |
| census, untruncated | 1,675,503 | 572 | 462,697 | −0.0016 | [−0.0051, +0.0015] | −0.0394 |
| census, untrunc. real-world | 1,667,902 | 571 | 460,651 | −0.0016 | [−0.0051, +0.0014] | −0.0394 |

**Three findings.**

1. **The brief's headline numbers are stale.** The "+14¢ in the 0.6–0.8 band"
   comes from the 2026-07-18 red-team, run on a 7,014-wallet ledger of thin,
   poller-observed samples. On today's ledger that band is **+1.35¢**, and the
   aggregate is +0.87¢, not +1.8¢. HANDOFF's Favorite-longshot section should be
   read with that correction.

2. **The ledger curve is positive in every single band** (+0.03¢ to +1.6¢, peaking
   mid-book). That is not what a favorite-longshot bias looks like — a true FLB is
   a *tilt*, negative at low prices and positive at high. A curve that is positive
   everywhere is the signature of *wallets selected for beating the price*, which
   is exactly how that ledger was built. It is not a curve anyone can trade.

3. **On unselected data the 0.6–0.8 band is −3.9¢**, and aggregate buyer edge is
   zero. The matched-market cut makes the point sharpest: over the 138 markets
   present in both datasets, the same period and the same price process, the
   selected wallets earn +21¢ to +38¢ in the 0.5–0.8 bands where the full tape of
   those markets earns +2.8¢ to +18¢. Same markets, different wallets, different
   answer.

**But a genuine FLB tilt does survive in the census** — just not where the brief
said. Longshots are overpriced and heavy favorites underpriced:

| band | n | markets | events | edge | 95% CI (event-block) |
|---|---:|---:|---:|---:|---|
| [0.02,0.05) | 80,985 | 359 | 143 | −0.0275 | [−0.0319, −0.0188] |
| [0.05,0.10) | 64,894 | 324 | 136 | −0.0456 | [−0.0659, −0.0041] |
| [0.10,0.20) | 33,010 | 259 | 119 | −0.0622 | [−0.1121, +0.0028] |
| … mid bands … | | | | ~0 | all span zero |
| [0.90,0.95) | 78,117 | 306 | 129 | +0.0435 | [+0.0001, +0.0676] |
| [0.95,0.98) | 154,226 | 378 | 141 | +0.0231 | [+0.0104, +0.0315] |
| [0.98,1.00) | 796,453 | 564 | 179 | +0.0040 | [+0.0035, +0.0047] |

The two ends **mirror each other**, which is a real internal consistency check:
buying NO at 0.96 *is* declining YES at 0.04, and the −2.75¢ measured at
[0.02,0.05) predicts the +2.31¢ measured at [0.95,0.98). So the tilt is coherent,
not a fluke of one band.

Controls applied: `/trades?market=` truncates at ~10,500 rows and drops the
*earliest* entries, so capped markets are their own near-resolution tail — removing
them changes nothing. Micro-crypto split out — changes nothing. And 573 `market_id`s
fold to **179 resolution events** (`src/sports_events.resolution_event`), so all CIs
above are event-blocked; the market→event correction costs a design effect of
1.1–2.7.

So the only surviving candidate is the heavy-favorite band, and it goes to G1.

---

## 2. G1 — only the top band survives out-of-sample

`scripts/audit_flb_tradeability.py`. Events split chronologically, 90 in-sample /
89 held out (events, not bets — a bet-level split leaks one event across the
boundary).

| band | in-sample edge | held-out edge | held-out 95% CI (event) | notional-wtd | sign |
|---|---:|---:|---|---:|---|
| [0.30,0.40) | −0.1192 | +0.0529 | [−0.1356, +0.3317] | +0.4315 | **FLIP** |
| [0.60,0.70) | +0.0385 | −0.1352 | [−0.3602, +0.0553] | −0.1022 | **FLIP** |
| [0.70,0.80) | +0.0939 | −0.1032 | [−0.2930, +0.0597] | −0.1213 | **FLIP** |
| [0.80,0.90) | +0.1116 | −0.0358 | [−0.1735, +0.0746] | −0.0978 | **FLIP** |
| [0.90,0.95) | +0.0647 | +0.0173 | [−0.0980, +0.0688] | +0.0282 | ok |
| [0.95,0.98) | +0.0271 | +0.0174 | [−0.0170, +0.0323] | +0.0220 | ok |
| **[0.98,1.00)** | **+0.0039** | **+0.0043** | **[+0.0035, +0.0052]** | **+0.0027** | ok |

Everything below 0.90 is noise — four bands flip sign. The two lower favorite
bands keep their sign but decay (6.5¢ → 1.7¢) into CIs spanning zero. **Only
[0.98,1.00) survives**, at +0.43¢ per share / +0.27¢ per dollar of notional.

---

## 3. G2a — the cost control, which this edge survives

Worth stating plainly because the honest prior said they would.

- **Tick size is 0.001, not 0.01.** 98.3% of prints at p ≥ 0.90 sit on the 0.1¢
  grid. My initial read that a 1¢ tick would obliterate a 0.4¢ edge was wrong.
- **Effective spread, from the complement identity** (buying NO at *q* ≡ selling
  YES at 1−*q*, so two contemporaneous BUY prints on complementary tokens give
  `p_yes + p_no − 1` directly — no order-book history needed):

  | band | n pairs | mean spread | median | p75 |
  |---|---:|---:|---:|---:|
  | [0.90,0.95) | 5,831 | 0.0062 | 0.0010 | 0.0050 |
  | [0.95,0.98) | 10,558 | 0.0023 | 0.0000 | 0.0020 |
  | [0.98,1.00) | 19,332 | 0.0016 | 0.0010 | 0.0010 |

  A taker crossing half that spread pays ~0.08% of notional against a 0.34%
  gross return.
- **Fees vanish exactly here.** Polymarket's taker fee is
  `shares × k × price × (1 − price)`, which peaks at 50¢ and approaches zero at
  either end. At p ≈ 0.996 the fee is ~3% of the candidate edge. Makers pay zero
  and earn a rebate.

Net: costs take roughly a quarter of the edge in the surviving band. Not fatal.

---

## 4. G2b — what actually kills it: the edge is not distinguishable from fair pricing

Buying at 0.99 is **selling insurance**: a small premium almost always, a
near-total loss rarely. The entire measured edge in [0.98,1.00) is the claim that
the tail was overpriced — priced loss rate **0.41%**, realized loss rate
**0.0067%**, a 60× gap. Everything rests on how often the tail actually fired.

It fired **4 times in 179 events**, and **4.35 additional average-sized tail
events would erase the entire measured excess over the 850-day tape.**

One benign explanation was checked and ruled out: this is **not** settlement dust.
If the favourite-band edge were just carrying near-certain contracts the last mile
to resolution, it would concentrate at the end of each market's tape. It does not —
across tape-position quintiles the [0.98,1.00) edge is flat at +0.0037 to +0.0046.
The edge is real in-sample and uniform across the market's life. That is what makes
the next test the one that matters.

This is precisely where a bootstrap fails. A market-block bootstrap resamples the
tail events that *occurred*; it cannot represent the ones that didn't. Its tight
CI ([+0.0035, +0.0047]) is anti-conservative by construction.

The correct test is a **fair-price null**: settle each market by one binary draw at
its own notional-weighted price, with events block-resampled so within-event
correlation is preserved. Under fair pricing E[P&L] = 0 exactly, so this asks
directly "would a fairly-priced market have produced a run this good?"

| cohort | band | realised return | null sd (return units) | **p(null ≥ real)** |
|---|---|---:|---:|---:|
| all events | [0.98,1.00) | 0.34% | 0.89% | **0.183** |
| all events | [0.90,1.00) | 0.64% | 0.94% | **0.086** |
| single-market events | [0.98,1.00) | 0.31% | 1.08% | **0.331** |
| single-market events | [0.90,1.00) | 0.41% | 1.18% | **0.471** |

The single-market rows are the trustworthy ones: within a multi-outcome event
exactly one outcome wins, so drawing legs independently misstates the variance —
restricting to the 133 single-market events makes the independence assumption
exactly right. There, **p = 0.47**. The band's whole 850-day record is what a fairly-priced
market produces about half the time.

**Methodological note this repo should keep.** Out-of-sample persistence — the
validation apparatus everything here is built on — is *powerless* against this
failure mode. Both halves under-observe the tail, so their agreement is nearly
automatic. [0.98,1.00) passing G1 cleanly and failing G2b is not a contradiction;
split-half persistence simply cannot test an insurance-shaped payoff.

---

## 5. G3 — not applicable

This gate asked about tradeable size. It is out of scope for a measurement
study and was dropped.

---

## 6. #2 cross-market arbitrage — not tractable from this corpus

Measured anyway, because §4's residual concentrates in multi-outcome events. It
**cannot be answered with the data we have**, for two independent reasons:

1. **The outcome sets are incomplete.** `discover.py` selects markets by volume, so
   we hold a volume-selected *subset* of each mutually-exclusive family — 28 of the
   World Series field, 25 "lead the NBA in scoring" candidates. Summing a subset
   gives sums like 0.12 or 0.23 that look like enormous arbitrage and are pure
   artifact. Median across 35 measurable events: 0.231.
2. **Prints are not quotes.** Requiring every leg to have traded within the hour,
   all legs are simultaneously live at only **1.6%** of hourly grid points.

Both are fixed by the same thing and only by that thing: a **live CLOB `/book`
snapshot across a complete outcome set** (public GET, no keys). That is a small,
genuinely read-only capture and it is the one live thread this gate leaves. It is
*not* evidence of an arb — it is an unanswered question.

#3 (stale-price / news-lag) was not run: it is blocked in kind, needing an external
news clock to establish that the market was slow rather than correctly unmoved.

---

## 7. What would change the verdict

Honestly stated, since the prior was low and the result confirms it:

- **More tail events.** The binding constraint is 179 resolution events with 4 tail
  realizations. Widening the census (`discover.py` is read-only and can enumerate
  far more markets) would tighten the fair-price null. But note the direction: the
  point estimate is +0.34%, and the null says that is ordinary. More data is as
  likely to erase it as confirm it.
- **The multi-outcome residual**, if a live book capture shows genuine coherent
  mispricing across complete outcome sets rather than the incomplete-set artifact
  measured here.

Neither is established by anything measured here.

---

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/audit_flb_selection.py       # G0
PYTHONPATH=. .venv/bin/python scripts/audit_flb_favorite_band.py   # G0b: cluster unit, tail count, settlement dust
PYTHONPATH=. .venv/bin/python scripts/audit_flb_tradeability.py    # G1 / G2a / G2b / G3
```

Outputs land in `data/interim/value/` (gitignored). Nothing in `data/processed/`
was touched; the Project 1/3 frozen artifacts and both forward scoreboards are
untouched.
