# Project 3 — The Slow-Market Pivot

**Status:** scoping + design spec, **plus the stage-1 gradient test, which has now RUN — see §10.
Result: flat.** Otherwise nothing built. Read-only in spirit, same discipline as the rest of the
repo. **Every number marked "measured" was measured 2026-07-22/23 against the real 4,695,081-row
ledger or live read-only API probes; repro commands in §9.**

**Prereqs:** `CLAUDE.md`, `HANDOFF.md` (esp. the 2026-07-22 edge-decay verdict),
`DECISIONS.md`, `docs/project2_forecaster_discovery.md`.

---

## 0. The reframe

The project has been unintentionally studying **fast** markets, and it was never a decision — it is
an artifact of the data source. `ingest.py` polls Polymarket's *global* `/trades` firehose, that
firehose is dominated by 5-minute BTC/ETH/HYPE "up or down" coin-flips, and so that is what the
ledger became:

| | markets | resolved BUY bets | share |
|---|---|---|---|
| micro-crypto (5-min up/down) | 229,092 | **3,335,883** | **81.96%** |
| everything else | 119,565 | **734,208** | **18.04%** |
| **total** | 348,657 | 4,070,091 | 100% |

*(measured; confirms the 82/18 framing exactly)*

That slice is quant/latency territory — sub-second market-making against 5-minute coin-flips. We
cannot compete there and we do not want to. Three years of this project's findings are, in
hindsight, all findings *about that slice*:

- the sharp wallets we find are ~all `high_frequency_micro_market`-flagged bots (28 of 36 persisters);
- their edges are 1–7¢/bet — real, ironclad (p to 1e-26), and economically negligible;
- and the 2026-07-22 edge-decay run showed **no copyable edge at any latency**, with wallet-agnostic
  placebos beating the wallet at 30s, 15m and 1h.

**The new thesis:** a genuine *information* edge can persist only where the market is too small and
slow to attract algorithmic competition. Obscurity is not a defect of such a market — it is the
mechanism. So the universe becomes **slow markets**, defined by speed, and the wallets we hunt are
forecasters, not scalpers.

### What this pivot does and does not overturn

The 2026-07-22 "no copyable edge at any latency" verdict was measured on a ledger that is a
**backfill of wallets selected for micro-crypto performance**. Its real-world slice is "real-world
bets that micro bots happened to place." That verdict must be **re-run on the new universe, and it
must not be assumed to carry over — nor assumed to be overturned.** It is an open question again,
on different data, and it stays open until re-measured with the same placebo controls.

---

## 1. Defining the slow-market universe

### 1.1 The three obvious definitions all fail — measured, not assumed

**(a) Topic keywords fail** (and the user's constraint here is correct on the merits). The current
`discover.is_micro_crypto()` keys on `"updown" in slug`. Live probe found
`spacex-ipo-closing-price-updown…`: **startDate 2026-06-09 → closedTime 2026-06-30, a 21-day
market**, classified `micro_crypto` today and therefore excluded from every real-world analysis.
Symmetrically, nothing in a keyword rule stops a 5-minute politics market from being admitted.
Topic is not speed.

**(b) Market metadata "lifespan" fails, in a way that would have silently poisoned everything.**
CLOB `/markets/{cid}` carries `accepting_order_timestamp` and `end_date_iso`, and both are traps:

- For micro-crypto, `accepting_order_timestamp` is **exactly −24h from the resolution epoch, every
  time** (measured across 8 sampled micro markets: −85,921 to −86,001 s). It is a series-creation
  artifact carrying literally zero information about the 5-minute trading window.
- `end_date_iso` is the **midnight floor of the resolution day** (measured: `end_ts − resolution
  epoch` ranged −6,900 to −85,200 s, i.e. always the preceding 00:00Z). It can therefore land
  *before* the market's own last trade.
- Net effect: sampled micro markets report a **median metadata lifespan of 13.5 h** (p10 5.0 h,
  p90 20.7 h) for markets that genuinely trade for 5 minutes. **A naive "lifespan ≥ 6h" filter would
  re-admit ~75% of the micro-crypto slice.** This is already half-known — `is_micro_crypto`'s
  docstring warns "those markets report a ~24h start→end window" — but the consequence for a
  speed-based universe definition was not.

**(c) Observed tape span fails on its own**, because our coverage is not uniform. Per-market observed
`t_max − t_min` in the ledger:

| | n markets | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| micro | 229,092 | 0.000 h | **0.035 h (2.1 min)** | 0.17 h | 0.90 h |
| real-world | 119,565 | 0.000 h | **0.000 h** | 25.7 h | 319 h |

The real-world median is 0.000 h because **50.8% of real-world markets have exactly one trade in
our ledger** — we see only the tracked wallets' fills, not the tape. Observed span is accurate for
micro (dense coverage) and near-useless for the real-world tail (sparse coverage).

### 1.2 The definition: three timestamp-derived axes, no keywords

**The ground truth we were missing is `closedTime`.** Gamma's `/markets` listing returns it, it is
a **precise** resolution timestamp (`"2026-06-30 22:15:35+00"`, not day-truncated), and it was
populated on **100/100** markets in a closed-listing page (measured). Combined with `startDate`,
that gives true lifespan without touching `accepting_order_timestamp`.

Three axes, all derived from timestamps and money, none from topic:

| axis | definition | source | what it captures |
|---|---|---|---|
| **L — lifespan** | `closedTime − startDate` | Gamma listing | how long the market existed to be reasoned about |
| **H — bet horizon** | `closedTime − entry_ts`, per bet | Gamma + ledger | was this a *forecast* or a near-resolution scalp |
| **V — velocity** | `volumeNum / L`, and trades/hour from the tape | Gamma + ledger | competition density — the actual operationalisation of "too small and slow to attract algorithms" |

**H is the operationally binding one** and deserves emphasis: L is a property of the market, but a
slow market still gets scalped in its final hour. Project 2 §1.2 already found near-resolution tape
bias inflating apparent skill **10–50×**. Gating on H is how that is killed at the source rather
than patched downstream.

**V is the axis the thesis actually rests on.** "Slow" is a proxy; "uncontested" is the mechanism.
A $109M World Cup market has a 9-day lifespan *and* heavy algorithmic participation — long-lived
but not obscure. V separates them, and it is free from data we already fetch.

**Soundness rule for combining sources.** The two speed sources have complementary failure modes,
so combine them as *lower bounds*, never as a single authority:

```
slow(market) := (Gamma lifespan L ≥ H_thresh)          # precise, but incomplete coverage
             OR (observed tape span ≥ H_thresh)         # complete coverage, lower bound only
unknown_speed := neither, and no Gamma record            # parked, NOT dropped; retried next enumeration
```

Both branches are sound for *inclusion* (each proves the market lived at least that long). Neither
is sound for *exclusion*, which is why the third bucket exists and why nothing is deleted from it.

### 1.3 Getting the metadata: date-sliced Gamma enumeration (mechanically works; coverage is NOT sufficient)

> 🔴 **Superseded in part by a 2026-07-23 measurement.** Everything below about *mechanics* holds —
> range filters bind, a day-slice fits under the offset cap, `closedTime` is precise. But the
> **coverage is only 41.7% of the ledger's real-world markets**, and widening the window makes it
> worse. See the first row of §6. The practical consequence: **CLOB per-market GET is the primary
> metadata path** (path-keyed, 100% coverage, ~23 req/s — 39,941 markets in 30 min, which is what
> §10 actually used), with Gamma enumeration as the source of `closedTime`, volume and the universe
> of markets we hold *no* bets in. Read this section as "how to enumerate", not "how to get
> metadata for markets we already have".

`DECISIONS.md` records that Gamma's query filters "don't work as documented." That finding is
correct **for identity filters** (`conditionId`, `condition_ids`, `slug` — all re-confirmed broken
or ignored today). **Range filters are different and they do work** (measured):

| filter | result |
|---|---|
| `end_date_min` / `end_date_max` | ✅ binds correctly (a `[2026-07-01, 2026-07-02)` slice returned only that day) |
| `start_date_min` | ✅ binds |
| `volume_num_min` | ✅ binds (lowest volume in page = 1,000,170 at `volume_num_min=1000000`) |
| `condition_ids` (any form) | ❌ returns 0 rows |
| `condition_id` singular | ❌ silently ignored → unrelated default listing |
| offset cap | **10,000 on `/markets`, ~5,000 on `/events`** (422 beyond) |

The offset cap kills volume-ranked full enumeration — but **date-window slicing sidesteps it
entirely**, because a one-day slice fits comfortably underneath:

- `closed=true, endDate ∈ [2026-07-01, 2026-07-02)` → **2,100 unique markets in 21 pages, not capped.**
- Full historical enumeration ≈ 21 pages/day × ~742 days ≈ **~15,600 requests, ~50 min one-time** at a
  polite 5 req/s, then a handful of pages nightly. Each row carries `conditionId`, `startDate`,
  `endDate`, **`closedTime`**, `volumeNum`, `liquidityNum`, `clobTokenIds`, `events`, `umaEndDate`,
  `gameStartTime` — so the whole metadata table is a byproduct of enumeration, with **zero
  per-market lookups**.

⚠️ **One coverage caveat, measured and unexplained.** That 2,100-market day slice contained **1**
micro-shaped market. Gamma's listing appears to exclude the recurring/hidden micro series (the CLOB
`tags` on those markets include `'Recurring'`, `'Hide From New'`). This is *convenient* — Gamma
enumeration natively inverts the 95%-crypto firehose problem — but it means **enumeration coverage
must be verified rather than assumed**, and the metadata table will have holes exactly where the
fast markets are. That is fine under the §1.2 soundness rule (absence from a slow enumeration never
proves fast; the observed-tape branch and the "unknown" bucket absorb it), but it must not be
quietly relied on.

**Cheap alternative worth knowing:** `update_resolutions` already GETs `clob.polymarket.com/markets/{cid}`
for every market, and that response already contains `end_date_iso`, `game_start_time`,
`accepting_order_timestamp` and `tags` — all currently parsed and thrown away. Persisting them costs
**zero extra requests** and gives `game_start_time` (present on **93.4%** of bet-weighted real-world
markets — a precise, event-anchored timestamp Gamma's `closedTime` complements). Do both: Gamma
enumeration for the universe, CLOB field capture as the free cross-check.

### 1.4 Thresholds — and testing the gradient instead of guessing the cliff

Bet-weighted lifespan distribution of the real-world slice (PPS sample of 500 markets drawn
proportional to resolved-BUY count, so unweighted sample stats estimate bet-weighted population
stats; ±4pp at 95%):

| lifespan ≥ | share of real-world bets | ≈ share of whole ledger |
|---|---|---|
| 1 h | 98.2% | 17.7% |
| 6 h | 95.6% | 17.2% |
| **1 d** | **77.2%** | **13.9%** |
| **3 d** | **44.2%** | **8.0%** |
| **7 d** | **24.2%** | **4.4%** |
| 30 d | 6.4% | 1.2% |

**Do not pick one threshold. Stratify and measure the gradient — it is the cheapest evidence we can
get about the pivot.** The thesis makes a directional prediction: *skill edge and its out-of-sample
persistence should rise as L rises and V falls.* Bucket the existing 734,208 real-world bets by
(L, V) and plot it. If the gradient is there, the pivot is earned and the thresholds fall out of
where it turns on. If it is flat, that is **ambiguous rather than falsifying** — this ledger's
population is selected under the old micro regime, so the wallets the thesis is about are mostly not
in it (§4 stage 1). Either way it costs hours and runs before any ingest work.

Working defaults until that measurement lands: **universe L ≥ 24 h, bet gate H ≥ 6 h**, with a
"deep-slow" tier at L ≥ 7 d for the core thesis. All config-tunable, none load-bearing.

---

## 2. Partition the ledger, or filter at read time?

### 2.1 What scoping actually buys — measured

Same seven columns (`wallet, market_id, side, resolved, timestamp, entry_price, resolved_value`),
deep pandas footprint over the real ledger:

| | rows | in-memory | ratio |
|---|---|---|---|
| whole ledger | 4,070,091 resolved BUY | **0.75 GB** | 1.0× |
| real-world slice | 734,208 | **0.12 GB** | **6.1× smaller** |
| L ≥ 7d tier | ~178k (est.) | ~0.03 GB | ~25× smaller |

Against the current VM baseline (e2-micro, 1 GB RAM, post-streaming-rewrite: features 773 MB /
26 min, validate+rank+report 28 min, nightly recompute ≈ 54 min; ingest is separate and already
down to ~48 s/tick since the delta-write fix):

**Estimated** scoped nightly recompute: **~10 min, well under 250 MB peak.** The 6.1× is measured;
the runtime projection is an estimate and must be confirmed by a real run.

**The payoff that matters is not the nightly job — it is the un-deferring of blocked work.**
Metric B (copyability) has been deferred since 2026-07-21 because its ~2.9 GB peak OOM'd the 3.7 GB
Chromebook *twice*. At 6× smaller that is ~500 MB — it runs on the free VM. Same for the
`audit_edge_decay_long` placebo battery, per-category price baselines, and per-wallet bootstrap CIs.
Scoping does not merely make the pipeline comfortable; **it converts "too expensive to measure" into
"routine," which is where the actual analytical debt is.**

### 2.2 Recommendation: one ledger, a metadata sidecar, and a persisted speed column

**Not** a physical split (it fights the atomicity discipline that cost an incident to get right —
`atomic_to_parquet` + single-file `os.replace`), and **not** a naive read-time `market_id in {120k
ids}` predicate (slow, and re-derived every run).

1. **`data/interim/market_meta.parquet`** — the sidecar: `market_id, start_ts, closed_ts, lifespan_s,
   volume, liquidity, velocity, game_start_ts, tags, speed_source ∈ {gamma, tape, unknown},
   first_seen, last_seen`. ~350k rows × ~12 cols ≈ 25–40 MB. Built by §1.3 enumeration, refreshed
   nightly, **never** deletes a row.
2. **Additive `speed_bucket` column on the ledger** (dictionary-encoded: `fast | slow | deep_slow |
   unknown`), populated from the sidecar during the nightly `--fold-delta` step, which already
   rewrites the ledger under the writer lock. Backward-compatible: pre-existing rows get `unknown`
   and are filled on the next fold, exactly as the `category` column was designed to work.
3. **Sort row groups by `speed_bucket`** so pyarrow's row-group statistics prune the fast rows at the
   *file* level — `load_ledger(columns=…, filters=[("speed_bucket","==","slow")])` then never reads
   those bytes. ⚠️ **Verify the pruning actually fires** (compare bytes-read / peak RSS with and
   without); parquet predicate pushdown is famously conditional. If it does not fire, fall back to a
   Hive-partitioned directory — and pay the atomicity cost deliberately, with a written procedure.

This satisfies the "nothing gets deleted" constraint literally: micro-crypto stays on disk, stays
ingested, stays scored, and remains addressable by flipping one filter. **We scope what we compute
over, not what we keep.**

### 2.3 Micro-crypto's two jobs, which is why it must stay

Not sentiment — it is load-bearing twice over:

- **Price baseline.** The favorite-longshot calibration curve `E[outcome | entry_price]` is what makes
  skill edge *skill*. Project 2 §2.0 already established it must be fit **per category** — a 0.75 NFL
  favorite is not a 0.75 crypto print. Micro is the largest, densest, best-calibrated curve we have,
  and it is the reference against which the slow curves are read.
- **Null control.** 3.3M bets of known-structural, known-un-copyable edge is the ideal negative
  control for every new metric. Any slow-market finding that also "fires" on micro is measuring
  plumbing, not skill. This is the cheapest guard we own and the pivot makes it *more* valuable.

---

## 3. Should ingest shift to market-first discovery?

### 3.1 The two corpora are broken in exactly opposite ways — measured

| | main ledger (wallet-first backfill) | discovery corpus (market-first, `discovery_trades.parquet`) |
|---|---|---|
| real-world resolved BUY bets | 734,208 | 1,685,995 |
| distinct wallets | **3,192** | **463,561** |
| distinct markets | 119,565 | **573** |
| median bets/wallet | 2 | 1 |
| wallets with ≥20 bets | 310 | 5,361 |
| median breadth of those | **95 markets** | **5 markets** |
| overlap between the two | — | 1,259 wallets |

**Main ledger:** deep per wallet (up to 14,942 real-world bets), but only 3,192 wallets — and those
wallets exist in it *because they were selected for micro-crypto performance*. It is a deep sample of
the wrong population.

**Discovery corpus:** 463k real-world wallets — the right population — but each is 1 bet deep across
573 markets. That concentration is precisely what killed Project 2 §1.5: both BH-FDR survivors turned
out to be single-event artifacts (one Sunday's 5 NFL games; 3 props of one Fed presser).

**Neither corpus can answer the question alone, and the missing piece is a chain that was specified
and never built** — §1.5 of the Project 2 spec: *feed discovered wallets into `backfill.py`'s
per-user deepening.* `/trades?user=` returns a wallet's near-complete cross-market history under the
same 10k cap. That is what converts "463k shallow real-world wallets" into "N deep real-world
forecasters," and it is the single highest-leverage unbuilt thing in the repo.

### 3.2 Recommendation: keep both, with different jobs

- **Keep `ingest.py` on the global firehose, unchanged.** It costs ~48 s per 5-min tick post-fix, it
  is the price-baseline and null-control feed (§2.3), and per constraint 2 it keeps running.
- **Promote `discover.py` from Project-2 side-experiment to the primary discovery path**, with two
  changes: (a) replace volume-ranked enumeration (10k-offset-capped, incomplete) with **date-sliced
  enumeration** (§1.3 — complete, verified); (b) replace its selection predicates
  (`in_volume_band`, keyword `skip_micro`) with the **speed/velocity gates from §1.2**. Both are
  small, contained edits to existing, tested code.
- **Build the discovery → backfill chain** (§3.1). This is the work item, not the enumeration.
- **Retarget `backfill.select_wallets`**: today it deepens top-500 *by current rank*, which is a
  ranking of micro-crypto bots. Under the pivot it must seed from the slow-market screen (§4, stage 1).

### 3.3 The unlock this hands back to Project 2

Project 2 §1.5 closed with a named blocker: *"the concentration guard misses correlated markets
sharing one resolution event — the discovery resolutions table has no timestamp, so resolution-time
clustering is the next guard."* **`closedTime` is that timestamp**, and §1.3 gets it in bulk. The
guard that was blocked is now buildable, and it is exactly the guard that would have caught both
§1.5 false positives directly rather than by inspection.

---

## 4. The funnel

Crude and wide at the top, expensive and narrow at the bottom, and — the important structural
property — **selection and validation run on disjoint data, not on disjoint halves of the same thin
data.**

### Stage 0 — Universe (cheap, no new ingest)
Build `market_meta.parquet` (§1.3). Tag the ledger with `speed_bucket` (§2.2). Nothing scored yet.

### Stage 1 — Gradient test: **one-way evidence**, not go/no-go (cheap, existing data)
Bucket the existing 734,208 real-world bets by (L, V) and measure skill edge + OOS persistence per
bucket, with the micro slice as the null control. Cost: hours. It runs before any new ingest is built.

**Read it asymmetrically, because the population is adversely selected.** The deep real-world bets in
this ledger belong to the top-500 wallets, and that ranking was produced under the old 82%-micro
regime — **28 of the 36 persisters carry `pattern_flag=high_frequency_micro_market`**. So nearly all
of this test's statistical power comes from micro-specialists trading real-world markets on the side.
The slow-market specialists the thesis is actually about are **largely absent from this ledger by
construction**: only 3,192 wallets have any real-world bet at all, and the market-first corpus that
does contain them is 1 bet deep per wallet (§3.1).

| result | reading |
|---|---|
| **rising gradient** | **strong evidence.** Edge appearing *despite* adverse selection — the population least likely to show it, showing it anyway. |
| **flat gradient** | **ambiguous, not falsification.** Consistent with "no slow-market edge" *and* with "the wallets that have it aren't in this sample." Cannot distinguish them. |

A flat result therefore does **not** stop the pivot. It routes to a cheap version of build steps 1–2
(metadata table + `speed_bucket`, no new ingest yet), then a re-test on a less biased population once
the discovery→backfill chain (stage 3) has deepened actual slow-market wallets. The measurement to
distrust is the one taken on a sample selected by the regime we are leaving.

### Stage 2 — Crude wide screen (skill edge only)
Over **every** wallet with ≥1 bet in the slow universe, across both corpora, compute the
**shrunk per-(wallet, category) residual skill edge**:

```
screen_score = n/(n+k) · mean(outcome − E_cat[outcome | entry_price])
```

- **Skill edge, never win rate, never profit** — hard constraint, and the shrinkage is what makes it
  usable at n=2 instead of letting one lucky bet top the table.
- `E_cat[·]` is the **per-category** favorite-longshot baseline (Project 2 §2.0), fit inside the slow
  universe, not inherited from the crypto-dominated global curve.
- Output is a **shortlist, not a finding.** It is never published, never ranked for a human, never
  written to `data/processed/`. Screening-stage numbers are selection noise by construction — that
  is the whole lesson of §1.5's ~40% estimated gate FDR.

### Stage 3 — Deepen the shortlist (the expensive step, and the only one that spends requests)
Per-user `/trades?user=` backfill for shortlisted wallets only. **This creates data the screen never
saw**, which is what makes stage 4 an honest test rather than a re-measurement of the screen's noise.

### Stage 4 — Validate on the deep data
The existing A/C/D stack, unchanged in shape: chronological OOS split, one-sided significance,
economic-magnitude floor, regime detection (metric D), `manufactured_record_flag` / `pattern_flag`
as metadata only. Plus, all of which already exist and simply get pointed at the new universe:
- shuffled-outcome null within (category, 1¢-price) + BH-FDR (`audit_forecaster_null.py`);
- concentration guard (`eff_breadth = 1/HHI ≥ 3`, distinct decision-days ≥ 3);
- **new: resolution-time clustering guard**, now buildable via `closedTime` (§3.3).

### Stage 5 — Copyability at *slow* latencies
Metric B at Δ = 1 h / 6 h / 24 h — the latencies a human actually operates at — with the full placebo
battery from `audit_edge_decay_long.py` (random-anchor, pre-entry anchor, cluster bootstrap by
market/family/day). Affordable now (§2.1). **The 2026-07-22 verdict is re-opened here, not assumed.**

⚠️ **Name the tension honestly: the slow thesis and the copy thesis pull against each other.** Low
velocity means thin tape means fewer follow-on prints to enter against — and 75% of real-world
positions already had *no* follow-on liquidity on the deep ledger. The resolution, if there is one,
is that in a slow market you are not copying a *fill*; you are receiving a *signal you have hours or
days to act on*, and the relevant question becomes "is the price still near their entry at Δ=6h?"
rather than "can I get filled at Δ=30s." That is measurable, and it is measured here.

### Stage 6 — Prospective freeze → forward test
Freeze the surviving shortlist with a timestamp, then score **only** on bets resolving after the
freeze. Per HANDOFF, the forward test is the only arbiter left; the freeze is what makes it one.

---

## 5. The small-sample trap — the honest part

The user is right that this is the trap that already collapsed a top-30 list (median held-out skill
edge **0.093 → 0.0007** when thin samples were deepened). Slow markets make it structurally worse:
fewer bets per wallet, by definition. Here is the size of the problem, measured, not hand-waved.

### 5.1 The power calculation

Residual (skill) edge in the real-world slice: mean −0.00000, **SD = 0.3758** (n=734,208, 20 quantile
price bins). Bets needed for 80% power at one-sided α=0.05:

| per-bet skill edge | bets needed **in the held-out half** | ≈ total |
|---|---|---|
| 1¢ | 8,733 | 17,465 |
| 2¢ | 2,183 | 4,366 |
| 3¢ | 970 | 1,941 |
| 5¢ | 349 | 699 |
| **10¢** | **87** | **175** |
| **20¢** | **22** | **44** |

A forecaster betting twice a month makes 24 bets/year. At the current `min_skill_edge = 0.02` floor
they would need **~180 years** of history. That is not a tuning problem; it is arithmetic.

### 5.2 Which forces a design conclusion, and it is a good one

**The slow universe is only viable if we hunt large edges, and that is exactly what the thesis
predicts exists there.** Uncontested markets should be mispriced by *cents-to-tens-of-cents*, not by
the 1–7¢ that survives in a market with algorithms in it. A 10¢ edge is detectable at n=87 — well
inside a real forecaster's lifetime record. So:

> **A ~10¢ magnitude floor for the slow universe** — versus the current 2¢. Not a tuned parameter but
> a joint statement about economics *and* detectability: below ~10¢ we cannot tell a slow-market
> forecaster from noise within a human career, so a lower floor produces unfalsifiable claims, not
> discoveries.

⚠️ **This is a new, scoped knob — it must NOT change `scoring.min_skill_edge`.** That global 2¢ floor
is load-bearing for Project 1: raising it to 10¢ would silently rewrite the existing results and
collapse most of the current 36-wallet persisted set, whose edges are 1–7¢/bet by design (the deep
micro persisters are *supposed* to be small-edge/large-n). Project 1's universe and Project 3's have
opposite statistical shapes and need opposite floors. Implement as a separate key —
`scoring.slow.min_skill_edge`, applied only on the slow-universe path — with the global default
untouched and Project 1's outputs bit-unchanged by the addition.

This also inverts a familiar complaint. Large-n micro wallets get certified at p=1e-26 on 0.5¢ edges
(statistically ironclad, economically meaningless); slow wallets will be the opposite — economically
large, statistically marginal. **Both gates must stay separate and both must be reported**, exactly as
metric C established.

### 5.3 Four defenses, in order of strength

1. **Disjoint selection and validation data (§4, stages 2→3).** The strongest and most under-used.
   The screen runs on shallow data; validation runs on *newly fetched* deep data. Selection noise in
   the screen cannot survive into a fresh sample. This is strictly stronger than the current
   chronological half-split, which selects and validates on two halves of the *same* thin history.
2. **Prospective freeze (§4, stage 6).** Eliminates selection bias outright. Costs calendar time,
   nothing else.
3. **Partial pooling instead of per-wallet isolation.** Hierarchical shrinkage to a per-category
   prior (already the screen statistic), plus testing *groups* — "wallets entering ≥7 days before
   resolution beat the price by X" is detectable at group n even when no individual is. Group
   findings are also more robust and more directly actionable than a wallet list.
4. **Exploit the variance structure.** Residual SD is strongly price-dependent, so cheap validation
   lives at the extremes:

   | entry price band | n | residual SD | bets needed for a 3¢ edge |
   |---|---|---|---|
   | [0.0, 0.1) | 53,241 | 0.2272 | **355** |
   | [0.1, 0.3) | 92,586 | 0.3977 | 1,087 |
   | [0.3, 0.5) | 128,605 | 0.4893 | 1,645 |
   | [0.5, 0.7) | 142,237 | 0.4860 | 1,623 |
   | [0.7, 0.9) | 119,710 | 0.3921 | 1,057 |
   | [0.9, 1.0) | 197,829 | 0.1650 | **187** |

   Longshot and heavy-favorite specialists are **4–9× cheaper to validate** than mid-price traders.
   This connects directly to the already-investigated black-swan/tail metric family
   (`audit_blackswan.py`), whose split-half forward validity (+0.218) already beat the mean-edge
   pipeline's (+0.105). In a slow universe that family may stop being a curiosity and start being
   the main event.

### 5.4 What we will *not* be able to answer, stated up front

Some genuinely skilled low-frequency forecasters are **permanently un-validatable** by this
machinery. A wallet with 40 lifetime bets and a true 4¢ edge cannot be distinguished from noise — not
with better ingest, not with better statistics. The honest response is to report them with their CI
and their power, in an explicitly-labelled "insufficient evidence" bucket, and never to quietly relax
a threshold to make the table look fuller. That failure mode — relaxing gates until winners appear —
is precisely what §1.5's shuffled null was built to catch, and it must stay pointed at us.

---

## 6. Risks and open questions

| risk | status |
|---|---|
| 🔴 **Gamma date-sliced enumeration covers only 41.7% of the ledger's real-world markets** (measured 2026-07-23: 90/216 across 8 day-slices). Misses are the recurring/hidden series — `highest-temperature-in-*`, esports, hourly crypto levels, even World Cup exact-score props. **Widening the window made it WORSE** (0/47 over a 7-day window that a 1-day slice hit 38% of), so the range filter also misbehaves at width | **open, and bigger than §1.3 assumed** — build step 1 needs a coverage *fix*, not just an implementation. §1.2's soundness rule contains the correctness risk (absence never proves fast), but a metadata table covering 42% of markets cannot define the universe on its own. CLOB per-market GET is the 100%-coverage fallback (path-keyed, no filter ambiguity) at ~23 req/s — the stage-1 run used exactly this for 39,941 markets in 30 min |
| `closedTime` semantics on multi-outcome / neg-risk markets | **unverified** — spot-check before trusting L. Note `end_date_iso` is *already known bad*: midnight-floored, and it put 87% of bets at a negative horizon (§10.7) |
| Predicate pushdown may not actually prune row groups | **must measure** (§2.2), Hive partition is the fallback |
| Slow ⇒ thin ⇒ un-copyable; the thesis may be true and unactionable | **acknowledged** (§4 stage 5) — measured, not assumed away |
| Discovery corpus concentration (573 markets) biases the screen | **contained** by stage-3 deepening + the resolution-time clustering guard |
| `/trades?user=` 10k cap may truncate high-volume discovered wallets | known; `backfill.py` already degrades gracefully (keeps newest ~9k on a 408) |
| **The gradient test is run on an adversely-selected population** — its power comes from micro-specialists (28/36 persisters are `high_frequency_micro_market`), so slow-market specialists are largely absent by construction | **structural, unfixable at stage 1.** Makes the test **one-way**: a rising gradient is strong evidence, a flat one is ambiguous and does not falsify. Flat → proceed to cheap build steps 1–2 and re-test after stage-3 deepening (§4 stage 1) |
| The premise may simply be wrong (no slow-market edge exists) | **cannot be settled on this ledger.** Only a re-test on a population not selected under the micro regime can distinguish it from the sampling artifact above. Stage 1 came back flat (§10) — consistent with both readings, as designed |
| 🔴 **The bet-level shuffled-outcome null used since Project 2 §1.5 is too permissive for wallet-level statistics** — it destroys market clustering, so correlation through a shared resolution event reads as skill. Measured: 8/12 cells "significant" against it vs **3/12** against a cluster-preserving null; variance reference 1.0 vs 1.75–5.78 | **found at stage 1, fix available now.** `permute_wallets_within_market` in `audit_speed_gradient.py` is the cluster-preserving null; it should become **standard for every wallet-level statistic in this repo**, and Project 2 §1.5's FDR estimate should be recomputed against it. **QUALIFIED 2026-07-23** (`docs/blackswan_cluster_null.md`): it is not a universal drop-in. In sparse strata (the ≤15¢ tail: median 2 tail bets/market) holding each wallet's per-market count fixed leaves its per-wallet reference **more constrained than independence** — design effect 0.17 vs 1.45 for a market-block bootstrap — so it manufactures its own per-wallet false positives. Its *count-level* comparison stays valid (same procedure both sides). The rule to adopt repo-wide is **measure the design effect of your null**, and use a market-block bootstrap for per-wallet arbitration where D<1 |

---

## 7. Build order

Each step commits separately, read-only, per the CLAUDE.md git convention.

1. **`market_meta.parquet` + date-sliced Gamma enumeration** — the metadata table (§1.3). Plus the
   free CLOB field capture in `update_resolutions`. Tests on a hand-built fixture with a known
   lifespan and a planted micro-metadata trap.
2. **`speed_bucket` on the ledger + pruning verification** (§2.2). Byte-identity harness against the
   existing baseline, as every ledger-writer change has been.
3. ~~**Stage-1 gradient test**~~ — **DONE 2026-07-23, `scripts/audit_speed_gradient.py`. Result: FLAT
   (§10).** Ran ahead of steps 1–2 using CLOB per-market metadata rather than the (42%-coverage)
   Gamma path. As designed, a flat result routes back through steps 1–2 for a re-test on a less
   biased population and stops nothing. Two spin-off items it created, both above in §6: the
   Gamma-coverage fix, and adopting the cluster-preserving null repo-wide.
4. **Per-category price baseline** inside the slow universe (Project 2 §2.0 — specified, never built).
5. **Screen + discovery→backfill chain** (§4 stages 2–3; Project 2 §1.5 — specified, never built).
6. **Point the A/C/D stack + null/FDR + concentration guards at the slow universe** (stage 4), and add
   the resolution-time clustering guard (§3.3).
7. **Metric B at slow latencies with the placebo battery** (stage 5) — un-deferred by §2.1.
8. **Prospective freeze + forward test** (stage 6).

---

## 8. Invariants preserved

- **Nothing is deleted, down-weighted, or dropped.** Micro-crypto keeps being ingested, keeps being
  scored, keeps its row in every table. `speed_bucket` is an additive column and a *view*; both
  partitions stay on disk. We scope what we compute over, not what we keep.
- **Skill edge, never win rate, never profit** — including at the top of the funnel, where the
  temptation is greatest and where a raw-mean screen at n=2 would repopulate the list with zero-skill
  wallets. Shrinkage, not a different metric.
- **Magnitude and significance stay separate gates**, both reported.
- **Flags stay metadata** and never touch score or rank order.
- **Read-only.** No keys, no order placement.

---

## 9. Measurements — repro

All read-only. Probe scripts live in this session's scratchpad; fold the keepers into `scripts/` when
step 1 is built.

| § | claim | how |
|---|---|---|
| 0, 1.1 | 82/18 split; per-category counts; observed-span distributions | batched `iter_batches` rollup over `bet_ledger.parquet` (~20 s, <1 GB) |
| 1.1 | micro `accepting_order_timestamp` = −24h constant; `end_date_iso` = midnight floor | `clob.polymarket.com/markets/{cid}` on 8 sampled micro markets vs. the slug's resolution epoch |
| 1.1, 1.4 | bet-weighted lifespan distribution | PPS sample of 500 real-world markets (∝ resolved-BUY count) → CLOB metadata |
| 1.3 | Gamma range filters bind; offset caps; 2,100 markets/day; `closedTime` populated 100/100 | live `gamma-api.polymarket.com/markets` probes |
| 2.1 | 0.75 GB → 0.12 GB (6.1×) | deep `memory_usage` over the same 7 columns, batched |
| 3.1 | two-corpus comparison | rollups over `bet_ledger.parquet` and `discovery/discovery_trades.parquet` |
| 5.1, 5.4 | residual SD 0.3758; power table; per-band SD | 20-quantile price baseline over the 734,208 real-world resolved BUY bets |
| 1.3, 6 | Gamma day-slice coverage of ledger markets = 41.7% (90/216); widening to 7 days → 0/47 | 8 day-slices enumerated live, cross-checked against sampled ledger markets' CLOB end dates |
| 10 | the whole stage-1 gradient result | `PYTHONPATH=. .venv/bin/python scripts/audit_speed_gradient.py --meta <clob_meta.parquet> --shuffles 200` (~22 min). Metadata: CLOB per-market GET for 39,941 markets, ~23 req/s, 30 min |
| 10 | the statistics themselves are validated on planted fixtures | no-skill → ρ=+0.06/varR=1.09 (ns); planted skill_sd=0.06 → ρ=+0.29/varR=1.86 (p=0.005). Detection floor ≈ skill_sd 0.02–0.04 at 300 wallets × 60 bets |

---

## 10. Stage-1 result — the speed gradient (measured 2026-07-23)

**Script:** `scripts/audit_speed_gradient.py` (read-only). Market metadata from
`clob.polymarket.com/markets/{cid}` for the 39,941 real-world markets with ≥3 observed trades;
39,650 had usable timestamps, covering **651,801 bets = 88.8% of the 734,208-bet real-world slice**.
200 shuffles per null.

**VERDICT: the gradient is FLAT. Skill does not strengthen as markets get slower in this ledger** —
not across market lifespan, not across bet horizon, and not within individual wallets. Once market
clustering is divided out, clustering-adjusted skill dispersion is **the same (1.49–2.59) in every
speed bucket and in 5-minute micro-crypto (1.66)** (§10.4). The within-wallet paired test, which
controls for wallet identity, is null in all three comparisons (§10.6).

Per §4 stage 1 this is **ambiguous, not falsifying** — the population is selected under the old
micro regime — so it routes to build steps 1–2 and a re-test, and stops nothing. What it *does*
establish is narrower and solid: **micro-specialists do not become sharper in slow markets.**

### 10.1 The methodological finding comes first, because it changes how to read everything else

Against the **bet-level** shuffled-outcome null (permute outcomes within 1¢ price bins — the null
this repo has used since Project 2 §1.5), the result looks strong: **8 of 12 cells** have a
significant split-half persistence ρ, and **12 of 12** have excess cross-wallet variance at the
p-floor.

Against the **cluster-preserving** null added for this run (permute *which wallet* made each bet,
within each market, leaving every market's prices and outcomes exactly intact), **only 3 of 12
survive.** The bet-level null's variance reference sits at ~1.0 while the cluster-preserving null's
sits at **1.75–5.78** — i.e. most of the apparent "excess" is bets sharing a market, and therefore a
single resolution event, not wallets differing in skill.

> This is the §1.5 trap in a new place. Had this run used only the established null it would have
> reported a strong, monotone-looking result. **The cluster-preserving null should become standard
> for every wallet-level statistic in this repo, not a one-off for this audit.**

### 10.2 Axis L — market lifespan (open → close)

| L | bets | wallets | eff. breadth | ρ | depth | ρ null (bet) | ρ null (cluster) | p (cluster) | LOMO ρ ≥ | varR | varR null (cluster) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <6h | 9,041 | 210 | 86.7 | +0.321 | 93 | +0.004 | +0.241 | 0.264 | +0.243 | 8.88 | 4.99 |
| 6–24h | 49,079 | 468 | 305.2 | +0.355 | 104 | +0.087 | +0.212 | **0.045** | +0.335 | 9.75 | 3.86 |
| 1–3d | 184,230 | 1,014 | 1537.4 | **+0.389** | 184 | +0.018 | +0.213 | **0.005** | +0.366 | 6.09 | 3.00 |
| 3–7d | 118,094 | 585 | 453.6 | +0.076 | 107 | +0.022 | +0.094 | 0.597 | +0.028 | 5.38 | 3.23 |
| 7–30d | 251,827 | 1,929 | 1183.1 | +0.108 | 44 | +0.022 | +0.051 | 0.224 | +0.051 | 4.54 | 1.75 |
| >30d | 39,530 | 306 | 64.8 | +0.188 | 36 | +0.022 | +0.065 | 0.075 | +0.165 | 4.02 | 2.36 |

ρ **peaks in the middle** (1–3 days) and falls away on both sides. The variance ratio *declines*
monotonically with lifespan (8.88 → 4.02), and its ratio to the cluster null has no trend at all
(1.67–2.59, unordered). Nothing here rises with slowness.

⚠️ **Depth confound, visible in the `depth` column:** ρ rises mechanically with per-wallet bet depth,
and the long-lifespan cells hold the shallowest wallets (36–44 bets vs 93–184). So the low ρ at
L ≥ 7d is *partly* an artifact of thin wallets, not necessarily absent skill. `varR` divides out
sampling noise and is the sounder cross-cell comparison — and it is flat-to-declining too.

### 10.3 Axis H — bet horizon (entry → resolution)

| H | bets | wallets | eff. breadth | ρ | depth | ρ null (bet) | ρ null (cluster) | p (cluster) | LOMO ρ ≥ | varR | varR null (cluster) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <1h | 174,420 | 647 | 2141.9 | +0.122 | 147 | +0.028 | +0.173 | 0.771 | +0.108 | 4.75 | 2.27 |
| 1–6h | 198,580 | 1,428 | 1039.6 | +0.111 | 101 | +0.012 | +0.066 | 0.229 | +0.086 | 4.32 | 2.25 |
| 6–24h | 62,892 | 455 | 1271.3 | +0.358 | 71 | +0.035 | +0.131 | **0.005** | +0.324 | 8.53 | 3.53 |
| 1–3d | 26,810 | 185 | 376.4 | +0.338 | 56 | +0.037 | +0.287 | 0.199 | +0.312 | 8.60 | 5.78 |
| 3–7d | 176,551 | 1,678 | 820.4 | +0.172 | 40 | +0.011 | +0.095 | 0.164 | +0.110 | 4.09 | 1.91 |
| >7d | 12,548 | 358 | 328.9 | +0.427 | 35 | +0.035 | +0.354 | 0.075 | +0.408 | 7.33 | 4.77 |

The **one directionally thesis-shaped hint** in the whole run: the two near-resolution buckets
(<1h, 1–6h) have the *lowest* ρ and do not clear either null, while 6–24h clears the cluster null
outright and >7d has the highest raw ρ (+0.427). But >7d's cluster null is +0.354 — so **most of
that headline number is market clustering**, and it lands at p=0.075. Non-monotone (3–7d is +0.172,
ns) and mostly not significant. Suggestive at best; not evidence.

### 10.4 The sharpest single statistic: genuine skill dispersion is *constant* in market speed

Excess cross-wallet skill dispersion is real everywhere — varR clears **both** nulls at the p-floor
in **12/12** real-world cells, and in the micro control too. But the *raw* varR is not comparable
across cells, because market clustering differs hugely between them (cluster nulls run 1.08 → 5.78).
Divide it out, and the picture collapses to one number:

| cell | varR | cluster null | **ratio** |
|---|---|---|---|
| L <6h / 6–24h / 1–3d / 3–7d / 7–30d / >30d | 8.88 / 9.75 / 6.09 / 5.38 / 4.54 / 4.02 | 4.99 / 3.86 / 3.00 / 3.23 / 1.75 / 2.36 | 1.78 / 2.52 / 2.03 / 1.67 / **2.59** / 1.70 |
| H <1h / 1–6h / 6–24h / 1–3d / 3–7d / >7d | 4.75 / 4.32 / 8.53 / 8.60 / 4.09 / 7.33 | 2.27 / 2.25 / 3.53 / 5.78 / 1.91 / 4.77 | 2.09 / 1.92 / 2.41 / **1.49** / 2.14 / 1.54 |
| **micro-crypto (control)** | **1.79** | **1.08** | **1.66** |

**Every cell sits in 1.49–2.59, unordered in L and in H — and micro-crypto, the un-copyable
5-minute slice, sits at 1.66, right in the middle.** The amount of *genuine* skill heterogeneity,
once market clustering is removed, is the same in 5-minute crypto coin-flips as in month-long
markets. The apparent 5× spread in raw varR across the table is clustering, not skill.

This is the flat gradient stated as one number, and it is the run's main result.

### 10.5 Micro control

`n=3,335,883, 8,095 wallets, 212,020 markets`. ρ=+0.119 (2,197 wallets, median depth 24) against a
cluster null of +0.015, p=0.0099; varR=1.79 against a cluster null of 1.078, p=0.0099. So micro
**does** carry real, significant skill dispersion — consistent with the 36 deep persisters Project 1
already validated — at the same clustering-adjusted magnitude as every real-world bucket.

### 10.6 Within-wallet paired test — the direct answer to adverse selection

The cleanest test in the run, because it controls for wallet identity: for wallets active in both a
fast and a slow bucket, does the *same wallet* beat the price more when the market is slower?

| comparison | wallets | mean(slow − fast) | median | t p | Wilcoxon p | improving in slow |
|---|---|---|---|---|---|---|
| L: <24h vs ≥7d | 81 | +0.0041 | +0.0003 | 0.746 | 0.931 | 41/81 (51%) |
| L: <3d vs ≥3d | 127 | −0.0020 | +0.0083 | 0.821 | 0.482 | 74/127 (58%) |
| H: <6h vs ≥3d | 109 | +0.0045 | +0.0009 | 0.598 | 0.695 | 55/109 (50%) |

**All three null.** Signs are inconsistent (+, −, +), every p sits between 0.48 and 0.93, and the
improving-in-slow counts are coin flips. Effect sizes are ≤0.45¢/bet — two orders of magnitude below
the ~10¢ floor §5.2 argues the slow universe would need.

⚠️ **But note what this test can and cannot say.** It is methodologically the cleanest result here
*and* it is subject to the exact population limit that makes the whole run one-way: these are the
same micro-specialist wallets. It establishes that **micro-specialists do not become sharper in slow
markets**. It says nothing about whether slow-market specialists exist — those wallets are not in
this ledger to be paired.

### 10.7 Anchoring note

`end_date_iso` is the midnight floor of the scheduled end day, which put **87% of bets at a negative
horizon** on the first pass. H is therefore anchored on `max(end_date_iso, last observed trade on
the market)` — a hard lower bound on when the market was still live, so H is non-negative by
construction and a conservative *under*-estimate of the true horizon. Caveat: the last bet in every
market has H ≈ 0 by construction, so the `<1h` bucket is partly "the last bets we observed" rather
than strictly "bets placed just before resolution". Gamma's `closedTime` (§1.3) is the real fix and
lands with build step 1.
