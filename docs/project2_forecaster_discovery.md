# Project 2 — Copyable Forecaster Discovery

**Status:** design spec, not built. Read-only / analysis-only in spirit, same discipline as
`src/ingest.py` and `src/backfill.py`. Nothing here places trades or touches keys.

**Prereqs to read first:** `CLAUDE.md`, `HANDOFF.md` (esp. the "Edge-left-at-detection
analysis" verdict), `DECISIONS.md`. This doc assumes the Project-1 machinery (skill/residual edge,
the significance-gated OOS split, the leakage-guarded forward-price proxy) exists and is trusted; it
extends that machinery rather than replacing it.

---

## 0. Thesis — the gap Project 2 attacks

Project 1 ("rank the bots") works: the pipeline reliably finds the sharpest wallets it can *see*.
But two findings from that work, taken together, are the whole reason Project 2 exists:

1. **The wallets we're good at finding are ~all high-frequency micro-crypto bots.** The global
   `/trades` feed is dominated (~95%) by the 5-minute BTC/ETH "up or down" markets, so those are the
   wallets that accumulate enough observed bets to clear the sample-size bar.
2. **Their edge is essentially un-copyable.** Per HANDOFF's edge-decay audit: on micro-crypto,
   the own-edge ceiling is only ~+1.7¢/bet, ~half of positions have *no* follow-on liquidity to enter
   against at any latency, and the raw follower edge at 30–60s does not clear the shuffled-outcome
   base rate. Micro edge is structural (favorite-longshot) and gone by the time anyone could act.

The copyable edge lives in the **~5% real-world slice** (sports / politics / econ / long-horizon
events): per the same audit, 75% of those positions have follow-on liquidity, and at ~30s latency the
skill-residual follower edge is **+3.1¢ (95% CI +0.6…+5.7¢, excludes 0)** with a strongly negative
shuffled-outcome null — a genuine, favorite-longshot-neutral, outcome-correlated edge. It is thin and
sub-minute, but real.

**So the wallets we can *find* and the wallets *worth copying* are nearly disjoint.** Project 2 attacks
that directly: **deliberately surface real-world markets and the wallets trading them, then score those
wallets not just on "is the edge real" but on "is the edge real, copyable, big enough to matter, and
still current."**

### What is reused vs. what is new

| Reused as-is | New in Project 2 |
|---|---|
| Skill (residual) edge, `fit_price_baseline`, `residual_edge_per_bet` | **Targeted real-world ingest** (market-first discovery) |
| OOS chronological split + one-sided significance test (`validate.py`) | **Copyability score** (edge-left-at-detection → per-wallet) |
| Leakage-guarded forward-price core (`features._forward_price_arrays`) | **Economic-magnitude floor** (separate from significance) |
| Reliability down-weighting, "nothing is ever dropped" invariant | **Recency / form windows + regime-change detection** |
| `tune_weights.py` forward-validation methodology | **Per-category** everything (incl. a per-category price baseline) |

---

## 1. Targeted real-world ingest

### 1.1 The discovery gap

`ingest.py` is a rolling-window poller of the *global* feed. Because that feed is ~95% micro-crypto
and the API hard-caps at the most-recent ~10k trades platform-wide (≈ 5–10 min of wall-clock), a
real-world forecaster who bets **twice a month** will essentially *never* appear in it — their trades
fall outside every 10k-row window. `backfill.py` fixes per-wallet *depth* but only for wallets we've
**already discovered**; it cannot surface a wallet the global feed never showed us. Discovery of
real-world wallets is therefore the binding gap, and it needs a **market-first** ingest path.

### 1.2 Market-first discovery — the core idea

Instead of "watch the firehose and hope a real-world trade floats by," enumerate the **real-world
markets** directly and pull each market's full trade tape:

```
enumerate real-world markets (Gamma) ──► for each: /trades?market=<conditionId>
        │                                          │
        │                                          ├─ every wallet that traded it  ──► discovery
        │                                          └─ the whole market's tape       ──► copyability liquidity
        └─ category label per market ─────────────────────────────────────────────► per-category scoring
```

Three properties, **verified by a live probe (2026-07-18 — see §1.9)**:

- **`/trades?market=` works, but hits the SAME ~10k per-query offset cap as the global feed** — this
  corrects the first draft, which wrongly claimed markets "almost always" sit under the cap. The
  probe found *every* high/mid-volume market tested (down to a $13.8M Premier-League market) has
  **≥ 9,900 trades and hits the cap**; only the long tail of smaller markets is fully retrievable
  (a $4.5M market: ~7,500 trades, complete). Two consequences:
  - You still get **that market's own** most-recent ~10k trades — vastly more than the global feed
    ever shows for it — but for the biggest markets those recent trades **cluster near resolution**
    (a mega election market's recent ~10k spanned just 0.3 days), i.e. exactly the near-resolution
    window the leakage guard *discards*, biasing discovery toward late/closing traders and starving
    the *early-entry* bets where forecaster skill lives.
  - **Mitigations, now load-bearing:** (a) favor **mid/low-volume markets**, where the retrievable
    tape covers most of the lifespan and the early entries survive; (b) ingest **open markets live**
    and incrementally (§1.4) so early trades are captured before they scroll past the per-market 10k
    window; (c) lean on **per-user backfill** (§1.5) to recover a discovered wallet's own early
    entries on a capped market (one wallet's trades on one market are few, well under the cap).
- **The whole tape, not just the target wallet.** Copyability (metric B) is computed from *other*
  wallets' trades on the same token after an entry. Per-*user* backfill gives a wallet's own bets but
  not the surrounding book activity; **market-first ingest gives the surrounding tape**, so it is a
  *prerequisite* for measuring copyability, not merely a discovery convenience.
- **It breaks the 1.4h-ledger ceiling — confirmed decisively.** The probe measured real-world market
  tapes spanning **9, 25, and 73 days** (vs. the 1.4h global-poll ledger). So the recency/regime
  (§2-D) and hour-to-day copyability (§2-B) analyses that were *unmeasurable by construction* on the
  old ledger become measurable on real-world market tapes — subject to the near-resolution bias above
  for the very largest markets.

### 1.3 Enumerating the market universe

**Primary source: the Gamma listing — verified mechanics (probe 2026-07-18, §1.9).** Enumerate two
complementary ways, both confirmed working:
- **Volume-ranked sweep — `gamma-api.polymarket.com/markets?closed=true&order=volumeNum&ascending=false`.**
  `order=volumeNum` genuinely sorts (top market $1.5B ↓; note the field is `volumeNum` — plain
  `order=volume` returned garbage $100 markets and must not be used). Offset paging is clean (0
  overlap page-to-page); the **effective page cap is 100** (`limit=500` silently returns 100), so page
  in 100s. **Key empirical payoff:** the entire top-40 by volume is real-world (politics / sports /
  econ / geopolitics / crypto-events) with **zero micro-crypto** — because micro markets have
  negligible *per-market* volume, volume-ranked enumeration inherently inverts the 95%-crypto feed
  problem. This is the discovery gap, solved.
- **Category slices — `/events?tag_slug=<category>`.** Confirmed working on `/events` (politics →
  election events, sports → NFL/NCAA events). **`/markets` ignores `tag_slug`/`tag`** (returns an
  unrelated default listing — consistent with DECISIONS.md's Gamma-filter warning), so category
  filtering must go through `/events`, then extract each event's markets → `conditionId`s.

Request both `closed=true` (resolved → historical skill signal) and `closed=false` (live → the
watch/copy layer and, critically, early-tape capture per §1.2/§1.4).

**Category labeling — verified, three-layer.**
1. **Per-market `tags`** via `?include_tag=true` on the listing, or the `/markets/{id}` detail — the
   probe confirmed these carry top-level labels (a market returned `['Trump','Politics','US
   Election',…]`). Primary labeler where present.
2. **`sportsMarketType`** — present and non-null on sports markets (e.g. `"totals"`); a reliable sports
   signal.
3. **Slug/question keyword classifier** as the fallback — an extension of the existing
   `audit_edge_decay.classify_horizon`. Naive keyword-only classification scored **~78% on the top-40
   by volume (22% unknown)**; the misses were geopolitics/novel-politics phrasings ("US forces enter
   Iran…", "Zelenskyy wear a suit") that layers 1–2 (tags/events) cover. Combined, classification is
   reliable enough to be useful; keyword-alone is not. The classifier must at minimum separate:
   - `micro_crypto` — the 5-min "up or down" coinflips (the existing `updown`/`up-or-down` rule);
   - real-world sub-categories: `sports_*` (and ideally league: NFL/NBA/soccer/UFC/MLB…),
     `politics`/`elections`, `econ_macro` (Fed/CPI/GDP/jobs), `crypto_event` (long-horizon "BTC hits
     $X by date" — real-world horizon, *distinct* from micro), `culture`, `other`.

   Sub-category granularity matters because "sharp at NBA" and "sharp at politics" are different
   skills a user would want to tell apart — the same reasoning that motivated the existing
   `high_frequency_micro_market` pattern flag. Keep the classifier small, tested against a hand-built
   fixture of known slugs, and conservative (unknown → `other`, never force a guess).

### 1.4 Ingesting the discovered markets' trades

For each enumerated market's `conditionId`, page `/trades?market=<conditionId>` newest-first into the
**same `bet_ledger.parquet`**, additive and dedup-safe on the existing key
`(tx_hash, wallet, token_id, side)`. Reuse `fold_trades_to_ledger`, `update_resolutions`,
`refresh_ledger_resolutions` unchanged. Incrementality mirrors `backfill.py`:

- **Per-market cursor** (`discover_cursors.json`, same `{max_timestamp, keys_at_max}` overlap-safe
  shape as ingest/watch) so re-runs fetch only new trades on still-open markets. Resolved markets are
  pulled once and marked done.
- **Poll open markets live to capture the early tape.** Because the per-market query hits the same
  ~10k offset cap (§1.2), a high-volume market's *early* trades scroll out of reach once it accrues
  >10k trades near resolution. Ingesting markets **while `closed=false`**, on a cadence, captures those
  early entries in real time before they fall off — the same rolling-overlap discipline `ingest.py`
  uses for the global feed, but scoped per-market (and far slower per market, so easily kept
  overlapping). This is what makes early-entry copyability measurable on popular markets.
- **Bounded per run.** Cap markets-per-run and reuse `update_resolutions(max_fetches=…)`; checkpoint
  cursors and loop nightly until the universe is covered and resolutions converge — exactly the
  pattern `backfill.py` established.
- **Prioritize** by liquidity and recency, but with the cap in mind: for **resolved** markets favor
  **mid/low-volume** ones (fully retrievable, early entries intact — see §1.2); for the **mega**
  markets, rely on the live open-market poll above plus per-user backfill (§1.5) rather than a
  one-shot historical pull that would return only the near-resolution slice.

### 1.5 Depth on discovered wallets

Discovery gives breadth (which wallets touch real-world markets); the existing **`backfill.py` gives
depth** (each wallet's full cross-market `/trades?user=` history). Chain them: after a discovery run,
feed the set of wallets that appeared in ≥1 real-world market into `backfill.select_wallets`'s
candidate pool (in addition to the current top-N-by-rank seed). A discovered wallet's full history may
of course include micro-crypto bets too — that's fine and wanted, because per-category scoring (§2)
partitions each wallet's bets by category, so a wallet is scored as a forecaster *only on its
real-world bets*.

### 1.6 Ledger / storage additions

- **`category` column on the ledger (additive, nullable).** Persist the classifier's label so
  downstream per-category work doesn't re-derive it every run. Backward-compatible: pre-existing rows
  get `NaN`/`"unknown"` and are backfilled from a market→category map on the next run. (Alternative:
  derive category at feature-time from `slug`/`question` with no schema change — simpler but recomputed
  every run; the persisted column is preferred for a large real-world backlog. Flagged as a decision to
  confirm.)
- **New small state files**, flat and gitignored like the existing cursors: `discover_cursors.json`
  (per-market), a `market_universe.parquet` (conditionId → category, volume, closed, first/last seen)
  as the enumeration cache so re-runs don't re-page all of Gamma.
- **Disk footprint is modest.** Real-world markets are far fewer and slower than micro, so the
  incremental compressed-Parquet cost is small; the existing `print_disk_usage_summary` cap logic
  covers it. Prune raw per-poll payloads as today; the ledger accumulates resolved bets forever.

### 1.7 Module + config sketch (Part 1)

New module `src/discover.py` (`run_discover()`): enumerate → classify → per-market ingest → hand
discovered wallets to backfill. Slots into `run.sh` before `backfill` in the nightly chain
(`discover → backfill → features → validate → rank → report`).

```yaml
discover:
  gamma_api_base: "https://gamma-api.polymarket.com"
  enum_order: "volumeNum"         # VERIFIED sort field (NOT "volume", which returns garbage)
  event_tag_slugs: [politics, sports, economy, crypto]  # /events?tag_slug= (works; /markets ignores it)
  gamma_page_size: 100            # effective cap: limit>100 silently returns 100
  markets_per_run: 300            # bound each run; loop nightly until universe covered
  min_market_volume: 5000         # skip near-empty markets (few wallets, no copyable liquidity)
  include_open_markets: true      # closed=false too — early-tape capture (§1.4) + live watch layer
  trades_page_size: 500           # /trades?market= page size (data-api)
  max_pages_per_market: 20        # 20*500 = the shared 10k /trades offset cap — HITS it for big markets
  max_resolution_fetches_per_run: 2000
  sleep_between_requests_sec: 0.1
```

### 1.8 Tradeoffs & uncertainties (Part 1)

- **RESOLVED by the probe (§1.9): Gamma enumeration is reliable; the approach holds.** Volume-ranked
  `/markets` paging and `/events?tag_slug=` category slices both work; the volume-ranked top is
  real-world-dominated (discovery gap solved). Classification is reliable *combining* tags + events +
  `sportsMarketType` + keyword (keyword-alone ~78%). No redesign needed.
- **CORRECTED by the probe: the 10k offset cap is NOT escaped per-market.** High/mid-volume markets
  (≥ ~10k trades) return only their most-recent ~10k, which for the biggest markets is a
  near-resolution slice — the early-entry bets that matter are unreachable in one historical pull.
  This is now a first-class design constraint, not a footnote: it drives the live open-market ingest
  (§1.4) and the mid-volume-market preference (§1.2). **Remaining uncertainty:** whether live polling
  of open markets + per-user backfill together recover *enough* early-entry tape on popular markets to
  measure their forecasters — only a multi-day live run will tell. Until then, expect the cleanest
  signal from the mid/low-volume long tail.
- **Category is fuzzy / multi-label.** A market can be both "crypto" and "econ" (the probe's tag lists
  are granular and overlapping, e.g. `['Trump','Politics','US Election',…]`). Store the single best
  label but keep the raw tags so re-labeling is cheap; treat category as a *view*, not a hard
  partition of identity.
- **Cost.** Thousands of real-world markets × a paged pull is a real but bounded backlog (same shape
  as the deep backfill; gamma page cap is 100, so more but cheap pages). Bounded-per-run + nightly loop
  handles it; log what was deferred (no silent truncation).
- **Still unverified (left for the build's verify gate):** exact `/trades?market=` behavior at
  offset ≥ 10000 (probe stopped at 9900; the cap is documented at 10000 and `backfill.py` already
  handles the 400), and whether `min_volume`/`volume_num_min`-style server-side volume *filters* work
  on `/markets` (the probe only confirmed volume *ordering`).

### 1.9 Probe results (verified 2026-07-18, read-only GETs; nothing written to `data/`)

Small live probe of `gamma-api.polymarket.com` (enumeration) and `data-api.polymarket.com/trades`
(tapes), a handful of requests, run while the backfill was live on a different host.

| Question | Result |
|---|---|
| Enumerate real-world markets (paged)? | **Yes.** `/markets?order=volumeNum&ascending=false` sorts by volume (top $1.5B); offset paging clean (0 overlap); page cap 100. Top-40 by volume = **all real-world, zero micro-crypto**. |
| Sort field | `volumeNum` works; **`volume` returns garbage** ($100 markets) — do not use. |
| Category filter | `/events?tag_slug=politics\|sports` **works**; `/markets?tag_slug=` **ignored** (unrelated default listing). |
| Classify locally? | Per-market `tags` (via `include_tag=true` / `/markets/{id}`) carry top-level labels (`'Politics'`, …); `sportsMarketType` flags sports; keyword-only ~**78%** on top-40 (misses geopolitics). Combined = reliable enough. |
| `/trades?market=` complete tape? | **Partly.** Same ~10k offset cap per query. $13.8M+ markets have ≥9,900 trades (capped, recent-slice only); a $4.5M market had ~7,500 (fully retrievable). |
| Breaks the 1.4h ceiling? | **Yes, decisively.** Tape spans measured: 73 days ($4.5M mkt), 25 days ($13.8M), 9 days ($109M) — and 0.3 days for a mega election market's recent-10k (near-resolution clustering). |

---

## 2. Per-category discovery + the A/B/C/D metric stack

Everything below is computed **per (wallet, category) cell** where sample allows, plus an all-category
aggregate. A wallet is a "forecaster" candidate *in a category* — the unit of a winner record is the
(wallet, category) pair, not the wallet alone.

### 2.0 The one change to metric A's foundation: a per-category price baseline

The favorite-longshot calibration curve `E[outcome | entry_price]` (`fit_price_baseline`) is currently
fit **market-wide across all bets** — which today means it is dominated by micro-crypto coinflips near
0.5. Sports/politics have a *different* price→outcome calibration (a 0.75 NFL favorite is not the same
animal as a 0.75 crypto-updown print). Residualizing real-world bets against a crypto-dominated curve
mis-measures skill. **Fix: fit the baseline per category** (at minimum per horizon micro vs real-world;
better per real-world sub-category where sample allows, falling back to the real-world-wide curve, then
the global curve, when a sub-category is too thin). This is a small, contained change to how the
baseline is fit and passed into `residual_edge_per_bet`, and it is a *correctness* fix, not a new
metric.

### A — Is the edge real? *(mostly built; reused)*

Per (wallet, category): in-sample residual (skill) edge > 0 selects a candidate; **persistence** =
held-out residual edge > 0 **and** one-sided t-test p < `oos_significance_alpha`. Reuse `validate.py`
verbatim, with the per-category baseline from §2.0 and bets filtered to the category.

- **Low-frequency is fine here.** A UFC forecaster at ~2 bets/month is ~24/year — over a multi-year
  backfill that is a valid sample, and the significance test already *demands a larger edge* from a
  smaller n (that's what the t-test does). Frequency ≠ sample size. What must be tuned per category is
  `min_bets_per_half`: 10/half suits fast categories; for genuinely low-frequency categories, prefer a
  *longer lookback* over lowering the threshold (lowering it re-admits the coin-flip problem the
  significance test was built to kill). Some legitimately-skilled but rare forecasters will remain
  un-validatable — an honest floor set by their real frequency, not fixable by more ingest.

### B — Copyability *(NEW — the metric that separates Project 2 from Project 1)*

Turns HANDOFF's edge-decay analysis from a pooled diagnostic into a **per-wallet, per-category score**.
Reuse `features._forward_price_arrays` with the exact leakage discipline (bounded window, resolution
guard, **no `resolved_value` fallback**) and the same-cohort / shuffled-null / bootstrap-CI rigor of
`audit_edge_decay.py`.

For a wallet's resolved BUY bet *i* on token *T*, entry time *tᵢ*, at operational latency **Δ** and
follower fill bandwidth **b**:

- `price_at_i` = size-weighted price of *other* wallets' trades on *T* in `(tᵢ+Δ, tᵢ+Δ+b]` before the
  guard cutoff; `reachable_i = 1` if any such trade exists.
- `follower_skill_i = resolved_value_i − E_cat[outcome | price_at_i]` (per-category baseline) — the
  favorite-longshot-neutral edge a copier keeps after entering Δ late at the prevailing price.

Two components, kept **separate** (fusing them is itself a weighting choice — defer, per project
guidance):

1. **Reachability(w, Δ, b)** = `mean_i reachable_i` — can a follower even get a fill? A wallet whose
   positions nobody trades after is un-copyable regardless of edge (this is the "54% vs 75%
   followable-liquidity" split from the audit, now per-wallet).
2. **Retained copyable edge(w, Δ, b)** = mean of `follower_skill_i` over reachable bets, with a
   bootstrap 95% CI and the shuffled-outcome null. This is the level that must clear the magnitude
   floor (C).

Report both at a small Δ sweep (e.g. 30s / 60s / 300s) plus Δ=0 (own-edge ceiling, same-cohort) so
the per-wallet **decay** is visible. If a single sortable scalar is needed,
`reachability × retained_edge(Δ=60s)` = *expected copyable edge per detected signal* (counting
un-fillable positions as 0) — offered as a convenience column, with the two components always present
so the user isn't forced to accept that composition.

- **Caveat (large):** copyability is sample-hungry and selection-prone (larger-Δ cohorts are biased
  toward longer-lived markets — hence same-cohort own-edge and CIs). On the *current* 1.4h global-poll
  ledger the minutes-to-hours latency a human operates at is unmeasurable — but the §1.9 probe
  **confirms the fix is real**: real-world market tapes span 9–73 days, so once §1's market-first
  ingest lands, hour-to-day copyability becomes measurable. The residual caveat is the near-resolution
  bias on mega markets (§1.2): their retrievable tape is late-clustered, so early-entry copyability
  there stays thin until live open-market polling accrues it. Until §1 lands, report only the
  sub-minute Δ the current data supports, and flag the rest "not yet measurable," never "no edge."

### C — Economic magnitude *(NEW)*

Significance answers *"is it real?"*; magnitude answers *"is it big enough to clear fees + slippage?"*
These are different questions, and conflating them is the trap the user flagged: **on huge backfilled
samples the t-test will start passing +1¢ edges** (large n → tiny SE → statistical significance on
economically-worthless effects). So magnitude is a **separate, independent gate**, not another p-value.

- **The floor.** A (wallet, category) qualifies only if its held-out **copyable** skill edge (B, at
  the operational Δ) is **both** significant (A) **and** ≥ `magnitude.min_copyable_edge_cents`. The
  floor is an *economic* threshold with a principled basis (cover trading cost), not a fitted weight —
  so proposing a starting value is defensible; it is still config-tunable.
- **What the cost actually is.** Polymarket's explicit trading fee is ~0 today (subject to change), so
  the real cost is **slippage**, of two kinds:
  1. *Latency slippage* — the price moved between the wallet acting and the copier filling. This is
     **already captured empirically** by using `price_at(tᵢ+Δ)` as the follower entry in B. Good — no
     estimate needed.
  2. *Impact + spread slippage* — the copier's own order walks the book, and pays the spread. The free
     tape has **no depth history**, so this is **not measurable offline**. Subtract a conservative
     configurable buffer (`magnitude.slippage_buffer_cents`) as a pre-filter, and be explicit that the
     true value is not knowable from this data at all. The magnitude floor is a pre-filter, not a
     measurement.
- **Units: residual-skill cents per bet, never $ profit or return %.** Restating the user's guidance
  and the project's founding principle — do **not** rank by raw returns / profit rate; that is the
  favorite-longshot + size + luck trap this whole engine exists to avoid.

### D — Recency / form + regime-change *(NEW)*

A forecaster's skill can go stale (roster changes, a market regime they had edge in ends). Two pieces,
both **count-aware** to respect low frequency:

- **Form windows.** Skill edge over trailing windows — but windowed by **both time and count**, so a
  2-bets/month forecaster isn't reported as "0 in the past week." Compute, per (wallet, category):
  `skill_edge` over {past 30d, past 90d, all-time} **and** {last 20 bets, last 50 bets}, each with its
  n and CI; render "insufficient recent data" rather than a spurious 0 when a window is empty. Form is
  a **layer on top of** validated all-time skill (A), never a replacement — recency alone is noise.
- **Regime-change detection.** HANDOFF flagged that the chronological select/validate split mishandles
  non-stationary wallets (its example: a wallet skill-*negative* in half 1, strongly skill-*positive*
  in half 2 — treated as "not a candidate"). A simple change-point test on the per-bet residual-edge
  series (compare mean skill edge before/after the best candidate breakpoint; or CUSUM) flags wallets
  whose recent regime differs significantly from their early regime, in **both** directions:
  - *improving* (early-negative → recent-positive) — currently lost by first-half selection; surface
    it so a genuinely-improved forecaster isn't silently dropped;
  - *decaying* (early-positive → recent-negative) — currently could still rank on all-time; surface it
    so the user doesn't follow a wallet that has stopped working.

  Consistent with the project's philosophy, regime-change is a **flag + the form columns**, not a hard
  exclusion — but the recommended default *view* (§3) uses the decaying flag as a default filter,
  because "worth copying *now*" is the product question.

### 2.x Combining the stack — gates + sortable columns, **not** premature weights

The user's explicit guidance, backed by `tune_weights.py`: on current data no weight scheme beats
another beyond bootstrap noise, so **do not hand-declare metric weights**. Project 2 honors this by
**not fusing A/B/C/D into one weighted scalar**. Instead:

- **Gates, not weights, define a "copyable forecaster."** A default candidate is a (wallet, category)
  that: is real (A: significant held-out skill edge), is reachable + retains edge (B: reachability ≥
  floor, retained copyable edge CI excludes 0), clears the economic floor (C), and is not flagged
  decaying (D). Gates are thresholds with principled bases (significance level, an economic cost floor,
  a liquidity floor) — far more defensible on noisy data than fitted relative weights.
- **Every metric is an independently-sortable column.** The engine collects everything and hides
  nothing (CLAUDE.md); the user sorts/filters to their own risk tolerance.
- **Any eventual composite weighting is deferred to forward-validation on richer data.** Extend
  `tune_weights.py`'s leakage-free methodology to the new target: predict each wallet's held-out
  **copyable** edge from its **first-half** features (skill edge, reachability, magnitude, form),
  bootstrap the difference between weightings, and adopt a composite score **only if its CI excludes
  0**. Until then, present the gates + columns, re-run as the ledger deepens.

### 2.y Module + config sketch (Part 2)

- Per-category baseline: extend `features.fit_price_baseline` callers to fit per category (a
  `{category: PriceBaseline}` map), and thread the right baseline into `residual_edge_per_bet` /
  `validate.py`.
- New `src/forecaster_metrics.py`: copyability (B), magnitude (C), recency/regime (D), per
  (wallet, category). Reuses `_forward_price_arrays`, `fit_price_baseline`, `expected_outcome`.
- New `src/rank_forecasters.py`: apply the gates, emit the per-category gated table (leaves Project 1's
  `ranked_wallets.parquet` untouched — the two coexist).
- Tests (per CLAUDE.md style): a hand-built fixture where the correct copyability, magnitude, and
  regime break are known by construction — e.g. a wallet whose edge is fully gone by Δ=60s (copyability
  ≈ 0 despite skill edge > 0), and a wallet with a planted change-point.

```yaml
copyability:
  operational_deltas_sec: [30, 60, 300]   # latencies to report; 0 == own-edge ceiling
  fill_bandwidth_sec: 60                   # follower fill window (matches watch.py poll)
  min_reachability: 0.5                    # gate: fraction of positions with any followable liquidity
magnitude:
  min_copyable_edge_cents: 2.0             # economic floor (cover slippage); tunable, NOT a fitted weight
  slippage_buffer_cents: 1.0               # conservative impact+spread haircut; not measurable from this data
recency:
  windows_days: [30, 90]
  windows_bets: [20, 50]
  regime_min_bets_each_side: 10            # change-point test needs points on both sides
```

### 2.z Uncertainties (Part 2)

- **Sample depth may stay the binding constraint** for low-frequency real-world forecasters even after
  full backfill — their real bet frequency caps n. Widen the lookback (multi-year), accept that some
  are un-validatable.
- **Copyability's operational-latency region is unmeasurable on the *current* 1.4h ledger** but becomes
  measurable once §1 lands — the §1.9 probe confirmed real-world tapes span 9–73 days. Numbers stay
  provisional until then, with a lasting near-resolution bias on the very largest markets (§1.2).
- **Impact/spread slippage is not measurable offline** — the magnitude floor is a pre-filter, and
  nothing in this repo can turn it into a measurement.
- **Regime-change on thin series is itself noisy** — a change-point test needs points; treat its output
  as a flag to inspect, not a verdict.

---

## 3. Output shape

### 3.1 The winner record

The unit is a **(wallet, category)** row. Schema (all columns; the dashboard renders a subset):

```
wallet, category
── A: is it real ───────────────────────────────────────────────
  sample_size, bets_per_month            # frequency shown, never used as a quality proxy
  skill_edge_all, skill_edge_p, edge_persisted
  in_sample_skill_edge, out_of_sample_skill_edge   # side by side, decay visible (CLAUDE.md)
  raw_edge_all                           # reported alongside skill edge, per CLAUDE.md
── B: copyability ──────────────────────────────────────────────
  reachability@60s
  copyable_skill_edge@30s / @60s / @300s  (+ 95% CI, shuffled-null)
  own_edge_ceiling (Δ=0, same-cohort)     # how much latency costs
  expected_copyable_edge_per_signal       # reachability × retained_edge, convenience only
── C: magnitude ────────────────────────────────────────────────
  copyable_edge_cents, clears_magnitude_floor (bool)
── D: recency / form ───────────────────────────────────────────
  skill_edge_30d / _90d / _last20 / _last50  (+ n each, "insufficient" when empty)
  form_trend (improving / steady / cooling), regime_change_flag
── metadata (never affects gates/sort) ─────────────────────────
  copy_window, breadth, time_consistency
  manufactured_record_flag, pattern_flag
  confidence (from sample_size + CI width)
```

A "winner" is a row that **clears all four gates** (§2.x). But — per the invariant preserved from
Project 1 — **no row is ever dropped**; non-qualifying (wallet, category) pairs stay in the table with
their metrics computed, sorted below, and the gates exposed as boolean columns the user can relax.

### 3.2 The dashboard view

- **Per-category tables**, each **sortable by any metric column** (the parquet is the source of truth;
  an optional self-contained, offline, sortable HTML table generated from it is the natural presentation
  layer — matches the repo's low-dependency ethos, ships nothing external).
- **Recency layered on validated skill**, as toggleable columns/views: *past week / past month /
  all-time* (and the count-based `last20/last50`), so the user sees "sharp all-time, but cooling" at a
  glance. Recency is a lens over the A-gated set, never a standalone ranking.
- **Recommended default filters (all user-adjustable — the engine hides nothing):**
  1. `category ∈ real-world` (exclude `micro_crypto` by default — the un-copyable slice);
  2. `edge_persisted = true` and `skill_edge_p < 0.05` *(A)*;
  3. `reachability@60s ≥ 0.5` and `copyable_skill_edge@60s` CI excludes 0 *(B)*;
  4. `clears_magnitude_floor = true` *(C)*;
  5. `regime_change_flag ≠ decaying` *(D)*.
  Each filter shows **why it exists** in the UI, and every one can be toggled off — a user who wants to
  see flagged/cooling/thin wallets can.
- **Default sort within a category:** by `copyable_skill_edge@60s` (the B∩C axis — real *and* copyable
  *and* big enough), **never** by raw returns, profit, or win rate. Restated because it is the single
  most important "don't": ranking by profit re-introduces the favorite-longshot + size + luck trap the
  entire project exists to avoid.

### 3.3 How it connects downstream

- **`watch.py` scopes to the Project-2 copyable set.** The edge-decay verdict narrowed what is even
  observable to real-world markets at short latency. Project 2 produces exactly the watchlist that
  makes that concrete: the real-world (wallet, category) rows clearing all four gates. `watch.py`
  stays read-only/alert-only; it just watches a better list.
- **Project 2 is the pre-filter; a forward reading is the arbiter.** These metrics narrow
  thousands of wallets to a shortlist of *candidate* copyable forecasters. Scoring that shortlist
  forward then validates it — running forward at true detection latency,
  logging the hypothetical entry at the real detected price, scored on resolution, no peeking. That is
  the only honest measurement of copyable edge, and it is what the whole stack feeds. Weighting,
  magnitude floors, and copyability thresholds all get their calibration from that forward loop,
  not from in-sample fitting.

---

## 4. Build order (staged; commit after each — reuses existing plumbing)

1. **`src/discover.py`** — Gamma enumeration + keyword classifier + per-market `/trades?market=`
   ingest into the ledger; add `category` column; chain into `run.sh`. *(Part 1.)*
2. **Per-category baseline** — fit `fit_price_baseline` per category; thread through `validate.py`.
   *(§2.0 — correctness.)*
3. **`src/forecaster_metrics.py`** — copyability (B), magnitude (C), recency/regime (D), per
   (wallet, category), with tests on planted fixtures. *(Part 2.)*
4. **`src/rank_forecasters.py`** + output — gated per-category table to `data/processed/forecasters/`;
   optional offline sortable HTML. *(Part 3.)*
5. **Extend `tune_weights.py`** — forward-validate any composite weighting against held-out *copyable*
   edge; adopt a composite only if its bootstrap CI excludes 0. *(§2.x.)*
6. **Re-scope `watch.py`'s watchlist** to the real-world gated set.

## 5. Invariants preserved (non-negotiable, from CLAUDE.md / HANDOFF.md)

- **Read-only, analysis-only.** No order placement, no key handling, no chain writes. Discovery and
  scoring only.
- **Nothing is ever dropped.** Gates and flags are additive/boolean columns and *views*; every
  (wallet, category) row stays in the table. Filtering is the user's, applied on top.
- **Flags stay metadata.** `manufactured_record_flag` / `pattern_flag` never feed a gate or a sort key.
  (The recency/regime *flags* are new and *do* inform the recommended default view — but as
  user-toggleable filters over a complete table, not as silent row removal or score edits.)
- **Skill edge, not raw edge; magnitude separate from significance; never rank on profit.**
