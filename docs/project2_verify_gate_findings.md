# Project 2 — Verify-Gate Findings

**Date:** 2026-07-19. **Method:** read-only live GETs against the public Polymarket
REST APIs (`gamma-api.polymarket.com`, `data-api.polymarket.com`), polite rate-limiting
(single-threaded, ~0.3–0.4s between requests, back-off on 429). **Nothing written to
`data/`.** This resolves the open empirical questions the design spec
(`docs/project2_forecaster_discovery.md` §1.8, §2.z) left "for the build's verify gate,"
and corrects/extends the earlier 2026-07-18 probe (§1.9) with the exact requests run and
the responses observed.

The probe scripts are in the session scratchpad (not committed); every request and its
result is transcribed below so the findings are reproducible.

---

## Summary — what changed vs. the spec's assumptions

| Question | Spec assumption | Verified result |
|---|---|---|
| Gamma volume-ranked enumeration | works via `order=volumeNum`, offset paging, cap 100 | **Confirmed** — but there is an **offset ceiling ~2400** (HTTP 422 → use `/markets/keyset`). Spec did not know this. |
| Deep enumeration past the top ~2400 | (not addressed) | **`/markets/keyset` exists** (cursor pagination); page-1→2 advances, but **reliable multi-page volume-ordered continuity was NOT confirmed** (stalled after page 2 in the probe). Treat as follow-on. |
| Server-side volume *filter* | left unverified (§1.8) | **Does NOT work.** `volume_num_min` / `volumeNum_min` / `min_volume` are silently ignored; `liquidity_num_min` → HTTP 500. Filter volume **client-side**. |
| `/trades?market=` filter correctness | works | **Confirmed exactly** — 0 wrong-market rows across a 10,500-row pull. |
| `/trades?market=` offset ≥ 10000 | "documented at 10000, probe stopped at 9900" (§1.8) | **Confirmed:** offset ≤ 10000 → 200; offset > 10000 → **HTTP 400 "max historical trades offset of 10000 exceeded"** (same cap as the global feed). |
| Micro-crypto vs real-world signal | tags / `sportsMarketType` / keyword | **Slug keyword is the only reliable signal.** `sportsMarketType` is `None` even on sports futures; Gamma metadata **lifespan is unreliable** (5-minute `updown` markets report a ~24h start→end window). |

**Bottom line: the market-first discovery approach is feasible and the design holds**, with
two corrections baked into `src/discover.py`: (1) enumerate volume-ranked via **offset paging
up to the ~2400 ceiling** (covers the entire high/mid-volume real-world universe — all
real-world, zero micro-crypto), keyset deep-pagination deferred; (2) apply the **min-volume
filter and micro-crypto exclusion client-side**, since neither is available server-side.

---

## 1. Gamma `/markets` enumeration

### 1.1 Volume ordering — `order=volumeNum` works, `order=volume` is garbage

Request:
```
GET gamma-api.polymarket.com/markets?closed=true&order=volumeNum&ascending=false&limit=5
```
→ HTTP 200, top 5 by volume (descending):
```
vol=1,531,479,285  will-donald-trump-win-the-2024-us-presidential-election
vol=1,037,039,118  will-kamala-harris-win-the-2024-us-presidential-election
vol=  400,409,527  will-donald-trump-be-inaugurated
vol=  378,011,507  will-the-sacramento-kings-win-the-2025-nba-finals
vol=  375,813,105  microstrategy-sells-any-bitcoin-by-may-31-2026
```
Request `...&order=volume&...` → HTTP 200 but returns **$100 markets** (`cs2-...total-games-2pt5`, …).
**Use `volumeNum`, never `volume`.** (Matches the 2026-07-18 probe.)

### 1.2 Page size cap 100; offset paging clean

`limit=500` → HTTP 200 but **list length 100** (silently capped). Offset paging is clean:
```
offset=0   limit=100 → vol 1.53B … 73.9M
offset=100 limit=100 → vol 73.8M … 42.8M   (overlap with page 0 = 0)
```

### 1.3 **Offset ceiling ~2400 → 422 (new finding)**

```
offset=2000 → HTTP 200 (vol ~4.75M … 4.54M)
offset=2400 → HTTP 422 {"type":"validation error","error":"offset too large, use /markets/keyset for deeper pagination"}
offset=2500/2600/4000 → same 422
```
So **offset-based volume enumeration is usable for offset 0…~2300** (≈2400 markets, 24 pages
at limit 100). Crucially, at offset 2000 the markets are **still ~$4.5M volume** — i.e. the top
~2400 markets are all substantial, real-world markets. That is more than enough candidate
real-world markets to seed discovery; the discovery gap (surfacing real-world markets at all)
is solved by the top slice alone.

### 1.4 `/markets/keyset` — deep pagination exists, continuity unconfirmed

```
GET /markets/keyset?closed=true&order=volumeNum&ascending=false&limit=5
→ 200 {"$schema":…, "markets":[…5…], "next_cursor":"_nkXZEL…"}
```
The cursor works **only when the filter params are dropped on subsequent pages** (the cursor
encodes `order`/`ascending`/`closed`): re-sending `order=volumeNum&ascending=false&closed=true`
alongside `next_cursor` returns page 1 again; sending `limit + next_cursor` **alone** advances
to a different page. However, paging further stalled (distinct markets stopped growing after
~2 pages, and the page-2 contents were not cleanly volume-ordered). **Verdict:** keyset is the
documented path past the offset ceiling and is worth a dedicated follow-on probe, but reliable
full-universe volume-ordered enumeration through it is **not yet proven**, so `discover.py`
relies on offset paging (0…2300) for now and logs that the long tail beyond the ceiling is
deferred (no silent truncation).

### 1.5 Server-side volume filter does NOT work

```
...&order=volumeNum&ascending=false&limit=5&volume_num_min=1000000   → 200, ignored (same top-5)
...&volumeNum_min=1000000                                            → 200, ignored
...&min_volume=1000000                                              → 200, ignored
...&liquidity_num_min=1000000                                       → HTTP 500
```
None filter. **`discover.py` filters `volumeNum >= min_market_volume` client-side.** (Consistent
with DECISIONS.md "Gamma markets API query filters don't work as documented.")

---

## 2. Category / classification signals

### 2.1 `/events?tag_slug=` works; `/markets?tag_slug=` ignored

```
GET /events?tag_slug=politics&closed=true&limit=5  → 200, events e.g. "Which party wins 2024 US Presidential Election?"; each event carries `markets` with conditionIds.
GET /events?tag_slug=sports&closed=true&limit=5    → 200, events e.g. "NFL Draft: Top 5 Picks Scenarios" (6 markets).
GET /markets?tag_slug=sports&closed=true&limit=5   → 200 but UNRELATED default listing (2020-era "will-joe-biden-get-coronavirus", "will-airbnb-begin-publicly-trading", …).
```
Category filtering must go through **`/events?tag_slug=`**, then expand each event's `markets`
→ `conditionId`. (Matches the 2026-07-18 probe.)

### 2.2 Tags are present but very granular; `sportsMarketType` unreliable

`?include_tag=true` on a slug lookup returns a `tags` list, but it is **long and hyper-granular**
(the Trump market carried ~30 tags: `hillary clinton`, `joe biden`, `USA Election`, `Trump`,
`Politics`, `Elections`, `republican party`, …). Top-level labels (`Politics`, `US Election`)
are buried among candidate-name tags, so tags need a *keyword rollup* to a top-level category,
not a direct read. `sportsMarketType` was **`None`** on every top-15 market including
`will-the-sacramento-kings-win-the-2025-nba-finals` and `will-argentina-win-the-2026-fifa-world-cup`
— it is only populated on game-level markets (totals/spreads), not futures, so it is a *weak
positive* signal for sports, never a negative one.

### 2.3 Metadata lifespan is NOT a micro-crypto signal — slug is

Open low-volume listing (`closed=false&order=volumeNum&ascending=true`) is dominated by
micro-crypto with slugs like:
```
eth-updown-5m-1766161800   startDate→endDate span reported ≈ 23.94h   sportsMarketType=None
btc-updown-5m-1766162100   span ≈ 23.95h
sol-updown-5m-1766162100   span ≈ 23.95h
```
The **"5m" is in the slug**, but the Gamma `startDate`/`endDate` report a ~24h window — so the
5-minute cadence is invisible in metadata. **Do not classify horizon from metadata lifespan**
(the spec's `audit_edge_decay.classify_horizon` already keys on the slug for exactly this
reason). The reliable micro-crypto signal is the slug: `updown`, `up-or-down`, or the
`<coin>-updown-5m-<unixts>` shape.

Real-world lifespan is genuinely long where metadata is trustworthy: the Trump market reports
`startDate=2024-01-04 … endDate=2024-11-05` (~10 months).

---

## 3. `/trades?market=<conditionId>` — the tape

### 3.1 Filter correctness — exact

Pulling `will-colombia-win-the-2026-fifa-world-cup` (vol $104.7M, conditionId
`0xe99cc59f…60e771`) across 21 pages / 10,500 rows: **distinct conditionIds in results = 1,
wrong-market rows = 0.** The `market=` filter is exact. Trade row schema (per fill) is identical
to the global `/trades` feed:
```
asset, conditionId, outcome, outcomeIndex, price, proxyWallet, side, size,
timestamp, title, slug, eventSlug, transactionHash, name, pseudonym, bio, icon, profileImage…
```
so the existing `fold_trades_to_ledger` column mapping and the
`(transactionHash, proxyWallet, asset, side)` dedup key apply unchanged (DECISIONS.md
"Duplicate trade rows").

### 3.2 Offset cap — same 10,000 ceiling as the global feed

```
market=<mega>&limit=500&offset=9500  → 200 (500 rows)
market=<mega>&limit=500&offset=10000 → 200 (500 rows)
market=<mega>&limit=500&offset=10500 → 400 {"error":"max historical trades offset of 10000 exceeded"}
```
The cap is on the **offset value** (must be ≤ 10000), giving up to ~10,500 retrievable rows per
market. High/mid-volume markets exceed this: the $104.7M World-Cup market and a $4.5M
swing-state market both **hit the 400 at offset 10500** after 10,500 rows. So the spec's §1.2
correction stands: for the biggest markets you get only the most-recent ~10.5k trades. Markets
below the ~10k-trade threshold return a **short/empty final page** and terminate cleanly (same
loop shape as `ingest`/`backfill`) — no complete-tape example was captured in-probe (the
ascending long tail is mostly zero-volume, never-traded markets), but the termination path is
the identical, already-tested newest-first loop.

### 3.3 Breaks the 1.4h ledger ceiling — confirmed

Measured tape spans (max_ts − min_ts of the retrievable rows):
```
$104.7M World-Cup market : 7.12 days
$4.5M swing-state market : 7.83 days
```
vs. the ~1.4h global-poll ledger. Real-world market tapes span days, so the hour-to-day
copyability (metric B) and recency/regime (metric D) analyses that were unmeasurable on the
old ledger become measurable — subject to the near-resolution recent-slice bias on the very
largest markets (§3.2).

---

## 4. Design consequences baked into `src/discover.py`

1. **Enumerate volume-ranked via offset paging 0…~2300** (`/markets?closed=true&order=volumeNum&ascending=false&limit=100&offset=N`), stopping cleanly on the 422 offset ceiling. Also enumerate `closed=false` for open-market early-tape capture. Keyset deep-pagination deferred (logged, not silently truncated).
2. **Filter `volumeNum >= min_market_volume` client-side** (no server-side filter).
3. **Classify each market by slug/question keyword** (`classify_market`), micro-crypto detected by the `updown` / `up-or-down` / `<coin>-updown-5m-` slug shape; **never** by metadata lifespan. Record every enumerated market's category in `market_universe.parquet` (hides nothing, per CLAUDE.md); by default pull tapes for **non-micro** markets (a targeting choice for the discovery ingest, re-runnable for micro if wanted).
4. **Pull each market's tape via `/trades?market=`**, newest-first, per-market overlap-safe cursor, stopping at the cursor / short page / offset-cap 400 — identical discipline to `backfill.fetch_user_trades`.
5. **Write to a SEPARATE discovery dataset** under `data/interim/discovery/` (compressed Parquet) — the shared `bet_ledger.parquet` is left untouched (it is being rebuilt by another process); merging discovery trades into the main ledger, and resolution enrichment against the shared read-only resolutions cache, are trivial documented follow-ons.
