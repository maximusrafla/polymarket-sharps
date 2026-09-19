# DECISIONS.md — modeling choices and why

This file records defaults picked where CLAUDE.md left a modeling choice open,
plus things discovered empirically while building against the real API that
shaped the design. Updated as each pipeline stage was built.

## Data source (src/ingest.py)

Two public, unauthenticated Polymarket REST APIs, no key required:

- **`https://data-api.polymarket.com/trades`** — per-fill trade history:
  wallet (`proxyWallet`), market (`conditionId`), outcome token (`asset` =
  CLOB token id), `side` (BUY/SELL), `price`, `size`, unix `timestamp`.
- **`https://clob.polymarket.com/markets/{condition_id}`** — ground-truth
  resolution: `closed` flag and a `tokens` list, each with `token_id`,
  `outcome`, settlement `price` (0 or 1), and `winner` (bool).

Rejected: the Goldsky subgraph (requires picking/tracking a subgraph deployment
ID that moves over time, more moving parts for a Chromebook cron job) and raw
Polygon RPC log scraping (would mean re-deriving fills from CTF Exchange events
by hand — far more code and RPC load for no benefit over the two REST APIs
above, which already do that work).

**Hard gate (passed):** verified against "Will Donald Trump win the 2024 US
Presidential Election?" (`condition_id`
`0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917`): resolved,
"Yes" token settled at price 1.0 (matches reality — Trump won), and sample
trades priced in `[0,1]` with timestamps on 2024-11-06, the day after election
day, consistent with the market converging to near-certainty once the race was
called. Reproducible via `python -m src.ingest --sanity-check`.

## `/trades` has no working time filter, and a hard 10,000-row offset cap

Empirically verified while building:
- `after=`/`before=` query params are **silently ignored** — passing
  `after=9999999999` (a timestamp in the far future) returned the same rows as
  no filter at all.
- `offset` is capped: `offset=10500&limit=500` returns
  `400 {"error":"max historical trades offset of 10000 exceeded"}`. So a single
  poll can only ever see the most recent ~10,000 trades platform-wide — there
  is no way to page an offset-based cursor deeper into history across runs,
  because `offset=N` points to a different historical trade every time new
  activity prepends the live feed.
- At current volume, one full 10,000-row poll spans roughly **5-10 minutes**
  of real time (it was ~7.5 minutes when measured during the build — the feed
  is dominated by 5-minute-cycle BTC/ETH/HYPE "up or down" markets that trade
  very fast).

**Decision:** `ingest.py` is a rolling-window poller, not a backfill tool.
Each run pages from offset 0 until it (a) overlaps with the previous run's
cursor, (b) hits the API's offset ceiling, or (c) reaches
`ingest.backfill_lookback_days`. Consequence: **it must run frequently enough
that consecutive polls overlap**, or trades in the gap are permanently missed
(there is no way to retrieve them later — they fall outside the next poll's
10k-row window once enough new trades have prepended). At current volume that
means roughly every 5 minutes, not once nightly as the original SETUP.md
example crontab suggested — that example has been updated accordingly. The
compact bet ledger (`data/interim/bet_ledger.parquet`) still accumulates every
resolved bet forever across runs; only the bulky raw per-poll payload
(`data/raw/trades/batch_*.parquet`) rolls off (`ingest.raw_batches_to_keep`).

Because gaps are possible if the box is offline for a while, wallet sample
sizes should be read as "resolved bets we observed," not "resolved bets the
wallet ever made" — this is inherent to polling a fast, filterless public feed
and is called out again in report.py's output.

### ⚠️ `/trades` pages are NOT time-ordered — and the poll cadence was losing trades (measured 2026-07-23, FIXED 2026-07-23)

Two findings from the first steady-state check on the VM after the migration.
**Both are fixed** — see "Fixes" at the end of this section; the diagnosis below
is kept because it is the reason ingest is shaped the way it is.

**(1) The global feed is not globally sorted by timestamp.** Sweeping all 20
pages of the live feed in one pass: **12 of 19 page boundaries are inversions**,
with excursions up to ~4 minutes (`offset=0` held trades at 01:13:21 while
`offset=500` held 01:17:22). Rows are tightly clustered *within* a 500-row page
(~15-25 s) but the pages themselves arrive out of order — presumably a sharded
backend. The full 10k window still spans a coherent ~8.3 min; it is only the
*ordering* that is unreliable.

This breaks an assumption in `fetch_new_trades`, which sets `stop=True` on the
**first** trade older than the cursor floor and abandons the remaining pages. With
unordered pages it can terminate mid-window while newer, never-seen trades sit at
deeper offsets. It is not biting while the cursor floor is old enough that all 20
pages clear it (today's cap-hitting runs), but it becomes the dominant leak the
moment the cadence is fixed and the floor gets recent. **Fix: sweep all 20 pages
and filter by timestamp; never break early on a stale row.**

**(2) Consecutive polls no longer overlap — trades are being lost now.** The three
retained raw batches show the newest trade of one poll and the oldest of the next
separated by a **gap**, not an overlap:

| batch | rows | trade-time span | vs. previous |
|---|---|---|---|
| 00:48 | 7,000 | 00:42:36 – 00:48:03 | — |
| 00:57 | 10,000 | 00:50:31 – 00:57:49 | **+2.5 min gap** |
| 01:07 | 10,000 | 00:59:25 – 01:07:52 | **+1.6 min gap** |

≈20% of platform trades are never seen. Cause is arithmetic: feed depth is
**~8.3 min**, but the *effective* poll period is **~10 min** — a single ingest
takes 5-8 min on the e2-micro and still holds the writer lock when the next
`*/5` tick fires, so that tick SKIPs. 10 − 8.3 ≈ the measured gaps.

Where the 5-8 min goes is **not** raw I/O: reading and rewriting the whole
4,727,576-row ledger on the VM benchmarks at **27 s / 705 MB peak**. The cost is
the per-batch work inside `stream_merge_ledger` — a resolutions merge plus an
object-dtype `MultiIndex.isin` superseded check, repeated across all 24 batches —
plus up to 500 resolution GETs. **Fix: get ingest under 5 min.** The highest-
leverage form is the "v1 incremental ledger write" the VM migration notes already
anticipated — ingest appends to a small delta parquet and the nightly recompute
folds it in — which takes ingest to ~1 min, restoring a 5-min cadence against an
8.3-min window.

Note the SKIP line in `cron.log` read "backfill running"; in practice the lock
holder is almost always the *previous ingest*, which made the overlap failure
easy to misread as healthy. (The message now names both possibilities.)

#### Fixes (2026-07-23)

**1. Incremental ("delta") ledger write — the `v1` from the VM migration notes.**
`ingest` no longer rewrites the ledger at all. It appends the run's new rows as
one small parquet part under `data/interim/ledger_delta/` (`append_ledger_delta`,
O(new rows), ~a second) and the nightly recompute folds the parts in as its first
step, under the writer lock, before anything reads the ledger
(`python -m src.ingest --fold-delta` → `fold_delta_into_ledger`). The fold is the
*same* `stream_merge_ledger` as before, so nothing about the merge semantics
changed — verified **byte-identical on the real 4,695,081-row ledger**
(old inline merge vs delta round-trip + fold: same sha256, 354,721,605 bytes) plus
7 new unit tests.

Details that matter:
- Parts are named `part_<zero-padded max trade ts>_<pid>.parquet`, so
  lexicographic order is chronological (later part supersedes earlier on a
  duplicate key, matching run-by-run `keep="last"`), and two writers can never
  collide on a filename. Written atomically like every other ledger write.
- The fold groups parts into passes of at most `ingest.delta_fold_rows_per_pass`
  rows (default 300k): a day of parts can be ~1M rows, and only the group — not
  the pending backlog — is held in memory, so the 1 GB host is safe. Extra passes
  just cost another ledger rewrite (~27 s at 4.7M rows).
- Parts are unlinked only after the replaced ledger is in place, so an interrupted
  fold re-folds rows that dedup to the same result (idempotent).
- The fold runs even with nothing pending: `refresh_ledger_resolutions` used to
  ride along on every ingest and still has to happen nightly.
- `load_ledger()` prints a NOTE when unfolded parts exist, so an ad-hoc read
  between the 5-min ingest and the nightly fold can't silently under-report.
- Guard: ingest warns if more than `ingest.delta_warn_rows` (default 5M) rows sit
  unfolded — i.e. the nightly recompute has stopped running.

**2. Pager sweeps the whole window.** `fetch_new_trades` no longer breaks on the
first stale row; it sweeps every page the API serves (offset ceiling, or a
short/empty page = end of feed) and filters purely on timestamp, de-duplicating
keys within the sweep because the feed shifts underneath it between page requests.
It still short-circuits on the 400 offset-cap and on a short page — those are real
end-of-feed signals, not ordering assumptions.

**Verified on the VM (2026-07-23, right after deploy).** Ingest wall time
**5.4 min → ~48 s**; every `*/5` tick now runs (no more alternating SKIPs), each
picking up ~5,000 fresh trades. Consecutive raw-batch spans went from
**+2.5 / +1.6 min gaps** to **+0.5 min / exact boundary overlap** — and a
read-only control sweep of one continuous 10,000-row window (7.6 min deep) shows
the live feed itself contains quiet slots of **52-54 s**, so a sub-minute hole
between polls is the feed's own burstiness, not loss. Captured volume rose from
~1,000 to ~1,060 rows/min. The nightly fold of 21,907 delta rows into the
4.74M-row ledger took **3m47s** and landed exactly +21,907 rows.

Residual known gap (accepted, not fixed): the cursor floor is the newest timestamp
of the last sweep, so a trade that only *enters* the feed after our sweep with an
older timestamp (the sharded backend's ~4-min inversion window) is still filtered
out next run. Fixing it means a lagged floor plus a per-key memory of the lag
window, which would also make the raw-batch spans overlap by construction and
destroy the cheapest health signal we have. Revisit only if the batch spans show
loss again after the cadence fix.

## Per-wallet backfill (src/backfill.py) — deepening the samples that matter

The global-feed limitation above caps *discovery*, but not per-wallet *depth*.
Verified empirically (2026-07-18): while `after`/`before` are ignored, the
`user=` and `market=` filters on `/trades` **do** work. `/trades?user=<wallet>`
returns that wallet's own fills, subject to the same ~10k offset cap — but for a
single wallet that ceiling is its (near-)complete history, not a sliver. Two
top-ranked wallets that the global feed had given us ~50 resolved bets each
turned out to have **10,000+** trades retrievable this way.

**Decision:** `backfill.py` deepens the top-`backfill.top_n_wallets` ranked
wallets by paging each one's `/trades?user=` history into the same ledger —
incremental (per-wallet cursor in `backfill_cursors.json`), resumable, additive,
dedup-safe. It is seeded from the current ranking so the wallets we report/watch
get accurate metrics; the global poller keeps discovering new wallets. Because a
deep backfill of active wallets surfaces thousands of new markets at once,
`update_resolutions` takes a `max_fetches` cap (`backfill.max_resolution_fetches_per_run`)
so one run stays bounded and the rest resolve over subsequent runs via
`resolution_recheck`. Run it nightly *before* the recompute (see README cron).

**Why this is the top lever (and a caution about the current ranking).** The
weight-tuning analysis showed forward validity is capped by per-wallet sample
size, not the weights. A smoke test deepening just the top 3 wallets was
decisive: their thin-sample "sharp, persisted" verdicts **did not survive**. On
1,200+ bets wallet #1's in-sample skill edge went slightly negative (it was not
even a candidate — its 25-bet in-sample edge had been noise); wallet #2's edge
was real but tiny and no longer significant (p=0.083); wallet #3's held-out
skill went negative. **So the current thin-sample top ranks should be treated as
provisional until a full backfill runs.** This is the intended payoff: more bets
per wallet turns noisy sign-flips into verdicts the significance test can
actually stand behind. (It also surfaced a modeling nuance for later — some
wallets are non-stationary across the first/second-half split, e.g. #1 was
negative-then-positive; the chronological select/validate split treats that as
"not a candidate," which is conservative but worth revisiting.)

### Memory-safe wallet batching (src/backfill.py) — added 2026-07-20

The corruption-recovery rebuild deepened the top-500 via *repeated* incremental
runs (cursors advance across process restarts, so each run only re-fetches a
little). But `run_backfill` itself accumulated **every** wallet's full
`/trades?user=` history in one in-memory `all_new` list before folding, so a
single **from-empty** `python -m src.backfill` at `top_n_wallets=500` would hold
millions of raw trade dicts at once and OOM the 3.7 GB box. The memory-safety
lived only in the *operational pattern* (restart-and-resume), not in the module.

**Decision.** `run_backfill` now processes wallets in batches of
`backfill.wallet_batch_size` (default 25). Each batch is fetched, **folded into
the ledger, and its raw dicts freed** before the next batch is fetched — so peak
memory is bounded by the ledger plus *one* batch of trades, regardless of run
size. The *fold* cadence (per batch, what bounds memory) is deliberately
**decoupled** from the *save/checkpoint* cadence (expensive: the ledger is a
~250 MB gzip'd parquet): the ledger + cursors are persisted only once every
`backfill.checkpoint_min_rows` newly-folded rows (default 100 000), plus a
mandatory final save. Consequences:
- A long **from-empty** run checkpoints ~every 100 k rows, so a crash costs at
  most that much re-fetch (and, thanks to atomic writes + the dedup on every
  save, never corruption). Cursors are written *after* the ledger, so a kill
  between the two leaves cursors *behind* the persisted ledger — a re-run
  re-fetches and the dedup absorbs the overlap; a cursor is never advanced past
  trades not yet in the persisted ledger.
- A **caught-up nightly** run folds far fewer than 100 k rows and so saves
  exactly **once at the end**, exactly as the pre-batching code did — no
  per-batch I/O regression. (Verified 2026-07-20: a real nightly-shaped run
  folded 68 126 rows across 20 batches with **zero** mid-run checkpoints.)

Residual ceiling: holding the full ledger (~3.1 M rows, ~2 GB resident) plus the
transient copy during concat/dedup is the remaining memory floor — the same
~3 GB ceiling the `features` stage already hits and survives on this box (4.6 GB
swap is the backstop). The batching removes the *unbounded* term (the raw dicts),
not the ledger itself. `--batch-size` overrides the config per run if memory is
tighter.

## `features`/`validate` memory footprint — the free e2-micro RAM wall (2026-07-22)

Context: migrating the pipeline off the 3.7 GB Chromebook (which began OOMing once
the ledger passed ~4.7 M rows) onto a free GCP e2-micro (**1 GB RAM**, 30 GB
*standard* disk, 16 GB swap).

At 4.7 M rows the `features` stage peaked at **~5.4 GB RSS** — too big for both
boxes. Profiled dominant costs, and the fixes (all verified **BYTE-IDENTICAL** on
the full 4.7 M-row ledger via a saved baseline + a per-function sample harness;
183 tests green):

1. **manufactured_flag pre-filter.** `compute_manufactured_record_flag` did
   `groupby("tx_hash")` over 4.7 M **unique** tx_hashes (unique per row on this
   data), iterating millions of *singleton* groups in Python — every one skipped by
   the `len(wallets)!=2` guard. Now `duplicated(keep=False)` restricts the groupby
   to tx_hashes on ≥2 rows first. Exact (singletons contribute nothing to `edges`),
   but skips the millions-of-groups loop. Top time+memory sink; was the pathology
   the "vectorize manufactured_record_flag" backlog item warned about.
2. **Column projection.** `common.load_ledger(columns=…)` pushes a projection to the
   parquet reader (falls back to a full read if a column is absent). `features`
   loads only the 10 columns it uses — dropping ~480 MB of `outcome`/`question`/
   `slug` that it never reads; `validate` loads only 6 (in the nightly path where
   features are cached; the standalone fallback still loads the full set to compute
   features).
3. **Categorical dtypes.** `market_id`/`token_id`/`side` → category in
   `features.main` (ledger 2.0 GB → 883 MB, compute **3.5× faster**). `wallet` is
   deliberately left `str` — it is the output index and a groupby key in nearly
   every metric, and keeping it str avoids categorical-index alignment subtleties on
   the join back to `all_wallets`. `_factorize_pair` makes `compute_forward_drift`
   dtype-agnostic: `pd.factorize` + `np.asarray(uniq)` — a *categorical* factorize
   returns *sorted* uniques, which must be flattened so the bet side maps in
   appearance order matching the trade side (otherwise codes diverge, Δ≈0.99).
4. **One-pass multi-window drift.** `compute_forward_drift_multi` builds the
   trade-side argsort + prefix sums once and computes both windows (copy_window +
   earliness) from them; `compute_forward_drift` is now a thin wrapper (API kept for
   the tests/validate that call it). `BIG` is sized for the larger window — it only
   scales the composite keys without changing their (token, ts) ordering, so each
   window is identical to computing it alone. Halves the drift cost and collapses
   two swap-thrashing peak phases into one.

**Net on the Chromebook (fits in RAM there): peak 5.4 GB → 2.8 GB, `features`
440 s → 105 s** — this alone fixes the Chromebook OOM that triggered the migration.

**The wall, and how it was cleared (2026-07-22).** The 2.8 GB peak still exceeded the
e2-micro's **1 GB RAM**, so ~1.8 GB lived in swap — and the free tier's **standard**
persistent disk has *very* low IOPS, so the argsort/prefix-sum random-access pattern
against swapped memory was catastrophic: measured **~98.5 % I/O-wait, 58 min elapsed
for 54 s of CPU** (≈2 h for `features` alone). A Balanced/SSD disk would fix the IOPS
but is not free-tier, so swap was a dead end; the cure had to be a peak **under ~1 GB**.

Two measurements shaped the fix, and both are counter-intuitive enough to record:
- **glibc never returns freed memory to the OS**, and `ru_maxrss` is a high-water
  mark. So *freeing* a column after allocating it cannot lower the peak — only
  *never allocating* it can. Dropping `tx_hash` after use, and dictionary-encoding at
  load, were both byte-identical and both moved the peak by <5 %.
- Peak RSS is set by the **largest momentary allocation**, so the win came from never
  holding the whole ledger, a ~93 % `only_buys` copy, and the drift arrays at once.

### The streaming rewrite (`compute_wallet_features_streaming`)

`features` now runs as four column-projected passes, each loading only what its
metrics read and freeing before the next (with `malloc_trim` in between). The peak
becomes the largest single pass, not the sum:
1. **edge metrics** — wallet/side/resolved/prices/timestamp.
2. **manufactured_record_flag** — only rows *sharing* a tx_hash can form a
   counterparty edge, so Arrow is asked whether any tx_hash repeats at all
   (distinct-count vs row-count). On this ledger all 4.7M are unique, so the pass
   short-circuits instead of loading six columns (~1.5 GB) and hashing 4.7M strings
   (a further ~880 MB spike — it had been setting the whole stage's peak).
3. **market-scoped** — breadth, market_windows, pattern_flag.
4. **forward drift** — read straight from Arrow into numpy (never building the
   ledger/bets frames, ~1.3 GB saved), with the bet side processed in batches: each
   bet's fair value is an independent searchsorted + prefix-difference, so batching
   is byte-identical while shrinking ~20 full-length per-window intermediates from
   ~700 MB to ~50 MB. Results group by wallet *code* and relabel the 23,956 unique
   wallets at the end rather than carrying 4.4M wallet strings as a groupby key.

Byte-identity across all of it rests on one trick worth remembering: `pd.factorize`
on Arrow's dictionary **codes** returns exactly the codes it would return on the
strings (the codes are a 1:1 relabeling), so sort order and prefix sums are
unchanged. Deriving codes the obvious way instead — `pd.factorize` on a *categorical*
column — silently returns **sorted** uniques while the ledger side is in appearance
order, which diverged results by ~0.99; `np.asarray(uniq)` flattening is what makes
it exact (see `_factorize_pair`).

### `ingest` (`stream_merge_ledger`) + the zstd codec

`ingest` rewrites the whole ledger every run (it also *refreshes resolutions* on
existing rows, so it is not append-only). It now streams: read a batch, apply the
row-wise resolution refresh, drop rows superseded by new keys, write a temp parquet,
atomically replace. Equivalence rests on an invariant verified against the live
ledger — it is already unique on `(tx_hash, wallet, token_id, side)` (0 duplicate
keys over 4,695,081 rows) — so `keep="last"` only ever means "a new row supersedes an
identical-key old row", and dropping superseded old rows then appending the
self-deduped new rows preserves both result and row order.

Ledger codec **gzip → zstd** (`common.LEDGER_COMPRESSION`): a full rewrite is
**192.7 s → 20.8 s (~9x)** *and* the file is smaller (395 MB → 354 MB). Since ingest
rewrites the ledger every run, write cost is what decides whether the 5-minute poll
fits its cadence. Readers are codec-agnostic, so existing gzip ledgers still load.

**Result — the free e2-micro now runs the whole pipeline (cron enabled 2026-07-22):**

| stage | VM peak RSS | VM wall | note |
|---|---|---|---|
| `ingest` (5-min) | 767 MB | 5.9 min | was impossible (~2 GB load) |
| `features` | 773 MB | 26 min | was ~2 h / never finished |
| `validate`→`rank`→`report` | — | 28 min | validate is now the slowest stage |

All outputs byte-identical to the pre-optimization pipeline (full 4.7M-row baseline
diff, worst float delta 0.0; 183 tests). The nightly recompute holds the writer lock
for its whole duration so a 5-min ingest can't run concurrently and blow RAM.
Remaining lever: `validate` has not had the streaming treatment (it still loads the
ledger and runs a per-wallet Python loop) — giving it the same shape would cut the
nightly from ~54 to ~35 min.

## Scheduling — cron split + shared writer lock (2026-07-20)

`run.sh` did the whole pipeline (ingest → backfill → features → validate → rank →
report) in one sequential shot, which is fine for a manual run but wrong for the
schedule: **ingest must run every ~5 min** (the global `/trades` feed only serves
the most recent ~10 k trades with no time filter, so consecutive polls must
overlap or trades in the gap are lost forever — see the ingest section above),
while the **recompute is nightly** (`features` peaks ~3 GB RSS; no reason to pay
that every 5 min).

Two cron targets (`scripts/crontab.example`, not auto-installed):
- `scripts/ingest.sh` — every 5 min, just `python -m src.ingest`.
- `scripts/recompute.sh` — nightly, `backfill` then `features/validate/rank/report`.

**The concurrency hazard and its fix.** ingest and backfill are both ledger
*writers*; two writers at once is exactly what corrupted the ledger before. So a
5-min ingest tick firing while the nightly backfill is mid-write must not become
a second writer. Both scripts serialize on a shared `flock`
(`data/interim/.ledger.lock`, gitignored):
- `ingest.sh` uses `flock -n` (non-blocking): if the nightly backfill holds the
  lock, the tick **skips** rather than double-writing. A skipped tick is harmless
  — the next tick's rolling window still overlaps the last successful poll.
- `recompute.sh` holds the lock only around **backfill** (the writer); the
  read-only `features/validate/rank/report` stages run **unlocked** (atomic
  writes mean a reader always sees a complete ledger), so the ~20-min re-rank
  doesn't freeze the 5-min ingest cadence.
- `run.sh` (the manual full run) routes its ingest and backfill through the same
  lock too, so a manual run can never collide with a live cron tick.

Writes are atomic regardless, so the lock is a *liveness/politeness* guarantee
(don't waste work, don't interleave two writers), not the corruption safety net —
that's `common.atomic_*`. `data/cron.log` collects all cron output and is rotated
nightly by `recompute.sh` (cap ~5 MB) so it can't grow unbounded on the tight
eMMC.

## Atomic persistence (src/common.py) — added after a corruption incident

During the top-500 deep backfill (2026-07-20) the bet ledger was corrupted: a
backfill process was killed mid-write and `save_ledger` used
`df.to_parquet(path)`, an **in-place** write. Parquet writes the footer last, so an
interrupted in-place write leaves a footer-less, unreadable file — and since the
save overwrites the only copy, the ledger was destroyed (recovered only because the
separate `resolutions.parquet`, written earlier in the same run, survived).

**Decision:** all durable state now writes **atomically** — `common.atomic_to_parquet`
and `common.atomic_write_json` serialize to a `*.tmp<pid>` file in the same directory,
then `os.replace()` it into place (atomic rename on one filesystem). A crash/kill
mid-write leaves the previous good file intact plus a stray tmp, never a half-written
target. `save_ledger`, `save_resolutions_cache`, and both cursor writers use them.
Operational rule that falls out of this: the read→transform→write pipeline stages
(`features`/`validate`/`rank`/`report`) only *read* the ledger, so they are always safe
to kill; only the *writers* (`ingest`/`backfill`) must not be interrupted mid-save
(and now even that only costs a re-run, not corruption).

## Parallel resolution fetch + the CLOB rate limit (src/ingest.py)

`update_resolutions` fetches each market's resolution with an independent, read-only
GET, so it was parallelized with a small thread pool (`ingest.resolution_fetch_workers`,
default 5). Empirically the CLOB `/markets/{id}` endpoint **rate-limits aggregate
throughput to ~20–22 req/s regardless of concurrency**: a clean burst of ~120 requests
at 8 workers returns 0 errors, but a sustained burst throttles hard (workers 3/4/6 all
converge to ~20 markets/s once the limiter engages; 12+ workers just draw 429s). So 5
workers is polite and near-optimal (~2.5× serial), and the session's existing
`Retry(429, backoff)` absorbs stragglers. Verified the concurrent path returns
byte-identical output to the serial one. `workers <= 1` restores exact serial behavior.

## Duplicate trade rows from the API

Observed: within a single `/trades` page, the same `(transactionHash,
proxyWallet, asset, side)` combination sometimes appears 2-3 times with
**byte-identical** `size`/`price`/`timestamp` (verified: in one 10,000-row
poll, 1,195 groups repeated this way, and in every single one all duplicate
rows matched exactly on size/price/timestamp — none differed). This looks like
the feed re-emitting the same fill rather than distinct partial fills (which
would be expected to differ in size across separate maker matches).
**Decision:** dedupe on `(transactionHash, wallet, token_id, side)`, keep last.
If this assumption is ever wrong for some future trade shape (a truly distinct
partial fill sharing all four keys), it would slightly *undercount* size —
flagged here as a known limitation rather than silently risking the reverse
(double-counting, which would inflate edge/volume).

## Gamma markets API query filters don't work as documented

`https://gamma-api.polymarket.com/markets?conditionId=...`,
`?condition_ids=...`, and `?slug=...` were all tested against known-correct
values and either returned an unrelated default listing or an empty list.
Root cause found for `slug`: the endpoint appears to apply an implicit
`closed=false` filter unless `closed=true` is passed explicitly, even when the
correct slug is supplied. Given this endpoint's filtering was unreliable in
testing, resolution lookups use the single-market CLOB endpoint
(`clob.polymarket.com/markets/{condition_id}`) instead, which takes the
condition id as a path parameter (no query-filter ambiguity) and returns
authoritative per-token settlement data directly.

## "Bets" = BUY-side fills only (for scoring)

A trade row can be `side=BUY` (opening/adding exposure to an outcome token) or
`side=SELL` (disposing of a token the wallet holds — normally closing/trimming
a previously-opened position, since Polymarket wallets sell tokens they hold
rather than shorting directly). Properly attributing a SELL's "entry" would
require reconstructing per-wallet, per-token position lineage (matching sells
back to the specific earlier buys they close, handling partial fills/FIFO vs
LIFO) — a much larger undertaking than this pass covers.

**Decision:** both BUY and SELL rows are ingested and kept in the bet ledger
in full (nothing is dropped, per CLAUDE.md), but `features.py` only treats
BUY rows as scoreable "entries" for the edge / earliness / copy-window
metrics, since those metrics are inherently about the skill of *entering* a
position. SELL rows remain in the ledger as descriptive data. This is a scope
limitation, not a data-quality flag — documented here so it's not mistaken for
an oversight.

## Fair value at entry / copy window (src/features.py)

CLAUDE.md leaves "fair value at entry" undefined. Full order-book depth
history isn't available from these free APIs, so fair value is proxied from
the trade tape itself:

**`fair_value(bet)`** = size-weighted average trade price for the same market
outcome token, over *other wallets'* trades in the window
`(entry_ts, entry_ts + copy_window_hours]` (default 24h, `scoring.copy_window_hours`).
This approximates "what the rest of the market converged to shortly after,"
independent of the (possibly much later) binary resolution.

`copy_window = fair_value - entry_price`. Positive and large means the wallet
routinely gets in before the market re-prices toward their side — the
"followable" signal CLAUDE.md describes.

### Forward-price leakage fix (audit 2026-07-18 — see HANDOFF.md)

The original proxy leaked the outcome three ways, quantified by an audit
(`scripts/audit_forward.py` in scratch; findings in HANDOFF.md): copy_window
correlated **~0.86 with the realized outcome**. As a forward price approaches
resolution it approaches the 0/1 answer, so an unbounded forward proxy is
look-ahead. The proxy was hardened:

1. **No `resolved_value` fallback.** The old chain fell back to the resolved
   value when a bet had no later trades — injecting the outcome *directly* as
   "fair value" (fired for ~2.4% of resolved bets). Removed.
2. **Strictly window-bounded.** The old "next trade at any horizon" fallback
   could grab a near-resolution price arbitrarily far out. Removed; only trades
   inside `(entry_ts, entry_ts + window]` count.
3. **Resolution guard.** Trades in the final `scoring.fair_value_resolution_guard`
   fraction (default 0.2) of a token's observed lifespan are excluded, so
   near-resolution prices can't stand in for fair value.

No qualifying trade → `NaN` (unknown, contributes 0 and is down-weighted), never
the outcome. After the fix, copy_window's correlation with the realized outcome
fell from ~0.86 to ~0.56 (the remainder is plausibly genuine early-drift signal),
and ~3.3k wallets that only had near-resolution "fair value" now correctly carry
`NaN` rather than a leaked one.

### `earliness` coincides with `copy_window` (not scored separately)

`earliness` uses the same forward-drift proxy at a shorter horizon
(`scoring.earliness_window_hours`, default 6h). CLAUDE.md intends copy_window
(24h, "where the crowd converged") and earliness (6h, "did price move their way
right away") as two horizons of one signal. But in this data source, **all
per-token trades cluster inside 6h**, so the two windows capture identical trades
and earliness equals copy_window for 100% of wallets (corr 1.0). Scoring both
would double-count one signal, so `rank.py` scores **copy_window only**
(`w_earliness` is 0 / unused). Earliness is still computed and reported for
transparency; re-introduce it to the score only if a richer trade history ever
makes the two horizons genuinely diverge.

### `copyable` flag (`ranking.copyable_window_floor`, metric B)

Skill and *followability* are independent. A wallet can beat the price it paid by
a real, significant, economically-meaningful margin (high skill edge) and yet have
a **non-positive `copy_window`** — its edge realizes at/near resolution rather than
drifting into the forward window, so a copier entering after it has no room to
capture. The deep backfill found several such wallets among the persisters (one at
`copy_window` −0.19). Skill edge (resolution outcome vs. price paid) and copy_window
(forward price drift a follower can ride) are genuinely different quantities: e.g.
`0x32ed517a` has only ~1.35¢ skill edge but +0.12 copy_window.

**Decision.** `rank.py` computes a first-class **`copyable`** column =
`copy_window > ranking.copyable_window_floor` (default **0.0** — strictly positive
followable room; raise it to demand more). It is deliberately **kept separate from
skill** (`edge_persisted`) and **not fed into the score** — `copy_window` already
contributes to the score via `w_copy_window`, so `copyable` is purely a sortable
filter the user applies, consistent with CLAUDE.md's "flags never alter rank order"
rule. `report.py` surfaces it per wallet and as an "Actionable set" summary
(persisted ∩ copyable = the subset for which the copyability question is even
well posed). On the current ledger 11 of the 14 persisters are copyable (all of the top
10, deep n≥768); the 3 non-copyable are sharp-but-unfollowable.

## Time-consistency (src/validate.py / features.py)

Each wallet's resolved BUY bets are split chronologically into
`scoring.time_consistency_buckets` (default 4) equal-count buckets.
`time_consistency = fraction of buckets with positive mean edge` (0 to 1)
rather than, say, 1 minus a coefficient of variation — chosen for
interpretability (directly readable as "in 3 of 4 periods this wallet was
profitable") and because it's insensitive to edge's scale, unlike a
CV-based measure which blows up near zero mean edge.

## Favorite-longshot residualization / "skill edge" (src/features.py)

**Why raw edge alone is not skill.** A 2026-07-18 red-team of the persistence
rate (see HANDOFF.md; reproduce with `scripts/audit_persistence.py`) found that
`resolved_value − entry_price` is dominated by a structural market bias, not
forecasting skill. In this data the market mean resolution (~0.62) exceeds the
mean entry price (~0.60), so ~62% of *all* bets have positive raw edge by
default, and edge is strongly price-dependent: buying favorites at 0.6–0.8 earns
~+0.14/bet while longshots at 0.2–0.4 lose ~−0.18/bet. A wallet's raw edge
therefore mostly reflects *which price band it habitually buys* — a persistent
per-wallet trait that trivially survives an out-of-sample split. A shuffled-
outcomes null test confirmed it: with outcomes randomized, the old validator
still reported ~62% persistence (real was ~69%), i.e. it was measuring the base
rate, not skill.

**Decision.** Fit a market-wide calibration curve `E[resolved_value | entry_price]`
over equal-count (quantile) entry-price bins (`scoring.price_baseline_bins`,
default 20) across all resolved BUY bets, and define per-bet **skill (residual)
edge** = `resolved_value − E[resolved_value | entry_price]`: how much the wallet
beat the price it actually paid, net of the free favorite-longshot edge available
to any buyer at that price. A single wallet contributes negligibly to a
market-wide bin, so the curve is fit on the full sample (leave-one-wallet-out
would not move a bin mean). Raw edge is still computed and reported (per CLAUDE.md)
but ranking and validation key on skill edge. Skill edge is *not* followable-alpha-
neutral the way raw edge is: a copier buying the same favorite at 0.7 gets the
structural edge regardless of whom they copied, so only the residual is worth
ranking on.

### Slow-market universe: scoped baseline, `speed_bucket`, and the sort-vs-bit-identity call (Project 3, 2026-07-25)

The Project 3 pivot (`docs/project3_slow_markets.md`) added a slow-market path,
built steps 1→2→4→5 (commits 67704d8→de97701). Decisions worth recording:

- **Per-category price baseline fit *inside* the slow universe**, not inherited from
  the crypto-dominated global curve (`src/slow_validate.py`, step 4). A 0.75 NFL
  favorite is not a 0.75 crypto print, so `E[outcome | entry_price]` is fit per
  category. ⚠️ **Caveat that is now the result's central open question: 67% of slow
  bets classify as `"other"`, so the baseline is coarse.** A sub-niche mispriced
  relative to that coarse average produces a large positive residual "skill edge" for
  *every* buyer in it — the favorite-longshot structural edge one level up, at the
  category level, un-stripped. This is un-separable from genuine skill by disjoint
  data, the cluster null, concentration guards, or the forward test; **only a finer
  baseline (sub-category / category×price) separates them.** Baseline scope is
  deliberately **pluggable** so the finer refit is a config change, not a rewrite. See
  HANDOFF.md "Project 3 slow-market pivot".
  - ✅ **RESOLVED 2026-07-25 (step 6, commit 0301985 —
    `docs/project3_finer_baseline.md`).** The refit was run at niche granularity
    (`src/slow_niche.py` frozen family × form partition, `src/slow_baseline.py`
    hierarchical partial pooling + leave-one-wallet-out) via
    `slow_validate --baseline {category,niche,niche_form}`. **The confound is real in
    kind but ~an order of magnitude too small to matter**: median edge shift −1.0¢,
    52 → 46 persisted, `+0.40…+0.54` band 8/8, and every niche's all-participant
    residual on the *unselected discovery corpus* lands within ±3.5¢ (`geo_iran`
    +1.0¢ over 128,433 bets). Median niche-explained fraction +6.6%.
  - ⚠️ **The refit surfaced a bigger confound: rolling-deadline ladders.** `geo_iran`
    is 11.5% of the slow universe as "US strikes Iran by \<rolling date\>" — dozens of
    markets, distinct resolution days, **one event**. Every existing guard treats them
    as independent. Resampling event complexes (`--cluster niche_l1`) costs 18 of the
    52, three times the baseline question's damage; 5 of the top 8 by edge have their
    whole held-out record in one complex. **Under both controls 30 survive.** Same
    lesson as "cluster-preserving null is required", one level up: the cluster unit
    must be the unit of independent *resolution*, which for a deadline ladder is the
    narrative, not the market. Adopting the complex block as the slow path's standard
    is the recommended next decision.
  - ❄️ **RESOLVED + FROZEN 2026-07-25 (step 7, commit 3123d9c — `docs/project3_freeze.md`).**
    The event complex is now the slow path's **standard cluster unit**
    (`STANDARD_BASELINE=niche_form`, `STANDARD_CLUSTER=niche_l1`, complex gates ON),
    and the concentration floors are re-read in complex units **alongside** the
    market-unit ones — no gate removed, complex columns additive metadata even when
    they do not gate. **Thresholds transposed UNCHANGED** (`min_oos_complexes ≥ 3`,
    `min_eff_breadth_complex ≥ 3.0`, `min_entry_days_complex ≥ 3`,
    `min_resolution_days_complex ≥ 3`): this corrects the unit, it does not re-tune
    the number. Dropping to ≥2 would have returned 19 rather than 12 — the
    "relax until the table fills" failure. **Clean count: 12.** Progression
    52 → 46 → 30 → 12; median top-complex share 0.61 → 0.37. Single-complex wallets
    fail BY RULE (one cluster → NaN bootstrap p → not significant), so the +0.54
    name is excluded mechanically rather than by hand.
  - **Narrative layer = diagnostic, never a gate.** A frozen versioned map above the
    complex measures correlated-complex betting (Iran + Israel + Hormuz = one bet).
    Reported **strict AND broad** because `commodity_crude` spikes on Hormuz risk but
    also trades on OPEC/demand — deciding it by fiat would hide a sensitivity that
    moves the answer. PRIMARY (12): **3.13 effective independent narratives**;
    SECONDARY (30): **2.96**, with 13 of 30 single-narrative. Both cohorts carry ~the
    same ~3 effective narratives, so the wider set is bigger without being broader.
    Deliberately NOT a hard gate: the confound regress does not terminate
    retrospectively, and the honest move is to measure the breadth, state it, and let
    the forward test break the regress.
  - **PRE-REGISTRATION (the decision that cannot be revisited).** Frozen
    2026-07-25T06:32:34Z at commit dcae30d5. **primary (12) is the headline**;
    secondary (30, inclusive) is frozen so its forward data is not wasted but is never
    the headline. The fitted pre-freeze baseline travels inside the manifest, so
    scoring never refits and the forward number cannot drift because the baseline
    moved. **Cohorts are not re-frozen in light of forward outcomes** — choosing the
    cohort after seeing which one worked would reintroduce the exact selection this
    chain exists to eliminate; `freeze` refuses to overwrite without `--force`.
    Accepted power cost, recorded at freeze time rather than offered afterwards: a
    mideast-quiet period may leave the 12 unscoreable for months. Integrity over
    power was the explicit call.
  - 📊 **WIDENED 2026-07-25 (step 8, commit 5abd782 — `docs/project3_forward_tiers.md`).**
    The forward test runs over **five pre-registered tiers** (t12/t30/t52/t87/t699)
    sharing ONE freeze cutoff, because 3 post-freeze bets cannot grade a wallet but
    3×N can grade a crowd. **A tier is a SELECTION rule only — forward scoring is
    identical across all of them.** `t12` stays the headline; wider tiers are
    secondary readings and are never promoted on forward results.
  - **Decision: keep the aggregate statistic EVENT-WEIGHTED, not bet-weighted.** The
    mean over event complexes of the within-complex mean, with the bootstrap
    resampling complexes rather than bets and returning NaN below 2 events. Rationale:
    the slow tape is dominated by rolling-deadline ladders, so a bet-weighted mean
    lets one hyperactive complex become the answer, and a bet-level bootstrap would
    let piling bets into the same complexes manufacture confidence. Pinned by a test
    that 100× the bets in the SAME events does not narrow the CI.
  - **Decision: report `effective_events` beside every bet count, always.** Measured
    at freeze time: **widening buys volume, not narrative breadth** — t699 has 58× the
    wallets and 195× the pre-freeze bets of t12 for only 3.47 vs 3.13 effective
    narratives, and t87/t52 are actually NARROWER than t12 (2.68 / 2.24) because the
    wide crowd is disproportionately mideast_escalation. Forward bet counts will grow
    far faster than forward evidence, and the reporting contract exists so that can
    never be misread. Also recorded: **the tiers are NOT a nested ladder** —
    `tiers_are_nested: false` in the manifest, since legacy t699⊃t87⊃t52 and standard
    t30⊃t12 were selected under different baselines.
  - **Decision: the placebo outranks the baseline as the real test.** Beating the
    frozen calibration curve is weak — a tier can clear it because the markets it
    traded post-freeze were mispriced for EVERYONE (the steps 6-7 niche confound
    arriving forward). So each tier is also scored against non-cohort wallets in the
    SAME markets, plus a **count-matched** random-crowd percentile (crowd size changes
    sampling variance, so an unmatched pool would be rigged). Market-tape fetching is
    capped per run and **any truncation is reported**, never silent.
  - **Amendment integrity.** `freeze --amend` preserves the original `freeze_ts` and
    logs the forward-observation count at the time. The step-8 widening ran at **0
    observations** (`legitimate_pre_registration: true`); `--amend` now refuses because
    647 observations exist, so the widening is permanently on record as pre-registration.
  - ℹ️ The plan assumed Gamma `tags` were persisted by the step-1 CLOB capture and
    could carry the sub-category scheme. **They were not** — the sidecar `tags` column
    is populated for **16 of 348,657** markets (written, never filled). The partition
    is derived from slug/question syntax instead, which is also strictly more
    informative here because it encodes bet *form* (spread vs total vs moneyline vs
    deadline ladder), which tags do not.
- **Scoped `scoring.slow.min_skill_edge = 0.10`**, a separate key from the global
  `scoring.min_skill_edge = 0.02`. Project 1's universe (large-n, small-edge micro
  wallets) and Project 3's (small-n, large-edge slow forecasters) have opposite
  statistical shapes and need opposite floors; the global default is **untouched** and
  Project 1's outputs are verified **bit-identical** by the addition (§5.2 of the plan).
- **`speed_bucket` is additive and the canonical ledger is NOT sorted by it.** §2.2 of
  the plan floated sorting row groups by `speed_bucket` so parquet row-group statistics
  prune fast rows at the file level — but sorting the ledger would change its byte
  layout and break the Project 1 bit-identity guarantee (§5.2). **Resolution: leave the
  canonical ledger row order untouched; do not sort for pruning.** The slow path reads
  via a column-projected filter on the additive `speed_bucket` column; the measured
  pruning benefit did not justify sacrificing bit-identity. (If pruning ever becomes a
  real cost, the fallback is a *separate* Hive-partitioned read copy, never a re-sort of
  the canonical file.)

## Out-of-sample validation (src/validate.py)

Split each wallet's resolved BUY bets in chronological order at
`scoring.oos_split` (default 0.5). *Selection* (which wallets look sharp) uses
first-half (in-sample) **skill edge**; a wallet is a candidate iff its in-sample
skill edge is positive. It **persists** only if its held-out (second-half) skill
edge is both positive **and statistically greater than 0** — a one-sided test at
Project 1's certification alpha (`scoring.project1.oos_significance_alpha`,
**0.005** since 2026-07-26; falls back to the global `scoring.oos_significance_alpha`
= 0.05 when absent), not the old sign-only "> 0" check. The test itself is the
market-block bootstrap, not the t-test — see "Cluster-robust significance" and
"Tightened significance threshold" below. Both raw and skill edges, in- and out-of-sample, are reported side by side
so both luck-decay and the favorite-longshot adjustment are visible.

**Why the significance test + minimum sample.** The old test required only 2 bets
per half and only that the mean be positive. On ~7-bet median samples that is a
coin flip dominated by the base rate above. Now each half must have at least
`scoring.min_bets_per_half` resolved bets (default 10, so ~30 total — consistent
with `min_sample_size`) before persistence is even tested; below that the edges
gate to `NaN` (not zero — zero would misleadingly read as "measured no edge"
rather than "insufficient data") and the wallet is down-weighted via the
sample-size term in `rank.py`, never excluded. After these two changes,
out-of-sample persistence fell from ~69% to ~27% (30 of 110 candidates), while
the shuffled-outcomes null through the *same* validator persists only ~5% (~6
wallets) — so the surviving signal is ~5× the null, versus ~1.1× before. Degenerate
zero-variance held-out halves (e.g. many replicated bets on one shared outcome)
yield a `NaN` p-value and are treated as not significant, which is conservative.

### Economic-magnitude gate (`scoring.min_skill_edge`, metric C)

The full deep backfill (2026-07-19) exposed a failure mode of significance-only
persistence: on ~5,000-bet halves the one-sided t-test certifies edges at
p≈1e-26 that are only ~1¢ in magnitude — statistically ironclad, substantively
meaningless (a 1¢ mean on a price in [0,1] is inside the noise of the tape it is
measured from, and nothing a watcher could act on survives it). Because
`edge_persisted` drives a wallet's reliability multiplier in `rank.py`, these
trivial-but-certain edges were ranking as validated skill.

**Decision.** Persistence now requires a wallet's held-out skill edge to clear an
economic-magnitude floor `scoring.min_skill_edge` (default **0.02**, i.e. 2¢)
*in addition to*, and independent of, the significance test. The two gates are
reported as separate columns — `edge_significant` (positive and t-test
p < α) and `edge_magnitude_ok` (held-out skill edge ≥ floor) — with
`edge_persisted = edge_significant AND edge_magnitude_ok`. The floor is set
*above* 1¢ deliberately, so the "1¢ at p≈1e-26" cases don't rank. Per CLAUDE.md's
no-drop rule this gates only the flag/score, never dataset membership: a
magnitude-failed wallet keeps its row and all metrics, just un-flagged. On the
current ledger this took the persisted set from 21 (significant) to 14 (both
gates); the 7 dropped are exactly the sub-2¢ edges, e.g. one at p=2.3e-6 over
5,062 bets but only 0.5¢ of edge. The floor is a judgement about effect size, not a
measured quantity — it should be re-derived per universe, which is exactly the
mistake the sports arm made (see "`scoring.sports.min_skill_edge = 0.10` was
MIS-SCOPED").

### Cluster-count gate (`scoring.min_oos_markets`) + stable split (IMPLEMENTED 2026-07-23)

A cluster-aware re-check of the persisted set (`scripts/audit_persistence_cluster.py`,
`docs/persistence_cluster_recheck.md`) found the significance t-test miscalibrated:
it treats a wallet's bets as independent, but bets in one market share one
resolution event, so the effective sample size is the number of *markets*, not
bets. Measured median design effect on the persisted set was **1.60** (max 230) —
i.e. t-test p-values too small by ~√D. A `high_frequency_micro_market` wallet with
3,098 held-out bets across 44 markets had t-test p=2e-15 but a cluster-robust p of
0.31.

**Decision (two parts).**
1. **Cluster-count floor.** Persistence now also requires
   `out_of_sample_markets ≥ scoring.min_oos_markets` (default **30**), the
   distinct held-out markets — the effective sample size, parallel to
   `min_bets_per_half` which counts bets. Reported as `edge_markets_ok` +
   `out_of_sample_markets`; `edge_persisted = edge_significant AND
   edge_magnitude_ok AND edge_markets_ok`. Per the no-drop rule it gates only the
   flag/score, never membership; legacy ledgers without `market_id` leave it inert
   (`out_of_sample_markets`=NaN, gate True). On the current ledger this took the
   persisted set from **118 → 70** (the 48 removed all have <30 held-out markets).
2. **Stable split.** `split_in_sample_out_of_sample` now sorts with
   `kind="mergesort"`. The default quicksort is unstable, so a wallet with tied
   timestamps at the split boundary had its halves decided by sort internals — 400
   such wallets on the live ledger, one flipping `edge_persisted`.

**Follow-up: the significance test itself is now cluster-robust (see next section).**
The floor removes too-few-cluster wallets but did not, on its own, fix the t-test
miscalibration *among* wallets that clear it — that is what the cluster-robust
bootstrap below does. The aggregate result still stands: the persisted count is
decisively above a cluster-preserving null (real 118 vs null mean 59, P<0.001),
just at a ~50% implied FDR.

### Cluster-robust significance (`scoring.oos_bootstrap_resamples`, IMPLEMENTED 2026-07-23)

The cluster-count floor above gates on the *number* of held-out markets but still
let the per-bet **t-test** decide significance among wallets that clear it — and
that t-test assumes bets are independent. They are not: bets in one market settle
on one resolution event, so on a `high_frequency_micro_market` wallet the t-test
counts one event dozens of times and certifies a mean that a few markets carry.
Measured design effect on the persisted set: median 1.60, max 230
(`scripts/audit_persistence_cluster.py`).

**Decision.** `edge_significant` is now gated by a **market-block bootstrap**
(`validate._cluster_bootstrap_p`), not the t-test. Per candidate with positive
held-out edge, the held-out residuals are aggregated to per-market (sum, count)
and whole markets are resampled with replacement `scoring.oos_bootstrap_resamples`
times (default 2000); the one-sided `out_of_sample_cluster_p =
(#resamples with mean ≤ 0 + 1)/(n+1)` gates at `validate.certification_alpha()`
(the scoped `scoring.project1.oos_significance_alpha`, 0.005 — see below; it was the
global 0.05 when this shipped). Fewer
than 2 markets → NaN → not significant (one event cannot establish skill). The
bootstrap is **deterministic**: each wallet seeds `default_rng([SEED, hash(addr)])`
so its verdict depends only on its own bets and never shifts as the universe grows.
The t-test p (`out_of_sample_residual_p`) is still computed and reported for
contrast but no longer gates.

`edge_persisted = edge_significant AND edge_magnitude_ok AND edge_markets_ok`, with
`edge_significant` now the cluster-robust condition. On the live ledger this took
the persisted set from **70 → 41** — the 29 removed are exactly the wallets whose
apparent significance rode on clustering (they clear ≥30 markets and the magnitude
floor, but their bootstrap p is ≥ 0.05). 41 matches the audit's independently
derived "~42 defensible" set. Cost: ~+16 s on the full validate (58 s total),
peak RSS unchanged at ~2 GB.

**Scope.** Legacy ledgers without `market_id` fall back to the t-test (behaviour
preserved). **Metric D keeps the t-test**: its regime sub-splits are small and
often span few markets, where a market bootstrap is mostly undefined, and D is
additive metadata that never certifies — so the extra conservatism would only
null out the watchlist for no gain.

### Tightened significance threshold — `scoring.project1.oos_significance_alpha` = 0.005 (IMPLEMENTED 2026-07-26)

**What changed.** Project 1's certification gate now requires
`out_of_sample_cluster_p < 0.005`, not `< 0.05`. The threshold moved to a **scoped**
key, `scoring.project1.oos_significance_alpha`, read only by `src/validate.py` through
the new `validate.certification_alpha()`. The global `scoring.oos_significance_alpha`
stays at **0.05**. Certified set: **41 → 26 wallets.**

**Why.** `docs/persistence_fdr_hardened.md` (2026-07-26) measured, for the first time,
the false-discovery rate of the certified set *under the hardened gate* — 200 shuffles
per bracket, real and null at identical settings, with the harness first asserted
bit-identical to the shipped `wallet_validated.parquet`. It came back at **55%** under
the strict cell-permutation null (80.6% under the weaker within-market bracket), and it
identified the cause: α=0.05 applied per wallet across a **691-candidate** search yields
~35 false positives from multiplicity alone. Significance was never corrected for the
size of the search. The measured trade-off, both arms at 2,000 resamples:

| α | certified | measured FDR (strict null) |
|---|---:|---:|
| 0.05 (was shipped) | 41 | 55% |
| 0.0171 (BH q=0.10) | 36 | 32% |
| 0.0073 (BH q=0.05) | 28 | 23% |
| **0.005 (now shipped)** | **26** | **20%** |
| 0.001 | 17 | 10% |

**Why 0.005 and not 0.001.** Three reasons, in order of weight.
1. **Resolution.** `oos_bootstrap_resamples = 2000` gives a p-floor of 1/2001 ≈ 5e-4, so
   below α=0.001 there are exactly **two** attainable p-values (0.0005, 0.0010): the gate
   would be quantized by the resample count rather than by the evidence, and moving it
   would mean paying 25× the bootstrap cost (50,000 resamples) on a box that is already
   memory-bound. At 0.005 there are ~10 attainable levels, which the audit's 50k-resample
   stability check supports (median cluster p moved only 0.00250 → 0.00220).
2. **It is the calibrated version of BH q=0.05.** BH over the 691-candidate family rejects
   at p ≤ 0.0073 for q=0.05, but the permutation says the bootstrap p is
   anti-conservative ~3× on this data, so the nominal 5% reads as a measured 23%. 0.005
   sits just inside that threshold and measures 20% — i.e. the honest form of "BH at
   q=0.05", chosen against the permutation curve rather than against a nominal guarantee.
3. **Cost per point of FDR.** 0.05 → 0.005 buys 35 percentage points of FDR for 15
   wallets. 0.005 → 0.001 buys 10 more points for 9 more wallets — a much worse rate, and
   it would leave a set too small to power any forward measurement, which is still the
   only real arbiter of copyability. Every FDR here also assumes π₀=1 (nobody has skill), so
   20% is an **upper** bound.

**The chosen point is not knife-edge.** On the live ledger the 78 candidates with a
defined cluster p have a gap in their p-distribution between 0.0035 and 0.0060, so the
certified set is **26 for any α in (0.0035, 0.0060)** — the count does not depend on the
third decimal place. (α=0.003 → 25, 0.0073 → 28, 0.01 → 31.)

**Why scoped, not global.** `scoring.oos_significance_alpha` is read by
`src/slow_validate.py`, `src/slow_forward.py`, `src/sports_forward.py` and
`src/forecaster_metrics.py` — the slow-forecaster and sports arms — and all three of their
**frozen, pre-registered forward tests** (`slow_freeze_manifest.json`,
`sports_freeze_manifest.json`, `sports_sig2c_freeze_manifest.json`) record it as part of
the selection rule they were frozen with. Changing the global would have silently re-tuned
three running experiments. Project 1 therefore gets its own nested block, exactly as
`scoring.slow.*` and `scoring.sports.*` already do; absent the block, `certification_alpha`
falls back to the global key so older configs behave identically. Non-interference is
**proved, not asserted**, in `tests/test_scoped_alpha.py`: `validate_slow` output is
bit-identical with and without a `project1` block carrying an absurd α (for both
`cfg_section="slow"` and `"sports"`, market- and complex-clustered), the live config's
global α and the 2¢ `min_skill_edge` are pinned, and the frozen manifests' recorded α is
asserted equal to what the code would record today.

**What did NOT change.** `min_skill_edge` (2¢), `min_oos_markets` (30),
`min_bets_per_half`, `oos_split`, `oos_bootstrap_resamples`, the ranking weights, and the
flag columns. **Metric D keeps the global 0.05** — it runs on the per-bet t-test, it is
additive descriptive metadata that never sets `edge_persisted`, and it is not part of the
691-wide certification search this correction is aimed at; `regime_flag` / `regime_watch` /
`persisted_recent` are asserted invariant under the change. Per CLAUDE.md **no wallet is
dropped**: all 23,956 keep their rows, metrics and rank positions; only the
`edge_persisted` flag (and through it the `non_persisted_penalty` multiplier) moves.

**Re-verified after the re-run.** `scripts/audit_persistence_fdr.py --shuffles 200 --boot
2000` against the new artifact: the harness reproduces the shipped 26 bit-identically
(691 candidates → 98 cluster-significant → 26 persisted, 0 mismatches), and the measured
FDR is **20.1% ± 0.6%** under the strict null (null mean 5.21, max 12, P(null≥26)=0.000) —
exactly where the curve predicted. Under the within-market bracket the ratio stays 70.3%
(that null holds each wallet's market footprint fixed and so contains a real component),
but its P(null≥real) improved from **0.045 at α=0.05 to 0.000** — the null's max over 200
shuffles is now 25 against a real 26, where at α=0.05 it was 48 against 41.

**Verified artifact-to-artifact.** Diffing `ranked_wallets.parquet` before and after:
23,956 rows and an identical wallet set both times; every measurement column
(`out_of_sample_residual_edge`, `out_of_sample_cluster_p`, `copy_window`,
`in_sample_residual_edge`, `out_of_sample_residual_p`), both flag columns, both other gate
columns and all three metric-D columns (`regime_watch` 53, `persisted_recent` 5) are
bit-identical; `edge_persisted` moved 41 → 26; exactly **15** wallets' scores changed — the
demoted ones, through `non_persisted_penalty`. The three frozen arms' manifests, frozen
sets and forward scoreboards are byte-identical (md5) and their tests pass unchanged.

**How to read the 15 wallets that left the certified set.** They are not judged fake. They
are the wallets whose held-out edge the tightened threshold can no longer distinguish from
chance *given how wide the search was* — their cluster p sits in [0.005, 0.05). They remain
fully in the dataset and the ranking with their metrics intact, and they are recoverable at
any time by sorting on `out_of_sample_cluster_p`. Re-verified after the re-run with
`scripts/audit_persistence_fdr.py` (which reads the same scoped α, so its bit-identity
check against the shipped artifact still holds).

### Metric D — non-stationary / regime-change wallets (IMPLEMENTED 2026-07-20)

**Status: shipped.** Three additive columns on `wallet_validated.parquet`
(`regime_flag`, `regime_watch`, `persisted_recent`), computed in
`src/validate.py::classify_regime`, surfaced by `rank.py`/`report.py`, gated by
`scoring.regime_min_bets` (default 50, the recent-half depth floor for the D3
sub-split) and `scoring.regime_trend_alpha` (default 0.05, the D1 detection
significance). None feed the score or `edge_persisted`. D1 labels a direction only
when a Welch two-sample half-difference AND a Spearman(residual, time) trend agree
at `regime_trend_alpha`; D3 promotes a confirmed recovery to
`regime_flag='improving_confirmed'`. The original design rationale follows.

The chronological select-then-validate gate (`edge_persisted = (in_sample_residual_edge
> 0) AND edge_significant AND edge_magnitude_ok`) has one structural blind spot: a wallet
whose skill edge is **non-stationary** in time. The `in_sample_residual_edge > 0`
candidate gate exists for anti-leakage (select on the first half, confirm on the second)
and MUST stay — but it silently excludes wallets that *became* sharp: the recent
(held-out) half is significant and clears the magnitude floor, yet the early-half
residual is ≤ 0, so the wallet is never a candidate and ranks as unvalidated.

Measured on the top-500 ledger (2026-07-20, read from `wallet_validated.parquet`): of 246
candidates → 36 persisted, but **13 wallets are "became-sharp"** (recent half significant
+ ≥ the 2¢ floor, early residual ≤ 0), **8 of them deep** (recent half ≥ 500 bets) with
recent-half p-values down to 9.4e-26-scale (e.g. `0x5d634050ad`: in-sample −0.024 →
out-of-sample +0.068 over ~3,915 bets each, p=9.4e-24). These are genuine regime changes,
not thin-sample luck, and they are exactly the "sharp right now" wallets the copy product
wants — currently buried at `edge_persisted=False`. Symmetrically, 210 candidates are
"decaying" (early-sharp, recent fails); the gate correctly does not persist them but does
not label *why* (real lost edge vs. early luck).

**Why not just relax the candidate gate.** Persisting on the recent half alone
re-introduces leakage — with ~250 non-candidates, ~α show a significant-positive recent
half by chance, and you would be *selecting* a wallet using the very data you then
*validate* on. From a single split a became-sharp wallet is genuinely un-retro-validatable
(its only sharp data is the data you would have to hold out). So metric D is **detection +
honest routing, not a gate relaxation.**

**Decision (design).** All new columns are additive metadata that never feed the score,
per CLAUDE.md's flag rule; `edge_persisted` semantics and the certified-set rank order are
unchanged.
- **D1 `regime_flag`** (enum, descriptive like `pattern_flag`): `improving` / `decaying` /
  `stable` / `insufficient`, from the sign+significance of (recent − early) residual edge,
  corroborated by a Spearman(residual, chronological order) trend p.
- **D2 routing.** Stationary-sharp → `edge_persisted` unchanged (the 36). Decaying →
  verdict unchanged (not persisted — don't copy a lost edge) but labeled. Became-sharp →
  **`regime_watch=True`**, a first-class sortable watchlist column for forward
  confirmation on NEW data, explicitly **not** `edge_persisted`.
- **D3 within-regime recovery.** If a became-sharp wallet's recent half is itself deep
  enough, sub-split the RECENT half 50/50 and require both recent sub-halves
  significant+material — a leakage-free OOS confirmation *conditional on* the detected
  regime change (the early/recent boundary is used only for detection; the two recent
  sub-halves do honest select/confirm). Such wallets get **`persisted_recent=True` and
  `regime_flag=improving_confirmed`, while `edge_persisted` stays False** — the canonical
  persisted set and its "stationary in both halves" provenance stay intact, and recoveries
  are opt-in via the separate column. The actionable copy set becomes
  `(edge_persisted OR persisted_recent) AND copyable`. D3 only exists because the deep
  backfill made ≥500-bet recent halves the norm (8 of the 13 qualify); it was impossible on
  the old ~7-bet-median samples.

New config: `scoring.regime_min_bets`, `scoring.regime_trend_alpha`; D3 reuses
`oos_significance_alpha` / `min_skill_edge` / `min_bets_per_half`. v1 detects at the
existing 50/50 boundary (zero new split machinery); v2 could add change-point detection
(CUSUM / max-split of cumulative residual) to locate the true regime boundary rather than
assume 50%. Reproduce the evidence with `PYTHONPATH=. python scripts/audit_nonstationary.py`
(read-only; `--deep` adds the per-bet trend + D3 recovery counts). No pipeline code changed
by this design pass.

## `manufactured_record_flag` (src/features.py)

The `/trades` API doesn't label counterparties directly, but every fill's
`transactionHash` is shared by both legs of that match (the taker and the
resting maker fill land in the same settlement transaction). This lets
counterparty pairs be reconstructed from the raw trade tape: group by
`transactionHash`, and where exactly two distinct wallets appear on opposite
sides of the same token, treat that as one observed counterparty match.

**Decision:** for each wallet, compute the share of its total matched notional
against its single most-frequent counterparty. If that share exceeds
`scoring.manufactured_flag_counterparty_share` (default 40%) **and** the
pairing recurs across at least `scoring.manufactured_flag_min_markets`
(default 5) distinct markets, set `manufactured_record_flag = True`. This is a
concentration heuristic, not proof of common control or intent — the column
name and report wording both say "flag," never "insider" or "colluding," per
CLAUDE.md's instruction not to infer intent as fact. It never changes the
wallet's score or rank.

## `pattern_flag` (src/features.py)

One descriptive enum value is populated when it applies, else null:

- `"late_concentrated_entry"` — the wallet's bets, relative to each market's
  own lifespan, land in the bottom `scoring.late_entry_percentile` (default
  10%) of time-to-resolution on average, *and* those late bets are
  larger than the wallet's own median bet size. Purely descriptive of a
  timing/sizing shape, not a claim about why.
- `"high_frequency_micro_market"` — the majority of the wallet's bets are on
  markets with a total observed lifespan under 30 minutes (the 5-minute
  crypto up/down markets that dominate the raw feed, see the ingest section
  above). Added because these bots would otherwise be indistinguishable in
  the ranked table from wallets skillfully picking longer-horizon real-world
  events, even though "sharp on a 5-minute coinflip market" and "sharp on
  election/sports markets" are different skills a user would want to tell
  apart when deciding who to follow.

Both flags are additive metadata columns only, computed after scoring, and
never feed back into edge/rank.

## Combining metrics into a score (src/rank.py)

CLAUDE.md's Scoring and Validation sections read as slightly in tension:
Validation says "only wallets whose edge survives out-of-sample are
reported," while Scoring says every wallet is "ingested, scored, and ranked"
with nothing ever dropped. `validate.py` already resolved this for the
insufficient-history case (NaN out-of-sample edge, down-weighted rather than
excluded); `rank.py` extends the same resolution to the general case: no
wallet is ever excluded from `ranked_wallets.parquet`, but a `reliability`
factor in `[0, 1]` scales every wallet's score down toward (not to) zero when
its out-of-sample evidence is weak, so non-validated wallets naturally sort
to the bottom instead of being removed.

`reliability = sample_confidence * breadth_credit * persisted_multiplier`:
- `sample_confidence = out_of_sample_n / (out_of_sample_n + scoring.min_sample_size)`
  — reuses `min_sample_size` (already documented in config.example.yaml as
  "bets required before a wallet is trusted") as the shrinkage scale, rather
  than adding a second, redundant sample-size config knob. Approaches 0 for
  thin out-of-sample history, approaches 1 as it grows past `min_sample_size`.
- `breadth_credit = min(breadth, ranking.breadth_full_credit) / ranking.breadth_full_credit`
  — soft cap so breadth (CLAUDE.md: "so one lucky event can't dominate")
  down-weights concentrated wallets without needing unbounded market count to
  reach full trust.
- `persisted_multiplier = 1.0 if edge_persisted else ranking.non_persisted_penalty`
  (default 0.3) — a wallet whose out-of-sample edge happened to be positive
  without ever looking sharp in-sample (see `test_wallet_not_a_candidate_when_in_sample_edge_negative`
  in test_validate.py) is exactly the kind of luck the out-of-sample split
  exists to catch, so it's heavily discounted, not fully zeroed (never 0, so
  it never functions as an exclusion).

`edge_score` is the weighted sum of the out-of-sample-validated edge with the
other variables CLAUDE.md names as first-class ranking inputs — `copy_window`
is explicit ("Compute per wallet as a first-class ranking variable"),
`earliness` and `time_consistency` are treated the same way for consistency.
`time_consistency` is centered on its neutral value (0.5) before weighting so
a wallet with unknown/neutral consistency contributes 0 rather than being
penalized relative to one confirmed consistent.

`score = reliability * edge_score`. Weights (`ranking.w_edge`,
`w_copy_window`, `w_earliness`, `w_time_consistency`) and the two down-weight
knobs (`breadth_full_credit`, `non_persisted_penalty`) live in `config.yaml`
so they're tunable without a code change. `manufactured_record_flag` and
`pattern_flag` are deliberately absent from every term above, per CLAUDE.md's
"flags MUST NOT feed into the score" rule — they pass through to
`ranked_wallets.parquet` untouched as metadata columns only.

### Ranking weights: validated against a forward target (2026-07-18)

The weights were originally hand-set. `scripts/tune_weights.py` validates them
against a leakage-free forward target — each wallet's **held-out (second-half)
skill edge**, predicted from features computed on its **first half only** (so the
target is never used to build the predictor). Findings on the current ledger:

- In-sample skill edge **does** predict held-out skill edge — Spearman rho +0.105
  (p=0.026) pooled over wallets with ≥5 bets/half, trending stronger with more
  bets/half (+0.209 at ≥20, +0.236 at ≥30). So the ranking has real, if modest,
  forward validity, and the ceiling on it is per-wallet sample size (the rolling-
  ingest limitation), not the weights.
- **No weight scheme beats another beyond noise.** Bootstrapping the difference in
  forward Spearman between the current weights and alternatives (skill-only,
  skill+small-cw, cw-only) gives a 95% CI that spans 0. Skill edge is the only
  feature with a non-trivial forward coefficient; copy_window and time_consistency
  add ~nothing to *prediction* of future skill.

**Decision: keep the current weights.** They are within noise of every alternative
and already skill-edge-dominant, which matches the one feature that carries forward
signal. Precision-tuning weights on ~170 noisy wallets would overfit. `copy_window`
is retained despite adding little forward prediction because it serves a different,
product-level purpose (CLAUDE.md's "followable" copy-window signal — how much room
a copier still has), not future-skill prediction. Re-run `tune_weights.py` as the
ledger grows; change config weights only if its bootstrap CI ever excludes 0.
`report.py` surfaces the forward Spearman ("Forward validity") so rank *order* is
presented as the coarse skill filter it is, not a precise ordering.

## The sports arm's cluster unit is the RESOLUTION EVENT (src/sports_events.py, 2026-07-25)

**Decision: cluster sports statistics on the resolution event — neither
`market_id` nor `niche_l1`.** Both alternatives were considered and both are
wrong, in opposite directions.

`market_id` over-counts. In the gate corpus, 284 market ids fold to ~25 events:
`will-the-<team>-win-the-2025-nba-finals` is 54 markets that settle on ONE
Finals with mutually exclusive outcomes, and `fifwc-esp-arg-2026-07-19-spread-*`
is 28 markets on one match. A market-block bootstrap reads a wallet's six World
Cup team-outrights as six independent draws. That is the concentrated
single-event artifact that killed Project 2 §1.5 and cut the forecasters 52→12.

`niche_l1` (the league) under-counts, and would have been the natural
generalization of the slow arm's event-complex unit. It is wrong here because
sports is not a ladder population: the ledger holds **25,390 distinct coded
games across 193 league codes**, each resolving independently. Collapsing every
NBA game into one cluster destroys the breadth that makes sports worth testing
at all — the opposite failure, and a silent one, because it looks conservative.

The event unit is derived from **slug syntax only** (never outcomes, prices,
residuals or wallet identity), by two rules: a date-coded slug collapses its
betting lines onto the match (`fifwc-esp-arg-2026-07-19`), and a prose outright
keys on its predicate (`win-2025-nba-finals`) so a whole championship field is
one event while a different season is not. A market matching neither becomes its
own event — permissive, so `event_kind` is reported and callers print the
unmatched share (1.1% of the gate corpus). Versioned as `SPORTS_EVENT_VERSION`
so a frozen sports cohort stays reproducible against its own partition.

**Measured consequence, and the honest part:** re-running the gate at the
corrected unit did NOT overturn its population claim — dispersion excess went
8.32× (from 7.29×) and the split-half null re-centred from +0.072 to −0.004.
The unit correction matters for *per-wallet* certification, not for the
population statistic. Recorded because the opposite was expected.

**Scoping:** thresholds live under `scoring.sports.*`, read only via
`validate_slow(cfg_section="sports")`. `validate_slow` gained two optional
parameters (`complex_col`, `cfg_section`) whose defaults leave the slow arm and
the frozen forecaster cohort bit-identical, rather than forking the stack. The
global 2¢ floor is untouched — asserted by running Project 1's
`compute_oos_validation` with and without an extreme sports block and comparing
the frames.

**Speed buckets are not applied to the sports arm.** The slow arm exists to
isolate slow markets; sports wants every sports market a shortlisted wallet
touched, and fast resolution is the property that makes its forward test
readable in days rather than months.

## `scoring.sports.min_skill_edge = 0.10` was MIS-SCOPED (corrected 2026-07-26)

**The error.** The sports arm inherited a 10¢ held-out magnitude floor from the
slow-forecaster arm. `config/config.yaml:58` states the reason in its own words —
*"same scoped rationale as the slow arm's"* — and that is precisely the bug: the
slow arm's rationale is a **small-n, large-edge** universe, explicitly contrasted
in this file with **large-n, small-edge** universes like Project 1, which uses 2¢.
The deep sports universe is unambiguously the latter: 1,135,069 bets across
40,256 resolution events, median candidate held-out n in the hundreds-to-thousands.
By this repo's own scoping logic the floor was wrong by 5×. Found by the
independent red-team audit (`docs/redteam_audit_2026-07-26.md` §1) and
re-derived independently by the coordinator; every number reproduces exactly.

**Why a magnitude floor on HELD-OUT edge is worse than a mis-set number.** It is a
**winner's-curse selector**. Candidacy is decided on the in-sample half and the
floor is then applied to the *held-out* half, so among wallets of equal true skill
it preferentially certifies whichever record was most upward-noisy out of sample.
The arm's single certified survivor is the textbook shape: **in-sample +0.1¢ →
held-out +18.8¢** on 182 events. Meanwhile it excluded wallets with stationary
in ≈ out records and up to **947** independent held-out events. Ranking the
evidence by breadth rather than by magnitude reverses the outcome entirely: under
a correct BH-FDR step-up across all 174 candidates (the arm's own doc checked only
the rank-1 threshold, which is not how BH works) there are **27 survivors** — the
certified one **fails**, and 6 of the excluded 12 **pass**.

**Rule that generalizes:** an economic-magnitude floor is a statement about the
*universe's* effect size, so it must be re-derived per universe and never copied
across arms. Where it is applied matters as much as its value — a floor on the
held-out half is a selection rule, not just an economic filter, and it interacts
with the split. Prefer floors that a wallet must clear in **both** halves, or
report the in-sample/held-out pair so the winner's-curse shape is visible.

**What was NOT done.** The parameter was not edited and the parent freeze was not
amended. `sports_freeze_manifest.json` has forward observations against it and is
correctly immutable; silently relaxing a live pre-registration is the exact failure
mode the freeze machinery exists to prevent. The correction is **forward-looking**:
`src/sports_sig2c.py` pre-registers the affected cohort as a NEW arm with its own
artifact and its own freeze timestamp (2026-07-26T06:03:11Z, 0 forward observations
at freeze), reusing the parent's fitted baseline verbatim so no refit can drift the
comparison. Parent artifacts verified byte-identical across the freeze.

**Guards on the re-opening**, because relaxing a threshold after seeing the table is
otherwise indistinguishable from table-filling:
- 2¢ is the repo's **pre-existing global** `scoring.min_skill_edge`, in force since
  2026-07-19 — not a value invented for this cohort.
- The wallet-count counterfactual is **smooth** (12 / 4 / 1 / 1 at 2 / 5 / 10 / 15¢),
  so there is no cliff to hunt.
- The justification is **floor-independent**: 37 of 174 candidates have
  event-clustered p<0.05 against ~8.7 expected under a global null, 26 have p<0.01
  against ~1.7. The population excess exists whatever floor is chosen.
- The headline tier applies a **stricter** standard than the parent arm did:
  survival at ≥2¢ with p<0.05 under **all five** saved baseline variants (10 of 12
  qualify), where the parent ran that re-check only for its single survivor.
- `magnitude_ok` is the **only** gate omitted; significance, market count and
  event-unit concentration all still apply, pinned by test.
- The selection-after-the-fact limitation is written into the manifest's own
  `known_limits` rather than left to a reader to notice.

## Real-world deepening is a PROBABILITY SAMPLE, not a shortlist (src/realworld_deepen.py, 2026-07-27)

**The problem it fixes.** Every "the real-world sample is too thin" conclusion in
this repo traces to one number: the shared ledger holds only ~3,192 real-world
wallets, 98 with 1000+ bets. That number is a **collection artifact**. `ingest.py`
is a global-firehose poller and ~82% of the firehose is 5-minute crypto, so
real-world traders barely register no matter how long it runs. Meanwhile
`data/interim/discovery/discovery_trades.parquet` — built **market-first** over
just 585 markets (0.6% of the 101,591 real-world markets in the ledger) — already
holds 542,397 wallets, **10,791 with ≥20 bets**, sitting unused on disk. So the
missing step was never more discovery. It was **deepening**: pulling
`/trades?user=` histories for wallets already discovered, which is what turns a
20-bet wallet into one that `validate.py` can actually test.

**Decision: select on ACTIVITY ONLY, and make it a probability sample.** The slow
and sports arms deepened a *screen shortlist* — wallets chosen for measured edge.
That is the right shape when chasing a named hypothesis, and both arms put a
disjoint-data firewall behind it. But this repo has since spent a lot of evidence
learning how easily a performance shortlist re-imports its own selection noise
(docs/redteam_audit_2026-07-26.md §1; the screening-surface commits 0642b98 /
00c5b5d, where the one "working" screen turned out to be a crypto-frequency
artifact). This arm therefore inverts it:

- The sampler is handed **two columns** — discovery bet count and distinct-market
  count. It never sees edge, profit, win rate or residual skill. A test asserts
  that adding a performance column that would flip any performance-based ranking
  moves **zero** wallets.
- Selection is a **stratified probability sample with known inclusion
  probabilities**, persisted per wallet. So any statistic computed on the deepened
  set reweights back to the full frame (Horvitz–Thompson; a test asserts
  `Σ 1/p == frame size`). This is what the red-team audit asked for when it flagged
  that the existing census is **volume-selected** (median market $7.1M, zero
  weather rows) and therefore cannot speak for the population.
- **A screen evaluated on this population is being tested, not re-measured.** That
  is the entire point: the owner's ROI screen and 2-D screen have only ever been
  run on a thin, non-representative real-world slice.

**The frame and the draw (frozen, seed 20260727).** Frame = discovery wallets with
≥20 bets **and** ≥5 distinct markets = 6,802. The three upper strata are censused
at p=1 (≥200 bets: 397; 100–199: 342; 50–99: 1,172) because they are small and
carry the power; the broad 20–49 band is sampled **589 of 4,891 at p=0.1204**.
Total drawn **2,500**. Per-stratum seed offsets mean resizing one stratum cannot
reshuffle another's draw. Wallets are fetched biggest-history-first, so if the
disk budget binds mid-run the most informative wallets are already in.

**Nothing is dropped** (CLAUDE.md): the whole 6,802-wallet frame is persisted to
`pool_census.parquet` with each wallet's stratum, inclusion probability and
`selected` flag, so the draw is auditable and reweightable rather than implicit.

**Reuse without bias.** 1,472 wallets already have deep histories from the slow
(1,299) and sports (293) arms; **383 of them fall in the draw**. They are flagged
`prior_arm` and **skipped at fetch time**, their data read from those directories
at analysis time. Ordering matters here and is deliberate: the **draw happens
first, the on-disk check second**, so inclusion probability is untouched. Checking
first — "sample from the not-yet-deepened wallets" — would have conditioned the
frame on a performance-correlated variable (those wallets were selected by edge
screens) and quietly biased everything downstream. Cost: 2,117 fetches instead of
2,500, for free.

**Disk is the binding constraint and is budgeted in code, not in a comment.**
4.82 GB free at 81% on a 32 GB eMMC. Measured cost: 0.27 MB/wallet (slow arm,
3,249 trades/wallet) to 0.67 MB/wallet (observed here — the deepest wallets return
exactly 10,000 rows, the per-user offset ceiling). The run therefore calls
`budget_check` at **every checkpoint** and stops cleanly — cursors saved, fully
resumable — the moment free space drops below **2.5 GB** or this dataset reaches
**1.6 GB**. It also refuses to *start* outside budget. The crypto ledger is **not**
deleted for room: it frees only ~290 MB and would destroy reproducibility of the
certified 26 (84% of their bets are crypto).

**Append-only shards instead of one rewritten parquet.** `slow_deepen` /
`sports_deepen` reload the entire deep parquet, concat, dedup and rewrite it on
**every** checkpoint. At 293 wallets that is fine; at 2,117 it is ~85 rewrites of a
file heading for ~1 GB, with the whole multi-million-row frame resident each time —
a real OOM risk on a box that has already been OOM-killed twice (see the
`features`/`validate` memory section). **Decision:** each checkpoint writes a new
`deep_trades/part-NNNNN.parquet` and never touches prior shards, so write memory
and write I/O are **flat in dataset size**. Dedup moves to **read** time
(`drop_duplicates(LEDGER_DEDUP_KEY, keep="last")` across shards), which makes a
re-fetched trade simply land in a later shard where the newer copy wins — the same
semantics the rewrite pattern had, without the rewrite. Resolution refresh is
likewise **shard-wise**, so peak memory there is one shard, not the dataset.

**Isolation.** Writes only to `data/interim/realworld/` (gitignored). The shared
`bet_ledger.parquet` is never opened for writing, so Project 1's certified set
stays bit-identical — asserted by a test, same discipline as `discover.py` /
`slow_deepen.py` / `sports_deepen.py`. Read-only public unauthenticated GETs: no
keys, no signing, nothing on-chain.

## src/watch.py (Stage 6)

**Per-wallet polling.** `data-api.polymarket.com/trades?user=<wallet>` was
verified empirically (unlike `after`/`before`, see the ingest.py section
above) to correctly filter to one wallet's trades, newest-first — same
ordering as the unfiltered global feed. So watch.py polls each watched
wallet individually with a small `limit` rather than scanning the global feed
and filtering client-side, which would miss a wallet's trades entirely once
global volume pushes them past the 10k-row offset ceiling within one poll
interval.

**Current-price proxy.** `asset=<token_id>` was tested against `/trades` and
does **not** filter (returns unfiltered results — verified empirically,
2026-07-18). Rather than re-deriving "current price" from the global feed,
watch.py uses the CLOB's `https://clob.polymarket.com/last-trade-price`
endpoint directly (`{"price": ..., "side": ...}` per token id, verified
working). This is a different current-price source than features.py's
historical fair-value proxy (which is trade-tape-based, by necessity, since
there's no way to ask the historical tape for "the price right now"), but
here a live authoritative endpoint exists and is simpler/cheaper than paging
the global feed per token per poll.

**New-position detection.** A wallet's cursor is `{max_timestamp,
keys_at_max}`, the same shape and overlap-safe logic ingest.py uses for the
global feed cursor (ties at the same timestamp are resolved via
`keys_at_max` rather than risking a double- or zero-alert at the boundary).
Each watched wallet gets its own cursor. When a wallet is first added to the
watchlist (e.g. a re-rank promotes it into the top N), its cursor starts at
"now," not at its existing trade history — the point of Stage 6 is alerting
on *new* positions, not replaying a wallet's past into the alert log the
moment it's added.

**BUY-only.** "New position" reuses the BUY/SELL convention established in
features.py (`only_buys`): a SELL is a position close, not a new entry to
copy, so only BUY trades are evaluated for alerts. SELL rows still advance
the wallet's cursor (via `advance_cursor` using the full fetched page, not
just alertable trades), so they aren't re-fetched and re-skipped every poll.

**Remaining window.** `estimate_fair_value(entry_price, historical_copy_window)
= entry_price + historical_copy_window` extrapolates the wallet's own
historical `copy_window` (already computed by features.py from *their* past
bets) onto this new entry, since there's no way to know a brand-new
position's realized fair value yet. `remaining_cents = (fair_value -
current_price) * 100`; the window is "open" when this is at least
`watch.min_remaining_window` (price units, 0-1 scale, default 0.02 = 2
cents) — deliberately not zero, so noise-level remaining edge doesn't
trigger an alert. A wallet whose historical `copy_window` is flat or
negative will simply almost never clear this bar, which is the desired
behavior without needing a separate sign check.

**Alert-once guarantee and crash safety.** `poll_once` takes `fetch_trades`/
`fetch_price` as injectable parameters (defaulting to the real HTTP-calling
functions) specifically so its new-position/window/dedup logic is unit
testable without a network call, per CLAUDE.md's Stage 6 test requirements.
`run_watch_loop` persists state to disk after every cycle (survives
restarts) and never lets a `requests` exception escape — it backs off
exponentially (capped at `MAX_BACKOFF_SECONDS`) and retries, per the "never
crash-loop" requirement.
