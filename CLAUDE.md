# CLAUDE.md — build & operating instructions for Claude Code

## What this project is
An **identification engine** for Polymarket. It ingests public on-chain trade
history and scores every wallet on how *genuinely sharp* it is — not raw win rate,
not raw profit. The output is a ranked table of wallets that consistently earn
positive edge versus the market price at the moment they entered, validated
out-of-sample so hot streaks don't masquerade as skill.

**Everything here is read-only.** The pipeline reads public, unauthenticated
endpoints, holds no keys, signs nothing and places no orders. The question it
exists to answer is whether the ranking is *true* — whether a wallet that looks
sharp is sharp, and whether that skill is visible to anyone watching from
outside. Keep new code read-only.

## Build order (do these in sequence, commit after each)
1. `src/ingest.py` — pull Polymarket trade history from the public data source
   (Polygon chain / Polymarket data API / Goldsky subgraph — pick the most
   reliable free one, document the choice). Persist raw trades to `data/raw/`.
   Each record needs at minimum: wallet, market_id, outcome, side, entry_price,
   size, timestamp, and the market's eventual resolution.
2. `src/features.py` — per wallet, compute the metrics in "Scoring" below.
3. `src/validate.py` — the out-of-sample split (see "Validation"). This is
   non-negotiable and must run before any ranking is emitted.
4. `src/rank.py` — combine validated metrics into a score, emit ranked table to
   `data/processed/`.
5. `src/report.py` — human-readable summary of the top wallets and why each ranks.

## Scoring — the metrics that matter
Raw win rate and raw profit are traps (favorite-betting inflates win rate; size
inflates profit). Center everything on these instead:
- **Edge over entry price**: for each bet, (realized outcome − entry price),
  aggregated. NOTE (added 2026-07-18 after a red-team, see HANDOFF.md /
  DECISIONS.md): raw mean edge turned out NOT to equal skill — it is dominated by
  a structural favorite-longshot base rate (favorites are underpriced, longshots
  overpriced), a persistent per-wallet price preference that trivially survives
  out-of-sample. Ranking and validation therefore key on **skill (residual) edge**
  = realized outcome − E[outcome | entry price], i.e. how much the wallet beat the
  price it paid. Raw edge is still computed and reported alongside.
- **Earliness**: did the wallet enter *before* the market price moved toward
  their side? Measure price drift in the window after their entry.
- **Copy window**: gap between their entry price and estimated fair value. Wallets
  with consistently wide windows are the followable ones. Compute per wallet as a
  first-class ranking variable.
- **Sample size**: number of resolved bets. Down-weight small samples hard.
- **Breadth**: distinct markets / categories, so one lucky event can't dominate.
- **Time-consistency**: does edge hold across time, or is it one spike?
- **Authenticity & pattern flags (metadata only — never exclude anything)**:
  EVERY wallet is ingested, scored, and ranked. Nothing is ever dropped, soft-deleted,
  down-weighted, or removed from the dataset or the ranking for any reason. Instead,
  attach non-destructive flag columns as metadata that the user can sort/filter on
  themselves later:
  - `manufactured_record_flag`: set when a wallet's record appears mechanically
    self-dealt (trading against its own linked/co-funded wallets), which can distort
    its edge numbers. This is a data-quality *note*, not a removal — the wallet stays
    fully in the ranking with its metrics computed as normal.
  - `pattern_flag` (free-text/enum): optional note for atypical shapes, e.g.
    concentrated bets timed just before events. This is purely descriptive.
  Rules: flags are additive columns only. They MUST NOT feed into the score, alter
  rank order, or filter rows out. Do not infer or record intent (no "insider"
  labels as fact). The goal is a complete dataset where the user can apply their own
  filters — the engine collects everything and hides nothing.

## Validation — the anti-self-deception step
With thousands of wallets, some sit in the "sharp" corner by pure chance.
For every candidate: fit/select on the **first half** of its bet history, then
measure whether edge **persists in the held-out second half**. Only wallets whose
edge survives out-of-sample are reported. Report both in-sample and out-of-sample
edge side by side so the decay is visible.

## Runtime
- Runs headless on low-end hardware (a Debian Chromebook, always-on). Design for
  that: modest memory, resumable ingest, no GPU.
- **Storage is tight (~32GB eMMC). Keep the disk footprint small and FLAT, not
  growing:**
  - Store raw and processed data as **compressed Parquet** (or gzipped), never
    plain uncompressed JSON/CSV on disk.
  - Keep `data/raw/` as an **incremental, pruned rolling cache** — once trades are
    folded into per-wallet metrics, do not retain unbounded raw history on disk.
    Keep only what's needed to recompute (e.g. a rolling window), and prune older
    raw once its metrics are persisted.
  - Persist compact **intermediate aggregates** (per-wallet rollups) so a re-run
    doesn't need to re-download or re-store the full history.
  - Print a disk-usage summary at the end of each run and warn if `data/` exceeds a
    configurable cap (default 5 GB).
  - Processed outputs (the ranked table, reports) go to GitHub; the bulky raw cache
    stays local and gitignored.
- The pipeline is plain Python and runs on a schedule (cron). The model is NOT in
  the hot loop — it writes and fixes this code; it does not re-reason each run.
- Idempotent + incremental: re-running should fetch only new data, not re-pull.

## Style
Small, testable functions. A test in `tests/` for every metric in `features.py`
using a tiny hand-built fixture where the correct edge is known by construction.
Log decisions to stdout. Keep secrets/config in `config/config.yaml` (gitignored).

## Git workflow
Commit directly to `master` for read-only analysis, research, and pipeline work —
this is a solo, fast-moving repo with linear history, and every such change is
reversible. Don't force-push or rewrite shared history when multiple sessions share
the working tree; if a push is rejected, `git fetch` and fast-forward.

---

## Stage 6 — `src/watch.py` (real-time monitor)

A monitoring layer built on top of the ranked output. It observes and notifies:
it watches public data and emits alerts, with no order placement or key handling.
Its point is measurement — it is the only part of the repo that sees a signal at
the latency an outside watcher would actually see it.

Behavior:
1. Load the top-N ranked wallets from `data/processed/` (produced by `rank.py`).
2. Poll public Polymarket data on an interval (configurable, default 60s) for new
   positions taken by those wallets. Track last-seen state so each position alerts once.
3. For each new position, compute whether the **current** market price is still
   inside that wallet's historical profitable copy-window (entry price → estimated
   fair value). Report remaining edge in cents and as a percentage.
4. Emit an alert when — and only when — the window is still open:
   wallet, market, side, their entry price, current price, remaining window,
   that wallet's validated out-of-sample edge and sample size.
5. Alert channels: stdout + append to `data/alerts.log`, plus an optional webhook
   URL from config (e.g. ntfy/Discord) for phone push. Never alert twice for the
   same position.
6. Runs as a long-lived process; resumable, and survives restarts via persisted
   last-seen state. Handle API errors with backoff, never crash-loop.

Config additions (`config/config.yaml`): `watch.poll_seconds`, `watch.top_n_wallets`,
`watch.min_remaining_window`, `watch.webhook_url`.
