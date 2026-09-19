#!/usr/bin/env bash
# Nightly recompute — cron target, runs once per night.
#
# Three phases:
#   0. fold-delta — the 5-min ingest only appends new rows to small delta parts
#      (it must stay ~1 min so the poll cadence holds); this folds them into the
#      bet ledger. Runs FIRST, under the writer lock, so every stage below reads a
#      complete ledger. See DECISIONS.md "incremental ledger write".
#   1. backfill  — deepens the top-ranked wallets' per-wallet history. This is a
#      ledger WRITER, so it runs under the shared writer lock (BLOCKING: it waits
#      for any in-flight 5-min ingest to finish, then holds the lock so ingest
#      ticks skip while it writes). It is memory-batched (src/backfill.py folds +
#      checkpoints per wallet-batch) so a from-empty top-500 run can't OOM.
#   2. features -> validate -> rank -> report — these only READ the ledger, and
#      the ledger is written atomically (common.atomic_*), so a reader always sees
#      a complete file. They therefore run UNLOCKED and don't block the 5-min
#      ingest — no reason to freeze discovery for the ~20 min the re-rank takes.
#
# features peaks ~3 GB RSS on the 3.7 GB box; nightly timing keeps it off the
# 5-min ingest's back. set -e so a stage failure stops the chain and is loud in
# the cron log; the next night retries.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK="data/interim/.ledger.lock"
mkdir -p data/interim

# Nightly rotation of the shared cron log (the 5-min ingest appends to it forever):
# cap at ~5 MB, trimming to the last ~1 MB. Cheap, keeps data/ under its disk cap.
LOG="data/cron.log"
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5000000 ]; then
  tail -c 1000000 "$LOG" >"$LOG.rot" && mv "$LOG.rot" "$LOG"
fi
exec >>"$LOG" 2>&1

source .venv/bin/activate
echo "=== [recompute] $(date -Is) START ==="

# The WHOLE recompute runs under the writer lock, so the 5-min ingest skips for
# its duration. Phase 2 is read-only and the ledger is written atomically, so a
# concurrent reader is safe *correctness*-wise — the reason to hold the lock is
# MEMORY: on the 1 GB always-on host, features peaks ~770 MB and ingest ~770 MB,
# and running them together would blow RAM into swap and slow both to a crawl.
# Serialising costs at most one skipped ingest window (~an hour at 03:17, the
# quietest time) and the rolling-window poll self-heals on the next tick.
flock "$LOCK" bash -c '
  source .venv/bin/activate
  python -m src.ingest --fold-delta   # fold the 5-min ingest deltas into the ledger FIRST
  python -m src.backfill
  # Project 3: refresh the per-market speed sidecar from the tape (cheap, no network)
  # and (re)derive the additive speed_bucket ledger column from it, AFTER the ledger
  # writers above so it classifies the final state. Both are streaming/low-RSS
  # (~22s + ~43s, <1 GB); populate preserves row order so Project 1 stays bit-identical.
  python -m src.market_meta
  python -m src.market_meta --populate
  python -m src.features
  python -m src.validate
  python -m src.rank
  python -m src.report
'

echo "=== [recompute] $(date -Is) DONE ==="
