#!/usr/bin/env bash
# Frequent rolling-window global ingest — cron target, runs every ~5 min.
#
# The global /trades feed only serves the most recent ~10k platform-wide trades
# and has no working time filter, so consecutive polls MUST overlap or trades in
# the gap are lost permanently (see DECISIONS.md "no working time filter"). Hence
# every-5-min, not nightly.
#
# ingest is a ledger WRITER, so it takes the shared writer lock. It uses -n
# (non-blocking): if the nightly recompute is holding the lock, this tick simply
# SKIPS rather than becoming a second concurrent ledger writer (the thing that
# corrupted the ledger once).
#
# A skip is NOT free: it doubles the effective poll period, and if that period
# exceeds the feed's ~8.3-min depth the window stops overlapping and trades are
# lost for good (measured 2026-07-23: ~20%). Ingest therefore writes new rows to
# a small DELTA part (data/interim/ledger_delta/) instead of rewriting the whole
# ledger, keeping a tick to ~a minute so it is finished long before the next one
# fires; the nightly recompute folds the delta in. If cron.log starts showing
# regular SKIPs again, the cadence — not the lock — is the thing to fix.
set -uo pipefail
cd "$(dirname "$0")/.."

LOCK="data/interim/.ledger.lock"       # gitignored (under data/interim/)
mkdir -p data/interim
exec >>"data/cron.log" 2>&1            # gitignored; single shared cron log

flock -n -E 99 "$LOCK" bash -c '
  source .venv/bin/activate
  echo "=== [ingest] $(date -Is) START ==="
  python -m src.ingest
  echo "=== [ingest] $(date -Is) DONE ==="
'
rc=$?
if [ "$rc" -eq 99 ]; then
  echo "=== [ingest] $(date -Is) SKIPPED — ledger writer lock held (previous ingest still running, or the nightly recompute) ==="
fi
exit 0
