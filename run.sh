#!/usr/bin/env bash
# Full manual pipeline: ingest -> backfill -> features -> validate -> rank -> report.
# Use for a one-shot end-to-end run. The recurring schedule instead splits this
# into scripts/ingest.sh (every ~5 min) and scripts/recompute.sh (nightly) — see
# scripts/crontab.example.
#
# The two WRITER stages (ingest, backfill) go through the shared writer lock so a
# manual run here never becomes a second concurrent ledger writer alongside a
# cron-fired ingest tick (blocking flock: it waits its turn). The read-only
# stages run unlocked.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

LOCK="data/interim/.ledger.lock"
mkdir -p data/interim

flock "$LOCK" python -m src.ingest
flock "$LOCK" python -m src.ingest --fold-delta  # fold the incremental delta parts into the ledger
flock "$LOCK" python -m src.backfill   # deepen top-ranked wallets before recomputing (see DECISIONS.md)
python -m src.features
python -m src.validate
python -m src.rank
python -m src.report
