#!/usr/bin/env bash
# Finish the pipeline unattended. One CLEAN context per stage (fresh process),
# so context never accumulates. Waits out rate limits and resumes.
# Stops when the build is DONE, or at a hard safety cap. No infinite churn.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

STAGES=(features validate rank report watch)
MAX_WAIT_CYCLES=12     # safety brake: at most 12 wait-and-retry cycles, ever
WAIT_SECONDS=3600      # how long to sleep when a limit is hit (1h)
cycle=0

for stage in "${STAGES[@]}"; do
  while true; do
    echo "[$(date)] === stage: $stage ==="
    if claude -p "Read CLAUDE.md. Implement src/$stage.py only, per its spec. Write tests in tests/ for each metric, run them until green, then git commit with a clear message. Do not touch other modules. If genuinely blocked (NOT a usage/rate limit), write the reason to BLOCKED.txt and exit nonzero." \
        --model sonnet --permission-mode acceptEdits 2>>data/agent_err.log; then
      echo "[$(date)] stage $stage complete"
      break
    fi
    if [ -f BLOCKED.txt ]; then
      echo "[$(date)] genuinely blocked on $stage. stopping. see BLOCKED.txt"
      exit 1
    fi
    cycle=$((cycle+1))
    if [ "$cycle" -ge "$MAX_WAIT_CYCLES" ]; then
      echo "[$(date)] hit safety cap ($MAX_WAIT_CYCLES cycles). stopping so it can't run away."
      exit 1
    fi
    echo "[$(date)] limit/transient on $stage. sleeping ${WAIT_SECONDS}s, then retry (cycle $cycle/$MAX_WAIT_CYCLES)."
    sleep "$WAIT_SECONDS"
  done
done

echo "[$(date)] all stages built. running end-to-end..."
if ./run.sh > data/final_run.log 2>&1; then
  echo "[$(date)] DONE. ranked table in data/processed/. summary in data/final_run.log"
else
  echo "[$(date)] stages built but ./run.sh errored. see data/final_run.log"
fi
