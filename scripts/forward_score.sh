#!/usr/bin/env bash
# forward_score.sh — periodic runner for the Project 3 forward paper tests.
#
# Runs the slow-forecaster and both sports forward scorers, which re-fetch each frozen
# wallet's post-freeze trades (public data-api /trades GETs — READ-ONLY, no keys,
# no order placement) and score whether the frozen cohorts beat their baseline on
# bets placed after the freeze.
#
# The scorers rewrite their scoreboards (…_forward_scoreboard.{md,parquet}) on
# every run and embed a run timestamp, so a straight run would leave the git tree
# perpetually dirty. This wrapper therefore REVERTS the forward artifacts after
# each run to keep the tree clean, and instead surfaces the signal in the log:
# it prints a loud line the moment a scorer stops reporting "awaiting data".
# When that happens, a human runs the scorer once by hand and commits a real
# snapshot. It never touches Project 1 outputs (only the *forward* files).
#
# Installed via cron (every 2 days). All output appended to data/cron.log
# (gitignored). Cadence: sports resolves in hours-days, forecasters in weeks, so
# every 2 days catches the fast side with low latency without hammering the API.
set -uo pipefail
REPO=/home/agent47/polymarket-sharps
PY="$REPO/.venv/bin/python"
LOG="$REPO/data/cron.log"
cd "$REPO" || exit 1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

{
  echo "======== forward_score $(ts) ========"
  git pull --rebase --autostash 2>&1 | tail -2 || true
  for mod in slow_forward sports_forward sports_sig2c; do
    echo "---- src.$mod score ----"
    if OUT=$(nice -n 19 ionice -c 3 "$PY" -m "src.$mod" score 2>&1); then
      echo "$OUT" | tail -25
      if ! echo "$OUT" | grep -qi "awaiting data"; then
        echo "*** [$mod] forward data accumulating — this fires on ANY resolved post-freeze bet, incl. the unvetted wide tier. Read the scoreboard's CI/p AND whether the HEADLINE tier has events before concluding anything. Commit a real snapshot only when it's worth reading. ***"
      fi
    else
      echo "[$mod] scorer exited non-zero:"; echo "$OUT" | tail -25
    fi
  done
  # Keep the tree clean: discard the scorers' timestamp churn on the *forward*
  # artifacts only. Real snapshots are committed by hand when data arrives.
  git checkout -- data/processed/*forward* 2>/dev/null || true
  git clean -fq -- data/processed/*forward* 2>/dev/null || true
  echo "======== done $(ts) ========"
} >> "$LOG" 2>&1
