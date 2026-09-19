#!/usr/bin/env bash
# Real-world deepening arm — draw the stratified sample, fetch it, resolve it.
#
# Long-running (~2-4 h for ~2,100 wallets at the observed ~6 s/wallet) and fully
# resumable: per-wallet cursors mean a re-run picks up exactly where it stopped,
# and the disk-budget guard stops the fetch cleanly rather than filling the eMMC.
# Re-run it as many times as it takes; each pass is idempotent.
#
# Writes ONLY to data/interim/realworld/. The shared bet_ledger is never opened
# for writing, so this needs no writer lock — but the stages are serialized here
# so a fetch and a resolve can never overlap.
#
# READ-ONLY / ANALYSIS-ONLY: public unauthenticated GETs, no keys, no signing.
#
# Usage:  ./scripts/run_realworld_deepen.sh [--skip-plan] [--skip-fetch] [--skip-resolve]
# Detached:  setsid nohup ./scripts/run_realworld_deepen.sh > ~/.pmrun/rw_deepen.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p "${HOME}/.pmrun"

SKIP_PLAN=0
SKIP_FETCH=0
SKIP_RESOLVE=0
for arg in "$@"; do
  case "$arg" in
    --skip-plan)    SKIP_PLAN=1 ;;
    --skip-fetch)   SKIP_FETCH=1 ;;
    --skip-resolve) SKIP_RESOLVE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

step() { echo -e "\n=== $* === $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }

# The plan is a frozen draw: re-running it with the same seed reproduces the same
# shortlist, so it is safe to leave in, but it is skipped once drawn to make the
# pre-registration obvious.
if [ "$SKIP_PLAN" -eq 0 ] && [ ! -f data/interim/realworld/shortlist.parquet ]; then
  step "1/3 plan — stratified, performance-blind draw (no network)"
  $PY -u -m src.realworld_deepen plan
else
  step "1/3 plan — shortlist already drawn, keeping it (a re-draw would break the pre-registration)"
fi

if [ "$SKIP_FETCH" -eq 0 ]; then
  step "2/3 deepen (fetch only; resumable, budget-guarded)"
  $PY -u -m src.realworld_deepen deepen --no-resolve --checkpoint-every 25
fi

if [ "$SKIP_RESOLVE" -eq 0 ]; then
  # Uncapped: a first deep pull surfaces tens of thousands of unseen markets. The
  # CLOB rate-limits to ~20 req/s regardless of concurrency, so this is the long
  # pole. Re-running drains whatever is left.
  step "3/3 resolve markets (long; drains across re-runs)"
  $PY -u -m src.realworld_deepen resolve
fi

step "status"
$PY -u -m src.realworld_deepen status

step "done"
