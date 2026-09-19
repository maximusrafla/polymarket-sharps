#!/usr/bin/env bash
# Project 3 sports arm — the whole chain, in order, resumable.
#
# Each stage is independently re-runnable and writes only to the isolated
# data/interim/sports/ tree (plus the freeze artifacts in data/processed/). The
# shared bet_ledger is never opened for writing, so this does NOT need the
# writer lock — but it must not run while a resolve and a fetch could overlap,
# which is why the stages are serialized here rather than launched separately.
#
# Usage:  ./scripts/run_sports_arm.sh [--skip-fetch] [--skip-resolve]
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG_DIR="${HOME}/.pmrun"
mkdir -p "$LOG_DIR"

SKIP_FETCH=0
SKIP_RESOLVE=0
for arg in "$@"; do
  case "$arg" in
    --skip-fetch)   SKIP_FETCH=1 ;;
    --skip-resolve) SKIP_RESOLVE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

step() { echo -e "\n=== $* === $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }

if [ "$SKIP_FETCH" -eq 0 ]; then
  step "1/5 deepen (fetch only; per-wallet cursors make this resumable)"
  $PY -u -m src.sports_deepen --no-resolve --checkpoint-every 25
fi

if [ "$SKIP_RESOLVE" -eq 0 ]; then
  # Uncapped: a first deep pull surfaces tens of thousands of unseen game
  # markets. The CLOB rate-limits to ~20 req/s regardless of concurrency, so
  # this is the long pole. Re-running drains whatever is left.
  step "2/5 resolve markets (long; drains across re-runs)"
  $PY -u -m src.sports_deepen --resolve-only
fi

step "3/5 validate on fresh deep data (EVENT clusters, scoped 10c floor)"
$PY -u -m src.sports_validate --fdr-q 0.10

step "4/5 MANDATORY finer-baseline re-check (league -> league|form -> +side, 20/40 bins)"
$PY -u -m src.sports_validate --finer-recheck --fdr-q 0.10

step "5/5 freeze the pre-registered forward tier + write the scoreboard"
if [ -f data/processed/sports_freeze_manifest.json ]; then
  echo "freeze artifact already exists — NOT overwriting (cohorts are never"
  echo "re-frozen in light of forward outcomes). Delete it deliberately, or pass"
  echo "--force to src.sports_forward freeze with a reason, if this is a"
  echo "legitimate pre-registration amendment."
else
  $PY -u -m src.sports_forward freeze
fi

step "done"
