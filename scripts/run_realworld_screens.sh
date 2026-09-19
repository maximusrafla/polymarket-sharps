#!/usr/bin/env bash
# Run the screens and the Project 1 gate on the DEEP REAL-WORLD sample.
#
# The rules are unchanged — this is the same certification gate and the same
# screening surface that ran on the shared ledger. Only the population changes:
# a stratified, performance-blind probability sample of 2,500 discovery wallets,
# deepened to near-complete histories (src/realworld_deepen.py). Every prior
# real-world verdict in this repo was measured on the ledger's real-world
# RESIDUE — what a firehose that is 82% 5-minute crypto leaves behind.
#
# SERIALIZED ON PURPOSE. Each stage peaks around 2.4 GB and this box has 3 GB
# plus 4 GB of swap; running them together would OOM (a naive full-tape concat
# already got OOM-killed here once, exit 137). Do not parallelize these.
#
# READ-ONLY: no network, no writes outside data/interim/realworld/.
#
# Usage:  ./scripts/run_realworld_screens.sh [--skip-validate] [--skip-screen]
# Detached:  setsid nohup ./scripts/run_realworld_screens.sh > ~/.pmrun/rw_screens.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p "${HOME}/.pmrun"

SKIP_VALIDATE=0
SKIP_SCREEN=0
for arg in "$@"; do
  case "$arg" in
    --skip-validate) SKIP_VALIDATE=1 ;;
    --skip-screen)   SKIP_SCREEN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

step() { echo -e "\n=== $* === $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }

# Validation first: it is the more interpretable of the two and the faster, so a
# long screen run never blocks the headline answer.
if [ "$SKIP_VALIDATE" -eq 0 ]; then
  step "1/2 Project 1 gate on the real-world sample (micro dropped, baseline refitted)"
  $PY -u -m src.realworld_validate
fi

if [ "$SKIP_SCREEN" -eq 0 ]; then
  # Pre-registered counts, exactly the defaults every prior run of this script
  # used: 500 null-P shuffles, 25 each for null O / null C / the gate rebuild,
  # 2000 cluster bootstrap resamples.
  step "2/2 screening surface (owner's ROI screen, 2-D screen, all three nulls)"
  $PY -u scripts/audit_screen_surface.py --real-world --tape realworld
fi

step "done"
