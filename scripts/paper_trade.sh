#!/usr/bin/env bash
# paper_trade.sh — cron runner for the forward PAPER trader (src/paper_trader.py).
#
# PAPER ONLY. This runs a read-only simulator: public unauthenticated GETs to
# data-api.polymarket.com (/trades) and clob.polymarket.com (/book, /markets),
# no private keys, no signing, no order placement of any kind. It writes
# hypothetical fills to a local parquet ledger under data/interim/paper/ and
# rewrites data/processed/paper_scoreboard.{md,parquet}.
#
# SEATBELTS, checked here as well as inside the module:
#   * KILL SWITCH — `touch data/KILL_PAPER_TRADER` and this script exits without
#     fetching anything. `rm` it to resume. Checked before every invocation.
#   * --commit is passed explicitly. The module's default is --dry-run; the cron
#     is the ONE place that opts into persisting, so nothing else can write the
#     ledger by accident.
#   * flock -n so two ticks can never interleave on the same ledger. A skipped
#     tick costs one poll, never corruption (all writes are atomic anyway).
#
# CADENCE. The poll interval is the paper trader's detection latency, and the
# `limit_noChase` arm's fill rate depends on it: the module can only see book
# snapshots at the moments it polls, so a wider interval misses transient offers
# and UNDER-counts fills. Every 5 minutes matches the ingest cadence and is a
# compromise between that and API politeness. Tighten it if the fill rate looks
# suspiciously low; the module is idempotent so a faster tick is safe.
#
# The scorer runs on a longer cadence than the poller (resolution GETs are the
# expensive part and markets do not settle every 5 minutes). It rewrites the
# scoreboard with a fresh timestamp on every run, which would leave the git tree
# perpetually dirty, so — exactly like scripts/forward_score.sh — this reverts
# the scoreboard afterwards and instead prints a loud line when it stops saying
# "AWAITING DATA". A human then runs `score` by hand and commits a real snapshot.
#
# Install (append to `crontab -e`):
#   */5 * * * *  /home/agent47/polymarket-sharps/scripts/paper_trade.sh poll
#   23  */6 * * *  /home/agent47/polymarket-sharps/scripts/paper_trade.sh score
set -uo pipefail
REPO=/home/agent47/polymarket-sharps
PY="$REPO/.venv/bin/python"
LOG="$REPO/data/cron.log"
KILL="$REPO/data/KILL_PAPER_TRADER"
LOCK="$REPO/data/interim/.paper.lock"
MODE="${1:-poll}"
cd "$REPO" || exit 1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

mkdir -p "$(dirname "$LOCK")"

if [ -e "$KILL" ]; then
  echo "[paper_trade $(ts)] KILL SWITCH present ($KILL) — not running." >> "$LOG"
  exit 0
fi

if [ ! -f "$REPO/data/processed/paper_freeze_manifest.json" ]; then
  echo "[paper_trade $(ts)] no freeze manifest — run \`python -m src.paper_trader freeze\` first." >> "$LOG"
  exit 0
fi

{
  echo "======== paper_trade ($MODE) $(ts) ========"
  if [ "$MODE" = "score" ]; then
    if OUT=$(flock -n "$LOCK" nice -n 19 ionice -c 3 "$PY" -m src.paper_trader score 2>&1); then
      echo "$OUT" | tail -25
      if ! grep -qi "AWAITING FORWARD DATA\|AWAITING DATA" data/processed/paper_scoreboard.md; then
        echo "*** [paper_trader] THE PAPER SCOREBOARD HAS DATA. This fires on the"
        echo "*** first scored signal in ANY arm — it is not a result. Before"
        echo "*** concluding anything, read (1) the FILL RATE per arm, (2)"
        echo "*** effective_events (not signal counts), (3) the placebo horizon"
        echo "*** sources and the no-fill reasons — a control that never fired is a"
        echo "*** broken control, not a win — and (4) limit_noChase vs"
        echo "*** placebo_random, which is the only comparison that decides anything."
        echo "*** Commit a snapshot by hand when it is worth reading."
      fi
    else
      echo "[paper_trader] scorer exited non-zero:"; echo "$OUT" | tail -25
    fi
    # keep the tree clean; real snapshots are committed by hand (see header)
    git checkout -- data/processed/paper_scoreboard.md data/processed/paper_scoreboard.parquet 2>/dev/null || true
    git clean -fq -- data/processed/paper_scoreboard.md data/processed/paper_scoreboard.parquet 2>/dev/null || true
  else
    flock -n "$LOCK" nice -n 19 ionice -c 3 "$PY" -m src.paper_trader run --commit 2>&1 | tail -12 \
      || echo "[paper_trader] poll skipped or failed (lock held, or transient error — it retries next tick)"
  fi
  echo "======== done $(ts) ========"
} >> "$LOG" 2>&1
