#!/usr/bin/env bash
# paper_rw.sh — cron runner for the REAL-WORLD forward paper arm (src/paper_rw.py).
#
# PAPER ONLY. This runs a read-only simulator: public unauthenticated GETs to
# data-api.polymarket.com (/trades), clob.polymarket.com (/book, /markets) and
# gamma-api.polymarket.com (/markets, for the real fee schedule). No private
# keys, no signing, no order placement of any kind. It writes hypothetical fills
# to a local parquet ledger under data/interim/paper_rw/ and rewrites
# data/processed/paper_rw_scoreboard.{md,parquet}.
#
# THIS IS A SECOND, INDEPENDENT ARM. The 38-wallet paper trader frozen
# 2026-07-26T22:48:29Z keeps running under scripts/paper_trade.sh and shares
# nothing with this one: different manifest, different watchlist, different
# ledger directory, different scoreboard, different lock. Do not merge them.
#
# SEATBELTS, checked here as well as inside the module:
#   * TWO KILL SWITCHES — the shared `data/KILL_PAPER_TRADER` halts every paper
#     arm on the box; `data/KILL_PAPER_RW` halts this one alone, leaving the
#     running 38-wallet experiment untouched. Either file stops this script
#     before it fetches anything. `rm` to resume.
#   * --commit is passed explicitly. The module's default is --dry-run; this
#     cron is the ONE place that opts into persisting, so nothing else can write
#     the ledger by accident.
#   * flock -n on ITS OWN lock so two ticks never interleave on this ledger, and
#     so this arm can never block or be blocked by the other one. A skipped tick
#     costs one poll, never corruption (all writes are atomic anyway).
#
# CADENCE. The poll interval is this arm's detection latency and it is therefore
# part of the experiment, not a free parameter — which is why the achieved lag is
# MEASURED and printed on the scoreboard rather than assumed. 5 minutes matches
# the other arm; the offset (2-59/5) keeps the two arms from hitting the same
# APIs in the same second. The scorer runs every 6 h: it is what turns a still-live
# order in a settled market into a terminal no-fill, so a long gap leaves resting
# limits re-checking books in markets whose outcome is already known.
#
# Like scripts/paper_trade.sh and scripts/forward_score.sh, the scorer reverts
# the scoreboard afterwards (it rewrites a fresh timestamp every run and would
# otherwise leave the tree permanently dirty) and instead prints a loud line when
# the page stops saying "AWAITING DATA". A human then runs `score` by hand and
# commits a real snapshot.
#
# Install (append to `crontab -e`):
#   2-59/5 *    * * *  /home/agent47/polymarket-sharps/scripts/paper_rw.sh poll
#   41     */6  * * *  /home/agent47/polymarket-sharps/scripts/paper_rw.sh score
set -uo pipefail
REPO=/home/agent47/polymarket-sharps
PY="$REPO/.venv/bin/python"
LOG="$REPO/data/cron.log"
KILL_SHARED="$REPO/data/KILL_PAPER_TRADER"
KILL_LOCAL="$REPO/data/KILL_PAPER_RW"
LOCK="$REPO/data/interim/.paper_rw.lock"
MODE="${1:-poll}"
cd "$REPO" || exit 1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

mkdir -p "$(dirname "$LOCK")"

for K in "$KILL_SHARED" "$KILL_LOCAL"; do
  if [ -e "$K" ]; then
    echo "[paper_rw $(ts)] KILL SWITCH present ($K) — not running." >> "$LOG"
    exit 0
  fi
done

if [ ! -f "$REPO/data/processed/paper_rw_freeze_manifest.json" ]; then
  echo "[paper_rw $(ts)] no freeze manifest — run \`python -m src.paper_rw freeze\` first." >> "$LOG"
  exit 0
fi

{
  echo "======== paper_rw ($MODE) $(ts) ========"
  if [ "$MODE" = "score" ]; then
    if OUT=$(flock -n "$LOCK" nice -n 19 ionice -c 3 "$PY" -m src.paper_rw score 2>&1); then
      echo "$OUT" | tail -25
      if ! grep -qi "AWAITING DATA" data/processed/paper_rw_scoreboard.md; then
        echo "*** [paper_rw] THE REAL-WORLD PAPER SCOREBOARD HAS DATA. This fires on"
        echo "*** the first scored signal in ANY arm — it is not a result. Before"
        echo "*** concluding anything, read (1) the FILL RATE and the ACHIEVED COPY"
        echo "*** LAG table, (2) effective_events, not signal counts, (3) the"
        echo "*** ABSOLUTE net dollars per arm (the SELECTION test) BEFORE the"
        echo "*** headline-minus-control difference (the TIMING test), and (4) the"
        echo "*** per-wallet rows — with 5 wallets at a measured 17.3% set FDR,"
        echo "*** about one of them is expected to be a false discovery."
        echo "*** Commit a snapshot by hand when it is worth reading."
      fi
    else
      echo "[paper_rw] scorer exited non-zero:"; echo "$OUT" | tail -25
    fi
    # keep the tree clean; real snapshots are committed by hand (see header)
    git checkout -- data/processed/paper_rw_scoreboard.md data/processed/paper_rw_scoreboard.parquet 2>/dev/null || true
    git clean -fq -- data/processed/paper_rw_scoreboard.md data/processed/paper_rw_scoreboard.parquet 2>/dev/null || true
  else
    flock -n "$LOCK" nice -n 19 ionice -c 3 "$PY" -m src.paper_rw run --commit 2>&1 | tail -12 \
      || echo "[paper_rw] poll skipped or failed (lock held, or transient error — it retries next tick)"
  fi
  echo "======== done $(ts) ========"
} >> "$LOG" 2>&1
