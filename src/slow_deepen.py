"""Deepen the slow-market screen shortlist — Project 3 build step 5b
(docs/project3_slow_markets.md §4 stage 3, §7.5).

The screen (5a) scored wallets on SHALLOW data (~1 bet/wallet in the discovery
corpus). This step pulls each shortlisted wallet's near-complete cross-market
history via `/trades?user=` — **creating data the screen never saw**, which is
exactly what makes the validation in 5c an honest out-of-sample test rather than a
re-measurement of the screen's own selection noise (§5.3 #1: disjoint selection and
validation data).

ISOLATION (decision #3): writes ONLY to `data/interim/slow_deepening/`. It NEVER
touches the shared `bet_ledger.parquet` — Project 1's certified 41-wallet set stays
bit-identical. Same discipline as `discover.py`. The per-user fetch, fold and
resolution steps reuse the tested `backfill` / `ingest` machinery verbatim.

READ-ONLY / ANALYSIS-ONLY: public unauthenticated GETs, no keys, nothing on-chain.
This is the one step in the chain that spends requests (§4 stage 3).
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.backfill import fetch_user_trades
from src.common import (
    INTERIM_DIR,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
    print_disk_usage_summary,
)
from src.ingest import (
    LEDGER_DEDUP_KEY,
    fold_trades_to_ledger,
    refresh_ledger_resolutions,
    trade_key,
    update_resolutions,
)
from src.common import LEDGER_COLUMNS
from src.slow_market import WALLET_SCORES_PATH

# Isolated deepening dataset (gitignored interim; never the shared ledger).
SLOW_DEEPEN_DIR = INTERIM_DIR / "slow_deepening"
DEEP_TRADES_PATH = SLOW_DEEPEN_DIR / "deep_trades.parquet"
DEEP_CURSORS_PATH = SLOW_DEEPEN_DIR / "deep_cursors.json"
DEEP_RESOLUTIONS_PATH = SLOW_DEEPEN_DIR / "deep_resolutions.parquet"
SHORTLIST_PATH = SLOW_DEEPEN_DIR / "shortlist.parquet"

RESOLUTION_COLUMNS = ["market_id", "token_id", "outcome", "resolved", "resolved_value", "closed"]

DEFAULT_SCREEN_GATE = 0.05          # confirmed with the coordinator (5a shortlist)
DEFAULT_CHECKPOINT_EVERY = 25       # wallets per fold+save+cursor checkpoint (resumable)


# ---------------------------------------------------------------------------
# Shortlist selection (from the 5a screen scores)
# ---------------------------------------------------------------------------

def select_shortlist(gate: float = DEFAULT_SCREEN_GATE, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """The wallets to deepen: screen_score >= gate, from the 5a screen output. The
    unit is the WALLET (deepening pulls a wallet's whole history), highest score
    first so a truncated run still deepens the strongest candidates."""
    if scores is None:
        if not WALLET_SCORES_PATH.exists():
            raise FileNotFoundError(
                f"{WALLET_SCORES_PATH} not found — run `python -m src.slow_market` (5a) first.")
        scores = pd.read_parquet(WALLET_SCORES_PATH)
    short = scores[scores["screen_score"] >= gate].copy()
    return short.sort_values("screen_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Isolated persistence
# ---------------------------------------------------------------------------

def load_deep_trades() -> pd.DataFrame:
    if DEEP_TRADES_PATH.exists():
        return pd.read_parquet(DEEP_TRADES_PATH)
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def save_deep_trades(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, DEEP_TRADES_PATH, compression="gzip")


def load_deep_cursors() -> dict:
    if DEEP_CURSORS_PATH.exists():
        with open(DEEP_CURSORS_PATH) as f:
            return json.load(f)
    return {}


def save_deep_cursors(cursors: dict) -> None:
    atomic_write_json(cursors, DEEP_CURSORS_PATH)


def load_deep_resolutions() -> pd.DataFrame:
    if DEEP_RESOLUTIONS_PATH.exists():
        return pd.read_parquet(DEEP_RESOLUTIONS_PATH)
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


def save_deep_resolutions(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, DEEP_RESOLUTIONS_PATH, compression="gzip")


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=LEDGER_DEDUP_KEY, keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deepening
# ---------------------------------------------------------------------------

def _checkpoint(batch: list[dict], cursors: dict) -> None:
    """Fold a batch of freshly-fetched trades into the isolated deep dataset and
    persist it + cursors (both atomic). Idempotent: dedup on the fill key + per-wallet
    cursors make repeated checkpoints safe and the run resumable after an interruption."""
    if batch:
        new_rows = fold_trades_to_ledger(batch, pd.DataFrame())  # resolutions applied later
        combined = _dedup(pd.concat([load_deep_trades(), new_rows], ignore_index=True))
        save_deep_trades(combined)
    save_deep_cursors(cursors)


def deepen_wallets(
    wallets: list[str],
    cfg: dict | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    session=None,
) -> int:
    """Pull each shortlisted wallet's full `/trades?user=` history into the isolated
    deep dataset. Incremental (per-wallet cursor = newest timestamp already stored),
    checkpointed every `checkpoint_every` wallets, resumable. Returns total new rows."""
    SLOW_DEEPEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = cfg or load_config()
    session = session or make_session()
    cursors = load_deep_cursors()

    batch: list[dict] = []
    total_new = 0
    t0 = time.time()
    for i, wallet in enumerate(wallets, 1):
        since_ts = int(cursors.get(wallet, {}).get("max_timestamp", 0))
        try:
            trades = fetch_user_trades(session, cfg, wallet, since_ts)
        except Exception as exc:  # noqa: BLE001 - keep deepening other wallets
            print(f"[slow_deepen] WARNING: fetch failed for {wallet}: {exc}", flush=True)
            continue
        if trades:
            batch.extend(trades)
            total_new += len(trades)
            newest = max(t["timestamp"] for t in trades)
            keys_at_max = [trade_key(t) for t in trades if t["timestamp"] == newest]
            cursors[wallet] = {"max_timestamp": max(since_ts, newest), "keys_at_max": keys_at_max}
        else:
            cursors.setdefault(wallet, {"max_timestamp": since_ts, "keys_at_max": []})
        if i % checkpoint_every == 0 or i == len(wallets):
            _checkpoint(batch, cursors)
            batch = []
            rate = i / (time.time() - t0) if time.time() > t0 else 0.0
            print(f"[slow_deepen]   {i}/{len(wallets)} wallets, {total_new:,} new trade rows "
                  f"({rate:.1f} wallets/s) — checkpointed", flush=True)
    return total_new


def resolve_deep_markets(cfg: dict | None = None, session=None, max_fetches: int | None = None) -> pd.DataFrame:
    """Resolve the deep dataset's markets, ISOLATED from the shared cache (read the
    shared resolutions read-only to avoid re-fetching, write only our own deep cache),
    then refresh deep_trades' resolved/resolved_value. Mirrors
    discover.resolve_discovery_markets."""
    from src.common import RESOLUTIONS_PATH
    cfg = cfg or load_config()
    session = session or make_session()

    trades = load_deep_trades()
    if trades.empty:
        print("[slow_deepen] no deep trades to resolve — run deepening first.")
        return trades

    shared = pd.read_parquet(RESOLUTIONS_PATH) if RESOLUTIONS_PATH.exists() else pd.DataFrame(columns=RESOLUTION_COLUMNS)
    deep_res = load_deep_resolutions()

    market_ids = set(trades["market_id"].dropna().unique())
    already = set()
    for r in (shared, deep_res):
        if not r.empty:
            already |= set(r.loc[r["closed"].fillna(False), "market_id"])
    need = market_ids - already
    print(f"[slow_deepen] resolving {len(market_ids):,} deep markets: "
          f"{len(market_ids) - len(need):,} already known, {len(need):,} to fetch from CLOB")

    deep_res = update_resolutions(session, cfg, need, deep_res, max_fetches=max_fetches)
    save_deep_resolutions(deep_res)

    combined = pd.concat(
        [shared[RESOLUTION_COLUMNS] if not shared.empty else shared, deep_res],
        ignore_index=True,
    )
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["market_id", "token_id"], keep="last")
    enriched = refresh_ledger_resolutions(trades, combined)
    save_deep_trades(enriched)

    n_res = int(enriched["resolved"].fillna(False).sum())
    print(f"[slow_deepen] deep dataset now {n_res:,}/{len(enriched):,} resolved bets "
          f"across {enriched.loc[enriched['resolved'].fillna(False), 'market_id'].nunique():,} markets")
    return enriched


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_deepen(gate: float = DEFAULT_SCREEN_GATE, checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
               max_wallets: int | None = None, resolve: bool = True) -> None:
    SLOW_DEEPEN_DIR.mkdir(parents=True, exist_ok=True)
    short = select_shortlist(gate)
    if max_wallets is not None:
        short = short.head(max_wallets)
    atomic_to_parquet(short, SHORTLIST_PATH, compression="gzip")
    print(f"[slow_deepen] shortlist: {len(short):,} wallets at screen_score >= {gate} "
          f"(persisted -> {SHORTLIST_PATH})")

    wallets = short["wallet"].tolist()
    total = deepen_wallets(wallets, checkpoint_every=checkpoint_every)
    deep = load_deep_trades()
    print(f"[slow_deepen] added {total:,} rows; deep dataset now {len(deep):,} trades "
          f"across {deep['market_id'].nunique():,} markets and {deep['wallet'].nunique():,} wallets")

    if resolve:
        resolve_deep_markets()
    print_disk_usage_summary()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deepen the slow-market screen shortlist via /trades?user= (Project 3 "
                    "step 5b). Isolated dataset; the shared ledger is never touched.")
    parser.add_argument("--gate", type=float, default=DEFAULT_SCREEN_GATE,
                        help=f"screen_score gate for the shortlist (default {DEFAULT_SCREEN_GATE})")
    parser.add_argument("--max-wallets", type=int, default=None,
                        help="cap wallets deepened this run (smoke tests; highest-score first)")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                        help="fold+save+cursor checkpoint every N wallets (resumable)")
    parser.add_argument("--resolve-only", action="store_true",
                        help="skip fetching; just (re)resolve the existing deep dataset")
    args = parser.parse_args()
    if args.resolve_only:
        resolve_deep_markets()
        print_disk_usage_summary()
    else:
        run_deepen(gate=args.gate, checkpoint_every=args.checkpoint_every,
                   max_wallets=args.max_wallets)


if __name__ == "__main__":
    main()
