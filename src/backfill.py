"""Targeted per-wallet history backfill — the top lever on forward validity.

The global `/trades` feed only surfaces the most recent ~10k platform-wide
trades, so any single wallet appears in it sparsely and per-wallet samples stay
thin (median ~7 resolved bets). The weight-tuning analysis (HANDOFF.md /
DECISIONS.md "Ranking weights") showed that thin per-wallet history — not the
weights — is the binding limit on how well the ranking predicts future skill.

But `/trades?user=<wallet>` returns *that wallet's own* history (same schema),
subject only to the same ~10k offset cap — which for a single wallet is its
(near-)complete record rather than a sliver. This stage deepens the wallets we
actually rank/report/watch by pulling their full per-user history into the same
`bet_ledger.parquet`. It is additive, incremental (per-wallet cursor so re-runs
fetch only new trades), resumable, and read-only against public data.

Bootstrapping note: which wallets get deepened is seeded from the current
ranking (top-N), so the first run deepens the initial (shallow-data) top-N, the
re-rank sharpens them, and the set stabilizes while the global feed keeps
discovering new wallets. This is intentional — we most want accurate metrics on
the wallets a user would actually consider following.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from src.common import (
    INTERIM_DIR,
    RANKED_WALLETS_PATH,
    atomic_write_json,
    ensure_dirs,
    load_config,
    load_ledger,
    make_session,
    print_disk_usage_summary,
    save_ledger,
)
from src.ingest import (
    fold_trades_to_ledger,
    load_resolutions_cache,
    refresh_ledger_resolutions,
    save_resolutions_cache,
    update_resolutions,
)

BACKFILL_CURSORS_PATH = INTERIM_DIR / "backfill_cursors.json"


def load_backfill_cursors() -> dict:
    if BACKFILL_CURSORS_PATH.exists():
        with open(BACKFILL_CURSORS_PATH) as f:
            return json.load(f)
    return {}


def save_backfill_cursors(cursors: dict) -> None:
    atomic_write_json(cursors, BACKFILL_CURSORS_PATH)


def select_wallets(cfg: dict, max_wallets: int | None = None) -> list[str]:
    """Wallets to deepen: the top-N by current rank (so we improve exactly the
    wallets we surface). Falls back to every wallet in the ledger when no ranking
    exists yet. `max_wallets` overrides the config cap (used for smoke tests)."""
    top_n = cfg.get("backfill", {}).get("top_n_wallets", 100)
    if max_wallets is not None:
        top_n = max_wallets
    if RANKED_WALLETS_PATH.exists():
        ranked = pd.read_parquet(RANKED_WALLETS_PATH).sort_values("rank")
        return ranked["wallet"].head(top_n).tolist()
    ledger = load_ledger()
    if ledger.empty:
        return []
    return ledger["wallet"].drop_duplicates().head(top_n).tolist()


def fetch_user_trades(session, cfg: dict, wallet: str, since_ts: int) -> list[dict]:
    """Page one wallet's `/trades?user=` history newest-first, stopping at the
    first trade at or before `since_ts` (already captured on a prior run) or the
    per-user offset cap. Returns only trades strictly newer than `since_ts`."""
    base = cfg["data_source"]["trades_api_base"]
    bf = cfg["backfill"]
    page_size = bf["page_size"]
    max_pages = bf["max_pages_per_wallet"]
    timeout = bf.get("request_timeout_sec", cfg["ingest"]["request_timeout_sec"])
    sleep_s = bf.get("sleep_between_requests_sec", 0.1)

    out: list[dict] = []
    offset = 0
    for _page in range(max_pages):
        try:
            resp = session.get(
                f"{base}/trades",
                params={"user": wallet, "limit": page_size, "offset": offset},
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # 400 = offset+limit past the per-user ceiling (~10k); a clean stop.
            # 408 = the API times out serving deep offsets for high-volume wallets
            # (observed reliably at offset~9000). Pages are newest-first, so by the
            # time we hit it we already have this wallet's most recent ~9k trades —
            # keep that partial rather than discard the whole wallet and re-pull it
            # (and re-trigger the 408) every run. The oldest sliver past the ceiling
            # is unreachable through this endpoint anyway.
            if status in (400, 408):
                if status == 408:
                    print(f"[backfill]   {wallet}: 408 at offset {offset}; keeping {len(out)} trades fetched so far")
                break
            raise
        page = resp.json()
        if not page:
            break
        stop = False
        for trade in page:
            if trade["timestamp"] <= since_ts:
                stop = True
                break
            out.append(trade)
        if stop or len(page) < page_size:
            break
        offset += page_size
        if sleep_s:
            time.sleep(sleep_s)
    return out


def run_backfill(
    max_wallets: int | None = None, batch_size: int | None = None
) -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    session = make_session(
        max_retries=cfg["ingest"]["max_retries"], backoff=cfg["ingest"]["retry_backoff_sec"]
    )

    bf = cfg.get("backfill", {})
    wallets = select_wallets(cfg, max_wallets)
    if batch_size is None:
        batch_size = int(bf.get("wallet_batch_size", 25))
    batch_size = max(1, batch_size)
    # Persist the ledger once every `checkpoint_min_rows` newly-folded rows (plus
    # a mandatory final save). This decouples the *fold* cadence (per batch, which
    # is what bounds memory) from the *save* cadence (which is expensive — the
    # ledger is a ~250 MB gzip'd parquet). A from-empty top-500 run checkpoints
    # ~every 100k rows so a crash costs at most that much re-fetch; a caught-up
    # nightly run adds far fewer than 100k rows and so saves exactly once at the
    # end, as the pre-batching code did — no per-batch I/O regression.
    checkpoint_min_rows = max(1, int(bf.get("checkpoint_min_rows", 100_000)))
    n_batches = (len(wallets) + batch_size - 1) // batch_size
    print(
        f"[backfill] deepening {len(wallets)} wallets via /trades?user= "
        f"(batch size {batch_size}, {n_batches} batches; checkpoint every "
        f"{checkpoint_min_rows} folded rows)"
    )
    cursors = load_backfill_cursors()

    # Load the ledger + resolutions ONCE and grow the ledger in place across
    # batches, folding each batch's trades in and freeing the raw dicts before
    # fetching the next batch. Peak memory is therefore bounded by the ledger
    # plus ONE batch of trades — not by the whole run. The old code accumulated
    # every wallet's full `/trades?user=` history (up to ~10k rows each) in a
    # single `all_new` list before folding, so a from-empty top-500 run would
    # hold millions of trade dicts at once and OOM the 3.7 GB box.
    ledger = load_ledger()
    resolutions = load_resolutions_cache()

    def _checkpoint() -> None:
        # Dedup, then write the ledger and cursors. Ledger is written first, then
        # cursors, both atomically (common.atomic_*): a kill in between leaves the
        # cursors *behind* the persisted ledger, so a re-run re-fetches those
        # trades and the dedup absorbs them — a re-run costs work, never
        # corruption. A cursor is never advanced on disk past trades not yet in
        # the persisted ledger.
        nonlocal ledger
        ledger = ledger.drop_duplicates(
            subset=["tx_hash", "wallet", "token_id", "side"], keep="last"
        )
        save_ledger(ledger)
        save_backfill_cursors(cursors)

    touched_condition_ids: set[str] = set()
    total_new_rows = 0
    rows_since_checkpoint = 0
    for bi in range(n_batches):
        batch = wallets[bi * batch_size : (bi + 1) * batch_size]
        batch_new: list[dict] = []
        for wallet in batch:
            since = int(cursors.get(wallet, 0))
            try:
                trades = fetch_user_trades(session, cfg, wallet, since)
            except Exception as exc:  # noqa: BLE001 - keep going on other wallets
                print(f"[backfill] WARNING: fetch failed for {wallet}: {exc}")
                continue
            if trades:
                cursors[wallet] = max(since, max(t["timestamp"] for t in trades))
                batch_new.extend(trades)

        if batch_new:
            touched_condition_ids |= {t["conditionId"] for t in batch_new}
            new_rows = fold_trades_to_ledger(batch_new, resolutions)
            total_new_rows += len(new_rows)
            rows_since_checkpoint += len(new_rows)
            ledger = pd.concat([ledger, new_rows], ignore_index=True)
            batch_new = []  # free the batch's raw trade dicts before the next fetch

        walcnt = min((bi + 1) * batch_size, len(wallets))
        if rows_since_checkpoint >= checkpoint_min_rows:
            _checkpoint()
            print(
                f"[backfill]   batch {bi + 1}/{n_batches} ({walcnt}/{len(wallets)} "
                f"wallets), {total_new_rows} new rows; CHECKPOINT -> ledger {len(ledger)} bets"
            )
            rows_since_checkpoint = 0
        else:
            print(
                f"[backfill]   batch {bi + 1}/{n_batches} ({walcnt}/{len(wallets)} "
                f"wallets), {total_new_rows} new rows folded"
            )

    # Resolution pass once, after all trades are folded. Queue the markets this
    # run touched PLUS every market still sitting unresolved in the ledger — not
    # just this run's new markets. A deep first backfill pulls tens of thousands
    # of markets but `max_resolution_fetches_per_run` only resolves a slice each
    # run; on later (incremental) runs the un-checked markets never reappear in
    # `touched_condition_ids`, so without re-queuing them they'd be orphaned
    # (pulled but never resolved) and the loop could never converge. Feeding them
    # back in drains the backlog deterministically at the per-run cap
    # (update_resolutions skips known-resolved markets and does unseen ones first).
    condition_ids = set(touched_condition_ids)
    if not ledger.empty:
        unresolved = ledger.loc[~ledger["resolved"].astype(bool), "market_id"]
        condition_ids |= set(unresolved.unique())
    max_res = bf.get("max_resolution_fetches_per_run")
    resolutions = update_resolutions(session, cfg, condition_ids, resolutions, max_fetches=max_res)
    save_resolutions_cache(resolutions)
    ledger = refresh_ledger_resolutions(ledger, resolutions)
    # Final save (always runs, like the pre-batching code): dedups then persists
    # the ledger + cursors, folding in any rows added since the last checkpoint
    # and any resolutions just refreshed onto existing rows.
    _checkpoint()

    n_resolved = int(ledger["resolved"].sum()) if not ledger.empty else 0
    n_wallets = ledger["wallet"].nunique() if not ledger.empty else 0
    print(
        f"[backfill] added {total_new_rows} rows; ledger now {len(ledger)} bets "
        f"({n_resolved} resolved) across {n_wallets} wallets"
    )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted per-wallet history backfill.")
    parser.add_argument(
        "--max-wallets", type=int, default=None,
        help="cap wallets this run (overrides backfill.top_n_wallets; for smoke tests)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="wallets fetched+folded+checkpointed per batch (overrides "
             "backfill.wallet_batch_size; lower it if memory is tight)",
    )
    args = parser.parse_args()
    run_backfill(max_wallets=args.max_wallets, batch_size=args.batch_size)
    print_disk_usage_summary()


if __name__ == "__main__":
    main()
