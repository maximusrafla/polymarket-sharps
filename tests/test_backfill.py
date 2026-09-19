import pandas as pd
import requests

import src.backfill as backfill
from src.backfill import fetch_user_trades
from src.common import LEDGER_COLUMNS
from src.ingest import fold_trades_to_ledger


def cfg(page_size=2, max_pages=10):
    return {
        "data_source": {"trades_api_base": "http://api"},
        "ingest": {"request_timeout_sec": 5},
        "backfill": {
            "page_size": page_size,
            "max_pages_per_wallet": max_pages,
            "sleep_between_requests_sec": 0,
        },
    }


class FakeResp:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError()
            err.response = self
            raise err


class FakeSession:
    """Serves a newest-first trade list by offset/limit; 400s past `cap`."""

    def __init__(self, trades_desc, cap=10_000):
        self.all = trades_desc
        self.cap = cap
        self.offsets = []
        self.params_seen = []

    def get(self, url, params=None, timeout=None):
        self.params_seen.append(params)
        off, lim = params["offset"], params["limit"]
        self.offsets.append(off)
        if off >= self.cap:
            return FakeResp(None, status=400)
        return FakeResp(self.all[off:off + lim], status=200)


def trades(tss):
    return [{"timestamp": ts, "transactionHash": f"tx{ts}"} for ts in tss]


def test_fetch_user_trades_pages_until_short_page():
    sess = FakeSession(trades([5, 4, 3]))
    out = fetch_user_trades(sess, cfg(page_size=2), "0xW", since_ts=0)
    assert [t["timestamp"] for t in out] == [5, 4, 3]
    assert sess.offsets == [0, 2]  # stopped on the short (final) page
    assert sess.params_seen[0]["user"] == "0xW"


def test_fetch_user_trades_stops_at_since_ts():
    sess = FakeSession(trades([5, 4, 3, 2, 1]))
    out = fetch_user_trades(sess, cfg(page_size=2), "0xW", since_ts=3)
    # stops as soon as it sees a trade at or before the cursor; nothing <= 3 kept
    assert [t["timestamp"] for t in out] == [5, 4]


def test_fetch_user_trades_stops_on_offset_cap_400():
    sess = FakeSession(trades(list(range(20, 0, -1))), cap=4)
    out = fetch_user_trades(sess, cfg(page_size=2, max_pages=100), "0xW", since_ts=0)
    # offset 0 and 2 serve pages; offset 4 hits the cap (400) and stops cleanly
    assert [t["timestamp"] for t in out] == [20, 19, 18, 17]
    assert sess.offsets == [0, 2, 4]


def test_fetch_user_trades_respects_max_pages():
    sess = FakeSession(trades(list(range(20, 0, -1))))
    out = fetch_user_trades(sess, cfg(page_size=2, max_pages=3), "0xW", since_ts=0)
    assert len(out) == 6  # 3 pages * 2
    assert sess.offsets == [0, 2, 4]


def test_fetch_user_trades_empty_history():
    sess = FakeSession([])
    out = fetch_user_trades(sess, cfg(), "0xW", since_ts=0)
    assert out == []


# --- run_backfill batching / per-batch checkpointing ---------------------------
#
# The from-empty-top-500 OOM guard: run_backfill must fetch+fold+checkpoint in
# wallet batches so peak memory is bounded by one batch of trades, never the whole
# run. These tests drive run_backfill with the network and disk stubbed out and
# assert the batching, per-batch checkpoint cadence, and correctness of the folded
# ledger + advanced cursors.


def _fake_trade(wallet, ts):
    """A trade dict shaped exactly like the /trades API rows fold_trades_to_ledger
    consumes."""
    return {
        "proxyWallet": wallet,
        "conditionId": f"mkt-{wallet}",
        "asset": f"tok-{wallet}",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.5,
        "size": 10.0,
        "timestamp": ts,
        "title": "Q?",
        "slug": "q",
        "transactionHash": f"tx-{wallet}-{ts}",
    }


class _BackfillHarness:
    """Stubs every I/O boundary run_backfill touches, keeping the real fold /
    dedup / refresh logic so the folded ledger is exercised for real. Records the
    ledger row-count and cursor snapshot at each checkpoint save."""

    def __init__(self, wallets, trades_per_wallet):
        self.wallets = wallets
        self.trades_per_wallet = trades_per_wallet  # wallet -> [ts, ...]
        self.ledger_saves = []   # row count captured at each save_ledger
        self.cursor_saves = []   # dict snapshot captured at each save_backfill_cursors
        self.fold_calls = 0
        self.last_ledger = pd.DataFrame()
        self.cursors = {}

    def install(self, monkeypatch, batch_size, checkpoint_min_rows=100_000):
        cfg = {
            "ingest": {"max_retries": 1, "retry_backoff_sec": 0},
            "backfill": {
                "wallet_batch_size": batch_size,
                "top_n_wallets": len(self.wallets),
                "checkpoint_min_rows": checkpoint_min_rows,
            },
        }
        monkeypatch.setattr(backfill, "ensure_dirs", lambda: None)
        monkeypatch.setattr(backfill, "load_config", lambda: cfg)
        monkeypatch.setattr(backfill, "make_session", lambda **k: None)
        monkeypatch.setattr(backfill, "select_wallets", lambda c, m=None: list(self.wallets))
        monkeypatch.setattr(backfill, "load_backfill_cursors", lambda: dict(self.cursors))
        monkeypatch.setattr(
            backfill, "load_ledger", lambda: pd.DataFrame(columns=LEDGER_COLUMNS),
        )
        monkeypatch.setattr(backfill, "load_resolutions_cache", lambda: pd.DataFrame())
        # resolution pass is a no-op that returns the (empty) cache unchanged
        monkeypatch.setattr(
            backfill, "update_resolutions",
            lambda sess, c, ids, existing, max_fetches=None: existing,
        )
        monkeypatch.setattr(backfill, "save_resolutions_cache", lambda df: None)

        def fake_fetch(session, c, wallet, since):
            return [
                _fake_trade(wallet, ts)
                for ts in self.trades_per_wallet.get(wallet, [])
                if ts > since
            ]

        monkeypatch.setattr(backfill, "fetch_user_trades", fake_fetch)

        real_fold = fold_trades_to_ledger

        def counting_fold(trades, resolutions):
            self.fold_calls += 1
            return real_fold(trades, resolutions)

        monkeypatch.setattr(backfill, "fold_trades_to_ledger", counting_fold)

        def fake_save_ledger(df):
            self.last_ledger = df
            self.ledger_saves.append(len(df))

        def fake_save_cursors(cur):
            self.cursor_saves.append(dict(cur))

        monkeypatch.setattr(backfill, "save_ledger", fake_save_ledger)
        monkeypatch.setattr(backfill, "save_backfill_cursors", fake_save_cursors)


def test_run_backfill_folds_per_batch_but_saves_once_below_threshold(monkeypatch):
    # The memory fix (fold per batch, freeing the raw dicts) is DECOUPLED from the
    # save cadence. Below checkpoint_min_rows the whole run persists exactly once
    # at the end — as the pre-batching code did — so a caught-up nightly run pays
    # no per-batch I/O penalty, yet folding still happens per batch.
    wallets = [f"0xW{i}" for i in range(5)]
    trades_per_wallet = {w: [100 + i] for i, w in enumerate(wallets)}
    h = _BackfillHarness(wallets, trades_per_wallet)
    h.install(monkeypatch, batch_size=2, checkpoint_min_rows=100_000)

    ledger = backfill.run_backfill()

    # 5 wallets / batch 2 -> 3 batches -> 3 per-batch fold calls (one per non-empty
    # batch, NOT one giant fold at the end: that's the memory-bounding property).
    assert h.fold_calls == 3
    # Only 5 total new rows (< threshold) -> a single final save, not per-batch.
    assert h.ledger_saves == [5]
    assert h.cursor_saves[-1] == {w: 100 + i for i, w in enumerate(wallets)}
    assert len(ledger) == 5
    assert set(ledger["wallet"]) == set(wallets)


def test_run_backfill_checkpoints_when_threshold_reached(monkeypatch):
    # With a tiny checkpoint threshold, each batch that folds rows triggers a
    # ledger+cursor checkpoint mid-run — bounding crash re-fetch on a long
    # from-empty run. Ledger grows monotonically across checkpoints.
    wallets = [f"0xW{i}" for i in range(5)]
    trades_per_wallet = {w: [100 + i] for i, w in enumerate(wallets)}
    h = _BackfillHarness(wallets, trades_per_wallet)
    h.install(monkeypatch, batch_size=2, checkpoint_min_rows=1)

    backfill.run_backfill()

    assert h.fold_calls == 3
    # batch1 folds 2 (>=1 -> save), batch2 -> 4, batch3 -> 5, then final save.
    assert h.ledger_saves == [2, 4, 5, 5]
    assert len(h.cursor_saves) == 4


def test_run_backfill_empty_batches_save_once(monkeypatch):
    # Wallets whose cursor is already caught up (no new trades) fold nothing and
    # trigger no mid-run checkpoint; the mandatory final save still runs once
    # (matching the pre-batching always-save-at-end behavior).
    wallets = [f"0xW{i}" for i in range(4)]
    trades_per_wallet = {w: [50] for w in wallets}
    h = _BackfillHarness(wallets, trades_per_wallet)
    h.cursors = {w: 50 for w in wallets}  # already at/after the only trade -> nothing new
    h.install(monkeypatch, batch_size=2)

    ledger = backfill.run_backfill()

    assert h.fold_calls == 0          # nothing to fold
    assert len(ledger) == 0
    assert h.ledger_saves == [0]      # single final save of the (empty) ledger
