"""Tests for src/slow_deepen.py — the shortlist deepening (Project 3 step 5b).

Covers shortlist selection, fill-key dedup, and the per-user fetch->fold->checkpoint
loop against a FakeSession (no network), including resumability via cursors and the
isolation invariant (writes only the deepening dir).
"""

import json

import pandas as pd
import requests

import src.slow_deepen as sd
from src.common import LEDGER_COLUMNS


# --- shortlist selection ---------------------------------------------------

def test_select_shortlist_filters_and_sorts():
    scores = pd.DataFrame({
        "wallet": ["a", "b", "c", "d"],
        "screen_score": [0.20, 0.04, 0.11, 0.05],
    })
    short = sd.select_shortlist(gate=0.05, scores=scores)
    assert list(short["wallet"]) == ["a", "c", "d"]     # b (0.04) excluded
    assert list(short["screen_score"]) == [0.20, 0.11, 0.05]  # sorted desc


# --- dedup on the fill-identity key ---------------------------------------

def test_dedup_keeps_last_on_fill_key():
    df = pd.DataFrame({
        "tx_hash": ["x", "x", "y"], "wallet": ["w", "w", "w"],
        "token_id": ["t", "t", "u"], "side": ["BUY", "BUY", "BUY"],
        "entry_price": [0.4, 0.5, 0.6],
    })
    out = sd._dedup(df)
    assert len(out) == 2
    assert out[out["tx_hash"] == "x"]["entry_price"].iloc[0] == 0.5  # kept last


# --- per-user deepening against a FakeSession -----------------------------

class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            e = requests.exceptions.HTTPError(); e.response = self; raise e


class _FakeUserSession:
    """Serves each wallet's /trades?user= tape newest-first by offset/limit."""

    def __init__(self, trades_by_wallet):
        self.trades = trades_by_wallet  # wallet -> [trade dicts newest-first]

    def get(self, url, params=None, timeout=None):
        w = params["user"]
        off, lim = params["offset"], params["limit"]
        return _Resp(self.trades.get(w, [])[off:off + lim])


def _t(wallet, ts):
    return {"proxyWallet": wallet, "conditionId": f"mkt-{wallet}", "asset": f"tok-{wallet}",
            "outcome": "Yes", "side": "BUY", "price": 0.5, "size": 10.0, "timestamp": ts,
            "title": "Q?", "slug": "q", "transactionHash": f"tx-{wallet}-{ts}"}


def _cfg():
    return {
        "data_source": {"trades_api_base": "http://data"},
        "ingest": {"request_timeout_sec": 5},
        "backfill": {"page_size": 2, "max_pages_per_wallet": 10, "sleep_between_requests_sec": 0},
    }


def _point_deepen_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "SLOW_DEEPEN_DIR", tmp_path, raising=False)
    monkeypatch.setattr(sd, "DEEP_TRADES_PATH", tmp_path / "deep_trades.parquet", raising=False)
    monkeypatch.setattr(sd, "DEEP_CURSORS_PATH", tmp_path / "deep_cursors.json", raising=False)


def test_deepen_wallets_folds_and_is_resumable(monkeypatch, tmp_path):
    _point_deepen_dir(monkeypatch, tmp_path)
    sess = _FakeUserSession({
        "wA": [_t("wA", 30), _t("wA", 20), _t("wA", 10)],
        "wB": [_t("wB", 5)],
    })
    n = sd.deepen_wallets(["wA", "wB"], cfg=_cfg(), checkpoint_every=1, session=sess)
    assert n == 4
    deep = sd.load_deep_trades()
    assert set(deep.columns) == set(LEDGER_COLUMNS)
    assert set(deep["wallet"]) == {"wA", "wB"}
    assert len(deep) == 4
    # Cursor advanced to each wallet's newest ts -> a re-run fetches nothing new.
    cursors = json.loads((tmp_path / "deep_cursors.json").read_text())
    assert cursors["wA"]["max_timestamp"] == 30
    n2 = sd.deepen_wallets(["wA", "wB"], cfg=_cfg(), checkpoint_every=1, session=sess)
    assert n2 == 0                       # incremental: nothing newer than the cursor
    assert len(sd.load_deep_trades()) == 4  # unchanged


def test_deepen_isolation_writes_only_deepen_dir(monkeypatch, tmp_path):
    # The shared ledger path is NOT among the files this step writes.
    _point_deepen_dir(monkeypatch, tmp_path)
    sess = _FakeUserSession({"wA": [_t("wA", 10)]})
    sd.deepen_wallets(["wA"], cfg=_cfg(), checkpoint_every=1, session=sess)
    written = {p.name for p in tmp_path.iterdir()}
    assert written == {"deep_trades.parquet", "deep_cursors.json"}
    assert "bet_ledger.parquet" not in written
