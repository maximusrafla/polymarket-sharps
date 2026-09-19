import os

import pandas as pd

from src import common, ingest
from src.common import LEDGER_COLUMNS
from src.ingest import fold_trades_to_ledger, trade_key, update_resolutions


def test_trade_key_is_unique_per_fill_identity():
    t1 = {"transactionHash": "0xabc", "proxyWallet": "0xw1", "asset": "1", "side": "BUY"}
    t2 = {"transactionHash": "0xabc", "proxyWallet": "0xw1", "asset": "1", "side": "SELL"}
    assert trade_key(t1) != trade_key(t2)
    assert trade_key(t1) == trade_key(dict(t1))


def test_fold_trades_to_ledger_attaches_resolution():
    trades = [
        {
            "proxyWallet": "0xwallet1",
            "conditionId": "0xmarket1",
            "asset": "token_yes",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.4,
            "size": 10.0,
            "timestamp": 1000,
            "title": "Will X happen?",
            "slug": "will-x-happen",
            "transactionHash": "0xtx1",
        },
        {
            "proxyWallet": "0xwallet2",
            "conditionId": "0xmarket2",
            "asset": "token_no",
            "outcome": "No",
            "side": "BUY",
            "price": 0.7,
            "size": 5.0,
            "timestamp": 1001,
            "title": "Will Y happen?",
            "slug": "will-y-happen",
            "transactionHash": "0xtx2",
        },
    ]
    resolutions = pd.DataFrame(
        [
            {
                "market_id": "0xmarket1",
                "token_id": "token_yes",
                "outcome": "Yes",
                "resolved": True,
                "resolved_value": 1.0,
                "closed": True,
            }
        ]
    )

    ledger = fold_trades_to_ledger(trades, resolutions)

    assert len(ledger) == 2
    row1 = ledger.loc[ledger["wallet"] == "0xwallet1"].iloc[0]
    assert row1["resolved"] == True  # noqa: E712
    assert row1["resolved_value"] == 1.0
    assert row1["entry_price"] == 0.4

    row2 = ledger.loc[ledger["wallet"] == "0xwallet2"].iloc[0]
    assert row2["resolved"] == False  # noqa: E712
    assert pd.isna(row2["resolved_value"])


def test_fold_trades_to_ledger_empty_input():
    ledger = fold_trades_to_ledger([], pd.DataFrame())
    assert ledger.empty


# --- update_resolutions: per-run cap + unseen-first ordering -------------------
#
# ingest caps resolution GETs per run (ingest.max_resolution_fetches_per_run) so it
# fits the ~5-min cron cadence; the recheck backlog drains across runs. Fresh
# (unseen) markets must always be resolved before rechecks of already-known-open
# ones, so a just-placed bet's market gets resolved promptly even under the cap.


class _RecordingSession:
    """Serves a minimal open-market resolution and records every market_id GET."""

    def __init__(self):
        self.fetched = []

    def get(self, url, timeout=None):
        cid = url.rstrip("/").rsplit("/", 1)[-1]
        self.fetched.append(cid)
        return _MarketResp(cid)


class _MarketResp:
    def __init__(self, cid):
        self.status_code = 200
        self._cid = cid

    def raise_for_status(self):
        pass

    def json(self):
        # An open market (closed=False) so it stays in the recheck pool.
        return {
            "closed": False,
            "tokens": [{"token_id": f"{self._cid}-t", "outcome": "Yes", "price": 0}],
        }


def _res_cfg():
    return {
        "data_source": {"clob_api_base": "http://clob"},
        "ingest": {
            "request_timeout_sec": 5,
            "resolution_recheck": True,
            "resolution_fetch_workers": 1,  # serial -> deterministic fetch order
        },
    }


def test_update_resolutions_caps_and_prioritizes_unseen():
    # existing cache: two already-known OPEN markets (rechecks) + one resolved.
    existing = pd.DataFrame(
        [
            {"market_id": "OPEN1", "token_id": "OPEN1-t", "outcome": "Yes",
             "resolved": False, "resolved_value": None, "closed": False},
            {"market_id": "OPEN2", "token_id": "OPEN2-t", "outcome": "Yes",
             "resolved": False, "resolved_value": None, "closed": False},
            {"market_id": "DONE", "token_id": "DONE-t", "outcome": "Yes",
             "resolved": True, "resolved_value": 1.0, "closed": True},
        ]
    )
    # This run just saw trades on two brand-new markets.
    new_markets = {"NEW1", "NEW2"}
    sess = _RecordingSession()

    update_resolutions(sess, _res_cfg(), new_markets, existing, max_fetches=2)

    # Exactly the cap was fetched, and both were the unseen (fresh) markets —
    # the already-known-open rechecks were deferred to a later run.
    assert len(sess.fetched) == 2
    assert set(sess.fetched) == {"NEW1", "NEW2"}
    assert "OPEN1" not in sess.fetched and "OPEN2" not in sess.fetched


def test_update_resolutions_uncapped_fetches_all():
    existing = pd.DataFrame(
        [
            {"market_id": "OPEN1", "token_id": "OPEN1-t", "outcome": "Yes",
             "resolved": False, "resolved_value": None, "closed": False},
        ]
    )
    sess = _RecordingSession()
    update_resolutions(sess, _res_cfg(), {"NEW1"}, existing, max_fetches=None)
    # No cap -> the new market AND the open-market recheck are both fetched.
    assert set(sess.fetched) == {"NEW1", "OPEN1"}


# --- free CLOB metadata capture: the resolution output must stay byte-identical --

class _MetaMarketResp:
    """A CLOSED market carrying the speed-metadata fields the free capture wants."""

    def __init__(self, cid):
        self.status_code = 200
        self._cid = cid

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "closed": True,
            "accepting_order_timestamp": "2026-06-29T00:00:00Z",
            "end_date_iso": "2026-06-30T00:00:00Z",
            "game_start_time": "2026-06-30T18:00:00Z",
            "tags": ["Sports", "Soccer"],
            "tokens": [
                {"token_id": f"{self._cid}-y", "outcome": "Yes", "price": 1},
                {"token_id": f"{self._cid}-n", "outcome": "No", "price": 0},
            ],
        }


class _MetaSession:
    def get(self, url, timeout=None):
        cid = url.rstrip("/").rsplit("/", 1)[-1]
        return _MetaMarketResp(cid)


def test_update_resolutions_meta_sink_is_byte_identical_and_captures():
    """The meta_sink is purely additive: the resolution frame with a sink must be
    byte-identical to the frame without one, and the sink collects the raw dicts."""
    markets = {"M1", "M2", "M3"}
    without = update_resolutions(_MetaSession(), _res_cfg(), markets, pd.DataFrame(),
                                 max_fetches=None)
    sink = {}
    with_sink = update_resolutions(_MetaSession(), _res_cfg(), markets, pd.DataFrame(),
                                   max_fetches=None, meta_sink=sink)
    # Resolution output unchanged (sort for order-independence across thread runs).
    a = without.sort_values(["market_id", "token_id"]).reset_index(drop=True)
    b = with_sink.sort_values(["market_id", "token_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # ...and the raw market dicts were captured, one per market.
    assert set(sink) == markets
    assert sink["M1"]["accepting_order_timestamp"] == "2026-06-29T00:00:00Z"


def test_update_resolutions_default_no_sink_unchanged():
    # Default (no meta_sink) leaves behavior exactly as before — no capture, no error.
    out = update_resolutions(_MetaSession(), _res_cfg(), {"M1"}, pd.DataFrame(), max_fetches=None)
    assert set(out["market_id"]) == {"M1"}
    assert len(out) == 2  # two tokens


# --- pager: the /trades feed is NOT time-ordered across pages ------------------
#
# Measured 2026-07-23: 12 of 19 page boundaries are inversions (excursions to
# ~4 min). fetch_new_trades must therefore sweep every page and filter purely by
# timestamp — breaking on the first stale row can abandon pages that still hold
# newer, never-seen trades. See DECISIONS.md "`/trades` pages are NOT time-ordered".


def _trade(tx, ts, wallet="0xw", asset="tok", side="BUY"):
    return {
        "transactionHash": tx,
        "proxyWallet": wallet,
        "asset": asset,
        "side": side,
        "timestamp": ts,
    }


def _pager_cfg(page_size=2, max_pages=5):
    return {"ingest": {"page_size": page_size, "max_pages_per_run": max_pages}}


def _serve(monkeypatch, pages):
    served = []

    def fake_page(session, cfg, offset, limit):
        served.append(offset)
        idx = offset // limit
        return pages[idx] if idx < len(pages) else []

    monkeypatch.setattr(ingest, "fetch_trades_page", fake_page)
    return served


def test_fetch_new_trades_does_not_stop_at_a_stale_row(monkeypatch):
    # Page 0 is entirely OLDER than the cursor floor; page 1 holds a NEW trade.
    # The old "break on first stale row" pager returned nothing here.
    pages = [
        [_trade("old1", 500), _trade("old2", 400)],
        [_trade("new1", 900), _trade("old3", 300)],
        [],
    ]
    served = _serve(monkeypatch, pages)
    cursor = {"max_timestamp": 800, "keys_at_max": []}

    new = ingest.fetch_new_trades(object(), _pager_cfg(), cursor)

    assert [t["transactionHash"] for t in new] == ["new1"]
    assert served[:2] == [0, 2]  # it kept sweeping past the stale page


def test_fetch_new_trades_filters_by_timestamp_and_dedups(monkeypatch):
    # ts == floor is only skipped for keys already recorded at the cursor; the
    # same fill can surface on two pages because the feed shifts under the sweep.
    at_max = _trade("seen", 800)
    dup = _trade("dup", 950)
    pages = [
        [at_max, _trade("fresh", 801)],
        [dup, dup],
        [_trade("older", 799)],
        [],
    ]
    _serve(monkeypatch, pages)
    cursor = {"max_timestamp": 800, "keys_at_max": [ingest.trade_key(at_max)]}

    new = ingest.fetch_new_trades(object(), _pager_cfg(), cursor)

    assert [t["transactionHash"] for t in new] == ["fresh", "dup"]


def test_fetch_new_trades_stops_on_short_page(monkeypatch):
    # A short page really is the end of the served feed — that break stays.
    pages = [[_trade("a", 900)], [_trade("b", 901), _trade("c", 902)]]
    served = _serve(monkeypatch, pages)
    cursor = {"max_timestamp": 0, "keys_at_max": []}

    new = ingest.fetch_new_trades(object(), _pager_cfg(), cursor)

    assert [t["transactionHash"] for t in new] == ["a"]
    assert served == [0]


# --- incremental (delta) ledger write ----------------------------------------


def _ledger_rows(n, start=0, resolved=False):
    return pd.DataFrame(
        {
            "wallet": [f"0xw{i % 3}" for i in range(start, start + n)],
            "market_id": [f"m{i % 5}" for i in range(start, start + n)],
            "token_id": [f"t{i % 7}" for i in range(start, start + n)],
            "outcome": ["Yes"] * n,
            "side": ["BUY"] * n,
            "entry_price": [0.4] * n,
            "size": [1.0] * n,
            "timestamp": list(range(start, start + n)),
            "resolved": [resolved] * n,
            "resolved_value": [1.0 if resolved else None] * n,
            "question": ["q?"] * n,
            "slug": ["q"] * n,
            "tx_hash": [f"0xtx{i}" for i in range(start, start + n)],
            "speed_bucket": ["unknown"] * n,
        }
    )[LEDGER_COLUMNS]


def _point_paths(monkeypatch, tmp_path):
    """Point both the ledger and the delta dir at a tmp dir (ingest re-exports
    the names, so both modules have to be patched)."""
    ledger = tmp_path / "bet_ledger.parquet"
    delta = tmp_path / "ledger_delta"
    for mod in (common, ingest):
        monkeypatch.setattr(mod, "BET_LEDGER_PATH", ledger, raising=False)
        monkeypatch.setattr(mod, "LEDGER_DELTA_DIR", delta, raising=False)
    return ledger, delta


def _inline_merge(ledger_path, new_rows, resolutions):
    """What ingest used to do inline on every run: stream the whole ledger through
    a rewrite with the new rows merged in."""
    if not ledger_path.exists():
        common.atomic_to_parquet(
            new_rows.drop_duplicates(subset=ingest.LEDGER_DEDUP_KEY, keep="last"),
            ledger_path,
            compression=common.LEDGER_COMPRESSION,
            index=False,
        )
        return
    tmp = ledger_path.with_suffix(".tmp")
    ingest.stream_merge_ledger(ledger_path, tmp, new_rows, resolutions)
    os.replace(tmp, ledger_path)


def test_delta_fold_is_byte_identical_to_the_inline_merge(monkeypatch, tmp_path):
    """The incremental path must produce exactly the ledger the old path did.

    Three "runs", the third re-sending a row from the first (the overlap a rolling
    poll always produces) so the dedup/supersede ordering is exercised."""
    resolutions = pd.DataFrame(
        columns=["market_id", "token_id", "resolved", "resolved_value"]
    )
    runs = [_ledger_rows(50), _ledger_rows(40, start=50), _ledger_rows(10, start=5)]

    # Old path.
    old_ledger = tmp_path / "old.parquet"
    for rows in runs:
        _inline_merge(old_ledger, ingest.normalize_ledger_dtypes(rows), resolutions)

    # New path: append a delta part per run, fold once at the end.
    ledger, delta = _point_paths(monkeypatch, tmp_path)
    for i, rows in enumerate(runs):
        ingest.append_ledger_delta(rows, batch_ts=i + 1)
    ingest.fold_delta_into_ledger(resolutions=resolutions)

    # Same rows, same order, same dtypes. (Parquet *bytes* are only comparable
    # when the two paths also share a row-group layout — they do in production,
    # where the ledger already exists and both paths stream the same source file;
    # see the full-ledger byte-identity harness in HANDOFF.md. Here the new path
    # creates the ledger from scratch in one write, so compare content.)
    old = pd.read_parquet(old_ledger).reset_index(drop=True)
    new = pd.read_parquet(ledger).reset_index(drop=True)
    pd.testing.assert_frame_equal(old, new)
    assert common.delta_part_paths() == []  # folded parts are cleaned up


def test_delta_fold_in_several_passes_matches_a_single_pass(monkeypatch, tmp_path):
    """Fold grouping bounds memory; it must not change the result."""
    resolutions = pd.DataFrame(
        columns=["market_id", "token_id", "resolved", "resolved_value"]
    )
    runs = [_ledger_rows(30), _ledger_rows(30, start=30), _ledger_rows(30, start=15)]

    ledger, delta = _point_paths(monkeypatch, tmp_path)
    for i, rows in enumerate(runs):
        ingest.append_ledger_delta(rows, batch_ts=i + 1)
    ingest.fold_delta_into_ledger(resolutions=resolutions, max_rows_per_pass=10**9)
    one_pass = pd.read_parquet(ledger).reset_index(drop=True)
    ledger.unlink()

    for i, rows in enumerate(runs):
        ingest.append_ledger_delta(rows, batch_ts=i + 1)
    # 30 rows per part -> one part per pass.
    ingest.fold_delta_into_ledger(resolutions=resolutions, max_rows_per_pass=31)

    pd.testing.assert_frame_equal(pd.read_parquet(ledger).reset_index(drop=True), one_pass)


def test_fold_delta_is_idempotent_and_applies_resolutions(monkeypatch, tmp_path):
    ledger, delta = _point_paths(monkeypatch, tmp_path)
    rows = _ledger_rows(20)
    ingest.append_ledger_delta(rows, batch_ts=1)
    ingest.fold_delta_into_ledger(
        resolutions=pd.DataFrame(
            columns=["market_id", "token_id", "resolved", "resolved_value"]
        )
    )
    first = pd.read_parquet(ledger)
    assert len(first) == 20
    assert not first["resolved"].any()

    # A no-op fold (nothing pending) still refreshes resolutions that landed since
    # — the refresh used to ride along on every ingest run.
    resolutions = pd.DataFrame(
        [{"market_id": "m0", "token_id": "t0", "resolved": True, "resolved_value": 1.0}]
    )
    ingest.fold_delta_into_ledger(resolutions=resolutions)
    after = pd.read_parquet(ledger)
    assert len(after) == 20
    resolved = after.loc[after["resolved"]]
    assert len(resolved) == len(
        first[(first["market_id"] == "m0") & (first["token_id"] == "t0")]
    ) > 0
    assert (resolved["resolved_value"] == 1.0).all()


def test_ingest_delta_part_naming_is_chronological(monkeypatch, tmp_path):
    _point_paths(monkeypatch, tmp_path)
    ingest.append_ledger_delta(_ledger_rows(2), batch_ts=1_800_000_000)
    ingest.append_ledger_delta(_ledger_rows(2, start=2), batch_ts=1_700_000_000)
    names = [p.name for p in common.delta_part_paths()]
    assert names == sorted(names)
    assert "1700000000" in names[0]
