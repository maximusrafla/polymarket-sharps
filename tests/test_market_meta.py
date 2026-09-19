"""Tests for src/market_meta.py — pure logic + Gamma pagination against a
FakeSession. No live network.

The load-bearing test is `test_micro_metadata_trap_*`: it proves
`combine_lifespan` never lets the poisoned CLOB accepting_order→end_date window
(a −24h / midnight-floor artifact) beat a short observed tape span — the exact
mechanism (§1.1b) that would otherwise re-admit ~75% of micro-crypto as "slow".
It fails loudly if a CLOB-derived lifespan ever wins over a 2-minute tape span.
"""

import math

import pandas as pd
import requests

import src.common as common
import src.market_meta as mm
from src.common import LEDGER_COLUMNS
from src.market_meta import (
    MARKET_META_COLUMNS,
    assign_speed_bucket,
    build_meta_row,
    classify_speed_bucket,
    combine_lifespan,
    enumerate_gamma_day,
    market_bucket_map,
    merge_meta,
    parse_clob_market_meta,
    parse_gamma_market_row,
    populate_speed_bucket,
    refresh_clob_fields,
    speed_thresholds,
    _iso_to_ts,
)

HOUR = 3600
DAY = 86400


# Anchor epoch: 2026-06-30 00:00:00Z. Timestamps below are offsets from it.
BASE = 1782777600  # 2026-06-30T00:00:00Z


# --- timestamp parsing ----------------------------------------------------

def test_iso_to_ts_handles_clob_and_gamma_shapes():
    # 'Z' suffix (CLOB), space + bare '+00' offset (Gamma closedTime), and a
    # naive string (assumed UTC) all parse to the same instant family.
    t_z = _iso_to_ts("2026-06-30T22:15:35Z")
    t_gamma = _iso_to_ts("2026-06-30 22:15:35+00")
    t_naive = _iso_to_ts("2026-06-30T22:15:35")
    assert abs(t_z - t_gamma) < 1e-6
    assert abs(t_z - t_naive) < 1e-6


def test_iso_to_ts_bad_input_is_nan_not_error():
    for bad in (None, "", "not-a-date", float("nan")):
        assert math.isnan(_iso_to_ts(bad))
    # Numeric epochs pass through.
    assert _iso_to_ts(1782777600) == 1782777600.0


# --- parsing --------------------------------------------------------------

def test_parse_clob_market_meta_extracts_fields():
    raw = {
        "accepting_order_timestamp": "2026-06-29T00:00:00Z",
        "end_date_iso": "2026-06-30T00:00:00Z",
        "game_start_time": "2026-06-30T18:00:00Z",
        "tags": ["Sports", "Soccer", "Recurring"],
        "closed": True,
    }
    row = parse_clob_market_meta("0xabc", raw)
    assert row["market_id"] == "0xabc"
    assert row["tags"] == "Sports|Soccer|Recurring"
    assert row["closed"] is True
    assert not math.isnan(row["open_ts_clob"])
    assert not math.isnan(row["game_start_ts"])


def test_parse_clob_market_meta_none_on_garbage():
    assert parse_clob_market_meta("0xabc", None) is None
    assert parse_clob_market_meta("0xabc", ["not", "a", "dict"]) is None


def test_parse_gamma_market_row_extracts_precise_fields():
    raw = {
        "conditionId": "0xdef",
        "startDate": "2026-06-01T00:00:00Z",
        "closedTime": "2026-06-30 22:15:35+00",
        "volumeNum": 1234567.0,
        "liquidityNum": 8900.0,
    }
    row = parse_gamma_market_row(raw)
    assert row["market_id"] == "0xdef"
    assert row["volume"] == 1234567.0
    assert row["closed_ts_gamma"] > row["start_ts_gamma"]


def test_parse_gamma_market_row_none_without_condition_id():
    assert parse_gamma_market_row({"startDate": "2026-06-01T00:00:00Z"}) is None


# --- combine_lifespan: the soundness rule ---------------------------------

def test_combine_lifespan_prefers_gamma_precise():
    gamma = {
        "start_ts_gamma": BASE,
        "closed_ts_gamma": BASE + 7 * 86400,  # a 7-day market
        "volume": 700000.0,
    }
    lifespan, velocity, source = combine_lifespan(clob_row={}, gamma_row=gamma,
                                                  tape_span=(BASE + 1000, BASE + 2000))
    assert source == "gamma"
    assert lifespan == 7 * 86400
    # velocity = volume / lifespan, computed ONLY on the precise branch.
    assert abs(velocity - 700000.0 / (7 * 86400)) < 1e-9


def test_combine_lifespan_falls_back_to_tape_lower_bound():
    # No Gamma record -> observed tape span is the lower bound; velocity NaN
    # (no sound denominator / no volume).
    lifespan, velocity, source = combine_lifespan(
        clob_row={}, gamma_row=None, tape_span=(BASE, BASE + 3 * 86400))
    assert source == "clob_tape"
    assert lifespan == 3 * 86400
    assert math.isnan(velocity)


def test_combine_lifespan_unknown_when_neither_available():
    # No Gamma, and a single observed trade (t_min == t_max -> zero span) -> parked.
    lifespan, velocity, source = combine_lifespan(
        clob_row={}, gamma_row=None, tape_span=(BASE, BASE))
    assert source == "unknown"
    assert math.isnan(lifespan)
    assert math.isnan(velocity)


def test_combine_lifespan_ignores_degenerate_gamma():
    # closedTime <= startDate is not a valid precise lifespan -> fall through to tape.
    gamma = {"start_ts_gamma": BASE + 100, "closed_ts_gamma": BASE, "volume": 1.0}
    lifespan, _, source = combine_lifespan(
        clob_row={}, gamma_row=gamma, tape_span=(BASE, BASE + 600))
    assert source == "clob_tape"
    assert lifespan == 600


# --- THE LOAD-BEARING TEST: the planted micro-metadata trap ----------------

def _micro_clob_raw():
    """A 5-minute micro-crypto market's real CLOB metadata shape (§1.1b):
    accepting_order_timestamp is exactly resolution −24h, end_date_iso is the
    midnight floor of the end day. Both are series-creation artifacts — the market
    actually traded for ~2 minutes."""
    resolution = BASE + 12 * 3600  # resolves midday
    return {
        "accepting_order_timestamp": pd.Timestamp(resolution - 86400, unit="s", tz="UTC").isoformat(),
        "end_date_iso": pd.Timestamp(BASE, unit="s", tz="UTC").isoformat(),  # midnight floor
        "game_start_time": None,
        "tags": ["Crypto", "Recurring", "Hide From New"],
        "closed": True,
    }


def test_micro_metadata_trap_tape_span_beats_clob_lifespan():
    """combine_lifespan must NOT return the ~24h CLOB accepting_order→end_date
    window for a market that only traded ~2 minutes. Gamma excludes micro, so the
    only signal is the 2-minute observed tape span -> lifespan ~120s, fast."""
    resolution = BASE + 12 * 3600
    # Observed tape: a 2-minute burst just before resolution (dense micro coverage).
    tape_span = (resolution - 120, resolution)
    clob = parse_clob_market_meta("0xmicro", _micro_clob_raw())

    lifespan, velocity, source = combine_lifespan(clob, gamma_row=None, tape_span=tape_span)

    # The CLOB window is ~24h; if it ever leaked into lifespan this assert fires.
    assert lifespan == 120, (
        f"micro trap: lifespan {lifespan}s should be the 120s tape span, not the "
        f"CLOB accepting_order→end_date window (~86400s)"
    )
    assert lifespan < 6 * 3600, "micro market must never clear a 6h slow gate"
    assert source == "clob_tape"
    assert math.isnan(velocity)


def test_micro_metadata_trap_end_to_end_row():
    """Same trap through build_meta_row: the assembled row carries the poisoned
    CLOB timestamps (for later H-anchoring) but its lifespan_s is the tape lower
    bound, and speed_source is honest about resting on that bound."""
    resolution = BASE + 12 * 3600
    row = build_meta_row(
        "0xmicro", _micro_clob_raw(), gamma_row=None,
        tape_span=(resolution - 120, resolution),
    )
    assert row["lifespan_s"] == 120
    assert row["speed_source"] == "clob_tape"
    # The poisoned CLOB fields are still persisted (H-anchor inputs), just not used
    # as a lifespan source — provenance is honest, nothing is silently dropped.
    assert not math.isnan(row["open_ts_clob"])
    assert not math.isnan(row["end_ts_clob"])
    # A naive CLOB lifespan would have been ~24h; prove we're nowhere near it.
    naive_clob_lifespan = row["end_ts_clob"] - row["open_ts_clob"]
    assert naive_clob_lifespan > 6 * 3600  # the trap really is ~half a day wide...
    assert row["lifespan_s"] < naive_clob_lifespan / 100  # ...and we did not fall for it


# --- build_meta_row: a genuine slow market --------------------------------

def test_build_meta_row_genuine_slow_market_uses_gamma():
    gamma = parse_gamma_market_row({
        "conditionId": "0xslow",
        "startDate": "2026-06-01T00:00:00Z",
        "closedTime": "2026-06-30T00:00:00Z",  # ~29-day market
        "volumeNum": 5_000_000.0,
        "liquidityNum": 40000.0,
    })
    clob = {
        "accepting_order_timestamp": "2026-06-01T00:00:00Z",
        "end_date_iso": "2026-06-30T00:00:00Z",
        "tags": ["Politics"],
        "closed": True,
    }
    row = build_meta_row("0xslow", clob, gamma, tape_span=(BASE - 20 * 86400, BASE - 1000))
    assert row["speed_source"] == "gamma"
    assert row["lifespan_s"] == 29 * 86400
    assert row["volume"] == 5_000_000.0
    assert not math.isnan(row["velocity"])
    assert set(row.keys()) - {"first_seen", "last_seen"} == set(MARKET_META_COLUMNS) - {"first_seen", "last_seen"}


# --- merge_meta: upsert, keep first_seen, never delete ---------------------

def test_merge_meta_upsert_preserves_first_seen_and_refreshes():
    existing = merge_meta(pd.DataFrame(), [
        build_meta_row("0xa", None, None, (BASE, BASE + 600)),
        build_meta_row("0xb", None, None, (BASE, BASE + 600)),
    ], now_ts=1000)
    assert set(existing["market_id"]) == {"0xa", "0xb"}
    assert list(existing.columns) == MARKET_META_COLUMNS

    # Re-see 0xa with a longer span; 0xb is absent this round (must NOT be deleted).
    updated = merge_meta(existing, [
        build_meta_row("0xa", None, None, (BASE, BASE + 5 * 86400)),
    ], now_ts=2000)
    assert set(updated["market_id"]) == {"0xa", "0xb"}  # no-drop invariant

    a = updated[updated["market_id"] == "0xa"].iloc[0]
    assert a["first_seen"] == 1000       # preserved
    assert a["last_seen"] == 2000        # refreshed
    assert a["lifespan_s"] == 5 * 86400  # mutable field updated

    b = updated[updated["market_id"] == "0xb"].iloc[0]
    assert b["first_seen"] == 1000
    assert b["last_seen"] == 1000        # untouched — not re-seen


def test_merge_meta_empty_inputs():
    assert merge_meta(pd.DataFrame(), [], now_ts=1).empty
    df = merge_meta(pd.DataFrame(), [build_meta_row("0xa", None, None, (BASE, BASE + 10))], now_ts=1)
    assert merge_meta(df, [], now_ts=2).equals(df) or set(merge_meta(df, [], now_ts=2)["market_id"]) == {"0xa"}


# --- Gamma pagination against a FakeSession -------------------------------

class _FakeResp:
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


class _FakeGammaSession:
    """Serves a fixed list of Gamma market dicts by offset/limit."""

    def __init__(self, markets):
        self.markets = markets
        self.offsets = []

    def get(self, url, params=None, timeout=None):
        off, lim = params["offset"], params["limit"]
        self.offsets.append(off)
        return _FakeResp(self.markets[off:off + lim], status=200)


# --- refresh_clob_fields: the free ingest-time capture, no clobbering -----

def _micro_raw_dict():
    return {
        "accepting_order_timestamp": "2026-06-29T00:00:00Z",
        "end_date_iso": "2026-06-30T00:00:00Z",
        "game_start_time": "2026-06-30T18:00:00Z",
        "tags": ["Crypto", "Recurring"],
        "closed": True,
    }


def test_refresh_clob_fields_preserves_ledger_derived():
    # An existing sidecar row whose lifespan came from the tape (clob_tape) — the
    # exact thing a cheap CLOB refresh must NOT overwrite.
    existing = merge_meta(pd.DataFrame(), [
        build_meta_row("0xa", None, None, (BASE, BASE + 4 * 86400)),
    ], now_ts=1000)
    assert existing.iloc[0]["speed_source"] == "clob_tape"
    assert existing.iloc[0]["lifespan_s"] == 4 * 86400

    out = refresh_clob_fields({"0xa": _micro_raw_dict()}, now_ts=2000, existing=existing)
    row = out[out["market_id"] == "0xa"].iloc[0]
    # CLOB fields now populated + last_seen refreshed...
    assert not math.isnan(row["open_ts_clob"])
    assert row["tags"] == "Crypto|Recurring"
    assert row["last_seen"] == 2000
    # ...but the ledger-derived fields and first_seen are untouched.
    assert row["lifespan_s"] == 4 * 86400
    assert row["speed_source"] == "clob_tape"
    assert row["tape_t_max"] == BASE + 4 * 86400
    assert row["first_seen"] == 1000


def test_refresh_clob_fields_inserts_new_market_parked():
    # A market not yet in the sidecar is inserted with CLOB fields but parked
    # (no tape span yet -> lifespan NaN, speed_source unknown) until the next full build.
    out = refresh_clob_fields({"0xnew": _micro_raw_dict()}, now_ts=5000,
                              existing=pd.DataFrame(columns=MARKET_META_COLUMNS))
    row = out[out["market_id"] == "0xnew"].iloc[0]
    assert row["speed_source"] == "unknown"
    assert math.isnan(row["lifespan_s"])
    assert not math.isnan(row["end_ts_clob"])  # CLOB fields captured
    assert row["first_seen"] == 5000


def test_refresh_clob_fields_empty_capture_is_noop():
    existing = merge_meta(pd.DataFrame(), [
        build_meta_row("0xa", None, None, (BASE, BASE + 600)),
    ], now_ts=1)
    assert refresh_clob_fields({}, now_ts=2, existing=existing).equals(existing)


# --- speed_bucket classification (step 2, §1.4/§2.2) ----------------------

def test_speed_thresholds_defaults_and_config():
    assert speed_thresholds({}) == (24 * HOUR, 7 * DAY)
    assert speed_thresholds(None) == (24 * HOUR, 7 * DAY)
    cfg = {"scoring": {"speed": {"slow_lifespan_hours": 12, "deep_slow_lifespan_days": 3}}}
    assert speed_thresholds(cfg) == (12 * HOUR, 3 * DAY)


def test_classify_speed_bucket_boundaries():
    slow_s, deep_s = 24 * HOUR, 7 * DAY
    assert classify_speed_bucket(1 * HOUR, slow_s, deep_s) == "fast"
    assert classify_speed_bucket(24 * HOUR - 1, slow_s, deep_s) == "fast"
    assert classify_speed_bucket(24 * HOUR, slow_s, deep_s) == "slow"       # inclusive
    assert classify_speed_bucket(3 * DAY, slow_s, deep_s) == "slow"
    assert classify_speed_bucket(7 * DAY, slow_s, deep_s) == "deep_slow"    # inclusive
    assert classify_speed_bucket(30 * DAY, slow_s, deep_s) == "deep_slow"


def test_classify_speed_bucket_unknown_never_fast():
    # No usable lifespan -> unknown (parked), never 'fast' — absence never proves fast.
    assert classify_speed_bucket(float("nan"), 24 * HOUR, 7 * DAY) == "unknown"
    assert classify_speed_bucket(None, 24 * HOUR, 7 * DAY) == "unknown"


def test_market_bucket_map_and_assign():
    meta = pd.DataFrame({
        "market_id": ["fastm", "slowm", "deepm", "parkm"],
        "lifespan_s": [600.0, 2 * DAY, 10 * DAY, float("nan")],
    })
    bmap = market_bucket_map(meta, 24 * HOUR, 7 * DAY)
    assert bmap == {"fastm": "fast", "slowm": "slow", "deepm": "deep_slow", "parkm": "unknown"}
    # An id absent from the sidecar -> unknown (not in the map).
    out = assign_speed_bucket(["fastm", "deepm", "absent"], bmap)
    assert list(out) == ["fast", "deep_slow", "unknown"]


# --- populate_speed_bucket: byte-identity harness + correct assignment ------

def _legacy_ledger_13col():
    """A pre-Project-3 ledger (no speed_bucket) with same-timestamp rows, to prove
    populate preserves row ORDER exactly (validate.py's split depends on it)."""
    return pd.DataFrame({
        "wallet": ["0xw0", "0xw1", "0xw0", "0xw2"],
        "market_id": ["slowm", "fastm", "deepm", "absent"],
        "token_id": ["t0", "t1", "t2", "t3"],
        "outcome": ["Yes"] * 4,
        "side": ["BUY"] * 4,
        "entry_price": [0.4, 0.5, 0.6, 0.7],
        "size": [1.0, 2.0, 3.0, 4.0],
        "timestamp": [100, 100, 200, 200],  # ties -> order must be preserved
        "resolved": [True, True, True, False],
        "resolved_value": [1.0, 0.0, 1.0, None],
        "question": ["q?"] * 4,
        "slug": ["s"] * 4,
        "tx_hash": ["0xa", "0xb", "0xc", "0xd"],
    })


def _point_ledger_and_sidecar(monkeypatch, tmp_path, ledger_df, meta_df):
    ledger_path = tmp_path / "bet_ledger.parquet"
    meta_path = tmp_path / "market_meta.parquet"
    ledger_df.to_parquet(ledger_path)
    meta_df.to_parquet(meta_path)
    for mod in (common, mm):
        monkeypatch.setattr(mod, "BET_LEDGER_PATH", ledger_path, raising=False)
    monkeypatch.setattr(mm, "MARKET_META_PATH", meta_path, raising=False)
    return ledger_path


def test_populate_speed_bucket_is_byte_identical_on_existing_columns(monkeypatch, tmp_path):
    base = _legacy_ledger_13col()  # 13 columns, no speed_bucket
    meta = pd.DataFrame({
        "market_id": ["slowm", "fastm", "deepm"],
        "lifespan_s": [2 * DAY, 600.0, 10 * DAY],
    })
    _point_ledger_and_sidecar(monkeypatch, tmp_path, base, meta)

    populate_speed_bucket(cfg={})  # code defaults 24h / 7d

    out = common.load_ledger()
    # The additive column is present and correctly classified from the sidecar.
    assert list(out["speed_bucket"]) == ["slow", "fast", "deep_slow", "unknown"]
    # BYTE-IDENTITY: every pre-existing column, same values AND same row order.
    after_13 = out.drop(columns=["speed_bucket"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(after_13, base.reset_index(drop=True))


def test_populate_speed_bucket_is_idempotent(monkeypatch, tmp_path):
    base = _legacy_ledger_13col()
    meta = pd.DataFrame({"market_id": ["slowm", "fastm", "deepm"],
                         "lifespan_s": [2 * DAY, 600.0, 10 * DAY]})
    _point_ledger_and_sidecar(monkeypatch, tmp_path, base, meta)
    populate_speed_bucket(cfg={})
    first = common.load_ledger()
    populate_speed_bucket(cfg={})  # re-run against an already-populated ledger
    second = common.load_ledger()
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_enumerate_gamma_day_pages_until_short_page(monkeypatch):
    import src.market_meta as mm
    monkeypatch.setattr(mm, "GAMMA_PAGE_SIZE", 2)
    monkeypatch.setattr(mm, "SLEEP_BETWEEN_REQUESTS_SEC", 0)
    markets = [
        {"conditionId": f"0x{i}", "startDate": "2026-06-01T00:00:00Z",
         "closedTime": "2026-06-30T00:00:00Z", "volumeNum": float(i)}
        for i in range(5)
    ]
    sess = _FakeGammaSession(markets)
    rows = enumerate_gamma_day(sess, "2026-06-30", "2026-07-01")
    assert [r["market_id"] for r in rows] == [f"0x{i}" for i in range(5)]
    assert sess.offsets == [0, 2, 4]  # stopped on the short final page
