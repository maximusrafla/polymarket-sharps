"""Tests for src/discover.py — pure logic only, no network calls.

Covers the classifier, min-volume/micro selection, the per-market pagination
cursor (overlap-safe), dedup, market-row parsing, universe upsert, and the
`/trades?market=` fetch loop against a FakeSession (mirrors test_backfill)."""

import pandas as pd
import requests

from src.discover import (
    DISCOVERY_LEDGER_COLUMNS,
    advance_cursor,
    classify_market,
    dedup_trades,
    fetch_market_trades,
    fold_discovery_trades,
    is_micro_crypto,
    is_real_world,
    merge_universe,
    parse_market_row,
    refresh_discovery_resolutions,
    select_candidate_markets,
    select_new_trades,
    _resolved_market_ids,
)


# --- classification -------------------------------------------------------

def test_micro_crypto_detected_by_slug_not_lifespan():
    # The 5-minute up/down coin-flips — even though Gamma metadata reports ~24h.
    assert is_micro_crypto("btc-updown-5m-1766162100", "Bitcoin up or down?")
    assert is_micro_crypto("eth-updown-5m-1766161800", None)
    assert is_micro_crypto(None, "Will ETH be up or down in the next 5 minutes?")
    assert not is_micro_crypto("will-argentina-win-the-2026-fifa-world-cup", "Argentina champions?")


def test_classify_market_real_world_subcategories():
    assert classify_market("will-donald-trump-win-the-2024-us-presidential-election",
                           "Will Donald Trump win?") == "politics"
    assert classify_market("will-argentina-win-the-2026-fifa-world-cup",
                           "Argentina wins World Cup?") == "sports_soccer"
    assert classify_market("nfl-chiefs-win-super-bowl", "Chiefs win Super Bowl?") == "sports_nfl"
    assert classify_market("fed-decreases-interest-rates-by-50-bps-after-jan",
                           "Fed cuts rates 50 bps?") == "econ_macro"
    assert classify_market("microstrategy-sells-any-bitcoin-by-may-31-2026",
                           "MicroStrategy sells BTC?") == "crypto_event"
    assert classify_market("new-playboi-carti-album-before-gta-vi",
                           "New album before GTA VI?") == "culture"


def test_classify_market_micro_beats_everything_and_unknown_is_other():
    # A crypto keyword present, but the updown slug shape must win -> micro_crypto.
    assert classify_market("btc-updown-5m-123", "Bitcoin up or down?") == "micro_crypto"
    # Nothing matches -> conservative 'other', never a forced guess.
    assert classify_market("will-it-rain-in-nowhere-tomorrow", "Random?") == "other"


def test_classify_market_sports_type_only_rescues_unknown():
    # sportsMarketType is a weak positive: it only rescues an otherwise-unknown
    # market, and never overrides a real keyword match.
    assert classify_market("some-obscure-match-2026", None, sports_market_type="totals") == "sports_other"
    assert classify_market("fed-cuts-rates", "Fed?", sports_market_type="totals") == "econ_macro"


def test_is_real_world():
    assert is_real_world("politics")
    assert is_real_world("sports_nfl")
    assert not is_real_world("micro_crypto")


# --- market row parsing ---------------------------------------------------

def test_parse_market_row_extracts_and_classifies():
    row = parse_market_row({
        "conditionId": "0xabc",
        "slug": "will-argentina-win-the-2026-fifa-world-cup",
        "question": "Argentina wins?",
        "volumeNum": 174315688.0,
        "closed": True,
        "startDate": "2025-01-01T00:00:00Z",
        "endDate": "2026-07-01T00:00:00Z",
        "sportsMarketType": None,
    })
    assert row["market_id"] == "0xabc"
    assert row["category"] == "sports_soccer"
    assert row["volume"] == 174315688.0
    assert row["closed"] is True


def test_parse_market_row_none_without_condition_id():
    assert parse_market_row({"slug": "x", "volumeNum": 10}) is None


def test_parse_market_row_bad_volume_defaults_zero():
    row = parse_market_row({"conditionId": "0x1", "slug": "s", "volumeNum": None})
    assert row["volume"] == 0.0


# --- candidate selection --------------------------------------------------

def _universe():
    return pd.DataFrame([
        {"market_id": "0xbig", "category": "politics", "volume": 1e9},
        {"market_id": "0xmid", "category": "sports_nfl", "volume": 50000.0},
        {"market_id": "0xsmall", "category": "politics", "volume": 100.0},   # below floor
        {"market_id": "0xmicro", "category": "micro_crypto", "volume": 2e6}, # skipped
    ])


def test_select_candidate_markets_filters_and_orders():
    got = select_candidate_markets(_universe(), min_volume=5000.0, skip_micro=True)
    assert got == ["0xbig", "0xmid"]  # small dropped (floor), micro dropped, volume-desc


def test_select_candidate_markets_can_keep_micro():
    got = select_candidate_markets(_universe(), min_volume=5000.0, skip_micro=False)
    assert got == ["0xbig", "0xmicro", "0xmid"]  # micro kept, volume-desc


def test_select_candidate_markets_empty():
    assert select_candidate_markets(pd.DataFrame(columns=["market_id", "category", "volume"])) == []


# --- pagination cursor (overlap-safe) -------------------------------------

def _t(ts, tx="0xtx", wallet="0xw", asset="a", side="BUY"):
    return {"timestamp": ts, "transactionHash": tx, "proxyWallet": wallet,
            "asset": asset, "side": side}


def test_select_new_trades_stops_below_floor():
    page = [_t(10, "x3"), _t(9, "x2"), _t(8, "x1")]
    picked, stop = select_new_trades(page, floor_ts=9, seen_at_max=set())
    # ts 10 kept; ts 9 == floor and not seen -> kept; ts 8 < floor -> stop
    assert [t["timestamp"] for t in picked] == [10, 9]
    assert stop is True


def test_select_new_trades_skips_seen_boundary_key():
    boundary = _t(9, "xdup")
    page = [_t(10, "x3"), boundary]
    seen = {f"{boundary['transactionHash']}:{boundary['proxyWallet']}:{boundary['asset']}:{boundary['side']}"}
    picked, stop = select_new_trades(page, floor_ts=9, seen_at_max=seen)
    # the already-recorded boundary trade is skipped; only the newer one kept
    assert [t["timestamp"] for t in picked] == [10]
    assert stop is False


def test_advance_cursor_records_keys_at_max():
    new = [_t(10, "a"), _t(10, "b"), _t(7, "c")]
    cur = advance_cursor(new, prev={})
    assert cur["max_timestamp"] == 10
    assert set(cur["keys_at_max"]) == {"a:0xw:a:BUY", "b:0xw:a:BUY"}


def test_advance_cursor_unchanged_when_no_new():
    prev = {"max_timestamp": 5, "keys_at_max": ["x"]}
    assert advance_cursor([], prev) == prev


def test_advance_cursor_unions_boundary_keys_when_max_unchanged():
    prev = {"max_timestamp": 10, "keys_at_max": ["a:0xw:a:BUY"]}
    new = [_t(10, "b")]
    cur = advance_cursor(new, prev)
    assert cur["max_timestamp"] == 10
    assert set(cur["keys_at_max"]) == {"a:0xw:a:BUY", "b:0xw:a:BUY"}


# --- dedup ----------------------------------------------------------------

def test_dedup_trades_keeps_last_on_fill_identity():
    df = pd.DataFrame([
        {"tx_hash": "0x1", "wallet": "w", "token_id": "t", "side": "BUY", "size": 10},
        {"tx_hash": "0x1", "wallet": "w", "token_id": "t", "side": "BUY", "size": 10},  # dup
        {"tx_hash": "0x1", "wallet": "w", "token_id": "t", "side": "SELL", "size": 3},  # distinct side
    ])
    out = dedup_trades(df)
    assert len(out) == 2


# --- fold -----------------------------------------------------------------

def _raw_trade(cid, slug, question, tx="0xtx", wallet="0xw"):
    return {
        "proxyWallet": wallet, "conditionId": cid, "asset": "tok", "outcome": "Yes",
        "side": "BUY", "price": 0.4, "size": 10.0, "timestamp": 1000,
        "title": question, "slug": slug, "transactionHash": tx,
    }


def test_fold_discovery_trades_attaches_category_from_map():
    trades = [_raw_trade("0xm1", "will-x-win-election", "Will X win?")]
    df = fold_discovery_trades(trades, {"0xm1": "politics"})
    assert df.iloc[0]["category"] == "politics"
    assert df.iloc[0]["resolved"] == False  # noqa: E712 - resolution is a downstream step
    assert list(df.columns)[-1] == "category"


def test_fold_discovery_trades_falls_back_to_row_classifier():
    # market missing from the map -> classify from the row's own slug/question
    trades = [_raw_trade("0xUnknown", "will-argentina-win-the-2026-fifa-world-cup", "Argentina?")]
    df = fold_discovery_trades(trades, {})
    assert df.iloc[0]["category"] == "sports_soccer"


def test_fold_discovery_trades_empty():
    assert fold_discovery_trades([], {}).empty


# --- universe upsert ------------------------------------------------------

def test_merge_universe_upsert_preserves_first_seen():
    existing = merge_universe(
        pd.DataFrame(),
        [{"market_id": "0xa", "slug": "s", "question": "q", "category": "politics",
          "volume": 100.0, "closed": False, "start_date": None, "end_date": None}],
        now_ts=1000,
    )
    assert existing.iloc[0]["first_seen"] == 1000
    # re-seen later with updated volume/closed: first_seen stays, last_seen advances
    updated = merge_universe(
        existing,
        [{"market_id": "0xa", "slug": "s", "question": "q", "category": "politics",
          "volume": 500.0, "closed": True, "start_date": None, "end_date": None}],
        now_ts=2000,
    )
    assert len(updated) == 1
    row = updated.iloc[0]
    assert row["first_seen"] == 1000
    assert row["last_seen"] == 2000
    assert row["volume"] == 500.0
    assert bool(row["closed"]) is True


# --- fetch loop against a FakeSession (no real network) -------------------

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


class FakeMarketSession:
    """Serves a market's newest-first tape by offset/limit; 400s past `cap`
    (the ~10k offset ceiling)."""

    def __init__(self, trades_desc, cap=10_000):
        self.all = trades_desc
        self.cap = cap
        self.offsets = []

    def get(self, url, params=None, timeout=None):
        off, lim = params["offset"], params["limit"]
        self.offsets.append(off)
        if off > self.cap:
            return FakeResp({"error": "max historical trades offset of 10000 exceeded"}, status=400)
        return FakeResp(self.all[off:off + lim], status=200)


def _tape(tss):
    return [_t(ts, tx=f"tx{ts}") for ts in tss]


def test_fetch_market_trades_pages_until_short_page():
    sess = FakeMarketSession(_tape([5, 4, 3]))
    import src.discover as d
    old = d.TRADES_PAGE_SIZE
    d.TRADES_PAGE_SIZE = 2
    try:
        trades, cursor, hit_cap = fetch_market_trades(sess, "0xm", {"max_timestamp": 0, "keys_at_max": []})
    finally:
        d.TRADES_PAGE_SIZE = old
    assert [t["timestamp"] for t in trades] == [5, 4, 3]
    assert cursor["max_timestamp"] == 5
    assert sess.offsets == [0, 2]  # stopped on the short final page
    assert hit_cap is False        # reached the tape's end -> not truncated


def test_fetch_market_trades_stops_at_cursor_excludes_seen_boundary():
    sess = FakeMarketSession(_tape([5, 4, 3, 2, 1]))
    import src.discover as d
    old = d.TRADES_PAGE_SIZE
    d.TRADES_PAGE_SIZE = 2
    try:
        # cursor knows the ts=3 boundary trade (tx3) -> it must be skipped, and
        # everything below it stopped. Overlap-safe: no re-fetch, no gap.
        trades, _, _ = fetch_market_trades(
            sess, "0xm", {"max_timestamp": 3, "keys_at_max": ["tx3:0xw:a:BUY"]})
    finally:
        d.TRADES_PAGE_SIZE = old
    assert [t["timestamp"] for t in trades] == [5, 4]


def test_fetch_market_trades_keeps_unseen_boundary():
    # Same floor, but the boundary key is unknown (empty keys_at_max): the ts=3
    # trade is kept rather than risk dropping it.
    sess = FakeMarketSession(_tape([5, 4, 3, 2, 1]))
    import src.discover as d
    old = d.TRADES_PAGE_SIZE
    d.TRADES_PAGE_SIZE = 2
    try:
        trades, _, _ = fetch_market_trades(sess, "0xm", {"max_timestamp": 3, "keys_at_max": []})
    finally:
        d.TRADES_PAGE_SIZE = old
    assert [t["timestamp"] for t in trades] == [5, 4, 3]


def test_fetch_market_trades_stops_on_offset_cap_400():
    sess = FakeMarketSession(_tape(list(range(20, 0, -1))), cap=4)
    import src.discover as d
    old_ps, old_mp = d.TRADES_PAGE_SIZE, d.MAX_PAGES_PER_MARKET
    d.TRADES_PAGE_SIZE, d.MAX_PAGES_PER_MARKET = 2, 100
    try:
        trades, _, hit_cap = fetch_market_trades(sess, "0xm", {"max_timestamp": 0, "keys_at_max": []})
    finally:
        d.TRADES_PAGE_SIZE, d.MAX_PAGES_PER_MARKET = old_ps, old_mp
    # offset 0,2,4 serve; offset 6 (>cap 4) 400s and stops cleanly
    assert [t["timestamp"] for t in trades] == [20, 19, 18, 17, 16, 15]
    assert sess.offsets == [0, 2, 4, 6]
    assert hit_cap is True  # 400 at the offset cap => tape truncated


# --- resolution enrichment (pure; shared cache stays read-only) -----------

def _disc_row(market_id, token_id, resolved=False, resolved_value=None, category="politics"):
    return {
        "wallet": "0xw", "market_id": market_id, "token_id": token_id, "outcome": "Yes",
        "side": "BUY", "entry_price": 0.4, "size": 10.0, "timestamp": 1000,
        "resolved": resolved, "resolved_value": resolved_value, "question": "q?",
        "slug": "s", "tx_hash": "0xtx", "category": category,
    }


def test_resolved_market_ids_reads_closed_flag():
    res = pd.DataFrame([
        {"market_id": "0xa", "token_id": "t", "outcome": "Yes", "resolved": True,
         "resolved_value": 1.0, "closed": True},
        {"market_id": "0xb", "token_id": "t", "outcome": "Yes", "resolved": False,
         "resolved_value": None, "closed": False},
    ])
    assert _resolved_market_ids(res) == {"0xa"}
    assert _resolved_market_ids(pd.DataFrame()) == set()


def test_refresh_discovery_resolutions_fills_and_preserves_category():
    trades = pd.DataFrame(
        [_disc_row("0xm", "tokYes", category="sports_nfl"),
         _disc_row("0xm", "tokNo", category="sports_nfl")],
        columns=DISCOVERY_LEDGER_COLUMNS,
    )
    resolutions = pd.DataFrame([
        {"market_id": "0xm", "token_id": "tokYes", "resolved": True, "resolved_value": 1.0},
        {"market_id": "0xm", "token_id": "tokNo", "resolved": True, "resolved_value": 0.0},
    ])
    out = refresh_discovery_resolutions(trades, resolutions)
    # resolution filled in per token...
    assert list(out["resolved"]) == [True, True]
    assert sorted(out["resolved_value"].tolist()) == [0.0, 1.0]
    # ...and the discovery-only `category` column survived (LEDGER_COLUMNS would drop it)
    assert list(out["category"]) == ["sports_nfl", "sports_nfl"]
    assert list(out.columns) == DISCOVERY_LEDGER_COLUMNS


def test_refresh_discovery_resolutions_no_op_when_empty():
    trades = pd.DataFrame([_disc_row("0xm", "t")], columns=DISCOVERY_LEDGER_COLUMNS)
    # empty resolutions -> unchanged
    assert refresh_discovery_resolutions(trades, pd.DataFrame()).equals(trades)


# --- §1.4 prong 2: events enumeration (the mid-volume full-tape band) ------

from src.discover import (  # noqa: E402
    enumerate_events_markets,
    in_volume_band,
    market_tape_complete,
    parse_event_markets,
    should_reenumerate,
    update_tape_stats,
)


def _mkt(cid, slug, vol, question=None, closed=True):
    return {"conditionId": cid, "slug": slug, "question": question,
            "volumeNum": vol, "closed": closed}


def test_parse_event_markets_backfills_question_and_tag_rescues_other():
    # A terse player-prop slug the keyword classifier can't place, under tag=nfl:
    # the event title backfills the question and the tag rescues it to sports_nfl.
    event = {
        "title": "NFL Week 1: Will Justin Jefferson play?",
        "markets": [
            _mkt("0xa", "will-jj-play", 200_000, question=None),          # -> tag rescue
            _mkt("0xb", "argentina-wc", 300_000,                           # keyword wins -> soccer
                 question="Will Argentina win the World Cup?"),
            {"slug": "no-cid"},                                            # dropped (no conditionId)
        ],
    }
    rows = parse_event_markets(event, tag_hint="nfl")
    by_id = {r["market_id"]: r for r in rows}
    assert set(by_id) == {"0xa", "0xb"}
    assert by_id["0xa"]["category"] == "sports_nfl"       # rescued from 'other'
    assert by_id["0xa"]["question"] == event["title"]     # title backfilled
    assert by_id["0xb"]["category"] == "sports_soccer"    # confident keyword NOT overridden


def test_in_volume_band():
    assert in_volume_band({"volume": 100_000.0}, 50_000, 4_000_000)
    assert not in_volume_band({"volume": 10_000.0}, 50_000, 4_000_000)   # below floor
    assert not in_volume_band({"volume": 9e6}, 50_000, 4_000_000)        # above ceiling (cap-truncated)
    assert not in_volume_band({"volume": None}, 50_000, 4_000_000)       # missing -> 0


class FakeEventsSession:
    """Serves /events pages (offset/limit) from a fixed event list, per tag."""

    def __init__(self, events_by_tag):
        self.events_by_tag = events_by_tag
        self.calls = []

    def get(self, url, params=None, timeout=None):
        tag = params["tag_slug"]
        off, lim = params["offset"], params["limit"]
        self.calls.append((tag, off))
        evs = self.events_by_tag.get(tag, [])
        return FakeResp(evs[off:off + lim], status=200)


def test_enumerate_events_markets_filters_band_and_dedups():
    events = {
        "nfl": [
            {"title": "NFL A", "markets": [
                _mkt("0xa", "nfl-a", 200_000),        # in band
                _mkt("0xmega", "nfl-mega", 9e6),      # above ceiling -> dropped
            ]},
            {"title": "NFL B", "markets": [
                _mkt("0xa", "nfl-a", 200_000),        # dup market_id -> dropped
                _mkt("0xtiny", "nfl-tiny", 1_000),    # below floor -> dropped
                _mkt("0xb", "nfl-b", 3_000_000),      # in band
            ]},
        ],
    }
    sess = FakeEventsSession(events)
    import src.discover as d
    old = d.GAMMA_PAGE_SIZE
    d.GAMMA_PAGE_SIZE = 100
    try:
        rows = enumerate_events_markets(sess, ["nfl"], closed=True, max_markets=50,
                                        vol_min=50_000, vol_max=4_000_000)
    finally:
        d.GAMMA_PAGE_SIZE = old
    ids = [r["market_id"] for r in rows]
    assert ids == ["0xa", "0xb"]  # band-only, deduped, order preserved


def test_enumerate_events_markets_respects_max_markets():
    events = {"nba": [{"title": "e", "markets": [_mkt(f"0x{i}", f"nba-{i}", 100_000)
                                                  for i in range(20)]}]}
    sess = FakeEventsSession(events)
    rows = enumerate_events_markets(sess, ["nba"], closed=True, max_markets=5,
                                    vol_min=50_000, vol_max=4_000_000)
    assert len(rows) == 5


def test_should_reenumerate():
    assert should_reenumerate(0, 12)       # always on the first tick
    assert should_reenumerate(12, 12)
    assert not should_reenumerate(5, 12)
    assert should_reenumerate(7, 0)        # every<=0 -> every tick


# --- tape-completeness signal (segment biased vs full tapes) ---------------

def test_update_tape_stats_sticky_or():
    existing = pd.DataFrame(
        [{"market_id": "0xa", "hit_cap": True, "last_pulled": 1},
         {"market_id": "0xb", "hit_cap": False, "last_pulled": 1}])
    # 0xa now reports NOT capped (incremental top-up) -> must STAY capped (sticky);
    # 0xb now hits the cap -> flips to capped; 0xc is new.
    updates = [{"market_id": "0xa", "hit_cap": False},
               {"market_id": "0xb", "hit_cap": True},
               {"market_id": "0xc", "hit_cap": False}]
    out = update_tape_stats(existing, updates, now_ts=99).set_index("market_id")
    assert bool(out.loc["0xa", "hit_cap"]) is True   # sticky
    assert bool(out.loc["0xb", "hit_cap"]) is True   # newly capped
    assert bool(out.loc["0xc", "hit_cap"]) is False
    assert int(out.loc["0xc", "last_pulled"]) == 99


def test_update_tape_stats_empty_updates_noop():
    existing = pd.DataFrame([{"market_id": "0xa", "hit_cap": True, "last_pulled": 1}])
    assert update_tape_stats(existing, [], now_ts=5) is existing


def test_market_tape_complete_prefers_stats_then_heuristic():
    import src.discover as d
    # 0xrec has a stats row (hit_cap True) -> incomplete regardless of length.
    # 0xleg has NO stats row -> heuristic: >= TAPE_CAP_ROWS => truncated.
    tape = pd.DataFrame({
        "market_id": (["0xrec"] * 3) + (["0xleg"] * (d.TAPE_CAP_ROWS + 5))
                     + (["0xfull"] * 10),
    })
    stats = pd.DataFrame([{"market_id": "0xrec", "hit_cap": True, "last_pulled": 1}])
    out = market_tape_complete(tape, stats).set_index("market_id")
    assert bool(out.loc["0xrec", "tape_complete"]) is False   # recorded hit_cap wins
    assert bool(out.loc["0xleg", "tape_complete"]) is False   # heuristic: long tape truncated
    assert bool(out.loc["0xfull", "tape_complete"]) is True    # short tape, no stats -> complete


def test_fetch_market_trades_hit_cap_on_page_budget_exhaustion():
    # More history than the page budget can page: no 400, but the loop exhausts
    # MAX_PAGES_PER_MARKET without a natural stop -> hit_cap via the for/else path.
    sess = FakeMarketSession(_tape(list(range(100, 0, -1))), cap=10_000)
    import src.discover as d
    old_ps, old_mp = d.TRADES_PAGE_SIZE, d.MAX_PAGES_PER_MARKET
    d.TRADES_PAGE_SIZE, d.MAX_PAGES_PER_MARKET = 2, 3  # 3*2=6 rows, tape has 100
    try:
        trades, _, hit_cap = fetch_market_trades(sess, "0xm", {"max_timestamp": 0, "keys_at_max": []})
    finally:
        d.TRADES_PAGE_SIZE, d.MAX_PAGES_PER_MARKET = old_ps, old_mp
    assert len(trades) == 6       # only the page budget's worth
    assert hit_cap is True        # truncated (more history than we could page)
