import pandas as pd
import pytest

from src.watch import (
    advance_cursor,
    build_alert,
    compute_remaining_window,
    default_cursor,
    estimate_fair_value,
    format_alert,
    new_trades_since,
    normalize_trade,
    poll_once,
    position_key,
    window_is_open,
)


def raw_trade(
    wallet="0xA",
    market_id="m1",
    token_id="t1",
    side="BUY",
    price=0.5,
    size=10,
    timestamp=1000,
    tx_hash="tx1",
    outcome="Yes",
    title="Will X happen?",
    slug="will-x-happen",
):
    return {
        "proxyWallet": wallet,
        "conditionId": market_id,
        "asset": token_id,
        "outcome": outcome,
        "side": side,
        "price": price,
        "size": size,
        "timestamp": timestamp,
        "title": title,
        "slug": slug,
        "transactionHash": tx_hash,
    }


DEFAULT_CFG = {"watch": {"min_remaining_window": 0.02}}


def wallet_row(wallet="0xA", copy_window=0.1, out_of_sample_edge=0.15, out_of_sample_n=40):
    return pd.Series(
        {
            "wallet": wallet,
            "copy_window": copy_window,
            "out_of_sample_edge": out_of_sample_edge,
            "out_of_sample_n": out_of_sample_n,
        }
    )


# --- normalize_trade ---------------------------------------------------

def test_normalize_trade_renames_fields():
    n = normalize_trade(raw_trade(wallet="0xA", market_id="m1", token_id="t1", price=0.42))
    assert n["wallet"] == "0xA"
    assert n["market_id"] == "m1"
    assert n["token_id"] == "t1"
    assert n["entry_price"] == pytest.approx(0.42)
    assert n["tx_hash"] == "tx1"


# --- position_key / cursor logic ---------------------------------------

def test_position_key_distinguishes_side_and_tx():
    t1 = normalize_trade(raw_trade(side="BUY", tx_hash="tx1"))
    t2 = normalize_trade(raw_trade(side="SELL", tx_hash="tx1"))
    assert position_key(t1) != position_key(t2)


def test_new_trades_since_excludes_trades_before_floor():
    cursor = {"max_timestamp": 1000, "keys_at_max": []}
    trades = [normalize_trade(raw_trade(timestamp=500)), normalize_trade(raw_trade(timestamp=1500))]
    fresh = new_trades_since(trades, cursor)
    assert [t["timestamp"] for t in fresh] == [1500]


def test_new_trades_since_excludes_already_seen_trade_at_floor_timestamp():
    seen = normalize_trade(raw_trade(timestamp=1000, tx_hash="seen"))
    unseen_same_ts = normalize_trade(raw_trade(timestamp=1000, tx_hash="unseen"))
    cursor = {"max_timestamp": 1000, "keys_at_max": [position_key(seen)]}
    fresh = new_trades_since([seen, unseen_same_ts], cursor)
    assert [t["tx_hash"] for t in fresh] == ["unseen"]


def test_advance_cursor_tracks_max_timestamp_and_ties():
    trades = [
        normalize_trade(raw_trade(timestamp=1000, tx_hash="a")),
        normalize_trade(raw_trade(timestamp=1500, tx_hash="b")),
        normalize_trade(raw_trade(timestamp=1500, tx_hash="c")),
    ]
    cursor = advance_cursor({"max_timestamp": 0, "keys_at_max": []}, trades)
    assert cursor["max_timestamp"] == 1500
    assert len(cursor["keys_at_max"]) == 2


def test_advance_cursor_no_trades_keeps_existing_cursor():
    existing = {"max_timestamp": 999, "keys_at_max": ["x"]}
    assert advance_cursor(existing, []) == existing


def test_advance_cursor_never_moves_backward():
    # A wallet's first poll can fetch a page that's entirely older than its
    # default_cursor("now") floor (most recent trade predates joining the
    # watchlist) — the cursor must stay put, not jump back into the past.
    cursor = {"max_timestamp": 1000, "keys_at_max": []}
    old_trades = [normalize_trade(raw_trade(timestamp=500, tx_hash="old"))]
    assert advance_cursor(cursor, old_trades) == cursor


def test_default_cursor_starts_at_now_not_zero():
    cursor = default_cursor(123456)
    assert cursor == {"max_timestamp": 123456, "keys_at_max": []}


# --- copy-window math ----------------------------------------------------

def test_estimate_fair_value_adds_historical_copy_window():
    assert estimate_fair_value(0.4, 0.1) == pytest.approx(0.5)


def test_window_open_when_remaining_exceeds_threshold():
    window = compute_remaining_window(entry_price=0.4, current_price=0.42, historical_copy_window=0.1)
    # fair_value=0.5, remaining=0.08 -> 8 cents
    assert window["remaining_cents"] == pytest.approx(8.0)
    assert window_is_open(window["remaining_cents"], min_remaining_window=0.02)


def test_window_closed_when_price_already_near_fair_value():
    window = compute_remaining_window(entry_price=0.4, current_price=0.495, historical_copy_window=0.1)
    # fair_value=0.5, remaining=0.005 -> 0.5 cents, below the 2-cent default threshold
    assert not window_is_open(window["remaining_cents"], min_remaining_window=0.02)


def test_window_closed_when_price_has_overshot_fair_value():
    window = compute_remaining_window(entry_price=0.4, current_price=0.55, historical_copy_window=0.1)
    assert window["remaining_cents"] < 0
    assert not window_is_open(window["remaining_cents"], min_remaining_window=0.02)


# --- build_alert / format_alert ------------------------------------------

def test_build_alert_carries_wallet_validated_edge():
    trade = normalize_trade(raw_trade())
    window = compute_remaining_window(0.5, 0.52, 0.1)
    alert = build_alert(wallet_row(out_of_sample_edge=0.22, out_of_sample_n=77), trade, 0.52, window)
    assert alert["wallet_out_of_sample_edge"] == pytest.approx(0.22)
    assert alert["wallet_out_of_sample_n"] == 77
    assert alert["entry_price"] == pytest.approx(0.5)
    assert alert["current_price"] == pytest.approx(0.52)


def test_format_alert_is_human_readable_string():
    trade = normalize_trade(raw_trade())
    window = compute_remaining_window(0.5, 0.52, 0.1)
    alert = build_alert(wallet_row(), trade, 0.52, window)
    text = format_alert(alert)
    assert "0xA" in text
    assert "ALERT" in text


# --- poll_once (network calls injected) -----------------------------------

def make_watchlist(rows):
    return pd.DataFrame(rows)


def test_poll_once_alerts_on_new_position_with_open_window():
    watchlist = make_watchlist([wallet_row(wallet="0xA", copy_window=0.1).to_dict()])
    fake_trades = {"0xA": [normalize_trade(raw_trade(wallet="0xA", timestamp=2000, price=0.4, tx_hash="tx1"))]}

    def fetch_trades(session, cfg, wallet, limit):
        return fake_trades[wallet]

    def fetch_price(session, cfg, token_id):
        return 0.42  # fair_value 0.5, remaining 8c -> open

    alerts, state = poll_once(
        None, DEFAULT_CFG, watchlist, state={}, now_ts=1000, fetch_trades=fetch_trades, fetch_price=fetch_price
    )
    assert len(alerts) == 1
    assert alerts[0]["wallet"] == "0xA"
    assert state["0xA"]["max_timestamp"] == 2000


def test_poll_once_suppresses_alert_when_window_closed():
    watchlist = make_watchlist([wallet_row(wallet="0xA", copy_window=0.1).to_dict()])
    fake_trades = {"0xA": [normalize_trade(raw_trade(wallet="0xA", timestamp=2000, price=0.4, tx_hash="tx1"))]}

    def fetch_trades(session, cfg, wallet, limit):
        return fake_trades[wallet]

    def fetch_price(session, cfg, token_id):
        return 0.499  # fair_value 0.5, remaining 0.1c -> closed (below default-style 2c threshold)

    alerts, state = poll_once(
        None, DEFAULT_CFG, watchlist, state={}, now_ts=1000, fetch_trades=fetch_trades, fetch_price=fetch_price
    )
    assert alerts == []
    # cursor still advances even though nothing alerted, so it isn't re-fetched forever
    assert state["0xA"]["max_timestamp"] == 2000


def test_poll_once_ignores_sell_trades():
    watchlist = make_watchlist([wallet_row(wallet="0xA", copy_window=0.1).to_dict()])
    fake_trades = {
        "0xA": [normalize_trade(raw_trade(wallet="0xA", side="SELL", timestamp=2000, price=0.4, tx_hash="tx1"))]
    }

    def fetch_trades(session, cfg, wallet, limit):
        return fake_trades[wallet]

    def fetch_price(session, cfg, token_id):
        raise AssertionError("price should never be fetched for a SELL trade")

    alerts, state = poll_once(
        None, DEFAULT_CFG, watchlist, state={}, now_ts=1000, fetch_trades=fetch_trades, fetch_price=fetch_price
    )
    assert alerts == []
    assert state["0xA"]["max_timestamp"] == 2000


def test_poll_once_never_alerts_twice_for_the_same_position():
    watchlist = make_watchlist([wallet_row(wallet="0xA", copy_window=0.1).to_dict()])
    trade = normalize_trade(raw_trade(wallet="0xA", timestamp=2000, price=0.4, tx_hash="tx1"))

    def fetch_trades(session, cfg, wallet, limit):
        return [trade]  # same page returned every poll, as the real API would

    def fetch_price(session, cfg, token_id):
        return 0.42  # window stays open throughout

    state = {}
    alerts1, state = poll_once(
        None, DEFAULT_CFG, watchlist, state=state, now_ts=1000, fetch_trades=fetch_trades, fetch_price=fetch_price
    )
    alerts2, state = poll_once(
        None, DEFAULT_CFG, watchlist, state=state, now_ts=1000, fetch_trades=fetch_trades, fetch_price=fetch_price
    )
    assert len(alerts1) == 1
    assert alerts2 == []


def test_poll_once_first_poll_ignores_history_before_watch_started():
    watchlist = make_watchlist([wallet_row(wallet="0xA", copy_window=0.1).to_dict()])
    old_trade = normalize_trade(raw_trade(wallet="0xA", timestamp=500, price=0.4, tx_hash="old"))

    def fetch_trades(session, cfg, wallet, limit):
        return [old_trade]

    def fetch_price(session, cfg, token_id):
        return 0.42

    alerts, state = poll_once(
        None, DEFAULT_CFG, watchlist, state={}, now_ts=1000, fetch_trades=fetch_trades, fetch_price=fetch_price
    )
    assert alerts == []  # trade predates the wallet joining the watchlist
    assert state["0xA"]["max_timestamp"] == 1000  # cursor stays at the join-time floor, doesn't rewind to 500
