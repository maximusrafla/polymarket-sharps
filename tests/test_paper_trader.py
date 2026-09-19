"""Tests for the forward paper trader (src/paper_trader.py).

Three things are pinned here, in order of how badly they would hurt if wrong:

1. THE FILL LOGIC. Paper trading lies by assuming fills. Every no-fill path is
   tested explicitly — empty book, size only above the limit, a fill under the
   token's minimum order size — as is walking the book into worse prices, the
   maker fill taking the LIMIT price rather than the aggressor's better price, and
   the fee formula's shape.
2. THE SEATBELTS. The kill switch, the position cap and the max-open cap are
   asserted to actually stop things, and the module's source is scanned for any
   signing / key / order-placement code path.
3. THE PRE-REGISTRATION. The rule text names the arms and the event unit, the
   freeze refuses to overwrite itself, an amendment is logged with the number of
   forward observations that existed at the time, and the scoreboard says
   "awaiting data" out loud instead of shipping an empty table that reads as $0.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.paper_trader as pt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def raw_book(asks=(), bids=(), tick="0.01", min_size="5"):
    return {
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "tick_size": tick, "min_order_size": min_size, "timestamp": "1700000000000",
    }


def rule(**over):
    base = {
        "notional_per_signal_usd": 100.0, "max_notional_per_position_usd": 100.0,
        "max_open_positions": 200, "taker_fee_k": 0.02, "taker_fee_k_stress": 0.10,
        "placebo_seed": "seed-v1", "placebo_fallback_horizon_hours": 6.0,
        "enforce_min_order_size": True,
    }
    base.update(over)
    return base


def signal(**over):
    base = {
        "signal_id": "sig1", "wallet": "0xaa", "cohorts": "s2_twelve",
        "market_id": "0xm1", "token_id": "tok1", "outcome": "Yes", "side": "BUY",
        "slug": "nba-bos-nyk-2026-01-02", "question": "Q?",
        "wallet_entry_price": 0.60, "wallet_size": 10.0, "wallet_ts": 1000,
        "detect_ts": 1000, "status": "accepted", "market_end_ts": 1000 + 86400,
        "game_start_ts": None, "placebo_horizon_ts": 1000 + 86400,
        "placebo_horizon_source": "end_date_iso",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# 1. BOOK PARSING
# ---------------------------------------------------------------------------

def test_parse_book_sorts_both_sides_best_first_whatever_the_api_order():
    """The CLOB serves both sides worst-price-first. If we ever trusted the raw
    order, `best_ask` would return the WORST ask and every fill would be wrong."""
    book = pt.parse_book(raw_book(asks=[(0.99, 10), (0.61, 20), (0.70, 5)],
                                  bids=[(0.10, 3), (0.55, 8)]))
    assert [p for p, _ in book["asks"]] == [0.61, 0.70, 0.99]
    assert [p for p, _ in book["bids"]] == [0.55, 0.10]
    assert pt.best_ask(book) == (0.61, 20.0)
    assert pt.best_bid(book) == (0.55, 8.0)


def test_parse_book_drops_zero_size_levels_because_they_are_not_liquidity():
    book = pt.parse_book(raw_book(asks=[(0.61, 0), (0.62, 4)]))
    assert book["asks"] == [(0.62, 4.0)]


def test_parse_book_of_nothing_is_an_empty_book_not_a_crash():
    for empty in (None, {}, raw_book()):
        book = pt.parse_book(empty)
        assert book["asks"] == [] and book["bids"] == []
        assert np.isnan(pt.best_ask(book)[0])


# ---------------------------------------------------------------------------
# 2. FILL LOGIC — the part that makes paper trading lie
# ---------------------------------------------------------------------------

def test_no_fill_when_the_book_has_no_size():
    book = pt.parse_book(raw_book(asks=[]))
    res = pt.fill_from_book(book, 100.0, mode="taker")
    assert res["shares"] == 0 and res["cost_usd"] == 0


def test_no_fill_when_every_level_is_above_the_limit():
    """The core of `limit_noChase`: the wallet paid 0.60, the book is offered at
    0.63, so we do NOT get to buy. A paper trader that fills here is lying."""
    book = pt.parse_book(raw_book(asks=[(0.63, 1000), (0.64, 1000)]))
    res = pt.fill_from_book(book, 100.0, mode="taker", limit_price=0.60)
    assert res["shares"] == 0
    assert res["cost_usd"] == 0


def test_limit_fill_stops_at_the_limit_and_ignores_deeper_size():
    book = pt.parse_book(raw_book(asks=[(0.58, 50), (0.60, 50), (0.61, 100000)]))
    res = pt.fill_from_book(book, 100.0, mode="taker", limit_price=0.60)
    # 50 @ 0.58 = $29, then 50 @ 0.60 = $30 -> 100 shares, $59; 0.61 unreachable
    assert res["shares"] == pytest.approx(100.0)
    assert res["cost_usd"] == pytest.approx(59.0)
    assert res["levels_used"] == 2


def test_walking_the_book_pays_worse_prices_as_size_runs_out():
    """Honest slippage: a thin top level means the average fill price rises."""
    book = pt.parse_book(raw_book(asks=[(0.50, 10), (0.60, 10), (0.70, 1000)]))
    res = pt.fill_from_book(book, 100.0, mode="taker")
    # 10 @ .50 = 5, 10 @ .60 = 6, then $89 of 0.70 = 127.142857 shares
    assert res["levels_used"] == 3
    assert res["cost_usd"] == pytest.approx(100.0)
    assert res["shares"] == pytest.approx(20 + 89 / 0.70)
    assert res["avg_price"] > 0.50          # NOT filled at the top of book


def test_partial_fill_when_the_whole_book_is_thinner_than_the_budget():
    book = pt.parse_book(raw_book(asks=[(0.50, 20), (0.55, 20)], min_size="1"))
    res = pt.fill_from_book(book, 100.0, mode="taker")
    assert res["shares"] == pytest.approx(40.0)
    assert res["cost_usd"] == pytest.approx(20 * 0.50 + 20 * 0.55)
    assert res["cost_usd"] < 100.0          # the unspent budget stays unspent


def test_min_order_size_rejects_a_dribble_of_liquidity():
    """5 shares is the exchange minimum on this token; 2 shares is not a trade."""
    book = pt.parse_book(raw_book(asks=[(0.50, 2)], min_size="5"))
    assert pt.fill_from_book(book, 100.0, mode="taker")["shares"] == 0
    assert pt.fill_from_book(book, 100.0, mode="taker")["rejected_min_size"] is True
    # ...and the check can be disabled to measure what it costs
    assert pt.fill_from_book(book, 100.0, mode="taker",
                             enforce_min_size=False)["shares"] == pytest.approx(2.0)


def test_maker_fill_takes_the_limit_price_not_the_better_price_that_swept_through():
    """THE classic paper-trading lie. Our bid rests at 0.60; the ask collapses to
    0.50. A real passive order was consumed on the way down AT 0.60. Awarding it
    0.50 invents free money out of the modelling."""
    book = pt.parse_book(raw_book(asks=[(0.50, 1000)]))
    res = pt.fill_from_book(book, 100.0, mode="maker", limit_price=0.60)
    assert res["avg_price"] == pytest.approx(0.60)
    assert res["shares"] == pytest.approx(100.0 / 0.60)
    assert res["is_taker"] is False


def test_maker_fill_is_capped_by_the_size_that_actually_traded_through():
    book = pt.parse_book(raw_book(asks=[(0.55, 30)], min_size="1"))
    res = pt.fill_from_book(book, 100.0, mode="maker", limit_price=0.60)
    assert res["shares"] == pytest.approx(30.0)     # not 100/0.60 = 166.7
    assert res["cost_usd"] == pytest.approx(30 * 0.60)


def test_maker_does_not_fill_when_nothing_reaches_the_limit():
    book = pt.parse_book(raw_book(asks=[(0.61, 10000)]))
    assert pt.fill_from_book(book, 100.0, mode="maker", limit_price=0.60)["shares"] == 0


def test_size_at_or_below_only_counts_reachable_levels():
    asks = [(0.58, 5), (0.60, 7), (0.601, 1000)]
    assert pt.size_at_or_below(asks, 0.60) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# 3. COSTS — tick grid and the fee curve
# ---------------------------------------------------------------------------

def test_taker_fee_matches_the_measured_formula_and_vanishes_at_both_ends():
    assert pt.taker_fee(100, 0.5, 0.02) == pytest.approx(100 * 0.02 * 0.25)
    assert pt.taker_fee(100, 0.99, 0.02) == pytest.approx(100 * 0.02 * 0.99 * 0.01)
    assert pt.taker_fee(100, 0.01, 0.02) < pt.taker_fee(100, 0.5, 0.02)
    assert pt.taker_fee(100, 0.5, 0.0) == 0.0


def test_fee_units_let_any_k_be_applied_without_rewalking_the_book():
    book = pt.parse_book(raw_book(asks=[(0.40, 10), (0.60, 10)], min_size="1"))
    res = pt.fill_from_book(book, 100.0, mode="taker")
    expected = 10 * 0.40 * 0.60 + 10 * 0.60 * 0.40
    assert res["fee_units"] == pytest.approx(expected)
    assert 0.02 * res["fee_units"] == pytest.approx(
        pt.taker_fee(10, 0.40, 0.02) + pt.taker_fee(10, 0.60, 0.02))


def test_limit_is_rounded_DOWN_to_the_tick_so_it_never_exceeds_their_price():
    """Rounding up would put our bid above what the wallet paid and quietly turn
    the no-chase arm into a chase."""
    assert pt.round_down_to_tick(0.6349, 0.01) == pytest.approx(0.63)
    assert pt.round_down_to_tick(0.6349, 0.001) == pytest.approx(0.634)
    assert pt.round_down_to_tick(0.63, 0.01) == pytest.approx(0.63)
    assert pt.round_down_to_tick(0.6349, None) == pytest.approx(0.6349)
    for tick in (0.01, 0.001):
        assert pt.round_down_to_tick(0.777, tick) <= 0.777 + 1e-12


def test_tick_size_comes_from_the_token_not_a_constant():
    """docs/project4_value_gate.md measured a 0.001 tick in the favourite band, but
    live game lines report 0.01. Hardcoding either one mis-prices the other."""
    coarse = pt.parse_book(raw_book(asks=[(0.60, 10)], tick="0.01"))
    fine = pt.parse_book(raw_book(asks=[(0.60, 10)], tick="0.001"))
    assert coarse["tick_size"] == 0.01 and fine["tick_size"] == 0.001
    assert pt.round_down_to_tick(0.6349, coarse["tick_size"]) == pytest.approx(0.63)
    assert pt.round_down_to_tick(0.6349, fine["tick_size"]) == pytest.approx(0.634)


# ---------------------------------------------------------------------------
# 4. THE THREE ARMS
# ---------------------------------------------------------------------------

def test_the_three_arms_are_fired_off_one_signal_so_they_are_paired():
    book = pt.parse_book(raw_book(asks=[(0.62, 1000)], min_size="1"))
    orders = pt.open_orders_for_signal(signal(), book, rule(), now_ts=1000)
    assert [o["arm"] for o in orders] == list(pt.ARMS)
    assert len({o["signal_id"] for o in orders}) == 1


def test_limit_arm_rests_unfilled_when_the_market_is_offered_above_their_price():
    book = pt.parse_book(raw_book(asks=[(0.62, 1000)], min_size="1"))
    o = pt.open_orders_for_signal(signal(), book, rule(), 1000)[0]
    assert o["arm"] == "limit_noChase"
    assert o["status"] == pt.ST_OPEN and o["shares"] == 0
    assert o["limit_price"] == pytest.approx(0.60)


def test_limit_arm_takes_immediately_when_already_offered_at_or_below_their_price():
    """Buying at or below what they paid is not a chase, so an immediately
    marketable limit crosses — as a taker, paying the taker fee."""
    book = pt.parse_book(raw_book(asks=[(0.59, 1000)], min_size="1"))
    o = pt.open_orders_for_signal(signal(), book, rule(), 1000)[0]
    assert o["status"] == pt.ST_FILLED
    assert o["is_taker"] is True
    assert o["avg_fill_price"] == pytest.approx(0.59)


def test_limit_arm_fills_on_a_LATER_poll_as_a_maker_at_its_own_price():
    book_now = pt.parse_book(raw_book(asks=[(0.62, 1000)], min_size="1"))
    o = pt.open_orders_for_signal(signal(), book_now, rule(), 1000)[0]
    later = pt.parse_book(raw_book(asks=[(0.55, 40)], min_size="1"))
    o2 = pt.advance_order(o, later, rule(), now_ts=2000)
    # 40 shares * $0.60 = $24 of the $100 budget, so the order is PARTIAL and stays
    # live — the remaining $76 has not been filled and must not be treated as if it had
    assert o2["status"] == pt.ST_PARTIAL and pt.ST_PARTIAL in pt.LIVE_STATUSES
    assert o2["is_taker"] is False                      # maker: no fee
    assert o2["avg_fill_price"] == pytest.approx(0.60)  # our limit, not their 0.55
    assert o2["shares"] == pytest.approx(40.0)
    assert o2["first_fill_ts"] == 2000

    # ...and it closes only once the budget is actually spent
    deep = pt.parse_book(raw_book(asks=[(0.60, 100000)], min_size="1"))
    o3 = pt.advance_order(o2, deep, rule(), now_ts=3000)
    assert o3["status"] == pt.ST_FILLED
    assert o3["cost_usd"] == pytest.approx(100.0)


def test_partial_maker_fills_accumulate_across_polls_at_a_weighted_price():
    o = pt.open_orders_for_signal(
        signal(), pt.parse_book(raw_book(asks=[(0.62, 10)], min_size="1")), rule(), 1000)[0]
    o = pt.advance_order(o, pt.parse_book(raw_book(asks=[(0.60, 30)], min_size="1")),
                         rule(), 2000)
    o = pt.advance_order(o, pt.parse_book(raw_book(asks=[(0.60, 25)], min_size="1")),
                         rule(), 3000)
    assert o["shares"] == pytest.approx(55.0)
    assert o["cost_usd"] == pytest.approx(55 * 0.60)
    assert o["cost_usd"] <= o["budget_usd"] + 1e-9


def test_a_filled_order_never_exceeds_its_budget_however_deep_the_book():
    book = pt.parse_book(raw_book(asks=[(0.10, 10_000_000)], min_size="1"))
    o = pt.open_orders_for_signal(signal(), book, rule(), 1000)[1]
    assert o["arm"] == "market_chase"
    assert o["cost_usd"] == pytest.approx(100.0)


def test_market_chase_is_an_immediate_no_fill_on_an_empty_book():
    o = pt.open_orders_for_signal(signal(), pt.parse_book(raw_book()), rule(), 1000)[1]
    assert o["status"] == pt.ST_NO_FILL
    assert o["no_fill_reason"] == "no_ask_size_at_detection"


def test_placebo_waits_for_its_trigger_and_then_takes_at_the_ask():
    book = pt.parse_book(raw_book(asks=[(0.62, 1000)], min_size="1"))
    o = pt.open_orders_for_signal(signal(), book, rule(), 1000)[2]
    assert o["status"] == pt.ST_PENDING and 1000 <= o["trigger_ts"] <= 1000 + 86400
    before = pt.advance_order(o, book, rule(), now_ts=o["trigger_ts"] - 1)
    assert before["status"] == pt.ST_PENDING and before["shares"] == 0
    after = pt.advance_order(o, book, rule(), now_ts=o["trigger_ts"])
    assert after["status"] == pt.ST_FILLED and after["is_taker"] is True
    assert after["avg_fill_price"] == pytest.approx(0.62)


def test_placebo_trigger_is_deterministic_reproducible_and_inside_the_window():
    a = pt.placebo_trigger_ts("sigX", 1000, 1000 + 86400, "seed-v1")
    b = pt.placebo_trigger_ts("sigX", 1000, 1000 + 86400, "seed-v1")
    c = pt.placebo_trigger_ts("sigX", 1000, 1000 + 86400, "seed-v2")
    assert a == b                      # cannot be redrawn after seeing an outcome
    assert a != c                      # ...but the frozen seed genuinely drives it
    assert 1000 <= a < 1000 + 86400
    draws = [pt.placebo_trigger_ts(f"s{i}", 1000, 1000 + 86400, "seed-v1")
             for i in range(400)]
    assert 0.35 < np.mean([(d - 1000) / 86400 for d in draws]) < 0.65   # ~uniform


def test_placebo_horizon_refuses_a_stale_end_date_and_says_which_source_it_used():
    """src/market_meta.py documents end_date_iso as the midnight FLOOR of the
    scheduled end day, and a live probe of 99 detected signals found 97 already past
    it. Drawing over a window that closed yesterday would fire the control
    instantly; drawing over a 7-day fallback would never fire it at all. Both would
    corrupt the only arm that says whether the wallets add anything."""
    now = 1_000_000
    # a future game start wins outright
    ts, src = pt.placebo_horizon(now, now + 3600, now + 1800, 6.0)
    assert (ts, src) == (now + 1800, "game_start_time")
    # no game time -> a future end date
    ts, src = pt.placebo_horizon(now, now + 3600, None, 6.0)
    assert (ts, src) == (now + 3600, "end_date_iso")
    # an end date already in the past is NOT used
    ts, src = pt.placebo_horizon(now, now - 86400, None, 6.0)
    assert src == "fallback_past_end" and ts == now + 6 * 3600
    # nothing at all
    ts, src = pt.placebo_horizon(now, None, None, 6.0)
    assert src == "fallback_no_end_date" and ts == now + 6 * 3600


def test_placebo_trigger_never_precedes_detection_even_on_a_stale_horizon():
    assert pt.placebo_trigger_ts("sigX", 1000, 500, "seed-v1") == 1000


# ---------------------------------------------------------------------------
# 5. NO-FILLS ARE FIRST-CLASS
# ---------------------------------------------------------------------------

def test_live_orders_in_a_settled_market_become_terminal_no_fills():
    """The single biggest way paper trading lies is quietly forgetting the limits
    that never got hit. They must survive into the ledger as $0 outcomes."""
    orders = pd.DataFrame([
        {"order_id": "a", "arm": "limit_noChase", "market_id": "m1",
         "status": pt.ST_OPEN, "no_fill_reason": None, "last_update_ts": 1},
        {"order_id": "b", "arm": "placebo_random", "market_id": "m1",
         "status": pt.ST_PENDING, "no_fill_reason": None, "last_update_ts": 1},
        {"order_id": "c", "arm": "limit_noChase", "market_id": "m2",
         "status": pt.ST_OPEN, "no_fill_reason": None, "last_update_ts": 1},
    ])
    orders["shares"] = 0.0
    out = pt.terminate_settled(orders, {"m1"}, now_ts=999)
    assert list(out["status"]) == [pt.ST_NO_FILL, pt.ST_NO_FILL, pt.ST_OPEN]
    assert out.loc[0, "no_fill_reason"] == "never_filled_before_settlement"
    assert out.loc[1, "no_fill_reason"] == "settled_before_trigger"


def test_a_partial_order_keeps_its_shares_when_the_market_settles():
    """Settlement closes a partially-filled limit as FILLED (it really did buy
    those shares); only an order that got nothing becomes a no-fill."""
    orders = pd.DataFrame([
        {"order_id": "a", "arm": "limit_noChase", "market_id": "m1",
         "status": pt.ST_PARTIAL, "shares": 12.0, "no_fill_reason": None,
         "last_update_ts": 1},
        {"order_id": "b", "arm": "limit_noChase", "market_id": "m1",
         "status": pt.ST_OPEN, "shares": 0.0, "no_fill_reason": None,
         "last_update_ts": 1},
    ])
    out = pt.terminate_settled(orders, {"m1"}, now_ts=999)
    assert out.loc[0, "status"] == pt.ST_FILLED and out.loc[0, "shares"] == 12.0
    assert out.loc[1, "status"] == pt.ST_NO_FILL
    assert pt.n_live_orders(out) == 0


def test_no_fills_score_zero_dollars_and_still_count_in_the_fill_rate():
    orders = pd.DataFrame([
        {"order_id": "1", "signal_id": "s1", "arm": "limit_noChase", "market_id": "m1",
         "token_id": "t1", "shares": 100.0, "cost_usd": 50.0, "fee_units": 0.0,
         "is_taker": False, "avg_fill_price": 0.50, "slug": None, "question": None},
        {"order_id": "2", "signal_id": "s2", "arm": "limit_noChase", "market_id": "m2",
         "token_id": "t2", "shares": 0.0, "cost_usd": 0.0, "fee_units": 0.0,
         "is_taker": False, "avg_fill_price": np.nan, "slug": None, "question": None},
    ])
    res = pd.DataFrame([
        {"market_id": "m1", "token_id": "t1", "resolved": True, "resolved_value": 1.0},
        {"market_id": "m2", "token_id": "t2", "resolved": True, "resolved_value": 1.0},
    ])
    sig = pd.DataFrame([{"signal_id": "s1", "slug": "a-b-2026-01-01", "question": "q",
                         "market_id": "m1"},
                        {"signal_id": "s2", "slug": "c-d-2026-01-02", "question": "q",
                         "market_id": "m2"}])
    out = pt.score_orders(orders, sig, res, rule(), n_boot=200)
    row = out["board"].iloc[0]
    assert row["n_signals"] == 2 and row["n_filled"] == 1
    assert row["fill_rate"] == pytest.approx(0.5)
    # the winner made $50 on $50; the no-fill made $0 -> $25/signal, not $50
    assert row["total_pnl_usd"] == pytest.approx(50.0)
    assert row["pnl_per_signal_usd"] == pytest.approx(25.0)
    assert row["return_on_filled_notional"] == pytest.approx(1.0)


def test_headline_dollars_per_signal_is_diluted_by_a_low_fill_rate():
    """A strategy that fills 10% of the time and wins on those is not profitable.
    The headline statistic has to say so."""
    rows = [{"order_id": str(i), "signal_id": f"s{i}", "arm": "limit_noChase",
             "market_id": f"m{i}", "token_id": f"t{i}",
             "shares": 100.0 if i == 0 else 0.0, "cost_usd": 50.0 if i == 0 else 0.0,
             "fee_units": 0.0, "is_taker": False, "avg_fill_price": 0.5,
             "slug": None, "question": None} for i in range(10)]
    res = pd.DataFrame([{"market_id": f"m{i}", "token_id": f"t{i}", "resolved": True,
                         "resolved_value": 1.0} for i in range(10)])
    sig = pd.DataFrame([{"signal_id": f"s{i}", "slug": f"x-y-2026-01-{i+1:02d}",
                         "question": "q", "market_id": f"m{i}"} for i in range(10)])
    row = pt.score_orders(pd.DataFrame(rows), sig, res, rule(), n_boot=200)["board"].iloc[0]
    assert row["fill_rate"] == pytest.approx(0.1)
    assert row["return_on_filled_notional"] == pytest.approx(1.0)   # looks brilliant
    assert row["pnl_per_signal_usd"] == pytest.approx(5.0)          # ...and is not


def test_taker_fee_is_charged_to_takers_and_not_to_makers():
    common = dict(signal_id="s1", market_id="m1", token_id="t1", shares=100.0,
                  cost_usd=50.0, fee_units=25.0, avg_fill_price=0.5, slug=None,
                  question=None)
    orders = pd.DataFrame([
        {"order_id": "1", "arm": "limit_noChase", "is_taker": False, **common},
        {"order_id": "2", "arm": "market_chase", "is_taker": True, **common},
    ])
    res = pd.DataFrame([{"market_id": "m1", "token_id": "t1", "resolved": True,
                         "resolved_value": 1.0}])
    sig = pd.DataFrame([{"signal_id": "s1", "slug": "a-b-2026-01-01", "question": "q",
                         "market_id": "m1"}])
    board = pt.score_orders(orders, sig, res, rule(taker_fee_k=0.02),
                            n_boot=200)["board"].set_index("arm")
    assert board.loc["limit_noChase", "total_fee_usd"] == pytest.approx(0.0)
    assert board.loc["market_chase", "total_fee_usd"] == pytest.approx(0.02 * 25.0)
    assert board.loc["market_chase", "total_pnl_usd"] == pytest.approx(50.0 - 0.5)


# ---------------------------------------------------------------------------
# 6. STATISTICS — the event unit
# ---------------------------------------------------------------------------

def test_event_bootstrap_resamples_events_not_signals():
    """Ten signals on one game are one observation. If the bootstrap resampled
    signals it would report a CI ten times too tight — the exact mistake that has
    burned this repo repeatedly."""
    vals = np.array([1.0] * 10 + [-1.0] * 10)
    one_event_each = np.arange(20)
    two_events = np.array([0] * 10 + [1] * 10)
    wide = pt.event_boot(vals, two_events, n_boot=2000)
    narrow = pt.event_boot(vals, one_event_each, n_boot=2000)
    assert wide["n_events"] == 2 and narrow["n_events"] == 20
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_event_bootstrap_is_undefined_below_two_events():
    out = pt.event_boot(np.array([1.0, 2.0]), np.array(["e", "e"]), n_boot=100)
    assert out["n_events"] == 1
    assert np.isnan(out["ci_low"]) and np.isnan(out["p_one_sided"])


def test_effective_events_collapses_when_one_event_dominates():
    assert pt.effective_events(np.arange(10)) == pytest.approx(10.0)
    assert pt.effective_events(np.array([0] * 9 + [1])) < 2.0
    assert pt.effective_events(np.array([])) == 0.0


def test_event_unit_is_the_game_for_sports_and_the_market_elsewhere():
    frame = pd.DataFrame([
        {"slug": "nba-bos-nyk-2026-01-02-spread-home-1pt5", "question": "q",
         "market_id": "m1"},
        {"slug": "nba-bos-nyk-2026-01-02-total-210pt5", "question": "q",
         "market_id": "m2"},
        {"slug": "will-gpt-6-be-released-in-2026", "question": "q", "market_id": "m3"},
    ])
    units = pt.assign_event_units(frame)
    assert units.iloc[0] == units.iloc[1]        # two lines, one game, one draw
    assert units.iloc[2] == "market:m3"


def test_the_board_is_cut_by_source_cohort_so_the_loud_one_cannot_masquerade():
    """Project 1's cohort is micro-crypto HFT and fires orders of magnitude more
    signals than the 12 sports wallets. Pooled, its number IS the pooled number."""
    def mk(sid, cohort, mid, pnl_shares):
        return {"order_id": f"{sid}:a", "signal_id": sid, "arm": "limit_noChase",
                "cohorts": cohort, "market_id": mid, "token_id": f"t{mid}",
                "shares": pnl_shares, "cost_usd": 50.0, "fee_units": 0.0,
                "is_taker": False, "avg_fill_price": 0.5, "slug": None,
                "question": None, "no_fill_reason": None}
    rows = [mk(f"p{i}", "p1_certified", f"m{i}", 0.0) for i in range(8)]        # -$0
    rows += [mk(f"s{i}", "s2_twelve|s2_core", f"n{i}", 200.0) for i in range(3)]  # +$150
    ids = [f"m{i}" for i in range(8)] + [f"n{i}" for i in range(3)]
    res = pd.DataFrame([{"market_id": m, "token_id": f"t{m}", "resolved": True,
                         "resolved_value": 1.0} for m in ids])
    sig = pd.DataFrame([{"signal_id": s, "slug": f"x-y-2026-03-{i+1:02d}",
                         "question": "q", "market_id": m}
                        for i, (s, m) in enumerate(
                            [(f"p{i}", f"m{i}") for i in range(8)]
                            + [(f"s{i}", f"n{i}") for i in range(3)])])
    board = pt.score_orders(pd.DataFrame(rows), sig, res, rule(),
                            n_boot=200)["board"].set_index(["stratum", "arm"])
    assert set(board.index.get_level_values("stratum")) == {
        "all", "p1_certified", "s2_twelve"}
    assert board.loc[("s2_twelve", "limit_noChase"), "n_signals"] == 3
    assert board.loc[("p1_certified", "limit_noChase"), "fill_rate"] == 0.0
    # the sports cohort is genuinely profitable; pooled it is diluted to near nothing
    assert board.loc[("s2_twelve", "limit_noChase"), "pnl_per_signal_usd"] > 100
    assert board.loc[("all", "limit_noChase"), "pnl_per_signal_usd"] < 50


def test_the_deciding_contrast_is_thesis_minus_placebo_on_shared_events():
    def mk(arm, sid, mid, shares, cost):
        return {"order_id": f"{sid}:{arm}", "signal_id": sid, "arm": arm,
                "market_id": mid, "token_id": f"t{mid}", "shares": shares,
                "cost_usd": cost, "fee_units": 0.0, "is_taker": False,
                "avg_fill_price": 0.5, "slug": None, "question": None}
    rows = []
    for i in range(6):
        rows.append(mk("limit_noChase", f"s{i}", f"m{i}", 100.0, 50.0))   # +$50
        rows.append(mk("placebo_random", f"s{i}", f"m{i}", 100.0, 80.0))  # +$20
    res = pd.DataFrame([{"market_id": f"m{i}", "token_id": f"tm{i}", "resolved": True,
                         "resolved_value": 1.0} for i in range(6)])
    sig = pd.DataFrame([{"signal_id": f"s{i}", "slug": f"x-y-2026-02-{i+1:02d}",
                         "question": "q", "market_id": f"m{i}"} for i in range(6)])
    out = pt.score_orders(pd.DataFrame(rows), sig, res, rule(), n_boot=500)
    c = out["contrast"]
    assert c["comparison"] == "limit_noChase - placebo_random"
    assert c["shared_events"] == 6
    assert c["mean_usd_per_signal"] == pytest.approx(30.0)
    assert c["p_one_sided"] < 0.05


# ---------------------------------------------------------------------------
# 7. SEATBELTS
# ---------------------------------------------------------------------------

def test_kill_switch_file_halts_the_run_before_any_fetch_or_write(monkeypatch, tmp_path):
    kill = tmp_path / "KILL"
    kill.write_text("stop")
    cfg = {"paper": {"kill_switch_path": str(kill)}}
    assert pt.kill_switch_engaged(cfg) is True

    def explode(*a, **k):                     # must never be reached
        raise AssertionError("run() fetched with the kill switch engaged")
    monkeypatch.setattr(pt, "load_freeze", lambda: (pd.DataFrame([{"wallet": "0xa"}]),
                                                    _fake_manifest()))
    monkeypatch.setattr(pt, "make_session", explode)
    monkeypatch.setattr(pt, "save_ledger", explode)
    out = pt.run(cfg, passes=1, dry_run=False)
    assert out == {"halted": True, "reason": "kill_switch"}

    kill.unlink()
    assert pt.kill_switch_engaged(cfg) is False


def test_kill_switch_defaults_to_a_path_inside_data():
    assert pt.kill_switch_path({}).name == "KILL_PAPER_TRADER"
    assert "data" in str(pt.kill_switch_path({}))


def test_the_committed_artifacts_record_a_repo_relative_kill_switch():
    """The manifest and scoreboard are read on other machines, so an absolute path
    from whatever box happened to freeze them is a broken instruction."""
    assert pt.kill_switch_label({}) == "data/KILL_PAPER_TRADER"
    assert pt.kill_switch_label({"paper": {"kill_switch_path": "/mnt/ram/STOP"}}) \
        == "/mnt/ram/STOP"


def test_per_position_notional_is_capped_below_the_configured_stake():
    assert pt.capped_budget(100.0, 25.0) == 25.0
    o = pt.open_orders_for_signal(
        signal(), pt.parse_book(raw_book(asks=[(0.10, 100000)], min_size="1")),
        rule(max_notional_per_position_usd=25.0), 1000)[1]
    assert o["cost_usd"] == pytest.approx(25.0)


def test_config_can_tighten_but_never_loosen_the_frozen_caps():
    manifest = _fake_manifest()
    manifest["seatbelts"]["max_open_positions"] = 200
    manifest["seatbelts"]["max_notional_per_position_usd"] = 100.0
    tight = pt.rule_from_manifest(manifest, {"paper": {"max_open_positions": 5,
                                                       "max_notional_per_position_usd": 10.0}})
    assert tight["max_open_positions"] == 5 and tight["max_notional_per_position_usd"] == 10.0
    loose = pt.rule_from_manifest(manifest, {"paper": {"max_open_positions": 99999,
                                                       "max_notional_per_position_usd": 1e9}})
    assert loose["max_open_positions"] == 200
    assert loose["max_notional_per_position_usd"] == 100.0


def test_max_open_positions_cap_records_the_signal_and_opens_nothing():
    """The cap must COST the strategy: a capped-out signal still counts in
    n_signals, so it cannot be used to quietly skip the hard ones."""
    cfg = {"paper": {"max_open_positions": 1}}
    watchlist = pd.DataFrame([{"wallet": "0xaa", "cohorts": "s2_twelve"}])
    trades = [
        {"wallet": "0xaa", "market_id": "m1", "token_id": "t1", "outcome": "Yes",
         "side": "BUY", "entry_price": 0.6, "size": 5, "timestamp": 2000,
         "question": "q1", "slug": "nba-a-b-2026-01-01", "tx_hash": "0x1"},
        {"wallet": "0xaa", "market_id": "m2", "token_id": "t2", "outcome": "Yes",
         "side": "BUY", "entry_price": 0.6, "size": 5, "timestamp": 2001,
         "question": "q2", "slug": "nba-c-d-2026-01-01", "tx_hash": "0x2"},
    ]
    sig, orders, state, stats = pt.poll_once(
        None, cfg, rule(), watchlist, pt._empty(pt.SIGNAL_COLUMNS),
        pt._empty(pt.ORDER_COLUMNS), {"cursors": {"0xaa": {"max_timestamp": 0,
                                                           "keys_at_max": []}}},
        now_ts=3000,
        fetch_positions=lambda *a, **k: trades,
        get_book=lambda *a, **k: raw_book(asks=[(0.62, 1000)], min_size="1"),
        get_market=lambda *a, **k: {"market_end_ts": 90000, "end_ts_source": "end_date_iso"})
    assert stats["new_signals"] == 2
    assert stats["skipped_cap"] == 1
    assert set(sig["status"]) == {"accepted", "skipped_cap"}
    assert set(orders["signal_id"]) == {sig.loc[sig["status"] == "accepted",
                                                "signal_id"].iloc[0]}


def test_dry_run_is_the_default_and_writes_nothing(monkeypatch):
    monkeypatch.setattr(pt, "load_freeze",
                        lambda: (pd.DataFrame(columns=["wallet", "cohorts"]),
                                 _fake_manifest()))
    monkeypatch.setattr(pt, "load_ledger",
                        lambda: (pt._empty(pt.SIGNAL_COLUMNS), pt._empty(pt.ORDER_COLUMNS)))
    monkeypatch.setattr(pt, "load_state", lambda: {"cursors": {}})
    monkeypatch.setattr(pt, "make_session", lambda **k: None)

    def boom(*a, **k):
        raise AssertionError("dry run persisted the ledger")
    monkeypatch.setattr(pt, "save_ledger", boom)
    out = pt.run({"paper": {"kill_switch_path": "/nonexistent/KILL"},
                  "ingest": {"max_retries": 1, "retry_backoff_sec": 0.1}},
                 passes=1)                    # dry_run defaults to True
    assert out["halted"] is False


def test_module_has_no_execution_code_path():
    """The hard guarantee: this file cannot place an order or touch a key, and a
    future edit that tries has to delete this test to do it."""
    src = (__import__("pathlib").Path(pt.__file__)).read_text()
    code = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    forbidden = [
        "private_key", "PRIVATE_KEY", "privateKey", "secret_key", "mnemonic",
        "eth_account", "web3", "py_clob_client", "ClobClient", "Account.from_key",
        "sign_message", "signTypedData", "sign_typed_data", "eth_sign",
        "session.post", "session.put", "session.delete", "requests.post",
        "requests.put", "requests.delete", ".post(", ".put(", ".delete(",
        "/order", "post_order", "create_order", "place_order", "cancel_order",
        "api_key", "api_secret", "passphrase", "L1_AUTH", "POLY_ADDRESS",
    ]
    hits = [tok for tok in forbidden if tok in code]
    assert not hits, f"execution-layer code path found in paper_trader.py: {hits}"
    # ...and every network call it does make is a GET on a public host
    import re
    verbs = set(re.findall(r"session\.(\w+)\(", code)) | set(re.findall(r"resp\w*\s*=\s*\w+\.(\w+)\(", code))
    assert verbs <= {"get"}, f"non-GET HTTP verb in paper_trader.py: {verbs}"


def test_the_docstring_states_the_guarantee_plainly():
    doc = pt.__doc__
    for phrase in ("PAPER ONLY", "READ-ONLY", "NO PRIVATE KEYS", "NO SIGNING",
                   "NO ORDERS"):
        assert phrase in doc


# ---------------------------------------------------------------------------
# 8. PRE-REGISTRATION
# ---------------------------------------------------------------------------

def _fake_manifest(ts: int = 1000) -> dict:
    return {
        "freeze_ts": ts, "freeze_utc": "2026-07-26T00:00:00Z", "git_commit": "deadbeef",
        "arms": list(pt.ARMS), "headline_arm": pt.HEADLINE_ARM,
        "control_arm": pt.CONTROL_ARM, "entry_rule": pt.ENTRY_RULE,
        "scoring_rule": pt.SCORING_RULE,
        "stake": {"notional_per_signal_usd": 100.0},
        "costs": {"taker_fee_k": 0.0, "taker_fee_k_stress": 0.10},
        "seatbelts": {"max_notional_per_position_usd": 100.0, "max_open_positions": 200,
                      "kill_switch_path": "/tmp/KILL"},
        "placebo": {"seed": "seed-v1", "fallback_horizon_hours": 6.0},
        "known_limits": ["limit"],
    }


def test_entry_rule_pins_the_three_arms_and_the_no_chase_condition():
    r = pt.ENTRY_RULE
    for arm in pt.ARMS:
        assert arm in r
    assert "rounded DOWN to the token's tick grid" in r
    assert "never above the price they paid" in r
    assert "MAKER at the limit price" in r
    assert "NO-FILL, which is an outcome, not a discard" in r
    assert "100.00 USD" in r


def test_scoring_rule_pins_the_event_unit_the_headline_and_the_control():
    r = pt.SCORING_RULE
    assert "DOLLARS PER DETECTED SIGNAL" in r
    assert "resamples EVENTS with replacement, never signals" in r
    assert "resolution EVENT" in r and "market_id elsewhere" in r
    assert "limit_noChase minus placebo_random" in r
    assert "fill_rate" in r and "effective_events" in r
    assert "machinery check" in r
    assert "STRATA, fixed here in advance" in r and "No further slicing is permitted" in r


def test_freeze_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(pt, "build_watchlist",
                        lambda cfg=None: (pd.DataFrame([{"wallet": "0xa",
                                                         "cohorts": "p1_certified",
                                                         "in_p1_certified": True,
                                                         "in_s2_twelve": False}]),
                                          {"watchlist_size": 1}))
    pt.freeze({}, freeze_ts=1000)
    with pytest.raises(SystemExit):
        pt.freeze({}, freeze_ts=2000)


def test_forced_refreeze_is_logged_with_the_observation_count(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(pt, "build_watchlist",
                        lambda cfg=None: (pd.DataFrame([{"wallet": "0xa",
                                                         "cohorts": "p1_certified",
                                                         "in_p1_certified": True,
                                                         "in_s2_twelve": False}]),
                                          {"watchlist_size": 1}))
    pt.freeze({}, freeze_ts=1000)
    monkeypatch.setattr(pt, "forward_observations", lambda: 37)
    m = pt.freeze({}, freeze_ts=1000, force=True, amend_reason="testing")
    a = m["amendments"][-1]
    assert a["forward_observations_at_amendment"] == 37
    assert a["legitimate_pre_registration"] is False
    assert a["reason"] == "testing"
    assert a["prior_entry_rule"] == pt.ENTRY_RULE


def test_freeze_records_the_funnel_and_the_seatbelts(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(pt, "build_watchlist",
                        lambda cfg=None: (pd.DataFrame([{"wallet": "0xa",
                                                         "cohorts": "p1_certified",
                                                         "in_p1_certified": True,
                                                         "in_s2_twelve": False}]),
                                          {"watchlist_size": 1, "p1_certified": 26,
                                           "p1_oos_significance_alpha": 0.01}))
    m = pt.freeze({}, freeze_ts=1000)
    assert m["forward_observations_at_freeze"] == 0
    assert m["funnel"]["p1_certified"] == 26
    assert m["seatbelts"]["order_placement"].startswith("none")
    assert m["seatbelts"]["keys"] == "none"
    assert m["seatbelts"]["dry_run_default"] is True
    assert m["placebo"]["seed"].endswith(":1000")
    with open(pt.FREEZE_MANIFEST_PATH) as f:
        assert json.load(f)["entry_rule"] == pt.ENTRY_RULE


def test_scoreboard_says_awaiting_data_out_loud_instead_of_shipping_a_zero(
        monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    pt.write_scoreboard({"board": pd.DataFrame(), "contrast": None, "n_unresolved": 4,
                         "note": "signals captured, none resolved yet"},
                        _fake_manifest(), pd.DataFrame([{"wallet": "0xa"}]),
                        pt._empty(pt.SIGNAL_COLUMNS), pt._empty(pt.ORDER_COLUMNS))
    text = pt.SCOREBOARD_MD.read_text()
    assert "AWAITING DATA" in text
    assert "not** a zero" in text
    assert "no keys, no orders" in text.lower() or "no keys, no orders" in text
    assert pt.ENTRY_RULE.split("\n")[0][:40] in text


def test_scoreboard_puts_the_fill_rate_next_to_every_dollar_figure(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    board = pd.DataFrame([{
        "arm": a, "n_signals": 10, "n_filled": 1, "fill_rate": 0.1, "n_events": 4,
        "effective_events": 3.2, "notional_filled_usd": 50.0, "total_pnl_usd": 5.0,
        "total_pnl_usd_fee_stress": 4.0, "return_on_filled_notional": 0.1,
        "pnl_per_signal_usd": 0.5, "pnl_per_signal_usd_event_wtd": 0.4,
        "ci_low": -0.2, "ci_high": 1.1, "p_one_sided": 0.2, "mean_fill_price": 0.5,
        "total_fee_usd": 0.0, "stratum": "all"} for a in pt.ARMS])
    pt.write_scoreboard({"board": board, "contrast": None, "n_unresolved": 0,
                         "note": None}, _fake_manifest(),
                        pd.DataFrame([{"wallet": "0xa"}]),
                        pt._empty(pt.SIGNAL_COLUMNS), pt._empty(pt.ORDER_COLUMNS))
    text = pt.SCOREBOARD_MD.read_text()
    assert "fill rate" in text and "10.0%" in text
    assert "$ / signal" in text
    assert "Never quote" in text and "return on filled" in text
    assert "Kill switch" in text


def _redirect(monkeypatch, tmp_path):
    """Point every artifact path at a tmpdir so tests never touch data/."""
    monkeypatch.setattr(pt, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pt, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(pt, "FREEZE_MANIFEST_PATH", tmp_path / "paper_freeze_manifest.json")
    monkeypatch.setattr(pt, "WATCHLIST_PATH", tmp_path / "paper_watchlist.parquet")
    monkeypatch.setattr(pt, "SCOREBOARD_MD", tmp_path / "paper_scoreboard.md")
    monkeypatch.setattr(pt, "SCOREBOARD_PARQUET", tmp_path / "paper_scoreboard.parquet")
    monkeypatch.setattr(pt, "SIGNALS_PATH", tmp_path / "paper" / "paper_signals.parquet")
    monkeypatch.setattr(pt, "ORDERS_PATH", tmp_path / "paper" / "paper_orders.parquet")
    monkeypatch.setattr(pt, "STATE_PATH", tmp_path / "paper" / "paper_state.json")


# ---------------------------------------------------------------------------
# 9. RESUMABILITY
# ---------------------------------------------------------------------------

def test_a_signal_is_never_traded_twice_across_polls():
    """Idempotence: re-serving the same trade (the cursor overlap deliberately
    does this) must not open a second set of orders."""
    cfg = {"paper": {}}
    watchlist = pd.DataFrame([{"wallet": "0xaa", "cohorts": "s2_twelve"}])
    trade = {"wallet": "0xaa", "market_id": "m1", "token_id": "t1", "outcome": "Yes",
             "side": "BUY", "entry_price": 0.6, "size": 5, "timestamp": 2000,
             "question": "q", "slug": "nba-a-b-2026-01-01", "tx_hash": "0x1"}
    kw = dict(fetch_positions=lambda *a, **k: [trade],
              get_book=lambda *a, **k: raw_book(asks=[(0.62, 1000)], min_size="1"),
              get_market=lambda *a, **k: {"market_end_ts": 90000})
    state = {"cursors": {"0xaa": {"max_timestamp": 0, "keys_at_max": []}}}
    sig, orders, state, _ = pt.poll_once(None, cfg, rule(), watchlist,
                                         pt._empty(pt.SIGNAL_COLUMNS),
                                         pt._empty(pt.ORDER_COLUMNS), state, 3000, **kw)
    assert len(sig) == 1 and len(orders) == 3
    sig2, orders2, state, stats = pt.poll_once(None, cfg, rule(), watchlist, sig,
                                               orders, state, 3060, **kw)
    assert len(sig2) == 1 and len(orders2) == 3
    assert stats["new_signals"] == 0


def test_sell_trades_are_not_signals():
    cfg = {"paper": {}}
    watchlist = pd.DataFrame([{"wallet": "0xaa", "cohorts": "x"}])
    sell = {"wallet": "0xaa", "market_id": "m1", "token_id": "t1", "outcome": "Yes",
            "side": "SELL", "entry_price": 0.6, "size": 5, "timestamp": 2000,
            "question": "q", "slug": "s", "tx_hash": "0x1"}
    sig, orders, _, stats = pt.poll_once(
        None, cfg, rule(), watchlist, pt._empty(pt.SIGNAL_COLUMNS),
        pt._empty(pt.ORDER_COLUMNS),
        {"cursors": {"0xaa": {"max_timestamp": 0, "keys_at_max": []}}}, 3000,
        fetch_positions=lambda *a, **k: [sell],
        get_book=lambda *a, **k: raw_book(asks=[(0.5, 10)]),
        get_market=lambda *a, **k: {})
    assert len(sig) == 0 and len(orders) == 0 and stats["new_signals"] == 0


def test_a_failing_book_fetch_records_the_signal_and_does_not_crash_the_cycle():
    cfg = {"paper": {}}
    watchlist = pd.DataFrame([{"wallet": "0xaa", "cohorts": "x"}])
    trade = {"wallet": "0xaa", "market_id": "m1", "token_id": "t1", "outcome": "Yes",
             "side": "BUY", "entry_price": 0.6, "size": 5, "timestamp": 2000,
             "question": "q", "slug": "s", "tx_hash": "0x1"}

    def boom(*a, **k):
        raise RuntimeError("502 from the CLOB")
    sig, orders, _, stats = pt.poll_once(
        None, cfg, rule(), watchlist, pt._empty(pt.SIGNAL_COLUMNS),
        pt._empty(pt.ORDER_COLUMNS),
        {"cursors": {"0xaa": {"max_timestamp": 0, "keys_at_max": []}}}, 3000,
        fetch_positions=lambda *a, **k: [trade], get_book=boom,
        get_market=lambda *a, **k: {})
    assert len(sig) == 1 and sig.iloc[0]["status"] == "skipped_book_error"
    assert len(orders) == 0 and stats["errors"] == 1
