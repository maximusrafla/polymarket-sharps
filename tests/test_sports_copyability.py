"""Tests for scripts/audit_sports_copyability.py — the metric-B / copyability
audit of the three slow-cadence sports wallets.

Hand-built fixtures where the answer is known by construction (CLAUDE.md test
style). No network, no data files.

The load-bearing invariants:
  - `guard_cutoff` drops exactly the final `guard` fraction of a token's OBSERVED
    lifespan, integer-floored — the same cutoff `audit_edge_decay_long`'s
    `FollowerPricer` uses, so the two audits guard identically.
  - `next_other_print` returns the FIRST *other*-wallet print strictly after the
    anchor and at or before the cutoff, and returns None (never a
    `resolved_value` fallback, never an own-wallet print) when there is none.
  - `event_weighted_mean` weights RESOLUTION EVENTS equally, so one championship
    field bought across 54 markets counts once — the freeze manifest's own
    scoring rule.
  - `design_effect` reports > 1 when bets inside a cluster are positively
    correlated (the cluster bootstrap is correctly charging for shared
    resolution) and is undefined below two clusters.
  - `TokenTape.price_at` is the production `features.forward_price`, so the
    follower price inherits the leakage discipline rather than re-implementing it.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.audit_sports_copyability import (
    TokenTape,
    design_effect,
    effective_spread,
    event_weighted_mean,
    guard_cutoff,
    next_other_print,
)
from src.features import forward_price


# ---------------------------------------------------------------------------
# guard_cutoff
# ---------------------------------------------------------------------------
def test_guard_cutoff_drops_the_final_fraction_of_observed_life():
    ts = np.array([100, 200, 600], dtype=float)   # span 500
    assert guard_cutoff(ts, 0.2) == 500.0         # 600 - 0.2*500
    assert guard_cutoff(ts, 0.0) == 600.0         # no guard -> last trade
    assert guard_cutoff(ts, 1.0) == 100.0         # full guard -> first trade


def test_guard_cutoff_floors_to_integer_seconds():
    ts = np.array([0, 7], dtype=float)            # 7 - 0.2*7 = 5.6
    assert guard_cutoff(ts, 0.2) == 5.0


def test_guard_cutoff_is_unaffected_by_trade_order():
    ts = np.array([600, 100, 200], dtype=float)
    assert guard_cutoff(ts, 0.2) == guard_cutoff(np.sort(ts), 0.2)


# ---------------------------------------------------------------------------
# next_other_print
# ---------------------------------------------------------------------------
@pytest.fixture
def tape_arrays():
    """One token: own wallet 'W' prints at t=0 and t=50; others at 10, 30, 90."""
    ts = np.array([0, 10, 30, 50, 90], dtype=float)
    wal = np.array(["W", "A", "B", "W", "C"])
    price = np.array([0.50, 0.55, 0.60, 0.62, 0.70])
    return ts, wal, price


def test_next_other_print_takes_the_first_other_wallet_print(tape_arrays):
    ts, wal, price = tape_arrays
    got = next_other_print(ts, wal, price, anchor_ts=0.0, exclude_wallet="W",
                           cutoff_ts=np.inf)
    assert got == (0.55, 10.0)          # t=10, not the own print at t=0 or 50


def test_next_other_print_skips_own_wallet_prints(tape_arrays):
    ts, wal, price = tape_arrays
    got = next_other_print(ts, wal, price, anchor_ts=30.0, exclude_wallet="W",
                           cutoff_ts=np.inf)
    assert got == (0.70, 60.0)          # t=50 is own; the next OTHER print is t=90


def test_next_other_print_respects_the_resolution_guard(tape_arrays):
    ts, wal, price = tape_arrays
    # cutoff 60 hides the t=90 print entirely -> nothing left after t=30
    assert next_other_print(ts, wal, price, 30.0, "W", cutoff_ts=60.0) is None
    # ...and the guard is inclusive at the boundary
    assert next_other_print(ts, wal, price, 30.0, "W", cutoff_ts=90.0) == (0.70, 60.0)


def test_next_other_print_is_strict_after_the_anchor(tape_arrays):
    ts, wal, price = tape_arrays
    # a print exactly AT the anchor does not count (matches forward_price's `>`)
    assert next_other_print(ts, wal, price, 10.0, "W", np.inf) == (0.60, 20.0)


def test_next_other_print_does_not_need_sorted_input(tape_arrays):
    ts, wal, price = tape_arrays
    o = np.array([4, 2, 0, 3, 1])
    assert (next_other_print(ts[o], wal[o], price[o], 0.0, "W", np.inf)
            == (0.55, 10.0))


def test_next_other_print_returns_none_when_only_own_prints_remain():
    ts = np.array([0, 10], dtype=float)
    wal = np.array(["W", "W"])
    price = np.array([0.4, 0.9])
    assert next_other_print(ts, wal, price, 0.0, "W", np.inf) is None


# ---------------------------------------------------------------------------
# event_weighted_mean
# ---------------------------------------------------------------------------
def test_event_weighted_mean_gives_one_event_one_vote():
    # event A: 4 bets all +1.0 ; event B: 1 bet at -1.0
    values = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
    events = np.array(["A", "A", "A", "A", "B"])
    assert values.mean() == pytest.approx(0.6)          # bet-weighted
    assert event_weighted_mean(values, events) == pytest.approx(0.0)


def test_event_weighted_mean_equals_the_plain_mean_when_events_are_singletons():
    values = np.array([0.1, -0.2, 0.3])
    events = np.array(["a", "b", "c"])
    assert event_weighted_mean(values, events) == pytest.approx(values.mean())


def test_event_weighted_mean_drops_nans_and_is_nan_when_nothing_is_left():
    values = np.array([np.nan, 0.5, np.nan])
    events = np.array(["a", "b", "c"])
    assert event_weighted_mean(values, events) == pytest.approx(0.5)
    assert np.isnan(event_weighted_mean(np.array([np.nan]), np.array(["a"])))


# ---------------------------------------------------------------------------
# design_effect
# ---------------------------------------------------------------------------
def test_design_effect_exceeds_one_when_bets_share_a_resolution():
    rng = np.random.default_rng(0)
    # 20 events x 20 bets; every bet in an event carries the SAME value, so a
    # bet resample sees 400 draws where only 20 are independent.
    per_event = rng.normal(size=20)
    values = np.repeat(per_event, 20)
    events = np.repeat(np.arange(20), 20)
    d = design_effect(values, events, np.random.default_rng(1), n=400)
    assert d > 5.0


def test_design_effect_is_about_one_when_clusters_carry_no_information():
    rng = np.random.default_rng(2)
    values = rng.normal(size=600)
    events = np.tile(np.arange(200), 3)     # cluster labels unrelated to values
    d = design_effect(values, events, np.random.default_rng(3), n=600)
    assert 0.6 < d < 1.6


def test_design_effect_is_undefined_below_two_clusters():
    rng = np.random.default_rng(4)
    values = np.array([0.1, 0.2, 0.3])
    assert np.isnan(design_effect(values, np.array(["a", "a", "a"]), rng, n=50))


# ---------------------------------------------------------------------------
# effective_spread
# ---------------------------------------------------------------------------
def test_effective_spread_reads_top_of_book_and_depth_within_two_cents():
    bids = [(0.40, 100.0), (0.39, 50.0)]
    asks = [(0.42, 10.0), (0.42, 5.0), (0.43, 20.0), (0.46, 999.0)]
    got = effective_spread(bids, asks)
    assert got["best_bid"] == pytest.approx(0.40)
    assert got["best_ask"] == pytest.approx(0.42)
    assert got["spread"] == pytest.approx(0.02)
    assert got["ask_size"] == pytest.approx(15.0)      # both levels AT 0.42
    assert got["ask_size_2c"] == pytest.approx(35.0)   # 0.42 + 0.43, not 0.46


def test_effective_spread_handles_a_one_sided_book():
    got = effective_spread([], [(0.9, 3.0)])
    assert np.isnan(got["best_bid"]) and np.isnan(got["spread"])
    assert got["best_ask"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TokenTape delegates to the production leakage-guarded core
# ---------------------------------------------------------------------------
@pytest.fixture
def tape_frame():
    return pd.DataFrame({
        "token_id": ["t"] * 5,
        "wallet": ["W", "A", "B", "W", "C"],
        "timestamp": [0, 10, 30, 50, 90],
        "entry_price": [0.50, 0.55, 0.60, 0.62, 0.70],
        "size": [1.0, 1.0, 3.0, 1.0, 1.0],
    })


def test_price_at_matches_features_forward_price_exactly(tape_frame):
    tape = TokenTape(tape_frame, guard=0.0)      # cutoff = 90, the last print
    got = tape.price_at("t", "W", entry_ts=0, delta=0, bandwidth=40)
    ref = forward_price(tape_frame, entry_ts=0, exclude_wallet="W",
                        window_seconds=40, cutoff_ts=90.0)
    # size-weighted over the OTHER prints at t=10 (0.55 x1) and t=30 (0.60 x3)
    assert got == pytest.approx(ref)
    assert got == pytest.approx((0.55 + 3 * 0.60) / 4)


def test_price_at_is_nan_rather_than_falling_back_to_the_outcome(tape_frame):
    tape = TokenTape(tape_frame, guard=0.0)
    # window (50, 60] contains no print at all
    assert np.isnan(tape.price_at("t", "W", entry_ts=50, delta=0, bandwidth=10))
    # unknown token
    assert np.isnan(tape.price_at("nope", "W", entry_ts=0, delta=0, bandwidth=10))


def test_covered_is_the_any_later_other_print_ceiling(tape_frame):
    tape = TokenTape(tape_frame, guard=0.0)
    assert tape.covered("t", "W", 0) is True
    assert tape.covered("t", "W", 95) is False     # nothing after the last print
    # C is the only wallet printing after t=50, so for C itself there is nothing
    assert tape.covered("t", "C", 50) is False


def test_guard_removes_the_late_print_from_both_anchors(tape_frame):
    # span 0..90, guard 0.2 -> cutoff 72, so the t=90 print is unusable
    tape = TokenTape(tape_frame, guard=0.2)
    assert np.isnan(tape.price_at("t", "W", entry_ts=72, delta=0, bandwidth=100))
    assert tape.next_print("t", "W", 72)[0] is not tape.next_print("t", "W", 0)[0]
    assert np.isnan(tape.next_print("t", "W", 72)[0])
    # ...but lifting the guard exposes it again (the coverage UPPER bound)
    assert tape.next_print("t", "W", 72, guarded=False)[0] == pytest.approx(0.70)
