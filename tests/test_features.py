import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_breadth,
    compute_edge,
    compute_forward_drift,
    compute_manufactured_record_flag,
    compute_market_windows,
    compute_pattern_flag,
    compute_sample_size,
    compute_skill_edge,
    compute_time_consistency,
    compute_wallet_features,
    expected_outcome,
    fit_price_baseline,
    forward_price,
    only_buys,
    residual_edge_per_bet,
)

LEDGER_COLS = [
    "wallet", "market_id", "token_id", "outcome", "side", "entry_price", "size",
    "timestamp", "resolved", "resolved_value", "question", "slug", "tx_hash",
]


def bet(wallet, market_id="m1", token_id="t1", side="BUY", entry_price=0.5, size=10,
        timestamp=0, resolved=False, resolved_value=None, tx_hash="tx", question="q", slug="s",
        outcome="Yes"):
    return {
        "wallet": wallet, "market_id": market_id, "token_id": token_id, "outcome": outcome,
        "side": side, "entry_price": entry_price, "size": size, "timestamp": timestamp,
        "resolved": resolved, "resolved_value": resolved_value, "question": question,
        "slug": slug, "tx_hash": tx_hash,
    }


def make_ledger(rows):
    return pd.DataFrame(rows, columns=LEDGER_COLS)


# --- edge -------------------------------------------------------------

def test_compute_edge_known_values():
    rows = [
        bet("A", entry_price=0.4, resolved=True, resolved_value=1.0),  # edge +0.6
        bet("A", entry_price=0.6, resolved=True, resolved_value=0.0),  # edge -0.6
        bet("A", entry_price=0.9, resolved=False),  # excluded: unresolved
        bet("B", entry_price=0.2, resolved=True, resolved_value=1.0),  # edge +0.8
    ]
    edge = compute_edge(make_ledger(rows))
    assert edge["A"] == pytest.approx(0.0)
    assert edge["B"] == pytest.approx(0.8)


def test_compute_edge_excludes_sell_side_via_only_buys():
    rows = [
        bet("A", side="BUY", entry_price=0.3, resolved=True, resolved_value=1.0),
        bet("A", side="SELL", entry_price=0.9, resolved=True, resolved_value=1.0),
    ]
    ledger = make_ledger(rows)
    bets = only_buys(ledger)
    assert len(bets) == 1
    edge = compute_edge(bets)
    assert edge["A"] == pytest.approx(0.7)


# --- favorite-longshot baseline / skill (residual) edge -----------------


def _calibration_pop():
    """Balanced market population: longshots at 0.3 win only 0.2 of the time
    (overpriced), favorites at 0.7 win 0.8 of the time (underpriced) — the
    favorite-longshot structure the residual edge is meant to neutralize."""
    rows = []
    for i in range(50):
        rows.append(bet(f"lo{i}", entry_price=0.3, resolved=True, resolved_value=1.0 if i < 10 else 0.0))
    for i in range(50):
        rows.append(bet(f"hi{i}", entry_price=0.7, resolved=True, resolved_value=1.0 if i < 40 else 0.0))
    return rows


def test_fit_price_baseline_recovers_calibration_curve():
    baseline = fit_price_baseline(make_ledger(_calibration_pop()), n_bins=2)
    assert expected_outcome(baseline, [0.3])[0] == pytest.approx(0.2)
    assert expected_outcome(baseline, [0.7])[0] == pytest.approx(0.8)


def test_price_baseline_falls_back_to_global_mean_when_too_few_bets():
    ledger = make_ledger([bet("A", entry_price=0.5, resolved=True, resolved_value=1.0)])
    baseline = fit_price_baseline(ledger, n_bins=20)
    # not enough data to bin -> every price maps to the global base rate (1.0)
    assert expected_outcome(baseline, [0.1, 0.9])[0] == pytest.approx(1.0)


def test_residual_edge_neutralizes_favorite_longshot_but_keeps_real_skill():
    baseline = fit_price_baseline(make_ledger(_calibration_pop()), n_bins=2)
    # favbuyer just rides the base rate at 0.7 (wins 0.8): raw edge looks sharp,
    # skill (residual) edge ~ 0. skilled wins ALL at 0.7: it beat the base rate.
    favbuyer = [bet("favbuyer", entry_price=0.7, resolved=True, resolved_value=1.0 if i < 8 else 0.0)
                for i in range(10)]
    skilled = [bet("skilled", entry_price=0.7, resolved=True, resolved_value=1.0) for _ in range(10)]
    bets = only_buys(make_ledger(favbuyer + skilled))

    raw = compute_edge(bets)
    skill = compute_skill_edge(bets, baseline)
    # raw edge flags the favorite-buyer as sharp purely from the base rate...
    assert raw["favbuyer"] == pytest.approx(0.1, abs=1e-9)
    # ...but the residual edge removes that structural component.
    assert skill["favbuyer"] == pytest.approx(0.0, abs=1e-9)
    # genuine skill (beating the price paid) survives residualization.
    assert skill["skilled"] == pytest.approx(0.2, abs=1e-9)
    assert skill["skilled"] > skill["favbuyer"]


def test_residual_edge_per_bet_matches_outcome_minus_expected():
    baseline = fit_price_baseline(make_ledger(_calibration_pop()), n_bins=2)
    bets = pd.DataFrame({"entry_price": [0.3, 0.7], "resolved_value": [1.0, 0.0]})
    resid = residual_edge_per_bet(bets, baseline)
    assert resid[0] == pytest.approx(1.0 - 0.2)   # won a longshot the market underrates
    assert resid[1] == pytest.approx(0.0 - 0.8)   # lost a favorite the market overrates


# --- sample size / breadth ---------------------------------------------

def test_compute_sample_size_counts_resolved_only():
    rows = [
        bet("A", resolved=True, resolved_value=1.0),
        bet("A", resolved=True, resolved_value=0.0),
        bet("A", resolved=False),
    ]
    n = compute_sample_size(make_ledger(rows))
    assert n["A"] == 2


def test_compute_breadth_counts_distinct_markets_incl_unresolved():
    rows = [
        bet("A", market_id="m1"),
        bet("A", market_id="m2", resolved=True, resolved_value=1.0),
        bet("A", market_id="m1", resolved=True, resolved_value=0.0),
    ]
    breadth = compute_breadth(make_ledger(rows))
    assert breadth["A"] == 2


# --- forward_price -------------------------------------------------------

def test_forward_price_uses_window_weighted_average():
    trades = pd.DataFrame([
        {"timestamp": 100, "entry_price": 0.5, "wallet": "A", "size": 10},
        {"timestamp": 150, "entry_price": 0.6, "wallet": "B", "size": 10},
        {"timestamp": 160, "entry_price": 0.8, "wallet": "C", "size": 30},
    ])
    fv = forward_price(trades, entry_ts=100, exclude_wallet="A", window_seconds=100)
    # weighted avg of 0.6 (size 10) and 0.8 (size 30) = (0.6*10+0.8*30)/40 = 0.75
    assert fv == pytest.approx(0.75)


def test_forward_price_excludes_own_wallet():
    trades = pd.DataFrame([
        {"timestamp": 100, "entry_price": 0.5, "wallet": "A", "size": 10},
        {"timestamp": 110, "entry_price": 0.9, "wallet": "A", "size": 999},
        {"timestamp": 120, "entry_price": 0.4, "wallet": "B", "size": 5},
    ])
    fv = forward_price(trades, entry_ts=100, exclude_wallet="A", window_seconds=1000)
    assert fv == pytest.approx(0.4)


def test_forward_price_returns_none_beyond_window_no_unbounded_fallback():
    # The only other-wallet trade is far past the window: no beyond-window reach
    # (that could grab a near-resolution price), so the result is None, not 0.7.
    trades = pd.DataFrame([
        {"timestamp": 100, "entry_price": 0.5, "wallet": "A", "size": 10},
        {"timestamp": 500, "entry_price": 0.7, "wallet": "B", "size": 5},
    ])
    fv = forward_price(trades, entry_ts=100, exclude_wallet="A", window_seconds=10)
    assert fv is None


def test_forward_price_never_injects_outcome_when_no_future_trades():
    # No other-wallet future trade -> None. There is deliberately no
    # resolved_value fallback, so the 0/1 outcome can never leak in as fair value.
    trades = pd.DataFrame([
        {"timestamp": 100, "entry_price": 0.5, "wallet": "A", "size": 10},
    ])
    fv = forward_price(trades, entry_ts=100, exclude_wallet="A", window_seconds=10)
    assert fv is None


def test_forward_price_cutoff_excludes_near_resolution_trades():
    trades = pd.DataFrame([
        {"timestamp": 100, "entry_price": 0.5, "wallet": "A", "size": 10},
        {"timestamp": 150, "entry_price": 0.6, "wallet": "B", "size": 10},  # before cutoff
        {"timestamp": 199, "entry_price": 0.99, "wallet": "C", "size": 10},  # near resolution
    ])
    # cutoff_ts=160 drops the 0.99 trade; only the 0.6 trade counts.
    fv = forward_price(trades, entry_ts=100, exclude_wallet="A", window_seconds=1000, cutoff_ts=160)
    assert fv == pytest.approx(0.6)


def test_compute_forward_drift_averages_per_wallet():
    rows = [
        bet("A", token_id="t1", timestamp=0, entry_price=0.5),
        bet("B", token_id="t1", timestamp=10, entry_price=0.6, size=10),
    ]
    ledger = make_ledger(rows)
    bets = only_buys(ledger)
    drift = compute_forward_drift(bets, ledger, window_hours=1000 / 3600)
    # A's bet: future other-wallet trade is B's at price 0.6 -> drift = 0.1
    assert drift["A"] == pytest.approx(0.1)


def test_compute_forward_drift_resolution_guard_drops_late_trades():
    # Token life spans ts 0..100. B's early trade (ts 20, 0.6) is fair value;
    # C's late trade (ts 95, 0.99) sits in the final 20% and must be excluded.
    rows = [
        bet("A", token_id="t1", timestamp=0, entry_price=0.5),
        bet("B", token_id="t1", timestamp=20, entry_price=0.6, size=10),
        bet("C", token_id="t1", timestamp=95, entry_price=0.99, size=10),
    ]
    ledger = make_ledger(rows)
    bets = only_buys(ledger)
    guarded = compute_forward_drift(bets, ledger, window_hours=1000 / 3600, resolution_guard=0.2)
    unguarded = compute_forward_drift(bets, ledger, window_hours=1000 / 3600, resolution_guard=0.0)
    # guarded: only B's 0.6 counts -> drift 0.1; unguarded: (0.6+0.99)/2 - 0.5 pulled up by leak
    assert guarded["A"] == pytest.approx(0.1)
    assert unguarded["A"] > guarded["A"]


# --- vectorized perf fix: numerical equivalence to the mask version ----------
# compute_forward_drift is now fully vectorized (integer composite keys +
# prefix sums, no Python per-token/per-bet loop) instead of rebuilding a boolean
# mask per bet. This reference is the exact pre-optimization mask logic; the
# optimized version must match it to a tight tolerance on arbitrary —
# importantly, non-timestamp-sorted — input, since the real ledger is not sorted
# within a token. It is numerically equivalent, not bit-identical: the size-
# weighted sums are now formed by differencing prefix sums accumulated in
# timestamp order rather than summing each masked subset in original row order,
# which differs only at the float-rounding (~1e-8) level, far below any decision
# threshold (copy_window is cents-scale; the copyable gate uses a magnitude
# floor). The window-membership / own-wallet-exclusion SET must be exactly right,
# which is what these tests pin down.

def _reference_forward_drift(bets, ledger, window_hours, resolution_guard=0.0):
    """Pre-optimization implementation: rebuild a boolean mask over the full
    per-token trade array for every bet. Kept here only to pin the optimized
    version's output."""
    window_seconds = window_hours * 3600
    drifts, wallets = [], []
    ledger_by_token = {tid: g for tid, g in ledger.groupby("token_id")}
    for token_id, token_bets in bets.groupby("token_id"):
        token_trades = ledger_by_token.get(token_id)
        if token_trades is None or token_bets.empty:
            continue
        ts = token_trades["timestamp"].to_numpy()
        price = token_trades["entry_price"].to_numpy()
        size = token_trades["size"].to_numpy()
        wallet_arr = token_trades["wallet"].to_numpy()
        t_min, t_max = ts.min(), ts.max()
        cutoff_ts = t_max - resolution_guard * (t_max - t_min)
        cols = token_bets[["wallet", "timestamp", "entry_price"]]
        for bet_wallet, bet_ts, bet_entry in cols.itertuples(index=False, name=None):
            mask = (
                (wallet_arr != bet_wallet)
                & (ts > bet_ts)
                & (ts <= bet_ts + window_seconds)
                & (ts <= cutoff_ts)
            )
            if not mask.any():
                continue
            cprice, csize = price[mask], size[mask]
            weight = csize.sum()
            fv = float(cprice.mean()) if weight <= 0 else float((cprice * csize).sum() / weight)
            drifts.append(fv - bet_entry)
            wallets.append(bet_wallet)
    if not drifts:
        return pd.Series(dtype=float)
    return pd.Series(drifts, index=wallets).groupby(level=0).mean()


def _scrambled_multiwallet_ledger(seed=0):
    """Several tokens, many wallets/bets each, deliberately shuffled so rows are
    NOT in timestamp order within a token (mirrors the real ledger) and each
    window holds several trades of varying size — so a mismatch in summation
    order would show up as a non-exact difference."""
    rng = np.random.default_rng(seed)
    rows = []
    for token in range(6):
        n = int(rng.integers(12, 40))
        for _ in range(n):
            w = f"w{int(rng.integers(0, 8))}"
            rows.append(bet(
                w, market_id=f"m{token}", token_id=f"tok{token}",
                entry_price=round(float(rng.uniform(0.05, 0.95)), 4),
                size=float(rng.integers(1, 500)),
                timestamp=int(rng.integers(0, 10_000)),
                resolved=True, resolved_value=float(rng.integers(0, 2)),
            ))
    rng.shuffle(rows)
    return make_ledger(rows)


def _assert_series_equivalent(a, b):
    a, b = a.sort_index(), b.sort_index()
    # same wallets survive (identical window / own-wallet-exclusion SET)...
    assert list(a.index) == list(b.index)
    # ...and identical values up to float-rounding of the summation order.
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("window_hours", [6, 24, 1000 / 3600])
@pytest.mark.parametrize("guard", [0.0, 0.2])
def test_compute_forward_drift_matches_mask_reference(window_hours, guard):
    ledger = _scrambled_multiwallet_ledger(seed=window_hours and 3 or 1)
    bets = only_buys(ledger)
    optimized = compute_forward_drift(bets, ledger, window_hours=window_hours, resolution_guard=guard)
    reference = _reference_forward_drift(bets, ledger, window_hours=window_hours, resolution_guard=guard)
    _assert_series_equivalent(optimized, reference)


def test_compute_forward_drift_matches_reference_across_many_seeds():
    for seed in range(25):
        ledger = _scrambled_multiwallet_ledger(seed=seed)
        bets = only_buys(ledger)
        optimized = compute_forward_drift(bets, ledger, window_hours=24, resolution_guard=0.2)
        reference = _reference_forward_drift(bets, ledger, window_hours=24, resolution_guard=0.2)
        _assert_series_equivalent(optimized, reference)


# --- time consistency ------------------------------------------------------

def test_compute_time_consistency_fraction_of_positive_buckets():
    rows = [
        bet("A", timestamp=0, resolved=True, entry_price=0.2, resolved_value=1.0),  # +0.8
        bet("A", timestamp=1, resolved=True, entry_price=0.9, resolved_value=1.0),  # +0.1
        bet("A", timestamp=2, resolved=True, entry_price=0.9, resolved_value=0.0),  # -0.9
        bet("A", timestamp=3, resolved=True, entry_price=0.8, resolved_value=0.0),  # -0.8
    ]
    tc = compute_time_consistency(make_ledger(rows), buckets=4)
    assert tc["A"] == pytest.approx(0.5)


def test_compute_time_consistency_nan_when_insufficient_bets():
    rows = [bet("A", timestamp=0, resolved=True, entry_price=0.2, resolved_value=1.0)]
    tc = compute_time_consistency(make_ledger(rows), buckets=4)
    assert pd.isna(tc["A"])


# --- market windows ----------------------------------------------------

def test_compute_market_windows():
    rows = [
        bet("A", market_id="m1", timestamp=100),
        bet("B", market_id="m1", timestamp=300),
    ]
    windows = compute_market_windows(make_ledger(rows))
    assert windows.loc["m1", "market_min_ts"] == 100
    assert windows.loc["m1", "market_max_ts"] == 300


# --- manufactured_record_flag -------------------------------------------

def test_manufactured_record_flag_true_for_concentrated_counterparty():
    rows = []
    for i in range(6):
        rows.append(bet("X", market_id=f"m{i}", side="BUY", entry_price=0.5, size=100, tx_hash=f"tx{i}"))
        rows.append(bet("Y", market_id=f"m{i}", side="SELL", entry_price=0.5, size=100, tx_hash=f"tx{i}"))
    flags = compute_manufactured_record_flag(make_ledger(rows), counterparty_share_threshold=0.4, min_markets=5)
    assert bool(flags["X"])
    assert bool(flags["Y"])


def test_manufactured_record_flag_false_for_diverse_counterparties():
    rows = []
    for i in range(6):
        rows.append(bet("X", market_id=f"m{i}", side="BUY", entry_price=0.5, size=100, tx_hash=f"tx{i}"))
        rows.append(bet(f"other{i}", market_id=f"m{i}", side="SELL", entry_price=0.5, size=100, tx_hash=f"tx{i}"))
    flags = compute_manufactured_record_flag(make_ledger(rows), counterparty_share_threshold=0.4, min_markets=5)
    assert not bool(flags["X"])


# --- pattern_flag --------------------------------------------------------

def test_pattern_flag_late_concentrated_entry():
    rows = [
        bet("A", market_id="m1", timestamp=1000, size=1),
        bet("A", market_id="m1", timestamp=2000, size=1),
        bet("A", market_id="m1", timestamp=3000, size=1),
        bet("A", market_id="m1", timestamp=4000, size=1),
        bet("A", market_id="m1", timestamp=95000, size=50),
        bet("A", market_id="m1", timestamp=96000, size=50),
        bet("A", market_id="m1", timestamp=97000, size=50),
    ]
    ledger = make_ledger(rows)
    bets = only_buys(ledger)
    windows = compute_market_windows(ledger)
    # extend window so market spans 0-100,000s (well over the micro-market cutoff)
    windows.loc["m1", "market_min_ts"] = 0
    windows.loc["m1", "market_max_ts"] = 100000
    flags = compute_pattern_flag(bets, windows, late_entry_percentile=0.1)
    assert flags["A"] == "late_concentrated_entry"


def test_pattern_flag_high_frequency_micro_market():
    rows = [bet("A", market_id="m1", timestamp=t, size=1) for t in range(5)]
    ledger = make_ledger(rows)
    bets = only_buys(ledger)
    windows = compute_market_windows(ledger)  # lifespan = 4 seconds, well under 1800
    flags = compute_pattern_flag(bets, windows, late_entry_percentile=0.1)
    assert flags["A"] == "high_frequency_micro_market"


def test_pattern_flag_none_for_ordinary_wallet():
    rows = [bet("A", market_id="m1", timestamp=t * 1000, size=1) for t in range(5)]
    ledger = make_ledger(rows)
    bets = only_buys(ledger)
    windows = compute_market_windows(ledger)
    windows.loc["m1", "market_min_ts"] = 0
    windows.loc["m1", "market_max_ts"] = 100000
    flags = compute_pattern_flag(bets, windows, late_entry_percentile=0.1)
    assert flags["A"] is None


# --- full integration: nothing is ever dropped --------------------------

def test_compute_wallet_features_keeps_every_wallet():
    rows = [
        bet("A", resolved=True, resolved_value=1.0, entry_price=0.3),
        bet("B", resolved=False),  # only unresolved bet — must still appear
    ]
    cfg = {
        "scoring": {
            "copy_window_hours": 24,
            "earliness_window_hours": 6,
            "time_consistency_buckets": 4,
            "manufactured_flag_counterparty_share": 0.4,
            "manufactured_flag_min_markets": 5,
            "late_entry_percentile": 0.1,
        }
    }
    features = compute_wallet_features(make_ledger(rows), cfg)
    assert set(features["wallet"]) == {"A", "B"}
    b_row = features.loc[features["wallet"] == "B"].iloc[0]
    assert b_row["sample_size"] == 0
    assert pd.isna(b_row["edge"])
