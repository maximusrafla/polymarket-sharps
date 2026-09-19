"""Tests for the event-weighted aggregate forward scorer (src/slow_aggregate.py).

The statistic exists to stop one hyperactive event complex becoming the answer,
so that is what is pinned hardest: an event with 1000 bets and an event with 3
bets must count the same, and the uncertainty must come from the number of
EVENTS rather than the number of bets.
"""

import numpy as np
import pandas as pd
import pytest

from src.slow_aggregate import (
    add_fine_event_column,
    effective_events,
    event_block_bootstrap,
    event_weighted_mean,
    resolution_speed_stratum,
    score_group,
    score_tier,
)


def _bets(spec, wallet="w"):
    """spec: {event: [residuals]} -> a frame."""
    rows = []
    for e, rs in spec.items():
        for r in rs:
            rows.append({"wallet": wallet, "niche_l1": e, "residual_skill": r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the weighting itself
# ---------------------------------------------------------------------------

def test_event_weighted_mean_counts_each_event_once():
    """Event A: 1000 bets at +1.0. Event B: 2 bets at -1.0.
    Bet-weighted is ~+1.0 (A drowns B). Event-weighted is exactly 0.0."""
    df = _bets({"A": [1.0] * 1000, "B": [-1.0] * 2})
    r = df["residual_skill"].to_numpy()
    e = df["niche_l1"].to_numpy()
    assert event_weighted_mean(r, e) == pytest.approx(0.0)
    assert np.mean(r) > 0.99


def test_bet_and_event_weighted_agree_on_a_balanced_sample():
    df = _bets({"A": [0.2] * 10, "B": [0.4] * 10})
    r = df["residual_skill"].to_numpy()
    e = df["niche_l1"].to_numpy()
    assert event_weighted_mean(r, e) == pytest.approx(np.mean(r))
    assert event_weighted_mean(r, e) == pytest.approx(0.3)


def test_effective_events_reflects_concentration_not_count():
    assert effective_events(["A"] * 100) == pytest.approx(1.0)
    assert effective_events(["A", "B", "C", "D"]) == pytest.approx(4.0)
    # 1000 bets but effectively ~1 event
    assert effective_events(["A"] * 1000 + ["B"]) < 1.01


# ---------------------------------------------------------------------------
# uncertainty comes from events, not bets
# ---------------------------------------------------------------------------

def test_bootstrap_is_undefined_with_a_single_event():
    """One event = one observation. Refuse by rule rather than return a 0 p."""
    df = _bets({"A": [0.5] * 500})
    out = event_block_bootstrap(df["residual_skill"].to_numpy(),
                                df["niche_l1"].to_numpy(),
                                np.random.default_rng(0), n_boot=200)
    assert out["n_events"] == 1
    assert np.isnan(out["p_one_sided"])
    assert np.isnan(out["ci_low"])


def test_more_bets_in_the_same_event_do_not_narrow_the_interval():
    """The failure mode this design exists to prevent: piling bets into the same
    complexes must not manufacture confidence."""
    few = _bets({"A": [0.2] * 5, "B": [0.3] * 5, "C": [0.25] * 5})
    many = _bets({"A": [0.2] * 500, "B": [0.3] * 500, "C": [0.25] * 500})
    kw = dict(rng=np.random.default_rng(0), n_boot=1000)
    a = event_block_bootstrap(few["residual_skill"].to_numpy(),
                              few["niche_l1"].to_numpy(), **kw)
    b = event_block_bootstrap(many["residual_skill"].to_numpy(),
                              many["niche_l1"].to_numpy(),
                              rng=np.random.default_rng(0), n_boot=1000)
    width_a = a["ci_high"] - a["ci_low"]
    width_b = b["ci_high"] - b["ci_low"]
    assert width_b == pytest.approx(width_a, rel=0.05)   # 100x the bets, same width


def test_more_events_do_narrow_the_interval():
    rng = np.random.default_rng(0)
    few = _bets({f"e{i}": [0.2 + 0.1 * (i % 3)] * 10 for i in range(3)})
    lots = _bets({f"e{i}": [0.2 + 0.1 * (i % 3)] * 10 for i in range(40)})
    a = event_block_bootstrap(few["residual_skill"].to_numpy(),
                              few["niche_l1"].to_numpy(), rng, n_boot=1000)
    b = event_block_bootstrap(lots["residual_skill"].to_numpy(),
                              lots["niche_l1"].to_numpy(),
                              np.random.default_rng(0), n_boot=1000)
    assert (b["ci_high"] - b["ci_low"]) < (a["ci_high"] - a["ci_low"])


def test_bootstrap_is_deterministic_for_a_given_seed():
    df = _bets({f"e{i}": [0.1 * i] * 4 for i in range(6)})
    kw = dict(n_boot=500)
    a = event_block_bootstrap(df["residual_skill"].to_numpy(),
                              df["niche_l1"].to_numpy(),
                              np.random.default_rng(7), **kw)
    b = event_block_bootstrap(df["residual_skill"].to_numpy(),
                              df["niche_l1"].to_numpy(),
                              np.random.default_rng(7), **kw)
    assert a == b


def test_a_real_crowd_edge_across_many_events_is_detected():
    df = _bets({f"e{i}": [0.25] * 8 for i in range(25)})
    out = score_group(df, seed=1, n_boot=1000)
    assert out["edge_event_weighted"] == pytest.approx(0.25)
    assert out["p_one_sided"] < 0.05
    assert out["ci_low"] > 0


def test_a_zero_edge_crowd_is_not_significant():
    rng = np.random.default_rng(3)
    spec = {f"e{i}": list(rng.normal(0, 0.3, 8)) for i in range(25)}
    out = score_group(_bets(spec), seed=1, n_boot=1000)
    assert out["p_one_sided"] > 0.05


# ---------------------------------------------------------------------------
# reporting contract
# ---------------------------------------------------------------------------

def test_score_group_always_pairs_bet_count_with_effective_events():
    """Volume must never be reportable without its evidentiary discount."""
    out = score_group(_bets({"A": [0.3] * 5000, "B": [0.3] * 2}), seed=0, n_boot=200)
    assert out["n_bets"] == 5002
    assert out["effective_events"] < 1.01        # ~1 event despite 5002 bets
    for k in ("n_bets", "n_events", "effective_events", "edge_event_weighted",
              "edge_bet_weighted", "ci_low", "ci_high", "p_one_sided"):
        assert k in out


def test_score_group_on_empty_input():
    out = score_group(_bets({}).assign(residual_skill=pd.Series(dtype=float)),
                      seed=0, n_boot=10)
    assert out["n_bets"] == 0
    assert np.isnan(out["edge_event_weighted"])


def test_nan_residuals_are_excluded_not_counted():
    df = _bets({"A": [0.2, np.nan, 0.4], "B": [0.3, 0.3]})
    out = score_group(df, seed=0, n_boot=200)
    assert out["n_bets"] == 4


# ---------------------------------------------------------------------------
# resolution-speed split
# ---------------------------------------------------------------------------

def test_resolution_speed_split_is_keyed_on_entry_to_resolution():
    """Keyed on the market's own speed, so a bet's stratum cannot change between
    scoring runs as time passes."""
    day = 86400
    df = pd.DataFrame({
        "timestamp": [0, 0, 0],
        "resolution_ts": [5 * day, 14 * day, 40 * day],
    })
    got = resolution_speed_stratum(df, 14)
    assert list(got) == ["fast", "fast", "slow"]


def test_score_tier_emits_overall_plus_each_stratum():
    day = 86400
    df = _bets({f"e{i}": [0.2] * 6 for i in range(5)})
    df["timestamp"] = 0
    df["resolution_ts"] = [day * (3 if i % 2 else 30) for i in range(len(df))]
    rows = score_tier(df, "t12", split_days=14, n_boot=200)
    strata = [r["stratum"] for r in rows]
    assert "all" in strata
    assert any(s.startswith("fast") for s in strata)
    assert any(s.startswith("slow") for s in strata)
    assert all(r["tier"] == "t12" for r in rows)


def test_fine_event_column_splits_a_complex_by_resolution_week():
    day = 86400
    df = pd.DataFrame({
        "niche_l1": ["geo_iran", "geo_iran"],
        "resolution_ts": [0, 200 * day],
    })
    out = add_fine_event_column(df)
    assert out["event_fine"].nunique() == 2
    assert all(v.startswith("geo_iran@") for v in out["event_fine"])
