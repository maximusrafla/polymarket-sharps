"""Tests for the wallet-agnostic placebo (src/slow_placebo.py).

The placebo is the control that separates "this crowd has an edge" from "these
markets were mispriced for everyone", so what is pinned is that it actually
discriminates those two cases, and that a capped run can never masquerade as a
complete one.

No test touches the network.
"""

import numpy as np
import pandas as pd
import pytest

from src.slow_placebo import matched_null, placebo_for_tier


def _tape(spec):
    """spec: list of (wallet, market, complex, residual)."""
    return pd.DataFrame(
        [{"wallet": w, "market_id": m, "niche_l1": c, "residual_skill": r}
         for w, m, c, r in spec])


def _crowd(wallets, markets, complexes, residual):
    out = []
    for w in wallets:
        for m, c in zip(markets, complexes):
            out.append((w, m, c, residual))
    return out


MK = ["m1", "m2", "m3", "m4", "m5"]
CX = ["c1", "c2", "c3", "c4", "c5"]


def test_a_market_wide_lift_is_caught_by_the_placebo():
    """The confound the placebo exists for: the cohort looks good in absolute
    terms, but everyone in those markets did just as well. Absolute edge is
    positive; the percentile against a matched crowd is unremarkable."""
    cohort = _tape(_crowd(["a", "b"], MK, CX, 0.20))
    others = _tape(_crowd([f"o{i}" for i in range(30)], MK, CX, 0.20))
    got = placebo_for_tier(cohort, others, "t12", n_wallets=2, seed=1, n_draws=300)
    assert got["placebo_edge"] == pytest.approx(0.20)
    # cohort is not distinguishable from the room
    assert got["placebo_percentile"] < 0.95


def test_a_genuine_wallet_edge_beats_the_matched_crowd():
    cohort = _tape(_crowd(["a", "b"], MK, CX, 0.40))
    others = _tape(_crowd([f"o{i}" for i in range(30)], MK, CX, 0.00))
    got = placebo_for_tier(cohort, others, "t12", n_wallets=2, seed=1, n_draws=300)
    assert got["placebo_edge"] == pytest.approx(0.0)
    assert got["placebo_percentile"] == pytest.approx(1.0)


def test_cohort_wallets_are_excluded_from_their_own_placebo():
    """Otherwise the control is contaminated by the thing it is controlling for."""
    cohort = _tape(_crowd(["a"], MK, CX, 0.50))
    both = _tape(_crowd(["a"], MK, CX, 0.50) + _crowd([f"o{i}" for i in range(10)],
                                                      MK, CX, 0.0))
    got = placebo_for_tier(cohort, both, "t12", n_wallets=1, seed=1, n_draws=200)
    assert got["placebo_edge"] == pytest.approx(0.0)   # 'a' did not leak in
    assert got["placebo_wallets"] == 10


def test_placebo_is_restricted_to_the_markets_the_cohort_actually_traded():
    cohort = _tape(_crowd(["a"], ["m1"], ["c1"], 0.30))
    others = _tape(_crowd(["o1"], ["m1"], ["c1"], 0.10)
                   + _crowd(["o2"], ["m9"], ["c9"], 9.99))   # unrelated market
    got = placebo_for_tier(cohort, others, "t12", n_wallets=1, seed=1, n_draws=100)
    assert got["placebo_edge"] == pytest.approx(0.10)
    assert got["placebo_bets"] == 1


def test_matched_null_respects_crowd_size():
    """A 2-wallet crowd and a 30-wallet crowd have different sampling variance, so
    the null must be built at the cohort's own size."""
    rng = np.random.default_rng(0)
    others = _tape([(f"o{i}", m, c, float(i % 7) / 10.0)
                    for i in range(30) for m, c in zip(MK, CX)])
    small = matched_null(others, 2, 400, np.random.default_rng(0))
    large = matched_null(others, 25, 400, np.random.default_rng(0))
    assert small.size and large.size
    assert np.std(small) > np.std(large)


def test_matched_null_is_empty_when_the_room_is_too_thin():
    others = _tape(_crowd(["o1", "o2"], MK, CX, 0.1))
    assert matched_null(others, 2, 100, np.random.default_rng(0)).size == 0


def test_placebo_returns_a_clean_empty_when_no_other_wallets_exist():
    cohort = _tape(_crowd(["a"], MK, CX, 0.3))
    got = placebo_for_tier(cohort, pd.DataFrame(), "t12", n_wallets=1, seed=1)
    assert np.isnan(got["placebo_edge"])
    assert got["placebo_wallets"] == 0


def test_placebo_is_event_weighted_like_the_cohort_statistic():
    """One hyperactive complex must not carry the placebo either, or the two sides
    of the comparison would not be computed the same way."""
    others = _tape([("o1", "m1", "c1", 1.0)] * 500 + [("o2", "m2", "c2", -1.0)] * 2
                   + [("o3", "m3", "c3", 0.0)] * 2)
    cohort = _tape(_crowd(["a"], ["m1", "m2", "m3"], ["c1", "c2", "c3"], 0.1))
    got = placebo_for_tier(cohort, others, "t12", n_wallets=1, seed=1, n_draws=100)
    # event-weighted over c1/c2/c3 = (1.0 - 1.0 + 0.0)/3 = 0.0, not ~+1.0
    assert got["placebo_edge"] == pytest.approx(0.0)


def test_deterministic_for_a_given_seed():
    cohort = _tape(_crowd(["a", "b"], MK, CX, 0.25))
    others = _tape(_crowd([f"o{i}" for i in range(20)], MK, CX, 0.05))
    kw = dict(tier="t12", n_wallets=2, seed=42, n_draws=200)
    assert placebo_for_tier(cohort, others, **kw) == placebo_for_tier(cohort, others, **kw)
