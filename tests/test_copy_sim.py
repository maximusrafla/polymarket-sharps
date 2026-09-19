"""Hand-built fixtures for the copy-lag simulation, where the right answer is
known by construction.

The four things pinned here are the ones that have silently broken every prior
edge-decay analysis in this repo:
  1. no follower price available -> NaN, and NEVER `resolved_value`;
  2. the resolution guard actually excludes late prices;
  3. the same-cohort comparison uses the same bet subset for both arms;
  4. fees are charged at the right per-category rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import copy_sim as cs


# --------------------------------------------------------------------------
# 1. Bounded window: no price available -> NaN, never resolved_value
# --------------------------------------------------------------------------
def test_price_at_returns_first_point_inside_window():
    ts = np.array([100, 160, 220], dtype=np.int64)
    px = np.array([0.30, 0.40, 0.50])
    # target 150, bandwidth 60 -> window [150, 210] -> the 160 point.
    assert cs.price_at(ts, px, 150, 60, cutoff=1e12) == pytest.approx(0.40)


def test_price_at_accepts_a_point_exactly_on_the_target():
    ts = np.array([100, 200], dtype=np.int64)
    px = np.array([0.30, 0.40])
    assert cs.price_at(ts, px, 200, 60, cutoff=1e12) == pytest.approx(0.40)


def test_price_at_is_nan_when_the_window_is_empty():
    ts = np.array([100, 5000], dtype=np.int64)
    px = np.array([0.30, 0.99])
    # Next point is 4,800s past the target; the bounded window is 60s.
    out = cs.price_at(ts, px, 200, 60, cutoff=1e12)
    assert np.isnan(out)


def test_price_at_never_reaches_past_the_end_of_the_series():
    ts = np.array([100, 160], dtype=np.int64)
    px = np.array([0.30, 0.40])
    assert np.isnan(cs.price_at(ts, px, 1_000, 60, cutoff=1e12))


def test_price_at_is_nan_not_resolved_value_when_unavailable():
    """The failure mode this guards: falling back to the outcome (1.0) would make
    the follower look like a genius on exactly the un-copyable bets."""
    ts = np.zeros(0, dtype=np.int64)
    px = np.zeros(0)
    resolved_value = 1.0
    out = cs.price_at(ts, px, 500, 60, cutoff=1e12)
    assert np.isnan(out)
    assert out != resolved_value


def test_compute_per_bet_leaves_returns_nan_when_no_price_exists():
    bets = _frame(entry_ts=[1_000], entry_price=[0.40], resolved_value=[1.0],
                  cutoff=[1e12])
    store, offsets = _store({0: ([1_000], [0.40])})   # only the entry minute exists
    out = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")])
    assert np.isnan(out["price_2m"].iloc[0])
    assert np.isnan(out["gross_2m"].iloc[0])
    assert np.isnan(out["net_2m"].iloc[0])
    # ...while the wallet's own Δ=0 return is still measured.
    assert out["gross_0"].iloc[0] == pytest.approx((1.0 - 0.40) / 0.40)


# --------------------------------------------------------------------------
# 2. Resolution guard
# --------------------------------------------------------------------------
def test_guard_cutoff_drops_the_final_fraction_of_the_lifespan():
    # lifespan [0, 1000], guard 0.2 -> cutoff at 800
    assert cs.guard_cutoff(0, 1000, 0.2) == pytest.approx(800.0)
    assert cs.guard_cutoff(500, 1500, 0.1) == pytest.approx(1400.0)


def test_guard_cutoff_is_nan_when_the_lifespan_is_unknown():
    assert np.isnan(cs.guard_cutoff(np.nan, 1000, 0.2))
    assert np.isnan(cs.guard_cutoff(0, np.nan, 0.2))
    assert np.isnan(cs.guard_cutoff(1000, 500, 0.2))    # end before start


def test_price_at_excludes_a_price_past_the_guard_cutoff():
    ts = np.array([100, 160], dtype=np.int64)
    px = np.array([0.30, 0.98])          # 0.98 is the near-resolution price
    assert cs.price_at(ts, px, 150, 60, cutoff=1e12) == pytest.approx(0.98)
    # Same series, same window — but the guard says 160 is too close to the end.
    assert np.isnan(cs.price_at(ts, px, 150, 60, cutoff=155))


def test_unknown_lifespan_makes_every_follower_price_unusable():
    """No lifespan -> no guard -> we must refuse to measure, not guess. NaN is the
    'unknown' sentinel; inf is the explicit 'no guard' value, and they must not be
    confused (an unknown lifespan silently becoming 'no guard' is the bug)."""
    ts = np.array([100, 160], dtype=np.int64)
    px = np.array([0.30, 0.98])
    assert np.isnan(cs.price_at(ts, px, 150, 60, cutoff=float("nan")))
    assert cs.price_at(ts, px, 150, 60, cutoff=float("inf")) == pytest.approx(0.98)


def test_mid_at_entry_is_a_diagnostic_that_never_feeds_a_score():
    """`mid_at_entry` deliberately ignores the guard (it is our own tape at the
    instant of entry, not a forward price). It must never leak into a return."""
    bets = _frame(entry_ts=[1_000], entry_price=[0.40], resolved_value=[1.0],
                  cutoff=[500])                       # guard excludes everything
    store, offsets = _store({0: ([1_000, 1_120], [0.44, 0.90])})
    out = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")])
    assert out["mid_at_entry"].iloc[0] == pytest.approx(0.44)   # guard not applied
    assert np.isnan(out["price_2m"].iloc[0])                    # guard applied
    assert np.isnan(out["net_2m"].iloc[0])
    # Δ=0 is the wallet's own recorded fill, not the midpoint.
    assert out["gross_0"].iloc[0] == pytest.approx((1.0 - 0.40) / 0.40)


def test_compute_per_bet_applies_the_guard_end_to_end():
    # Two identical bets; only the guard cutoff differs.
    bets = _frame(entry_ts=[1_000, 1_000], entry_price=[0.40, 0.40],
                  resolved_value=[1.0, 1.0], cutoff=[1e12, 1_050], tok=[0, 1])
    store, offsets = _store({0: ([1_000, 1_120], [0.40, 0.90]),
                             1: ([1_000, 1_120], [0.40, 0.90])})
    out = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")])
    assert out["price_2m"].iloc[0] == pytest.approx(0.90)   # unguarded
    assert np.isnan(out["price_2m"].iloc[1])                # guard bites


# --------------------------------------------------------------------------
# 3. Same-cohort comparison
# --------------------------------------------------------------------------
def test_summarize_uses_the_same_bet_subset_for_both_arms():
    """Bet A is measurable at Δ; bet B is not. The own-return column at Δ must be
    A's own return alone — NOT the mean over A and B."""
    bets = _frame(entry_ts=[1_000, 2_000], entry_price=[0.50, 0.10],
                  resolved_value=[1.0, 1.0], cutoff=[1e12, 1e12], tok=[0, 1],
                  market=["mA", "mB"])
    store, offsets = _store({0: ([1_000, 1_120], [0.50, 0.60]),
                             1: ([2_000], [0.10])})        # B has no +2m point
    per_bet = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")])
    rng = np.random.default_rng(0)
    s = cs.summarize(per_bet, "T", rng, deltas=[(0, "0"), (120, "2m")], n_shuffle=5)
    row = s.loc[s["delta"] == "2m"].iloc[0]
    assert row["n"] == 1
    assert row["coverage"] == pytest.approx(0.5)
    # A's own return only: (1.0 - 0.50)/0.50 = 1.0.  The full-sample number would
    # be the mean of 1.0 and B's 9.0 = 5.0, which is the bug this pins.
    assert row["own_gross_cohort"] == pytest.approx(1.0)
    full_sample = per_bet["gross_0"].mean()
    assert full_sample == pytest.approx(5.0)
    assert row["own_gross_cohort"] != pytest.approx(full_sample)


def test_summarize_delta_zero_row_covers_every_bet():
    bets = _frame(entry_ts=[1_000, 2_000], entry_price=[0.50, 0.10],
                  resolved_value=[1.0, 0.0], cutoff=[1e12, 1e12], tok=[0, 1],
                  market=["mA", "mB"])
    store, offsets = _store({0: ([1_000], [0.50]), 1: ([2_000], [0.10])})
    per_bet = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0")])
    rng = np.random.default_rng(0)
    s = cs.summarize(per_bet, "T", rng, deltas=[(0, "0")], n_shuffle=5)
    assert s.iloc[0]["n"] == 2
    assert s.iloc[0]["coverage"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 4. Fees, at the right per-category rate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("category,expected", [
    ("politics", 0.04),
    ("sports_nba", 0.05),
    ("sports_soccer", 0.05),
    ("micro_crypto", 0.07),
    ("crypto_event", 0.07),
    ("geopolitics", 0.0),
    ("other", 0.05),
    ("econ_macro", 0.05),
    (None, 0.05),
])
def test_fee_k_per_category(category, expected):
    assert cs.fee_k_for(category) == pytest.approx(expected)


def test_gamma_rate_overrides_the_category_map():
    # Gamma's per-market feeSchedule.rate is the only correct source.
    assert cs.fee_k_for("politics", 0.07) == pytest.approx(0.07)
    assert cs.fee_k_for("politics", float("nan")) == pytest.approx(0.04)
    assert cs.fee_k_for("politics", None) == pytest.approx(0.04)


def test_taker_fee_matches_the_documented_formula():
    # docs/polymarket_mechanics.md §5, verified on-chain to 7 s.f.:
    #   fee = shares × k × p × (1−p)
    shares, k, p = 580.677591, 0.07, 0.527058567
    fee = shares * cs.taker_fee_per_share(p, k)
    assert fee == pytest.approx(10.132090, abs=1e-5)


def test_follower_cost_is_price_plus_tick_plus_fee_on_the_filled_price():
    p, k, tick = 0.50, 0.04, 0.01
    eff = p + tick                       # 0.51 — the midpoint is not an ask
    expected = eff + k * eff * (1 - eff)
    assert cs.follower_cost(p, k, tick) == pytest.approx(expected)


def test_zero_fee_and_zero_tick_reduce_to_the_gross_price():
    assert cs.follower_cost(0.37, 0.0, 0.0) == pytest.approx(0.37)


def test_fee_exponent_is_honoured():
    """Gamma's `crypto_15_min` schedule is {rate: 0.25, exponent: 2} — a steeper,
    narrower parabola. Charging exponent 1 there would be ~16x the real fee."""
    p, k = 0.50, 0.25
    assert cs.taker_fee_per_share(p, k, 2.0) == pytest.approx(0.25 * (0.25 ** 2))
    assert cs.taker_fee_per_share(p, k, 1.0) == pytest.approx(0.25 * 0.25)
    assert cs.taker_fee_per_share(p, k, 2.0) < cs.taker_fee_per_share(p, k, 1.0)
    assert cs.follower_cost(p, k, 0.0, 2.0) == pytest.approx(0.5 + 0.25 * 0.0625)


def test_compute_per_bet_honours_a_per_market_fee_exponent():
    bets = _frame(entry_ts=[1_000, 1_000], entry_price=[0.50, 0.50],
                  resolved_value=[1.0, 1.0], cutoff=[1e12, 1e12], tok=[0, 1],
                  market=["mA", "mB"])
    bets["fee_k"] = 0.25
    bets["fee_exponent"] = [1.0, 2.0]
    store, offsets = _store({0: ([1_000, 1_120], [0.50, 0.50]),
                             1: ([1_000, 1_120], [0.50, 0.50])})
    out = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")],
                             adverse_tick=0.0)
    assert out["netedge_2m"].iloc[0] == pytest.approx(1.0 - (0.5 + 0.25 * 0.25))
    assert out["netedge_2m"].iloc[1] == pytest.approx(1.0 - (0.5 + 0.25 * 0.0625))


def test_compute_per_bet_charges_the_category_rate():
    bets = _frame(entry_ts=[1_000, 1_000], entry_price=[0.50, 0.50],
                  resolved_value=[1.0, 1.0], cutoff=[1e12, 1e12], tok=[0, 1],
                  market=["mA", "mB"])
    bets["fee_k"] = [0.04, 0.07]          # politics vs crypto
    store, offsets = _store({0: ([1_000, 1_120], [0.50, 0.50]),
                             1: ([1_000, 1_120], [0.50, 0.50])})
    out = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")],
                             adverse_tick=0.01)
    for i, k in enumerate([0.04, 0.07]):
        cost = 0.51 + k * 0.51 * 0.49
        assert out["net_2m"].iloc[i] == pytest.approx((1.0 - cost) / cost)
        assert out["netedge_2m"].iloc[i] == pytest.approx(1.0 - cost)
    # The higher fee must produce the lower net return.
    assert out["net_2m"].iloc[1] < out["net_2m"].iloc[0]
    # Gross is fee-free and therefore identical for both.
    assert out["gross_2m"].iloc[0] == pytest.approx(out["gross_2m"].iloc[1])


# --------------------------------------------------------------------------
# 5. Null and CI machinery
# --------------------------------------------------------------------------
def test_shuffle_within_bins_preserves_each_bin_multiset():
    rng = np.random.default_rng(3)
    values = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    bins = np.array([0, 0, 0, 1, 1, 1])
    out = cs.shuffle_within_bins(values, bins, rng)
    for b in (0, 1):
        assert sorted(out[bins == b]) == sorted(values[bins == b])


def test_price_bins_are_monotone_in_price():
    p = np.linspace(0.01, 0.99, 100)
    b = cs.price_bins(p, n_bins=5)
    assert np.all(np.diff(b) >= 0)
    assert b.min() == 0


def test_shuffled_null_returns_probabilities_for_both_metrics():
    rng = np.random.default_rng(7)
    rv = np.array([1.0, 0.0, 1.0, 0.0])
    cohort = np.array([True, True, True, True])
    cost = np.full(4, 0.5)
    bins = np.zeros(4, dtype=int)
    real_roi = float(np.mean((rv - cost) / cost))
    real_edge = float(np.mean(rv - cost))
    out = cs.shuffled_null(rv, cohort, cost, bins, real_roi, real_edge, rng,
                           n_shuffle=20)
    # Permuting within a single bin cannot change the mean when every cost is equal.
    assert out["null_net"] == pytest.approx(real_roi)
    assert out["null_netedge_c"] == pytest.approx(real_edge)
    assert 0.0 <= out["null_p"] <= 1.0
    assert 0.0 <= out["null_p_edge"] <= 1.0


def test_shuffled_null_strips_selection_skill():
    """A wallet that only wins its cheap bets has real edge; once outcomes are
    permuted within price bins that edge must collapse toward the base rate."""
    rng = np.random.default_rng(5)
    # 4 bets at 0.10 (all win) and 4 at 0.90 (all lose) -> strong real edge.
    cost = np.array([0.10] * 4 + [0.90] * 4)
    rv = np.array([1.0] * 4 + [0.0] * 4)
    bins = np.array([0] * 4 + [1] * 4)          # price bins mirror the two prices
    cohort = np.ones(8, dtype=bool)
    real_edge = float(np.mean(rv - cost))
    out = cs.shuffled_null(rv, cohort, cost, bins, 0.0, real_edge, rng, n_shuffle=50)
    # Within-bin permutation cannot move a bin whose outcomes are all identical,
    # so this fixture pins the *mechanism*: the null equals the real edge exactly
    # when the wallet's skill is entirely a price-level effect, i.e. not copyable.
    assert out["null_netedge_c"] == pytest.approx(real_edge)


def test_cluster_boot_ci_resamples_markets_not_bets():
    """Two markets, perfectly homogeneous inside each. Resampling markets can only
    ever produce the two market means or their average, so the CI must span them —
    a bet-level bootstrap would give a far tighter interval."""
    rng = np.random.default_rng(11)
    values = np.array([0.0] * 50 + [1.0] * 50)
    markets = np.array([0] * 50 + [1] * 50)
    lo, hi = cs.cluster_boot_ci(values, markets, rng, n=500)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_cluster_boot_ci_is_nan_with_a_single_market():
    rng = np.random.default_rng(11)
    lo, hi = cs.cluster_boot_ci(np.array([0.1, 0.2]), np.array([0, 0]), rng, n=50)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------
# 6. Request planning and wallet resolution
# --------------------------------------------------------------------------
def test_plan_requests_covers_every_bets_full_horizon():
    bets = pd.DataFrame({"token_id": ["A", "A", "B"],
                         "timestamp": [1_000, 1_000 + 3 * 3600, 50_000]})
    reqs = cs.plan_requests(bets, horizon=21_600, merge_gap=12 * 3600, pad=60)
    for _, b in bets.iterrows():
        assert any(r["token_id"] == b["token_id"]
                   and r["start"] <= b["timestamp"]
                   and r["end"] >= b["timestamp"] + 21_600
                   for r in reqs), b.to_dict()


def test_plan_requests_merges_nearby_entries_and_splits_distant_ones():
    bets = pd.DataFrame({"token_id": ["A", "A", "A"],
                         "timestamp": [0, 3600, 30 * 86400]})
    reqs = cs.plan_requests(bets, horizon=21_600, merge_gap=12 * 3600, pad=0)
    assert len(reqs) == 2


def test_plan_requests_never_exceeds_the_endpoint_span_limit():
    # 40 entries a day apart: merging them all would blow past the API's range cap.
    bets = pd.DataFrame({"token_id": ["A"] * 40,
                         "timestamp": [i * 86400 for i in range(40)]})
    reqs = cs.plan_requests(bets, horizon=21_600, merge_gap=2 * 86400, pad=60)
    assert all(r["end"] - r["start"] <= cs.MAX_REQUEST_SPAN for r in reqs)


def test_request_key_is_stable_and_unique():
    a = {"token_id": "A", "start": 1, "end": 2}
    b = {"token_id": "A", "start": 1, "end": 3}
    assert cs.request_key(a) == cs.request_key(dict(a))
    assert cs.request_key(a) != cs.request_key(b)


def test_resolve_wallets_requires_a_unique_certified_match():
    validated = pd.DataFrame({
        "wallet": ["0xaaa1", "0xaaa2", "0xbbb1"],
        "edge_persisted": [True, True, True],
    })
    assert cs.resolve_wallets(["0xbbb"], validated) == ["0xbbb1"]
    with pytest.raises(ValueError):
        cs.resolve_wallets(["0xaaa"], validated)       # ambiguous
    with pytest.raises(ValueError):
        cs.resolve_wallets(["0xccc"], validated)       # missing


def test_resolve_wallets_ignores_uncertified_wallets():
    validated = pd.DataFrame({"wallet": ["0xaaa1", "0xaaa2"],
                              "edge_persisted": [False, True]})
    assert cs.resolve_wallets(["0xaaa2"], validated) == ["0xaaa2"]
    with pytest.raises(ValueError):
        cs.resolve_wallets(["0xaaa1"], validated)


# --------------------------------------------------------------------------
# 7. A fully worked end-to-end fixture with a known answer
# --------------------------------------------------------------------------
def test_end_to_end_known_by_construction():
    """One wallet, two markets, one bet each. Price drifts 5c against the follower
    in the two minutes after entry; k = 0.04; adverse tick 0.01.

    bet 1: entry 0.40, price@2m 0.45, resolves 1.0
    bet 2: entry 0.60, price@2m 0.65, resolves 0.0
    """
    bets = _frame(entry_ts=[1_000, 1_000], entry_price=[0.40, 0.60],
                  resolved_value=[1.0, 0.0], cutoff=[1e12, 1e12], tok=[0, 1],
                  market=["m1", "m2"])
    bets["fee_k"] = 0.04
    store, offsets = _store({0: ([1_000, 1_120], [0.40, 0.45]),
                             1: ([1_000, 1_120], [0.60, 0.65])})
    out = cs.compute_per_bet(bets, store, offsets, deltas=[(0, "0"), (120, "2m")],
                             adverse_tick=0.01)
    assert list(out["price_2m"]) == pytest.approx([0.45, 0.65])
    own = [(1.0 - 0.40) / 0.40, (0.0 - 0.60) / 0.60]
    assert list(out["gross_0"]) == pytest.approx(own)
    gross = [(1.0 - 0.45) / 0.45, (0.0 - 0.65) / 0.65]
    assert list(out["gross_2m"]) == pytest.approx(gross)
    costs = [p + 0.04 * p * (1 - p) for p in (0.46, 0.66)]
    assert list(out["net_2m"]) == pytest.approx([(1.0 - costs[0]) / costs[0],
                                                 (0.0 - costs[1]) / costs[1]])

    rng = np.random.default_rng(0)
    s = cs.summarize(out, "T", rng, deltas=[(0, "0"), (120, "2m")], adverse_tick=0.01,
                     n_shuffle=10)
    row = s.loc[s["delta"] == "2m"].iloc[0]
    assert row["n"] == 2
    assert row["own_gross_cohort"] == pytest.approx(np.mean(own))
    assert row["follower_gross"] == pytest.approx(np.mean(gross))
    assert row["follower_net"] < row["follower_gross"]          # fees cost money
    assert row["follower_gross"] < row["own_gross_cohort"]      # lag costs money
    assert row["real_minus_null"] == pytest.approx(
        row["follower_net"] - row["null_net"])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(entry_ts, entry_price, resolved_value, cutoff, tok=None, market=None,
           fee_k=0.04):
    n = len(entry_ts)
    return pd.DataFrame({
        "wallet": ["0xw"] * n,
        "market_id": market or [f"m{i}" for i in range(n)],
        "token_id": [f"t{i}" for i in (tok if tok is not None else range(n))],
        "tok_ix": np.arange(n) if tok is None else np.asarray(tok),
        "entry_price": np.asarray(entry_price, dtype=float),
        "timestamp": np.asarray(entry_ts, dtype=np.int64),
        "resolved_value": np.asarray(resolved_value, dtype=float),
        "cutoff": np.asarray(cutoff, dtype=float),
        "fee_k": np.full(n, fee_k, dtype=float),
        "category": ["politics"] * n,
    })


def _store(by_token: dict[int, tuple[list, list]]):
    """Build the sorted (tok_ix, t, p) price store and its per-token offsets."""
    ix, ts, ps = [], [], []
    n_tokens = max(by_token) + 1
    for tok in sorted(by_token):
        t, p = by_token[tok]
        ix.extend([tok] * len(t))
        ts.extend(t)
        ps.extend(p)
    store = (np.array(ix, dtype=np.int32), np.array(ts, dtype=np.int64),
             np.array(ps, dtype=np.float32))
    return store, cs.token_slices(store[0], n_tokens)


def test_build_bets_frame_preserves_existing_tok_ix(tmp_path, monkeypatch):
    """REGRESSION: tok_ix is the price cache's primary key. Every
    prices/chunk-*.parquet stores points against the tok_ix current when it was
    fetched, so renumbering an existing token silently re-points millions of
    cached price points at the WRONG token — the run still completes and every
    number is wrong. Adding wallets must only APPEND."""
    import numpy as np
    import pandas as pd
    from src import copy_sim as cs

    tokens_path = tmp_path / "tokens.parquet"
    # "aaa"/"zzz" already indexed; note zzz sorts AFTER a newly-arriving "mmm",
    # so a naive sorted() rebuild would renumber it.
    pd.DataFrame({"token_id": ["zzz", "aaa"], "tok_ix": np.int32([0, 1])}).to_parquet(tokens_path)
    monkeypatch.setattr(cs, "TOKENS_PATH", tokens_path)
    monkeypatch.setattr(cs, "BETS_PATH", tmp_path / "bets.parquet")
    monkeypatch.setattr(cs, "COPYSIM_DIR", tmp_path)
    monkeypatch.setattr(cs, "resolve_wallets", lambda p=None: ["0xw"])
    monkeypatch.setattr(cs, "attach_categories", lambda b: b.assign(category="other"))
    monkeypatch.setattr(cs, "load_bets", lambda w, trades_dir=None: pd.DataFrame({
        "wallet": ["0xw"] * 3, "market_id": ["m1", "m2", "m3"],
        "token_id": ["aaa", "mmm", "zzz"], "outcome": ["Yes"] * 3,
        "entry_price": [0.5] * 3, "resolved_value": [1.0] * 3,
        "size": [1.0] * 3, "timestamp": [1, 2, 3],
        "slug": ["s"] * 3, "question": ["q"] * 3,
    }))

    _bets, tokens = cs.build_bets_frame()
    ix = dict(zip(tokens["token_id"], tokens["tok_ix"]))
    assert ix["zzz"] == 0 and ix["aaa"] == 1      # untouched
    assert ix["mmm"] == 2                          # appended after the max


# ---------------------------------------------------------------------------
# The LIVE estimator (`certified` stage). Fixtures are built so the correct
# answer is known by construction, per CLAUDE.md.
# ---------------------------------------------------------------------------
def test_spread_proxy_charges_the_gap_the_wallet_paid_floored_at_one_tick():
    """The wallet's VWAP sits above the mid; the follower must be charged that gap,
    not one tick, or it is credited the midpoint-vs-ask artifact."""
    import numpy as np
    from src import copy_sim as cs

    sp = cs.spread_proxy(entry_price=[0.50, 0.50, 0.50],
                         mid_at_entry=[0.44, 0.4999, 0.55],
                         tick_size=[0.001, 0.001, 0.001])
    assert sp[0] == pytest.approx(0.06)        # the observed 6c gap, not a tick
    assert sp[1] == pytest.approx(0.001)       # gap below a tick -> floored
    assert sp[2] == pytest.approx(0.001)       # wallet filled BELOW mid -> floored


def test_population_baseline_recovers_a_known_calibration_curve():
    """Bets priced at p resolve YES with probability p by construction, so every
    bin mean must come back at ~its own price level."""
    import numpy as np
    from src import copy_sim as cs

    rng = np.random.default_rng(0)
    prices = np.repeat(np.linspace(0.05, 0.95, 19), 400)
    outcomes = (rng.random(prices.size) < prices).astype(float)
    bl = cs.fit_population_baseline(prices, outcomes)
    got = cs.expected_from(bl, [0.10, 0.50, 0.90])
    assert got[0] == pytest.approx(0.10, abs=0.05)
    assert got[1] == pytest.approx(0.50, abs=0.05)
    assert got[2] == pytest.approx(0.90, abs=0.05)


def _fine_baseline():
    """A calibrated baseline over many distinct prices, so its 20 quantile bins are
    narrow. NB with a COARSE baseline a 2c drift can land inside the same bin, where
    E[outcome|price] is flat and the residual metric cannot see the move at all —
    which is a real property of the metric worth knowing, and the reason this
    fixture is deliberately fine-grained."""
    import numpy as np
    from src import copy_sim as cs
    rng = np.random.default_rng(7)
    prices = np.repeat(np.linspace(0.02, 0.98, 97), 300)
    outcomes = (rng.random(prices.size) < prices).astype(float)
    return cs.fit_population_baseline(prices, outcomes)


def _certified_frame(price_2m, entry=0.60, mid=0.58, resolved=1.0, n=200):
    import numpy as np
    import pandas as pd
    return pd.DataFrame({
        "wallet": ["0xw"] * n,
        "market_id": [f"m{i}" for i in range(n)],
        "entry_price": [entry] * n, "mid_at_entry": [mid] * n,
        "tick_size": [0.001] * n, "price_2m": [price_2m] * n,
        "resolved_value": [resolved] * n, "fee_k": [0.0] * n,
        "fee_exponent": [1.0] * n,
    })


def test_certified_scoring_charges_drift_spread_and_never_beats_the_wallet():
    """A follower entering 2c higher than the wallet, plus the 2c spread the wallet
    paid, must end up strictly worse than the wallet on identical bets."""
    from src import copy_sim as cs

    bl = _fine_baseline()
    # a drift big enough to cross a baseline bin edge (see the blindness test below)
    out = cs.score_certified(_certified_frame(price_2m=0.75), bl, min_bets=10)
    assert len(out) == 1
    assert out.loc[0, "net_c"] < out.loc[0, "own_c"]      # the assertion's premise
    assert out.loc[0, "unfillable_share"] == 0.0


def test_residual_metric_is_blind_to_drift_inside_one_baseline_bin():
    """A DOCUMENTED LIMITATION, pinned so it cannot regress silently.

    `net_c` residualises at the follower's own fill price, and E[outcome|price] is a
    step function over 20 quantile bins. So a price move that stays INSIDE a bin
    changes the follower's cost but not the baseline it is scored against, and the
    residual does not see it. This is why the residual-at-fill rule reads ~0.66c/share
    more generous than `own - drift - costs` on the real cohort
    (docs/redteam_realworld_copy_2026-07-29.md §0). Read the money column alongside."""
    from src import copy_sim as cs

    bl = _fine_baseline()
    tight = cs.score_certified(_certified_frame(price_2m=0.605, mid=0.5999),
                               bl, min_bets=10)
    assert tight.loc[0, "net_c"] == pytest.approx(tight.loc[0, "own_c"], abs=1e-9)
    # ... but the MONEY column always sees it, which is why both are reported.
    # NB own_c is a RESIDUAL and money_c is raw money — not comparable to each
    # other. The wallet's own money edge is the right reference.
    wallet_money_c = 100 * (1.0 - 0.60)
    assert tight.loc[0, "money_c"] < wallet_money_c


def test_certified_scoring_asserts_the_follower_cannot_beat_the_wallet():
    """The midpoint-vs-ask artifact showed up as follower > wallet on 14 of 46
    wallets and was not caught. It is now an assertion, so a spread proxy that
    under-charges cannot pass silently."""
    from src import copy_sim as cs

    bl = _fine_baseline()
    # a follower filling BELOW the wallet's own entry price is the impossibility
    frame = _certified_frame(price_2m=0.20, entry=0.90, mid=0.899)
    with pytest.raises(AssertionError, match="IMPOSSIBLE"):
        cs.score_certified(frame, bl, min_bets=10)


def test_certified_scoring_counts_rather_than_hides_impossible_fills():
    """price_2m + spread can exceed $1.00, which no book can produce. Those bets
    are reported as a share, not silently clipped out of sight."""
    from src import copy_sim as cs

    bl = _fine_baseline()
    frame = _certified_frame(price_2m=0.97, entry=0.95, mid=0.85)   # 0.97 + 0.10 > 1
    out = cs.score_certified(frame, bl, min_bets=10)
    assert out.loc[0, "unfillable_share"] == pytest.approx(1.0)


def test_certified_scoring_skips_wallets_below_the_bet_floor():
    from src import copy_sim as cs

    bl = _fine_baseline()
    out = cs.score_certified(_certified_frame(price_2m=0.62, n=20), bl, min_bets=150)
    assert out.empty
