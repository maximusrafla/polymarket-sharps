"""Tests for scripts/audit_screen_surface.py — the screening-surface audit.

The script's job is to decide whether a screening rule beats its own null, so
every number it prints rests on four things being exactly right: the three-way
chronological cut, the segment sums that turn a 4M-row tape into per-wallet
window statistics, the rectangle search that finds the optimum, and the null
permutation that must move targets only WITHIN depth strata and only among
eligible wallets. Each is tested on a fixture where the answer is known by
construction, and the segment sums are additionally checked against a naive
Python loop on random data (the reduceat pairing is easy to get subtly wrong at
the array end).
"""

import numpy as np
import pytest

from scripts.audit_screen_surface import (
    DEPTH_THRESHOLDS,
    DETAIL_DEPTHS,
    REALWORLD_DEPTH_THRESHOLDS,
    REALWORLD_DETAIL_DEPTHS,
    REALWORLD_EVENT_FLOORS,
    UNCOPYABLE_CATEGORY,
    all_rectangles,
    best_rectangle,
    bin_by_edges,
    certify_on_window,
    cluster_bootstrap_ci,
    decile_profile,
    fee_fraction_of_stake,
    fee_rate_for_categories,
    depth_strata_codes,
    gate_passes,
    grid_accumulate,
    null_p_permutations,
    pooled_roi,
    quantile_edges,
    rect_sums,
    rect_window,
    screen_w3_null_p,
    size_matched_median,
    screen_w3_stats,
    segment_sums,
    set_population,
    three_way_bounds,
    tick_for_price,
    wallet_bootstrap_ci,
)


# --------------------------------------------------------------------------- #
# The three-way chronological cut                                              #
# --------------------------------------------------------------------------- #
def test_three_way_bounds_is_disjoint_and_exhaustive():
    starts = np.array([0, 7, 7, 20])
    counts = np.array([7, 0, 13, 1])
    b0, b1, b2, b3 = three_way_bounds(starts, counts)
    assert list(b0) == [0, 7, 7, 20]
    assert list(b3) == [7, 7, 20, 21]
    # windows never overlap and jointly cover the block
    assert np.all(b0 <= b1) and np.all(b1 <= b2) and np.all(b2 <= b3)
    assert np.all((b1 - b0) + (b2 - b1) + (b3 - b2) == counts)


def test_three_way_bounds_uses_floor_thirds():
    b0, b1, b2, b3 = three_way_bounds(np.array([0]), np.array([10]))
    # 10 bets -> 3 / 3 / 4: the remainder goes to the LAST window, matching
    # src.validate's floor cut which gives the held-out half the odd bet.
    assert (int(b1[0]), int(b2[0]), int(b3[0])) == (3, 6, 10)


def test_a_wallet_below_three_bets_gets_empty_early_windows():
    _, b1, b2, b3 = three_way_bounds(np.array([0]), np.array([2]))
    assert int(b1[0]) == 0 and int(b2[0]) == 1 and int(b3[0]) == 2


# --------------------------------------------------------------------------- #
# segment_sums                                                                 #
# --------------------------------------------------------------------------- #
def test_segment_sums_matches_a_naive_loop_including_the_array_end():
    rng = np.random.default_rng(7)
    values = rng.normal(size=97)
    lo = np.array([0, 5, 5, 40, 90, 97])
    hi = np.array([3, 5, 40, 41, 97, 97])
    got = segment_sums(values, lo, hi)
    want = np.array([values[a:b].sum() for a, b in zip(lo, hi)])
    assert np.allclose(got, want)


def test_segment_sums_empty_segments_are_zero_not_the_element_at_the_index():
    values = np.array([5.0, 7.0, 11.0])
    # raw np.add.reduceat would return values[1] == 7.0 for the empty segment
    assert segment_sums(values, np.array([1]), np.array([1]))[0] == 0.0


def test_segment_sums_handles_an_empty_value_array():
    assert list(segment_sums(np.array([]), np.array([0]), np.array([0]))) == [0.0]


# --------------------------------------------------------------------------- #
# Binning                                                                      #
# --------------------------------------------------------------------------- #
def test_quantile_edges_are_open_ended_so_nothing_falls_outside():
    edges = quantile_edges(np.arange(100.0), 10)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    b = bin_by_edges(np.array([-1e9, 50.0, 1e9]), edges)
    assert b.min() >= 0 and b.max() <= edges.size - 2


def test_quantile_edges_collapse_on_a_degenerate_axis():
    edges = quantile_edges(np.full(50, 0.3), 10)
    assert edges.size == 2
    assert list(bin_by_edges(np.array([0.3, 99.0]), edges)) == [0, 0]


def test_bin_by_edges_is_half_open_on_the_right():
    edges = np.array([-np.inf, 0.0, 1.0, np.inf])
    assert list(bin_by_edges(np.array([-0.1, 0.0, 0.999, 1.0]), edges)) == [0, 1, 1, 2]


# --------------------------------------------------------------------------- #
# The rectangle search                                                         #
# --------------------------------------------------------------------------- #
def test_rect_window_recovers_any_sub_rectangle_sum():
    rng = np.random.default_rng(3)
    grid = rng.normal(size=(4, 5))
    p = rect_sums(grid)
    for r0 in range(4):
        for r1 in range(r0 + 1, 5):
            for c0 in range(5):
                for c1 in range(c0 + 1, 6):
                    assert np.isclose(rect_window(p, r0, r1, c0, c1),
                                      grid[r0:r1, c0:c1].sum())


def test_all_rectangles_enumerates_every_axis_aligned_box():
    r0, r1, c0, c1 = all_rectangles(3, 4)
    assert r0.size == 6 * 10                            # C(4,2) * C(5,2)
    assert np.all(r0 < r1) and np.all(c0 < c1)
    assert len({(a, b, c, e) for a, b, c, e in zip(r0, r1, c0, c1)}) == r0.size


def test_best_rectangle_finds_the_planted_optimum():
    # one 2x2 block earns 50% on $100 a cell; everything else is flat at 0%. The
    # wallet floor is set to the block's own size so that no strict sub-rectangle
    # (which would tie on ROI) can win it on enumeration order.
    profit = np.zeros((4, 4))
    spend = np.full((4, 4), 100.0)
    count = np.full((4, 4), 20.0)
    profit[2:4, 1:3] = 50.0
    rects = all_rectangles(4, 4)
    roi, rect, k, prof, sp = best_rectangle(profit, spend, count, rects, min_wallets=80)
    assert rect == (2, 4, 1, 3)
    assert np.isclose(roi, 0.5) and k == 80 and np.isclose(prof, 200.0) and np.isclose(sp, 400.0)


def test_best_rectangle_honours_the_wallet_floor():
    profit = np.zeros((4, 4))
    spend = np.full((4, 4), 10.0)
    count = np.full((4, 4), 1.0)
    profit[0, 0] = 3.0          # +30% on one lucky wallet, in a far corner
    profit[2:4, 2:4] = 1.0      # +10% on a diversified block of four
    rects = all_rectangles(4, 4)
    _, rect_small, _, _, _ = best_rectangle(profit, spend, count, rects, min_wallets=1)
    _, rect_big, _, _, _ = best_rectangle(profit, spend, count, rects, min_wallets=4)
    assert rect_small == (0, 1, 0, 1)
    assert rect_big == (2, 4, 2, 4)


def test_best_rectangle_honours_the_spend_floor():
    profit = np.zeros((2, 2))
    spend = np.array([[1.0, 100.0], [100.0, 100.0]])
    count = np.full((2, 2), 50.0)
    profit[0, 0] = 0.9          # +90% on one dollar
    profit[1, 1] = 5.0          # +5% on a hundred
    rects = all_rectangles(2, 2)
    _, loose, _, _, _ = best_rectangle(profit, spend, count, rects, 1, min_spend=0.0)
    _, tight, _, _, _ = best_rectangle(profit, spend, count, rects, 1, min_spend=50.0)
    assert loose == (0, 1, 0, 1)
    assert tight == (1, 2, 1, 2)


def test_best_rectangle_returns_nothing_when_no_rectangle_qualifies():
    grid = np.zeros((2, 2))
    roi, rect, k, prof, sp = best_rectangle(grid, grid, grid, all_rectangles(2, 2), 1)
    assert rect is None and k == 0 and np.isnan(roi)


def test_grid_accumulate_sums_duplicates_into_one_cell():
    got = grid_accumulate(np.array([0, 0, 1]), np.array([1, 1, 0]),
                          np.array([2.0, 3.0, 7.0]), (2, 2))
    assert got.tolist() == [[0.0, 5.0], [7.0, 0.0]]


# --------------------------------------------------------------------------- #
# Dollar statistics                                                            #
# --------------------------------------------------------------------------- #
def test_pooled_roi_is_dollar_weighted_not_an_average_of_ratios():
    # a $1 bet that doubles and a $99 bet that loses 1%: the naive mean of the
    # two ROIs is +49.5%, the dollar-weighted truth is +0.01%.
    profit = np.array([1.0, -0.99])
    spend = np.array([1.0, 99.0])
    assert np.isclose(pooled_roi(profit, spend), 0.01 / 100.0)
    assert np.isnan(pooled_roi(np.array([1.0]), np.array([0.0])))


def test_cluster_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(11)
    profit = rng.normal(5.0, 1.0, size=200)
    spend = np.full(200, 100.0)
    lo, hi = cluster_bootstrap_ci(profit, spend, rng, 500)
    assert lo < pooled_roi(profit, spend) < hi


def test_cluster_bootstrap_ci_refuses_a_single_cluster():
    rng = np.random.default_rng(1)
    lo, hi = cluster_bootstrap_ci(np.array([1.0]), np.array([10.0]), rng, 100)
    assert np.isnan(lo) and np.isnan(hi)


def test_tick_is_per_price_band_not_a_constant():
    got = tick_for_price(np.array([0.02, 0.5, 0.97]))
    assert got.tolist() == [0.001, 0.01, 0.001]


# --------------------------------------------------------------------------- #
# The decile screen                                                            #
# --------------------------------------------------------------------------- #
def test_decile_profile_reads_the_planted_gradient():
    n = 100
    w1 = {"roi": np.linspace(-0.5, 0.5, n)}
    # target ROI equals +10% for the top tenth, 0 elsewhere
    spend = np.full(n, 100.0)
    profit = np.where(np.arange(n) >= 90, 10.0, 0.0)
    pool = np.ones(n, dtype=bool)
    prof = decile_profile(w1, {"profit": profit, "spend": spend}, pool, "roi")
    assert np.isclose(prof["roi"][-1], 0.10)
    assert np.isclose(prof["roi"][0], 0.0)
    assert np.isclose(prof["pool_roi"], 0.01)
    assert np.isclose(prof["top_minus_pool"], 0.09)
    assert np.isclose(prof["top_minus_bottom"], 0.10)


# --------------------------------------------------------------------------- #
# Null P                                                                       #
# --------------------------------------------------------------------------- #
def test_null_p_permutes_only_eligible_wallets_and_only_within_depth_strata():
    n = 60
    depths = np.repeat([6, 60, 600], 20)          # three clearly separate strata
    eligible = np.ones(n, dtype=bool)
    eligible[::10] = False                        # a few ineligible wallets
    sp = {"w1": {"n": depths}, "eligible": eligible}
    strata = depth_strata_codes(depths)
    rng = np.random.default_rng(5)
    perms = list(null_p_permutations(sp, rng, 8))
    assert len(perms) == 8
    for perm in perms:
        assert np.all(perm[~eligible] == np.flatnonzero(~eligible))   # untouched
        assert np.all(strata[perm] == strata)                         # stratum-preserving
        assert sorted(perm.tolist()) == list(range(n))                # a permutation
    assert any(not np.array_equal(p, np.arange(n)) for p in perms)     # and not the identity


# --------------------------------------------------------------------------- #
# The in-window certification gate                                             #
# --------------------------------------------------------------------------- #
def _cert_fixture(n_per=80):
    """Two wallets: one that beats its price everywhere, one that is exactly fair.
    Prices are 0.5 and the baseline expectation is 0.5, so the residual is the
    whole signal and the gate's verdict is known by construction."""
    rng = np.random.default_rng(2)
    value = np.concatenate([rng.random(n_per) < 0.75, rng.random(n_per) < 0.50])
    d = {
        "value": value.astype(float),
        "resid_bet": value.astype(float) - 0.5,
        "ecode": np.concatenate([np.arange(n_per), np.arange(n_per)]),
        "n_wallets": 2,
    }
    lo = np.array([0, n_per])
    hi = np.array([n_per, 2 * n_per])
    return d, lo, hi


def test_certify_on_window_separates_a_sharp_wallet_from_a_fair_one():
    d, lo, hi = _cert_fixture()
    passes, diag = certify_on_window(d, lo, hi, np.ones(2, dtype=bool),
                                     (0.05,), 0.02, 5, 500, seed=1)
    assert passes[0.05][0] and not passes[0.05][1]
    assert diag["held_edge"][0] > diag["held_edge"][1]


def test_certify_on_window_alpha_one_skips_the_significance_test():
    d, lo, hi = _cert_fixture()
    passes, _ = certify_on_window(d, lo, hi, np.ones(2, dtype=bool),
                                  (0.005, 1.0), 0.02, 5, 300, seed=1)
    # alpha=1.0 is the pure magnitude+breadth screen, so it can only be wider
    assert passes[1.0][passes[0.005]].all()
    assert passes[1.0].sum() >= passes[0.005].sum()


def test_certify_on_window_event_floor_bites():
    d, lo, hi = _cert_fixture()
    passes, _ = certify_on_window(d, lo, hi, np.ones(2, dtype=bool),
                                  (0.05,), 0.02, 10_000, 300, seed=1)
    assert passes[0.05].sum() == 0


@pytest.mark.parametrize("n", [0, 1, 3])
def test_certify_on_window_ignores_wallets_too_short_to_split(n):
    d = {"value": np.ones(n), "resid_bet": np.ones(n),
         "ecode": np.arange(n), "n_wallets": 1}
    passes, _ = certify_on_window(d, np.array([0]), np.array([n]),
                                  np.ones(1, dtype=bool), (0.05,), 0.0, 1, 50, seed=1)
    assert passes[0.05].sum() == 0


# --------------------------------------------------------------------------- #
# The real-world restriction and its recalibrated ladders                       #
# --------------------------------------------------------------------------- #
def test_gate_passes_reproduces_certify_on_window_at_the_floor_it_was_run_at():
    d, lo, hi = _cert_fixture()
    passes, diag = certify_on_window(d, lo, hi, np.ones(2, dtype=bool),
                                     (0.005, 0.05, 1.0), 0.02, 5, 500, seed=1)
    for a in (0.005, 0.05, 1.0):
        assert np.array_equal(gate_passes(diag, 0.02, 5, a), passes[a])


def test_gate_passes_event_floor_only_ever_narrows_the_set():
    """The whole point of sweeping the floor: raising it can only remove wallets,
    so one bootstrap run at the lowest floor answers the entire ladder."""
    d, lo, hi = _cert_fixture()
    _, diag = certify_on_window(d, lo, hi, np.ones(2, dtype=bool),
                                (0.05,), 0.02, 3, 500, seed=1)
    prev = gate_passes(diag, 0.02, 3, 0.05)
    for ev in (5, 10, 10_000):
        cur = gate_passes(diag, 0.02, ev, 0.05)
        assert cur[cur].size == 0 or prev[cur].all()
        prev = cur
    assert gate_passes(diag, 0.02, 10_000, 0.05).sum() == 0


def test_gate_passes_refuses_a_diag_computed_at_a_higher_floor():
    """Guards the one way this shortcut could lie: a diag from a run with a HIGHER
    floor never computed `p` for the wallets a lower floor would admit, so it
    would silently under-certify rather than error."""
    d, lo, hi = _cert_fixture()
    _, diag = certify_on_window(d, lo, hi, np.ones(2, dtype=bool),
                                (0.05,), 0.02, 20, 300, seed=1)
    with pytest.raises(ValueError):
        gate_passes(diag, 0.02, 5, 0.05)


def test_set_population_swaps_both_ladders_and_restores_them():
    # read through the module, not through a from-import: `set_population` rebinds
    # module globals, which a from-imported name would not see.
    import scripts.audit_screen_surface as mod

    try:
        set_population(True)
        assert mod.DEPTHS == REALWORLD_DEPTH_THRESHOLDS
        assert mod.DETAILS == REALWORLD_DETAIL_DEPTHS
        set_population(False)
        assert mod.DEPTHS == DEPTH_THRESHOLDS
        assert mod.DETAILS == DETAIL_DEPTHS
    finally:
        set_population(False)


def test_the_real_world_ladder_reaches_shallower_than_the_unrestricted_one():
    """A real-world wallet placing 30 bets is a serious trader; a 5-minute-tape bot
    does that in an hour. If the ladders were the same the real-world arm would be
    asking about depth strata that do not exist in its population."""
    assert min(REALWORLD_DEPTH_THRESHOLDS) < min(DEPTH_THRESHOLDS)
    assert max(REALWORLD_DEPTH_THRESHOLDS) < max(DEPTH_THRESHOLDS)
    assert min(REALWORLD_EVENT_FLOORS) < 30      # the production floor is swept, not assumed
    assert 30 in REALWORLD_EVENT_FLOORS          # ...and the production floor is still on it


def test_screen_w3_stats_reports_all_three_weightings_and_the_sample_floor():
    """Pooled follows the big stake, the equal-weight mean does not, and the
    min-target floor drops the wallet with too few W3 bets — planted so all three
    disagree by construction."""
    w3 = {"profit": np.array([100.0, -1.0, 999.0]),
          "spend": np.array([1000.0, 10.0, 1.0]),
          "n": np.array([9, 9, 1])}
    sel = np.ones(3, dtype=bool)
    pooled, eqw, med, k, stake = screen_w3_stats(sel, w3, min_target=5)
    assert k == 2 and stake == pytest.approx(1010.0)
    assert pooled == pytest.approx(99.0 / 1010.0)          # the $1000 wallet dominates
    assert eqw == pytest.approx((0.10 + -0.10) / 2)        # ...and here it does not
    assert med == pytest.approx(0.0)


def test_screen_w3_null_p_holds_the_screen_and_redraws_only_the_linkage():
    """An identity permutation must reproduce the real statistic exactly, and a
    screen that is genuinely the best wallets must beat random re-linkings."""
    rng = np.random.default_rng(0)
    n = 60
    profit = np.concatenate([np.full(10, 50.0), rng.normal(0, 1, n - 10)])
    w3 = {"profit": profit, "spend": np.full(n, 100.0), "n": np.full(n, 9)}
    sel = np.zeros(n, dtype=bool)
    sel[:10] = True                                   # the screen picks the winners
    ident = [np.arange(n)]
    real, ps = screen_w3_null_p(sel, w3, ident)
    assert ps == [pytest.approx(1.0)] * 3             # identity perm ties everything
    perms = [rng.permutation(n) for _ in range(99)]
    real2, ps2 = screen_w3_null_p(sel, w3, perms)
    assert real2 == real                              # the real stat cannot move
    assert all(p < 0.05 for p in ps2)


def test_screen_w3_stats_honours_the_profit_key_so_costs_reach_the_median():
    """A per-wallet statistic must be chargeable for costs, or the equal-weight and
    median views quietly report a gross number while the pooled one pays."""
    w3 = {"profit": np.array([10.0, 10.0]), "net_profit": np.array([-5.0, -5.0]),
          "spend": np.array([100.0, 100.0]), "n": np.array([9, 9])}
    sel = np.ones(2, dtype=bool)
    assert screen_w3_stats(sel, w3)[2] == pytest.approx(0.10)
    assert screen_w3_stats(sel, w3, profit_key="net_profit")[2] == pytest.approx(-0.05)


def test_size_matched_median_reprices_a_pure_size_effect_to_zero_excess():
    """Plant a world where ROI depends ONLY on wallet size and the screen picks the
    small wallets: the raw median then looks great and the excess must be zero."""
    small_r, big_r = 0.20, -0.20
    w3 = {"spend": np.concatenate([np.full(20, 10.0), np.full(20, 1e6)]),
          "profit": np.concatenate([np.full(20, 10.0 * small_r),
                                    np.full(20, 1e6 * big_r)]),
          "n": np.full(40, 9)}
    pool = np.ones(40, dtype=bool)
    sel = np.zeros(40, dtype=bool)
    sel[:20] = True                                   # the screen takes small books
    assert screen_w3_stats(sel, w3)[2] == pytest.approx(small_r)
    assert size_matched_median(sel, pool, w3) == pytest.approx(small_r)


def test_size_matched_median_keeps_a_real_effect_inside_a_size_bin():
    """The other direction: if the screen beats the pool WITHIN its own size band,
    the index must not explain that away."""
    rng = np.random.default_rng(3)
    n = 200
    spend = 10.0 ** rng.uniform(1, 6, n)
    roi = rng.normal(0.0, 0.01, n)
    sel = np.zeros(n, dtype=bool)
    sel[rng.choice(n, 40, replace=False)] = True      # size-blind pick...
    roi[sel] += 0.5                                   # ...that genuinely earns more
    w3 = {"spend": spend, "profit": roi * spend, "n": np.full(n, 9)}
    excess = screen_w3_stats(sel, w3)[2] - size_matched_median(sel, np.ones(n, bool), w3)
    assert excess > 0.4


def test_screen_w3_null_p_finds_nothing_when_the_screen_is_arbitrary():
    """The other direction: a screen unrelated to the target must NOT be
    significant, or the test has no specificity."""
    rng = np.random.default_rng(7)
    n = 80
    w3 = {"profit": rng.normal(0, 1, n), "spend": np.full(n, 100.0),
          "n": np.full(n, 9)}
    sel = np.zeros(n, dtype=bool)
    sel[::4] = True
    _, ps = screen_w3_null_p(sel, w3, [rng.permutation(n) for _ in range(199)])
    assert all(0.0 < p <= 1.0 for p in ps)
    assert max(ps) > 0.05


def test_is_real_world_matches_the_category_the_other_arms_exclude():
    """`--real-world` must mean exactly what the sports / slow-forecaster /
    edge-decay arms mean by it, or the results are not comparable to them."""
    from src.discover import is_real_world

    assert not is_real_world(UNCOPYABLE_CATEGORY)
    assert UNCOPYABLE_CATEGORY == "micro_crypto"
    for cat in ("sports_nfl", "politics", "econ_macro", "culture", "crypto_event",
                "other"):
        assert is_real_world(cat)


# --------------------------------------------------------------------------- #
# The real fee model (docs/polymarket_mechanics.md §5)                          #
# --------------------------------------------------------------------------- #
def test_fee_rate_maps_the_live_categories_and_defaults_to_sports_general():
    got = fee_rate_for_categories(
        ["micro_crypto", "crypto_event", "politics", "econ_macro", "culture",
         "other", "sports_nfl", "sports_other", "a-label-that-does-not-exist"])
    assert list(got) == [0.07, 0.07, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]


def test_fee_fraction_of_stake_is_rate_times_one_minus_price():
    # The on-chain fee is `shares * rate * p * (1-p)` and the stake is `shares * p`,
    # so as a FRACTION of stake the shares and one power of p cancel. This is the
    # sign flip that matters: the tax is heaviest on cheap longshots.
    rate, price = 0.05, np.array([0.05, 0.50, 0.95])
    frac = fee_fraction_of_stake(np.full(3, rate), price)
    assert frac == pytest.approx([0.0475, 0.025, 0.0025])
    assert frac[0] > frac[1] > frac[2]


def test_fee_fraction_reproduces_the_on_chain_worked_example():
    # docs/polymarket_mechanics.md: tx 0x9ddaccdf…82343d, btc-updown-5m, rate 0.07,
    # shares 580.677591 @ 0.527058567 -> fee 10.132090 USDC, exact to 7 s.f.
    # The doc quotes the CHARGED fee, which the venue floors to 5 dp; the raw
    # product is 10.1320971..., so 5-dp truncation is what reproduces the receipt.
    shares, price, rate = 580.677591, 0.527058567, 0.07
    fee = float(fee_fraction_of_stake(rate, price)) * shares * price
    assert fee == pytest.approx(10.132097, abs=1e-6)
    assert np.floor(fee * 1e5) / 1e5 == pytest.approx(10.13209, abs=1e-9)


def test_wallet_bootstrap_ci_widens_when_one_wallet_carries_the_book():
    rng = np.random.default_rng(0)
    # same pooled ROI both ways; the concentrated book has all its profit in one
    # wallet, so a wallet resample must be far less certain about it
    spread_p, spread_s = np.full(20, 5.0), np.full(20, 100.0)
    conc_p, conc_s = np.array([100.0] + [0.0] * 19), np.full(20, 100.0)
    assert pooled_roi(spread_p, spread_s) == pytest.approx(pooled_roi(conc_p, conc_s))
    a_lo, a_hi = wallet_bootstrap_ci(spread_p, spread_s, rng, 2000)
    b_lo, b_hi = wallet_bootstrap_ci(conc_p, conc_s, rng, 2000)
    assert (b_hi - b_lo) > 5 * (a_hi - a_lo)
    assert a_lo > 0 and b_lo == 0.0


def test_price_bands_cover_the_whole_unit_interval_without_gaps():
    from scripts.audit_screen_surface import PRICE_BANDS
    edges = np.array(PRICE_BANDS)
    assert edges[0] == 0.0 and edges[-1] == 1.0
    assert np.all(np.diff(edges) > 0)
    # every legal entry price lands in a band, including the clamped extremes
    b = bin_by_edges(np.array([0.001, 0.5, 0.999]), edges)
    assert list(b) == [0, 3, 5]
