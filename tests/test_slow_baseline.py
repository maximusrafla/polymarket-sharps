"""Tests for the hierarchical leave-one-wallet-out baseline (src/slow_baseline.py).

Every fixture is hand-built so the correct shrunk / LOWO value is known by
construction and asserted exactly. The three properties that decide whether the
finer-baseline re-check is interpretable at all get their own tests:

  1. partial pooling really is (n*ybar + k*parent)/(n+k);
  2. a thin cell takes its parent's value (no fitting on noise);
  3. leave-one-wallet-out really excludes the wallet — the guard against the
     false negative where a wallet dominating a niche residualizes against
     itself and collapses to zero by construction.
"""

import numpy as np
import pandas as pd
import pytest

from src.slow_baseline import (
    DEFAULT_MIN_BIN,
    assign_bins,
    fit_bin_edges,
    fit_hierarchical_residuals,
)


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["wallet", "category", "niche_l1",
                                       "entry_price", "resolved_value"])


# ---------------------------------------------------------------------------
# bins
# ---------------------------------------------------------------------------

def test_bin_edges_are_quantiles_and_shared():
    prices = np.linspace(0.0, 1.0, 1000)
    edges = fit_bin_edges(prices, n_bins=10)
    assert edges.size == 11
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(1.0)


def test_bin_edges_degenerate_input():
    assert fit_bin_edges([0.5] * 100, n_bins=10).size == 0
    assert fit_bin_edges([], n_bins=10).size == 0


def test_assign_bins_clamps_outside_range():
    edges = np.array([0.2, 0.4, 0.6])
    got = assign_bins([0.0, 0.3, 0.5, 1.0], edges)
    assert list(got) == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# partial pooling — the exact shrinkage identity
# ---------------------------------------------------------------------------

def test_shrinkage_is_exact_weighted_blend():
    """One price bin, one category, two niches. With k pinned, the fine estimate
    must be exactly (n*ybar + k*parent)/(n+k).

    Category has 200 bets, 100 of them 1 -> parent = 0.50.
    Niche A has 100 bets, 80 of them 1  -> own ybar = 0.80.
    With k = 100:  (100*0.80 + 100*0.50) / (100 + 100) = 0.65.
    """
    rows = []
    for i in range(100):                       # niche A: 80 ones
        rows.append(["wA%d" % i, "c", "A", 0.5, 1.0 if i < 80 else 0.0])
    for i in range(100):                       # niche B: 20 ones
        rows.append(["wB%d" % i, "c", "B", 0.5, 1.0 if i < 20 else 0.0])
    df = _frame(rows)

    out, info = fit_hierarchical_residuals(
        df, levels=("category", "niche_l1"), n_bins=2, min_bin=1,
        lowo=False, ks={"category": 0.0, "niche_l1": 100.0})

    a = out[out["niche_l1"] == "A"]["expected_outcome"].unique()
    b = out[out["niche_l1"] == "B"]["expected_outcome"].unique()
    assert a.size == 1 and b.size == 1
    assert a[0] == pytest.approx(0.65)
    assert b[0] == pytest.approx(0.35)         # (100*0.20 + 100*0.50)/200
    assert info.ks["niche_l1"] == 100.0


def test_larger_k_pools_harder():
    rows = ([["w%d" % i, "c", "A", 0.5, 1.0] for i in range(50)]
            + [["v%d" % i, "c", "B", 0.5, 0.0] for i in range(50)])
    df = _frame(rows)
    kw = dict(levels=("category", "niche_l1"), n_bins=2, min_bin=1, lowo=False)

    loose, _ = fit_hierarchical_residuals(df, ks={"category": 0.0, "niche_l1": 1.0}, **kw)
    tight, _ = fit_hierarchical_residuals(df, ks={"category": 0.0, "niche_l1": 10_000.0}, **kw)

    a_loose = loose[loose["niche_l1"] == "A"]["expected_outcome"].iloc[0]
    a_tight = tight[tight["niche_l1"] == "A"]["expected_outcome"].iloc[0]
    assert a_loose > a_tight                    # loose trusts the niche's own 1.0
    assert a_tight == pytest.approx(0.5, abs=0.02)   # tight collapses to the parent


# ---------------------------------------------------------------------------
# the min-bin floor — thin cells are never fit on noise
# ---------------------------------------------------------------------------

def test_thin_cell_takes_parent_value_exactly():
    """Niche B has 3 bets, all winners. With min_bin=50 it must NOT earn a 1.0
    estimate; it must take the parent's value exactly — the false-negative guard."""
    rows = ([["w%d" % i, "c", "A", 0.5, 1.0 if i < 60 else 0.0] for i in range(120)]
            + [["z%d" % i, "c", "B", 0.5, 1.0] for i in range(3)])
    df = _frame(rows)
    out, info = fit_hierarchical_residuals(
        df, levels=("category", "niche_l1"), n_bins=2, min_bin=50, lowo=False,
        ks={"category": 0.0, "niche_l1": 10.0})

    parent = out[out["niche_l1"] == "A"]["expected_outcome"]
    b = out[out["niche_l1"] == "B"]["expected_outcome"]
    # category mean = 63/123; B is pooled to it, NOT to its own 1.0
    assert b.nunique() == 1
    assert b.iloc[0] == pytest.approx(63.0 / 123.0)
    assert b.iloc[0] < 0.99
    # ...while A, which cleared the floor, did earn its own (shrunk) value
    assert parent.iloc[0] != pytest.approx(63.0 / 123.0)


def test_pooling_report_shape():
    rows = [["w%d" % i, "c", "A" if i % 2 else "B", 0.5, float(i % 2)] for i in range(200)]
    out, info = fit_hierarchical_residuals(_frame(rows), levels=("category", "niche_l1"),
                                           n_bins=2, min_bin=10, lowo=False)
    rep = info.pooling_report()
    assert set(rep["level"]) == {"_global", "category", "niche_l1"}
    assert (rep["bets_own_share"] <= 1.0).all()


# ---------------------------------------------------------------------------
# leave-one-wallet-out — the safeguard that makes a "collapse" verdict readable
# ---------------------------------------------------------------------------

def test_lowo_excludes_the_wallets_own_bets():
    """Niche A: 100 bets in one price bin. Wallet `big` owns 50 of them, all
    winners; 50 others are split 30 losers / 20 winners.

    Without LOWO the niche mean is 70/100 = 0.70 and `big`'s residual is only
    0.30. With LOWO the mean `big` faces is (70-50)/(100-50) = 0.40, so its
    residual is 0.60 — the honest number. Pinning k=0 and min_bin=1 isolates
    the exclusion arithmetic from the shrinkage.
    """
    rows = ([["big", "c", "A", 0.5, 1.0] for _ in range(50)]
            + [["o%d" % i, "c", "A", 0.5, 0.0] for i in range(30)]
            + [["p%d" % i, "c", "A", 0.5, 1.0] for i in range(20)])
    df = _frame(rows)
    kw = dict(levels=("category", "niche_l1"), n_bins=2, min_bin=1,
              ks={"category": 0.0, "niche_l1": 0.0})

    plain, _ = fit_hierarchical_residuals(df, lowo=False, **kw)
    loo, _ = fit_hierarchical_residuals(df, lowo=True, **kw)

    big_plain = plain[plain["wallet"] == "big"]
    big_loo = loo[loo["wallet"] == "big"]
    assert big_plain["expected_outcome"].iloc[0] == pytest.approx(0.70)
    assert big_plain["residual_skill"].mean() == pytest.approx(0.30)
    assert big_loo["expected_outcome"].iloc[0] == pytest.approx(0.40)
    assert big_loo["residual_skill"].mean() == pytest.approx(0.60)


def test_lowo_is_a_no_op_for_a_negligible_wallet():
    """The property that justifies the existing market-wide baseline ignoring
    LOWO: a wallet holding 1 of 1000 bets barely moves its own baseline."""
    rows = [["w%d" % i, "c", "A", 0.5, 1.0 if i < 500 else 0.0] for i in range(1000)]
    df = _frame(rows)
    kw = dict(levels=("category", "niche_l1"), n_bins=2, min_bin=1,
              ks={"category": 0.0, "niche_l1": 0.0})
    plain, _ = fit_hierarchical_residuals(df, lowo=False, **kw)
    loo, _ = fit_hierarchical_residuals(df, lowo=True, **kw)
    assert abs(plain["expected_outcome"].iloc[0] - loo["expected_outcome"].iloc[0]) < 0.002


def test_lowo_recurses_up_the_parent_chain():
    """A wallet that dominates BOTH its niche and its whole category must be
    excluded at both levels, not just the finest one."""
    rows = ([["big", "c", "A", 0.5, 1.0] for _ in range(80)]
            + [["o%d" % i, "c", "A", 0.5, 0.0] for i in range(20)])
    df = _frame(rows)
    out, _ = fit_hierarchical_residuals(
        df, levels=("category", "niche_l1"), n_bins=2, min_bin=1, lowo=True,
        ks={"category": 0.0, "niche_l1": 0.0})
    # excluding `big` entirely, every remaining bet in c/A is a loser -> 0.0
    assert out[out["wallet"] == "big"]["expected_outcome"].iloc[0] == pytest.approx(0.0)
    assert out[out["wallet"] == "big"]["residual_skill"].mean() == pytest.approx(1.0)


def test_sole_occupant_of_a_cell_falls_back_to_parent_not_nan():
    """If excluding the wallet empties its cell, the estimate must degrade to the
    parent rather than produce NaN or a divide-by-zero."""
    rows = ([["solo", "c", "A", 0.5, 1.0]]
            + [["o%d" % i, "c", "B", 0.5, 0.0] for i in range(60)])
    df = _frame(rows)
    out, _ = fit_hierarchical_residuals(df, levels=("category", "niche_l1"),
                                        n_bins=2, min_bin=1, lowo=True)
    v = out[out["wallet"] == "solo"]["expected_outcome"].iloc[0]
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# k estimation and invariants
# ---------------------------------------------------------------------------

def test_estimated_k_pools_hard_when_niches_do_not_differ():
    """All niches share one true rate: between-niche variance is ~0, so the
    estimator should choose a large k (pool hard) rather than chase noise."""
    rng = np.random.default_rng(0)
    rows = []
    for niche in range(30):
        for i in range(200):
            rows.append(["w%d_%d" % (niche, i), "c", "n%d" % niche, 0.5,
                         float(rng.random() < 0.5)])
    out, info = fit_hierarchical_residuals(_frame(rows), levels=("category", "niche_l1"),
                                           n_bins=2, min_bin=10, lowo=False)
    assert info.ks["niche_l1"] > 500


def test_estimated_k_trusts_niches_when_they_genuinely_differ():
    """Niches have wildly different true rates: the estimator should choose a
    small k so each niche is allowed its own value."""
    rows = []
    for niche in range(30):
        rate = 0.05 if niche % 2 else 0.95
        for i in range(200):
            rows.append(["w%d_%d" % (niche, i), "c", "n%d" % niche, 0.5,
                         float(i < 200 * rate)])
    out, info = fit_hierarchical_residuals(_frame(rows), levels=("category", "niche_l1"),
                                           n_bins=2, min_bin=10, lowo=False)
    assert info.ks["niche_l1"] < 10


def test_residual_is_outcome_minus_expectation_and_frame_is_additive():
    rows = [["w%d" % i, "c", "A", 0.1 + 0.8 * (i % 5) / 5, float(i % 2)] for i in range(300)]
    src = _frame(rows)
    out, _ = fit_hierarchical_residuals(src, levels=("category", "niche_l1"))
    assert len(out) == len(src)
    for c in src.columns:
        assert list(out[c]) == list(src[c])
    assert np.allclose(out["residual_skill"],
                       out["resolved_value"] - out["expected_outcome"])


def test_nan_price_gets_nan_expectation_not_a_clamped_guess():
    rows = [["w%d" % i, "c", "A", 0.5, float(i % 2)] for i in range(100)]
    df = _frame(rows)
    df.loc[0, "entry_price"] = np.nan
    out, _ = fit_hierarchical_residuals(df, levels=("category", "niche_l1"), min_bin=1)
    assert np.isnan(out.loc[0, "expected_outcome"])
    assert np.isnan(out.loc[0, "residual_skill"])
    assert out["expected_outcome"].iloc[1:].notna().all()


def test_empty_frame():
    out, info = fit_hierarchical_residuals(_frame([]))
    assert len(out) == 0
    assert "residual_skill" in out.columns
    assert info.n_bets == 0


def test_missing_level_column_is_skipped_not_fatal():
    rows = [["w%d" % i, "c", "A", 0.5, float(i % 2)] for i in range(100)]
    out, info = fit_hierarchical_residuals(
        _frame(rows), levels=("category", "niche_l1", "niche_l2"), min_bin=1)
    assert "niche_l2" not in info.ks
    assert out["expected_outcome"].notna().all()


def test_default_min_bin_is_conservative():
    assert DEFAULT_MIN_BIN >= 30


# ---------------------------------------------------------------------------
# Persistable fit / apply split (step 7c) — forward scoring
# ---------------------------------------------------------------------------

import json  # noqa: E402

from src.slow_baseline import (  # noqa: E402
    expected_outcome_hier,
    fit_hierarchical_baseline,
)


def test_fitted_baseline_is_json_serialisable():
    """The freeze manifest carries it, so a re-score months later needs no refit
    and no access to the original tape."""
    rows = [["w%d" % i, "c", "A" if i % 2 else "B", 0.5, float(i % 2)] for i in range(400)]
    bl = fit_hierarchical_baseline(_frame(rows), levels=("category", "niche_l1"), min_bin=10)
    s = json.dumps(bl)
    assert json.loads(s)["levels"] == ["category", "niche_l1"]


def test_apply_reproduces_the_in_sample_fit_when_lowo_is_off():
    """Same data, same answer: the split is a re-plumbing, not a different model."""
    rows = []
    for niche in range(6):
        for i in range(200):
            rows.append(["w%d_%d" % (niche, i), "c", "n%d" % niche,
                         0.05 + 0.9 * (i % 10) / 10.0, float(i % 3 == 0)])
    df = _frame(rows)
    fused, _ = fit_hierarchical_residuals(df, levels=("category", "niche_l1"),
                                          lowo=False, min_bin=50)
    bl = fit_hierarchical_baseline(df, levels=("category", "niche_l1"), min_bin=50)
    applied = expected_outcome_hier(bl, df)
    assert np.allclose(fused["expected_outcome"].to_numpy(), applied, atol=1e-9)


def test_apply_to_unseen_rows_uses_the_frozen_curve():
    rows = [["w%d" % i, "c", "A", 0.5, 1.0 if i < 150 else 0.0] for i in range(200)]
    bl = fit_hierarchical_baseline(_frame(rows), levels=("category", "niche_l1"),
                                   min_bin=10)
    fresh = _frame([["new_wallet", "c", "A", 0.5, 0.0]])
    got = expected_outcome_hier(bl, fresh)
    # the frozen A@bin estimate is ~0.75 and is applied unchanged to a wallet
    # that did not exist when it was fitted
    assert 0.7 < got[0] < 0.8


def test_unseen_niche_falls_through_to_the_parent():
    rows = [["w%d" % i, "c", "A", 0.5, 1.0 if i < 100 else 0.0] for i in range(200)]
    bl = fit_hierarchical_baseline(_frame(rows), levels=("category", "niche_l1"),
                                   min_bin=10)
    fresh = _frame([["nw", "c", "NEVER_SEEN", 0.5, 0.0]])
    got = expected_outcome_hier(bl, fresh)
    assert np.isfinite(got[0])
    assert got[0] == pytest.approx(0.5, abs=0.05)   # the category/global value


def test_thin_cells_are_absent_so_application_pools_to_parent():
    rows = ([["w%d" % i, "c", "A", 0.5, 1.0 if i < 60 else 0.0] for i in range(120)]
            + [["z%d" % i, "c", "B", 0.5, 1.0] for i in range(3)])
    bl = fit_hierarchical_baseline(_frame(rows), levels=("category", "niche_l1"),
                                   min_bin=50)
    assert not any(k.startswith("B@") for k in bl["cells"]["niche_l1"])
    got = expected_outcome_hier(bl, _frame([["x", "c", "B", 0.5, 0.0]]))
    assert got[0] < 0.99                     # pooled, not B's own 1.0


def test_apply_nan_price_stays_nan():
    rows = [["w%d" % i, "c", "A", 0.5, float(i % 2)] for i in range(100)]
    bl = fit_hierarchical_baseline(_frame(rows), levels=("category", "niche_l1"), min_bin=10)
    fresh = _frame([["x", "c", "A", np.nan, 0.0]])
    assert np.isnan(expected_outcome_hier(bl, fresh)[0])


def test_fit_empty_frame():
    bl = fit_hierarchical_baseline(_frame([]))
    assert bl["n_bets"] == 0
