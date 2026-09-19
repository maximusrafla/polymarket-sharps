"""Tests for scripts/audit_forecaster_null.py — the shuffled-outcome null + BH-FDR
gate for the Project-2 forecaster table. Hand-built fixtures where the answer is
known by construction (CLAUDE.md test style). No network.

The load-bearing invariants:
  - `score_cells` reproduces validate.py's one-sided OOS t-test gate exactly, AND
    NaN-outs degenerate (~zero-variance) cells in BOTH the p and the t (the bug the
    empirical-p FDR path would otherwise leak a blown-up ~1e16 t through).
  - the within-(category, cent) shuffle preserves every bucket's outcome multiset
    (the favorite-longshot base rate) while permuting the wallet<->outcome link.
  - `build_cells` builds the per-wallet all_real_world aggregate for multi-category
    wallets and splits each cell chronologically at floor(n * oos_split).
"""

import numpy as np
import pandas as pd
from scipy import stats

from scripts.audit_forecaster_null import (
    ALL_CELL,
    build_cells,
    make_shuffler,
    score_cells,
    shuffled_residual,
)


def _one_cell(in_resid, out_resid):
    """A `cells` dict for a single cell whose in/out halves are the given residuals,
    laid out contiguously in a base residual vector [in..., out...]."""
    ni, no = len(in_resid), len(out_resid)
    resid_full = np.concatenate([in_resid, out_resid]).astype(float)
    cells = {
        "C": 1,
        "in_code": np.zeros(ni, dtype=int), "in_pos": np.arange(ni),
        "in_n": np.array([float(ni)]),
        "out_code": np.zeros(no, dtype=int), "out_pos": np.arange(ni, ni + no),
        "out_n": np.array([float(no)]),
    }
    return resid_full, cells


def test_score_cells_matches_scipy_ttest():
    rng = np.random.default_rng(0)
    out = rng.normal(0.05, 0.1, size=30)
    in_ = rng.normal(0.05, 0.1, size=30)
    resid, cells = _one_cell(in_, out)
    in_m, out_m, t, p, sig, pers = score_cells(resid, cells, alpha=0.05, min_half=10, min_edge=0.02)

    exp = stats.ttest_1samp(out, 0.0, alternative="greater")
    assert np.isclose(out_m[0], out.mean())
    assert np.isclose(in_m[0], in_.mean())
    assert np.isclose(t[0], exp.statistic, rtol=1e-9)
    assert np.isclose(p[0], exp.pvalue, rtol=1e-9)
    # candidate (in>0) & out>0 & p<alpha => matches the analytic verdict
    assert bool(sig[0]) == (in_.mean() > 0 and out.mean() > 0 and exp.pvalue < 0.05)


def test_score_cells_degenerate_variance_is_nan_not_blownup():
    # out-half all identical (a wallet whose held-out bets all resolved the same way):
    # variance 0 -> the t-test is undefined. validate._oos_significance returns NaN;
    # score_cells MUST NaN BOTH t and p (a finite ~1e16 t would leak into the FDR).
    resid, cells = _one_cell(np.full(12, 0.3), np.full(12, 0.3))
    in_m, out_m, t, p, sig, pers = score_cells(resid, cells, alpha=0.05, min_half=10, min_edge=0.02)
    assert np.isnan(t[0])
    assert np.isnan(p[0])
    assert not bool(sig[0]) and not bool(pers[0])


def test_score_cells_gates_thin_halves_to_not_significant():
    # out-half below min_half -> untestable -> never significant, no matter the mean.
    resid, cells = _one_cell(np.full(10, 0.1), np.array([0.9, 0.9, 0.9]))
    _, _, t, p, sig, pers = score_cells(resid, cells, alpha=0.05, min_half=10, min_edge=0.02)
    assert np.isnan(p[0]) and not bool(sig[0])


def test_score_cells_magnitude_floor_separates_persisted_from_significant():
    rng = np.random.default_rng(1)
    # tiny but ultra-consistent edge: significant (huge n, small SE) yet below the 2c
    # magnitude floor -> significant True, persisted False (the C-gate does its job).
    out = np.full(400, 0.005) + rng.normal(0, 1e-4, size=400)
    resid, cells = _one_cell(np.full(400, 0.005), out)
    _, out_m, t, p, sig, pers = score_cells(resid, cells, alpha=0.05, min_half=10, min_edge=0.02)
    assert bool(sig[0]) is True
    assert out_m[0] < 0.02 and not bool(pers[0])


def _bets():
    # two categories, two cents; a wallet spanning both categories (-> all cell).
    rows = []
    ts = 0
    for cat in ("politics", "sports_nba"):
        for cent_price in (0.30, 0.31):
            for k in range(8):
                ts += 1
                rows.append({
                    "wallet": f"0x{'a'*39}{k%2}",  # two wallets, both multi-category
                    "category": cat, "entry_price": cent_price,
                    "rv": float(k % 2), "expected": 0.4,
                    "residual": float(k % 2) - 0.4,
                    "cent": int(np.floor(cent_price / 0.01)),
                    "timestamp": float(ts), "market_id": f"m{cat}{k%3}",
                    "tape_complete": True,
                })
    return pd.DataFrame(rows)


def test_shuffle_preserves_bucket_multiset_and_permutes():
    bets = _bets()
    group_id, base_order, rv, expected = make_shuffler(bets)
    rng = np.random.default_rng(7)
    resid = shuffled_residual(group_id, base_order, rv, expected, rng)
    shuffled_rv = resid + expected  # invert residual = shuffled_rv - expected

    key = bets["category"].astype(str) + "|" + bets["cent"].astype(str)
    for _, idx in pd.Series(np.arange(len(bets))).groupby(key.to_numpy()):
        orig = np.sort(rv[idx.to_numpy()])
        shuf = np.sort(shuffled_rv[idx.to_numpy()])
        # base rate (multiset of outcomes) preserved exactly within each bucket
        assert np.allclose(orig, shuf)
    # and it is a real permutation, not identity (with this seed some row moved)
    assert not np.allclose(rv, shuffled_rv)


def test_build_cells_makes_all_cell_and_half_split():
    bets = _bets()
    cfg_vals = {"min_score_bets": 5, "oos_split": 0.5}
    cells = build_cells(bets, cfg_vals)
    attr = cells["attr"]
    # multi-category wallets get an all_real_world aggregate cell
    assert (attr["category"] == ALL_CELL).any()
    # every cell's in/out halves sum to its sample_size, split at floor(n/2)
    tot = cells["in_n"] + cells["out_n"]
    assert np.allclose(tot, attr["sample_size"].to_numpy())
    # in-sample half is floor(n * 0.5)
    assert np.allclose(cells["in_n"], np.floor(attr["sample_size"].to_numpy() * 0.5))


def test_concentration_metrics_catch_fill_inflation():
    # A single-market cell replicated across 20 fills on ONE day: breadth=1 counts it
    # as 1 market, and eff_breadth (1/HHI) collapses to 1, entry_days to 1 — the
    # concentration guard's job. A balanced 4-market cell over 4 days scores ~4 / 4.
    day = 86400
    rows = []
    for k in range(20):  # concentrated: one market, one day
        rows.append({"wallet": "0x" + "1" * 40, "category": "politics", "entry_price": 0.5,
                     "rv": 1.0, "expected": 0.5, "residual": 0.5, "cent": 50,
                     "timestamp": float(1000 + k), "market_id": "mkt_one", "tape_complete": True})
    for k in range(20):  # spread: 4 markets, 4 distinct days
        rows.append({"wallet": "0x" + "2" * 40, "category": "politics", "entry_price": 0.5,
                     "rv": 1.0, "expected": 0.5, "residual": 0.5, "cent": 50,
                     "timestamp": float(1000 + (k % 4) * day), "market_id": f"mkt_{k % 4}",
                     "tape_complete": True})
    attr = build_cells(pd.DataFrame(rows), {"min_score_bets": 5, "oos_split": 0.5})["attr"]
    conc = attr[attr["category"] == "politics"].set_index("wallet")
    w1, w2 = "0x" + "1" * 40, "0x" + "2" * 40
    assert conc.loc[w1, "eff_breadth"] == 1.0 and conc.loc[w1, "entry_days"] == 1
    assert conc.loc[w1, "top_market_share"] == 1.0
    assert np.isclose(conc.loc[w2, "eff_breadth"], 4.0) and conc.loc[w2, "entry_days"] == 4
