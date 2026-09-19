"""Tests for src/slow_market.py — the slow-universe loader + the per-category price
baseline fit inside the slow universe (Project 3 step 4).

The §2.0 baseline fallback hierarchy itself is covered by test_forecaster_metrics;
these tests cover the slow-universe GLUE: the speed_bucket filter, category
derivation on the main ledger, and the residualization math.
"""

import numpy as np
import pandas as pd

from src.features import PriceBaseline
from src.forecaster_metrics import GLOBAL_KEY, REAL_WORLD_KEY
from src.slow_market import (
    SLOW_BUCKETS,
    baseline_summary,
    derive_categories,
    fit_slow_category_baselines,
    load_slow_universe_bets,
    residualize,
    screen_wallets,
    shortlist_size_table,
)


# --- category derivation --------------------------------------------------

def test_derive_categories_labels():
    slugs = ["will-the-chiefs-win-the-super-bowl", "btc-updown-5m-123",
             "2028-us-presidential-election", "some-random-market"]
    qs = ["Chiefs win?", "Bitcoin up or down?", "Who wins?", "?"]
    cats = derive_categories(slugs, qs)
    assert list(cats) == ["sports_nfl", "micro_crypto", "politics", "other"]


# --- slow-universe loader -------------------------------------------------

def _ledger_with_buckets():
    rows = []
    specs = [
        # (market, bucket, side, resolved, slug)
        ("mfast", "fast", "BUY", True, "btc-updown-5m-1"),
        ("mslow", "slow", "BUY", True, "nfl-chiefs-win"),
        ("mdeep", "deep_slow", "BUY", True, "2028-election"),
        ("munk", "unknown", "BUY", True, "mystery-market"),
        ("mslow2", "slow", "SELL", True, "nba-lakers"),        # SELL -> excluded
        ("mslow3", "slow", "BUY", False, "nfl-bills-win"),      # unresolved -> excluded
    ]
    for i, (mkt, bucket, side, res, slug) in enumerate(specs):
        rows.append({
            "wallet": f"0xw{i}", "market_id": mkt, "token_id": f"t{i}",
            "outcome": "Yes", "side": side, "entry_price": 0.5, "size": 1.0,
            "timestamp": 1000 + i, "resolved": res,
            "resolved_value": 1.0 if res else None,
            "question": "?", "slug": slug, "tx_hash": f"0x{i}",
            "speed_bucket": bucket,
        })
    return pd.DataFrame(rows)


def test_load_slow_universe_filters_buckets_buys_resolved(tmp_path):
    led = tmp_path / "ledger.parquet"
    _ledger_with_buckets().to_parquet(led)
    bets = load_slow_universe_bets(ledger_path=led)
    # Only the slow + deep_slow resolved BUY bets survive.
    assert set(bets["market_id"]) == {"mslow", "mdeep"}
    assert "category" in bets.columns
    assert set(bets["category"]) == {"sports_nfl", "politics"}


def test_load_slow_universe_custom_buckets(tmp_path):
    led = tmp_path / "ledger.parquet"
    _ledger_with_buckets().to_parquet(led)
    # deep_slow only.
    bets = load_slow_universe_bets(ledger_path=led, buckets=("deep_slow",))
    assert set(bets["market_id"]) == {"mdeep"}


def test_load_slow_universe_legacy_ledger_is_empty(tmp_path):
    # A pre-step-2 ledger without speed_bucket -> empty slow universe (nothing proven slow).
    led = tmp_path / "legacy.parquet"
    df = _ledger_with_buckets().drop(columns=["speed_bucket"])
    df.to_parquet(led)
    bets = load_slow_universe_bets(ledger_path=led)
    assert bets.empty


def test_load_slow_universe_missing_ledger_is_empty(tmp_path):
    assert load_slow_universe_bets(ledger_path=tmp_path / "nope.parquet").empty


# --- residualization math -------------------------------------------------

def test_residualize_subtracts_category_baseline():
    # A single-category baseline: E[outcome | price] = 0.6 for the fitted bin.
    bl = {
        GLOBAL_KEY: PriceBaseline(np.array([0.0, 1.0]), np.array([0.6]), 0.6),
        REAL_WORLD_KEY: PriceBaseline(np.array([0.0, 1.0]), np.array([0.6]), 0.6),
        "sports_nfl": PriceBaseline(np.array([0.0, 1.0]), np.array([0.6]), 0.6),
    }
    bets = pd.DataFrame({
        "category": ["sports_nfl", "sports_nfl"],
        "entry_price": [0.5, 0.5],
        "resolved_value": [1.0, 0.0],
        "resolved": [True, True],
    })
    out = residualize(bets, bl)
    # expected outcome is the category curve's bin mean (0.6); residual = outcome - 0.6.
    assert np.allclose(out["expected_outcome"], [0.6, 0.6])
    assert np.allclose(out["residual_skill"], [0.4, -0.6])


# --- baseline fit + summary over a constructed slow universe --------------

def _slow_universe_fixture():
    """600 NFL bets (clears the 500-bet own-curve floor) with a planted
    favorite-longshot structure, plus a handful of thin 'other' bets that must fall
    back. NFL favorites at 0.7 resolve YES 70% of the time -> zero structural skill,
    so a correct per-category baseline residualizes them to ~0 on average."""
    rng = np.random.default_rng(0)
    n = 600
    price = np.where(rng.random(n) < 0.5, 0.7, 0.3)
    outcome = (rng.random(n) < price).astype(float)  # calibrated: P(YES)=price
    nfl = pd.DataFrame({
        "wallet": [f"0xw{i%20}" for i in range(n)],
        "market_id": [f"nfl{i%40}" for i in range(n)],
        "category": "sports_nfl", "entry_price": price, "resolved_value": outcome,
        "resolved": True,
    })
    other = pd.DataFrame({
        "wallet": ["0xz"] * 4, "market_id": [f"o{i}" for i in range(4)],
        "category": "culture", "entry_price": [0.4, 0.6, 0.5, 0.55],
        "resolved_value": [1.0, 0.0, 1.0, 0.0], "resolved": [True] * 4,
    })
    return pd.concat([nfl, other], ignore_index=True)


def test_fit_slow_category_baselines_own_and_fallback():
    bets = _slow_universe_fixture()
    bl = fit_slow_category_baselines(bets, cfg={"scoring": {"price_baseline_bins": 5}})
    assert GLOBAL_KEY in bl and REAL_WORLD_KEY in bl
    # NFL has >=500 bets -> its OWN curve; culture has 4 -> falls back to real-world-wide.
    assert bl["sports_nfl"] is not bl[REAL_WORLD_KEY]
    assert bl.get("culture", bl[REAL_WORLD_KEY]) is bl[REAL_WORLD_KEY]
    # Calibrated NFL bets -> mean residual skill ~ 0 (no structural edge left).
    resid = residualize(bets[bets.category == "sports_nfl"], bl)
    assert abs(resid["residual_skill"].mean()) < 0.05


def test_baseline_summary_columns_and_own_flag():
    bets = _slow_universe_fixture()
    bl = fit_slow_category_baselines(bets, cfg={"scoring": {"price_baseline_bins": 5}})
    summ = baseline_summary(bets, bl)
    assert set(summ.columns) == {"category", "n_resolved", "own_curve",
                                 "mean_outcome", "mean_raw_edge", "mean_residual_skill"}
    nfl = summ[summ["category"] == "sports_nfl"].iloc[0]
    assert nfl["own_curve"] is True or nfl["own_curve"] == True  # noqa: E712
    assert nfl["n_resolved"] == 600


# --- the crude wide screen (5a) -------------------------------------------

_SCREEN_CFG = {"scoring": {"slow": {"screen_shrinkage_k": 5, "screen_min_bets": 2}}}


def _calibrated_baseline():
    """A FIXED, perfectly-calibrated market baseline E[outcome | price] = price (on a
    0.1 grid), so the screen logic is tested independent of baseline fitting (in a
    tiny fixture a skilled wallet would otherwise dominate its own price bin and the
    fitted baseline would absorb its edge — a small-sample artifact that does not
    occur at 463k wallets, where one wallet is negligible to a bin mean)."""
    edges = np.round(np.arange(0.0, 1.01, 0.1), 2)
    means = edges[:-1]  # price 0.1 -> 0.1, 0.5 -> 0.5, 0.9 -> 0.9
    bl = PriceBaseline(edges, means, float(means.mean()))
    return {GLOBAL_KEY: bl, REAL_WORLD_KEY: bl, "other": bl}


def _screen_fixture():
    """Three wallets against a flat baseline E[outcome|price]=0.5:
      - skilled:  10 bets, always beats price by +0.4 (residual +0.4) -> high score
      - lucky1:   1 bet, residual +1.0 -> excluded (below screen_min_bets) OR shrunk hard
      - noise:   10 bets, residual ~0 -> ~0 score
    A single flat 'other' category keeps E_cat = 0.5 so residual = outcome - 0.5.
    """
    rows = []
    for i in range(10):  # skilled: pays 0.1, wins (outcome 1) -> residual +0.5
        rows.append(("skilled", f"m{i}", "other", 0.1, 1.0))
    rows.append(("lucky1", "mx", "other", 0.1, 1.0))  # 1 bet only
    for i in range(10):  # noise: pays 0.5, half win -> residual ~0
        rows.append(("noise", f"n{i}", "other", 0.5, float(i % 2)))
    df = pd.DataFrame(rows, columns=["wallet", "market_id", "category", "entry_price", "resolved_value"])
    df["resolved"] = True
    df["corpus"] = "discovery"
    return df


def test_screen_scores_skill_above_noise_and_shrinks_thin():
    bets = _screen_fixture()
    scores = screen_wallets(bets, _calibrated_baseline(), cfg=_SCREEN_CFG)
    s = scores.set_index("wallet")
    # skilled (pays 0.1, always wins -> residual +0.9, n=10, k=5 -> shrunk ~0.6).
    assert s.loc["skilled", "screen_score"] > 0.2
    assert s.loc["skilled", "screen_score"] > s.loc["noise", "screen_score"]
    # noise ~ 0.
    assert abs(s.loc["noise", "screen_score"]) < 0.1
    # lucky1 (1 bet) is excluded by screen_min_bets=2 -> not scored.
    assert "lucky1" not in s.index
    # corpus breakdown carried through.
    assert s.loc["skilled", "n_discovery"] == 10


def test_screen_score_never_uses_win_rate_only_skill():
    # A wallet that only buys 0.9 favorites (high win rate, ~zero skill) must NOT
    # score high: against a calibrated baseline (0.9 resolves YES 90% of the time),
    # its residual is ~0 regardless of its 90% win rate.
    rows = [("fav", f"m{i}", "other", 0.9, float(i % 10 != 0)) for i in range(50)]  # 90% win
    df = pd.DataFrame(rows, columns=["wallet", "market_id", "category", "entry_price", "resolved_value"])
    df["resolved"] = True
    df["corpus"] = "main"
    scores = screen_wallets(df, _calibrated_baseline(), cfg=_SCREEN_CFG)
    # ~90% win rate but ~zero SKILL -> screen_score near 0.
    assert abs(scores.set_index("wallet").loc["fav", "screen_score"]) < 0.1


def test_shortlist_size_table_monotone():
    bets = _screen_fixture()
    scores = screen_wallets(bets, _calibrated_baseline(), cfg=_SCREEN_CFG)
    tbl = shortlist_size_table(scores, gates=(0.0, 0.2, 0.5))
    sizes = tbl.set_index("gate")["wallets"]
    # more permissive gate -> at least as many wallets.
    assert sizes[0.0] >= sizes[0.2] >= sizes[0.5]
