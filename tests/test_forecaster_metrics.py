"""Tests for src/forecaster_metrics.py — the Project-2 per-(wallet, category)
metric stack. Hand-built fixtures where the correct skill edge, copyability, and
magnitude are known by construction (CLAUDE.md test style).

No network: every test drives the pure scoring functions on small frames.
"""

import numpy as np
import pandas as pd

from src.features import expected_outcome
from src.forecaster_metrics import (
    ALL_CELL,
    GLOBAL_KEY,
    REAL_WORLD_KEY,
    build_token_arrays,
    category_baseline,
    cell_metrics,
    compute_forecaster_table,
    expected_outcome_by_category,
    fit_category_baselines,
    follower_prices,
    _config_values,
)

DISCOVERY_COLS = [
    "wallet", "market_id", "token_id", "outcome", "side", "entry_price",
    "size", "timestamp", "resolved", "resolved_value", "question", "slug",
    "tx_hash", "category",
]


def _bet(wallet, market, token, price, ts, resolved_value, category,
         side="BUY", size=10.0):
    return {
        "wallet": wallet, "market_id": market, "token_id": token, "outcome": "Yes",
        "side": side, "entry_price": price, "size": size, "timestamp": ts,
        "resolved": True, "resolved_value": float(resolved_value),
        "question": "q?", "slug": "s", "tx_hash": f"{wallet}:{token}:{ts}",
        "category": category,
    }


def _cfg():
    # Small thresholds so tiny fixtures can actually validate.
    return {
        "scoring": {"price_baseline_bins": 20, "min_bets_per_half": 5,
                    "oos_significance_alpha": 0.05, "min_skill_edge": 0.02,
                    "oos_split": 0.5, "fair_value_resolution_guard": 0.2,
                    "min_bets_for_category_baseline": 500},
        "copyability": {"fill_bandwidth_sec": 60, "min_reachability": 0.5},
        "magnitude": {"min_copyable_edge_cents": 2.0, "slippage_buffer_cents": 1.0},
    }


# --- §2.0 per-category baseline + fallback --------------------------------

def test_fit_category_baselines_thin_subcategory_falls_back():
    rng = np.random.default_rng(0)
    rows = []
    # 600 politics bets with varied prices -> own fitted curve.
    prices = np.round(rng.uniform(0.1, 0.9, 600), 2)
    for i, p in enumerate(prices):
        rows.append(_bet("f", "mp", f"tp{i}", p, i, rng.random() < p, "politics"))
    # 10 sports bets -> too thin for its own curve.
    for i in range(10):
        rows.append(_bet("f", "ms", f"ts{i}", 0.5, i, i % 2, "sports_nfl"))
    resolved = pd.DataFrame(rows)

    bl = fit_category_baselines(resolved, n_bins=20, min_bets_for_own=500)
    # politics got its own fitted curve (real bins)...
    assert bl["politics"].edges.size > 0
    assert bl["politics"] is not bl[REAL_WORLD_KEY]
    # ...sports fell back to the real-world-wide baseline object (identity).
    assert bl["sports_nfl"] is bl[REAL_WORLD_KEY]
    # unknown category resolves to real-world-wide; micro to global.
    assert category_baseline(bl, "never_seen") is bl[REAL_WORLD_KEY]
    assert category_baseline(bl, "micro_crypto") is bl[GLOBAL_KEY]


def test_expected_outcome_by_category_uses_each_rows_baseline():
    # Two categories with opposite calibration at the same price.
    rows = []
    for i in range(600):
        # politics: high outcomes; sports: low outcomes, both at prices {0.3,0.7}
        p = 0.3 if i % 2 else 0.7
        rows.append(_bet("f", "mp", f"tp{i}", p, i, 1, "politics"))
        rows.append(_bet("f", "ms", f"ts{i}", p, i, 0, "sports_nfl"))
    resolved = pd.DataFrame(rows)
    bl = fit_category_baselines(resolved, n_bins=20, min_bets_for_own=500)

    got = expected_outcome_by_category(bl, ["politics", "sports_nfl"], [0.7, 0.7])
    # politics curve ~1, sports curve ~0 at 0.7 — each row used its own baseline.
    assert got[0] > 0.8
    assert got[1] < 0.2
    # matches a direct per-baseline call
    assert np.isclose(got[0], expected_outcome(bl["politics"], [0.7])[0])


def test_expected_outcome_by_category_keeps_nan_prices_nan():
    bl = {GLOBAL_KEY: fit_category_baselines(pd.DataFrame())[GLOBAL_KEY],
          REAL_WORLD_KEY: fit_category_baselines(pd.DataFrame())[REAL_WORLD_KEY]}
    got = expected_outcome_by_category(bl, ["politics"], [np.nan])
    assert np.isnan(got[0])


# --- metric B: leakage-guarded follower price -----------------------------

def test_follower_prices_reachability_and_value():
    # token T: our bet at t=0; a later OTHER-wallet trade at t=45 @0.7 sits inside
    # (0+Δ30, +bw60] = (30,90]; a lone bet with no follow-on is unreachable (NaN).
    # An anchor trade far in the future extends the token's observed lifespan (as
    # real markets have) so the t=45 trade isn't inside the final-20% resolution
    # guard — otherwise a 2-trade fixture guards its own only follow-on away.
    tape = pd.DataFrame([
        _bet("W", "m", "T", 0.5, 0, 1, "politics"),
        _bet("OTHER", "m", "T", 0.7, 45, 1, "politics"),
        _bet("ANCHOR", "m", "T", 0.7, 100000, 1, "politics"),  # extends lifespan
        # far-future trade only, past the window -> should not rescue reachability
        _bet("W", "m", "U", 0.5, 0, 1, "politics"),
        _bet("OTHER", "m", "U", 0.9, 100000, 1, "politics"),
    ])
    tok = build_token_arrays(tape, guard=0.2)
    bets = tape[(tape["wallet"] == "W")].reset_index(drop=True)
    pa30 = follower_prices(bets, tok, delta=30, bandwidth=60)
    # bet on T reachable at 0.7; bet on U not reachable in-window before guard
    assert np.isclose(pa30[0], 0.7)
    assert np.isnan(pa30[1])


def test_follower_prices_excludes_own_wallet():
    # only the entering wallet trades after entry -> no followable liquidity.
    tape = pd.DataFrame([
        _bet("W", "m", "T", 0.5, 0, 1, "politics"),
        _bet("W", "m", "T", 0.6, 45, 1, "politics"),  # own follow-on: excluded
    ])
    tok = build_token_arrays(tape, guard=0.2)
    bets = tape.iloc[[0]].reset_index(drop=True)
    pa = follower_prices(bets, tok, delta=30, bandwidth=60)
    assert np.isnan(pa[0])


# --- end-to-end: known winner vs. sharp-but-uncopyable --------------------

def _skill_cell(wallet, category, n, base_ts, followable):
    """n resolved BUY bets, entry 0.5, ~92% resolve YES (strong, low-variance
    positive skill vs. a ~0.5 base rate). If `followable`, plant an OTHER-wallet
    trade 45s later @0.5 on each token (price hasn't moved, so a copier entering
    late keeps the edge); else no follow-on liquidity at all (un-copyable)."""
    rows = []
    for i in range(n):
        tok = f"{wallet}_{category}_{i}"
        rv = 0 if i % 12 == 0 else 1  # ~92% YES
        ts = base_ts + i * 1000
        rows.append(_bet(wallet, f"mk_{tok}", tok, 0.5, ts, rv, category))
        # anchor far in the future so the token's observed lifespan is long and the
        # +45s follow-on isn't guarded away as "near resolution" (real tokens live
        # for days; a 3-trade fixture otherwise guards its own follow-on).
        rows.append(_bet("ANCHOR", f"mk_{tok}", tok, 0.5, ts + 500000, rv, category))
        if followable:
            # +75s: inside both the Δ=30 (30,90] and Δ=60 (60,120] follower windows.
            rows.append(_bet("CROWD", f"mk_{tok}", tok, 0.5, ts + 75, rv, category))
    return rows


def _filler(n, base_ts):
    """Filler bets at 0.5 resolving ~50% so the FL base rate at 0.5 is ~0.5 and a
    92%-YES wallet shows a large positive residual (skill) edge. Made large so the
    sharp wallets don't themselves pull the 0.5 base rate up."""
    return [_bet("filler", f"fm{i}", f"ft{i}", 0.5, base_ts + i, i % 2, "politics")
            for i in range(n)]


def test_compute_forecaster_table_winner_vs_uncopyable():
    rows = []
    rows += _filler(700, 0)
    # W_copy: strong skill AND followable liquidity -> a winner.
    rows += _skill_cell("W_copy", "politics", 24, 1_000_000, followable=True)
    # W_nocopy: identical strong skill but NO follow-on liquidity -> not copyable.
    rows += _skill_cell("W_nocopy", "politics", 24, 2_000_000, followable=False)
    tape = pd.DataFrame(rows)[DISCOVERY_COLS]

    table = compute_forecaster_table(tape, _cfg())

    def cell(w):
        return table[(table["wallet"] == w) & (table["category"] == "politics")].iloc[0]

    wc, wn = cell("W_copy"), cell("W_nocopy")

    # Both are genuinely sharp: strong positive residual (skill) edge, persisted.
    assert wc["skill_edge_all"] > 0.25 and wn["skill_edge_all"] > 0.25
    assert bool(wc["edge_persisted"]) and bool(wn["edge_persisted"])

    # Copyability separates them: W_copy has follow-on liquidity and retains edge;
    # W_nocopy has none -> reachability 0, not a winner despite the skill.
    assert wc["reachability_60"] >= 0.5 and wc["copyable_skill_edge_60"] > 0.15
    assert bool(wc["is_winner"]) is True
    assert wn["reachability_60"] == 0.0
    assert bool(wn["is_winner"]) is False


def test_magnitude_floor_blocks_trivial_copyable_edge():
    # A cell whose copyable edge is a fraction of a cent must fail metric C even
    # if reachable and persisted-shaped.
    cfg_vals = _config_values(_cfg())
    # one reachable bet with ~0 follower skill
    cell = pd.DataFrame([{
        "wallet": "w", "market_id": "m", "timestamp": 1, "entry_price": 0.5,
        "resolved_value": 1.0, "residual_edge": 0.001,
        "price_at_30": 0.999, "fskill_30": 0.001,
        "price_at_60": 0.999, "fskill_60": 0.001,
        "price_at_300": 0.999, "fskill_300": 0.001,
    }])
    rec = cell_metrics(cell, "politics", cfg_vals)
    assert rec["clears_magnitude_floor"] is False
    assert rec["is_winner"] is False


def test_full_tape_gate_debiases_winner_on_truncated_tapes():
    # A genuine A+B+C winner, but ALL its markets' tapes are cap-truncated (§1.2):
    # is_winner stays True (gate semantics unchanged) yet winner_full_tape flips to
    # False and frac_full_tape -> 0, so the de-biased default view excludes it.
    rows = _filler(700, 0)
    rows += _skill_cell("W", "politics", 24, 1_000_000, followable=True)
    tape = pd.DataFrame(rows)[DISCOVERY_COLS]
    # Mark every market in the tape as cap-truncated.
    stats = pd.DataFrame({"market_id": tape["market_id"].unique()})
    stats["hit_cap"] = True
    stats["last_pulled"] = 1

    table = compute_forecaster_table(tape, _cfg(), tape_stats=stats)
    w = table[(table["wallet"] == "W") & (table["category"] == "politics")].iloc[0]
    assert bool(w["is_winner"]) is True          # A+B+C unchanged
    assert w["frac_full_tape"] == 0.0            # all bets on truncated tapes
    assert bool(w["full_tape"]) is False
    assert bool(w["winner_full_tape"]) is False  # excluded from the de-biased view

    # With the SAME tape but no truncation recorded, it is a full-tape winner.
    table2 = compute_forecaster_table(tape, _cfg(), tape_stats=None)
    w2 = table2[(table2["wallet"] == "W") & (table2["category"] == "politics")].iloc[0]
    assert w2["frac_full_tape"] == 1.0 and bool(w2["winner_full_tape"]) is True


def test_nothing_dropped_all_cells_present():
    rows = _filler(10, 0)
    rows += [_bet("solo", "m1", "t1", 0.5, 1, 1, "sports_nba")]  # single thin bet
    tape = pd.DataFrame(rows)[DISCOVERY_COLS]
    table = compute_forecaster_table(tape, _cfg())
    # the single-bet wallet still appears with its (un-validatable) row
    assert ((table["wallet"] == "solo") & (table["category"] == "sports_nba")).any()


def test_all_real_world_aggregate_cell_added_for_multi_category_wallet():
    rows = _filler(40, 0)
    rows += _skill_cell("multi", "politics", 12, 1_000_000, followable=True)
    rows += _skill_cell("multi", "sports_nba", 12, 3_000_000, followable=True)
    tape = pd.DataFrame(rows)[DISCOVERY_COLS]
    table = compute_forecaster_table(tape, _cfg())
    cats = set(table[table["wallet"] == "multi"]["category"])
    assert {"politics", "sports_nba", ALL_CELL} <= cats
