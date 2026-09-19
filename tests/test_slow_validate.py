"""Tests for src/slow_validate.py — validation of the deepened slow shortlist
(Project 3 step 5c). Covers the concentration/FDR helpers and the per-wallet gate
logic (persistence, the 10c floor, and each concentration guard) on fixtures whose
correct verdict is known by construction, plus the deep slow-universe loader.
"""

import numpy as np
import pandas as pd

from src.features import PriceBaseline
from src.forecaster_metrics import GLOBAL_KEY, REAL_WORLD_KEY
from src.slow_validate import (
    bh_reject,
    eff_breadth,
    load_deep_slow_bets,
    slow_verdict,
    validate_slow,
)

DAY = 86400

# Config with easy-to-hit floors for small fixtures, real gates intact.
_CFG = {"scoring": {"oos_split": 0.5, "oos_bootstrap_resamples": 500,
                    "oos_significance_alpha": 0.05,
                    "slow": {"min_bets_per_half": 5, "min_oos_markets": 3,
                             "min_eff_breadth": 3.0, "min_entry_days": 3,
                             "min_resolution_days": 3, "min_skill_edge": 0.10}}}


def _calibrated_baseline():
    edges = np.round(np.arange(0.0, 1.01, 0.1), 2)
    bl = PriceBaseline(edges, edges[:-1], float(edges[:-1].mean()))
    return {GLOBAL_KEY: bl, REAL_WORLD_KEY: bl, "other": bl}


def _wallet_bets(wallet, n, price, win_rate, one_market=False, base_ts=0):
    """n bets in `other` at `price`, wins DISTRIBUTED evenly across the ordering (so
    the chronological OOS split gives both halves the same win rate — a stationary
    edge, not a decaying one). Spread across n distinct markets/days (or one market)."""
    win_per_block = int(round(win_rate * 20))
    rows = []
    for i in range(n):
        rows.append({
            "wallet": wallet,
            "market_id": f"{wallet}-solo" if one_market else f"{wallet}-m{i}",
            "category": "other", "entry_price": price,
            "resolved_value": 1.0 if (i % 20) < win_per_block else 0.0, "resolved": True,
            "timestamp": base_ts + i * DAY,          # distinct entry days
            "res_ts_proxy": base_ts + i * DAY + 3 * DAY,  # distinct resolution days
        })
    return rows


# --- helpers ---------------------------------------------------------------

def test_eff_breadth():
    assert eff_breadth(["a", "a", "a"]) == 1.0                 # one market
    assert abs(eff_breadth(["a", "b", "c", "d"]) - 4.0) < 1e-9  # even across 4
    assert 1.0 < eff_breadth(["a", "a", "b"]) < 2.0            # concentrated


def test_bh_reject_controls_and_handles_nan():
    # Clearly-significant p's reject; a large p does not; NaN never rejects.
    p = np.array([0.001, 0.002, 0.9, np.nan])
    rej = bh_reject(p, q=0.10)
    assert rej[0] and rej[1] and not rej[2] and not rej[3]
    assert not bh_reject(np.array([np.nan, np.nan]), 0.10).any()


# --- per-wallet validation -------------------------------------------------

def test_broad_persistent_10c_wallet_persists():
    bets = pd.DataFrame(_wallet_bets("skilled", 40, price=0.4, win_rate=0.9))  # residual ~ +0.5
    table = validate_slow(bets, _calibrated_baseline(), cfg=_CFG)
    r = table.set_index("wallet").loc["skilled"]
    assert r["out_sample_skill"] >= 0.10
    assert r["significant"] and r["magnitude_ok"] and r["concentration_ok"]
    assert r["edge_persisted"]


def test_concentrated_wallet_rejected_even_with_edge():
    # High edge, but ALL bets in one market -> eff_breadth 1 -> concentration fails.
    bets = pd.DataFrame(_wallet_bets("solo", 40, price=0.4, win_rate=0.9, one_market=True))
    table = validate_slow(bets, _calibrated_baseline(), cfg=_CFG)
    r = table.set_index("wallet").loc["solo"]
    assert r["eff_breadth"] < 3.0
    assert not r["concentration_ok"]
    assert not r["edge_persisted"]


def test_below_10c_floor_does_not_persist():
    # Real but small edge (~+0.05) -> fails the 10c magnitude floor.
    # price 0.45 -> baseline bin mean 0.4; win_rate 0.45 -> residual ~ +0.05.
    bets = pd.DataFrame(_wallet_bets("small", 40, price=0.45, win_rate=0.45))
    table = validate_slow(bets, _calibrated_baseline(), cfg=_CFG)
    r = table.set_index("wallet").loc["small"]
    assert 0.0 < r["out_sample_skill"] < 0.10
    assert not r["magnitude_ok"]
    assert not r["edge_persisted"]


def test_noise_wallet_does_not_persist():
    bets = pd.DataFrame(_wallet_bets("noise", 40, price=0.5, win_rate=0.5))  # residual ~0
    table = validate_slow(bets, _calibrated_baseline(), cfg=_CFG)
    r = table.set_index("wallet").loc["noise"]
    assert not r["edge_persisted"]


def test_thin_wallet_skipped():
    # Below 2*min_bets_per_half -> not tested at all (no row).
    bets = pd.DataFrame(_wallet_bets("thin", 6, price=0.4, win_rate=0.9))
    table = validate_slow(bets, _calibrated_baseline(), cfg=_CFG)
    assert "thin" not in set(table.get("wallet", []))


def test_slow_verdict_counts_and_fdr():
    bets = pd.concat([
        pd.DataFrame(_wallet_bets("skilled", 40, 0.4, 0.9, base_ts=0)),
        pd.DataFrame(_wallet_bets("noise", 40, 0.5, 0.5, base_ts=1000 * DAY)),
    ], ignore_index=True)
    table = validate_slow(bets, _calibrated_baseline(), cfg=_CFG)
    v = slow_verdict(table, cfg=_CFG, fdr_q=0.10)
    assert v["persisted"] >= 1
    assert v["fdr_survivors"] >= 1


# --- deep slow-universe loader (speed classification) ----------------------

def test_load_deep_slow_bets_classifies_speed(monkeypatch):
    import src.slow_validate as sv
    # Deep dataset: one slow market (span 3 days) + one fast (span 2 min).
    deep = pd.DataFrame({
        "wallet": ["w1", "w1", "w2", "w2"],
        "market_id": ["slowm", "slowm", "fastm", "fastm"],
        "token_id": ["t"] * 4, "outcome": ["Yes"] * 4, "side": ["BUY"] * 4,
        "entry_price": [0.5] * 4, "size": [1.0] * 4,
        "timestamp": [0, 3 * DAY, 100, 220],
        "resolved": [True] * 4, "resolved_value": [1.0, 0.0, 1.0, 0.0],
        "question": ["?"] * 4, "slug": ["nfl-x", "nfl-x", "btc-updown", "btc-updown"],
        "speed_bucket": ["unknown"] * 4,
    })
    monkeypatch.setattr(sv, "load_deep_trades", lambda: deep)
    monkeypatch.setattr(sv, "load_market_meta", lambda columns=None: pd.DataFrame(columns=["market_id", "lifespan_s"]))
    out = load_deep_slow_bets(cfg={"scoring": {"speed": {}}})
    # Only the slow market's bets survive (tape span 3d >= 24h; fast 2min excluded).
    assert set(out["market_id"]) == {"slowm"}
    assert "category" in out.columns and "res_ts_proxy" in out.columns


# ---------------------------------------------------------------------------
# Complex-unit gates (step 7) — the event complex is the slow path's real cluster
# ---------------------------------------------------------------------------

_CFG_CX = {"scoring": {"oos_split": 0.5, "oos_bootstrap_resamples": 500,
                       "oos_significance_alpha": 0.05,
                       "slow": {"min_bets_per_half": 5, "min_oos_markets": 3,
                                "min_eff_breadth": 3.0, "min_entry_days": 3,
                                "min_resolution_days": 3, "min_skill_edge": 0.10,
                                "min_oos_complexes": 3,
                                "min_eff_breadth_complex": 3.0,
                                "min_entry_days_complex": 3,
                                "min_resolution_days_complex": 3}}}


def _cx_bets(wallet, n, price, win_rate, complexes, base_ts=0):
    """Like `_wallet_bets` but assigns each bet an event complex round-robin over
    `complexes`, so complex-unit breadth is known by construction."""
    rows = _wallet_bets(wallet, n, price, win_rate, base_ts=base_ts)
    for i, r in enumerate(rows):
        r["niche_l1"] = complexes[i % len(complexes)]
        r["niche_l2"] = f"{r['niche_l1']}|deadline"
    return rows


def _run(rows, complex_gates, cluster_col="niche_l1"):
    df = pd.DataFrame(rows)
    return validate_slow(df, _calibrated_baseline(), cfg=_CFG_CX,
                         cluster_col=cluster_col, complex_gates=complex_gates)


def test_complex_columns_are_reported_even_when_not_gating():
    """Additive metadata: the complex-unit view is always computed when the column
    is present, whether or not it gates. No-drop invariant."""
    rows = _cx_bets("w", 40, 0.3, 0.75, ["a", "b", "c", "d"])
    t = _run(rows, complex_gates=False)
    for c in ("out_complexes", "eff_breadth_complex", "entry_days_complex",
              "resolution_days_complex", "top_complex", "top_complex_share",
              "complex_concentration_ok"):
        assert c in t.columns, c
    assert t.loc[0, "out_complexes"] == 4


def test_single_complex_wallet_is_not_persisted_by_rule():
    """A wallet whose entire held-out record sits in ONE event complex must fall
    out mechanically: the complex bootstrap has a single cluster, returns NaN, and
    `significant` requires a non-NaN p. This is the +0.54 single-ladder name."""
    rows = _cx_bets("solo", 40, 0.2, 0.95, ["only_complex"])
    t = _run(rows, complex_gates=True)
    assert t.loc[0, "out_complexes"] == 1
    assert np.isnan(t.loc[0, "cluster_p"])
    assert not t.loc[0, "significant"]
    assert not t.loc[0, "edge_persisted"]
    # ...and it is excluded for the same reason even with the gates switched off,
    # i.e. by the significance rule, not by the concentration gate.
    assert not _run(rows, complex_gates=True).loc[0, "edge_persisted"]


def test_complex_breadth_gate_rejects_a_market_broad_but_complex_narrow_wallet():
    """The whole point of step 7: 40 distinct markets and 40 distinct days looks
    broad in market units, but if 90% of the bets are one ladder the wallet is
    effectively a single bet. Market-unit gates pass it; complex-unit gates do not."""
    rows = _cx_bets("ladder", 40, 0.3, 0.80, ["big"] * 9 + ["other"])
    t_off = _run(rows, complex_gates=False)
    t_on = _run(rows, complex_gates=True)
    assert t_off.loc[0, "eff_breadth"] >= 3.0          # broad over markets
    assert t_on.loc[0, "eff_breadth_complex"] < 3.0    # narrow over complexes
    assert not t_on.loc[0, "complex_concentration_ok"]
    assert t_on.loc[0, "concentration_ok"]             # market gate still passes
    assert not t_on.loc[0, "edge_persisted"]


def test_complex_gates_pass_a_genuinely_complex_broad_wallet():
    rows = _cx_bets("broad", 60, 0.3, 0.80, ["a", "b", "c", "d", "e"])
    t = _run(rows, complex_gates=True)
    assert t.loc[0, "out_complexes"] == 5
    assert t.loc[0, "eff_breadth_complex"] >= 3.0
    assert t.loc[0, "complex_concentration_ok"]
    assert t.loc[0, "edge_persisted"]


def test_market_unit_gates_are_still_applied_alongside():
    """Nothing was removed: a wallet that is complex-broad but piles every bet into
    one market still fails the market-unit breadth gate."""
    rows = _cx_bets("onemkt", 60, 0.3, 0.80, ["a", "b", "c", "d", "e"])
    for r in rows:
        r["market_id"] = "single"
    t = _run(rows, complex_gates=True)
    assert not t.loc[0, "concentration_ok"]
    assert not t.loc[0, "edge_persisted"]


def test_entry_days_complex_counts_complex_day_pairs():
    """entry_days >= 3 can be met by ONE complex entered on three days; the
    complex-unit version counts (complex, day) pairs instead."""
    rows = _cx_bets("w", 40, 0.3, 0.75, ["a", "b"])
    t = _run(rows, complex_gates=False)
    assert t.loc[0, "entry_days_complex"] >= t.loc[0, "out_complexes"]


def test_complex_gates_without_the_column_raises_not_silently_passes():
    rows = _wallet_bets("w", 40, 0.3, 0.75)
    df = pd.DataFrame(rows)
    try:
        validate_slow(df, _calibrated_baseline(), cfg=_CFG_CX,
                      cluster_col="market_id", complex_gates=True)
    except KeyError as e:
        assert "niche_l1" in str(e)
    else:
        raise AssertionError("expected KeyError when complex_gates lacks its column")
