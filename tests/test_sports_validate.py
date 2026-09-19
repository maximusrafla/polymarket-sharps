"""Tests for the sports validation stack (src/sports_validate.py).

The crux is the cluster unit. These fixtures are built so the SAME wallet record
is certified under the gate's setting (`market_id` clusters, no event gates) and
rejected under the corrected one (`event` clusters + event-unit concentration
gates) — which is exactly the failure the gate corpus would have produced at
scale, and the reason the sports arm re-validates instead of inheriting.

Also pinned: the scoped `scoring.sports` block cannot reach Project 1, and the
finer level is strictly finer.
"""

import numpy as np
import pandas as pd
import pytest

from src.slow_validate import validate_slow
from src.sports_validate import (
    BASELINE_MODES,
    CFG_SECTION,
    CLUSTER_COL,
    STANDARD_BASELINE,
    residualize_for,
)

DAY = 86400


def _cfg(**sports):
    """A config whose sports block is deliberately DIFFERENT from the global and
    slow blocks, so a test that passes can only be reading the right one."""
    block = {"min_skill_edge": 0.10, "min_oos_markets": 5, "min_eff_breadth": 3.0,
             "min_entry_days": 3, "min_resolution_days": 3, "min_bets_per_half": 10,
             "min_oos_complexes": 3, "min_eff_breadth_complex": 3.0,
             "min_entry_days_complex": 3, "min_resolution_days_complex": 3}
    block.update(sports)
    return {"scoring": {"oos_split": 0.5, "oos_significance_alpha": 0.05,
                        "oos_bootstrap_resamples": 500, "min_skill_edge": 0.02,
                        "slow": {"min_skill_edge": 0.99},   # must never be read here
                        CFG_SECTION: block}}


def _wallet_bets(wallet, events, per_event, resid, start_day=0):
    """One wallet's bets: `per_event` bets in each of `events`, each bet in its own
    market (the outright-field shape: many markets, one resolution)."""
    rows = []
    for e, ev in enumerate(events):
        for i in range(per_event):
            day = start_day + e * 3 + i
            rows.append({"wallet": wallet, "event": ev,
                         "market_id": f"{ev}-team{i}", "residual_skill": resid,
                         "timestamp": day * DAY, "res_ts_proxy": (day + 30) * DAY})
    return pd.DataFrame(rows)


def _one_event_wallet():
    """40 bets, all on ONE championship, spread over 40 distinct team markets —
    the shape that dominates the gate corpus."""
    return _wallet_bets("w_one_event", ["win-2025-nba-finals"], 40, 0.30)


def _many_event_wallet():
    """40 bets across 20 genuinely independent games."""
    return _wallet_bets("w_many_events", [f"nba-g{i}-2026-01-{i:02d}" for i in range(1, 21)],
                        2, 0.30)


# ---------------------------------------------------------------------------
# The crux: one championship is one draw, not forty
# ---------------------------------------------------------------------------

def test_one_event_wallet_is_certified_under_the_gates_setting():
    """Baseline for the contrast below: with market clusters and no event gates —
    the setting the sports gate used — a whole outright field reads as 40
    independent draws and the wallet is certified."""
    out = validate_slow(_one_event_wallet(), baselines={}, cfg=_cfg(),
                        pre_residualized=True, cluster_col="market_id",
                        complex_gates=False, complex_col="event",
                        cfg_section=CFG_SECTION)
    assert bool(out.iloc[0]["edge_persisted"])
    assert out.iloc[0]["out_markets"] == 20


def test_one_event_wallet_is_rejected_at_the_event_unit():
    """Corrected: one resolution event cannot establish significance no matter
    how many of its markets were bought. Rejection is BY RULE — the bootstrap has
    a single cluster and returns NaN."""
    out = validate_slow(_one_event_wallet(), baselines={}, cfg=_cfg(),
                        pre_residualized=True, cluster_col=CLUSTER_COL,
                        complex_gates=True, complex_col=CLUSTER_COL,
                        cfg_section=CFG_SECTION)
    row = out.iloc[0]
    assert not bool(row["edge_persisted"])
    assert np.isnan(row["cluster_p"])
    assert row["out_complexes"] == 1
    assert not bool(row["complex_concentration_ok"])


def test_genuinely_broad_wallet_survives_the_event_unit():
    """The correction must not simply kill everything: real game breadth passes."""
    out = validate_slow(_many_event_wallet(), baselines={}, cfg=_cfg(),
                        pre_residualized=True, cluster_col=CLUSTER_COL,
                        complex_gates=True, complex_col=CLUSTER_COL,
                        cfg_section=CFG_SECTION)
    row = out.iloc[0]
    assert bool(row["edge_persisted"])
    assert row["out_complexes"] >= 3
    assert row["cluster_p"] < 0.05


def test_event_breadth_does_not_rescue_a_small_edge():
    """The scoped 10c floor still bites: broad but 3c is not certified."""
    bets = _wallet_bets("w_small", [f"nba-g{i}-2026-01-{i:02d}" for i in range(1, 21)],
                        2, 0.03)
    out = validate_slow(bets, baselines={}, cfg=_cfg(), pre_residualized=True,
                        cluster_col=CLUSTER_COL, complex_gates=True,
                        complex_col=CLUSTER_COL, cfg_section=CFG_SECTION)
    row = out.iloc[0]
    assert not bool(row["magnitude_ok"])
    assert not bool(row["edge_persisted"])


# ---------------------------------------------------------------------------
# Scoping — the sports block must not leak into Project 1 or the slow arm
# ---------------------------------------------------------------------------

def test_sports_block_is_the_one_being_read():
    """The fixture config sets scoring.slow.min_skill_edge to 0.99 and the global
    to 0.02. A wallet at +0.30 is certified only if the 0.10 SPORTS floor is what
    gated it."""
    cfg = _cfg()
    assert cfg["scoring"]["slow"]["min_skill_edge"] == 0.99
    assert cfg["scoring"]["min_skill_edge"] == 0.02
    out = validate_slow(_many_event_wallet(), baselines={}, cfg=cfg,
                        pre_residualized=True, cluster_col=CLUSTER_COL,
                        complex_gates=True, complex_col=CLUSTER_COL,
                        cfg_section=CFG_SECTION)
    assert bool(out.iloc[0]["edge_persisted"])

    # and the same call WITHOUT the section override reads scoring.slow (0.99)
    out_slow = validate_slow(_many_event_wallet(), baselines={}, cfg=cfg,
                             pre_residualized=True, cluster_col=CLUSTER_COL,
                             complex_gates=True, complex_col=CLUSTER_COL)
    assert not bool(out_slow.iloc[0]["edge_persisted"])


def test_project1_output_is_bit_identical_with_and_without_the_sports_block():
    """The scoped block must be INERT for Project 1, not merely unread by
    inspection. Same ledger through `compute_oos_validation` twice — once with a
    sports block whose every threshold is deliberately extreme — and the two
    frames must be equal, column for column."""
    from src.validate import compute_oos_validation

    rng = np.random.default_rng(7)
    n = 600
    ledger = pd.DataFrame({
        "wallet": [f"w{i % 12}" for i in range(n)],
        "market_id": [f"m{i % 40}" for i in range(n)],
        "side": ["BUY"] * n,
        "resolved": [True] * n,
        "entry_price": rng.uniform(0.05, 0.95, n),
        "resolved_value": rng.integers(0, 2, n).astype(float),
        "timestamp": np.sort(rng.integers(1_700_000_000, 1_800_000_000, n)),
    })
    base = {"scoring": {"oos_split": 0.5, "min_skill_edge": 0.02, "min_bets_per_half": 10,
                        "oos_significance_alpha": 0.05, "price_baseline_bins": 20,
                        "oos_bootstrap_resamples": 200, "min_oos_markets": 5}}
    with_sports = {"scoring": dict(base["scoring"],
                                   sports={"min_skill_edge": 0.99, "min_oos_markets": 999,
                                           "min_oos_complexes": 999, "min_bets_per_half": 999})}

    a = compute_oos_validation(ledger, base)
    b = compute_oos_validation(ledger, with_sports)
    pd.testing.assert_frame_equal(a, b)
    assert int(a["edge_persisted"].sum()) == int(b["edge_persisted"].sum())


def test_live_config_keeps_the_arms_separated():
    """The shipped config must carry the 10c sports floor WITHOUT moving the
    global 2c one — the invariant the whole scoping exists to protect."""
    from src.common import load_config
    scoring = load_config().get("scoring", {})
    assert scoring.get("min_skill_edge") == 0.02
    assert scoring.get(CFG_SECTION, {}).get("min_skill_edge") == 0.10
    assert scoring.get(CFG_SECTION, {}).get("min_oos_complexes") == 3


# ---------------------------------------------------------------------------
# Baseline modes
# ---------------------------------------------------------------------------

def test_new_validate_slow_parameters_are_behaviour_preserving():
    """The sports arm reuses `validate_slow` by adding two parameters. Their
    defaults must reproduce the pre-change behaviour EXACTLY, or the frozen
    forecaster cohort silently moves. Same frame, called the old way (defaults)
    and the new way (explicitly passing what used to be hardcoded)."""
    from src.slow_validate import COMPLEX_COL

    bets = _wallet_bets("w", [f"nba-g{i}-2026-01-{i:02d}" for i in range(1, 21)], 2, 0.30)
    bets["niche_l1"] = ["fam_a", "fam_b"] * (len(bets) // 2)
    cfg = {"scoring": {"oos_split": 0.5, "oos_significance_alpha": 0.05,
                       "oos_bootstrap_resamples": 500,
                       "slow": {"min_skill_edge": 0.10}}}

    for cluster, gates in (("market_id", False), (COMPLEX_COL, True)):
        old = validate_slow(bets, baselines={}, cfg=cfg, pre_residualized=True,
                            cluster_col=cluster, complex_gates=gates)
        new = validate_slow(bets, baselines={}, cfg=cfg, pre_residualized=True,
                            cluster_col=cluster, complex_gates=gates,
                            complex_col=COMPLEX_COL, cfg_section="slow")
        pd.testing.assert_frame_equal(old, new)


def test_magnitude_sensitivity_is_diagnostic_and_monotone():
    """A null result must be readable: 'nobody cleared 10c' and 'nobody cleared
    2c either' are different findings. The counts must fall as the floor rises,
    and the function must never change the gate itself."""
    from src.sports_validate import magnitude_sensitivity

    t = pd.DataFrame({
        "wallet": [f"w{i}" for i in range(5)],
        "out_sample_skill": [0.30, 0.12, 0.07, 0.03, 0.01],
        "candidate": True, "significant": True, "markets_ok": True,
        "concentration_ok": True, "complex_concentration_ok": True,
        "magnitude_ok": [True, True, False, False, False],
        "edge_persisted": [True, True, False, False, False],
    })
    s = magnitude_sensitivity(t).set_index("floor")["n_clearing"]
    assert list(s) == sorted(s, reverse=True)
    assert s[0.02] == 4 and s[0.05] == 3 and s[0.10] == 2 and s[0.15] == 1
    # unchanged input — a diagnostic never mutates the table it reads
    assert int(t["edge_persisted"].sum()) == 2


def test_baseline_modes_are_nested_coarse_to_fine():
    assert BASELINE_MODES[STANDARD_BASELINE] == ("niche_l1", "niche_l2")
    for coarse, fine in (("league", "league_form"), ("league_form", "sub_form")):
        assert BASELINE_MODES[coarse] == BASELINE_MODES[fine][:len(BASELINE_MODES[coarse])]


def test_residualize_for_rejects_unknown_mode():
    with pytest.raises(ValueError):
        residualize_for(pd.DataFrame(), "not_a_mode")


def test_residualize_for_produces_residual_skill():
    n = 400
    rng = np.random.default_rng(0)
    bets = pd.DataFrame({
        "wallet": [f"w{i % 20}" for i in range(n)],
        "entry_price": rng.uniform(0.05, 0.95, n),
        "resolved_value": rng.integers(0, 2, n).astype(float),
        "niche_l1": ["lg_nba"] * n,
        "niche_l2": ["lg_nba|spread"] * n,
        "niche_l3": ["lg_nba|spread|home"] * (n // 2) + ["lg_nba|spread|away"] * (n // 2),
    })
    for mode in BASELINE_MODES:
        out, info = residualize_for(bets, mode)
        assert "residual_skill" in out.columns
        assert out["residual_skill"].notna().all()
        assert len(out) == n
