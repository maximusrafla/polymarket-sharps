"""Tests for the sports forward freeze (src/sports_forward.py).

A freeze is a pre-registration: its value is entirely in the properties that make
it unfalsifiable-after-the-fact. Those are what is pinned here — the selection
rules parse against the real validator schema, the ladder is genuinely nested,
the artifact refuses to be overwritten, an overwrite is recorded with the number
of forward observations that existed at the time, and the scoreboard says
"awaiting data" out loud instead of shipping an empty table.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.sports_forward as sf


def _validator_like_table():
    """A frame with the columns `slow_validate.validate_slow` actually emits, so a
    typo in a tier selector fails here rather than silently freezing nobody."""
    rows = []
    for i in range(9):
        rows.append({
            "wallet": f"w{i}", "n_bets": 100,
            "in_sample_skill": 0.2, "out_sample_skill": 0.30 - 0.03 * i,
            "out_n": 50, "out_markets": 20, "cluster_p": 0.01 * (i + 1),
            "eff_breadth": 10.0, "entry_days": 10, "resolution_days": 10,
            "candidate": True, "significant": i < 3, "magnitude_ok": i < 6,
            "markets_ok": True, "concentration_ok": True,
            "out_complexes": 8, "eff_breadth_complex": 6.0,
            "entry_days_complex": 8, "resolution_days_complex": 8,
            "top_complex": "nba-x-y-2026-01-01", "top_complex_share": 0.2,
            "complex_concentration_ok": i < 7,
            "edge_persisted": i < 3,
        })
    return pd.DataFrame(rows)


def test_tier_selectors_parse_and_produce_a_nested_ladder():
    t = _validator_like_table()
    members = {tier: set(t.query(spec["selector"])["wallet"])
               for tier, spec in sf.TIER_SPECS.items()}
    assert members["s_core"] == {"w0", "w1", "w2"}
    assert members["s_core"] < members["s_wide"] < members["s_all"]
    assert sf.HEADLINE_TIER == "s_core"
    assert sf.TIER_SPECS[sf.HEADLINE_TIER]["headline"] is True
    assert sum(s["headline"] for s in sf.TIER_SPECS.values()) == 1


def test_scoring_rule_pins_the_event_unit_and_the_crowd_statistic():
    """The rule is the pre-commitment. If someone later scores bet-weighted, or
    per-wallet, or at market level, this text is what they violated."""
    rule = sf.SCORING_RULE
    assert "EVENT-WEIGHTED" in rule
    assert "RESOLUTION EVENTS" in rule
    assert "one championship a whole outright field settles on" in rule
    assert "ONE observation, not one per team" in rule
    assert "WITHOUT refitting" in rule
    assert "never wallet-by-wallet" in rule
    assert "undefined below two events" in rule


def _freeze_with_stub(monkeypatch, tmp_path, frozen=None, ts=1_800_000_000):
    frozen = _validator_like_table() if frozen is None else frozen
    frozen = frozen.assign(pre_freeze_bets=100, pre_freeze_events=8, pre_freeze_leagues=3)
    for tier, spec in sf.TIER_SPECS.items():
        frozen[tier] = frozen["wallet"].isin(set(frozen.query(spec["selector"])["wallet"]))
    stats = {"tiers": {t: {"n_wallets": int(frozen[t].sum()),
                           "median_out_sample_skill": 0.2,
                           "median_held_out_events": 8.0,
                           "distinct_pre_freeze_events": 40,
                           "distinct_pre_freeze_leagues": 5,
                           "pre_freeze_bets": 900} for t in sf.TIER_ORDER},
             "subset_of": {}}
    baseline = {"edges": [0.0, 0.5, 1.0], "levels": ["niche_l1"], "cells": {},
                "ks": {}, "grand": 0.5, "min_bin": 50, "n_bets": 900}
    monkeypatch.setattr(sf, "build_tiers", lambda cfg=None: (frozen, baseline, stats))
    monkeypatch.setattr(sf, "FROZEN_SET_PATH", tmp_path / "sports_frozen_set.parquet")
    monkeypatch.setattr(sf, "FREEZE_MANIFEST_PATH", tmp_path / "sports_freeze_manifest.json")
    monkeypatch.setattr(sf, "FORWARD_TRADES_PATH", tmp_path / "forward_trades.parquet")
    monkeypatch.setattr(sf, "SCOREBOARD_MD", tmp_path / "board.md")
    monkeypatch.setattr(sf, "SCOREBOARD_PARQUET", tmp_path / "board.parquet")
    return sf.freeze(cfg={"scoring": {}}, freeze_ts=ts)


def test_freeze_records_the_pre_registration(monkeypatch, tmp_path):
    m = _freeze_with_stub(monkeypatch, tmp_path)
    assert m["arm"] == "sports"
    assert m["freeze_ts"] == 1_800_000_000
    assert m["forward_observations_at_freeze"] == 0     # the whole point
    assert m["gates"]["cluster_unit"] == "event"
    assert m["gates"]["min_skill_edge"] == 0.10
    assert m["gates"]["global_min_skill_edge_untouched"] == 0.02
    assert m["sports_event_version"] and m["niche_partition_version"]
    assert m["baseline"]["levels"] == ["niche_l1"]      # persisted, not refitted later
    assert (tmp_path / "sports_frozen_set.parquet").exists()
    assert json.loads((tmp_path / "sports_freeze_manifest.json").read_text())["arm"] == "sports"


def test_freeze_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    _freeze_with_stub(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        _freeze_with_stub(monkeypatch, tmp_path)


def test_forced_refreeze_is_logged_with_the_observation_count(monkeypatch, tmp_path):
    """An amendment made after forward outcomes exist must be visibly marked as
    NOT legitimate pre-registration rather than quietly accepted."""
    _freeze_with_stub(monkeypatch, tmp_path)
    pd.DataFrame({"timestamp": [1, 2, 3]}).to_parquet(tmp_path / "forward_trades.parquet")

    frozen = _validator_like_table()
    for tier, spec in sf.TIER_SPECS.items():
        frozen[tier] = frozen["wallet"].isin(set(frozen.query(spec["selector"])["wallet"]))
    frozen = frozen.assign(pre_freeze_bets=100, pre_freeze_events=8, pre_freeze_leagues=3)
    monkeypatch.setattr(sf, "build_tiers", lambda cfg=None: (
        frozen,
        {"edges": [], "levels": [], "cells": {}, "ks": {}, "grand": 0.5,
         "min_bin": 50, "n_bets": 0},
        {"tiers": {t: {"n_wallets": 1, "median_out_sample_skill": 0.2,
                       "median_held_out_events": 8.0,
                       "distinct_pre_freeze_events": 40,
                       "distinct_pre_freeze_leagues": 5,
                       "pre_freeze_bets": 900} for t in sf.TIER_ORDER},
         "subset_of": {}}))
    m = sf.freeze(cfg={"scoring": {}}, force=True, amend_reason="test")
    amend = m["amendments"][-1]
    assert amend["forward_observations_at_amendment"] == 3
    assert amend["legitimate_pre_registration"] is False
    assert amend["freeze_ts_preserved"] is True         # cutoff never moves


def test_awaiting_scoreboard_says_so_out_loud(monkeypatch, tmp_path):
    m = _freeze_with_stub(monkeypatch, tmp_path)
    frozen = pd.read_parquet(tmp_path / "sports_frozen_set.parquet")
    sf._await_data(frozen, m, "no post-freeze trades yet")
    md = (tmp_path / "board.md").read_text()
    assert "AWAITING FORWARD DATA" in md
    assert "no post-freeze trades yet" in md
    assert "**s_core**" in md            # headline is marked as such
    assert "EVENT-WEIGHTED" in md        # the verbatim rule travels with the board
    board = pd.read_parquet(tmp_path / "board.parquet")
    assert set(board["tier"]) == set(sf.TIER_ORDER)
    assert board["edge_event_weighted"].isna().all()


def test_fast_split_is_tighter_than_the_slow_arms():
    """Sports resolves in hours-to-days; a 14-day split would put everything in
    one stratum and say nothing."""
    from src.slow_forward import FAST_RESOLUTION_DAYS as SLOW_SPLIT
    assert sf.FAST_RESOLUTION_DAYS < SLOW_SPLIT
    assert sf.FAST_RESOLUTION_DAYS == 2


def test_sports_freeze_is_a_separate_artifact_from_the_slow_one():
    """Bolting sports onto the slow manifest would either break its amendment
    rule (647 observations exist against it) or force a meaningless shared
    cutoff."""
    from src.slow_forward import FREEZE_MANIFEST_PATH as SLOW_MANIFEST
    assert sf.FREEZE_MANIFEST_PATH != SLOW_MANIFEST
    assert sf.FROZEN_SET_PATH.name == "sports_frozen_set.parquet"
