"""Tests for the sports `s_sig2c` arm (src/sports_sig2c.py).

This arm exists because a mis-scoped magnitude floor excluded a certifiable
cohort (docs/redteam_audit_2026-07-26.md §1). Re-opening a closed verdict is
exactly the move that goes wrong when it is done sloppily, so what is pinned
here is the discipline rather than the numbers:

  * the cohort is selected WITHOUT the magnitude gate but with every other gate
    still applied — a wallet that fails significance or concentration can never
    enter, no matter how large its edge;
  * the headline tier is the BASELINE-ROBUST subset, i.e. a stricter standard
    than the parent arm applied, not a looser one;
  * the tiers are genuinely nested, so a wider tier cannot smuggle in a name;
  * the parent freeze artifact is never written — its immutability is the whole
    reason this is a separate arm;
  * the baseline is copied verbatim from the parent, so no refit can drift the
    forward number;
  * the artifact refuses to be overwritten, and any overwrite is recorded with
    the number of forward observations that existed at the time.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.sports_sig2c as s2


def _variant_table(edges=None, ps=None, n=8):
    """A frame with the columns the sports validator actually emits."""
    edges = edges if edges is not None else [0.30 - 0.03 * i for i in range(n)]
    ps = ps if ps is not None else [0.01 * (i + 1) for i in range(n)]
    rows = []
    for i in range(n):
        rows.append({
            "wallet": f"0xw{i}", "n_bets": 100,
            "in_sample_skill": 0.02, "out_sample_skill": edges[i],
            "out_n": 50, "out_markets": 20, "cluster_p": ps[i],
            "eff_breadth": 10.0, "entry_days": 10, "resolution_days": 10,
            "candidate": True, "significant": True, "magnitude_ok": edges[i] >= 0.10,
            "markets_ok": True, "concentration_ok": True,
            "out_complexes": 8, "eff_breadth_complex": 6.0,
            "entry_days_complex": 8, "resolution_days_complex": 8,
            "top_complex": "nba-x-y-2026-01-01", "top_complex_share": 0.2,
            "complex_concentration_ok": True, "edge_persisted": edges[i] >= 0.10,
        })
    return pd.DataFrame(rows)


# --- selection discipline --------------------------------------------------

def test_magnitude_gate_is_the_only_gate_omitted():
    """The arm corrects ONE parameter. Every other gate must still be applied —
    otherwise this is table-filling rather than a unit correction."""
    assert "magnitude_ok" not in s2.NON_MAGNITUDE_GATES
    for gate in ("candidate", "significant", "markets_ok",
                 "concentration_ok", "complex_concentration_ok"):
        assert gate in s2.NON_MAGNITUDE_GATES


def test_a_huge_edge_cannot_enter_without_significance():
    """A 50c edge that fails the event-clustered bootstrap stays out. The floor
    was wrong; the significance test was not."""
    t = _variant_table()
    t.loc[0, "out_sample_skill"] = 0.50
    t.loc[0, "significant"] = False
    keep = s2._gate_mask(t) & (t["out_sample_skill"] >= s2.MIN_SKILL_EDGE)
    assert "0xw0" not in set(t[keep]["wallet"])


def test_a_huge_edge_cannot_enter_without_event_breadth():
    t = _variant_table()
    t.loc[1, "out_sample_skill"] = 0.50
    t.loc[1, "complex_concentration_ok"] = False
    keep = s2._gate_mask(t) & (t["out_sample_skill"] >= s2.MIN_SKILL_EDGE)
    assert "0xw1" not in set(t[keep]["wallet"])


def test_floor_is_project_ones_global_value_not_a_bespoke_number():
    """The defence against threshold-hunting is that 2c was already the repo's
    global floor (config `scoring.min_skill_edge`, Project 1) before this arm
    existed — and that the scoped sports 0.10 it replaces is documented in the
    config itself as borrowed from the slow arm's rationale."""
    from src.common import load_config
    cfg = load_config()
    assert s2.MIN_SKILL_EDGE == float(cfg["scoring"]["min_skill_edge"]) == 0.02
    # the value this arm deliberately does NOT use
    assert float(cfg["scoring"]["sports"]["min_skill_edge"]) == 0.10
    assert s2.MIN_SKILL_EDGE < float(cfg["scoring"]["sports"]["min_skill_edge"])


# --- the headline tier is STRICTER, not looser -----------------------------

def test_headline_tier_requires_every_baseline_variant():
    """s2_core must survive all five saved baselines. A wallet that is only
    significant under the standard fit is demoted to s2_twelve, never headlined."""
    assert s2.TIER_SPECS[s2.HEADLINE_TIER]["headline"] is True
    assert len(s2.BASELINE_VARIANTS) == 5
    assert s2.STANDARD_VARIANT in s2.BASELINE_VARIANTS


def test_baseline_fragile_wallet_is_excluded_from_core(monkeypatch):
    """Fragility under ANY variant demotes. This is the check the parent arm ran
    only for its single survivor."""
    strong = _variant_table(edges=[0.05] * 3, ps=[0.01] * 3, n=3)
    fragile = strong.copy()
    fragile.loc[2, "cluster_p"] = 0.20          # fails under one variant only
    variants = {name: (fragile if name == "sub_form_40bins" else strong).copy()
                for name in s2.BASELINE_VARIANTS}
    monkeypatch.setattr(s2, "load_variants", lambda: variants)
    monkeypatch.setattr(s2, "held_out_cadence", lambda ws: pd.DataFrame(
        {"wallet": ws, "held_out_bets": 10, "held_out_span_days": 10.0,
         "held_out_bets_per_day": 1.0}))
    monkeypatch.setattr(s2, "PARENT_FROZEN_SET_PATH", s2.PROCESSED_DIR / "does_not_exist")
    frozen, stats, members = s2.build_tiers()
    assert members["s2_twelve"] == {"0xw0", "0xw1", "0xw2"}
    assert members["s2_core"] == {"0xw0", "0xw1"}


# --- nesting ---------------------------------------------------------------

def test_tiers_are_nested(monkeypatch):
    t = _variant_table(edges=[0.05, 0.04, 0.03], ps=[0.01, 0.01, 0.01], n=3)
    monkeypatch.setattr(s2, "load_variants",
                        lambda: {n: t.copy() for n in s2.BASELINE_VARIANTS})
    monkeypatch.setattr(s2, "held_out_cadence", lambda ws: pd.DataFrame(
        {"wallet": ws, "held_out_bets": 10, "held_out_span_days": 10.0,
         "held_out_bets_per_day": [1.0, 500.0, 500.0]}))
    monkeypatch.setattr(s2, "PARENT_FROZEN_SET_PATH", s2.PROCESSED_DIR / "does_not_exist")
    _, stats, members = s2.build_tiers()
    assert members["s2_core"] <= members["s2_twelve"]
    assert members["s2_slow"] <= members["s2_twelve"]
    assert members["s2_slow"] == {"0xw0"}
    assert "s2_twelve" in stats["subset_of"]["s2_core"]


def test_slow_tier_uses_cadence_not_edge(monkeypatch):
    """s2_slow is about whether a follower could ACT, so it must key on cadence
    alone — a big edge must not buy a seat in it."""
    t = _variant_table(edges=[0.50, 0.03], ps=[0.001, 0.01], n=2)
    monkeypatch.setattr(s2, "load_variants",
                        lambda: {n: t.copy() for n in s2.BASELINE_VARIANTS})
    monkeypatch.setattr(s2, "held_out_cadence", lambda ws: pd.DataFrame(
        {"wallet": ws, "held_out_bets": 10, "held_out_span_days": 10.0,
         "held_out_bets_per_day": [900.0, 2.0]}))
    monkeypatch.setattr(s2, "PARENT_FROZEN_SET_PATH", s2.PROCESSED_DIR / "does_not_exist")
    _, _, members = s2.build_tiers()
    assert members["s2_slow"] == {"0xw1"}


def test_empty_cohort_refuses_to_freeze_an_empty_artifact(monkeypatch):
    t = _variant_table(edges=[0.001, 0.001], ps=[0.01, 0.01], n=2)
    monkeypatch.setattr(s2, "load_variants",
                        lambda: {n: t.copy() for n in s2.BASELINE_VARIANTS})
    with pytest.raises(RuntimeError, match="nothing to pre-register"):
        s2.build_tiers()


# --- the parent artifact is immutable --------------------------------------

def test_parent_paths_are_distinct_from_this_arms_paths():
    """Every written path must differ from the parent arm's, or a freeze here
    would silently clobber a running pre-registration."""
    import src.sports_forward as sf
    written = {s2.FROZEN_SET_PATH, s2.FREEZE_MANIFEST_PATH,
               s2.SCOREBOARD_PARQUET, s2.SCOREBOARD_MD,
               s2.FORWARD_TRADES_PATH, s2.FORWARD_RESOLUTIONS_PATH}
    parent = {sf.FROZEN_SET_PATH, sf.FREEZE_MANIFEST_PATH,
              sf.SCOREBOARD_PARQUET, sf.SCOREBOARD_MD,
              sf.FORWARD_TRADES_PATH, sf.FORWARD_RESOLUTIONS_PATH}
    assert written.isdisjoint(parent)
    assert s2.PARENT_MANIFEST_PATH == sf.FREEZE_MANIFEST_PATH


def test_freeze_refuses_to_overwrite_and_logs_amendments(tmp_path, monkeypatch):
    manifest_path = tmp_path / "sig2c_manifest.json"
    frozen_path = tmp_path / "sig2c_frozen.parquet"
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(
        {"baseline": {"levels": ["league"], "grand": 0.5},
         "freeze_utc": "2026-07-25T23:25:48Z", "git_commit": "deadbeef"}))
    monkeypatch.setattr(s2, "FREEZE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(s2, "FROZEN_SET_PATH", frozen_path)
    monkeypatch.setattr(s2, "PARENT_MANIFEST_PATH", parent_path)
    monkeypatch.setattr(s2, "FORWARD_TRADES_PATH", tmp_path / "fwd.parquet")

    t = _variant_table(edges=[0.05, 0.04], ps=[0.01, 0.01], n=2)
    monkeypatch.setattr(s2, "load_variants",
                        lambda: {n: t.copy() for n in s2.BASELINE_VARIANTS})
    monkeypatch.setattr(s2, "held_out_cadence", lambda ws: pd.DataFrame(
        {"wallet": ws, "held_out_bets": 10, "held_out_span_days": 10.0,
         "held_out_bets_per_day": 1.0}))
    monkeypatch.setattr(s2, "PARENT_FROZEN_SET_PATH", tmp_path / "nope.parquet")

    m = s2.freeze(freeze_ts=1_700_000_000)
    assert m["freeze_ts"] == 1_700_000_000
    assert m["amendments"] == []
    # the baseline is the parent's, verbatim
    assert m["baseline"] == {"levels": ["league"], "grand": 0.5}

    with pytest.raises(SystemExit, match="already exists"):
        s2.freeze()

    m2 = s2.freeze(force=True, amend_reason="test", freeze_ts=1_700_000_000)
    assert len(m2["amendments"]) == 1
    assert m2["amendments"][0]["reason"] == "test"
    assert m2["amendments"][0]["forward_observations_at_freeze"
                               if "forward_observations_at_freeze"
                               in m2["amendments"][0]
                               else "forward_observations_at_amendment"] == 0
    assert m2["amendments"][0]["legitimate_pre_registration"] is True


def test_manifest_records_the_selection_after_the_fact_limitation(tmp_path, monkeypatch):
    """The single most important honesty property of this artifact: it must say
    out loud that its floor was chosen after the parent table existed."""
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps({"baseline": {"grand": 0.5}}))
    monkeypatch.setattr(s2, "FREEZE_MANIFEST_PATH", tmp_path / "m.json")
    monkeypatch.setattr(s2, "FROZEN_SET_PATH", tmp_path / "f.parquet")
    monkeypatch.setattr(s2, "PARENT_MANIFEST_PATH", parent_path)
    monkeypatch.setattr(s2, "FORWARD_TRADES_PATH", tmp_path / "fwd.parquet")
    monkeypatch.setattr(s2, "PARENT_FROZEN_SET_PATH", tmp_path / "nope.parquet")
    t = _variant_table(edges=[0.05], ps=[0.01], n=1)
    monkeypatch.setattr(s2, "load_variants",
                        lambda: {n: t.copy() for n in s2.BASELINE_VARIANTS})
    monkeypatch.setattr(s2, "held_out_cadence", lambda ws: pd.DataFrame(
        {"wallet": ws, "held_out_bets": 10, "held_out_span_days": 10.0,
         "held_out_bets_per_day": 1.0}))
    m = s2.freeze(freeze_ts=1)
    limits = " ".join(m["known_limits"]).lower()
    assert "selection-after-the-fact" in limits
    assert "copyability" in limits
    assert m["selection"]["gate_deliberately_omitted"].startswith("magnitude_ok")
    assert m["baseline_source"].startswith("copied verbatim")


# --- scoreboard ------------------------------------------------------------

def test_awaiting_data_board_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(s2, "SCOREBOARD_MD", tmp_path / "b.md")
    monkeypatch.setattr(s2, "SCOREBOARD_PARQUET", tmp_path / "b.parquet")
    manifest = {
        "freeze_utc": "2026-07-26T06:00:00Z", "git_commit": "abcdef1234",
        "sports_event_version": "1.0.0", "headline_tier": s2.HEADLINE_TIER,
        "scoring_rule": "rule.", "known_limits": ["limit one"],
        "tiers": {t: {"n_wallets": 3, "headline": t == s2.HEADLINE_TIER,
                      "distinct_pre_freeze_events": 100} for t in s2.TIER_ORDER},
    }
    s2._await_data(manifest, "no post-freeze trades yet")
    text = (tmp_path / "b.md").read_text()
    assert "AWAITING FORWARD DATA" in text
    assert "does not amend" in text
    assert f"**{s2.HEADLINE_TIER}**" in text
