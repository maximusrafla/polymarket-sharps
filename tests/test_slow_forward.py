"""Tests for the pre-registered freeze and forward-scoring harness
(src/slow_forward.py).

The freeze is the one artifact in this project whose VALUE comes entirely from
not being changed after the fact, so what is pinned here is mostly integrity
properties: the manifest records everything needed to reproduce and re-score,
the cohort semantics cannot be misread, and re-freezing is refused.

No test touches the network.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.slow_forward as sf
from src.slow_baseline import expected_outcome_hier, fit_hierarchical_baseline


@pytest.fixture
def frozen_env(tmp_path, monkeypatch):
    """A tiny hand-built cohort standing in for the real pipeline output."""
    frozen = pd.DataFrame({
        "wallet": ["wA", "wB", "wC", "wD"],
        "out_sample_skill": [0.30, 0.20, 0.15, 0.05],
        "out_n": [50, 40, 30, 20],
        "cluster_p": [0.001, 0.01, 0.02, 0.40],
        "out_complexes": [5, 4, 3, 2],
        "eff_breadth_complex": [4.0, 3.5, 3.1, 1.2],
        "edge_persisted": [True, True, True, False],
        "cohort": ["primary", "primary", "secondary", "wider"],
        "t12": [True, True, False, False],
        "t30": [True, True, True, False],
        "t52": [True, False, True, False],
        "t87": [True, True, True, False],
        "t699": [True, True, True, True],
        "top_narrative_broad": ["mideast_escalation", "sports", "sports", "sports"],
        "top_narrative_share_broad": [0.5, 0.4, 0.9, 0.95],
        "top_narrative_share_strict": [0.5, 0.4, 0.9, 0.95],
        "single_narrative_flag": [False, False, True, True],
        "n_narratives_broad": [4, 5, 2, 1],
        "eff_narratives_broad": [3.0, 3.5, 1.2, 1.0],
    })
    fit_rows = pd.DataFrame({
        "wallet": [f"w{i}" for i in range(400)],
        "category": ["other"] * 400,
        "niche_l1": ["geo_iran" if i % 2 else "sports" for i in range(400)],
        "niche_l2": ["geo_iran|deadline" if i % 2 else "sports|match_line" for i in range(400)],
        "entry_price": [0.3] * 400,
        "resolved_value": [float(i % 4 == 0) for i in range(400)],
    })
    baseline = fit_hierarchical_baseline(
        fit_rows, levels=("category", "niche_l1", "niche_l2"), min_bin=10)

    monkeypatch.setattr(sf, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(sf, "FROZEN_SET_PATH", tmp_path / "slow_frozen_set.parquet")
    monkeypatch.setattr(sf, "FREEZE_MANIFEST_PATH", tmp_path / "slow_freeze_manifest.json")
    monkeypatch.setattr(sf, "FORWARD_TRADES_PATH", tmp_path / "forward_trades.parquet")
    stats = {
        "tiers": {t: {"n_wallets": int(frozen[t].sum()),
                      "median_out_sample_skill": 0.2,
                      "effective_independent_narratives": 3.0,
                      "median_top_narrative_share": 0.5,
                      "single_narrative_wallets": 1,
                      "pre_freeze_slow_bets": 1000}
                  for t in sf.TIER_ORDER},
        "subset_of": {"t12": ["t30", "t87", "t699"]},
    }
    monkeypatch.setattr(sf, "build_tiers", lambda cfg=None: (frozen, baseline, stats))
    return tmp_path, frozen, baseline


def test_freeze_records_everything_needed_to_reproduce(frozen_env):
    _, _, _ = frozen_env
    m = sf.freeze(cfg={"scoring": {}}, freeze_ts=1_800_000_000)
    # provenance
    assert m["freeze_ts"] == 1_800_000_000
    assert m["freeze_utc"].endswith("Z")
    assert m["niche_partition_version"]
    assert m["narrative_map_version"]
    assert "git_commit" in m
    # every gate parameter, so the cohort can be re-derived
    for k in ("min_skill_edge", "min_oos_complexes", "min_eff_breadth_complex",
              "min_entry_days_complex", "min_resolution_days_complex",
              "oos_significance_alpha", "oos_split", "baseline", "cluster_unit",
              "complex_gates"):
        assert k in m["gates"], k
    assert m["gates"]["complex_gates"] is True
    # the fitted baseline travels with the freeze so scoring needs no refit
    assert m["baseline"]["levels"]
    assert m["scoring_rule"] == sf.SCORING_RULE


def test_manifest_is_json_serialisable_on_disk(frozen_env):
    tmp, _, _ = frozen_env
    sf.freeze(cfg={"scoring": {}}, freeze_ts=1_800_000_000)
    loaded = json.loads((tmp / "slow_freeze_manifest.json").read_text())
    assert loaded["headline_cohort"] == "primary"


def test_headline_tier_and_tier_membership_are_unambiguous(frozen_env):
    _, _, _ = frozen_env
    m = sf.freeze(cfg={"scoring": {}}, freeze_ts=1)
    assert m["headline_tier"] == "t12"
    assert [t for t in m["tiers"] if m["tiers"][t]["headline"]] == ["t12"]
    assert m["tiers"]["t12"]["n_wallets"] == 2
    assert m["tiers"]["t699"]["n_wallets"] == 4
    # the tiers are NOT one monotone ladder and the manifest says so, so nobody
    # reads t52 as a superset of t30
    assert m["tiers_are_nested"] is False
    assert "NOT one monotone ladder" in m["tier_note"] or "not one monotone" in m["tier_note"].lower()
    # step-7 keys still resolve for older readers
    assert m["cohorts"]["primary"]["equals_tier"] == "t12"


def test_known_limits_are_recorded_at_freeze_time(frozen_env):
    """The power limit has to be on the record BEFORE the forward data exists,
    not offered as an explanation afterwards."""
    m = sf.freeze(cfg={"scoring": {}}, freeze_ts=1)
    limits = " ".join(m["known_limits"]).lower()
    assert "narrative" in limits
    assert "months" in limits or "weeks" in limits
    assert "copyab" in limits          # identification != copyability
    # the measured "widening buys volume, not breadth" finding must be recorded
    # BEFORE forward data exists, so it cannot later look like a rationalisation
    assert "volume" in limits and "breadth" in limits
    assert "smaller than the retrospective" in limits


def test_scoring_rule_pins_the_pre_registration():
    r = sf.SCORING_RULE.lower()
    assert "after freeze_ts" in r
    assert "without" in r and "refit" in r          # frozen baseline, no refit
    assert "t12" in r and "headline" in r
    assert "event-weighted" in r                     # the aggregate statistic
    assert "resamples event complexes" in r          # clustered by event, not bet
    assert "placebo" in r
    assert "not re-frozen" in r


def test_frozen_set_round_trips_with_its_tier_columns(frozen_env):
    tmp, _, _ = frozen_env
    sf.freeze(cfg={"scoring": {}}, freeze_ts=1)
    got = pd.read_parquet(tmp / "slow_frozen_set.parquet")
    assert len(got) == 4
    for t in sf.TIER_ORDER:
        assert got[t].dtype == bool
    assert got["single_narrative_flag"].dtype == bool


def test_load_freeze_without_a_freeze_is_a_clear_error(frozen_env):
    with pytest.raises(FileNotFoundError, match="freeze"):
        sf.load_freeze()


def test_frozen_baseline_scores_unseen_forward_bets_without_refitting(frozen_env):
    """The core forward-scoring property: a bet that did not exist at freeze time
    is scored by the frozen curve, so the forward number cannot drift because the
    baseline moved."""
    _, _, baseline = frozen_env
    forward = pd.DataFrame({
        "category": ["other", "other"],
        "niche_l1": ["geo_iran", "never_seen_complex"],
        "niche_l2": ["geo_iran|deadline", "never_seen_complex|deadline"],
        "entry_price": [0.3, 0.3],
        "resolved_value": [1.0, 1.0],
    })
    exp = expected_outcome_hier(baseline, forward)
    assert np.isfinite(exp).all()
    # the unseen complex falls through to the parent rather than erroring
    assert 0.0 <= exp[1] <= 1.0
    resid = forward["resolved_value"].to_numpy() - exp
    assert np.isfinite(resid).all()


# ---------------------------------------------------------------------------
# Amendment integrity (step 8) — widening tiers is only pre-registration while
# no forward outcome has been observable
# ---------------------------------------------------------------------------

def test_amendment_preserves_the_freeze_ts_so_all_tiers_share_one_cutoff(frozen_env):
    tmp, _, _ = frozen_env
    sf.freeze(cfg={"scoring": {}}, freeze_ts=1_800_000_000)
    amended = sf.freeze(cfg={"scoring": {}}, amend_reason="add wider tiers")
    assert amended["freeze_ts"] == 1_800_000_000
    a = amended["amendments"][-1]
    assert a["freeze_ts_preserved"] is True
    assert a["reason"] == "add wider tiers"


def test_amendment_records_whether_it_was_still_pre_registration(frozen_env):
    """The audit trail that makes the claim checkable rather than asserted."""
    tmp, _, _ = frozen_env
    sf.freeze(cfg={"scoring": {}}, freeze_ts=1)
    clean = sf.freeze(cfg={"scoring": {}}, amend_reason="no forward data yet")
    a = clean["amendments"][-1]
    assert a["forward_observations_at_amendment"] == 0
    assert a["legitimate_pre_registration"] is True
    assert a["prior_scoring_rule"]              # the superseded rule is retained

    # ...and once forward observations exist, the record says so
    pd.DataFrame({"timestamp": [1, 2, 3]}).to_parquet(sf.FORWARD_TRADES_PATH)
    dirty = sf.freeze(cfg={"scoring": {}}, amend_reason="too late")
    b = dirty["amendments"][-1]
    assert b["forward_observations_at_amendment"] == 3
    assert b["legitimate_pre_registration"] is False


def test_amendments_accumulate_rather_than_overwrite(frozen_env):
    sf.freeze(cfg={"scoring": {}}, freeze_ts=1)
    sf.freeze(cfg={"scoring": {}}, amend_reason="first")
    m = sf.freeze(cfg={"scoring": {}}, amend_reason="second")
    assert [a["reason"] for a in m["amendments"]] == ["first", "second"]


def test_tier_specs_declare_exactly_one_headline():
    assert sum(1 for s in sf.TIER_SPECS.values() if s["headline"]) == 1
    assert sf.TIER_SPECS[sf.HEADLINE_TIER]["headline"] is True
    assert set(sf.TIER_ORDER) == set(sf.TIER_SPECS)


def test_wider_tiers_are_labelled_as_under_vetted():
    """t87/t699 must carry their own health warning in the manifest, so a good
    forward number there is never read as if it came from t12."""
    assert "vetted" in sf.TIER_SPECS["t699"]["description"].lower()
    assert "noise" in sf.TIER_SPECS["t699"]["description"].lower()
    assert "significance" in sf.TIER_SPECS["t87"]["description"].lower()
