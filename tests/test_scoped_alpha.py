"""Project 1's certification alpha is SCOPED — and provably inert elsewhere.

`scoring.oos_significance_alpha` is a GLOBAL key. Project 1 (`src/validate.py`)
tightened its certification threshold to 0.005 on 2026-07-26 after
`docs/persistence_fdr_hardened.md` measured the certified set's false-discovery
rate (41 wallets @ 55% FDR at alpha=0.05 -> 26 @ 20% at 0.005). The slow-forecaster
and sports arms read the SAME global key and have live, pre-registered forward
tests frozen against it, so the tightening had to be scoped to
`scoring.project1.oos_significance_alpha`.

This file is the proof, not the assertion:
  1. the resolver reads the scoped key, falls back to the global, then to the default;
  2. the scoped key actually moves Project 1's `edge_persisted`;
  3. metric D does NOT move (it stays on the global alpha — additive, never certifies);
  4. `validate_slow` (slow AND sports arms) is BIT-IDENTICAL with and without the block;
  5. the forward arms' recorded `gates` still read 0.05, matching the frozen manifests
     on disk — an artifact-level drift guard on the three running experiments.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.common import PROCESSED_DIR, load_config
from src.slow_validate import validate_slow
from src.validate import DEFAULT_OOS_ALPHA, certification_alpha, compute_oos_validation

DAY = 86400


# --------------------------------------------------------------------------- #
# 1. the resolver
# --------------------------------------------------------------------------- #

def test_certification_alpha_prefers_the_scoped_key():
    scoring = {"oos_significance_alpha": 0.05, "project1": {"oos_significance_alpha": 0.005}}
    assert certification_alpha(scoring) == 0.005


def test_certification_alpha_falls_back_to_the_global_key():
    """A config written before the scoped block existed must validate EXACTLY as
    it did — the fallback is what makes this change non-breaking."""
    assert certification_alpha({"oos_significance_alpha": 0.05}) == 0.05
    assert certification_alpha({"oos_significance_alpha": 0.01, "project1": {}}) == 0.01
    assert certification_alpha({}) == DEFAULT_OOS_ALPHA


def test_certification_alpha_ignores_other_arms_blocks():
    scoring = {"oos_significance_alpha": 0.05,
               "slow": {"oos_significance_alpha": 0.99},
               "sports": {"oos_significance_alpha": 0.99}}
    assert certification_alpha(scoring) == 0.05


# --------------------------------------------------------------------------- #
# 2./3. the scoped key moves certification, and ONLY certification
# --------------------------------------------------------------------------- #

def _ledger(seed=11, n_wallets=8, per_wallet=140):
    """A ledger with edge-carrying wallets: each wallet buys at 0.5 and wins at a
    per-wallet rate, one market per bet (so the cluster bootstrap has clusters).
    Wallet 0 is built to be a strong persister, the rest span weaker rates so the
    certified set is sensitive to alpha."""
    rng = np.random.default_rng(seed)
    rows = []
    ts = 1_700_000_000
    for w in range(n_wallets):
        rate = 0.50 + 0.03 * w          # 0.50 .. 0.71
        for i in range(per_wallet):
            rows.append({
                "wallet": f"w{w}", "market_id": f"w{w}_m{i}", "side": "BUY",
                "resolved": True, "entry_price": 0.5,
                "resolved_value": float(rng.random() < rate),
                "timestamp": ts + w * per_wallet * 60 + i * 60,
            })
    # ...plus one became-sharp wallet (bad early half, strong recent half) so the
    # metric D columns below are exercised rather than trivially all-False.
    for i in range(per_wallet):
        rate = 0.30 if i < per_wallet // 2 else 0.80
        rows.append({
            "wallet": "w_became_sharp", "market_id": f"ws_m{i}", "side": "BUY",
            "resolved": True, "entry_price": 0.5,
            "resolved_value": float(rng.random() < rate),
            "timestamp": ts + n_wallets * per_wallet * 60 + i * 60,
        })
    return pd.DataFrame(rows)


def _cfg(alpha_global=0.05, project1=None, **over):
    scoring = {"oos_split": 0.5, "min_bets_per_half": 10, "price_baseline_bins": 20,
               "oos_significance_alpha": alpha_global, "min_skill_edge": 0.02,
               "min_oos_markets": 5, "oos_bootstrap_resamples": 500,
               "regime_min_bets": 50}
    scoring.update(over)
    if project1 is not None:
        scoring["project1"] = project1
    return {"scoring": scoring}


def test_scoped_alpha_tightens_project1_certification():
    """Same ledger, same global alpha, only the scoped block differs -> fewer
    certified wallets. If validate.py were still reading the global this test
    would see no difference at all."""
    ledger = _ledger()
    loose = compute_oos_validation(ledger, _cfg())
    tight = compute_oos_validation(ledger, _cfg(project1={"oos_significance_alpha": 1e-9}))

    assert int(loose["edge_persisted"].sum()) > 0, "fixture must certify someone at 0.05"
    assert int(tight["edge_persisted"].sum()) == 0
    assert int(tight["edge_significant"].sum()) == 0
    # tightening can only REMOVE certifications, never add
    assert set(tight.loc[tight["edge_persisted"], "wallet"]) <= set(
        loose.loc[loose["edge_persisted"], "wallet"])


def test_shipped_alpha_thins_the_certified_set_without_emptying_it():
    """The production shape of the change: at the shipped 0.005 the certified set
    is a strict, non-empty subset of the 0.05 one — the same 41 -> 26 move the
    live ledger makes."""
    ledger = _ledger()
    loose = compute_oos_validation(ledger, _cfg())
    shipped = compute_oos_validation(ledger, _cfg(project1={"oos_significance_alpha": 0.005}))

    loose_set = set(loose.loc[loose["edge_persisted"], "wallet"])
    shipped_set = set(shipped.loc[shipped["edge_persisted"], "wallet"])
    assert shipped_set < loose_set and shipped_set
    # the wallets it drops are the ones with the weakest cluster p, by construction
    dropped = loose_set - shipped_set
    p = loose.set_index("wallet")["out_of_sample_cluster_p"]
    assert all(0.005 <= p[w] < 0.05 for w in dropped)


def test_tightening_never_drops_a_wallet_from_the_dataset():
    """CLAUDE.md's ironclad rule: every wallet keeps its row and its metrics; only
    the flag moves."""
    ledger = _ledger()
    loose = compute_oos_validation(ledger, _cfg())
    tight = compute_oos_validation(ledger, _cfg(project1={"oos_significance_alpha": 1e-9}))

    assert list(loose["wallet"]) == list(tight["wallet"])
    assert len(tight) == ledger["wallet"].nunique()
    # every non-gate column is untouched — the edge/p/n measurements do not depend
    # on the threshold they are compared against.
    for col in ("in_sample_edge", "out_of_sample_edge", "in_sample_residual_edge",
                "out_of_sample_residual_edge", "out_of_sample_residual_p",
                "out_of_sample_cluster_p", "in_sample_n", "out_of_sample_n",
                "out_of_sample_markets"):
        pd.testing.assert_series_equal(loose[col], tight[col])


def test_metric_d_is_unaffected_by_the_scoped_alpha():
    """Metric D deliberately keeps the GLOBAL alpha: it is additive descriptive
    metadata, it uses the per-bet t-test, and it never certifies. Tightening the
    certification gate must not silently move regime_flag / regime_watch /
    persisted_recent."""
    ledger = _ledger()
    loose = compute_oos_validation(ledger, _cfg())
    tight = compute_oos_validation(ledger, _cfg(project1={"oos_significance_alpha": 1e-9}))

    for col in ("regime_flag", "regime_watch", "persisted_recent"):
        pd.testing.assert_series_equal(loose[col], tight[col], check_names=False)
    # non-vacuous: metric D actually fired on this fixture (a became-sharp wallet
    # is flagged and D3-confirmed) — so an accidental re-wiring of D onto the
    # tightened alpha would be caught.
    assert bool(loose["regime_watch"].any())
    assert bool(loose["persisted_recent"].any())
    assert (loose["regime_flag"] != "insufficient").any()
    # ...and the certification flag DID move, so the comparison above has teeth
    assert int(loose["edge_persisted"].sum()) != int(tight["edge_persisted"].sum())


# --------------------------------------------------------------------------- #
# 4. non-interference with the slow / sports arms
# --------------------------------------------------------------------------- #

def _arm_frame(wallet="w", n_events=12, per_event=4, resid=0.30):
    rows = []
    for e in range(n_events):
        for i in range(per_event):
            day = e * 3 + i
            rows.append({"wallet": wallet, "event": f"ev{e}", "niche_l1": f"fam{e % 3}",
                         "market_id": f"ev{e}-m{i}", "residual_skill": resid,
                         "timestamp": day * DAY, "res_ts_proxy": (day + 30) * DAY})
    return pd.DataFrame(rows)


def _arm_cfg(project1=None):
    scoring = {
        "oos_split": 0.5, "oos_significance_alpha": 0.05, "oos_bootstrap_resamples": 500,
        "min_skill_edge": 0.02,
        "slow": {"min_skill_edge": 0.10, "min_oos_markets": 5, "min_eff_breadth": 3.0,
                 "min_entry_days": 3, "min_resolution_days": 3, "min_bets_per_half": 10,
                 "min_oos_complexes": 3, "min_eff_breadth_complex": 3.0,
                 "min_entry_days_complex": 3, "min_resolution_days_complex": 3},
    }
    scoring["sports"] = dict(scoring["slow"])
    if project1 is not None:
        scoring["project1"] = project1
    return {"scoring": scoring}


@pytest.mark.parametrize("section,cluster,gates", [("slow", "niche_l1", True),
                                                   ("sports", "event", True),
                                                   ("slow", "market_id", False)])
def test_slow_and_sports_arms_are_bit_identical_with_and_without_the_block(section, cluster, gates):
    """The scoped block must be INERT for the frozen arms, not merely unread by
    inspection — the same standard
    tests/test_sports_validate.py::test_project1_output_is_bit_identical_with_and_without_the_sports_block
    holds the sports block to. A project1 block with an absurd alpha must not
    perturb a single value of the arms' output."""
    frame = _arm_frame()
    without = validate_slow(frame, baselines={}, cfg=_arm_cfg(), pre_residualized=True,
                            cluster_col=cluster, complex_gates=gates, complex_col="event",
                            cfg_section=section)
    with_p1 = validate_slow(frame, baselines={}, cfg=_arm_cfg(
        project1={"oos_significance_alpha": 1e-12}), pre_residualized=True,
        cluster_col=cluster, complex_gates=gates, complex_col="event", cfg_section=section)
    pd.testing.assert_frame_equal(without, with_p1)
    assert bool(without["edge_persisted"].any()), "fixture must certify someone, or this is vacuous"


# --------------------------------------------------------------------------- #
# 5. the live config and the frozen forward experiments
# --------------------------------------------------------------------------- #

def test_live_config_scopes_the_tightening_to_project1():
    scoring = load_config().get("scoring", {})
    # Project 1's certification gate moved...
    assert scoring.get("project1", {}).get("oos_significance_alpha") == 0.005
    assert certification_alpha(scoring) == 0.005
    # ...and nothing else did.
    assert scoring.get("oos_significance_alpha") == 0.05      # the arms' key
    assert scoring.get("min_skill_edge") == 0.02              # the 2c economic floor
    assert scoring.get("sports", {}).get("min_skill_edge") == 0.10


@pytest.mark.parametrize("arm", ["slow", "sports"])
def test_frozen_forward_manifests_still_match_the_live_global_alpha(arm):
    """Artifact-level drift guard. `slow_forward`/`sports_forward` record
    `scoring.oos_significance_alpha` in the manifest `gates` of a FROZEN,
    pre-registered forward test. If a future edit moves the global key, the value
    the code would record today stops matching what those experiments were frozen
    with — and this test fails loudly instead of the experiment being corrupted
    silently."""
    path = PROCESSED_DIR / f"{arm}_freeze_manifest.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    gates = json.loads(path.read_text()).get("gates", {})
    if "oos_significance_alpha" not in gates:
        pytest.skip("manifest records no alpha")
    live_global = float(load_config().get("scoring", {}).get("oos_significance_alpha", 0.05))
    assert gates["oos_significance_alpha"] == live_global
