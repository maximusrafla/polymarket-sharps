"""Tests for the Project-1 gate on the deep real-world sample (src/realworld_validate.py).

What is new here is not the gate — that is `validate.compute_oos_validation`, tested
in test_validate.py and deliberately untouched — but the two things wrapped around it:

  1. the population assembled for it (the micro restriction, and the 383 reused
     wallets whose bets live in the prior arms), and
  2. the Horvitz-Thompson reweighting that turns "k of the sample certified" into
     "about K of the frame would".

Both are places where a silent mistake produces a plausible number, so they are
pinned with fixtures whose right answer is known by construction.
"""

import numpy as np
import pandas as pd
import pytest

from src import realworld_validate as rv


# ---------------------------------------------------------------------------
# The micro restriction
# ---------------------------------------------------------------------------

def _bets(rows):
    """rows: [(wallet, market_id, slug)]"""
    return pd.DataFrame([{"wallet": w, "market_id": m, "slug": s, "question": "",
                          "side": "BUY", "resolved": True, "entry_price": 0.5,
                          "resolved_value": 1.0, "timestamp": 1_700_000_000 + i}
                         for i, (w, m, s) in enumerate(rows)])


def test_restriction_drops_micro_and_keeps_everything_else():
    bets = _bets([
        ("0xa", "m1", "will-trump-win-the-2024-election"),
        ("0xa", "m2", "bitcoin-up-or-down-october-3-5pm-et"),
        ("0xb", "m3", "will-the-lakers-beat-the-celtics"),
        ("0xb", "m4", "ethereum-up-or-down-october-3-5pm-et"),
    ])
    kept, report = rv.restrict_to_real_world(bets)
    assert set(kept["market_id"]) == {"m1", "m3"}
    assert report["bets_before"] == 4 and report["bets_after"] == 2
    assert report["markets_before"] == 4 and report["markets_after"] == 2


def test_restriction_keeps_the_other_fallback_bucket():
    """`other` is real-world by definition (not micro_crypto). Sampling it showed
    geopolitics, golf, esports and unknown league codes — a granularity gap, not a
    micro leak — so dropping it would silently delete half the tape."""
    bets = _bets([("0xa", "m1", "sea-udi-fio-2026-03-02-draw"),
                  ("0xa", "m2", "will-jordan-strike-iran-by-march-31")])
    kept, _ = rv.restrict_to_real_world(bets)
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# Horvitz-Thompson reweighting
# ---------------------------------------------------------------------------

def _validated(spec):
    """spec: [(wallet, inclusion_prob, stratum, flag)]"""
    df = pd.DataFrame([{"wallet": w, "inclusion_prob": p, "stratum": s, "flag": f}
                       for w, p, s, f in spec])
    df["design_weight"] = 1.0 / df["inclusion_prob"]
    return df


def test_censused_stratum_is_its_own_answer():
    """p=1 means the stratum was not sampled at all, so the frame total is the
    observed count exactly and carries zero sampling variance."""
    v = _validated([(f"0x{i}", 1.0, "s4", i < 3) for i in range(10)])
    est = rv.ht_estimate(v, "flag")
    assert est["sample_count"] == 3
    assert est["frame_total"] == pytest.approx(3.0)
    assert est["frame_n"] == pytest.approx(10.0)
    assert est["frame_total_se"] == pytest.approx(0.0)


def test_sampled_stratum_scales_up_by_the_inverse_probability():
    v = _validated([(f"0x{i}", 0.1, "s1", i < 2) for i in range(20)])
    est = rv.ht_estimate(v, "flag")
    assert est["sample_count"] == 2
    assert est["frame_total"] == pytest.approx(20.0)     # 2 * (1/0.1)
    assert est["frame_n"] == pytest.approx(200.0)
    assert est["frame_total_se"] > 0                     # a draw, so it has variance


def test_mixed_strata_combine_and_only_the_sampled_band_carries_variance():
    v = pd.concat([
        _validated([(f"0xc{i}", 1.0, "s4", i < 5) for i in range(10)]),
        _validated([(f"0xs{i}", 0.2, "s1", i < 1) for i in range(10)]),
    ], ignore_index=True)
    est = rv.ht_estimate(v, "flag")
    assert est["frame_total"] == pytest.approx(5 + 1 * 5)      # 5 censused + 5 scaled
    assert est["frame_n"] == pytest.approx(10 + 50)
    solo = rv.ht_estimate(v[v["stratum"] == "s1"], "flag")
    assert est["frame_total_se"] == pytest.approx(solo["frame_total_se"])


def test_rate_is_a_share_of_the_frame_not_of_the_sample():
    v = pd.concat([
        _validated([(f"0xc{i}", 1.0, "s4", True) for i in range(10)]),
        _validated([(f"0xs{i}", 0.1, "s1", False) for i in range(10)]),
    ], ignore_index=True)
    est = rv.ht_estimate(v, "flag")
    # 10 of 20 sampled wallets carry the flag, but the frame is 10 + 100 = 110.
    assert est["sample_count"] == 10
    assert est["frame_rate"] == pytest.approx(10 / 110)


def test_wallets_without_a_weight_are_excluded_not_counted_as_zero():
    """A wallet with no census row (weight NaN) must not silently dilute the rate."""
    v = _validated([("0xa", 1.0, "s4", True), ("0xb", 1.0, "s4", False)])
    v.loc[2] = {"wallet": "0xc", "inclusion_prob": np.nan, "stratum": None,
                "flag": True, "design_weight": np.nan}
    est = rv.ht_estimate(v, "flag")
    assert est["sample_n"] == 2 and est["frame_n"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Memory shape: read fat, keep slim
# ---------------------------------------------------------------------------

def _fat(rows):
    """rows: [(wallet, market_id, slug, tx_hash)] -> a tape-shaped frame."""
    return pd.DataFrame([{
        "wallet": w, "market_id": m, "slug": s, "question": "", "tx_hash": tx,
        "token_id": "t", "side": "BUY", "resolved": True, "entry_price": 0.5,
        "resolved_value": 1.0, "timestamp": 1_700_000_000 + i,
    } for i, (w, m, s, tx) in enumerate(rows)])


def test_slim_drops_the_wide_columns_it_read():
    """slug/question/tx_hash/token_id exist only to classify and dedup; keeping them
    is what OOM-killed the first version."""
    out = rv._slim(_fat([("0xa", "m1", "will-trump-win-the-2024-election", "tx1")]), {})
    assert list(out.columns) == rv.SLIM_COLUMNS
    for wide in ("slug", "question", "tx_hash", "token_id"):
        assert wide not in out.columns


def test_slim_drops_micro_and_dedups_fills():
    fat = _fat([
        ("0xa", "m1", "will-trump-win-the-2024-election", "tx1"),
        ("0xa", "m1", "will-trump-win-the-2024-election", "tx1"),   # duplicate fill
        ("0xa", "m2", "bitcoin-up-or-down-october-3-5pm-et", "tx2"),
    ])
    out = rv._slim(fat, {})
    assert len(out) == 1 and out["market_id"].iloc[0] == "m1"


def test_slim_cache_is_shared_across_calls_so_each_market_classifies_once():
    cache = {}
    rv._slim(_fat([("0xa", "m1", "will-trump-win-the-2024-election", "tx1")]), cache)
    assert cache == {"m1": True}
    rv._slim(_fat([("0xb", "m1", "", "tx2")]), cache)   # blank slug, would misclassify
    assert cache == {"m1": True}                        # cached verdict reused, not redone


def test_slim_handles_an_empty_frame():
    out = rv._slim(pd.DataFrame(columns=rv.READ_COLUMNS), {})
    assert out.empty and list(out.columns) == rv.SLIM_COLUMNS


def test_assembled_market_id_is_dictionary_encoded(monkeypatch, tmp_path):
    """market_id is a 66-char hex string over ~306k distinct values in ~4.9M rows.
    Carried as objects it cost ~600 MB and pushed the first real run 2 GB into swap;
    validate.main reads it dictionary-encoded for exactly this reason."""
    from src import realworld_deepen as rwd

    shard_dir = tmp_path / "deep_trades"
    shard_dir.mkdir(parents=True)
    fat = _fat([(f"0xw{i}", f"0x{'a' * 64}{i}", "will-trump-win-the-2024-election", f"tx{i}")
                for i in range(4)])
    fat.to_parquet(shard_dir / "part-00000.parquet")
    monkeypatch.setattr(rwd, "SHARD_DIR", shard_dir)
    short = pd.DataFrame({"wallet": ["0xw0"], "prior_arm": [""]})
    short.to_parquet(tmp_path / "shortlist.parquet")
    monkeypatch.setattr(rv, "SHORTLIST_PATH", tmp_path / "shortlist.parquet")
    monkeypatch.setattr(rv, "PRIOR_ARM_TAPES", {})

    bets, report = rv.assemble_population(real_world=True)
    assert str(bets["market_id"].dtype) == "category"
    assert report["assembled_rows"] == 4
