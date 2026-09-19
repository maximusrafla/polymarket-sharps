"""Tests for the real-world stratified deepening sampler (src/realworld_deepen.py).

Hand-built fixtures where the correct stratum membership, inclusion probability
and budget behaviour are known by construction. Three properties are load-bearing
and each is pinned here:

  1. The draw is PERFORMANCE-BLIND and reproducible — it can only see activity,
     and the same seed gives the same wallets.
  2. Inclusion probabilities are correct, so a statistic on the sample can be
     reweighted back to the frame. This is the whole reason to sample rather than
     take a top-N slice.
  3. The disk budget actually stops the run, and stopping loses no rows and no
     resumability — the binding constraint on a 32 GB eMMC box.

The fetch/fold/resolve half is the tested backfill/ingest machinery with different
paths (exercised in test_backfill/test_ingest/test_slow_deepen); what is new here
is the append-only SHARD store, so its round-trip and dedup are pinned too.
"""

import json

import pandas as pd
import pytest

from src import realworld_deepen as rw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _activity(spec):
    """spec: {wallet: (n_bets, n_markets)} -> the frame load_discovery_activity returns."""
    return pd.DataFrame(
        [{"wallet": w, "n_discovery_bets": b, "n_discovery_markets": m}
         for w, (b, m) in spec.items()]
    ).sort_values("wallet", kind="mergesort").reset_index(drop=True)


def _big_activity(n_per_band=200):
    spec = {}
    for band, bets in (("a", 500), ("b", 150), ("c", 70), ("d", 30)):
        for i in range(n_per_band):
            spec[f"0x{band}{i:04d}"] = (bets, 9)
    return _activity(spec)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point every write at a tmp dir so tests never touch the real dataset."""
    root = tmp_path / "realworld"
    monkeypatch.setattr(rw, "RW_DIR", root)
    monkeypatch.setattr(rw, "SHARD_DIR", root / "deep_trades")
    monkeypatch.setattr(rw, "DEEP_CURSORS_PATH", root / "deep_cursors.json")
    monkeypatch.setattr(rw, "DEEP_RESOLUTIONS_PATH", root / "deep_resolutions.parquet")
    monkeypatch.setattr(rw, "SHORTLIST_PATH", root / "shortlist.parquet")
    monkeypatch.setattr(rw, "POOL_CENSUS_PATH", root / "pool_census.parquet")
    monkeypatch.setattr(rw, "PRIOR_ARMS", {})
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Frame + strata
# ---------------------------------------------------------------------------

def test_frame_excludes_thin_and_narrow_wallets():
    act = _activity({
        "0xdeep": (500, 20),    # in
        "0xthin": (19, 20),     # too few bets
        "0xnarrow": (500, 4),   # too few distinct markets
        "0xedge": (20, 5),      # exactly at both floors -> in
    })
    frame = rw.assign_strata(act)
    assert set(frame["wallet"]) == {"0xdeep", "0xedge"}


def test_strata_boundaries_are_half_open():
    act = _activity({f"0x{n:04d}": (n, 9) for n in (20, 49, 50, 99, 100, 199, 200, 5000)})
    frame = rw.assign_strata(act).set_index("wallet")["stratum"].to_dict()
    assert frame["0x0020"] == frame["0x0049"] == "s1_20_49"
    assert frame["0x0050"] == frame["0x0099"] == "s2_50_99"
    assert frame["0x0100"] == frame["0x0199"] == "s3_100_199"
    assert frame["0x0200"] == frame["0x5000"] == "s4_200plus"


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def test_upper_strata_are_censused_and_bottom_band_is_sampled():
    frame = rw.assign_strata(_big_activity(200))   # 200 per band, 800 total
    census = rw.draw_sample(frame, target=700)
    drawn = census[census["selected"]]
    by = drawn.groupby("stratum").size().to_dict()
    assert by["s4_200plus"] == 200 and by["s3_100_199"] == 200 and by["s2_50_99"] == 200
    assert by["s1_20_49"] == 100                     # 700 - 600 spills into the sampled band
    assert len(drawn) == 700


def test_inclusion_probabilities_reweight_to_the_frame():
    frame = rw.assign_strata(_big_activity(200))
    census = rw.draw_sample(frame, target=700)
    drawn = census[census["selected"]]
    # Horvitz-Thompson: summing 1/p over the sample must recover the frame size.
    assert (1.0 / drawn["inclusion_prob"]).sum() == pytest.approx(len(frame))
    censused = drawn[drawn["stratum"] != "s1_20_49"]
    assert (censused["inclusion_prob"] == 1.0).all()
    sampled = drawn[drawn["stratum"] == "s1_20_49"]
    assert sampled["inclusion_prob"].unique().tolist() == [100 / 200]


def test_unselected_wallets_are_retained_not_dropped():
    """CLAUDE.md: nothing is ever removed from the dataset. The census keeps the
    whole frame so the sample can be audited and reweighted."""
    frame = rw.assign_strata(_big_activity(200))
    census = rw.draw_sample(frame, target=700)
    assert len(census) == len(frame)
    assert (~census["selected"]).sum() == len(frame) - 700


def test_draw_is_reproducible_and_seed_sensitive():
    frame = rw.assign_strata(_big_activity(200))
    a = rw.draw_sample(frame, target=700, seed=1)
    b = rw.draw_sample(frame, target=700, seed=1)
    c = rw.draw_sample(frame, target=700, seed=2)
    sel = lambda d: set(d.loc[d["selected"], "wallet"])  # noqa: E731
    assert sel(a) == sel(b)
    assert sel(a) != sel(c)


def test_draw_cannot_see_performance():
    """The sampler is handed activity only; adding an edge column that would flip
    any performance-based ranking must not move a single wallet."""
    frame = rw.assign_strata(_big_activity(200))
    baseline = rw.draw_sample(frame, target=700)
    poisoned = frame.copy()
    poisoned["mean_resid"] = [0.9 if i % 2 else -0.9 for i in range(len(poisoned))]
    after = rw.draw_sample(poisoned, target=700)
    assert (set(baseline.loc[baseline["selected"], "wallet"])
            == set(after.loc[after["selected"], "wallet"]))


def test_prior_arm_wallets_are_flagged_but_still_drawn(monkeypatch, tmp_path):
    """Reuse must not bias the sample: the draw happens first, the on-disk check
    second, so inclusion probability is untouched."""
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"0xa0000": {}, "0xd0000": {}}))
    monkeypatch.setattr(rw, "PRIOR_ARMS", {"slow": prior})
    frame = rw.assign_strata(_big_activity(200))
    census = rw.draw_sample(frame, target=800)      # everything drawn
    flagged = census.loc[census["prior_arm"] == "slow", "wallet"]
    assert set(flagged) == {"0xa0000", "0xd0000"}
    assert census.loc[census["wallet"].isin(flagged), "selected"].all()
    assert (census.loc[census["wallet"].isin(flagged), "inclusion_prob"] == 1.0).all()


# ---------------------------------------------------------------------------
# Shard store
# ---------------------------------------------------------------------------

def _rows(tx_prefix, n, resolved=False):
    return pd.DataFrame([{
        "wallet": "0xw", "market_id": "m1", "token_id": "t1", "outcome": "Yes",
        "side": "BUY", "entry_price": 0.4, "size": 10.0, "timestamp": 1_700_000_000 + i,
        "resolved": resolved, "resolved_value": 1.0 if resolved else None,
        "question": "q", "slug": "s", "tx_hash": f"{tx_prefix}{i}", "speed_bucket": "unknown",
    } for i in range(n)])


def test_shards_append_and_read_back_whole():
    rw.write_shard(_rows("a", 3))
    rw.write_shard(_rows("b", 2))
    assert len(rw.shard_paths()) == 2
    assert len(rw.load_deep_trades()) == 5


def test_shard_read_dedups_on_the_fill_key_newest_wins():
    rw.write_shard(_rows("a", 2, resolved=False))
    rw.write_shard(_rows("a", 2, resolved=True))     # same tx_hashes, re-fetched
    out = rw.load_deep_trades()
    assert len(out) == 2
    assert out["resolved"].all()                      # the later shard wins


def test_write_shard_ignores_empty_batches():
    assert rw.write_shard(pd.DataFrame()) is None
    assert rw.shard_paths() == []


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------

def test_budget_trips_on_low_free_space(monkeypatch):
    monkeypatch.setattr(rw, "free_bytes", lambda: int(1.0 * 1024 ** 3))
    ok, reason = rw.budget_check(min_free_gb=2.5, max_new_gb=99)
    assert not ok and "free" in reason


def test_budget_trips_on_dataset_cap(monkeypatch):
    monkeypatch.setattr(rw, "free_bytes", lambda: int(50 * 1024 ** 3))
    monkeypatch.setattr(rw, "dataset_bytes", lambda: int(2.0 * 1024 ** 3))
    ok, reason = rw.budget_check(min_free_gb=1.0, max_new_gb=1.6)
    assert not ok and "cap" in reason


def test_budget_passes_with_headroom(monkeypatch):
    monkeypatch.setattr(rw, "free_bytes", lambda: int(10 * 1024 ** 3))
    monkeypatch.setattr(rw, "dataset_bytes", lambda: 0)
    ok, _ = rw.budget_check(min_free_gb=2.5, max_new_gb=1.6)
    assert ok


# ---------------------------------------------------------------------------
# Deepening loop: budget stop, resumability, no lost rows
# ---------------------------------------------------------------------------

def _fake_trade(wallet, i):
    return {"proxyWallet": wallet, "conditionId": f"m{i}", "asset": f"t{i}",
            "outcome": "Yes", "side": "BUY", "price": 0.5, "size": 1.0,
            "timestamp": 1_700_000_000 + i, "title": "q", "slug": "s",
            "transactionHash": f"{wallet}-tx{i}"}


def _patch_fetch(monkeypatch, calls):
    def fake(session, cfg, wallet, since_ts):
        calls.append(wallet)
        return [_fake_trade(wallet, 0)]
    monkeypatch.setattr(rw, "fetch_user_trades", fake)
    monkeypatch.setattr(rw, "make_session", lambda: None)
    monkeypatch.setattr(rw, "load_config", lambda: {})


def test_deepen_stops_at_the_budget_and_keeps_what_it_fetched(monkeypatch):
    calls = []
    _patch_fetch(monkeypatch, calls)
    monkeypatch.setattr(rw, "free_bytes", lambda: int(0.5 * 1024 ** 3))  # already below floor
    wallets = [f"0xw{i}" for i in range(10)]
    stats = rw.deepen_wallets(wallets, checkpoint_every=2, min_free_gb=2.5, max_new_gb=99)
    assert stats["stopped_early"] and "free" in stats["stop_reason"]
    assert len(calls) == 2                       # stopped at the first checkpoint
    assert len(rw.load_deep_trades()) == 2       # and the fetched rows are on disk
    assert set(json.loads(rw.DEEP_CURSORS_PATH.read_text())) == {"0xw0", "0xw1"}


def test_deepen_resumes_and_does_not_refetch_completed_wallets(monkeypatch):
    calls = []
    _patch_fetch(monkeypatch, calls)
    monkeypatch.setattr(rw, "free_bytes", lambda: int(50 * 1024 ** 3))
    monkeypatch.setattr(rw, "dataset_bytes", lambda: 0)
    wallets = [f"0xw{i}" for i in range(4)]
    rw.deepen_wallets(wallets, checkpoint_every=2)
    assert len(calls) == 4
    calls.clear()
    stats = rw.deepen_wallets(wallets, checkpoint_every=2)   # second pass
    assert calls == [] and stats["skipped_done"] == 4


def test_deepen_survives_a_failing_wallet(monkeypatch):
    monkeypatch.setattr(rw, "free_bytes", lambda: int(50 * 1024 ** 3))
    monkeypatch.setattr(rw, "dataset_bytes", lambda: 0)
    monkeypatch.setattr(rw, "make_session", lambda: None)
    monkeypatch.setattr(rw, "load_config", lambda: {})

    def fake(session, cfg, wallet, since_ts):
        if wallet == "0xbad":
            raise RuntimeError("boom")
        return [_fake_trade(wallet, 0)]
    monkeypatch.setattr(rw, "fetch_user_trades", fake)

    stats = rw.deepen_wallets(["0xa", "0xbad", "0xc"], checkpoint_every=10)
    assert stats["errors"] == 1 and stats["fetched"] == 2
    assert set(rw.load_deep_trades()["wallet"]) == {"0xa", "0xc"}
    assert "0xbad" not in json.loads(rw.DEEP_CURSORS_PATH.read_text())


def test_deepen_never_touches_the_shared_ledger(monkeypatch):
    """Isolation invariant: Project 1's certified set must stay bit-identical."""
    from src.common import BET_LEDGER_PATH
    before = BET_LEDGER_PATH.stat().st_mtime if BET_LEDGER_PATH.exists() else None
    calls = []
    _patch_fetch(monkeypatch, calls)
    monkeypatch.setattr(rw, "free_bytes", lambda: int(50 * 1024 ** 3))
    monkeypatch.setattr(rw, "dataset_bytes", lambda: 0)
    rw.deepen_wallets(["0xa", "0xb"], checkpoint_every=1)
    after = BET_LEDGER_PATH.stat().st_mtime if BET_LEDGER_PATH.exists() else None
    assert before == after
    assert str(rw.RW_DIR) not in str(BET_LEDGER_PATH)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def _tape(rows):
    """rows: [(wallet, market_id, slug, side, resolved)] -> a shard-shaped frame."""
    return pd.DataFrame([{
        "wallet": w, "market_id": m, "token_id": "t", "outcome": "Yes", "side": s,
        "entry_price": 0.5, "size": 1.0, "timestamp": 1_700_000_000 + i,
        "resolved": r, "resolved_value": 1.0 if r else None,
        "question": "", "slug": slug, "tx_hash": f"tx{i}", "speed_bucket": "unknown",
    } for i, (w, m, slug, s, r) in enumerate(rows)])


def test_profile_counts_only_resolved_buys_and_splits_micro_from_real_world():
    rw.write_shard(_tape([
        ("0xa", "m1", "will-trump-win-the-2024-election", "BUY", True),
        ("0xa", "m2", "bitcoin-up-or-down-october-3-5pm-et", "BUY", True),
        ("0xa", "m3", "will-the-lakers-beat-the-celtics", "SELL", True),    # not a bet
        ("0xa", "m4", "will-putin-be-reelected", "BUY", False),             # unresolved
        ("0xb", "m2", "bitcoin-up-or-down-october-3-5pm-et", "BUY", True),
    ]))
    prof = rw.profile_wallets().set_index("wallet")
    assert prof.loc["0xa", "total_bets"] == 2      # the SELL and the unresolved row drop out
    assert prof.loc["0xa", "micro_bets"] == 1
    assert prof.loc["0xa", "real_world_bets"] == 1
    assert prof.loc["0xb", "real_world_bets"] == 0 and prof.loc["0xb", "micro_bets"] == 1


def test_profile_is_shard_order_independent():
    """Streaming per shard must give the same answer as one pass — the whole reason
    the profile is streamed is that a whole-dataset concat OOM-killed this box."""
    rows = [("0xa", f"m{i}", "will-trump-win-the-2024-election", "BUY", True) for i in range(6)]
    rw.write_shard(_tape(rows[:2]))
    rw.write_shard(_tape(rows[2:]))
    prof = rw.profile_wallets().set_index("wallet")
    assert prof.loc["0xa", "real_world_bets"] == 6


def test_classify_deep_markets_labels_each_market_once():
    rw.write_shard(_tape([
        ("0xa", "m1", "bitcoin-up-or-down-october-3-5pm-et", "BUY", True),
        ("0xb", "m1", "bitcoin-up-or-down-october-3-5pm-et", "BUY", True),
        ("0xa", "m2", "will-trump-win-the-2024-election", "BUY", True),
    ]))
    cats = rw.classify_deep_markets()
    assert len(cats) == 2
    assert cats.set_index("market_id").loc["m1", "category"] == "micro_crypto"


def test_load_deep_tape_dedups_on_the_FULL_key_even_under_projection():
    """Regression: pruning the dedup key to the projected columns deduped on
    (wallet, side) and collapsed the real 7.05M-row tape to 4,077 rows. The key is
    read in full regardless of the projection, then dropped again."""
    rw.write_shard(_rows("a", 4))
    got = rw.load_deep_tape(columns=["wallet", "side", "entry_price"])
    assert len(got) == 4                                  # not 1
    assert list(got.columns) == ["wallet", "side", "entry_price"]   # key not leaked


def test_load_deep_tape_removes_true_duplicate_fills():
    dup = pd.concat([_rows("a", 3), _rows("a", 3)], ignore_index=True)
    rw.write_shard(dup)
    assert len(rw.load_deep_tape(columns=["wallet", "side"])) == 3


def test_load_deep_tape_dictionary_encodes_requested_columns():
    rw.write_shard(_rows("a", 5))
    got = rw.load_deep_tape(columns=["wallet", "slug", "entry_price"],
                            categorical=["wallet", "slug"])
    assert str(got["wallet"].dtype) == "category"
    assert str(got["slug"].dtype) == "category"
    assert str(got["entry_price"].dtype).startswith("float")
