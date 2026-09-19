import numpy as np
import pandas as pd
import pytest

from src.validate import (
    _oos_significance,
    _regime_flag,
    classify_regime,
    compute_oos_validation,
    split_in_sample_out_of_sample,
)

LEDGER_COLS = [
    "wallet", "market_id", "token_id", "outcome", "side", "entry_price", "size",
    "timestamp", "resolved", "resolved_value", "question", "slug", "tx_hash",
]


def bet(wallet, market_id="m1", token_id="t1", side="BUY", entry_price=0.5, size=10,
        timestamp=0, resolved=False, resolved_value=None, tx_hash="tx", question="q", slug="s",
        outcome="Yes"):
    return {
        "wallet": wallet, "market_id": market_id, "token_id": token_id, "outcome": outcome,
        "side": side, "entry_price": entry_price, "size": size, "timestamp": timestamp,
        "resolved": resolved, "resolved_value": resolved_value, "question": question,
        "slug": slug, "tx_hash": tx_hash,
    }


def make_ledger(rows):
    return pd.DataFrame(rows, columns=LEDGER_COLS)


def cfg(oos_split=0.5, min_bets_per_half=10, oos_significance_alpha=0.05, price_baseline_bins=20,
        min_skill_edge=0.0, min_oos_markets=0, oos_bootstrap_resamples=1000):
    # min_oos_markets defaults to 0 here so the significance/magnitude/regime tests
    # stay isolated from the cluster-count floor (its own dedicated tests set it).
    # oos_bootstrap_resamples is small for test speed; the bootstrap is seeded, so
    # results are deterministic regardless.
    return {"scoring": {
        "oos_split": oos_split,
        "min_bets_per_half": min_bets_per_half,
        "oos_significance_alpha": oos_significance_alpha,
        "price_baseline_bins": price_baseline_bins,
        "min_skill_edge": min_skill_edge,
        "min_oos_markets": min_oos_markets,
        "oos_bootstrap_resamples": oos_bootstrap_resamples,
    }}


def fair_market(n=40, price=0.5):
    """Filler at a single price with a 50% win rate. All one price, so the fitted
    baseline falls back to the global base rate ~= that win rate: a fair market
    against which a test wallet's *residual* (skill) edge is how much it beat the
    price it paid. Each filler wallet has one bet, so none become candidates."""
    return [bet(f"mkt{i}", market_id=f"fm{i}", entry_price=price, timestamp=i, resolved=True,
                resolved_value=1.0 if i % 2 == 0 else 0.0) for i in range(n)]


def wallet_bets(wallet, in_wins, out_wins, n_in=10, n_out=10, price=0.5, t0=1000,
                single_market=False):
    """One wallet's in/out bets at a single price. By default each bet lands in its
    OWN market, so the held-out half has `n_out` distinct markets — enough clusters
    for the cluster-robust significance test and the cluster-count floor to be
    defined (which is the realistic case). Set `single_market` to force every bet
    into `m1`, the degenerate one-cluster case used to exercise those gates."""
    rows, ts = [], t0
    for i in range(n_in):
        mkt = "m1" if single_market else f"{wallet}_in_{i}"
        rows.append(bet(wallet, market_id=mkt, entry_price=price, timestamp=ts, resolved=True,
                        resolved_value=1.0 if i < in_wins else 0.0)); ts += 1
    for i in range(n_out):
        mkt = "m1" if single_market else f"{wallet}_out_{i}"
        rows.append(bet(wallet, market_id=mkt, entry_price=price, timestamp=ts, resolved=True,
                        resolved_value=1.0 if i < out_wins else 0.0)); ts += 1
    return rows


# --- split_in_sample_out_of_sample --------------------------------------

def test_split_in_sample_out_of_sample_even_count():
    rows = pd.DataFrame({"timestamp": [5, 1, 4, 2, 3, 0]})
    in_sample, out_of_sample = split_in_sample_out_of_sample(rows, oos_split=0.5)
    assert list(in_sample["timestamp"]) == [0, 1, 2]
    assert list(out_of_sample["timestamp"]) == [3, 4, 5]


def test_split_in_sample_out_of_sample_odd_count_favors_out_of_sample():
    rows = pd.DataFrame({"timestamp": [0, 1, 2, 3, 4]})
    in_sample, out_of_sample = split_in_sample_out_of_sample(rows, oos_split=0.5)
    assert len(in_sample) == 2
    assert len(out_of_sample) == 3


# --- compute_oos_validation: persistence on SKILL (residual) edge ----------

def test_edge_persists_when_skill_edge_is_significantly_positive_out_of_sample():
    rows = fair_market() + wallet_bets("P", in_wins=9, out_wins=9)
    row = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet").loc["P"]
    assert row["in_sample_n"] == 10 and row["out_of_sample_n"] == 10
    # beats the ~0.5 price it paid in both halves...
    assert row["in_sample_residual_edge"] > 0
    assert row["out_of_sample_residual_edge"] > 0
    # ...and the held-out edge is statistically > 0, so it persists.
    assert row["out_of_sample_residual_p"] < 0.05
    assert bool(row["edge_persisted"])
    # raw edge is still reported side by side (9/10 wins at 0.5 -> +0.4).
    assert row["in_sample_edge"] == pytest.approx(0.4)


def test_edge_does_not_persist_when_out_of_sample_skill_edge_goes_negative():
    rows = fair_market() + wallet_bets("D", in_wins=9, out_wins=1)
    row = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet").loc["D"]
    assert row["in_sample_residual_edge"] > 0          # looked sharp in-sample
    assert row["out_of_sample_residual_edge"] < 0      # decayed out-of-sample
    assert not bool(row["edge_persisted"])


def test_edge_does_not_persist_when_positive_but_not_significant():
    # Wins just over half out-of-sample: residual edge is positive but the
    # one-sided t-test can't reject 0, so the stricter gate rejects it.
    rows = fair_market() + wallet_bets("W", in_wins=9, out_wins=6)
    row = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet").loc["W"]
    assert row["out_of_sample_residual_edge"] > 0
    assert row["out_of_sample_residual_p"] >= 0.05
    assert not bool(row["edge_persisted"])


# --- economic-magnitude gate (metric C): edge_persisted needs BOTH gates -------

def test_edge_magnitude_gate_blocks_small_but_significant_edge():
    # "P" beats its price significantly out-of-sample, but by less than a high
    # magnitude floor: statistically real, economically trivial -> not persisted.
    rows = fair_market() + wallet_bets("P", in_wins=9, out_wins=9)
    row = compute_oos_validation(make_ledger(rows), cfg(min_skill_edge=0.5)).set_index("wallet").loc["P"]
    assert row["out_of_sample_residual_edge"] > 0
    assert bool(row["edge_significant"])          # clears the significance gate
    assert not bool(row["edge_magnitude_ok"])     # but not the magnitude floor
    assert not bool(row["edge_persisted"])        # persistence needs both


def test_edge_persisted_requires_both_significance_and_magnitude():
    rows = fair_market() + wallet_bets("P", in_wins=9, out_wins=9)
    row = compute_oos_validation(make_ledger(rows), cfg(min_skill_edge=0.02)).set_index("wallet").loc["P"]
    assert bool(row["edge_significant"]) and bool(row["edge_magnitude_ok"])
    assert bool(row["edge_persisted"])


def test_edge_magnitude_ok_is_independent_of_significance():
    # positive-but-not-significant: clears a zero magnitude floor yet does not
    # persist, showing the two gates are reported independently.
    rows = fair_market() + wallet_bets("W", in_wins=9, out_wins=6)
    row = compute_oos_validation(make_ledger(rows), cfg(min_skill_edge=0.0)).set_index("wallet").loc["W"]
    assert row["out_of_sample_residual_edge"] > 0
    assert bool(row["edge_magnitude_ok"])         # positive edge clears a 0 floor
    assert not bool(row["edge_significant"])       # but the t-test can't reject 0
    assert not bool(row["edge_persisted"])


def test_magnitude_gate_never_drops_a_wallet():
    # gating is flag-only: a magnitude-failed wallet still has a row and metrics.
    rows = fair_market() + wallet_bets("P", in_wins=9, out_wins=9)
    validated = compute_oos_validation(make_ledger(rows), cfg(min_skill_edge=0.9))
    assert "P" in set(validated["wallet"])
    row = validated.set_index("wallet").loc["P"]
    assert not bool(row["edge_persisted"])
    assert row["out_of_sample_n"] == 10  # metrics intact, just un-flagged


def test_wallet_not_a_candidate_when_in_sample_skill_edge_negative():
    # Loses in-sample (not a candidate) but wins out-of-sample: nothing to
    # "persist", so it is never counted regardless of held-out performance.
    rows = fair_market() + wallet_bets("C", in_wins=1, out_wins=9)
    row = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet").loc["C"]
    assert row["in_sample_residual_edge"] < 0
    assert not bool(row["edge_persisted"])


def test_wallet_below_min_bets_per_half_is_gated_to_nan_not_dropped():
    # 4 bets per half < min_bets_per_half (10): edges gate to NaN (insufficient
    # data), the wallet is not a candidate, but it is still present.
    rows = fair_market() + wallet_bets("F", in_wins=4, out_wins=4, n_in=4, n_out=4)
    validated = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet")
    assert "F" in validated.index
    row = validated.loc["F"]
    assert np.isnan(row["in_sample_residual_edge"])
    assert np.isnan(row["out_of_sample_residual_edge"])
    assert not bool(row["edge_persisted"])


def test_wallet_with_only_unresolved_bets_is_present_with_nan():
    rows = fair_market() + [bet("E", timestamp=5000, resolved=False)]
    validated = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet")
    assert "E" in validated.index
    row = validated.loc["E"]
    assert np.isnan(row["in_sample_edge"])
    assert row["in_sample_n"] == 0
    assert not bool(row["edge_persisted"])


def test_every_wallet_present_regardless_of_validation_outcome():
    rows = (
        fair_market()
        + wallet_bets("P", in_wins=9, out_wins=9)
        + wallet_bets("F", in_wins=1, out_wins=1, n_in=1, n_out=1)
        + [bet("E", timestamp=6000, resolved=False)]
    )
    validated = compute_oos_validation(make_ledger(rows), cfg())
    assert {"P", "F", "E"}.issubset(set(validated["wallet"]))


# --- metric D: non-stationary / regime-change detection --------------------
# The regime is known BY CONSTRUCTION in every fixture below: residual (skill)
# edge is +0.5 for a win, -0.5 for a loss, so a half's mean residual is exactly
# 0.5*(2*winrate - 1). Wins are spread evenly (never a constant sub-window) so
# the significance/trend tests have variance to work with.

def resid(win_pattern):
    """Residual-edge series from an explicit per-bet win/loss pattern."""
    return np.array([0.5 if w else -0.5 for w in win_pattern], dtype=float)


def wins_spread(n, k):
    """Length-n boolean win list with exactly k wins spread as evenly as possible
    (Bresenham) — so no contiguous sub-window is all-wins/all-losses and every
    sub-half keeps variance. 0 < k < n."""
    return [((i * k) // n) != (((i + 1) * k) // n) for i in range(n)]


REGIME_KW = dict(min_per_half=10, alpha=0.05, min_skill_edge=0.02,
                 regime_alpha=0.05, regime_min_bets=50)


def _classify(early_wins, recent_wins, **overrides):
    """Run classify_regime on constructed win patterns, deriving recent_mean and
    recent_p exactly as compute_oos_validation does."""
    kw = {**REGIME_KW, **overrides}
    early, recent = resid(early_wins), resid(recent_wins)
    recent_mean = float(recent.mean()) if recent.size else np.nan
    recent_p = _oos_significance(recent, kw["min_per_half"])
    return classify_regime(early, recent, recent_mean, recent_p, **kw)


def test_regime_flag_insufficient_when_a_half_is_too_small():
    # 5 bets in the early half < min_per_half (10): untestable -> insufficient,
    # and nothing downstream (watch/recovery) can fire.
    flag, watch, recent = _classify(wins_spread(5, 2), wins_spread(60, 54))
    assert flag == "insufficient"
    assert watch is False and recent is False


def test_regime_flag_stable_for_a_stationary_sharp_wallet():
    # Sharp and steady in BOTH halves (80% each): no shifted mean, no trend ->
    # stable, and it is NOT on the became-sharp watch (early edge already > 0).
    flag, watch, recent = _classify(wins_spread(60, 48), wins_spread(60, 48))
    assert flag == "stable"
    assert watch is False and recent is False


def test_regime_flag_decaying_when_edge_erodes():
    # Early sharp (90%), recent poor (10%): mean drops and the time-trend is
    # negative -> decaying. Not a became-sharp wallet, so no watch/recovery.
    flag, watch, recent = _classify(wins_spread(60, 54), wins_spread(60, 6))
    assert flag == "decaying"
    assert watch is False and recent is False


def test_became_sharp_sets_regime_watch_and_d3_confirms_when_recent_is_deep():
    # Early not sharp (30%), recent strongly sharp (90%) over a deep (>=50) recent
    # half whose two sub-halves each independently clear significance + magnitude:
    # improving_confirmed, regime_watch True, persisted_recent True.
    flag, watch, recent = _classify(wins_spread(60, 18), wins_spread(60, 54))
    assert flag == "improving_confirmed"
    assert watch is True and recent is True


def test_became_sharp_watch_without_recovery_when_recent_too_shallow():
    # Same became-sharp shape but a recent half of 40 < regime_min_bets (50):
    # on the watch and flagged improving, but D3 is not attempted -> no recovery.
    flag, watch, recent = _classify(wins_spread(40, 12), wins_spread(40, 36))
    assert watch is True
    assert recent is False
    assert flag == "improving"  # detected, but not sub-split-confirmed


def test_d3_recovery_blocked_when_a_recent_sub_half_has_no_variance():
    # Deep became-sharp recent half, but its FIRST sub-half is all wins (constant,
    # so the one-sided t-test is undefined): D3 cannot certify it -> no recovery,
    # yet the wallet stays on the watch and is flagged improving (not confirmed).
    recent_pattern = [True] * 30 + wins_spread(30, 27)
    flag, watch, recent = _classify(wins_spread(60, 18), recent_pattern)
    assert watch is True
    assert recent is False
    assert flag == "improving"


def test_regime_flag_helper_matches_directional_construction():
    # _regime_flag alone (no D3 promotion) on the raw halves.
    assert _regime_flag(resid(wins_spread(60, 18)), resid(wins_spread(60, 54)), 10, 0.05) == "improving"
    assert _regime_flag(resid(wins_spread(60, 54)), resid(wins_spread(60, 6)), 10, 0.05) == "decaying"
    assert _regime_flag(resid(wins_spread(60, 42)), resid(wins_spread(60, 42)), 10, 0.05) == "stable"
    assert _regime_flag(resid(wins_spread(5, 2)), resid(wins_spread(60, 54)), 10, 0.05) == "insufficient"


# --- metric D through the full validation table (wiring + invariants) ------

def regime_wallet(wallet, early_wins, recent_wins, price=0.5, t0=1000):
    """Lay out a wallet chronologically: the early half's per-bet win/loss list
    then the recent half's. Equal-length halves so the 50/50 chronological split
    in compute_oos_validation lands exactly on the early|recent boundary. Each bet
    is in its own market so the cluster-robust significance test has enough
    clusters (regime residuals don't depend on the market labels)."""
    assert len(early_wins) == len(recent_wins)
    rows, ts = [], t0
    for i, w in enumerate(list(early_wins) + list(recent_wins)):
        rows.append(bet(wallet, market_id=f"{wallet}_m{i}", entry_price=price, timestamp=ts,
                        resolved=True, resolved_value=1.0 if w else 0.0))
        ts += 1
    return rows


def test_metric_d_columns_wired_and_additive_through_validation():
    rows = (
        fair_market(200)
        # became-sharp: early 30% -> recent 90%, recent half deep enough for D3
        + regime_wallet("B", wins_spread(60, 18), wins_spread(60, 54))
        # stationary sharp: 80% in both halves
        + regime_wallet("P", wins_spread(60, 48), wins_spread(60, 48))
    )
    validated = compute_oos_validation(make_ledger(rows), cfg(min_skill_edge=0.02))
    # new columns are additive and present for every wallet; nothing dropped.
    for col in ("regime_flag", "regime_watch", "persisted_recent"):
        assert col in validated.columns
    assert not validated["regime_flag"].isna().any()
    v = validated.set_index("wallet")
    assert {"B", "P"}.issubset(set(v.index))

    # became-sharp wallet: routed to the watchlist, NOT edge_persisted (the
    # candidate gate excludes it because the early half is not sharp), and its
    # deep recent regime is D3-confirmed.
    b = v.loc["B"]
    assert bool(b["regime_watch"]) is True
    assert bool(b["persisted_recent"]) is True
    assert b["regime_flag"] == "improving_confirmed"
    assert bool(b["edge_persisted"]) is False  # invariant: recovery never sets edge_persisted

    # stationary sharp wallet: persists the ordinary way, stable regime, off the
    # became-sharp watch.
    p = v.loc["P"]
    assert bool(p["edge_persisted"]) is True
    assert bool(p["regime_watch"]) is False
    assert bool(p["persisted_recent"]) is False
    assert p["regime_flag"] == "stable"


def test_metric_d_never_flags_or_watches_a_wallet_without_resolved_bets():
    rows = fair_market() + [bet("E", timestamp=9000, resolved=False)]
    v = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet")
    assert v.loc["E", "regime_flag"] == "insufficient"
    assert bool(v.loc["E", "regime_watch"]) is False
    assert bool(v.loc["E", "persisted_recent"]) is False


# --- cluster-count floor (metric: distinct held-out markets) ------------------
# Bets sharing a market share one resolution event, so the persistence t-test
# over bets overstates the evidence. edge_persisted therefore also requires
# `min_oos_markets` distinct held-out markets. See docs/persistence_cluster_recheck.md.

def test_out_of_sample_markets_is_reported():
    # 30 held-out bets, each in its own market -> 30 distinct held-out markets.
    rows = fair_market() + wallet_bets("P", in_wins=27, out_wins=27, n_in=30, n_out=30)
    row = compute_oos_validation(make_ledger(rows), cfg()).set_index("wallet").loc["P"]
    assert row["out_of_sample_markets"] == 30
    assert bool(row["edge_markets_ok"]) is True


def test_cluster_floor_blocks_wallet_below_the_market_count_floor():
    # Cluster-SIGNIFICANT (10 distinct held-out markets, mostly winning) but below
    # the 30-market floor: the significance gate fires, the count floor does not,
    # so it is not certified. Isolates the floor from the significance test.
    rows = fair_market() + wallet_bets("C", in_wins=9, out_wins=9, n_in=10, n_out=10)
    row = compute_oos_validation(
        make_ledger(rows), cfg(min_oos_markets=30)).set_index("wallet").loc["C"]
    assert row["out_of_sample_markets"] == 10
    assert bool(row["edge_significant"]) is True       # 10 markets, 9 winners -> significant
    assert bool(row["edge_magnitude_ok"]) is True
    assert bool(row["edge_markets_ok"]) is False       # ...but under the 30-market floor...
    assert bool(row["edge_persisted"]) is False        # ...so not certified.


def test_cluster_floor_passes_when_breadth_is_adequate():
    # Same edge, but spread across 30 distinct held-out markets -> certified.
    rows = fair_market() + wallet_bets("W", in_wins=27, out_wins=27, n_in=30, n_out=30)
    row = compute_oos_validation(
        make_ledger(rows), cfg(min_oos_markets=30)).set_index("wallet").loc["W"]
    assert row["out_of_sample_markets"] == 30
    assert bool(row["edge_markets_ok"]) is True
    assert bool(row["edge_persisted"]) is True


def test_cluster_floor_is_inert_without_market_id_column():
    # Legacy ledgers with no market_id must validate exactly as before: the floor
    # goes inert (markets NaN, gate True) AND significance falls back to the t-test.
    rows = fair_market() + wallet_bets("P", in_wins=18, out_wins=18, n_in=20, n_out=20)
    ledger = make_ledger(rows).drop(columns=["market_id"])
    row = compute_oos_validation(
        ledger, cfg(min_oos_markets=30)).set_index("wallet").loc["P"]
    assert pd.isna(row["out_of_sample_markets"])
    assert bool(row["edge_markets_ok"]) is True
    # cluster p falls back to the t-test p on a legacy ledger, so it still certifies
    assert row["out_of_sample_cluster_p"] == pytest.approx(row["out_of_sample_residual_p"])
    assert bool(row["edge_persisted"]) is True


# --- split determinism (stable sort) ------------------------------------------

def test_split_is_deterministic_under_tied_boundary_timestamps():
    # All timestamps tied at the split boundary: an unstable sort would let row
    # order decide the halves. A stable sort makes the split a function of input
    # order, so a fixed input gives a fixed split regardless of pandas internals.
    df = pd.DataFrame({"timestamp": [7, 7, 7, 7, 7, 7], "id": [0, 1, 2, 3, 4, 5]})
    ins, out = split_in_sample_out_of_sample(df, oos_split=0.5)
    assert list(ins["id"]) == [0, 1, 2]      # stable: first-seen rows stay in-sample
    assert list(out["id"]) == [3, 4, 5]
    # and it agrees with a manual stable sort of the same frame
    ins2, out2 = split_in_sample_out_of_sample(df.iloc[::-1].copy(), oos_split=0.5)
    assert list(ins2["id"]) == [5, 4, 3]     # reversed input -> reversed, but deterministic


# --- cluster-robust significance: the _cluster_bootstrap_p primitive -----------

def test_cluster_bootstrap_p_significant_when_edge_is_spread_across_markets():
    from src.validate import _cluster_bootstrap_p
    rng = np.random.default_rng(1)
    # 40 markets, each one bet, almost all positive -> unambiguously significant.
    resid = np.array([0.1] * 38 + [-0.1] * 2, dtype=float)
    labels = np.arange(40)
    p = _cluster_bootstrap_p(resid, labels, rng, n_boot=2000)
    assert p < 0.05


def test_cluster_bootstrap_p_not_significant_when_edge_is_concentrated():
    from src.validate import _cluster_bootstrap_p
    rng = np.random.default_rng(2)
    # Positive bet-level mean, but it rides on ONE market out of five: resamples
    # that miss that market go negative, so the mean crosses 0 well above 5%.
    resid = np.array([0.5] * 20 + [-0.05] * 40, dtype=float)   # mean = +0.133
    labels = np.array([0] * 20 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10)
    assert resid.mean() > 0
    p = _cluster_bootstrap_p(resid, labels, rng, n_boot=4000)
    assert p > 0.05          # clustering dissolves the apparent significance


def test_cluster_bootstrap_p_is_nan_below_two_markets():
    from src.validate import _cluster_bootstrap_p
    rng = np.random.default_rng(3)
    p = _cluster_bootstrap_p(np.array([0.2, 0.3, 0.1]), np.array([0, 0, 0]), rng, n_boot=500)
    assert np.isnan(p)       # one resolution event cannot establish significance


def test_cluster_bootstrap_p_is_deterministic_for_a_fixed_seed():
    from src.validate import _cluster_bootstrap_p
    resid = np.array([0.1] * 25 + [-0.1] * 15, dtype=float)
    labels = np.arange(40)
    p1 = _cluster_bootstrap_p(resid, labels, np.random.default_rng([12345, 7]), 2000)
    p2 = _cluster_bootstrap_p(resid, labels, np.random.default_rng([12345, 7]), 2000)
    assert p1 == p2


# --- cluster-robust significance: end-to-end through compute_oos_validation ----

def concentrated_wallet(wallet, t0=1000):
    """A wallet whose held-out edge is real per-bet but rides on ONE dominant
    market: 1 big winning market + several small losing ones. The per-bet t-test
    (which counts each bet as independent evidence) certifies it; the market-block
    bootstrap does not, because dropping the one market flips the sign. In-sample
    half is broadly winning so it qualifies as a candidate."""
    rows, ts = [], t0
    # in-sample: 30 winning bets across 30 distinct markets (candidate, in_resid>0)
    for i in range(30):
        rows.append(bet(wallet, market_id=f"{wallet}_in_{i}", entry_price=0.5, timestamp=ts,
                        resolved=True, resolved_value=1.0)); ts += 1
    # held-out: 1 market with 26 winning bets, 4 markets with 1 losing bet each
    for i in range(26):
        rows.append(bet(wallet, market_id=f"{wallet}_big", entry_price=0.5, timestamp=ts,
                        resolved=True, resolved_value=1.0)); ts += 1
    for j in range(4):
        rows.append(bet(wallet, market_id=f"{wallet}_lose_{j}", entry_price=0.5, timestamp=ts,
                        resolved=True, resolved_value=0.0)); ts += 1
    return rows


def test_cluster_significance_gates_edge_significant_not_the_ttest():
    # The whole point of the cluster-robust test: a wallet the per-bet t-test
    # certifies but whose edge rides on a single resolution event is NOT certified.
    rows = fair_market(n=200) + concentrated_wallet("K")
    row = compute_oos_validation(
        make_ledger(rows), cfg(min_oos_markets=0)).set_index("wallet").loc["K"]
    assert row["out_of_sample_residual_edge"] > 0
    assert row["out_of_sample_residual_p"] < 0.05      # the t-test is fooled...
    assert row["out_of_sample_cluster_p"] > 0.05       # ...the cluster bootstrap is not
    assert bool(row["edge_significant"]) is False       # edge_significant follows the cluster p
    assert bool(row["edge_persisted"]) is False


def test_cluster_p_reported_and_gates_a_clean_multi_market_winner():
    rows = fair_market() + wallet_bets("P", in_wins=27, out_wins=27, n_in=30, n_out=30)
    row = compute_oos_validation(
        make_ledger(rows), cfg(min_oos_markets=30)).set_index("wallet").loc["P"]
    assert not pd.isna(row["out_of_sample_cluster_p"])
    assert row["out_of_sample_cluster_p"] < 0.05
    assert bool(row["edge_significant"]) is True
    assert bool(row["edge_persisted"]) is True


def test_single_market_wallet_is_not_cluster_significant():
    # Many bets, all in one market: cluster p is NaN (one event), so despite a big
    # raw edge the wallet is not certified — the degenerate end of the same rule.
    rows = fair_market() + wallet_bets("S", in_wins=9, out_wins=10, n_in=10, n_out=10,
                                       single_market=True)
    row = compute_oos_validation(
        make_ledger(rows), cfg(min_oos_markets=0)).set_index("wallet").loc["S"]
    assert row["out_of_sample_markets"] == 1
    assert pd.isna(row["out_of_sample_cluster_p"])
    assert bool(row["edge_significant"]) is False
    assert bool(row["edge_persisted"]) is False
