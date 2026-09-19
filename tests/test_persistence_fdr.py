"""Tests for scripts/audit_persistence_fdr.py — the hardened-gate cluster null.

The whole point of that script is that its vectorized replica of the CURRENT
`src.validate` gate is exact: an FDR measured against a lookalike gate is the
mistake the script exists to correct. So the load-bearing test is an equality
assert against `compute_oos_validation` itself on hand-built fixtures where the
gate outcomes are known by construction.

Also covered: the lexicographic market-code renumbering (which is what makes the
replica's bootstrap p-values bit-identical rather than merely distributionally
equal), the held-out distinct-market count, and the two pure summary statistics
(design effect, Monte-Carlo error).
"""

import numpy as np
import pandas as pd

from scripts.audit_persistence_fdr import (
    _lexicographic_codes,
    benjamini_hochberg,
    build_cells,
    permute_cells_within_size,
    structural_retention,
    design_effect,
    hardened_gate,
    held_out_market_counts,
    mc_error,
    split_stats,
)
from scripts.audit_speed_gradient import permute_wallets_within_market
from src.features import fit_price_baseline, residual_edge_per_bet
from src.validate import (
    CLUSTER_BOOTSTRAP_SEED,
    _wallet_seed,
    compute_oos_validation,
)

LEDGER_COLS = [
    "wallet", "market_id", "token_id", "outcome", "side", "entry_price", "size",
    "timestamp", "resolved", "resolved_value", "question", "slug", "tx_hash",
]


def bet(wallet, market_id, entry_price, resolved_value, timestamp, side="BUY", resolved=True):
    return {
        "wallet": wallet, "market_id": market_id, "token_id": "t", "outcome": "Yes",
        "side": side, "entry_price": entry_price, "size": 10.0, "timestamp": timestamp,
        "resolved": resolved, "resolved_value": resolved_value, "question": "q",
        "slug": "s", "tx_hash": "tx",
    }


def cfg(**over):
    scoring = {
        "oos_split": 0.5,
        "min_bets_per_half": 4,
        "oos_significance_alpha": 0.05,
        "price_baseline_bins": 4,
        "min_skill_edge": 0.02,
        "min_oos_markets": 3,
        "oos_bootstrap_resamples": 500,
    }
    scoring.update(over)
    return {"scoring": scoring}


def as_data(ledger, config):
    """Build the flat-array `data` dict `hardened_gate` consumes, exactly as
    `prepare()` does from the real ledger (categorical wallet/market columns,
    lexicographic market codes, residual edge off the same price baseline)."""
    led = ledger.copy()
    led["wallet"] = led["wallet"].astype("category")
    led["market_id"] = led["market_id"].astype("category")
    keep = (led["side"] == "BUY").to_numpy() & led["resolved"].to_numpy(dtype=bool)
    wallets = led["wallet"].cat.categories.to_numpy()
    wcode = led["wallet"].cat.codes.to_numpy().astype(np.int64)[keep]
    mcode = _lexicographic_codes(led["market_id"])[keep]
    ts = led["timestamp"].to_numpy(dtype=float)[keep]
    frame = pd.DataFrame({
        "entry_price": led["entry_price"].to_numpy(dtype=float)[keep],
        "resolved_value": led["resolved_value"].to_numpy(dtype=float)[keep],
    })
    baseline = fit_price_baseline(frame, config["scoring"]["price_baseline_bins"])
    resid = residual_edge_per_bet(frame, baseline)
    return {
        "wcode": wcode, "mcode": mcode, "ts": ts, "resid": resid, "wallets": wallets,
        "n_wallets": wallets.size, "n_markets": int(mcode.max()) + 1 if mcode.size else 1,
    }


def params_from(config):
    sc = config["scoring"]
    return {
        "oos_split": sc["oos_split"], "min_per_half": sc["min_bets_per_half"],
        "alpha": sc["oos_significance_alpha"], "min_skill_edge": sc["min_skill_edge"],
        "min_oos_markets": sc["min_oos_markets"], "n_boot": sc["oos_bootstrap_resamples"],
    }


def run_both(ledger, config):
    """(replica flags, wallet order, src.validate output) for the same ledger."""
    data = as_data(ledger, config)
    seed = lambda wid: np.random.default_rng(  # noqa: E731
        [CLUSTER_BOOTSTRAP_SEED, _wallet_seed(data["wallets"][wid])])
    flags = hardened_gate(data, params_from(config), seed, full=True)
    truth = compute_oos_validation(ledger, config).set_index("wallet")
    return flags, data["wallets"], truth


# --------------------------------------------------------------------------- #
# The load-bearing equality assert                                             #
# --------------------------------------------------------------------------- #
def _mixed_ledger():
    """Three wallets, each failing (or clearing) the gate for a different reason.

    Everyone buys at 0.5, so the price baseline collapses to the global win rate
    (96/120 = 0.8) and each wallet's residual edge is just (win rate - 0.8).

    - `sharp`: 40 bets over 20 markets; 90% wins in-sample, all 20 held-out bets
      (10 markets) win -> +0.2 held-out residual, clears every gate.
    - `narrow`: the identical record piled into 2 markets -> significant and
      material, but blocked by the cluster-count floor.
    - `flat`: wins half the time -> negative residual, never a candidate.
    """
    rows = []
    t = 0
    for name, market in (("sharp", lambda i: f"s{i//2:02d}"),
                         ("narrow", lambda i: f"n{i % 2}")):
        for i in range(40):
            # in-sample (first 20) 90% wins; held-out (last 20) all wins
            win = 1.0 if (i >= 20 or i % 10 != 9) else 0.0
            rows.append(bet(name, market(i), 0.5, win, t)); t += 1
    for i in range(40):
        rows.append(bet("flat", f"f{i//2:02d}", 0.5, float(i % 2), t)); t += 1
    return pd.DataFrame(rows, columns=LEDGER_COLS)


def test_replica_matches_src_validate_on_mixed_ledger():
    config = cfg()
    flags, wallets, truth = run_both(_mixed_ledger(), config)
    for i, w in enumerate(wallets):
        for col in ("edge_significant", "edge_magnitude_ok", "edge_markets_ok",
                    "edge_persisted"):
            assert bool(flags[col][i]) == bool(truth.loc[w, col]), (w, col)


def test_replica_reproduces_cluster_p_exactly():
    """Not merely 'the same gate outcome' — the identical bootstrap draw. This is
    what the lexicographic market renumbering buys, and it is why the audit can
    claim its real arm IS the shipped pipeline rather than a re-derivation."""
    config = cfg()
    flags, wallets, truth = run_both(_mixed_ledger(), config)
    for i, w in enumerate(wallets):
        mine, theirs = flags["cluster_p"][i], truth.loc[w, "out_of_sample_cluster_p"]
        if np.isnan(theirs):
            assert np.isnan(mine), w
        else:
            assert mine == theirs, (w, mine, theirs)


def test_mixed_ledger_verdicts_are_as_constructed():
    """Guards the fixture itself: if all three wallets ever agree, the equality
    test above would pass vacuously."""
    config = cfg()
    flags, wallets, _ = run_both(_mixed_ledger(), config)
    idx = {w: i for i, w in enumerate(wallets)}
    per = flags["edge_persisted"]
    assert per[idx["sharp"]]
    # narrow is blocked by the CLUSTER-COUNT floor alone: it is significant and
    # material, so if the market gate ever stopped biting this would go green.
    assert not per[idx["narrow"]]
    assert flags["edge_significant"][idx["narrow"]]
    assert flags["edge_magnitude_ok"][idx["narrow"]]
    assert not flags["edge_markets_ok"][idx["narrow"]]
    # flat never even becomes a candidate
    assert not per[idx["flat"]] and not flags["candidate"][idx["flat"]]
    mm = config["scoring"]["min_oos_markets"]
    assert flags["out_markets"][idx["narrow"]] < mm <= flags["out_markets"][idx["sharp"]]


def test_sell_and_unresolved_rows_are_excluded_like_the_validator():
    config = cfg()
    base = _mixed_ledger()
    noise = pd.DataFrame(
        [bet("sharp", "zz", 0.9, 0.0, 999, side="SELL")]
        + [bet("sharp", "zy", 0.9, None, 1000, resolved=False)],
        columns=LEDGER_COLS,
    )
    flags_a, wallets, _ = run_both(base, config)
    flags_b, wallets_b, truth_b = run_both(pd.concat([base, noise], ignore_index=True), config)
    assert list(wallets) == [w for w in wallets_b if w in set(wallets)]
    i = list(wallets).index("sharp")
    j = list(wallets_b).index("sharp")
    assert flags_a["out_mean"][i] == flags_b["out_mean"][j]
    assert bool(flags_b["edge_persisted"][j]) == bool(truth_b.loc["sharp", "edge_persisted"])


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def test_lexicographic_codes_sort_like_their_strings():
    s = pd.Series(pd.Categorical(["m10", "m2", "m1", "m2", "m10"]))
    codes = _lexicographic_codes(s)
    order = np.argsort(codes, kind="stable")
    assert list(s.to_numpy()[order]) == sorted(s.to_numpy())
    # same string -> same code, different string -> different code
    assert codes[1] == codes[3] and codes[0] == codes[4]
    assert len(set(codes.tolist())) == 3


def test_split_stats_splits_and_gates_like_the_validator():
    # one wallet, 10 bets, residuals 0..9 in time order; oos_split 0.5 -> 5/5
    wcode = np.zeros(10, dtype=np.int64)
    ts = np.arange(10, dtype=float)
    resid = np.arange(10, dtype=float)
    _, starts, k, m, in_mean, out_mean, out_var = split_stats(
        wcode, ts, resid, 1, 0.5, min_per_half=5)
    assert starts[0] == 0 and k[0] == 5 and m[0] == 5
    assert in_mean[0] == 2.0 and out_mean[0] == 7.0
    assert np.isclose(out_var[0], np.var(np.arange(5, 10), ddof=1))
    # raising the floor above the half size NaNs both means (the validator's
    # "insufficient data", never a measured zero)
    _, _, _, _, in2, out2, _ = split_stats(wcode, ts, resid, 1, 0.5, min_per_half=6)
    assert np.isnan(in2[0]) and np.isnan(out2[0])


def test_split_stats_is_stable_on_tied_timestamps():
    """Ties must resolve to input row order, matching validate.py's mergesort —
    otherwise the split (and the certified set) is not reproducible."""
    wcode = np.zeros(4, dtype=np.int64)
    ts = np.zeros(4)
    resid = np.array([1.0, 2.0, 3.0, 4.0])
    _, _, k, _, in_mean, out_mean, _ = split_stats(wcode, ts, resid, 1, 0.5, min_per_half=1)
    assert k[0] == 2
    assert in_mean[0] == 1.5 and out_mean[0] == 3.5


def test_held_out_market_counts_counts_distinct_markets_in_the_held_out_half():
    # wallet 0: 4 bets, held-out half = last 2, both in market 7 -> 1 market
    # wallet 1: 4 bets, held-out half = markets 8 and 9        -> 2 markets
    wcode = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    mcode = np.array([1, 2, 7, 7, 3, 4, 8, 9], dtype=np.int64)
    ts = np.arange(8, dtype=float)
    resid = np.zeros(8)
    order, starts, k, m, *_ = split_stats(wcode, ts, resid, 2, 0.5, min_per_half=1)
    counts = held_out_market_counts(order, starts, k, m, wcode, mcode, 2, 10)
    assert counts.tolist() == [1, 2]


def test_design_effect_is_one_when_the_null_matches_independence():
    # two "wallets": running sums crafted so Var(null) == mean(s^2/m) exactly.
    # Var(null) is the UNBIASED (ddof=1) variance over replicates — with a
    # population divisor D would read (S-1)/S too small, a 33% error at S=3.
    n = 4
    means = np.array([[1.0, -1.0], [3.0, 5.0], [1.0, -1.0], [3.0, 5.0]])  # 4 shuffles x 2
    mean_sum = means.sum(axis=0)
    mean_sq = (means ** 2).sum(axis=0)
    var_null = means.var(axis=0, ddof=1)
    indep_sum = var_null * n                  # so mean(indep) == var_null -> D == 1
    D = design_effect(mean_sq, mean_sum, indep_sum, n)
    assert np.allclose(D, 1.0)
    # halving the independence reference doubles D
    assert np.allclose(design_effect(mean_sq, mean_sum, indep_sum / 2, n), 2.0)
    # an over-constrained null (no spread across shuffles) reads D == 0
    flat = np.zeros(2)
    assert np.allclose(design_effect(flat, flat, indep_sum, n), 0.0)


def test_design_effect_needs_two_shuffles():
    assert np.isnan(design_effect(np.zeros(2), np.zeros(2), np.ones(2), 1)).all()


def test_mc_error_is_the_standard_error_of_the_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(mc_error(x), x.std(ddof=1) / 2.0)
    assert np.isnan(mc_error([5.0]))


# --------------------------------------------------------------------------- #
# Null B2 — the cell permutation and the frozenness diagnostic                 #
# --------------------------------------------------------------------------- #
def _cell_tape():
    """wallet 0: 3 bets in market 10, 1 in 11; wallet 1: 1 each in 12 and 13;
    wallet 2: 1 in 10, 2 in 14."""
    wcode = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    mcode = np.array([10, 10, 10, 11, 12, 13, 10, 14, 14], dtype=np.int64)
    return wcode, mcode


def test_build_cells_finds_the_wallet_market_blocks():
    wcode, mcode = _cell_tape()
    order, owner, size = build_cells(wcode, mcode)
    assert sorted(zip(owner.tolist(), size.tolist())) == [
        (0, 1), (0, 3), (1, 1), (1, 1), (2, 1), (2, 2)]
    # `order` must lay the cells out contiguously
    assert np.all(np.diff(np.lexsort((mcode[order], wcode[order]))) == 1)


def _disjoint_tape():
    """Three wallets on markets no one else touches, so a permuted cell can never
    land on a market its new owner already holds — the one regime where B2's
    cell-size profile is preserved EXACTLY."""
    wcode = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    mcode = np.array([10, 10, 10, 11, 20, 20, 21, 30, 31, 32], dtype=np.int64)
    return wcode, mcode


def test_permute_cells_preserves_bet_counts_and_cell_size_profile():
    wcode, mcode = _disjoint_tape()
    order, owner, size = build_cells(wcode, mcode)
    rng = np.random.default_rng(7)
    seen_moved = False
    for _ in range(40):
        out = permute_cells_within_size(order, owner, size, rng)
        # exact bet count per wallet
        assert np.array_equal(np.bincount(out, minlength=3), np.bincount(wcode, minlength=3))
        # exact cell-size profile per wallet
        _, o2, s2 = build_cells(out, mcode)
        for w in range(3):
            assert sorted(s2[o2 == w].tolist()) == sorted(size[owner == w].tolist())
        # bets of one original (wallet, market) cell must move together
        for lo, hi in zip(np.append(0, np.cumsum(size))[:-1], np.cumsum(size)):
            idx = order[lo:hi]
            assert len(set(out[idx].tolist())) == 1
        seen_moved |= bool((out != wcode).any())
    assert seen_moved, "the permutation never moved anything — it is not a null"


def test_permute_cells_preserves_bet_counts_even_when_cells_collide():
    """Two same-size cells of the SAME market can land on one wallet, merging into
    a bigger cell and costing that wallet a distinct market. Bet counts are still
    exact, and the effect is conservative (fewer held-out markets = harder to clear
    the cluster-count floor). Documented in `permute_cells_within_size`."""
    wcode, mcode = _cell_tape()
    order, owner, size = build_cells(wcode, mcode)
    rng = np.random.default_rng(5)
    for _ in range(40):
        out = permute_cells_within_size(order, owner, size, rng)
        assert np.array_equal(np.bincount(out, minlength=3), np.bincount(wcode, minlength=3))
        for lo, hi in zip(np.append(0, np.cumsum(size))[:-1], np.cumsum(size)):
            assert len(set(out[order[lo:hi]].tolist())) == 1
        # markets a wallet holds can only ever shrink, never grow, via a merge
        _, o2, _ = build_cells(out, mcode)
        for w in range(3):
            assert int((o2 == w).sum()) <= int((owner == w).sum())


def test_permute_cells_only_relabels_it_never_rewrites_the_tape():
    """B2's whole claim is that it moves ATTRIBUTION, never the bets. The output
    must be a rearrangement of the same wallet labels over the same rows."""
    wcode, mcode = _cell_tape()
    order, owner, size = build_cells(wcode, mcode)
    out = permute_cells_within_size(order, owner, size, np.random.default_rng(3))
    assert out.shape == wcode.shape
    assert sorted(out.tolist()) == sorted(wcode.tolist())


def test_structural_retention_is_the_wallets_share_of_each_market():
    wcode, mcode = _cell_tape()
    ret = structural_retention(wcode, mcode, 3)
    # wallet 0: 3 bets in a 4-bet market (share 3/4) + 1 bet in a market it owns
    # alone (share 1) -> (3*0.75 + 1*1)/4
    assert np.isclose(ret[0], (3 * 0.75 + 1 * 1.0) / 4)
    # wallet 1 is alone in both its markets -> fully frozen
    assert np.isclose(ret[1], 1.0)
    # wallet 2: 1 bet in the 4-bet market (share 1/4) + 2 bets it owns alone
    assert np.isclose(ret[2], (1 * 0.25 + 2 * 1.0) / 3)


def test_structural_retention_matches_the_realised_permutation():
    """The closed form must agree with what the permutation actually does, or the
    frozenness argument rests on algebra nobody checked."""
    rng = np.random.default_rng(11)
    n = 4000
    mcode = rng.integers(0, 200, size=n).astype(np.int64)
    wcode = rng.integers(0, 12, size=n).astype(np.int64)
    ret = structural_retention(wcode, mcode, 12)
    kept = np.zeros(12)
    trials = 60
    for _ in range(trials):
        out = permute_wallets_within_market(wcode, mcode, rng)
        kept += np.bincount(wcode[out == wcode], minlength=12)
    realised = kept / (trials * np.bincount(wcode, minlength=12))
    assert np.allclose(realised, ret, atol=0.03)


# --------------------------------------------------------------------------- #
# BH step-up                                                                   #
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_rejects_the_step_up_prefix():
    # m=5, q=0.1 -> thresholds .02 .04 .06 .08 .10. STEP-UP: find the largest rank
    # whose p clears its own threshold and reject everything at or below it — so
    # .05 (3rd, threshold .06) pulls in .039 even though .039 > its own .04.
    p = np.array([0.001, 0.039, 0.05, 0.5, 0.9])
    assert benjamini_hochberg(p, 0.1).tolist() == [True, True, True, False, False]
    # nothing beyond rank 1 clears -> only the single smallest is rejected
    p2 = np.array([0.001, 0.9, 0.5, 0.7, 0.8])
    assert benjamini_hochberg(p2, 0.1).tolist() == [True, False, False, False, False]


def test_benjamini_hochberg_rejects_nothing_when_no_p_clears_its_threshold():
    assert not benjamini_hochberg(np.array([0.03, 0.4, 0.6]), 0.05).any()
    assert benjamini_hochberg(np.array([]), 0.1).size == 0


def test_benjamini_hochberg_treats_nan_as_never_rejected():
    p = np.array([0.0001, np.nan, np.nan, np.nan])
    out = benjamini_hochberg(p, 0.1)
    assert out.tolist() == [True, False, False, False]


def test_benjamini_hochberg_p_floor_costs_resolution_at_the_smallest_ranks():
    """Why `--bh-boot` exists. At 2000 resamples the p-floor is 1/2001, ABOVE BH's
    rank-1 threshold q/m for a 691-wide family — so a lone strong wallet cannot be
    rejected at all, while a higher-resolution p for the identical evidence can."""
    m, q = 691, 0.10
    lone_at_floor = np.full(m, 1.0)
    lone_at_floor[0] = 1.0 / 2001
    assert not benjamini_hochberg(lone_at_floor, q).any()
    lone_resolved = np.full(m, 1.0)
    lone_resolved[0] = 1.0 / 50001
    assert benjamini_hochberg(lone_resolved, q).sum() == 1
    # with enough wallets tied at the floor the step-up climbs past it, so the
    # floor is a resolution limit at the top ranks, not a blanket veto
    many_at_floor = np.full(m, 1.0)
    many_at_floor[:50] = 1.0 / 2001
    assert benjamini_hochberg(many_at_floor, q).sum() == 50
