import numpy as np
import pandas as pd
import pytest

from src.rank import (
    compute_copyable,
    compute_edge_score,
    compute_reliability,
    compute_score,
    rank_wallets,
)


def cfg(**overrides):
    base = {
        "scoring": {"min_sample_size": 30},
        "ranking": {
            "w_edge": 1.0,
            "w_copy_window": 0.5,
            "w_earliness": 0.25,
            "w_time_consistency": 0.1,
            "breadth_full_credit": 10,
            "non_persisted_penalty": 0.3,
            "copyable_window_floor": 0.0,
        },
    }
    for section, values in overrides.items():
        base[section].update(values)
    return base


def wallet_row(
    wallet="A",
    out_of_sample_n=30,
    out_of_sample_residual_edge=0.1,
    out_of_sample_edge=0.1,
    in_sample_edge=0.1,
    breadth=10,
    edge_persisted=True,
    copy_window=0.0,
    earliness=0.0,
    time_consistency=0.5,
    manufactured_record_flag=False,
    pattern_flag=None,
):
    return {
        "wallet": wallet,
        "out_of_sample_n": out_of_sample_n,
        "out_of_sample_residual_edge": out_of_sample_residual_edge,
        "out_of_sample_edge": out_of_sample_edge,
        "in_sample_residual_edge": out_of_sample_residual_edge,
        "in_sample_edge": in_sample_edge,
        "in_sample_n": out_of_sample_n,
        "breadth": breadth,
        "edge_persisted": edge_persisted,
        "copy_window": copy_window,
        "earliness": earliness,
        "time_consistency": time_consistency,
        "manufactured_record_flag": manufactured_record_flag,
        "pattern_flag": pattern_flag,
        "edge": in_sample_edge,
        "skill_edge": out_of_sample_residual_edge,
        "sample_size": out_of_sample_n,
    }


def make_df(rows):
    return pd.DataFrame(rows)


# --- compute_reliability --------------------------------------------------

def test_reliability_zero_with_no_out_of_sample_bets():
    df = make_df([wallet_row(out_of_sample_n=0, edge_persisted=False)])
    reliability = compute_reliability(df, cfg())
    assert reliability.iloc[0] == pytest.approx(0.0)


def test_reliability_increases_with_sample_size():
    df = make_df([wallet_row(wallet="small", out_of_sample_n=5), wallet_row(wallet="big", out_of_sample_n=300)])
    reliability = compute_reliability(df, cfg())
    assert reliability.iloc[1] > reliability.iloc[0]


def test_reliability_capped_by_breadth_full_credit():
    df = make_df([wallet_row(wallet="a", breadth=10), wallet_row(wallet="b", breadth=1000)])
    reliability = compute_reliability(df, cfg())
    # breadth beyond breadth_full_credit gives no extra credit
    assert reliability.iloc[0] == pytest.approx(reliability.iloc[1])


def test_reliability_discounted_but_not_zeroed_when_not_persisted():
    persisted = make_df([wallet_row(edge_persisted=True)])
    not_persisted = make_df([wallet_row(edge_persisted=False)])
    r_persisted = compute_reliability(persisted, cfg()).iloc[0]
    r_not_persisted = compute_reliability(not_persisted, cfg()).iloc[0]
    assert r_not_persisted == pytest.approx(r_persisted * 0.3)
    assert r_not_persisted > 0


# --- compute_edge_score -----------------------------------------------------

def test_edge_score_uses_out_of_sample_residual_edge_not_raw_or_in_sample():
    # scoring keys on the favorite-longshot-neutralized OOS skill edge (0.2),
    # not the raw OOS edge (0.9) and not the in-sample edge (0.9).
    df = make_df([wallet_row(out_of_sample_residual_edge=0.2, out_of_sample_edge=0.9, in_sample_edge=0.9)])
    score = compute_edge_score(df, cfg())
    assert score.iloc[0] == pytest.approx(0.2)


def test_edge_score_includes_copy_window_as_first_class_term():
    baseline = compute_edge_score(make_df([wallet_row()]), cfg()).iloc[0]
    with_copy_window = compute_edge_score(make_df([wallet_row(copy_window=0.4)]), cfg()).iloc[0]
    assert with_copy_window == pytest.approx(baseline + 0.5 * 0.4)


def test_edge_score_does_not_double_count_earliness():
    # earliness coincides with copy_window in this data, so it must NOT enter the
    # score independently (that would double-count one forward-drift signal).
    baseline = compute_edge_score(make_df([wallet_row(earliness=0.0)]), cfg()).iloc[0]
    with_earliness = compute_edge_score(make_df([wallet_row(earliness=0.4)]), cfg()).iloc[0]
    assert with_earliness == pytest.approx(baseline)


def test_edge_score_time_consistency_centered_on_neutral():
    neutral = compute_edge_score(make_df([wallet_row(time_consistency=0.5)]), cfg()).iloc[0]
    nan_consistency = compute_edge_score(make_df([wallet_row(time_consistency=np.nan)]), cfg()).iloc[0]
    consistent = compute_edge_score(make_df([wallet_row(time_consistency=1.0)]), cfg()).iloc[0]
    inconsistent = compute_edge_score(make_df([wallet_row(time_consistency=0.0)]), cfg()).iloc[0]
    assert nan_consistency == pytest.approx(neutral)
    assert consistent > neutral
    assert inconsistent < neutral


def test_edge_score_missing_values_contribute_zero_not_penalty():
    df = make_df([wallet_row(out_of_sample_residual_edge=np.nan, copy_window=np.nan, earliness=np.nan, time_consistency=np.nan)])
    score = compute_edge_score(df, cfg())
    assert score.iloc[0] == pytest.approx(0.0)


# --- compute_copyable (metric B) -------------------------------------------

def test_copyable_true_only_for_positive_copy_window():
    df = make_df([
        wallet_row(wallet="pos", copy_window=0.05),
        wallet_row(wallet="zero", copy_window=0.0),
        wallet_row(wallet="neg", copy_window=-0.1),
        wallet_row(wallet="nan", copy_window=np.nan),
    ])
    copyable = compute_copyable(df, cfg())
    assert list(copyable) == [True, False, False, False]


def test_copyable_respects_configurable_floor():
    df = make_df([wallet_row(wallet="a", copy_window=0.01), wallet_row(wallet="b", copy_window=0.05)])
    copyable = compute_copyable(df, cfg(ranking={"copyable_window_floor": 0.02}))
    assert list(copyable) == [False, True]


def test_copyable_is_independent_of_persistence_and_score():
    # a sharp-but-unfollowable wallet: persisted skill but non-positive copy_window
    sharp_unfollowable = compute_score(
        make_df([wallet_row(out_of_sample_residual_edge=0.3, edge_persisted=True, copy_window=-0.05)]), cfg()
    )
    assert not bool(sharp_unfollowable["copyable"].iloc[0])
    assert sharp_unfollowable["score"].iloc[0] > 0  # still ranks on skill


def test_copyable_flag_does_not_change_score_or_rank_order():
    # copyable is a metadata filter, never a score input: two wallets identical
    # except copy_window sign-at-zero must keep score determined only by the
    # existing terms (copy_window itself still feeds the score, as before).
    a = compute_score(make_df([wallet_row(wallet="a", copy_window=0.0)]), cfg())
    assert "copyable" in a.columns
    # score does not include a separate copyable term (copy_window=0 contributes 0)
    assert a["score"].iloc[0] == pytest.approx(
        a["reliability"].iloc[0] * a["edge_score"].iloc[0]
    )


# --- compute_score / rank_wallets ------------------------------------------

def test_score_is_reliability_times_edge_score():
    df = make_df([wallet_row()])
    scored = compute_score(df, cfg())
    assert scored["score"].iloc[0] == pytest.approx(
        scored["reliability"].iloc[0] * scored["edge_score"].iloc[0]
    )


def test_rank_wallets_sorts_descending_and_assigns_rank():
    df = make_df(
        [
            wallet_row(wallet="low", out_of_sample_residual_edge=0.01),
            wallet_row(wallet="high", out_of_sample_residual_edge=0.5),
            wallet_row(wallet="mid", out_of_sample_residual_edge=0.1),
        ]
    )
    ranked = rank_wallets(df, cfg())
    assert list(ranked["wallet"]) == ["high", "mid", "low"]
    assert list(ranked["rank"]) == [1, 2, 3]


def test_rank_wallets_keeps_every_wallet_even_with_zero_reliability():
    df = make_df(
        [
            wallet_row(wallet="strong", out_of_sample_n=300, out_of_sample_residual_edge=0.3),
            wallet_row(wallet="untested", out_of_sample_n=0, out_of_sample_residual_edge=np.nan, edge_persisted=False),
        ]
    )
    ranked = rank_wallets(df, cfg())
    assert set(ranked["wallet"]) == {"strong", "untested"}
    assert ranked.loc[ranked["wallet"] == "untested", "score"].iloc[0] == pytest.approx(0.0)


def test_rank_wallets_flags_never_affect_score():
    flagged = make_df([wallet_row(wallet="flagged", manufactured_record_flag=True, pattern_flag="late_concentrated_entry")])
    unflagged = make_df([wallet_row(wallet="clean")])
    flagged_score = compute_score(flagged, cfg())["score"].iloc[0]
    unflagged_score = compute_score(unflagged, cfg())["score"].iloc[0]
    assert flagged_score == pytest.approx(unflagged_score)


def test_rank_wallets_non_persisted_positive_edge_ranks_below_persisted_equal_edge():
    df = make_df(
        [
            wallet_row(wallet="persisted", out_of_sample_residual_edge=0.2, edge_persisted=True),
            wallet_row(wallet="lucky", out_of_sample_residual_edge=0.2, edge_persisted=False),
        ]
    )
    ranked = rank_wallets(df, cfg())
    assert list(ranked["wallet"]) == ["persisted", "lucky"]
