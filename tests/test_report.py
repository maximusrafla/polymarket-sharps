import numpy as np
import pandas as pd
import pytest

from src.report import build_why, forward_validity, render_report


def cfg(top_n=200):
    return {"ranking": {"top_n": top_n}}


def ranked_row(
    rank=1,
    wallet="0xA",
    score=0.5,
    out_of_sample_residual_edge=0.15,
    out_of_sample_residual_p=0.02,
    in_sample_residual_edge=0.18,
    out_of_sample_edge=0.2,
    out_of_sample_n=40,
    in_sample_edge=0.25,
    in_sample_n=40,
    edge_persisted=True,
    edge_significant=True,
    edge_magnitude_ok=True,
    copyable=True,
    copy_window=0.1,
    earliness=0.05,
    breadth=8,
    time_consistency=0.75,
    manufactured_record_flag=False,
    pattern_flag=None,
    regime_flag="stable",
    regime_watch=False,
    persisted_recent=False,
):
    return {
        "rank": rank,
        "wallet": wallet,
        "score": score,
        "out_of_sample_residual_edge": out_of_sample_residual_edge,
        "out_of_sample_residual_p": out_of_sample_residual_p,
        "in_sample_residual_edge": in_sample_residual_edge,
        "out_of_sample_edge": out_of_sample_edge,
        "out_of_sample_n": out_of_sample_n,
        "in_sample_edge": in_sample_edge,
        "in_sample_n": in_sample_n,
        "edge_persisted": edge_persisted,
        "edge_significant": edge_significant,
        "edge_magnitude_ok": edge_magnitude_ok,
        "copyable": copyable,
        "copy_window": copy_window,
        "earliness": earliness,
        "breadth": breadth,
        "time_consistency": time_consistency,
        "manufactured_record_flag": manufactured_record_flag,
        "pattern_flag": pattern_flag,
        "regime_flag": regime_flag,
        "regime_watch": regime_watch,
        "persisted_recent": persisted_recent,
    }


def make_ranked(rows):
    return pd.DataFrame(rows)


# --- build_why --------------------------------------------------------------

def test_build_why_shows_in_sample_and_out_of_sample_side_by_side():
    row = pd.Series(ranked_row(out_of_sample_edge=0.123, in_sample_edge=0.456))
    why = build_why(row)
    assert "0.123" in why
    assert "0.456" in why


def test_build_why_leads_with_skill_edge_and_shows_raw_edge_separately():
    row = pd.Series(ranked_row(out_of_sample_residual_edge=0.077, out_of_sample_edge=0.222))
    why = build_why(row)
    assert "skill edge" in why
    assert "0.077" in why                       # the residual (ranking) signal
    assert "favorite-longshot" in why           # raw edge is labelled as pre-adjustment
    assert "0.222" in why


def test_build_why_reports_persistence_status():
    persisted = build_why(pd.Series(ranked_row(edge_persisted=True)))
    not_persisted = build_why(pd.Series(ranked_row(edge_persisted=False)))
    assert "did NOT persist" not in persisted
    assert "did NOT persist" in not_persisted


def test_build_why_reports_copyability():
    copyable = build_why(pd.Series(ranked_row(copyable=True)))
    not_copyable = build_why(pd.Series(ranked_row(copyable=False)))
    assert "COPYABLE" in copyable
    assert "NOT copyable" in not_copyable


def test_build_why_distinguishes_below_magnitude_floor_from_noise():
    # significant but below the economic-magnitude floor -> reported distinctly
    row = pd.Series(ranked_row(edge_persisted=False, edge_significant=True, edge_magnitude_ok=False))
    why = build_why(row)
    assert "below the economic-magnitude floor" in why
    # plain non-persister (not significant) keeps the generic wording
    plain = build_why(pd.Series(ranked_row(edge_persisted=False, edge_significant=False, edge_magnitude_ok=False)))
    assert "below the economic-magnitude floor" not in plain
    assert "did NOT persist" in plain


def test_build_why_handles_unknown_time_consistency():
    row = pd.Series(ranked_row(time_consistency=np.nan))
    why = build_why(row)
    assert "unknown (too few bets)" in why


def test_build_why_omits_flag_notes_when_absent():
    row = pd.Series(ranked_row(manufactured_record_flag=False, pattern_flag=None))
    why = build_why(row)
    assert "manufactured_record_flag" not in why
    assert "pattern_flag" not in why


def test_build_why_omits_pattern_flag_note_when_nan_after_parquet_roundtrip():
    # A parquet round-trip of an object column mixing None with strings can
    # come back as NaN (float) rather than None — truthiness alone (`if
    # pattern_flag:`) would wrongly treat NaN as a real flag value.
    row = pd.Series(ranked_row(pattern_flag=np.nan))
    why = build_why(row)
    assert "pattern_flag" not in why


def test_build_why_includes_manufactured_flag_note_when_set():
    row = pd.Series(ranked_row(manufactured_record_flag=True))
    why = build_why(row)
    assert "manufactured_record_flag is set" in why
    assert "not an exclusion" in why


def test_build_why_includes_pattern_flag_note_when_set():
    row = pd.Series(ranked_row(pattern_flag="late_concentrated_entry"))
    why = build_why(row)
    assert "pattern_flag = late_concentrated_entry" in why


# --- build_why / render_report: metric D (regime change) --------------------

def test_build_why_flags_became_sharp_watch_without_recovery():
    row = pd.Series(ranked_row(edge_persisted=False, regime_watch=True,
                               persisted_recent=False, regime_flag="improving"))
    why = build_why(row)
    assert "BECAME sharp" in why
    assert "regime_watch" in why
    assert "persisted_recent" not in why  # not recovered, so no recovery clause


def test_build_why_notes_d3_recovery_on_watched_wallet():
    row = pd.Series(ranked_row(edge_persisted=False, regime_watch=True,
                               persisted_recent=True, regime_flag="improving_confirmed"))
    why = build_why(row)
    assert "BECAME sharp" in why
    assert "persisted_recent" in why  # recovery clause present


def test_build_why_notes_decaying_regime_without_watch():
    row = pd.Series(ranked_row(regime_watch=False, regime_flag="decaying"))
    why = build_why(row)
    assert "regime_flag = decaying" in why


def test_build_why_omits_regime_note_for_stable_wallets():
    row = pd.Series(ranked_row(regime_watch=False, regime_flag="stable"))
    why = build_why(row)
    assert "regime_flag" not in why
    assert "BECAME sharp" not in why


def test_render_report_summarizes_regime_watch_and_actionable_set():
    rows = [
        # certified persister, copyable -> in the actionable set
        ranked_row(rank=1, wallet="0xpersist", edge_persisted=True, copyable=True,
                   regime_watch=False, persisted_recent=False, regime_flag="stable"),
        # recovered became-sharp, copyable -> widens the actionable set, NOT persisted
        ranked_row(rank=2, wallet="0xrecovered", edge_persisted=False, copyable=True,
                   regime_watch=True, persisted_recent=True, regime_flag="improving_confirmed"),
        # on watch but not recovered -> counted in watch, not actionable
        ranked_row(rank=3, wallet="0xwatch", edge_persisted=False, copyable=True,
                   regime_watch=True, persisted_recent=False, regime_flag="improving"),
    ]
    report = render_report(make_ranked(rows), cfg(top_n=10))
    assert "Regime change (metric D):" in report
    assert "2 wallets are on the became-sharp" in report          # two regime_watch=True
    assert "1 earned `persisted_recent`" in report                # one recovered
    # actionable = (edge_persisted OR persisted_recent) AND copyable = persist + recovered = 2
    assert "**2 wallets**" in report


# --- forward_validity ---------------------------------------------------------

def test_forward_validity_positive_when_in_sample_predicts_held_out():
    # in-sample skill edge monotonically predicts held-out skill edge -> rho ~ +1
    rows = [ranked_row(wallet=f"0x{i}", in_sample_residual_edge=i / 10, out_of_sample_residual_edge=i / 10)
            for i in range(10)]
    rho, p, n = forward_validity(make_ranked(rows))
    assert n == 10
    assert rho == pytest.approx(1.0)


def test_forward_validity_ignores_wallets_missing_either_half():
    rows = [ranked_row(wallet=f"0x{i}", in_sample_residual_edge=i / 10, out_of_sample_residual_edge=i / 10)
            for i in range(9)]
    rows.append(ranked_row(wallet="0xNaNout", in_sample_residual_edge=0.5, out_of_sample_residual_edge=np.nan))
    rho, p, n = forward_validity(make_ranked(rows))
    assert n == 9  # the wallet with no held-out edge is dropped, not counted


def test_forward_validity_nan_when_too_few_points():
    rows = [ranked_row(wallet=f"0x{i}", in_sample_residual_edge=i, out_of_sample_residual_edge=i)
            for i in range(3)]
    rho, p, n = forward_validity(make_ranked(rows))
    assert np.isnan(rho)
    assert n == 3


# --- render_report ------------------------------------------------------------

def test_render_report_summarizes_persisted_and_copyable_counts():
    rows = [
        ranked_row(rank=1, wallet="0xsharp_copyable", edge_persisted=True, copyable=True),
        ranked_row(rank=2, wallet="0xsharp_unfollowable", edge_persisted=True, copyable=False),
        ranked_row(rank=3, wallet="0xnope", edge_persisted=False, copyable=False),
    ]
    report = render_report(make_ranked(rows), cfg(top_n=10))
    assert "Actionable set:" in report
    assert "2 wallets have validated skill edge" in report
    assert "1 are also" in report  # 1 of the 2 persisters is copyable


def test_render_report_truncates_to_top_n():
    rows = [ranked_row(rank=i, wallet=f"0x{i}") for i in range(1, 6)]
    report = render_report(make_ranked(rows), cfg(top_n=2))
    assert "0x1" in report
    assert "0x2" in report
    assert "0x3" not in report


def test_render_report_notes_total_ranked_count_even_when_truncated():
    rows = [ranked_row(rank=i, wallet=f"0x{i}") for i in range(1, 6)]
    report = render_report(make_ranked(rows), cfg(top_n=2))
    assert "top 2 of 5" in report


def test_render_report_orders_by_rank_not_input_order():
    rows = [ranked_row(rank=2, wallet="second"), ranked_row(rank=1, wallet="first")]
    report = render_report(make_ranked(rows), cfg(top_n=10))
    assert report.index("first") < report.index("second")


def test_render_report_includes_every_wallet_when_top_n_covers_all():
    rows = [ranked_row(rank=i, wallet=f"0x{i}") for i in range(1, 4)]
    report = render_report(make_ranked(rows), cfg(top_n=200))
    for i in range(1, 4):
        assert f"0x{i}" in report
