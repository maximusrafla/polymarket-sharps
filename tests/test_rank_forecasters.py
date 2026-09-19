"""Tests for src/rank_forecasters.py — funnel/selection logic and that the two
renderers produce output without error on a small hand-built table."""

import pandas as pd

from src.rank_forecasters import category_funnel, default_view, render_html, render_markdown


def _table():
    # three cells: one full winner, one persisted-but-uncopyable, one thin.
    return pd.DataFrame([
        {"wallet": "0x" + "a" * 40, "category": "politics", "sample_size": 40, "breadth": 30,
         "bets_per_month": 4.0, "skill_edge_all": 0.06, "raw_edge_all": 0.05,
         "out_of_sample_skill_edge": 0.05, "out_of_sample_skill_p": 0.001,
         "edge_persisted": True, "reachability_60": 0.8,
         "copyable_skill_edge_60": 0.05, "copyable_skill_edge_60_cents": 5.0,
         "clears_magnitude_floor": True, "is_winner": True},
        {"wallet": "0x" + "b" * 40, "category": "politics", "sample_size": 35, "breadth": 20,
         "bets_per_month": 2.0, "skill_edge_all": 0.04, "raw_edge_all": 0.03,
         "out_of_sample_skill_edge": 0.03, "out_of_sample_skill_p": 0.01,
         "edge_persisted": True, "reachability_60": 0.1,
         "copyable_skill_edge_60": float("nan"), "copyable_skill_edge_60_cents": float("nan"),
         "clears_magnitude_floor": False, "is_winner": False},
        {"wallet": "0x" + "c" * 40, "category": "sports_nba", "sample_size": 3, "breadth": 3,
         "bets_per_month": float("nan"), "skill_edge_all": 0.2, "raw_edge_all": 0.2,
         "out_of_sample_skill_edge": float("nan"), "out_of_sample_skill_p": float("nan"),
         "edge_persisted": False, "reachability_60": 0.0,
         "copyable_skill_edge_60": float("nan"), "copyable_skill_edge_60_cents": float("nan"),
         "clears_magnitude_floor": False, "is_winner": False},
    ])


def test_category_funnel_counts():
    f = category_funnel(_table()).set_index("category")
    assert f.loc["politics", "cells"] == 2
    assert f.loc["politics", "persisted(A+C)"] == 2
    assert f.loc["politics", "winners(A+B+C)"] == 1
    assert f.loc["sports_nba", "winners(A+B+C)"] == 0


def test_default_view_is_winners_only_sorted():
    v = default_view(_table())
    assert len(v) == 1
    assert bool(v.iloc[0]["is_winner"]) is True


def test_renderers_run():
    t = _table()
    md = render_markdown(t, {})
    html_out = render_html(t, {})
    assert "Copyable Forecasters" in md and "winners" in md.lower()
    assert "<table" in html_out and "sortTable" in html_out
    # never advertises profit/returns as the sort key
    assert "never" in md.lower()
