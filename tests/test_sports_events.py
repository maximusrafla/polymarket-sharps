"""Tests for the sports resolution-event unit (src/sports_events.py).

Every fixture is a real slug shape taken from the ledger or the sports gate
corpus, with the correct event known by construction. The unit is load-bearing:
it is the cluster the significance bootstrap resamples and the unit the
concentration gates count, so both directions of error are pinned here —

  * OVER-COLLAPSE (the league-unit error): two different NBA games, or two
    different seasons of the same championship, must NOT share an event.
  * UNDER-COLLAPSE (the market-unit error that inflated the gate): every team
    outright in one championship, and every derivative line on one match, MUST
    share an event.
"""

import pandas as pd

from src.sports_events import (
    EVENT_COLUMNS,
    assign_events,
    event_summary,
    is_sports_market,
    resolution_event,
)


# ---------------------------------------------------------------------------
# Rule 1 — coded game slugs: the line collapses onto the match
# ---------------------------------------------------------------------------

def test_derivative_lines_collapse_to_one_game():
    """The 28 `fifwc-esp-arg-2026-07-19-*` markets are one match. This is the
    exact shape that let the gate count one game as 28 independent draws."""
    lines = [
        "fifwc-esp-arg-2026-07-19-spread-away-1pt5",
        "fifwc-esp-arg-2026-07-19-spread-home-2pt5",
        "fifwc-esp-arg-2026-07-19-total-3pt5",
        "fifwc-esp-arg-2026-07-19-btts",
        "fifwc-esp-arg-2026-07-19-team-to-advance",
        "fifwc-esp-arg-2026-07-19",
    ]
    keys = {resolution_event(s) for s in lines}
    assert keys == {("fifwc-esp-arg-2026-07-19", "game")}


def test_different_games_stay_distinct():
    """The league is NOT the unit: two matches are two events, which is the
    breadth that market-level clustering gets right and league-level destroys."""
    a, _ = resolution_event("nba-sas-nyk-2026-06-10-spread-home-1pt5")
    b, _ = resolution_event("nba-nyk-sas-2026-06-13")
    assert a != b
    assert a == "nba-sas-nyk-2026-06-10"
    assert b == "nba-nyk-sas-2026-06-13"


def test_motorsport_field_is_one_race():
    """`-winner-<driver>-` folds away: a field of drivers in one Grand Prix is
    one resolution event, not one per driver."""
    lec = resolution_event("f1-belgian-grand-prix-winner-leclerc-2026-07-19")
    ver = resolution_event("f1-belgian-grand-prix-winner-verstappen-2026-07-19")
    assert lec == ver == ("f1-belgian-grand-prix-2026-07-19", "game")


def test_trailing_capture_id_is_stripped():
    a = resolution_event("mlb-chc-cin-2026-07-11-total-8pt5-1743783173")
    b = resolution_event("mlb-chc-cin-2026-07-11")
    assert a == b == ("mlb-chc-cin-2026-07-11", "game")


# ---------------------------------------------------------------------------
# Rule 2 — prose outrights: the field collapses onto the championship
# ---------------------------------------------------------------------------

def test_championship_field_is_one_event():
    """54 NBA team outrights resolve on ONE Finals. Exactly one pays, so their
    residuals are mechanically anti-correlated — the opposite of independent."""
    teams = [
        "will-the-sacramento-kings-win-the-2025-nba-finals",
        "will-the-toronto-raptors-win-the-2025-nba-finals",
        "will-the-miami-heat-win-the-2025-nba-finals",
    ]
    keys = {resolution_event(s) for s in teams}
    assert keys == {("win-2025-nba-finals", "competition")}


def test_the_is_folded_so_phrasing_variants_agree():
    a, _ = resolution_event("will-the-new-york-yankees-win-the-2025-world-series")
    b, _ = resolution_event("will-boston-red-sox-win-2025-world-series")
    assert a == b == "win-2025-world-series"


def test_different_seasons_are_different_events():
    """Under-collapsing across seasons would merge two genuinely independent
    championships and silently halve the cluster count."""
    a, _ = resolution_event("will-the-titans-win-super-bowl-2025")
    b, _ = resolution_event("will-the-titans-win-super-bowl-2026")
    assert a != b


def test_season_props_collapse_by_predicate():
    scoring = {resolution_event(f"will-{p}-lead-the-nba-in-scoring")
               for p in ("nikola-jokic", "lebron-james", "another-player")}
    assists = resolution_event("will-trae-young-lead-the-nba-in-assists")
    assert scoring == {("lead-nba-in-scoring", "competition")}
    assert assists == ("lead-nba-in-assists", "competition")
    assert assists[0] not in {k for k, _ in scoring}


# ---------------------------------------------------------------------------
# Fallback — permissive, and labelled so the caller can report it
# ---------------------------------------------------------------------------

def test_unmatched_market_is_its_own_event_and_labelled():
    key, kind = resolution_event("some-unparseable-slug", None, "0xdeadbeef")
    assert kind == "market"
    assert key == "market:0xdeadbeef"


def test_empty_slug_falls_back_to_market():
    assert resolution_event(None, None, "0xabc") == ("market:0xabc", "market")


# ---------------------------------------------------------------------------
# Sports membership
# ---------------------------------------------------------------------------

def test_sports_membership_covers_coded_leagues_and_outrights():
    assert is_sports_market("atp-collign-sonego-2026-07-15")
    assert is_sports_market("cs2-mis-ent2-2026-07-18")
    assert is_sports_market("will-the-miami-heat-win-the-2025-nba-finals",
                            "Will the Miami Heat win the 2025 NBA Finals?",
                            "sports_nba")


def test_weather_temperature_is_not_sports():
    """The temperature generator carries a date-coded slug too. It must not leak
    into the sports universe on the strength of slug shape alone."""
    assert not is_sports_market("highest-temperature-in-miami-on-june-26-2026-92-93f",
                                "Highest temperature in Miami on June 26?")
    assert not is_sports_market("will-us-strike-iran-by-march-31",
                                "Will the US strike Iran by March 31?")


# ---------------------------------------------------------------------------
# Frame-level assignment
# ---------------------------------------------------------------------------

def test_assign_events_is_additive_and_preserves_rows():
    df = pd.DataFrame({
        "market_id": ["m1", "m2", "m3", "m4"],
        "slug": ["fifwc-esp-arg-2026-07-19-btts",
                 "fifwc-esp-arg-2026-07-19-total-3pt5",
                 "will-the-miami-heat-win-the-2025-nba-finals",
                 "unparseable"],
        "question": [None, None, None, None],
        "niche_l1": ["lg_fifwc", "lg_fifwc", "sports_nba", "other"],
    })
    out = assign_events(df)
    assert len(out) == len(df)
    assert list(out["market_id"]) == list(df["market_id"])
    for c in EVENT_COLUMNS:
        assert c in out.columns
    assert out.loc[0, "event"] == out.loc[1, "event"]      # one match
    assert out["event"].nunique() == 3                      # 4 markets, 3 events
    assert list(out["event_kind"]) == ["game", "game", "competition", "market"]


def test_assign_events_on_empty_frame():
    out = assign_events(pd.DataFrame(columns=["market_id", "slug", "question"]))
    assert out.empty
    for c in EVENT_COLUMNS:
        assert c in out.columns


def test_sub_form_extracts_the_side_and_nothing_else():
    """The finer level must add the SIDE (where home-favourite miscalibration
    would live) without keying on team codes or handicap numbers, which would
    push the baseline cell toward the market itself."""
    from src.sports_events import sub_form
    assert sub_form("nba-sas-nyk-2026-06-10-spread-home-1pt5") == "home"
    assert sub_form("mls-lag-laf-2026-07-17-spread-away-2pt5") == "away"
    assert sub_form("lal-bil-cel-2026-05-17-draw") == "draw"
    assert sub_form("mlb-chc-cin-2026-07-11-total-8pt5") == "na"
    assert sub_form("mex-que-ame-2026-07-18-ame") == "na"      # team code, not a side
    assert sub_form("will-the-miami-heat-win-the-2025-nba-finals") == "na"


def test_assign_sub_forms_builds_niche_l3_under_niche_l2():
    from src.sports_events import assign_sub_forms
    df = pd.DataFrame({
        "slug": ["nba-sas-nyk-2026-06-10-spread-home-1pt5",
                 "nba-sas-nyk-2026-06-10-spread-away-1pt5",
                 "mlb-chc-cin-2026-07-11-total-8pt5"],
        "niche_l2": ["lg_nba|spread", "lg_nba|spread", "lg_mlb|total"],
    })
    out = assign_sub_forms(df)
    assert list(out["niche_l3"]) == ["lg_nba|spread|home", "lg_nba|spread|away",
                                     "lg_mlb|total|na"]
    # strictly finer: it never merges two niche_l2 cells
    assert out.groupby("niche_l3")["niche_l2"].nunique().max() == 1


def test_event_summary_counts_markets_per_event():
    df = pd.DataFrame({
        "market_id": ["m1", "m2", "m3"],
        "slug": ["nba-sas-nyk-2026-06-10-total-217pt5",
                 "nba-sas-nyk-2026-06-10-spread-home-1pt5",
                 "nba-nyk-sas-2026-06-13"],
        "question": [None, None, None],
    })
    tbl = event_summary(assign_events(df))
    assert list(tbl["n_markets"]) == [2, 1]
    assert tbl.loc[0, "kind"] == "game"
