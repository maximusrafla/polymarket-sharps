"""Tests for the slow-universe niche partition (src/slow_niche.py).

Every case is a hand-built fixture whose correct label is known by construction
from real slug/question shapes observed in the deep slow tape. The partition is
load-bearing for the finer-baseline re-check, so its behaviour is pinned here —
in particular the ORDERING guarantees (a Hormuz ship-transit market must not be
swallowed by the geo_iran complex, a crude-oil market must not be swallowed by
`geo_other`'s "war"/"ceasefire" keywords).
"""

import pandas as pd

from src.slow_niche import (
    NICHE_COLUMNS,
    assign_niches,
    bet_form,
    market_family,
    niche_summary,
)


# ---------------------------------------------------------------------------
# bet_form — coded sports slugs carry the form in the tail after the date
# ---------------------------------------------------------------------------

def test_bet_form_coded_slug_tails():
    cases = {
        "bra2-rec-ope-2026-07-18-team-total-away-2pt5": "team_total",
        "lol-hle1-ns-2026-04-22-total-games-2pt5": "total_games",
        "cbb-ark-ala-2026-02-18-total-182pt5": "total",
        "bun-pau-koe-2026-04-17-spread-away-1pt5": "spread",
        "chi-tie-ton-2026-07-04-exact-score-3-2": "exact_score",
        "fifwc-arg-egy-2026-07-07-team-to-advance": "advance",
        "sud-cue-san-2026-04-08-draw": "draw",
        "val-oas-boom-2026-06-11-game2": "map_game",
        "lol-edg-nip-2025-12-19": "match_line",
        "ere-gro-nec-2026-05-10-gro": "moneyline",
    }
    for slug, expected in cases.items():
        assert bet_form(slug, "") == expected, slug


def test_bet_form_prose_shapes():
    assert bet_form("", "US strikes Iran by February 28, 2026?") == "deadline"
    assert bet_form("", "Will the US next strike Iran on March 2, 2026 (ET)?") == "on_date"
    assert bet_form("", "Will Luis Antonio Tagle be the next pope?") == "outright"
    assert bet_form("", "Will Ludvig Aberg win the 2025 FedEx Cup?") == "outright"
    assert bet_form("", "Will Crude Oil (CL) hit (HIGH) $105 by end of March?") == "threshold"
    assert bet_form("", "Will the government shutdown last 14 days or more?") == "duration"
    assert bet_form("highest-temperature-in-seoul-on-july-21-2026-27c",
                    "Will the highest temperature in Seoul be 27°C on July 21?") == "threshold"
    assert bet_form("", "Will 20 ships transit the Strait of Hormuz on any day in March?") == "threshold"


def test_bet_form_epoch_suffix_is_stripped():
    # Some slugs carry a trailing capture id; it must not defeat the date match.
    assert bet_form("luiz-henrique-20260609165807842",
                    "Will Luiz Henrique be in Brazil's Starting 11?") == "misc"


def test_bet_form_unmatched_is_misc_never_raises():
    assert bet_form(None, None) == "misc"
    assert bet_form("", "") == "misc"


# ---------------------------------------------------------------------------
# market_family — generators, then event complexes, in that order
# ---------------------------------------------------------------------------

def test_family_temperature_generator():
    assert market_family("highest-temperature-in-moscow-on-april-12-2026-13c",
                         "Will the highest temperature in Moscow be 13°C on April 12?") == "weather_temp"
    assert market_family("lowest-temperature-in-seoul-on-july-21-2026-21c",
                         "Will the lowest temperature in Seoul be 21°C on July 21?") == "weather_temp"


def test_family_coded_league_slug():
    assert market_family("fifwc-tun-jpn-2026-06-21-draw",
                         "Will Tunisia vs. Japan end in a draw?") == "lg_fifwc"
    assert market_family("cbb-cita-chat-2026-02-07-total-140pt5",
                         "The Citadel Bulldogs vs. Chattanooga Mocs: O/U 140.5") == "lg_cbb"


def test_family_ordering_hormuz_shipping_beats_geo_iran():
    """The ship-transit COUNT markets are a mechanical generator with their own
    price structure; they must not be pooled into the Iran event complex even
    though their text contains 'Strait of Hormuz'."""
    q = "Will 40-44 ships transit the Strait of Hormuz between March 10-16?"
    assert market_family("", q) == "hormuz_shipping"


def test_family_ordering_crude_beats_geo_keywords():
    q = "Will Crude Oil (CL) hit (HIGH) $140 by end of June?"
    assert market_family("", q) == "commodity_crude"


def test_family_iran_complex_is_one_family():
    """The rolling-deadline Iran ladder is ONE complex, however the slug is
    tokenised — this is exactly what the coarse partition failed to capture."""
    for q in (
        "US strikes Iran by February 28, 2026?",
        "US x Iran permanent peace deal by June 15, 2026?",
        "Will the Iranian regime fall by June 30?",
        "Khamenei out as Supreme Leader of Iran by February 28?",
        "Will Iran close the Strait of Hormuz by March 31?",
        "Strait of Hormuz traffic returns to normal by end of June?",
        "Fordow nuclear facility destroyed before July?",
        "Kharg Island no longer under Iranian control by April 30?",
    ):
        assert market_family("", q) == "geo_iran", q


def test_family_distinguishes_neighbouring_complexes():
    assert market_family("", "Israel x Hezbollah ceasefire by April 30, 2026?") == "geo_israel"
    assert market_family("", "Russia x Ukraine ceasefire by May 31, 2026?") == "geo_ukraine"
    assert market_family("", "US x Venezuela military engagement by December 31?") == "geo_venezuela"
    assert market_family("", "Will Luis Antonio Tagle be the next pope?") == "conclave"
    assert market_family("", "Will the government shutdown last 6 days or more?") == "us_shutdown"
    assert market_family("", "Lighter Airdrop on December 18?") == "crypto_airdrop"


def test_family_falls_back_to_category_then_misc():
    # A non-`other` category with no finer rule keeps its category as its family.
    assert market_family("some-slug", "An unmatched question", "sports_nba") == "sports_nba"
    # An `other` market with no rule at all pools to misc (never to noise).
    assert market_family("zzz", "An unmatched question", "other") == "misc"
    assert market_family(None, None, None) == "misc"


# ---------------------------------------------------------------------------
# assign_niches — frame level, and the no-drop / no-null invariant
# ---------------------------------------------------------------------------

def _fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "market_id": ["m1", "m2", "m3", "m4", "m5"],
        "wallet": ["w1", "w1", "w2", "w2", "w3"],
        "slug": ["", "", "fifwc-tun-jpn-2026-06-21-draw",
                 "highest-temperature-in-moscow-on-april-12-2026-13c", "zzz"],
        "question": ["US strikes Iran by February 28, 2026?",
                     "Will Crude Oil (CL) hit (HIGH) $105 by end of March?",
                     "Will Tunisia vs. Japan end in a draw?",
                     "Will the highest temperature in Moscow be 13°C on April 12?",
                     "An unmatched question"],
        "category": ["other", "other", "other", "other", "other"],
    })


def test_assign_niches_labels_and_levels():
    out = assign_niches(_fixture())
    for c in NICHE_COLUMNS:
        assert c in out.columns
    assert list(out["niche_family"]) == [
        "geo_iran", "commodity_crude", "lg_fifwc", "weather_temp", "misc"]
    assert list(out["niche_form"]) == [
        "deadline", "threshold", "draw", "threshold", "misc"]
    assert out.loc[0, "niche_l1"] == "geo_iran"
    assert out.loc[0, "niche_l2"] == "geo_iran|deadline"


def test_assign_niches_is_additive_and_total():
    """No row is dropped and no label is null — the no-drop invariant."""
    src = _fixture()
    out = assign_niches(src)
    assert len(out) == len(src)
    assert list(out["market_id"]) == list(src["market_id"])
    for c in NICHE_COLUMNS:
        assert out[c].notna().all()
    # original columns survive untouched
    for c in src.columns:
        assert list(out[c]) == list(src[c])


def test_assign_niches_empty_frame():
    out = assign_niches(pd.DataFrame(columns=["slug", "question", "category"]))
    assert len(out) == 0
    for c in NICHE_COLUMNS:
        assert c in out.columns


def test_assign_niches_handles_missing_category_column():
    src = _fixture().drop(columns=["category"])
    out = assign_niches(src)
    assert out["niche_family"].notna().all()


def test_assign_niches_dedups_by_market_text():
    """Repeating the same market many times must not change its label — the
    per-unique-triple computation is a performance optimisation only."""
    src = pd.concat([_fixture()] * 20, ignore_index=True)
    out = assign_niches(src)
    assert out["niche_l2"].nunique() == 5
    assert (out.groupby("market_id")["niche_l2"].nunique() == 1).all()


def test_niche_summary_counts():
    out = assign_niches(_fixture())
    tbl = niche_summary(out, level="niche_l1")
    assert set(tbl["niche_l1"]) == {"geo_iran", "commodity_crude", "lg_fifwc",
                                    "weather_temp", "misc"}
    assert tbl["n_bets"].sum() == 5
    assert abs(tbl["share"].sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Narrative layer (step 7b) — one level above the event complex
# ---------------------------------------------------------------------------

from src.slow_niche import (  # noqa: E402
    NARRATIVE_MAP_VERSION,
    NICHE_PARTITION_VERSION,
    assign_narratives,
    narrative,
)


def test_versions_are_declared():
    """The freeze manifest records these, so a frozen cohort stays reproducible
    against the exact partition that produced it."""
    assert NICHE_PARTITION_VERSION
    assert NARRATIVE_MAP_VERSION


def test_correlated_complexes_collapse_to_one_narrative():
    """The whole point: three DISTINCT event complexes that are one bet."""
    for fam in ("geo_iran", "geo_israel", "hormuz_shipping"):
        assert narrative(fam) == "mideast_escalation", fam


def test_unrelated_complexes_stay_separate():
    assert narrative("geo_ukraine") == "ukraine_war"
    assert narrative("us_politics") == "us_domestic"
    assert narrative("macro_rates") == "macro"
    assert narrative("conclave") == "culture"


def test_crude_strict_vs_broad_is_the_documented_judgement_call():
    """Crude spikes on Hormuz risk but also trades on OPEC/demand. Reported both
    ways rather than decided by fiat — the sensitivity is part of the finding."""
    assert narrative("commodity_crude", "strict") == "energy_commodities"
    assert narrative("commodity_crude", "broad") == "mideast_escalation"
    # ...and the choice must not leak into anything else
    assert narrative("geo_iran", "strict") == narrative("geo_iran", "broad")
    assert narrative("macro_rates", "strict") == narrative("macro_rates", "broad")


def test_sports_leagues_collapse_and_unmapped_is_a_real_bucket():
    assert narrative("lg_fifwc") == "sports"
    assert narrative("lg_nba") == "sports"
    assert narrative("sports_soccer") == "sports"
    assert narrative("misc") == "other"          # a bucket, not a drop
    assert narrative(None) == "other"


def test_assign_narratives_is_additive_and_total():
    src = pd.DataFrame({
        "wallet": ["w1", "w1", "w2"],
        "niche_l1": ["geo_iran", "commodity_crude", "lg_nba"],
    })
    out = assign_narratives(src)
    assert len(out) == len(src)
    assert list(out["narrative_strict"]) == [
        "mideast_escalation", "energy_commodities", "sports"]
    assert list(out["narrative_broad"]) == [
        "mideast_escalation", "mideast_escalation", "sports"]
    for c in src.columns:
        assert list(out[c]) == list(src[c])
    assert out[["narrative_strict", "narrative_broad"]].notna().all().all()


def test_assign_narratives_empty_frame():
    out = assign_narratives(pd.DataFrame(columns=["niche_l1"]))
    assert len(out) == 0
    assert "narrative_strict" in out.columns
