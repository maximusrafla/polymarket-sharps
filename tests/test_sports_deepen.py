"""Tests for the sports shortlist screen (src/sports_deepen.py).

Hand-built fixtures where the correct arm membership is known by construction.
The screen is a SELECTION device behind a disjoint-data firewall, so what these
pin is not "the right wallets" but the property that makes the three arms worth
running: the bet-weighted score can be manufactured by one event, and the
event-weighted score cannot.

The fetch/fold/resolve half of the module is the tested slow_deepen/ingest
machinery with different paths, and is exercised there.
"""

import numpy as np
import pandas as pd

from src.sports_deepen import (
    BREADTH_MIN_EVENTS,
    CORE_GATE,
    DEFAULT_GATE,
    MIN_SCREEN_BETS,
    screen_wallets,
    select_shortlist,
)


def _bets(wallet, resid_by_event):
    """One wallet's bets: {event: [residuals]} -> long frame."""
    rows = []
    t = 1_700_000_000
    for ev, resids in resid_by_event.items():
        for i, r in enumerate(resids):
            rows.append({"wallet": wallet, "event": ev, "market_id": f"{ev}-m{i % 3}",
                         "residual_skill": r, "timestamp": t})
            t += 3600
    return pd.DataFrame(rows)


def _corpus():
    return pd.concat([
        # one event, big edge -> bet-weighted score is high; this is the shape the
        # gate corpus is full of (one tournament, many team markets)
        _bets("w_core", {"win-2025-nba-finals": [0.5] * 40}),
        # one event, moderate edge -> clears 2c but not 10c
        _bets("w_default", {"win-2025-world-series": [0.06] * 30}),
        # five events, small positive edge in each -> low bet-weighted score,
        # but it cannot be one lucky tournament
        _bets("w_breadth", {f"game-{i}": [0.012] * 12 for i in range(5)}),
        # too thin to be findable at all
        _bets("w_thin", {"win-2025-stanley-cup": [0.9] * 10}),
        # deep and negative
        _bets("w_neg", {"win-2026-fifa-world-cup": [-0.2] * 40}),
    ], ignore_index=True)


def test_thin_wallets_are_not_findable():
    scores = screen_wallets(_corpus())
    assert "w_thin" not in set(scores["wallet"])
    assert (scores["n_bets"] >= MIN_SCREEN_BETS).all()


def test_arms_select_the_wallets_they_are_meant_to():
    short = select_shortlist(screen_wallets(_corpus()))
    by = short.set_index("wallet")

    assert bool(by.loc["w_core", "arm_core"])
    assert bool(by.loc["w_core", "arm_default"])          # 10c implies 2c

    assert "w_default" in by.index
    assert not bool(by.loc["w_default", "arm_core"])
    assert bool(by.loc["w_default", "arm_default"])

    # the breadth wallet's bet-weighted score is below the 2c gate, so the gate's
    # own arms would never have deepened it
    assert by.loc["w_breadth", "shrunk_skill"] < DEFAULT_GATE
    assert bool(by.loc["w_breadth", "arm_breadth"])

    assert "w_neg" not in by.index


def test_event_weighted_score_is_not_inflated_by_one_event():
    """The crux of the breadth arm: piling more markets into ONE event raises the
    bet-weighted score but leaves the event-weighted score where it was."""
    base = _bets("w", {"one-event": [0.3] * 25})
    piled = _bets("w", {"one-event": [0.3] * 200})
    s_base = screen_wallets(base).iloc[0]
    s_piled = screen_wallets(piled).iloc[0]

    assert s_piled["shrunk_skill"] > s_base["shrunk_skill"] * 1.5
    assert np.isclose(s_piled["shrunk_event_skill"], s_base["shrunk_event_skill"])


def test_breadth_arm_requires_multiple_events():
    """A single-event wallet can never enter the breadth arm no matter how large
    its edge — that is the whole point of the arm."""
    corpus = _bets("w_huge_one_event", {"one-event": [0.9] * 500})
    short = select_shortlist(screen_wallets(corpus))
    row = short.set_index("wallet").loc["w_huge_one_event"]
    assert row["n_events"] < BREADTH_MIN_EVENTS
    assert not bool(row["arm_breadth"])
    assert bool(row["arm_core"])        # still deepened, via the core arm


def test_top_event_share_is_reported():
    short = select_shortlist(screen_wallets(_corpus())).set_index("wallet")
    assert short.loc["w_core", "top_event_share"] == 1.0
    assert np.isclose(short.loc["w_breadth", "top_event_share"], 0.2)


def test_core_arm_is_deepened_first():
    """A truncated run must still cover the most-vetted wallets."""
    short = select_shortlist(screen_wallets(_corpus()))
    assert short.iloc[0]["wallet"] == "w_core"
    assert list(short["priority"]) == sorted(short["priority"])


def test_gate_constants_are_the_ones_documented():
    assert (CORE_GATE, DEFAULT_GATE) == (0.10, 0.02)
