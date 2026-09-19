"""Tests for the REAL-WORLD forward paper arm (src/paper_rw.py).

Four things are pinned here, in order of how badly they would hurt if wrong:

1. ISOLATION. The 38-wallet paper trader has been live since 2026-07-26 and is a
   pre-registered forward experiment. If this arm writes ANY of its files the
   record is destroyed. Path disjointness is asserted, and a full poll+score cycle
   is run with the live arm's own path constants pointed at an empty temp dir,
   which is then asserted to still be empty.
2. THE UNIVERSE. Micro-crypto signals must be SKIPPED, counted, and excluded from
   every dollar figure — and an unclassifiable market must be skipped too rather
   than assumed real-world, which is the failure mode that would quietly re-admit
   the 5-minute tape into an explicitly real-world-only experiment.
3. THE COSTS. Per-category fees, taker-only, live `feeSchedule.rate` taking
   precedence over the frozen map, and a missing k never becoming a free trade.
4. THE PRE-REGISTRATION. The freeze refuses to overwrite itself, an amendment is
   logged with the number of forward observations that existed at the time, the
   per-wallet strata exist before any data does, and a position fills at most once.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.paper_rw as rw
import src.paper_trader as pt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def raw_book(asks=(), bids=(), tick="0.01", min_size="5"):
    return {
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "tick_size": tick, "min_order_size": min_size, "timestamp": "1700000000000",
    }


def rule(**over):
    base = {
        "notional_per_signal_usd": 100.0, "max_notional_per_position_usd": 100.0,
        "max_open_positions": 200, "taker_fee_k": 0.0, "taker_fee_k_stress": 0.07,
        "placebo_seed": "rw-seed-v1", "placebo_fallback_horizon_hours": 48.0,
        "enforce_min_order_size": True,
    }
    base.update(over)
    return base


def watchlist(labels=("0xaaaa",)):
    return pd.DataFrame([{"wallet": f"{lab}0000", "wallet_label": lab}
                         for lab in labels])


def trade(**over):
    base = {
        "wallet": "0xaaaa0000", "market_id": "0xm1", "token_id": "tok1",
        "outcome": "Yes", "side": "BUY", "entry_price": 0.60, "size": 10.0,
        "timestamp": 900, "question": "Will the Senate pass the bill?",
        "slug": "senate-passes-bill-2026", "tx_hash": "0xtx1",
    }
    base.update(over)
    return base


def fetchers(trades, book=None, gamma=None, meta=None):
    """Injectable stand-ins for the four network calls. No network, ever."""
    book = raw_book(asks=[(0.60, 500)]) if book is None else book
    gamma = {} if gamma is None else gamma
    meta = {"market_end_ts": 10_000_000, "game_start_ts": None} if meta is None else meta
    return {
        "fetch_positions": lambda s, c, w, cur: list(trades),
        "get_book": lambda s, c, tok: book,
        "get_market": lambda s, c, m: dict(meta),
        "get_gamma": lambda s, m: dict(gamma),
    }


def poll(trades, *, signals=None, orders=None, state=None, now=1000, r=None, wl=None,
         **fk):
    signals = pt._empty(rw.SIGNAL_COLUMNS) if signals is None else signals
    orders = pt._empty(rw.ORDER_COLUMNS) if orders is None else orders
    state = {"cursors": {"0xaaaa0000": {"max_timestamp": 0, "keys_at_max": []}},
             "polls": 0} if state is None else state
    return rw.poll_once(None, {}, r or rule(), wl if wl is not None else watchlist(),
                        signals, orders, state, now, **fetchers(trades, **fk))


# ---------------------------------------------------------------------------
# 1. ISOLATION FROM THE LIVE ARM  (the thing that must never break)
# ---------------------------------------------------------------------------

def test_every_artifact_path_is_disjoint_from_the_live_paper_trader():
    """The running 38-wallet experiment's files are its record. Sharing even one
    path would let this arm overwrite it."""
    mine = {rw.RW_DIR, rw.STATE_PATH, rw.SIGNALS_PATH, rw.ORDERS_PATH,
            rw.RESOLUTIONS_PATH, rw.FREEZE_MANIFEST_PATH, rw.WATCHLIST_PATH,
            rw.SCOREBOARD_MD, rw.SCOREBOARD_PARQUET}
    theirs = {pt.PAPER_DIR, pt.STATE_PATH, pt.SIGNALS_PATH, pt.ORDERS_PATH,
              pt.RESOLUTIONS_PATH, pt.FREEZE_MANIFEST_PATH, pt.WATCHLIST_PATH,
              pt.SCOREBOARD_MD, pt.SCOREBOARD_PARQUET}
    assert mine.isdisjoint(theirs)
    # ...and nothing of mine sits INSIDE their ledger directory either
    assert not any(pt.PAPER_DIR in p.parents for p in mine)


def test_the_live_arms_artifacts_are_never_touched_by_a_full_cycle(monkeypatch, tmp_path):
    """Point the LIVE arm's path constants at an empty directory, run this arm end
    to end, and assert the directory is still empty. This catches a stray write
    that path-comparison alone would miss."""
    live = tmp_path / "live"
    live.mkdir()
    for name in ("PAPER_DIR",):
        monkeypatch.setattr(pt, name, live / "paper")
    for name, fn in (("STATE_PATH", "paper_state.json"),
                     ("SIGNALS_PATH", "paper_signals.parquet"),
                     ("ORDERS_PATH", "paper_orders.parquet"),
                     ("RESOLUTIONS_PATH", "paper_resolutions.parquet"),
                     ("FREEZE_MANIFEST_PATH", "paper_freeze_manifest.json"),
                     ("WATCHLIST_PATH", "paper_watchlist.parquet"),
                     ("SCOREBOARD_MD", "paper_scoreboard.md"),
                     ("SCOREBOARD_PARQUET", "paper_scoreboard.parquet")):
        monkeypatch.setattr(pt, name, live / fn)

    mine = tmp_path / "rw"
    mine.mkdir()
    monkeypatch.setattr(rw, "RW_DIR", mine)
    monkeypatch.setattr(rw, "SIGNALS_PATH", mine / "rw_signals.parquet")
    monkeypatch.setattr(rw, "ORDERS_PATH", mine / "rw_orders.parquet")
    monkeypatch.setattr(rw, "STATE_PATH", mine / "rw_state.json")
    monkeypatch.setattr(rw, "SCOREBOARD_MD", mine / "rw_scoreboard.md")
    monkeypatch.setattr(rw, "SCOREBOARD_PARQUET", mine / "rw_scoreboard.parquet")

    signals, orders, state, _ = poll([trade()])
    rw.save_ledger(signals, orders, state)
    out = rw.score_orders(orders, signals, pd.DataFrame(
        columns=["market_id", "token_id", "resolved", "resolved_value"]), rule())
    rw.write_scoreboard(out, _manifest_stub(), watchlist(), signals, orders)

    assert list(live.iterdir()) == [], f"the live arm's directory was written: {list(live.iterdir())}"
    assert (mine / "rw_signals.parquet").exists()


def test_module_never_names_the_live_arms_artifact_files():
    """A source scan: this module may import the live arm's FUNCTIONS but must
    never name its WRITE TARGETS. (`paper_scoreboard.md` appears once, in prose on
    this arm's own scoreboard, pointing a reader at the other experiment.)"""
    src = open(rw.__file__).read()
    for forbidden in ("paper_state.json", "paper_signals.parquet",
                      "paper_orders.parquet", "paper_resolutions.parquet",
                      "paper_freeze_manifest.json", "paper_watchlist.parquet",
                      "paper_scoreboard.parquet"):
        assert forbidden not in src, forbidden
    for p in (rw.FREEZE_MANIFEST_PATH, rw.WATCHLIST_PATH, rw.SCOREBOARD_MD,
              rw.SCOREBOARD_PARQUET):
        assert p.name.startswith("paper_rw_"), p


# ---------------------------------------------------------------------------
# 2. THE REAL-WORLD UNIVERSE
# ---------------------------------------------------------------------------

def test_micro_crypto_signals_are_skipped_counted_and_open_no_orders():
    """The whole arm is scoped to slow / real-world markets. A 5-minute up/down
    coin flip must be recorded (nothing is ever dropped, per CLAUDE.md) and must
    open nothing."""
    signals, orders, _, stats = poll([trade(slug="btc-updown-5m-1738000000",
                                            question="Bitcoin Up or Down?")])
    assert stats["skipped_micro_crypto"] == 1
    assert stats["signals_traded"] == 0
    assert len(signals) == 1 and signals.iloc[0]["status"] == rw.SKIP_MICRO
    assert signals.iloc[0]["category"] == "micro_crypto"
    assert bool(signals.iloc[0]["is_real_world"]) is False
    assert len(orders) == 0


def test_a_real_world_signal_opens_all_three_arms():
    signals, orders, _, stats = poll([trade()])
    assert stats["signals_traded"] == 1
    assert set(orders["arm"]) == set(rw.ARMS)
    assert signals.iloc[0]["category"] == "politics"
    assert bool(signals.iloc[0]["is_real_world"]) is True


def test_an_unclassifiable_market_is_skipped_not_assumed_real_world():
    """Assuming 'unknown means real-world' is how micro-crypto would leak back in.
    Gamma is asked first (a missing slug is a data gap, not evidence); if Gamma has
    nothing either, the signal is skipped."""
    signals, orders, _, stats = poll([trade(slug=None, question=None)])
    assert stats["skipped_unclassified"] == 1
    assert signals.iloc[0]["status"] == rw.SKIP_UNCLASSIFIED
    assert len(orders) == 0


def test_gamma_rescues_a_slug_missing_from_the_tape():
    signals, orders, _, stats = poll(
        [trade(slug=None, question=None)],
        gamma={"gamma_slug": "nba-bos-nyk-2026-01-02", "gamma_question": "Celtics?"})
    assert stats["signals_traded"] == 1
    assert signals.iloc[0]["category"].startswith("sports")
    assert signals.iloc[0]["slug"] == "nba-bos-nyk-2026-01-02"


def test_the_real_world_definition_is_the_repos_own_not_a_local_copy():
    from src.discover import classify_market, is_real_world
    for slug, q in (("btc-updown-5m-1", "Up or Down?"),
                    ("senate-passes-bill-2026", "Will the Senate pass?"),
                    ("nba-bos-nyk", "Celtics win?")):
        cat, tradeable = rw.classify_signal(slug, q)
        assert cat == classify_market(slug, q)
        assert tradeable == is_real_world(cat)


# ---------------------------------------------------------------------------
# 3. THE FEES
# ---------------------------------------------------------------------------

def test_category_fee_map_matches_the_verified_live_rates():
    """docs/polymarket_mechanics.md's live Gamma sweep: politics 0.04, sports 0.05,
    crypto 0.07, geopolitics 0."""
    assert rw.fee_k_for_category("politics") == 0.04
    assert rw.fee_k_for_category("sports_nba") == 0.05
    assert rw.fee_k_for_category("sports_soccer") == 0.05
    assert rw.fee_k_for_category("crypto_event") == 0.07
    assert rw.fee_k_for_category("geopolitics") == 0.0
    assert rw.fee_k_for_category("econ_macro") == 0.05
    assert rw.fee_k_for_category(None) == rw.FEE_K_DEFAULT
    assert rw.fee_k_for_category(float("nan")) == rw.FEE_K_DEFAULT


def test_live_fee_schedule_beats_the_frozen_category_map():
    k, src = rw.resolve_fee_k(
        {"fees_enabled": True, "fee_rate_live": 0.07, "fee_exponent": 1.0},
        "politics")
    assert (k, src) == (0.07, "gamma_feeSchedule")


def test_fees_disabled_on_gamma_means_zero_not_the_fallback():
    k, src = rw.resolve_fee_k({"fees_enabled": False}, "politics")
    assert (k, src) == (0.0, "gamma_fees_disabled")


def test_an_unexpected_fee_exponent_falls_back_instead_of_misapplying_the_formula():
    """The frozen formula is the exponent-1 form. A market advertising a different
    exponent is not covered by it, so its live rate must not be used."""
    k, src = rw.resolve_fee_k(
        {"fees_enabled": True, "fee_rate_live": 0.09, "fee_exponent": 2.0}, "politics")
    assert (k, src) == (0.04, "category_map")


def test_no_gamma_at_all_falls_back_to_the_category_map():
    assert rw.resolve_fee_k({}, "sports_nfl") == (0.05, "category_map")
    assert rw.resolve_fee_k(None, "crypto_event") == (0.07, "category_map")


def test_fee_is_charged_per_category_to_takers_and_never_to_makers():
    """Two identical taker fills in different categories pay different fees, and a
    maker fill pays none."""
    orders = pd.DataFrame([
        {"order_id": "a", "signal_id": "s1", "arm": "market_chase", "wallet": "0xw",
         "wallet_label": "0xw", "market_id": "m1", "token_id": "t1",
         "category": "politics", "fee_k": 0.04, "shares": 100.0, "cost_usd": 50.0,
         "fee_units": 25.0, "is_taker": True, "status": pt.ST_FILLED,
         "no_fill_reason": None, "detect_lag_s": 60, "fill_lag_s": 60,
         "slug": "s", "question": "q", "wallet_ts": 0, "detect_ts": 60,
         "first_fill_ts": 60, "avg_fill_price": 0.5, "wallet_entry_price": 0.45},
        {"order_id": "b", "signal_id": "s2", "arm": "market_chase", "wallet": "0xw",
         "wallet_label": "0xw", "market_id": "m2", "token_id": "t2",
         "category": "crypto_event", "fee_k": 0.07, "shares": 100.0, "cost_usd": 50.0,
         "fee_units": 25.0, "is_taker": True, "status": pt.ST_FILLED,
         "no_fill_reason": None, "detect_lag_s": 60, "fill_lag_s": 60,
         "slug": "s", "question": "q", "wallet_ts": 0, "detect_ts": 60,
         "first_fill_ts": 60, "avg_fill_price": 0.5, "wallet_entry_price": 0.45},
        {"order_id": "c", "signal_id": "s3", "arm": "limit_noChase", "wallet": "0xw",
         "wallet_label": "0xw", "market_id": "m3", "token_id": "t3",
         "category": "politics", "fee_k": 0.04, "shares": 100.0, "cost_usd": 50.0,
         "fee_units": 25.0, "is_taker": False, "status": pt.ST_FILLED,
         "no_fill_reason": None, "detect_lag_s": 60, "fill_lag_s": 600,
         "slug": "s", "question": "q", "wallet_ts": 0, "detect_ts": 60,
         "first_fill_ts": 600, "avg_fill_price": 0.5, "wallet_entry_price": 0.45},
    ])
    res = pd.DataFrame([{"market_id": f"m{i}", "token_id": f"t{i}", "resolved": True,
                         "resolved_value": 1.0} for i in (1, 2, 3)])
    signals = pd.DataFrame([{"signal_id": f"s{i}", "slug": "s", "question": "q",
                             "market_id": f"m{i}", "status": "accepted"}
                            for i in (1, 2, 3)])
    out = rw.score_orders(orders, signals, res, rule(), n_boot=50)
    scored = out["scored"].set_index("order_id")
    assert scored.loc["a", "fee_usd"] == pytest.approx(0.04 * 25.0)
    assert scored.loc["b", "fee_usd"] == pytest.approx(0.07 * 25.0)
    assert scored.loc["c", "fee_usd"] == 0.0
    # gross is identical for all three; only net differs
    assert set(np.round(scored["pnl_usd_gross"], 6)) == {50.0}
    assert scored.loc["a", "pnl_usd_net"] > scored.loc["b", "pnl_usd_net"]
    assert scored.loc["c", "pnl_usd_net"] == pytest.approx(50.0)


def test_a_missing_k_falls_back_and_never_becomes_a_free_trade():
    orders = pd.DataFrame([
        {"order_id": "a", "signal_id": "s1", "arm": "market_chase", "wallet": "0xw",
         "wallet_label": "0xw", "market_id": "m1", "token_id": "t1",
         "category": "sports_nba", "fee_k": None, "shares": 100.0, "cost_usd": 50.0,
         "fee_units": 25.0, "is_taker": True, "status": pt.ST_FILLED,
         "no_fill_reason": None, "detect_lag_s": 60, "fill_lag_s": 60,
         "slug": "s", "question": "q", "wallet_ts": 0, "detect_ts": 60,
         "first_fill_ts": 60, "avg_fill_price": 0.5, "wallet_entry_price": 0.45}])
    res = pd.DataFrame([{"market_id": "m1", "token_id": "t1", "resolved": True,
                         "resolved_value": 1.0}])
    signals = pd.DataFrame([{"signal_id": "s1", "slug": "s", "question": "q",
                             "market_id": "m1", "status": "accepted"}])
    out = rw.score_orders(orders, signals, res, rule(), n_boot=50)
    assert out["scored"].iloc[0]["fee_usd"] == pytest.approx(0.05 * 25.0)


def test_the_frozen_fee_is_the_verified_shape_not_a_flat_percentage():
    """`shares * k * p * (1-p)` vanishes at both ends and peaks mid-book. Reused
    verbatim from the live arm so the two cannot drift apart."""
    assert pt.taker_fee(100, 0.5, 0.04) == pytest.approx(1.0)
    assert pt.taker_fee(100, 0.99, 0.04) == pytest.approx(0.0396)
    assert pt.taker_fee(100, 0.01, 0.04) == pytest.approx(0.0396)


# ---------------------------------------------------------------------------
# 4. COPY LAG
# ---------------------------------------------------------------------------

def test_copy_lag_is_recorded_per_order_from_their_trade_not_from_detection():
    signals, orders, _, _ = poll([trade(timestamp=900)], now=1000)
    assert (orders["detect_lag_s"] == 100).all()
    chase = orders[orders["arm"] == "market_chase"].iloc[0]
    assert chase["first_fill_ts"] == 1000
    assert chase["fill_lag_s"] == 100


def test_a_later_maker_fill_records_the_LATER_lag():
    """A resting limit that fills two hours after their trade is not the trade the
    wallet made, and the ledger has to say so."""
    signals, orders, state, _ = poll(
        [trade(entry_price=0.55)], now=1000, book=raw_book(asks=[(0.60, 500)]))
    limit = orders[orders["arm"] == "limit_noChase"].iloc[0]
    assert limit["status"] == pt.ST_OPEN and pd.isna(limit["fill_lag_s"])
    _, orders2, _, _ = poll([], signals=signals, orders=orders, state=state,
                            now=1000 + 7200, book=raw_book(asks=[(0.50, 500)]))
    limit2 = orders2[orders2["arm"] == "limit_noChase"].iloc[0]
    assert limit2["shares"] > 0
    assert limit2["fill_lag_s"] == pytest.approx(7300)   # 1000+7200-900


def test_copy_lag_columns_are_idempotent():
    orders = pd.DataFrame([{"wallet_ts": 100, "detect_ts": 160, "first_fill_ts": 260}])
    once = rw.copy_lag_columns(orders)
    twice = rw.copy_lag_columns(once)
    assert once["detect_lag_s"].iloc[0] == 60 and once["fill_lag_s"].iloc[0] == 160
    pd.testing.assert_frame_equal(once, twice)


def test_the_scoreboard_reports_the_achieved_lag_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(rw, "SCOREBOARD_MD", tmp_path / "sb.md")
    monkeypatch.setattr(rw, "SCOREBOARD_PARQUET", tmp_path / "sb.parquet")
    out = _scored_out()
    rw.write_scoreboard(out, _manifest_stub(), watchlist(), out["_signals"],
                        out["_orders"])
    text = (tmp_path / "sb.md").read_text()
    assert "Achieved copy lag (measured, not assumed)" in text
    assert "fill lag median" in text


# ---------------------------------------------------------------------------
# 5. FILL-ONCE / DEDUPE / SEATBELTS
# ---------------------------------------------------------------------------

def test_a_signal_is_never_traded_twice_across_polls():
    signals, orders, state, _ = poll([trade()])
    assert len(signals) == 1 and len(orders) == 3
    signals2, orders2, _, stats2 = poll([trade()], signals=signals, orders=orders,
                                        state=state, now=1100)
    assert len(signals2) == 1 and len(orders2) == 3
    assert stats2["signals_traded"] == 0


def test_a_market_order_fills_once_and_never_tops_itself_up():
    _, orders, state, _ = poll([trade()], now=1000)
    chase = orders[orders["arm"] == "market_chase"].iloc[0]
    assert chase["status"] == pt.ST_FILLED
    shares, cost = float(chase["shares"]), float(chase["cost_usd"])
    _, orders2, _, _ = poll([], orders=orders, state=state, now=1300,
                            book=raw_book(asks=[(0.10, 9999)]))
    chase2 = orders2[orders2["arm"] == "market_chase"].iloc[0]
    assert float(chase2["shares"]) == shares and float(chase2["cost_usd"]) == cost


def test_sell_trades_are_not_signals():
    signals, orders, _, stats = poll([trade(side="SELL")])
    assert len(signals) == 0 and len(orders) == 0 and stats["signals_seen"] == 0


def test_max_open_positions_cap_records_the_signal_and_opens_nothing():
    _, orders, state, _ = poll([trade()])
    signals, orders2, _, stats = poll(
        [trade(tx_hash="0xtx2", market_id="0xm2")], orders=orders, state=state,
        now=1100, r=rule(max_open_positions=1))
    assert stats["skipped_cap"] == 1
    assert signals.iloc[0]["status"] == "skipped_cap"
    assert len(orders2) == 3          # no new orders opened


def test_either_kill_switch_halts_the_arm(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "KILL_SWITCH_PATH", tmp_path / "KILL_PAPER_RW")
    monkeypatch.setattr(pt, "KILL_SWITCH_PATH", tmp_path / "KILL_PAPER_TRADER")
    assert not rw.kill_switch_engaged({})
    (tmp_path / "KILL_PAPER_TRADER").touch()
    assert rw.kill_switch_engaged({})          # the shared switch stops this arm too
    (tmp_path / "KILL_PAPER_TRADER").unlink()
    (tmp_path / "KILL_PAPER_RW").touch()
    assert rw.kill_switch_engaged({})          # ...and so does its own


def test_kill_switch_halts_run_before_any_fetch_or_write(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "KILL_SWITCH_PATH", tmp_path / "KILL_PAPER_RW")
    (tmp_path / "KILL_PAPER_RW").touch()

    def boom(*a, **k):
        raise AssertionError("network touched with the kill switch engaged")

    monkeypatch.setattr(rw, "make_session", boom)
    monkeypatch.setattr(rw, "load_freeze",
                        lambda: (watchlist(), _manifest_stub()))
    assert rw.run({"ingest": {}}, passes=1, dry_run=False)["halted"] is True


def test_per_position_notional_is_capped_below_the_configured_stake():
    r = rule(notional_per_signal_usd=100.0, max_notional_per_position_usd=25.0)
    _, orders, _, _ = poll([trade()], r=r, book=raw_book(asks=[(0.50, 100000)]))
    assert (orders["budget_usd"] == 25.0).all()
    assert float(orders[orders["arm"] == "market_chase"].iloc[0]["cost_usd"]) <= 25.0 + 1e-9


def test_config_can_tighten_but_never_loosen_the_frozen_caps():
    m = _manifest_stub()
    tight = rw.rule_from_manifest(m, {"paper_rw": {"max_open_positions": 5,
                                                   "max_notional_per_position_usd": 10.0}})
    assert tight["max_open_positions"] == 5 and tight["max_notional_per_position_usd"] == 10.0
    loose = rw.rule_from_manifest(m, {"paper_rw": {"max_open_positions": 10**9,
                                                   "max_notional_per_position_usd": 10**9}})
    assert loose["max_open_positions"] == m["seatbelts"]["max_open_positions"]
    assert loose["max_notional_per_position_usd"] == m["seatbelts"]["max_notional_per_position_usd"]


def test_module_has_no_execution_code_path():
    """The seatbelt that matters most: this file cannot place an order. No signing
    library, no key, no authenticated endpoint, and `get` is the only HTTP verb."""
    src = open(rw.__file__).read().lower()
    for token in ("private_key", "privatekey", "signer", "eth_account", "web3",
                  "sign_typed", "eip712", "post_order", "place_order", "create_order",
                  "l1_headers", "l2_headers", "api_secret", "passphrase", "mnemonic",
                  ".post(", ".put(", ".delete(", ".patch("):
        assert token not in src, f"execution-shaped token {token!r} found"
    assert ".get(" in src


# ---------------------------------------------------------------------------
# 6. THE PRE-REGISTRATION
# ---------------------------------------------------------------------------

def _validated_stub(tmp_path, persisted=True):
    path = tmp_path / "validated.parquet"
    rows = []
    for i, pre in enumerate(rw.WALLET_PREFIXES):
        rows.append({
            "wallet": pre + "f" * (42 - len(pre)),
            "edge_persisted": persisted, "out_of_sample_residual_edge": 0.05 + i / 100,
            "out_of_sample_edge": 0.03, "out_of_sample_n": 1000 + i,
            "out_of_sample_markets": 300 + i, "out_of_sample_cluster_p": 0.0005,
            "in_sample_residual_edge": 0.06, "in_sample_n": 1000,
            "stratum": "s2_50_99", "prior_arm": "", "n_discovery_bets": 80,
        })
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_resolve_wallets_refuses_a_prefix_that_matches_nothing(tmp_path):
    path = _validated_stub(tmp_path)
    with pytest.raises(KeyError):
        rw.resolve_wallets(prefixes=("0xdeadbeef",), validated_path=path)


def test_resolve_wallets_refuses_an_uncertified_wallet(tmp_path):
    path = _validated_stub(tmp_path, persisted=False)
    with pytest.raises(ValueError):
        rw.resolve_wallets(validated_path=path)


def test_resolve_wallets_refuses_an_ambiguous_prefix(tmp_path):
    path = tmp_path / "v.parquet"
    pd.DataFrame([{"wallet": "0xaa" + c * 38, "edge_persisted": True,
                   "out_of_sample_residual_edge": 0.05, "out_of_sample_edge": 0.03,
                   "out_of_sample_n": 10, "out_of_sample_markets": 10,
                   "out_of_sample_cluster_p": 0.001, "in_sample_residual_edge": 0.05,
                   "in_sample_n": 10, "stratum": "s", "prior_arm": "",
                   "n_discovery_bets": 10} for c in "12"]).to_parquet(path)
    with pytest.raises(ValueError):
        rw.resolve_wallets(prefixes=("0xaa",), validated_path=path)


def test_the_watchlist_prefixes_resolve_against_the_real_validated_table():
    """Guards the actual watchlist: every prefix resolves to exactly one distinct,
    certified wallet. AMENDMENT 1 grew this from 5 to 9 (the original five, kept,
    plus the four copy-validated survivors), so it asserts against the constant
    rather than a magic number."""
    if not rw.VALIDATED_PATH.exists():
        pytest.skip("deep real-world validation table not present on this box")
    wl = rw.resolve_wallets()
    assert len(wl) == len(rw.WALLET_PREFIXES)
    assert wl["edge_persisted"].all()
    assert wl["wallet"].nunique() == len(rw.WALLET_PREFIXES)


def test_the_headline_tier_is_a_subset_of_the_watchlist():
    """AMENDMENT 1 regression: the headline tier named four wallets that were not
    being polled. A tier that references an unwatched wallet silently scores an
    empty stratum forever."""
    assert set(rw.COPY_VALIDATED_WALLETS) <= set(rw.WALLET_PREFIXES)
    assert set(rw.LEGACY_WALLETS) <= set(rw.WALLET_PREFIXES)


def test_freeze_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "FREEZE_MANIFEST_PATH", tmp_path / "m.json")
    monkeypatch.setattr(rw, "WATCHLIST_PATH", tmp_path / "w.parquet")
    monkeypatch.setattr(rw, "SIGNALS_PATH", tmp_path / "sig.parquet")
    monkeypatch.setattr(rw, "VALIDATED_PATH", _validated_stub(tmp_path))
    monkeypatch.setattr(rw, "WALLET_PROFILE_PATH", tmp_path / "missing.parquet")
    m = rw.freeze({}, freeze_ts=1000)
    assert m["forward_observations_at_freeze"] == 0
    with pytest.raises(SystemExit):
        rw.freeze({}, freeze_ts=1000)


def test_forced_refreeze_is_logged_with_the_observation_count(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "FREEZE_MANIFEST_PATH", tmp_path / "m.json")
    monkeypatch.setattr(rw, "WATCHLIST_PATH", tmp_path / "w.parquet")
    monkeypatch.setattr(rw, "SIGNALS_PATH", tmp_path / "sig.parquet")
    monkeypatch.setattr(rw, "VALIDATED_PATH", _validated_stub(tmp_path))
    monkeypatch.setattr(rw, "WALLET_PROFILE_PATH", tmp_path / "missing.parquet")
    rw.freeze({}, freeze_ts=1000)
    pd.DataFrame({"signal_id": ["a", "b", "c"]}).to_parquet(tmp_path / "sig.parquet")
    m = rw.freeze({}, freeze_ts=1000, force=True, amend_reason="test")
    assert len(m["amendments"]) == 1
    a = m["amendments"][0]
    assert a["forward_observations_at_amendment"] == 3
    assert a["legitimate_pre_registration"] is False


def test_the_manifest_pre_registers_the_per_wallet_strata_before_any_data(
        monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "FREEZE_MANIFEST_PATH", tmp_path / "m.json")
    monkeypatch.setattr(rw, "WATCHLIST_PATH", tmp_path / "w.parquet")
    monkeypatch.setattr(rw, "SIGNALS_PATH", tmp_path / "sig.parquet")
    monkeypatch.setattr(rw, "VALIDATED_PATH", _validated_stub(tmp_path))
    monkeypatch.setattr(rw, "WALLET_PROFILE_PATH", tmp_path / "missing.parquet")
    m = rw.freeze({}, freeze_ts=1000)
    assert m["strata"]["pooled"] == rw.POOLED_STRATUM
    assert m["strata"]["per_wallet"] == list(rw.WALLET_PREFIXES)
    assert m["forward_observations_at_freeze"] == 0
    assert m["headline_arm"] == "market_chase"
    assert m["control_arm"] == "placebo_random"
    assert len(m["wallets"]) == len(rw.WALLET_PREFIXES)
    # the rule text is in the manifest, not only in the code
    assert "is_real_world" in m["entry_rule"]
    assert "micro_crypto" in m["entry_rule"]
    assert "detect_lag_s" in m["entry_rule"]
    json.loads(json.dumps(m))     # must be serializable exactly as written


def test_the_entry_rule_pins_the_universe_the_fees_and_the_lag():
    for phrase in ("REAL-WORLD FILTER", "skipped_micro_crypto", "COPY LAG",
                   "feeSchedule.rate", "shares * k * price * (1 - price)",
                   "market_chase (THE HEADLINE FOR THIS ARM)"):
        assert phrase in rw.ENTRY_RULE, phrase


def test_the_scoring_rule_pins_the_strata_and_the_reading_order():
    for phrase in ("rw5_pooled", "one stratum per frozen wallet",
                   "SELECTION test", "TIMING test", "EVENTS with replacement"):
        assert phrase in rw.SCORING_RULE, phrase


# ---------------------------------------------------------------------------
# 7. SCORING
# ---------------------------------------------------------------------------

def _manifest_stub():
    return {
        "arm": "paper_rw", "freeze_utc": "2026-07-28T00:00:00Z",
        "git_commit": "deadbeefdeadbeef", "headline_arm": rw.HEADLINE_ARM,
        "control_arm": rw.CONTROL_ARM, "hypothesis": rw.HYPOTHESIS,
        "entry_rule": rw.ENTRY_RULE, "scoring_rule": rw.SCORING_RULE,
        "stake": {"notional_per_signal_usd": 100.0},
        "strata": {"pooled": rw.POOLED_STRATUM, "per_wallet": ["0xaaaa", "0xbbbb"]},
        "costs": {"taker_fee_k_stress": 0.07},
        "placebo": {"seed": "rw-seed-v1", "fallback_horizon_hours": 48.0},
        "seatbelts": {"max_notional_per_position_usd": 100.0,
                      "max_open_positions": 2000,
                      "kill_switch_paths": ["data/KILL_PAPER_TRADER",
                                            "data/KILL_PAPER_RW"],
                      "writes_only": ["data/interim/paper_rw"]},
        "known_limits": ["small samples are expected"],
    }


def _scored_out(n_boot=50):
    """Two wallets x two markets x three arms, one no-fill, hand-built so the
    dollar figures are known by construction."""
    rows, signals, res = [], [], []
    for i, (label, mkt) in enumerate([("0xaaaa", "m1"), ("0xbbbb", "m2")]):
        for arm in rw.ARMS:
            filled = not (arm == "limit_noChase" and label == "0xbbbb")
            rows.append({
                "order_id": f"{mkt}:{arm}", "signal_id": f"s{mkt}", "arm": arm,
                "wallet": label + "0000", "wallet_label": label, "market_id": mkt,
                "token_id": f"t{mkt}", "category": "politics", "fee_k": 0.04,
                "shares": 100.0 if filled else 0.0,
                "cost_usd": 60.0 if filled else 0.0,
                "fee_units": 24.0 if filled else 0.0,
                "is_taker": arm != "limit_noChase",
                "status": pt.ST_FILLED if filled else pt.ST_NO_FILL,
                "no_fill_reason": None if filled else "never_filled_before_settlement",
                "avg_fill_price": 0.60 if filled else float("nan"),
                "wallet_entry_price": 0.50,
                "wallet_ts": 1000, "detect_ts": 1120,
                "first_fill_ts": 1120 if filled else None,
                "slug": f"senate-{mkt}", "question": "Will it pass?",
            })
        signals.append({"signal_id": f"s{mkt}", "slug": f"senate-{mkt}",
                        "question": "Will it pass?", "market_id": mkt,
                        "status": "accepted"})
        res.append({"market_id": mkt, "token_id": f"t{mkt}", "resolved": True,
                    "resolved_value": 1.0 if i == 0 else 0.0})
    orders = rw.copy_lag_columns(pd.DataFrame(rows))
    out = rw.score_orders(orders, pd.DataFrame(signals), pd.DataFrame(res), rule(),
                          n_boot=n_boot)
    out["_orders"], out["_signals"] = orders, pd.DataFrame(signals)
    return out


def test_no_fills_score_zero_dollars_and_still_count_in_the_fill_rate():
    out = _scored_out()
    board = out["board"]
    limit = board[(board["arm"] == "limit_noChase") &
                  (board["stratum"] == rw.POOLED_STRATUM)].iloc[0]
    assert limit["n_signals"] == 2 and limit["n_filled"] == 1
    assert limit["fill_rate"] == pytest.approx(0.5)
    nf = out["scored"]
    assert (nf.loc[~nf["filled"], "pnl_usd_net"] == 0).all()


def test_gross_and_net_are_both_reported_and_net_is_the_smaller():
    board = _scored_out()["board"]
    chase = board[(board["arm"] == "market_chase") &
                  (board["stratum"] == rw.POOLED_STRATUM)].iloc[0]
    assert chase["pnl_per_signal_usd_gross"] > chase["pnl_per_signal_usd_net"]
    assert chase["total_fee_usd"] == pytest.approx(2 * 0.04 * 24.0)
    assert chase["total_pnl_usd_net_stress"] < chase["total_pnl_usd_net"]


def test_the_board_carries_a_row_per_wallet_and_the_pooled_row():
    board = _scored_out()["board"]
    assert set(board["stratum"]) == {rw.POOLED_STRATUM, "0xaaaa", "0xbbbb"}
    pooled = board[(board["stratum"] == rw.POOLED_STRATUM) &
                   (board["arm"] == "market_chase")].iloc[0]
    per = board[(board["stratum"] != rw.POOLED_STRATUM) &
                (board["arm"] == "market_chase")]
    assert pooled["n_signals"] == int(per["n_signals"].sum())


def test_a_single_wallet_cannot_hide_inside_the_pooled_mean():
    """0xaaaa's market resolves YES, 0xbbbb's NO. The pooled mean is ~0; the
    per-wallet rows are +/-, which is the whole reason they are pre-registered."""
    board = _scored_out()["board"]
    a = board[(board["stratum"] == "0xaaaa") & (board["arm"] == "market_chase")].iloc[0]
    b = board[(board["stratum"] == "0xbbbb") & (board["arm"] == "market_chase")].iloc[0]
    assert a["pnl_per_signal_usd_net"] > 0 > b["pnl_per_signal_usd_net"]


def test_scoreboard_says_awaiting_data_instead_of_shipping_a_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "SCOREBOARD_MD", tmp_path / "sb.md")
    monkeypatch.setattr(rw, "SCOREBOARD_PARQUET", tmp_path / "sb.parquet")
    out = {"board": pd.DataFrame(), "contrast": None, "n_unresolved": 4,
           "note": "signals captured, none of their markets resolved yet",
           "signal_status_counts": {"accepted": 4}}
    rw.write_scoreboard(out, _manifest_stub(), watchlist(),
                        pd.DataFrame({"signal_id": ["a"], "status": ["accepted"]}),
                        pt._empty(rw.ORDER_COLUMNS))
    text = (tmp_path / "sb.md").read_text()
    assert "AWAITING DATA" in text
    assert "not** a zero" in text


def test_scoreboard_puts_the_fill_rate_and_both_arms_of_the_read_next_to_dollars(
        monkeypatch, tmp_path):
    monkeypatch.setattr(rw, "SCOREBOARD_MD", tmp_path / "sb.md")
    monkeypatch.setattr(rw, "SCOREBOARD_PARQUET", tmp_path / "sb.parquet")
    out = _scored_out()
    rw.write_scoreboard(out, _manifest_stub(), watchlist(), out["_signals"],
                        out["_orders"])
    text = (tmp_path / "sb.md").read_text()
    assert "**fill rate**" in text
    assert "Per wallet — the breadwinner cut" in text
    assert "SELECTION" in text and "TIMING" in text
    assert "net $/sig" in text and "gross $/sig" in text


def test_the_scoreboard_shows_what_the_copy_paid_versus_what_the_wallet_paid(
        monkeypatch, tmp_path):
    """A market order into a thin book can pay several times the wallet's price.
    That is the most likely way this arm comes back negative, so it is reported
    rather than buried inside the P&L."""
    monkeypatch.setattr(rw, "SCOREBOARD_MD", tmp_path / "sb.md")
    monkeypatch.setattr(rw, "SCOREBOARD_PARQUET", tmp_path / "sb.parquet")
    out = _scored_out()
    chase = out["board"][(out["board"]["stratum"] == rw.POOLED_STRATUM) &
                         (out["board"]["arm"] == "market_chase")].iloc[0]
    # fixture: filled at 0.60, wallet paid 0.50
    assert chase["mean_fill_price"] == pytest.approx(0.60)
    assert chase["mean_wallet_entry_price"] == pytest.approx(0.50)
    assert chase["mean_slippage_vs_wallet"] == pytest.approx(0.10)
    rw.write_scoreboard(out, _manifest_stub(), watchlist(), out["_signals"],
                        out["_orders"])
    text = (tmp_path / "sb.md").read_text()
    assert "What the copy actually paid, versus what the wallet paid" in text
    assert "bracket the truth" in text


def test_skipped_micro_crypto_signals_never_enter_a_dollar_figure():
    """The filter is pre-registered as an exclusion, so a skipped signal must not
    appear in n_signals on any arm — but it must still be visible on the page."""
    out = _scored_out()
    sig = pd.concat([out["_signals"],
                     pd.DataFrame([{"signal_id": "sX", "slug": "btc-updown-5m-1",
                                    "question": "Up or Down?", "market_id": "mX",
                                    "status": rw.SKIP_MICRO}])], ignore_index=True)
    out2 = rw.score_orders(out["_orders"], sig, pd.DataFrame(
        [{"market_id": m, "token_id": f"t{m}", "resolved": True,
          "resolved_value": 1.0} for m in ("m1", "m2")]), rule(), n_boot=50)
    pooled = out2["board"][out2["board"]["stratum"] == rw.POOLED_STRATUM]
    assert set(pooled["n_signals"]) == {2}
    assert out2["signal_status_counts"][rw.SKIP_MICRO] == 1
