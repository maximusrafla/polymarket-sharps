"""Stage 1 of the execution roadmap — the FORWARD PAPER TRADER.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PAPER ONLY. READ-ONLY. NO PRIVATE KEYS. NO SIGNING. NO ORDERS.      │
    │                                                                      │
    │  This module places nothing. It issues public, unauthenticated GETs  │
    │  to data-api.polymarket.com (/trades) and clob.polymarket.com        │
    │  (/book, /markets/<condition_id>) and writes hypothetical fills to a │
    │  local parquet ledger. It imports no signing library, holds no key,  │
    │  and makes no POST/PUT/DELETE of any kind. A test                    │
    │  (tests/test_paper_trader.py::test_module_has_no_execution_code_path)│
    │  asserts that by scanning this file's source.                        │
    │                                                                      │
    │  Seatbelts are built now, before they are needed, because CLAUDE.md  │
    │  says they must not be bolted on later: a kill-switch FILE whose     │
    │  presence halts the loop, a per-position notional cap, a cap on      │
    │  simultaneously-open positions, and a --dry-run that is the DEFAULT  │
    │  (persisting anything requires an explicit --commit).                │
    └──────────────────────────────────────────────────────────────────────┘

WHY THIS EXISTS
---------------
This repo has ~110 commits of retrospective analysis and has never once tested
whether following its wallets makes money — not real, not paper. Every
retrospective copyability probe has been poisoned by the same defect (see
`docs/redteam_audit_2026-07-26.md` C9 and `docs/project3_sports_metric_b.md` §8):
the trade tape we own is ~98% our OWN tracked wallets, so the "market price" a
retrospective follower is priced against is other tracked wallets' prints rather
than the order book. This module removes that defect by construction — it prices
every fill against the LIVE `/book`, going forward, and it never peeks.

THE THREE ARMS (all fired off the SAME detected signals, so they are paired)
---------------------------------------------------------------------------
* `limit_noChase`  — THE THESIS. A resting buy limit at or below the wallet's own
                     entry price. Fills only when real size appears at that price.
                     "Take the selection, refuse the chase", the strategy
                     `docs/project3_sports_metric_b.md` §7 left explicitly open.
* `market_chase`   — a taker order at the best ask at detection. The repo has
                     already shown this is dead, so it is a SANITY CHECK ON THIS
                     MACHINERY: if it comes out strongly profitable, suspect a bug
                     before believing it.
* `placebo_random` — same market, same side, entry triggered at a pre-registered
                     random moment in the market's remaining life, ignoring the
                     wallet's timing entirely. THIS is the arm that says whether
                     the wallets add anything. If `limit_noChase` does not beat
                     it, the wallets are not the signal.

THE THREE THINGS THAT MAKE PAPER TRADING LIE, AND WHERE EACH IS BLOCKED
-----------------------------------------------------------------------
1. ASSUMING FILLS. A limit that never gets hit is a real cost. No-fills are
   first-class rows with a terminal status and a reason; `fill_rate` is reported
   per arm; and the HEADLINE statistic is dollars *per detected signal* with
   unfilled signals contributing exactly $0 — so a strategy that fills 5% of the
   time and wins on those cannot look profitable here.
2. PRICING OFF THE TAPE. Every fill decision in this module prices against
   `/book` with real size at the ask. `fill_from_book` is the only path that can
   produce a fill and it takes a book, never a print. If the book has no size at
   your price, you did not fill.
3. CHOOSING THE RULE AFTER SEEING RESULTS. `freeze` writes a manifest holding the
   entry rule verbatim, the stake, the watchlist, the scoring rule, the fee model,
   the placebo seed, the freeze timestamp and `forward_observations_at_freeze`;
   it refuses to overwrite without `--force` and logs every amendment with the
   number of forward observations that existed at the time.

STATISTICS
----------
Scored at the unit of independent resolution: the sports resolution EVENT
(`src/sports_events.py`) for sports markets, the market elsewhere. Bootstraps
resample EVENTS, never bets. `n_signals`, `n_filled`, `fill_rate` and
`effective_events` sit next to every dollar figure, because volume is not
evidence. Samples will be tiny at first and the scoreboard says "AWAITING DATA"
out loud rather than shipping an empty table that reads as a zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time

import numpy as np
import pandas as pd

from src.common import (
    DATA_DIR,
    REPO_ROOT,
    INTERIM_DIR,
    PROCESSED_DIR,
    RANKED_WALLETS_PATH,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
)
from src.watch import (
    advance_cursor,
    default_cursor,
    new_trades_since,
    normalize_trade,
    position_key,
)

# --- paths (state is deliberately tiny; the box has ~2.3 GB free) -----------
PAPER_DIR = INTERIM_DIR / "paper"
STATE_PATH = PAPER_DIR / "paper_state.json"
SIGNALS_PATH = PAPER_DIR / "paper_signals.parquet"
ORDERS_PATH = PAPER_DIR / "paper_orders.parquet"
RESOLUTIONS_PATH = PAPER_DIR / "paper_resolutions.parquet"

FREEZE_MANIFEST_PATH = PROCESSED_DIR / "paper_freeze_manifest.json"
WATCHLIST_PATH = PROCESSED_DIR / "paper_watchlist.parquet"
SCOREBOARD_MD = PROCESSED_DIR / "paper_scoreboard.md"
SCOREBOARD_PARQUET = PROCESSED_DIR / "paper_scoreboard.parquet"

# SEATBELT: create this file and the loop halts on its next check, before any
# fetch and before any write. `touch data/KILL_PAPER_TRADER` is the stop button.
KILL_SWITCH_PATH = DATA_DIR / "KILL_PAPER_TRADER"

SPORTS_ARMS = ("limit_noChase", "market_chase", "placebo_random")
ARMS = SPORTS_ARMS
HEADLINE_ARM = "limit_noChase"
CONTROL_ARM = "placebo_random"

# Seatbelt defaults. The frozen manifest overrides the economic ones (stake, fee
# model) so the rule cannot drift after the fact; the operational ones (caps,
# cadence) are read live from config so an operator can tighten but not loosen a
# pre-registration.
DEFAULT_NOTIONAL_USD = 100.0
DEFAULT_MAX_POSITION_USD = 100.0
# Deliberately high for the PAPER stage. The cap is real scaffolding that stage 2
# will set low, but at paper stage a binding cap would truncate the sample exactly
# during busy markets — a selection effect on WHICH signals get tested. Any
# `skipped_cap` is reported on the scoreboard so a binding cap is never silent.
DEFAULT_MAX_OPEN_POSITIONS = 5000
DEFAULT_TAKER_FEE_K = 0.0
DEFAULT_TAKER_FEE_K_STRESS = 0.10
DEFAULT_PLACEBO_HORIZON_HOURS = 6.0
DEFAULT_TRADES_FETCH_OVERLAP_S = 2
BOOK_LEVELS_RECORDED = 5

# Order statuses. `partial` is LIVE on purpose: a resting limit that got 30 of the
# 55 shares it wanted still has an unfilled remainder, and pretending otherwise
# would quietly convert an unfilled remainder into a fill it never got.
ST_OPEN = "open"            # resting limit, nothing filled yet
ST_PENDING = "pending"      # placebo waiting for its pre-drawn trigger time
ST_PARTIAL = "partial"      # some shares filled, budget not exhausted, still resting
ST_FILLED = "filled"        # terminal: budget spent, or a one-shot taker arm is done
ST_NO_FILL = "no_fill"      # terminal: market settled without a single share
LIVE_STATUSES = (ST_OPEN, ST_PENDING, ST_PARTIAL)
FILLED_STATUSES = (ST_PARTIAL, ST_FILLED)


# ---------------------------------------------------------------------------
# THE PRE-REGISTERED RULE. Frozen verbatim into the manifest by `freeze`.
# ---------------------------------------------------------------------------

ENTRY_RULE = (
    "SIGNAL. A signal is a new BUY trade by a wallet on the frozen watchlist, "
    "detected by polling the public data-api /trades?user= endpoint. The last-seen "
    "cursor floors at the moment each wallet joins the watchlist, so pre-existing "
    "history is never traded. Every signal is recorded, including ones that produce "
    "no order. SELL trades are ignored (a position is an entry). "
    "\n"
    "BOOK. At detection, the live CLOB order book for that outcome token is fetched "
    "(GET /book?token_id=) and persisted with the signal: best bid/ask, the size at "
    "each, the top levels of both sides, the token's tick size and minimum order "
    "size. Every fill decision in every arm is made against a book snapshot with "
    "real size at the price; no fill is ever inferred from the trade tape. "
    "\n"
    "STAKE. Fixed 100.00 USD of notional per signal per arm — never scaled by "
    "conviction, wallet, price or bankroll — capped by the per-position notional cap. "
    "A fill may be partial: you get the shares the book actually offered, and the "
    "unspent budget stays unspent. "
    "\n"
    "ARM 1, limit_noChase (THE THESIS, HEADLINE ARM). A resting buy limit at the "
    "wallet's own entry price, rounded DOWN to the token's tick grid so the limit is "
    "never above the price they paid. If the best ask is already at or below the "
    "limit at detection, it fills immediately as a TAKER, walking the book across "
    "levels priced at or below the limit and paying the taker fee. Otherwise the "
    "order rests and every subsequent poll re-checks the live book: whenever size "
    "appears at or below the limit, it fills as a MAKER at the limit price (a passive "
    "bid is hit AT its own price, not at the better price that swept through it) and "
    "pays no fee. Partial fills accumulate across polls against the same budget. If "
    "the market settles with the order unfilled it is recorded as a NO-FILL, which is "
    "an outcome, not a discard. "
    "\n"
    "ARM 2, market_chase (SANITY CHECK). A taker order at detection that walks the "
    "book from the best ask upward until the 100 USD budget is spent, at whatever "
    "prices are there, paying the taker fee. No limit. If the book has no asks at "
    "detection it is a NO-FILL. "
    "\n"
    "ARM 3, placebo_random (THE CONTROL). Same market, same outcome token, same "
    "stake, but the wallet's timing is discarded: at detection a trigger time is "
    "drawn ONCE, deterministically, as "
    "trigger = detect_ts + u * (horizon_ts - detect_ts) where u in [0,1) is "
    "SHA-256(placebo_seed || signal_id) read as a fraction. horizon_ts is the first "
    "of the CLOB's game_start_time and end_date_iso that is still in the FUTURE at "
    "detection, and otherwise detect_ts plus the frozen fallback horizon; "
    "end_date_iso alone is not usable as a horizon because it is the midnight floor "
    "of the scheduled end day and is usually already past. The source of the horizon "
    "is recorded per signal. On the first poll at or after the trigger the arm "
    "takes at the best ask exactly like market_chase. If the market settles before "
    "the trigger fires it is a NO-FILL. The draw is a pure function of the frozen "
    "seed and the signal id, so it is reproducible and could not have been redrawn "
    "after seeing an outcome. "
    "\n"
    "COSTS. The taker fee is shares * k * price * (1 - price), summed level by level "
    "across a walked fill, with k frozen in this manifest; maker fills pay zero. "
    "Prices are the prices actually resting on the book, so slippage is whatever "
    "walking that book costs. The tick grid is the token's own minimum_tick_size "
    "from the CLOB, not a constant. A fill smaller than the token's minimum order "
    "size is not a fill. "
    "\n"
    "SEATBELTS. Paper only: no keys, no signing, no order placement, public GETs "
    "only. The presence of the kill-switch file halts the loop before any fetch or "
    "write. Notional per position is capped. When the number of simultaneously-live "
    "orders is at the cap, a new signal is recorded with status skipped_cap and "
    "generates no orders — it still counts in n_signals, so the cap costs the "
    "strategy rather than flattering it. --dry-run is the default and persists "
    "nothing."
)

SCORING_RULE = (
    "Every arm is scored on the SAME set of detected signals, so the arms are paired "
    "signal by signal and event by event. A signal enters scoring once its market has "
    "a resolution; unresolved signals are excluded from every arm identically and "
    "counted as n_unresolved. "
    "\n"
    "PER ORDER. net P&L in USD = shares * resolved_value - cost_usd - fee_usd, where "
    "cost_usd is what the walked book actually charged and fee_usd is the frozen "
    "taker-fee model applied to taker fills only. A NO-FILL scores exactly 0.00 USD "
    "on 0.00 USD of notional — it is never dropped. "
    "\n"
    "THE HEADLINE STATISTIC IS DOLLARS PER DETECTED SIGNAL, with unfilled signals "
    "contributing 0. This is the number that makes the fill rate bite: an arm that "
    "fills 5% of the time and wins on those earns 5% of the dollars, and this "
    "statistic says so. Return on FILLED notional is reported alongside, and must "
    "never be quoted without the fill rate next to it. "
    "\n"
    "CLUSTERING. The unit of independent resolution is the sports resolution EVENT "
    "(src/sports_events.py, version recorded in this manifest) for markets the sports "
    "labeller recognises, and the market_id elsewhere. The headline is EVENT-WEIGHTED: "
    "the mean over events of the within-event mean dollars-per-signal. Bootstrap "
    "resamples EVENTS with replacement, never signals, giving a 95% CI and a one-sided "
    "p versus zero; it is undefined below two events. n_signals, n_filled, fill_rate, "
    "n_events and effective_events (Kish) are reported next to every dollar figure. "
    "\n"
    "THE DECIDING COMPARISON is limit_noChase minus placebo_random on the events where "
    "both arms have a scored signal, event-weighted, with the difference bootstrapped "
    "over those shared events. If that difference does not clear zero, the wallets are "
    "not the signal and the selection thesis is dead. market_chase is a machinery "
    "check, not a candidate: this repo has already shown chasing is dead, so a "
    "strongly profitable market_chase indicates a bug, not an edge. "
    "\n"
    "STRATA, fixed here in advance: the pooled book (`all`, the headline) plus one cut "
    "per source cohort (`s2_twelve`, `p1_certified`). The cohorts are different "
    "populations with different resolution speeds and wildly different signal rates, "
    "so a pooled number is dominated by whichever is louder; the cuts exist so that "
    "cannot be mistaken for a result. No further slicing is permitted. "
    "\n"
    "Arms are never re-defined, re-weighted or added in light of forward results, and "
    "the watchlist is never re-frozen."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# SEATBELTS
# ---------------------------------------------------------------------------

def kill_switch_path(cfg: dict | None = None):
    """Where the stop button lives. Configurable so an operator can point it at a
    tmpfs or a shared mount, but it always defaults to a path inside data/."""
    cfg = cfg or {}
    custom = (cfg.get("paper", {}) or {}).get("kill_switch_path")
    if custom:
        from pathlib import Path
        return Path(custom)
    return KILL_SWITCH_PATH


def kill_switch_engaged(cfg: dict | None = None) -> bool:
    return kill_switch_path(cfg).exists()


def kill_switch_label(cfg: dict | None = None) -> str:
    """The kill switch as a REPO-RELATIVE path where possible. The manifest and the
    scoreboard are committed artifacts read on other machines (and this module was
    developed in a git worktree), so recording an absolute path there would print an
    instruction that does not exist on the reader's box."""
    path = kill_switch_path(cfg)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def paper_settings(cfg: dict | None = None) -> dict:
    """Operational settings (caps, cadence). Economic settings that define the RULE
    (stake, fee k, placebo seed) live in the frozen manifest, not here."""
    p = ((cfg or {}).get("paper", {}) or {})
    return {
        "notional_per_signal_usd": float(p.get("notional_per_signal_usd", DEFAULT_NOTIONAL_USD)),
        "max_notional_per_position_usd": float(
            p.get("max_notional_per_position_usd", DEFAULT_MAX_POSITION_USD)),
        "max_open_positions": int(p.get("max_open_positions", DEFAULT_MAX_OPEN_POSITIONS)),
        "poll_seconds": int(p.get("poll_seconds", 60)),
        "taker_fee_k": float(p.get("taker_fee_k", DEFAULT_TAKER_FEE_K)),
        "taker_fee_k_stress": float(p.get("taker_fee_k_stress", DEFAULT_TAKER_FEE_K_STRESS)),
        "placebo_fallback_horizon_hours": float(
            p.get("placebo_fallback_horizon_hours", DEFAULT_PLACEBO_HORIZON_HOURS)),
        "enforce_min_order_size": bool(p.get("enforce_min_order_size", True)),
    }


def capped_budget(notional: float, cap: float) -> float:
    """SEATBELT: no single paper position may exceed the per-position notional cap,
    whatever the configured stake says."""
    return float(min(notional, cap))


# ---------------------------------------------------------------------------
# THE BOOK — the only thing allowed to produce a fill
# ---------------------------------------------------------------------------

def parse_book(raw: dict | None) -> dict:
    """Normalize a CLOB /book payload.

    The endpoint returns both sides worst-price-first; we sort explicitly rather
    than trust that, so `asks[0]` is always the best (lowest) ask and `bids[0]` the
    best (highest) bid. Zero-size and out-of-range levels are dropped: a level with
    no size is not liquidity."""
    empty = {"asks": [], "bids": [], "tick_size": None, "min_order_size": 0.0,
             "book_ts": None}
    if not raw:
        return empty

    def levels(key: str, reverse: bool) -> list[tuple[float, float]]:
        out = []
        for lv in (raw.get(key) or []):
            try:
                price, size = float(lv["price"]), float(lv["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if size > 0 and 0.0 < price <= 1.0:
                out.append((price, size))
        out.sort(key=lambda ps: ps[0], reverse=reverse)
        return out

    def num(key, default=None):
        try:
            return float(raw[key])
        except (KeyError, TypeError, ValueError):
            return default

    return {
        "asks": levels("asks", reverse=False),
        "bids": levels("bids", reverse=True),
        "tick_size": num("tick_size"),
        "min_order_size": num("min_order_size", 0.0) or 0.0,
        "book_ts": raw.get("timestamp"),
    }


def best_ask(book: dict) -> tuple[float, float]:
    return book["asks"][0] if book["asks"] else (float("nan"), 0.0)


def best_bid(book: dict) -> tuple[float, float]:
    return book["bids"][0] if book["bids"] else (float("nan"), 0.0)


def round_down_to_tick(price: float, tick: float | None) -> float:
    """Round a BUY limit DOWN to the token's tick grid. Down, always: rounding up
    would put the limit above the wallet's own entry price and quietly turn
    `limit_noChase` into a small chase."""
    if not tick or tick <= 0 or not np.isfinite(price):
        return float(price)
    return float(math.floor(price / tick + 1e-9) * tick)


def taker_fee(shares: float, price: float, k: float) -> float:
    """Polymarket's taker fee shape, as measured in docs/project4_value_gate.md §3:
    `shares * k * price * (1 - price)`. It vanishes toward 0c and 100c and is near
    its maximum mid-book, which is why it barely touched the 0.99 value-betting band
    and would bite hard on a 0.5 game line. `k` is frozen in the manifest; it is NOT
    pinned by any public Polymarket doc this repo has verified, so the scoreboard
    also reports a stressed-k column."""
    return float(shares) * float(k) * float(price) * (1.0 - float(price))


def walk_book(asks: list[tuple[float, float]], budget_usd: float,
              limit_price: float | None = None) -> dict:
    """Buy up to `budget_usd` of notional by walking `asks` from the best price
    upward, taking only what is actually resting at each level.

    This is where slippage is modelled honestly: if the top level is thin, the fill
    continues into worse prices and the average fill price rises. If a `limit_price`
    is given, levels above it are simply not reachable — the walk stops there, which
    is how a limit order behaves and how a no-fill arises.

    Returns shares, cost, average price, levels consumed, and `fee_units` =
    sum(shares_i * p_i * (1 - p_i)) so a fee at any k is `k * fee_units` without
    re-walking the book."""
    shares = cost = fee_units = 0.0
    levels_used = 0
    budget = max(0.0, float(budget_usd))
    for price, size in asks:
        if limit_price is not None and price > limit_price + 1e-12:
            break
        remaining = budget - cost
        if remaining <= 1e-9:
            break
        take = min(float(size), remaining / price)
        if take <= 0:
            break
        shares += take
        cost += take * price
        fee_units += take * price * (1.0 - price)
        levels_used += 1
    return {
        "shares": shares,
        "cost_usd": cost,
        "avg_price": (cost / shares) if shares > 0 else float("nan"),
        "levels_used": levels_used,
        "fee_units": fee_units,
    }


def size_at_or_below(asks: list[tuple[float, float]], limit_price: float) -> float:
    """Shares resting at or below `limit_price`.

    A resting bid at L cannot coexist with an offer at or below L — the two would
    have matched. So seeing such an offer in a snapshot is direct evidence that size
    traded through L, i.e. that our resting order would have been filled for at
    least that much."""
    return float(sum(s for p, s in asks if p <= limit_price + 1e-12))


def maker_fill(asks: list[tuple[float, float]], budget_usd: float,
               limit_price: float) -> dict:
    """Fill a RESTING limit as the passive side: at the limit price, not at the
    better price that swept through it.

    If the ask falls from 0.60 to 0.50 our bid at 0.60 was consumed on the way down,
    at 0.60. Assuming we would have got 0.50 is the classic paper-trading lie of
    giving the passive order the aggressor's price."""
    if limit_price <= 0 or not np.isfinite(limit_price):
        return {"shares": 0.0, "cost_usd": 0.0, "avg_price": float("nan"),
                "levels_used": 0, "fee_units": 0.0}
    available = size_at_or_below(asks, limit_price)
    shares = min(available, max(0.0, float(budget_usd)) / limit_price)
    cost = shares * limit_price
    return {
        "shares": shares,
        "cost_usd": cost,
        "avg_price": limit_price if shares > 0 else float("nan"),
        "levels_used": 1 if shares > 0 else 0,
        "fee_units": shares * limit_price * (1.0 - limit_price),
    }


def fill_from_book(book: dict, budget_usd: float, *, mode: str,
                   limit_price: float | None = None,
                   enforce_min_size: bool = True) -> dict:
    """THE ONLY PATH THAT CAN PRODUCE A FILL. Takes a book, never a print.

    mode="taker" walks the ask side (optionally capped by `limit_price`) and is
    charged the taker fee. mode="maker" fills passively at `limit_price`.

    A fill below the token's minimum order size is not a fill: the exchange would
    have rejected it. That check is what stops a 3-share dribble of liquidity from
    being scored as a successful strategy."""
    if mode not in ("taker", "maker"):
        raise ValueError(f"unknown fill mode {mode!r}")
    if mode == "taker":
        res = walk_book(book.get("asks", []), budget_usd, limit_price=limit_price)
    else:
        res = maker_fill(book.get("asks", []), budget_usd, float(limit_price))
    res["is_taker"] = mode == "taker"
    min_size = float(book.get("min_order_size") or 0.0)
    if enforce_min_size and 0 < res["shares"] < min_size:
        return {"shares": 0.0, "cost_usd": 0.0, "avg_price": float("nan"),
                "levels_used": 0, "fee_units": 0.0, "is_taker": res["is_taker"],
                "rejected_min_size": True}
    res["rejected_min_size"] = False
    return res


# ---------------------------------------------------------------------------
# PLACEBO — drawn once, deterministically, at detection
# ---------------------------------------------------------------------------

def placebo_trigger_ts(signal_id: str, detect_ts: int, horizon_ts: int,
                       seed: str) -> int:
    """A uniform draw over the market's remaining life, keyed on the frozen seed and
    the signal id. Deterministic and reproducible: a reviewer can recompute every
    trigger time from the manifest, so no trigger can have been redrawn after seeing
    an outcome."""
    end = max(int(horizon_ts), int(detect_ts))
    digest = hashlib.sha256(f"{seed}:{signal_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return int(detect_ts + u * (end - int(detect_ts)))


def signal_id_for(trade: dict) -> str:
    """Stable id for a detected position: a hash of watch.py's own position key, so
    the same trade is the same signal across restarts."""
    return hashlib.sha1(position_key(trade).encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# NETWORK — public unauthenticated GETs only
# ---------------------------------------------------------------------------

def fetch_book(session, cfg: dict, token_id: str) -> dict | None:
    """GET clob.polymarket.com/book?token_id=... — public, unauthenticated, read-only."""
    base = cfg["data_source"]["clob_api_base"]
    resp = session.get(f"{base}/book", params={"token_id": token_id},
                       timeout=cfg["ingest"]["request_timeout_sec"])
    if resp.status_code in (404, 400):
        return None
    resp.raise_for_status()
    return resp.json()


def _iso_ts(value) -> int | None:
    if not value:
        return None
    try:
        return int(pd.Timestamp(value).timestamp())
    except (ValueError, TypeError):
        return None


def fetch_market_meta(session, cfg: dict, market_id: str) -> dict:
    """GET clob.polymarket.com/markets/<condition_id> for the market's scheduled end
    (the placebo's horizon), its tick and minimum size (a fallback for when /book is
    unavailable), plus a free capture of the exchange's own advertised fee fields,
    which this repo has never pinned. Read-only."""
    from src.ingest import fetch_market_resolution

    raw = fetch_market_resolution(session, cfg, market_id) or {}
    return {
        "market_end_ts": _iso_ts(raw.get("end_date_iso")),
        "game_start_ts": _iso_ts(raw.get("game_start_time")),
        "taker_base_fee_raw": raw.get("taker_base_fee"),
        "maker_base_fee_raw": raw.get("maker_base_fee"),
        "min_tick_size_meta": raw.get("minimum_tick_size"),
        "min_order_size_meta": raw.get("minimum_order_size"),
    }


def placebo_horizon(detect_ts: int, market_end_ts: int | None,
                    game_start_ts: int | None,
                    fallback_hours: float) -> tuple[int, str]:
    """The window the placebo's random entry is drawn over, and where it came from.

    `end_date_iso` CANNOT be trusted as a horizon on its own: `src/market_meta.py`
    §1.3 documents that it is the midnight FLOOR of the scheduled end day, and a live
    probe of 99 detected signals found 97 of them already past it. So the horizon is
    taken from the first source that is actually in the future, and falls back to a
    short frozen window otherwise — short, because the universe this watches is
    hourly-to-daily markets, and a multi-day fallback would simply never fire and
    would hand the control arm a fake no-fill.

    The source is recorded per signal so the scoreboard can be cut by whether the
    control had a real horizon or a guessed one."""
    for ts, name in ((game_start_ts, "game_start_time"), (market_end_ts, "end_date_iso")):
        if ts is not None and np.isfinite(ts) and int(ts) > int(detect_ts):
            return int(ts), name
    source = "fallback_past_end" if market_end_ts is not None else "fallback_no_end_date"
    return int(detect_ts + fallback_hours * 3600), source


def fetch_new_positions(session, cfg: dict, wallet: str, cursor: dict) -> list[dict]:
    """New BUY-or-SELL trades for one wallet since its cursor, via the tested paged
    `/trades?user=` fetcher (not watch.py's fixed 25-row page, which a fast sports
    wallet can outrun between polls). `since_ts - overlap` so trades sharing the
    cursor's exact timestamp are re-served and de-duplicated by
    `watch.new_trades_since` rather than lost."""
    from src.backfill import fetch_user_trades

    since = int(cursor["max_timestamp"]) - DEFAULT_TRADES_FETCH_OVERLAP_S
    return [normalize_trade(t) for t in fetch_user_trades(session, cfg, wallet, since)]


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

SIGNAL_COLUMNS = [
    "signal_id", "wallet", "cohorts", "market_id", "token_id", "outcome", "side",
    "slug", "question", "wallet_entry_price", "wallet_size", "wallet_ts",
    "detect_ts", "status", "best_ask", "best_ask_size", "best_bid", "best_bid_size",
    "tick_size", "min_order_size", "book_json", "market_end_ts", "game_start_ts",
    "placebo_horizon_ts", "placebo_horizon_source", "taker_base_fee_raw",
    "maker_base_fee_raw",
]

ORDER_COLUMNS = [
    "order_id", "signal_id", "arm", "cohorts", "wallet", "market_id", "token_id", "slug",
    "question", "wallet_entry_price", "detect_ts", "budget_usd", "limit_price",
    "trigger_ts", "status", "shares", "cost_usd", "avg_fill_price", "fee_units",
    "is_taker", "first_fill_ts", "last_update_ts", "no_fill_reason",
]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"cursors": {}, "last_poll_ts": None, "polls": 0}


def load_ledger() -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = pd.read_parquet(SIGNALS_PATH) if SIGNALS_PATH.exists() else _empty(SIGNAL_COLUMNS)
    orders = pd.read_parquet(ORDERS_PATH) if ORDERS_PATH.exists() else _empty(ORDER_COLUMNS)
    return signals, orders


def save_ledger(signals: pd.DataFrame, orders: pd.DataFrame, state: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(signals, SIGNALS_PATH, compression="gzip")
    atomic_to_parquet(orders, ORDERS_PATH, compression="gzip")
    atomic_write_json(state, STATE_PATH)


def n_live_orders(orders: pd.DataFrame) -> int:
    if orders.empty:
        return 0
    return int(orders["status"].isin(LIVE_STATUSES).sum())


# ---------------------------------------------------------------------------
# ORDER CONSTRUCTION + PROGRESSION
# ---------------------------------------------------------------------------

def _blank_order(signal: dict, arm: str, budget: float) -> dict:
    return {
        "order_id": f"{signal['signal_id']}:{arm}",
        "signal_id": signal["signal_id"], "arm": arm,
        "cohorts": signal.get("cohorts", ""), "wallet": signal["wallet"],
        "market_id": signal["market_id"], "token_id": signal["token_id"],
        "slug": signal.get("slug"), "question": signal.get("question"),
        "wallet_entry_price": signal["wallet_entry_price"],
        "detect_ts": signal["detect_ts"], "budget_usd": budget,
        "limit_price": float("nan"), "trigger_ts": None, "status": ST_OPEN,
        "shares": 0.0, "cost_usd": 0.0, "avg_fill_price": float("nan"),
        "fee_units": 0.0, "is_taker": False, "first_fill_ts": None,
        "last_update_ts": signal["detect_ts"], "no_fill_reason": None,
    }


def _apply_fill(order: dict, fill: dict, now_ts: int) -> dict:
    """Accumulate a (possibly partial) fill onto an order. Average fill price is
    notional-weighted across every partial, so a fill that completed over three
    polls at three prices reports what it actually cost."""
    if fill["shares"] <= 0:
        return order
    order = dict(order)
    order["shares"] = float(order["shares"]) + fill["shares"]
    order["cost_usd"] = float(order["cost_usd"]) + fill["cost_usd"]
    order["fee_units"] = float(order["fee_units"]) + fill["fee_units"]
    order["avg_fill_price"] = order["cost_usd"] / order["shares"]
    order["is_taker"] = bool(order["is_taker"]) or bool(fill.get("is_taker"))
    if order["first_fill_ts"] is None:
        order["first_fill_ts"] = int(now_ts)
    order["last_update_ts"] = int(now_ts)
    order["status"] = ST_FILLED
    return order


def _rest_or_close(order: dict) -> dict:
    """Decide whether a RESTING limit order is done or still working.

    The one-shot taker arms (`market_chase`, `placebo_random`) are terminal the
    moment they act; only `limit_noChase` keeps working, and it keeps working until
    its budget is spent or its market settles. Getting this wrong in the generous
    direction — closing a partial as if it were complete — would silently credit the
    arm with shares it never obtained."""
    out = dict(order)
    if float(out["cost_usd"]) >= float(out["budget_usd"]) - 1e-9:
        out["status"] = ST_FILLED
    elif float(out["shares"]) > 0:
        out["status"] = ST_PARTIAL
    else:
        out["status"] = ST_OPEN
    return out


def open_orders_for_signal(signal: dict, book: dict, rule: dict, now_ts: int) -> list[dict]:
    """Create the three paper orders for one detected signal and take whatever fills
    are available at detection. Every arm sees the same signal and the same book."""
    budget = capped_budget(rule["notional_per_signal_usd"],
                           rule["max_notional_per_position_usd"])
    enforce = rule["enforce_min_order_size"]
    asks = book.get("asks", [])
    out = []

    # --- ARM 1: resting limit at or below the wallet's own price ---
    limit = round_down_to_tick(float(signal["wallet_entry_price"]), book.get("tick_size"))
    o = _blank_order(signal, "limit_noChase", budget)
    o["limit_price"] = limit
    if asks and asks[0][0] <= limit + 1e-12:
        # already marketable: not a chase (we pay no more than they did), so take it
        fill = fill_from_book(book, budget, mode="taker", limit_price=limit,
                              enforce_min_size=enforce)
        o = _apply_fill(o, fill, now_ts)
    o = _rest_or_close(o)
    out.append(o)

    # --- ARM 2: taker at the best ask, right now ---
    o = _blank_order(signal, "market_chase", budget)
    fill = fill_from_book(book, budget, mode="taker", enforce_min_size=enforce)
    if fill["shares"] > 0:
        o = _apply_fill(o, fill, now_ts)
    else:
        o["status"] = ST_NO_FILL
        o["no_fill_reason"] = ("min_order_size" if fill.get("rejected_min_size")
                               else "no_ask_size_at_detection")
    out.append(o)

    # --- ARM 3: the control — same market and side, the wallet's timing discarded ---
    o = _blank_order(signal, "placebo_random", budget)
    o["status"] = ST_PENDING
    o["trigger_ts"] = placebo_trigger_ts(
        signal["signal_id"], int(signal["detect_ts"]),
        int(signal["placebo_horizon_ts"]), rule["placebo_seed"])
    out.append(o)
    return out


def advance_order(order: dict, book: dict, rule: dict, now_ts: int) -> dict:
    """Re-check ONE live order against a freshly fetched book. Pure; no network."""
    status = order["status"]
    if status not in LIVE_STATUSES:
        return order
    remaining = float(order["budget_usd"]) - float(order["cost_usd"])
    if remaining <= 1e-9:
        return order
    enforce = rule["enforce_min_order_size"]

    if order["arm"] == "limit_noChase":
        limit = float(order["limit_price"])
        if not book.get("asks") or book["asks"][0][0] > limit + 1e-12:
            return order          # nobody offered at our price this poll
        fill = fill_from_book(book, remaining, mode="maker", limit_price=limit,
                              enforce_min_size=enforce)
        return _rest_or_close(_apply_fill(order, fill, now_ts))

    if order["arm"] == "placebo_random":
        if order["trigger_ts"] is None or now_ts < int(order["trigger_ts"]):
            return order
        fill = fill_from_book(book, remaining, mode="taker", enforce_min_size=enforce)
        if fill["shares"] > 0:
            return _apply_fill(order, fill, now_ts)
        out = dict(order)
        out["status"] = ST_NO_FILL
        out["no_fill_reason"] = ("min_order_size" if fill.get("rejected_min_size")
                                 else "no_ask_size_at_trigger")
        out["last_update_ts"] = int(now_ts)
        return out

    return order


def terminate_settled(orders: pd.DataFrame, settled_markets: set[str],
                      now_ts: int) -> pd.DataFrame:
    """Turn still-live orders in settled markets into terminal NO-FILLS.

    This is the step that stops the ledger from quietly forgetting the limits that
    never got hit — the single biggest way paper trading lies."""
    if orders.empty:
        return orders
    out = orders.copy()
    live = out["status"].isin(LIVE_STATUSES) & out["market_id"].isin(settled_markets)
    if not live.any():
        return out
    shares = out["shares"].astype(float).fillna(0.0) if "shares" in out else 0.0
    # A partially filled order keeps the shares it actually got; only the ones that
    # got nothing become no-fills. Both are terminal once the market has settled.
    empty = live & (shares <= 0)
    partial = live & (shares > 0)
    reason = np.where(out.loc[empty, "arm"] == "placebo_random",
                      "settled_before_trigger", "never_filled_before_settlement")
    out.loc[empty, "no_fill_reason"] = reason
    out.loc[empty, "status"] = ST_NO_FILL
    out.loc[partial, "status"] = ST_FILLED
    out.loc[live, "last_update_ts"] = int(now_ts)
    return out


# ---------------------------------------------------------------------------
# THE POLL CYCLE
# ---------------------------------------------------------------------------

def poll_once(session, cfg: dict, rule: dict, watchlist: pd.DataFrame,
              signals: pd.DataFrame, orders: pd.DataFrame, state: dict, now_ts: int,
              fetch_positions=fetch_new_positions, get_book=fetch_book,
              get_market=fetch_market_meta) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """One full cycle: detect new signals, open their orders, then re-check every
    still-live order against a fresh book.

    Pure apart from the three injectable fetchers, so tests drive it with canned
    books instead of the network. Never mutates its inputs."""
    settings = paper_settings(cfg)
    cursors = dict(state.get("cursors", {}))
    new_signals: list[dict] = []
    new_orders: list[dict] = []
    stats = {"polled_wallets": 0, "new_signals": 0, "skipped_cap": 0,
             "skipped_book_error": 0, "no_book": 0, "orders_opened": 0,
             "fills": 0, "errors": 0}

    live_count = n_live_orders(orders)
    known = set(signals["signal_id"]) if len(signals) else set()

    # ---- 1. detect new positions -----------------------------------------
    for _, row in watchlist.iterrows():
        wallet = row["wallet"]
        cursor = cursors.get(wallet, default_cursor(now_ts))
        try:
            trades = fetch_positions(session, cfg, wallet, cursor)
        except Exception as exc:  # noqa: BLE001 — one wallet must not abort the cycle
            print(f"[paper] WARNING: {wallet[:12]} trade fetch failed: {exc}")
            stats["errors"] += 1
            continue
        stats["polled_wallets"] += 1
        fresh = new_trades_since(trades, cursor)
        cursors[wallet] = advance_cursor(cursor, trades)

        for trade in fresh:
            if trade["side"] != "BUY":
                continue
            sid = signal_id_for(trade)
            if sid in known:
                continue
            known.add(sid)
            stats["new_signals"] += 1

            sig = {
                "signal_id": sid, "wallet": wallet,
                "cohorts": row.get("cohorts", ""),
                "market_id": trade["market_id"], "token_id": trade["token_id"],
                "outcome": trade.get("outcome"), "side": trade["side"],
                "slug": trade.get("slug"), "question": trade.get("question"),
                "wallet_entry_price": float(trade["entry_price"]),
                "wallet_size": float(trade["size"]), "wallet_ts": int(trade["timestamp"]),
                "detect_ts": int(now_ts), "status": "accepted",
                "best_ask": float("nan"), "best_ask_size": 0.0,
                "best_bid": float("nan"), "best_bid_size": 0.0,
                "tick_size": None, "min_order_size": 0.0, "book_json": None,
                "market_end_ts": None, "game_start_ts": None,
                "placebo_horizon_ts": None, "placebo_horizon_source": None,
                "taker_base_fee_raw": None, "maker_base_fee_raw": None,
            }

            # SEATBELT: at the open-position cap we record the signal and open
            # nothing. It still counts in n_signals, so the cap costs the strategy.
            if live_count >= settings["max_open_positions"]:
                sig["status"] = "skipped_cap"
                new_signals.append(sig)
                stats["skipped_cap"] += 1
                continue

            # A book fetch that ERRORS is an observability failure and we open
            # nothing — we genuinely do not know what was tradeable. A book that
            # comes back EMPTY or 404 is information: there was nothing to buy. The
            # arms still open, so that shows up as a no-fill rather than vanishing.
            try:
                raw_book = get_book(session, cfg, trade["token_id"])
            except Exception as exc:  # noqa: BLE001
                print(f"[paper] WARNING: book fetch failed for {sid}: {exc}")
                sig["status"] = "skipped_book_error"
                new_signals.append(sig)
                stats["errors"] += 1
                stats["skipped_book_error"] += 1
                continue

            book = parse_book(raw_book)
            if not book["asks"]:
                sig["status"] = "no_book_at_detection"
                stats["no_book"] += 1
            ba, bas = best_ask(book)
            bb, bbs = best_bid(book)
            sig.update({
                "best_ask": ba, "best_ask_size": bas, "best_bid": bb,
                "best_bid_size": bbs, "tick_size": book["tick_size"],
                "min_order_size": book["min_order_size"],
                "book_json": json.dumps({"asks": book["asks"][:BOOK_LEVELS_RECORDED],
                                         "bids": book["bids"][:BOOK_LEVELS_RECORDED],
                                         "book_ts": book["book_ts"]}),
            })
            try:
                meta = get_market(session, cfg, trade["market_id"])
            except Exception as exc:  # noqa: BLE001
                print(f"[paper] WARNING: market meta failed for {sid}: {exc}")
                meta, stats["errors"] = {}, stats["errors"] + 1
            sig.update({k: meta.get(k) for k in
                        ("market_end_ts", "game_start_ts", "taker_base_fee_raw",
                         "maker_base_fee_raw") if k in meta})
            # the CLOB's market metadata backstops the book's tick / min size, which
            # matters precisely when the book came back empty
            if book["tick_size"] is None and meta.get("min_tick_size_meta"):
                book["tick_size"] = float(meta["min_tick_size_meta"])
                sig["tick_size"] = book["tick_size"]
            if not book["min_order_size"] and meta.get("min_order_size_meta"):
                book["min_order_size"] = float(meta["min_order_size_meta"])
                sig["min_order_size"] = book["min_order_size"]
            horizon, hsource = placebo_horizon(
                int(now_ts), meta.get("market_end_ts"), meta.get("game_start_ts"),
                rule["placebo_fallback_horizon_hours"])
            sig["placebo_horizon_ts"] = horizon
            sig["placebo_horizon_source"] = hsource

            made = open_orders_for_signal(sig, book, rule, now_ts)
            new_orders.extend(made)
            new_signals.append(sig)
            stats["orders_opened"] += len(made)
            stats["fills"] += sum(1 for o in made if o["status"] == ST_FILLED)
            live_count += sum(1 for o in made if o["status"] in LIVE_STATUSES)

    # ---- 2. re-check every still-live order against a fresh book ----------
    orders_out = pd.concat([orders, pd.DataFrame(new_orders, columns=ORDER_COLUMNS)],
                           ignore_index=True) if new_orders else orders.copy()
    if len(orders_out):
        live_mask = orders_out["status"].isin(LIVE_STATUSES)
        # one book fetch per distinct token, shared by every live order on it
        books: dict[str, dict] = {}
        for idx in orders_out.index[live_mask]:
            order = orders_out.loc[idx].to_dict()
            if order["arm"] == "placebo_random" and (
                    order["trigger_ts"] is None or now_ts < int(order["trigger_ts"])):
                continue                      # not due; don't spend a request
            token = order["token_id"]
            if token not in books:
                try:
                    books[token] = parse_book(get_book(session, cfg, token))
                except Exception as exc:  # noqa: BLE001
                    print(f"[paper] WARNING: book re-check failed for {token[:12]}: {exc}")
                    books[token] = parse_book(None)
                    stats["errors"] += 1
            before = float(order["shares"])
            updated = advance_order(order, books[token], rule, now_ts)
            if float(updated["shares"]) > before:
                stats["fills"] += 1
            for col, val in updated.items():
                orders_out.at[idx, col] = val

    signals_out = (pd.concat([signals, pd.DataFrame(new_signals, columns=SIGNAL_COLUMNS)],
                             ignore_index=True) if new_signals else signals.copy())
    new_state = dict(state)
    new_state["cursors"] = cursors
    new_state["last_poll_ts"] = int(now_ts)
    new_state["polls"] = int(state.get("polls", 0)) + 1
    return signals_out, orders_out, new_state, stats


# ---------------------------------------------------------------------------
# FREEZE — the pre-registration
# ---------------------------------------------------------------------------

def build_watchlist(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The pre-registered watchlist, read FRESH at freeze time.

    Two disjoint sources, tagged so the scoreboard can be cut by cohort later
    without re-freezing anything:
      * `s2_twelve` — the 12 sports wallets of the sig2c freeze, with `s2_core` and
        `s2_slow` membership flagged. Sports resolves in DAYS, so this is where a
        fast read comes from.
      * Project 1's certified set — `edge_persisted == True` in ranked_wallets.parquet,
        whatever it says at the moment of freezing.
    Returns (watchlist, funnel) where `funnel` is the counts actually observed, so
    the selection is reproducible against the recorded git commit."""
    cfg = cfg or load_config()
    funnel: dict = {}

    sig2c_path = PROCESSED_DIR / "sports_sig2c_freeze_manifest.json"
    sports: dict[str, set] = {"s2_twelve": set(), "s2_core": set(), "s2_slow": set()}
    if sig2c_path.exists():
        with open(sig2c_path) as f:
            m = json.load(f)
        for tier in sports:
            sports[tier] = set(m.get("tiers", {}).get(tier, {}).get("wallets", []))
        funnel["sports_freeze_commit"] = m.get("git_commit")
        funnel["sports_freeze_utc"] = m.get("freeze_utc")
    funnel.update({f"n_{t}": len(v) for t, v in sports.items()})

    p1_cols = ["wallet", "rank", "edge_persisted", "copyable", "out_of_sample_edge",
               "out_of_sample_residual_edge", "out_of_sample_n", "out_of_sample_markets",
               "copy_window"]
    if RANKED_WALLETS_PATH.exists():
        ranked = pd.read_parquet(RANKED_WALLETS_PATH, columns=p1_cols)
    else:
        ranked = pd.DataFrame(columns=p1_cols)
    funnel["p1_wallets_ranked"] = int(len(ranked))
    certified = ranked[ranked["edge_persisted"].fillna(False)].copy() if len(ranked) else ranked
    funnel["p1_certified"] = int(len(certified))
    funnel["p1_certified_and_copyable"] = int(certified["copyable"].fillna(False).sum()) if len(certified) else 0
    # Read the SAME alpha the certification actually used — Project 1's scoped
    # `scoring.project1.oos_significance_alpha` (tightened to 0.005 on 2026-07-26,
    # cutting the certified set 41 -> 26), not the global key. Recording the wrong
    # one would make the funnel unreproducible.
    from src.validate import certification_alpha

    funnel["p1_oos_significance_alpha"] = float(certification_alpha(cfg.get("scoring", {}) or {}))
    funnel["p1_min_skill_edge"] = float(cfg.get("scoring", {}).get("min_skill_edge", 0.02))

    wallets = sorted(set(certified["wallet"]) | sports["s2_twelve"])
    rows = []
    ranked_idx = ranked.set_index("wallet") if len(ranked) else pd.DataFrame()
    for w in wallets:
        tags = [t for t in ("s2_core", "s2_twelve", "s2_slow") if w in sports[t]]
        p1 = w in set(certified["wallet"]) if len(certified) else False
        if p1:
            tags.append("p1_certified")
        r = ranked_idx.loc[w] if len(ranked_idx) and w in ranked_idx.index else None

        def _num(col, cast=float):
            """Sports wallets need not appear in Project 1's ranking at all, and a
            ranked row can carry NaN where a metric was undefined. Both are recorded
            as null rather than coerced."""
            if r is None or col not in r or pd.isna(r[col]):
                return None
            return cast(r[col])

        rows.append({
            "wallet": w, "cohorts": "|".join(tags),
            "in_s2_twelve": w in sports["s2_twelve"],
            "in_s2_core": w in sports["s2_core"],
            "in_s2_slow": w in sports["s2_slow"],
            "in_p1_certified": bool(p1),
            "p1_rank": _num("rank", int),
            "p1_copyable": _num("copyable", bool),
            "p1_out_of_sample_residual_edge": _num("out_of_sample_residual_edge"),
            "p1_out_of_sample_n": _num("out_of_sample_n", int),
            "p1_out_of_sample_markets": _num("out_of_sample_markets", int),
        })
    watchlist = pd.DataFrame(rows)
    funnel["watchlist_size"] = int(len(watchlist))
    funnel["watchlist_sports_only"] = int((~watchlist["in_p1_certified"]).sum()) if len(watchlist) else 0
    funnel["watchlist_p1_only"] = int((~watchlist["in_s2_twelve"]).sum()) if len(watchlist) else 0
    funnel["watchlist_both"] = int((watchlist["in_p1_certified"] & watchlist["in_s2_twelve"]).sum()) if len(watchlist) else 0
    return watchlist, funnel


def forward_observations() -> int:
    """How many signals the paper trader has already recorded. A freeze is only
    pre-registration while this is ZERO."""
    if not SIGNALS_PATH.exists():
        return 0
    try:
        return int(len(pd.read_parquet(SIGNALS_PATH, columns=["signal_id"])))
    except Exception:  # noqa: BLE001
        return -1


def freeze(cfg: dict | None = None, freeze_ts: int | None = None, force: bool = False,
           amend_reason: str | None = None) -> dict:
    """Write the pre-registered manifest + watchlist.

    Refuses to overwrite an existing freeze without `--force`, and records every
    re-freeze in an `amendments` log with the number of forward observations that
    existed at the time — the audit trail that makes "this was still
    pre-registration" checkable rather than asserted."""
    from src.sports_events import SPORTS_EVENT_VERSION

    cfg = cfg or load_config()
    prior = None
    if FREEZE_MANIFEST_PATH.exists():
        with open(FREEZE_MANIFEST_PATH) as f:
            prior = json.load(f)
        if not force:
            raise SystemExit(
                f"{FREEZE_MANIFEST_PATH} already exists (frozen "
                f"{prior.get('freeze_utc')}). The rule is frozen BEFORE the data — "
                f"pass --force with a --reason if this is a legitimate "
                f"pre-registration amendment.")

    watchlist, funnel = build_watchlist(cfg)
    if watchlist.empty:
        raise RuntimeError(
            "watchlist is empty — nothing to pre-register. Check that "
            "data/processed/ranked_wallets.parquet and "
            "sports_sig2c_freeze_manifest.json exist.")

    settings = paper_settings(cfg)
    ts = int(freeze_ts if freeze_ts is not None
             else (prior["freeze_ts"] if prior else time.time()))
    obs = forward_observations()
    amendments = list(prior.get("amendments", [])) if prior else []
    if prior is not None:
        amendments.append({
            "amended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "amended_commit": _git_commit(),
            "reason": amend_reason or "unspecified",
            "forward_observations_at_amendment": obs,
            "legitimate_pre_registration": obs == 0,
            "prior_entry_rule": prior.get("entry_rule"),
            "prior_scoring_rule": prior.get("scoring_rule"),
            "prior_watchlist_size": prior.get("funnel", {}).get("watchlist_size"),
        })

    manifest = {
        "arm": "paper_trader",
        "stage": "execution roadmap stage 1 — forward paper test",
        "freeze_ts": ts,
        "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "git_commit": _git_commit(),
        "sports_event_version": SPORTS_EVENT_VERSION,
        "arms": list(ARMS),
        "headline_arm": HEADLINE_ARM,
        "control_arm": CONTROL_ARM,
        "arm_roles": {
            "limit_noChase": "THE THESIS — take the selection, refuse the chase. Headline.",
            "market_chase": ("SANITY CHECK on the machinery. Already shown dead "
                             "retrospectively; strong profit here means a bug, not an edge."),
            "placebo_random": ("THE CONTROL — same market and side, wallet timing "
                               "discarded. If the thesis arm does not beat this, the "
                               "wallets are not the signal."),
        },
        "entry_rule": ENTRY_RULE,
        "scoring_rule": SCORING_RULE,
        "stake": {
            "notional_per_signal_usd": settings["notional_per_signal_usd"],
            "sizing": "fixed notional per signal per arm; never scaled by anything",
        },
        "costs": {
            "taker_fee_model": "shares * k * price * (1 - price)",
            "taker_fee_k": settings["taker_fee_k"],
            "taker_fee_k_stress": settings["taker_fee_k_stress"],
            "maker_fee": 0.0,
            "tick_source": "the token's own tick_size from GET /book (NOT a constant)",
            "slippage": "walk the live book level by level; no assumed top-of-book size",
            "fee_k_caveat": (
                "k is NOT pinned by anything this repo has verified. Gamma reports "
                "fee: None on live sports lines (docs/project3_sports_metric_b.md §5) "
                "and the CLOB advertises taker_base_fee/maker_base_fee in unexplained "
                "units, captured per signal for later calibration. The scoreboard "
                "therefore reports P&L at the frozen k AND at the stressed k."),
        },
        "seatbelts": {
            "paper_only": True, "keys": "none", "signing": "none",
            "order_placement": "none — public unauthenticated GETs only",
            "kill_switch_path": kill_switch_label(cfg),
            "max_notional_per_position_usd": settings["max_notional_per_position_usd"],
            "max_open_positions": settings["max_open_positions"],
            "dry_run_default": True,
        },
        "placebo": {
            "seed": f"paper_trader_placebo_v1:{ts}",
            "draw": ("u = SHA256(seed || signal_id)[:8] / 2**64, "
                     "trigger = detect + u * (horizon - detect)"),
            "horizon": ("first of game_start_time / end_date_iso that is in the "
                        "FUTURE at detection, else the fallback below. end_date_iso "
                        "alone is unusable: src/market_meta.py documents it as the "
                        "midnight FLOOR of the scheduled end day, and a live probe of "
                        "99 detected signals found 97 already past it."),
            "fallback_horizon_hours": settings["placebo_fallback_horizon_hours"],
        },
        "watchlist_sources": {
            "sports": ("s2_twelve from data/processed/sports_sig2c_freeze_manifest.json "
                       "(s2_core and s2_slow membership flagged)"),
            "project1": "edge_persisted == True in data/processed/ranked_wallets.parquet",
        },
        "funnel": funnel,
        "forward_observations_at_freeze": obs,
        "amendments": amendments,
        "known_limits": [
            "POLLING IS DISCRETE. Fills are decided from book snapshots taken at the "
            "poll cadence, so a transient offer between polls is missed (under-counts "
            "fills) while a resting bid could in reality be hit by flow no snapshot "
            "shows (over- or under-counts, direction unknown). This is the largest "
            "modelling error in the module and it is not signed.",
            "MARKET IMPACT IS NOT MODELLED. A real $100 order joins the queue and can "
            "move a thin book; here it consumes displayed size without disturbing it. "
            "At $100 on books showing thousands of shares this is small, but it is a "
            "one-directional optimism.",
            "QUEUE POSITION IS NOT MODELLED. A resting maker order is assumed filled "
            "whenever size appears at or below its price. Real queue priority would "
            "fill LESS often, so the thesis arm's fill rate here is an upper bound.",
            "THE FEE CONSTANT k IS UNPINNED (see costs.fee_k_caveat). Read the stressed "
            "column before quoting a net number.",
            "SPORTS RESOLVES IN DAYS; the Project 1 cohort's markets can take weeks or "
            "months. The first readable scoreboard will be almost entirely sports.",
            "SMALL SAMPLES ARE EXPECTED AND THE SCOREBOARD SAYS SO. effective_events, "
            "not signal counts, is what a reader should size the evidence by.",
            "The watchlist inherits every selection caveat of the cohorts it is built "
            "from. Freezing it does not make those cohorts more likely to be real; it "
            "only stops this test from being tuned to them.",
        ],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(watchlist, WATCHLIST_PATH, compression="gzip")
    atomic_write_json(manifest, FREEZE_MANIFEST_PATH)
    return manifest


def load_freeze() -> tuple[pd.DataFrame, dict]:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{FREEZE_MANIFEST_PATH} not found — run "
            f"`python -m src.paper_trader freeze` first.")
    with open(FREEZE_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return pd.read_parquet(WATCHLIST_PATH), manifest


def rule_from_manifest(manifest: dict, cfg: dict) -> dict:
    """The economic rule comes from the FROZEN manifest (stake, fee, placebo seed);
    only the operational caps come from live config, and only so they can be
    tightened. This is what stops the rule drifting after the data arrives."""
    settings = paper_settings(cfg)
    return {
        "notional_per_signal_usd": float(manifest["stake"]["notional_per_signal_usd"]),
        "max_notional_per_position_usd": min(
            float(manifest["seatbelts"]["max_notional_per_position_usd"]),
            settings["max_notional_per_position_usd"]),
        "max_open_positions": min(int(manifest["seatbelts"]["max_open_positions"]),
                                  settings["max_open_positions"]),
        "taker_fee_k": float(manifest["costs"]["taker_fee_k"]),
        "taker_fee_k_stress": float(manifest["costs"]["taker_fee_k_stress"]),
        "placebo_seed": manifest["placebo"]["seed"],
        "placebo_fallback_horizon_hours": float(
            manifest["placebo"]["fallback_horizon_hours"]),
        "enforce_min_order_size": settings["enforce_min_order_size"],
    }


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

def run(cfg: dict | None = None, passes: int = 1, dry_run: bool = True) -> dict:
    """Poll the frozen watchlist and progress the paper book.

    Idempotent and resumable: cursors, signals and orders are persisted after every
    pass, so a restart resumes rather than re-trades. Never crash-loops — a failing
    wallet or book is logged and the cycle continues."""
    cfg = cfg or load_config()
    watchlist, manifest = load_freeze()
    rule = rule_from_manifest(manifest, cfg)
    settings = paper_settings(cfg)

    if kill_switch_engaged(cfg):
        print(f"[paper] KILL SWITCH ENGAGED ({kill_switch_path(cfg)}) — halting "
              f"before any fetch or write.")
        return {"halted": True, "reason": "kill_switch"}

    session = make_session(max_retries=cfg["ingest"]["max_retries"],
                           backoff=cfg["ingest"]["retry_backoff_sec"])
    signals, orders = load_ledger()
    state = load_state()
    print(f"[paper] freeze {manifest['freeze_utc']} ({manifest['git_commit'][:8]}) — "
          f"{len(watchlist)} wallets, {len(signals)} signals, "
          f"{n_live_orders(orders)} live orders. "
          f"{'DRY RUN (nothing will be written)' if dry_run else 'COMMIT'}")

    total = {}
    for i in range(max(1, passes)):
        if kill_switch_engaged(cfg):
            print("[paper] KILL SWITCH ENGAGED mid-run — stopping.")
            break
        try:
            signals, orders, state, stats = poll_once(
                session, cfg, rule, watchlist, signals, orders, state, int(time.time()))
        except Exception as exc:  # noqa: BLE001 — never crash-loop
            print(f"[paper] WARNING: poll cycle failed ({exc}); will retry next pass")
            time.sleep(min(settings["poll_seconds"], 60))
            continue
        total = stats
        print(f"[paper] pass {i+1}/{passes}: {stats}")
        if not dry_run:
            save_ledger(signals, orders, state)
        if i < passes - 1:
            time.sleep(settings["poll_seconds"])

    if dry_run:
        print("[paper] DRY RUN — no ledger written. Re-run with --commit to persist.")
    return {"halted": False, "signals": len(signals), "orders": len(orders),
            "live": n_live_orders(orders), "stats": total}


# ---------------------------------------------------------------------------
# SCORE
# ---------------------------------------------------------------------------

def effective_events(codes) -> float:
    """Kish effective cluster count: (Sum n)^2 / Sum n^2. Equals the true count when
    signals spread evenly across events and collapses toward 1 when one event
    dominates — which is exactly the artifact that killed Project 2 §1.5."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return 0.0
    counts = np.bincount(np.unique(codes, return_inverse=True)[1]).astype(float)
    return float(counts.sum() ** 2 / (counts**2).sum())


def event_boot(values: np.ndarray, event_codes: np.ndarray, n_boot: int = 2000,
               seed: int = 12345) -> dict:
    """Event-weighted mean with a cluster bootstrap that resamples EVENTS.

    The statistic is the mean over events of the within-event mean, so a wallet that
    fires ten signals into one game contributes one observation, not ten. Undefined
    below two events, which is honest rather than a zero."""
    values = np.asarray(values, dtype=float)
    codes = np.asarray(event_codes)
    ok = ~np.isnan(values)
    values, codes = values[ok], codes[ok]
    if values.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_one_sided": np.nan, "n_events": 0, "effective_events": 0.0}
    g = np.unique(codes, return_inverse=True)[1]
    G = int(g.max()) + 1
    sums = np.bincount(g, weights=values, minlength=G)
    cnts = np.bincount(g, minlength=G).astype(float)
    means = sums / cnts
    point = float(means.mean())
    eff = effective_events(codes)
    if G < 2:
        return {"mean": point, "ci_low": np.nan, "ci_high": np.nan,
                "p_one_sided": np.nan, "n_events": G, "effective_events": eff}
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, G, size=(n_boot, G))].mean(axis=1)
    return {
        "mean": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_one_sided": float((np.sum(draws <= 0) + 1) / (n_boot + 1)),
        "n_events": G,
        "effective_events": eff,
    }


def assign_event_units(frame: pd.DataFrame) -> pd.Series:
    """The unit of independent resolution: the sports resolution event where the
    sports labeller recognises the market, the market_id everywhere else."""
    from src.slow_market import derive_categories
    from src.sports_events import assign_events, is_sports_market

    if frame.empty:
        return pd.Series(dtype=object)
    df = frame.copy()
    cats = derive_categories(df["slug"], df["question"])
    sports = pd.Series([is_sports_market(s, q, c) for s, q, c in
                        zip(df["slug"], df["question"], cats)], index=df.index)
    df = assign_events(df)
    return pd.Series(np.where(sports, df["event"], "market:" + df["market_id"].astype(str)),
                     index=df.index)


def _arm_row(sub: pd.DataFrame, arm: str, stratum: str, n_boot: int) -> dict:
    """One scoreboard row. `n_signals`, `n_filled`, `fill_rate`, `n_events` and
    `effective_events` sit beside every dollar figure on purpose — a dollar figure
    without them has burned this repo before."""
    boot = event_boot(sub["pnl_usd"].to_numpy(), sub["event_unit"].to_numpy(),
                      n_boot=n_boot)
    notional = float(sub["cost_usd"].sum())
    pnl = float(sub["pnl_usd"].sum())
    return {
        "arm": arm, "stratum": stratum,
        "n_signals": int(len(sub)),
        "n_filled": int(sub["filled"].sum()),
        "fill_rate": float(sub["filled"].mean()),
        "n_events": boot["n_events"],
        "effective_events": boot["effective_events"],
        "notional_filled_usd": notional,
        "total_pnl_usd": pnl,
        "total_pnl_usd_fee_stress": float(sub["pnl_usd_fee_stress"].sum()),
        "return_on_filled_notional": (pnl / notional if notional > 0 else float("nan")),
        "pnl_per_signal_usd": float(sub["pnl_usd"].mean()),
        "pnl_per_signal_usd_event_wtd": boot["mean"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "p_one_sided": boot["p_one_sided"],
        "mean_fill_price": (float(sub.loc[sub["filled"], "avg_fill_price"].mean())
                            if sub["filled"].any() else float("nan")),
        "total_fee_usd": float(sub["fee_usd"].sum()),
    }


def score_orders(orders: pd.DataFrame, signals: pd.DataFrame, resolutions: pd.DataFrame,
                 rule: dict, n_boot: int = 2000) -> dict:
    """Turn the paper ledger into dollars, per arm, at the event unit.

    Every arm sees the same signals, so no-fills cannot be hidden: an unfilled order
    scores exactly $0 on $0 of notional and still counts in n_signals."""
    if orders.empty:
        return {"board": pd.DataFrame(), "contrast": None, "n_unresolved": 0,
                "note": "no paper orders recorded yet"}

    res = resolutions[["market_id", "token_id", "resolved", "resolved_value"]].copy() \
        if len(resolutions) else pd.DataFrame(
            columns=["market_id", "token_id", "resolved", "resolved_value"])
    df = orders.merge(res, on=["market_id", "token_id"], how="left")
    df["resolved"] = df["resolved"].fillna(False).astype(bool)

    n_unresolved_signals = int(df.loc[~df["resolved"], "signal_id"].nunique())
    scored = df[df["resolved"]].copy()
    if scored.empty:
        return {"board": pd.DataFrame(), "contrast": None,
                "n_unresolved": n_unresolved_signals,
                "note": "signals captured, none of their markets resolved yet"}

    scored["shares"] = scored["shares"].astype(float).fillna(0.0)
    scored["cost_usd"] = scored["cost_usd"].astype(float).fillna(0.0)
    scored["fee_units"] = scored["fee_units"].astype(float).fillna(0.0)
    scored["resolved_value"] = scored["resolved_value"].astype(float)
    taker = scored["is_taker"].fillna(False).astype(bool)
    scored["fee_usd"] = np.where(taker, rule["taker_fee_k"] * scored["fee_units"], 0.0)
    scored["fee_usd_stress"] = np.where(
        taker, rule["taker_fee_k_stress"] * scored["fee_units"], 0.0)
    payout = scored["shares"] * scored["resolved_value"]
    scored["pnl_usd"] = payout - scored["cost_usd"] - scored["fee_usd"]
    scored["pnl_usd_fee_stress"] = payout - scored["cost_usd"] - scored["fee_usd_stress"]
    scored["filled"] = scored["shares"] > 0

    meta = signals[["signal_id", "slug", "question", "market_id"]].drop_duplicates("signal_id") \
        if len(signals) else pd.DataFrame(columns=["signal_id", "slug", "question", "market_id"])
    if len(meta):
        scored = scored.drop(columns=["slug", "question"], errors="ignore").merge(
            meta, on=["signal_id", "market_id"], how="left")
    scored["event_unit"] = assign_event_units(scored)

    # STRATA, pre-registered: the whole book, and each source cohort separately. The
    # two cohorts are not comparable populations — the sports wallets trade games
    # that settle in days, Project 1's are dominated by micro-crypto HFT that fires
    # orders of magnitude more signals — so a pooled number is mostly the loudest
    # cohort. `all` stays the headline; the cuts stop it being misread.
    cohorts = scored["cohorts"].fillna("") if "cohorts" in scored else pd.Series("", index=scored.index)
    strata = {"all": pd.Series(True, index=scored.index)}
    for tag in ("s2_twelve", "p1_certified"):
        mask = cohorts.str.contains(tag, regex=False)
        if mask.any():
            strata[tag] = mask

    rows, per_event = [], {}
    for stratum, mask in strata.items():
        for arm in ARMS:
            sub = scored[mask & (scored["arm"] == arm)]
            if sub.empty:
                continue
            rows.append(_arm_row(sub, arm, stratum, n_boot))
            if stratum == "all":
                per_event[arm] = sub.groupby("event_unit")["pnl_usd"].mean()
    board = pd.DataFrame(rows)

    contrast = None
    if HEADLINE_ARM in per_event and CONTROL_ARM in per_event:
        a, b = per_event[HEADLINE_ARM], per_event[CONTROL_ARM]
        shared = a.index.intersection(b.index)
        if len(shared):
            diff = (a.loc[shared] - b.loc[shared]).to_numpy()
            boot = event_boot(diff, np.asarray(shared), n_boot=n_boot, seed=999)
            contrast = {
                "comparison": f"{HEADLINE_ARM} - {CONTROL_ARM}",
                "shared_events": int(len(shared)),
                "mean_usd_per_signal": boot["mean"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "p_one_sided": boot["p_one_sided"],
                "effective_events": boot["effective_events"],
            }
    # WHY a no-fill happened is as important as how often: "the book never came to
    # my price" and "the market settled before the control fired" are different
    # findings, and the second one would mean the control is broken rather than the
    # strategy good.
    nf = scored[~scored["filled"]].copy()
    if "no_fill_reason" not in nf.columns:
        nf["no_fill_reason"] = None
    reasons = (nf.groupby(["arm", nf["no_fill_reason"].fillna("unrecorded")])
               .size().rename("n").reset_index() if len(nf) else pd.DataFrame())
    return {"board": board, "contrast": contrast, "n_unresolved": n_unresolved_signals,
            "note": None, "scored": scored, "no_fill_reasons": reasons}


def score(cfg: dict | None = None, refetch: bool = True, n_boot: int = 2000) -> dict:
    """Refresh resolutions for every market in the paper ledger, score it, and write
    the scoreboard. Read-only against the network; writes only the paper artifacts."""
    from src.ingest import update_resolutions

    cfg = cfg or load_config()
    watchlist, manifest = load_freeze()
    rule = rule_from_manifest(manifest, cfg)
    signals, orders = load_ledger()

    resolutions = (pd.read_parquet(RESOLUTIONS_PATH) if RESOLUTIONS_PATH.exists()
                   else pd.DataFrame(columns=["market_id", "token_id", "outcome",
                                              "resolved", "resolved_value", "closed"]))
    if refetch and len(orders):
        session = make_session()
        resolutions = update_resolutions(session, cfg, set(orders["market_id"].unique()),
                                         resolutions)
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(resolutions, RESOLUTIONS_PATH, compression="gzip")

    settled = set(resolutions.loc[resolutions["resolved"].fillna(False), "market_id"]) \
        if len(resolutions) else set()
    orders = terminate_settled(orders, settled, int(time.time()))
    if len(orders) and refetch:
        atomic_to_parquet(orders, ORDERS_PATH, compression="gzip")

    out = score_orders(orders, signals, resolutions, rule, n_boot=n_boot)
    write_scoreboard(out, manifest, watchlist, signals, orders)
    return out


def _fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
        return "—"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def write_scoreboard(out: dict, manifest: dict, watchlist: pd.DataFrame,
                     signals: pd.DataFrame, orders: pd.DataFrame) -> None:
    """Write data/processed/paper_scoreboard.md (+ parquet).

    An empty table reads as a zero, so when there is nothing to score this says
    AWAITING DATA in the loudest place on the page."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    board = out.get("board", pd.DataFrame())
    atomic_to_parquet(board if len(board) else pd.DataFrame({"arm": pd.Series(dtype=object)}),
                      SCOREBOARD_PARQUET, compression="gzip")

    n_sig = len(signals)
    live = n_live_orders(orders)
    lines = [
        "# Forward paper-trade scoreboard (PAPER ONLY — no keys, no orders)",
        "",
        f"**Frozen {manifest['freeze_utc']}** at commit `{manifest['git_commit'][:8]}`. "
        f"Watchlist: {len(watchlist)} wallets. Stake: "
        f"${manifest['stake']['notional_per_signal_usd']:.2f} notional per signal per arm. "
        f"Headline arm: `{manifest['headline_arm']}`; control: `{manifest['control_arm']}`.",
        "",
        f"Signals detected: **{n_sig}** · live orders: **{live}** · "
        f"signals whose market has not resolved: **{out.get('n_unresolved', 0)}**.",
        "",
    ]
    if n_sig:
        counts = signals["status"].value_counts()
        lines += ["Signal disposition at detection: "
                  + " · ".join(f"`{k}` {v}" for k, v in counts.items()),
                  "",
                  "> `no_book_at_detection` still opens all three arms (the resting "
                  "limit can fill later); `skipped_book_error` opens nothing — we do "
                  "not know what was tradeable — and `skipped_cap` is the "
                  "max-open-positions seatbelt refusing a signal, which counts "
                  "against the strategy rather than for it.",
                  ""]
        horizons = signals["placebo_horizon_source"].dropna()
        if len(horizons):
            hc = horizons.value_counts()
            lines += ["Placebo horizon source: "
                      + " · ".join(f"`{k}` {v}" for k, v in hc.items()),
                      "", ]
    if out.get("note") or not len(board):
        lines += [
            f"> ## ⏳ AWAITING DATA — {out.get('note') or 'nothing scored yet'}.",
            ">",
            "> This is **not** a zero and **not** a negative result. Sports markets on "
            "the watchlist resolve in days; the Project 1 cohort can take weeks. Read "
            "`effective_events` before reading any dollar figure.",
            "",
        ]
    else:
        lines += [
            "| arm | signals | filled | **fill rate** | events | eff. events | "
            "notional filled | **$ / signal** | event-wtd $/signal | 95% CI | p | "
            "total P&L | return on filled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            r = board[(board["arm"] == arm) & (board["stratum"] == "all")]
            if not len(r):
                lines.append(f"| {arm} | 0 | 0 | — | 0 | — | — | — | — | — | — | — | — |")
                continue
            r = r.iloc[0]
            head = "**" if arm == manifest["headline_arm"] else ""
            ci = ("—" if pd.isna(r["ci_low"])
                  else f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
            lines.append(
                f"| {head}{arm}{head} | {int(r['n_signals'])} | {int(r['n_filled'])} "
                f"| **{r['fill_rate']:.1%}** | {int(r['n_events'])} "
                f"| {_fmt(r['effective_events'], '.2f')} "
                f"| ${_fmt(r['notional_filled_usd'], ',.2f')} "
                f"| **${_fmt(r['pnl_per_signal_usd'], '+,.3f')}** "
                f"| ${_fmt(r['pnl_per_signal_usd_event_wtd'], '+,.3f')} "
                f"| {ci} | {_fmt(r['p_one_sided'], '.3f')} "
                f"| ${_fmt(r['total_pnl_usd'], '+,.2f')} "
                f"| {_fmt(r['return_on_filled_notional'], '+.2%')} |")
        lines += [
            "",
            "`$ / signal` is the headline: **dollars per DETECTED signal, with "
            "unfilled signals contributing $0.** An arm that fills rarely earns "
            "proportionally less here, by construction. Never quote "
            "`return on filled` without the fill rate beside it.",
            "",
        ]
        cuts = [s for s in board["stratum"].unique() if s != "all"]
        if cuts:
            lines += [
                "### By source cohort",
                "",
                "The two cohorts are different populations — the sports wallets trade "
                "games that settle in days; Project 1's are dominated by micro-crypto "
                "that fires orders of magnitude more signals — so the pooled row above "
                "is mostly whichever cohort is louder. These cuts are pre-registered, "
                "not post-hoc slicing, and `all` remains the headline.",
                "",
                "| cohort | arm | signals | filled | fill rate | eff. events | "
                "$ / signal | 95% CI |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for stratum in cuts:
                for arm in ARMS:
                    r = board[(board["arm"] == arm) & (board["stratum"] == stratum)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    ci = ("—" if pd.isna(r["ci_low"])
                          else f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
                    lines.append(
                        f"| `{stratum}` | {arm} | {int(r['n_signals'])} "
                        f"| {int(r['n_filled'])} | {r['fill_rate']:.1%} "
                        f"| {_fmt(r['effective_events'], '.2f')} "
                        f"| ${_fmt(r['pnl_per_signal_usd'], '+,.3f')} | {ci} |")
            lines.append("")
        reasons = out.get("no_fill_reasons")
        if reasons is not None and len(reasons):
            lines += ["### Why the no-fills did not fill", "",
                      "| arm | reason | n |", "|---|---|---:|"]
            for _, rr in reasons.iterrows():
                lines.append(f"| {rr['arm']} | `{rr.iloc[1]}` | {int(rr['n'])} |")
            lines += ["", "A high `settled_before_trigger` count on `placebo_random` "
                      "means the CONTROL is broken, not that the thesis is good — the "
                      "placebo's horizon was wrong and it never got to act. Check "
                      "the placebo horizon sources above before reading the contrast.",
                      ""]

        c = out.get("contrast")
        lines += ["## The deciding comparison", ""]
        if c:
            ci = ("—" if c["ci_low"] is None or pd.isna(c["ci_low"])
                  else f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")
            lines += [
                f"**{c['comparison']}** on {c['shared_events']} shared events "
                f"(effective {c['effective_events']:.2f}): "
                f"**${c['mean_usd_per_signal']:+,.3f} per signal**, 95% CI {ci}, "
                f"one-sided p = {_fmt(c['p_one_sided'], '.3f')}.",
                "",
                "If this does not clear zero, the wallets are not the signal: taking "
                "their market and side at a random moment did as well as following "
                "them. `market_chase` is a machinery check — strong profit there "
                "means a bug, not an edge.",
                "",
            ]
        else:
            lines += ["Not computable yet — needs scored signals in both the headline "
                      "and the control arm.", ""]

    lines += ["## The pre-registered entry rule, verbatim from the manifest", "",
              "> " + manifest["entry_rule"].replace("\n", "\n>\n> "), "",
              "## The pre-registered scoring rule, verbatim from the manifest", "",
              "> " + manifest["scoring_rule"].replace("\n", "\n>\n> "), "",
              "## Seatbelts", ""]
    sb = manifest["seatbelts"]
    lines += [
        f"- Paper only; **no keys, no signing, no order placement** — public "
        f"unauthenticated GETs only.",
        f"- Kill switch: `touch {sb['kill_switch_path']}` halts the loop before any "
        f"fetch or write.",
        f"- Per-position notional cap: ${sb['max_notional_per_position_usd']:.2f}.",
        f"- Max simultaneously-live orders: {sb['max_open_positions']} "
        f"(signals over the cap are recorded as `skipped_cap` and still counted).",
        "- `--dry-run` is the default; persisting requires an explicit `--commit`.",
        "",
        "## Known limits, recorded at freeze time", "",
    ]
    lines += [f"- {lim}" for lim in manifest["known_limits"]]
    SCOREBOARD_MD.write_text("\n".join(lines) + "\n")
    print(f"[paper] -> {SCOREBOARD_MD}\n[paper] -> {SCOREBOARD_PARQUET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=("Forward PAPER trader (execution roadmap stage 1). PAPER ONLY: "
                     "no keys, no signing, no order placement, public GETs only."))
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="pre-register the rule and the watchlist")
    f.add_argument("--force", action="store_true",
                   help="overwrite an existing freeze (records an amendment)")
    f.add_argument("--reason", type=str, default=None)

    r = sub.add_parser("run", help="poll the watchlist and progress the paper book")
    r.add_argument("--commit", action="store_true",
                   help="persist the ledger (WITHOUT this, --dry-run is the default)")
    r.add_argument("--dry-run", action="store_true", help="explicit no-op default")
    r.add_argument("--passes", type=int, default=1)

    s = sub.add_parser("score", help="score the paper ledger and write the scoreboard")
    s.add_argument("--no-refetch", action="store_true")
    s.add_argument("--boot", type=int, default=2000)

    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "freeze":
        m = freeze(cfg, force=args.force, amend_reason=args.reason)
        print(f"[paper] FROZEN {m['freeze_utc']} @ {m['git_commit'][:8]}")
        for k, v in m["funnel"].items():
            print(f"    {k:32s} {v}")
        print(f"  -> {WATCHLIST_PATH}\n  -> {FREEZE_MANIFEST_PATH}")
        score(cfg, refetch=False)      # write the live 'awaiting data' scoreboard
    elif args.cmd == "run":
        if args.commit and args.dry_run:
            raise SystemExit("--commit and --dry-run are mutually exclusive.")
        run(cfg, passes=args.passes, dry_run=not args.commit)
    else:
        score(cfg, refetch=not args.no_refetch, n_boot=args.boot)


if __name__ == "__main__":
    main()
