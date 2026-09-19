"""FORWARD PAPER TRADER — REAL-WORLD ARM (`paper_rw`). Execution roadmap stage 1b.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PAPER ONLY. READ-ONLY. NO PRIVATE KEYS. NO SIGNING. NO ORDERS.      │
    │                                                                      │
    │  Public unauthenticated GETs to data-api.polymarket.com (/trades),   │
    │  clob.polymarket.com (/book, /markets/<condition_id>) and            │
    │  gamma-api.polymarket.com (/markets?condition_ids=, for the REAL fee │
    │  schedule). Writes hypothetical fills to a local parquet ledger. It  │
    │  imports no signing library, holds no key, and makes no POST/PUT/    │
    │  DELETE of any kind — asserted by a source scan in                   │
    │  tests/test_paper_rw.py::test_module_has_no_execution_code_path.     │
    └──────────────────────────────────────────────────────────────────────┘

WHY A SECOND ARM RATHER THAN A CHANGE TO THE FIRST
--------------------------------------------------
`src/paper_trader.py` has been LIVE since 2026-07-26T22:48:29Z on a frozen
watchlist of 38 wallets. It is a pre-registered forward experiment; touching its
watchlist, its state or its manifest would destroy the record. So this module is
**additive and byte-isolated**: its own freeze manifest, its own state file, its
own ledger directory (`data/interim/paper_rw/`), its own scoreboard, its own kill
switch. It shares no path with the running arm and
`tests/test_paper_rw.py::test_the_live_arms_artifacts_are_never_touched` pins that.

It shares the *machinery*, though: every fill decision, the book parser, the tick
grid, the taker-fee shape, the placebo draw, the order-progression state machine,
the settled-market terminator, the event-clustered bootstrap and the event unit
are all imported from `src.paper_trader`. The only new code here is the four
things this arm needs and that one does not.

WHAT THIS ARM ASKS
------------------
Can the FIVE wallets certified by the Project 1 gate on the deep, performance-
blind real-world sample (`docs/realworld_deep_sample.md`) actually be COPIED, in
slow / real-world markets, in dollars, forward, against a live order book?

The four differences from the running arm:

1. REAL-WORLD ONLY. A signal in a micro-crypto market (`discover.is_real_world`,
   i.e. `classify_market(...) != "micro_crypto"` — the same definition the rest of
   the repo uses) is recorded and SKIPPED. It still counts in `n_signals_seen`, so
   the filter is visible rather than silent. These five wallets are 94-100%
   real-world by `wallet_profile.parquet`, so the filter should bite rarely; if it
   bites often that is itself worth seeing.
2. REALISTIC COPY LAG, MEASURED. Every order records `detect_lag_s` (their trade →
   our detection) and, once filled, `fill_lag_s` (their trade → our fill). The
   scoreboard reports the ACHIEVED lag distribution per arm instead of assuming
   one. A copy strategy whose fills land two hours late is a different strategy.
3. REAL FEES, PER CATEGORY. `fee = shares * k * price * (1 - price)`, taker only.
   `k` is read LIVE from Gamma's `feeSchedule.rate` per market where available
   (`docs/polymarket_mechanics.md` §"Fee field to read"), and falls back to the
   frozen per-category map (0.04 politics/tech · 0.05 sports/econ/culture/other ·
   0.07 crypto · 0 geopolitics) otherwise. GROSS and NET are both reported, plus a
   stressed k. The running arm froze `k = 0.0`, which `docs/polymarket_mechanics.md`
   lists as WRONG assumption #1; this arm does not repeat it.
4. PER-WALLET SCOREBOARD, not just the pooled one. Five wallets is few enough that
   the pooled mean can be one wallet's, and the owner's question is about
   individual breadwinners. Per-wallet strata are pre-registered in the manifest
   BEFORE any data arrives, so they are not post-hoc slicing.

THE THREE ARMS (same three, same code, roles re-read for this population)
------------------------------------------------------------------------
* `market_chase`   — **THE HEADLINE HERE.** A taker order at the best ask at
                     detection. In micro-crypto this is "chasing" and the repo has
                     shown it dead; in a market that lives for days, buying at
                     market within one poll of the wallet IS what copying means,
                     and it is the only arm whose fills need no queue assumption.
* `limit_noChase`  — secondary: a resting limit at or below their own price. Its
                     fill rate is an UPPER BOUND (queue position is not modelled),
                     so it cannot be the headline for an actionability question.
* `placebo_random` — THE CONTROL. Same market, same side, their timing discarded.
                     Per the READING CORRECTION in `src/paper_trader.py`, read the
                     ABSOLUTE dollars per arm FIRST (that is the selection test)
                     and the difference SECOND (that is the timing test). The live
                     prior from three independent 2026-07-26 analyses is
                     difference ~ 0 with the control positive.

HONEST PRIOR: every copy thesis in this repo has come back negative or thin. The
rule is frozen before the data so a negative cannot be argued away and a positive
cannot be manufactured.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.common import (
    DATA_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    REPO_ROOT,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
)
from src.discover import classify_market, is_real_world
from src.watch import advance_cursor, default_cursor, new_trades_since

import src.paper_trader as pt

# --- paths: DISJOINT from the live arm's, deliberately and testably ---------
RW_DIR = INTERIM_DIR / "paper_rw"
STATE_PATH = RW_DIR / "rw_state.json"
SIGNALS_PATH = RW_DIR / "rw_signals.parquet"
ORDERS_PATH = RW_DIR / "rw_orders.parquet"
RESOLUTIONS_PATH = RW_DIR / "rw_resolutions.parquet"

FREEZE_MANIFEST_PATH = PROCESSED_DIR / "paper_rw_freeze_manifest.json"
WATCHLIST_PATH = PROCESSED_DIR / "paper_rw_watchlist.parquet"
SCOREBOARD_MD = PROCESSED_DIR / "paper_rw_scoreboard.md"
SCOREBOARD_PARQUET = PROCESSED_DIR / "paper_rw_scoreboard.parquet"

# SEATBELT: EITHER file halts this arm. The shared one is the project-wide stop
# button (it already halts the live arm), the local one stops this arm alone
# without disturbing the running 38-wallet experiment.
KILL_SWITCH_PATH = DATA_DIR / "KILL_PAPER_RW"
SHARED_KILL_SWITCH_PATH = pt.KILL_SWITCH_PATH        # data/KILL_PAPER_TRADER

VALIDATED_PATH = INTERIM_DIR / "realworld" / "validated.parquet"
WALLET_PROFILE_PATH = INTERIM_DIR / "realworld" / "wallet_profile.parquet"

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

ARMS = pt.ARMS                    # same three arms, same code
HEADLINE_ARM = "market_chase"     # ...but a different headline; see the docstring
CONTROL_ARM = "placebo_random"
POOLED_STRATUM = "rw5_pooled"

# --- COPY-VALIDATED TIER (amended 2026-07-28, still at 0 RESOLVED observations) --
# AMENDMENT 1. The first version of this tier was built on a copy simulation that
# credited the follower at the book MIDPOINT while charging the wallet its actual
# post-slippage taker VWAP — worth ~3.13c/share of unearned edge (the wallet paid
# above the mid on 92.8% of 133,003 bets, mean +3.23c, against a 0.10c tick).
# Correcting that and re-running across all 46 certified wallets in the deep tape
# (docs/copy_verdict.md "CORRECTION 2") cut the copyable set from 38 to 11 and
# REVERSED the original headline pick: 0x1ee9a5fc09 falls to +1.61c [-1.05, +4.22]
# and no longer qualifies.
#
# The headline is now the five survivors with the TIGHTEST intervals — selected on
# CI width, not point estimate, which is why 0x5ac8f582 (+16.4c but [+0.13, +30.21],
# i.e. almost no information) is excluded:
#   0x253da81575  +6.56c [+2.62, +10.10]   221 markets
#   0x9fc0432877  +5.61c [+2.93,  +7.82]   114 markets
#   0x69ea0d77ef  +5.23c [+3.07,  +7.51]   447 markets  <- the only original survivor
#   0xd06f0f7719  +3.52c [+1.24,  +5.49]   816 markets
#   0x05c5aab002  +3.35c [+2.02,  +4.84]   165 markets
#
# LEGITIMACY. This amends on better BACKTEST evidence while zero forward
# observations have RESOLVED, so no forward outcome informed it. The moment one
# resolves, this list is frozen for good.
# NOTHING IS DROPPED (CLAUDE.md): the original five stay in the watchlist with
# their per-wallet rows and their registered predictions intact, so the earlier
# tier remains auditable rather than rewritten away.
COPY_VALIDATED_STRATUM = "rw5_copyable_v2"
COPY_VALIDATED_WALLETS = (
    "0x253da81575",
    "0x9fc0432877",
    "0x69ea0d77ef",
    "0xd06f0f7719",
    "0x05c5aab002",
)
# The superseded tier, retained as a declared stratum so the swap is visible in
# every scoreboard rather than buried in git history.
LEGACY_STRATUM = "rw2_copyable_v1"
LEGACY_WALLETS = ("0x1ee9a5fc09", "0x69ea0d77ef")

# THE FIVE. Prefixes are what the task named; the full addresses are resolved from
# data/interim/realworld/validated.parquet at freeze time and frozen into the
# manifest, so the manifest is self-contained and the resolution is auditable.
WALLET_PREFIXES = (
    # the original five (kept: nothing is dropped, their predictions stand)
    "0xe542afd388",
    "0x83255595ba",
    "0x1ee9a5fc09",
    "0x69ea0d77ef",
    "0x09bed19766",
    # added by AMENDMENT 1 — the copy-validated survivors from the 46-wallet rerun
    "0x253da81575",
    "0x9fc0432877",
    "0xd06f0f7719",
    "0x05c5aab002",
)

# Defaults. Economic ones are frozen into the manifest at freeze time; operational
# ones are read live from config so an operator can TIGHTEN but never loosen.
DEFAULT_NOTIONAL_USD = 100.0
DEFAULT_MAX_POSITION_USD = 100.0
# Deliberately high for the PAPER stage, and for the same reason the live arm's is:
# a binding cap truncates the sample exactly during the busiest markets, which is a
# selection effect on WHICH signals get tested. Any `skipped_cap` is reported on the
# scoreboard so a binding cap is never silent. It is not free — one of the five
# wallets fires several hundred in-play signals in a single football match — but the
# per-poll cost is one book fetch per distinct LIVE TOKEN, and those same hundreds of
# signals collapse onto ~10 tokens.
DEFAULT_MAX_OPEN_POSITIONS = 5000
# Real-world markets live for days to months, so the running arm's 6-hour fallback
# would draw the control's "random moment in the market's remaining life" from the
# first six hours of it — which is not a random moment, it is a slightly delayed
# copy. 48h is the frozen fallback, and it is only ever used when neither Gamma's
# precise endDate nor the CLOB's game_start_time is in the future.
DEFAULT_PLACEBO_HORIZON_HOURS = 48.0
# The stressed rate is the HIGHEST live rate on the platform (crypto, 0.07): the
# pessimistic "what if every market charged the worst rate" column.
DEFAULT_TAKER_FEE_K_STRESS = 0.07

# Frozen fallback fee map, from docs/polymarket_mechanics.md's live Gamma sweep
# (300 markets): politics_fees 0.04 · tech_fees 0.04 · sports_fees_v2 0.05 ·
# general_fees 0.05 · culture_fees 0.05 · weather_fees 0.05 · economics_fees 0.05 ·
# crypto_fees_v2 0.07 · geopolitics 0. Keyed on `discover.classify_market` labels.
FEE_K_BY_CATEGORY = {
    "politics": 0.04,
    "econ_macro": 0.05,
    "crypto_event": 0.07,
    "culture": 0.05,
    "micro_crypto": 0.07,
    "geopolitics": 0.0,
    "other": 0.05,
}
FEE_K_SPORTS = 0.05
FEE_K_DEFAULT = 0.05


def fee_k_for_category(category: str | None) -> float:
    """The frozen per-category taker rate — used only when the live `feeSchedule`
    is unavailable.

    NOTE, recorded because it is a KNOWN conservative error: `classify_market` has
    no `geopolitics` label, so a geopolitics market (true rate 0) falls into
    `other` and is charged 0.05 here. That overstates cost, i.e. it can only make
    this arm look WORSE than reality, never better. The live `feeSchedule.rate`
    read from Gamma has no such gap and takes precedence whenever it is present."""
    cat = category if isinstance(category, str) and category else "other"
    if cat.startswith("sports"):
        return FEE_K_SPORTS
    return float(FEE_K_BY_CATEGORY.get(cat, FEE_K_DEFAULT))


def resolve_fee_k(fee_meta: dict | None, category: str | None) -> tuple[float, str]:
    """Pick the taker `k` for one market and say where it came from.

    Precedence, frozen in the manifest:
      1. `feesEnabled == False`            -> k = 0, source `gamma_fees_disabled`
      2. `feeSchedule.rate` with `exponent == 1` -> that rate, `gamma_feeSchedule`
      3. anything else                     -> the frozen category map, `category_map`

    Exponent 1 is checked rather than assumed: the fee formula frozen here is
    `shares * k * p * (1-p)`, which IS the exponent-1 form. A market advertising a
    different exponent is not covered by that formula, so its live rate is not
    usable and the category fallback is taken (and the exponent is persisted per
    signal so a reader can see it happened)."""
    meta = fee_meta or {}
    if meta.get("fees_enabled") is False:
        return 0.0, "gamma_fees_disabled"
    rate, exponent = meta.get("fee_rate_live"), meta.get("fee_exponent")
    if rate is not None and np.isfinite(rate) and rate >= 0 and (
            exponent is None or float(exponent) == 1.0):
        return float(rate), "gamma_feeSchedule"
    return fee_k_for_category(category), "category_map"


# ---------------------------------------------------------------------------
# THE PRE-REGISTERED RULE (frozen verbatim into the manifest by `freeze`)
# ---------------------------------------------------------------------------

HYPOTHESIS = (
    "H1 (THE OWNER'S QUESTION). The five wallets certified by the Project 1 gate on "
    "the deep, performance-blind real-world sample (docs/realworld_deep_sample.md; "
    "set-level measured FDR 17.3%, so ~4 of these 5 are expected to be genuinely "
    "sharp) can be COPIED profitably in SLOW / REAL-WORLD markets: a follower who "
    "buys the same outcome token at market within one poll of their trade, at a "
    "fixed 100 USD stake, earns strictly positive dollars PER DETECTED SIGNAL after "
    "real per-category taker fees. The null is <= 0 USD per detected signal. "
    "\n"
    "H2 (SELECTION vs TIMING, per the READING CORRECTION in src/paper_trader.py). "
    "placebo_random's ABSOLUTE dollars test SELECTION — does taking their market and "
    "side at an arbitrary moment still make money? The DIFFERENCE headline minus "
    "control tests TIMING — does entering promptly beat entering at a random moment "
    "in the market they chose? The repo's live prior, from three independent "
    "2026-07-26 analyses, is difference ~ 0 with the control positive: a "
    "market-discovery signal, not a timing signal. Absolute per-arm dollars are read "
    "FIRST and the difference SECOND. "
    "\n"
    "H3 (PER WALLET). Each of the five is reported separately, pre-registered here "
    "before any data exists. Five wallets is few enough that a pooled mean can be one "
    "wallet's; the owner's question is about individual breadwinners, not a portfolio "
    "average. "
    "\n"
    "H4 (COPYABILITY TIER). The HEADLINE stratum is the one named in "
    "`strata.headline`, which `score` reads and which is authoritative: the wallets "
    "whose edge survived a MEASURED 2-minute copy lag on past data, net of real costs "
    "and net of the spread the wallet demonstrably paid (docs/copy_verdict.md "
    "CORRECTION 2). The full pool is SECONDARY and is retained undropped, because the "
    "backtest predicts the rest FAIL as copy targets — 0x09bed19766 unproven at n=307, "
    "0xe542afd388 ~zero (it buys at 27c and longshot prices move on size), "
    "0x83255595ba NEGATIVE (in-play football, where a 2-minute lag is fatal), "
    "0x1ee9a5fc09 ~zero (it was the original headline pick at '+31.9%'; that was the "
    "midpoint-vs-ask artifact). Those are falsifiable predictions made before any data "
    "exists, not exclusions. Neither stratum is ever displaced by whichever wallet "
    "happens to look best once data arrives. "
    "⚠️ AMENDMENT 1 (2026-07-28) moved the headline from a two-wallet `rw2_copyable` "
    "to the current stratum and EXPANDED the watchlist from 5 wallets to 9; see "
    "`amendments`. This text names no specific stratum precisely so that a future "
    "amendment cannot leave it contradicting `strata.headline`, which is what happened "
    "to its predecessor."
)

ENTRY_RULE = (
    "SIGNAL. A signal is a new BUY trade by one of the five frozen wallets, detected "
    "by polling the public data-api /trades?user= endpoint. The last-seen cursor "
    "floors at the moment the wallet joins this watchlist, so pre-existing history is "
    "never traded. SELL trades are ignored (a position is an entry). Every signal is "
    "recorded, including ones that produce no order. "
    "\n"
    "REAL-WORLD FILTER. The market is classified with discover.classify_market on its "
    "slug and question — the same classifier the rest of the repo uses — and a signal "
    "is traded only if discover.is_real_world(category) is True, i.e. the category is "
    "not micro_crypto. A micro-crypto signal is recorded with status "
    "skipped_micro_crypto and opens NO orders; a signal whose market cannot be "
    "classified at all (no slug and no question from either the tape or Gamma) is "
    "recorded as skipped_unclassified and also opens no orders. Both still count in "
    "n_signals_seen, so the filter is visible and can never be mistaken for an absence "
    "of activity. Skipped signals are EXCLUDED from every dollar statistic, because "
    "this arm's question is explicitly about slow / real-world markets only. "
    "\n"
    "BOOK. At detection the live CLOB order book for that outcome token is fetched "
    "(GET /book?token_id=) and persisted with the signal: best bid/ask, size at each, "
    "the top levels of both sides, the token's tick size and minimum order size. Every "
    "fill decision in every arm is made against a book snapshot with real size at the "
    "price; no fill is ever inferred from the trade tape. "
    "\n"
    "STAKE. Fixed 100.00 USD of notional per signal per arm, never scaled by "
    "conviction, wallet, price or bankroll, capped by the per-position notional cap. "
    "A fill may be partial: you get the shares the book actually offered and the "
    "unspent budget stays unspent. "
    "\n"
    "ARM market_chase (THE HEADLINE FOR THIS ARM). A taker order at detection that "
    "walks the book from the best ask upward until the budget is spent, at whatever "
    "prices are there, paying the taker fee. No limit. This is what copying a wallet "
    "in a slow market actually is, and it is the only arm whose fill needs no "
    "assumption about queue position. If the book has no asks at detection it is a "
    "NO-FILL. "
    "\n"
    "ARM limit_noChase (SECONDARY). A resting buy limit at the wallet's own entry "
    "price, rounded DOWN to the token's tick grid so the limit is never above the "
    "price they paid. If the best ask is already at or below the limit at detection it "
    "fills immediately as a TAKER, walking levels at or below the limit and paying the "
    "taker fee. Otherwise it rests and every subsequent poll re-checks the live book: "
    "whenever size appears at or below the limit it fills as a MAKER at the limit "
    "price and pays no fee. Partial fills accumulate across polls against the same "
    "budget. If the market settles unfilled it is a NO-FILL, which is an outcome and "
    "not a discard. Its fill rate is an UPPER BOUND because queue position is not "
    "modelled, which is exactly why it is not the headline. "
    "\n"
    "ARM placebo_random (THE CONTROL). Same market, same outcome token, same stake, "
    "the wallet's timing discarded: at detection a trigger time is drawn ONCE, "
    "deterministically, as trigger = detect_ts + u * (horizon_ts - detect_ts) where u "
    "in [0,1) is SHA-256(placebo_seed || signal_id) read as a fraction. horizon_ts is "
    "the first of the CLOB's game_start_time, Gamma's precise endDate and the CLOB's "
    "end_date_iso that is still in the FUTURE at detection, and otherwise detect_ts "
    "plus the frozen fallback horizon. The source is recorded per signal. On the first "
    "poll at or after the trigger the arm takes at the best ask exactly like "
    "market_chase. If the market settles before the trigger fires it is a NO-FILL. The "
    "draw is a pure function of the frozen seed and the signal id, so it is "
    "reproducible and could not have been redrawn after seeing an outcome. "
    "\n"
    "COPY LAG. Every order records detect_lag_s = detect_ts - wallet_trade_ts and, on "
    "its first fill, fill_lag_s = first_fill_ts - wallet_trade_ts. The scoreboard "
    "reports the ACHIEVED lag distribution per arm (median, quartiles, max). No lag is "
    "assumed anywhere; the poll cadence is an input to the experiment, not a "
    "parameter of the result. "
    "\n"
    "COSTS. The taker fee is shares * k * price * (1 - price), summed level by level "
    "across a walked fill, taker fills only; makers pay zero. k is read PER MARKET "
    "from Gamma's feeSchedule.rate at detection when feesEnabled is true and the "
    "advertised exponent is 1, and otherwise from the frozen per-category map in this "
    "manifest. The k actually used and its source are persisted per signal. GROSS "
    "(pre-fee) and NET (post-fee) dollars are BOTH reported, plus a stressed column at "
    "the platform's highest live rate. Prices are the prices actually resting on the "
    "book, so slippage is whatever walking that book costs. The tick grid is the "
    "token's own tick size from the CLOB, not a constant. A fill smaller than the "
    "token's minimum order size is not a fill. "
    "\n"
    "SEATBELTS. Paper only: no keys, no signing, no order placement, public "
    "unauthenticated GETs only. The presence of EITHER kill-switch file (the shared "
    "one or this arm's own) halts the loop before any fetch or write. Notional per "
    "position is capped. At the open-position cap a new signal is recorded as "
    "skipped_cap and generates no orders, still counted, so the cap costs the strategy "
    "rather than flattering it. --dry-run is the default and persists nothing. This "
    "arm writes only its own artifacts and never touches the running paper trader's "
    "state, watchlist, manifest or scoreboard."
)

SCORING_RULE = (
    "Every arm is scored on the SAME set of traded signals, so the arms are paired "
    "signal by signal and event by event. A signal enters scoring once its market has "
    "a resolution; unresolved signals are excluded from every arm identically and "
    "counted as n_unresolved. Signals skipped by the real-world filter or the cap are "
    "excluded from every arm identically and reported separately. "
    "\n"
    "PER ORDER. gross P&L in USD = shares * resolved_value - cost_usd. net P&L = gross "
    "- fee_usd, where fee_usd = k * sum_over_levels(shares_i * p_i * (1 - p_i)) on "
    "taker fills only, at the per-market k resolved at detection. A NO-FILL scores "
    "exactly 0.00 USD on 0.00 USD of notional and is never dropped. "
    "\n"
    "THE HEADLINE STATISTIC IS NET DOLLARS PER DETECTED (traded) SIGNAL, with unfilled "
    "signals contributing 0. An arm that fills 5% of the time and wins on those earns "
    "5% of the dollars and this statistic says so. Return on FILLED notional is "
    "reported alongside and must never be quoted without the fill rate next to it. "
    "\n"
    "CLUSTERING. The unit of independent resolution is the sports resolution EVENT "
    "(src/sports_events.py, version recorded in this manifest) where the sports "
    "labeller recognises the market, and the market_id elsewhere. The headline is "
    "EVENT-WEIGHTED: the mean over events of the within-event mean dollars per signal. "
    "The bootstrap resamples EVENTS with replacement, never signals, giving a 95% CI "
    "and a one-sided p versus zero; it is undefined below two events and is reported "
    "as a dash rather than as a zero. n_signals, n_filled, fill_rate, n_events and "
    "effective_events (Kish) sit beside every dollar figure. "
    "\n"
    "STRATA, fixed here in advance and never extended: the pooled book (rw5_pooled, "
    "THE HEADLINE) plus exactly one stratum per frozen wallet. No other slicing is "
    "permitted — not by category, not by price band, not by time window — because any "
    "such cut chosen after the data arrives is a search. Per-wallet rows are reported "
    "always, including when they are unreadably small, and the pooled row is never "
    "displaced by whichever wallet happens to look best. "
    "\n"
    "READING ORDER, pre-registered. (1) the fill rate and the achieved copy-lag "
    "distribution; (2) effective_events, not signal counts; (3) the ABSOLUTE net "
    "dollars per arm, which is the SELECTION test; (4) headline minus control on "
    "shared events, which is the TIMING test; (5) the per-wallet rows. A high "
    "settled_before_trigger count on placebo_random means the CONTROL is broken, not "
    "that the thesis is good: stop reading the contrast at that point. "
    "\n"
    "Arms are never re-defined, re-weighted or added in light of forward results, the "
    "watchlist is never re-frozen, and the fee model is never relaxed."
)


# ---------------------------------------------------------------------------
# SEATBELTS
# ---------------------------------------------------------------------------

def kill_switch_paths(cfg: dict | None = None) -> list:
    """Both stop buttons. The shared one halts every paper arm on the box; the
    local one halts this arm only, so an operator can stop this experiment without
    disturbing the running 38-wallet one (or vice versa)."""
    cfg = cfg or {}
    custom = (cfg.get("paper_rw", {}) or {}).get("kill_switch_path")
    local = KILL_SWITCH_PATH
    if custom:
        from pathlib import Path
        local = Path(custom)
    return [pt.kill_switch_path(cfg), local]


def kill_switch_engaged(cfg: dict | None = None) -> bool:
    return any(p.exists() for p in kill_switch_paths(cfg))


def _repo_relative(path) -> str:
    """A repo-relative label where possible. The manifest and the scoreboard are
    committed artifacts read on other machines, so recording an absolute path there
    would print an instruction that does not exist on the reader's box."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def kill_switch_labels(cfg: dict | None = None) -> list[str]:
    return [_repo_relative(p) for p in kill_switch_paths(cfg)]


def rw_settings(cfg: dict | None = None) -> dict:
    """Operational settings, read live from config so they can be tightened. The
    economic rule (stake, fee model, placebo seed) lives in the frozen manifest."""
    p = ((cfg or {}).get("paper_rw", {}) or {})
    return {
        "notional_per_signal_usd": float(p.get("notional_per_signal_usd",
                                               DEFAULT_NOTIONAL_USD)),
        "max_notional_per_position_usd": float(p.get("max_notional_per_position_usd",
                                                     DEFAULT_MAX_POSITION_USD)),
        "max_open_positions": int(p.get("max_open_positions", DEFAULT_MAX_OPEN_POSITIONS)),
        "poll_seconds": int(p.get("poll_seconds", 300)),
        "taker_fee_k_stress": float(p.get("taker_fee_k_stress", DEFAULT_TAKER_FEE_K_STRESS)),
        "placebo_fallback_horizon_hours": float(
            p.get("placebo_fallback_horizon_hours", DEFAULT_PLACEBO_HORIZON_HOURS)),
        "enforce_min_order_size": bool(p.get("enforce_min_order_size", True)),
    }


# ---------------------------------------------------------------------------
# CLASSIFICATION + FEES — the two things this arm adds at detection
# ---------------------------------------------------------------------------

def classify_signal(slug: str | None, question: str | None) -> tuple[str | None, bool]:
    """(category, tradeable). Returns (None, False) when there is nothing to
    classify on — an unclassifiable market is NOT assumed real-world, because
    assuming it would quietly let micro-crypto into an explicitly real-world-only
    experiment."""
    if not (slug or question):
        return None, False
    category = classify_market(slug, question)
    return category, bool(is_real_world(category))


def fetch_gamma_market(session, condition_id: str, timeout: int = 25) -> dict:
    """GET gamma-api.polymarket.com/markets?condition_ids=<cid> — public,
    unauthenticated, read-only.

    Gamma is the ONLY source for two things this arm needs and the CLOB does not
    serve: the real `feeSchedule` (docs/polymarket_mechanics.md: read
    `feeSchedule.rate`, NEVER `taker_base_fee`, which is identically 1000 across
    categories whose real rates are 0.04/0.05/0.07) and a PRECISE `endDate`
    (the CLOB's `end_date_iso` is the midnight floor of the scheduled end day).

    Returns {} on any failure; the caller falls back to the category fee map and
    the CLOB horizon, so a Gamma outage degrades this arm, it does not stop it."""
    resp = session.get(f"{GAMMA_API_BASE}/markets",
                       params={"condition_ids": condition_id}, timeout=timeout)
    if resp.status_code in (400, 404):
        return {}
    resp.raise_for_status()
    rows = resp.json() or []
    if not rows:
        return {}
    m = rows[0]
    sched = m.get("feeSchedule") or {}

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "gamma_slug": m.get("slug"),
        "gamma_question": m.get("question"),
        "gamma_end_ts": pt._iso_ts(m.get("endDate")),
        "fees_enabled": m.get("feesEnabled"),
        "fee_type": m.get("feeType"),
        "fee_rate_live": _f(sched.get("rate")),
        "fee_exponent": _f(sched.get("exponent")),
        "fee_taker_only": sched.get("takerOnly"),
    }


def copy_lag_columns(orders: pd.DataFrame) -> pd.DataFrame:
    """Stamp the ACHIEVED copy lag onto every order. Idempotent and vectorized.

    `detect_lag_s` is how long after the wallet's own trade we saw it (the poll
    cadence plus the API's own lag); `fill_lag_s` is how long after their trade our
    first share actually filled. Recording both is the difference between measuring
    a copy strategy and assuming one — a resting limit that fills six hours later is
    not the same trade the wallet made, and the scoreboard must be able to say so."""
    if orders.empty:
        return orders
    out = orders.copy()
    wallet_ts = pd.to_numeric(out["wallet_ts"], errors="coerce")
    detect_ts = pd.to_numeric(out["detect_ts"], errors="coerce")
    first_fill = pd.to_numeric(out["first_fill_ts"], errors="coerce")
    out["detect_lag_s"] = detect_ts - wallet_ts
    out["fill_lag_s"] = first_fill - wallet_ts
    return out


def _order_for_advance(order: dict) -> dict:
    """Round-tripping an order through a parquet-backed object column turns a
    `None` into a `NaN`, and `paper_trader._apply_fill` stamps `first_fill_ts` only
    when it finds exactly `None`. Left alone, an order that fills on a LATER poll
    would silently never record when it filled — which is precisely the measurement
    this arm exists to make. Normalizing here (rather than editing the running
    arm's module) keeps the live experiment untouched.

    NOTE for the live arm: `src/paper_trader.py` has the same NaN-vs-None
    behaviour. It costs it nothing — nothing there reads `first_fill_ts` — but its
    later-poll maker fills carry a null fill timestamp for the same reason."""
    out = dict(order)
    if out.get("first_fill_ts") is not None and pd.isna(out["first_fill_ts"]):
        out["first_fill_ts"] = None
    return out


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

SIGNAL_COLUMNS = [
    "signal_id", "wallet", "wallet_label", "cohorts", "market_id", "token_id",
    "outcome", "side", "slug", "question", "category", "is_real_world",
    "wallet_entry_price", "wallet_size", "wallet_ts", "detect_ts", "detect_lag_s",
    "status", "best_ask", "best_ask_size", "best_bid", "best_bid_size", "tick_size",
    "min_order_size", "book_json", "market_end_ts", "game_start_ts", "gamma_end_ts",
    "placebo_horizon_ts", "placebo_horizon_source", "fee_k", "fee_k_source",
    "fee_rate_live", "fee_type", "fees_enabled", "fee_exponent", "fee_taker_only",
    "taker_base_fee_raw", "maker_base_fee_raw",
]

ORDER_COLUMNS = pt.ORDER_COLUMNS + [
    "wallet_label", "category", "fee_k", "wallet_ts", "detect_lag_s", "fill_lag_s",
]

# statuses a signal can carry (a superset of the live arm's, plus the two skips)
SKIP_MICRO = "skipped_micro_crypto"
SKIP_UNCLASSIFIED = "skipped_unclassified"
TRADED_STATUSES = ("accepted", "no_book_at_detection")


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"cursors": {}, "last_poll_ts": None, "polls": 0}


def load_ledger() -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = (pd.read_parquet(SIGNALS_PATH) if SIGNALS_PATH.exists()
               else pt._empty(SIGNAL_COLUMNS))
    orders = (pd.read_parquet(ORDERS_PATH) if ORDERS_PATH.exists()
              else pt._empty(ORDER_COLUMNS))
    return signals, orders


def save_ledger(signals: pd.DataFrame, orders: pd.DataFrame, state: dict) -> None:
    RW_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(signals, SIGNALS_PATH, compression="gzip")
    atomic_to_parquet(orders, ORDERS_PATH, compression="gzip")
    atomic_write_json(state, STATE_PATH)


# ---------------------------------------------------------------------------
# THE POLL CYCLE
# ---------------------------------------------------------------------------

def poll_once(session, cfg: dict, rule: dict, watchlist: pd.DataFrame,
              signals: pd.DataFrame, orders: pd.DataFrame, state: dict, now_ts: int,
              fetch_positions=pt.fetch_new_positions, get_book=pt.fetch_book,
              get_market=pt.fetch_market_meta, get_gamma=fetch_gamma_market
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """One full cycle: detect new signals, apply the real-world filter, open the
    three orders for the ones that survive it, then re-check every still-live order
    against a fresh book.

    Pure apart from the four injectable fetchers, so tests drive it with canned
    books instead of the network. Never mutates its inputs. One wallet's or one
    market's failure is logged and skipped; it never aborts the cycle."""
    cursors = dict(state.get("cursors", {}))
    new_signals: list[dict] = []
    new_orders: list[dict] = []
    stats = {"polled_wallets": 0, "signals_seen": 0, "signals_traded": 0,
             "skipped_micro_crypto": 0, "skipped_unclassified": 0, "skipped_cap": 0,
             "skipped_book_error": 0, "no_book": 0, "orders_opened": 0, "fills": 0,
             "errors": 0}

    live_count = int(orders["status"].isin(pt.LIVE_STATUSES).sum()) if len(orders) else 0
    known = set(signals["signal_id"]) if len(signals) else set()

    # ---- 1. detect new positions -----------------------------------------
    for _, row in watchlist.iterrows():
        wallet = row["wallet"]
        cursor = cursors.get(wallet, default_cursor(now_ts))
        try:
            trades = fetch_positions(session, cfg, wallet, cursor)
        except Exception as exc:  # noqa: BLE001 — one wallet must not abort the cycle
            print(f"[paper_rw] WARNING: {wallet[:12]} trade fetch failed: {exc}")
            stats["errors"] += 1
            continue
        stats["polled_wallets"] += 1
        fresh = new_trades_since(trades, cursor)
        cursors[wallet] = advance_cursor(cursor, trades)

        for trade in fresh:
            if trade["side"] != "BUY":
                continue
            sid = pt.signal_id_for(trade)
            if sid in known:
                continue           # a position is detected, and traded, exactly once
            known.add(sid)
            stats["signals_seen"] += 1

            sig = {
                "signal_id": sid, "wallet": wallet,
                "wallet_label": row.get("wallet_label", wallet[:12]),
                "cohorts": "p1_rw_certified",
                "market_id": trade["market_id"], "token_id": trade["token_id"],
                "outcome": trade.get("outcome"), "side": trade["side"],
                "slug": trade.get("slug"), "question": trade.get("question"),
                "category": None, "is_real_world": None,
                "wallet_entry_price": float(trade["entry_price"]),
                "wallet_size": float(trade["size"]),
                "wallet_ts": int(trade["timestamp"]),
                "detect_ts": int(now_ts),
                "detect_lag_s": int(now_ts) - int(trade["timestamp"]),
                "status": "accepted",
                "best_ask": float("nan"), "best_ask_size": 0.0,
                "best_bid": float("nan"), "best_bid_size": 0.0,
                "tick_size": None, "min_order_size": 0.0, "book_json": None,
                "market_end_ts": None, "game_start_ts": None, "gamma_end_ts": None,
                "placebo_horizon_ts": None, "placebo_horizon_source": None,
                "fee_k": None, "fee_k_source": None, "fee_rate_live": None,
                "fee_type": None, "fees_enabled": None, "fee_exponent": None,
                "fee_taker_only": None,
                "taker_base_fee_raw": None, "maker_base_fee_raw": None,
            }

            # ---- THE REAL-WORLD FILTER, before any network spend -----------
            category, tradeable = classify_signal(trade.get("slug"), trade.get("question"))
            gamma: dict = {}
            if category is None:
                # nothing to classify on from the tape — ask Gamma for the slug
                # before giving up, since a missing slug is a data gap, not evidence
                try:
                    gamma = get_gamma(session, trade["market_id"]) or {}
                except Exception as exc:  # noqa: BLE001
                    print(f"[paper_rw] WARNING: gamma lookup failed for {sid}: {exc}")
                    stats["errors"] += 1
                category, tradeable = classify_signal(
                    gamma.get("gamma_slug"), gamma.get("gamma_question"))
                if gamma.get("gamma_slug") and not sig["slug"]:
                    sig["slug"] = gamma.get("gamma_slug")
                if gamma.get("gamma_question") and not sig["question"]:
                    sig["question"] = gamma.get("gamma_question")
            sig["category"] = category
            sig["is_real_world"] = bool(tradeable)
            if category is not None and not tradeable:
                sig["status"] = SKIP_MICRO
                new_signals.append(sig)
                stats["skipped_micro_crypto"] += 1
                continue
            if category is None:
                sig["status"] = SKIP_UNCLASSIFIED
                new_signals.append(sig)
                stats["skipped_unclassified"] += 1
                continue

            # SEATBELT: at the open-position cap, record and open nothing.
            if live_count >= rule["max_open_positions"]:
                sig["status"] = "skipped_cap"
                new_signals.append(sig)
                stats["skipped_cap"] += 1
                continue

            # A book fetch that ERRORS is an observability failure: we do not know
            # what was tradeable, so we open nothing. A book that comes back EMPTY
            # is information (there was nothing to buy) and the arms still open, so
            # it shows up as a no-fill rather than vanishing.
            try:
                raw_book = get_book(session, cfg, trade["token_id"])
            except Exception as exc:  # noqa: BLE001
                print(f"[paper_rw] WARNING: book fetch failed for {sid}: {exc}")
                sig["status"] = "skipped_book_error"
                new_signals.append(sig)
                stats["errors"] += 1
                stats["skipped_book_error"] += 1
                continue

            book = pt.parse_book(raw_book)
            if not book["asks"]:
                sig["status"] = "no_book_at_detection"
                stats["no_book"] += 1
            ba, bas = pt.best_ask(book)
            bb, bbs = pt.best_bid(book)
            sig.update({
                "best_ask": ba, "best_ask_size": bas, "best_bid": bb,
                "best_bid_size": bbs, "tick_size": book["tick_size"],
                "min_order_size": book["min_order_size"],
                "book_json": json.dumps(
                    {"asks": book["asks"][:pt.BOOK_LEVELS_RECORDED],
                     "bids": book["bids"][:pt.BOOK_LEVELS_RECORDED],
                     "book_ts": book["book_ts"]}),
            })

            try:
                meta = get_market(session, cfg, trade["market_id"])
            except Exception as exc:  # noqa: BLE001
                print(f"[paper_rw] WARNING: market meta failed for {sid}: {exc}")
                meta, stats["errors"] = {}, stats["errors"] + 1
            sig.update({k: meta.get(k) for k in
                        ("market_end_ts", "game_start_ts", "taker_base_fee_raw",
                         "maker_base_fee_raw") if k in meta})
            if not gamma:
                try:
                    gamma = get_gamma(session, trade["market_id"]) or {}
                except Exception as exc:  # noqa: BLE001
                    print(f"[paper_rw] WARNING: gamma lookup failed for {sid}: {exc}")
                    stats["errors"] += 1
            sig.update({k: gamma.get(k) for k in
                        ("gamma_end_ts", "fees_enabled", "fee_type", "fee_rate_live",
                         "fee_exponent", "fee_taker_only") if k in gamma})

            # THE FEE, resolved once at detection and frozen onto the signal.
            k, k_source = resolve_fee_k(gamma, category)
            sig["fee_k"], sig["fee_k_source"] = float(k), k_source

            # the CLOB's market metadata backstops the book's tick / min size, which
            # matters precisely when the book came back empty
            if book["tick_size"] is None and meta.get("min_tick_size_meta"):
                book["tick_size"] = float(meta["min_tick_size_meta"])
                sig["tick_size"] = book["tick_size"]
            if not book["min_order_size"] and meta.get("min_order_size_meta"):
                book["min_order_size"] = float(meta["min_order_size_meta"])
                sig["min_order_size"] = book["min_order_size"]

            # Gamma's endDate is PRECISE; the CLOB's end_date_iso is the midnight
            # floor of the scheduled end day. Prefer the precise one when present.
            end_ts = sig.get("gamma_end_ts") if sig.get("gamma_end_ts") else sig.get("market_end_ts")
            horizon, hsource = pt.placebo_horizon(
                int(now_ts), end_ts, sig.get("game_start_ts"),
                rule["placebo_fallback_horizon_hours"])
            if hsource == "end_date_iso":
                hsource = ("gamma_endDate" if sig.get("gamma_end_ts")
                           else "clob_end_date_iso")
            sig["placebo_horizon_ts"] = horizon
            sig["placebo_horizon_source"] = hsource

            made = pt.open_orders_for_signal(sig, book, rule, now_ts)
            for o in made:                       # this arm's extra order columns
                o["wallet_label"] = sig["wallet_label"]
                o["category"] = category
                o["fee_k"] = float(k)
                o["wallet_ts"] = sig["wallet_ts"]
                o["detect_lag_s"] = sig["detect_lag_s"]
                o["fill_lag_s"] = None
            new_orders.extend(made)
            new_signals.append(sig)
            stats["signals_traded"] += 1
            stats["orders_opened"] += len(made)
            stats["fills"] += sum(1 for o in made if o["status"] == pt.ST_FILLED)
            live_count += sum(1 for o in made if o["status"] in pt.LIVE_STATUSES)

    # ---- 2. re-check every still-live order against a fresh book ----------
    orders_out = (pd.concat([orders, pd.DataFrame(new_orders, columns=ORDER_COLUMNS)],
                            ignore_index=True) if new_orders else orders.copy())
    if len(orders_out):
        live_mask = orders_out["status"].isin(pt.LIVE_STATUSES)
        books: dict[str, dict] = {}
        for idx in orders_out.index[live_mask]:
            order = _order_for_advance(orders_out.loc[idx].to_dict())
            if order["arm"] == "placebo_random" and (
                    order["trigger_ts"] is None or now_ts < int(order["trigger_ts"])):
                continue                       # not due; don't spend a request
            token = order["token_id"]
            if token not in books:
                try:
                    books[token] = pt.parse_book(get_book(session, cfg, token))
                except Exception as exc:  # noqa: BLE001
                    print(f"[paper_rw] WARNING: book re-check failed for "
                          f"{str(token)[:12]}: {exc}")
                    books[token] = pt.parse_book(None)
                    stats["errors"] += 1
            before = float(order["shares"])
            updated = pt.advance_order(order, books[token], rule, now_ts)
            if float(updated["shares"]) > before:
                stats["fills"] += 1
            for col, val in updated.items():
                orders_out.at[idx, col] = val
        orders_out = copy_lag_columns(orders_out)

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

def resolve_wallets(prefixes=WALLET_PREFIXES, validated_path=None) -> pd.DataFrame:
    """Resolve the five prefixes to full addresses from the deep real-world
    validation table, and REFUSE anything that is not certified.

    Three ways this could silently go wrong, all made loud:
      * a prefix that matches nothing -> KeyError, not a quietly shorter watchlist;
      * a prefix that matches more than one wallet -> ValueError, because a
        watchlist built on an ambiguous prefix is not a pre-registration;
      * a match whose `edge_persisted` is False -> ValueError, because the whole
        claim being tested is that these are the CERTIFIED ones."""
    path = validated_path or VALIDATED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — this arm's watchlist is defined by the deep "
            f"real-world validation table (docs/realworld_deep_sample.md).")
    cols = ["wallet", "edge_persisted", "out_of_sample_residual_edge",
            "out_of_sample_edge", "out_of_sample_n", "out_of_sample_markets",
            "out_of_sample_cluster_p", "in_sample_residual_edge", "in_sample_n",
            "stratum", "prior_arm", "n_discovery_bets"]
    df = pd.read_parquet(path, columns=cols)
    rows = []
    for prefix in prefixes:
        hit = df[df["wallet"].str.startswith(prefix)]
        if hit.empty:
            raise KeyError(f"wallet prefix {prefix} matches nothing in {path}")
        if len(hit) > 1:
            raise ValueError(f"wallet prefix {prefix} is ambiguous: {len(hit)} matches")
        r = hit.iloc[0]
        if not bool(r["edge_persisted"]):
            raise ValueError(
                f"{prefix} is NOT certified (edge_persisted is False) — this arm "
                f"tests the certified set and must not silently include a "
                f"non-member.")
        rows.append({"wallet": r["wallet"], "wallet_label": prefix,
                     **{c: r[c] for c in cols if c != "wallet"}})
    return pd.DataFrame(rows)


def build_watchlist(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The pre-registered watchlist: the five certified real-world wallets, read
    fresh at freeze time, with their certification metrics and (where available)
    their category mix so a reader can see they are real-world traders rather than
    taking it on trust."""
    watchlist = resolve_wallets()
    funnel = {
        "source": _repo_relative(VALIDATED_PATH),
        "prefixes": list(WALLET_PREFIXES),
        "n_wallets": int(len(watchlist)),
        "all_certified": bool(watchlist["edge_persisted"].all()),
    }
    if WALLET_PROFILE_PATH.exists():
        prof = pd.read_parquet(WALLET_PROFILE_PATH,
                               columns=["wallet", "real_world_bets", "micro_bets",
                                        "total_bets"])
        watchlist = watchlist.merge(prof, on="wallet", how="left")
        share = (watchlist["real_world_bets"] / watchlist["total_bets"]).astype(float)
        watchlist["real_world_share"] = share
        funnel["min_real_world_share"] = float(np.nanmin(share)) if len(share) else None
    funnel["median_out_of_sample_residual_edge"] = float(
        watchlist["out_of_sample_residual_edge"].median())
    funnel["median_out_of_sample_markets"] = float(
        watchlist["out_of_sample_markets"].median())
    return watchlist, funnel


def forward_observations() -> int:
    """How many signals this arm has already recorded. A freeze is only
    pre-registration while this is ZERO."""
    if not SIGNALS_PATH.exists():
        return 0
    try:
        return int(len(pd.read_parquet(SIGNALS_PATH, columns=["signal_id"])))
    except Exception:  # noqa: BLE001
        return -1


def freeze(cfg: dict | None = None, freeze_ts: int | None = None, force: bool = False,
           amend_reason: str | None = None) -> dict:
    """Write the pre-registered manifest + watchlist for this arm.

    Refuses to overwrite an existing freeze without `--force`, and records every
    re-freeze in an `amendments` log together with the number of forward
    observations that existed at the time — the audit trail that makes "this was
    still pre-registration" checkable rather than asserted."""
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
        raise RuntimeError("watchlist is empty — nothing to pre-register.")

    settings = rw_settings(cfg)
    ts = int(freeze_ts if freeze_ts is not None
             else (prior["freeze_ts"] if prior else time.time()))
    obs = forward_observations()
    amendments = list(prior.get("amendments", [])) if prior else []
    if prior is not None:
        amendments.append({
            "amended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "amended_commit": pt._git_commit(),
            "reason": amend_reason or "unspecified",
            "forward_observations_at_amendment": obs,
            "legitimate_pre_registration": obs == 0,
            "prior_entry_rule": prior.get("entry_rule"),
            "prior_scoring_rule": prior.get("scoring_rule"),
            "prior_wallets": prior.get("wallets"),
        })

    wallets = [
        {
            "wallet": r["wallet"], "label": r["wallet_label"],
            "stratum_in_sample_design": r["stratum"],
            "out_of_sample_residual_edge": float(r["out_of_sample_residual_edge"]),
            "out_of_sample_edge": float(r["out_of_sample_edge"]),
            "out_of_sample_n": int(r["out_of_sample_n"]),
            "out_of_sample_markets": int(r["out_of_sample_markets"]),
            "out_of_sample_cluster_p": float(r["out_of_sample_cluster_p"]),
            "real_world_share": (float(r["real_world_share"])
                                 if "real_world_share" in r and pd.notna(r["real_world_share"])
                                 else None),
        }
        for _, r in watchlist.iterrows()
    ]

    manifest = {
        "arm": "paper_rw",
        "stage": ("execution roadmap stage 1b — forward PAPER test of the five "
                  "certified REAL-WORLD wallets"),
        "freeze_ts": ts,
        "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "git_commit": pt._git_commit(),
        "sports_event_version": SPORTS_EVENT_VERSION,
        "independent_of": {
            "arm": "paper_trader",
            "note": ("this arm shares NO state, watchlist, manifest or scoreboard "
                     "with the running 38-wallet paper trader frozen "
                     "2026-07-26T22:48:29Z; it only imports its pure functions."),
        },
        "hypothesis": HYPOTHESIS,
        "arms": list(ARMS),
        "headline_arm": HEADLINE_ARM,
        "control_arm": CONTROL_ARM,
        "arm_roles": {
            "market_chase": ("THE HEADLINE FOR THIS ARM — buy at market within one "
                             "poll of the wallet. In a market that lives for days "
                             "this IS copying, and it needs no queue assumption."),
            "limit_noChase": ("SECONDARY — a resting limit at or below their price. "
                              "Its fill rate is an UPPER BOUND because queue "
                              "position is not modelled."),
            "placebo_random": ("THE CONTROL — same market and side, their timing "
                               "discarded. Its ABSOLUTE dollars test SELECTION; the "
                               "difference tests TIMING."),
        },
        "universe": {
            "rule": "discover.is_real_world(discover.classify_market(slug, question))",
            "excluded": "micro_crypto (the 5-minute up/down coin-flips)",
            "skipped_statuses": [SKIP_MICRO, SKIP_UNCLASSIFIED],
            "note": ("an unclassifiable market is SKIPPED, not assumed real-world; "
                     "assuming it would let micro-crypto into an explicitly "
                     "real-world-only experiment."),
        },
        "entry_rule": ENTRY_RULE,
        "scoring_rule": SCORING_RULE,
        "strata": {
            "headline": COPY_VALIDATED_STRATUM,
            "copy_validated": list(COPY_VALIDATED_WALLETS),
            "legacy_superseded": {LEGACY_STRATUM: list(LEGACY_WALLETS)},
            "pooled": POOLED_STRATUM,
            "per_wallet": [w["label"] for w in wallets],
            "note": ("pre-registered BEFORE any data exists. HEADLINE is "
                     f"{COPY_VALIDATED_STRATUM} ({len(COPY_VALIDATED_WALLETS)} wallets: "
                     f"{', '.join(COPY_VALIDATED_WALLETS)}) — those whose edge survived a "
                     "measured 2-minute copy lag on past data, net of the spread the "
                     "wallet demonstrably paid (docs/copy_verdict.md CORRECTION 2). "
                     f"{POOLED_STRATUM} (all {len(wallets)}) is SECONDARY and is retained "
                     "so that 'the rest fail forward too' stays a falsifiable prediction "
                     "rather than a quiet exclusion — nothing is dropped, per CLAUDE.md. "
                     "Neither is ever displaced by whichever wallet looks best once data "
                     "arrives. No further slicing is permitted. "
                     "⚠️ READ `rw5_pooled`'s ABSOLUTE dollars FIRST: the red-team audit of "
                     "2026-07-29 found the per-wallet ranking indistinguishable from "
                     "chance under a cluster-preserving null (P=0.23), so the named-wallet "
                     "stratum is the weaker object. Reading correction, not a rule change."),
        },
        "prior_expectation": {
            "note": ("Registered before any forward observation RESOLVED. Backtested "
                     "follower net edge in CENTS PER SHARE at a 2-minute lag, charging "
                     "the follower the same spread the wallet demonstrably paid plus "
                     "real per-category fees (docs/copy_verdict.md CORRECTION 2). "
                     "Stated so the forward test can FALSIFY it."),
            "_superseded": ("AMENDMENT 1 replaced an earlier set of expectations that "
                            "were inflated by a midpoint-vs-ask artifact worth ~3.13c/"
                            "share. Those numbers are preserved in git at 26d4944."),
            "0x253da81575": "+6.56c [+2.62, +10.10] — expect positive (headline)",
            "0x9fc0432877": "+5.61c [+2.93, +7.82] — expect positive (headline)",
            "0x69ea0d77ef": "+5.23c [+3.07, +7.51] — expect positive (headline)",
            "0xd06f0f7719": "+3.52c [+1.24, +5.49] — expect positive (headline)",
            "0x05c5aab002": "+3.35c [+2.02, +4.84] — expect positive (headline)",
            "0x1ee9a5fc09": "+1.61c [-1.05, +4.22] — expect ~ZERO. Was the original "
                            "headline pick at '+31.9%'; that was the artifact.",
            "0xe542afd388": "-1.78c [-3.99, +0.57] — expect ~zero to negative; buys at "
                            "27c and longshot prices move on size",
            "0x83255595ba": "in-play football — expect NEGATIVE; a 2-minute lag is "
                            "fatal to a live line",
            "0x09bed19766": "too few bets survive the guard to score — unproven",
        },
        "wallets": wallets,
        "stake": {
            "notional_per_signal_usd": settings["notional_per_signal_usd"],
            "sizing": "fixed notional per signal per arm; never scaled by anything",
        },
        "costs": {
            "taker_fee_model": "shares * k * price * (1 - price), taker fills only",
            "k_source_precedence": [
                "gamma feesEnabled == false -> k = 0",
                "gamma feeSchedule.rate with exponent == 1 -> that rate",
                "frozen category map below",
            ],
            "category_fee_k": {**FEE_K_BY_CATEGORY, "sports_*": FEE_K_SPORTS,
                               "_default": FEE_K_DEFAULT},
            "taker_fee_k_stress": settings["taker_fee_k_stress"],
            "maker_fee": 0.0,
            "reference": ("docs/polymarket_mechanics.md — fees are LIVE since "
                          "2026-01-05 and verified on-chain to 7 s.f.; read "
                          "feeSchedule.rate, NEVER taker_base_fee (identically 1000 "
                          "across categories whose real rates are 0.04/0.05/0.07)."),
            "known_conservatism": ("classify_market has no `geopolitics` label, so a "
                                   "geopolitics market whose true rate is 0 falls "
                                   "into `other` and is charged 0.05 whenever the "
                                   "live feeSchedule is unavailable. That can only "
                                   "make this arm look worse, never better."),
            "tick_source": "the token's own tick_size from GET /book (NOT a constant)",
            "slippage": "walk the live book level by level; no assumed top-of-book size",
        },
        "seatbelts": {
            "paper_only": True, "keys": "none", "signing": "none",
            "order_placement": "none — public unauthenticated GETs only",
            "kill_switch_paths": kill_switch_labels(cfg),
            "max_notional_per_position_usd": settings["max_notional_per_position_usd"],
            "max_open_positions": settings["max_open_positions"],
            "dry_run_default": True,
            "writes_only": [_repo_relative(p) for p in
                            (RW_DIR, FREEZE_MANIFEST_PATH, WATCHLIST_PATH,
                             SCOREBOARD_MD, SCOREBOARD_PARQUET)],
        },
        "placebo": {
            "seed": f"paper_rw_placebo_v1:{ts}",
            "draw": ("u = SHA256(seed || signal_id)[:8] / 2**64, "
                     "trigger = detect + u * (horizon - detect)"),
            "horizon": ("first of game_start_time, Gamma's precise endDate and the "
                        "CLOB's end_date_iso that is in the FUTURE at detection, else "
                        "the fallback below. The CLOB's end_date_iso alone is "
                        "unusable: src/market_meta.py documents it as the midnight "
                        "FLOOR of the scheduled end day."),
            "fallback_horizon_hours": settings["placebo_fallback_horizon_hours"],
            "fallback_rationale": ("real-world markets live for days to months, so "
                                   "the live arm's 6h fallback would draw the "
                                   "'random moment' from the first 6 hours of the "
                                   "market's life, which is a delayed copy rather "
                                   "than a control."),
        },
        "copy_lag": {
            "measured": True,
            "columns": ["detect_lag_s", "fill_lag_s"],
            "definition": ("detect_lag_s = detect_ts - wallet_trade_ts; fill_lag_s = "
                           "first_fill_ts - wallet_trade_ts. The ACHIEVED "
                           "distribution is reported per arm; no lag is assumed."),
            "poll_seconds_at_freeze": settings["poll_seconds"],
        },
        "evidence_for_these_five": {
            "source": "docs/realworld_deep_sample.md",
            "gate": ("Project 1 gate on a deep, performance-blind probability sample: "
                     "candidacy -> cluster-robust market-block bootstrap at alpha "
                     "0.005 -> 2c magnitude floor -> >=30 held-out markets"),
            "set_level_measured_fdr": 0.173,
            "expected_genuine_of_five": round(5 * (1 - 0.173), 2),
            "caveat": ("IDENTIFICATION is what that evidence establishes — whose bets "
                       "beat the price they paid. COPYABILITY is what this arm tests "
                       "and it is not a corollary. Every copy thesis in this repo has "
                       "come back negative or thin."),
        },
        "funnel": funnel,
        "forward_observations_at_freeze": obs,
        "amendments": amendments,
        "known_limits": [
            "POLLING IS DISCRETE. Fills are decided from book snapshots taken at the "
            "poll cadence, so a transient offer between polls is missed (under-counts "
            "fills) while a resting bid could in reality be hit by flow no snapshot "
            "shows. Not signed. It is also why the achieved copy lag is MEASURED and "
            "reported rather than assumed.",
            "MARKET IMPACT IS NOT MODELLED. A real $100 order joins the queue and can "
            "move a thin book; here it consumes displayed size without disturbing it. "
            "A one-directional optimism, small at $100 on a deep book and NOT small "
            "on the thin real-world books this arm will often meet.",
            "QUEUE POSITION IS NOT MODELLED, so limit_noChase's fill rate is an upper "
            "bound. That is exactly why market_chase, not the limit arm, is the "
            "headline for an actionability question.",
            "THE FEE k IS LIVE-READ PER MARKET but Gamma reports TODAY's schedule, "
            "not the one in force when the market opened. fee_k_source is persisted "
            "per signal so the fallback rate is never mistaken for a measured one.",
            "REAL-WORLD MARKETS RESOLVE SLOWLY — weeks to months. This scoreboard "
            "will read AWAITING DATA for a long time, and that is not a zero.",
            "FIVE WALLETS IS FIVE WALLETS. The measured 17.3% set-level FDR says ~1 "
            "of these 5 is expected to be a false discovery, and a per-wallet row "
            "with two events is not evidence about that wallet.",
            "THE US IS CLOSE-ONLY on polymarket.com (docs/polymarket_mechanics.md), "
            "so even a positive result here does not by itself make this actionable "
            "for the owner.",
            "ONE OF THE FIVE IS AN IN-PLAY TRADER AND WILL DOMINATE THE POOLED ROW. A "
            "pre-freeze probe of the last 7 days (no forward data, no scoring) found "
            "0x83255595ba placing ~3,200 buys into ~10 live football over/under "
            "markets, i.e. ~85% of all signals from ~2% of the distinct markets. "
            "'Real-world' is not the same as 'slow': an in-play line moves in seconds "
            "and a 5-minute copy lag is fatal to it. This is exactly why the "
            "per-wallet strata are pre-registered and why the achieved copy lag is "
            "measured. Read the per-wallet rows and effective_events before the "
            "pooled row, always.",
            "THE CATEGORY LABEL IS COARSE. classify_market parks ~half of the "
            "real-world tape in `other` (docs/realworld_deep_sample.md §3), including "
            "geopolitics, golf, cycling and unknown league codes. That is a "
            "GRANULARITY gap, not a micro-crypto leak — the micro_crypto rule keys on "
            "the slug shape and is unaffected — but it does mean the category fee "
            "fallback is coarse, which is why the LIVE feeSchedule takes precedence.",
            "THE TWO TAKER ARMS BRACKET A REAL COPIER; NEITHER IS ONE. market_chase "
            "is an UNCAPPED market order and will walk a thin book to whatever price "
            "is resting there — a pre-freeze probe saw a $100 take average 9.6c on a "
            "market where the wallet had paid 0.8c — so it is a LOWER bound on a "
            "careful copier. limit_noChase never pays more than they did but assumes "
            "queue priority it would not have, so it is an UPPER bound. The obvious "
            "refinement is a third taker arm with a max-slippage cap relative to the "
            "best ask; it was deliberately NOT added here so that the arm "
            "construction stays byte-identical to the running experiment's, and the "
            "mean fill price versus the wallet's own entry price is reported per arm "
            "so the gap can never be invisible.",
            "A RESTING LIMIT CAN FILL AFTER THE EVENT IS DECIDED but before the market "
            "resolves, because still-live orders are only terminated when `score` "
            "runs. The direction is CONSERVATIVE — such a fill is adversely selected, "
            "it is the losing side being dumped onto our bid — and fill_lag_s makes it "
            "visible. Run `score` on a cadence that keeps the settled-order tail short.",
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
            f"`python -m src.paper_rw freeze` first.")
    with open(FREEZE_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return pd.read_parquet(WATCHLIST_PATH), manifest


def rule_from_manifest(manifest: dict, cfg: dict) -> dict:
    """The economic rule comes from the FROZEN manifest; only the operational caps
    come from live config, and only so they can be TIGHTENED. This is what stops
    the rule drifting after the data arrives."""
    settings = rw_settings(cfg)
    return {
        "notional_per_signal_usd": float(manifest["stake"]["notional_per_signal_usd"]),
        "max_notional_per_position_usd": min(
            float(manifest["seatbelts"]["max_notional_per_position_usd"]),
            settings["max_notional_per_position_usd"]),
        "max_open_positions": min(int(manifest["seatbelts"]["max_open_positions"]),
                                  settings["max_open_positions"]),
        # kept for pt.open_orders_for_signal / pt.advance_order compatibility; this
        # arm's fee is per-market and lives on the order rows, not in the rule
        "taker_fee_k": 0.0,
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
    """Poll the five frozen wallets and progress this arm's paper book.

    Idempotent and resumable: cursors, signals and orders are persisted after every
    pass, so a restart resumes rather than re-trades. Never crash-loops — a failing
    wallet, book or Gamma lookup is logged and the cycle continues."""
    cfg = cfg or load_config()
    watchlist, manifest = load_freeze()
    rule = rule_from_manifest(manifest, cfg)
    settings = rw_settings(cfg)

    if kill_switch_engaged(cfg):
        print(f"[paper_rw] KILL SWITCH ENGAGED "
              f"({', '.join(kill_switch_labels(cfg))}) — halting before any fetch "
              f"or write.")
        return {"halted": True, "reason": "kill_switch"}

    session = make_session(max_retries=cfg["ingest"]["max_retries"],
                           backoff=cfg["ingest"]["retry_backoff_sec"])
    signals, orders = load_ledger()
    state = load_state()
    print(f"[paper_rw] freeze {manifest['freeze_utc']} "
          f"({manifest['git_commit'][:8]}) — {len(watchlist)} wallets, "
          f"{len(signals)} signals, "
          f"{int(orders['status'].isin(pt.LIVE_STATUSES).sum()) if len(orders) else 0} "
          f"live orders. "
          f"{'DRY RUN (nothing will be written)' if dry_run else 'COMMIT'}")

    total = {}
    for i in range(max(1, passes)):
        if kill_switch_engaged(cfg):
            print("[paper_rw] KILL SWITCH ENGAGED mid-run — stopping.")
            break
        try:
            signals, orders, state, stats = poll_once(
                session, cfg, rule, watchlist, signals, orders, state,
                int(time.time()))
        except Exception as exc:  # noqa: BLE001 — never crash-loop
            print(f"[paper_rw] WARNING: poll cycle failed ({exc}); retrying next pass")
            time.sleep(min(settings["poll_seconds"], 60))
            continue
        total = stats
        print(f"[paper_rw] pass {i+1}/{passes}: {stats}")
        if not dry_run:
            save_ledger(signals, orders, state)
        if i < passes - 1:
            time.sleep(settings["poll_seconds"])

    if dry_run:
        print("[paper_rw] DRY RUN — no ledger written. Re-run with --commit.")
    live = int(orders["status"].isin(pt.LIVE_STATUSES).sum()) if len(orders) else 0
    return {"halted": False, "signals": len(signals), "orders": len(orders),
            "live": live, "stats": total}


# ---------------------------------------------------------------------------
# SCORE
# ---------------------------------------------------------------------------

def _lag_stats(sub: pd.DataFrame, col: str) -> dict:
    vals = pd.to_numeric(sub.get(col), errors="coerce").dropna() if col in sub else pd.Series(dtype=float)
    if vals.empty:
        return {f"{col}_median": float("nan"), f"{col}_p25": float("nan"),
                f"{col}_p75": float("nan"), f"{col}_max": float("nan")}
    return {
        f"{col}_median": float(vals.median()),
        f"{col}_p25": float(vals.quantile(0.25)),
        f"{col}_p75": float(vals.quantile(0.75)),
        f"{col}_max": float(vals.max()),
    }


def _arm_row(sub: pd.DataFrame, arm: str, stratum: str, n_boot: int) -> dict:
    """One scoreboard row. `n_signals`, `n_filled`, `fill_rate`, `n_events` and
    `effective_events` sit beside every dollar figure on purpose — a dollar figure
    without them has burned this repo before. GROSS and NET are both here because
    the running arm froze k = 0 and the whole point of this one is that fees are
    real."""
    boot = pt.event_boot(sub["pnl_usd_net"].to_numpy(), sub["event_unit"].to_numpy(),
                         n_boot=n_boot)
    boot_gross = pt.event_boot(sub["pnl_usd_gross"].to_numpy(),
                               sub["event_unit"].to_numpy(), n_boot=n_boot, seed=4242)
    notional = float(sub["cost_usd"].sum())
    net = float(sub["pnl_usd_net"].sum())
    row = {
        "arm": arm, "stratum": stratum,
        "n_signals": int(len(sub)),
        "n_filled": int(sub["filled"].sum()),
        "fill_rate": float(sub["filled"].mean()),
        "n_events": boot["n_events"],
        "effective_events": boot["effective_events"],
        "notional_filled_usd": notional,
        "total_pnl_usd_gross": float(sub["pnl_usd_gross"].sum()),
        "total_pnl_usd_net": net,
        "total_pnl_usd_net_stress": float(sub["pnl_usd_net_stress"].sum()),
        "total_fee_usd": float(sub["fee_usd"].sum()),
        "return_on_filled_notional_net": (net / notional if notional > 0 else float("nan")),
        "pnl_per_signal_usd_gross": float(sub["pnl_usd_gross"].mean()),
        "pnl_per_signal_usd_net": float(sub["pnl_usd_net"].mean()),
        "pnl_per_signal_usd_net_event_wtd": boot["mean"],
        "pnl_per_signal_usd_gross_event_wtd": boot_gross["mean"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "p_one_sided": boot["p_one_sided"],
        "mean_fill_price": (float(sub.loc[sub["filled"], "avg_fill_price"].mean())
                            if sub["filled"].any() else float("nan")),
        "mean_fee_k": float(pd.to_numeric(sub["fee_k"], errors="coerce").mean()),
    }
    filled = sub[sub["filled"]]
    # What the copy actually cost RELATIVE TO THE WALLET. A market order into a thin
    # book can pay many times what they paid, and that is the single most likely way
    # this arm comes back negative. It is not allowed to be invisible.
    if len(filled):
        wp = pd.to_numeric(filled["wallet_entry_price"], errors="coerce")
        fp = pd.to_numeric(filled["avg_fill_price"], errors="coerce")
        row["mean_wallet_entry_price"] = float(wp.mean())
        row["mean_slippage_vs_wallet"] = float((fp - wp).mean())
    else:
        row["mean_wallet_entry_price"] = float("nan")
        row["mean_slippage_vs_wallet"] = float("nan")
    row.update(_lag_stats(sub, "detect_lag_s"))
    row.update(_lag_stats(filled, "fill_lag_s"))
    return row


def score_orders(orders: pd.DataFrame, signals: pd.DataFrame,
                 resolutions: pd.DataFrame, rule: dict, n_boot: int = 2000) -> dict:
    """Turn this arm's paper ledger into dollars — GROSS and NET — per arm, at the
    event unit, pooled and per wallet.

    Every arm sees the same traded signals, so no-fills cannot be hidden: an
    unfilled order scores exactly $0 on $0 of notional and still counts in
    n_signals."""
    skipped = {}
    if len(signals):
        counts = signals["status"].value_counts()
        skipped = {k: int(v) for k, v in counts.items()}
    if orders.empty:
        return {"board": pd.DataFrame(), "contrast": None, "n_unresolved": 0,
                "signal_status_counts": skipped,
                "note": "no paper orders recorded yet"}

    res = (resolutions[["market_id", "token_id", "resolved", "resolved_value"]].copy()
           if len(resolutions) else
           pd.DataFrame(columns=["market_id", "token_id", "resolved", "resolved_value"]))
    df = orders.merge(res, on=["market_id", "token_id"], how="left")
    df["resolved"] = df["resolved"].fillna(False).astype(bool)

    n_unresolved_signals = int(df.loc[~df["resolved"], "signal_id"].nunique())
    scored = df[df["resolved"]].copy()
    if scored.empty:
        return {"board": pd.DataFrame(), "contrast": None,
                "n_unresolved": n_unresolved_signals,
                "signal_status_counts": skipped,
                "note": "signals captured, none of their markets resolved yet"}

    scored["shares"] = scored["shares"].astype(float).fillna(0.0)
    scored["cost_usd"] = scored["cost_usd"].astype(float).fillna(0.0)
    scored["fee_units"] = scored["fee_units"].astype(float).fillna(0.0)
    scored["resolved_value"] = scored["resolved_value"].astype(float)
    # A missing per-market k must NEVER become a free trade: fall back to the
    # frozen category map, which is the same rule the poller applied.
    k = pd.to_numeric(scored["fee_k"], errors="coerce")
    k = k.fillna(scored["category"].map(fee_k_for_category)).fillna(FEE_K_DEFAULT)
    scored["fee_k"] = k
    taker = scored["is_taker"].fillna(False).astype(bool)
    scored["fee_usd"] = np.where(taker, k * scored["fee_units"], 0.0)
    scored["fee_usd_stress"] = np.where(
        taker, rule["taker_fee_k_stress"] * scored["fee_units"], 0.0)
    payout = scored["shares"] * scored["resolved_value"]
    scored["pnl_usd_gross"] = payout - scored["cost_usd"]
    scored["pnl_usd_net"] = scored["pnl_usd_gross"] - scored["fee_usd"]
    scored["pnl_usd_net_stress"] = scored["pnl_usd_gross"] - scored["fee_usd_stress"]
    scored["filled"] = scored["shares"] > 0

    meta = (signals[["signal_id", "slug", "question", "market_id"]]
            .drop_duplicates("signal_id") if len(signals) else
            pd.DataFrame(columns=["signal_id", "slug", "question", "market_id"]))
    if len(meta):
        scored = scored.drop(columns=["slug", "question"], errors="ignore").merge(
            meta, on=["signal_id", "market_id"], how="left")
    scored["event_unit"] = pt.assign_event_units(scored)

    # STRATA, pre-registered in the manifest: the copy-validated headline, the
    # five-wallet pool, and exactly one row per wallet. No other slicing.
    strata = {
        COPY_VALIDATED_STRATUM: scored["wallet_label"].isin(COPY_VALIDATED_WALLETS),
        LEGACY_STRATUM: scored["wallet_label"].isin(LEGACY_WALLETS),
        POOLED_STRATUM: pd.Series(True, index=scored.index),
    }
    for label in sorted(scored["wallet_label"].dropna().unique()):
        strata[str(label)] = scored["wallet_label"] == label

    rows, per_event = [], {}
    for stratum, mask in strata.items():
        for arm in ARMS:
            sub = scored[mask & (scored["arm"] == arm)]
            if sub.empty:
                continue
            rows.append(_arm_row(sub, arm, stratum, n_boot))
            if stratum == POOLED_STRATUM:
                per_event[arm] = sub.groupby("event_unit")["pnl_usd_net"].mean()
    board = pd.DataFrame(rows)

    contrast = None
    if HEADLINE_ARM in per_event and CONTROL_ARM in per_event:
        a, b = per_event[HEADLINE_ARM], per_event[CONTROL_ARM]
        shared = a.index.intersection(b.index)
        if len(shared):
            diff = (a.loc[shared] - b.loc[shared]).to_numpy()
            boot = pt.event_boot(diff, np.asarray(shared), n_boot=n_boot, seed=999)
            contrast = {
                "comparison": f"{HEADLINE_ARM} - {CONTROL_ARM} (net)",
                "shared_events": int(len(shared)),
                "mean_usd_per_signal": boot["mean"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "p_one_sided": boot["p_one_sided"],
                "effective_events": boot["effective_events"],
            }

    nf = scored[~scored["filled"]].copy()
    if "no_fill_reason" not in nf.columns:
        nf["no_fill_reason"] = None
    reasons = ((nf.groupby(["arm", nf["no_fill_reason"].fillna("unrecorded")])
                .size().rename("n").reset_index()) if len(nf) else pd.DataFrame())
    fee_sources = (scored.groupby("fee_k_source").size().rename("n").reset_index()
                   if "fee_k_source" in scored else pd.DataFrame())
    return {"board": board, "contrast": contrast, "n_unresolved": n_unresolved_signals,
            "signal_status_counts": skipped, "note": None, "scored": scored,
            "no_fill_reasons": reasons, "fee_sources": fee_sources}


def score(cfg: dict | None = None, refetch: bool = True, n_boot: int = 2000) -> dict:
    """Refresh resolutions for every market in this arm's ledger, score it, and
    write the scoreboard. Read-only against the network; writes only this arm's
    artifacts."""
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
        resolutions = update_resolutions(session, cfg,
                                         set(orders["market_id"].unique()), resolutions)
        RW_DIR.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(resolutions, RESOLUTIONS_PATH, compression="gzip")

    settled = (set(resolutions.loc[resolutions["resolved"].fillna(False), "market_id"])
               if len(resolutions) else set())
    orders = pt.terminate_settled(orders, settled, int(time.time()))
    orders = copy_lag_columns(orders)
    if len(orders) and refetch:
        atomic_to_parquet(orders, ORDERS_PATH, compression="gzip")

    out = score_orders(orders, signals, resolutions, rule, n_boot=n_boot)
    write_scoreboard(out, manifest, watchlist, signals, orders)
    return out


def _hms(seconds) -> str:
    """A lag in seconds, printed as something a human can read at a glance."""
    if seconds is None or (isinstance(seconds, float) and not np.isfinite(seconds)):
        return "—"
    s = int(round(float(seconds)))
    sign = "-" if s < 0 else ""
    s = abs(s)
    if s < 60:
        return f"{sign}{s}s"
    if s < 3600:
        return f"{sign}{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{sign}{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{sign}{s // 86400}d{(s % 86400) // 3600:02d}h"


def write_scoreboard(out: dict, manifest: dict, watchlist: pd.DataFrame,
                     signals: pd.DataFrame, orders: pd.DataFrame) -> None:
    """Write data/processed/paper_rw_scoreboard.{md,parquet}.

    An empty table reads as a zero, so when there is nothing to score this says
    AWAITING DATA in the loudest place on the page."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    board = out.get("board", pd.DataFrame())
    atomic_to_parquet(board if len(board) else
                      pd.DataFrame({"arm": pd.Series(dtype=object)}),
                      SCOREBOARD_PARQUET, compression="gzip")

    fmt = pt._fmt
    status = out.get("signal_status_counts", {}) or {}
    n_seen = len(signals)
    n_traded = sum(v for k, v in status.items() if k in TRADED_STATUSES)
    live = int(orders["status"].isin(pt.LIVE_STATUSES).sum()) if len(orders) else 0

    lines = [
        "# Forward paper trade — the five certified REAL-WORLD wallets "
        "(PAPER ONLY: no keys, no orders)",
        "",
        f"**Frozen {manifest['freeze_utc']}** at commit "
        f"`{manifest['git_commit'][:8]}`. Watchlist: {len(watchlist)} wallets, "
        f"real-world markets only. Stake: "
        f"${manifest['stake']['notional_per_signal_usd']:.2f} notional per signal "
        f"per arm. Headline arm: `{manifest['headline_arm']}`; control: "
        f"`{manifest['control_arm']}`.",
        "",
        "> This arm is **independent of** the running 38-wallet paper trader "
        "(`data/processed/paper_scoreboard.md`), which it shares no state with.",
        "",
        f"Signals seen: **{n_seen}** · traded (real-world): **{n_traded}** · "
        f"live orders: **{live}** · signals whose market has not resolved: "
        f"**{out.get('n_unresolved', 0)}**.",
        "",
    ]
    if status:
        lines += ["Signal disposition at detection: "
                  + " · ".join(f"`{k}` {v}" for k, v in sorted(status.items())),
                  "",
                  "> `skipped_micro_crypto` is the REAL-WORLD FILTER doing its job — "
                  "those signals are excluded from every dollar figure by "
                  "pre-registration. `skipped_unclassified` means neither the tape "
                  "nor Gamma gave a slug to classify on, and is skipped rather than "
                  "assumed real-world. `no_book_at_detection` still opens all three "
                  "arms (the resting limit can fill later); `skipped_book_error` "
                  "opens nothing; `skipped_cap` is the seatbelt refusing a signal.",
                  ""]
        horizons = (signals["placebo_horizon_source"].dropna()
                    if "placebo_horizon_source" in signals else pd.Series(dtype=object))
        if len(horizons):
            lines += ["Placebo horizon source: "
                      + " · ".join(f"`{k}` {v}" for k, v in
                                   horizons.value_counts().items()), ""]
        fk = out.get("fee_sources")
        if fk is not None and len(fk):
            lines += ["Fee `k` source: "
                      + " · ".join(f"`{r['fee_k_source']}` {int(r['n'])}"
                                   for _, r in fk.iterrows()), ""]

    if out.get("note") or not len(board):
        lines += [
            f"> ## ⏳ AWAITING DATA — {out.get('note') or 'nothing scored yet'}.",
            ">",
            "> This is **not** a zero and **not** a negative result. Real-world "
            "markets resolve in weeks to months, so this page is expected to stay "
            "here for a long time. Read `effective_events` before reading any dollar "
            "figure.",
            "",
        ]
    else:
        headline_stratum = manifest["strata"].get("headline", manifest["strata"]["pooled"])
        pooled = board[board["stratum"] == headline_stratum]
        lines += [
            f"## `{headline_stratum}` — the headline",
            "",
            ("The two wallets whose edge survived a measured 2-minute copy lag on past "
             "data (docs/copy_verdict.md). The five-wallet pool and every individual "
             "wallet are reported below, unchanged and undropped — three of the five "
             "were predicted BEFORE any data to fail forward, which is what makes this "
             "falsifiable rather than a quiet exclusion."
             if headline_stratum != manifest["strata"]["pooled"] else ""),
            "",
            "| arm | signals | filled | **fill rate** | events | eff. events | "
            "notional | gross $/sig | **net $/sig** | event-wtd net | 95% CI | p | "
            "total net | net stress k | return on filled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            r = pooled[pooled["arm"] == arm]
            if not len(r):
                lines.append(f"| {arm} | 0 | 0 | — | 0 | — | — | — | — | — | — | — | "
                             f"— | — | — |")
                continue
            r = r.iloc[0]
            head = "**" if arm == manifest["headline_arm"] else ""
            ci = ("—" if pd.isna(r["ci_low"])
                  else f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
            lines.append(
                f"| {head}{arm}{head} | {int(r['n_signals'])} "
                f"| {int(r['n_filled'])} | **{r['fill_rate']:.1%}** "
                f"| {int(r['n_events'])} | {fmt(r['effective_events'], '.2f')} "
                f"| ${fmt(r['notional_filled_usd'], ',.2f')} "
                f"| ${fmt(r['pnl_per_signal_usd_gross'], '+,.3f')} "
                f"| **${fmt(r['pnl_per_signal_usd_net'], '+,.3f')}** "
                f"| ${fmt(r['pnl_per_signal_usd_net_event_wtd'], '+,.3f')} "
                f"| {ci} | {fmt(r['p_one_sided'], '.3f')} "
                f"| ${fmt(r['total_pnl_usd_net'], '+,.2f')} "
                f"| ${fmt(r['total_pnl_usd_net_stress'], '+,.2f')} "
                f"| {fmt(r['return_on_filled_notional_net'], '+.2%')} |")
        lines += [
            "",
            "`net $/sig` is the headline: **net dollars per DETECTED (traded) signal, "
            "with unfilled signals contributing $0.** An arm that fills rarely earns "
            "proportionally less here, by construction. Never quote "
            "`return on filled` without the fill rate beside it. Read the ABSOLUTE "
            "per-arm numbers first (that is the SELECTION test) and the difference "
            "second (that is the TIMING test).",
            "",
            "### What the copy actually paid, versus what the wallet paid",
            "",
            "| arm | mean fill price | their mean entry price | **mean slippage** |",
            "|---|---:|---:|---:|",
        ]
        for arm in ARMS:
            r = pooled[pooled["arm"] == arm]
            if not len(r):
                continue
            r = r.iloc[0]
            lines.append(
                f"| {arm} | {fmt(r['mean_fill_price'], '.4f')} "
                f"| {fmt(r['mean_wallet_entry_price'], '.4f')} "
                f"| **{fmt(r['mean_slippage_vs_wallet'], '+.4f')}** |")
        lines += [
            "",
            "> ⚠️ **The two taker arms bracket the truth; neither is a copier.** "
            "`market_chase` is an **uncapped** market order: it walks a thin book to "
            "whatever price is there, so on an illiquid longshot it can pay several "
            "times what the wallet paid. A real copier would refuse that, so its "
            "dollars are a **LOWER bound**. `limit_noChase` never pays more than they "
            "did, but assumes queue priority it would not have, so its dollars are an "
            "**UPPER bound**. A careful copier with a slippage cap sits between them. "
            "Read the bracket, not either end alone.",
            "",
            "### Achieved copy lag (measured, not assumed)",
            "",
            "| arm | detect lag median | detect p75 | detect max | "
            "**fill lag median** | fill p75 | fill max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            r = pooled[pooled["arm"] == arm]
            if not len(r):
                continue
            r = r.iloc[0]
            lines.append(
                f"| {arm} | {_hms(r['detect_lag_s_median'])} "
                f"| {_hms(r['detect_lag_s_p75'])} | {_hms(r['detect_lag_s_max'])} "
                f"| **{_hms(r['fill_lag_s_median'])}** "
                f"| {_hms(r['fill_lag_s_p75'])} | {_hms(r['fill_lag_s_max'])} |")
        lines += [
            "",
            "`detect lag` is their trade → our detection (the poll cadence plus the "
            "API's own lag); `fill lag` is their trade → our first filled share. A "
            "`limit_noChase` fill hours later is **not the trade the wallet made**, "
            "and no dollar figure should be read without this table beside it.",
            "",
            "### Per wallet — the breadwinner cut (pre-registered)",
            "",
            "| wallet | arm | signals | filled | fill rate | eff. events | "
            "net $/signal | 95% CI | total net |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label in manifest["strata"]["per_wallet"]:
            sub = board[board["stratum"] == label]
            if not len(sub):
                lines.append(f"| `{label}` | — | 0 | 0 | — | — | — | — | — |")
                continue
            for arm in ARMS:
                r = sub[sub["arm"] == arm]
                if not len(r):
                    continue
                r = r.iloc[0]
                ci = ("—" if pd.isna(r["ci_low"])
                      else f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
                lines.append(
                    f"| `{label}` | {arm} | {int(r['n_signals'])} "
                    f"| {int(r['n_filled'])} | {r['fill_rate']:.1%} "
                    f"| {fmt(r['effective_events'], '.2f')} "
                    f"| ${fmt(r['pnl_per_signal_usd_net'], '+,.3f')} | {ci} "
                    f"| ${fmt(r['total_pnl_usd_net'], '+,.2f')} |")
        lines += [
            "",
            "The owner's question is about individual breadwinners, so these rows are "
            "reported always — including when they are too small to read. With five "
            "wallets and a measured set-level FDR of 17.3%, **about one of these five "
            "is expected to be a false discovery**; a per-wallet row with two events "
            "cannot tell you which.",
            "",
        ]
        reasons = out.get("no_fill_reasons")
        if reasons is not None and len(reasons):
            lines += ["### Why the no-fills did not fill", "",
                      "| arm | reason | n |", "|---|---|---:|"]
            for _, rr in reasons.iterrows():
                lines.append(f"| {rr['arm']} | `{rr.iloc[1]}` | {int(rr['n'])} |")
            lines += ["", "A high `settled_before_trigger` count on `placebo_random` "
                      "means the CONTROL is broken, not that the thesis is good — "
                      "stop reading the contrast at that point.", ""]

        c = out.get("contrast")
        lines += ["## The timing test (read AFTER the absolute numbers)", ""]
        if c:
            ci = ("—" if c["ci_low"] is None or pd.isna(c["ci_low"])
                  else f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")
            lines += [
                f"**{c['comparison']}** on {c['shared_events']} shared events "
                f"(effective {c['effective_events']:.2f}): "
                f"**${c['mean_usd_per_signal']:+,.3f} per signal**, 95% CI {ci}, "
                f"one-sided p = {fmt(c['p_one_sided'], '.3f')}.",
                "",
                "This difference tests **TIMING** — both arms take the wallet's "
                "market and side, so the only thing that differs is when. The "
                "**SELECTION** test is `placebo_random`'s absolute net dollars in the "
                "pooled table above. The repo's live prior is difference ≈ 0 with the "
                "control positive: a market-discovery signal, not a timing signal. "
                "Difference ≈ 0 is therefore **not** a failure of the copy thesis on "
                "its own.",
                "",
            ]
        else:
            lines += ["Not computable yet — needs scored signals in both the headline "
                      "and the control arm.", ""]

    lines += ["## The pre-registered hypothesis, verbatim from the manifest", "",
              "> " + manifest["hypothesis"].replace("\n", "\n>\n> "), "",
              "## The pre-registered entry rule, verbatim from the manifest", "",
              "> " + manifest["entry_rule"].replace("\n", "\n>\n> "), "",
              "## The pre-registered scoring rule, verbatim from the manifest", "",
              "> " + manifest["scoring_rule"].replace("\n", "\n>\n> "), "",
              "## Seatbelts", ""]
    sb = manifest["seatbelts"]
    lines += [
        "- Paper only; **no keys, no signing, no order placement** — public "
        "unauthenticated GETs only.",
        "- Kill switches (either one halts this arm before any fetch or write): "
        + " · ".join(f"`touch {p}`" for p in sb["kill_switch_paths"]) + ".",
        f"- Per-position notional cap: ${sb['max_notional_per_position_usd']:.2f}.",
        f"- Max simultaneously-live orders: {sb['max_open_positions']} "
        f"(signals over the cap are recorded as `skipped_cap` and still counted).",
        "- `--dry-run` is the default; persisting requires an explicit `--commit`.",
        "- Writes only: " + ", ".join(f"`{p}`" for p in sb["writes_only"]) + ".",
        "",
        "## Known limits, recorded at freeze time", "",
    ]
    lines += [f"- {lim}" for lim in manifest["known_limits"]]
    SCOREBOARD_MD.write_text("\n".join(lines) + "\n")
    print(f"[paper_rw] -> {SCOREBOARD_MD}\n[paper_rw] -> {SCOREBOARD_PARQUET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=("Forward PAPER trader, REAL-WORLD arm (execution roadmap stage "
                     "1b). PAPER ONLY: no keys, no signing, no order placement, "
                     "public GETs only."))
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="pre-register the rule and the five wallets")
    f.add_argument("--force", action="store_true",
                   help="overwrite an existing freeze (records an amendment)")
    f.add_argument("--reason", type=str, default=None)

    r = sub.add_parser("run", help="poll the five wallets and progress the paper book")
    r.add_argument("--commit", action="store_true",
                   help="persist the ledger (WITHOUT this, --dry-run is the default)")
    r.add_argument("--dry-run", action="store_true", help="explicit no-op default")
    r.add_argument("--passes", type=int, default=1)

    s = sub.add_parser("score", help="score this arm's ledger and write its scoreboard")
    s.add_argument("--no-refetch", action="store_true")
    s.add_argument("--boot", type=int, default=2000)

    args = ap.parse_args()
    cfg = load_config()

    if args.cmd == "freeze":
        m = freeze(cfg, force=args.force, amend_reason=args.reason)
        print(f"[paper_rw] FROZEN {m['freeze_utc']} @ {m['git_commit'][:8]}")
        for w in m["wallets"]:
            print(f"    {w['label']}  oos_resid {w['out_of_sample_residual_edge']:+.4f}"
                  f"  markets {w['out_of_sample_markets']}"
                  f"  p {w['out_of_sample_cluster_p']:.5f}")
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
