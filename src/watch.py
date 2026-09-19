"""Stage 6 — real-time monitor. See CLAUDE.md "Stage 6".

Watches the top `watch.top_n_wallets` ranked wallets for new BUY positions and
alerts (stdout + data/alerts.log + optional webhook) while the current market
price is still inside that wallet's historical profitable copy-window. It is
read-only against public data plus outbound notifications. See
DECISIONS.md "src/watch.py" for the current-price proxy and cursor design.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from src.common import (
    ALERTS_LOG_PATH,
    RANKED_WALLETS_PATH,
    WATCH_STATE_PATH,
    ensure_dirs,
    load_config,
    make_session,
)

TRADES_FETCH_LIMIT = 25  # recent trades to pull per wallet per poll
MAX_BACKOFF_SECONDS = 1800  # cap on error backoff so it never sleeps forever


# --- trade fetch/normalization ---------------------------------------------

def normalize_trade(raw: dict) -> dict:
    """Rename a raw /trades API row to this codebase's column names — same
    mapping as ingest.py's fold_trades_to_ledger, so a "position" here means
    the same thing as a ledger row."""
    return {
        "wallet": raw["proxyWallet"],
        "market_id": raw["conditionId"],
        "token_id": raw["asset"],
        "outcome": raw.get("outcome"),
        "side": raw["side"],
        "entry_price": float(raw["price"]),
        "size": float(raw["size"]),
        "timestamp": int(raw["timestamp"]),
        "question": raw.get("title"),
        "slug": raw.get("slug"),
        "tx_hash": raw["transactionHash"],
    }


def fetch_wallet_trades(session, cfg: dict, wallet: str, limit: int) -> list[dict]:
    """Most recent trades for one wallet, newest-first (verified empirically —
    same ordering as the global feed). `user=` is a working server-side
    filter on this endpoint (unlike `after`/`before`, see DECISIONS.md)."""
    base = cfg["data_source"]["trades_api_base"]
    resp = session.get(
        f"{base}/trades",
        params={"user": wallet, "limit": limit},
        timeout=cfg["ingest"]["request_timeout_sec"],
    )
    resp.raise_for_status()
    return [normalize_trade(t) for t in resp.json()]


def fetch_current_price(session, cfg: dict, token_id: str) -> float | None:
    """Current market price proxy for one outcome token, via the CLOB's
    authoritative last-trade-price endpoint (see DECISIONS.md for why this is
    used here instead of the trade-tape proxy features.py uses for
    historical fair value)."""
    base = cfg["data_source"]["clob_api_base"]
    resp = session.get(
        f"{base}/last-trade-price",
        params={"token_id": token_id},
        timeout=cfg["ingest"]["request_timeout_sec"],
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    price = resp.json().get("price")
    return float(price) if price is not None else None


# --- last-seen state / new-position detection -------------------------------

def position_key(trade: dict) -> str:
    return f"{trade['wallet']}:{trade['market_id']}:{trade['token_id']}:{trade['side']}:{trade['tx_hash']}"


def default_cursor(now_ts: int) -> dict:
    """A wallet's cursor when first watched: floor is "now", so pre-existing
    history already in the ranked table is never (re-)alerted — only
    positions taken from this moment forward."""
    return {"max_timestamp": now_ts, "keys_at_max": []}


def new_trades_since(trades: list[dict], cursor: dict) -> list[dict]:
    """Same overlap-safe logic as ingest.py's fetch_new_trades: trades below
    the cursor's floor are old; trades exactly at the floor are old only if
    already recorded in keys_at_max (handles multiple trades sharing a
    timestamp, which is common on this feed)."""
    floor_ts = cursor["max_timestamp"]
    seen_at_max = set(cursor.get("keys_at_max", []))
    fresh = []
    for trade in trades:
        if trade["timestamp"] < floor_ts:
            continue
        if trade["timestamp"] == floor_ts and position_key(trade) in seen_at_max:
            continue
        fresh.append(trade)
    return fresh


def advance_cursor(cursor: dict, trades: list[dict]) -> dict:
    """Move the cursor to the newest timestamp seen in this poll's fetched
    page (not just the fresh subset) — this is what makes each position
    alert at most once even across restarts. Never moves the cursor
    *backward*: a wallet's first poll can fetch a page that's entirely older
    than its default_cursor("now") floor (its most recent trade predates
    joining the watchlist), and that must not un-skip that old history on a
    later poll."""
    if not trades:
        return cursor
    max_ts = max(t["timestamp"] for t in trades)
    if max_ts < cursor["max_timestamp"]:
        return cursor
    new_keys_at_max = {position_key(t) for t in trades if t["timestamp"] == max_ts}
    if max_ts == cursor["max_timestamp"]:
        new_keys_at_max |= set(cursor.get("keys_at_max", []))
    return {"max_timestamp": max_ts, "keys_at_max": sorted(new_keys_at_max)}


def load_watch_state() -> dict:
    if WATCH_STATE_PATH.exists():
        with open(WATCH_STATE_PATH) as f:
            return json.load(f)
    return {}


def save_watch_state(state: dict) -> None:
    WATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCH_STATE_PATH, "w") as f:
        json.dump(state, f)


# --- copy-window math ---------------------------------------------------

def estimate_fair_value(entry_price: float, historical_copy_window: float) -> float:
    """Extrapolate this wallet's typical (fair_value - entry_price) drift
    (from features.py's copy_window, computed over their past bets) onto
    this new entry price."""
    return entry_price + historical_copy_window


def compute_remaining_window(entry_price: float, current_price: float, historical_copy_window: float) -> dict:
    """Remaining edge between the current price and this position's
    estimated fair value, in price-cents (0-100 scale) and as a percentage
    of the current price (what you'd pay to copy the position now)."""
    fair_value = estimate_fair_value(entry_price, historical_copy_window)
    remaining_price = fair_value - current_price
    remaining_pct = (remaining_price / current_price * 100) if current_price else float("nan")
    return {
        "fair_value": fair_value,
        "remaining_cents": remaining_price * 100,
        "remaining_pct": remaining_pct,
    }


def window_is_open(remaining_cents: float, min_remaining_window: float) -> bool:
    """min_remaining_window is in price units (0-1 scale, same as
    copy_window/entry_price) — converted to cents here for comparison."""
    return remaining_cents >= min_remaining_window * 100


# --- alerting ---------------------------------------------------------------

def build_alert(wallet_row: pd.Series, trade: dict, current_price: float, window: dict) -> dict:
    return {
        "wallet": trade["wallet"],
        "market_id": trade["market_id"],
        "market": trade.get("question"),
        "side": trade["side"],
        "entry_price": trade["entry_price"],
        "current_price": current_price,
        "remaining_cents": window["remaining_cents"],
        "remaining_pct": window["remaining_pct"],
        "wallet_out_of_sample_edge": wallet_row["out_of_sample_edge"],
        "wallet_out_of_sample_n": int(wallet_row["out_of_sample_n"]),
        "timestamp": trade["timestamp"],
    }


def format_alert(alert: dict) -> str:
    return (
        f"[ALERT] {alert['wallet']} bought \"{alert['market']}\" @ {alert['entry_price']:.3f} "
        f"(now {alert['current_price']:.3f}) — remaining window "
        f"{alert['remaining_cents']:+.1f}c ({alert['remaining_pct']:+.1f}%). "
        f"Wallet's out-of-sample edge: {alert['wallet_out_of_sample_edge']:+.3f} "
        f"over {alert['wallet_out_of_sample_n']} bets."
    )


def append_alert_log(alert: dict) -> None:
    ALERTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_LOG_PATH, "a") as f:
        f.write(json.dumps(alert) + "\n")


def send_webhook(session, webhook_url: str | None, alert: dict, timeout: float) -> None:
    if not webhook_url:
        return
    try:
        session.post(webhook_url, json={"text": format_alert(alert)}, timeout=timeout)
    except requests.exceptions.RequestException as exc:  # noqa: BLE001 - never crash the watcher
        print(f"[watch] WARNING: webhook delivery failed: {exc}")


def emit_alert(session, cfg: dict, alert: dict) -> None:
    print(format_alert(alert))
    append_alert_log(alert)
    send_webhook(session, cfg["watch"].get("webhook_url"), alert, cfg["ingest"]["request_timeout_sec"])


# --- poll cycle --------------------------------------------------------

def load_watchlist(cfg: dict) -> pd.DataFrame:
    """Top `watch.top_n_wallets` ranked wallets, best rank first."""
    if not RANKED_WALLETS_PATH.exists():
        return pd.DataFrame()
    ranked = pd.read_parquet(RANKED_WALLETS_PATH)
    if ranked.empty:
        return ranked
    top_n = cfg["watch"]["top_n_wallets"]
    return ranked.sort_values("rank").head(top_n)


def poll_once(
    session,
    cfg: dict,
    watchlist: pd.DataFrame,
    state: dict,
    now_ts: int,
    fetch_trades=fetch_wallet_trades,
    fetch_price=fetch_current_price,
) -> tuple[list[dict], dict]:
    """One poll cycle over every watched wallet. Pure aside from the two
    injectable fetch functions, so tests can supply canned data instead of
    hitting the network. Returns (alerts, updated_state) — never mutates
    `state` in place, so a caller can discard it on failure."""
    new_state = dict(state)
    alerts = []

    for _, wallet_row in watchlist.iterrows():
        wallet = wallet_row["wallet"]
        cursor = new_state.get(wallet, default_cursor(now_ts))
        trades = fetch_trades(session, cfg, wallet, TRADES_FETCH_LIMIT)
        fresh = new_trades_since(trades, cursor)
        new_state[wallet] = advance_cursor(cursor, trades)

        for trade in fresh:
            if trade["side"] != "BUY":  # positions = entries; see DECISIONS.md
                continue
            current_price = fetch_price(session, cfg, trade["token_id"])
            if current_price is None:
                continue
            window = compute_remaining_window(trade["entry_price"], current_price, wallet_row["copy_window"])
            if not window_is_open(window["remaining_cents"], cfg["watch"]["min_remaining_window"]):
                continue
            alerts.append(build_alert(wallet_row, trade, current_price, window))

    return alerts, new_state


def run_watch_loop(cfg: dict) -> None:
    """Long-lived poll loop. Resumable (state persisted to disk each cycle)
    and never crash-loops: transient errors back off exponentially up to
    MAX_BACKOFF_SECONDS and retry, rather than raising."""
    session = make_session(max_retries=cfg["ingest"]["max_retries"], backoff=cfg["ingest"]["retry_backoff_sec"])
    state = load_watch_state()
    poll_seconds = cfg["watch"]["poll_seconds"]
    backoff_seconds = poll_seconds

    while True:
        try:
            watchlist = load_watchlist(cfg)
            if watchlist.empty:
                print("[watch] no ranked wallets to watch yet — run src.rank first.")
            else:
                alerts, state = poll_once(session, cfg, watchlist, state, int(time.time()))
                save_watch_state(state)
                for alert in alerts:
                    emit_alert(session, cfg, alert)
                if alerts:
                    print(f"[watch] {len(alerts)} alert(s) this cycle")
            backoff_seconds = poll_seconds
        except requests.exceptions.RequestException as exc:
            print(f"[watch] WARNING: poll cycle failed ({exc}); retrying in {backoff_seconds}s")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)
            continue

        time.sleep(poll_seconds)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    print(
        f"[watch] starting — polling every {cfg['watch']['poll_seconds']}s "
        f"for the top {cfg['watch']['top_n_wallets']} ranked wallets"
    )
    run_watch_loop(cfg)


if __name__ == "__main__":
    main()
