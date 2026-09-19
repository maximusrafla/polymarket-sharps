"""Shortlist + deepen the sports candidates — Project 3 sports step 3.

THE FIREWALL IS THE POINT. The gate screened wallets on a SHALLOW, slow-filtered
discovery corpus whose 284 markets fold to ~25 resolution events (see
src/sports_events.py). Selection on that corpus is selection noise by
construction — 79% of its findable wallets span <=2 events, so "high residual
skill" there mostly means "bought the team that won one tournament". This step
pulls each shortlisted wallet's near-complete history via `/trades?user=`,
creating data the screen never saw, so the validation in step 4 is an honest
out-of-sample test rather than a re-measurement of the screen. The screen is a
SELECTION DEVICE, never evidence; a bad screen costs power, not validity.

WHY THREE ARMS. The two arms the gate implies (>=2c, >=10c shrunk residual
skill) are both dominated by single-event selection. The BREADTH arm selects on
event-weighted skill among wallets spanning >=3 distinct events, so it is the
one arm whose members cannot be a single lucky tournament — and therefore the
one most likely to produce forward-testable candidates with real game breadth.
All three are kept and labelled; nothing is dropped, and arm membership is an
additive column so any of them can be read separately later.

ISOLATION: writes ONLY to `data/interim/sports/`. It NEVER touches the shared
`bet_ledger.parquet` — Project 1's certified set stays bit-identical. Same
discipline as `discover.py` and `slow_deepen.py`, whose per-user fetch / fold /
resolution machinery this reuses verbatim.

READ-ONLY / ANALYSIS-ONLY: public unauthenticated GETs, no keys, nothing
on-chain. This is the one step in the sports chain that spends requests.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.backfill import fetch_user_trades
from src.common import (
    INTERIM_DIR,
    LEDGER_COLUMNS,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
    print_disk_usage_summary,
)
from src.ingest import (
    LEDGER_DEDUP_KEY,
    fold_trades_to_ledger,
    refresh_ledger_resolutions,
    trade_key,
    update_resolutions,
)
from src.slow_baseline import fit_hierarchical_residuals
from src.sports_events import assign_events

SPORTS_DIR = INTERIM_DIR / "sports"
GATE_DIR = SPORTS_DIR / "gate"
DEEP_TRADES_PATH = SPORTS_DIR / "deep_trades.parquet"
DEEP_CURSORS_PATH = SPORTS_DIR / "deep_cursors.json"
DEEP_RESOLUTIONS_PATH = SPORTS_DIR / "deep_resolutions.parquet"
SHORTLIST_PATH = SPORTS_DIR / "shortlist.parquet"

RESOLUTION_COLUMNS = ["market_id", "token_id", "outcome", "resolved", "resolved_value", "closed"]

# --- screen parameters (pinned, not tuned) ---------------------------------
MIN_SCREEN_BETS = 20        # the gate's "findable" rule
SHRINK_K = 50.0             # bets a wallet needs before it earns half its own mean
SHRINK_K_EVENTS = 2.0       # events, for the event-weighted breadth arm
CORE_GATE = 0.10            # arm 1: the gate's 10c core
DEFAULT_GATE = 0.02         # arm 2: the gate's 2c shortlist
BREADTH_MIN_EVENTS = 3      # arm 3: cannot be one lucky tournament
# The breadth arm takes its WHOLE qualifying pool (>=3 events and positive
# event-weighted skill) rather than a top-N slice: a rank cut inside an arm whose
# whole purpose is coverage would just re-import the selection noise it exists to
# avoid. The pool is 212 wallets, so the three-arm union is ~250. The cap below is
# a runaway guard, not a selection rule; it prints loudly if it ever binds.
BREADTH_MAX = 400

DEFAULT_CHECKPOINT_EVERY = 25


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------

def load_gate_corpus(lowo: bool = False, refresh: bool = False) -> pd.DataFrame:
    """The gate's sports bets with `event` attached and a per-bet residual skill
    from a league / league|form baseline.

    Non-LOWO by default: this is a screen, and the population baseline is the
    conservative direction (a wallet heavy enough to move its own baseline has
    its residual pulled toward zero). Validation in step 4 uses the LOWO fit.

    Cached — the slug join reads the 153 MB discovery tape and the fit is a full
    pass over the corpus, neither of which changes between runs. This is the one
    implementation; `scripts/audit_sports_event_unit.py` imports it rather than
    keeping a second copy that could drift."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cache = GATE_DIR / f"corpus_resid{'_lowo' if lowo else ''}.parquet"
    if cache.exists() and not refresh:
        print(f"[sports_deepen] cached gate corpus -> {cache}")
        return pd.read_parquet(cache)

    bets = pd.read_parquet(GATE_DIR / "sports_bets_all.parquet")
    markets = pd.read_parquet(GATE_DIR / "sports_markets.parquet")
    t = pq.read_table(INTERIM_DIR / "discovery" / "discovery_trades.parquet",
                      columns=["market_id", "slug", "question"])
    t = t.filter(pc.is_in(t["market_id"], value_set=pa.array(list(markets["market_id"]))))
    markets = assign_events(markets.merge(t.to_pandas().drop_duplicates("market_id"),
                                          on="market_id", how="left"))
    bets = bets.merge(markets[["market_id", "event", "event_kind"]],
                      on="market_id", how="left")
    resid, _ = fit_hierarchical_residuals(bets, levels=("niche_l1", "niche_l2"),
                                          lowo=lowo)
    keep = ["wallet", "market_id", "event", "event_kind", "niche_l1", "niche_l2",
            "entry_price", "resolved_value", "timestamp", "residual_skill"]
    resid = resid[[c for c in keep if c in resid.columns]]
    resid = resid[~resid["residual_skill"].isna()].reset_index(drop=True)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(resid, cache, compression="gzip")
    print(f"[sports_deepen] built gate corpus -> {cache}")
    return resid


def screen_wallets(corpus: pd.DataFrame) -> pd.DataFrame:
    """Per-wallet screen scores on the gate corpus. One row per findable wallet,
    with BOTH the bet-weighted score the gate used and the event-weighted score
    the breadth arm uses — the second is the one a single tournament cannot
    inflate, because each event contributes once regardless of how many of its
    markets the wallet bought."""
    g = corpus.groupby("wallet")
    tbl = g.agg(n_bets=("residual_skill", "size"),
                mean_resid=("residual_skill", "mean"),
                n_events=("event", "nunique"),
                n_markets=("market_id", "nunique")).reset_index()
    tbl = tbl[tbl["n_bets"] >= MIN_SCREEN_BETS].copy()

    # event-weighted: mean over events of the within-event mean residual
    per_ev = corpus.groupby(["wallet", "event"])["residual_skill"].mean().reset_index()
    ev_mean = per_ev.groupby("wallet")["residual_skill"].mean().rename("event_mean_resid")
    tbl = tbl.merge(ev_mean, on="wallet", how="left")

    tbl["shrunk_skill"] = tbl["mean_resid"] * tbl["n_bets"] / (tbl["n_bets"] + SHRINK_K)
    tbl["shrunk_event_skill"] = (tbl["event_mean_resid"] * tbl["n_events"]
                                 / (tbl["n_events"] + SHRINK_K_EVENTS))
    top_event = corpus.groupby(["wallet", "event"]).size().rename("n").reset_index()
    top_share = (top_event.sort_values("n", ascending=False).drop_duplicates("wallet")
                 .set_index("wallet"))
    tbl["top_event"] = tbl["wallet"].map(top_share["event"])
    tbl["top_event_share"] = tbl["wallet"].map(top_share["n"]) / tbl["n_bets"]
    return tbl.sort_values("shrunk_skill", ascending=False).reset_index(drop=True)


def select_shortlist(scores: pd.DataFrame) -> pd.DataFrame:
    """The three arms, unioned. Arm membership is additive metadata: a wallet in
    two arms keeps both flags, and the union is what gets deepened."""
    s = scores.copy()
    s["arm_core"] = s["shrunk_skill"] >= CORE_GATE
    s["arm_default"] = s["shrunk_skill"] >= DEFAULT_GATE
    breadth_pool = s[(s["n_events"] >= BREADTH_MIN_EVENTS)
                     & (s["shrunk_event_skill"] > 0)]
    if len(breadth_pool) > BREADTH_MAX:
        print(f"[sports_deepen] WARNING: breadth pool {len(breadth_pool):,} exceeds the "
              f"{BREADTH_MAX} runaway guard — TRUNCATED by event-weighted skill. "
              f"{len(breadth_pool) - BREADTH_MAX:,} qualifying wallets are NOT deepened.")
        breadth_pool = breadth_pool.sort_values("shrunk_event_skill",
                                                ascending=False).head(BREADTH_MAX)
    breadth = set(breadth_pool["wallet"])
    s["arm_breadth"] = s["wallet"].isin(breadth)
    short = s[s["arm_core"] | s["arm_default"] | s["arm_breadth"]].copy()
    # deepen the most-vetted first, so a truncated run still covers the core
    short["priority"] = np.where(short["arm_core"], 0,
                                 np.where(short["arm_breadth"], 1, 2))
    return short.sort_values(["priority", "shrunk_skill"], ascending=[True, False]
                             ).reset_index(drop=True)


def build_shortlist() -> pd.DataFrame:
    corpus = load_gate_corpus()
    scores = screen_wallets(corpus)
    short = select_shortlist(scores)
    SPORTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(short, SHORTLIST_PATH, compression="gzip")
    print(f"[sports_deepen] gate corpus: {len(corpus):,} bets, "
          f"{corpus['wallet'].nunique():,} wallets, {corpus['event'].nunique():,} events")
    print(f"[sports_deepen] findable (>={MIN_SCREEN_BETS} bets): {len(scores):,}")
    print(f"[sports_deepen] shortlist: {len(short):,} wallets  "
          f"[core>={CORE_GATE:.2f}: {int(short['arm_core'].sum())}, "
          f"default>={DEFAULT_GATE:.2f}: {int(short['arm_default'].sum())}, "
          f"breadth(>={BREADTH_MIN_EVENTS} events): {int(short['arm_breadth'].sum())}]")
    print(f"[sports_deepen] shortlist event breadth: median {short['n_events'].median():.0f}, "
          f"share spanning 1 event {np.mean(short['n_events'] == 1):.1%}")
    print(f"[sports_deepen] -> {SHORTLIST_PATH}")
    return short


# ---------------------------------------------------------------------------
# Isolated persistence (never the shared ledger)
# ---------------------------------------------------------------------------

def load_deep_trades() -> pd.DataFrame:
    if DEEP_TRADES_PATH.exists():
        return pd.read_parquet(DEEP_TRADES_PATH)
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def save_deep_trades(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, DEEP_TRADES_PATH, compression="gzip")


def load_deep_cursors() -> dict:
    if DEEP_CURSORS_PATH.exists():
        with open(DEEP_CURSORS_PATH) as f:
            return json.load(f)
    return {}


def load_deep_resolutions() -> pd.DataFrame:
    if DEEP_RESOLUTIONS_PATH.exists():
        return pd.read_parquet(DEEP_RESOLUTIONS_PATH)
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=LEDGER_DEDUP_KEY, keep="last").reset_index(drop=True)


def _checkpoint(batch: list[dict], cursors: dict) -> None:
    """Fold a batch into the isolated dataset and persist it + cursors (both
    atomic). Idempotent: the fill-key dedup plus per-wallet cursors make repeated
    checkpoints safe and the run resumable after an interruption."""
    if batch:
        new_rows = fold_trades_to_ledger(batch, pd.DataFrame())   # resolutions applied later
        combined = _dedup(pd.concat([load_deep_trades(), new_rows], ignore_index=True))
        save_deep_trades(combined)
    atomic_write_json(cursors, DEEP_CURSORS_PATH)


def deepen_wallets(wallets: list[str], cfg: dict | None = None,
                   checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY, session=None) -> int:
    """Pull each shortlisted wallet's full `/trades?user=` history into the
    isolated sports dataset. Per-wallet cursors make it incremental and
    resumable. Returns total new rows."""
    SPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = cfg or load_config()
    session = session or make_session()
    cursors = load_deep_cursors()

    batch: list[dict] = []
    total_new = 0
    t0 = time.time()
    for i, wallet in enumerate(wallets, 1):
        since_ts = int(cursors.get(wallet, {}).get("max_timestamp", 0))
        try:
            trades = fetch_user_trades(session, cfg, wallet, since_ts)
        except Exception as exc:  # noqa: BLE001 - keep deepening the other wallets
            print(f"[sports_deepen] WARNING: fetch failed for {wallet}: {exc}", flush=True)
            continue
        if trades:
            batch.extend(trades)
            total_new += len(trades)
            newest = max(t["timestamp"] for t in trades)
            keys_at_max = [trade_key(t) for t in trades if t["timestamp"] == newest]
            cursors[wallet] = {"max_timestamp": max(since_ts, newest), "keys_at_max": keys_at_max}
        else:
            cursors.setdefault(wallet, {"max_timestamp": since_ts, "keys_at_max": []})
        if i % checkpoint_every == 0 or i == len(wallets):
            _checkpoint(batch, cursors)
            batch = []
            rate = i / max(time.time() - t0, 1e-9)
            print(f"[sports_deepen]   {i}/{len(wallets)} wallets, {total_new:,} new trade rows "
                  f"({rate:.1f} wallets/s) — checkpointed", flush=True)
    return total_new


def resolve_deep_markets(cfg: dict | None = None, session=None,
                         max_fetches: int | None = None,
                         sports_only: bool = True) -> pd.DataFrame:
    """Resolve the sports dataset's markets, ISOLATED from the shared cache: read
    the shared resolutions read-only to avoid re-fetching what is already known,
    write only our own. Mirrors slow_deepen.resolve_deep_markets.

    `sports_only` (the default) restricts the CLOB fetches to markets the sports
    arm will actually score. A deep pull returns each wallet's WHOLE history, and
    measured on the first 125 wallets only 61% of its 133k markets are sports —
    resolving the rest is pure request spend on data this arm never reads, and
    the CLOB rate-limits to ~20 req/s regardless of concurrency, so it is the
    difference between ~50 and ~80 minutes. Non-sports rows simply stay
    unresolved; pass `sports_only=False` to complete the tape for another arm."""
    from src.common import RESOLUTIONS_PATH
    cfg = cfg or load_config()
    session = session or make_session()

    trades = load_deep_trades()
    if trades.empty:
        print("[sports_deepen] no deep trades to resolve — run deepening first.")
        return trades

    shared = (pd.read_parquet(RESOLUTIONS_PATH) if RESOLUTIONS_PATH.exists()
              else pd.DataFrame(columns=RESOLUTION_COLUMNS))
    deep_res = load_deep_resolutions()

    market_ids = set(trades["market_id"].dropna().unique())
    if sports_only:
        from src.slow_market import derive_categories
        from src.sports_events import is_sports_market
        idx = trades.drop_duplicates("market_id")[["market_id", "slug", "question"]]
        cats = derive_categories(idx["slug"], idx["question"])
        keep = {m for m, s, q, c in zip(idx["market_id"], idx["slug"], idx["question"], cats)
                if is_sports_market(s, q, c)}
        print(f"[sports_deepen] sports-only resolution: {len(keep):,} of "
              f"{len(market_ids):,} markets ({len(keep) / max(len(market_ids), 1):.0%}); "
              f"the rest stay unresolved and are never scored by this arm")
        market_ids = market_ids & keep
    already = set()
    for r in (shared, deep_res):
        if not r.empty:
            already |= set(r.loc[r["closed"].fillna(False), "market_id"])
    need = market_ids - already
    print(f"[sports_deepen] resolving {len(market_ids):,} markets: "
          f"{len(market_ids) - len(need):,} already known, {len(need):,} to fetch from CLOB")

    deep_res = update_resolutions(session, cfg, need, deep_res, max_fetches=max_fetches)
    atomic_to_parquet(deep_res, DEEP_RESOLUTIONS_PATH, compression="gzip")

    combined = pd.concat([shared[RESOLUTION_COLUMNS] if not shared.empty else shared,
                          deep_res], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["market_id", "token_id"], keep="last")
    enriched = refresh_ledger_resolutions(trades, combined)
    save_deep_trades(enriched)

    n_res = int(enriched["resolved"].fillna(False).sum())
    print(f"[sports_deepen] deep dataset now {n_res:,}/{len(enriched):,} resolved bets "
          f"across {enriched.loc[enriched['resolved'].fillna(False), 'market_id'].nunique():,} markets")
    return enriched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_deepen(max_wallets: int | None = None, checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
               resolve: bool = True, max_resolution_fetches: int | None = None,
               sports_only: bool = True) -> None:
    short = build_shortlist()
    if max_wallets is not None:
        short = short.head(max_wallets)
    total = deepen_wallets(short["wallet"].tolist(), checkpoint_every=checkpoint_every)
    deep = load_deep_trades()
    print(f"[sports_deepen] added {total:,} rows; dataset now {len(deep):,} trades across "
          f"{deep['market_id'].nunique():,} markets and {deep['wallet'].nunique():,} wallets")
    if resolve:
        resolve_deep_markets(max_fetches=max_resolution_fetches, sports_only=sports_only)
    print_disk_usage_summary()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shortlist + deepen the sports candidates via /trades?user= "
                    "(Project 3 sports step 3). Isolated dataset; the shared ledger "
                    "is never touched.")
    ap.add_argument("--shortlist-only", action="store_true",
                    help="build and persist the shortlist without spending any requests")
    ap.add_argument("--resolve-only", action="store_true",
                    help="skip fetching; just (re)resolve the existing dataset")
    ap.add_argument("--max-wallets", type=int, default=None,
                    help="cap wallets deepened this run (highest-priority arm first)")
    ap.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    ap.add_argument("--all-markets", action="store_true",
                    help="resolve every market in the deep tape, not just the sports "
                         "markets this arm scores (slower; for reusing the tape)")
    ap.add_argument("--no-resolve", action="store_true",
                    help="fetch only; resolve in a separate (long, uncapped) run")
    ap.add_argument("--max-resolution-fetches", type=int, default=None,
                    help="cap CLOB resolution GETs this run (the rest drain on re-runs)")
    args = ap.parse_args()

    if args.shortlist_only:
        build_shortlist()
    elif args.resolve_only:
        resolve_deep_markets(max_fetches=args.max_resolution_fetches,
                             sports_only=not args.all_markets)
        print_disk_usage_summary()
    else:
        run_deepen(max_wallets=args.max_wallets, checkpoint_every=args.checkpoint_every,
                   resolve=not args.no_resolve,
                   max_resolution_fetches=args.max_resolution_fetches,
                   sports_only=not args.all_markets)


if __name__ == "__main__":
    main()
