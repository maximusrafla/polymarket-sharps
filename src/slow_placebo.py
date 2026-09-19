"""Wallet-agnostic placebo for the forward test — Project 3 step 8c.

THE CONTROL
-----------
Beating the frozen baseline is the weak test: the baseline is a price-calibration
curve fit on pre-freeze data, so a tier can clear it simply because the markets it
happened to trade after the freeze were mispriced relative to that curve — a
regime shift, a newly-mispriced niche, anything that lifts EVERY participant in
those markets. That is the same "the edge lives in the niche, not the wallet"
confound that steps 6-7 spent their whole length on, arriving in forward data.

The strong test is therefore relative: for the SAME resolving markets, how did
everyone ELSE do? If the frozen crowd's event-weighted edge is indistinguishable
from a random same-market crowd's, the tier has no wallet-level edge no matter
how positive its absolute number looks.

Two readouts, both against the identical frozen baseline and the identical
event-weighted statistic:

  placebo_edge       — the event-weighted edge of ALL non-cohort BUY bets in the
                       same markets. The "what did the room do" number.
  placebo_percentile — the tier's edge against a null built by resampling
                       COUNT-MATCHED random wallet crowds from those same markets.
                       This is the one that matters: it controls for crowd size as
                       well as for market selection.

COST AND HONESTY
----------------
This needs each involved market's full tape via `/trades?market=`, which is the
only place in the forward path that spends more than one request per wallet. The
market count is capped per run and **any truncation is logged explicitly** — a
silently capped placebo would understate the control and read as a pass.

READ-ONLY: public unauthenticated GETs only, same endpoint discover.py uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import INTERIM_DIR, atomic_to_parquet, load_config, make_session
from src.slow_aggregate import event_weighted_mean

PLACEBO_TAPE_PATH = INTERIM_DIR / "slow_forward" / "placebo_tapes.parquet"

DEFAULT_MAX_MARKETS = 300      # per run; truncation is always logged
DEFAULT_NULL_DRAWS = 2000
MIN_PLACEBO_WALLETS = 5        # below this the matched null is meaningless


def fetch_market_tapes(market_ids, max_markets: int = DEFAULT_MAX_MARKETS,
                       cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Pull the full public tape for each market. Returns (trades, coverage) where
    `coverage` records exactly how many markets were requested, fetched and
    skipped — so a capped run can never be mistaken for a complete one."""
    from src.discover import fetch_market_trades

    cfg = cfg or load_config()
    ids = list(dict.fromkeys(market_ids))
    requested = len(ids)
    truncated = requested > max_markets
    use = ids[:max_markets]

    session = make_session()
    rows, failed = [], 0
    for i, mid in enumerate(use, 1):
        try:
            trades, _cursor, _cap = fetch_market_trades(session, mid, {})
        except Exception as e:  # noqa: BLE001 — one market must not abort the run
            failed += 1
            print(f"  [placebo {i}/{len(use)}] {mid[:12]} FAILED: {e}")
            continue
        rows.extend(trades)
        if i % 25 == 0 or i == len(use):
            print(f"  [placebo {i}/{len(use)}] {len(rows):,} tape rows")

    coverage = {
        "markets_requested": requested,
        "markets_fetched": len(use) - failed,
        "markets_skipped_by_cap": max(0, requested - max_markets),
        "markets_failed": failed,
        "truncated": bool(truncated),
        "max_markets": max_markets,
    }
    if truncated:
        print(f"  [placebo] TRUNCATED: {requested} markets involved, capped at "
              f"{max_markets}. The placebo covers a SUBSET — reported, not silent.")
    return pd.DataFrame(rows), coverage


def score_placebo_tape(tape: pd.DataFrame, baseline: dict, freeze_ts: int,
                       cfg: dict | None = None) -> pd.DataFrame:
    """Turn a raw market tape into resolved, niche-labelled, residualised bets
    scored against the SAME frozen baseline the cohorts are scored against."""
    from src.ingest import fold_trades_to_ledger, update_resolutions
    from src.slow_baseline import expected_outcome_hier
    from src.slow_market import derive_categories
    from src.slow_niche import assign_narratives, assign_niches

    cfg = cfg or load_config()
    if tape.empty:
        return pd.DataFrame()

    session = make_session()
    cids = set(tape["conditionId"].dropna().unique())
    res = update_resolutions(session, cfg, cids,
                             pd.DataFrame(columns=["market_id", "token_id", "resolved",
                                                   "resolved_value", "closed"]))
    bets = fold_trades_to_ledger(tape.to_dict("records"), res)
    bets = bets[(bets["side"] == "BUY") & bets["resolved"].fillna(False)]
    bets = bets[bets["timestamp"] > freeze_ts].reset_index(drop=True)
    if bets.empty:
        return bets

    bets["category"] = derive_categories(bets["slug"], bets["question"])
    bets = assign_narratives(assign_niches(bets))
    exp = expected_outcome_hier(baseline, bets)
    bets["expected_outcome"] = exp
    bets["residual_skill"] = bets["resolved_value"].to_numpy(dtype=float) - exp
    bets["resolution_ts"] = bets.groupby("market_id")["timestamp"].transform("max")
    return bets


def matched_null(placebo_bets: pd.DataFrame, n_wallets: int, n_draws: int,
                 rng: np.random.Generator, event_col: str = "niche_l1") -> np.ndarray:
    """Null distribution of the event-weighted edge for a random crowd of
    `n_wallets` wallets drawn from the same markets.

    Matching on crowd SIZE matters: a 12-wallet crowd and a 699-wallet crowd have
    very different sampling variance, so comparing either to an unmatched pool
    would be a rigged test in one direction or the other."""
    if placebo_bets.empty or n_wallets <= 0:
        return np.array([])
    wallets = pd.unique(placebo_bets["wallet"])
    if wallets.size < max(MIN_PLACEBO_WALLETS, 2):
        return np.array([])
    take = min(n_wallets, wallets.size)
    by_wallet = {w: g for w, g in placebo_bets.groupby("wallet", sort=False)}
    out = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        pick = rng.choice(wallets, size=take, replace=False)
        sub = pd.concat([by_wallet[w] for w in pick], ignore_index=True)
        r = sub["residual_skill"].to_numpy(dtype=float)
        ok = ~np.isnan(r)
        out[i] = (event_weighted_mean(r[ok], sub[event_col].to_numpy()[ok])
                  if ok.any() else np.nan)
    return out[~np.isnan(out)]


def placebo_for_tier(cohort_bets: pd.DataFrame, placebo_bets: pd.DataFrame,
                     tier: str, n_wallets: int, seed: int,
                     n_draws: int = DEFAULT_NULL_DRAWS,
                     event_col: str = "niche_l1") -> dict:
    """The two placebo readouts for one tier, restricted to the markets the tier
    actually traded after the freeze."""
    empty = {"tier": tier, "stratum": "all", "event_unit": event_col,
             "placebo_edge": float("nan"), "placebo_percentile": float("nan"),
             "placebo_wallets": 0, "placebo_bets": 0}
    if cohort_bets.empty or placebo_bets.empty:
        return empty

    markets = set(cohort_bets["market_id"].unique())
    cohort_wallets = set(cohort_bets["wallet"].unique())
    others = placebo_bets[placebo_bets["market_id"].isin(markets)
                          & ~placebo_bets["wallet"].isin(cohort_wallets)]
    if others.empty:
        return empty

    r = others["residual_skill"].to_numpy(dtype=float)
    ok = ~np.isnan(r)
    edge = (event_weighted_mean(r[ok], others[event_col].to_numpy()[ok])
            if ok.any() else float("nan"))

    cr = cohort_bets["residual_skill"].to_numpy(dtype=float)
    cok = ~np.isnan(cr)
    cohort_edge = (event_weighted_mean(cr[cok], cohort_bets[event_col].to_numpy()[cok])
                   if cok.any() else float("nan"))

    rng = np.random.default_rng(seed)
    null = matched_null(others, n_wallets, n_draws, rng, event_col=event_col)
    pct = (float(np.mean(null < cohort_edge)) if null.size and not np.isnan(cohort_edge)
           else float("nan"))
    return {"tier": tier, "stratum": "all", "event_unit": event_col,
            "placebo_edge": edge, "placebo_percentile": pct,
            "placebo_wallets": int(others["wallet"].nunique()),
            "placebo_bets": int(len(others))}


def run_placebo(bets: pd.DataFrame, frozen: pd.DataFrame, manifest: dict,
                tier_order, max_markets: int = DEFAULT_MAX_MARKETS,
                cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fetch the tapes for every market the frozen crowd traded post-freeze, then
    compute the placebo readouts per tier."""
    cfg = cfg or load_config()
    markets = sorted(bets["market_id"].dropna().unique())
    if not markets:
        return pd.DataFrame(), {"markets_requested": 0}
    print(f"[placebo] {len(markets)} markets involved")
    tape, coverage = fetch_market_tapes(markets, max_markets=max_markets, cfg=cfg)
    scored = score_placebo_tape(tape, manifest["baseline"],
                                int(manifest["freeze_ts"]), cfg=cfg)
    if not scored.empty:
        PLACEBO_TAPE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(scored, PLACEBO_TAPE_PATH, compression="gzip")

    rows = []
    for tier in tier_order:
        if tier not in frozen.columns:
            continue
        members = set(frozen.loc[frozen[tier], "wallet"])
        sub = bets[bets["wallet"].isin(members)]
        rows.append(placebo_for_tier(sub, scored, tier, n_wallets=len(members),
                                     seed=abs(hash(tier)) % (2 ** 32)))
    return pd.DataFrame(rows), coverage
