"""Slow-market universe: the shared bet loader + the per-category price baseline
fit INSIDE the slow universe — Project 3 build step 4 (docs/project3_slow_markets.md
§4 stage 2, §7.4; Project 2 §2.0).

The slow universe is the set of ledger bets whose market is classified `slow` or
`deep_slow` by the `speed_bucket` column (step 2), i.e. lifespan L ≥ the slow
threshold. This module:

  1. `load_slow_universe_bets` — resolved BUY bets from the ledger, filtered to the
     slow buckets via parquet predicate pushdown (measured ~675 MB peak, no
     row-group pruning on the unsorted ledger — see step 2), with a `category`
     column derived from slug/question. This is the loader step 5's screen reuses.
  2. `fit_slow_category_baselines` — the §2.0 per-category favorite-longshot
     calibration curve E[outcome | entry_price], fit inside the slow universe so it
     is NOT inherited from the crypto-dominated global curve (§4 stage 2). Reuses
     `forecaster_metrics.fit_category_baselines` verbatim (own-category → real-world-
     wide → global fallback hierarchy); the only new thing is the slow-universe scope.
  3. `residualize` — per-bet skill edge = outcome − E_cat[outcome | entry_price],
     the residual the step-5 screen aggregates per (wallet, category).

READ-ONLY / ANALYSIS-ONLY. Screening-stage baselines and residuals are a selection
foundation, NOT a published finding — nothing here is ranked for a human or written
to data/processed/ (§4 stage 2: screening numbers are selection noise by construction).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import BET_LEDGER_PATH, INTERIM_DIR, atomic_to_parquet, load_config
from src.discover import classify_market, load_discovery_trades
from src.forecaster_metrics import (
    GLOBAL_KEY,
    REAL_WORLD_KEY,
    fit_category_baselines,
    expected_outcome_by_category,
)
from src.market_meta import classify_speed_bucket, speed_thresholds

# The speed buckets that constitute the slow universe (step 2's `speed_bucket`).
# `unknown` is deliberately EXCLUDED — a bet parked as unknown-speed is not proven
# slow (§1.2: absence of a lifespan never proves slow OR fast), so it must not enter
# the slow-universe screen; it re-enters once its lifespan is established. Tunable.
SLOW_BUCKETS = ("slow", "deep_slow")

# Screen output (isolated interim dir, gitignored — NEVER data/processed/: the screen
# is a selection step whose numbers are selection noise by construction, §4 stage 2).
SLOW_SCREEN_DIR = INTERIM_DIR / "slow_screen"
WALLET_SCORES_PATH = SLOW_SCREEN_DIR / "wallet_scores.parquet"

# Columns the slow-universe loader reads. slug+question are needed only to derive
# `category`; the heavy text stays out of everything downstream.
_LOAD_COLUMNS = ["wallet", "market_id", "side", "entry_price", "resolved",
                 "resolved_value", "timestamp", "slug", "question", "speed_bucket"]


# ---------------------------------------------------------------------------
# Category derivation (pure; no network)
# ---------------------------------------------------------------------------

def derive_categories(slugs, questions) -> np.ndarray:
    """Per-bet category label from slug/question via the tested `discover.classify_market`
    keyword classifier (the main ledger has no persisted `category` column, unlike the
    discovery tape). Returns an object array aligned to the inputs."""
    slugs = list(slugs)
    questions = list(questions)
    return np.array([classify_market(s, q) for s, q in zip(slugs, questions)], dtype=object)


# ---------------------------------------------------------------------------
# Slow-universe bet loader (the shared foundation for step 5's screen)
# ---------------------------------------------------------------------------

def _has_speed_bucket(path) -> bool:
    import pyarrow.parquet as pq
    try:
        return "speed_bucket" in pq.ParquetFile(path).schema_arrow.names
    except Exception:  # noqa: BLE001
        return False


def load_slow_universe_bets(
    buckets=SLOW_BUCKETS,
    ledger_path=None,
    resolved_only: bool = True,
    buys_only: bool = True,
) -> pd.DataFrame:
    """Resolved BUY bets whose market is in the slow universe, with `category`.

    Filters `speed_bucket ∈ buckets` at the parquet layer (predicate pushdown), so
    only the slow slice is materialized (~370k rows, ~675 MB peak — step 2's
    measurement). Requires the step-2 `speed_bucket` column; on a legacy ledger
    without it, returns empty with a warning (nothing is proven slow, so the honest
    result is an empty universe until populate runs)."""
    ledger_path = ledger_path or BET_LEDGER_PATH
    if not pd.io.common.file_exists(str(ledger_path)):
        return pd.DataFrame(columns=_LOAD_COLUMNS + ["category"])
    if not _has_speed_bucket(ledger_path):
        print("[slow_market] ledger has no speed_bucket column — run "
              "`python -m src.market_meta && python -m src.market_meta --populate` first. "
              "Returning an empty slow universe.")
        return pd.DataFrame(columns=_LOAD_COLUMNS + ["category"])

    filters = [("speed_bucket", "in", list(buckets))]
    bets = pd.read_parquet(ledger_path, columns=_LOAD_COLUMNS, filters=filters)
    if buys_only:
        bets = bets[bets["side"] == "BUY"]
    if resolved_only:
        bets = bets[bets["resolved"].fillna(False)]
    bets = bets.reset_index(drop=True)
    bets["category"] = derive_categories(bets["slug"], bets["question"])
    return bets


def load_discovery_slow_bets(buckets=SLOW_BUCKETS, cfg: dict | None = None) -> pd.DataFrame:
    """Slow-universe resolved BUY bets from the market-first discovery corpus (the
    broad real-world population, §3.1). The discovery tape already carries `category`
    but no `speed_bucket`; its markets have FULL tapes (pulled via /trades?market=), so
    the observed tape span is a good lifespan estimate — classify each market with the
    same tape-lower-bound rule as the sidecar and keep the slow buckets."""
    cfg = cfg or load_config()
    slow_s, deep_s = speed_thresholds(cfg)
    cols = ["wallet", "market_id", "side", "entry_price", "resolved",
            "resolved_value", "timestamp", "category"]
    df = load_discovery_trades(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols + ["speed_bucket"])
    # Lifespan lower bound = observed span over ALL of the market's discovery trades.
    span = df.groupby("market_id")["timestamp"].agg(["min", "max"])
    lifespan = (span["max"] - span["min"]).to_dict()
    bucket_of = {m: classify_speed_bucket(L, slow_s, deep_s) for m, L in lifespan.items()}
    bets = df[(df["side"] == "BUY") & df["resolved"].fillna(False)].copy()
    bets["speed_bucket"] = bets["market_id"].map(bucket_of)
    bets = bets[bets["speed_bucket"].isin(buckets)].reset_index(drop=True)
    return bets


def load_combined_slow_universe(buckets=SLOW_BUCKETS, cfg: dict | None = None) -> pd.DataFrame:
    """The slow universe across BOTH corpora (§4 stage 2): the main wallet-first
    ledger (deep per wallet, but adversely selected for micro performance) and the
    market-first discovery corpus (shallow per wallet, but the right population). A
    `corpus` column marks the source. This is what the screen scores over."""
    main = load_slow_universe_bets(buckets=buckets)
    disc = load_discovery_slow_bets(buckets=buckets, cfg=cfg)
    keep = ["wallet", "market_id", "category", "entry_price", "resolved",
            "resolved_value", "timestamp"]
    main = main[[c for c in keep if c in main.columns]].copy()
    disc = disc[[c for c in keep if c in disc.columns]].copy()
    main["corpus"] = "main"
    disc["corpus"] = "discovery"
    combined = pd.concat([main, disc], ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Per-category price baseline, fit inside the slow universe (§2.0 / §4 stage 2)
# ---------------------------------------------------------------------------

def fit_slow_category_baselines(slow_bets: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Fit E[outcome | entry_price] per category over the SLOW universe (reusing the
    §2.0 fallback hierarchy from forecaster_metrics). `slow_bets` must carry
    `category`, `entry_price`, `resolved`, `resolved_value` — the resolved BUY bets
    from `load_slow_universe_bets`. Returns the category→PriceBaseline map."""
    cfg = cfg or load_config()
    scoring = cfg.get("scoring", {})
    n_bins = int(scoring.get("price_baseline_bins", 20))
    resolved = slow_bets.loc[slow_bets["resolved"].fillna(False)] if "resolved" in slow_bets else slow_bets
    return fit_category_baselines(resolved, n_bins=n_bins)


def residualize(slow_bets: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Attach `expected_outcome` (per-category structural E[outcome|price]) and
    `residual_skill` (= resolved_value − expected_outcome) to the slow bets. This
    residual — favorite-longshot-neutralized against the RIGHT per-category curve —
    is exactly what step 5's screen aggregates, shrunk, per (wallet, category)."""
    out = slow_bets.copy()
    exp = expected_outcome_by_category(baselines, out["category"].to_numpy(),
                                       out["entry_price"].to_numpy())
    out["expected_outcome"] = exp
    out["residual_skill"] = out["resolved_value"].to_numpy(dtype=float) - exp
    return out


def baseline_summary(slow_bets: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Per-category diagnostic (NOT a published finding): n resolved bets, whether the
    category got its OWN fitted curve or fell back, mean raw outcome, mean raw edge,
    and mean residual (skill) edge. Lets a reviewer see if the slow universe is too
    thin for a category to earn its own calibration (§4 stage 2 thin-sample concern)."""
    resolved = slow_bets.loc[slow_bets["resolved"].fillna(False)].copy()
    if resolved.empty:
        return pd.DataFrame()
    resid = residualize(resolved, baselines)
    rows = []
    for cat, g in resid.groupby("category"):
        own = cat in baselines and baselines[cat] is not baselines.get(REAL_WORLD_KEY) \
            and baselines[cat] is not baselines.get(GLOBAL_KEY)
        rows.append({
            "category": cat,
            "n_resolved": len(g),
            "own_curve": bool(own),
            "mean_outcome": float(g["resolved_value"].mean()),
            "mean_raw_edge": float((g["resolved_value"] - g["entry_price"]).mean()),
            "mean_residual_skill": float(g["residual_skill"].mean()),
        })
    return pd.DataFrame(rows).sort_values("n_resolved", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The crude wide screen (§4 stage 2) — a SHORTLIST, never a finding
# ---------------------------------------------------------------------------

def screen_wallets(slow_bets: pd.DataFrame, baselines: dict, cfg: dict | None = None) -> pd.DataFrame:
    """Per-wallet slow-market screen score. For every (wallet, category) cell with
    >= screen_min_bets slow bets, the shrunk residual skill edge

        cell_score = n/(n+k) * mean(outcome - E_cat[outcome | entry_price])

    (skill edge only, NEVER win rate or profit; shrinkage k makes it usable at small
    n without letting one lucky bet top the list). Each wallet's `screen_score` is the
    max of its best single-category cell and its pooled-across-categories cell (strong
    in one category OR broadly across slow markets). Returns one row per wallet.

    This is a SELECTION statistic on shallow data — never ranked for a human, never
    written to data/processed/. Its whole job is to pick a shortlist to DEEPEN (5b),
    after which validation on the fresh deep data (5c) is the honest test."""
    cfg = cfg or load_config()
    slow_cfg = cfg.get("scoring", {}).get("slow", {}) or {}
    k = float(slow_cfg.get("screen_shrinkage_k", cfg.get("scoring", {}).get("min_sample_size", 30)))
    min_bets = int(slow_cfg.get("screen_min_bets", 2))

    resid = residualize(slow_bets, baselines)

    # Per (wallet, category) cell.
    cell = (resid.groupby(["wallet", "category"])
            .agg(n=("residual_skill", "size"), mean_resid=("residual_skill", "mean"))
            .reset_index())
    cell = cell[cell["n"] >= min_bets].copy()
    cell["shrunk"] = cell["n"] / (cell["n"] + k) * cell["mean_resid"]
    best = (cell.sort_values("shrunk").groupby("wallet").tail(1)
            [["wallet", "category", "n", "shrunk"]]
            .rename(columns={"category": "best_category", "n": "best_n", "shrunk": "best_shrunk"}))

    # Per-wallet pooled-across-categories cell (residuals are already category-neutral,
    # so pooling them is well-defined), plus a corpus breakdown for interpretation.
    pooled = (resid.groupby("wallet")
              .agg(total_n=("residual_skill", "size"),
                   mean_resid=("residual_skill", "mean"),
                   n_markets=("market_id", "nunique"))
              .reset_index())
    pooled = pooled[pooled["total_n"] >= min_bets].copy()
    pooled["pooled_shrunk"] = pooled["total_n"] / (pooled["total_n"] + k) * pooled["mean_resid"]
    # Corpus split — vectorized (a per-group python apply over ~463k wallets is a
    # performance trap; a size-crosstab is the same result in one pass).
    corpus = (resid.groupby(["wallet", "corpus"]).size().unstack(fill_value=0).reset_index())
    for c in ("main", "discovery"):
        if c not in corpus.columns:
            corpus[c] = 0
    corpus = corpus.rename(columns={"main": "n_main", "discovery": "n_discovery"})[
        ["wallet", "n_main", "n_discovery"]]

    scores = pooled.merge(best, on="wallet", how="left").merge(corpus, on="wallet", how="left")
    scores["screen_score"] = scores[["best_shrunk", "pooled_shrunk"]].max(axis=1)
    return scores.sort_values("screen_score", ascending=False).reset_index(drop=True)


def shortlist_size_table(scores: pd.DataFrame,
                         gates=(0.02, 0.05, 0.08, 0.10, 0.15)) -> pd.DataFrame:
    """How many wallets the shortlist holds at each candidate screen gate — the input
    to choosing the deepening (5b) budget. Reported, not auto-applied."""
    rows = [{"gate": g, "wallets": int((scores["screen_score"] >= g).sum())} for g in gates]
    return pd.DataFrame(rows)


def save_wallet_scores(scores: pd.DataFrame) -> None:
    SLOW_SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(scores, WALLET_SCORES_PATH, compression="gzip")


# ---------------------------------------------------------------------------
# CLI: run the screen, persist scores, report shortlist sizes (writes nothing to
# data/processed/ — the shortlist is a selection artifact, not a finding).
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slow-market crude wide screen (Project 3 step 5a): shrunk per-(wallet, "
                    "category) residual skill edge over both corpora -> a shortlist to deepen. "
                    "Read-only; persists only the isolated interim screen scores.")
    parser.add_argument("--buckets", type=str, default=",".join(SLOW_BUCKETS),
                        help=f"speed_buckets defining the slow universe (default: {','.join(SLOW_BUCKETS)})")
    parser.add_argument("--baseline-summary", action="store_true",
                        help="also print the per-category baseline coverage diagnostic (step 4)")
    args = parser.parse_args()
    buckets = tuple(b.strip() for b in args.buckets.split(",") if b.strip())

    bets = load_combined_slow_universe(buckets=buckets)
    if bets.empty:
        print("[slow_market] combined slow universe is empty — nothing to screen.")
        return
    cc = bets["corpus"].value_counts().to_dict()
    print(f"[slow_market] slow universe ({'/'.join(buckets)}): {len(bets):,} resolved BUY bets, "
          f"{bets['wallet'].nunique():,} wallets, {bets['market_id'].nunique():,} markets "
          f"(main={cc.get('main', 0):,}, discovery={cc.get('discovery', 0):,})")

    baselines = fit_slow_category_baselines(bets)
    if args.baseline_summary:
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print("\n[baseline coverage]\n" + baseline_summary(bets, baselines).to_string(index=False))

    scores = screen_wallets(bets, baselines)
    save_wallet_scores(scores)
    print(f"\n[slow_market] screened {len(scores):,} wallets (>= screen_min_bets slow bets); "
          f"scores -> {WALLET_SCORES_PATH}")
    print("\n[shortlist size at candidate gates] (screen_score = shrunk residual skill edge)")
    print(shortlist_size_table(scores).to_string(index=False))
    print("\n[top 15 by screen_score]")
    cols = ["wallet", "screen_score", "best_category", "best_n", "total_n", "n_markets",
            "n_main", "n_discovery"]
    with pd.option_context("display.width", 200):
        print(scores[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
