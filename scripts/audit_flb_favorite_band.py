"""G0b: is the favorite-band edge in the unselected census a real, independent
signal — or a handful of resolution events plus near-settlement dust?

Read-only.

G0a (`audit_flb_selection.py`) found the tradeable-looking part of the census
calibration curve is NOT the 0.6-0.8 band the brief quotes (that band is
negative on unselected data) but the HEAVY FAVOURITE bands, 0.90-1.00, where
E[y] > p with market-block CIs excluding zero. Before that counts as an edge
source it has to survive three things this repo has been burned by before:

  1. CLUSTER UNIT. 585 `market_id`s are not 585 independent draws — season
     outrights, derivative lines on one game and rolling-deadline ladders all
     resolve together. `src/sports_events.resolution_event` folds them to the
     unit that actually resolves. Re-bootstrap on THAT.

  2. TAIL COUNT. Buying at 0.98 is selling insurance: you collect ~0.4c almost
     always and lose ~98c rarely. The mean is only as trustworthy as the number
     of times the tail actually fired IN THIS SAMPLE. Count the losses at the
     event level, not the bet level.

  3. SETTLEMENT DUST. If the favourite-band edge sits in the last slice of a
     market's life, it is not a mispricing you can deploy into — it is carrying
     a near-certain contract to settlement for a few ticks, with capital locked
     and no capacity. Split by position in the market's tape.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/audit_flb_favorite_band.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.sports_events import assign_events

CENSUS = Path("data/interim/discovery/discovery_trades.parquet")
OUTDIR = Path("data/interim/value")

BAND_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0]
TAPE_CAP_ROWS = 10_400


def _label(iv) -> str:
    return f"[{iv.left:.2f},{iv.right:.2f})"


def block_ci(df: pd.DataFrame, unit: str, resamples: int, seed: int) -> pd.DataFrame:
    """Per-band mean edge with a bootstrap that resamples whole `unit` blocks."""
    g = df.groupby([unit, "band"], observed=True)["edge"].agg(["sum", "count"])
    sums, cnts = g["sum"].unstack(fill_value=0.0), g["count"].unstack(fill_value=0)
    bands = list(sums.columns)
    S, C = sums.to_numpy(float), cnts.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(bands)))
    for i in range(resamples):
        idx = rng.integers(0, S.shape[0], S.shape[0])
        c = C[idx].sum(axis=0)
        draws[i] = np.where(c > 0, S[idx].sum(axis=0) / np.where(c > 0, c, 1), np.nan)
    tot_c, tot_s = C.sum(axis=0), S.sum(axis=0)
    return pd.DataFrame({
        "band": [_label(b) for b in bands],
        "n_bets": tot_c.astype(int),
        f"n_{unit}": (C > 0).sum(axis=0).astype(int),
        "edge": np.where(tot_c > 0, tot_s / np.where(tot_c > 0, tot_c, 1), np.nan),
        "ci_lo": np.nanpercentile(draws, 2.5, axis=0),
        "ci_hi": np.nanpercentile(draws, 97.5, axis=0),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CENSUS, columns=[
        "wallet", "market_id", "slug", "question", "token_id", "side",
        "entry_price", "size", "timestamp", "resolved", "resolved_value", "category"])
    df = df[(df["side"] == "BUY") & df["resolved"].fillna(False)
            & df["resolved_value"].notna() & df["entry_price"].between(0, 1, "neither")].copy()
    df["edge"] = df["resolved_value"] - df["entry_price"]
    df["band"] = pd.cut(df["entry_price"], BAND_EDGES, right=False)

    # --- 1. cluster unit -----------------------------------------------------
    mk = df[["market_id", "slug", "question"]].drop_duplicates("market_id")
    mk = assign_events(mk)
    df = df.merge(mk[["market_id", "event", "event_kind"]], on="market_id", how="left")
    n_mkt, n_evt = df["market_id"].nunique(), df["event"].nunique()
    print(f"census: {len(df):,} resolved BUY bets, {n_mkt} markets -> {n_evt} resolution events")
    top = (df.drop_duplicates("market_id").groupby("event").size()
             .sort_values(ascending=False).head(8))
    print("largest events (markets folded):", top.to_dict())

    mkt_ci = block_ci(df, "market_id", args.resamples, args.seed)
    evt_ci = block_ci(df, "event", args.resamples, args.seed)
    cmp = mkt_ci.merge(evt_ci.drop(columns=["n_bets", "edge"]), on="band",
                       suffixes=("_mkt", "_evt"))
    cmp["width_mkt"] = cmp["ci_hi_mkt"] - cmp["ci_lo_mkt"]
    cmp["width_evt"] = cmp["ci_hi_evt"] - cmp["ci_lo_evt"]
    cmp["design_effect"] = (cmp["width_evt"] / cmp["width_mkt"]) ** 2

    print("\n=== band edge: MARKET-block vs EVENT-block bootstrap ===")
    print(f"{'band':<14}{'n':>10}{'mkts':>6}{'evts':>6}{'edge':>9}"
          f"{'CI(market)':>21}{'CI(event)':>21}{'DE':>7}")
    for _, r in cmp.iterrows():
        a = f"[{r.ci_lo_mkt:+.4f},{r.ci_hi_mkt:+.4f}]"
        b = f"[{r.ci_lo_evt:+.4f},{r.ci_hi_evt:+.4f}]"
        print(f"{r.band:<14}{r.n_bets:>10,}{r.n_market_id:>6}{r.n_event:>6}"
              f"{r.edge:>+9.4f}{a:>21}{b:>21}{r.design_effect:>7.2f}")

    # --- 2. tail count: how often did the favourite actually lose? -----------
    fav = df[df["entry_price"] >= 0.90].copy()
    fav["lost"] = fav["resolved_value"] < 0.5
    per_evt = fav.groupby("event").agg(
        n=("edge", "size"), n_lost=("lost", "sum"), edge=("edge", "mean"))
    print("\n=== TAIL REALISATIONS in the 0.90+ favourite bands ===")
    print(f"bets {len(fav):,} | losing bets {int(fav['lost'].sum()):,} "
          f"({fav['lost'].mean():.4%}) | events {len(per_evt)} | "
          f"events with >=1 loss {int((per_evt['n_lost'] > 0).sum())}")
    for lo, hi in [(0.90, 0.95), (0.95, 0.98), (0.98, 1.0)]:
        s = fav[(fav.entry_price >= lo) & (fav.entry_price < hi)]
        pe = s.groupby("event")["edge"].mean()
        n_loss_evt = int((s.groupby("event")["lost"].sum() > 0).sum())
        # break-even loss rate: at price p you need P(win) > p to profit
        print(f"  [{lo},{hi}) n={len(s):>8,} events={len(pe):>4} events_with_loss={n_loss_evt:>3} "
              f"realised_loss_rate={s['lost'].mean():.4%} break_even_loss_rate={1-s.entry_price.mean():.4%} "
              f"edge={s['edge'].mean():+.4f} | event-weighted edge={pe.mean():+.4f}")

    # --- 3. settlement dust: where in the market's life is the edge? ---------
    t = df.groupby("market_id")["timestamp"].agg(["min", "max"])
    df = df.join(t, on="market_id")
    span = (df["max"] - df["min"]).replace(0, np.nan)
    df["rel_t"] = ((df["timestamp"] - df["min"]) / span).clip(0, 1)
    rows_per_mkt = df.groupby("market_id")["edge"].size()
    df["truncated"] = df["market_id"].map(rows_per_mkt >= TAPE_CAP_ROWS)
    df["life_q"] = pd.cut(df["rel_t"], [0, .25, .5, .75, .9, 1.0],
                          labels=["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"],
                          include_lowest=True)
    print("\n=== favourite-band edge by POSITION IN THE MARKET'S TAPE (untruncated only) ===")
    print("(rel_t = 1.0 is the last trade before resolution; a positive edge that lives")
    print(" only in the last slice is settlement carry, not a deployable mispricing)")
    u = df[~df["truncated"] & (df["entry_price"] >= 0.90)]
    tab = u.groupby(["band", "life_q"], observed=True).agg(
        n=("edge", "size"), evts=("event", "nunique"), edge=("edge", "mean"))
    print(tab.reset_index().to_string(index=False))

    print("\n=== all-band edge by tape position (untruncated) ===")
    tab2 = df[~df["truncated"]].groupby("life_q", observed=True).agg(
        n=("edge", "size"), evts=("event", "nunique"),
        mean_price=("entry_price", "mean"), edge=("edge", "mean"))
    print(tab2.to_string())

    cmp.to_parquet(OUTDIR / "g0b_cluster_unit.parquet", index=False)
    (OUTDIR / "g0b_summary.json").write_text(json.dumps({
        "n_markets": int(n_mkt), "n_events": int(n_evt),
        "fav_bets": int(len(fav)), "fav_losing_bets": int(fav["lost"].sum()),
        "fav_events": int(len(per_evt)),
        "fav_events_with_loss": int((per_evt["n_lost"] > 0).sum()),
    }, indent=2))
    print(f"\nwrote {OUTDIR/'g0b_cluster_unit.parquet'}")


if __name__ == "__main__":
    main()
