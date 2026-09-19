"""G0 of the market-centric value-betting gate: is the favorite-longshot "edge"
a property of the MARKET, or of the WALLETS we selected?

Read-only. Touches nothing but parquet reads.

Why this runs first
-------------------
The value-betting thesis needs a structural mispricing to trade. The one we have
"already measured" is E[outcome | entry_price] != entry_price, quoted at ~+1.8c
average buyer's edge and ~+14c in the 0.6-0.8 band. Those numbers come from
`data/interim/bet_ledger.parquet`, which is 500 wallets *chosen by this repo's own
sharpness ranking* and then deep-backfilled. A calibration curve fit on wallets
selected for having positive edge will show positive edge by construction, and it
is not a curve anyone can trade.

`data/interim/discovery/discovery_trades.parquet` is the control: `/trades?market=`
tapes, i.e. EVERY wallet that traded those markets. It is unselected with respect
to wallet skill, which is exactly the bias in question.

Two known biases in the census, both controlled here:
  - Truncation. `/trades?market=` caps at ~10,500 rows and drops the EARLIEST
    entries, so a capped market's tape is its near-resolution tail, where prices
    sit at 0/1. Markets at the cap are flagged and reported separately.
  - Regime mix. Micro-crypto and real-world markets are different price processes;
    reported split, never pooled silently.

The sharpest cut is the MATCHED-MARKET test: markets present in both datasets, so
the market, the period and the price process are held fixed and the only thing
that varies is which wallets you look at.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/audit_flb_selection.py [--resamples 500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CENSUS = Path("data/interim/discovery/discovery_trades.parquet")
LEDGER = Path("data/interim/bet_ledger.parquet")
OUTDIR = Path("data/interim/value")

# Tape truncation: /trades?market= paging stops at offset 10000 with page size 500.
# A market at (or fractionally under) that cap lost its early history.
TAPE_CAP_ROWS = 10_400

# Fixed price bands. Deliberately not quantile bands: we need to read the SAME
# band across datasets whose price distributions differ, and the brief's claim is
# stated for a fixed band (0.6-0.8).
BAND_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0]

# The band the brief quotes at +14c, kept as its own headline row.
HEADLINE_BAND = (0.60, 0.80)


def load_buys(path: Path, columns: list[str]) -> pd.DataFrame:
    """Resolved BUY prints with a usable price. SELLs are position exits, not
    directional entries at a price, so `resolved_value - entry_price` is not their
    P&L and they are excluded from every calibration here."""
    df = pd.read_parquet(path, columns=columns)
    df = df[
        (df["side"] == "BUY")
        & df["resolved"].fillna(False)
        & df["resolved_value"].notna()
        & df["entry_price"].notna()
        & (df["entry_price"] > 0.0)
        & (df["entry_price"] < 1.0)
    ].copy()
    df["edge"] = df["resolved_value"] - df["entry_price"]
    df["band"] = pd.cut(df["entry_price"], BAND_EDGES, right=False)
    return df


def _band_label(iv) -> str:
    return f"[{iv.left:.2f},{iv.right:.2f})"


def cluster_band_stats(df: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    """Mean raw edge per band with a MARKET-BLOCK bootstrap CI.

    Bets in one market share a single resolution, so a per-bet CI is badly
    overstated (this repo has measured design effects up to 230). Resampling
    whole markets is the same correction `validate._cluster_bootstrap_p` applies.

    Implemented on per-(market, band) sums so a resample is an index into a small
    matrix rather than a re-scan of millions of rows.
    """
    g = df.groupby(["market_id", "band"], observed=True)["edge"].agg(["sum", "count"])
    if g.empty:
        return pd.DataFrame()
    sums = g["sum"].unstack(fill_value=0.0)
    cnts = g["count"].unstack(fill_value=0)
    bands = list(sums.columns)
    S, C = sums.to_numpy(float), cnts.to_numpy(float)
    n_mkt = S.shape[0]

    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(bands)), float)
    for i in range(resamples):
        idx = rng.integers(0, n_mkt, n_mkt)
        c = C[idx].sum(axis=0)
        draws[i] = np.where(c > 0, S[idx].sum(axis=0) / np.where(c > 0, c, 1), np.nan)

    tot_c, tot_s = C.sum(axis=0), S.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.where(tot_c > 0, tot_s / tot_c, np.nan)
    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)

    out = pd.DataFrame({
        "band": [_band_label(b) for b in bands],
        "n_bets": tot_c.astype(int),
        "n_markets": (C > 0).sum(axis=0).astype(int),
        "mean_price": np.nan,
        "mean_outcome": np.nan,
        "edge": point,
        "ci_lo": lo,
        "ci_hi": hi,
    })
    px = df.groupby("band", observed=True).agg(
        mean_price=("entry_price", "mean"), mean_outcome=("resolved_value", "mean"))
    px.index = [_band_label(b) for b in px.index]
    out["mean_price"] = out["band"].map(px["mean_price"])
    out["mean_outcome"] = out["band"].map(px["mean_outcome"])
    return out[out["n_bets"] > 0].reset_index(drop=True)


def overall(df: pd.DataFrame, resamples: int, seed: int) -> dict:
    """Aggregate edge + market-block CI, plus the headline 0.6-0.8 band."""
    g = df.groupby("market_id")["edge"].agg(["sum", "count"])
    S, C = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, len(S), len(S))
        draws[i] = S[idx].sum() / C[idx].sum()
    h = df[(df["entry_price"] >= HEADLINE_BAND[0]) & (df["entry_price"] < HEADLINE_BAND[1])]
    return {
        "n_bets": int(len(df)),
        "n_markets": int(df["market_id"].nunique()),
        "n_wallets": int(df["wallet"].nunique()),
        "edge": float(df["edge"].mean()),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "size_wtd_edge": float(np.average(df["edge"], weights=df["size"]))
        if df["size"].notna().any() and df["size"].sum() > 0 else float("nan"),
        "headline_band_n": int(len(h)),
        "headline_band_edge": float(h["edge"].mean()) if len(h) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cols = ["wallet", "market_id", "side", "entry_price", "size",
            "timestamp", "resolved", "resolved_value"]
    print("loading census tape (market-centric, unselected wallets) ...")
    cen = load_buys(CENSUS, cols + ["category"])
    rows_per_mkt = cen.groupby("market_id")["edge"].size()
    cen["truncated"] = cen["market_id"].map(rows_per_mkt >= TAPE_CAP_ROWS)
    cen["real_world"] = cen["category"].ne("crypto_event")

    print("loading wallet-selected ledger ...")
    led = load_buys(LEDGER, cols)

    matched = sorted(set(cen["market_id"]) & set(led["market_id"]))
    print(f"census {len(cen):,} bets / {cen.market_id.nunique()} markets; "
          f"ledger {len(led):,} bets; matched markets {len(matched)}")

    cohorts: dict[str, pd.DataFrame] = {
        "ledger_selected_wallets": led,
        "census_all": cen,
        "census_untruncated": cen[~cen["truncated"]],
        "census_untrunc_realworld": cen[~cen["truncated"] & cen["real_world"]],
        "matched_census": cen[cen["market_id"].isin(matched)],
        "matched_ledger": led[led["market_id"].isin(matched)],
    }

    summary, band_frames = {}, []
    for i, (name, df) in enumerate(cohorts.items()):
        if df.empty:
            continue
        summary[name] = overall(df, args.resamples, args.seed + i)
        bf = cluster_band_stats(df, args.resamples, args.seed + 100 + i)
        bf.insert(0, "cohort", name)
        band_frames.append(bf)

    bands = pd.concat(band_frames, ignore_index=True)
    bands.to_parquet(OUTDIR / "g0_flb_bands.parquet", index=False)
    (OUTDIR / "g0_flb_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== AGGREGATE BUY-SIDE RAW EDGE (market-block bootstrap 95% CI) ===")
    print(f"{'cohort':<26}{'bets':>11}{'mkts':>7}{'wallets':>9}"
          f"{'edge':>9}{'95% CI':>20}{'0.6-0.8':>10}")
    for name, s in summary.items():
        ci = f"[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]"
        print(f"{name:<26}{s['n_bets']:>11,}{s['n_markets']:>7,}{s['n_wallets']:>9,}"
              f"{s['edge']:>+9.4f}{ci:>20}{s['headline_band_edge']:>+10.4f}")

    for name in cohorts:
        sub = bands[bands["cohort"] == name]
        if sub.empty:
            continue
        print(f"\n--- {name} ---")
        print(f"{'band':<14}{'n':>10}{'mkts':>7}{'E[y]':>8}{'p':>8}{'edge':>9}{'95% CI':>20}")
        for _, r in sub.iterrows():
            ci = f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
            print(f"{r.band:<14}{r.n_bets:>10,}{r.n_markets:>7,}{r.mean_outcome:>8.3f}"
                  f"{r.mean_price:>8.3f}{r.edge:>+9.4f}{ci:>20}")

    print(f"\nwrote {OUTDIR/'g0_flb_bands.parquet'} and {OUTDIR/'g0_flb_summary.json'}")


if __name__ == "__main__":
    main()
