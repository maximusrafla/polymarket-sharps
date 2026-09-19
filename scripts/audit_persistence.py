"""Red-team audit of the out-of-sample persistence rate reported by validate.py.

Runs read-only against data/interim/bet_ledger.parquet. Writes nothing to
data/. Reproduces the numbers documented in HANDOFF.md ("Red-team results").

Two questions:
  1. Is the high persistence rate look-ahead LEAKAGE, or a weak test?
  2. How much of it survives a shuffled-outcomes null and favorite-longshot
     neutralization — i.e. how much is genuine skill vs. structure?

Usage:  PYTHONPATH=. python scripts/audit_persistence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import load_config, load_ledger
from src.features import only_buys

SEED = 12345           # fixed for reproducibility (no Date/random in pipeline)
MIN_BETS_PER_HALF = 2  # mirrors validate.MIN_BETS_PER_HALF
N_SHUFFLES = 20


def _persistence(df: pd.DataFrame, oos: float, value_col: str = "resolved_value"):
    """Reproduce validate.compute_oos_validation's candidate/persist counts."""
    cand = persist = 0
    for _w, g in df.groupby("wallet"):
        g = g.sort_values("timestamp")
        k = int(np.floor(len(g) * oos))
        ins, out = g.iloc[:k], g.iloc[k:]
        ie = (ins[value_col] - ins["entry_price"]).mean() if len(ins) >= MIN_BETS_PER_HALF else np.nan
        oe = (out[value_col] - out["entry_price"]).mean() if len(out) >= MIN_BETS_PER_HALF else np.nan
        if (not np.isnan(ie)) and ie > 0:
            cand += 1
            if (not np.isnan(oe)) and oe > 0:
                persist += 1
    return cand, persist


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg = load_config()
    oos = cfg["scoring"]["oos_split"]
    r = only_buys(load_ledger())
    r = r.loc[r["resolved"]].copy().sort_values("timestamp")
    r["edge"] = r["resolved_value"] - r["entry_price"]

    print("=== 1. REAL DATA ===")
    cand, persist = _persistence(r, oos)
    print(f"candidates={cand}  persisted={persist}  rate={persist/cand:.1%}")

    print("\n=== 2. GLOBAL STRUCTURE (why the null floor is not 50%) ===")
    print(f"mean resolved_value={r['resolved_value'].mean():.4f}  "
          f"mean entry_price={r['entry_price'].mean():.4f}  "
          f"mean edge={r['edge'].mean():+.4f}")
    print(f"share of individual bets with positive edge: {(r['edge'] > 0).mean():.1%}")

    print("\n=== 3. FAVORITE-LONGSHOT: mean edge by entry-price bucket ===")
    r["pxb"] = pd.cut(r["entry_price"], [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    print(r.groupby("pxb", observed=True)["edge"].agg(["size", "mean"]).to_string())

    print("\n=== 4. SPLIT INTEGRITY (is there cross-half market leakage?) ===")
    spans = []
    for _w, g in r.groupby("wallet"):
        g = g.sort_values("timestamp")
        k = int(np.floor(len(g) * oos))
        im, om = set(g.iloc[:k]["market_id"]), set(g.iloc[k:]["market_id"])
        if om:
            spans.append(len(im & om) / len(om))
    print(f"median share of held-out markets also seen in-sample: {np.median(spans):.1%} "
          "(0% => chronological split does not leak outcomes across halves)")

    print("\n=== 5. SHUFFLED-OUTCOMES NULL TEST ===")
    for mode in ("global", "by_market"):
        rates = []
        for _ in range(N_SHUFFLES):
            rr = r.copy()
            if mode == "global":
                rr["resolved_value"] = rng.permutation(rr["resolved_value"].to_numpy())
            else:  # preserve shared-outcome-within-market structure
                mkt = rr.groupby("market_id")["resolved_value"].first()
                shuf = pd.Series(rng.permutation(mkt.to_numpy()), index=mkt.index)
                rr["resolved_value"] = rr["market_id"].map(shuf)
            c, p = _persistence(rr, oos)
            rates.append(p / c)
        rates = np.array(rates)
        print(f"  {mode:9s}: persistence {rates.mean():.1%} "
              f"(min {rates.min():.1%}, max {rates.max():.1%})")

    print("\n=== 6. FAVORITE-LONGSHOT-NEUTRALIZED (residual) persistence ===")
    r["pxb5"] = (r["entry_price"] * 20).round() / 20  # 5-cent buckets
    r["edge_resid"] = r["edge"] - r.groupby("pxb5")["edge"].transform("mean")
    cand, persist = _persistence(r.assign(resolved_value=r["edge_resid"] + r["entry_price"]), oos)
    print(f"real residual: candidates={cand} persisted={persist} rate={persist/cand:.1%}")
    rates = []
    for _ in range(N_SHUFFLES):
        rr = r.copy()
        shuf_resid = rng.permutation(rr["edge_resid"].to_numpy())
        rr["resolved_value"] = shuf_resid + rr["entry_price"]
        c, p = _persistence(rr, oos)
        rates.append(p / c)
    rates = np.array(rates)
    print(f"null residual: persistence {rates.mean():.1%} (min {rates.min():.1%}, max {rates.max():.1%})")
    print("\n=> genuine-skill signal ~= (real residual - null residual) persistence points.")


if __name__ == "__main__":
    main()
