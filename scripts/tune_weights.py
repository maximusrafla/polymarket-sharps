"""Validate (and, when the data supports it, re-tune) the ranking weights against
a leakage-free forward target: each wallet's HELD-OUT (second-half) skill edge,
predicted from features measured on its FIRST half only. Read-only; writes nothing.

The question this answers is NOT "what weights fit the data" (that overfits ~100
noisy wallets) but "does any weight scheme predict future skill better than another,
beyond noise?" As of 2026-07-18 the answer is no — see HANDOFF.md / DECISIONS.md
"Ranking weights". Re-run this as the ledger grows: if the bootstrap CI on a
scheme difference ever excludes 0, that is the signal to change config weights.

Usage:  PYTHONPATH=. python scripts/tune_weights.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.common import load_config, load_ledger
from src.features import fit_price_baseline, only_buys, residual_edge_per_bet


def _per_bet_forward_drift(resolved: pd.DataFrame, ledger: pd.DataFrame,
                           window_hours: float, guard: float) -> pd.Series:
    """Leakage-guarded copy_window drift per bet (mirrors features.compute_forward_drift
    but keyed to each bet's row index so it can be split in/out of sample)."""
    drift = pd.Series(np.nan, index=resolved.index)
    ws = window_hours * 3600
    by_token = {t: g for t, g in ledger.groupby("token_id")}
    for tid, tb in resolved.groupby("token_id"):
        tt = by_token.get(tid)
        if tt is None:
            continue
        ts = tt["timestamp"].to_numpy(); pr = tt["entry_price"].to_numpy()
        sz = tt["size"].to_numpy(); wl = tt["wallet"].to_numpy()
        cutoff = ts.max() - guard * (ts.max() - ts.min())
        for idx, bw, bts, be in tb[["wallet", "timestamp", "entry_price"]].itertuples(name=None):
            m = (wl != bw) & (ts > bts) & (ts <= bts + ws) & (ts <= cutoff)
            if not m.any():
                continue
            w = sz[m].sum()
            fv = (pr[m] * sz[m]).sum() / w if w > 0 else pr[m].mean()
            drift.iloc[idx] = fv - be
    return drift


def build_dataset(min_half: int = 5) -> pd.DataFrame:
    cfg = load_config(); s = cfg["scoring"]
    oos = s["oos_split"]; guard = s.get("fair_value_resolution_guard", 0.2)
    ledger = load_ledger()
    resolved = (only_buys(ledger).pipe(lambda b: b.loc[b["resolved"]])
                .sort_values("timestamp").reset_index(drop=True))
    baseline = fit_price_baseline(resolved, s.get("price_baseline_bins", 20))
    resolved["skill"] = residual_edge_per_bet(resolved, baseline)
    resolved["drift"] = _per_bet_forward_drift(resolved, ledger, s["copy_window_hours"], guard)

    rows = []
    for w, g in resolved.groupby("wallet"):
        g = g.sort_values("timestamp"); k = math.floor(len(g) * oos)
        ins, out = g.iloc[:k], g.iloc[k:]
        if len(ins) < min_half or len(out) < min_half:
            continue
        tc = np.nan
        if len(ins) >= 4:
            b = pd.qcut(range(len(ins)), 4, labels=False)
            tc = float((ins["skill"].groupby(b).mean() > 0).mean())
        rows.append({
            "skill_in": ins["skill"].mean(),
            "drift_in": ins["drift"].mean(),
            "tc_in": tc,
            "skill_out": out["skill"].mean(),   # forward target
        })
    df = pd.DataFrame(rows)
    df["drift_in"] = df["drift_in"].fillna(0.0)
    df["tc_in"] = df["tc_in"].fillna(0.5)
    return df


def _score(d, w_edge, w_cw, w_tc):
    return w_edge * d["skill_in"] + w_cw * d["drift_in"] + w_tc * (d["tc_in"] - 0.5)


def main() -> None:
    cfg = load_config(); r = cfg["ranking"]
    current = (r["w_edge"], r["w_copy_window"], r["w_time_consistency"])

    print("=== forward predictability of in-sample skill edge, by bets/half ===")
    for mh in (5, 10, 20, 30):
        d = build_dataset(mh)
        if len(d) < 8:
            print(f"  >={mh:2d}/half: n={len(d)} (too few)"); continue
        rho, p = stats.spearmanr(d["skill_in"], d["skill_out"])
        print(f"  >={mh:2d}/half: wallets={len(d):3d}  rho(skill_in -> skill_out)={rho:+.3f}  p={p:.3f}")

    d = build_dataset(10)
    print(f"\n=== weight schemes: Spearman(in-sample score, held-out skill), n={len(d)} ===")
    schemes = {
        f"current {tuple(round(x,2) for x in current)}": current,
        "skill-only (1,0,0)": (1.0, 0.0, 0.0),
        "skill+small-cw (1,0.25,0)": (1.0, 0.25, 0.0),
        "cw-only (0,1,0)": (0.0, 1.0, 0.0),
    }
    for name, (a, b, c) in schemes.items():
        rho, _ = stats.spearmanr(_score(d, a, b, c), d["skill_out"])
        print(f"  {name:30s}: rho={rho:+.3f}")

    # Bootstrap: does the best alternative beat 'current' beyond noise?
    rng = np.random.default_rng(1); diffs = []
    for _ in range(2000):
        db = d.iloc[rng.integers(0, len(d), len(d))]
        r_cur, _ = stats.spearmanr(_score(db, *current), db["skill_out"])
        r_alt, _ = stats.spearmanr(_score(db, 1.0, 0.25, 0.0), db["skill_out"])
        diffs.append(r_alt - r_cur)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    changed = lo > 0 or hi < 0
    print(f"\n=== bootstrap: best-alt minus current (95% CI) ===")
    print(f"  mean={np.mean(diffs):+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")
    print("  VERDICT:", "a scheme beats current -> consider re-tuning config weights"
          if changed else "no scheme beats current beyond noise -> KEEP current weights")


if __name__ == "__main__":
    main()
