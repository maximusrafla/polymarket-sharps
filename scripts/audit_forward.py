"""Audit forward-price leakage in copy_window / earliness (features.forward_price).
Read-only against data/interim/bet_ledger.parquet; writes nothing. Reproduces the
findings documented in HANDOFF.md / DECISIONS.md "Forward-price leakage fix":
copy_window's correlation with the realized outcome, before vs after the
resolution guard, and the exact copy_window == earliness coincidence.

Usage:  PYTHONPATH=. python scripts/audit_forward.py"""
import numpy as np, pandas as pd
from src.common import load_ledger, load_config
from src.features import only_buys, compute_forward_drift, fit_price_baseline, residual_edge_per_bet

cfg = load_config()
ledger = load_ledger()
bets = only_buys(ledger)
resolved_bets = bets.loc[bets["resolved"]].copy()

CW_H = cfg["scoring"]["copy_window_hours"]     # 24
EA_H = cfg["scoring"]["earliness_window_hours"] # 6

# --- instrumented replica of _forward_price_arrays path selection ----------
def classify_paths(bets, ledger, window_hours):
    window_seconds = window_hours * 3600
    ledger_by_token = {t: g for t, g in ledger.groupby("token_id")}
    counts = {"no_future": 0, "in_window": 0, "beyond_window": 0}
    outcome_when_fallback = []   # (fair_value_used, resolved_value) for no_future -> uses resolved_value
    beyond_gap_hours = []
    for token_id, tb in bets.groupby("token_id"):
        tt = ledger_by_token.get(token_id)
        if tt is None: continue
        ts = tt["timestamp"].to_numpy(); wal = tt["wallet"].to_numpy()
        for bw, bts, bres, brv in tb[["wallet","timestamp","resolved","resolved_value"]].itertuples(index=False, name=None):
            m = (wal != bw) & (ts > bts)
            if not m.any():
                counts["no_future"] += 1
                if bres: outcome_when_fallback.append(brv)
                continue
            fts = ts[m]
            inw = fts <= bts + window_seconds
            if inw.any():
                counts["in_window"] += 1
            else:
                counts["beyond_window"] += 1
                beyond_gap_hours.append((fts.min() - bts)/3600)
    return counts, outcome_when_fallback, beyond_gap_hours

for name, wh in [("copy_window(24h)", CW_H), ("earliness(6h)", EA_H)]:
    c, ofb, gaps = classify_paths(resolved_bets, ledger, wh)
    tot = sum(c.values())
    print(f"\n=== {name}: path taken per resolved bet (n={tot}) ===")
    for k,v in c.items():
        print(f"  {k:14s}: {v:6d} ({v/tot:.1%})")
    if ofb:
        print(f"  -> no_future fallback injects resolved_value as 'fair value' for {len(ofb)} resolved bets; mean outcome {np.mean(ofb):.3f}")
    if gaps:
        print(f"  -> beyond_window nearest-trade gap hours: median {np.median(gaps):.1f}, p90 {np.percentile(gaps,90):.1f} (unbounded horizon)")

# --- outcome-leakage correlation, BEFORE (guard=0) vs AFTER (config guard) -----
baseline = fit_price_baseline(resolved_bets, cfg["scoring"].get("price_baseline_bins",20))
resolved_bets["resid"] = residual_edge_per_bet(resolved_bets, baseline)
edge = (resolved_bets["resolved_value"]-resolved_bets["entry_price"]).groupby(resolved_bets["wallet"]).mean().rename("raw_edge")
skill = resolved_bets.groupby("wallet")["resid"].mean().rename("skill_edge")
cfg_guard = cfg["scoring"].get("fair_value_resolution_guard", 0.2)
print("\n=== copy_window leakage: correlation with realized outcome, guard sweep ===")
for label, guard in [("BEFORE (guard=0.0)", 0.0), (f"AFTER  (guard={cfg_guard})", cfg_guard)]:
    cw = compute_forward_drift(bets, ledger, CW_H, guard).rename("copy_window")
    ea = compute_forward_drift(bets, ledger, EA_H, guard).rename("earliness")
    df = pd.concat([cw, ea, edge, skill], axis=1).dropna()
    ident = np.isclose(df["copy_window"], df["earliness"]).mean() if len(df) else float("nan")
    print(f"  {label}: wallets={len(df):4d}  corr(cw,raw_edge)={df['copy_window'].corr(df['raw_edge']):.3f}  "
          f"corr(cw,skill)={df['copy_window'].corr(df['skill_edge']):.3f}  copy==early={ident:.1%}")
print("\n=> guard cuts the outcome correlation; copy==early stays 100% (windows are")
print("   redundant in this clustered-trade data), which is why only copy_window is scored.")
print("DONE")
