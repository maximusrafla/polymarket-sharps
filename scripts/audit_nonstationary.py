"""Read-only audit for metric D — non-stationary / regime-change wallets.

Quantifies how the chronological select-then-validate gate in src/validate.py
mishandles wallets whose skill (residual) edge is non-stationary in time:

  - "became-sharp": recent (held-out) half is significant AND clears the
    magnitude floor, but the early half residual is <= 0, so the wallet is never
    a *candidate* (in_sample_residual_edge > 0) and ranks as unvalidated even
    though it is sharp *now*.
  - "decaying": early-sharp candidate whose recent half fails — correctly not
    persisted, but the gate does not label real-decay vs. early-luck.

See DECISIONS.md "Metric D" and HANDOFF.md for the design this evidences.
Writes nothing; the ledger is only read. Reproduce with:

    PYTHONPATH=. python scripts/audit_nonstationary.py         # fast, from wallet_validated.parquet
    PYTHONPATH=. python scripts/audit_nonstationary.py --deep  # + per-bet trend + D3 recovery (loads the ledger)
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.common import WALLET_VALIDATED_PATH, load_config, load_ledger
from src.features import fit_price_baseline, only_buys, residual_edge_per_bet
from src.validate import DEFAULT_MIN_BETS_PER_HALF, DEFAULT_MIN_SKILL_EDGE, DEFAULT_OOS_ALPHA


def _thresholds(cfg: dict) -> tuple[float, float, int, float]:
    s = cfg["scoring"]
    return (
        # Deliberately the GLOBAL alpha, not Project 1's scoped certification
        # alpha (validate.certification_alpha): everything below is metric D,
        # which runs on the per-bet t-test p and never certifies, so it was left
        # on 0.05 when the certification gate was tightened to 0.005. See
        # DECISIONS.md "Tightened significance threshold".
        s.get("oos_significance_alpha", DEFAULT_OOS_ALPHA),
        s.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE),
        s.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF),
        s["oos_split"],
    )


def fast_populations(cfg: dict) -> None:
    """Core populations straight from the cached validated table — no ledger
    load, so this is instant and robust. Everything needed for the became-sharp
    / decaying counts is already columns in wallet_validated.parquet."""
    alpha, floor, min_half, _ = _thresholds(cfg)
    if not WALLET_VALIDATED_PATH.exists():
        print(f"[audit] {WALLET_VALIDATED_PATH} not found — run src.validate first.")
        return
    v = pd.read_parquet(WALLET_VALIDATED_PATH)
    ir, orr = v["in_sample_residual_edge"], v["out_of_sample_residual_edge"]
    op, inn, on = v["out_of_sample_residual_p"], v["in_sample_n"], v["out_of_sample_n"]

    testable = (inn >= min_half) & (on >= min_half)
    candidate = testable & (ir > 0)
    out_sig = (orr > 0) & (op < alpha)
    out_mag = orr >= floor
    persisted = candidate & out_sig & out_mag
    became = testable & ~(ir > 0) & out_sig & out_mag
    decaying = candidate & ~(out_sig & out_mag)

    print(f"=== POPULATION (alpha={alpha}, floor={floor}, min_half={min_half}) ===")
    print(f"total wallets:                          {len(v)}")
    print(f"testable (both halves >= {min_half} bets):   {int(testable.sum())}")
    print(f"candidates (in_sample_residual>0):      {int(candidate.sum())}")
    print(f"persisted (edge_persisted):             {int(v['edge_persisted'].sum())}"
          f"  (recomputed here: {int(persisted.sum())})")
    print(f"BECAME-SHARP (recent sig+material, early residual<=0): {int(became.sum())}")
    print(f"  of those, recent half >=100 bets:     {int((became & (on >= 100)).sum())}")
    print(f"  of those, recent half >=500 bets:     {int((became & (on >= 500)).sum())}  <- D3-recoverable")
    print(f"DECAYING (candidate, recent fails):     {int(decaying.sum())}")

    if became.any():
        show = v.loc[became, ["wallet", "in_sample_n", "out_of_sample_n",
                              "in_sample_residual_edge", "out_of_sample_residual_edge",
                              "out_of_sample_residual_p"]].copy()
        show["wallet"] = show["wallet"].str.slice(0, 12)
        show = show.sort_values("out_of_sample_residual_edge", ascending=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print("\n--- became-sharp detail (by out-of-sample residual edge) ---")
            print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def _gated_mean(vals, min_n):
    a = np.asarray(vals, dtype=float)
    return (np.nan if a.size < min_n else float(a.mean())), a.size


def _oos_p(resid, min_n):
    r = np.asarray(resid, dtype=float)
    if r.size < max(min_n, 2) or np.allclose(r.std(), 0.0):
        return np.nan
    return float(stats.ttest_1samp(r, 0.0, alternative="greater").pvalue)


def deep_pass(cfg: dict) -> None:
    """Per-bet trend classification + D3 within-recent-regime recovery counts.
    Loads the ledger and residualizes (memory-safe: lazy groupby, no full
    materialization) so it runs on the Chromebook."""
    alpha, floor, min_half, oos_split = _thresholds(cfg)
    ledger = load_ledger()
    if ledger.empty:
        print("[audit] ledger empty — skip --deep.")
        return
    resolved = only_buys(ledger)
    resolved = resolved.loc[resolved["resolved"]].copy()
    baseline = fit_price_baseline(resolved, cfg["scoring"].get("price_baseline_bins", 20))
    resolved["residual_edge"] = residual_edge_per_bet(resolved, baseline)

    n_improve = n_decay = n_flat = 0
    became_n = recov_n = 0
    for _, g in resolved.groupby("wallet", sort=False):  # lazy → low memory
        g = g.sort_values("timestamp")
        n = len(g)
        si = math.floor(n * oos_split)
        ins, out = g.iloc[:si], g.iloc[si:]
        in_r, in_n = _gated_mean(ins["residual_edge"], min_half)
        out_r, out_n = _gated_mean(out["residual_edge"], min_half)
        if in_n < min_half or out_n < min_half:
            continue
        # monotonic trend over the whole per-bet residual series
        if n >= 8:
            rho, p = stats.spearmanr(np.arange(n), g["residual_edge"].to_numpy())
            if p < 0.05 and rho > 0:
                n_improve += 1
            elif p < 0.05 and rho < 0:
                n_decay += 1
            else:
                n_flat += 1
        # D3 recovery: became-sharp AND both recent sub-halves sig+material
        out_p = _oos_p(out["residual_edge"], min_half)
        became = (not np.isnan(in_r) and in_r <= 0
                  and not np.isnan(out_r) and out_r > 0 and out_r >= floor
                  and not np.isnan(out_p) and out_p < alpha)
        if became:
            became_n += 1
            h = math.floor(len(out) * 0.5)
            a, b = out.iloc[:h], out.iloc[h:]
            ea, _ = _gated_mean(a["residual_edge"], min_half)
            eb, _ = _gated_mean(b["residual_edge"], min_half)
            pa, pb = _oos_p(a["residual_edge"], min_half), _oos_p(b["residual_edge"], min_half)
            if (not np.isnan(ea) and ea >= floor and not np.isnan(pa) and pa < alpha
                    and not np.isnan(eb) and eb >= floor and not np.isnan(pb) and pb < alpha):
                recov_n += 1

    print("\n=== DEEP PASS (per-bet, from the ledger) ===")
    print(f"Spearman residual-vs-time trend:  improving={n_improve}  decaying={n_decay}  flat={n_flat}")
    print(f"D3 recovery: of {became_n} became-sharp, both recent sub-halves "
          f"significant+material: {recov_n}")
    print("  (these are the ones a recency-anchored validator could certify leakage-free)")


def main() -> None:
    cfg = load_config()
    fast_populations(cfg)
    if "--deep" in sys.argv:
        deep_pass(cfg)
    else:
        print("\n(pass --deep for the per-bet trend + D3 recovery counts — loads the ledger)")


if __name__ == "__main__":
    main()
