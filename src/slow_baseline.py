"""Hierarchical, leave-one-wallet-out price baseline for the slow universe —
Project 3 step 6b.

WHAT THIS REPLACES
------------------
Step 5c residualizes against `slow_market.fit_slow_category_baselines`: one
E[outcome | entry_price] curve per `category`, each on its own quantile bins.
With 59% of slow bets in `other`, a mispriced sub-niche inside `other` leaves a
positive residual on every buyer in it. This module fits the same object at
niche granularity without falling into the opposite trap.

THE TWO TRAPS, AND WHAT HANDLES EACH
------------------------------------
1. TOO COARSE (the 5c confound): the niche's own mispricing is left in the
   residual and reads as wallet skill.  ->  handled by fitting at niche level.

2. TOO FINE (the false negative): a bin holding a handful of bets has a mean
   that IS the local outcome rate, so residuals collapse to ~0 by construction
   and every wallet looks skill-less.  ->  handled by THREE things:

   (a) PARTIAL POOLING. Each cell's estimate is shrunk toward its parent's
       estimate in the SAME price bin:

           m[level, bin] = (n*ybar[level,bin] + k*m[parent, bin]) / (n + k)

       so a thin cell is mostly its parent and only earns its own value as
       evidence accumulates. `k` is estimated per level by method of moments
       (k = within-cell variance / between-cell variance) rather than hand-set,
       so it is not a free knob; `pooling_report` exposes the realized weight
       n/(n+k) per cell.

   (b) A HARD MIN-BIN FLOOR. Below `min_bin` effective bets a cell takes its
       parent's value exactly and is reported as pooled — belt and braces on top
       of (a), and it makes "this niche was never fit on noise" checkable.

   (c) LEAVE-ONE-WALLET-OUT. This is the one that actually decides the question.
       `features.fit_price_baseline` documents (correctly) that LOWO is
       unnecessary market-wide, because one wallet is a negligible share of a
       market-wide bin. At NICHE granularity that breaks: in the deep slow tape
       a single wallet can be a large share of one niche's volume, so its own
       outcomes would enter its own baseline and its residual would collapse
       toward zero mechanically — a false negative indistinguishable from "no
       skill". Every cell mean here is therefore computed EXCLUDING the
       evaluated wallet's own bets:

           ybar[level,bin]^(-w) = (S - S_w) / (n - n_w)

       recursed up the whole parent chain, so the exclusion is exact at every
       level. Without this a "survivors collapse" verdict would be uninterpretable.

BIN EDGES ARE SHARED ACROSS LEVELS. Partial pooling needs "the parent's value in
the same price bin" to be well defined, so one set of quantile edges is fit on
the whole sample and used at every level (5c gave each category its own edges).

SELECTION-BIAS NOTE. Fit over the deep slow tape, the baseline inherits that
tape's selection: the deepened wallets were screened for positive residuals, so
the fitted E[outcome|price] sits ABOVE the true population curve and residuals
are biased DOWN. That direction is conservative for a "survivors persist"
verdict and anti-conservative for a "survivors collapse" one — which is why the
step-6c niche diagnostic reads its niche averages off the unbiased discovery
corpus wherever coverage allows.

READ-ONLY / ANALYSIS-ONLY. No writes, no network, no ledger contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Coarse -> fine. The implicit root above `category` is the global per-bin curve.
DEFAULT_LEVELS = ("category", "niche_l1", "niche_l2")

DEFAULT_N_BINS = 20
DEFAULT_MIN_BIN = 50        # effective bets a cell needs before it earns its own value
_K_FLOOR = 1.0              # never let an estimated k imply "no pooling at all"
_K_CEIL = 100_000.0
_MIN_CELLS_FOR_K = 20       # too few sibling cells to estimate tau^2 -> fall back
_FALLBACK_K = 100.0


@dataclass
class HierBaselineInfo:
    """Diagnostics for a fitted hierarchical baseline (never a finding —
    this is the object a reviewer inspects to see the bins were not fit on noise)."""
    edges: np.ndarray
    levels: tuple[str, ...]
    ks: dict[str, float]
    n_bets: int
    lowo: bool
    min_bin: int
    per_level: list[dict] = field(default_factory=list)

    def pooling_report(self) -> pd.DataFrame:
        """Per level: cells, k, how many cells cleared the min-bin floor, and the
        realized pooling weight n/(n+k) — i.e. how much of each fine estimate is
        genuinely its own data rather than its parent's."""
        return pd.DataFrame(self.per_level)


def fit_bin_edges(prices, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """Shared quantile price-bin edges, fit once over the whole sample and used
    at EVERY level of the hierarchy (see module docstring)."""
    p = np.asarray(prices, dtype=float)
    p = p[~np.isnan(p)]
    if p.size < n_bins or np.unique(p).size < 2:
        return np.array([])
    return np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))


def assign_bins(prices, edges: np.ndarray) -> np.ndarray:
    """Bin index per price; prices outside the fitted range clamp to the end bins
    (same convention as `features.expected_outcome`). Empty edges -> all zeros."""
    p = np.asarray(prices, dtype=float)
    if edges.size < 2:
        return np.zeros(p.shape, dtype=np.int64)
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)


def _cell_sums(codes: np.ndarray, y: np.ndarray, valid: np.ndarray,
               minlength: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(sum of outcomes, count) per integer cell code, over valid bets only."""
    ml = minlength if minlength is not None else int(codes.max()) + 1
    s = np.bincount(codes[valid], weights=y[valid], minlength=ml)
    c = np.bincount(codes[valid], minlength=ml).astype(float)
    return s, c


def _pair_codes(cell: np.ndarray, w_codes: np.ndarray, n_w: int) -> np.ndarray:
    """Dense codes for (cell, wallet). Built from integer arithmetic and then
    compacted — string keys over a 1.4M-row tape are a needless memory spike."""
    return pd.factorize(cell.astype(np.int64) * n_w + w_codes)[0]


def _estimate_k(sum_y: np.ndarray, n: np.ndarray, parent_est: np.ndarray) -> float:
    """Method-of-moments empirical-Bayes shrinkage constant for one level.

    Cells deviate from their parent by a true amount (variance tau^2) plus
    sampling noise (variance sigma^2/n). The observed spread of deviations
    therefore over-states tau^2 by the average sampling term:

        tau^2 = mean(d^2) - mean(sigma^2 / n),    k = sigma^2_bar / tau^2

    A large k means siblings genuinely look like their parent, so pool hard; a
    small k means the niches really do differ, so trust their own data. Estimated
    rather than chosen, so the granularity knob is not tuned by hand."""
    use = n >= 2
    if use.sum() < _MIN_CELLS_FOR_K:
        return _FALLBACK_K
    ybar = sum_y[use] / n[use]
    d = ybar - parent_est[use]
    sig2 = np.clip(ybar * (1.0 - ybar), 1e-6, 0.25)      # Bernoulli within-cell
    tau2 = float(np.mean(d ** 2) - np.mean(sig2 / n[use]))
    sig2_bar = float(np.mean(sig2))
    if not np.isfinite(tau2) or tau2 <= 1e-9:
        return _K_CEIL                                    # no real between-niche spread
    return float(np.clip(sig2_bar / tau2, _K_FLOOR, _K_CEIL))


def fit_hierarchical_residuals(
    bets: pd.DataFrame,
    levels: tuple[str, ...] = DEFAULT_LEVELS,
    n_bins: int = DEFAULT_N_BINS,
    min_bin: int = DEFAULT_MIN_BIN,
    lowo: bool = True,
    ks: dict[str, float] | None = None,
    wallet_col: str = "wallet",
    price_col: str = "entry_price",
    outcome_col: str = "resolved_value",
) -> tuple[pd.DataFrame, HierBaselineInfo]:
    """Fit the hierarchy and residualize the SAME frame in one pass.

    Fitting and application are deliberately fused: the leave-one-wallet-out
    correction is defined only against the sample the cell means were computed
    from, so there is no honest way to hand back a reusable "baseline object" and
    still exclude each wallet exactly. Returns (frame + `expected_outcome` +
    `residual_skill`, diagnostics).

    Set `lowo=False` for the population view used by the niche diagnostic ("what
    does the average participant in this niche earn?"), where no single wallet
    should be excluded.
    """
    out = bets.copy()
    n = len(out)
    if n == 0:
        out["expected_outcome"] = pd.Series(dtype=float)
        out["residual_skill"] = pd.Series(dtype=float)
        return out, HierBaselineInfo(np.array([]), tuple(levels), {}, 0, lowo, min_bin)

    y = out[outcome_col].to_numpy(dtype=float)
    prices = out[price_col].to_numpy(dtype=float)
    edges = fit_bin_edges(prices, n_bins)
    bin_idx = assign_bins(prices, edges)
    n_bin = int(edges.size - 1) if edges.size >= 2 else 1

    w_codes = (pd.factorize(out[wallet_col].to_numpy())[0]
               if (lowo and wallet_col in out.columns) else np.zeros(n, dtype=np.int64))
    n_w = int(w_codes.max()) + 1 if n else 1

    # A bet contributes to a cell mean only if BOTH its outcome and its price are
    # known; a NaN price has no bin, so it gets a NaN expectation rather than a
    # clamped guess (same convention as forecaster_metrics.expected_outcome_by_category).
    valid = ~np.isnan(y) & ~np.isnan(prices)

    # --- root: the global per-bin curve, LOWO'd -----------------------------
    S_bin = np.bincount(bin_idx[valid], weights=y[valid], minlength=n_bin)
    N_bin = np.bincount(bin_idx[valid], minlength=n_bin).astype(float)
    grand = float(np.mean(y[valid])) if valid.any() else 0.0

    if lowo:
        pair = _pair_codes(bin_idx, w_codes, n_w)
        S_pair, N_pair = _cell_sums(pair, y, valid)
        num = S_bin[bin_idx] - S_pair[pair]
        den = N_bin[bin_idx] - N_pair[pair]
    else:
        num = S_bin[bin_idx]
        den = N_bin[bin_idx].copy()
    est = np.where(den > 0, num / np.where(den > 0, den, 1.0), grand)

    info = HierBaselineInfo(edges=edges, levels=tuple(levels), ks={}, n_bets=n,
                            lowo=lowo, min_bin=min_bin)
    info.per_level.append({
        "level": "_global", "cells": int((N_bin > 0).sum()), "k": float("nan"),
        "cells_own": int((N_bin >= min_bin).sum()),
        "bets_own_share": float(N_bin[N_bin >= min_bin].sum() / max(N_bin.sum(), 1)),
        "mean_pool_weight": 1.0,
    })

    # --- each level, coarse -> fine, shrunk toward the level above ----------
    for lvl in levels:
        if lvl not in out.columns:
            continue
        lvl_codes = pd.factorize(out[lvl].to_numpy())[0]
        cell = lvl_codes.astype(np.int64) * n_bin + bin_idx
        n_cells = int(cell.max()) + 1
        S_cell, N_cell = _cell_sums(cell, y, valid, minlength=n_cells)

        if ks and lvl in ks:
            k = float(ks[lvl])
        else:
            # k is estimated against the parent estimate each cell actually sees.
            # All bets in a cell share a level value AND a price bin, so they share
            # a parent cell; taking the first bet's parent estimate is exact up to
            # the (negligible, second-order) LOWO difference between wallets.
            first = np.zeros(n_cells, dtype=np.int64)
            first[cell[::-1]] = np.arange(n - 1, -1, -1)
            k = _estimate_k(S_cell, N_cell, est[first])
        info.ks[lvl] = k

        if lowo:
            pair = _pair_codes(cell, w_codes, n_w)
            S_pair, N_pair = _cell_sums(pair, y, valid)
            s_i = S_cell[cell] - S_pair[pair]
            n_i = N_cell[cell] - N_pair[pair]
        else:
            s_i = S_cell[cell]
            n_i = N_cell[cell].copy()

        shrunk = (s_i + k * est) / (n_i + k)
        # hard floor: a cell that has not cleared min_bin (after exclusion) takes
        # its parent's value exactly, and is reported as pooled.
        own = n_i >= min_bin
        est = np.where(own, shrunk, est)

        info.per_level.append({
            "level": lvl,
            "cells": n_cells,
            "k": k,
            "cells_own": int(np.unique(cell[own]).size),
            "bets_own_share": float(own.mean()),
            "mean_pool_weight": float(np.mean((n_i / (n_i + k))[own])) if own.any() else 0.0,
        })

    # A bet with no usable price gets no expectation (and so a NaN residual)
    # rather than a clamped guess it never earned.
    est = np.where(np.isnan(prices), np.nan, est)
    out["expected_outcome"] = est
    out["residual_skill"] = y - est
    return out, info


# ---------------------------------------------------------------------------
# Persistable fit / apply split — for FORWARD scoring only
# ---------------------------------------------------------------------------
#
# `fit_hierarchical_residuals` fuses fit and apply because leave-one-wallet-out is
# only defined against the sample the cell means came from. Forward scoring has
# the opposite requirement: a curve fit ONLY on pre-freeze data, applied unchanged
# to bets that did not exist when it was fitted. LOWO is irrelevant there — a
# forward bet is not in the fitting sample, so there is nothing to exclude — which
# is exactly why the split is safe here and would not be safe in-sample.
#
# The fitted object is plain JSON (edges + per-level cell -> estimate), so the
# freeze manifest can carry it and a re-score months later needs no refit and no
# access to the original tape.

def fit_hierarchical_baseline(
    bets: pd.DataFrame,
    levels: tuple[str, ...] = DEFAULT_LEVELS,
    n_bins: int = DEFAULT_N_BINS,
    min_bin: int = DEFAULT_MIN_BIN,
    ks: dict[str, float] | None = None,
    price_col: str = "entry_price",
    outcome_col: str = "resolved_value",
) -> dict:
    """Fit the hierarchy WITHOUT leave-one-wallet-out and return a JSON-safe dict
    that `expected_outcome_hier` can apply to unseen bets.

    Cells that never cleared `min_bin` are simply absent, so application falls
    through to the parent — the same rule the in-sample fit uses."""
    n = len(bets)
    if n == 0:
        return {"edges": [], "levels": [], "cells": {}, "ks": {},
                "grand": 0.0, "min_bin": min_bin, "n_bets": 0}

    y = bets[outcome_col].to_numpy(dtype=float)
    prices = bets[price_col].to_numpy(dtype=float)
    edges = fit_bin_edges(prices, n_bins)
    bin_idx = assign_bins(prices, edges)
    n_bin = int(edges.size - 1) if edges.size >= 2 else 1
    valid = ~np.isnan(y) & ~np.isnan(prices)
    grand = float(np.mean(y[valid])) if valid.any() else 0.0

    S_bin = np.bincount(bin_idx[valid], weights=y[valid], minlength=n_bin)
    N_bin = np.bincount(bin_idx[valid], minlength=n_bin).astype(float)
    est = np.where(N_bin[bin_idx] > 0, S_bin[bin_idx] / np.where(N_bin[bin_idx] > 0, N_bin[bin_idx], 1.0), grand)

    out_cells: dict[str, dict[str, float]] = {}
    out_ks: dict[str, float] = {}
    root = {str(b): float(S_bin[b] / N_bin[b]) if N_bin[b] > 0 else grand
            for b in range(n_bin)}
    out_cells["_global"] = root

    used_levels = []
    for lvl in levels:
        if lvl not in bets.columns:
            continue
        used_levels.append(lvl)
        lvl_vals = bets[lvl].astype(str).to_numpy()
        lvl_codes, uniques = pd.factorize(lvl_vals)
        cell = lvl_codes.astype(np.int64) * n_bin + bin_idx
        n_cells = int(cell.max()) + 1
        S_cell, N_cell = _cell_sums(cell, y, valid, minlength=n_cells)

        if ks and lvl in ks:
            k = float(ks[lvl])
        else:
            first = np.zeros(n_cells, dtype=np.int64)
            first[cell[::-1]] = np.arange(n - 1, -1, -1)
            k = _estimate_k(S_cell, N_cell, est[first])
        out_ks[lvl] = k

        shrunk = (S_cell[cell] + k * est) / (N_cell[cell] + k)
        own = N_cell[cell] >= min_bin
        est = np.where(own, shrunk, est)

        table: dict[str, float] = {}
        keep = np.where(own)[0]
        for i in keep:
            table[f"{uniques[lvl_codes[i]]}@{bin_idx[i]}"] = float(est[i])
        out_cells[lvl] = table

    return {"edges": [float(e) for e in edges], "levels": used_levels,
            "cells": out_cells, "ks": out_ks, "grand": grand,
            "min_bin": min_bin, "n_bets": int(n)}


def expected_outcome_hier(baseline: dict, bets: pd.DataFrame,
                          price_col: str = "entry_price") -> np.ndarray:
    """Apply a fitted baseline to unseen bets: take the FINEST level that has an
    entry for the bet's (level-value, price-bin) cell, else fall through to the
    parent, ending at the global per-bin curve. NaN price -> NaN expectation."""
    prices = bets[price_col].to_numpy(dtype=float)
    edges = np.asarray(baseline.get("edges", []), dtype=float)
    bin_idx = assign_bins(prices, edges)
    grand = float(baseline.get("grand", 0.0))
    cells = baseline.get("cells", {})

    root = cells.get("_global", {})
    est = pd.Series(bin_idx).astype(str).map(root).to_numpy(dtype=float)
    est = np.where(np.isnan(est), grand, est)

    bin_str = pd.Series(bin_idx).astype(str)
    for lvl in baseline.get("levels", []):          # coarse -> fine; finest wins
        table = cells.get(lvl, {})
        if not table or lvl not in bets.columns:
            continue
        keys = bets[lvl].astype(str).reset_index(drop=True) + "@" + bin_str
        hit = keys.map(table).to_numpy(dtype=float)
        est = np.where(np.isnan(hit), est, hit)
    return np.where(np.isnan(prices), np.nan, est)
