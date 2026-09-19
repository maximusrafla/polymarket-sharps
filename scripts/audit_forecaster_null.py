"""Shuffled-outcome null + Benjamini-Hochberg FDR gate for the Project-2
forecaster table (`src/forecaster_metrics.py`). THE decisive step that decides
whether ANY (wallet, category) forecaster "winner" is real skill or a
multiple-testing mirage across the ~hundreds-of-thousands of scored cells.

WHY THIS EXISTS (the §1.4 crux — docs/project2_section14_findings.md). The
de-biased re-score found the apparent forecaster skill is NOT near-resolution
tape bias: on honest full tapes the *pooled* skill edge is ~0 at every entry
time. Yet the chronological OOS persistence gate still fires on ~25% of testable
cells — 5x the 5% chance rate — with a median +12c edge among survivors. That is
the signature of SELECTION plus residual per-wallet baseline structure surviving
the split (the "favorite-longshot survives out-of-sample" failure mode
CLAUDE.md/HANDOFF flag), searched across ~4.5k testable cells. The analytic
one-sided t-test the gate uses tests H0: OOS residual mean = 0 — but that null is
anti-conservative here, because a wallet with a persistent finer-than-baseline
price preference has a *positive* structural residual with no skill. So we need a
null that preserves exactly that structure and destroys only skill.

THE NULL (lifted from scripts/audit_blackswan.py's posture + `bh_reject`).
Permute resolved outcomes WITHIN each (category, 1-cent price) bucket over ALL
resolved BUY bets. This preserves (a) the per-category favorite-longshot base
rate the residual is measured against and (b) every wallet's price mix, while
destroying only the wallet<->outcome link — i.e. genuine forecasting skill. Under
this null a cell's OOS residual is drawn from its own price/category pool, so any
"edge" it shows is exactly the baseline-structure-survives-selection artifact. We
re-score metric A on each shuffle and:

  1. COUNT-NULL (robust headline; the audit_blackswan approach). The null
     distribution of the number of cells passing the gate (OOS significant &
     magnitude). Real >> null 95th pct => the aggregate is above chance; the ratio
     null_mean/real estimates the gate's false-discovery rate. No cross-cell
     exchangeability assumption.

  2. POOLED-NULL EMPIRICAL p + BH-FDR (per-cell verdict). The per-cell OOS
     t-statistic is pivotal (comparable across cells of different n), so we pool
     all null t's across cells and shuffles into one empirical null, score each
     real cell's one-sided empirical p against it, and apply Benjamini-Hochberg at
     q. This corrects BOTH multiple testing AND the anti-conservative analytic
     null in one step. Survivors are the individually-defensible cells, if any.

DECISION. FDR survivors that also clear the economic/quality gates (magnitude >=
2c, breadth >= 3, full-tape, real-world category) are the provisional publishable
winner list -> metric B (copyability) then runs on just those. If nothing
survives, the §1.4 winners are selection noise and the dashboard/findings say so.

Read-only. Reads data/interim/discovery/ (tape + forecasters.parquet), writes
ONLY data/interim/discovery/forecaster_null_fdr.parquet (the per-cell survivor
flags rank_forecasters joins). Touches no ledger, no Project-1 code.

Usage:  PYTHONPATH=. python scripts/audit_forecaster_null.py [--shuffles N]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.common import load_config
from src.discover import (
    DISCOVERY_DIR,
    load_discovery_trades,
    load_tape_stats,
    market_tape_complete,
)
from src.forecaster_metrics import (
    ALL_CELL,
    _SCORING_COLUMNS,
    _config_values,
    expected_outcome_by_category,
    fit_category_baselines,
    lean_tape,
)
from scripts.audit_blackswan import bh_reject  # reuse the BH machinery, don't rebuild

# Fixed knobs (no Date/random in the pipeline; mirror audit_blackswan's SEED style).
SEED = 12345
N_SHUFFLES = 200          # null replications: count-null p95 stable, pooled null ~1e-6 resolution
CENT_BIN = 0.01           # price-bucket resolution for the within-bucket shuffle (== audit_blackswan)
FDR_QS = (0.05, 0.10)     # BH levels to report
# Concentration / independence guard on the WINNER definition (NOT the statistical
# null — the shuffle can't see it). The exchangeable within-bucket shuffle treats
# every fill as independent, so it cannot detect a cell whose "edge" is really a
# handful of CORRELATED events replicated across many fills (e.g. 27 fills on 3
# props of one Fed press conference, or 60 fills on one Sunday's 5 NFL games — both
# survived FDR with an absurd ~50c edge). A copyable forecaster's track record must
# instead come from several INDEPENDENT positions over several distinct days:
#   - eff_breadth = 1/HHI of per-market fill shares (effective # of independent
#     markets; robust to fill-count inflation — buying one winner 27x -> ~1, not 27);
#   - entry_days = distinct calendar days with >=1 bet (temporal independence — one
#     correlated slate/event is ~1 macro-decision, not a track record).
# Additive minimum-track-record floors (principled, not fitted weights), like the
# existing breadth>=3 floor, of which eff_breadth is the concentration-robust form.
MIN_EFF_BREADTH = 3.0     # >= this many effective independent markets contributing the edge
MIN_ENTRY_DAYS = 3        # >= this many distinct decision-days (kills single-slate/-event cells)
# Economic / quality gates applied ON TOP of FDR significance to define a winner
# candidate (metric-A side). Pulled from the same config the scorer uses.
NULL_FDR_PATH = DISCOVERY_DIR / "forecaster_null_fdr.parquet"
NULL_SUMMARY_PATH = DISCOVERY_DIR / "forecaster_null_summary.json"

_SEP = "\x1f"  # cell-key join char (never appears in a wallet/category string)


# --------------------------------------------------------------------------- #
# Prep: resolved BUY bets, fixed per-category baseline, per-bet residual        #
# --------------------------------------------------------------------------- #
def prepare(cfg_vals: dict):
    """Rebuild EXACTLY the metric-A per-bet residual the production scorer
    (`compute_forecaster_table`) computes: per-category favorite-longshot baseline
    over all resolved BUY bets, residual = resolved_value - E_cat[outcome|price].
    Returns the full resolved-BUY frame with `residual`, `expected`, `cent`, and a
    `category` column. The baseline is FIXED (fit on the real outcomes) and reused
    for the null residuals so real and null are compared against one identical
    model — the shuffle only moves outcomes, exactly the effect we want to test."""
    tape = lean_tape(load_discovery_trades(columns=_SCORING_COLUMNS))
    stats_df = load_tape_stats()
    bets = tape.loc[(tape["side"] == "BUY") & tape["resolved"].fillna(False)].copy()
    bets = bets[bets["resolved_value"].notna()].reset_index(drop=True)

    comp = market_tape_complete(tape, stats_df if stats_df is not None else pd.DataFrame())
    cmap = dict(zip(comp["market_id"], comp["tape_complete"]))
    bets["tape_complete"] = bets["market_id"].map(cmap).fillna(True).astype(bool)
    del tape

    baselines = fit_category_baselines(bets, cfg_vals["n_bins"], cfg_vals["min_bets_for_category_baseline"])
    expected = expected_outcome_by_category(baselines, bets["category"], bets["entry_price"])
    bets["expected"] = expected
    bets["rv"] = bets["resolved_value"].to_numpy(dtype=float)
    bets["residual"] = bets["rv"].to_numpy(dtype=float) - expected
    bets["cent"] = np.floor(bets["entry_price"].to_numpy(dtype=float) / CENT_BIN).astype(int)
    return bets


# --------------------------------------------------------------------------- #
# Cell structure (fixed: timestamps/prices/categories don't move under shuffle) #
# --------------------------------------------------------------------------- #
def build_cells(bets: pd.DataFrame, cfg_vals: dict):
    """Expand resolved BUY bets from score-eligible wallets into (wallet, category)
    cells PLUS each multi-category wallet's `all_real_world` aggregate — the exact
    cell set the production scorer builds — and precompute the FIXED chronological
    in/out split membership per cell (only outcomes shuffle, so the split, breadth,
    full-tape fraction and cell membership are all constant across shuffles).

    Returns a dict of numpy arrays the per-shuffle scorer needs (integer cell codes
    + base-frame positions for the in/out halves), plus a per-cell attribute frame."""
    min_score = cfg_vals["min_score_bets"]
    oos = cfg_vals["oos_split"]

    bets = bets.reset_index(drop=True)
    bets["base_pos"] = np.arange(len(bets))
    wc = bets.groupby("wallet").size()
    score_wallets = wc[wc >= min_score].index
    sb = bets[bets["wallet"].isin(score_wallets)].copy()

    cols = ["wallet", "category", "base_pos", "timestamp", "market_id", "tape_complete"]
    sub = sb[cols].copy()
    sub["cell_cat"] = sub["category"]
    ncat = sb.groupby("wallet")["category"].transform("nunique")
    allc = sb[ncat >= 2][cols].copy()
    allc["cell_cat"] = ALL_CELL
    exp = pd.concat([sub, allc], ignore_index=True)
    exp["cell"] = exp["wallet"].astype(str) + _SEP + exp["cell_cat"].astype(str)

    # chronological order within cell -> half (floor(n*oos) in-sample), like validate.py
    exp = exp.sort_values(["cell", "timestamp"], kind="stable").reset_index(drop=True)
    exp["day"] = (exp["timestamp"].to_numpy(dtype="int64") // 86400)
    g = exp.groupby("cell", sort=False)
    pos = g.cumcount().to_numpy()
    n = g["cell"].transform("size").to_numpy()
    split_idx = np.floor(n * oos).astype(int)
    half = (pos >= split_idx).astype(int)  # 0=in-sample, 1=out-of-sample

    # integer cell codes
    cell_codes, cell_index = pd.factorize(exp["cell"], sort=False)
    C = len(cell_index)

    in_mask = half == 0
    out_mask = half == 1
    base_pos = exp["base_pos"].to_numpy()

    in_code = cell_codes[in_mask]
    in_pos = base_pos[in_mask]
    out_code = cell_codes[out_mask]
    out_pos = base_pos[out_mask]
    in_n = np.bincount(in_code, minlength=C).astype(float)
    out_n = np.bincount(out_code, minlength=C).astype(float)

    # per-cell fixed attributes
    attr = exp.groupby("cell", sort=False).agg(
        breadth=("market_id", "nunique"),
        frac_full_tape=("tape_complete", "mean"),
        sample_size=("cell", "size"),
        entry_days=("day", "nunique"),
    )
    # eff_breadth = 1/HHI of per-market fill shares (concentration-robust breadth).
    mkt_counts = exp.groupby(["cell", "market_id"], sort=False).size()
    tot = mkt_counts.groupby("cell").sum()
    hhi = (mkt_counts.pow(2).groupby("cell").sum()) / tot.pow(2)
    attr["eff_breadth"] = 1.0 / hhi
    attr["top_market_share"] = (mkt_counts.groupby("cell").max()) / tot
    attr = attr.reindex(cell_index)  # align to code order
    wallet_cat = pd.Series(cell_index).str.split(_SEP, expand=True)
    attr = attr.reset_index(drop=True)
    attr["wallet"] = wallet_cat[0].to_numpy()
    attr["category"] = wallet_cat[1].to_numpy()

    return {
        "C": C,
        "in_code": in_code, "in_pos": in_pos, "in_n": in_n,
        "out_code": out_code, "out_pos": out_pos, "out_n": out_n,
        "attr": attr,
    }


# --------------------------------------------------------------------------- #
# One-pass vectorized metric-A gate on a residual vector                        #
# --------------------------------------------------------------------------- #
def score_cells(resid_full: np.ndarray, cells: dict, alpha: float, min_half: int, min_edge: float):
    """Vectorized metric-A per-cell statistics from a base-frame residual vector.
    Returns (in_mean, out_mean, out_t, out_p, sig, persisted) as length-C arrays.
    Byte-for-byte the same gate as forecaster_metrics.cell_metrics / validate.py
    (_gated_mean + one-sided ttest_1samp greater + magnitude floor), just computed
    for all cells at once via bincount."""
    C = cells["C"]
    in_n, out_n = cells["in_n"], cells["out_n"]

    in_sum = np.bincount(cells["in_code"], weights=resid_full[cells["in_pos"]], minlength=C)
    out_code = cells["out_code"]
    out_r = resid_full[cells["out_pos"]]
    out_sum = np.bincount(out_code, weights=out_r, minlength=C)

    with np.errstate(invalid="ignore", divide="ignore"):
        in_mean = in_sum / in_n
        out_mean = out_sum / out_n
        # Unbiased (ddof=1) variance via CENTERED sum-of-squares — numerically stable
        # (the sumsq/n - mean^2 shortcut catastrophically cancels for residuals near
        # +-1 with small spread, flipping borderline cells vs scipy). Matches
        # pandas .std(ddof=1) / scipy ttest_1samp to float precision.
        cent = out_r - out_mean[out_code]
        out_ss = np.bincount(out_code, weights=cent * cent, minlength=C)
        out_var_u = out_ss / (out_n - 1.0)
        out_std = np.sqrt(np.clip(out_var_u, 0.0, None))
        out_t = out_mean / (out_std / np.sqrt(out_n))

    # The t-test is UNDEFINED where there are too few points or ~zero variance
    # (degenerate replicated bets — a wallet's out-half all resolved the same way).
    # validate._oos_significance returns NaN there ("not significant", conservative).
    # We must NaN-out BOTH the p AND the t: a ~zero-variance cell has a blown-up but
    # FINITE t (mean / ~0 -> ~1e16) that would otherwise sail through the empirical-p
    # FDR path AND pollute the pooled null. NaN-ing t drops it from both consistently.
    bad = (out_n < max(min_half, 2)) | ~(out_std > 1e-12)
    out_t = np.where(bad, np.nan, out_t)
    out_p = np.full(C, np.nan)
    good = ~bad
    out_p[good] = stats.t.sf(out_t[good], out_n[good] - 1.0)

    in_ok = in_n >= min_half
    out_ok = out_n >= min_half
    cand = in_ok & (in_mean > 0)
    sig = cand & out_ok & (out_mean > 0) & (out_p < alpha)
    persisted = sig & (out_mean >= min_edge)
    return in_mean, out_mean, out_t, out_p, np.nan_to_num(sig, nan=False).astype(bool), \
        np.nan_to_num(persisted, nan=False).astype(bool)


# --------------------------------------------------------------------------- #
# Within-(category,cent)-bucket outcome shuffle (vectorized)                     #
# --------------------------------------------------------------------------- #
def make_shuffler(bets: pd.DataFrame):
    """Return (group_id, base_order, rv, expected) for a fast vectorized
    within-(category, cent) permutation of resolved outcomes. Preserves each
    bucket's base rate and each wallet's price mix; destroys only wallet<->outcome.
    A shuffle is one lexsort: assign each row a random key, order rows by
    (bucket, key), and scatter outcomes back into bucket-grouped slots."""
    gkey = bets["category"].astype(str) + _SEP + bets["cent"].astype(str)
    group_id = pd.factorize(gkey, sort=False)[0]
    base_order = np.argsort(group_id, kind="stable")  # positions grouped by bucket (fixed)
    rv = bets["rv"].to_numpy(dtype=float)
    expected = bets["expected"].to_numpy(dtype=float)
    return group_id, base_order, rv, expected


def shuffled_residual(group_id, base_order, rv, expected, rng) -> np.ndarray:
    """One within-bucket permutation of `rv`, then residual = shuffled_rv - expected
    against the FIXED baseline."""
    rand = rng.random(rv.size)
    perm = np.lexsort((rand, group_id))          # rows ordered by (bucket, random)
    shuffled = np.empty(rv.size, dtype=float)
    shuffled[base_order] = rv[perm]               # scatter into bucket-grouped slots
    return shuffled - expected


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Shuffled-outcome null + BH-FDR gate for Project-2 forecasters.")
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLES, help="null replications (default 200)")
    ap.add_argument("--min-breadth", type=int, default=3, help="winner breadth floor (metric-A side)")
    args = ap.parse_args()

    cfg = load_config()
    cv = _config_values(cfg)
    alpha, min_half, min_edge = cv["alpha"], cv["min_per_half"], cv["min_skill_edge"]
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("PROJECT-2 FORECASTER NULL + FDR  (read-only; shuffle-within-(cat,cent))")
    print("=" * 78)
    bets = prepare(cv)
    print(f"resolved BUY bets: {len(bets):,}   wallets: {bets['wallet'].nunique():,}")
    cells = build_cells(bets, cv)
    attr = cells["attr"]
    C = cells["C"]
    testable = (cells["in_n"] >= min_half) & (cells["out_n"] >= min_half)
    print(f"(wallet,category) cells scored: {C:,}   testable (both halves >= {min_half}): {int(testable.sum()):,}")

    # -- Real gate --------------------------------------------------------------
    resid_real = bets["residual"].to_numpy(dtype=float)
    in_m0, out_m0, t0, p0, sig0, pers0 = score_cells(resid_real, cells, alpha, min_half, min_edge)
    # Winner candidate (metric-A side): persisted + breadth + full-tape + real-world.
    real_world = attr["category"].to_numpy() != "micro_crypto"
    full_tape = attr["frac_full_tape"].to_numpy() >= cv["min_frac_full_tape"]
    breadth_ok = attr["breadth"].to_numpy() >= args.min_breadth
    winnerA = pers0 & breadth_ok & full_tape & real_world
    # Concentration / independence guard (folded into the FINAL winner definition,
    # not the statistical gate): the edge must come from >= MIN_EFF_BREADTH effective
    # independent markets over >= MIN_ENTRY_DAYS distinct days. Kills fill-inflated
    # single-slate / single-event cells the exchangeable shuffle can't detect.
    conc_ok = (attr["eff_breadth"].to_numpy() >= MIN_EFF_BREADTH) & \
              (attr["entry_days"].to_numpy() >= MIN_ENTRY_DAYS)

    n_test = int(testable.sum())
    print("\n--- Real gate (metric A) ---")
    print(f"  significant (OOS p<{alpha}, positive):        {int(sig0.sum()):,}   "
          f"({int(sig0.sum())/n_test:.1%} of testable)")
    print(f"  persisted   (+ magnitude >= {min_edge:.2f}):        {int(pers0.sum()):,}   "
          f"({int(pers0.sum())/n_test:.1%} of testable)")
    print(f"  winnerA (persisted & breadth>={args.min_breadth} & full-tape & real-world): {int(winnerA.sum()):,}")

    # -- Shuffle null: count-null + pooled null t's -----------------------------
    group_id, base_order, rv, expected = make_shuffler(bets)
    K = args.shuffles
    null_sig = np.empty(K, dtype=int)
    null_pers = np.empty(K, dtype=int)
    null_winnerA = np.empty(K, dtype=int)
    pooled_null_t = []  # testable-cell null t's, pooled across shuffles for the empirical p
    print(f"\n--- Shuffling {K}x (within (category, cent) buckets) ... ---", flush=True)
    for s in range(K):
        resid = shuffled_residual(group_id, base_order, rv, expected, rng)
        _, out_m, t_s, _, sig_s, pers_s = score_cells(resid, cells, alpha, min_half, min_edge)
        null_sig[s] = int(sig_s.sum())
        null_pers[s] = int(pers_s.sum())
        null_winnerA[s] = int((pers_s & breadth_ok & full_tape & real_world).sum())
        tt = t_s[testable]
        pooled_null_t.append(tt[np.isfinite(tt)])
        if (s + 1) % 50 == 0:
            print(f"  {s+1}/{K}  (null persisted this shuffle: {null_pers[s]})", flush=True)
    pooled_null_t = np.concatenate(pooled_null_t)
    pooled_null_t.sort()

    summary = {"n_shuffles": K, "n_testable": n_test, "min_half": min_half,
               "alpha": alpha, "min_skill_edge": min_edge, "min_breadth": args.min_breadth,
               "count_null": {}}

    def null_line(name, key, real, null):
        exceed = float((null >= real).mean())
        fdr = null.mean() / real if real > 0 else float("nan")
        summary["count_null"][key] = {
            "real": int(real), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "null_max": int(null.max()),
            "p_null_ge_real": exceed, "est_fdr": float(fdr),
        }
        print(f"  {name:<26} real {real:>6,} | null mean {null.mean():>8.1f}  "
              f"p95 {np.percentile(null,95):>6.0f}  max {null.max():>5}  | "
              f"P(null>=real) {exceed:5.3f}  est.FDR {fdr:5.2f}")

    print(f"\n--- Count-null verdict (the aggregate arbiter) ---")
    null_line("significant", "significant", int(sig0.sum()), null_sig)
    null_line("persisted (A+C)", "persisted", int(pers0.sum()), null_pers)
    null_line("winnerA (A+C+breadth+FT+RW)", "winnerA", int(winnerA.sum()), null_winnerA)

    # -- Pooled-null empirical p + BH-FDR (per-cell verdict) --------------------
    # Empirical one-sided p for each testable real cell: fraction of the pooled null
    # t-distribution at or above the cell's real OOS t (add-one smoothing).
    t_real = t0[testable]
    finite = np.isfinite(t_real)
    Np = pooled_null_t.size
    emp_p = np.ones(t_real.size)
    # searchsorted: # null t >= real_t = Np - left_insertion_index
    ge = Np - np.searchsorted(pooled_null_t, t_real[finite], side="left")
    emp_p[finite] = (1.0 + ge) / (1.0 + Np)

    test_idx = np.where(testable)[0]
    emp_full = np.full(C, np.nan)
    emp_full[test_idx] = emp_p

    print(f"\n--- BH-FDR over {int(finite.sum()):,} testable cells "
          f"(empirical p vs pooled null of {Np:,} t's; min emp_p {1.0/(1.0+Np):.2e}) ---")
    summary["fdr"] = {"m_tested": int(finite.sum()), "pooled_null_size": int(Np),
                      "min_emp_p": float(1.0 / (1.0 + Np)), "by_q": {}}
    surv_cols = {}
    for q in FDR_QS:
        rej = bh_reject(emp_p[finite], q)
        surv = np.zeros(C, dtype=bool)
        surv[test_idx[finite][rej]] = True
        surv_cols[q] = surv
        winners = surv & breadth_ok & full_tape & real_world & (out_m0 >= min_edge) & conc_ok
        exp_fp = q * int(rej.sum())  # BH controls E[FP] <= q * (#rejections)
        summary["fdr"]["by_q"][str(q)] = {"survive": int(rej.sum()),
                                          "expected_false": float(exp_fp),
                                          "null_winners": int(winners.sum())}
        print(f"  q={q:<4}: {int(rej.sum()):>4} cells survive FDR (empirical, structural null); "
              f"expected false among them <= {exp_fp:.1f}")
        print(f"           of those, WINNERS (& magnitude & breadth>={args.min_breadth} & full-tape "
              f"& real-world & eff_breadth>={MIN_EFF_BREADTH:g} & entry_days>={MIN_ENTRY_DAYS}): "
              f"{int(winners.sum())}")

    # For contrast: BH on the ANALYTIC p (H0: mean=0) — the anti-conservative one.
    rej_an = bh_reject(np.nan_to_num(p0[testable], nan=1.0), 0.05)
    print(f"\n  [contrast] BH q=0.05 on the ANALYTIC t-test p (H0:mean=0): "
          f"{int(rej_an.sum()):,} 'survive' — this is the inflated count the structural "
          f"null corrects.")

    # -- Persist per-cell survivor flags (rank_forecasters joins these) ---------
    out = attr.copy()
    out["testable"] = testable
    out["real_oos_skill_edge"] = out_m0
    out["real_oos_t"] = t0
    out["real_oos_p_analytic"] = p0
    out["empirical_p_structural_null"] = emp_full
    out["edge_significant"] = sig0
    out["edge_persisted"] = pers0
    out["concentration_ok"] = conc_ok
    for q in FDR_QS:
        tag = f"survives_fdr_{int(q*100):02d}"
        out[tag] = surv_cols[q]
        winners = surv_cols[q] & breadth_ok & full_tape & real_world & (out_m0 >= min_edge) & conc_ok
        out[f"null_winner_fdr_{int(q*100):02d}"] = winners
    out.attrs = {}
    from src.common import atomic_to_parquet
    NULL_FDR_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(out, NULL_FDR_PATH, compression="gzip")

    # Analytic-BH contrast + verdict scalars into the summary sidecar the dashboard cites.
    summary["analytic_bh_q05_survive"] = int(rej_an.sum())
    summary["null_winner_fdr_05"] = int(out["null_winner_fdr_05"].sum())  # AFTER concentration guard
    summary["fdr05_winner_pre_concentration"] = int(
        (surv_cols[0.05] & breadth_ok & full_tape & real_world & (out_m0 >= min_edge)).sum())
    summary["real_significant"] = int(sig0.sum())
    summary["real_persisted"] = int(pers0.sum())
    summary["real_winnerA"] = int(winnerA.sum())
    summary["concentration_guard"] = {"min_eff_breadth": MIN_EFF_BREADTH, "min_entry_days": MIN_ENTRY_DAYS}
    NULL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    n_surv = int(surv_cols[0.05].sum())
    n_fdr_pre = int((surv_cols[0.05] & breadth_ok & full_tape & real_world & (out_m0 >= min_edge)).sum())
    n_win = int(out["null_winner_fdr_05"].sum())
    print("\n" + "=" * 78)
    if n_win == 0:
        print(f"VERDICT: NO credible copyable forecaster survives at q=0.05.")
        print(f"  {int(sig0.sum())} testable cells look significant, but the structural shuffle null")
        print(f"  reproduces ~{summary['count_null']['persisted']['null_mean']:.0f} 'persisted' cells by chance "
              f"(est. gate FDR {summary['count_null']['winnerA']['est_fdr']:.0%}).")
        print(f"  {n_surv} cells survive BH-FDR; {n_fdr_pre} of those clear magnitude/breadth/full-tape/")
        print(f"  real-world, but ALL are single-slate / single-event fill-inflated cells the")
        print(f"  exchangeable shuffle can't see -> rejected by the concentration guard")
        print(f"  (eff_breadth>={MIN_EFF_BREADTH:g} & entry_days>={MIN_ENTRY_DAYS}). The §1.4 winners are SELECTION")
        print(f"  NOISE + concentrated-event artifacts. Withhold the list; do NOT run metric B.")
    else:
        print(f"VERDICT: {n_win} cell(s) survive FDR (q=0.05) AND clear magnitude/breadth/full-tape/")
        print(f"  real-world AND the concentration guard. PROVISIONAL publishable winners; run")
        print(f"  metric B (copyability) on just this set. Still gated by the forward paper-test.")
    print(f"per-cell flags -> {NULL_FDR_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
