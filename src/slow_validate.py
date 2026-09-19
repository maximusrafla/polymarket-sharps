"""Validate the deepened slow-market shortlist — Project 3 build step 5c
(docs/project3_slow_markets.md §4 stage 4, §7.6).

The honest test: the screen (5a) selected on SHALLOW data; deepening (5b) fetched
each shortlisted wallet's full history — data the screen never saw. This step runs
the A/C stack on that fresh deep data and asks whether a slow-market skill edge
survives, under the standards the rest of the repo now enforces:

  - chronological OOS split (validate.split_in_sample_out_of_sample — the stable
    mergesort split), in-sample residual skill > 0 to be a candidate;
  - held-out significance by the **market-block bootstrap** (validate._cluster_bootstrap_p),
    NOT a bet-level shuffled null — bets sharing a market share one resolution event,
    and the cluster-preserving standard is what the repo adopted repo-wide
    (cluster-preserving-null-required; §6, §10.1). It also reports the per-wallet
    design effect so a D<1 wallet is read honestly;
  - the SCOPED ~10c economic-magnitude floor scoring.slow.min_skill_edge (§5.2) —
    NEVER the global 2c;
  - concentration guards: eff_breadth = 1/HHI of held-out market shares >= 3, distinct
    entry days >= 3, and a resolution-time clustering guard (distinct held-out
    resolution-days >= 3, §3.3) so a wallet whose held-out markets all settle on one
    event is rejected — the §1.5 false-positive shape;
  - BH-FDR across candidates, and the honest "insufficient evidence" bucket for
    wallets too thin to certify either way (§5.4).

Residualization uses the per-category slow baseline fit INSIDE the deep slow universe
(step 4) — the right E[outcome|price] per category, not the crypto-dominated global.

READ-ONLY / provisional: this is NOT the final arbiter. Per HANDOFF the forward paper
test (step 6, out of scope here) is. Output goes to the isolated interim dir, never
data/processed/. Nothing is dropped — every candidate keeps its row with its gates.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import INTERIM_DIR, atomic_to_parquet, load_config
from src.market_meta import classify_speed_bucket, load_market_meta, speed_thresholds
from src.slow_deepen import load_deep_trades
from src.slow_market import (
    SLOW_BUCKETS,
    derive_categories,
    fit_slow_category_baselines,
    residualize,
)
from src.validate import _cluster_bootstrap_p, _wallet_seed, split_in_sample_out_of_sample

SLOW_VALIDATED_PATH = INTERIM_DIR / "slow_deepening" / "slow_validated.parquet"

# Scoped defaults (config-tunable via scoring.slow.*). None load-bearing on Project 1.
DEFAULT_MIN_SKILL_EDGE = 0.10        # ~10c floor (§5.2)
DEFAULT_MIN_OOS_MARKETS = 5          # held-out distinct markets (slow wallets are thinner than P1)
DEFAULT_MIN_EFF_BREADTH = 3.0        # concentration-robust breadth (1/HHI)
DEFAULT_MIN_ENTRY_DAYS = 3           # temporal independence of entries
DEFAULT_MIN_RESOLUTION_DAYS = 3      # resolution-time clustering guard (§3.3)
DEFAULT_MIN_BETS_PER_HALF = 10

# --- complex-unit gates (step 7) -------------------------------------------
# Step 6 showed the market is the WRONG unit for the slow path: a rolling-deadline
# ladder ("US strikes Iran by Feb 28 / Mar 1 / Mar 15 / Mar 31 ...") is dozens of
# distinct markets with distinct resolution days but ONE event. The market-unit
# gates below are all still applied — nothing is removed — but the same thresholds
# are ALSO enforced over event complexes, which is where they were always meant to
# bite. The numbers are deliberately UNCHANGED from their market-unit values: this
# corrects the unit, it does not re-tune the threshold. Re-tuning to reach a
# fuller table is the "relax until it fills" failure the repo guards against.
DEFAULT_MIN_OOS_COMPLEXES = 3        # cluster-count floor on the REAL cluster
DEFAULT_MIN_EFF_BREADTH_COMPLEX = 3.0
DEFAULT_MIN_ENTRY_DAYS_COMPLEX = 3       # distinct (complex, entry-day) pairs
DEFAULT_MIN_RESOLUTION_DAYS_COMPLEX = 3  # distinct (complex, resolution-day) pairs

# The frozen event-complex unit. `niche_l1` is the family level of the versioned
# slow_niche partition, i.e. the event complex / market generator.
COMPLEX_COL = "niche_l1"


# ---------------------------------------------------------------------------
# Deep slow-universe loader (speed classified from sidecar ∪ deep tape span)
# ---------------------------------------------------------------------------

def load_deep_slow_bets(buckets=SLOW_BUCKETS, cfg: dict | None = None,
                        deep: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolved BUY bets from the DEEP dataset whose market is slow, with `category`
    and a per-market resolution-time proxy.

    Speed is classified from the market's lifespan, preferring the market_meta
    sidecar's `lifespan_s` (built over the full main ledger) and falling back to the
    deep dataset's own observed tape span for markets the sidecar doesn't cover.
    `res_ts_proxy` = the market's last observed deep trade (a lower bound on
    resolution time) — used only for the resolution-day clustering guard."""
    cfg = cfg or load_config()
    slow_s, deep_s = speed_thresholds(cfg)
    if deep is None:
        deep = load_deep_trades()
    if deep.empty:
        return deep.assign(category=pd.Series(dtype=object)) if "category" not in deep else deep

    # Per-market lifespan lower bound from the deep tape (all rows), + resolution proxy.
    span = deep.groupby("market_id")["timestamp"].agg(["min", "max"])
    deep_life = (span["max"] - span["min"]).to_dict()
    res_ts = span["max"].to_dict()

    meta = load_market_meta(columns=["market_id", "lifespan_s"])
    sidecar_life = dict(zip(meta["market_id"], meta["lifespan_s"])) if not meta.empty else {}

    def _bucket(mid):
        L = sidecar_life.get(mid)
        if L is None or (isinstance(L, float) and np.isnan(L)):
            L = deep_life.get(mid, np.nan)
        return classify_speed_bucket(L, slow_s, deep_s)

    bets = deep[(deep["side"] == "BUY") & deep["resolved"].fillna(False)].copy()
    bets["speed_bucket"] = bets["market_id"].map(_bucket)
    bets = bets[bets["speed_bucket"].isin(buckets)].reset_index(drop=True)
    if bets.empty:
        return bets.assign(category=pd.Series(dtype=object))
    bets["category"] = derive_categories(bets["slug"], bets["question"])
    bets["res_ts_proxy"] = bets["market_id"].map(res_ts)
    return bets


# ---------------------------------------------------------------------------
# Memory-lean deep slow loader, with niche labels (step 6)
# ---------------------------------------------------------------------------
#
# `load_deep_slow_bets` above materializes every column of the 4.2M-row deep tape
# (measured peak RSS 2.97 GB on the 3 GB box — it survives only via swap). The
# finer-baseline work fits a hierarchy on top of that frame, so it needs the same
# universe at a fraction of the footprint. This loader does the per-market
# indexing in Arrow and projects away the columns nothing downstream reads
# (token_id, outcome, size, tx_hash, and — after `category`/niche are derived —
# slug and question).
#
# It is deliberately written ALONGSIDE the original rather than refactored out of
# it: `load_deep_slow_bets` is what produced the 5c verdict, and leaving it
# byte-for-byte untouched keeps that result reproducible.

_LEAN_BET_COLUMNS = ["wallet", "market_id", "side", "entry_price", "resolved",
                     "resolved_value", "timestamp"]


def _deep_market_index() -> pd.DataFrame:
    """Per-market slug/question and observed tape span, computed in Arrow so the
    4.2M-row tape is never materialized as Python objects. slug/question are
    constant within a market, so `min` picks the value."""
    import pyarrow.parquet as pq

    from src.slow_deepen import DEEP_TRADES_PATH

    span = (pq.read_table(DEEP_TRADES_PATH, columns=["market_id", "timestamp"])
            .group_by("market_id")
            .aggregate([("timestamp", "min"), ("timestamp", "max")])
            .to_pandas())
    text = (pq.read_table(DEEP_TRADES_PATH, columns=["market_id", "slug", "question"])
            .group_by("market_id")
            .aggregate([("slug", "min"), ("question", "min")])
            .to_pandas())
    idx = span.merge(text, on="market_id", how="left")
    return idx.rename(columns={"timestamp_min": "t_min", "timestamp_max": "t_max",
                               "slug_min": "slug", "question_min": "question"})


def load_deep_slow_bets_niched(buckets=SLOW_BUCKETS, cfg: dict | None = None) -> pd.DataFrame:
    """Resolved BUY bets of the deep slow universe with `category`, `niche_l1`,
    `niche_l2` and `res_ts_proxy` attached — the same universe as
    `load_deep_slow_bets`, at a fraction of the memory.

    Speed classification is identical: the market_meta sidecar's `lifespan_s`
    where present, else the deep tape's own observed span."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from src.slow_deepen import DEEP_TRADES_PATH
    from src.slow_niche import assign_niches

    cfg = cfg or load_config()
    slow_s, deep_s = speed_thresholds(cfg)

    idx = _deep_market_index()
    meta = load_market_meta(columns=["market_id", "lifespan_s"])
    sidecar = dict(zip(meta["market_id"], meta["lifespan_s"])) if not meta.empty else {}

    span_life = (idx["t_max"] - idx["t_min"]).to_numpy(dtype="float64")
    side_life = np.array([sidecar.get(m, np.nan) for m in idx["market_id"]], dtype="float64")
    lifespan = np.where(np.isnan(side_life), span_life, side_life)
    idx["speed_bucket"] = [classify_speed_bucket(L, slow_s, deep_s) for L in lifespan]
    idx = idx[idx["speed_bucket"].isin(buckets)].reset_index(drop=True)
    if idx.empty:
        return pd.DataFrame(columns=_LEAN_BET_COLUMNS + ["category", "niche_l1", "niche_l2"])

    idx["category"] = derive_categories(idx["slug"], idx["question"])
    idx = assign_niches(idx)

    tbl = pq.read_table(DEEP_TRADES_PATH, columns=_LEAN_BET_COLUMNS,
                        filters=[("side", "==", "BUY"), ("resolved", "==", True)])
    tbl = tbl.filter(pc.is_in(tbl["market_id"],
                              value_set=pa.array(idx["market_id"].to_numpy())))
    bets = tbl.to_pandas()
    del tbl

    keep = ["market_id", "category", "niche_family", "niche_form", "niche_l1",
            "niche_l2", "t_max"]
    bets = bets.merge(idx[keep], on="market_id", how="left")
    return bets.rename(columns={"t_max": "res_ts_proxy"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Concentration + FDR helpers (small; kept local rather than imported from scripts/)
# ---------------------------------------------------------------------------

def eff_breadth(market_labels) -> float:
    """Effective number of independent markets = 1 / HHI of per-market bet shares.
    A wallet with all bets in one market -> 1.0; evenly across k markets -> k."""
    labels = pd.Series(list(market_labels))
    if labels.empty:
        return 0.0
    shares = labels.value_counts(normalize=True).to_numpy()
    return float(1.0 / np.sum(shares ** 2))


def _day(ts_array) -> np.ndarray:
    return (np.asarray(ts_array, dtype="float64") // 86400).astype("int64")


def _n_pairs(labels, ts_array) -> int:
    """Distinct (label, calendar-day) pairs — the complex-unit analogue of a
    plain day count. `entry_days >= 3` can be satisfied by one complex entered on
    three days; this requires the spread to survive being read per complex."""
    days = _day(ts_array)
    return int(pd.unique(pd.Series(list(zip(list(labels), days.tolist())))).size)


def bh_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of rejections controlling FDR at level q.
    NaN p-values never reject. Standard step-up procedure."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return np.zeros(p.shape, dtype=bool)
    order = idx[np.argsort(p[idx])]
    m = idx.size
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    reject = np.zeros(p.shape, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[:kmax + 1]] = True
    return reject


# ---------------------------------------------------------------------------
# Per-wallet validation on the deep slow data
# ---------------------------------------------------------------------------

def validate_slow(deep_slow: pd.DataFrame, baselines: dict, cfg: dict | None = None,
                  pre_residualized: bool = False,
                  cluster_col: str = "market_id",
                  complex_gates: bool = False,
                  complex_col: str = COMPLEX_COL,
                  cfg_section: str = "slow") -> pd.DataFrame:
    """Per-wallet OOS validation on the deep slow bets. One row per wallet with all
    the gate columns (additive — nothing is dropped). `edge_persisted` is the AND of
    every gate; BH-FDR is applied by `slow_verdict` across candidates afterwards.

    `pre_residualized` accepts a frame that already carries `residual_skill` (the
    finer hierarchical baseline computes it in one fused fit+apply pass, since
    leave-one-wallet-out is only defined against the sample it was fitted on).

    `cluster_col` is the unit the significance bootstrap resamples. The default
    `market_id` is the repo-wide cluster-preserving standard. Passing `niche_l1`
    instead resamples EVENT COMPLEXES: the slow tape is full of rolling-deadline
    ladders ("US strikes Iran by Feb 28 / Mar 1 / Mar 15 / Mar 31 ..."), which are
    dozens of distinct markets with distinct resolution days but ONE underlying
    event. Market blocks treat those as independent draws; complex blocks do not.
    That is a strictly harsher test and is reported separately, never as the
    headline gate.

    `complex_gates` additionally enforces the concentration floors over EVENT
    COMPLEXES (step 7). The complex-unit columns are computed and reported
    whenever the complex column is present — they are additive metadata either
    way — but they only enter `edge_persisted` when this is True. Nothing is ever
    removed: the market-unit gates continue to apply unchanged alongside them.

    A wallet whose entire held-out record sits in ONE complex fails BY RULE, not
    by hand: the complex bootstrap has a single cluster, returns NaN, and
    `significant` requires a non-NaN p. See
    `test_single_complex_wallet_is_not_persisted_by_rule`.

    `complex_col` and `cfg_section` exist so a sibling arm can reuse this stack
    with its own cluster unit and its own scoped thresholds. The SPORTS arm
    passes `complex_col="event"` (a resolution event: one game, or one
    championship the whole outright field settles on) and `cfg_section="sports"`.
    Both default to the slow path's own values, so every existing caller — and
    the frozen forecaster cohort — is bit-identical."""
    cfg = cfg or load_config()
    scoring = cfg.get("scoring", {})
    slow_cfg = scoring.get(cfg_section, {}) or {}
    oos_split = float(scoring.get("oos_split", 0.5))
    n_boot = int(scoring.get("oos_bootstrap_resamples", 2000))
    alpha = float(scoring.get("oos_significance_alpha", 0.05))
    min_half = int(slow_cfg.get("min_bets_per_half", scoring.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF)))
    min_edge = float(slow_cfg.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE))
    min_markets = int(slow_cfg.get("min_oos_markets", DEFAULT_MIN_OOS_MARKETS))
    min_breadth = float(slow_cfg.get("min_eff_breadth", DEFAULT_MIN_EFF_BREADTH))
    min_entry_days = int(slow_cfg.get("min_entry_days", DEFAULT_MIN_ENTRY_DAYS))
    min_res_days = int(slow_cfg.get("min_resolution_days", DEFAULT_MIN_RESOLUTION_DAYS))
    min_complexes = int(slow_cfg.get("min_oos_complexes", DEFAULT_MIN_OOS_COMPLEXES))
    min_breadth_cx = float(slow_cfg.get("min_eff_breadth_complex",
                                        DEFAULT_MIN_EFF_BREADTH_COMPLEX))
    min_entry_days_cx = int(slow_cfg.get("min_entry_days_complex",
                                         DEFAULT_MIN_ENTRY_DAYS_COMPLEX))
    min_res_days_cx = int(slow_cfg.get("min_resolution_days_complex",
                                       DEFAULT_MIN_RESOLUTION_DAYS_COMPLEX))

    resid = deep_slow if pre_residualized else residualize(deep_slow, baselines)
    if cluster_col not in resid.columns:
        # Silently falling back would turn a deliberately harsher test into a
        # no-op that reads as "the harsher test changed nothing".
        raise KeyError(
            f"cluster_col {cluster_col!r} is not a column of the slow frame "
            f"(have: {sorted(resid.columns)}). Load via load_deep_slow_bets_niched.")
    has_complex = complex_col in resid.columns
    if complex_gates and not has_complex:
        raise KeyError(
            f"complex_gates=True but {complex_col!r} is not a column of the "
            f"frame. Load via load_deep_slow_bets_niched.")
    rows = []
    for wallet, g in resid.groupby("wallet", sort=False):
        if len(g) < 2 * min_half:
            continue
        ins, oos = split_in_sample_out_of_sample(g, oos_split)
        if len(ins) < min_half or len(oos) < min_half:
            continue
        in_mean = float(ins["residual_skill"].mean())
        out_resid = oos["residual_skill"].to_numpy(dtype=float)
        out_markets = oos["market_id"].to_numpy()
        out_blocks = oos[cluster_col].to_numpy()
        out_mean = float(out_resid.mean())
        n_markets = int(pd.unique(out_markets).size)
        rng = np.random.default_rng(_wallet_seed(wallet))
        cluster_p = _cluster_bootstrap_p(out_resid, out_blocks, rng, n_boot)
        breadth = eff_breadth(out_markets)
        entry_days = int(np.unique(_day(oos["timestamp"])).size)
        res_days = int(np.unique(_day(oos["res_ts_proxy"])).size) if "res_ts_proxy" in oos else entry_days

        candidate = in_mean > 0 and len(ins) >= min_half
        conc_ok = breadth >= min_breadth and entry_days >= min_entry_days and res_days >= min_res_days
        sig_ok = (not np.isnan(cluster_p)) and cluster_p < alpha
        mag_ok = out_mean >= min_edge
        markets_ok = n_markets >= min_markets

        row = {
            "wallet": wallet, "n_bets": len(g),
            "in_sample_skill": in_mean, "out_sample_skill": out_mean,
            "out_n": len(oos), "out_markets": n_markets,
            "cluster_p": cluster_p, "eff_breadth": breadth,
            "entry_days": entry_days, "resolution_days": res_days,
            "candidate": bool(candidate), "significant": bool(sig_ok),
            "magnitude_ok": bool(mag_ok), "markets_ok": bool(markets_ok),
            "concentration_ok": bool(conc_ok),
        }

        # --- complex-unit view: always reported when available, gated only on
        # request. Additive metadata, per the no-drop invariant.
        cx_ok = True
        if has_complex:
            cx = oos[complex_col].to_numpy()
            n_complexes = int(pd.unique(cx).size)
            breadth_cx = eff_breadth(cx)
            entry_days_cx = _n_pairs(cx, oos["timestamp"])
            res_days_cx = (_n_pairs(cx, oos["res_ts_proxy"]) if "res_ts_proxy" in oos
                           else entry_days_cx)
            counts = pd.Series(cx).value_counts()
            cx_ok = (n_complexes >= min_complexes and breadth_cx >= min_breadth_cx
                     and entry_days_cx >= min_entry_days_cx
                     and res_days_cx >= min_res_days_cx)
            row.update({
                "out_complexes": n_complexes,
                "eff_breadth_complex": breadth_cx,
                "entry_days_complex": entry_days_cx,
                "resolution_days_complex": res_days_cx,
                "top_complex": str(counts.index[0]),
                "top_complex_share": float(counts.iloc[0] / len(cx)),
                "complex_concentration_ok": bool(cx_ok),
            })

        persisted = bool(candidate and out_mean > 0 and sig_ok and mag_ok
                         and markets_ok and conc_ok
                         and (cx_ok if complex_gates else True))
        row["edge_persisted"] = persisted
        rows.append(row)
    return pd.DataFrame(rows)


def slow_verdict(table: pd.DataFrame, cfg: dict | None = None, fdr_q: float = 0.10) -> dict:
    """Apply BH-FDR across candidate cluster p-values and summarise. Returns a verdict
    dict (counts + FDR survivors). The per-gate columns already live on `table`."""
    cfg = cfg or load_config()
    if table.empty:
        return {"candidates": 0, "persisted": 0, "fdr_survivors": 0, "verdict": "no candidates"}
    cand = table[table["candidate"]].copy()
    persisted = int(table["edge_persisted"].sum())
    # BH-FDR over candidates that clear the economic + concentration gates (so FDR is
    # applied to the economically-meaningful set, not the whole thin tail).
    gated = cand[cand["magnitude_ok"] & cand["markets_ok"] & cand["concentration_ok"]].copy()
    fdr_survivors = 0
    if not gated.empty:
        rej = bh_reject(gated["cluster_p"].to_numpy(), fdr_q)
        gated = gated.assign(fdr_reject=rej)
        fdr_survivors = int(rej.sum())
        table = table.merge(gated[["wallet", "fdr_reject"]], on="wallet", how="left")
        table["fdr_reject"] = table["fdr_reject"].fillna(False)
    return {
        "candidates": int(table["candidate"].sum()),
        "gated_candidates": int(len(gated)),
        "persisted": persisted,
        "fdr_survivors": fdr_survivors,
        "fdr_q": fdr_q,
        "table": table,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Baseline granularity. `category` is the 5c default and is left bit-identical;
# the finer modes fit the hierarchical leave-one-wallet-out baseline (step 6b).
BASELINE_MODES = {
    "category": ("category",),
    "niche": ("category", "niche_l1"),
    "niche_form": ("category", "niche_l1", "niche_l2"),
}


def residualize_for(deep_slow: pd.DataFrame, mode: str, cfg: dict | None = None):
    """Residualize the slow universe at the requested baseline granularity.
    Returns (frame with `residual_skill`, info-or-None)."""
    cfg = cfg or load_config()
    if mode == "category":
        return residualize(deep_slow, fit_slow_category_baselines(deep_slow, cfg=cfg)), None
    if mode not in BASELINE_MODES:
        raise ValueError(f"unknown baseline mode {mode!r}; choose from {sorted(BASELINE_MODES)}")
    from src.slow_baseline import fit_hierarchical_residuals
    return fit_hierarchical_residuals(deep_slow, levels=BASELINE_MODES[mode], lowo=True)


# The slow path's STANDARD configuration as of step 7. The event complex is the
# cluster unit and the concentration gates are read in complex units; the finer
# hierarchical baseline is the default residualizer. Reproduce the historical 5c
# verdict with `--baseline category --cluster market_id --no-complex-gates`.
STANDARD_BASELINE = "niche_form"
STANDARD_CLUSTER = COMPLEX_COL


def run_validate(buckets=SLOW_BUCKETS, fdr_q: float = 0.10,
                 baseline: str = STANDARD_BASELINE, cluster: str = STANDARD_CLUSTER,
                 complex_gates: bool = True) -> dict:
    cfg = load_config()
    # The niched loader is needed whenever ANYTHING downstream reads a niche
    # column — the finer baseline, a non-market cluster unit, OR the complex
    # gates. Getting this wrong makes a deliberately harsher test a silent no-op.
    plain = baseline == "category" and cluster == "market_id" and not complex_gates
    loader = load_deep_slow_bets if plain else load_deep_slow_bets_niched
    deep_slow = loader(buckets=buckets, cfg=cfg)
    if deep_slow.empty:
        print("[slow_validate] no deep slow bets — run 5b (slow_deepen) first.")
        return {}
    print(f"[slow_validate] deep slow universe: {len(deep_slow):,} resolved BUY bets, "
          f"{deep_slow['wallet'].nunique():,} wallets, {deep_slow['market_id'].nunique():,} markets")
    print(f"[slow_validate] baseline={baseline}  significance clusters={cluster}  "
          f"complex-unit gates={'ON' if complex_gates else 'OFF'}")
    resid, info = residualize_for(deep_slow, baseline, cfg=cfg)
    if info is not None:
        print(f"[slow_validate] shrinkage k per level: "
              f"{ {k: round(v, 2) for k, v in info.ks.items()} }")
        print(info.pooling_report().to_string(index=False))
    table = validate_slow(resid, baselines={}, cfg=cfg, pre_residualized=True,
                          cluster_col=cluster, complex_gates=complex_gates)
    verdict = slow_verdict(table, cfg=cfg, fdr_q=fdr_q)
    out = verdict.get("table", table)
    # The STANDARD configuration owns the canonical filename; every variant
    # (including the historical 5c settings) writes beside it, so no run can
    # silently overwrite another's published numbers.
    standard = (baseline == STANDARD_BASELINE and cluster == STANDARD_CLUSTER
                and complex_gates)
    suffix = "" if standard else \
        f"_{baseline}_{cluster}{'' if complex_gates else '_nocxgates'}"
    path = (SLOW_VALIDATED_PATH if not suffix else
            SLOW_VALIDATED_PATH.with_name(f"slow_validated{suffix}.parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(out, path, compression="gzip")

    print(f"\n[slow_validate] candidates (in-sample skill>0): {verdict['candidates']:,}")
    print(f"[slow_validate] clearing 10c + markets + concentration gates: {verdict.get('gated_candidates', 0):,}")
    print(f"[slow_validate] edge_persisted (all gates incl. cluster-robust sig): {verdict['persisted']:,}")
    print(f"[slow_validate] BH-FDR survivors @ q={fdr_q}: {verdict['fdr_survivors']:,}")
    print(f"[slow_validate] -> {path}")
    if verdict["persisted"]:
        cols = ["wallet", "out_sample_skill", "out_n", "out_markets", "cluster_p",
                "eff_breadth", "entry_days", "resolution_days"]
        if "out_complexes" in out.columns:
            cols += ["out_complexes", "eff_breadth_complex", "top_complex",
                     "top_complex_share"]
        top = out[out["edge_persisted"]].sort_values("out_sample_skill", ascending=False)
        with pd.option_context("display.width", 240):
            print("\n[persisted slow-market forecasters]\n"
                  + top[cols].round(4).to_string(index=False))
    else:
        print("\n[slow_validate] VERDICT: no wallet clears the slow-universe gates — "
              "consistent with the flat stage-1 gradient. The honest null.")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate the deepened slow shortlist on fresh deep data. The "
                    "STANDARD configuration (step 7) resamples event complexes and reads "
                    "the concentration gates in complex units. Read-only; the forward "
                    "paper test is the real arbiter. Reproduce the historical 5c verdict "
                    "with `--baseline category --cluster market_id --no-complex-gates`.")
    ap.add_argument("--buckets", type=str, default=",".join(SLOW_BUCKETS))
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--baseline", choices=sorted(BASELINE_MODES), default=STANDARD_BASELINE,
                    help="price-baseline granularity. 'niche_form' is the standard "
                         "(hierarchical leave-one-wallet-out, step 6b); 'category' is "
                         "the historical 5c setting.")
    ap.add_argument("--cluster", choices=["market_id", COMPLEX_COL], default=STANDARD_CLUSTER,
                    help="unit the significance bootstrap resamples. 'niche_l1' (event "
                         "complexes) is the standard; 'market_id' is the historical "
                         "setting, which counts a rolling-deadline ladder as dozens of "
                         "independent draws.")
    ap.add_argument("--no-complex-gates", action="store_true",
                    help="report the complex-unit concentration columns but do not let "
                         "them gate edge_persisted (the pre-step-7 behaviour).")
    args = ap.parse_args()
    run_validate(buckets=tuple(b.strip() for b in args.buckets.split(",") if b.strip()),
                 fdr_q=args.fdr_q, baseline=args.baseline, cluster=args.cluster,
                 complex_gates=not args.no_complex_gates)


if __name__ == "__main__":
    main()
