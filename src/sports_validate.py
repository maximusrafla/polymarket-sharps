"""Validate the deepened sports shortlist on fresh deep data — Project 3 sports step 4.

THE DISJOINT-DATA FIREWALL. The screen (step 3) selected on the gate's shallow,
slow-filtered corpus — 765k bets whose 284 markets fold to ~25 resolution events,
with 81% of findable wallets spanning <=2 of them. Every number from that corpus
is selection noise until it survives here. Deepening (step 3) fetched each
shortlisted wallet's full `/trades?user=` history, which is dominated by
per-game markets the screen never saw: the main ledger alone carries 25,390
distinct coded games across 193 league codes.

WHAT IS DIFFERENT FROM THE SLOW ARM
-----------------------------------
The stack is `slow_validate.validate_slow` — the same OOS split, market-block
bootstrap, magnitude floor, concentration guards, FDR and no-drop invariant —
with two substitutions:

  cluster unit = `event` (src/sports_events.py), NOT `market_id` and NOT the
      league. A game is its own cluster, so 25k games stay 25k independent
      draws and the breadth that makes sports worth testing survives; but the
      54 team markets of one NBA Finals collapse to the one event they settle
      on, so a wallet cannot buy a whole field and be credited with 54 draws.

  scoped thresholds under `scoring.sports.*`. The global 2c floor stays exactly
      where it is — Project 1 reads `scoring.min_skill_edge` and never this
      block, which is asserted in the tests.

Speed buckets are NOT applied. The slow arm exists to isolate slow markets; the
sports arm wants every sports market a shortlisted wallet touched, and a game
resolving in hours is the fast-resolving property that makes its forward test
readable in days rather than months.

THE FINER-BASELINE RE-CHECK IS MANDATORY, NOT OPTIONAL (`--baseline sub_form`,
`--bins 40`). A league-level baseline leaves any sub-league miscalibration —
home-favourite bias is the classic one — in the residual, where it reads as
wallet skill on every buyer of that side. That is the confound that cut the
forecasters 52 -> 12, and it is checked the same way: refit one level finer and
report what survives.

READ-ONLY / provisional: the forward test is the arbiter. Output goes to the
isolated interim dir, never data/processed/. Nothing is dropped — every wallet
keeps its row with its gates.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import atomic_to_parquet, load_config
from src.slow_baseline import fit_hierarchical_residuals
from src.slow_market import derive_categories
from src.slow_niche import assign_niches
from src.slow_validate import bh_reject, slow_verdict, validate_slow
from src.sports_deepen import DEEP_TRADES_PATH, SPORTS_DIR
from src.sports_events import assign_events, assign_sub_forms, is_sports_market

SPORTS_VALIDATED_PATH = SPORTS_DIR / "sports_validated.parquet"

CLUSTER_COL = "event"
CFG_SECTION = "sports"

# Baseline granularity. `league_form` is the standard; `sub_form` is the finer
# re-check (adds the line SIDE); `league` is the coarse reference that shows how
# much of the result the finer levels are already stripping.
BASELINE_MODES = {
    "league": ("niche_l1",),
    "league_form": ("niche_l1", "niche_l2"),
    "sub_form": ("niche_l1", "niche_l2", "niche_l3"),
}
STANDARD_BASELINE = "league_form"

_LEAN_BET_COLUMNS = ["wallet", "market_id", "side", "entry_price", "resolved",
                     "resolved_value", "timestamp"]


# ---------------------------------------------------------------------------
# Deep sports universe
# ---------------------------------------------------------------------------

def _deep_market_index(path=None) -> pd.DataFrame:
    """Per-market slug/question and observed tape span, computed in Arrow so the
    multi-million-row deep tape is never materialized as Python objects.
    slug/question are constant within a market, so `min` picks the value."""
    import pyarrow.parquet as pq

    path = path or DEEP_TRADES_PATH
    span = (pq.read_table(path, columns=["market_id", "timestamp"])
            .group_by("market_id")
            .aggregate([("timestamp", "min"), ("timestamp", "max")])
            .to_pandas())
    text = (pq.read_table(path, columns=["market_id", "slug", "question"])
            .group_by("market_id")
            .aggregate([("slug", "min"), ("question", "min")])
            .to_pandas())
    idx = span.merge(text, on="market_id", how="left")
    return idx.rename(columns={"timestamp_min": "t_min", "timestamp_max": "t_max",
                               "slug_min": "slug", "question_min": "question"})


def sports_market_index(path=None) -> pd.DataFrame:
    """Every SPORTS market in the deep tape, labelled with category, niche levels,
    the resolution event and a resolution-time proxy (`res_ts_proxy` = the last
    observed trade, a lower bound on resolution, used only by the resolution-day
    clustering guard)."""
    idx = _deep_market_index(path)
    if idx.empty:
        return idx
    idx["category"] = derive_categories(idx["slug"], idx["question"])
    keep = [is_sports_market(s, q, c) for s, q, c in
            zip(idx["slug"], idx["question"], idx["category"])]
    idx = idx[np.array(keep, dtype=bool)].reset_index(drop=True)
    if idx.empty:
        return idx
    idx = assign_niches(idx)
    idx = assign_sub_forms(idx)
    idx = assign_events(idx)
    return idx.rename(columns={"t_max": "res_ts_proxy"})


def load_deep_sports_bets(cfg: dict | None = None, path=None) -> pd.DataFrame:
    """Resolved BUY bets of the deep sports universe with every label attached."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = path or DEEP_TRADES_PATH
    idx = sports_market_index(path)
    if idx.empty:
        return pd.DataFrame(columns=_LEAN_BET_COLUMNS + ["event"])

    tbl = pq.read_table(path, columns=_LEAN_BET_COLUMNS,
                        filters=[("side", "==", "BUY"), ("resolved", "==", True)])
    tbl = tbl.filter(pc.is_in(tbl["market_id"],
                              value_set=pa.array(idx["market_id"].to_numpy())))
    bets = tbl.to_pandas()
    del tbl
    cols = ["market_id", "category", "niche_family", "niche_form", "niche_l1",
            "niche_l2", "niche_l3", "sub_form", "event", "event_kind", "res_ts_proxy"]
    return bets.merge(idx[cols], on="market_id", how="left").reset_index(drop=True)


def describe_universe(bets: pd.DataFrame) -> None:
    n_mkt = bets["market_id"].nunique()
    n_ev = bets["event"].nunique()
    kinds = bets.drop_duplicates("market_id")["event_kind"].value_counts()
    print(f"[sports_validate] deep sports universe: {len(bets):,} resolved BUY bets, "
          f"{bets['wallet'].nunique():,} wallets, {n_mkt:,} markets -> {n_ev:,} EVENTS")
    print(f"[sports_validate] market kinds: {kinds.to_dict()}  "
          f"(unmatched, i.e. own-event fallback: {kinds.get('market', 0) / max(n_mkt, 1):.1%})")
    top = (bets.groupby("niche_l1").size().sort_values(ascending=False) / len(bets)).head(8)
    print("[sports_validate] top leagues by bet share: "
          + ", ".join(f"{k}={v:.1%}" for k, v in top.items()))


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------

def residualize_for(bets: pd.DataFrame, mode: str, n_bins: int = 20,
                    cfg: dict | None = None):
    """Hierarchical leave-one-wallet-out residual skill at the requested
    granularity. LOWO matters here for the same reason it does on the slow path:
    at league|form|side granularity one deep wallet can be a large share of a
    cell, and letting its own outcomes into its own baseline would collapse its
    residual toward zero — a false negative indistinguishable from no skill."""
    if mode not in BASELINE_MODES:
        raise ValueError(f"unknown baseline mode {mode!r}; choose from {sorted(BASELINE_MODES)}")
    return fit_hierarchical_residuals(bets, levels=BASELINE_MODES[mode],
                                      n_bins=n_bins, lowo=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def magnitude_sensitivity(table: pd.DataFrame,
                          floors=(0.02, 0.05, 0.10, 0.15)) -> pd.DataFrame:
    """How many wallets clear EVERY gate except the magnitude floor, at several
    floors. DIAGNOSTIC ONLY — the pre-registered floor is `scoring.sports.
    min_skill_edge` and this never moves it. It exists so a null result can be
    read honestly: "nobody cleared 10c" and "nobody cleared 2c either" are very
    different findings, and printing only the first would hide which one holds."""
    if table.empty:
        return pd.DataFrame()
    base = table[table["candidate"] & table["significant"] & table["markets_ok"]
                 & table["concentration_ok"]]
    if "complex_concentration_ok" in table.columns:
        base = base[base["complex_concentration_ok"]]
    return pd.DataFrame([{"floor": f, "n_clearing": int((base["out_sample_skill"] >= f).sum())}
                         for f in floors])


def validate_slow_sports(resid: pd.DataFrame, cfg: dict | None = None,
                         cluster: str = CLUSTER_COL) -> pd.DataFrame:
    """The stage-4 validator wired for sports: EVENT clusters, event-unit
    concentration gates, `scoring.sports.*` thresholds. One call site for the
    settings so the freeze cannot drift from the validation that produced it."""
    return validate_slow(resid, baselines={}, cfg=cfg or load_config(),
                         pre_residualized=True, cluster_col=cluster,
                         complex_gates=True, complex_col=CLUSTER_COL,
                         cfg_section=CFG_SECTION)


def run_validate(baseline: str = STANDARD_BASELINE, n_bins: int = 20,
                 fdr_q: float = 0.10, cluster: str = CLUSTER_COL,
                 bets: pd.DataFrame | None = None, save: bool = True) -> dict:
    cfg = load_config()
    bets = load_deep_sports_bets(cfg=cfg) if bets is None else bets
    if bets.empty:
        print("[sports_validate] no deep sports bets — run step 3 (sports_deepen) first.")
        return {}
    describe_universe(bets)
    print(f"[sports_validate] baseline={baseline} ({'x'.join(BASELINE_MODES[baseline])}, "
          f"{n_bins} price bins)  clusters={cluster}  scoped thresholds=scoring.{CFG_SECTION}")

    resid, info = residualize_for(bets, baseline, n_bins=n_bins, cfg=cfg)
    print(f"[sports_validate] shrinkage k per level: "
          f"{ {k: round(v, 2) for k, v in info.ks.items()} }")
    print(info.pooling_report().to_string(index=False))

    table = validate_slow_sports(resid, cfg=cfg, cluster=cluster)
    if table.empty:
        print("[sports_validate] no wallet has enough resolved sports bets to test.")
        return {}
    verdict = slow_verdict(table, cfg=cfg, fdr_q=fdr_q)
    out = verdict.get("table", table)
    out = out.assign(baseline=baseline, price_bins=n_bins, cluster=cluster)

    if save:
        standard = baseline == STANDARD_BASELINE and n_bins == 20 and cluster == CLUSTER_COL
        suffix = "" if standard else f"_{baseline}_{n_bins}bins_{cluster}"
        path = (SPORTS_VALIDATED_PATH if not suffix else
                SPORTS_VALIDATED_PATH.with_name(f"sports_validated{suffix}.parquet"))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(out, path, compression="gzip")
        print(f"[sports_validate] -> {path}")

    print(f"\n[sports_validate] candidates (in-sample skill>0): {verdict['candidates']:,}")
    print(f"[sports_validate] clearing magnitude + markets + concentration: "
          f"{verdict.get('gated_candidates', 0):,}")
    print(f"[sports_validate] edge_persisted (all gates incl. cluster-robust sig): "
          f"{verdict['persisted']:,}")
    print(f"[sports_validate] BH-FDR survivors @ q={fdr_q}: {verdict['fdr_survivors']:,}")
    sens = magnitude_sensitivity(out)
    if not sens.empty:
        print("[sports_validate] magnitude sensitivity (all other gates cleared) — "
              "DIAGNOSTIC, the pre-registered floor does not move:")
        print("  " + "  ".join(f"{r.floor:.2f}c-floor: {r.n_clearing}"
                               for r in sens.itertuples()))
    held = out.loc[out["candidate"], "out_sample_skill"]
    if len(held):
        qs = held.quantile([0.5, 0.9, 0.99, 1.0]).round(4).to_dict()
        print(f"[sports_validate] held-out skill edge across candidates: "
              f"median {qs[0.5]:+.4f}, p90 {qs[0.9]:+.4f}, p99 {qs[0.99]:+.4f}, "
              f"max {qs[1.0]:+.4f}")

    if verdict["persisted"]:
        cols = ["wallet", "out_sample_skill", "out_n", "out_markets", "out_complexes",
                "cluster_p", "eff_breadth_complex", "top_complex", "top_complex_share",
                "entry_days", "resolution_days"]
        top = out[out["edge_persisted"]].sort_values("out_sample_skill", ascending=False)
        with pd.option_context("display.width", 240):
            print("\n[persisted sports wallets]\n"
                  + top[[c for c in cols if c in top.columns]].round(4).to_string(index=False))
    else:
        print("\n[sports_validate] VERDICT: no wallet clears the sports gates on fresh "
              "deep data — the screen's shortlist was selection noise.")
    verdict["table"] = out
    return verdict


def compare_baselines(fdr_q: float = 0.10) -> pd.DataFrame:
    """The mandatory finer-baseline re-check: the standard fit, one level finer
    (the line side), and finer price bins — survivors and per-wallet edge shift
    side by side. Loads the universe once and refits on it, so the three runs
    differ ONLY in the baseline."""
    cfg = load_config()
    bets = load_deep_sports_bets(cfg=cfg)
    if bets.empty:
        print("[sports_validate] no deep sports bets — run step 3 first.")
        return pd.DataFrame()

    runs = [("league", 20), ("league_form", 20), ("sub_form", 20), ("league_form", 40),
            ("sub_form", 40)]
    tables = {}
    for mode, bins in runs:
        print(f"\n{'=' * 74}\nBASELINE {mode} / {bins} bins\n{'=' * 74}")
        v = run_validate(baseline=mode, n_bins=bins, fdr_q=fdr_q, bets=bets, save=True)
        if v:
            tables[(mode, bins)] = v["table"]

    if not tables:
        return pd.DataFrame()
    ref = tables.get((STANDARD_BASELINE, 20))
    rows = []
    for (mode, bins), t in tables.items():
        row = {"baseline": mode, "price_bins": bins,
               "candidates": int(t["candidate"].sum()),
               "persisted": int(t["edge_persisted"].sum()),
               "fdr_survivors": int(t.get("fdr_reject", pd.Series(dtype=bool)).sum())}
        if ref is not None:
            m = ref[["wallet", "out_sample_skill", "edge_persisted"]].merge(
                t[["wallet", "out_sample_skill", "edge_persisted"]], on="wallet",
                suffixes=("_ref", ""))
            shift = m["out_sample_skill"] - m["out_sample_skill_ref"]
            row["median_edge_shift_vs_standard"] = float(shift.median())
            surv = m[m["edge_persisted_ref"]]
            row["standard_survivors_held"] = int(surv["edge_persisted"].sum())
            row["standard_survivors"] = int(len(surv))
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'=' * 74}\nFINER-BASELINE RE-CHECK\n{'=' * 74}")
    print(summary.to_string(index=False))
    atomic_to_parquet(summary, SPORTS_DIR / "sports_baseline_grid.parquet",
                      compression="gzip")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate the deepened sports shortlist on fresh deep data "
                    "(Project 3 sports step 4). Clusters resolution EVENTS, scoped "
                    "to scoring.sports.*; the global floor is untouched.")
    ap.add_argument("--baseline", choices=sorted(BASELINE_MODES), default=STANDARD_BASELINE)
    ap.add_argument("--bins", type=int, default=20, help="price-baseline quantile bins")
    ap.add_argument("--cluster", choices=["event", "market_id"], default=CLUSTER_COL,
                    help="unit the significance bootstrap resamples. 'event' is the "
                         "standard; 'market_id' is the gate's setting, which counts "
                         "one championship's whole outright field as many draws.")
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--finer-recheck", action="store_true",
                    help="run the mandatory finer-baseline grid and summarise it")
    args = ap.parse_args()

    if args.finer_recheck:
        compare_baselines(fdr_q=args.fdr_q)
    else:
        run_validate(baseline=args.baseline, n_bins=args.bins, fdr_q=args.fdr_q,
                     cluster=args.cluster)


if __name__ == "__main__":
    main()
