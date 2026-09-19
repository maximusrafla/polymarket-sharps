"""Project-1 validation on the deep REAL-WORLD sample (src/realworld_deepen.py).

WHAT THIS ANSWERS. Project 1's certification gate — candidacy, cluster-robust
market-block bootstrap significance at the scoped α, an economic-magnitude floor,
and a held-out distinct-market floor — has only ever been run on the shared bet
ledger. That ledger is what a global firehose that is 82% 5-minute crypto leaves
behind, so its real-world side is 3,192 wallets with 98 at 1000+ bets: a residue,
not a sample. This runs THE SAME GATE, unchanged, on a population that is both
deep and drawn without reference to performance.

WHY THE ANSWER IS REWEIGHTABLE, AND WHAT IT IS REWEIGHTABLE *TO*. The wallets were
drawn by stratified probability sampling with known inclusion probabilities, so a
Horvitz–Thompson estimate turns "k of the sampled wallets certify" into "≈K of the
frame would certify". ⚠️ The frame is **not** "real-world Polymarket traders". It
is *wallets with ≥20 bets and ≥5 distinct markets inside the 585-market discovery
corpus* — 6,802 of them. Generalizing past that is unsupported by this design, and
the temptation to do it anyway is exactly the error the volume-selected census made.

THE MICRO RESTRICTION IS APPLIED FIRST, ON PURPOSE. `/trades?user=` returns each
wallet's whole history, micro-crypto included (11.9% of this tape). Restricting to
`discover.is_real_world` markets BEFORE anything downstream means the
favorite-longshot price baseline, the chronological halves, the clusters and the
significance test are all fitted inside the real-world population rather than
inherited from a tape shaped by the fast market. The unrestricted variant is
reported beside it, never as the headline.

THE 383 REUSED WALLETS ARE INCLUDED, AND THEY HAVE TO BE. 383 of the 2,500 drawn
wallets were already deepened by the slow/sports arms, so `realworld_deepen` never
re-fetched them and their bets live in those directories. Dropping them here would
not be a neutral simplification: those arms selected on measured edge, so the
excluded set is performance-correlated and the remainder is no longer the sample
that was drawn. This module reads them back from the prior arms so the estimator
sees all 2,500.

READ-ONLY: no network, no writes outside `data/interim/realworld/`.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import INTERIM_DIR, atomic_to_parquet, load_config
from src.discover import is_real_world
from src.realworld_deepen import (
    POOL_CENSUS_PATH,
    RW_DIR,
    SHORTLIST_PATH,
    load_deep_tape,
)
from src.validate import VALIDATE_LEDGER_COLUMNS, certification_alpha, compute_oos_validation

VALIDATED_PATH = RW_DIR / "validated.parquet"

# Where the 383 reused wallets' bets actually live.
PRIOR_ARM_TAPES = {
    "slow": INTERIM_DIR / "slow_deepening" / "deep_trades.parquet",
    "sports": INTERIM_DIR / "sports" / "deep_trades.parquet",
}

TAPE_COLUMNS = sorted(set(VALIDATE_LEDGER_COLUMNS) | {"slug", "question"})


# ---------------------------------------------------------------------------
# Population assembly
# ---------------------------------------------------------------------------

# Read fat, keep slim. `slug`/`question` are needed only to classify a market and
# `tx_hash`/`token_id` only to dedup a fill; all four are the widest columns in the
# tape. Reading them inside a per-file scope and dropping them before anything is
# accumulated is what keeps peak RSS flat — a straight concat of the deepened tape
# (7.05M rows) and the prior-arm tape (2.83M) was OOM-killed on this 3 GB box.
READ_COLUMNS = sorted(set(VALIDATE_LEDGER_COLUMNS) | {"slug", "question", "tx_hash", "token_id"})
SLIM_COLUMNS = list(VALIDATE_LEDGER_COLUMNS)

# Only resolved BUY fills ever reach the gate (`validate.only_buys`, then
# `resolved`), so the filter is pushed down to the parquet reader rather than
# applied after materializing 6M rows that are then discarded.
BET_FILTER = [("side", "==", "BUY"), ("resolved", "==", True)]


def _slim(df: pd.DataFrame, cache: dict[str, bool]) -> pd.DataFrame:
    """Dedup fills, drop micro-crypto, and return only the gate's columns.

    `cache` memoizes market_id -> is_real_world across every file, so the slug
    classifier runs once per market instead of once per file that touches it."""
    from src.discover import classify_market

    if df.empty:
        return pd.DataFrame(columns=SLIM_COLUMNS)
    key = [c for c in ("tx_hash", "wallet", "token_id", "side") if c in df.columns]
    if key:
        df = df.drop_duplicates(subset=key, keep="last")
    unseen = df.loc[df["market_id"].map(cache).isna()].drop_duplicates("market_id")
    for mid, slug, question in zip(unseen["market_id"], unseen["slug"], unseen["question"]):
        cache[mid] = is_real_world(classify_market(slug or "", question or "", None))
    df = df.loc[df["market_id"].map(cache).astype(bool)]
    return df[SLIM_COLUMNS].reset_index(drop=True)


def load_prior_arm_bets(wallets: set[str], cache: dict[str, bool] | None = None) -> pd.DataFrame:
    """The reused wallets' bets, read back from the arms that deepened them.

    Filtered at the parquet reader so neither prior tape (4.2M and 2.0M rows) is
    ever fully materialized, and slimmed per file so the wide columns never
    accumulate."""
    import pyarrow.parquet as pq

    cache = {} if cache is None else cache
    if not wallets:
        return pd.DataFrame(columns=SLIM_COLUMNS)
    frames = []
    for path in PRIOR_ARM_TAPES.values():
        if not path.exists():
            continue
        table = pq.read_table(path, columns=READ_COLUMNS,
                              filters=BET_FILTER + [("wallet", "in", list(wallets))])
        if table.num_rows:
            frames.append(_slim(table.to_pandas(), cache))
        del table
    if not frames:
        return pd.DataFrame(columns=SLIM_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    # A wallet can sit in BOTH prior arms (120 overlap), so the same fill can
    # arrive twice here even though each arm's own file is internally deduped.
    return out.drop_duplicates().reset_index(drop=True)


def restrict_to_real_world(bets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop micro-crypto bets, classifying by slug/question.

    `market_meta`'s speed_bucket cannot do this job on the deep tape (it covers
    116k of 410k markets), and a tape-derived lifespan would be actively wrong —
    we hold only our own wallets' fills, so a market touched once looks
    zero-second and reads as micro. See realworld_deepen.classify_deep_markets."""
    from src.discover import classify_market

    per_market = bets.drop_duplicates("market_id")[["market_id", "slug", "question"]]
    keep_market = {
        mid: is_real_world(classify_market(slug or "", question or "", None))
        for mid, slug, question in zip(per_market["market_id"], per_market["slug"],
                                       per_market["question"])
    }
    mask = bets["market_id"].map(keep_market).astype(bool)
    report = {
        "bets_before": int(len(bets)), "bets_after": int(mask.sum()),
        "wallets_before": int(bets["wallet"].nunique()),
        "wallets_after": int(bets.loc[mask, "wallet"].nunique()),
        "markets_before": int(bets["market_id"].nunique()),
        "markets_after": int(bets.loc[mask, "market_id"].nunique()),
    }
    return bets.loc[mask].reset_index(drop=True), report


def assemble_population(real_world: bool = True) -> tuple[pd.DataFrame, dict]:
    """All 2,500 drawn wallets' resolved BUY bets: the freshly deepened 2,117 plus
    the 383 read back from the prior arms.

    Streamed SHARD BY SHARD, each one filtered, deduped, micro-dropped and slimmed
    before the next is read. A straight concat of the two tapes (7.05M + 2.83M rows,
    all columns) was OOM-killed here; what accumulates now is ~3.4M rows of six
    narrow columns."""
    import gc

    import pyarrow.parquet as pq

    from src.realworld_deepen import shard_paths

    cache: dict[str, bool] = {}
    kept, raw_rows = [], 0
    for p in shard_paths():
        table = pq.read_table(p, columns=READ_COLUMNS, filters=BET_FILTER)
        raw_rows += table.num_rows
        df = table.to_pandas()
        del table
        kept.append(_slim(df, cache) if real_world else df[SLIM_COLUMNS])
        del df
        gc.collect()
    deep = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=SLIM_COLUMNS)
    del kept
    gc.collect()

    short = pd.read_parquet(SHORTLIST_PATH)
    reused = set(short.loc[short["prior_arm"] != "", "wallet"])
    prior = load_prior_arm_bets(reused, cache)
    print(f"[rw_validate] deepened {len(deep):,} of {raw_rows:,} resolved BUY rows "
          f"({deep['wallet'].nunique():,} wallets) + prior-arm {len(prior):,} rows "
          f"({prior['wallet'].nunique() if len(prior) else 0:,} reused wallets)", flush=True)

    bets = pd.concat([deep, prior], ignore_index=True)
    del deep, prior
    gc.collect()
    # market_id is a 66-char hex string with only ~306k distinct values across
    # ~4.9M rows, so carrying it as objects costs ~600 MB and pushed the first run
    # 2 GB into swap. Dictionary-encoding it is the same thing validate.main does
    # (`load_ledger(..., categorical=["market_id"])`) and is the configuration
    # compute_oos_validation is tested against. `wallet` is deliberately left as
    # objects to match main() exactly rather than guess at a groupby interaction.
    bets["market_id"] = bets["market_id"].astype("category")
    gc.collect()
    report = {
        "assembled_rows": int(len(bets)),
        "assembled_wallets": int(bets["wallet"].nunique()),
        "bets_before": int(raw_rows), "bets_after": int(len(bets)),
        "markets_after": int(bets["market_id"].nunique()),
        "micro_dropped": bool(real_world),
    }
    return bets, report


# ---------------------------------------------------------------------------
# Horvitz–Thompson reweighting
# ---------------------------------------------------------------------------

def attach_weights(validated: pd.DataFrame) -> pd.DataFrame:
    """Attach each wallet's stratum and design weight 1/inclusion_prob."""
    census = pd.read_parquet(POOL_CENSUS_PATH)
    cols = ["wallet", "stratum", "inclusion_prob", "selected", "prior_arm",
            "n_discovery_bets"]
    out = validated.merge(census[cols], on="wallet", how="left")
    out["design_weight"] = np.where(out["inclusion_prob"].fillna(0) > 0,
                                    1.0 / out["inclusion_prob"], np.nan)
    return out


def ht_estimate(validated: pd.DataFrame, flag: str) -> dict:
    """Horvitz–Thompson total and rate for a boolean per-wallet flag, over the frame.

    The variance is the standard stratified-sampling one: strata taken at p=1
    contribute ZERO sampling variance (they are a census, not a draw), so all the
    uncertainty comes from the sampled band. This is a design-based interval about
    *the frame*, and says nothing about wallets outside it."""
    d = validated.dropna(subset=["design_weight"])
    total = float((d[flag].astype(float) * d["design_weight"]).sum())
    frame_n = float(d["design_weight"].sum())
    var = 0.0
    for _stratum, grp in d.groupby("stratum"):
        n, p = len(grp), float(grp["inclusion_prob"].iloc[0])
        if n < 2 or p >= 1.0:
            continue                      # censused stratum: no sampling variance
        N = n / p
        s2 = float(grp[flag].astype(float).var(ddof=1))
        var += N * N * (1 - p) * s2 / n   # finite-population-corrected
    se = float(np.sqrt(var))
    return {"sample_count": int(d[flag].sum()), "sample_n": int(len(d)),
            "frame_total": total, "frame_n": frame_n,
            "frame_rate": total / frame_n if frame_n else float("nan"),
            "frame_total_se": se,
            "frame_total_lo": total - 1.96 * se, "frame_total_hi": total + 1.96 * se}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(real_world: bool = True, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    bets, report = assemble_population(real_world=real_world)
    print(f"[rw_validate] population: {report['bets_after']:,} resolved BUY bets across "
          f"{report['assembled_wallets']:,} wallets and {report['markets_after']:,} markets"
          + ("  (micro-crypto dropped before the baseline was fitted)"
             if report["micro_dropped"] else "  (UNRESTRICTED — micro included)"))

    print(f"[rw_validate] running the Project 1 gate (alpha={certification_alpha(cfg['scoring'])}, "
          f"min_skill_edge={cfg['scoring'].get('min_skill_edge')}, "
          f"min_oos_markets={cfg['scoring'].get('min_oos_markets')}) on "
          f"{bets['wallet'].nunique():,} wallets...", flush=True)
    validated = compute_oos_validation(bets, cfg)
    del bets
    validated = attach_weights(validated)
    atomic_to_parquet(validated, VALIDATED_PATH, compression="gzip")

    n = len(validated)
    cand = validated["in_sample_residual_edge"] > 0
    print(f"\n[rw_validate] {n:,} wallets validated -> {VALIDATED_PATH}")
    print(f"  candidates (in-sample residual edge > 0): {int(cand.sum()):,} "
          f"({100 * cand.mean():.1f}%)")
    for flag in ("edge_significant", "edge_magnitude_ok", "edge_markets_ok", "edge_persisted"):
        if flag not in validated.columns:
            continue
        est = ht_estimate(validated, flag)
        print(f"  {flag:<20} sample {est['sample_count']:>5,}/{est['sample_n']:<5,}"
              f"   frame estimate {est['frame_total']:>8.0f} of {est['frame_n']:,.0f}"
              f"  ({100 * est['frame_rate']:.2f}%)"
              f"   95% CI [{max(est['frame_total_lo'], 0):.0f}, {est['frame_total_hi']:.0f}]")

    if "edge_persisted" in validated.columns:
        top = validated.loc[validated["edge_persisted"]].nlargest(
            15, "out_of_sample_residual_edge")
        show = [c for c in ("wallet", "stratum", "in_sample_residual_edge",
                            "out_of_sample_residual_edge", "out_of_sample_cluster_p",
                            "out_of_sample_markets", "prior_arm") if c in top.columns]
        if len(top):
            print("\n[rw_validate] certified wallets (top 15 by held-out skill edge):")
            print(top[show].to_string(index=False))
        else:
            print("\n[rw_validate] NO wallets cleared the Project 1 gate on this population.")
    return validated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unrestricted", action="store_true",
                    help="do NOT drop micro-crypto bets (reported beside the "
                         "real-world headline, never as it)")
    args = ap.parse_args()
    run(real_world=not args.unrestricted)


if __name__ == "__main__":
    main()
