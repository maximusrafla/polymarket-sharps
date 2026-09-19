"""Project 3 step 6c — the "would a copier get this by trading the niche?" audit.

THE QUESTION
------------
Step 5c found 52 slow-market wallets whose held-out skill edge persists. Their
residual is measured against E[outcome | price, category], and 59% of slow bets
are `category == "other"`. So a sub-niche inside `other` that is simply mispriced
hands a positive residual to EVERY buyer in it. That would survive disjoint data,
the cluster null, the concentration guards and the horizon check — and it is not
followable wallet alpha, because a copier would collect it by trading the niche
directly rather than by following anyone.

THE TEST
--------
For each survivor, decompose its edge against a common parent (category)
baseline into a niche part and a wallet part:

    wallet_vs_parent  ~=  niche_population_residual  +  wallet_vs_niche

  niche_population_residual — the mean residual of ALL participants in that
      niche. This is exactly "what a copier gets by trading the niche blind".
  wallet_vs_niche — the survivor's residual against its own niche's fair rate,
      leave-one-wallet-out. This is what is left over that is actually the wallet.

If a survivor's edge is ~entirely the first term, its edge is niche-level and not
copyable skill. If it beats its own niche, that is genuine.

THE UNBIASED READ
-----------------
The niche population average is computed TWICE:

  - on the DISCOVERY corpus (market-first over 542k wallets) — an unselected
    population sample, and the number that should be believed;
  - on the DEEP tape (1,295 screen-selected wallets) — reported for contrast.

The deep tape's wallets were screened FOR positive residuals, so its niche
averages are biased UP, which biases the decomposition toward "it's all niche".
Using it alone would manufacture a false collapse. Where discovery covers a
niche, discovery is the number reported.

READ-ONLY: reads the interim deep/discovery caches, writes nothing but stdout
and (optionally) one interim parquet. Never touches data/processed/.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/agent47/polymarket-sharps")

from src.common import INTERIM_DIR, atomic_to_parquet, load_config  # noqa: E402
from src.slow_baseline import fit_hierarchical_residuals  # noqa: E402
from src.slow_market import fit_slow_category_baselines, residualize  # noqa: E402
from src.slow_niche import assign_niches  # noqa: E402
from src.validate import split_in_sample_out_of_sample  # noqa: E402

OUT_PATH = INTERIM_DIR / "slow_deepening" / "niche_decomposition.parquet"
DISCOVERY_TRADES = INTERIM_DIR / "discovery" / "discovery_trades.parquet"


# ---------------------------------------------------------------------------
# The unbiased population view of each niche
# ---------------------------------------------------------------------------

def discovery_niche_residuals(min_bets: int = 200) -> pd.DataFrame:
    """Mean residual per niche over the DISCOVERY corpus, against a category
    baseline fit on that same unselected corpus.

    A large positive value means: everybody buying in this niche beat the
    category curve, i.e. the niche is mispriced relative to its parent and the
    edge is available to anyone — no wallet-following required."""
    cols = ["wallet", "market_id", "side", "entry_price", "resolved",
            "resolved_value", "category", "slug", "question"]
    df = pd.read_parquet(DISCOVERY_TRADES, columns=cols)
    df = df[(df["side"] == "BUY") & df["resolved"].fillna(False)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    df = assign_niches(df)
    # Parent = category curve fit on the unbiased corpus; no LOWO (we WANT the
    # whole population, including every participant, in this number).
    scored, _ = fit_hierarchical_residuals(df, levels=("category",), lowo=False)
    for lvl in ("niche_l1", "niche_l2"):
        scored[lvl] = df[lvl].to_numpy()

    out = []
    for lvl in ("niche_l1", "niche_l2"):
        g = (scored.groupby(lvl)
             .agg(pop_resid=("residual_skill", "mean"),
                  pop_bets=("residual_skill", "size"),
                  pop_wallets=("wallet", "nunique"),
                  pop_markets=("market_id", "nunique"))
             .reset_index().rename(columns={lvl: "niche"}))
        g["level"] = lvl
        out.append(g[g["pop_bets"] >= min_bets])
    return pd.concat(out, ignore_index=True)


def deep_niche_residuals(deep_scored: pd.DataFrame, min_bets: int = 200) -> pd.DataFrame:
    """Same statistic over the deep tape — biased UP by the screen, reported only
    for contrast with the discovery number."""
    out = []
    for lvl in ("niche_l1", "niche_l2"):
        g = (deep_scored.groupby(lvl)
             .agg(deep_resid=("residual_skill", "mean"),
                  deep_bets=("residual_skill", "size"),
                  deep_wallets=("wallet", "nunique"))
             .reset_index().rename(columns={lvl: "niche"}))
        g["level"] = lvl
        out.append(g[g["deep_bets"] >= min_bets])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-survivor decomposition
# ---------------------------------------------------------------------------

def decompose(bets: pd.DataFrame, survivors: set[str], level: str = "niche_l1",
              cfg: dict | None = None, top_niches: int = 3) -> pd.DataFrame:
    """Per (survivor, niche) decomposition over each wallet's HELD-OUT half —
    the same half 5c certified on, so the numbers are comparable to its verdict."""
    cfg = cfg or load_config()
    oos_split = float(cfg.get("scoring", {}).get("oos_split", 0.5))

    coarse = residualize(bets, fit_slow_category_baselines(bets))["residual_skill"].to_numpy()
    fine, _ = fit_hierarchical_residuals(
        bets, levels=("category", "niche_l1", "niche_l2"), lowo=True)
    work = bets.copy()
    work["resid_parent"] = coarse
    work["resid_niche"] = fine["residual_skill"].to_numpy()

    rows = []
    for w, g in work[work["wallet"].isin(survivors)].groupby("wallet", sort=False):
        _, oos = split_in_sample_out_of_sample(g, oos_split)
        if oos.empty:
            continue
        n_all = len(oos)
        for niche, gn in oos.groupby(level):
            rows.append({
                "wallet": w, "level": level, "niche": niche,
                "wallet_bets": len(gn), "wallet_share": len(gn) / n_all,
                "wallet_vs_parent": float(gn["resid_parent"].mean()),
                "wallet_vs_niche": float(gn["resid_niche"].mean()),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values(["wallet", "wallet_bets"], ascending=[True, False])
            .groupby("wallet").head(top_niches).reset_index(drop=True))


def summarise(dec: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted per-survivor rollup with the niche-explained fraction.
    `dec` must already carry `pop_resid` (merged by the caller)."""
    d = dec
    rows = []
    for w, g in d.groupby("wallet"):
        wt = g["wallet_bets"].to_numpy(dtype=float)
        wt = wt / wt.sum()
        vs_parent = float(np.sum(wt * g["wallet_vs_parent"]))
        vs_niche = float(np.sum(wt * g["wallet_vs_niche"]))
        pop_r = g["pop_resid"].to_numpy(dtype=float)
        cov = ~np.isnan(pop_r)
        pop_w = float(np.sum(wt[cov] * pop_r[cov]) / wt[cov].sum()) if cov.any() else np.nan
        rows.append({
            "wallet": w,
            "wallet_vs_parent": vs_parent,
            "wallet_vs_niche": vs_niche,
            "niche_pop_resid": pop_w,
            "pop_coverage": float(wt[cov].sum()),
            "niche_explained": (1.0 - vs_niche / vs_parent) if abs(vs_parent) > 1e-9 else np.nan,
            "top_niche": g.iloc[0]["niche"],
            "top_niche_share": float(g.iloc[0]["wallet_share"]),
        })
    return pd.DataFrame(rows).sort_values("wallet_vs_parent", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", default="niche_l1", choices=["niche_l1", "niche_l2"])
    ap.add_argument("--top-niches", type=int, default=3)
    ap.add_argument("--min-pop-bets", type=int, default=200)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--validated", default=None,
                    help="path to the slow_validated parquet whose edge_persisted set "
                         "is audited. Defaults to the 5c reference, since the published "
                         "step-6c numbers are about THAT set; pass the standard file to "
                         "audit the current pipeline output instead.")
    args = ap.parse_args()

    import src.slow_validate as sv

    bets = sv.load_deep_slow_bets_niched()
    print(f"[universe] {len(bets):,} bets, {bets['wallet'].nunique():,} wallets, "
          f"{bets['market_id'].nunique():,} markets")

    # The step-7 standard took over the canonical filename, so the historical 5c
    # set now lives beside it. Auditing whichever file happened to be written last
    # would silently change what the published numbers refer to.
    default_val = sv.SLOW_VALIDATED_PATH.with_name(
        "slow_validated_category_market_id_nocxgates.parquet")
    val_path = args.validated or (default_val if default_val.exists()
                                  else sv.SLOW_VALIDATED_PATH)
    val = pd.read_parquet(val_path)
    survivors = set(val.loc[val["edge_persisted"], "wallet"])
    print(f"[survivors] {len(survivors)} edge_persisted from {val_path.name}")

    print("\n[niche population residuals — DISCOVERY corpus, unselected]")
    disc = discovery_niche_residuals(min_bets=args.min_pop_bets)
    d1 = disc[disc["level"] == args.level].sort_values("pop_resid", ascending=False)
    print(f"  {len(d1)} niches with >= {args.min_pop_bets} discovery bets")
    with pd.option_context("display.width", 200):
        print(d1.head(20).to_string(index=False))
        print("  ... most negative:")
        print(d1.tail(8).to_string(index=False))

    fine, _ = fit_hierarchical_residuals(bets, levels=("category",), lowo=False)
    fine["niche_l1"] = bets["niche_l1"].to_numpy()
    fine["niche_l2"] = bets["niche_l2"].to_numpy()
    deep_pop = deep_niche_residuals(fine, min_bets=args.min_pop_bets)

    pop = disc[disc["level"] == args.level][["level", "niche", "pop_resid",
                                             "pop_bets", "pop_wallets"]]
    dec = decompose(bets, survivors, level=args.level, top_niches=args.top_niches)
    if dec.empty:
        print("\n[decompose] no survivor held-out bets — nothing to report.")
        return

    dec = dec.merge(pop, on=["level", "niche"], how="left").merge(
        deep_pop[deep_pop["level"] == args.level][["level", "niche", "deep_resid"]],
        on=["level", "niche"], how="left")

    print(f"\n[per-(survivor, niche) decomposition — top {args.top_niches} niches each]")
    with pd.option_context("display.width", 240, "display.max_rows", None):
        show = dec.copy()
        show["wallet"] = show["wallet"].str[:12]
        print(show[["wallet", "niche", "wallet_bets", "wallet_share",
                    "wallet_vs_parent", "wallet_vs_niche", "pop_resid",
                    "deep_resid", "pop_bets"]].round(4).to_string(index=False))

    summ = summarise(dec)
    print("\n[per-survivor rollup] "
          "niche_explained = share of the coarse edge that the niche baseline absorbs")
    with pd.option_context("display.width", 240, "display.max_rows", None):
        s = summ.copy()
        s["wallet"] = s["wallet"].str[:12]
        print(s.round(4).to_string(index=False))

    ok = summ.dropna(subset=["niche_explained"])
    print(f"\n[summary] survivors decomposed: {len(summ)}")
    print(f"  median niche_explained fraction: {ok['niche_explained'].median():+.3f}")
    print(f"  survivors whose edge is >50% niche-explained: "
          f"{int((ok['niche_explained'] > 0.5).sum())} of {len(ok)}")
    print(f"  survivors still beating their own niche by >=10c: "
          f"{int((summ['wallet_vs_niche'] >= 0.10).sum())} of {len(summ)}")
    cov = summ["pop_coverage"].mean()
    print(f"  mean discovery coverage of survivor niche volume: {cov:.1%}")

    if args.save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(dec, OUT_PATH, compression="gzip")
        print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
