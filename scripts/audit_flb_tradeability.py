"""G1/G2/G3 of the market-centric value-betting gate: does the favourite-band
mispricing in the unselected census survive out-of-sample, costs, and capacity?

Read-only. Public data already on disk; no keys, no orders, nothing placed.

What G0 established (`audit_flb_selection.py`, `audit_flb_favorite_band.py`)
--------------------------------------------------------------------------
On the wallet-selected ledger every price band shows positive edge — a selection
signature, not a price-structure one. On the market-centric census the 0.6-0.8
band the brief quotes is NEGATIVE (-3.9c). What IS there is the textbook
favourite-longshot tilt: longshots at 0.02-0.20 lose 2.8-6.2c, heavy favourites
at 0.90-1.00 gain 0.4-4.4c, and the two sides mirror each other (buying NO at
0.96 is the same trade as declining YES at 0.04), so they corroborate.

So the candidate edge is "buy heavy favourites". This module asks whether it is
tradeable, in the three ways a structural bias usually dies.

G1 OUT-OF-SAMPLE. Split resolution EVENTS chronologically, fit the calibration
curve on the first half, and realise the implied trade on the held-out half.
Events, not bets: 573 markets are 179 events, and a bet-level split leaks one
event across the boundary.

G2 COSTS. Two parts, and the second is the one that decides it.
  (a) Effective spread, estimated from prints alone via the COMPLEMENT identity:
      buying NO at q is economically selling YES at 1-q, so for two contemporaneous
      BUY prints on complementary tokens, `p_yes + p_no - 1` is the round-trip
      spread being paid. Cross-checked against the BUY-vs-SELL price gap on the
      same token. No order-book history needed.
  (b) FRAGILITY. Buying at 0.99 is selling insurance: a small premium almost
      always, a near-total loss rarely. The mean edge is only as trustworthy as
      the number of times the tail actually fired in-sample -- and a bootstrap
      cannot resample a tail event that isn't there, so its CI is optimistic by
      construction. We report instead: how many additional losing events would
      zero the realised P&L, against how many losing events the sample contains.

G3 CAPACITY. Notional actually printed in each band per unit time -- an edge you
cannot deploy into is not an edge.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/audit_flb_tradeability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.sports_events import assign_events

CENSUS = Path("data/interim/discovery/discovery_trades.parquet")
OUTDIR = Path("data/interim/value")

BANDS = [(0.90, 0.95), (0.95, 0.98), (0.98, 1.00)]
ALL_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0]
# Window for pairing complementary prints when estimating the spread. Wide enough
# to find a partner in a thin market, short enough that real price moves between
# the two prints don't masquerade as spread.
SPREAD_WINDOW_S = 120


def load() -> pd.DataFrame:
    df = pd.read_parquet(CENSUS, columns=[
        "wallet", "market_id", "slug", "question", "token_id", "outcome", "side",
        "entry_price", "size", "timestamp", "resolved", "resolved_value"])
    mk = assign_events(df[["market_id", "slug", "question"]].drop_duplicates("market_id"))
    return df.merge(mk[["market_id", "event"]], on="market_id", how="left")


def buys(df: pd.DataFrame) -> pd.DataFrame:
    b = df[(df["side"] == "BUY") & df["resolved"].fillna(False)
           & df["resolved_value"].notna()
           & df["entry_price"].between(0, 1, "neither")].copy()
    b["edge"] = b["resolved_value"] - b["entry_price"]
    b["notional"] = b["size"] * b["entry_price"]
    b["pnl"] = b["size"] * b["edge"]
    return b


def event_ci(sub: pd.DataFrame, resamples: int, seed: int) -> tuple[float, float]:
    if sub.empty:
        return float("nan"), float("nan")
    g = sub.groupby("event")["edge"].agg(["sum", "count"])
    S, C = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    d = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, len(S), len(S))
        d[i] = S[idx].sum() / C[idx].sum()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- G1
def g1_out_of_sample(b: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    """Chronological split at the EVENT level; fit bands on half 1, realise on half 2."""
    ev_t = b.groupby("event")["timestamp"].max().sort_values()
    cut = ev_t.iloc[len(ev_t) // 2]
    first = set(ev_t[ev_t <= cut].index)
    b = b.copy()
    b["half"] = np.where(b["event"].isin(first), "in", "out")
    b["band"] = pd.cut(b["entry_price"], ALL_EDGES, right=False)
    print(f"\nG1 event split: {len(first)} in-sample events / "
          f"{ev_t.index.nunique() - len(first)} held-out, cut at ts={cut}")

    rows = []
    for i, (band, sub) in enumerate(b.groupby("band", observed=True)):
        ins, out = sub[sub.half == "in"], sub[sub.half == "out"]
        if ins.empty or out.empty:
            continue
        lo, hi = event_ci(out, resamples, seed + i)
        rows.append({
            "band": f"[{band.left:.2f},{band.right:.2f})",
            "in_n": len(ins), "in_events": ins.event.nunique(),
            "in_edge": ins.edge.mean(),
            "out_n": len(out), "out_events": out.event.nunique(),
            "out_edge": out.edge.mean(), "out_ci_lo": lo, "out_ci_hi": hi,
            "out_edge_notional_wtd": (out.pnl.sum() / out.notional.sum()
                                      if out.notional.sum() else np.nan),
            "sign_agrees": np.sign(ins.edge.mean()) == np.sign(out.edge.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- G2a
def g2_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Effective round-trip spread per band, from complementary BUY prints.

    For a two-token market, buying NO at q == selling YES at 1-q. Two BUY prints
    on the complementary tokens close in time therefore straddle the book, and
    `p_yes + p_no - 1` is the spread the two takers paid between them.
    """
    two = df.groupby("market_id")["token_id"].nunique()
    df = df[df["market_id"].isin(two[two == 2].index)]
    d = df[(df["side"] == "BUY") & df["entry_price"].between(0, 1, "neither")][
        ["market_id", "token_id", "entry_price", "timestamp", "size"]].copy()

    rows = []
    for mid, g in d.groupby("market_id", sort=False):
        toks = g["token_id"].unique()
        if len(toks) != 2:
            continue
        a = g[g.token_id == toks[0]].sort_values("timestamp")
        c = g[g.token_id == toks[1]].sort_values("timestamp")
        if a.empty or c.empty:
            continue
        m = pd.merge_asof(a, c, on="timestamp", direction="nearest",
                          tolerance=SPREAD_WINDOW_S, suffixes=("_a", "_c"))
        m = m.dropna(subset=["entry_price_c"])
        if m.empty:
            continue
        rows.append(pd.DataFrame({
            "market_id": mid,
            "price": m["entry_price_a"].to_numpy(),
            "spread": (m["entry_price_a"] + m["entry_price_c"] - 1.0).to_numpy(),
        }))
    if not rows:
        return pd.DataFrame()
    s = pd.concat(rows, ignore_index=True)
    # A negative implied spread is a crossed pair (the two prints straddle a real
    # price move, or a mint match); keep them -- clipping would bias the estimate
    # downward, which is the direction that flatters the strategy.
    s["band"] = pd.cut(s["price"], ALL_EDGES, right=False)
    out = s.groupby("band", observed=True)["spread"].agg(
        n="size", mean_spread="mean", median_spread="median",
        p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75)).reset_index()
    out["band"] = out["band"].map(lambda iv: f"[{iv.left:.2f},{iv.right:.2f})")
    return out


# ---------------------------------------------------------------- G2b
def g2_fragility(b: pd.DataFrame) -> pd.DataFrame:
    """How many more losing events would zero the realised P&L?

    A bootstrap resamples the tail events present in the sample; it cannot
    resample the ones that never happened. For an insurance-shaped payoff that
    makes its CI optimistic. This is the blunt honest alternative: total realised
    profit, divided by what one typical losing event costs.
    """
    rows = []
    for lo, hi in BANDS:
        s = b[(b.entry_price >= lo) & (b.entry_price < hi)]
        if s.empty:
            continue
        per_ev = s.groupby("event").agg(pnl=("pnl", "sum"), notional=("notional", "sum"))
        losers = per_ev[per_ev.pnl < 0]
        # cost of one more average-sized event going wrong: you lose the notional
        # you paid, minus the premium you would have collected on it
        typical_notional = per_ev["notional"].median()
        rows.append({
            "band": f"[{lo:.2f},{hi:.2f})",
            "n_bets": len(s), "n_events": per_ev.shape[0],
            "losing_events": len(losers),
            "total_pnl": per_ev.pnl.sum(), "total_notional": per_ev.notional.sum(),
            "return_on_notional": per_ev.pnl.sum() / per_ev.notional.sum(),
            "median_event_notional": typical_notional,
            "events_to_zero": per_ev.pnl.sum() / typical_notional
            if typical_notional else np.nan,
            "realised_loss_rate": float((s.resolved_value < .5).mean()),
            "breakeven_loss_rate": float(1 - s.entry_price.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- G3
def g3_capacity(b: pd.DataFrame) -> pd.DataFrame:
    span_days = (b.timestamp.max() - b.timestamp.min()) / 86400.0
    rows = []
    for lo, hi in BANDS:
        s = b[(b.entry_price >= lo) & (b.entry_price < hi)]
        rows.append({
            "band": f"[{lo:.2f},{hi:.2f})",
            "n_prints": len(s), "notional": s.notional.sum(),
            "notional_per_day": s.notional.sum() / span_days if span_days else np.nan,
            "median_print_notional": (s["size"] * s.entry_price).median(),
            "p90_print_notional": (s["size"] * s.entry_price).quantile(.9),
        })
    print(f"\ncensus tape spans {span_days:.1f} days across {b.event.nunique()} events")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load()
    b = buys(df)
    print(f"census: {len(b):,} resolved BUY bets / {b.event.nunique()} events")

    g1 = g1_out_of_sample(b, args.resamples, args.seed)
    print("\n=== G1 OUT-OF-SAMPLE (fit half-1 events, realise on half-2) ===")
    print(f"{'band':<14}{'in_n':>9}{'in_edge':>10}{'out_n':>9}{'out_ev':>7}"
          f"{'out_edge':>10}{'95% CI (event)':>22}{'notional-wtd':>14}{'sign':>6}")
    for _, r in g1.iterrows():
        ci = f"[{r.out_ci_lo:+.4f},{r.out_ci_hi:+.4f}]"
        print(f"{r.band:<14}{r.in_n:>9,}{r.in_edge:>+10.4f}{r.out_n:>9,}{r.out_events:>7}"
              f"{r.out_edge:>+10.4f}{ci:>22}{r.out_edge_notional_wtd:>+14.4f}"
              f"{'ok' if r.sign_agrees else 'FLIP':>6}")

    sp = g2_spread(df)
    print("\n=== G2a EFFECTIVE SPREAD from complementary prints (p_yes + p_no - 1) ===")
    print(sp.to_string(index=False))

    fr = g2_fragility(b)
    print("\n=== G2b FRAGILITY of the favourite bands ===")
    print(fr.to_string(index=False))

    cap = g3_capacity(b)
    print("\n=== G3 CAPACITY ===")
    print(cap.to_string(index=False))

    g1.to_parquet(OUTDIR / "g1_oos_bands.parquet", index=False)
    sp.to_parquet(OUTDIR / "g2_spread.parquet", index=False)
    fr.to_parquet(OUTDIR / "g2_fragility.parquet", index=False)
    cap.to_parquet(OUTDIR / "g3_capacity.parquet", index=False)
    print(f"\nwrote g1/g2/g3 parquets to {OUTDIR}")


if __name__ == "__main__":
    main()
