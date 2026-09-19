"""Export the forensics artifacts to one compact JSON for the dashboard.

Columnar (parallel arrays, not row objects) and aggressively rounded, so 1,600+
wallets with their band / category / year breakdowns fit in a self-contained page.
Read-only. Writes data/processed/forensics_dashboard.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data" / "interim" / "forensics"
OUT = ROOT / "data" / "processed" / "forensics_dashboard.json"

MIN_BETS, MIN_EVENTS = 100, 30
BAND_ORDER = [f"{a}-{a+10}c" for a in range(0, 100, 10)]


def r(x, nd=3):
    """Round, mapping NaN/inf to None so JSON stays valid."""
    if x is None:
        return None
    v = float(x)
    return None if not np.isfinite(v) else round(v, nd)


def main() -> None:
    prof = pd.read_parquet(F / "wallet_profile.parquet")
    bands = pd.read_parquet(F / "wallet_bands.parquet")
    cats = pd.read_parquet(F / "wallet_categories.parquet")
    years = pd.read_parquet(F / "wallet_years.parquet")
    pop = json.loads((F / "population.json").read_text())
    checks = json.loads((F / "checks.json").read_text()) if (F / "checks.json").exists() else {}

    # event-disjoint halves (scripts/forensics_persistence.py). The raw halves in
    # wallet_profile share events across the split, which manufactures agreement;
    # every persistence figure on the dashboard must use the disjoint ones.
    pers = pd.read_parquet(F / "persistence.parquet")
    prof = prof.merge(pers[["wallet", "h1_disjoint", "h2_disjoint", "n1_disjoint",
                            "n2_disjoint", "n_shared_events"]], on="wallet", how="left")

    k = prof[(prof.n_bets >= MIN_BETS) & (prof.n_events >= MIN_EVENTS)].copy()
    k = k.sort_values("resid_event_c", ascending=False).reset_index(drop=True)
    keep = set(k.wallet)
    print(f"wallets exported: {len(k):,}")

    cols = ["n_bets", "n_markets", "n_events", "eff_events", "n_categories",
            "mean_price", "notional", "raw_edge_c", "resid_bet_c", "resid_event_c",
            "resid_event_se", "z_event", "median_bet_c", "trimmed_bet_c",
            "skew_bet", "top1pct_pos_share", "drop_best_event_c",
            "frac_events_positive", "conviction_c", "h1_resid_c", "h2_resid_c",
            "h1_events", "h2_events", "t_event",
            "h1_disjoint", "h2_disjoint", "n1_disjoint", "n2_disjoint",
            "n_shared_events"]
    nd = {"n_bets": 0, "n_markets": 0, "n_events": 0, "n_categories": 0,
          "eff_events": 1, "notional": 0, "h1_events": 0, "h2_events": 0,
          "n1_disjoint": 0, "n2_disjoint": 0, "n_shared_events": 0}
    wallets = {"wallet": k.wallet.tolist()}
    for c in cols:
        wallets[c] = [r(v, nd.get(c, 3)) for v in k[c]]

    def pack(df, keycol, order=None):
        """{wallet: [[key, n_bets, n_events, resid_event_c, se, mean_price], ...]}"""
        d = df[df.wallet.isin(keep)].copy()
        out: dict[str, list] = {}
        for w, g in d.groupby("wallet", observed=True):
            if order is not None:
                g = g.set_index(keycol).reindex(
                    [o for o in order if o in set(g[keycol])]).reset_index()
            else:
                g = g.sort_values("n_bets", ascending=False)
            g = g.dropna(subset=["n_bets"])
            out[w] = [[str(a), int(b), int(c), r(d, 2), r(e, 2), r(f, 3)]
                      for a, b, c, d, e, f in zip(
                          g[keycol], g["n_bets"], g["n_events"],
                          g["resid_event_c"], g["resid_event_se"], g["mean_price"])]
        return out

    band_order = [b for b in bands.band.unique()]
    band_order = sorted(band_order, key=lambda s: int(s.split("-")[0]))
    payload = {
        "generated_for": "per-wallet forensics",
        "population": pop,
        "checks": checks,
        "min_bets": MIN_BETS, "min_events": MIN_EVENTS,
        "band_order": band_order,
        "wallets": wallets,
        "bands": pack(bands, "band", band_order),
        "categories": pack(cats, "category"),
        "years": pack(years, "year"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} — {OUT.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
