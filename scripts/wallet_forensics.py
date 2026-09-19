"""Per-wallet forensics — the distributional view the repo has never had.

WHY THIS EXISTS (owner, 2026-07-30): every conclusion in this project is a MEAN
in cents with a CI. A mean of +2c can be 500 quiet +2c bets, or 499 flat bets and
one 10c longshot that hit, or strongly positive in one price band and negative in
another. Those are different animals and no artifact in this repo can tell them
apart. This script computes, per wallet, the SHAPE of the edge rather than its
average, and writes it to disk for a dashboard.

DESIGN NOTES (fixed before results were read):

  Universe   All resolved real-world BUY bets of the deepened performance-blind
             sample (micro_crypto excluded). NO time-to-close filter: the funnel's
             own tape-derived filter conditions on post-entry activity and is
             forward-looking (docs/funnel_conditional_2026-07-30.md). This is a
             DESCRIPTIVE artifact, so the widest honest universe is the right one.
  Baseline   E[resolved_value | entry_price] over 20 quantile bins fit on the whole
             universe, i.e. src.features.fit_price_baseline semantics. Per-bet
             residual = rv - E[rv|price]; reported in cents/share.
  Unit       The resolution event (src.sports_events.resolution_event), else the
             market. Every CI is event-level: SE = sd(event means)/sqrt(n_events),
             which is the cluster-robust SE by construction. Kish effective events
             is reported beside every wallet so thin records read as thin.
  Novelties  Four statistics this repo has never computed, all shape rather than level:
             1. STANDARDIZED residual z = (rv - E)/sqrt(E(1-E)). Mean-in-cents
                treats a 5c bet and a 95c bet as equally informative; they have
                wildly different Bernoulli variance. z is the precision-weighted
                version, and ranking on it is a genuinely different ranking.
             2. JACKKNIFE robustness: the wallet's edge recomputed with its single
                best event removed. Directly answers "is this one crazy win".
             3. BAND PROFILE: residual per wallet x 10c price band, so a wallet
                that is sharp on longshots and bad on favorites is visible instead
                of averaged away.
             4. CONVICTION: notional-weighted residual minus equal-weighted. Does
                the wallet bet BIGGER when it is right? `size` is on the shards and
                no analysis in this repo has ever used it.
  Null       Cluster-preserving: permute wallet labels across (wallet, event) cells,
             holding each wallet's event count fixed. Destroys wallet identity,
             preserves the market clustering that makes bet-level nulls too
             permissive (docs/... cluster-preserving-null). 200 reps.

Read-only against every shared artifact. Writes only:
  data/interim/forensics/wallet_profile.parquet     one row per wallet
  data/interim/forensics/wallet_bands.parquet       wallet x price band
  data/interim/forensics/wallet_categories.parquet  wallet x category
  data/interim/forensics/wallet_years.parquet       wallet x calendar year
  data/interim/forensics/population.json            baseline curve + null + totals
Run:  flock data/interim/.analysis.lock -c \
        'PYTHONPATH=. .venv/bin/python scripts/wallet_forensics.py'
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sports_events import resolution_event

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "data" / "interim" / "realworld"
OUT = ROOT / "data" / "interim" / "forensics"
SHARDS = sorted(glob.glob(str(RW / "deep_trades" / "*.parquet")))

SEED = 20260730
N_BINS = 20               # quantile bins for E[outcome|price], as features.py
N_NULL = 200              # cluster-preserving permutations
PRICE_BANDS = np.arange(0.0, 1.01, 0.10)
BAND_LABELS = [f"{int(a*100)}-{int(b*100)}c"
               for a, b in zip(PRICE_BANDS[:-1], PRICE_BANDS[1:])]
MIN_BETS = 30             # a wallet below this cannot support any shape claim


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data build
def build_universe() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Stream the deep shards into one compact integer-coded frame. Strings never
    accumulate: wallets/markets are dict-coded per shard and market attributes are
    joined by code at the end. Carries `size`, which the funnel builder drops."""
    cat = pd.read_parquet(RW / "market_category.parquet")
    cat_of = dict(zip(cat["market_id"], cat["category"]))

    cols = ["wallet", "market_id", "side", "entry_price", "size", "timestamp",
            "resolved", "resolved_value", "slug"]
    wcode: dict[str, int] = {}
    mcode: dict[str, int] = {}
    slug_by_mcode: dict[int, str] = {}
    parts = []
    for i, p in enumerate(SHARDS):
        df = pd.read_parquet(p, columns=cols)
        df = df[(df["side"] == "BUY") & df["resolved"].fillna(False)
                & df["resolved_value"].notna()]
        if not len(df):
            continue
        w = np.fromiter((wcode.setdefault(x, len(wcode))
                         for x in df["wallet"].to_numpy()),
                        dtype=np.int32, count=len(df))
        m = np.fromiter((mcode.setdefault(x, len(mcode))
                         for x in df["market_id"].to_numpy()),
                        dtype=np.int32, count=len(df))
        for mc, s in zip(m, df["slug"].to_numpy()):
            if mc not in slug_by_mcode and isinstance(s, str):
                slug_by_mcode[int(mc)] = s
        parts.append(pd.DataFrame({
            "w": w, "m": m,
            "price": df["entry_price"].to_numpy(np.float32),
            "size": df["size"].to_numpy(np.float32),
            "ts": df["timestamp"].to_numpy(np.int64),
            "rv": df["resolved_value"].to_numpy(np.float32),
        }))
        if (i + 1) % 20 == 0:
            log(f"  [shards {i+1}/{len(SHARDS)}] rows so far "
                f"{sum(len(x) for x in parts):,}")
    bets = pd.concat(parts, ignore_index=True)
    del parts
    log(f"  resolved BUY bets, all wallets: {len(bets):,}")

    markets = np.empty(len(mcode), dtype=object)
    for s, c in mcode.items():
        markets[c] = s
    m_cat = np.array([cat_of.get(s, "other") for s in markets], dtype=object)
    bets["category"] = pd.Categorical(m_cat[bets["m"].to_numpy()])
    bets = bets[bets["category"] != "micro_crypto"].copy()
    bets["category"] = bets["category"].cat.remove_unused_categories()
    log(f"  after micro_crypto excluded:    {len(bets):,}")

    # resolution event per surviving market, coded
    ev_key: dict[int, str] = {}
    for c in bets["m"].unique():
        ev_key[int(c)] = resolution_event(slug_by_mcode.get(int(c)),
                                          market_id=markets[c])[0]
    ecode: dict[str, int] = {}
    bets["e"] = np.fromiter(
        (ecode.setdefault(ev_key[int(c)], len(ecode))
         for c in bets["m"].to_numpy()),
        dtype=np.int32, count=len(bets))
    events = np.empty(len(ecode), dtype=object)
    for s, c in ecode.items():
        events[c] = s
    wallets = np.empty(len(wcode), dtype=object)
    for s, c in wcode.items():
        wallets[c] = s
    log(f"  markets {len(mcode):,} | resolution events {len(ecode):,} | "
        f"wallets {len(wcode):,}")
    return bets.reset_index(drop=True), wallets, events


# ---------------------------------------------------------------- baseline
def fit_baseline(price: np.ndarray, rv: np.ndarray, n_bins: int = N_BINS):
    """E[rv | price] over equal-count quantile bins — features.fit_price_baseline
    semantics, reimplemented on raw arrays so the 3.4M-row frame is never copied."""
    edges = np.unique(np.quantile(price, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, price, side="right") - 1, 0, edges.size - 2)
    sums = np.bincount(idx, weights=rv, minlength=edges.size - 1)
    cnts = np.bincount(idx, minlength=edges.size - 1)
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), rv.mean())
    return edges, means, cnts, idx


# ---------------------------------------------------------------- aggregation
def group_stats(keys: np.ndarray, vals: np.ndarray, n: int):
    """Fast per-group (sum, count, sumsq) for integer keys in [0, n)."""
    cnt = np.bincount(keys, minlength=n).astype(np.float64)
    s = np.bincount(keys, weights=vals, minlength=n)
    ss = np.bincount(keys, weights=vals * vals, minlength=n)
    return s, cnt, ss


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    log("[1/6] building universe from deep shards")
    bets, wallets, events = build_universe()

    log("[2/6] fitting E[outcome|price] and per-bet residuals")
    price = bets["price"].to_numpy(np.float64)
    rv = bets["rv"].to_numpy(np.float64)
    edges, bmeans, bcnts, bidx = fit_baseline(price, rv)
    exp = bmeans[bidx]
    resid_c = 100.0 * (rv - exp)                    # skill edge, cents/share
    raw_c = 100.0 * (rv - price)                    # raw edge, cents/share
    # precision-weighted: Bernoulli SD at the market's own expectation. Clipped so
    # a degenerate bin (exp ~ 0 or 1) cannot manufacture an unbounded z.
    sd = np.sqrt(np.clip(exp * (1.0 - exp), 1e-3, None))
    z = (rv - exp) / sd
    bets["resid_c"] = resid_c.astype(np.float32)
    bets["raw_c"] = raw_c.astype(np.float32)
    bets["z"] = z.astype(np.float32)
    log(f"  baseline bins {len(bmeans)} | pooled raw edge {raw_c.mean():+.3f}c "
        f"| pooled residual {resid_c.mean():+.4f}c (must be ~0 by construction)")

    w = bets["w"].to_numpy()
    e = bets["e"].to_numpy()
    nW, nE = len(wallets), len(events)

    log("[3/6] per (wallet, event) cells — the clustering unit")
    # cell id = dense code over observed (w, e) pairs
    cell_key = w.astype(np.int64) * np.int64(nE) + e
    uniq, cell = np.unique(cell_key, return_inverse=True)
    nC = len(uniq)
    cell_w = (uniq // nE).astype(np.int32)
    c_sum, c_cnt, _ = group_stats(cell, resid_c, nC)
    c_mean = c_sum / c_cnt                          # within-event mean residual
    cz_sum, _, _ = group_stats(cell, z, nC)
    cz_mean = cz_sum / c_cnt
    log(f"  (wallet, event) cells: {nC:,}")

    log("[4/6] per-wallet profile")
    # --- event-weighted mean + cluster-robust SE, vectorised over all wallets
    ev_n = np.bincount(cell_w, minlength=nW).astype(np.float64)
    ev_s = np.bincount(cell_w, weights=c_mean, minlength=nW)
    ev_ss = np.bincount(cell_w, weights=c_mean ** 2, minlength=nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        ew_mean = ev_s / ev_n
        ev_var = np.maximum(ev_ss / ev_n - ew_mean ** 2, 0.0) * ev_n / np.maximum(ev_n - 1, 1)
        ew_se = np.sqrt(ev_var / np.maximum(ev_n, 1))
    ez_s = np.bincount(cell_w, weights=cz_mean, minlength=nW)
    ez_ss = np.bincount(cell_w, weights=cz_mean ** 2, minlength=nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        ez_mean = ez_s / ev_n
        ez_var = np.maximum(ez_ss / ev_n - ez_mean ** 2, 0.0) * ev_n / np.maximum(ev_n - 1, 1)
        ez_se = np.sqrt(ez_var / np.maximum(ev_n, 1))

    # --- Kish effective events, and concentration of the record
    b_per_cell = c_cnt
    kish_num = np.bincount(cell_w, weights=b_per_cell, minlength=nW) ** 2
    kish_den = np.bincount(cell_w, weights=b_per_cell ** 2, minlength=nW)
    eff_events = np.where(kish_den > 0, kish_num / np.maximum(kish_den, 1e-9), 0.0)

    # --- jackknife: drop the single best event (by its contribution to the
    #     event-weighted mean). "Is this one crazy win?"
    order = np.lexsort((c_mean, cell_w))            # per wallet, ascending c_mean
    sorted_w = cell_w[order]
    sorted_m = c_mean[order]
    last_of_w = np.r_[np.where(np.diff(sorted_w) != 0)[0], len(sorted_w) - 1]
    first_of_w = np.r_[0, last_of_w[:-1] + 1]
    best_mean = np.full(nW, np.nan)
    worst_mean = np.full(nW, np.nan)
    best_mean[sorted_w[last_of_w]] = sorted_m[last_of_w]
    worst_mean[sorted_w[first_of_w]] = sorted_m[first_of_w]
    with np.errstate(invalid="ignore", divide="ignore"):
        drop_best = (ev_s - np.nan_to_num(best_mean)) / np.maximum(ev_n - 1, 1)
        drop_worst = (ev_s - np.nan_to_num(worst_mean)) / np.maximum(ev_n - 1, 1)
    drop_best = np.where(ev_n >= 2, drop_best, np.nan)
    drop_worst = np.where(ev_n >= 2, drop_worst, np.nan)

    # --- sign breadth: fraction of the wallet's events with positive mean residual
    frac_ev_pos = np.bincount(cell_w, weights=(c_mean > 0).astype(float),
                              minlength=nW) / np.maximum(ev_n, 1)

    # --- bet-level shape
    b_s, b_n, b_ss = group_stats(w, resid_c, nW)
    bet_mean = b_s / np.maximum(b_n, 1)
    raw_s, _, _ = group_stats(w, raw_c, nW)
    price_s, _, _ = group_stats(w, price, nW)
    size_arr = np.nan_to_num(bets["size"].to_numpy(np.float64), nan=0.0)
    notional = size_arr * price
    not_s, _, _ = group_stats(w, notional, nW)
    # conviction: notional-weighted residual minus equal-weighted residual
    nw_s, _, _ = group_stats(w, notional * resid_c, nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        conviction = nw_s / np.maximum(not_s, 1e-9) - bet_mean
    conviction = np.where(not_s > 0, conviction, np.nan)

    # --- robust location: per-wallet median and 10% trimmed mean of bet residuals
    med = np.full(nW, np.nan)
    trim = np.full(nW, np.nan)
    skewv = np.full(nW, np.nan)
    top1pct_share = np.full(nW, np.nan)
    ordw = np.argsort(w, kind="stable")
    ws, rs = w[ordw], resid_c[ordw]
    bnd = np.r_[0, np.where(np.diff(ws) != 0)[0] + 1, len(ws)]
    for a, b in zip(bnd[:-1], bnd[1:]):
        wi = ws[a]
        v = np.sort(rs[a:b])
        n = len(v)
        med[wi] = np.median(v)
        k = int(n * 0.10)
        trim[wi] = v[k:n - k].mean() if n - 2 * k > 0 else v.mean()
        sd_ = v.std()
        skewv[wi] = (((v - v.mean()) ** 3).mean() / sd_ ** 3) if sd_ > 1e-9 else np.nan
        # share of total positive residual coming from the top 1% of bets
        pos = v[v > 0]
        if pos.size:
            ktop = max(1, int(np.ceil(n * 0.01)))
            top1pct_share[wi] = pos[-min(ktop, pos.size):].sum() / pos.sum()

    # --- split-half persistence (chronological, per wallet)
    ordt = np.lexsort((bets["ts"].to_numpy(), w))
    ws2 = w[ordt]
    rs2 = resid_c[ordt]
    es2 = e[ordt]
    b2 = np.r_[0, np.where(np.diff(ws2) != 0)[0] + 1, len(ws2)]
    h1 = np.full(nW, np.nan)
    h2 = np.full(nW, np.nan)
    h1_ev = np.zeros(nW)
    h2_ev = np.zeros(nW)
    for a, b in zip(b2[:-1], b2[1:]):
        wi = ws2[a]
        n = b - a
        k = int(np.ceil(n / 2))
        for half, sl, arr, evc in ((0, slice(a, a + k), h1, h1_ev),
                                   (1, slice(a + k, b), h2, h2_ev)):
            r, ev = rs2[sl], es2[sl]
            if r.size == 0:
                continue
            u, inv = np.unique(ev, return_inverse=True)
            m_ = np.bincount(inv, weights=r) / np.bincount(inv)
            arr[wi] = m_.mean()
            evc[wi] = len(u)

    n_markets = pd.Series(bets["m"].to_numpy()).groupby(w).nunique() \
        .reindex(range(nW)).to_numpy()
    n_cats = bets.groupby("w", observed=True)["category"].nunique() \
        .reindex(range(nW)).to_numpy()

    prof = pd.DataFrame({
        "wallet": wallets,
        "n_bets": b_n.astype(int),
        "n_markets": np.nan_to_num(n_markets).astype(int),
        "n_events": ev_n.astype(int),
        "eff_events": eff_events,
        "n_categories": np.nan_to_num(n_cats).astype(int),
        "mean_price": price_s / np.maximum(b_n, 1),
        "notional": not_s,
        "raw_edge_c": raw_s / np.maximum(b_n, 1),
        "resid_bet_c": bet_mean,
        "resid_event_c": ew_mean,
        "resid_event_se": ew_se,
        "z_event": ez_mean,
        "z_event_se": ez_se,
        "median_bet_c": med,
        "trimmed_bet_c": trim,
        "skew_bet": skewv,
        "top1pct_pos_share": top1pct_share,
        "drop_best_event_c": drop_best,
        "drop_worst_event_c": drop_worst,
        "frac_events_positive": frac_ev_pos,
        "conviction_c": conviction,
        "h1_resid_c": h1, "h2_resid_c": h2,
        "h1_events": h1_ev, "h2_events": h2_ev,
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        prof["t_event"] = prof["resid_event_c"] / prof["resid_event_se"]
        prof["t_z"] = prof["z_event"] / prof["z_event_se"]
        prof["fragility"] = 1.0 - (prof["drop_best_event_c"] /
                                   prof["resid_event_c"].replace(0, np.nan))
    prof.to_parquet(OUT / "wallet_profile.parquet", index=False)
    log(f"  wrote wallet_profile.parquet — {len(prof):,} wallets")

    log("[5/6] band / category / year breakdowns")
    band_idx = np.clip(np.searchsorted(PRICE_BANDS, price, side="right") - 1,
                       0, len(BAND_LABELS) - 1)
    bets["band_i"] = band_idx.astype(np.int8)
    year = pd.to_datetime(bets["ts"], unit="s", utc=True).dt.year.to_numpy()
    bets["year"] = year.astype(np.int16)

    def breakdown(col: str, labels=None) -> pd.DataFrame:
        g = bets.groupby(["w", col], observed=True)
        out = g.agg(n_bets=("resid_c", "size"),
                    resid_bet_c=("resid_c", "mean"),
                    raw_c=("raw_c", "mean"),
                    z=("z", "mean"),
                    mean_price=("price", "mean"),
                    n_events=("e", "nunique")).reset_index()
        # event-weighted mean + cluster-robust SE within each cell
        ge = bets.groupby(["w", col, "e"], observed=True)["resid_c"].mean() \
            .reset_index()
        agg = ge.groupby(["w", col], observed=True)["resid_c"] \
            .agg(["mean", "std", "size"]).reset_index()
        agg = agg.rename(columns={"mean": "resid_event_c", "std": "_sd",
                                  "size": "_ne"})
        agg["resid_event_se"] = agg["_sd"] / np.sqrt(agg["_ne"].clip(lower=1))
        out = out.merge(agg[["w", col, "resid_event_c", "resid_event_se"]],
                        on=["w", col], how="left")
        out["wallet"] = wallets[out["w"].to_numpy()]
        if labels is not None:
            out[col] = [labels[i] for i in out[col].to_numpy()]
        return out.drop(columns=["w"])

    bands = breakdown("band_i", BAND_LABELS).rename(columns={"band_i": "band"})
    bands.to_parquet(OUT / "wallet_bands.parquet", index=False)
    cats = breakdown("category")
    cats.to_parquet(OUT / "wallet_categories.parquet", index=False)
    years = breakdown("year")
    years.to_parquet(OUT / "wallet_years.parquet", index=False)
    log(f"  bands {len(bands):,} | categories {len(cats):,} | years {len(years):,}")

    log("[6/6] cluster-preserving null over wallet labels")
    # permute the wallet label of each (wallet, event) cell, holding each wallet's
    # event count fixed. Preserves clustering; destroys wallet identity.
    real_disp = np.nanstd(ew_mean[b_n >= MIN_BETS])
    keep = b_n >= MIN_BETS
    null_disp = np.empty(N_NULL)
    null_tail = np.empty(N_NULL)
    obs_t = np.abs(prof["t_event"].to_numpy())
    thresh = 2.5
    for r in range(N_NULL):
        perm = rng.permutation(cell_w)
        s_ = np.bincount(perm, weights=c_mean, minlength=nW)
        n_ = np.bincount(perm, minlength=nW).astype(np.float64)
        ss_ = np.bincount(perm, weights=c_mean ** 2, minlength=nW)
        with np.errstate(invalid="ignore", divide="ignore"):
            m_ = s_ / n_
            v_ = np.maximum(ss_ / n_ - m_ ** 2, 0.0) * n_ / np.maximum(n_ - 1, 1)
            t_ = m_ / np.sqrt(np.maximum(v_ / np.maximum(n_, 1), 1e-12))
        null_disp[r] = np.nanstd(m_[keep])
        null_tail[r] = np.nansum(np.abs(t_[keep]) > thresh)
    obs_tail = float(np.nansum(obs_t[keep] > thresh))
    log(f"  wallets with >={MIN_BETS} bets: {int(keep.sum()):,}")
    log(f"  per-wallet edge dispersion: real {real_disp:.3f}c vs "
        f"null {null_disp.mean():.3f}c  (P={float((null_disp >= real_disp).mean()):.3f})")
    log(f"  wallets with |t|>{thresh}: real {obs_tail:.0f} vs "
        f"null {null_tail.mean():.1f}  (P={float((null_tail >= obs_tail).mean()):.3f})")

    pop = {
        "n_bets": int(len(bets)),
        "n_wallets": int(nW),
        "n_markets": int(bets['m'].nunique()),
        "n_events": int(nE),
        "ts_min": int(bets["ts"].min()), "ts_max": int(bets["ts"].max()),
        "pooled_raw_edge_c": float(raw_c.mean()),
        "baseline_edges": edges.tolist(),
        "baseline_means": bmeans.tolist(),
        "baseline_counts": bcnts.tolist(),
        "min_bets": MIN_BETS,
        "n_wallets_scored": int(keep.sum()),
        "disp_real": float(real_disp),
        "disp_null_mean": float(null_disp.mean()),
        "disp_p": float((null_disp >= real_disp).mean()),
        "tail_thresh": thresh,
        "tail_real": obs_tail,
        "tail_null_mean": float(null_tail.mean()),
        "tail_p": float((null_tail >= obs_tail).mean()),
        "n_null": N_NULL, "seed": SEED,
    }
    (OUT / "population.json").write_text(json.dumps(pop, indent=2))
    log("  wrote population.json")
    log("DONE")


if __name__ == "__main__":
    main()
