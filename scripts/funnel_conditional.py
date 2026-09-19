"""Conditional analysis over the 74 funnel survivors — 2026-07-30.

THE QUESTION (from docs/wallet_funnel_2026-07-29.md, "BRIEF FOR THE NEXT
SESSION"): not *is this wallet good* but **what is the cohort good at**. Across
the 74 structural survivors, does copyable edge concentrate by market category,
entry-price band, or time-to-close at entry? Cohort-level only — the funnel's own
power arithmetic (34 effective events per half) says per-wallet claims below
~15 c/share cannot be supported on this data, so no per-wallet conclusion is
drawn here.

PRE-REGISTERED DESIGN, fixed before any cell was computed:

  Universe    Resolved BUY bets of the 74 survivors (and, for the placebo, of the
              other deepened wallets in the same performance-blind sample), in
              REAL-WORLD markets (micro_crypto excluded), restricted to COPYABLE
              bets: time-to-close at entry >= 24 h, measured as
              tape_t_max(market) - entry_ts exactly as funnel F1 did. Bets whose
              market has no tape_t_max are excluded from cells (reported).
  Unit        The resolution event (src/sports_events.resolution_event), else the
              market. All uncertainty is a bootstrap over EVENTS, never bets.
  Statistic   Edge in cents/share = 100 * (resolved_value - entry_price).
              Bet-weighted mean reported alongside the EVENT-WEIGHTED mean (mean
              over events of the within-event mean); CIs and p's belong to the
              event-weighted number. Kish effective events beside every cell, and
              the minimum detectable edge 100/sqrt(eff_events) printed with it.
  Base rate   The pooled cohort number is printed FIRST; a cell is only ever read
              against it, never against zero alone (funnel trap #3).
  Halves      Each cohort wallet's universe bets are split chronologically in two
              (stable mergesort, first ceil(n/2) = H1). Selection happens on H1
              only; H2 is the exam.
  Selection   A cell is FLAGGED on H1 if eff_events >= 30 AND the event-bootstrap
              95% CI sits entirely above 0. BH-FDR (q=0.10) across all H1 cells
              is reported beside the flags as the multiplicity check.
  Exam        A flagged cell PASSES if on H2 its CI_low > 0 AND its point estimate
              keeps at least half of the H1 point (the repo's "kept >= half"
              convention).
  Placebo     For every PASSING cell: the same statistic over the NON-cohort
              wallets of the deep sample in the same cell (both halves pooled),
              and the paired contrast on SHARED events (within-event cohort mean
              minus placebo mean, event-weighted, bootstrapped over events).
              Reading: contrast CI_low > 0  -> wallet-conditional edge;
              contrast straddles 0 with placebo level > 0 -> market-level rule
              (tradeable without copying anyone, but must then survive a census
              check before belief — Project 4 closed the liquid core only).
  Cells       Marginals: category; entry-price band (10 x 0.10); time-to-close
              {1-3d, 3-7d, 7-30d, >30d}. Two-dimensional: category x coarse price
              band {<0.20, 0.20-0.80, >=0.80} only — cell count is controlled by
              design, not pruned after looking.
  Fees        Each cell carries the frozen per-category taker fee at its mean
              price, k*p*(1-p) (k: sports .05, politics/finance/tech .04, crypto
              .07, other .05 — the paper_rw map, incl. its known conservative
              geopolitics-in-other error). Net = edge - fee is what a taker
              copier would keep, before spread.

Read-only against every shared artifact. Writes only:
  data/interim/funnel/conditional_cells.parquet   (every cell, both halves)
  data/interim/funnel/conditional_bets.parquet    (the compact bet universe)
Run:  flock data/interim/.analysis.lock -c \
        'PYTHONPATH=. .venv/bin/python scripts/funnel_conditional.py'
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper_rw import fee_k_for_category           # frozen category fee map
from src.sports_events import resolution_event

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "data" / "interim" / "realworld"
FUNNEL = ROOT / "data" / "interim" / "funnel"
SHARDS = sorted(glob.glob(str(RW / "deep_trades" / "*.parquet")))
SEED = 20260730
N_BOOT = 1000
BOOT_CHUNK = 100          # bootstrap resamples per allocation, bounds memory
MIN_EFF_EVENTS = 30.0
BH_Q = 0.10

PRICE_BANDS = np.arange(0.0, 1.01, 0.10)
COARSE_BANDS = [0.0, 0.20, 0.80, 1.01]
COARSE_LABELS = ["long(<0.2)", "mid(0.2-0.8)", "fav(>=0.8)"]
HORIZON_EDGES_D = [1.0, 3.0, 7.0, 30.0, np.inf]
HORIZON_LABELS = ["1-3d", "3-7d", "7-30d", ">30d"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data build
def build_universe(close_mode: str = "tape") -> pd.DataFrame:
    """Stream the deep shards into one compact, integer-coded frame of resolved
    BUY bets. Strings never accumulate: wallets and markets are dict-coded per
    shard, and market-level attributes are joined by code at the end.

    close_mode — how time-to-close is measured, and whether it filters:
      "tape"  (registered): tape_t_max - entry >= 24h, exactly as funnel F1.
              ⚠ tape_t_max is the last observed trade, so this CONDITIONS ON
              POST-ENTRY ACTIVITY — a forward-looking universe rule.
      "gamma": scheduled close (closed_ts_gamma, else end_ts_clob) - entry
              >= 24h — outcome-INdependent; the artifact check.
      "none":  no filter at all (all resolved real-world BUYs).
    """
    cat = pd.read_parquet(RW / "market_category.parquet")
    cat_of = dict(zip(cat["market_id"], cat["category"]))
    meta = pd.read_parquet(ROOT / "data" / "interim" / "market_meta.parquet",
                           columns=["market_id", "tape_t_max", "end_ts_clob",
                                    "closed_ts_gamma"])
    if close_mode == "gamma":
        close = meta["closed_ts_gamma"].fillna(meta["end_ts_clob"])
    else:
        close = meta["tape_t_max"]
    tmax_of = dict(zip(meta["market_id"], close))

    cols = ["wallet", "market_id", "side", "entry_price", "timestamp",
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
        mids = df["market_id"].to_numpy()
        m = np.fromiter((mcode.setdefault(x, len(mcode)) for x in mids),
                        dtype=np.int32, count=len(df))
        for mc, s in zip(m, df["slug"].to_numpy()):
            if mc not in slug_by_mcode and isinstance(s, str):
                slug_by_mcode[int(mc)] = s
        parts.append(pd.DataFrame({
            "w": w, "m": m,
            "price": df["entry_price"].to_numpy(np.float32),
            "ts": df["timestamp"].to_numpy(np.int64),
            "rv": df["resolved_value"].to_numpy(np.float32),
        }))
        if (i + 1) % 20 == 0:
            log(f"  [shards {i+1}/{len(SHARDS)}] rows so far "
                f"{sum(len(x) for x in parts):,}")
    bets = pd.concat(parts, ignore_index=True)
    del parts
    log(f"  resolved BUY bets, all wallets: {len(bets):,}")

    # market-level attribute arrays, indexed by market code
    markets = np.empty(len(mcode), dtype=object)
    for s, c in mcode.items():
        markets[c] = s
    m_cat = np.array([cat_of.get(s, "other") for s in markets], dtype=object)
    m_tmax = np.array([tmax_of.get(s, np.nan) for s in markets], dtype=np.float64)

    bets["category"] = pd.Categorical(m_cat[bets["m"].to_numpy()])
    bets = bets[bets["category"] != "micro_crypto"].copy()
    log(f"  after micro_crypto excluded:    {len(bets):,}")

    bets["ttc_h"] = ((m_tmax[bets["m"].to_numpy()] - bets["ts"].to_numpy())
                     / 3600.0).astype(np.float32)
    n_unk = int(np.isnan(bets["ttc_h"]).sum())
    if close_mode != "none":
        bets = bets[bets["ttc_h"] >= 24.0].copy()
        log(f"  copyable (ttc>=24h via {close_mode}; {n_unk:,} unmeasurable "
            f"dropped): {len(bets):,}")
    else:
        log(f"  no time-to-close filter (ttc unmeasurable for {n_unk:,}): "
            f"{len(bets):,}")

    # resolution event per surviving market, coded
    ev_key: dict[int, str] = {}
    for c in bets["m"].unique():
        s = markets[c]
        ev_key[int(c)] = resolution_event(slug_by_mcode.get(int(c)),
                                          market_id=s)[0]
    ecode: dict[str, int] = {}
    bets["e"] = np.fromiter(
        (ecode.setdefault(ev_key[int(c)], len(ecode))
         for c in bets["m"].to_numpy()),
        dtype=np.int32, count=len(bets))
    event_names = np.empty(len(ecode), dtype=object)
    for s, c in ecode.items():
        event_names[c] = s
    build_universe.last_event_names = event_names  # for callers needing strings

    bets["edge_c"] = (100.0 * (bets["rv"] - bets["price"])).astype(np.float32)
    bets["band"] = pd.cut(bets["price"], PRICE_BANDS, right=False,
                          include_lowest=True).astype(str).astype("category")
    bets["coarse"] = pd.cut(bets["price"], COARSE_BANDS, right=False,
                            labels=COARSE_LABELS, include_lowest=True)
    bets["horizon"] = pd.cut(bets["ttc_h"] / 24.0, HORIZON_EDGES_D,
                             right=False, labels=HORIZON_LABELS)

    wallets = np.empty(len(wcode), dtype=object)
    for s, c in wcode.items():
        wallets[c] = s
    return bets.reset_index(drop=True), wallets


def split_halves(bets: pd.DataFrame, wallets: np.ndarray,
                 cohort: set[str]) -> pd.DataFrame:
    """half = 'H1'/'H2' for cohort wallets (stable chronological split),
    'placebo' for everyone else. Operates on full-frame positions throughout."""
    bets = bets.sort_values(["w", "ts"], kind="mergesort").reset_index(drop=True)
    coh_codes = {c for c, s in enumerate(wallets) if s in cohort}
    half = np.full(len(bets), "placebo", dtype=object)
    wcol = bets["w"].to_numpy()
    pos = np.where(np.isin(wcol, list(coh_codes)))[0]
    # pos is sorted; within it, rows are (wallet, ts)-sorted
    if len(pos):
        wp = wcol[pos]
        # group boundaries where the wallet code changes
        starts = np.r_[0, np.where(np.diff(wp) != 0)[0] + 1, len(wp)]
        for a, b in zip(starts[:-1], starts[1:]):
            gpos = pos[a:b]
            k = int(np.ceil(len(gpos) / 2))
            half[gpos[:k]] = "H1"
            half[gpos[k:]] = "H2"
    bets["half"] = half
    return bets


# ---------------------------------------------------------------- statistics
def kish(counts: np.ndarray) -> float:
    s = float(counts.sum())
    return s * s / float((counts.astype(np.float64) ** 2).sum()) if len(counts) else 0.0


def _boot_means(vec: np.ndarray, rng: np.random.Generator,
                n_boot: int = N_BOOT) -> np.ndarray:
    """Bootstrap means of `vec` resampled with replacement, chunked so the
    index matrix never exceeds BOOT_CHUNK x len(vec)."""
    out = np.empty(n_boot, dtype=np.float64)
    done = 0
    n = len(vec)
    while done < n_boot:
        k = min(BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        out[done:done + k] = vec[idx].mean(axis=1)
        done += k
    return out


def cell_stats(sub: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Event-weighted mean edge with an event bootstrap; bet-weighted alongside."""
    if not len(sub):
        return dict(n_bets=0, n_wallets=0, n_events=0, eff_events=0.0,
                    bet_wtd=np.nan, event_wtd=np.nan, ci_low=np.nan,
                    ci_high=np.nan, p_gt0=np.nan, mean_price=np.nan)
    g = sub.groupby("e", observed=True)["edge_c"]
    ev_mean = g.mean().to_numpy(np.float64)
    ev_n = g.size().to_numpy()
    stat = float(ev_mean.mean())
    if len(ev_mean) >= 2:
        boot = _boot_means(ev_mean, rng)
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        p_gt0 = float((boot <= 0).mean())
    else:
        ci_low = ci_high = p_gt0 = np.nan
    return dict(n_bets=int(len(sub)),
                n_wallets=int(sub["w"].nunique()),
                n_events=int(len(ev_mean)), eff_events=kish(ev_n),
                bet_wtd=float(sub["edge_c"].mean()), event_wtd=stat,
                ci_low=float(ci_low), ci_high=float(ci_high), p_gt0=p_gt0,
                mean_price=float(sub["price"].mean()))


def paired_contrast(coh: pd.DataFrame, pla: pd.DataFrame,
                    rng: np.random.Generator) -> dict:
    """Within-event cohort-minus-placebo mean edge on SHARED events."""
    a = coh.groupby("e", observed=True)["edge_c"].mean()
    b = pla.groupby("e", observed=True)["edge_c"].mean()
    shared = a.index.intersection(b.index)
    if len(shared) < 2:
        return dict(shared_events=int(len(shared)), diff=np.nan,
                    d_ci_low=np.nan, d_ci_high=np.nan)
    d = (a.loc[shared] - b.loc[shared]).to_numpy(np.float64)
    boot = _boot_means(d, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(shared_events=int(len(shared)), diff=float(d.mean()),
                d_ci_low=float(lo), d_ci_high=float(hi))


def fee_cents(category: str, mean_price: float) -> float:
    if not np.isfinite(mean_price):
        return np.nan
    k = fee_k_for_category(str(category))
    return 100.0 * k * mean_price * (1.0 - mean_price)


def bh_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up over defined p-values."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        out[order[: kmax + 1]] = True
    return out


COLMAP = {"category": "category", "price_band": "band", "horizon": "horizon"}


# ------------------------------------------------------------------- driver
def main() -> None:
    rng = np.random.default_rng(SEED)
    surv = pd.read_parquet(FUNNEL / "structural_survivors_74.parquet")
    cohort = set(surv["wallet"])
    log(f"cohort: {len(cohort)} structural survivors")

    log("building bet universe from the deep shards …")
    bets, wallets = build_universe()
    bets = split_halves(bets, wallets, cohort)
    found = bets.loc[bets["half"] != "placebo", "w"].nunique()
    log(f"cohort wallets found in shards: {found}/74")
    cov = bets[bets["half"] != "placebo"].groupby("w", observed=True).size()
    if len(cov) and int(cov.min()) < 50:
        log(f"  ⚠ {(cov < 50).sum()} cohort wallets have <50 universe bets here "
            f"(min {int(cov.min())}) — coverage caveat, reported not fixed")
    bets.to_parquet(FUNNEL / "conditional_bets.parquet", index=False)

    coh = bets[bets["half"] != "placebo"]
    pla = bets[bets["half"] == "placebo"]
    h1 = bets[bets["half"] == "H1"]
    h2 = bets[bets["half"] == "H2"]

    log("\n================ BASE RATES FIRST (funnel trap #3) ================")
    for name, sub in [("cohort ALL", coh), ("cohort H1", h1),
                      ("cohort H2", h2), ("placebo (non-cohort)", pla)]:
        s = cell_stats(sub, rng)
        log(f"{name:>22}: bet-wtd {s['bet_wtd']:+.2f}c | event-wtd "
            f"{s['event_wtd']:+.2f}c [{s['ci_low']:+.2f},{s['ci_high']:+.2f}] "
            f"| {s['n_bets']:,} bets, {s['n_events']:,} events "
            f"(eff {s['eff_events']:,.0f}), {s['n_wallets']} wallets")
    base_h1 = cell_stats(h1, rng)["event_wtd"]
    pc_all = paired_contrast(coh, pla, rng)
    log(f"overall paired contrast coh−pla: {pc_all['diff']:+.2f}c "
        f"[{pc_all['d_ci_low']:+.2f},{pc_all['d_ci_high']:+.2f}] on "
        f"{pc_all['shared_events']:,} shared events")

    dims = [("category", "category"), ("price_band", "band"),
            ("horizon", "horizon")]
    rows = []
    log("\n================ MARGINALS (H1 -> H2, cohort) =====================")
    for dim_name, col in dims:
        log(f"\n--- by {dim_name} ---")
        for val in [v for v in h1[col].dropna().unique()]:
            r = {"cell": f"{dim_name}={val}", "dim": dim_name, "value": str(val)}
            s1 = cell_stats(h1[h1[col] == val], rng)
            s2 = cell_stats(h2[h2[col] == val], rng)
            sp = cell_stats(pla[pla[col] == val], rng)
            r.update({f"h1_{k}": v for k, v in s1.items()})
            r.update({f"h2_{k}": v for k, v in s2.items()})
            r.update({f"pl_{k}": v for k, v in sp.items()})
            cat_for_fee = str(val) if dim_name == "category" else "other"
            r["fee_c"] = fee_cents(cat_for_fee, s1["mean_price"])
            r["mde_c"] = (100.0 / np.sqrt(s1["eff_events"])
                          if s1["eff_events"] else np.nan)
            rows.append(r)
            log(f"  {str(val)[:28]:28} H1 {s1['event_wtd']:+6.2f}c "
                f"[{s1['ci_low']:+6.2f},{s1['ci_high']:+6.2f}] eff {s1['eff_events']:8,.0f} "
                f"(mde {r['mde_c']:5.1f}c) | H2 {s2['event_wtd']:+6.2f}c "
                f"[{s2['ci_low']:+6.2f},{s2['ci_high']:+6.2f}] eff {s2['eff_events']:8,.0f} "
                f"| placebo {sp['event_wtd']:+6.2f}c")

    log("\n================ 2-D: category x coarse price band ================")
    for cat_v in [v for v in h1["category"].dropna().unique()]:
        for band_v in COARSE_LABELS:
            m1 = (h1["category"] == cat_v) & (h1["coarse"] == band_v)
            if int(m1.sum()) < 50:
                continue
            r = {"cell": f"{cat_v} x {band_v}", "dim": "cat_x_band",
                 "value": f"{cat_v}|{band_v}"}
            s1 = cell_stats(h1[m1], rng)
            s2 = cell_stats(h2[(h2["category"] == cat_v)
                               & (h2["coarse"] == band_v)], rng)
            sp = cell_stats(pla[(pla["category"] == cat_v)
                                & (pla["coarse"] == band_v)], rng)
            r.update({f"h1_{k}": v for k, v in s1.items()})
            r.update({f"h2_{k}": v for k, v in s2.items()})
            r.update({f"pl_{k}": v for k, v in sp.items()})
            r["fee_c"] = fee_cents(str(cat_v), s1["mean_price"])
            r["mde_c"] = (100.0 / np.sqrt(s1["eff_events"])
                          if s1["eff_events"] else np.nan)
            rows.append(r)
            log(f"  {str(cat_v)[:16]:16} {band_v:13} H1 {s1['event_wtd']:+6.2f}c "
                f"[{s1['ci_low']:+6.2f},{s1['ci_high']:+6.2f}] eff {s1['eff_events']:8,.0f} "
                f"| H2 {s2['event_wtd']:+6.2f}c [{s2['ci_low']:+6.2f},{s2['ci_high']:+6.2f}] "
                f"| placebo {sp['event_wtd']:+6.2f}c")

    cells = pd.DataFrame(rows)
    cells["h1_flag"] = ((cells["h1_eff_events"] >= MIN_EFF_EVENTS)
                        & (cells["h1_ci_low"] > 0))
    cells["bh_pass"] = bh_reject(cells["h1_p_gt0"].to_numpy(), BH_Q)
    cells["h2_pass"] = (cells["h1_flag"] & (cells["h2_ci_low"] > 0)
                        & (cells["h2_event_wtd"] >= 0.5 * cells["h1_event_wtd"]))

    log("\n================ SELECTION -> EXAM ================================")
    log(f"H1 base rate (event-wtd): {base_h1:+.2f}c — cells are read against "
        f"this, not against zero")
    flagged = cells[cells["h1_flag"]]
    log(f"flagged on H1 (eff>={MIN_EFF_EVENTS:.0f} & CI>0): {len(flagged)} of "
        f"{len(cells)} cells | BH q={BH_Q}: {int(cells['bh_pass'].sum())} pass")
    passing = cells[cells["h2_pass"]]
    log(f"PASS the H2 exam: {len(passing)}")

    if len(passing):
        log("\n---- placebo contrast on passing cells (paired, shared events) ----")
        for ridx, r in passing.iterrows():
            if r["dim"] == "cat_x_band":
                cv, bv = r["value"].split("|")
                mc = (coh["category"].astype(str) == cv) & (coh["coarse"].astype(str) == bv)
                mp = (pla["category"].astype(str) == cv) & (pla["coarse"].astype(str) == bv)
            else:
                col = COLMAP[r["dim"]]
                mc = coh[col].astype(str) == r["value"]
                mp = pla[col].astype(str) == r["value"]
            pc = paired_contrast(coh[mc], pla[mp], rng)
            verdict = ("WALLET-conditional"
                       if (np.isfinite(pc["d_ci_low"]) and pc["d_ci_low"] > 0)
                       else ("market-level rule?" if r["pl_event_wtd"] > 0
                             else "unclear"))
            log(f"  {r['cell'][:44]:44} coh-pla {pc['diff']:+6.2f}c "
                f"[{pc['d_ci_low']:+6.2f},{pc['d_ci_high']:+6.2f}] on "
                f"{pc['shared_events']:,} shared events -> {verdict}")
            for k, v in pc.items():
                cells.loc[ridx, f"pc_{k}"] = v

    cells.to_parquet(FUNNEL / "conditional_cells.parquet", index=False)
    log(f"\nwrote {FUNNEL / 'conditional_cells.parquet'} ({len(cells)} cells)")

    # where do the six provisional wallets sit? (descriptive only)
    six_prefixes = {"0x933ca00f56", "0x630f096a63", "0x4478d7bd8a",
                    "0xd501dd1c72", "0x44c1dfe432", "0x1ee9a5fc09"}
    six_codes = [c for c, s in enumerate(wallets)
                 if isinstance(s, str) and s[:12] in six_prefixes]
    if len(passing) and six_codes:
        log("\n---- the six provisional wallets vs passing cells (descriptive) ----")
        sixb = coh[coh["w"].isin(six_codes)]
        for _, r in passing.iterrows():
            if r["dim"] == "cat_x_band":
                cv, bv = r["value"].split("|")
                s6 = float(((sixb["category"].astype(str) == cv)
                            & (sixb["coarse"].astype(str) == bv)).mean()) if len(sixb) else np.nan
                sc = float(((coh["category"].astype(str) == cv)
                            & (coh["coarse"].astype(str) == bv)).mean())
            else:
                col = COLMAP[r["dim"]]
                s6 = float((sixb[col].astype(str) == r["value"]).mean()) if len(sixb) else np.nan
                sc = float((coh[col].astype(str) == r["value"]).mean())
            log(f"  {r['cell'][:44]:44} six {s6:5.1%} of bets vs cohort {sc:5.1%}")


def check() -> None:
    """ARTIFACT CHECK for the registered run's universe rule. tape_t_max is the
    last observed trade, so `tape_t_max - entry >= 24h` conditions each bet on
    POST-ENTRY MARKET ACTIVITY — a forward-looking rule that could manufacture
    a longshot-positive / favorite-negative gradient by keeping the longshots
    that stayed interesting and dropping the ones that died. Two counter-
    universes: an outcome-INdependent scheduled close ("gamma"), and no filter
    at all ("none"). If the band structure only exists under "tape", it is the
    filter, not the market."""
    rng = np.random.default_rng(SEED + 1)
    surv = pd.read_parquet(FUNNEL / "structural_survivors_74.parquet")
    cohort = set(surv["wallet"])
    for mode in ("gamma", "none"):
        log(f"\n########## close_mode = {mode} ##########")
        bets, wallets = build_universe(mode)
        bets = split_halves(bets, wallets, cohort)
        coh = bets[bets["half"] != "placebo"]
        pla = bets[bets["half"] == "placebo"]
        for name, sub in [("cohort", coh), ("placebo", pla)]:
            s = cell_stats(sub, rng)
            log(f"  {name:>8} pooled: bet-wtd {s['bet_wtd']:+.2f}c | event-wtd "
                f"{s['event_wtd']:+.2f}c [{s['ci_low']:+.2f},{s['ci_high']:+.2f}] "
                f"| {s['n_bets']:,} bets / {s['n_events']:,} events")
        pc = paired_contrast(coh, pla, rng)
        log(f"  overall paired contrast coh−pla: {pc['diff']:+.2f}c "
            f"[{pc['d_ci_low']:+.2f},{pc['d_ci_high']:+.2f}] on "
            f"{pc['shared_events']:,} shared events")
        log("  --- price bands (cohort, both halves | placebo) ---")
        for val in sorted(coh["band"].dropna().unique(), key=str):
            sc = cell_stats(coh[coh["band"] == val], rng)
            sp = cell_stats(pla[pla["band"] == val], rng)
            log(f"    {str(val):10} coh {sc['event_wtd']:+6.2f}c "
                f"[{sc['ci_low']:+6.2f},{sc['ci_high']:+6.2f}] eff {sc['eff_events']:8,.0f} "
                f"| pla {sp['event_wtd']:+6.2f}c [{sp['ci_low']:+6.2f},{sp['ci_high']:+6.2f}]")
        del bets, coh, pla


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
