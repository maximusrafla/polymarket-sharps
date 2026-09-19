"""METRIC B — copyability of the three slow-cadence sports wallets (tier `s2_slow`).

THE QUESTION
------------
Not "do these wallets have skill" — that is already established retrospectively
(`docs/project3_sports_validation.md`, `docs/redteam_audit_2026-07-26.md` §1, and
the `s2_slow` tier of `data/processed/sports_sig2c_freeze_manifest.json`). The
question here is the one the whole project exists for:

    if you had seen each of their entries N seconds/minutes/hours LATER and
    entered then, at the price actually available, would you have captured any
    of their edge?

The subjects are the three members of `s2_slow` — the only wallets in the sports
cohort whose held-out cadence (3.1 / 5.8 / 7.3 bets/day) is slow enough that a
human or a bot could plausibly act on them. Every other member of the sports
cohort is in-play HFT and uncopyable by inspection.

THE TRAP THIS SCRIPT EXISTS TO AVOID
------------------------------------
`scripts/audit_edge_decay_long.py` (2026-07-22) found a large, apparently robust
"+12.3¢ follower edge" at Δ=1h that survived concentration guards, cluster
bootstraps and leave-one-family-out — and was still an artifact. It was caught
ONLY by two wallet-agnostic placebo anchors, which both BEAT copying:

  * **random anchor** — price the same token at a uniformly random moment in its
    observed pre-guard life ("buy this market whenever", not conditioned on the
    wallet at all);
  * **pre-entry anchor** — price at entry MINUS Δ, i.e. a follower who acted
    before the wallet and therefore could not have been copying it.

So both placebos are run here, PAIRED on the bets where every anchor exists, and
no positive follower number is believed unless it beats them.

BRACKETS
--------
* Δ ∈ {0, 30s, 1m, 5m, 15m, 1h, 6h, 24h, 3d}. These wallets enter days before
  resolution, so unlike the HFT cohort the LONG Δs are the economically relevant
  ones; the sweep does not stop at 1h.
* Fill bandwidth ∈ {60s, 300s (primary), 3600s}, plus a deliberately generous
  **next-print** anchor (the first other-wallet print at or after entry+Δ,
  unbounded up to the resolution guard) so a NO-GO cannot be an artifact of a
  too-narrow fill window.
* Baseline: the hierarchical `niche_l1` / `niche_l2` (league / league|form)
  baseline copied VERBATIM from `sports_sig2c_freeze_manifest.json`. Never
  refit — applied with `src.slow_baseline.expected_outcome_hier`.
* Cluster unit: the RESOLUTION EVENT (`src.sports_events.assign_events`), not
  `market_id` (over-counts ~11x) and not league (under-counts). Bootstraps
  resample EVENTS. The headline aggregate is the EVENT-WEIGHTED mean, matching
  the freeze manifest's own `scoring_rule`; the bet-weighted mean is reported
  alongside.
* Leakage discipline is `src/features.py`'s: bounded forward window, the
  `scoring.fair_value_resolution_guard` resolution guard, and NEVER a
  `resolved_value` fallback. The follower price is computed with the production
  scalar core `features._forward_price_arrays` itself — at ~3.6k bets the
  vectorized `audit_edge_decay_long.FollowerPricer` buys nothing, and calling the
  production function directly removes any reimplementation risk (it is the very
  function that script's `--verify` mode checks itself against). The statistics
  (`cluster_boot_ci`, `effective_breadth`) ARE imported from that script, so the
  bootstrap machinery is literally shared.

KNOWN SYSTEMATIC LIMIT — stated on every retrospective number below
-------------------------------------------------------------------
The deep sports tape is a backfill of *selected* wallets (`/trades?user=` for 293
sports-screened wallets), so the "market tape" a follower is priced against is
other TRACKED wallets' prints, not the real order book. Coverage measured here is
therefore a LOWER BOUND on real fill opportunity and the prices are a biased
sample of the book. `--enrich-ledger` (default on) adds every print on the same
tokens from `data/interim/bet_ledger.parquet` and reports how much that moves
coverage — it is a direct measurement of how tape-limited the number is. Section 5
probes the LIVE read-only CLOB book because that is the only unbiased liquidity
evidence available.

n = 3 WALLETS. Whatever this finds is DIRECTIONAL, not decisive.

Read-only. Writes NOTHING. No keys, no orders. Public GETs only (section 5).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/audit_sports_copyability.py
    PYTHONPATH=. .venv/bin/python scripts/audit_sports_copyability.py --no-ledger --no-live

Reading the ledger (default) touches `data/interim/bet_ledger.parquet`; take the
shared analysis lock when another heavy job may be running:

    flock data/interim/.analysis.lock -c 'PYTHONPATH=. .venv/bin/python \
        scripts/audit_sports_copyability.py > /tmp/metricb.log 2>&1'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_edge_decay_long import cluster_boot_ci, effective_breadth, fmt
from src.common import load_config
from src.features import _forward_price_arrays
from src.slow_baseline import expected_outcome_hier
from src.sports_validate import DEEP_TRADES_PATH, sports_market_index

SEED = 20260726
N_BOOT = 2000
MANIFEST = Path("data/processed/sports_sig2c_freeze_manifest.json")
LEDGER_PATH = Path("data/interim/bet_ledger.parquet")
TIER = "s2_slow"

DELTAS = [
    (0, "0"),
    (30, "30s"),
    (60, "1m"),
    (300, "5m"),
    (900, "15m"),
    (3600, "1h"),
    (21600, "6h"),
    (86400, "24h"),
    (259200, "3d"),
]
BANDWIDTHS = [60, 300, 3600]
PRIMARY_BANDWIDTH = 300
PLACEBO_DELTAS = [30, 300, 3600, 86400]
TAPE_COLUMNS = ["token_id", "wallet", "timestamp", "entry_price", "size", "tx_hash"]

CLOB_BOOK = "https://clob.polymarket.com/book"
DATA_POSITIONS = "https://data-api.polymarket.com/positions"
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"


# ---------------------------------------------------------------------------
# Pure helpers (tested in tests/test_sports_copyability.py)
# ---------------------------------------------------------------------------
def guard_cutoff(ts: np.ndarray, guard: float) -> float:
    """Resolution-guard cutoff for one token: drop the final `guard` fraction of
    its OBSERVED lifespan, so near-resolution prints cannot proxy fair value.

    Identical to `audit_edge_decay_long.FollowerPricer`'s cutoff, which is the
    integer-floored form of the guard `features.compute_forward_drift` applies."""
    t_min, t_max = float(np.min(ts)), float(np.max(ts))
    return float(np.floor(t_max - guard * (t_max - t_min)))


def next_other_print(ts: np.ndarray, wallet: np.ndarray, price: np.ndarray,
                     anchor_ts: float, exclude_wallet: str,
                     cutoff_ts: float) -> tuple[float, float] | None:
    """The FIRST other-wallet print strictly after `anchor_ts` and at or before
    `cutoff_ts`, as (price, delay_seconds). None when no such print exists.

    This is the maximally generous follower anchor: it assumes the copier can
    wait indefinitely (up to the guard) for the next tradeable print rather than
    needing one inside a bounded fill window. `ts` need not be sorted."""
    mask = (wallet != exclude_wallet) & (ts > anchor_ts) & (ts <= cutoff_ts)
    if not mask.any():
        return None
    idx = int(np.flatnonzero(mask)[np.argmin(ts[mask])])
    return float(price[idx]), float(ts[idx] - anchor_ts)


def event_weighted_mean(values: np.ndarray, events: np.ndarray) -> float:
    """Mean over RESOLUTION EVENTS of the within-event mean — the freeze
    manifest's own headline statistic. Buying every team in one field is one
    observation, not one per team. NaN values are dropped; NaN if nothing left."""
    values = np.asarray(values, dtype=float)
    events = np.asarray(events)
    ok = ~np.isnan(values)
    values, events = values[ok], events[ok]
    if values.size == 0:
        return float("nan")
    codes = np.unique(events, return_inverse=True)[1]
    n = np.bincount(codes).astype(float)
    s = np.bincount(codes, weights=values)
    return float((s / n).mean())


def design_effect(values: np.ndarray, clusters: np.ndarray, rng,
                  n: int = 500) -> float:
    """D = Var(cluster-bootstrap mean) / Var(iid-bootstrap mean).

    D > 1 means the cluster null is correctly charging for shared resolution;
    D < 1 means it is OVER-CONSTRAINED and unusable for per-unit arbitration
    (this repo's `permute_wallets_within_market` pathology sat at D = 0.17 — see
    `docs/blackswan_cluster_null.md`). NaN below two clusters."""
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    values, clusters = values[ok], np.asarray(clusters)[ok]
    codes = np.unique(clusters, return_inverse=True)[1]
    g = int(codes.max()) + 1 if codes.size else 0
    if g < 2 or values.size < 2:
        return float("nan")
    sums = np.bincount(codes, weights=values, minlength=g)
    cnts = np.bincount(codes, minlength=g).astype(float)
    ci = rng.integers(0, g, size=(n, g))
    cl_means = sums[ci].sum(axis=1) / cnts[ci].sum(axis=1)
    ii = rng.integers(0, values.size, size=(n, values.size))
    iid_means = values[ii].mean(axis=1)
    v_iid = float(np.var(iid_means))
    if v_iid <= 0:
        return float("nan")
    return float(np.var(cl_means) / v_iid)


def effective_spread(bids: list[tuple[float, float]],
                     asks: list[tuple[float, float]]) -> dict:
    """Top-of-book summary of a CLOB `/book` response, in price units.

    Returns best_bid / best_ask / spread / ask_size (size resting AT the best
    ask) / ask_size_2c (cumulative size within 2¢ of the best ask). NaN fields
    when a side is empty. Purely arithmetic — no network."""
    b = sorted([p for p, _ in bids], reverse=True)
    a = sorted([p for p, _ in asks])
    best_bid = b[0] if b else float("nan")
    best_ask = a[0] if a else float("nan")
    ask_size = float(sum(s for p, s in asks if p == best_ask)) if a else float("nan")
    ask_2c = (float(sum(s for p, s in asks if p <= best_ask + 0.02))
              if a else float("nan"))
    return {"best_bid": best_bid, "best_ask": best_ask,
            "spread": best_ask - best_bid if (a and b) else float("nan"),
            "ask_size": ask_size, "ask_size_2c": ask_2c}


# ---------------------------------------------------------------------------
# Tape
# ---------------------------------------------------------------------------
class TokenTape:
    """Per-token price path: every print we can see on that token, any wallet,
    any side. Small enough (thousands of tokens) to keep as plain numpy per
    token, which lets every anchor be evaluated by the PRODUCTION core."""

    def __init__(self, trades: pd.DataFrame, guard: float):
        self.guard = guard
        self.g: dict[str, tuple] = {}
        for tok, grp in trades.groupby("token_id", sort=False):
            grp = grp.sort_values("timestamp", kind="stable")
            ts = grp["timestamp"].to_numpy(dtype=float)
            self.g[tok] = (
                ts,
                grp["entry_price"].to_numpy(dtype=float),
                grp["size"].to_numpy(dtype=float),
                grp["wallet"].to_numpy(),
                guard_cutoff(ts, guard),
                float(ts[0]),
            )

    def price_at(self, tok, wallet, entry_ts, delta, bandwidth) -> float:
        """Size-weighted mean OTHER-wallet price in (entry+Δ, entry+Δ+bandwidth],
        at or before the guard cutoff. NaN when nothing qualifies — never a
        `resolved_value` fallback."""
        e = self.g.get(tok)
        if e is None:
            return np.nan
        ts, price, size, wal, cutoff, _first = e
        out = _forward_price_arrays(ts, price, size, wal,
                                    float(entry_ts) + float(delta), wallet,
                                    float(bandwidth), cutoff)
        return np.nan if out is None else out

    def next_print(self, tok, wallet, anchor_ts, guarded: bool = True):
        """(price, delay, ts) of the first other-wallet print after `anchor_ts`,
        before the guard (`guarded=False` lifts the guard, giving the coverage
        UPPER bound). (nan, nan, nan) when there is none."""
        e = self.g.get(tok)
        if e is None:
            return np.nan, np.nan, np.nan
        ts, price, size, wal, cutoff, _first = e
        r = next_other_print(ts, wal, price, float(anchor_ts), wallet,
                             cutoff if guarded else np.inf)
        if r is None:
            return np.nan, np.nan, np.nan
        return r[0], r[1], float(anchor_ts) + r[1]

    def covered(self, tok, wallet, entry_ts) -> bool:
        """Copyability ceiling: is there ANY other-wallet print after entry and
        before the guard? If not, the position is un-copyable at any latency."""
        p, _d, _t = self.next_print(tok, wallet, entry_ts)
        return not np.isnan(p)

    def span(self, tok) -> tuple[float, float]:
        e = self.g.get(tok)
        return (np.nan, np.nan) if e is None else (e[5], e[4])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_manifest(path: Path = MANIFEST) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_bets(wallets: list[str], idx: pd.DataFrame) -> pd.DataFrame:
    """The subjects' resolved BUY bets in the deep SPORTS universe, labelled with
    the frozen baseline's levels and the resolution event."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(
        DEEP_TRADES_PATH,
        columns=["wallet", "market_id", "token_id", "entry_price", "size",
                 "timestamp", "resolved", "resolved_value", "side"],
        filters=[("wallet", "in", wallets), ("side", "==", "BUY"),
                 ("resolved", "==", True)],
    )
    bets = tbl.to_pandas()
    cols = ["market_id", "niche_l1", "niche_l2", "niche_l3", "event",
            "event_kind", "slug", "res_ts_proxy"]
    bets = bets.merge(idx[cols], on="market_id", how="inner")
    return bets.dropna(subset=["resolved_value", "entry_price"]).reset_index(drop=True)


def load_tape(tokens, use_ledger: bool) -> tuple[pd.DataFrame, dict]:
    """Every print we can see on the subjects' tokens: the deep sports tape,
    optionally unioned with the main bet ledger (deduped on the full print)."""
    import pyarrow.parquet as pq

    toks = list(tokens)
    deep = pq.read_table(DEEP_TRADES_PATH, columns=TAPE_COLUMNS,
                         filters=[("token_id", "in", toks)]).to_pandas()
    stats = {"deep_rows": len(deep), "deep_wallets": deep["wallet"].nunique(),
             "ledger_rows": 0, "added_rows": 0, "ledger_wallets": 0}
    tape = deep
    if use_ledger and LEDGER_PATH.exists():
        tset = set(toks)
        chunks = []
        pf = pq.ParquetFile(LEDGER_PATH)
        for batch in pf.iter_batches(batch_size=250_000, columns=TAPE_COLUMNS):
            df = batch.to_pandas()
            df = df[df["token_id"].isin(tset)]
            if len(df):
                chunks.append(df)
        if chunks:
            led = pd.concat(chunks, ignore_index=True)
            stats["ledger_rows"] = len(led)
            stats["ledger_wallets"] = led["wallet"].nunique()
            before = len(tape)
            tape = pd.concat([tape, led], ignore_index=True).drop_duplicates(
                subset=TAPE_COLUMNS)
            stats["added_rows"] = len(tape) - before
    return tape.reset_index(drop=True), stats


def skill(baseline: dict, prices: np.ndarray, l1: np.ndarray, l2: np.ndarray,
          rv: np.ndarray) -> np.ndarray:
    """resolved_value − E[outcome | price, league, league|form] under the FROZEN
    baseline. NaN price -> NaN skill."""
    df = pd.DataFrame({"entry_price": np.asarray(prices, dtype=float),
                       "niche_l1": np.asarray(l1), "niche_l2": np.asarray(l2)})
    return np.asarray(rv, dtype=float) - expected_outcome_hier(baseline, df)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_coverage(bets: pd.DataFrame, tape: TokenTape, rng) -> None:
    print("\n" + "-" * 92)
    print("1. COVERAGE / FOLLOWABILITY CEILING")
    print("   share of resolved BUY bets with ANY other-wallet print after entry and before the")
    print("   resolution guard. This is the ceiling: no later print => un-copyable at any latency.")
    print("   LOWER BOUND — the visible tape is tracked wallets' prints, not the order book.")
    print("-" * 92)
    print("   `no-guard` lifts the resolution guard — the coverage UPPER bound, the convention")
    print("   the HANDOFF Δ-coverage probe used.")
    print("-" * 92)
    print(f"  {'wallet':>12} | {'bets':>6} {'events':>7} {'effEv':>6} | {'covered':>7} "
          f"{'n_cov':>6} {'no-guard':>8} | {'delay to 1st other print (s)':>32}")
    for w, g in list(bets.groupby("wallet")) + [("POOLED", bets)]:
        cov, cov_ng, delay = [], [], []
        for tok, wal, ts in zip(g["token_id"], g["wallet"], g["timestamp"]):
            p, d, _t = tape.next_print(tok, wal, ts)
            cov.append(not np.isnan(p))
            cov_ng.append(not np.isnan(tape.next_print(tok, wal, ts, guarded=False)[0]))
            if not np.isnan(d):
                delay.append(d)
        cov = np.asarray(cov)
        delay = np.asarray(delay)
        q = (np.percentile(delay, [25, 50, 75]) if delay.size
             else np.array([np.nan] * 3))
        print(f"  {w[:12]:>12} | {len(g):>6} {g['event'].nunique():>7} "
              f"{effective_breadth(g['event'].to_numpy()):>6.1f} | {cov.mean():>7.1%} "
              f"{int(cov.sum()):>6} {np.mean(cov_ng):>8.1%} | "
              f"p25 {q[0]:>9,.0f} p50 {q[1]:>9,.0f} p75 {q[2]:>9,.0f}")
    print("\n  reference: the deep real-world ledger's ceiling was 25% "
          "(HANDOFF 'Long-latency edge-decay re-run').")


def section_copy_window(bets: pd.DataFrame, tape: TokenTape, baseline: dict,
                        window_hours: float) -> None:
    print("\n" + "-" * 92)
    print(f"2. COPY WINDOW — entry price vs guarded forward fair value ({window_hours:.0f}h window)")
    print("   copy_window = fwd_price − entry_price, exactly features.compute_forward_drift's")
    print("   quantity (bounded window, resolution guard, NO resolved_value fallback).")
    print("-" * 92)
    win = float(window_hours) * 3600.0
    fwd = np.array([tape.price_at(t, w, ts, 0, win)
                    for t, w, ts in zip(bets["token_id"], bets["wallet"], bets["timestamp"])])
    cw = fwd - bets["entry_price"].to_numpy(dtype=float)
    bets = bets.assign(_cw=cw, _fwd=fwd)
    print(f"  {'wallet':>12} | {'defined':>7} | {'entry':>6} {'fwd':>6} | "
          f"{'copy_window':>11} {'evt-wtd':>8} {'median':>7} {'share>0':>7}")
    for w, g in list(bets.groupby("wallet")) + [("POOLED", bets)]:
        v = g["_cw"].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if not ok.any():
            print(f"  {w[:12]:>12} | {'0.0%':>7} | (no guarded forward price anywhere)")
            continue
        print(f"  {w[:12]:>12} | {ok.mean():>7.1%} | {g['entry_price'].mean():>6.3f} "
              f"{np.nanmean(g['_fwd']):>6.3f} | {np.nanmean(v):>+11.4f} "
              f"{event_weighted_mean(v, g['event'].to_numpy()):>+8.4f} "
              f"{np.nanmedian(v):>+7.4f} {(v[ok] > 0).mean():>7.1%}")


def section_decay(bets: pd.DataFrame, tape: TokenTape, baseline: dict,
                  rng, bandwidth: int) -> dict:
    """Follower skill edge at each Δ, with coverage reported alongside — coverage
    collapse is what manufactured the fake +12.3¢ in the 2026-07-22 run."""
    tok = bets["token_id"].to_numpy()
    wal = bets["wallet"].to_numpy()
    ts = bets["timestamp"].to_numpy(dtype=float)
    ep = bets["entry_price"].to_numpy(dtype=float)
    rv = bets["resolved_value"].to_numpy(dtype=float)
    l1, l2 = bets["niche_l1"].to_numpy(), bets["niche_l2"].to_numpy()
    ev = bets["event"].to_numpy()
    n_all = len(bets)
    out = {}

    print(f"\n  bandwidth = {bandwidth}s")
    print(f"  {'Δ':>4} | {'cov':>6} {'n':>6} {'events':>6} {'effEv':>6} | {'own(coh)':>8} "
          f"{'follow_raw':>10} {'skill(bet)':>10} {'skill(evt)':>10} {'event CI':>18} {'D':>5}")
    for delta, label in DELTAS:
        if delta == 0:
            sk = skill(baseline, ep, l1, l2, rv)
            lo, hi = cluster_boot_ci(sk, ev, rng, n=N_BOOT)
            print(f"  {label:>4} | {'100%':>6} {n_all:>6} {np.unique(ev).size:>6} "
                  f"{effective_breadth(ev):>6.1f} | {(rv - ep).mean():>+8.4f} "
                  f"{(rv - ep).mean():>+10.4f} {sk.mean():>+10.4f} "
                  f"{event_weighted_mean(sk, ev):>+10.4f} "
                  f"[{fmt(lo)},{fmt(hi)}] {design_effect(sk, ev, rng):>5.2f}")
            out[0] = (np.ones(n_all, dtype=bool), sk)
            continue
        pv = np.array([tape.price_at(t, w, s, delta, bandwidth)
                       for t, w, s in zip(tok, wal, ts)])
        c = ~np.isnan(pv)
        n = int(c.sum())
        if n == 0:
            print(f"  {label:>4} | {'0.0%':>6} {0:>6} | (no follower liquidity reaches this Δ)")
            continue
        sk = skill(baseline, pv[c], l1[c], l2[c], rv[c])
        lo, hi = cluster_boot_ci(sk, ev[c], rng, n=N_BOOT)
        print(f"  {label:>4} | {c.mean():>6.1%} {n:>6} {np.unique(ev[c]).size:>6} "
              f"{effective_breadth(ev[c]):>6.1f} | {(rv[c] - ep[c]).mean():>+8.4f} "
              f"{(rv[c] - pv[c]).mean():>+10.4f} {sk.mean():>+10.4f} "
              f"{event_weighted_mean(sk, ev[c]):>+10.4f} "
              f"[{fmt(lo)},{fmt(hi)}] {design_effect(sk, ev[c], rng):>5.2f}")
        out[delta] = (c, sk)
    return out


def section_next_print(bets: pd.DataFrame, tape: TokenTape, baseline: dict, rng) -> None:
    """The most generous anchor there is: take the NEXT other-wallet print at or
    after entry+Δ, however long you have to wait (up to the guard)."""
    print("\n" + "-" * 92)
    print("3b. NEXT-PRINT ANCHOR — unbounded fill window (the friendliest possible copier)")
    print("    price = first other-wallet print after entry+Δ, before the guard. If copying")
    print("    fails HERE it is not failing because the fill window was drawn too narrow.")
    print("-" * 92)
    tok = bets["token_id"].to_numpy()
    wal = bets["wallet"].to_numpy()
    ts = bets["timestamp"].to_numpy(dtype=float)
    ep = bets["entry_price"].to_numpy(dtype=float)
    rv = bets["resolved_value"].to_numpy(dtype=float)
    l1, l2 = bets["niche_l1"].to_numpy(), bets["niche_l2"].to_numpy()
    ev = bets["event"].to_numpy()
    print("    own_* is recomputed on the SAME cohort at every Δ, so the wallet and its")
    print("    follower are always compared on identical bets.")
    print(f"  {'Δ':>4} | {'cov':>6} {'n':>6} | {'wait p50 (s)':>12} | {'own_sk(bet)':>11} "
          f"{'own_sk(evt)':>11} | {'fol_raw':>8} {'fol_sk(bet)':>11} {'fol_sk(evt)':>11} "
          f"{'event CI':>18} {'retain':>7}")
    for delta, label in DELTAS:
        pr, dl = [], []
        for t, w, s in zip(tok, wal, ts):
            p, d, _t = tape.next_print(t, w, s + delta)
            pr.append(p)
            dl.append(d)
        pv, dl = np.asarray(pr), np.asarray(dl)
        c = ~np.isnan(pv)
        if not c.any():
            print(f"  {label:>4} | {'0.0%':>6} — no print")
            continue
        sk = skill(baseline, pv[c], l1[c], l2[c], rv[c])
        own_sk = skill(baseline, ep[c], l1[c], l2[c], rv[c])
        lo, hi = cluster_boot_ci(sk, ev[c], rng, n=N_BOOT)
        o_evt = event_weighted_mean(own_sk, ev[c])
        f_evt = event_weighted_mean(sk, ev[c])
        print(f"  {label:>4} | {c.mean():>6.1%} {int(c.sum()):>6} | "
              f"{np.nanmedian(dl[c]):>12,.0f} | {own_sk.mean():>+11.4f} {o_evt:>+11.4f} | "
              f"{(rv[c] - pv[c]).mean():>+8.4f} {sk.mean():>+11.4f} {f_evt:>+11.4f} "
              f"[{fmt(lo)},{fmt(hi)}] "
              f"{('nan' if o_evt == 0 else f'{f_evt / o_evt:.0%}'):>7}")


def placebo_block_nextprint(bets: pd.DataFrame, tape: TokenTape, baseline: dict,
                            rng, deltas=PLACEBO_DELTAS) -> None:
    """The placebo comparison run on the NEXT-PRINT anchor, which is the only
    anchor with enough coverage on this tape to make the paired test testable at
    all (the bounded-bandwidth version above pairs 0-9 bets — see section 4).

    Anchors, all of them "take the first other-wallet print after time X":
      real      X = entry + Δ            (following the wallet)
      random    X ~ U[token first print, guard cutoff]   (not conditioned on the wallet)
      pre-entry X = entry − Δ            (acted BEFORE the wallet; cannot be copying)

    On a sparse tape the pre-entry anchor often resolves to the SAME print as the
    real one — if no other wallet printed between entry−Δ and the follower's fill,
    both strategies buy the identical fill. That collision share is reported, and
    the difference is re-reported on the DISCORDANT pairs only, because a
    mechanically-zero difference is not evidence of anything."""
    tok = bets["token_id"].to_numpy()
    wal = bets["wallet"].to_numpy()
    ts = bets["timestamp"].to_numpy(dtype=float)
    rv = bets["resolved_value"].to_numpy(dtype=float)
    l1, l2 = bets["niche_l1"].to_numpy(), bets["niche_l2"].to_numpy()
    ev = bets["event"].to_numpy()
    spans = np.array([tape.span(t) for t in tok])

    for delta in deltas:
        label = dict(DELTAS)[delta]
        real = np.array([tape.next_print(t, w, s + delta) for t, w, s in zip(tok, wal, ts)])
        width = np.maximum(spans[:, 1] - spans[:, 0], 0)
        anchor = spans[:, 0] + rng.random(width.size) * width
        rand = np.array([tape.next_print(t, w, a) for t, w, a in zip(tok, wal, anchor)])
        pre = np.array([tape.next_print(t, w, s - delta) for t, w, s in zip(tok, wal, ts)])
        both = ~np.isnan(real[:, 0]) & ~np.isnan(rand[:, 0]) & ~np.isnan(pre[:, 0])
        print(f"\n  [Δ={label}, next-print anchor]  marginal coverage: "
              f"real {np.mean(~np.isnan(real[:, 0])):.1%} | "
              f"random {np.mean(~np.isnan(rand[:, 0])):.1%} | "
              f"pre-entry {np.mean(~np.isnan(pre[:, 0])):.1%}")
        if both.sum() < 10:
            print(f"    PAIRED: only {int(both.sum())} bets carry all three anchors "
                  "— NOT TESTABLE")
            continue
        r = skill(baseline, real[both, 0], l1[both], l2[both], rv[both])
        a = skill(baseline, rand[both, 0], l1[both], l2[both], rv[both])
        p_ = skill(baseline, pre[both, 0], l1[both], l2[both], rv[both])
        e = ev[both]
        d1_lo, d1_hi = cluster_boot_ci(r - a, e, rng, n=N_BOOT)
        d2_lo, d2_hi = cluster_boot_ci(r - p_, e, rng, n=N_BOOT)
        same_pre = float(np.mean(real[both, 2] == pre[both, 2]))
        same_rand = float(np.mean(real[both, 2] == rand[both, 2]))
        print(f"    PAIRED n={int(both.sum()):,} over {np.unique(e).size} events "
              f"(effective {effective_breadth(e):.1f}); D(real−random)="
              f"{design_effect(r - a, e, rng):.2f}")
        print(f"      real {r.mean():+.4f} | random {a.mean():+.4f} | "
              f"pre-entry {p_.mean():+.4f}   (bet-weighted)")
        print(f"      real {event_weighted_mean(r, e):+.4f} | "
              f"random {event_weighted_mean(a, e):+.4f} | "
              f"pre-entry {event_weighted_mean(p_, e):+.4f}   (EVENT-weighted)")
        print(f"      real − random    = {r.mean() - a.mean():+.4f} "
              f"[event CI {fmt(d1_lo)},{fmt(d1_hi)}]   "
              f"evt-wtd {event_weighted_mean(r - a, e):+.4f}   "
              f"(identical fill in {same_rand:.1%} of pairs)")
        print(f"      real − pre-entry = {r.mean() - p_.mean():+.4f} "
              f"[event CI {fmt(d2_lo)},{fmt(d2_hi)}]   "
              f"evt-wtd {event_weighted_mean(r - p_, e):+.4f}   "
              f"(identical fill in {same_pre:.1%} of pairs)")
        disc = real[both, 2] != pre[both, 2]
        if disc.sum() >= 10:
            dl_, dh_ = cluster_boot_ci((r - p_)[disc], e[disc], rng, n=N_BOOT)
            print(f"      real − pre-entry on the {int(disc.sum()):,} DISCORDANT pairs "
                  f"= {(r - p_)[disc].mean():+.4f} [event CI {fmt(dl_)},{fmt(dh_)}]")


def placebo_block(bets: pd.DataFrame, tape: TokenTape, baseline: dict, rng,
                  bandwidth: int) -> None:
    """THE DECISIVE TEST. Identical bets, identical outcomes — only the entry
    anchor moves. If the wallet's entry time carries copyable information, real
    must beat BOTH wallet-agnostic placebos."""
    tok = bets["token_id"].to_numpy()
    wal = bets["wallet"].to_numpy()
    ts = bets["timestamp"].to_numpy(dtype=float)
    rv = bets["resolved_value"].to_numpy(dtype=float)
    l1, l2 = bets["niche_l1"].to_numpy(), bets["niche_l2"].to_numpy()
    ev = bets["event"].to_numpy()

    spans = np.array([tape.span(t) for t in tok])
    for delta in PLACEBO_DELTAS:
        label = dict(DELTAS)[delta]
        pv = np.array([tape.price_at(t, w, s, delta, bandwidth)
                       for t, w, s in zip(tok, wal, ts)])
        lo_a = spans[:, 0]
        hi_a = spans[:, 1] - bandwidth
        width = np.maximum(hi_a - lo_a, 0)
        anchor = lo_a + rng.random(width.size) * width
        pr = np.array([tape.price_at(t, w, s, a - s, bandwidth)
                       for t, w, s, a in zip(tok, wal, ts, anchor)])
        pp = np.array([tape.price_at(t, w, s, -delta, bandwidth)
                       for t, w, s in zip(tok, wal, ts)])
        both = ~np.isnan(pv) & ~np.isnan(pr) & ~np.isnan(pp)
        print(f"\n  [Δ={label}, bandwidth={bandwidth}s]  marginal coverage: "
              f"real {np.mean(~np.isnan(pv)):.1%} | random {np.mean(~np.isnan(pr)):.1%} | "
              f"pre-entry {np.mean(~np.isnan(pp)):.1%}")
        if both.sum() < 10:
            print(f"    PAIRED: only {int(both.sum())} bets carry all three anchors "
                  "— NOT TESTABLE")
            continue
        r = skill(baseline, pv[both], l1[both], l2[both], rv[both])
        a = skill(baseline, pr[both], l1[both], l2[both], rv[both])
        p_ = skill(baseline, pp[both], l1[both], l2[both], rv[both])
        e = ev[both]
        d1_lo, d1_hi = cluster_boot_ci(r - a, e, rng, n=N_BOOT)
        d2_lo, d2_hi = cluster_boot_ci(r - p_, e, rng, n=N_BOOT)
        print(f"    PAIRED n={int(both.sum()):,} over {np.unique(e).size} events "
              f"(effective {effective_breadth(e):.1f}); D(real−random)="
              f"{design_effect(r - a, e, rng):.2f}")
        print(f"      real {r.mean():+.4f} | random {a.mean():+.4f} | "
              f"pre-entry {p_.mean():+.4f}   (bet-weighted)")
        print(f"      real {event_weighted_mean(r, e):+.4f} | "
              f"random {event_weighted_mean(a, e):+.4f} | "
              f"pre-entry {event_weighted_mean(p_, e):+.4f}   (EVENT-weighted)")
        print(f"      real − random    = {r.mean() - a.mean():+.4f} "
              f"[event CI {fmt(d1_lo)},{fmt(d1_hi)}]   "
              f"evt-wtd {event_weighted_mean(r - a, e):+.4f}")
        print(f"      real − pre-entry = {r.mean() - p_.mean():+.4f} "
              f"[event CI {fmt(d2_lo)},{fmt(d2_hi)}]   "
              f"evt-wtd {event_weighted_mean(r - p_, e):+.4f}")


def section_placebos(bets: pd.DataFrame, tape: TokenTape, baseline: dict, rng,
                     bandwidth: int) -> None:
    print("\n" + "-" * 92)
    print("4. PLACEBO ANCHORS — paired on the bets where ALL THREE anchors exist")
    print("   real         = price at entry+Δ (following the wallet)")
    print("   random       = price at a uniformly random moment in the token's pre-guard life")
    print("                  ('buy this market whenever' — not conditioned on the wallet)")
    print("   pre-entry    = price at entry−Δ (a 'follower' who acted BEFORE the wallet and")
    print("                  therefore cannot have been copying it)")
    print("   A positive real edge that does not beat its own placebos is NOT copyability.")
    print("-" * 92)
    placebo_block(bets, tape, baseline, rng, bandwidth)


# ---------------------------------------------------------------------------
# Section 5 — live order book (read-only public GETs)
# ---------------------------------------------------------------------------
def _get(session, url, params=None, timeout=20):
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:  # network is best-effort; never crash the audit
        print(f"    [warn] {url} failed: {exc}")
        return None


def _book_row(session, token_id):
    raw = _get(session, CLOB_BOOK, {"token_id": token_id})
    if not raw:
        return None
    bids = [(float(x["price"]), float(x["size"])) for x in raw.get("bids", [])]
    asks = [(float(x["price"]), float(x["size"])) for x in raw.get("asks", [])]
    return effective_spread(bids, asks)


def section_live_book(wallets: list[str], bets: pd.DataFrame, do_live: bool,
                      sample: int = 24) -> None:
    print("\n" + "-" * 92)
    print("5. LIVE ORDERBOOK REALITY CHECK (read-only public GETs; no keys, no orders)")
    print("-" * 92)
    if not do_live:
        print("  skipped (--no-live)")
        return
    try:
        import requests
    except ImportError:
        print("  requests unavailable — skipped")
        return
    session = requests.Session()
    session.headers.update({"User-Agent": "polymarket-sharps/audit (read-only)"})

    open_tokens = []
    for w in wallets:
        pos = _get(session, DATA_POSITIONS, {"user": w, "limit": 500, "sizeThreshold": 1})
        if pos is None:
            print(f"  {w[:12]}: positions endpoint unavailable")
            continue
        live = [p for p in pos
                if not p.get("redeemable") and 0 < float(p.get("curPrice") or 0) < 1]
        print(f"  {w[:12]}: {len(pos)} positions on record, {len(live)} still open")
        for p in live:
            open_tokens.append((w, p))
    if open_tokens:
        print("\n  books for the currently-open positions:")
        print(f"    {'wallet':>12} {'slug':<44} {'bid':>6} {'ask':>6} {'spread':>7} "
              f"{'ask_sz':>10} {'sz<=+2c':>10}")
        for w, p in open_tokens:
            row = _book_row(session, p["asset"])
            time.sleep(0.1)
            if row is None:
                continue
            print(f"    {w[:12]:>12} {str(p.get('slug'))[:44]:<44} "
                  f"{row['best_bid']:>6.3f} {row['best_ask']:>6.3f} "
                  f"{row['spread']:>7.3f} {row['ask_size']:>10,.0f} "
                  f"{row['ask_size_2c']:>10,.0f}")
        from src.slow_market import derive_categories as _dc
        from src.sports_events import is_sports_market as _iss
        n_sports = 0
        for _w, p in open_tokens:
            s, q = p.get("slug"), p.get("title")
            if _iss(s, q, _dc(pd.Series([s]), pd.Series([q]))[0]):
                n_sports += 1
        print(f"\n  of those, SPORTS positions: {n_sports}")
    else:
        print("\n  NO open positions at all across the three wallets — there is no live")
        print("  position of theirs to price. The substitute probe below measures whether")
        print("  the venue offers takeable size in the kind of market they trade.")

    # substitute probe: live sports books at the entry prices these wallets use
    print("\n  substitute probe — live books in currently-open SPORTS GAME markets")
    print("  (event_kind=='game': the match lines these wallets actually trade, not the")
    print("   long-dated season outrights that dominate a volume-sorted sweep):")
    from src.slow_market import derive_categories
    from src.sports_events import is_sports_market, resolution_event

    rows, offset = [], 0
    while len(rows) < sample and offset < 2000:
        page = _get(session, GAMMA_MARKETS,
                    {"closed": "false", "limit": 100, "offset": offset,
                     "order": "volumeNum", "ascending": "false"})
        offset += 100
        if not page:
            break
        for m in page:
            slug, q = m.get("slug"), m.get("question")
            cat = derive_categories(pd.Series([slug]), pd.Series([q]))[0]
            if not is_sports_market(slug, q, cat):
                continue
            if resolution_event(slug, q, m.get("id"))[1] != "game":
                continue
            toks = m.get("clobTokenIds")
            if isinstance(toks, str):
                try:
                    toks = json.loads(toks)
                except Exception:
                    continue
            if not toks:
                continue
            rows.append((slug, toks[0], m.get("volumeNum"), m.get("fee")))
            if len(rows) >= sample:
                break
    if not rows:
        print("    no open sports GAME markets returned by gamma — skipped")
        return
    print(f"    {'slug':<44} {'bid':>6} {'ask':>6} {'spread':>7} {'ask_sz':>10} "
          f"{'sz<=+2c':>10} {'fee':>6}")
    spreads, sizes, asks = [], [], []
    for slug, tok, _vol, fee in rows:
        row = _book_row(session, tok)
        time.sleep(0.1)
        if row is None:
            continue
        spreads.append(row["spread"])
        sizes.append(row["ask_size"])
        asks.append(row["best_ask"])
        print(f"    {str(slug)[:44]:<44} {row['best_bid']:>6.3f} {row['best_ask']:>6.3f} "
              f"{row['spread']:>7.3f} {row['ask_size']:>10,.0f} "
              f"{row['ask_size_2c']:>10,.0f} {str(fee):>6}")
    if spreads:
        s = np.asarray(spreads, dtype=float)
        z = np.asarray(sizes, dtype=float)
        a = np.asarray(asks, dtype=float)
        print(f"\n    n={s.size} books | median spread {np.nanmedian(s):.4f} "
              f"({np.nanmedian(s) * 100:.2f}¢) | median size at best ask "
              f"{np.nanmedian(z):,.0f} shares | median best ask {np.nanmedian(a):.3f}")
        print("    taker fee model (docs/project4_value_gate.md): shares × k × price × (1−price);")
        print("    k is not pinned in this repo, so the fee is reported as the p(1−p) factor only.")
        print(f"    fee factor p(1−p) at the median ask: {np.nanmedian(a) * (1 - np.nanmedian(a)):.4f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-ledger", action="store_true",
                    help="skip the bet_ledger tape enrichment")
    ap.add_argument("--no-live", action="store_true",
                    help="skip section 5 (network)")
    ap.add_argument("--tier", default=TIER)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    cfg = load_config()
    guard = cfg["scoring"].get("fair_value_resolution_guard", 0.2)
    cw_hours = cfg["scoring"].get("copy_window_hours", 24)

    print("=" * 92)
    print("METRIC B — COPYABILITY OF THE SLOW-CADENCE SPORTS WALLETS (tier "
          f"{args.tier})")
    print("=" * 92)

    man = load_manifest()
    wallets = list(man["tiers"][args.tier]["wallets"])
    baseline = man["baseline"]
    print(f"freeze {man['freeze_utc']} @ {man['git_commit'][:12]} | tier {args.tier} "
          f"= {len(wallets)} wallets")
    print(f"baseline: FROZEN, copied verbatim, levels={baseline['levels']}, "
          f"n_bets={baseline['n_bets']:,}, ks={ {k: round(v, 2) for k, v in baseline['ks'].items()} }")
    print(f"guard={guard} | copy_window_hours={cw_hours} | seed={SEED}")
    for w in wallets:
        print(f"  subject {w}")

    idx = sports_market_index()
    bets = load_bets(wallets, idx)
    print(f"\nsubjects' resolved BUY SPORTS bets: {len(bets):,} across "
          f"{bets['market_id'].nunique():,} markets -> {bets['event'].nunique():,} events "
          f"({bets['timestamp'].min()} .. {bets['timestamp'].max()})")

    tape_df, tstats = load_tape(bets["token_id"].unique(), not args.no_ledger)
    print(f"tape on those {bets['token_id'].nunique():,} tokens: deep "
          f"{tstats['deep_rows']:,} prints / {tstats['deep_wallets']} wallets"
          + (f" + ledger {tstats['ledger_rows']:,} prints / {tstats['ledger_wallets']} "
             f"wallets ({tstats['added_rows']:,} new after dedupe)"
             if not args.no_ledger else " (ledger enrichment OFF)"))
    print("LIMIT: this tape is TRACKED WALLETS' prints, not the order book. Every")
    print("coverage number below is a LOWER BOUND and every follower price is a biased")
    print("sample of what was actually quotable. Section 5 is the unbiased check.")
    tape = TokenTape(tape_df, guard)
    del tape_df

    section_coverage(bets, tape, rng)
    section_copy_window(bets, tape, baseline, cw_hours)

    print("\n" + "-" * 92)
    print("3. LATENCY DECAY — follower skill edge vs Δ, coverage reported ALONGSIDE")
    print("   own(coh) = the wallet's own raw edge on the SAME bets (same-cohort, so cohort")
    print("   selection cannot masquerade as decay). skill = resolved − E[outcome|price,")
    print("   league, league|form] under the frozen baseline. CI resamples EVENTS.")
    print("   D = design effect (cluster var / iid var); D<1 would mean over-constrained.")
    print("-" * 92)
    for bw in BANDWIDTHS:
        section_decay(bets, tape, baseline, rng, bw)

    section_next_print(bets, tape, baseline, rng)
    section_placebos(bets, tape, baseline, rng, PRIMARY_BANDWIDTH)

    print("\n" + "-" * 92)
    print("4b. PLACEBOS ON THE NEXT-PRINT ANCHOR — THE DECISIVE TEST")
    print("    The bounded-bandwidth placebos above are not testable on this tape (0-9 paired")
    print("    bets). The next-print anchor is the only one with enough coverage, and it is")
    print("    also the most generous to copying, so this is the strongest form of the test.")
    print("-" * 92)
    placebo_block_nextprint(bets, tape, baseline, rng)

    print("\n" + "-" * 92)
    print("4c. PER-WALLET, next-print anchor (n=3 wallets — DIRECTIONAL, not decisive)")
    print("-" * 92)
    for w, g in bets.groupby("wallet"):
        print(f"\n  == {w}  (n={len(g)}, events={g['event'].nunique()})")
        placebo_block_nextprint(g, tape, baseline, rng)

    section_live_book(wallets, bets, not args.no_live)

    print("\n" + "=" * 92)
    print("Verdict lives in docs/project3_sports_metric_b.md. This script writes NOTHING.")
    print("=" * 92)


if __name__ == "__main__":
    main()
