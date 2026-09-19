"""Local, read-only server for the forensics dashboard with market drill-down.

The standalone HTML file carries per-wallet aggregates only: full per-bet detail
for the scored population is 3.27M rows (~129 MB as JSON) and cannot be inlined.
Served instead, one wallet at a time, it is ~60 KB a request. Nothing leaves the
machine: binds 127.0.0.1 by default, opens the parquets read-only, and has no
write path of any kind.

Run:  .venv/bin/python scripts/serve_forensics.py
      → http://127.0.0.1:8765/

  --port N      listen on another port (default 8765)
  --host H      bind address. Default 127.0.0.1 (this machine only). Pass
                0.0.0.0 to reach it from a phone on the same wifi — that exposes
                it to everyone on the network, so only do it on a network you trust.

Endpoints:
  GET /                     the dashboard (built on demand if missing)
  GET /api/wallet/<address> that wallet's markets and every bet in each
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paper_rw import fee_k_for_category            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data" / "interim" / "forensics"
RW = ROOT / "data" / "interim" / "realworld"
HTML = ROOT / "forensics_dashboard.html"
ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{6,64}$")

_INDEX = json.loads((F / "shard_index.json").read_text())
_POP = json.loads((F / "population.json").read_text())
_EDGES = np.array(_POP["baseline_edges"], dtype=float)
_MEANS = np.array(_POP["baseline_means"], dtype=float)
_CATS = pd.read_parquet(RW / "market_category.parquet")
_CAT_OF = dict(zip(_CATS["market_id"], _CATS["category"]))


def expected(price: np.ndarray) -> np.ndarray:
    """E[outcome | entry price] from the same 20-bin baseline the dashboard uses."""
    i = np.clip(np.searchsorted(_EDGES, price, side="right") - 1, 0, _MEANS.size - 1)
    return _MEANS[i]


@lru_cache(maxsize=12)
def _shard(i: int) -> pd.DataFrame:
    """One deep_trades shard, resolved real-world BUYs only. Cached: consecutive
    lookups of wallets in the same shard cost one read."""
    d = pd.read_parquet(ROOT / _INDEX["shards"][i],
                        columns=["wallet", "market_id", "outcome", "side",
                                 "entry_price", "size", "timestamp", "resolved",
                                 "resolved_value", "question", "slug"])
    d = d[(d["side"] == "BUY") & d["resolved"].fillna(False)
          & d["resolved_value"].notna()]
    keep = np.array([_CAT_OF.get(x, "other") != "micro_crypto" for x in d["market_id"]])
    return d.loc[keep].reset_index(drop=True)


def wallet_detail(addr: str) -> dict:
    ids = _INDEX["wallets"].get(addr)
    if not ids:
        return {"error": "wallet not in the scored population"}
    d = pd.concat([_shard(i) for i in ids], ignore_index=True)
    d = d[d["wallet"] == addr]
    if not len(d):
        return {"error": "no resolved real-world bets"}

    price = d["entry_price"].to_numpy(float)
    rv = d["resolved_value"].to_numpy(float)
    size = np.nan_to_num(d["size"].to_numpy(float), nan=0.0)
    cats = [_CAT_OF.get(m, "other") for m in d["market_id"]]
    # A share pays $1 if the outcome happens and $0 if not, so profit on a bet is
    # shares x (outcome - price): 50 shares at 36c returns +$32 on a win, -$18 on a
    # loss. Fee is the frozen taker rate k*p*(1-p) per share, charged either way —
    # this repo spent months reporting gross numbers with k=0, so net is shown
    # beside gross everywhere rather than left to the reader.
    k = np.array([fee_k_for_category(c) for c in cats], dtype=float)
    fee_usd = k * size * price * (1.0 - price)
    d = d.assign(resid_c=100.0 * (rv - expected(price)),
                 raw_c=100.0 * (rv - price),
                 category=cats,
                 stake_usd=size * price,
                 profit_usd=size * (rv - price),
                 fee_usd=fee_usd,
                 net_usd=size * (rv - price) - fee_usd)

    mcode = {m: i for i, m in enumerate(d["market_id"].unique())}
    d = d.assign(mc=[mcode[m] for m in d["market_id"]])

    label = {}
    for mc, q, s in zip(d["mc"], d["question"], d["slug"]):
        if mc not in label:
            label[mc] = q if isinstance(q, str) and q.strip() else (
                s if isinstance(s, str) else "(unnamed market)")

    g = d.groupby("mc", observed=True)
    markets = []
    for mc, gg in g:
        markets.append([
            int(mc), label[mc], gg["category"].iloc[0], int(len(gg)),
            int((gg["resolved_value"] > 0.5).sum()),
            round(float(gg["entry_price"].mean()), 4),
            round(float(gg["resid_c"].mean()), 2),
            round(float(gg["raw_c"].mean()), 2),
            round(float(gg["stake_usd"].sum()), 2),
            int(gg["timestamp"].min()),
            round(float(gg["profit_usd"].sum()), 2),      # [10] gross profit $
            round(float(gg["net_usd"].sum()), 2),         # [11] after taker fee
            round(float(gg["fee_usd"].sum()), 2),         # [12] fee $
        ])
    markets.sort(key=lambda r: (-r[3], r[9]))

    bets: dict[str, list] = {}
    for mc, gg in g:
        gg = gg.sort_values("timestamp")
        bets[str(int(mc))] = [
            [int(t), round(float(p), 4), round(float(sz), 2), float(v),
             (o if isinstance(o, str) else ""), round(float(rc), 2),
             round(float(pr), 2), round(float(fe), 2), round(float(ne), 2)]
            for t, p, sz, v, o, rc, pr, fe, ne in zip(
                gg["timestamp"], gg["entry_price"], gg["size"],
                gg["resolved_value"], gg["outcome"], gg["resid_c"],
                gg["profit_usd"], gg["fee_usd"], gg["net_usd"])
        ]
    return {"wallet": addr, "n_bets": int(len(d)), "n_markets": len(markets),
            "markets": markets, "bets": bets, "curve": equity_curve(d, label),
            "totals": {"stake": round(float(d["stake_usd"].sum()), 2),
                       "profit": round(float(d["profit_usd"].sum()), 2),
                       "fee": round(float(d["fee_usd"].sum()), 2),
                       "net": round(float(d["net_usd"].sum()), 2)}}


MAX_CURVE_POINTS = 2500


def equity_curve(d: pd.DataFrame, label: dict) -> dict:
    """Cumulative skill edge and cumulative gross dollars, in bet order.

    The year panel cannot tell a steady slope from a step function, and that is the
    whole "is this record one lucky event" question. Downsampling keeps every large
    single-bet move (the steps ARE the signal), so a spike is never smoothed away.
    Dollars are GROSS: size x (outcome - price), before the taker fee.
    """
    s = d.sort_values("timestamp")
    ts = s["timestamp"].to_numpy(np.int64)
    cum_c = np.cumsum(s["resid_c"].to_numpy(float))
    usd = s["size"].to_numpy(float) * (s["resolved_value"].to_numpy(float)
                                       - s["entry_price"].to_numpy(float))
    cum_usd = np.cumsum(usd)

    keep = np.arange(len(ts))
    if len(ts) > MAX_CURVE_POINTS:
        step = int(np.ceil(len(ts) / MAX_CURVE_POINTS))
        idx = set(range(0, len(ts), step)) | {len(ts) - 1}
        # never drop the biggest moves — a step function must survive downsampling
        big = np.argsort(-np.abs(usd))[:200]
        idx |= set(int(i) for i in big)
        idx |= set(int(i) for i in np.argsort(-np.abs(s["resid_c"].to_numpy()))[:200])
        keep = np.array(sorted(idx))

    # which resolution event contributes most to the cumulative skill edge
    by_ev = s.groupby("mc", observed=True)["resid_c"].sum()
    top_mc = int(by_ev.idxmax()) if len(by_ev) else None
    top = None
    if top_mc is not None:
        g = s[s["mc"] == top_mc]
        top = {"label": label.get(top_mc, ""), "mc": top_mc,
               "contrib_c": round(float(by_ev.max()), 1),
               "share": round(float(by_ev.max() / max(cum_c[-1], 1e-9)), 3),
               "t0": int(g["timestamp"].min()), "t1": int(g["timestamp"].max()),
               "n": int(len(g))}
    return {"ts": [int(x) for x in ts[keep]],
            "cum_c": [round(float(x), 1) for x in cum_c[keep]],
            "cum_usd": [round(float(x), 2) for x in cum_usd[keep]],
            "n_points": len(keep), "n_bets": len(ts), "top_event": top}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                      # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            if not HTML.exists():
                subprocess.run([sys.executable, str(ROOT / "scripts" /
                                "build_forensics_dashboard.py"), str(HTML)], check=True)
            self._send(HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path.startswith("/api/wallet/"):
            addr = path[len("/api/wallet/"):]
            if not ADDR_RE.match(addr):            # reject anything not an address
                self._send(b'{"error":"bad address"}', "application/json", 400)
                return
            try:
                body = json.dumps(wallet_detail(addr), separators=(",", ":")).encode()
            except Exception as exc:               # never take the server down
                body = json.dumps({"error": str(exc)}).encode()
            self._send(body, "application/json")
            return
        self._send(b"not found", "text/plain", 404)

    def log_message(self, fmt, *args):             # one tidy line per request
        sys.stderr.write("  %s\n" % (fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    where = "this machine only" if a.host == "127.0.0.1" else \
            "REACHABLE FROM THE NETWORK — anyone on this wifi can open it"
    print(f"forensics dashboard → http://{a.host}:{a.port}/   ({where})")
    print(f"  {len(_INDEX['wallets']):,} wallets indexed · read-only · Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
