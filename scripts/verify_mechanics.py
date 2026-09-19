#!/usr/bin/env python3
"""Re-verify the load-bearing claims in docs/polymarket_mechanics.md.

READ-ONLY. Public unauthenticated GETs plus a public Polygon RPC. No keys, no
orders, no writes to data/. Run this when you want to know whether the mechanics
reference has gone stale.

    python3 scripts/verify_mechanics.py            # fast checks only
    python3 scripts/verify_mechanics.py --onchain  # also decode fees on-chain

Each check prints PASS / FAIL / INFO and the evidence. A FAIL means the doc needs
updating, not necessarily that anything is broken.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import urllib.request

CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
RPC = "https://polygon.drpc.org"

# CTFExchangeV2 / NegRiskCTFExchangeV2 -- see docs/polymarket_mechanics.md sec.10
EXCHANGES_V2 = {
    "0xe111180000d2663c0091e4f400237545b87b996b": "CTFExchangeV2",
    "0xe2222d279d744050d28e00520010520000310f59": "NegRiskCTFExchangeV2",
}
ORDER_FILLED_PREFIX = "0xd543adfd945773f1a6"
FEE_TOPIC = "0x55bb3cade9d43b798a4fe5ffdd05024b2d7870df53920673bfc7e68047cd0ab1"

# Documented taker fee rates by category (docs.polymarket.com/trading/fees).
DOC_FEE_RATES = {0.04, 0.05, 0.07}

_results: list[tuple[str, str]] = []


def _get(url: str, timeout: int = 40, tries: int = 3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pm-sharps-verify/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _rpc(method: str, params: list):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    try:
        req = urllib.request.Request(
            RPC, data=body,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read().decode()).get("result")
    except Exception:
        return None


def report(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name))
    print(f"[{status:^4}] {name}")
    for line in detail.splitlines():
        if line.strip():
            print(f"         {line}")


def live_markets(limit: int = 500) -> list:
    return _get(f"{GAMMA}/markets?closed=false&archived=false&limit={limit}") or []


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_fee_schedule(mk: list) -> None:
    """feeSchedule is the real fee source; *_base_fee is the 10% ceiling."""
    sched = [m["feeSchedule"] for m in mk if m.get("feeSchedule")]
    if not sched:
        report("FAIL", "feeSchedule present on live markets", "no market returned a feeSchedule")
        return
    rates = collections.Counter(s.get("rate") for s in sched)
    taker_only = all(s.get("takerOnly") for s in sched)
    unknown = set(rates) - DOC_FEE_RATES
    detail = f"rates seen: {dict(rates)}\ntakerOnly on all: {taker_only}"
    report("PASS" if not unknown and taker_only else "FAIL",
           "feeSchedule.rate matches documented categories (0.04/0.05/0.07)",
           detail + (f"\nUNDOCUMENTED RATES: {unknown}" if unknown else ""))

    base = collections.Counter(m.get("takerBaseFee") for m in mk if m.get("takerBaseFee") is not None)
    only_ceiling = set(base) <= {1000}
    report("PASS" if only_ceiling else "INFO",
           "takerBaseFee is the flat 1000 bps ceiling, not a rate",
           f"takerBaseFee values: {dict(base)}\n"
           f"(1000 == MAX_FEE_RATE_BIPS in ctf-exchange Fees.sol; never use it as a rate)")


def check_tick_size(mk: list, sample: int = 40) -> None:
    """Tick size is per-market config; static metadata == live CLOB value."""
    checked = mismatched = 0
    seen: collections.Counter = collections.Counter()
    for m in mk:
        if checked >= sample:
            break
        if not (m.get("enableOrderBook") and m.get("acceptingOrders")):
            continue
        try:
            tid = json.loads(m["clobTokenIds"])[0]
        except Exception:
            continue
        live = _get(f"{CLOB}/tick-size?token_id={tid}")
        if not isinstance(live, dict) or "minimum_tick_size" not in live:
            continue
        checked += 1
        lt = live["minimum_tick_size"]
        seen[lt] += 1
        if float(lt) != float(m.get("orderPriceMinTickSize") or -1):
            mismatched += 1
        time.sleep(0.03)
    report("PASS" if checked and not mismatched else "FAIL",
           "static orderPriceMinTickSize == live CLOB tick-size",
           f"checked {checked} markets, {mismatched} mismatched\n"
           f"live tick values: {dict(seen)}  (tick is PER-MARKET, never a constant)")


def check_min_order_size(mk: list) -> None:
    sizes = collections.Counter(
        m.get("orderMinSize") for m in mk if m.get("enableOrderBook") and m.get("acceptingOrders")
    )
    report("INFO", "orderMinSize on live order-book markets", f"{dict(sizes)}  (doc says 5 shares)")


def check_end_date(mk: list) -> None:
    """endDateIso is a bare DATE; end_date_iso is that date + T00:00:00Z."""
    iso = [m.get("endDateIso") for m in mk if m.get("endDateIso")]
    date_only = sum(1 for v in iso if "T" not in str(v))
    report("PASS" if iso and date_only == len(iso) else "FAIL",
           "endDateIso is a bare calendar date (NOT a timestamp)",
           f"{date_only}/{len(iso)} are date-only -- never use as a time boundary")


def check_book_mirror(mk: list) -> None:
    """The NO book is the exact mirror of the YES book, reflected at 1.0."""
    for m in sorted(mk, key=lambda x: -(x.get("liquidityNum") or 0)):
        try:
            tids = json.loads(m["clobTokenIds"])
        except Exception:
            continue
        if len(tids) != 2:
            continue
        yes = _get(f"{CLOB}/book?token_id={tids[0]}")
        no = _get(f"{CLOB}/book?token_id={tids[1]}")
        if not (yes and no and yes.get("bids") and no.get("asks")):
            continue
        ybid = {(round(1 - float(x["price"]), 6), float(x["size"])) for x in yes["bids"]}
        nask = {(round(float(x["price"]), 6), float(x["size"])) for x in no["asks"]}
        ok = bool(ybid) and ybid == nask
        report("PASS" if ok else "INFO",
               "NO book mirrors YES book (buying NO == selling YES)",
               f"market: {m['slug'][:60]}\n"
               f"YES bids reflected at 1.0: {sorted(ybid)[:3]}\n"
               f"NO  asks                 : {sorted(nask)[:3]}")
        return
    report("INFO", "NO book mirrors YES book", "no suitable two-sided market found")


def check_trade_tape() -> None:
    """price is a VWAP across many fills; there is no fee field."""
    tr = _get(f"{DATA}/trades?limit=500") or []
    if not tr:
        report("FAIL", "data-api /trades reachable", "no rows returned")
        return
    fee_fields = [k for k in tr[0] if "fee" in k.lower()]
    report("PASS" if not fee_fields else "INFO",
           "data-api /trades has NO fee field (fees must be modelled)",
           f"fields: {sorted(tr[0].keys())}")

    off = sum(1 for t in tr if abs(round(t["price"] / 0.001) * 0.001 - t["price"]) > 1e-7)
    report("PASS" if off else "INFO",
           "tape price is a size-weighted VWAP, not a book level",
           f"{off}/{len(tr)} prints sit off the 0.001 tick grid "
           f"(e.g. {[t['price'] for t in tr if abs(round(t['price']/0.001)*0.001-t['price'])>1e-7][:3]})")


def check_onchain_fee(n: int = 12) -> None:
    """Solve k = fee / (shares * p * (1-p)) against real settled transactions."""
    tr = _get(f"{DATA}/trades?limit=300") or []
    ks, fills, exchanges, done = [], [], collections.Counter(), 0
    for t in tr:
        if done >= n:
            break
        rec = _rpc("eth_getTransactionReceipt", [t["transactionHash"]])
        if not rec:
            continue
        exchanges[EXCHANGES_V2.get(rec["to"].lower(), rec["to"].lower())] += 1
        fee_raw = sum(
            int(l["data"][2:66], 16) for l in rec["logs"] if l["topics"][0] == FEE_TOPIC
        )
        n_of = sum(
            1 for l in rec["logs"]
            if l["address"].lower() in EXCHANGES_V2 and l["topics"][0].startswith(ORDER_FILLED_PREFIX)
        )
        denom = t["size"] * t["price"] * (1 - t["price"])
        if denom > 0 and fee_raw:
            ks.append(fee_raw / 1e6 / denom)
        if n_of:
            fills.append(n_of)
        done += 1
        time.sleep(0.05)

    if not ks:
        report("INFO", "on-chain taker fee", "no fee events decoded (RPC may be rate-limited)")
        return
    rounded = collections.Counter(round(k, 2) for k in ks)
    ok = set(rounded) <= DOC_FEE_RATES
    report("PASS" if ok else "FAIL",
           "on-chain fee matches shares * k * p * (1-p) with documented k",
           f"implied k (2dp): {dict(rounded)}\n"
           f"exchange contracts used: {dict(exchanges)}")
    if fills:
        report("PASS" if max(fills) > 1 else "INFO",
               "one /trades row aggregates MANY on-chain fills",
               f"OrderFilled events per tape row: {dict(collections.Counter(fills))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onchain", action="store_true",
                    help="also decode fees from Polygon receipts (slower)")
    ap.add_argument("--tick-sample", type=int, default=40)
    args = ap.parse_args()

    print("Verifying docs/polymarket_mechanics.md against live Polymarket APIs")
    print("read-only: public GETs + public Polygon RPC\n")

    mk = live_markets()
    if not mk:
        print("could not reach gamma-api; aborting")
        return 2
    print(f"live markets fetched: {len(mk)}\n")

    check_fee_schedule(mk)
    check_tick_size(mk, args.tick_sample)
    check_min_order_size(mk)
    check_end_date(mk)
    check_book_mirror(mk)
    check_trade_tape()
    if args.onchain:
        check_onchain_fee()

    fails = [n for s, n in _results if s == "FAIL"]
    print(f"\n{len(_results)} checks | {len(fails)} FAIL")
    for n in fails:
        print(f"  FAIL: {n}")
    if fails:
        print("\nA FAIL means docs/polymarket_mechanics.md is stale -- update it.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
