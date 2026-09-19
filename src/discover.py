"""Market-first discovery of real-world (non-micro-crypto) markets and the
wallets that trade them — the binding gap Project 2 attacks (see
`docs/project2_forecaster_discovery.md` §1 and `docs/project2_verify_gate_findings.md`).

The global `/trades` feed is ~95% 5-minute micro-crypto, so a real-world
forecaster who bets twice a month essentially never appears in it. Instead of
watching the firehose, this module enumerates real-world markets directly
(Gamma, volume-ranked) and pulls each market's `/trades?market=` tape:
  - every wallet that traded it   -> discovery of real-world forecasters
  - the whole market's tape       -> the surrounding book activity metric B
    (copyability) needs, which per-user backfill cannot give.

READ-ONLY / ANALYSIS-ONLY, same discipline as `ingest.py` / `backfill.py`:
public unauthenticated GETs, no keys, nothing placed on-chain.

ISOLATION (important): this writes ONLY to `data/interim/discovery/`. It does
NOT touch the shared `bet_ledger.parquet`, `backfill_cursors.json`, or
`resolutions.parquet` — merging the discovery tape into the main ledger and
enriching it against the shared (read-only) resolutions cache are deliberate,
trivial downstream steps, kept separate so discovery can run fully decoupled.

Verify-gate facts this module is built on (see the findings doc for evidence):
  - Gamma `/markets?order=volumeNum&ascending=false` sorts by volume; `order=volume`
    returns garbage; `limit>100` is silently capped at 100; offset paging is clean
    but hits a ~2400 offset ceiling (HTTP 422 -> use `/markets/keyset`, deferred).
  - Server-side volume *filters* (`volume_num_min`, ...) are IGNORED -> filter
    volume client-side.
  - `/trades?market=<conditionId>` filters exactly (0 wrong-market rows) and hits
    the same ~10,000 offset cap as the global feed (offset>10000 -> HTTP 400).
  - Micro-crypto is identified by SLUG (`updown` / `up-or-down` / `<coin>-updown-5m-`),
    never by metadata lifespan (5-minute markets report a ~24h start->end window).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.common import (
    INTERIM_DIR,
    RESOLUTIONS_PATH,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
    print_disk_usage_summary,
)
from src.ingest import (
    fold_trades_to_ledger,
    load_resolutions_cache,
    trade_key,
    update_resolutions,
)

# --- Constants (kept local, NOT in config/config.yaml, which has pending edits
#     from another workstream). Mirror the spec's §1.7 config sketch. ---------
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"

ENUM_ORDER = "volumeNum"          # VERIFIED sort field (NOT "volume", which is garbage)
GAMMA_PAGE_SIZE = 100             # effective cap: limit>100 silently returns 100
GAMMA_OFFSET_CEILING = 2400       # /markets 422s at offset>=~2400 ("use /markets/keyset")
MARKETS_PER_RUN = 300             # bound each run; loop nightly until universe covered
MIN_MARKET_VOLUME = 5000.0        # skip near-empty markets (client-side; server filter ignored)
INCLUDE_OPEN_MARKETS = True       # closed=false too -> early-tape capture + live watch layer
TRADES_PAGE_SIZE = 500            # /trades?market= page size (data-api)
MAX_PAGES_PER_MARKET = 21         # 21*500 covers the shared ~10k offset cap (offset<=10000)
REQUEST_TIMEOUT_SEC = 20
SLEEP_BETWEEN_REQUESTS_SEC = 0.15  # polite: shared API, aggregate ~20-22 req/s cap
SKIP_MICRO_CRYPTO = True          # target real-world tapes by default (still recorded in universe)

# --- Events-based enumeration (§1.4 prong 2: the mid/low-volume FULL-tape tail).
# The volume-ranked /markets offset sweep can ONLY reach the top ~2000 markets
# (all >= ~$4.5M) and EVERY one of those hits the ~10.5k-trade offset cap — so its
# retrievable tape is the near-resolution slice (verified live 2026-07-20). The
# full-tape markets live BELOW ~$4.5M, and offset paging can't reach them:
# ascending order is all $0 dead markets, descending 422s at the ~$4.5M ceiling.
# `/events?tag_slug=` is the ONLY path to that band — it surfaces hundreds of
# mid-volume markets per category (nfl 435, nba 707, politics 567, soccer 787 in
# the $100k-$4M closed band on a single probe) WITH full, early-entry-intact tapes.
EVENT_TAG_SLUGS = ["nfl", "nba", "soccer", "politics", "economy", "mlb", "nhl",
                   "ufc", "tennis", "crypto"]
EVENTS_ORDER = "volume"           # /events sorts by `volume` (NOT `volumeNum`, which is the /markets field)
EVENTS_VOLUME_MIN = 50_000.0      # skip near-dead markets (thin liquidity -> no copyability signal)
EVENTS_VOLUME_MAX = 4_000_000.0   # favor full tapes: markets above ~$4M start hitting the 10.5k cap
EVENTS_MAX_PAGES_PER_TAG = 40     # bound work per tag (40*100 = 4000 events/tag scanned)

# tag_slug -> category, used ONLY to rescue a market the keyword classifier scored
# "other" (a terse player-prop slug the classifier misses is still known to be NFL
# because we enumerated it under tag_slug=nfl). Never overrides a confident keyword
# match, so the classifier stays the primary, consistent labeler (spec §1.3).
_TAG_CATEGORY = {
    "nfl": "sports_nfl", "nba": "sports_nba", "soccer": "sports_soccer",
    "mlb": "sports_mlb", "nhl": "sports_nhl", "ufc": "sports_ufc",
    "tennis": "sports_tennis", "cricket": "sports_cricket",
    "politics": "politics", "economy": "econ_macro", "crypto": "crypto_event",
    "sports": "sports_other", "culture": "culture",
}

# Discovery-only outputs (isolated from the shared pipeline state).
DISCOVERY_DIR = INTERIM_DIR / "discovery"
MARKET_UNIVERSE_PATH = DISCOVERY_DIR / "market_universe.parquet"
DISCOVERY_TRADES_PATH = DISCOVERY_DIR / "discovery_trades.parquet"
DISCOVER_CURSORS_PATH = DISCOVERY_DIR / "discover_cursors.json"
# Resolutions we fetch ourselves for discovery markets that the SHARED, read-only
# cache (data/raw/resolutions.parquet) doesn't already have. We NEVER write the
# shared cache — only this isolated discovery-owned file.
DISCOVERY_RESOLUTIONS_PATH = DISCOVERY_DIR / "discovery_resolutions.parquet"
RESOLUTION_COLUMNS = ["market_id", "token_id", "outcome", "resolved", "resolved_value", "closed"]
# Per-market tape-completeness record. Its one load-bearing signal is `hit_cap`:
# whether a market's tape was truncated at the ~10.5k offset cap (its early entries
# unreachable by historical pull -> near-resolution-biased). Sticky (OR across
# pulls): once truncated, later incremental top-ups never clear it. Lets scoring
# segment honest full-tape skill from near-resolution-inflated skill (§1.2). Markets
# pulled before this file existed fall back to the n_trades>=cap heuristic.
MARKET_TAPE_STATS_PATH = DISCOVERY_DIR / "market_tape_stats.parquet"
TAPE_STATS_COLUMNS = ["market_id", "hit_cap", "last_pulled"]
# Tape-length heuristic for legacy markets with no tape-stats row: the API serves
# at most ~10,500 rows/market (offset<=10000, page 500), so a market whose tape
# holds >= this many trades was truncated at the cap; fewer => fully retrieved.
TAPE_CAP_ROWS = 10450

DISCOVERY_LEDGER_COLUMNS = [
    "wallet", "market_id", "token_id", "outcome", "side", "entry_price",
    "size", "timestamp", "resolved", "resolved_value", "question", "slug",
    "tx_hash", "category",
]
MARKET_UNIVERSE_COLUMNS = [
    "market_id", "slug", "question", "category", "volume", "closed",
    "start_date", "end_date", "first_seen", "last_seen",
]

# ---------------------------------------------------------------------------
# Pure logic (unit-tested; no network)
# ---------------------------------------------------------------------------

# Ordered keyword rules -> category. First match wins; unknown -> "other".
# Sub-category granularity matters: "sharp at NBA" and "sharp at politics" are
# different skills. Conservative: never force a guess (spec §1.3).
_SPORTS_LEAGUE_KEYWORDS = {
    "sports_nfl": ("nfl", "super bowl", "super-bowl"),
    "sports_nba": ("nba", "playoffs", "finals"),  # 'finals'/'playoffs' overlap; NBA-leaning
    "sports_soccer": ("premier league", "premier-league", "la liga", "la-liga",
                       "champions league", "champions-league", "fifa", "world cup",
                       "world-cup", "epl", "serie a", "bundesliga", "mls", "uefa"),
    "sports_mlb": ("mlb", "world series", "world-series"),
    "sports_ufc": ("ufc", "mma"),
    "sports_nhl": ("nhl", "stanley cup", "stanley-cup"),
    "sports_cricket": ("ipl", "cricket"),
    "sports_tennis": ("wimbledon", "atp", "wta", "us open", "roland garros"),
}
_POLITICS_KEYWORDS = (
    "election", "president", "presidential", "senate", "congress", "governor",
    "prime minister", "parliament", "primary", "republican", "democrat",
    "inaugurat", "impeach", "nominee", "vote", "referendum", "poll",
)
_ECON_KEYWORDS = (
    "fed ", "interest rate", "interest-rate", "rate hike", "rate cut", "fomc",
    "cpi", "inflation", "gdp", "jobs report", "unemployment", "recession",
    "basis points", "bps", "nonfarm", "payroll",
)
# Long-horizon crypto EVENTS (distinct from micro up/down): "BTC hits $X by <date>".
_CRYPTO_EVENT_KEYWORDS = (
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "microstrategy",
    "coinbase", "xrp", "dogecoin", "hits $", "reach $", "all-time high", "flippening",
)
_CULTURE_KEYWORDS = (
    "album", "movie", "oscar", "grammy", "box office", "box-office", "gta",
    "spotify", "netflix", "taylor swift", "celebrity", "twitter", "tweet", "song",
)


def is_micro_crypto(slug: str | None, question: str | None) -> bool:
    """The 5-minute BTC/ETH 'up or down' coin-flips. Keyed on the slug shape
    (`updown` / `up-or-down` / `<coin>-updown-5m-<ts>`), NOT metadata lifespan —
    those markets report a ~24h start->end window (verify-gate finding)."""
    s = (slug or "").lower()
    q = (question or "").lower()
    return (
        "updown" in s
        or "up-or-down" in s
        or "up or down" in q
        or "-updown-" in s
    )


def classify_market(
    slug: str | None,
    question: str | None,
    sports_market_type: str | None = None,
) -> str:
    """Map a market to a single best category label. Ordered, conservative,
    keyword-based (the reliable signal per the verify gate); unknown -> 'other'.

    `sports_market_type` (Gamma `sportsMarketType`) is a weak positive sports
    signal only — it is None on sports futures, so it never vetoes a non-sports
    label, it only rescues an otherwise-unknown market into generic 'sports'.
    """
    s = (slug or "").lower()
    q = (question or "").lower()
    # Normalize hyphens to spaces so space-delimited keywords ("fed ", "world
    # cup") match slugs, which are hyphen-delimited ("fed-cuts-rates").
    text = f"{s} {q}".replace("-", " ")

    if is_micro_crypto(slug, question):
        return "micro_crypto"

    for league, kws in _SPORTS_LEAGUE_KEYWORDS.items():
        if any(k in text for k in kws):
            return league

    if any(k in text for k in _POLITICS_KEYWORDS):
        return "politics"
    if any(k in text for k in _ECON_KEYWORDS):
        return "econ_macro"
    if any(k in text for k in _CRYPTO_EVENT_KEYWORDS):
        return "crypto_event"
    if any(k in text for k in _CULTURE_KEYWORDS):
        return "culture"

    # Weak sports rescue: only when nothing else matched.
    if sports_market_type:
        return "sports_other"

    return "other"


def is_real_world(category: str) -> bool:
    """Everything that is not the un-copyable micro-crypto slice."""
    return category != "micro_crypto"


def parse_market_row(m: dict) -> dict | None:
    """Extract the fields we persist from a raw Gamma market dict. Returns None
    if it lacks a conditionId (nothing to key trades on)."""
    cid = m.get("conditionId")
    if not cid:
        return None
    vol = m.get("volumeNum")
    try:
        vol = float(vol) if vol is not None else 0.0
    except (TypeError, ValueError):
        vol = 0.0
    slug = m.get("slug")
    question = m.get("question")
    category = classify_market(slug, question, m.get("sportsMarketType"))
    return {
        "market_id": cid,
        "slug": slug,
        "question": question,
        "category": category,
        "volume": vol,
        "closed": bool(m.get("closed", False)),
        "start_date": m.get("startDate"),
        "end_date": m.get("endDate"),
    }


def parse_event_markets(event: dict, tag_hint: str | None = None) -> list[dict]:
    """Expand one Gamma `/events` entry into parsed market rows. An event carries a
    `markets` list of market dicts (same schema `parse_market_row` reads). The event
    `title` backfills a market's missing `question` (so a terse market slug still
    classifies), and `tag_hint` (the tag_slug we enumerated under) rescues an
    otherwise-`other` market into that tag's category — never overriding a confident
    keyword match (spec §1.3)."""
    out: list[dict] = []
    title = event.get("title")
    for m in (event.get("markets") or []):
        if not m.get("question") and title:
            m = {**m, "question": title}
        row = parse_market_row(m)
        if row is None:
            continue
        if tag_hint and row["category"] == "other":
            hinted = _TAG_CATEGORY.get(tag_hint)
            if hinted:
                row["category"] = hinted
        out.append(row)
    return out


def in_volume_band(row: dict, vol_min: float, vol_max: float) -> bool:
    """Keep markets whose volume is in [vol_min, vol_max] — enough liquidity for a
    copyability signal, but under the ~$4.5M line above which tapes hit the cap."""
    v = row.get("volume") or 0.0
    return vol_min <= v <= vol_max


def select_candidate_markets(
    universe: pd.DataFrame,
    min_volume: float = MIN_MARKET_VOLUME,
    skip_micro: bool = SKIP_MICRO_CRYPTO,
) -> list[str]:
    """From the enumerated universe, the market_ids whose tape we should pull:
    volume >= floor, and (by default) not micro-crypto. Highest-volume first."""
    if universe.empty:
        return []
    df = universe[universe["volume"].fillna(0.0) >= min_volume]
    if skip_micro:
        df = df[df["category"] != "micro_crypto"]
    df = df.sort_values("volume", ascending=False)
    return df["market_id"].tolist()


def select_new_trades(page: list[dict], floor_ts: int, seen_at_max: set[str]) -> tuple[list[dict], bool]:
    """Given a newest-first page and the on-disk cursor (floor timestamp +
    the trade keys already recorded at that exact timestamp), return
    (trades strictly newer than the cursor, stop?). `stop` is True once a trade
    at or before the floor is reached. Overlap-safe: trades exactly at floor_ts
    whose key was already seen are skipped (dedup at the boundary), the rest at
    floor_ts are kept. Mirrors `ingest.fetch_new_trades`."""
    out: list[dict] = []
    stop = False
    for t in page:
        ts = t["timestamp"]
        if ts < floor_ts:
            stop = True
            break
        if ts == floor_ts and trade_key(t) in seen_at_max:
            continue
        out.append(t)
    return out, stop


def advance_cursor(new_trades: list[dict], prev: dict) -> dict:
    """Compute the next per-market cursor from the trades just fetched. Keeps the
    overlap-safe {max_timestamp, keys_at_max} shape used by ingest/watch. If no
    new trades, the cursor is unchanged."""
    if not new_trades:
        return prev
    max_ts = max(t["timestamp"] for t in new_trades)
    keys_at_max = [trade_key(t) for t in new_trades if t["timestamp"] == max_ts]
    # If this run didn't advance past the previous max, preserve prior boundary keys.
    if prev and prev.get("max_timestamp") == max_ts:
        keys_at_max = sorted(set(keys_at_max) | set(prev.get("keys_at_max", [])))
    return {"max_timestamp": max_ts, "keys_at_max": keys_at_max}


def dedup_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup on the fill-identity key, keep last — same discipline as the main
    ledger (DECISIONS.md 'Duplicate trade rows'). The feed re-emits byte-identical
    fills; keeping last avoids double-counting size."""
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["tx_hash", "wallet", "token_id", "side"], keep="last"
    ).reset_index(drop=True)


def fold_discovery_trades(trades: list[dict], category_by_market: dict[str, str]) -> pd.DataFrame:
    """Trades -> discovery ledger rows. Reuses the tested `fold_trades_to_ledger`
    column mapping (with empty resolutions, so resolved=False / resolved_value=None
    — resolution enrichment against the shared read-only cache is a downstream
    step), then attaches the per-market `category`."""
    if not trades:
        return pd.DataFrame(columns=DISCOVERY_LEDGER_COLUMNS)
    df = fold_trades_to_ledger(trades, pd.DataFrame())
    # Category from the enumerated universe map; fall back to classifying from
    # the row's own slug/question if the market wasn't enumerated (e.g. added
    # mid-run). Built as a plain list to keep the column object-typed.
    cats = []
    for mid, slug, question in zip(df["market_id"], df["slug"], df["question"]):
        cat = category_by_market.get(mid)
        if cat is None or (isinstance(cat, float) and pd.isna(cat)):
            cat = classify_market(slug, question)
        cats.append(cat)
    df["category"] = cats
    return df[DISCOVERY_LEDGER_COLUMNS]


def merge_universe(existing: pd.DataFrame, fresh_rows: list[dict], now_ts: int) -> pd.DataFrame:
    """Upsert enumerated markets into the universe cache: new markets get
    first_seen=last_seen=now; re-seen markets keep first_seen, refresh last_seen
    and their mutable fields (volume/closed/category). Idempotent."""
    fresh = pd.DataFrame(fresh_rows)
    if not fresh.empty:
        fresh["first_seen"] = now_ts
        fresh["last_seen"] = now_ts
    if existing.empty:
        combined = fresh
    elif fresh.empty:
        combined = existing
    else:
        prev_first = dict(zip(existing["market_id"], existing["first_seen"]))
        fresh["first_seen"] = fresh["market_id"].map(prev_first).fillna(fresh["first_seen"])
        # fresh rows win for mutable fields; keep='last' after concat.
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["market_id"], keep="last")
    if combined.empty:
        return pd.DataFrame(columns=MARKET_UNIVERSE_COLUMNS)
    for col in MARKET_UNIVERSE_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[MARKET_UNIVERSE_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Network I/O (not unit-tested against the live API; exercised via FakeSession)
# ---------------------------------------------------------------------------

def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def enumerate_markets(session, closed: bool, max_markets: int) -> list[dict]:
    """Volume-ranked sweep of Gamma `/markets` via offset paging (0…ceiling).
    Returns parsed market rows (highest volume first). Stops at max_markets, an
    empty/short page, or the ~2400 offset ceiling (HTTP 422 -> logged, deferred:
    the long tail past the ceiling needs /markets/keyset, a documented follow-on)."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < max_markets and offset < GAMMA_OFFSET_CEILING:
        try:
            resp = session.get(
                f"{GAMMA_API_BASE}/markets",
                params={
                    "closed": "true" if closed else "false",
                    "order": ENUM_ORDER,
                    "ascending": "false",
                    "limit": GAMMA_PAGE_SIZE,
                    "offset": offset,
                },
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 422:  # offset ceiling: deeper pages need /markets/keyset
                print(f"[discover]   hit Gamma offset ceiling at offset={offset} "
                      f"(closed={closed}); deeper tail deferred to /markets/keyset")
                break
            raise
        page = resp.json()
        if not page:
            break
        for m in page:
            parsed = parse_market_row(m)
            if parsed is not None:
                rows.append(parsed)
        if len(page) < GAMMA_PAGE_SIZE:
            break
        offset += GAMMA_PAGE_SIZE
        if SLEEP_BETWEEN_REQUESTS_SEC:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    return rows[:max_markets]


def enumerate_events_markets(
    session,
    tag_slugs: list[str],
    closed: bool,
    max_markets: int,
    vol_min: float = EVENTS_VOLUME_MIN,
    vol_max: float = EVENTS_VOLUME_MAX,
    max_pages_per_tag: int = EVENTS_MAX_PAGES_PER_TAG,
) -> list[dict]:
    """Enumerate the mid/low-volume FULL-tape band via `/events?tag_slug=` (the only
    path to it — see the EVENTS_* constants). For each tag, page `/events` (volume
    DESC), expand every event's markets, and keep the ones in [vol_min, vol_max],
    deduped across tags. Returns parsed market rows. Category filtering must go
    through `/events` — `/markets?tag_slug=` is silently ignored (verify gate §2.1)."""
    rows: list[dict] = []
    seen: set[str] = set()
    for tag in tag_slugs:
        offset = 0
        for _page in range(max_pages_per_tag):
            if len(rows) >= max_markets:
                break
            try:
                resp = session.get(
                    f"{GAMMA_API_BASE}/events",
                    params={
                        "tag_slug": tag,
                        "closed": "true" if closed else "false",
                        "order": EVENTS_ORDER,
                        "ascending": "false",
                        "limit": GAMMA_PAGE_SIZE,
                        "offset": offset,
                    },
                    timeout=REQUEST_TIMEOUT_SEC,
                )
                resp.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 422:  # events offset ceiling; deeper tail deferred
                    print(f"[discover]   events tag={tag} hit offset ceiling at offset={offset}")
                    break
                raise
            page = resp.json()
            if not page:
                break
            for ev in page:
                for row in parse_event_markets(ev, tag_hint=tag):
                    cid = row["market_id"]
                    if cid in seen or not in_volume_band(row, vol_min, vol_max):
                        continue
                    seen.add(cid)
                    rows.append(row)
            if len(page) < GAMMA_PAGE_SIZE:
                break
            offset += GAMMA_PAGE_SIZE
            if SLEEP_BETWEEN_REQUESTS_SEC:
                time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
        if len(rows) >= max_markets:
            break
    return rows[:max_markets]


def fetch_market_trades(session, market_id: str, cursor: dict) -> tuple[list[dict], dict, bool]:
    """Page `/trades?market=<conditionId>` newest-first until overlap with the
    per-market cursor, a short page, or the ~10k offset cap (HTTP 400). Returns
    (new trades strictly newer than the cursor, updated cursor, hit_cap). `hit_cap`
    is True when the pull was truncated at the offset cap (400) or exhausted
    MAX_PAGES_PER_MARKET without reaching the tape's end — i.e. this market has more
    history than is retrievable, so its early entries are unreachable (spec §1.2).
    Mirrors `backfill.fetch_user_trades` / `ingest.fetch_new_trades`."""
    floor_ts = int(cursor.get("max_timestamp", 0))
    seen_at_max = set(cursor.get("keys_at_max", []))
    new_trades: list[dict] = []
    offset = 0
    hit_cap = False
    for _page in range(MAX_PAGES_PER_MARKET):
        try:
            resp = session.get(
                f"{DATA_API_BASE}/trades",
                params={"market": market_id, "limit": TRADES_PAGE_SIZE, "offset": offset},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # 400 = past the ~10k offset cap (offset>10000): tape truncated at the
            # cap. Pages are newest-first, so we hold only the most-recent ~10.5k.
            if status == 400:
                hit_cap = True
                break
            raise
        page = resp.json()
        if not page:
            break
        picked, stop = select_new_trades(page, floor_ts, seen_at_max)
        new_trades.extend(picked)
        if stop or len(page) < TRADES_PAGE_SIZE:
            break
        offset += TRADES_PAGE_SIZE
        if SLEEP_BETWEEN_REQUESTS_SEC:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    else:
        # Ran the full page budget without a natural stop -> more history than we
        # could page => truncated at the cap (same consequence as the 400).
        hit_cap = True
    return new_trades, advance_cursor(new_trades, cursor), hit_cap


# ---------------------------------------------------------------------------
# Persistence (discovery dir only)
# ---------------------------------------------------------------------------

def load_universe() -> pd.DataFrame:
    if MARKET_UNIVERSE_PATH.exists():
        return pd.read_parquet(MARKET_UNIVERSE_PATH)
    return pd.DataFrame(columns=MARKET_UNIVERSE_COLUMNS)


def save_universe(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, MARKET_UNIVERSE_PATH, compression="gzip")


def load_discovery_trades(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the discovery tape. `columns` projects the read (parquet reads only
    those column chunks) — scoring passes just the columns it needs so the heavy
    unused text columns (question/slug/tx_hash) are never materialized, which keeps
    the low-end box from OOMing on the grown tape."""
    if DISCOVERY_TRADES_PATH.exists():
        return pd.read_parquet(DISCOVERY_TRADES_PATH, columns=columns)
    cols = columns or DISCOVERY_LEDGER_COLUMNS
    return pd.DataFrame(columns=cols)


def save_discovery_trades(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, DISCOVERY_TRADES_PATH, compression="gzip")


def load_cursors() -> dict:
    if DISCOVER_CURSORS_PATH.exists():
        with open(DISCOVER_CURSORS_PATH) as f:
            return json.load(f)
    return {}


def save_cursors(cursors: dict) -> None:
    atomic_write_json(cursors, DISCOVER_CURSORS_PATH)


def load_tape_stats() -> pd.DataFrame:
    if MARKET_TAPE_STATS_PATH.exists():
        return pd.read_parquet(MARKET_TAPE_STATS_PATH)
    return pd.DataFrame(columns=TAPE_STATS_COLUMNS)


def save_tape_stats(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, MARKET_TAPE_STATS_PATH, compression="gzip")


def update_tape_stats(existing: pd.DataFrame, updates: list[dict], now_ts: int) -> pd.DataFrame:
    """Upsert per-market {market_id, hit_cap} pull results. `hit_cap` is STICKY —
    OR'd with any prior value — because a tape truncated at the offset cap once has
    early history that no later historical pull can recover (only live capture from
    when the market was young avoids truncation, and that path never sets hit_cap).
    Pure/idempotent; `last_pulled` refreshed to now_ts."""
    if not updates:
        return existing
    upd = pd.DataFrame(updates)
    upd = upd.drop_duplicates(subset=["market_id"], keep="last")
    if not existing.empty:
        prev = dict(zip(existing["market_id"], existing["hit_cap"].fillna(False)))
        upd["hit_cap"] = [
            bool(hc) or bool(prev.get(mid, False))
            for mid, hc in zip(upd["market_id"], upd["hit_cap"])
        ]
    upd["last_pulled"] = now_ts
    combined = upd if existing.empty else pd.concat([existing, upd], ignore_index=True)
    combined = combined.drop_duplicates(subset=["market_id"], keep="last")
    for col in TAPE_STATS_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    return combined[TAPE_STATS_COLUMNS].reset_index(drop=True)


def market_tape_complete(tape: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Per-market tape-completeness for scoring segmentation. Returns a frame
    [market_id, n_trades, tape_complete]: a tape is COMPLETE (early entries intact)
    when it was never truncated at the cap. Prefer the recorded `hit_cap` signal;
    for markets with no stats row (e.g. the first-run pull), fall back to the
    tape-length heuristic (>= TAPE_CAP_ROWS rows => truncated)."""
    if tape.empty:
        return pd.DataFrame(columns=["market_id", "n_trades", "tape_complete"])
    n = tape.groupby("market_id").size().rename("n_trades").reset_index()
    hit = {}
    if stats is not None and not stats.empty:
        hit = dict(zip(stats["market_id"], stats["hit_cap"].fillna(False)))
    # A tape is COMPLETE only if it passes BOTH checks: the recorded hit_cap flag
    # says not-truncated (default not-recorded -> not truncated), AND the tape holds
    # fewer than the cap's worth of rows. Requiring both is robust to the recorded
    # flag being wrongly cleared — e.g. a run killed after checkpointing a truncated
    # tape but before persisting its stats records hit_cap=False on resume; the
    # 10.5k-row length still flags it truncated.
    complete = [
        (not bool(hit.get(mid, False))) and (cnt < TAPE_CAP_ROWS)
        for mid, cnt in zip(n["market_id"], n["n_trades"])
    ]
    n["tape_complete"] = complete
    return n


def load_discovery_resolutions() -> pd.DataFrame:
    if DISCOVERY_RESOLUTIONS_PATH.exists():
        return pd.read_parquet(DISCOVERY_RESOLUTIONS_PATH)
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


def save_discovery_resolutions(df: pd.DataFrame) -> None:
    atomic_to_parquet(df, DISCOVERY_RESOLUTIONS_PATH, compression="gzip")


def load_shared_resolutions_readonly() -> pd.DataFrame:
    """Read the SHARED resolutions cache read-only. Another process (the main
    pipeline) owns and writes this file; we only ever read it, to avoid re-fetching
    resolutions it already has. Returns an empty frame if it doesn't exist yet."""
    if RESOLUTIONS_PATH.exists():
        return pd.read_parquet(RESOLUTIONS_PATH)
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


# ---------------------------------------------------------------------------
# Resolution enrichment (discovery dir only; shared cache read-only)
# ---------------------------------------------------------------------------

def _resolved_market_ids(resolutions: pd.DataFrame) -> set[str]:
    """Market ids marked closed/resolved in a resolutions frame."""
    if resolutions.empty:
        return set()
    return set(resolutions.loc[resolutions["closed"].fillna(False), "market_id"])


def refresh_discovery_resolutions(trades: pd.DataFrame, resolutions: pd.DataFrame) -> pd.DataFrame:
    """Backfill `resolved`/`resolved_value` on the discovery tape from a
    resolutions frame, per (market_id, token_id). Like ingest.refresh_ledger_
    resolutions but preserves the discovery-only `category` column (that helper
    reprojects to the 13-col LEDGER_COLUMNS, which would drop it)."""
    if trades.empty or resolutions.empty:
        return trades
    res = resolutions[["market_id", "token_id", "resolved", "resolved_value"]].rename(
        columns={"resolved": "resolved_new", "resolved_value": "resolved_value_new"}
    )
    merged = trades.merge(res, on=["market_id", "token_id"], how="left")
    needs = merged["resolved_new"].fillna(False) & ~merged["resolved"].fillna(False)
    merged.loc[needs, "resolved"] = True
    merged.loc[needs, "resolved_value"] = merged.loc[needs, "resolved_value_new"]
    return merged[DISCOVERY_LEDGER_COLUMNS]


def resolve_discovery_markets(session=None, max_fetches: int | None = None) -> pd.DataFrame:
    """Enrich the discovery tape with market resolutions, ISOLATED from the shared
    cache.

    Flow (nothing but discovery-dir files is written):
      1. Load our discovery tape + the SHARED resolutions cache (READ-ONLY).
      2. Fetch, from CLOB, resolutions ONLY for discovery markets the shared cache
         doesn't already resolve — reusing the tested `ingest.update_resolutions`
         with our OWN discovery_resolutions.parquet as its base, so shared rows are
         never copied in and the shared file is never written.
      3. Refresh the tape's resolved/resolved_value from shared ∪ discovery
         resolutions and persist (atomic).
    """
    cfg = load_config()
    if session is None:
        session = make_session()

    trades = load_discovery_trades()
    if trades.empty:
        print("[discover] no discovery tape to resolve — run discovery first.")
        return trades

    shared = load_shared_resolutions_readonly()
    discovery_res = load_discovery_resolutions()

    market_ids = set(trades["market_id"].dropna().unique())
    already = _resolved_market_ids(shared) | _resolved_market_ids(discovery_res)
    need = market_ids - already
    print(
        f"[discover] resolving {len(market_ids)} discovery markets: "
        f"{len(market_ids) - len(need)} already resolved (shared+discovery caches), "
        f"{len(need)} to fetch from CLOB"
    )

    # update_resolutions fetches `need` (unseen) + rechecks anything still-open in
    # our discovery cache, capped at max_fetches, and returns discovery_res ∪ fresh.
    discovery_res = update_resolutions(session, cfg, need, discovery_res, max_fetches=max_fetches)
    save_discovery_resolutions(discovery_res)

    # Enrich the tape from BOTH caches (shared read-only + our fetched rows). They
    # cover disjoint markets by construction, but dedup keep-last is safe anyway.
    combined = pd.concat(
        [shared[RESOLUTION_COLUMNS] if not shared.empty else shared, discovery_res],
        ignore_index=True,
    )
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["market_id", "token_id"], keep="last")
    enriched = refresh_discovery_resolutions(trades, combined)
    save_discovery_trades(enriched)

    n_resolved = int(enriched["resolved"].fillna(False).sum())
    n_markets_resolved = enriched.loc[enriched["resolved"].fillna(False), "market_id"].nunique()
    print(
        f"[discover] discovery tape now {n_resolved}/{len(enriched)} resolved bets "
        f"across {n_markets_resolved} resolved markets; "
        f"fetched-resolutions file has {discovery_res['market_id'].nunique()} markets"
    )
    return enriched


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _checkpoint_trades(batch: list[dict], category_by_market: dict[str, str], cursors: dict) -> None:
    """Fold a batch of freshly-fetched trades into the on-disk discovery tape and
    persist it + the cursors (both atomic). Called periodically so an interruption
    never loses a long tape pull — run_discover used to hold everything in memory
    and save only at the end, so a mid-run kill discarded the whole pull. Dedup +
    per-market cursors make repeated checkpoints idempotent and the run resumable."""
    if batch:
        new_rows = fold_discovery_trades(batch, category_by_market)
        combined = dedup_trades(pd.concat([load_discovery_trades(), new_rows], ignore_index=True))
        save_discovery_trades(combined)
    save_cursors(cursors)


def _enumerate_universe(
    session, enum_mode: str, include_open: bool, markets_per_run: int,
    tag_slugs: list[str], vol_min: float, vol_max: float,
) -> list[dict]:
    """Enumerate markets for one pass, by mode: 'volume' = the volume-ranked
    /markets head (top ~2000, all high-volume, mostly cap-truncated tapes);
    'events' = the /events?tag_slug= mid-volume band (§1.4 prong 2: full tapes).
    closed = historical skill signal; open (include_open) = early-tape capture."""
    if enum_mode == "events":
        enumerated = enumerate_events_markets(
            session, tag_slugs, closed=True, max_markets=markets_per_run,
            vol_min=vol_min, vol_max=vol_max)
        if include_open:
            enumerated += enumerate_events_markets(
                session, tag_slugs, closed=False, max_markets=markets_per_run,
                vol_min=vol_min, vol_max=vol_max)
        print(f"[discover] enumerated {len(enumerated)} markets via /events "
              f"(tags={','.join(tag_slugs)}; ${vol_min:,.0f}-${vol_max:,.0f} band)")
    else:
        enumerated = enumerate_markets(session, closed=True, max_markets=markets_per_run)
        if include_open:
            enumerated += enumerate_markets(session, closed=False, max_markets=markets_per_run)
        print(f"[discover] enumerated {len(enumerated)} markets (volume-ranked /markets)")
    return enumerated


def _pull_tapes(
    session, candidates: list[str], category_by_market: dict[str, str],
    cursors: dict, checkpoint_every: int, label: str = "discover",
) -> tuple[int, int]:
    """Pull each candidate market's incremental tape (per-market cursor, overlap-
    safe), checkpointing the folded tape + cursors + per-market tape stats every
    `checkpoint_every` markets so an interruption never loses progress and the next
    run resumes. tape stats are persisted on the SAME cadence as the tape (not only
    at the end) so the crash-safe hit_cap signal can't be lost by a kill mid-run.
    Returns (total_new_rows, n_errors). Shared by the historical sweep
    (run_discover) and the live open-market loop (run_live_capture)."""
    batch: list[dict] = []
    stats_updates: list[dict] = []
    total_new = 0
    errors = 0
    for i, cid in enumerate(candidates, 1):
        cursor = cursors.get(cid, {"max_timestamp": 0, "keys_at_max": []})
        try:
            trades, cursor, hit_cap = fetch_market_trades(session, cid, cursor)
        except Exception as exc:  # noqa: BLE001 - keep going on other markets
            errors += 1
            print(f"[{label}] WARNING: tape fetch failed for {cid}: {exc}", flush=True)
            continue
        cursors[cid] = cursor
        stats_updates.append({"market_id": cid, "hit_cap": hit_cap})
        if trades:
            batch.extend(trades)
            total_new += len(trades)
        if i % checkpoint_every == 0 or i == len(candidates):
            _checkpoint_trades(batch, category_by_market, cursors)
            save_tape_stats(update_tape_stats(load_tape_stats(), stats_updates, _now_ts()))
            batch = []
            stats_updates = []
            print(f"[{label}]   {i}/{len(candidates)} markets, {total_new} new trade rows "
                  f"(checkpointed)", flush=True)
    return total_new, errors


def run_discover(
    markets_per_run: int = MARKETS_PER_RUN,
    max_markets_to_pull: int | None = None,
    include_open: bool = INCLUDE_OPEN_MARKETS,
    checkpoint_every: int = 20,
    enum_mode: str = "volume",
    tag_slugs: list[str] | None = None,
    vol_min: float = EVENTS_VOLUME_MIN,
    vol_max: float = EVENTS_VOLUME_MAX,
) -> pd.DataFrame:
    """One discovery pass: enumerate markets -> update the universe cache -> pull
    the tapes of the selected real-world markets into the isolated discovery
    dataset. Incremental (per-market cursor), idempotent, and checkpointed every
    `checkpoint_every` markets so an interruption is safe and the run resumes.

    `enum_mode='volume'` (default) enumerates the volume-ranked /markets head — the
    original path, high-volume tapes that hit the cap near resolution.
    `enum_mode='events'` enumerates the mid-volume FULL-tape band via /events
    (§1.4 prong 2): those tapes keep their early entries, so metric-A skill is
    measured honestly instead of on near-resolution endgame entries.

    `include_open` also enumerates/pulls still-open markets. For a first *historical*
    output, set it False: only closed markets carry a resolution, so a closed-only
    sweep spends the tape budget entirely on scoreable (resolved) bets. Continuous
    open-market early-tape capture (§1.4 prong 1) is `run_live_capture`."""
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    now = _now_ts()
    tag_slugs = tag_slugs or EVENT_TAG_SLUGS

    # 1. Enumerate + upsert the universe.
    enumerated = _enumerate_universe(
        session, enum_mode, include_open, markets_per_run, tag_slugs, vol_min, vol_max)
    universe = merge_universe(load_universe(), enumerated, now)
    save_universe(universe)
    n_micro = int((universe["category"] == "micro_crypto").sum())
    print(f"[discover] market universe now {len(universe)} markets "
          f"({len(universe) - n_micro} real-world, {n_micro} micro-crypto)")

    # 2. Which tapes to pull this run. In events mode, pull exactly the band markets
    #    enumerated this pass (don't re-walk the whole universe / re-touch the mega
    #    head every events run); in volume mode, the volume-floor+desc candidate set.
    if enum_mode == "events":
        seen: set[str] = set()
        candidates = []
        for r in enumerated:
            cid = r["market_id"]
            if r["category"] != "micro_crypto" and cid not in seen:
                seen.add(cid)
                candidates.append(cid)
    else:
        candidates = select_candidate_markets(universe)
    if max_markets_to_pull is not None:
        candidates = candidates[:max_markets_to_pull]
    print(f"[discover] pulling tapes for {len(candidates)} real-world markets")

    category_by_market = dict(zip(universe["market_id"], universe["category"]))
    cursors = load_cursors()
    total_new, _ = _pull_tapes(
        session, candidates, category_by_market, cursors, checkpoint_every)

    combined = load_discovery_trades()
    n_wallets = combined["wallet"].nunique() if not combined.empty else 0
    n_markets = combined["market_id"].nunique() if not combined.empty else 0
    print(f"[discover] added {total_new} rows; discovery tape now {len(combined)} trades "
          f"across {n_markets} markets and {n_wallets} wallets", flush=True)
    return combined


def enumerate_open_realworld(
    session, markets_per_enum: int, tag_slugs: list[str], vol_min: float, vol_max: float,
) -> list[dict]:
    """Open real-world markets to live-capture: the /events mid-volume band (the
    forecaster-rich full-tape tail) plus the still-open volume head. Deduped,
    micro-crypto excluded. Both closed=false sweeps."""
    rows = enumerate_events_markets(
        session, tag_slugs, closed=False, max_markets=markets_per_enum,
        vol_min=vol_min, vol_max=vol_max)
    rows += enumerate_markets(session, closed=False, max_markets=markets_per_enum)
    seen: set[str] = set()
    out = []
    for r in rows:
        cid = r["market_id"]
        if cid in seen or r["category"] == "micro_crypto":
            continue
        seen.add(cid)
        out.append(r)
    return out


def should_reenumerate(iteration: int, every: int) -> bool:
    """Re-enumerate the open-market set on iteration 0 and every `every` ticks
    (new markets open, others close). Pure so the cadence is unit-testable."""
    return every <= 0 or iteration % every == 0


def run_live_capture(
    poll_seconds: int = 300,
    reenumerate_every: int = 12,
    max_iterations: int | None = None,
    max_minutes: float | None = None,
    markets_per_enum: int = 300,
    checkpoint_every: int = 25,
    tag_slugs: list[str] | None = None,
    vol_min: float = EVENTS_VOLUME_MIN,
    vol_max: float = EVENTS_VOLUME_MAX,
) -> None:
    """Continuous early-tape capture on OPEN real-world markets (§1.4 prong 1).

    A high-volume market's early trades scroll out of reach once it accrues >10.5k
    trades near resolution (the near-resolution bias §1.2). Polling open markets on
    a cadence with the overlap-safe per-market cursor captures each market's new
    trades every tick, so as long as fewer than ~10.5k arrive between ticks the FULL
    tape — including the early entries where forecaster skill lives — is recorded
    before it can scroll past the cap. Same rolling-overlap discipline `ingest.py`
    uses for the global feed, but scoped per-market (far slower per market, so
    trivially kept overlapping).

    Long-lived, checkpointed, resumable (cursors + tape + universe all persisted;
    a restart re-enumerates and resumes from the cursors), and backoff-guarded so a
    network blip never crash-loops. Bounded by `max_iterations` and/or `max_minutes`
    (either hit ends the loop); both None = run until externally stopped. Meant to
    run detached (setsid/nohup) or under a supervisor — the model is not in the loop."""
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    tag_slugs = tag_slugs or EVENT_TAG_SLUGS
    deadline = None if max_minutes is None else _now_ts() + max_minutes * 60.0

    cursors = load_cursors()
    open_markets: list[str] = []
    category_by_market: dict[str, str] = {}
    iteration = 0
    backoff = float(poll_seconds)
    print(f"[live] starting: poll={poll_seconds}s, reenumerate_every={reenumerate_every}, "
          f"max_iterations={max_iterations}, max_minutes={max_minutes}", flush=True)

    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        if deadline is not None and _now_ts() >= deadline:
            break

        # (Re)enumerate the open real-world set periodically.
        if should_reenumerate(iteration, reenumerate_every):
            try:
                enumerated = enumerate_open_realworld(
                    session, markets_per_enum, tag_slugs, vol_min, vol_max)
                universe = merge_universe(load_universe(), enumerated, _now_ts())
                save_universe(universe)
                category_by_market = dict(zip(universe["market_id"], universe["category"]))
                open_markets = [r["market_id"] for r in enumerated]
                print(f"[live] iter {iteration}: watching {len(open_markets)} open real-world markets",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] WARNING: enumeration failed ({exc}); "
                      f"keeping previous {len(open_markets)} markets", flush=True)

        # Incremental per-market cursor pull (tape stats persisted inside _pull_tapes).
        total_new, errors = _pull_tapes(
            session, open_markets, category_by_market, cursors, checkpoint_every, label="live")
        print(f"[live] iter {iteration}: +{total_new} new trades, {errors} market errors "
              f"({len(open_markets) - errors} ok)", flush=True)

        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        if deadline is not None and _now_ts() >= deadline:
            break

        # Backoff only when the whole tick failed (network down); else steady cadence.
        if open_markets and errors == len(open_markets):
            backoff = min(backoff * 2, 3600.0)
            print(f"[live] all {errors} markets errored — backing off to {backoff:.0f}s", flush=True)
        else:
            backoff = float(poll_seconds)
        time.sleep(backoff)

    print(f"[live] stopped after {iteration} iterations.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-first real-world forecaster discovery.")
    parser.add_argument("--markets-per-run", type=int, default=MARKETS_PER_RUN,
                        help="cap enumerated markets per closed/open sweep")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="cap markets whose tape is pulled this run (smoke tests)")
    parser.add_argument("--closed-only", action="store_true",
                        help="skip open markets — pull only resolved markets (a first "
                             "historical output; open-market live capture is layered on later)")
    parser.add_argument("--resolve", action="store_true",
                        help="resolution-enrich the discovery tape instead of pulling more "
                             "tapes: fetch missing resolutions (shared cache read-only) into "
                             "discovery_resolutions.parquet and refresh discovery_trades")
    parser.add_argument("--max-resolution-fetches", type=int, default=None,
                        help="cap CLOB resolution GETs this run (for --resolve)")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="save the tape + cursors every N markets (interruption-safe/resumable)")
    parser.add_argument("--events", action="store_true",
                        help="enumerate the mid-volume FULL-tape band via /events?tag_slug= "
                             "(§1.4 prong 2) instead of the volume-ranked /markets head — its "
                             "tapes keep early entries, so metric-A skill is measured honestly")
    parser.add_argument("--tags", type=str, default=None,
                        help="comma-separated tag_slugs for --events/--live (default: "
                             + ",".join(EVENT_TAG_SLUGS) + ")")
    parser.add_argument("--vol-min", type=float, default=EVENTS_VOLUME_MIN,
                        help="min market volume for the events band (skip near-dead markets)")
    parser.add_argument("--vol-max", type=float, default=EVENTS_VOLUME_MAX,
                        help="max market volume for the events band (above this, tapes hit the cap)")
    parser.add_argument("--live", action="store_true",
                        help="continuous early-tape capture on OPEN real-world markets (§1.4 prong 1): "
                             "poll on a cadence so early entries are recorded before they scroll past "
                             "the 10.5k cap. Long-lived/checkpointed — run detached (setsid/nohup)")
    parser.add_argument("--poll-seconds", type=int, default=300, help="[--live] cadence between ticks")
    parser.add_argument("--reenumerate-every", type=int, default=12,
                        help="[--live] re-enumerate the open-market set every N ticks")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="[--live] stop after N ticks (default: run until stopped)")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="[--live] stop after N minutes (default: run until stopped)")
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    if args.resolve:
        resolve_discovery_markets(max_fetches=args.max_resolution_fetches)
    elif args.live:
        run_live_capture(
            poll_seconds=args.poll_seconds,
            reenumerate_every=args.reenumerate_every,
            max_iterations=args.max_iterations,
            max_minutes=args.max_minutes,
            markets_per_enum=args.markets_per_run,
            checkpoint_every=args.checkpoint_every,
            tag_slugs=tags,
            vol_min=args.vol_min,
            vol_max=args.vol_max,
        )
    else:
        run_discover(
            markets_per_run=args.markets_per_run,
            max_markets_to_pull=args.max_markets,
            include_open=not args.closed_only,
            checkpoint_every=args.checkpoint_every,
            enum_mode="events" if args.events else "volume",
            tag_slugs=tags,
            vol_min=args.vol_min,
            vol_max=args.vol_max,
        )
    print_disk_usage_summary()


if __name__ == "__main__":
    main()
