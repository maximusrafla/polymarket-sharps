"""Deepen REAL-WORLD wallets from the existing discovery corpus.

WHY THIS EXISTS. `src/ingest.py` is a global-firehose poller and ~82% of that
firehose is 5-minute crypto, so real-world traders barely register: the shared
ledger holds only ~3,192 real-world wallets. Every "the real-world sample is too
thin" conclusion in this repo rests on that number — and it is a COLLECTION
ARTIFACT, not a fact about the population. `data/interim/discovery/
discovery_trades.parquet`, built market-first over just 585 markets (0.6% of the
real-world markets in the ledger), already holds 542,397 wallets, 10,791 of them
with >=20 bets. The fix is therefore not more discovery. It is DEEPENING wallets
already discovered — pulling their near-complete `/trades?user=` histories, which
is what turns a 20-bet wallet into one that can actually be validated.

WHY THE SELECTION IS PERFORMANCE-BLIND (the important design choice). The slow
and sports arms deepened a *screen shortlist*: wallets picked for measured edge.
That is right when you are chasing a specific hypothesis, and this repo has since
learned (docs/redteam_audit_2026-07-26.md; the screening-surface commits) how
easily such a shortlist re-imports its own selection noise. This module does the
opposite: it draws a STRATIFIED PROBABILITY SAMPLE on ACTIVITY ONLY — discovery
bet count and distinct-market count, never edge, never profit, never win rate.
Consequences that matter:

  * The deepened set is an unbiased sample of *active real-world traders*, so a
    screen evaluated on it is being tested, not re-measured.
  * Inclusion probabilities are known and persisted per wallet, so any statistic
    computed on the sample can be reweighted back to the full 6.8k-wallet frame.
    That is the "representative census" the red-team audit listed as outstanding
    process debt (the existing census is volume-selected).
  * Nothing is dropped: the FULL frame is persisted to `pool_census.parquet` with
    each wallet's stratum and inclusion probability, whether or not it was drawn.

REUSE, NOT RE-FETCH. 1,472 wallets already have deep histories on disk from the
slow (1,299) and sports (293) arms. Drawn wallets that are already deepened there
are flagged `prior_arm` and SKIPPED at fetch time — their data is read from those
directories at analysis time. This costs no disk and no requests, and it does not
bias the sample: the draw happened before the check, so inclusion probability is
unchanged.

DISK IS THE BINDING CONSTRAINT. ~4.9 GB free at 81% on a 32 GB eMMC; measured
cost is 0.27 MB/wallet (slow arm) to 0.59 MB/wallet (sports arm). The run is
therefore budgeted explicitly and checks the budget at EVERY checkpoint, stopping
cleanly (cursors saved, fully resumable) the moment either guard trips. Trades are
written as APPEND-ONLY SHARDS rather than one rewritten parquet: the slow/sports
pattern reloads and rewrites the whole multi-million-row file on every checkpoint,
which at this scale would mean ~100 rewrites of a ~1 GB file and a real OOM risk
on a box that has already been OOM-killed twice.

ISOLATION: writes ONLY to `data/interim/realworld/`. It NEVER touches the shared
`bet_ledger.parquet`, so Project 1's certified set stays bit-identical — the same
discipline as `discover.py`, `slow_deepen.py` and `sports_deepen.py`, whose
per-user fetch / fold / resolution machinery this reuses verbatim.

READ-ONLY / ANALYSIS-ONLY: public unauthenticated GETs, no keys, no signing,
nothing on-chain.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from src.backfill import fetch_user_trades
from src.common import (
    INTERIM_DIR,
    LEDGER_COLUMNS,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
    print_disk_usage_summary,
)
from src.ingest import (
    LEDGER_DEDUP_KEY,
    fold_trades_to_ledger,
    refresh_ledger_resolutions,
    trade_key,
    update_resolutions,
)

# --- isolated dataset paths -------------------------------------------------
RW_DIR = INTERIM_DIR / "realworld"
SHARD_DIR = RW_DIR / "deep_trades"            # append-only part-*.parquet shards
DEEP_CURSORS_PATH = RW_DIR / "deep_cursors.json"
DEEP_RESOLUTIONS_PATH = RW_DIR / "deep_resolutions.parquet"
SHORTLIST_PATH = RW_DIR / "shortlist.parquet"
POOL_CENSUS_PATH = RW_DIR / "pool_census.parquet"
WALLET_PROFILE_PATH = RW_DIR / "wallet_profile.parquet"
MARKET_CATEGORY_PATH = RW_DIR / "market_category.parquet"

DISCOVERY_TRADES_PATH = INTERIM_DIR / "discovery" / "discovery_trades.parquet"

# Prior arms whose deepened wallets are reused instead of re-fetched.
PRIOR_ARMS = {
    "slow": INTERIM_DIR / "slow_deepening" / "deep_cursors.json",
    "sports": INTERIM_DIR / "sports" / "deep_cursors.json",
}

RESOLUTION_COLUMNS = ["market_id", "token_id", "outcome", "resolved", "resolved_value", "closed"]

# --- sampling frame (pinned; activity only, never performance) ---------------
MIN_DISCOVERY_BETS = 20      # the discovery corpus' own "findable" rule
MIN_DISCOVERY_MARKETS = 5    # breadth, so a one-market wallet can't enter the frame
SAMPLE_SEED = 20260727       # frozen: the date this sample was drawn

# Strata over discovery bet count. `take=None` means take the whole stratum
# (inclusion probability 1.0); an int means draw that many at random. The three
# upper strata are censused because they are small and carry the statistical
# power; the broad 20-49 band is sampled, which is the only place a weight is
# needed. Ordered high-to-low so the budget, if it binds, bites the cheapest and
# least informative band first.
STRATA: list[tuple[str, int, int | None, int | str | None]] = [
    ("s4_200plus", 200, None, None),
    ("s3_100_199", 100, 200, None),
    ("s2_50_99", 50, 100, None),
    ("s1_20_49", 20, 50, "fill"),  # "fill" = whatever the target leaves over
]

DEFAULT_TARGET_WALLETS = 2500
DEFAULT_CHECKPOINT_EVERY = 25

# --- disk budget (the binding constraint; see module docstring) --------------
DEFAULT_MIN_FREE_GB = 2.5    # stop if the filesystem drops below this
DEFAULT_MAX_NEW_GB = 1.6     # stop once THIS dataset reaches this size
MB_PER_WALLET_ESTIMATE = 0.5  # measured: 0.27 (slow) .. 0.59 (sports)


# ---------------------------------------------------------------------------
# Sampling frame
# ---------------------------------------------------------------------------

def load_discovery_activity(path: Path | None = None) -> pd.DataFrame:
    """Per-wallet ACTIVITY in the discovery corpus: bet count and distinct markets.

    Projected read (two columns of a 2.7M-row file) so this stays cheap on a
    RAM-tight box. Deliberately reads nothing about outcomes or prices — the
    selection must not be able to see performance."""
    path = Path(path) if path is not None else DISCOVERY_TRADES_PATH
    df = pd.read_parquet(path, columns=["wallet", "market_id"])
    act = (
        df.groupby("wallet")
        .agg(n_discovery_bets=("market_id", "size"), n_discovery_markets=("market_id", "nunique"))
        .reset_index()
    )
    return act.sort_values("wallet", kind="mergesort").reset_index(drop=True)


def prior_arm_wallets() -> dict[str, str]:
    """wallet -> arm name, for wallets already deepened by an earlier arm."""
    out: dict[str, str] = {}
    for arm, cursor_path in PRIOR_ARMS.items():
        if not Path(cursor_path).exists():
            continue
        with open(cursor_path) as f:
            for wallet in json.load(f):
                out.setdefault(wallet, arm)
    return out


def assign_strata(activity: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the frame and label each wallet's activity stratum."""
    frame = activity[
        (activity["n_discovery_bets"] >= MIN_DISCOVERY_BETS)
        & (activity["n_discovery_markets"] >= MIN_DISCOVERY_MARKETS)
    ].copy()
    frame["stratum"] = pd.NA
    for name, lo, hi, _take in STRATA:
        mask = frame["n_discovery_bets"] >= lo
        if hi is not None:
            mask &= frame["n_discovery_bets"] < hi
        frame.loc[mask, "stratum"] = name
    return frame.dropna(subset=["stratum"]).reset_index(drop=True)


def draw_sample(
    frame: pd.DataFrame,
    target: int = DEFAULT_TARGET_WALLETS,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """Draw the stratified sample and record every wallet's inclusion probability.

    Returns the WHOLE frame (nothing dropped) with `selected`, `inclusion_prob`,
    `stratum_size` and `stratum_take` columns. Persisting the unselected rows is
    what makes reweighting to the population auditable later."""
    out = frame.copy()
    out["selected"] = False
    out["inclusion_prob"] = 0.0
    out["stratum_size"] = 0
    out["stratum_take"] = 0

    remaining = int(target)
    for name, _lo, _hi, take in STRATA:
        idx = out.index[out["stratum"] == name]
        size = len(idx)
        out.loc[idx, "stratum_size"] = size
        if size == 0:
            continue
        if take is None:                      # census this stratum
            n_take = min(size, remaining)
        elif take == "fill":                  # the sampled band absorbs what's left
            n_take = min(size, remaining)
        else:
            n_take = min(int(take), size, remaining)
        n_take = max(n_take, 0)
        if n_take == 0:
            continue
        if n_take >= size:
            chosen = idx
        else:
            # Per-stratum seed offset so shrinking/growing one stratum cannot
            # reshuffle another one's draw.
            stratum_seed = seed + sum(ord(c) for c in name)
            chosen = out.loc[idx].sample(n=n_take, random_state=stratum_seed).index
        out.loc[idx, "stratum_take"] = n_take
        out.loc[idx, "inclusion_prob"] = n_take / size
        out.loc[chosen, "selected"] = True
        remaining -= n_take
        if remaining <= 0:
            break

    priors = prior_arm_wallets()
    out["prior_arm"] = out["wallet"].map(priors).fillna("")
    out["sample_seed"] = seed
    # Deepen the biggest histories first: if the disk budget binds mid-run we
    # keep the most informative wallets, and the stratum weights stay valid for
    # whichever strata completed (the shortfall is recorded by `status`).
    return out.sort_values(
        ["selected", "n_discovery_bets"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)


def build_plan(
    target: int = DEFAULT_TARGET_WALLETS,
    seed: int = SAMPLE_SEED,
    activity_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(census, shortlist) — census is the whole frame, shortlist the wallets to fetch."""
    activity = load_discovery_activity(activity_path)
    frame = assign_strata(activity)
    census = draw_sample(frame, target=target, seed=seed)
    shortlist = census[census["selected"]].reset_index(drop=True)
    return census, shortlist


# ---------------------------------------------------------------------------
# Sharded storage (append-only; bounded memory)
# ---------------------------------------------------------------------------

def shard_paths() -> list[Path]:
    if not SHARD_DIR.exists():
        return []
    return sorted(SHARD_DIR.glob("part-*.parquet"))


def _next_shard_path() -> Path:
    existing = shard_paths()
    n = 0
    if existing:
        n = max(int(p.stem.split("-")[1]) for p in existing) + 1
    return SHARD_DIR / f"part-{n:05d}.parquet"


def write_shard(df: pd.DataFrame) -> Path | None:
    """Append one checkpoint's rows as a new shard. Never rewrites prior shards."""
    if df is None or df.empty:
        return None
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    path = _next_shard_path()
    atomic_to_parquet(df, path, compression="gzip")
    return path


def load_deep_trades(columns: list[str] | None = None) -> pd.DataFrame:
    """Read every shard and dedup on the fill key.

    Dedup at READ time (not write time) is what lets writes stay append-only: a
    re-fetched trade lands in a later shard and the newer copy wins here."""
    paths = shard_paths()
    if not paths:
        return pd.DataFrame(columns=columns or LEDGER_COLUMNS)
    frames = [pd.read_parquet(p, columns=columns) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    key = [c for c in LEDGER_DEDUP_KEY if c in combined.columns]
    if key:
        combined = combined.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    return combined


def load_deep_tape(columns: list[str] | None = None,
                   categorical: list[str] | None = None) -> pd.DataFrame:
    """Ledger-shaped read of the sharded deep tape.

    Mirrors `common.load_ledger`'s contract — column projection pushed down to the
    parquet reader, and `categorical` columns dictionary-encoded in Arrow — so a
    ledger-shaped analysis (e.g. `scripts/audit_screen_surface.py`) can be pointed
    at this arm without changing what it computes. The dictionary encoding is not
    cosmetic: glibc does not return freed memory to the OS, so materializing a big
    repetitive string column as Python objects sets a high-water RSS mark that
    later frees cannot lower. This box has 3.8 GB and has OOM-killed analyses on
    tapes this size.

    Dedup is PER SHARD. A wallet's whole history arrives in one fetch and is
    written to one shard, so cross-shard duplicates cannot arise from a completed
    run — only within-shard ones can (the API does serve duplicate trade rows; see
    DECISIONS "Duplicate trade rows from the API"). The assumption is checked, not
    assumed: if any wallet is found spanning shards the read falls back to a global
    dedup rather than silently under-deduping."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = shard_paths()
    if not paths:
        return pd.DataFrame(columns=columns or LEDGER_COLUMNS)

    # The dedup key is read in FULL regardless of the projection and dropped again
    # at the end. Pruning it to the requested columns would silently dedup on a
    # partial key — projecting to (wallet, side) collapsed the 7.05M-row tape to
    # 4,077 rows in testing, which is wrong in a way no downstream stage detects.
    dedup_cols = list(LEDGER_DEDUP_KEY)
    read_cols = None if columns is None else sorted(set(columns) | set(dedup_cols))

    tables, wallet_shards = [], {}
    for i, p in enumerate(paths):
        t = pq.read_table(p, columns=read_cols)
        if dedup_cols:
            df = t.to_pandas()
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
            t = pa.Table.from_pandas(df, preserve_index=False)
            del df
        if "wallet" in t.column_names:
            for w in t.column("wallet").unique().to_pylist():
                wallet_shards.setdefault(w, set()).add(i)
        tables.append(t)

    table = pa.concat_tables(tables, promote_options="default")
    del tables
    if categorical:
        for name in categorical:
            if name in table.column_names:
                idx = table.column_names.index(name)
                table = table.set_column(idx, name, table.column(name).dictionary_encode())
    out = table.to_pandas()
    del table

    spanning = [w for w, s in wallet_shards.items() if len(s) > 1]
    if spanning:
        print(f"[rw_deepen] NOTE: {len(spanning):,} wallets span multiple shards "
              "(a resumed run) — falling back to a global dedup.")
        out = out.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    if columns is not None:
        out = out[[c for c in columns if c in out.columns]]
    return out


def load_deep_cursors() -> dict:
    if DEEP_CURSORS_PATH.exists():
        with open(DEEP_CURSORS_PATH) as f:
            return json.load(f)
    return {}


def save_deep_cursors(cursors: dict) -> None:
    atomic_write_json(cursors, DEEP_CURSORS_PATH)


def load_deep_resolutions() -> pd.DataFrame:
    if DEEP_RESOLUTIONS_PATH.exists():
        return pd.read_parquet(DEEP_RESOLUTIONS_PATH)
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------

def dataset_bytes() -> int:
    total = 0
    for p in shard_paths():
        total += p.stat().st_size
    for p in (DEEP_RESOLUTIONS_PATH, SHORTLIST_PATH, POOL_CENSUS_PATH, DEEP_CURSORS_PATH):
        if p.exists():
            total += p.stat().st_size
    return total


def free_bytes() -> int:
    RW_DIR.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(RW_DIR).free


def budget_check(
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    max_new_gb: float = DEFAULT_MAX_NEW_GB,
) -> tuple[bool, str]:
    """(ok, reason). Checked at every checkpoint; a False stops the run cleanly."""
    gb = 1024 ** 3
    free = free_bytes() / gb
    used = dataset_bytes() / gb
    if free < min_free_gb:
        return False, f"filesystem free {free:.2f} GB < floor {min_free_gb:.2f} GB"
    if used >= max_new_gb:
        return False, f"realworld dataset {used:.2f} GB >= cap {max_new_gb:.2f} GB"
    return True, f"free {free:.2f} GB, dataset {used:.2f} GB"


# ---------------------------------------------------------------------------
# Deepening
# ---------------------------------------------------------------------------

def deepen_wallets(
    wallets: list[str],
    cfg: dict | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    session=None,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    max_new_gb: float = DEFAULT_MAX_NEW_GB,
) -> dict:
    """Pull each wallet's `/trades?user=` history into the isolated shard set.

    Incremental (per-wallet cursor = newest timestamp already stored), checkpointed,
    resumable, and budget-guarded. Returns a summary dict."""
    RW_DIR.mkdir(parents=True, exist_ok=True)
    cfg = cfg or load_config()
    session = session or make_session()
    cursors = load_deep_cursors()

    batch: list[dict] = []
    stats = {"attempted": 0, "fetched": 0, "skipped_done": 0, "errors": 0,
             "new_rows": 0, "stopped_early": False, "stop_reason": ""}
    t0 = time.time()

    def flush(i: int, n: int) -> None:
        nonlocal batch
        if batch:
            rows = fold_trades_to_ledger(batch, pd.DataFrame())  # resolutions applied later
            write_shard(rows)
            batch = []
        save_deep_cursors(cursors)
        elapsed = max(time.time() - t0, 1e-9)
        ok, reason = budget_check(min_free_gb, max_new_gb)
        print(f"[rw_deepen]   {i}/{n} wallets, {stats['new_rows']:,} new rows "
              f"({stats['attempted'] / elapsed:.2f} wallets/s) — checkpointed [{reason}]",
              flush=True)
        if not ok:
            stats["stopped_early"] = True
            stats["stop_reason"] = reason

    for i, wallet in enumerate(wallets, 1):
        if wallet in cursors and cursors[wallet].get("complete"):
            stats["skipped_done"] += 1
            continue
        stats["attempted"] += 1
        since_ts = int(cursors.get(wallet, {}).get("max_timestamp", 0))
        try:
            trades = fetch_user_trades(session, cfg, wallet, since_ts)
        except Exception as exc:  # noqa: BLE001 — one bad wallet must not end the run
            print(f"[rw_deepen] WARNING: fetch failed for {wallet}: {exc}", flush=True)
            stats["errors"] += 1
            continue
        stats["fetched"] += 1
        if trades:
            batch.extend(trades)
            stats["new_rows"] += len(trades)
            newest = max(t["timestamp"] for t in trades)
            keys_at_max = [trade_key(t) for t in trades if t["timestamp"] == newest]
            cursors[wallet] = {"max_timestamp": max(since_ts, newest),
                               "keys_at_max": keys_at_max, "complete": True}
        else:
            prior = cursors.get(wallet, {})
            cursors[wallet] = {"max_timestamp": prior.get("max_timestamp", since_ts),
                               "keys_at_max": prior.get("keys_at_max", []), "complete": True}

        if stats["attempted"] % checkpoint_every == 0 or i == len(wallets):
            flush(i, len(wallets))
            if stats["stopped_early"]:
                print(f"[rw_deepen] BUDGET STOP after {i}/{len(wallets)} wallets: "
                      f"{stats['stop_reason']}. Cursors saved — rerun to resume once "
                      "space is freed.", flush=True)
                break

    if batch:  # a break mid-batch must not lose rows
        flush(len(wallets), len(wallets))
    return stats


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_deep_markets(cfg: dict | None = None, session=None,
                         max_fetches: int | None = None) -> int:
    """Resolve the deep dataset's markets and refresh each shard in place.

    Reads the SHARED resolution cache read-only (so known markets aren't re-fetched)
    and writes only this arm's own cache. Shard-wise so peak memory stays at one
    shard, not the whole dataset."""
    from src.common import RESOLUTIONS_PATH
    cfg = cfg or load_config()
    session = session or make_session()

    paths = shard_paths()
    if not paths:
        print("[rw_deepen] no deep trades to resolve — run `deepen` first.")
        return 0

    market_ids: set[str] = set()
    for p in paths:
        market_ids |= set(pd.read_parquet(p, columns=["market_id"])["market_id"].dropna().unique())

    shared = (pd.read_parquet(RESOLUTIONS_PATH) if RESOLUTIONS_PATH.exists()
              else pd.DataFrame(columns=RESOLUTION_COLUMNS))
    deep_res = load_deep_resolutions()
    already: set[str] = set()
    for r in (shared, deep_res):
        if not r.empty:
            already |= set(r.loc[r["closed"].fillna(False), "market_id"])
    need = market_ids - already
    print(f"[rw_deepen] resolving {len(market_ids):,} markets: "
          f"{len(market_ids) - len(need):,} already known, {len(need):,} to fetch from CLOB",
          flush=True)

    deep_res = update_resolutions(session, cfg, need, deep_res, max_fetches=max_fetches)
    atomic_to_parquet(deep_res, DEEP_RESOLUTIONS_PATH, compression="gzip")

    combined = pd.concat(
        [shared[RESOLUTION_COLUMNS] if not shared.empty else shared, deep_res], ignore_index=True
    )
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["market_id", "token_id"], keep="last")

    n_resolved = 0
    for p in paths:
        shard = pd.read_parquet(p)
        enriched = refresh_ledger_resolutions(shard, combined)
        atomic_to_parquet(enriched, p, compression="gzip")
        n_resolved += int(enriched["resolved"].fillna(False).sum())
    print(f"[rw_deepen] deep dataset now {n_resolved:,} resolved bets across "
          f"{len(paths)} shards", flush=True)
    return n_resolved


# ---------------------------------------------------------------------------
# Profiling — how much of the deepened tape is actually real-world
# ---------------------------------------------------------------------------
#
# The wallets were SELECTED from real-world markets, but `/trades?user=` returns
# each one's WHOLE history, micro-crypto included. So "how many real-world bets
# did we actually buy with 0.5 GB" is a question the fetch cannot answer and this
# step must.
#
# The classifier is `discover.classify_market` on slug/question — the same one
# that built the discovery corpus, so the labels are consistent with it, and it
# needs no network. The `market_meta` sidecar's `speed_bucket` is NOT usable here:
# it covers only 116k of these 410k markets (294,529 absent, 32,476 with a NaN
# lifespan), which would park 66% of the tape as "unknown". Deriving a tape
# lifespan from our own trades instead would be actively WRONG — we hold only our
# 2,117 wallets' fills in a market, so a market we touched once would show a
# zero-second span and be misread as micro. Slug/question has no such failure mode.
#
# `is_real_world` is "not micro_crypto", so the fallback `other` bucket counts as
# real-world. That is correct for THIS question and imprecise for finer ones:
# sampling `other` shows geopolitics, golf, esports and league codes the keyword
# rules don't know (fl1, itc, fifwc, lol) — a granularity gap, not a micro leak.

def classify_deep_markets() -> pd.DataFrame:
    """market_id -> category, from slug/question, streamed one shard at a time."""
    from src.discover import classify_market
    seen: dict[str, str] = {}
    for p in shard_paths():
        d = pd.read_parquet(p, columns=["market_id", "slug", "question"]).drop_duplicates("market_id")
        for mid, slug, question in zip(d["market_id"], d["slug"], d["question"]):
            if mid not in seen:
                seen[mid] = classify_market(slug or "", question or "", None)
        del d
    return pd.DataFrame({"market_id": list(seen), "category": list(seen.values())})


def profile_wallets(categories: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-wallet resolved-BUY bet counts, split real-world vs micro and by category.

    Streamed per shard so peak memory is one shard — a whole-dataset concat of the
    7.05M-row tape OOM-killed this box (exit 137) while writing this."""
    from src.discover import is_real_world
    if categories is None:
        categories = classify_deep_markets()
    cat_by_mkt = dict(zip(categories["market_id"], categories["category"]))

    counts: dict[str, dict[str, int]] = {}
    for p in shard_paths():
        d = pd.read_parquet(p, columns=["wallet", "market_id", "side", "resolved"])
        d = d[(d["side"] == "BUY") & d["resolved"].fillna(False)]
        if d.empty:
            continue
        d["category"] = d["market_id"].map(cat_by_mkt).fillna("other")
        for (w, c), n in d.groupby(["wallet", "category"]).size().items():
            per_wallet = counts.setdefault(w, {})
            per_wallet[c] = per_wallet.get(c, 0) + int(n)
        del d

    prof = pd.DataFrame(counts).T.fillna(0).astype(int)
    prof.index.name = "wallet"
    prof = prof.reset_index()
    cat_cols = [c for c in prof.columns if c != "wallet"]
    rw_cols = [c for c in cat_cols if is_real_world(c)]
    prof["real_world_bets"] = prof[rw_cols].sum(axis=1) if rw_cols else 0
    prof["micro_bets"] = prof[[c for c in cat_cols if not is_real_world(c)]].sum(axis=1) \
        if any(not is_real_world(c) for c in cat_cols) else 0
    prof["total_bets"] = prof[cat_cols].sum(axis=1)
    return prof.sort_values("real_world_bets", ascending=False).reset_index(drop=True)


def run_profile() -> pd.DataFrame:
    cats = classify_deep_markets()
    atomic_to_parquet(cats, MARKET_CATEGORY_PATH, compression="gzip")
    prof = profile_wallets(cats)
    atomic_to_parquet(prof, WALLET_PROFILE_PATH, compression="gzip")

    tot = int(prof["total_bets"].sum())
    rw = int(prof["real_world_bets"].sum())
    print(f"[rw_deepen] {len(cats):,} markets classified; {len(prof):,} wallets; "
          f"{tot:,} resolved BUY bets")
    print(f"[rw_deepen] REAL-WORLD: {rw:,} ({100 * rw / max(tot, 1):.1f}%)   "
          f"micro: {tot - rw:,} ({100 * (tot - rw) / max(tot, 1):.1f}%)")
    by_cat = prof.drop(columns=["wallet", "real_world_bets", "micro_bets", "total_bets"]).sum()
    for c, n in by_cat.sort_values(ascending=False).items():
        print(f"    {c:<16}{int(n):>10,}{100 * int(n) / max(tot, 1):>7.1f}%")
    print("\n[rw_deepen] wallets by REAL-WORLD resolved bets:")
    for t in (20, 50, 100, 200, 500, 1000):
        print(f"    >={t:<5} {int((prof['real_world_bets'] >= t).sum()):>6,}")
    return prof


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------

def run_plan(target: int = DEFAULT_TARGET_WALLETS, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    RW_DIR.mkdir(parents=True, exist_ok=True)
    census, shortlist = build_plan(target=target, seed=seed)
    atomic_to_parquet(census, POOL_CENSUS_PATH, compression="gzip")
    atomic_to_parquet(shortlist, SHORTLIST_PATH, compression="gzip")

    print(f"[rw_deepen] sampling frame: {len(census):,} wallets "
          f"(>={MIN_DISCOVERY_BETS} discovery bets, >={MIN_DISCOVERY_MARKETS} markets)")
    print(f"[rw_deepen] drawn: {len(shortlist):,} wallets (seed {seed})")
    print(f"{'stratum':<14}{'frame':>8}{'drawn':>8}{'incl_p':>9}{'reuse':>8}")
    for name, _lo, _hi, _take in STRATA:
        sub = census[census["stratum"] == name]
        drawn = sub[sub["selected"]]
        reuse = int((drawn["prior_arm"] != "").sum())
        p = float(sub["inclusion_prob"].iloc[0]) if len(sub) else 0.0
        print(f"{name:<14}{len(sub):>8,}{len(drawn):>8,}{p:>9.4f}{reuse:>8,}")

    to_fetch = shortlist[shortlist["prior_arm"] == ""]
    est_gb = len(to_fetch) * MB_PER_WALLET_ESTIMATE / 1024
    gb = 1024 ** 3
    print(f"\n[rw_deepen] {len(shortlist) - len(to_fetch):,} drawn wallets are already deepened "
          f"by a prior arm and will be REUSED, not re-fetched.")
    print(f"[rw_deepen] to fetch: {len(to_fetch):,} wallets "
          f"~= {est_gb:.2f} GB at {MB_PER_WALLET_ESTIMATE} MB/wallet")
    print(f"[rw_deepen] budget: {free_bytes() / gb:.2f} GB free now, floor "
          f"{DEFAULT_MIN_FREE_GB} GB, dataset cap {DEFAULT_MAX_NEW_GB} GB")
    print(f"[rw_deepen] wrote {SHORTLIST_PATH} and {POOL_CENSUS_PATH}")
    return shortlist


def fetch_targets() -> list[str]:
    """Wallets the deepen step should actually request (drawn, not already deepened
    elsewhere), in shortlist order."""
    if not SHORTLIST_PATH.exists():
        raise SystemExit("[rw_deepen] no shortlist — run `plan` first.")
    short = pd.read_parquet(SHORTLIST_PATH)
    return short.loc[short["prior_arm"] == "", "wallet"].tolist()


def run_deepen(checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
               max_wallets: int | None = None,
               min_free_gb: float = DEFAULT_MIN_FREE_GB,
               max_new_gb: float = DEFAULT_MAX_NEW_GB,
               resolve: bool = True) -> dict:
    wallets = fetch_targets()
    if max_wallets is not None:
        wallets = wallets[:max_wallets]
    ok, reason = budget_check(min_free_gb, max_new_gb)
    if not ok:
        raise SystemExit(f"[rw_deepen] refusing to start — {reason}")
    print(f"[rw_deepen] deepening {len(wallets):,} wallets [{reason}]", flush=True)
    stats = deepen_wallets(wallets, checkpoint_every=checkpoint_every,
                           min_free_gb=min_free_gb, max_new_gb=max_new_gb)
    print(f"[rw_deepen] {stats}", flush=True)
    if resolve and not stats["stopped_early"]:
        resolve_deep_markets()
    print_disk_usage_summary()
    return stats


def run_status() -> None:
    gb = 1024 ** 3
    cursors = load_deep_cursors()
    done = sum(1 for c in cursors.values() if c.get("complete"))
    shards = shard_paths()
    print(f"[rw_deepen] shards: {len(shards)}  dataset: {dataset_bytes() / gb:.3f} GB  "
          f"free: {free_bytes() / gb:.2f} GB")
    print(f"[rw_deepen] wallets deepened: {done:,}")
    if SHORTLIST_PATH.exists():
        short = pd.read_parquet(SHORTLIST_PATH)
        todo = [w for w in short.loc[short["prior_arm"] == "", "wallet"] if w not in cursors]
        print(f"[rw_deepen] shortlist {len(short):,}; remaining to fetch: {len(todo):,}")
    if shards:
        rows = sum(pd.read_parquet(p, columns=["wallet"]).shape[0] for p in shards)
        print(f"[rw_deepen] rows on disk (pre-dedup): {rows:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deepen a stratified, performance-blind sample of real-world wallets "
                    "from the discovery corpus. Isolated dataset; the shared ledger is "
                    "never touched. Read-only public GETs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="draw the sample and persist shortlist + census (no network)")
    p_plan.add_argument("--target", type=int, default=DEFAULT_TARGET_WALLETS)
    p_plan.add_argument("--seed", type=int, default=SAMPLE_SEED)

    p_deep = sub.add_parser("deepen", help="fetch the shortlist's histories (resumable)")
    p_deep.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    p_deep.add_argument("--max-wallets", type=int, default=None, help="cap this run (smoke tests)")
    p_deep.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    p_deep.add_argument("--max-new-gb", type=float, default=DEFAULT_MAX_NEW_GB)
    p_deep.add_argument("--no-resolve", action="store_true")

    p_res = sub.add_parser("resolve", help="(re)resolve the existing deep dataset")
    p_res.add_argument("--max-fetches", type=int, default=None)

    sub.add_parser("profile", help="classify the deep tape and write the per-wallet profile")
    sub.add_parser("status", help="print progress and disk budget")

    args = parser.parse_args()
    if args.cmd == "plan":
        run_plan(target=args.target, seed=args.seed)
    elif args.cmd == "deepen":
        run_deepen(checkpoint_every=args.checkpoint_every, max_wallets=args.max_wallets,
                   min_free_gb=args.min_free_gb, max_new_gb=args.max_new_gb,
                   resolve=not args.no_resolve)
    elif args.cmd == "resolve":
        resolve_deep_markets(max_fetches=args.max_fetches)
        print_disk_usage_summary()
    elif args.cmd == "profile":
        run_profile()
    elif args.cmd == "status":
        run_status()


if __name__ == "__main__":
    main()
