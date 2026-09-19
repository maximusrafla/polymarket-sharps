"""Freeze the sports survivors as a pre-registered forward tier — Project 3 sports step 5.

WHY A SEPARATE FREEZE ARTIFACT, NOT AN AMENDMENT
------------------------------------------------
The forecaster freeze (`slow_freeze_manifest.json`) already has forward
observations against it, and `slow_forward freeze --amend` correctly REFUSES once
that is true: changing cohorts after outcomes are observable is not
pre-registration. Bolting a sports tier onto it would either break that rule or
force a shared cutoff that means nothing for a population frozen months later.

So sports gets its own artifact, with its own `freeze_ts` set at the moment of
freezing, its own persisted pre-freeze baseline, and its own verbatim scoring
rule. The two scoreboards are deliberately separate objects: different
populations, different cluster units, different resolution speeds. What they
share is the discipline — selection rules fixed before any forward outcome
exists, a baseline fit on pre-freeze data and applied WITHOUT refitting, and
tiers that are never re-frozen in light of forward results.

THE TIERS ARE A VETTING LADDER, NOTHING ELSE
--------------------------------------------
Forward scoring is IDENTICAL across tiers; a tier only says how heavily vetted
its members are. `s_core` is the pre-registered headline and is never displaced
by a wider tier's forward number.

WHAT IS DIFFERENT FROM THE SLOW FORWARD TEST
--------------------------------------------
The event unit is the resolution EVENT (a game, or the championship a whole
outright field settles on), not the event complex. And sports resolves fast:
this test is expected to produce a readable sample in DAYS to WEEKS, where the
forecaster one takes months. That is the whole reason the arm was opened.

READ-ONLY / paper-only: public unauthenticated GETs, no keys, no order placement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from src.common import (
    INTERIM_DIR,
    PROCESSED_DIR,
    atomic_to_parquet,
    atomic_write_json,
    load_config,
    make_session,
)
from src.slow_niche import NICHE_PARTITION_VERSION
from src.sports_events import SPORTS_EVENT_VERSION

FROZEN_SET_PATH = PROCESSED_DIR / "sports_frozen_set.parquet"
FREEZE_MANIFEST_PATH = PROCESSED_DIR / "sports_freeze_manifest.json"
FORWARD_TRADES_PATH = INTERIM_DIR / "sports" / "forward" / "forward_trades.parquet"
FORWARD_RESOLUTIONS_PATH = INTERIM_DIR / "sports" / "forward" / "forward_resolutions.parquet"
SCOREBOARD_PARQUET = PROCESSED_DIR / "sports_forward_scoreboard.parquet"
SCOREBOARD_MD = PROCESSED_DIR / "sports_forward_scoreboard.md"

# Sports resolve in hours-to-days, so the split that is informative here is much
# tighter than the slow arm's 14 days.
FAST_RESOLUTION_DAYS = 2

TIER_SPECS: dict[str, dict] = {
    "s_core": {
        "selector": "edge_persisted",
        "vetting": "highest",
        "headline": True,
        "description": ("league|form hierarchical LOWO baseline, EVENT clusters, "
                        "event-unit concentration gates, 10c magnitude floor, "
                        "cluster-robust significance. THE pre-registered headline."),
    },
    "s_wide": {
        "selector": "candidate & magnitude_ok & markets_ok & concentration_ok & complex_concentration_ok",
        "vetting": "low",
        "headline": False,
        "description": ("clears the magnitude, market-count and event-unit "
                        "concentration gates but NOT any significance test."),
    },
    "s_all": {
        "selector": "candidate",
        "vetting": "least — volume only",
        "headline": False,
        "description": ("in-sample skill > 0 and nothing else. LEAST VETTED, frozen "
                        "for forward VOLUME. Expected to be mostly noise — grading "
                        "that is the point."),
    },
}
HEADLINE_TIER = "s_core"
TIER_ORDER = ["s_core", "s_wide", "s_all"]

SCORING_RULE = (
    "For each frozen wallet, take its resolved BUY bets in SPORTS markets ENTERED "
    "STRICTLY AFTER freeze_ts. Per-bet skill edge = resolved_value - E[outcome | "
    "entry_price, league, league|form], where E is the hierarchical baseline frozen "
    "in this manifest (fit on pre-freeze data only, applied WITHOUT refitting). Each "
    "TIER is scored in AGGREGATE as a crowd, never wallet-by-wallet: the headline "
    "statistic is the EVENT-WEIGHTED mean, i.e. the mean over RESOLUTION EVENTS of "
    "the within-event mean residual, where an event is one game, or the one "
    "championship a whole outright field settles on (src/sports_events.py, version "
    "recorded in this manifest). Buying every team in a field is therefore ONE "
    "observation, not one per team. The bet-weighted mean is reported alongside. "
    "Uncertainty is a cluster bootstrap that resamples EVENTS with replacement (not "
    "bets), giving a 95% CI and a one-sided p vs 0, and is undefined below two "
    "events. Every bet count is reported with n_events and effective_events so "
    "volume is not mistaken for evidence. Results are additionally split by "
    "resolution speed (<=2 days from entry to resolution vs longer). s_core is the "
    "pre-registered headline tier; wider tiers are secondary readings and are NEVER "
    "promoted on the basis of forward results. Tiers are not re-frozen in light of "
    "forward outcomes."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROCESSED_DIR.parent.parent),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# FREEZE
# ---------------------------------------------------------------------------

def build_tiers(cfg: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Re-derive every pre-registered sports tier from pre-freeze deep data.

    Returns (frozen table with one row per wallet in the union plus a boolean
    column per tier, the fitted pre-freeze baseline, per-tier stats)."""
    import src.sports_validate as sv
    from src.slow_baseline import fit_hierarchical_baseline
    from src.slow_validate import slow_verdict
    from src.sports_events import assign_events  # noqa: F401  (documents the dependency)

    cfg = cfg or load_config()
    bets = sv.load_deep_sports_bets(cfg=cfg)
    if bets.empty:
        raise RuntimeError("no deep sports bets — run `python -m src.sports_deepen` first.")

    resid, _ = sv.residualize_for(bets, sv.STANDARD_BASELINE, cfg=cfg)
    table = sv.validate_slow_sports(resid, cfg=cfg)
    table = slow_verdict(table, cfg=cfg).get("table", table)

    members = {t: set(table.query(spec["selector"])["wallet"])
               for t, spec in TIER_SPECS.items()}
    union = set().union(*members.values())
    if not union:
        raise RuntimeError(
            "no wallet qualifies for ANY tier — not even s_all (in-sample skill > 0). "
            "There is nothing to pre-register; report the null instead of freezing "
            "an empty artifact.")
    frozen = table[table["wallet"].isin(union)].copy()
    for tier in TIER_ORDER:
        frozen[tier] = frozen["wallet"].isin(members[tier])

    # per-wallet pre-freeze event breadth — the number that says how much
    # independent evidence each tier actually carries
    ev = bets.groupby("wallet").agg(pre_freeze_bets=("event", "size"),
                                    pre_freeze_events=("event", "nunique"),
                                    pre_freeze_leagues=("niche_l1", "nunique"))
    frozen = frozen.merge(ev, on="wallet", how="left")

    stats = {}
    for tier in TIER_ORDER:
        sub = frozen[frozen[tier]]
        tier_bets = bets[bets["wallet"].isin(members[tier])]
        stats[tier] = {
            "n_wallets": int(len(sub)),
            "median_out_sample_skill": (float(sub["out_sample_skill"].median())
                                        if len(sub) else None),
            "median_held_out_events": (float(sub["out_complexes"].median())
                                       if len(sub) and "out_complexes" in sub else None),
            "distinct_pre_freeze_events": int(tier_bets["event"].nunique()),
            "distinct_pre_freeze_leagues": int(tier_bets["niche_l1"].nunique()),
            "pre_freeze_bets": int(len(tier_bets)),
        }

    nesting = {a: [b for b in TIER_ORDER
                   if a != b and members[a] and members[a] <= members[b]]
               for a in TIER_ORDER}
    baseline = fit_hierarchical_baseline(
        resid, levels=sv.BASELINE_MODES[sv.STANDARD_BASELINE])
    frozen = frozen.sort_values("out_sample_skill", ascending=False).reset_index(drop=True)
    return frozen, baseline, {"tiers": stats, "subset_of": nesting}


def _forward_observations_exist() -> int:
    """How many post-freeze bets have already been fetched. A re-freeze is only
    legitimate pre-registration while this is ZERO."""
    if not FORWARD_TRADES_PATH.exists():
        return 0
    try:
        return int(len(pd.read_parquet(FORWARD_TRADES_PATH, columns=["timestamp"])))
    except Exception:  # noqa: BLE001
        return -1


def freeze(cfg: dict | None = None, freeze_ts: int | None = None,
           force: bool = False, amend_reason: str | None = None) -> dict:
    """Write the pre-registered sports freeze artifact + manifest.

    Refuses to overwrite an existing freeze unless `force`, and records every
    re-freeze in an `amendments` log together with the number of forward
    observations that existed at the time — the audit trail that makes "this was
    still pre-registration" checkable rather than asserted."""
    import src.sports_validate as sv

    cfg = cfg or load_config()
    prior = None
    if FREEZE_MANIFEST_PATH.exists():
        with open(FREEZE_MANIFEST_PATH) as fh:
            prior = json.load(fh)
        if not force:
            raise SystemExit(
                f"{FREEZE_MANIFEST_PATH} already exists (frozen "
                f"{prior.get('freeze_utc')}). Cohorts are NOT re-frozen in light of "
                f"forward outcomes — pass --force with a reason if this is a "
                f"legitimate pre-registration amendment.")

    frozen, baseline, tier_stats = build_tiers(cfg=cfg)
    ts = int(freeze_ts if freeze_ts is not None
             else (prior["freeze_ts"] if prior else time.time()))

    sports_cfg = (cfg.get("scoring", {}).get("sports", {}) or {})
    from src.slow_validate import (
        DEFAULT_MIN_BETS_PER_HALF,
        DEFAULT_MIN_EFF_BREADTH,
        DEFAULT_MIN_EFF_BREADTH_COMPLEX,
        DEFAULT_MIN_ENTRY_DAYS,
        DEFAULT_MIN_ENTRY_DAYS_COMPLEX,
        DEFAULT_MIN_OOS_COMPLEXES,
        DEFAULT_MIN_OOS_MARKETS,
        DEFAULT_MIN_RESOLUTION_DAYS,
        DEFAULT_MIN_RESOLUTION_DAYS_COMPLEX,
        DEFAULT_MIN_SKILL_EDGE,
    )
    scoring = cfg.get("scoring", {})
    gates = {
        "min_skill_edge": float(sports_cfg.get("min_skill_edge", DEFAULT_MIN_SKILL_EDGE)),
        "min_oos_markets": int(sports_cfg.get("min_oos_markets", DEFAULT_MIN_OOS_MARKETS)),
        "min_oos_events": int(sports_cfg.get("min_oos_complexes", DEFAULT_MIN_OOS_COMPLEXES)),
        "min_eff_breadth": float(sports_cfg.get("min_eff_breadth", DEFAULT_MIN_EFF_BREADTH)),
        "min_eff_breadth_event": float(sports_cfg.get("min_eff_breadth_complex",
                                                      DEFAULT_MIN_EFF_BREADTH_COMPLEX)),
        "min_entry_days": int(sports_cfg.get("min_entry_days", DEFAULT_MIN_ENTRY_DAYS)),
        "min_entry_days_event": int(sports_cfg.get("min_entry_days_complex",
                                                   DEFAULT_MIN_ENTRY_DAYS_COMPLEX)),
        "min_resolution_days": int(sports_cfg.get("min_resolution_days",
                                                  DEFAULT_MIN_RESOLUTION_DAYS)),
        "min_resolution_days_event": int(sports_cfg.get("min_resolution_days_complex",
                                                        DEFAULT_MIN_RESOLUTION_DAYS_COMPLEX)),
        "min_bets_per_half": int(sports_cfg.get("min_bets_per_half", DEFAULT_MIN_BETS_PER_HALF)),
        "oos_significance_alpha": float(scoring.get("oos_significance_alpha", 0.05)),
        "oos_bootstrap_resamples": int(scoring.get("oos_bootstrap_resamples", 2000)),
        "oos_split": float(scoring.get("oos_split", 0.5)),
        "baseline": sv.STANDARD_BASELINE,
        "baseline_levels": list(sv.BASELINE_MODES[sv.STANDARD_BASELINE]),
        "cluster_unit": sv.CLUSTER_COL,
        "event_gates": True,
        "config_section": f"scoring.{sv.CFG_SECTION}",
        "global_min_skill_edge_untouched": float(scoring.get("min_skill_edge", 0.02)),
    }

    tiers = {t: {**tier_stats["tiers"][t], "selection_rule": TIER_SPECS[t]["selector"],
                 "vetting": TIER_SPECS[t]["vetting"],
                 "headline": bool(TIER_SPECS[t]["headline"]),
                 "description": TIER_SPECS[t]["description"],
                 "subset_of": tier_stats["subset_of"].get(t, [])}
             for t in TIER_ORDER}

    obs = _forward_observations_exist()
    amendments = list(prior.get("amendments", [])) if prior else []
    if prior is not None:
        amendments.append({
            "amended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "amended_commit": _git_commit(),
            "reason": amend_reason or "unspecified",
            "forward_observations_at_amendment": obs,
            "legitimate_pre_registration": obs == 0,
            "prior_scoring_rule": prior.get("scoring_rule"),
            "freeze_ts_preserved": int(prior["freeze_ts"]) == ts,
        })

    manifest = {
        "arm": "sports",
        "freeze_ts": ts,
        "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "git_commit": _git_commit(),
        "sports_event_version": SPORTS_EVENT_VERSION,
        "niche_partition_version": NICHE_PARTITION_VERSION,
        "headline_tier": HEADLINE_TIER,
        "tier_order": TIER_ORDER,
        "tiers": tiers,
        "tier_note": (
            "Tiers are SELECTION rules only; forward scoring is identical across all "
            "of them. s_core is the headline and is never displaced by a wider tier's "
            "forward number."),
        "gates": gates,
        "resolution_speed_split_days": FAST_RESOLUTION_DAYS,
        "scoring_rule": SCORING_RULE,
        "baseline": baseline,
        "forward_observations_at_freeze": obs,
        "amendments": amendments,
        "known_limits": [
            "The SCREEN that produced these candidates ran on a corpus whose 284 "
            "markets fold to ~25 resolution events, with 81% of findable wallets "
            "spanning <=2 of them. It is a selection device only; everything here "
            "rests on the disjoint deep data the screen never saw.",
            "The gate's population statistics (8.32x dispersion excess, split-half "
            "rho +0.493) are POPULATION claims about 233k wallets, not evidence "
            "about any frozen wallet. Between-event persistence — the statistic that "
            "asks the copyable question — is much weaker at rho=+0.180.",
            "Identification only. Copyability (metric B) is UNTESTED: sports lines "
            "move fast and an edge that is real may not be fillable. This is the "
            "priority next step, and sports is the one population where it can be "
            "answered in days rather than months.",
            "Wider tiers (s_wide, s_all) apply no significance test and are expected "
            "to contain many false positives. Grading that is the point of freezing "
            "them, not a defect.",
            "Forward margins are expected to be SMALLER than the retrospective ones. "
            "That is the consequence of removing selection, not a failure.",
        ],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(frozen, FROZEN_SET_PATH, compression="gzip")
    atomic_write_json(manifest, FREEZE_MANIFEST_PATH)
    return manifest


def load_freeze() -> tuple[pd.DataFrame, dict]:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{FREEZE_MANIFEST_PATH} not found — run "
            f"`python -m src.sports_forward freeze` first.")
    with open(FREEZE_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return pd.read_parquet(FROZEN_SET_PATH), manifest


# ---------------------------------------------------------------------------
# SCORE
# ---------------------------------------------------------------------------

def score(cfg: dict | None = None, refetch: bool = True, n_boot: int = 2000) -> dict:
    """Score the frozen tiers on SPORTS bets entered after the freeze timestamp.

    The baseline is the one persisted in the manifest, applied without refitting;
    the event unit is the frozen `sports_event_version` partition."""
    import src.sports_validate as sv
    from src.ingest import fold_trades_to_ledger, update_resolutions
    from src.slow_aggregate import score_tier
    from src.slow_baseline import expected_outcome_hier
    from src.slow_forward import fetch_forward_trades
    from src.slow_market import derive_categories
    from src.slow_niche import assign_niches
    from src.sports_events import assign_events, assign_sub_forms, is_sports_market

    cfg = cfg or load_config()
    frozen, manifest = load_freeze()
    freeze_ts = int(manifest["freeze_ts"])
    wallets = list(frozen["wallet"].unique())
    print(f"[sports_forward] freeze {manifest['freeze_utc']} "
          f"({manifest['git_commit'][:8]}) — {len(wallets)} frozen wallets "
          f"(headline {manifest['tiers'][HEADLINE_TIER]['n_wallets']})")

    FORWARD_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refetch or not FORWARD_TRADES_PATH.exists():
        raw = fetch_forward_trades(wallets, freeze_ts, cfg=cfg)
        if raw.empty:
            return _await_data(frozen, manifest, "no post-freeze trades yet")
        atomic_to_parquet(raw, FORWARD_TRADES_PATH, compression="gzip")
    else:
        raw = pd.read_parquet(FORWARD_TRADES_PATH)

    session = make_session()
    existing = (pd.read_parquet(FORWARD_RESOLUTIONS_PATH)
                if FORWARD_RESOLUTIONS_PATH.exists()
                else pd.DataFrame(columns=["market_id", "token_id", "resolved",
                                           "resolved_value", "closed"]))
    cids = set(raw["conditionId"].dropna().unique()) if "conditionId" in raw else set()
    res = update_resolutions(session, cfg, cids, existing)
    atomic_to_parquet(res, FORWARD_RESOLUTIONS_PATH, compression="gzip")

    bets = fold_trades_to_ledger(raw.to_dict("records"), res)
    bets = bets[(bets["side"] == "BUY") & bets["resolved"].fillna(False)
                & (bets["timestamp"] > freeze_ts)].copy()
    if bets.empty:
        return _await_data(frozen, manifest, "post-freeze trades captured, none resolved yet")

    bets["category"] = derive_categories(bets["slug"], bets["question"])
    bets = bets[[is_sports_market(s, q, c) for s, q, c in
                 zip(bets["slug"], bets["question"], bets["category"])]].copy()
    if bets.empty:
        return _await_data(frozen, manifest, "no resolved SPORTS bets yet")
    bets = assign_events(assign_sub_forms(assign_niches(bets)))
    exp = expected_outcome_hier(manifest["baseline"], bets)
    bets["expected_outcome"] = exp
    bets["residual_skill"] = bets["resolved_value"].to_numpy(dtype=float) - exp
    # resolution time proxy: the last observed trade in that market post-freeze
    bets["resolution_ts"] = bets.groupby("market_id")["timestamp"].transform("max")

    rows = []
    for tier in TIER_ORDER:
        members = set(frozen.loc[frozen[tier], "wallet"])
        rows.extend(score_tier(bets[bets["wallet"].isin(members)], tier,
                               FAST_RESOLUTION_DAYS, event_col=sv.CLUSTER_COL,
                               n_boot=n_boot))
    board = pd.DataFrame(rows)
    _write_board(board, manifest, note=None)
    return {"board": board, "manifest": manifest, "bets": len(bets)}


def _await_data(frozen: pd.DataFrame, manifest: dict, why: str) -> dict:
    """Emit a live scoreboard that says, explicitly, that it is waiting — so
    'no data yet' is a visible state rather than an empty file."""
    rows = [{"tier": t, "event_unit": "event", "stratum": "all", "n_bets": 0,
             "n_wallets": int(manifest["tiers"][t]["n_wallets"]), "n_events": 0,
             "effective_events": 0.0, "edge_event_weighted": np.nan,
             "edge_bet_weighted": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "p_one_sided": np.nan} for t in TIER_ORDER]
    board = pd.DataFrame(rows)
    _write_board(board, manifest, note=why)
    print(f"[sports_forward] {why} — scoreboard written in 'awaiting data' state.")
    return {"board": board, "manifest": manifest, "bets": 0}


def _write_board(board: pd.DataFrame, manifest: dict, note: str | None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(board, SCOREBOARD_PARQUET, compression="gzip")
    lines = [
        "# Sports forward scoreboard (paper, read-only)",
        "",
        f"**Frozen {manifest['freeze_utc']}** at commit `{manifest['git_commit'][:8]}`, "
        f"event partition v{manifest['sports_event_version']}. "
        f"Headline tier: `{manifest['headline_tier']}`.",
        "",
    ]
    if note:
        lines += [f"> **AWAITING FORWARD DATA** — {note}. Sports resolves in "
                  f"hours-to-days, so this should populate within days, not months.",
                  ""]
    lines += ["| tier | wallets | pre-freeze events | bets | events | eff. events | "
              "edge (event-wtd) | 95% CI | p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def _num(v, fmt):
        return "—" if v is None or pd.isna(v) else format(v, fmt)

    for t in TIER_ORDER:
        row = board[(board["tier"] == t) & (board["stratum"] == "all")]
        r = row.iloc[0] if len(row) else None
        tier = manifest["tiers"][t]
        head = "**" if tier["headline"] else ""
        get = (lambda k: None) if r is None else (lambda k: r[k])
        ci = "—"
        if r is not None and not pd.isna(r["ci_low"]):
            ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        lines.append(
            f"| {head}{t}{head} | {tier['n_wallets']} "
            f"| {tier['distinct_pre_freeze_events']} "
            f"| {0 if r is None else int(r['n_bets'])} "
            f"| {0 if r is None else int(r['n_events'])} "
            f"| {_num(get('effective_events'), '.2f')} "
            f"| {_num(get('edge_event_weighted'), '+.4f')} "
            f"| {ci} "
            f"| {_num(get('p_one_sided'), '.3f')} |")
    lines += ["", "## The scoring rule, verbatim from the manifest", "",
              "> " + manifest["scoring_rule"].replace(". ", ".\n> "), "",
              "## Known limits, recorded at freeze time", ""]
    lines += [f"- {lim}" for lim in manifest["known_limits"]]
    SCOREBOARD_MD.write_text("\n".join(lines) + "\n")
    print(f"[sports_forward] -> {SCOREBOARD_MD}\n[sports_forward] -> {SCOREBOARD_PARQUET}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Freeze / score the sports forward tier (Project 3 sports step 5).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="write the pre-registered freeze artifact")
    f.add_argument("--force", action="store_true",
                   help="overwrite an existing freeze (records an amendment)")
    f.add_argument("--reason", type=str, default=None)
    s = sub.add_parser("score", help="score post-freeze sports bets")
    s.add_argument("--no-refetch", action="store_true")
    s.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    if args.cmd == "freeze":
        m = freeze(force=args.force, amend_reason=args.reason)
        print(f"[sports_forward] FROZEN {m['freeze_utc']} @ {m['git_commit'][:8]}")
        for t in TIER_ORDER:
            ti = m["tiers"][t]
            print(f"  {t:7s} n={ti['n_wallets']:4d}  events={ti['distinct_pre_freeze_events']:5d}  "
                  f"leagues={ti['distinct_pre_freeze_leagues']:3d}  "
                  f"median edge={ti['median_out_sample_skill']}")
        print(f"  -> {FROZEN_SET_PATH}\n  -> {FREEZE_MANIFEST_PATH}")
        score(refetch=False)   # write the live 'awaiting data' scoreboard
    else:
        score(refetch=not args.no_refetch, n_boot=args.boot)


if __name__ == "__main__":
    main()
