"""Project 3 steps 7d/8 — the pre-registered FREEZE and the forward paper test.

WHY THIS EXISTS
---------------
Every retrospective control this repo has run answers "was this edge already
there?" — never "does it continue?". Steps 5–7 walked a confound regress: raw
edge -> favorite-longshot -> category miscalibration -> niche miscalibration ->
rolling-deadline ladders -> correlated narratives. Each level was closable, and
each revealed another above it. That regress does not terminate retrospectively.
The forward test is what breaks it, because a wallet cannot have selected itself
on data that does not exist yet.

THE PRE-REGISTRATION
--------------------
`freeze` writes a timestamped, git-tracked artifact naming the tiers, their
metrics, every gate parameter, the partition versions, the fitted baseline, and
the scoring rule — BEFORE any forward data exists. `score` later pulls those
wallets' bets ENTERED AFTER the freeze timestamp and scores them under exactly
that rule.

FIVE pre-registered tiers (step 8), all vetted on the same pre-freeze data and
sharing one freeze cutoff. See TIER_SPECS: t12 is the HEADLINE; t30/t52/t87/t699
are progressively wider and progressively less vetted, frozen so forward evidence
accumulates in aggregate rather than trickling from one thin cohort.

THE UNIT OF INFERENCE IS THE TIER, NOT THE WALLET. Three post-freeze bets cannot
grade a wallet; 3xN bets can grade a crowd, because luck-selected wallets wash to
zero forward. Scoring is therefore aggregate and EVENT-WEIGHTED (src/slow_aggregate)
so one hyperactive rolling-deadline ladder cannot become the answer.

The freeze is pre-committed in the strong sense: **the forward result is scored
on whatever is frozen here, and the tiers are not re-frozen in light of forward
outcomes.** Picking the tier after seeing which one worked would reintroduce
exactly the selection this whole chain exists to eliminate. `--amend` is
permitted ONLY while zero forward observations exist, and every amendment is
logged with the observation count at the time so the claim stays checkable.

Scoring uses the baseline FIT ON PRE-FREEZE DATA ONLY, persisted in the manifest.
No refit at score time — otherwise the forward number could drift because the
baseline moved rather than because the wallets did.

READ-ONLY
---------
`score` makes public unauthenticated GETs (`/trades?user=`, the CLOB resolution
endpoint) — the same calls `backfill`/`ingest` already make. No keys, no signing,
no order placement, nothing on-chain. This is a measurement instrument.
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
from src.slow_niche import (
    NARRATIVE_MAP_VERSION,
    NICHE_PARTITION_VERSION,
    assign_narratives,
    assign_niches,
    narrative_spread,
)

FROZEN_SET_PATH = PROCESSED_DIR / "slow_frozen_set.parquet"
FREEZE_MANIFEST_PATH = PROCESSED_DIR / "slow_freeze_manifest.json"
FORWARD_TRADES_PATH = INTERIM_DIR / "slow_forward" / "forward_trades.parquet"
FORWARD_RESOLUTIONS_PATH = INTERIM_DIR / "slow_forward" / "forward_resolutions.parquet"
FORWARD_SCORES_PATH = INTERIM_DIR / "slow_forward" / "forward_scores.parquet"
SCOREBOARD_PARQUET = PROCESSED_DIR / "slow_forward_scoreboard.parquet"
SCOREBOARD_MD = PROCESSED_DIR / "slow_forward_scoreboard.md"

# --- pre-registered tiers ---------------------------------------------------
#
# A tier is a SELECTION rule applied to pre-freeze data, nothing more. Forward
# scoring is IDENTICAL across all tiers, so a tier is only a statement about how
# heavily vetted its members are — it never changes how a bet is scored.
#
# The tiers are NOT a clean nested ladder. There are two overlapping ones, because
# they were selected under different baselines:
#     legacy   t699 > t87 > t52   (coarse category baseline, market clusters)
#     standard t30  > t12         (niche_form baseline, event-complex clusters)
# t30 is NOT a subset of t52 (7 wallets outside) and t12 is not either (2 outside).
# Presenting them as one monotone scale would be wrong.
#
# MEASURED AT FREEZE TIME: widening buys VOLUME, not narrative breadth. t699 has
# 58x the wallets of t12 and 195x the pre-freeze bets, but only 3.47 vs 3.13
# effective independent narratives — and t87/t52 are actually NARROWER than t12
# (2.68 / 2.24), because the wide crowd is disproportionately mideast_escalation.
# Every reported bet count is therefore paired with an effective-event count, so
# volume is never mistaken for evidence.
TIER_SPECS: dict[str, dict] = {
    "t12": {
        "config": "standard_cx",
        "selector": "edge_persisted",
        "vetting": "highest",
        "headline": True,
        "description": ("niche_form baseline, event-complex clusters, complex-unit "
                        "concentration gates. THE pre-registered headline."),
    },
    "t30": {
        "config": "standard",
        "selector": "edge_persisted",
        "vetting": "high",
        "headline": False,
        "description": ("niche_form baseline, event-complex clusters, WITHOUT the "
                        "complex-unit concentration gates. Re-admits single-complex "
                        "concentration."),
    },
    "t52": {
        "config": "legacy",
        "selector": "edge_persisted",
        "vetting": "medium",
        "headline": False,
        "description": ("the 5c rule: coarse category baseline, market-block "
                        "clusters. Counts a rolling-deadline ladder as many "
                        "independent draws."),
    },
    "t87": {
        "config": "legacy",
        "selector": "candidate & magnitude_ok & markets_ok & concentration_ok",
        "vetting": "low",
        "headline": False,
        "description": ("the wide vetted crowd: clears the 10c magnitude, held-out "
                        "market-count and concentration gates, but NOT any "
                        "significance test."),
    },
    "t699": {
        "config": "legacy",
        "selector": "candidate",
        "vetting": "least — volume only",
        "headline": False,
        "description": ("in-sample residual skill > 0 and nothing else. LEAST "
                        "VETTED; frozen for forward VOLUME. Expected to be mostly "
                        "noise — grading that is the point."),
    },
}

HEADLINE_TIER = "t12"
TIER_ORDER = ["t12", "t30", "t52", "t87", "t699"]

# Resolution-speed split, fixed here so it cannot drift between runs.
FAST_RESOLUTION_DAYS = 14

# The pre-committed scoring rule, recorded verbatim in the manifest.
SCORING_RULE = (
    "For each frozen wallet, take its resolved BUY bets in slow-bucket markets "
    "ENTERED STRICTLY AFTER freeze_ts. Per-bet skill edge = resolved_value - "
    "E[outcome | entry_price, niche], where E is the hierarchical baseline frozen "
    "in this manifest (fit on pre-freeze data only, applied WITHOUT refitting). "
    "Each TIER is scored in AGGREGATE as a crowd, never wallet-by-wallet: the "
    "headline statistic is the EVENT-WEIGHTED mean, i.e. the mean over event "
    "complexes of the within-complex mean residual, so one hyperactive ladder "
    "cannot carry a tier. The bet-weighted mean is reported alongside. "
    "Uncertainty is a cluster bootstrap that resamples EVENT COMPLEXES with "
    "replacement (not bets), giving a 95% CI and a one-sided p vs 0. Every bet "
    "count is reported with n_events and effective_events so volume is not "
    "mistaken for evidence. Results are additionally split by resolution speed "
    "(<=14 days from entry to resolution vs longer) and compared against a "
    "wallet-agnostic placebo: the same statistic computed over non-cohort wallets "
    "trading the same markets, plus a count-matched random-crowd null. "
    "t12 is the pre-registered headline tier; all other tiers are secondary "
    "readings and are NEVER promoted on the basis of forward results. Tiers are "
    "not re-frozen in light of forward outcomes."
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
    """Re-derive every pre-registered tier from pre-freeze data.

    Returns (frozen table with one row per wallet in the union and a boolean
    column per tier, fitted baseline, per-tier stats). All tiers are vetted on
    the SAME pre-freeze data and share the SAME freeze cutoff."""
    import src.slow_validate as sv
    from src.slow_baseline import fit_hierarchical_baseline
    from src.slow_niche import effective_n

    cfg = cfg or load_config()
    oos_split = float(cfg.get("scoring", {}).get("oos_split", 0.5))
    bets = sv.load_deep_slow_bets_niched(cfg=cfg)

    # The two selection configurations behind the two overlapping ladders.
    legacy_resid, _ = sv.residualize_for(bets, "category", cfg=cfg)
    tables = {
        "legacy": sv.validate_slow(legacy_resid, baselines={}, cfg=cfg,
                                   pre_residualized=True, cluster_col="market_id",
                                   complex_gates=False),
    }
    del legacy_resid

    resid, _ = sv.residualize_for(bets, sv.STANDARD_BASELINE, cfg=cfg)
    resid = assign_narratives(resid)
    tables["standard"] = sv.validate_slow(resid, baselines={}, cfg=cfg,
                                          pre_residualized=True,
                                          cluster_col=sv.STANDARD_CLUSTER,
                                          complex_gates=False)
    tables["standard_cx"] = sv.validate_slow(resid, baselines={}, cfg=cfg,
                                             pre_residualized=True,
                                             cluster_col=sv.STANDARD_CLUSTER,
                                             complex_gates=True)

    members: dict[str, set] = {}
    for tier, spec in TIER_SPECS.items():
        t = tables[spec["config"]]
        members[tier] = set(t.query(spec["selector"])["wallet"])

    union = set().union(*members.values())
    # The STANDARD-config row is the canonical retrospective metric set for every
    # frozen wallet, whichever ladder selected it; the legacy metrics ride along
    # suffixed so a legacy-tier member's selection numbers stay inspectable.
    frozen = tables["standard"][tables["standard"]["wallet"].isin(union)].copy()
    legacy_cols = ["wallet", "out_sample_skill", "cluster_p", "eff_breadth",
                   "out_markets", "candidate", "magnitude_ok", "markets_ok",
                   "concentration_ok", "edge_persisted"]
    frozen = frozen.merge(
        tables["legacy"][legacy_cols].add_suffix("_legacy").rename(
            columns={"wallet_legacy": "wallet"}), on="wallet", how="left")

    for tier in TIER_ORDER:
        frozen[tier] = frozen["wallet"].isin(members[tier])
    # continuity with the step-7 manifest
    frozen["cohort"] = np.where(frozen["t12"], "primary",
                                np.where(frozen["t30"], "secondary", "wider"))

    spread = narrative_spread(resid, union, oos_split=oos_split)
    frozen = frozen.merge(
        spread[["wallet", "n_narratives_broad", "eff_narratives_broad",
                "top_narrative_broad", "top_narrative_share_broad",
                "top_narrative_share_strict", "single_narrative_flag"]],
        on="wallet", how="left")

    pre_freeze_bets = bets.groupby("wallet").size()
    stats = {}
    for tier in TIER_ORDER:
        sub = frozen[frozen[tier]]
        sp = spread[spread["wallet"].isin(members[tier])]
        stats[tier] = {
            "n_wallets": int(len(sub)),
            "median_out_sample_skill": float(sub["out_sample_skill"].median()) if len(sub) else None,
            "effective_independent_narratives": (
                float(effective_n(sp["top_narrative_broad"])) if len(sp) else 0.0),
            "median_top_narrative_share": (
                float(sp["top_narrative_share_broad"].median()) if len(sp) else None),
            "single_narrative_wallets": int(sp["single_narrative_flag"].sum()) if len(sp) else 0,
            "pre_freeze_slow_bets": int(pre_freeze_bets.reindex(list(members[tier])).fillna(0).sum()),
        }

    # honest record of the non-nesting, so nobody reads the tiers as one scale
    nesting = {}
    for a in TIER_ORDER:
        for b in TIER_ORDER:
            if a != b and members[a] and members[a] <= members[b]:
                nesting.setdefault(a, []).append(b)

    baseline = fit_hierarchical_baseline(
        resid, levels=sv.BASELINE_MODES[sv.STANDARD_BASELINE])
    frozen = frozen.sort_values("out_sample_skill", ascending=False).reset_index(drop=True)
    return frozen, baseline, {"tiers": stats, "subset_of": nesting}


def _forward_observations_exist() -> int:
    """How many post-freeze bets have already been fetched. An amendment is only
    legitimate pre-registration while this is ZERO — once forward outcomes are
    observable, changing the cohorts is no longer a pre-commitment."""
    if not FORWARD_TRADES_PATH.exists():
        return 0
    try:
        return int(len(pd.read_parquet(FORWARD_TRADES_PATH, columns=["timestamp"])))
    except Exception:  # noqa: BLE001
        return -1


def freeze(cfg: dict | None = None, freeze_ts: int | None = None,
           amend_reason: str | None = None) -> dict:
    """Write the pre-registered freeze artifact + manifest. Read-only w.r.t. every
    other pipeline output.

    When a manifest already exists, its `freeze_ts` is REUSED so every tier shares
    one cutoff, and the change is recorded in an `amendments` log together with the
    number of forward observations that existed at the time — the audit trail that
    makes "this was still pre-registration" checkable rather than asserted."""
    import src.slow_validate as sv

    cfg = cfg or load_config()
    prior = None
    if FREEZE_MANIFEST_PATH.exists():
        with open(FREEZE_MANIFEST_PATH) as fh:
            prior = json.load(fh)

    frozen, baseline, tier_stats = build_tiers(cfg=cfg)
    # Reuse the original cutoff on amendment so all tiers share one timestamp.
    ts = int(freeze_ts if freeze_ts is not None
             else (prior["freeze_ts"] if prior else time.time()))

    slow_cfg = (cfg.get("scoring", {}).get("slow", {}) or {})
    gates = {
        "min_skill_edge": float(slow_cfg.get("min_skill_edge", sv.DEFAULT_MIN_SKILL_EDGE)),
        "min_oos_markets": int(slow_cfg.get("min_oos_markets", sv.DEFAULT_MIN_OOS_MARKETS)),
        "min_eff_breadth": float(slow_cfg.get("min_eff_breadth", sv.DEFAULT_MIN_EFF_BREADTH)),
        "min_entry_days": int(slow_cfg.get("min_entry_days", sv.DEFAULT_MIN_ENTRY_DAYS)),
        "min_resolution_days": int(slow_cfg.get("min_resolution_days", sv.DEFAULT_MIN_RESOLUTION_DAYS)),
        "min_oos_complexes": int(slow_cfg.get("min_oos_complexes", sv.DEFAULT_MIN_OOS_COMPLEXES)),
        "min_eff_breadth_complex": float(slow_cfg.get("min_eff_breadth_complex", sv.DEFAULT_MIN_EFF_BREADTH_COMPLEX)),
        "min_entry_days_complex": int(slow_cfg.get("min_entry_days_complex", sv.DEFAULT_MIN_ENTRY_DAYS_COMPLEX)),
        "min_resolution_days_complex": int(slow_cfg.get("min_resolution_days_complex", sv.DEFAULT_MIN_RESOLUTION_DAYS_COMPLEX)),
        "oos_significance_alpha": float(cfg.get("scoring", {}).get("oos_significance_alpha", 0.05)),
        "oos_bootstrap_resamples": int(cfg.get("scoring", {}).get("oos_bootstrap_resamples", 2000)),
        "oos_split": float(cfg.get("scoring", {}).get("oos_split", 0.5)),
        "baseline": sv.STANDARD_BASELINE,
        "cluster_unit": sv.STANDARD_CLUSTER,
        "complex_gates": True,
    }

    tiers = {}
    for tier in TIER_ORDER:
        spec = TIER_SPECS[tier]
        tiers[tier] = {
            **tier_stats["tiers"][tier],
            "selection_rule": spec["selector"],
            "selection_config": spec["config"],
            "vetting": spec["vetting"],
            "headline": bool(spec["headline"]),
            "description": spec["description"],
            "subset_of": tier_stats["subset_of"].get(tier, []),
        }

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
            "prior_tiers": sorted((prior.get("tiers") or prior.get("cohorts") or {}).keys()),
            "freeze_ts_preserved": int(prior["freeze_ts"]) == ts,
        })

    p = frozen[frozen["t12"]]
    s = frozen[frozen["t30"]]
    manifest = {
        "freeze_ts": ts,
        "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "git_commit": _git_commit(),
        "niche_partition_version": NICHE_PARTITION_VERSION,
        "narrative_map_version": NARRATIVE_MAP_VERSION,
        "headline_tier": HEADLINE_TIER,
        "tier_order": TIER_ORDER,
        "tiers": tiers,
        "tiers_are_nested": False,
        "tier_note": (
            "Tiers are SELECTION rules only; forward scoring is identical across all "
            "of them. They are NOT one monotone ladder: t699>t87>t52 were selected "
            "under the legacy coarse baseline with market clusters, while t30>t12 use "
            "the niche_form baseline with event-complex clusters, so t30 and t12 are "
            "NOT subsets of t52. Widening buys VOLUME, not narrative breadth — see "
            "effective_independent_narratives on each tier."),
        # step-7 keys kept so the earlier manifest's readers still resolve
        "headline_cohort": "primary",
        "cohorts": {
            "primary": {"n_wallets": int(len(p)), "equals_tier": "t12"},
            "secondary": {"n_wallets": int(len(s)), "equals_tier": "t30",
                          "inclusive_of_primary": True},
        },
        "gates": gates,
        "resolution_speed_split_days": FAST_RESOLUTION_DAYS,
        "scoring_rule": SCORING_RULE,
        "baseline": baseline,
        "amendments": amendments,
        "known_limits": [
            "WIDENING BUYS VOLUME, NOT BREADTH. Measured at freeze time: t699 has 58x "
            "the wallets and 195x the pre-freeze bets of t12, but only 3.47 vs 3.13 "
            "effective independent narratives — and t87/t52 are NARROWER than t12 "
            "(2.68 / 2.24) because the wide crowd is disproportionately "
            "mideast_escalation. Forward bet counts will grow far faster than forward "
            "evidence. Always read n_bets next to effective_events.",
            "Every tier carries only ~2-3.5 EFFECTIVE independent narratives, so all "
            "of them are far fewer independent pieces of evidence than n suggests.",
            "t12 is mideast_escalation-weighted (5 of 12 dominant). If that narrative "
            "goes quiet the headline tier may have nothing to score for months. Known "
            "and accepted at freeze time; it is why the wider tiers are frozen too.",
            "t87 and t699 are explicitly under-vetted (t699 applies no significance "
            "test at all) and are expected to contain many false positives. Grading "
            "that is the point of freezing them, not a defect.",
            "Slow markets resolve over days to weeks, so a scoreable forward sample "
            "takes months to accumulate. This is not a real-time signal.",
            "Forward margins are expected to be SMALLER than the retrospective ones. "
            "That is the expected outcome of removing selection, not a failure.",
            "~50% aggregate FDR context from the retrospective stage still applies to "
            "individual names; the tier aggregate is the unit of inference.",
            "Identification only. Copyability (metric B) is untested — an edge that is "
            "real is not thereby fillable.",
        ],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(frozen, FROZEN_SET_PATH, compression="gzip")
    atomic_write_json(manifest, FREEZE_MANIFEST_PATH)
    return manifest


# ---------------------------------------------------------------------------
# SCORE (run later — needs no state beyond the manifest)
# ---------------------------------------------------------------------------

def load_freeze() -> tuple[pd.DataFrame, dict]:
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{FREEZE_MANIFEST_PATH} not found — run `python -m src.slow_forward freeze` first.")
    with open(FREEZE_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return pd.read_parquet(FROZEN_SET_PATH), manifest


def fetch_forward_trades(wallets, freeze_ts: int, cfg: dict | None = None) -> pd.DataFrame:
    """Pull each frozen wallet's trades newer than the freeze timestamp via the
    public `/trades?user=` endpoint. Read-only; reuses the tested backfill pager."""
    from src.backfill import fetch_user_trades

    cfg = cfg or load_config()
    session = make_session()
    rows = []
    for i, w in enumerate(wallets, 1):
        try:
            got = fetch_user_trades(session, cfg, w, since_ts=freeze_ts)
        except Exception as e:  # noqa: BLE001 — one wallet must not abort the run
            print(f"  [{i}/{len(wallets)}] {w[:12]} FAILED: {e}")
            continue
        rows.extend(got)
        print(f"  [{i}/{len(wallets)}] {w[:12]} +{len(got)} trades")
    return pd.DataFrame(rows)


def score(cfg: dict | None = None, refetch: bool = True,
          with_placebo: bool = True,
          placebo_max_markets: int = 300) -> dict:
    """Score the frozen cohorts on bets entered after the freeze timestamp."""
    import src.slow_validate as sv
    from src.ingest import fold_trades_to_ledger, update_resolutions
    from src.market_meta import classify_speed_bucket, speed_thresholds
    from src.slow_baseline import expected_outcome_hier

    cfg = cfg or load_config()
    frozen, manifest = load_freeze()
    freeze_ts = int(manifest["freeze_ts"])
    wallets = list(frozen["wallet"].unique())
    print(f"[forward] freeze {manifest['freeze_utc']} ({manifest['git_commit'][:8]}) — "
          f"{len(wallets)} frozen wallets "
          f"(primary {manifest['cohorts']['primary']['n_wallets']}, "
          f"secondary {manifest['cohorts']['secondary']['n_wallets']})")

    FORWARD_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refetch or not FORWARD_TRADES_PATH.exists():
        raw = fetch_forward_trades(wallets, freeze_ts, cfg=cfg)
        if raw.empty:
            print("[forward] no post-freeze trades yet — nothing to score. "
                  "Slow markets resolve over weeks; re-run later.")
            return _await_data(frozen, manifest)
        atomic_to_parquet(raw, FORWARD_TRADES_PATH, compression="gzip")
    else:
        raw = pd.read_parquet(FORWARD_TRADES_PATH)

    session = make_session()
    existing = (pd.read_parquet(FORWARD_RESOLUTIONS_PATH)
                if FORWARD_RESOLUTIONS_PATH.exists()
                else pd.DataFrame(columns=["market_id", "token_id", "resolved",
                                           "resolved_value", "closed"]))
    cids = set(raw["conditionId"].dropna().unique())
    res = update_resolutions(session, cfg, cids, existing)
    atomic_to_parquet(res, FORWARD_RESOLUTIONS_PATH, compression="gzip")

    bets = fold_trades_to_ledger(raw.to_dict("records"), res)
    bets = bets[(bets["side"] == "BUY") & bets["resolved"].fillna(False)]
    bets = bets[bets["timestamp"] > freeze_ts].reset_index(drop=True)
    if bets.empty:
        print("[forward] post-freeze trades exist but none have resolved yet. "
              "Re-run later.")
        return _await_data(frozen, manifest)

    # slow-bucket filter, from each market's observed forward span
    slow_s, deep_s = speed_thresholds(cfg)
    span = bets.groupby("market_id")["timestamp"].agg(["min", "max"])
    life = (span["max"] - span["min"]).to_dict()
    bets["speed_bucket"] = bets["market_id"].map(
        lambda m: classify_speed_bucket(life.get(m, np.nan), slow_s, deep_s))
    bets = bets[bets["speed_bucket"].isin(sv.SLOW_BUCKETS)].reset_index(drop=True)
    if bets.empty:
        print("[forward] no resolved SLOW post-freeze bets yet. Re-run later.")
        return _await_data(frozen, manifest)

    from src.slow_market import derive_categories
    bets["category"] = derive_categories(bets["slug"], bets["question"])
    bets = assign_narratives(assign_niches(bets))

    # score against the FROZEN baseline — no refit
    exp = expected_outcome_hier(manifest["baseline"], bets)
    bets["expected_outcome"] = exp
    bets["residual_skill"] = bets["resolved_value"].to_numpy(dtype=float) - exp

    # Resolution time proxy: the market's last observed trade. Same lower-bound
    # convention as `res_ts_proxy` in the retrospective path — used only for the
    # resolution-speed split, never for scoring.
    last_seen = bets.groupby("market_id")["timestamp"].transform("max")
    bets["resolution_ts"] = last_seen

    bets = bets[bets["wallet"].isin(set(frozen["wallet"]))].reset_index(drop=True)
    if bets.empty:
        print("[forward] no resolved slow post-freeze bets from frozen wallets yet.")
        return _await_data(frozen, manifest)

    placebo, coverage = (pd.DataFrame(), {})
    if with_placebo:
        from src.slow_placebo import run_placebo
        placebo, coverage = run_placebo(
            bets, frozen, manifest, manifest.get("tier_order", TIER_ORDER),
            max_markets=placebo_max_markets, cfg=cfg)

    return report_scores(bets, frozen, manifest, cfg=cfg, placebo=placebo,
                         placebo_coverage=coverage)


def _await_data(frozen: pd.DataFrame, manifest: dict) -> dict:
    """Publish an empty-but-live scoreboard. The scoreboard exists from day one so
    "no data yet" is a visible state rather than a missing file."""
    tier_order = manifest.get("tier_order", TIER_ORDER)
    board = pd.DataFrame([{
        "tier": t, "stratum": "all", "event_unit": "niche_l1", "n_bets": 0,
        "n_wallets": 0, "n_events": 0, "effective_events": 0.0,
        "edge_event_weighted": float("nan"), "edge_bet_weighted": float("nan"),
        "ci_low": float("nan"), "ci_high": float("nan"),
        "p_one_sided": float("nan"),
        "frozen_wallets": int(frozen[t].sum()) if t in frozen.columns else 0,
        "headline": t == manifest.get("headline_tier", HEADLINE_TIER),
    } for t in tier_order])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(board, SCOREBOARD_PARQUET, compression="gzip")
    write_scoreboard_md(board, manifest, bets=None)
    print(f"[forward] scoreboard published (awaiting data) -> {SCOREBOARD_MD}")
    return {"forward_bets": 0, "resolved": 0, "board": board}


def report_scores(bets: pd.DataFrame, frozen: pd.DataFrame, manifest: dict,
                  cfg: dict | None = None, placebo: pd.DataFrame | None = None,
                  placebo_coverage: dict | None = None) -> dict:
    """Aggregate, event-weighted readout per pre-registered tier, plus the
    per-wallet detail and the scoreboard artifacts."""
    from src.slow_aggregate import add_fine_event_column, score_group, score_tier

    cfg = cfg or load_config()
    split_days = int(manifest.get("resolution_speed_split_days", FAST_RESOLUTION_DAYS))
    n_boot = int(manifest["gates"]["oos_bootstrap_resamples"])
    tier_order = manifest.get("tier_order", TIER_ORDER)

    rows = []
    for tier in tier_order:
        if tier not in frozen.columns:
            continue
        members = set(frozen.loc[frozen[tier], "wallet"])
        sub = bets[bets["wallet"].isin(members)]
        if sub.empty:
            rows.append({"tier": tier, "stratum": "all", "event_unit": "niche_l1",
                         "n_bets": 0, "n_wallets": 0, "n_events": 0,
                         "effective_events": 0.0,
                         "edge_event_weighted": float("nan"),
                         "edge_bet_weighted": float("nan"),
                         "ci_low": float("nan"), "ci_high": float("nan"),
                         "p_one_sided": float("nan")})
            continue
        rows.extend(score_tier(sub, tier, split_days, n_boot=n_boot))
        # secondary view on the finer event unit — reported so a reader can see how
        # much of the uncertainty is the unit choice, NEVER as the headline
        fine = add_fine_event_column(sub)
        r = score_group(fine, event_col="event_fine",
                        seed=abs(hash(tier + "fine")) % (2 ** 32), n_boot=n_boot)
        rows.append({"tier": tier, "stratum": "all", "event_unit": "event_fine", **r})

    board = pd.DataFrame(rows)
    board["frozen_wallets"] = board["tier"].map(
        {t: int(frozen[t].sum()) for t in tier_order if t in frozen.columns})
    board["headline"] = board["tier"] == manifest.get("headline_tier", HEADLINE_TIER)
    if placebo is not None and not placebo.empty:
        board = board.merge(placebo, on=["tier", "stratum", "event_unit"], how="left")

    per_wallet = (bets.groupby("wallet")
                  .agg(forward_bets=("residual_skill", "size"),
                       forward_edge=("residual_skill", "mean"),
                       complexes=("niche_l1", "nunique"))
                  .reset_index()
                  .merge(frozen[["wallet", "out_sample_skill"] +
                                [t for t in tier_order if t in frozen.columns]],
                         on="wallet", how="left"))
    per_wallet["edge_delta"] = per_wallet["forward_edge"] - per_wallet["out_sample_skill"]

    FORWARD_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(per_wallet, FORWARD_SCORES_PATH, compression="gzip")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(board, SCOREBOARD_PARQUET, compression="gzip")
    write_scoreboard_md(board, manifest, bets, placebo_coverage=placebo_coverage)

    print(f"\n[forward] {len(bets):,} resolved slow post-freeze bets "
          f"from {bets['wallet'].nunique()} frozen wallets")
    _print_board(board, manifest)
    return {"forward_bets": int(len(bets)), "resolved": int(len(bets)),
            "board": board}


def _print_board(board: pd.DataFrame, manifest: dict) -> None:
    head = manifest.get("headline_tier", HEADLINE_TIER)
    show = board[board["event_unit"] == "niche_l1"]
    cols = ["tier", "stratum", "n_bets", "n_wallets", "n_events", "effective_events",
            "edge_event_weighted", "edge_bet_weighted", "ci_low", "ci_high",
            "p_one_sided"]
    cols = [c for c in cols if c in show.columns]
    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(show[cols].round(4).to_string(index=False))
    print(f"\n  headline tier: {head} (all other tiers are secondary readings and are "
          f"never promoted on forward results)")
    print("  n_bets is VOLUME; effective_events is EVIDENCE. A tier with many bets "
          "across few events\n  carries the weight of the events, not the bets.")


def write_scoreboard_md(board: pd.DataFrame, manifest: dict,
                        bets: pd.DataFrame | None = None,
                        placebo_coverage: dict | None = None) -> None:
    """Human-readable running scoreboard. Regenerated on every scoring run."""
    head = manifest.get("headline_tier", HEADLINE_TIER)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    L = [
        "# Project 3 — forward paper test scoreboard",
        "",
        f"*Updated {now}. Frozen {manifest['freeze_utc']} at commit "
        f"`{manifest['git_commit'][:8]}`, partition v{manifest['niche_partition_version']}.*",
        "",
        f"**Headline tier: `{head}`.** All other tiers are pre-registered secondary "
        "readings and are never promoted on the basis of forward results.",
        "",
        "`n_bets` is **volume**; `effective_events` is **evidence**. A tier with many "
        "bets across few event complexes carries the weight of the events, not the "
        "bets — widening the crowd was measured at freeze time to buy volume, not "
        "narrative breadth.",
        "",
    ]
    total = 0 if bets is None else len(bets)
    if total == 0:
        L += ["## Status: awaiting forward data",
              "",
              "No resolved post-freeze slow bets yet. Slow markets resolve over days "
              "to weeks, so the first scoreable sample is expected to take weeks and "
              "a decisive one months. Re-run `python -m src.slow_forward score`.",
              ""]
    else:
        L += [f"## {total:,} resolved post-freeze slow bets", ""]

    show = board[board["event_unit"] == "niche_l1"] if "event_unit" in board else board
    if not show.empty:
        cols = [c for c in ["tier", "stratum", "n_bets", "n_wallets", "n_events",
                            "effective_events", "edge_event_weighted",
                            "edge_bet_weighted", "ci_low", "ci_high", "p_one_sided",
                            "placebo_edge", "placebo_percentile"]
                if c in show.columns]
        L += ["| " + " | ".join(cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in show.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append("—" if pd.isna(v) else f"{v:+.4f}"
                                 if c.startswith(("edge", "ci", "placebo_edge")) else f"{v:.4g}")
                else:
                    cells.append(str(v))
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    if placebo_coverage and placebo_coverage.get("markets_requested"):
        c = placebo_coverage
        L += ["## Placebo coverage", "",
              f"- markets involved: {c['markets_requested']}",
              f"- markets fetched: {c.get('markets_fetched', 0)}",
              f"- skipped by the per-run cap: {c.get('markets_skipped_by_cap', 0)}"
              + ("  **(TRUNCATED — the placebo covers a subset)**"
                 if c.get("truncated") else ""),
              f"- markets failed: {c.get('markets_failed', 0)}",
              ""]

    L += ["## Pre-registered scoring rule", "", "> " + manifest["scoring_rule"], "",
          "## Known limits (recorded at freeze time)", ""]
    L += [f"- {x}" for x in manifest.get("known_limits", [])]
    L.append("")
    SCOREBOARD_MD.parent.mkdir(parents=True, exist_ok=True)
    SCOREBOARD_MD.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="write the pre-registered freeze artifact")
    f.add_argument("--force", action="store_true",
                   help="overwrite an existing freeze. Refuses by default: "
                        "re-freezing after forward data exists destroys the "
                        "pre-registration.")
    f.add_argument("--amend", metavar="REASON",
                   help="amend an existing freeze IN PLACE, preserving its freeze_ts "
                        "so all tiers share one cutoff, and logging the change. "
                        "Only permitted while ZERO forward observations exist.")
    s = sub.add_parser("score", help="score frozen wallets on post-freeze bets")
    s.add_argument("--no-refetch", action="store_true",
                   help="reuse the cached forward trades instead of re-pulling")
    s.add_argument("--no-placebo", action="store_true",
                   help="skip the wallet-agnostic placebo (it is the only step that "
                        "fetches whole market tapes, so it dominates run time)")
    s.add_argument("--placebo-max-markets", type=int, default=300,
                   help="cap on market tapes fetched per run; any truncation is "
                        "reported in the scoreboard, never silent")
    args = ap.parse_args()

    if args.cmd == "freeze":
        exists = FREEZE_MANIFEST_PATH.exists()
        if exists and not (args.force or args.amend):
            with open(FREEZE_MANIFEST_PATH) as fh:
                existing = json.load(fh)
            print(f"[freeze] REFUSING — a freeze already exists from "
                  f"{existing.get('freeze_utc')} ({existing.get('git_commit', '')[:8]}).\n"
                  f"         Re-freezing after forward data exists would destroy the "
                  f"pre-registration.\n         Use --amend REASON to add tiers while "
                  f"zero forward data exists, or --force to replace outright.")
            return
        if args.amend:
            obs = _forward_observations_exist()
            if obs != 0:
                print(f"[freeze] REFUSING to amend — {obs} forward observations already "
                      f"exist.\n         Amending cohorts once forward outcomes are "
                      f"observable is no longer pre-registration.\n         Use --force "
                      f"only if you intend to abandon the existing pre-registration.")
                return
            print(f"[freeze] amending in place (0 forward observations — still "
                  f"pre-registration)")
        m = freeze(amend_reason=args.amend)
        print(f"[freeze] {m['freeze_utc']}  commit {m['git_commit'][:8]}  "
              f"partition v{m['niche_partition_version']}")
        print(f"  {'tier':<7}{'n':>6}{'median_edge':>14}{'eff_narr':>10}"
              f"{'med_top_share':>15}{'single_narr':>13}{'pre_freeze_bets':>17}  vetting")
        for t in m["tier_order"]:
            d = m["tiers"][t]
            star = " *HEADLINE*" if d["headline"] else ""
            print(f"  {t:<7}{d['n_wallets']:>6}{d['median_out_sample_skill']:>14.4f}"
                  f"{d['effective_independent_narratives']:>10.2f}"
                  f"{d['median_top_narrative_share']:>15.2f}"
                  f"{d['single_narrative_wallets']:>13}"
                  f"{d['pre_freeze_slow_bets']:>17,}  {d['vetting']}{star}")
        print(f"  headline tier: {m['headline_tier']}   tiers nested: "
              f"{m['tiers_are_nested']}")
        if m.get("amendments"):
            a = m["amendments"][-1]
            print(f"  amendment logged: {a['amended_utc']} "
                  f"(forward obs at amendment: {a['forward_observations_at_amendment']}, "
                  f"legitimate: {a['legitimate_pre_registration']}, "
                  f"freeze_ts preserved: {a['freeze_ts_preserved']})")
        print(f"  -> {FROZEN_SET_PATH}\n  -> {FREEZE_MANIFEST_PATH}")
    else:
        score(refetch=not args.no_refetch,
              with_placebo=not args.no_placebo,
              placebo_max_markets=args.placebo_max_markets)


if __name__ == "__main__":
    main()
