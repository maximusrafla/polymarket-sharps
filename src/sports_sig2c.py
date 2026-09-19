"""Project 3 sports — the `s_sig2c` arm: a SECOND pre-registered sports freeze.

Why this exists
---------------
The first sports freeze (`src/sports_forward.py`) gated on
`scoring.sports.min_skill_edge = 0.10`. The 2026-07-26 red-team audit
(`docs/redteam_audit_2026-07-26.md` §1) established that this floor was
mis-transposed: it was copied from the SLOW forecaster arm, whose own recorded
rationale is a "small-n, large-edge" universe, and applied to a large-n one
(1,135,069 deep bets / 40,256 events) where the repo's own scoping logic calls
for Project 1's 2c floor. Applied to *held-out* edge it is additionally a
winner's-curse selector: at fixed true skill it picks the most upward-noisy
record. It did exactly that — the certified `s_core` wallet runs in-sample
+0.1c -> held-out +18.8c on 182 events, while wallets with stationary in~=out
records and 3-5x more independent events were excluded.

Twelve wallets clear EVERY other gate (in-sample candidacy, event-clustered
bootstrap significance, market count, event-unit concentration) at >= 2c and
have no tier in the original freeze, so the running forward test cannot read
them: `s_wide` is 10c-gated and `s_all` dilutes them among 162 unvetted names.

What this module is NOT
-----------------------
It is **not** a retroactive relaxation of a pre-registered floor to fill a
table, and it does not touch `sports_freeze_manifest.json` — that artifact
already has forward observations against it and is correctly immutable. This is
a NEW cohort in a NEW artifact with its OWN freeze timestamp, exactly as the
sports freeze was itself legitimately created beside the forecaster freeze. Its
forward window therefore starts later and is strictly uncontaminated.

The selection is made entirely from PRE-freeze artifacts:
  * membership   — the five saved `sports_validated*.parquet` variant tables,
                   all computed before the original freeze;
  * pre-freeze   — `sports_frozen_set.parquet` (all 12 are candidates, so all
    breadth      are already in that union with their pre-freeze stats);
  * the baseline — reused VERBATIM from `sports_freeze_manifest.json`, never
                   refit, so forward numbers cannot drift and the two arms stay
                   exactly comparable.

Discipline note recorded at build time: the aggregate `s_all` forward number
(+3.13c, 95% CI [-0.004, +0.066], 18.8 effective events) had been observed
before this cohort was frozen. That is why the freeze timestamp is set to the
moment of freezing rather than the original arm's, and why NO per-wallet
decomposition of that aggregate was computed beforehand: the cohort is selected
purely on pre-freeze retrospective data, and its forward window begins after
the only forward figure anyone had seen.

Read-only: public GETs, paper only, no keys, no order placement.
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

SPORTS_INTERIM = INTERIM_DIR / "sports"
FROZEN_SET_PATH = PROCESSED_DIR / "sports_sig2c_frozen_set.parquet"
FREEZE_MANIFEST_PATH = PROCESSED_DIR / "sports_sig2c_freeze_manifest.json"
FORWARD_TRADES_PATH = SPORTS_INTERIM / "sig2c_forward" / "forward_trades.parquet"
FORWARD_RESOLUTIONS_PATH = SPORTS_INTERIM / "sig2c_forward" / "forward_resolutions.parquet"
SCOREBOARD_PARQUET = PROCESSED_DIR / "sports_sig2c_forward_scoreboard.parquet"
SCOREBOARD_MD = PROCESSED_DIR / "sports_sig2c_forward_scoreboard.md"

# The ORIGINAL sports freeze — read for its frozen baseline and pre-freeze
# per-wallet breadth. NEVER written by this module.
PARENT_MANIFEST_PATH = PROCESSED_DIR / "sports_freeze_manifest.json"
PARENT_FROZEN_SET_PATH = PROCESSED_DIR / "sports_frozen_set.parquet"

FAST_RESOLUTION_DAYS = 2

# --- pre-registered selection constants -----------------------------------
# 2c is Project 1's own global `scoring.min_skill_edge`, in force in this repo
# since 2026-07-19. It is NOT a number chosen to fill this table: the
# counterfactual is smooth (12 / 4 / 1 / 1 wallets at 2 / 5 / 10 / 15c), so
# there is no threshold cliff to hunt for.
MIN_SKILL_EDGE = 0.02
ALPHA = 0.05

# A wallet can only be COPIED if it trades slowly enough that a follower can act.
# Held-out cadence splits the twelve across a ~7x gap with nothing inside it:
# 3.1 / 8.4 / 9.8 bets-per-day, then 71.3 / 83.9 / 217 / 226 / 337 / 420 /
# 1185 / 1745. Any cut inside that gap yields the same three wallets; 20 is its
# midpoint on a log scale. Cadence is a PRE-freeze property of the deep tape.
SLOW_MAX_BETS_PER_DAY = 20.0

# Every saved baseline variant from the arm's mandatory finer-baseline re-check.
# Requiring survival in ALL of them is a STRICTER standard than the original
# arm applied, not a looser one.
BASELINE_VARIANTS: dict[str, str] = {
    "standard_league_form_20bins": "sports_validated.parquet",
    "league_20bins": "sports_validated_league_20bins_event.parquet",
    "league_form_40bins": "sports_validated_league_form_40bins_event.parquet",
    "sub_form_20bins": "sports_validated_sub_form_20bins_event.parquet",
    "sub_form_40bins": "sports_validated_sub_form_40bins_event.parquet",
}
STANDARD_VARIANT = "standard_league_form_20bins"

# Gates every tier here must clear. Note what is ABSENT: `magnitude_ok`, which
# is the 10c floor this arm exists to correct. Everything else is unchanged.
NON_MAGNITUDE_GATES = ["candidate", "significant", "markets_ok",
                       "concentration_ok", "complex_concentration_ok"]

TIER_SPECS: dict[str, dict] = {
    "s2_core": {
        "vetting": "highest",
        "headline": True,
        "description": (
            "Clears every non-magnitude gate at >=2c held-out skill AND is "
            "BASELINE-ROBUST: >=2c with event-clustered p<0.05 under all five "
            "saved baseline variants (league-only/20, league|form/20 and /40, "
            "sub-league|form/20 and /40). THE pre-registered headline."),
    },
    "s2_twelve": {
        "vetting": "high",
        "headline": False,
        "description": (
            "Clears every non-magnitude gate at >=2c under the STANDARD "
            "baseline only. The cohort named by the red-team audit; a "
            "superset of s2_core by exactly the baseline-fragile names."),
    },
    "s2_slow": {
        "vetting": "high, copy-relevant",
        "headline": False,
        "description": (
            "The subset of s2_twelve whose held-out cadence is <= 20 bets/day "
            "— slow enough that a follower could plausibly act. This is the "
            "only tier that speaks to COPYABILITY rather than identification; "
            "the rest of the cohort is in-play high-frequency."),
    },
}
TIER_ORDER = ["s2_core", "s2_twelve", "s2_slow"]
HEADLINE_TIER = "s2_core"

SCORING_RULE = (
    "For each frozen wallet, take its resolved BUY bets in SPORTS markets ENTERED "
    "STRICTLY AFTER this artifact's own freeze_ts. Per-bet skill edge = "
    "resolved_value - E[outcome | entry_price, league, league|form], where E is the "
    "hierarchical baseline copied VERBATIM from the parent sports freeze manifest "
    "(fit on pre-freeze data only, applied WITHOUT refitting here or there). Each "
    "TIER is scored in AGGREGATE as a crowd, never wallet-by-wallet: the headline "
    "statistic is the EVENT-WEIGHTED mean, i.e. the mean over RESOLUTION EVENTS of "
    "the within-event mean residual, where an event is one game, or the one "
    "championship a whole outright field settles on. Buying every team in a field "
    "is therefore ONE observation, not one per team. The bet-weighted mean is "
    "reported alongside. Uncertainty is a cluster bootstrap that resamples EVENTS "
    "with replacement (not bets), giving a 95% CI and a one-sided p vs 0, and is "
    "undefined below two events. Every bet count is reported with n_events and "
    "effective_events so volume is not mistaken for evidence. Results are split by "
    "resolution speed (<=2 days from entry to resolution vs longer). s2_core is the "
    "pre-registered headline tier and is NEVER displaced by a wider tier's forward "
    "number. Tiers are not re-frozen in light of forward outcomes."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROCESSED_DIR.parent.parent),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# SELECTION — entirely from pre-freeze artifacts
# ---------------------------------------------------------------------------

def load_variants() -> dict[str, pd.DataFrame]:
    """The five saved pre-freeze validation tables, keyed by baseline variant."""
    out = {}
    for name, fname in BASELINE_VARIANTS.items():
        path = SPORTS_INTERIM / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — the sports arm's saved baseline variants are "
                f"required to derive this cohort. Re-run `python -m src.sports_validate`.")
        df = out[name] = pd.read_parquet(path)
        df["wallet"] = df["wallet"].str.lower()
    return out


def _gate_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in NON_MAGNITUDE_GATES:
        mask &= df[col].fillna(False).astype(bool)
    return mask


def held_out_cadence(wallets: list[str]) -> pd.DataFrame:
    """Bets-per-day in each wallet's held-out half of the PRE-freeze deep tape.

    Mirrors the validator's split exactly: resolved BUY bets, stable mergesort
    by timestamp, second half held out."""
    path = SPORTS_INTERIM / "deep_trades.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — needed for the cadence tier.")
    d = pd.read_parquet(path, columns=["wallet", "timestamp", "side", "resolved_value"])
    d = d[(d["side"].str.upper() == "BUY") & d["resolved_value"].notna()]
    d["wallet"] = d["wallet"].str.lower()
    rows = []
    for w in wallets:
        g = d[d["wallet"] == w].sort_values("timestamp", kind="mergesort")
        if g.empty:
            rows.append({"wallet": w, "held_out_bets": 0, "held_out_span_days": np.nan,
                         "held_out_bets_per_day": np.nan})
            continue
        out = g.iloc[len(g) // 2:]
        span = (out["timestamp"].max() - out["timestamp"].min()) / 86400.0
        rows.append({
            "wallet": w,
            "held_out_bets": int(len(out)),
            "held_out_span_days": round(float(span), 2),
            "held_out_bets_per_day": (float(len(out)) / span) if span > 0 else np.inf,
        })
    return pd.DataFrame(rows)


def build_tiers() -> tuple[pd.DataFrame, dict, dict]:
    """Derive the cohort and its tiers from pre-freeze artifacts only."""
    variants = load_variants()
    std = variants[STANDARD_VARIANT]

    gated = std[_gate_mask(std) & (std["out_sample_skill"] >= MIN_SKILL_EDGE)]
    twelve = sorted(gated["wallet"].unique())
    if not twelve:
        raise RuntimeError(
            "no wallet clears the non-magnitude gates at the 2c floor — there is "
            "nothing to pre-register. Report the null instead of freezing an "
            "empty artifact.")

    # baseline robustness: >=2c AND p<alpha in EVERY variant
    robust: dict[str, dict[str, float]] = {}
    for name, df in variants.items():
        idx = df.set_index("wallet").reindex(twelve)
        robust[name] = {
            "edge": idx["out_sample_skill"].to_dict(),
            "p": idx["cluster_p"].to_dict(),
        }
    core = [w for w in twelve
            if all((robust[n]["edge"].get(w, np.nan) >= MIN_SKILL_EDGE)
                   and (robust[n]["p"].get(w, np.nan) < ALPHA)
                   for n in BASELINE_VARIANTS)]

    cadence = held_out_cadence(twelve).set_index("wallet")
    slow = [w for w in twelve
            if cadence.loc[w, "held_out_bets_per_day"] <= SLOW_MAX_BETS_PER_DAY]

    frozen = std[std["wallet"].isin(twelve)].copy()
    frozen = frozen.merge(cadence.reset_index(), on="wallet", how="left")
    for name in BASELINE_VARIANTS:
        frozen[f"edge_{name}"] = frozen["wallet"].map(robust[name]["edge"])
        frozen[f"p_{name}"] = frozen["wallet"].map(robust[name]["p"])
    frozen["edge_min_across_baselines"] = frozen[
        [f"edge_{n}" for n in BASELINE_VARIANTS]].min(axis=1)
    frozen["p_max_across_baselines"] = frozen[
        [f"p_{n}" for n in BASELINE_VARIANTS]].max(axis=1)

    members = {"s2_core": set(core), "s2_twelve": set(twelve), "s2_slow": set(slow)}
    for tier in TIER_ORDER:
        frozen[tier] = frozen["wallet"].isin(members[tier])

    # pre-freeze breadth, carried over from the parent freeze (all twelve are
    # candidates, so all are present in its union)
    if PARENT_FROZEN_SET_PATH.exists():
        parent = pd.read_parquet(PARENT_FROZEN_SET_PATH)
        parent["wallet"] = parent["wallet"].str.lower()
        keep = [c for c in ("wallet", "pre_freeze_bets", "pre_freeze_events",
                            "pre_freeze_leagues") if c in parent.columns]
        frozen = frozen.merge(parent[keep], on="wallet", how="left")

    stats = {}
    for tier in TIER_ORDER:
        sub = frozen[frozen[tier]]
        stats[tier] = {
            "n_wallets": int(len(sub)),
            "wallets": sorted(sub["wallet"].tolist()),
            "median_out_sample_skill": (float(sub["out_sample_skill"].median())
                                        if len(sub) else None),
            "median_in_sample_skill": (float(sub["in_sample_skill"].median())
                                       if len(sub) else None),
            "median_held_out_events": (float(sub["out_complexes"].median())
                                       if len(sub) and "out_complexes" in sub else None),
            "max_cluster_p": (float(sub["cluster_p"].max()) if len(sub) else None),
            "distinct_pre_freeze_events": (int(sub["pre_freeze_events"].sum())
                                           if "pre_freeze_events" in sub and len(sub)
                                           else None),
            "pre_freeze_bets": (int(sub["pre_freeze_bets"].sum())
                                if "pre_freeze_bets" in sub and len(sub) else None),
        }
    nesting = {a: [b for b in TIER_ORDER
                   if a != b and members[a] and members[a] <= members[b]]
               for a in TIER_ORDER}
    frozen = frozen.sort_values("out_sample_skill", ascending=False).reset_index(drop=True)
    return frozen, {"tiers": stats, "subset_of": nesting}, members


# ---------------------------------------------------------------------------
# FREEZE
# ---------------------------------------------------------------------------

def _forward_observations_exist() -> int:
    if not FORWARD_TRADES_PATH.exists():
        return 0
    try:
        return int(len(pd.read_parquet(FORWARD_TRADES_PATH, columns=["timestamp"])))
    except Exception:  # noqa: BLE001
        return -1


def freeze(freeze_ts: int | None = None, force: bool = False,
           amend_reason: str | None = None) -> dict:
    """Write this arm's own pre-registered freeze artifact + manifest."""
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

    if not PARENT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{PARENT_MANIFEST_PATH} not found — this arm reuses the parent sports "
            f"freeze's baseline verbatim and cannot be built without it.")
    with open(PARENT_MANIFEST_PATH) as fh:
        parent = json.load(fh)

    frozen, tier_stats, _ = build_tiers()
    ts = int(freeze_ts if freeze_ts is not None
             else (prior["freeze_ts"] if prior else time.time()))

    tiers = {t: {**tier_stats["tiers"][t],
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
            "freeze_ts_preserved": int(prior["freeze_ts"]) == ts,
        })

    manifest = {
        "arm": "sports_sig2c",
        "freeze_ts": ts,
        "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "git_commit": _git_commit(),
        "sports_event_version": SPORTS_EVENT_VERSION,
        "niche_partition_version": NICHE_PARTITION_VERSION,
        "headline_tier": HEADLINE_TIER,
        "tier_order": TIER_ORDER,
        "tiers": tiers,
        "parent_freeze": {
            "manifest": PARENT_MANIFEST_PATH.name,
            "freeze_utc": parent.get("freeze_utc"),
            "git_commit": parent.get("git_commit"),
            "note": ("The parent artifact is NOT amended by this arm. Its baseline "
                     "is reused verbatim; its tiers, timestamps and scoreboard are "
                     "untouched."),
        },
        "why_this_arm_exists": (
            "docs/redteam_audit_2026-07-26.md section 1: the parent arm's "
            "scoring.sports.min_skill_edge=0.10 floor was mis-transposed from the "
            "small-n slow-forecaster arm onto a large-n universe, and applied to "
            "HELD-OUT edge it is a winner's-curse selector. These wallets clear "
            "every other gate and had no tier that could read them."),
        "selection": {
            "min_skill_edge": MIN_SKILL_EDGE,
            "min_skill_edge_provenance": (
                "Project 1's global scoring.min_skill_edge, in force since "
                "2026-07-19 — not a threshold chosen for this table. Counterfactual "
                "is smooth: 12 / 4 / 1 / 1 wallets at 2 / 5 / 10 / 15c."),
            "alpha": ALPHA,
            "gates_applied": NON_MAGNITUDE_GATES,
            "gate_deliberately_omitted": "magnitude_ok (the 10c floor being corrected)",
            "baseline_variants_required": list(BASELINE_VARIANTS),
            "slow_max_bets_per_day": SLOW_MAX_BETS_PER_DAY,
            "slow_threshold_provenance": (
                "Held-out cadence of the twelve splits across a ~7x gap with "
                "nothing inside it (3.1/8.4/9.8 then 71.3/83.9/217/...); any cut "
                "in that gap selects the same three wallets."),
            "derived_from": ["sports_validated*.parquet (5 pre-freeze variants)",
                             "deep_trades.parquet (pre-freeze cadence)",
                             "sports_frozen_set.parquet (pre-freeze breadth)"],
        },
        "population_evidence_independent_of_the_floor": (
            "Among the parent arm's 174 candidates, 37 have event-clustered p<0.05 "
            "against ~8.7 expected under a global null and 26 have p<0.01 against "
            "~1.7 expected; BH-FDR across all 174 yields 27 survivors at q=0.10. "
            "The excess signal does not depend on which magnitude floor is chosen, "
            "which is what makes re-opening this legitimate rather than "
            "threshold-hunting."),
        "tier_note": (
            "Tiers are SELECTION rules only; forward scoring is identical across "
            "all of them. s2_core is the headline and is never displaced by a "
            "wider tier's forward number."),
        "resolution_speed_split_days": FAST_RESOLUTION_DAYS,
        "scoring_rule": SCORING_RULE,
        "baseline": parent["baseline"],
        "baseline_source": "copied verbatim from sports_freeze_manifest.json",
        "forward_observations_at_freeze": obs,
        "amendments": amendments,
        "known_limits": [
            "SELECTION-AFTER-THE-FACT, stated plainly: the 2c floor was chosen "
            "after the parent arm's table existed. The mitigations are that 2c is "
            "the repo's own pre-existing global floor, that the wallet-count "
            "counterfactual is smooth with no cliff, that the population-level "
            "excess signal is floor-independent, and that this cohort's forward "
            "window starts at THIS artifact's freeze_ts. The forward number is "
            "the arbiter; the retrospective number is not evidence for it.",
            "The aggregate s_all forward reading (+0.031, CI [-0.004,+0.066], 18.8 "
            "effective events) was known before this freeze. No per-wallet "
            "decomposition of it was computed, deliberately, so that membership "
            "could not be conditioned on forward outcomes. It remains true that "
            "these wallets sit inside a pool whose aggregate was seen to lean "
            "positive — treat that as a mild prior, not as evidence.",
            "Identification only. Copyability (metric B) is UNTESTED for every "
            "tier here. s2_slow exists because cadence is the precondition for "
            "copyability, not because copyability has been demonstrated.",
            "The 10k /trades?user= page cap truncates high-frequency wallets, so "
            "held-out spans for the fast members of s2_twelve are days, not "
            "months. This is a data artifact of the whole profile, not a property "
            "of any one wallet.",
            "s2_slow is THREE wallets. A three-wallet crowd carries very little "
            "independent evidence however many bets it places; read "
            "effective_events, never n_bets.",
            "Forward margins are expected to be SMALLER than the retrospective "
            "ones. That is the consequence of removing selection, not a failure.",
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
            f"`python -m src.sports_sig2c freeze` first.")
    with open(FREEZE_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return pd.read_parquet(FROZEN_SET_PATH), manifest


# ---------------------------------------------------------------------------
# SCORE — identical machinery to the parent arm, its own paths and window
# ---------------------------------------------------------------------------

def score(cfg: dict | None = None, refetch: bool = True, n_boot: int = 2000) -> dict:
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
    print(f"[sig2c] freeze {manifest['freeze_utc']} ({manifest['git_commit'][:8]}) — "
          f"{len(wallets)} frozen wallets "
          f"(headline {manifest['tiers'][HEADLINE_TIER]['n_wallets']})")

    FORWARD_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refetch or not FORWARD_TRADES_PATH.exists():
        raw = fetch_forward_trades(wallets, freeze_ts, cfg=cfg)
        if raw.empty:
            return _await_data(manifest, "no post-freeze trades yet")
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
        return _await_data(manifest, "post-freeze trades captured, none resolved yet")

    bets["category"] = derive_categories(bets["slug"], bets["question"])
    bets = bets[[is_sports_market(s, q, c) for s, q, c in
                 zip(bets["slug"], bets["question"], bets["category"])]].copy()
    if bets.empty:
        return _await_data(manifest, "no resolved SPORTS bets yet")
    bets = assign_events(assign_sub_forms(assign_niches(bets)))
    bets["wallet"] = bets["wallet"].str.lower()
    exp = expected_outcome_hier(manifest["baseline"], bets)
    bets["expected_outcome"] = exp
    bets["residual_skill"] = bets["resolved_value"].to_numpy(dtype=float) - exp
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


def _await_data(manifest: dict, why: str) -> dict:
    rows = [{"tier": t, "event_unit": "event", "stratum": "all", "n_bets": 0,
             "n_wallets": int(manifest["tiers"][t]["n_wallets"]), "n_events": 0,
             "effective_events": 0.0, "edge_event_weighted": np.nan,
             "edge_bet_weighted": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "p_one_sided": np.nan} for t in TIER_ORDER]
    board = pd.DataFrame(rows)
    _write_board(board, manifest, note=why)
    print(f"[sig2c] {why} — scoreboard written in 'awaiting data' state.")
    return {"board": board, "manifest": manifest, "bets": 0}


def _write_board(board: pd.DataFrame, manifest: dict, note: str | None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    atomic_to_parquet(board, SCOREBOARD_PARQUET, compression="gzip")
    lines = [
        "# Sports `s_sig2c` forward scoreboard (paper, read-only)",
        "",
        f"**Frozen {manifest['freeze_utc']}** at commit "
        f"`{manifest['git_commit'][:8]}`, event partition "
        f"v{manifest['sports_event_version']}. "
        f"Headline tier: `{manifest['headline_tier']}`.",
        "",
        "This is a **second, separate** sports pre-registration. It does not amend "
        "`sports_freeze_manifest.json`, whose forward window opened earlier and "
        "whose tiers are unchanged. See `docs/redteam_audit_2026-07-26.md` §1 for "
        "why this cohort exists.",
        "",
    ]
    if note:
        lines += [f"> **AWAITING FORWARD DATA** — {note}. Sports resolves in "
                  f"hours-to-days, so this should populate within days.", ""]
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
        pfe = tier.get("distinct_pre_freeze_events")
        lines.append(
            f"| {head}{t}{head} | {tier['n_wallets']} "
            f"| {'—' if pfe is None else pfe} "
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
    print(f"[sig2c] -> {SCOREBOARD_MD}\n[sig2c] -> {SCOREBOARD_PARQUET}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Freeze / score the sports s_sig2c forward arm.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="write the pre-registered freeze artifact")
    f.add_argument("--force", action="store_true",
                   help="overwrite an existing freeze (records an amendment)")
    f.add_argument("--reason", type=str, default=None)
    p = sub.add_parser("preview", help="show the cohort WITHOUT writing anything")
    p.add_argument("--verbose", action="store_true")
    s = sub.add_parser("score", help="score post-freeze sports bets")
    s.add_argument("--no-refetch", action="store_true")
    s.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    if args.cmd == "preview":
        frozen, stats, _ = build_tiers()
        for t in TIER_ORDER:
            st = stats["tiers"][t]
            print(f"{t:10s} n={st['n_wallets']:3d}  median edge="
                  f"{st['median_out_sample_skill']}  max p={st['max_cluster_p']}")
        cols = ["wallet", "in_sample_skill", "out_sample_skill", "cluster_p",
                "out_complexes", "eff_breadth_complex", "held_out_bets_per_day",
                "edge_min_across_baselines", "p_max_across_baselines", *TIER_ORDER]
        print(frozen[[c for c in cols if c in frozen]].to_string(index=False))
    elif args.cmd == "freeze":
        m = freeze(force=args.force, amend_reason=args.reason)
        print(f"[sig2c] FROZEN {m['freeze_utc']} @ {m['git_commit'][:8]}")
        for t in TIER_ORDER:
            ti = m["tiers"][t]
            print(f"  {t:10s} n={ti['n_wallets']:3d}  "
                  f"median edge={ti['median_out_sample_skill']}  "
                  f"pre-freeze events={ti['distinct_pre_freeze_events']}")
        print(f"  -> {FROZEN_SET_PATH}\n  -> {FREEZE_MANIFEST_PATH}")
        score(refetch=False)
    else:
        score(refetch=not args.no_refetch, n_boot=args.boot)


if __name__ == "__main__":
    main()
