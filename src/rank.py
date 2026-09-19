"""Combine validated per-wallet metrics into a single score and emit the
ranked table. See CLAUDE.md "Scoring"/"Validation" and DECISIONS.md
"Combining metrics into a score (src/rank.py)" for the exact formula and the
reasoning behind it. Every wallet present in wallet_validated.parquet gets a
row and a score here — nothing is dropped, per CLAUDE.md; wallets with thin
samples, narrow breadth, or edge that didn't survive out-of-sample validation
are down-weighted toward zero instead of excluded. `manufactured_record_flag`
and `pattern_flag` are metadata only and never enter the formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import RANKED_WALLETS_PATH, WALLET_VALIDATED_PATH, ensure_dirs, load_config


def compute_reliability(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Down-weighting factor in [0, 1]. Shrinks toward 0 for wallets with few
    out-of-sample bets (`scoring.min_sample_size` sets the shrinkage scale),
    narrow market breadth, or an out-of-sample edge that didn't persist.
    Never zero purely from the persistence term (`ranking.non_persisted_penalty`
    keeps a floor) — only thin samples/breadth can push it to exactly 0."""
    scoring = cfg["scoring"]
    ranking = cfg["ranking"]
    k = scoring["min_sample_size"]

    sample_confidence = df["out_of_sample_n"] / (df["out_of_sample_n"] + k)
    breadth_credit = df["breadth"].clip(upper=ranking["breadth_full_credit"]) / ranking["breadth_full_credit"]
    persisted_multiplier = np.where(df["edge_persisted"], 1.0, ranking["non_persisted_penalty"])

    return sample_confidence * breadth_credit * persisted_multiplier


def compute_edge_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Weighted combination of the validated out-of-sample **skill (residual)
    edge** — raw edge minus the favorite-longshot base rate, per HANDOFF.md —
    with the other first-class ranking variables CLAUDE.md calls out
    (copy_window, earliness, time_consistency), before reliability down-weighting
    is applied. Missing values (insufficient history) contribute 0 (edge/
    copy_window/earliness) or the neutral midpoint (time_consistency), never a
    penalty beyond what `compute_reliability` already applies for too little
    data."""
    ranking = cfg["ranking"]
    time_consistency_centered = df["time_consistency"].fillna(0.5) - 0.5

    # NOTE (2026-07-18 forward-price audit, see DECISIONS.md / HANDOFF.md):
    # `earliness` and `copy_window` are the same forward-drift measurement at two
    # window lengths, and in this data source they are numerically identical
    # (100% of wallets, corr 1.0) because all per-token trades cluster inside the
    # shorter window. Scoring both would double-count one signal, so only
    # `copy_window` (CLAUDE.md's first-class variable) enters the score.
    # `earliness` is still computed and reported; re-add it here only if a richer
    # history ever makes the two windows genuinely diverge (`w_earliness`).
    return (
        ranking["w_edge"] * df["out_of_sample_residual_edge"].fillna(0.0)
        + ranking["w_copy_window"] * df["copy_window"].fillna(0.0)
        + ranking["w_time_consistency"] * time_consistency_centered
    )


def compute_copyable(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """`copyable` flag (metric B): True when a wallet's `copy_window` — the
    followable room a copier still has after the wallet entered — clears
    `ranking.copyable_window_floor` (default 0.0, i.e. strictly positive). This
    is orthogonal to skill: a wallet can have real, significant skill edge yet a
    non-positive copy_window (its edge realizes at resolution, not within the
    forward window), which makes it unfollowable even though it is genuinely
    sharp. Kept SEPARATE from `edge_persisted` and deliberately NOT fed into the
    score (copy_window already contributes there via `w_copy_window`); this is a
    first-class, sortable filter the user applies themselves. NaN copy_window
    (insufficient forward history) -> not copyable."""
    floor = cfg["ranking"].get("copyable_window_floor", 0.0)
    cw = df["copy_window"]
    return (cw > floor) & cw.notna()


def compute_score(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach `reliability`, `edge_score`, `score` (= reliability * edge_score),
    and the `copyable` filter column. Does not sort or filter."""
    df = df.copy()
    df["reliability"] = compute_reliability(df, cfg)
    df["edge_score"] = compute_edge_score(df, cfg)
    df["score"] = df["reliability"] * df["edge_score"]
    df["copyable"] = compute_copyable(df, cfg)
    return df


def rank_wallets(validated: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score every wallet and sort descending by score. All wallets are kept
    (per CLAUDE.md); `rank` is simply the 1-indexed sort position."""
    scored = compute_score(validated, cfg)
    ranked = scored.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    if not WALLET_VALIDATED_PATH.exists():
        print("[rank] no validated wallet data — run src.validate first.")
        return
    validated = pd.read_parquet(WALLET_VALIDATED_PATH)
    if validated.empty:
        print("[rank] validated wallet table is empty — run src.validate first.")
        return

    ranked = rank_wallets(validated, cfg)
    RANKED_WALLETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(RANKED_WALLETS_PATH, compression="gzip")

    n_persisted = int(validated["edge_persisted"].sum())
    n_copyable_persisted = int((ranked["edge_persisted"] & ranked["copyable"]).sum())
    msg = (
        f"[rank] ranked {len(ranked)} wallets ({n_persisted} with persisted edge, "
        f"{n_copyable_persisted} of them copyable)"
    )
    # metric D: the copy-actionable set widens to (edge_persisted OR persisted_recent)
    if "persisted_recent" in ranked.columns:
        actionable = int(((ranked["edge_persisted"] | ranked["persisted_recent"]) & ranked["copyable"]).sum())
        n_recent = int(ranked["persisted_recent"].sum())
        msg += f"; metric-D actionable (persisted OR persisted_recent, copyable)={actionable} (+{n_recent} recent)"
    print(f"{msg} -> {RANKED_WALLETS_PATH}")


if __name__ == "__main__":
    main()
