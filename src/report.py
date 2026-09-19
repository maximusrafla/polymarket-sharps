"""Human-readable summary of the ranked wallet table. See CLAUDE.md
"src/report.py — human-readable summary of the top wallets and why each
ranks." Reads the full ranked table (nothing was excluded upstream) and
renders the top `ranking.top_n` wallets to Markdown, with in-sample and
out-of-sample edge shown side by side per CLAUDE.md's Validation section
("Report both ... so the decay is visible"), plus the descriptive
`manufactured_record_flag`/`pattern_flag` metadata called out where set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.common import RANKED_WALLETS_PATH, REPORT_PATH, ensure_dirs, load_config


def forward_validity(ranked: pd.DataFrame) -> tuple[float, float, int]:
    """How well does in-sample skill edge predict HELD-OUT skill edge across the
    evaluable wallets — i.e. how much rank *order* is worth trusting. Spearman
    rank correlation between `in_sample_residual_edge` and
    `out_of_sample_residual_edge` over wallets where both are defined (>=
    min_bets_per_half in each half). Returns (rho, p_value, n); NaNs when there
    are too few points. This is honest expectation-setting, not a score input —
    see HANDOFF.md / DECISIONS.md "Ranking weights"."""
    if "in_sample_residual_edge" not in ranked or "out_of_sample_residual_edge" not in ranked:
        return float("nan"), float("nan"), 0
    pair = ranked[["in_sample_residual_edge", "out_of_sample_residual_edge"]].dropna()
    if len(pair) < 8:
        return float("nan"), float("nan"), len(pair)
    rho, p = stats.spearmanr(pair["in_sample_residual_edge"], pair["out_of_sample_residual_edge"])
    return float(rho), float(p), len(pair)


def _fmt(value, digits: int = 3) -> str:
    """Signed fixed-point, or 'n/a' when the value is missing (insufficient
    history) — avoids rendering a bare 'nan' in the report."""
    return "n/a" if pd.isna(value) else f"{value:+.{digits}f}"


def _copyable_txt(row: pd.Series) -> str:
    """Followability note for a wallet: whether its copy_window leaves a copier
    room (metric B, kept separate from skill). Falls back gracefully if the
    `copyable` column is absent (older ranked tables)."""
    if "copyable" not in row or pd.isna(row.get("copyable")):
        return "copyability unknown"
    if bool(row["copyable"]):
        return "COPYABLE (positive followable room)"
    return "NOT copyable (edge realizes at/near resolution, no room to follow)"


def build_why(row: pd.Series) -> str:
    """One short paragraph explaining why a wallet ranked where it did: the
    validated **skill (residual) edge** — the ranking signal, raw edge minus the
    favorite-longshot base rate — with raw edge and significance alongside, the
    other first-class ranking variables, and any descriptive flags. Flags are
    reported, never framed as having affected the score."""
    if row["edge_persisted"]:
        persisted_txt = "persisted out-of-sample"
    elif bool(row.get("edge_significant")) and not bool(row.get("edge_magnitude_ok", True)):
        # real, statistically-significant skill edge that is simply too small to
        # matter economically — flagged distinctly from noise (see the magnitude
        # floor in DECISIONS.md / HANDOFF.md).
        persisted_txt = "did NOT persist out-of-sample (significant but below the economic-magnitude floor)"
    else:
        persisted_txt = "did NOT persist out-of-sample"
    time_consistency_txt = (
        "unknown (too few bets)" if pd.isna(row["time_consistency"]) else f"{row['time_consistency']:.2f}"
    )
    # Report the cluster-robust p (the gate on edge_significant); fall back to the
    # t-test p for legacy rows that predate it. See src/validate.py.
    pval = row.get("out_of_sample_cluster_p")
    if pval is None or pd.isna(pval):
        pval = row.get("out_of_sample_residual_p")
    sig_txt = "" if pd.isna(pval) else f", cluster p={pval:.3f}"
    parts = [
        f"Out-of-sample skill edge {_fmt(row['out_of_sample_residual_edge'])} over "
        f"{int(row['out_of_sample_n'])} held-out bets ({persisted_txt}{sig_txt}; in-sample "
        f"skill edge was {_fmt(row['in_sample_residual_edge'])} over {int(row['in_sample_n'])} bets).",
        f"Raw edge (pre favorite-longshot adjustment) was {_fmt(row['out_of_sample_edge'])} "
        f"out-of-sample vs {_fmt(row['in_sample_edge'])} in-sample.",
        f"Copy window {_fmt(row['copy_window'])} (leakage-guarded forward fair-value gap; "
        f"earliness is the same signal at a shorter window, not scored separately) — "
        f"{_copyable_txt(row)}; "
        f"breadth {int(row['breadth'])} markets, time-consistency {time_consistency_txt}.",
    ]
    if bool(row.get("regime_watch")):
        recovered = " and its recent regime survived a leakage-free sub-split (persisted_recent)" if bool(
            row.get("persisted_recent")
        ) else ""
        parts.append(
            "Regime change (metric D): this wallet BECAME sharp — its early half was not sharp so the "
            "chronological candidate gate excludes it from `edge_persisted`, but its recent half is "
            f"significant and material, so it is on the `regime_watch` list{recovered}. Watchlist "
            "signal, not a certified persister."
        )
    elif pd.notna(row.get("regime_flag")) and row.get("regime_flag") not in (None, "stable", "insufficient"):
        parts.append(f"Note: regime_flag = {row['regime_flag']} (descriptive; does not affect score).")
    if row["manufactured_record_flag"]:
        parts.append(
            "Note: manufactured_record_flag is set (matched notional concentrated against one "
            "recurring counterparty) — a data-quality note, not an exclusion; verify before following."
        )
    pattern_flag = row.get("pattern_flag")
    if pd.notna(pattern_flag):
        parts.append(f"Note: pattern_flag = {pattern_flag}.")
    return " ".join(parts)


def _persisted_copyable_line(ranked: pd.DataFrame) -> str:
    """Summary of the actionable set: persisted (validated skill, clearing both
    the significance and economic-magnitude gates) split by copyability. The two
    are orthogonal — a wallet can be sharp yet unfollowable — so the copyable
    persisters are the recommended default filter for the copy-trading use case."""
    if "edge_persisted" not in ranked:
        return ""
    persisted = ranked[ranked["edge_persisted"]]
    n_persisted = len(persisted)
    if "copyable" not in ranked:
        return f"**Persisted (validated skill):** {n_persisted} wallets."
    n_copyable = int(persisted["copyable"].sum())
    return (
        f"**Actionable set:** {n_persisted} wallets have validated skill edge (persisted: clears "
        f"both the significance and economic-magnitude gates). Of those, **{n_copyable} are also "
        f"copyable** (positive followable copy-window) — the recommended default filter for "
        f"copy-trading; the other {n_persisted - n_copyable} are genuinely sharp but realize their "
        "edge at/near resolution, leaving a follower no room. Skill and copyability are "
        "independent signals (`edge_persisted` vs `copyable`); sort/filter on either."
    )


def _regime_line(ranked: pd.DataFrame) -> str:
    """Metric-D summary: the actionable set widened by regime-recovered wallets.
    A wallet that *became* sharp (early residual <= 0, recent half significant +
    material) never becomes a candidate under the chronological gate, so its edge
    can't `persist` off a single split without leakage — it is routed to
    `regime_watch` instead. When such a wallet's recent half is itself deep enough
    to survive a leakage-free within-recent sub-split it earns `persisted_recent`
    (while `edge_persisted` deliberately stays False). The copy-actionable set is
    therefore `(edge_persisted OR persisted_recent) AND copyable`."""
    if "regime_watch" not in ranked or "persisted_recent" not in ranked:
        return ""
    n_watch = int(ranked["regime_watch"].sum())
    n_recent = int(ranked["persisted_recent"].sum())
    parts = [
        f"**Regime change (metric D):** {n_watch} wallets are on the became-sharp "
        "`regime_watch` (early residual <= 0 but a significant, material recent half) — the "
        "candidate gate buries these, so they are a sortable watchlist for the forward paper "
        f"test, **not** `edge_persisted`. Of them, **{n_recent} earned `persisted_recent`** "
        "(their recent regime survived a leakage-free within-recent sub-split)."
    ]
    if "edge_persisted" in ranked and "copyable" in ranked:
        actionable = ranked[(ranked["edge_persisted"] | ranked["persisted_recent"]) & ranked["copyable"]]
        parts.append(
            f" The copy-actionable set — `(edge_persisted OR persisted_recent) AND copyable` — is "
            f"**{len(actionable)} wallets**."
        )
    return "".join(parts)


def render_report(ranked: pd.DataFrame, cfg: dict) -> str:
    """Render the top `ranking.top_n` wallets (by the `rank` column already
    assigned in rank.py) to a Markdown report. `ranked` itself is expected to
    contain every scored wallet; only the rendered slice is truncated."""
    top_n = cfg["ranking"]["top_n"]
    top = ranked.sort_values("rank").head(top_n)

    rho, p, n = forward_validity(ranked)
    if n >= 8 and not np.isnan(rho):
        validity_line = (
            f"**Forward validity:** across the {n} wallets with enough history to test, "
            f"in-sample skill edge predicts held-out skill edge with Spearman rho={rho:+.2f} "
            f"(p={p:.2f}). Treat this as a **coarse skill filter**: *which* wallets clear the "
            "persistence gate carries more signal than the fine rank order among them, and the "
            "order is sample-limited (see DECISIONS.md 'Ranking weights')."
        )
    else:
        validity_line = (
            "**Forward validity:** too few wallets with enough held-out history to measure yet "
            "(see DECISIONS.md 'Ranking weights')."
        )

    persisted_line = _persisted_copyable_line(ranked)
    regime_line = _regime_line(ranked)

    lines = [
        "# Polymarket Sharps — Ranked Wallets",
        "",
        f"Showing the top {len(top)} of {len(ranked)} ranked wallets. Every wallet ingested by "
        "this pipeline is scored and ranked — none are excluded (see CLAUDE.md). Sample sizes "
        "reflect resolved bets *observed* during this pipeline's polling window, not necessarily "
        "a wallet's complete trading history (see DECISIONS.md).",
        "",
        validity_line,
        "",
        persisted_line,
        "",
        regime_line,
        "",
    ]
    for _, row in top.iterrows():
        lines.append(f"## #{int(row['rank'])} — `{row['wallet']}` (score {row['score']:+.4f})")
        lines.append(build_why(row))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    if not RANKED_WALLETS_PATH.exists():
        print("[report] no ranked wallet data — run src.rank first.")
        return
    ranked = pd.read_parquet(RANKED_WALLETS_PATH)
    if ranked.empty:
        print("[report] ranked wallet table is empty — run src.rank first.")
        return

    report_md = render_report(ranked, cfg)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md)

    top_n = min(cfg["ranking"]["top_n"], len(ranked))
    print(f"[report] wrote top {top_n} of {len(ranked)} wallets -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
