"""Per-category dashboard for the Project-2 forecaster table
(`src/forecaster_metrics.py`). Turns the (wallet, category) winner-record parquet
into a sortable, human-readable dashboard.

Per `docs/project2_forecaster_discovery.md` §2.x/§3: the engine does NOT fuse
A/B/C into one weighted scalar (on noisy data no weighting beats another beyond
bootstrap noise — `tune_weights.py`). Instead it applies **gates** (real edge,
reachable, big enough) and exposes **every metric as an independently-sortable
column**. Nothing is dropped: non-qualifying cells stay in the table, sorted
below, with the gate booleans exposed so the user can relax any filter.

The single most important "don't": **never sort by raw returns / profit / win
rate** — that re-introduces the favorite-longshot + size + luck trap the whole
engine exists to avoid. The default sort is copyable **skill** edge (residual,
favorite-longshot-neutral), in cents.

Outputs (discovery dir only; data/interim is gitignored/local, so the committed
deliverable is the code + a findings doc, not these artifacts):
  - forecasters_dashboard.md   — per-category funnels + sorted top tables
  - forecasters_dashboard.html — self-contained, client-side click-to-sort tables
"""

from __future__ import annotations

import html
import json

import pandas as pd

from src.common import load_config
from src.forecaster_metrics import ALL_CELL, FORECASTER_TABLE_PATH
from src.discover import DISCOVERY_DIR

DASHBOARD_MD_PATH = DISCOVERY_DIR / "forecasters_dashboard.md"
DASHBOARD_HTML_PATH = DISCOVERY_DIR / "forecasters_dashboard.html"

# The shuffled-outcome null + BH-FDR gate (scripts/audit_forecaster_null.py). When
# present it is the arbiter of whether a "winner" is real skill or selection noise:
# a null-surviving winner (`winner_null`) is one that ALSO beats the structural
# favorite-longshot shuffle null after FDR correction. See docs/project2_section14.
NULL_FDR_PATH = DISCOVERY_DIR / "forecaster_null_fdr.parquet"
NULL_SUMMARY_PATH = DISCOVERY_DIR / "forecaster_null_summary.json"


def load_null_summary() -> dict | None:
    """The count-null / FDR headline numbers (est. gate FDR, survivor counts) the
    audit script wrote, or None if the null gate hasn't been run yet."""
    if not NULL_SUMMARY_PATH.exists():
        return None
    try:
        return json.loads(NULL_SUMMARY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def join_null_fdr(table: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Left-join the per-cell shuffled-null + FDR survivor flags onto the forecaster
    table (additive columns; nothing dropped). Adds `winner_null` = a raw A+B+C
    winner (or, since metric B is deferred, a metric-A `null_winner_fdr_05`) that
    ALSO survives the structural-null FDR at q=0.05 — the trustworthy set. Returns
    (table, has_null): when the null gate hasn't been run, columns default to False/
    NaN and `has_null` is False so the dashboard falls back to the pre-null view."""
    cols = ["survives_fdr_05", "survives_fdr_10", "null_winner_fdr_05",
            "empirical_p_structural_null", "real_oos_t"]
    if not NULL_FDR_PATH.exists():
        for c in cols:
            table[c] = False if c.startswith(("survives", "null_winner")) else float("nan")
        table["winner_null"] = False
        return table, False
    nf = pd.read_parquet(NULL_FDR_PATH)[["wallet", "category"] + cols]
    table = table.merge(nf, on=["wallet", "category"], how="left")
    for c in ("survives_fdr_05", "survives_fdr_10", "null_winner_fdr_05"):
        table[c] = table[c].fillna(False).astype(bool)
    # A null-surviving winner: the metric-A winner candidate (persisted + magnitude +
    # breadth + full-tape + real-world) that also cleared FDR. Once metric B runs on
    # this set, gate it further with is_winner; today B is deferred so this IS the set.
    table["winner_null"] = table["null_winner_fdr_05"]
    return table, True

# Columns shown in the dashboard (order = display order). The parquet keeps every
# metric; this is the readable subset.
DISPLAY_COLUMNS = [
    ("wallet", "wallet"),
    ("sample_size", "n"),
    ("breadth", "mkts"),
    ("win_rate", "win%"),
    ("frac_full_tape", "full%"),
    ("bets_per_month", "bets/mo"),
    ("skill_edge_all", "skill_edge"),
    ("out_of_sample_skill_edge", "oos_skill"),
    ("out_of_sample_skill_p", "oos_p"),
    ("edge_persisted", "A:persist"),
    ("reachability_60", "B:reach60"),
    ("copyable_skill_edge_60_cents", "B:copy_edge¢"),
    ("clears_magnitude_floor", "C:mag_ok"),
    ("full_tape", "FT:full"),
    ("empirical_p_structural_null", "null_p"),
    ("is_winner", "WINNER"),
    ("winner_full_tape", "WIN·FT"),
    ("winner_null", "WIN·NULL"),
]


def category_funnel(table: pd.DataFrame) -> pd.DataFrame:
    """Per-category gate funnel: how many (wallet, category) cells clear each stage
    (positive skill -> persisted (A+C on skill) -> winner (A+B+C)). This is the
    honest 'how thin is the signal' summary."""
    has_ft = "winner_full_tape" in table.columns
    has_null = "winner_null" in table.columns
    rows = []
    for cat, g in table.groupby("category"):
        row = {
            "category": cat,
            "cells": len(g),
            "wallets": g["wallet"].nunique(),
            "skill>0": int((g["skill_edge_all"] > 0).sum()),
            "persisted(A+C)": int(g["edge_persisted"].sum()),
            "reachable(B)": int((g["reachability_60"] >= 0.5).sum()),
            "winners(A+B+C)": int(g["is_winner"].sum()),
        }
        if has_ft:
            row["full_tape_winners"] = int(g["winner_full_tape"].sum())
        if has_null:
            row["null_survivors(FDR)"] = int(g["winner_null"].sum())
        rows.append(row)
    sort_key = ("null_survivors(FDR)" if has_null else
                "full_tape_winners" if has_ft else "winners(A+B+C)")
    funnel = pd.DataFrame(rows).sort_values(sort_key, ascending=False)
    return funnel.reset_index(drop=True)


def default_view(table: pd.DataFrame) -> pd.DataFrame:
    """The recommended default filter (§3.2), now gated by the shuffled-outcome null:
    a trustworthy winner must be real (persisted) + big enough (magnitude) + on FULL
    (untruncated) tapes + a real-world category + breadth ≥ 3 AND survive the
    structural-null BH-FDR (`winner_null`) — i.e. beat the favorite-longshot shuffle,
    not just the analytic t-test. Kept as a *view*; the full table on disk keeps every
    cell. Falls back to winner_full_tape / is_winner when the null gate hasn't run."""
    for key in ("winner_null", "winner_full_tape", "is_winner"):
        if key in table.columns:
            break
    winners = table[table[key]].copy()
    return winners.sort_values("copyable_skill_edge_60_cents", ascending=False).reset_index(drop=True)


def _fmt_cell(col: str, val) -> str:
    if pd.isna(val):
        return "n/a"
    if col == "wallet":
        return f"{val[:10]}…{val[-4:]}" if isinstance(val, str) and len(val) > 16 else str(val)
    if col in ("edge_persisted", "clears_magnitude_floor", "is_winner",
               "full_tape", "winner_full_tape", "winner_null"):
        return "✓" if bool(val) else "·"
    if col in ("sample_size", "breadth"):
        return str(int(val))
    if col in ("out_of_sample_skill_p", "empirical_p_structural_null"):
        return f"{val:.3f}" if val >= 1e-3 else f"{val:.1e}"
    if col in ("win_rate", "frac_full_tape"):
        return f"{val:.0%}"
    if col in ("skill_edge_all", "out_of_sample_skill_edge", "reachability_60"):
        return f"{val:+.3f}" if col != "reachability_60" else f"{val:.2f}"
    if col in ("bets_per_month", "copyable_skill_edge_60_cents"):
        return f"{val:+.2f}" if col == "copyable_skill_edge_60_cents" else f"{val:.1f}"
    return f"{val:+.3f}"


def _category_table_md(g: pd.DataFrame, top: int) -> str:
    # Sort null-survivors first, then the B∩C axis (copyable skill edge), then all-time
    # skill; NEVER profit. winner_null leads so the trustworthy set floats to the top.
    sort_cols = [c for c in ("winner_null", "is_winner", "copyable_skill_edge_60_cents",
                             "skill_edge_all") if c in g.columns]
    g = g.sort_values(sort_cols, ascending=False).head(top)
    header = "| " + " | ".join(label for _, label in DISPLAY_COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in DISPLAY_COLUMNS) + "|"
    lines = [header, sep]
    for _, row in g.iterrows():
        cells = [_fmt_cell(col, row.get(col)) for col, _ in DISPLAY_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _null_verdict_banner(s: dict | None, n_null: int) -> str:
    """The one-line headline verdict from the shuffled-outcome null + FDR gate. This
    is the crux of §1.4: it states plainly whether ANY winner is real skill or the
    whole list is statistical selection across ~hundreds-of-thousands of cells."""
    if s is None:
        return ("> ℹ️ **The shuffled-outcome null + FDR gate has not been run.** Run "
                "`python scripts/audit_forecaster_null.py` to decide whether any winner "
                "survives; until then the WINNER/WIN·FT columns are UNVALIDATED against "
                "multiple testing.")
    cn = s.get("count_null", {})
    persisted = cn.get("persisted", {})
    winnerA = cn.get("winnerA", {})
    est_fdr = winnerA.get("est_fdr", persisted.get("est_fdr", float("nan")))
    K = s.get("n_shuffles", "?")
    fdr05 = s.get("fdr", {}).get("by_q", {}).get("0.05", {})
    pre_conc = s.get("fdr05_winner_pre_concentration", 0)
    n_win_fdr = s.get("null_winner_fdr_05", n_null)
    if n_win_fdr == 0:
        return (
            f"> 🛑 **VERDICT — SELECTION NOISE. No credible copyable forecaster survives "
            f"(q=0.05).** Shuffling resolved outcomes within each (category, 1¢-price) bucket "
            f"{K}× — preserving the favorite-longshot base rate and every wallet's price mix, "
            f"destroying only forecasting skill — the OOS persistence gate still fires "
            f"**{persisted.get('real','?')}** times vs a null mean of "
            f"**{persisted.get('null_mean',0):.0f}** (est. gate FDR **{est_fdr:.0%}**), so most "
            f"'winners' are the selected tail of a multiple-comparison search. "
            f"**{fdr05.get('survive','?')}** cells survive BH-FDR and **{pre_conc}** clear the "
            f"economic gates — but on inspection those are single-slate / single-event cells "
            f"(many fills on a handful of correlated positions, ~50¢ 'edge') that the "
            f"exchangeable shuffle cannot detect; the concentration guard "
            f"(eff_breadth ≥ 3 & entry_days ≥ 3) rejects them, leaving **0**. "
            f"**The winner list is withheld; metric B (copyability) is NOT run.** The WINNER / "
            f"WIN·FT columns below are retained as sortable diagnostics only — treat them as "
            f"noise. See docs/project2_section14_findings.md.")
    return (
        f"> ✅ **VERDICT — {n_win_fdr} provisional winner(s) survive the shuffled-outcome "
        f"null + BH-FDR (q=0.05)** out of {fdr05.get('survive','?')} FDR-surviving cells "
        f"(est. gate FDR on the raw persisted set was {est_fdr:.0%}, so most apparent "
        f"winners are noise — these are the exception). This is the **provisional** "
        f"publishable set (`WIN·NULL`); metric B (copyability) runs on just these, and the "
        f"forward paper-test remains the real arbiter. See docs/project2_section14_findings.md.")


def render_markdown(table: pd.DataFrame, cfg: dict, top_per_category: int = 15,
                    null_summary: dict | None = None) -> str:
    n_cells = len(table)
    n_wallets = table["wallet"].nunique()
    n_winners = int(table["is_winner"].sum())
    n_winners_ft = int(table["winner_full_tape"].sum()) if "winner_full_tape" in table else 0
    n_null = int(table["winner_null"].sum()) if "winner_null" in table else 0
    funnel = category_funnel(table)

    lines = [
        "# Project 2 — Copyable Forecasters (per category)",
        "",
        f"{n_cells} (wallet, category) cells over {n_wallets} wallets from the market-first "
        "discovery tape. Every cell is scored and kept — nothing is dropped (CLAUDE.md); the "
        "gates below are additive boolean columns you can relax.",
        "",
        _null_verdict_banner(null_summary, n_null),
        "",
        "**Gates:** A = skill edge is real (persisted out-of-sample: significant **and** clears "
        "the skill-magnitude floor). B = copyable (a follower entering ~60s late can still get a "
        "fill — reachability ≥ 0.5). C = economically big enough (copyable **skill** edge clears "
        "the cents floor, net of a slippage buffer). A **winner** clears A+B+C. **NULL = survives "
        "the shuffled-outcome null + BH-FDR** (`scripts/audit_forecaster_null.py`) — the gate that "
        "separates real skill from a multiple-testing mirage. **`WIN·NULL` is the only trustworthy "
        "column.**",
        "",
        "**Sort:** copyable skill edge (residual, favorite-longshot-neutral) — **never** raw "
        "returns / profit / win rate (the trap this engine exists to avoid).",
        "",
        "> ⚠️ **`full%` is the de-biasing column (§1.4).** The top-volume markets' retrievable tape "
        "is only the most-recent ~10.5k trades — clustered **near resolution** — which inflates "
        "metric-A skill 10–50× (endgame entries on the winning side scored as forecasts; look for "
        "`win%` near 100%). `full%` is the fraction of a cell's bets on **untruncated** tapes, where "
        "early entries survive and skill is honest. **`WIN·FT` = the de-biased winner** (clears "
        "A+B+C **and** `full_tape`, i.e. `full%` ≥ 80%). Prefer WIN·FT over WINNER; sort/relax "
        "`full%` yourself. Full tapes come from the mid-volume /events tail + live open-market "
        "capture — see docs/project2_section14_findings.md.",
        "",
        f"**{n_null} null-surviving winners** (WIN·NULL) — the trustworthy set — of "
        f"{n_winners_ft} de-biased (WIN·FT) / {n_winners} raw A+B+C winners. Full funnel by "
        "category:",
        "",
        "| " + " | ".join(funnel.columns) + " |",
        "|" + "|".join("---" for _ in funnel.columns) + "|",
    ]
    for _, r in funnel.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in funnel.columns) + " |")
    lines.append("")

    # Winners table (the actionable default view), then per-category detail.
    winners = default_view(table)
    lines += ["## Null-surviving winners (default view: A ∩ C ∩ full-tape ∩ real-world ∩ "
              "breadth ∩ FDR, sorted by copyable skill edge)", ""]
    if winners.empty:
        lines += [
            "_No cell survives the shuffled-outcome null + BH-FDR — see the verdict banner above._ "
            "The apparent winners on the `WINNER` / `WIN·FT` columns are the selected tail of a "
            "multiple-comparison search across the testable cells: the OOS persistence gate fires "
            "~5× the chance rate, but shuffling outcomes within each (category, price) bucket "
            "reproduces that excess, so no individual cell is distinguishable from the "
            "favorite-longshot null. **The winner list is withheld** (§1.4). The forward paper-test "
            "(spec §3.3) remains the only real arbiter; the durable deliverables are the discovery "
            "mechanisms + this null gate, not a published list.", ""]
    else:
        lines += [_category_table_md(winners, top=50), ""]

    shown = table[table["scored"]] if "scored" in table else table
    for cat in [c for c in funnel["category"] if c != ALL_CELL] + [ALL_CELL]:
        g = shown[shown["category"] == cat]
        if g.empty:
            continue
        n_all = int((table["category"] == cat).sum())
        lines += [f"## {cat}  ({n_all} cells total, {len(g)} scored, "
                  f"{int(g['is_winner'].sum())} winners) — top {top_per_category} scored", ""]
        lines += [_category_table_md(g, top_per_category), ""]
    return "\n".join(lines)


def render_html(table: pd.DataFrame, cfg: dict, top_per_category: int = 200,
                null_summary: dict | None = None) -> str:
    """Self-contained, offline, client-side click-to-sort tables (one per category
    tab). No external assets — matches the repo's low-dependency ethos.

    Only *scored* cells (>= min_forecaster_score_bets) are shown, capped per
    category: the ~300k thin 1-9-bet cells are noise in a dashboard (a 1-bet 100%-
    win cell has skill_edge_all ~0.9 and would top every sort) and would bloat the
    file to >100 MB. The full table (every cell) stays in forecasters.parquet."""
    funnel = category_funnel(table)  # funnel counts over the FULL table
    shown = table[table["scored"]] if "scored" in table else table
    cats = [c for c in funnel["category"] if c != ALL_CELL] + [ALL_CELL]
    cats = [c for c in cats if (shown["category"] == c).any()]

    sort_cols = [c for c in ("winner_null", "winner_full_tape", "is_winner",
                             "copyable_skill_edge_60_cents", "skill_edge_all") if c in table.columns]

    def table_html(g: pd.DataFrame, tid: str) -> str:
        g = g.sort_values(sort_cols, ascending=False).head(top_per_category)
        head = "".join(
            f'<th onclick="sortTable(\'{tid}\',{i})">{html.escape(label)}</th>'
            for i, (_, label) in enumerate(DISPLAY_COLUMNS)
        )
        body_rows = []
        for _, row in g.iterrows():
            cls = "winner" if bool(row.get("winner_null", row.get("winner_full_tape", row.get("is_winner")))) else ""
            tds = []
            for col, _ in DISPLAY_COLUMNS:
                val = row.get(col)
                sort_val = "" if pd.isna(val) else (float(val) if isinstance(val, (int, float, bool)) else str(val))
                tds.append(f'<td data-sort="{html.escape(str(sort_val))}">{html.escape(_fmt_cell(col, val))}</td>')
            body_rows.append(f'<tr class="{cls}">' + "".join(tds) + "</tr>")
        return (f'<table id="{tid}"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body_rows)}</tbody></table>')

    tabs = "".join(
        f'<button class="tab" onclick="showTab(\'{html.escape(c)}\')">{html.escape(c)} '
        f'({int((shown["category"] == c).sum())})</button>'
        for c in cats
    )
    sections = "".join(
        f'<div class="cat" id="cat-{html.escape(c)}" style="display:none">'
        f'<h2>{html.escape(c)} — showing top {top_per_category} scored cells</h2>'
        f'{table_html(shown[shown["category"] == c], "t_" + str(i))}</div>'
        for i, c in enumerate(cats)
    )
    n_winners = int(table["is_winner"].sum())
    n_winners_ft = int(table["winner_full_tape"].sum()) if "winner_full_tape" in table else 0
    n_null = int(table["winner_null"].sum()) if "winner_null" in table else 0
    # Verdict banner: reuse the markdown verdict text (strip the leading "> ").
    verdict = html.escape(_null_verdict_banner(null_summary, n_null).lstrip("> "))
    banner_bg = "#fdecea" if n_null == 0 else "#e7f6e7"
    funnel_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r[col]))}</td>" for col in funnel.columns) + "</tr>"
        for _, r in funnel.iterrows()
    )
    funnel_head = "".join(f"<th>{html.escape(c)}</th>" for c in funnel.columns)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Project 2 — Copyable Forecasters</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:1.5rem;color:#111;background:#fff}}
 h1{{font-size:1.3rem}} h2{{font-size:1.05rem;margin-top:.5rem}}
 .note{{color:#444;max-width:60rem;font-size:.9rem;line-height:1.4}}
 table{{border-collapse:collapse;margin:.5rem 0;font-size:.82rem;width:100%}}
 th,td{{border:1px solid #ddd;padding:.25rem .45rem;text-align:right;white-space:nowrap}}
 th:first-child,td:first-child{{text-align:left;font-family:monospace}}
 th{{background:#f3f3f3;cursor:pointer;position:sticky;top:0}}
 tr.winner{{background:#e7f6e7;font-weight:600}}
 .tab{{margin:.1rem;padding:.3rem .6rem;border:1px solid #ccc;background:#fafafa;cursor:pointer;border-radius:4px}}
 .funnel td,.funnel th{{text-align:center}}
 .verdict{{max-width:60rem;padding:.6rem .8rem;border-radius:6px;margin:.6rem 0;
   background:{banner_bg};font-size:.95rem;line-height:1.45}}
 @media(prefers-color-scheme:dark){{
   body{{background:#111;color:#eee}} th{{background:#222}} th,td{{border-color:#333}}
   tr.winner{{background:#173d17}} .tab{{background:#1c1c1c;border-color:#333;color:#eee}}
   .verdict{{background:#3a1a1a}}
 }}
</style></head><body>
<h1>Project 2 — Copyable Forecasters (per category)</h1>
<div class="verdict">{verdict}</div>
<p class="note"><b>{n_null} null-surviving winners</b> (WIN·NULL, the trustworthy set) of
{n_winners_ft} de-biased (WIN·FT) / {n_winners} raw A+B+C winners.
Gates: <b>A</b> skill edge real (persisted out-of-sample), <b>B</b> copyable (follower reachability
≥ 0.5 at ~60s), <b>C</b> copyable skill edge clears the cents floor, <b>FT</b> measured on FULL
(untruncated) tapes — <b>full%</b> ≥ 80% (§1.4: top-volume tapes are truncated to a near-resolution
slice that inflates skill 10–50×; <b>win%</b> near 100% flags the artifact), <b>NULL</b> survives the
shuffled-outcome null + BH-FDR (the multiple-testing gate; <b>null_p</b> = empirical p vs the
structural shuffle). Click any header to
sort. Sorted by copyable skill edge by default — <b>never</b> by raw returns/profit. Nothing is
dropped; relax any filter by sorting its column. Skill edge = favorite-longshot-neutralized residual,
per category baseline.</p>
<h2>Funnel by category</h2>
<table class="funnel"><thead><tr>{funnel_head}</tr></thead><tbody>{funnel_rows}</tbody></table>
<div>{tabs}</div>
{sections}
<script>
 function showTab(c){{document.querySelectorAll('.cat').forEach(d=>d.style.display='none');
   document.getElementById('cat-'+c).style.display='block';}}
 function sortTable(tid,col){{
   var t=document.getElementById(tid),tb=t.tBodies[0],rows=Array.from(tb.rows);
   var asc=t.getAttribute('data-sortcol')==col&&t.getAttribute('data-asc')!='1';
   rows.sort(function(a,b){{
     var x=a.cells[col].getAttribute('data-sort'),y=b.cells[col].getAttribute('data-sort');
     var nx=parseFloat(x),ny=parseFloat(y);
     if(!isNaN(nx)&&!isNaN(ny)){{return asc?nx-ny:ny-nx;}}
     return asc?(x>y?1:-1):(x<y?1:-1);
   }});
   rows.forEach(r=>tb.appendChild(r));
   t.setAttribute('data-sortcol',col);t.setAttribute('data-asc',asc?'1':'0');
 }}
 var first=document.querySelector('.cat');if(first)first.style.display='block';
</script></body></html>"""


def main() -> None:
    cfg = load_config()
    if not FORECASTER_TABLE_PATH.exists():
        print("[rank_forecasters] no forecaster table — run `python -m src.forecaster_metrics` first.")
        return
    table = pd.read_parquet(FORECASTER_TABLE_PATH)
    if table.empty:
        print("[rank_forecasters] forecaster table is empty.")
        return

    table, has_null = join_null_fdr(table)
    null_summary = load_null_summary()

    md = render_markdown(table, cfg, null_summary=null_summary)
    html_out = render_html(table, cfg, null_summary=null_summary)
    DASHBOARD_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_MD_PATH.write_text(md)
    DASHBOARD_HTML_PATH.write_text(html_out)

    n_winners = int(table["is_winner"].sum())
    n_null = int(table["winner_null"].sum())
    null_note = (f"{n_null} survive the shuffled-null+FDR gate" if has_null
                 else "null+FDR gate NOT run (run scripts/audit_forecaster_null.py)")
    print(
        f"[rank_forecasters] {len(table)} cells, {n_winners} raw A+B+C winners; "
        f"{null_note} -> {DASHBOARD_MD_PATH.name} + {DASHBOARD_HTML_PATH.name} (in {DISCOVERY_DIR})"
    )


if __name__ == "__main__":
    main()
