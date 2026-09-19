"""Resolution-event unit for the sports arm — Project 3 sports step 1.

WHY THIS EXISTS
---------------
The sports gate scored a corpus of 284 `market_id`s and called them 284
independent games. They are not. Joining those ids to their slugs shows

  * 117 season outrights — `will-the-<team>-win-the-2025-nba-finals` (54 NBA),
    `-win-the-2025-world-series` (40 MLB), `-win-the-2025-stanley-cup` (20 NHL) …
    Each block of them resolves on ONE championship, with mutually exclusive
    outcomes: exactly one pays.
  * 42 season props — `will-<player>-lead-the-nba-in-scoring` / `-in-assists`.
  * 83 derivative lines on THREE matches — `fifwc-esp-arg-2026-07-19-spread-*`
    (28), `fifwc-fra-esp-2026-07-14-*` (27), `nfl-sea-ne-2026-02-08-*` (14),
    `nba-sas-nyk-2026-06-10-*` (14).

Folded to what actually resolves, 284 markets are ~25 events. A market-block
bootstrap over that corpus treats one wallet's six World Cup team-outrights as
six independent draws when they are one bet on one tournament — the concentrated
single-event artifact that killed Project 2 §1.5 and cut the forecasters 52→12,
one level up. So `market_id` is the wrong cluster unit here.

`niche_l1` (the league) is the wrong unit in the OTHER direction: the main ledger
holds 25,390 distinct coded games across 193 league codes, and collapsing every
NBA game into one cluster would destroy exactly the breadth that makes sports
worth testing.

This module supplies the unit in between — the **resolution event**:

    coded game slug   ->  league + participants + date   (the match)
    prose outright    ->  the contest being bet on       (the championship)
    neither           ->  the market itself              (reported, see below)

TWO RULES, SYNTAX ONLY
----------------------
1. CODED (`mlb-chc-cin-2026-07-11-total-8pt5`): everything through the first
   date is the contest; the tail is the betting line. All spread/total/prop
   lines on one match collapse to that match. A `-winner-<name>-` infix (F1,
   IndyCar: `f1-belgian-grand-prix-winner-leclerc-2026-07-19`) folds away, so a
   field of drivers in one race is one event rather than one event per driver.

2. PROSE (`will-the-new-york-yankees-win-the-2025-world-series`): drop the
   leading entity and key on the predicate — `win-2025-world-series`. Every team
   in the field maps to the same key; a different season keys differently
   because the year is part of the predicate. This is deliberately a generic
   syntactic rule rather than a hand-maintained competition list, so it cannot
   be tuned per finding.

FALLBACK IS PERMISSIVE, AND REPORTED. A market matching neither rule becomes its
own event. That direction over-counts independence (it is the market-unit
assumption for that row), so `assign_events` also returns an `event_kind` column
and callers print the unmatched share: a cluster count is only as honest as the
share of it that was actually proven to share a resolution.

Computed from slug/question TEXT ONLY — never from outcomes, prices, residuals
or wallet identity, so the partition is blind to the answer it is used to test.

READ-ONLY / ANALYSIS-ONLY: pure functions, no network, no writes.
"""

from __future__ import annotations

import re

import pandas as pd

from src.slow_niche import market_family, narrative

# Bump when a rule change moves a market between event keys. The freeze manifest
# records this so a frozen sports cohort is reproducible against its own partition.
SPORTS_EVENT_VERSION = "1.0.0"

EVENT_COLUMNS = ("event", "event_kind")

# Trailing capture-id: either a 10+ digit epoch (`-1743783173`) or the 3-digit
# disambiguator Polymarket appends to duplicate slugs (`-win-the-2026-fifa-world-cup-464`).
# A bare 4-digit tail is a SEASON YEAR and must survive, or two seasons of one
# championship would collapse into a single event and halve the cluster count.
_EPOCH_SUFFIX_RE = re.compile(r"-(?:\d{3}|\d{10,})$")
_DATE_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})")
_WINNER_RE = re.compile(r"-winner-[a-z0-9-]+?(?=-\d{4}-\d{2}-\d{2})")

# The predicate markers a prose sports market hangs on. First hit wins, so
# "will-the-panthers-win-super-bowl-2025" keys on "win-super-bowl-2025".
_PREDICATES = ("-win-", "-lead-", "-score-", "-reach-", "-advance-",
               "-qualify-", "-finish-", "-make-", "-be-", "-cry-")

_LEADING_WILL_RE = re.compile(r"^will-")
_THE_RE = re.compile(r"-the-")


def _normalize(slug: str | None) -> str:
    """Lower-cased slug with the trailing capture-id stripped. `-the-` is folded
    out so `win-the-2025-world-series` and `win-2025-world-series` are one key."""
    s = _EPOCH_SUFFIX_RE.sub("", str(slug or "").lower())
    return _THE_RE.sub("-", s)


def resolution_event(slug: str | None, question: str | None = None,
                     market_id: str | None = None) -> tuple[str, str]:
    """The (event key, event kind) a market resolves on.

    Returns one of:
      (`<code>-<participants>-<date>`, "game")        — a single match
      (`<predicate>`,                  "competition") — a championship / season prop
      (`market:<market_id>`,           "market")      — unmatched; own event
    """
    s = _normalize(slug)
    if not s:
        return f"market:{market_id}", "market"

    # -- rule 1: coded game slug -------------------------------------------
    folded = _WINNER_RE.sub("", s)          # f1/indycar: drop the driver field
    m = _DATE_RE.search(folded)
    if m:
        return folded[: m.end()], "game"

    # -- rule 2: prose outright / season prop -------------------------------
    body = _LEADING_WILL_RE.sub("-", s)
    for pred in _PREDICATES:
        i = body.find(pred)
        if i != -1:
            key = body[i + 1:]
            if key:
                return key, "competition"

    return f"market:{market_id}", "market"


# The line SIDE inside a form — the axis a sub-league miscalibration would live
# on (home-favourite bias is the classic one). Deliberately a small closed
# allowlist: the remaining tail tokens are team codes and handicap numbers, and
# keying on those would push the baseline cell toward the market itself, which is
# the "too fine" false negative slow_baseline warns about.
_SIDES = ("home", "away", "draw", "over", "under", "yes", "no",
          "1h", "2h", "ot", "btts")


def sub_form(slug: str | None) -> str:
    """The side token within a market's form, or `na` when it has none.

    `nba-sas-nyk-2026-06-10-spread-home-1pt5` -> `home`
    `mlb-chc-cin-2026-07-11-total-8pt5`       -> `na`   (side lives on the token,
                                                         not the slug)
    prose outrights                           -> `na`
    """
    s = _normalize(slug)
    m = _DATE_RE.search(s)
    if not m:
        return "na"
    for tok in s[m.end():].split("-"):
        if tok in _SIDES:
            return tok
    return "na"


def assign_sub_forms(markets: pd.DataFrame, slug_col: str = "slug",
                     l2_col: str = "niche_l2") -> pd.DataFrame:
    """Attach `sub_form` and `niche_l3` (= niche_l2 | sub_form) — the finer level
    the mandatory finer-baseline re-check refits against."""
    out = markets.copy()
    if out.empty:
        out["sub_form"] = pd.Series(dtype=object)
        out["niche_l3"] = pd.Series(dtype=object)
        return out
    out["sub_form"] = [sub_form(s) for s in out[slug_col]]
    parent = out[l2_col] if l2_col in out.columns else pd.Series("", index=out.index)
    out["niche_l3"] = parent.astype(str) + "|" + out["sub_form"].astype(str)
    return out


def is_sports_market(slug: str | None, question: str | None = None,
                     category: str | None = None) -> bool:
    """Whether a market belongs to the sports universe.

    Delegates to the tested `slow_niche` labelling rather than re-deriving it:
    a market is sports when its event-complex family maps to the `sports`
    narrative. That covers coded league slugs (`lg_*` — 193 codes in the ledger,
    from `atp` to `cs2` to `nwsl`), the `sports_*` categories, and golf / chess /
    combat sports / esports. The weather-temperature generator also carries a
    date-coded slug but is caught as `weather_temp` upstream, so it does not leak
    in here."""
    return narrative(market_family(slug, question, category)) == "sports"


def assign_events(markets: pd.DataFrame, slug_col: str = "slug",
                  question_col: str = "question",
                  market_id_col: str = "market_id") -> pd.DataFrame:
    """Attach `event` and `event_kind` to a per-market frame.

    Labelling is computed once per distinct slug and mapped back, so a multi-
    million-row tape costs one pass over its markets rather than per-bet regex
    evaluation. Additive — nothing is dropped or reordered."""
    out = markets.copy()
    if out.empty:
        out["event"] = pd.Series(dtype=object)
        out["event_kind"] = pd.Series(dtype=object)
        return out

    slugs = out[slug_col] if slug_col in out.columns else pd.Series([None] * len(out), index=out.index)
    questions = (out[question_col] if question_col in out.columns
                 else pd.Series([None] * len(out), index=out.index))
    mids = (out[market_id_col] if market_id_col in out.columns
            else pd.Series([None] * len(out), index=out.index))

    keys = pd.DataFrame({"slug": slugs.astype(object).to_numpy(),
                         "question": questions.astype(object).to_numpy(),
                         "market_id": mids.astype(object).to_numpy()})
    pairs = [resolution_event(s, q, m) for s, q, m in
             zip(keys["slug"], keys["question"], keys["market_id"])]
    out["event"] = [p[0] for p in pairs]
    out["event_kind"] = [p[1] for p in pairs]
    return out


def event_summary(markets: pd.DataFrame) -> pd.DataFrame:
    """Per-event market counts and kind — the diagnostic a reviewer reads before
    trusting any cluster count built on this partition."""
    if markets.empty or "event" not in markets.columns:
        return pd.DataFrame()
    agg = {"n_markets": ("event", "size"), "kind": ("event_kind", "first")}
    if "niche_l1" in markets.columns:
        agg["league"] = ("niche_l1", "first")
    tbl = markets.groupby("event").agg(**agg).reset_index()
    return tbl.sort_values("n_markets", ascending=False).reset_index(drop=True)
