"""Sub-category ("niche") assignment for the slow universe — Project 3 step 6a
(docs/project3_slow_verdict.md "Coarse baseline" limit; HANDOFF "Project 3
slow-market pivot").

WHY THIS EXISTS
---------------
The step-5c positive residualizes against E[outcome | entry_price, category],
where `category` is the 14-value `discover.classify_market` label. 59% of slow
bets land in `other`. A structurally-mispriced sub-niche *inside* `other` hands
every buyer in it a large positive residual — the favorite-longshot structural
edge one level up, at the category level, un-stripped. That survives disjoint
data, the cluster null, the concentration guards and the horizon check, because
the mispricing sits in both halves and across genuinely-broad markets. It is not
followable wallet alpha: a copier would get it by trading the niche directly.

This module supplies the finer partition needed to strip it.

DESIGN: TWO AXES, DERIVED FROM SYNTAX ONLY
------------------------------------------
  family   — the market generator / event complex ("US strikes Iran by <date>"
             markets are one complex; "highest temperature in <city>" markets are
             one generator; `lg_fifwc` is one tournament).
  form     — the market's structural shape (moneyline vs spread vs over/under vs
             a "by <deadline>" ladder vs an outright-candidate field). Two markets
             in the same complex at the same price can still have very different
             true rates if one is a spread and the other an outright.

Both are computed from slug/question TEXT ONLY — never from outcomes, residuals,
prices, or wallet identity. The partition is therefore blind to the answer it is
used to test, and cannot be tuned toward "survivors live" or "survivors die".

The Gamma `tags` field would have been the natural external partition, but it is
populated for 16 of 348,657 markets in the sidecar (measured 2026-07-25) — the
column was written but never filled. Slug syntax is also strictly more
informative here, since it encodes `form`, which tags do not.

NOTHING IS DROPPED. Every market gets a family and a form; unmatched markets get
`misc` and pool to their parent category. This is an additive labelling module.

READ-ONLY / ANALYSIS-ONLY: pure functions, no network, no writes.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Bump when the partition changes in any way that moves a market between labels.
# The freeze manifest records this, so a frozen cohort is always reproducible
# against the exact partition that produced it.
NICHE_PARTITION_VERSION = "1.0.0"
NARRATIVE_MAP_VERSION = "1.0.0"

# --- slug shape ------------------------------------------------------------
_DATE_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})-?")
_EPOCH_SUFFIX_RE = re.compile(r"-\d{10,}$")      # trailing capture-id on some slugs
_TEMP_SLUG_RE = re.compile(r"^(highest|lowest)-temperature-in-")

# --- question shape (prose markets) ----------------------------------------
_MONTHS = (r"january|february|march|april|may|june|july|august|september|"
           r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_BY_DEADLINE_RE = re.compile(rf"\b(by|before|through)\b.*\b(\d{{4}}|{_MONTHS}|today|tomorrow|"
                             r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
_ON_DATE_RE = re.compile(rf"\bon\b\s+({_MONTHS})\s+\d{{1,2}}")
_OUTRIGHT_RE = re.compile(r"\bbe the next\b|\bwin the\b|\bwinner\b|\bbe elected\b|"
                          r"\bbe named\b|\bbe the\b.*\bnominee\b")
_THRESHOLD_RE = re.compile(r"\bhit\b|\bclose between\b|\bbetween\b\s*\$|\babove\b|\bbelow\b|"
                           r"\bor more\b|\bor fewer\b|\bo/u\b|\bover/under\b|"
                           r"°c|°f|\bexceed\b|\breach\b")
_DURATION_RE = re.compile(r"\blast\b.*\b(days?|weeks?|hours?)\b.*\b(or more|or fewer|or longer)\b")
_COUNT_RE = re.compile(r"\b\d+([-–]\d+)?\s+(ships?|days?|seats?|games?|goals?|points?)\b")


def bet_form(slug: str | None, question: str | None) -> str:
    """Structural shape of the market.

    Coded sports slugs carry the form in the tail after the date
    (`...-2026-05-10-spread-away-1pt5`); prose markets carry it in the question
    shape ("... by <date>?" is a deadline ladder; "Will X be the next pope?" is
    an outright candidate field). Returns `misc` when nothing matches."""
    s = _EPOCH_SUFFIX_RE.sub("", slug or "")
    q = (question or "").lower()

    # -- coded slug: the tail after the date is the form ---------------------
    m = _DATE_RE.search(s)
    if m:
        tail = s[m.end():]
        if tail.startswith("team-total"):
            return "team_total"
        if tail.startswith("total-games"):
            return "total_games"
        if tail.startswith("total"):
            return "total"
        if tail.startswith("spread"):
            return "spread"
        if tail.startswith("exact-score"):
            return "exact_score"
        if tail.startswith("team-to-advance"):
            return "advance"
        if tail.startswith("btts"):
            return "btts"
        if tail == "draw":
            return "draw"
        if re.fullmatch(r"game\d+", tail):
            return "map_game"
        if tail == "":
            return "match_line"
        if re.fullmatch(r"[a-z0-9]{2,12}", tail):
            return "moneyline"

    # -- prose: the question shape -------------------------------------------
    if _TEMP_SLUG_RE.match(s):
        return "threshold"
    if _DURATION_RE.search(q):
        return "duration"
    if _THRESHOLD_RE.search(q) or _COUNT_RE.search(q):
        return "threshold"
    if _OUTRIGHT_RE.search(q):
        return "outright"
    if _ON_DATE_RE.search(q):
        return "on_date"
    if _BY_DEADLINE_RE.search(q):
        return "deadline"
    return "misc"


# ---------------------------------------------------------------------------
# Family: the market generator / event complex.
#
# ORDERED rules — first match wins, so the more specific generator (Hormuz ship
# transit counts, crude-oil thresholds) is caught before the broad geopolitical
# complex that shares its keywords. Keyword sets are in the same idiom as the
# tested `discover.classify_market`.
# ---------------------------------------------------------------------------

_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # -- mechanical generators (must precede the event complexes) ------------
    ("hormuz_shipping", ("ships transit", "ship transit", "vessels transit",
                         "transit the strait", "targets shipping")),
    ("commodity_crude", ("crude oil", "wti", "brent", "opec")),
    ("commodity_metal", ("gold close", "will gold", "silver close", "gold hit",
                         "gold price", "silver hit")),
    ("commodity_energy", ("natural gas", "gasoline price", "henry hub")),
    ("crypto_airdrop", ("airdrop",)),
    ("conclave", ("next pope", "be elected pope", "new pope", "papal conclave")),
    # -- geopolitical event complexes ---------------------------------------
    ("geo_iran", ("iran", "iranian", "khamenei", "hormuz", "fordow", "kharg",
                  "tehran", "persian gulf", "irgc")),
    ("geo_israel", ("israel", "hezbollah", "hamas", "gaza", "lebanon",
                    "netanyahu", "idf", "west bank")),
    ("geo_ukraine", ("ukraine", "russia", "zelensky", "zelenskyy", "putin",
                     "kremlin", "moscow", "donbas")),
    ("geo_venezuela", ("venezuela", "maduro", "caracas")),
    ("geo_asia", ("china", "taiwan", "north korea", "kim jong", "xi jinping",
                  "india", "pakistan")),
    ("geo_other", ("military action", "ceasefire", "peace deal", "invade",
                   "airstrike", "war ", "coup", "nato")),
    # -- domestic politics ---------------------------------------------------
    ("us_shutdown", ("government shutdown", "shutdown last", "shutdown end")),
    ("us_politics", ("trump", "biden", "epstein", "congress", "senate",
                     "supreme court", "pardon", "white house", "cabinet",
                     "impeach", "house of representatives", "electoral")),
    ("intl_politics", ("starmer", "government of canada", "prime minister",
                       "chancellor", "president of", "leader out", "out as",
                       "officially leave office", "election")),
    # -- other real-world ----------------------------------------------------
    ("ai_tech", ("ai model", "gpt", "openai", "anthropic", "claude", "gemini",
                 "llm", "artificial intelligence")),
    ("corporate", ("ceo", " ipo", "acquire", "acquisition", "merger", "bankrupt",
                   "earnings")),
    ("awards_culture", ("oscar", "grammy", "emmy", "eurovision", "nobel",
                        "box office", "rotten tomatoes", "time person")),
    ("weather_storm", ("hurricane", "tropical storm", "typhoon", "tornado")),
    ("disaster", ("wildfire", "earthquake", "volcano", "flood")),
    ("golf", ("fedex cup", "pga", "the masters", "open championship",
              "invitational", "ryder cup")),
    ("chess", ("chess", "sinquefield", "grandmaster", "candidates tournament")),
    ("combat_sports", ("boxing", "usyk", "canelo", "heavyweight title")),
    ("esports", ("valorant", "counter-strike", "dota", "league of legends")),
    ("macro_rates", ("treasury yield", "fed ", "interest rate", "cpi ",
                     "inflation", "recession", "gdp")),
)


def market_family(slug: str | None, question: str | None,
                  category: str | None = None) -> str:
    """The market's generator / event complex.

    Precedence:
      1. the temperature generator (slug-shaped, very large and very mechanical);
      2. a coded league slug -> `lg_<code>` (the token before the date);
      3. the ordered keyword complexes above;
      4. `misc` — which pools to the parent category, never to noise.

    `category` is accepted so a caller can keep a non-`other` category's own label
    as its family when no finer rule fires (a sports_nba market with no keyword hit
    is still meaningfully "NBA")."""
    s = _EPOCH_SUFFIX_RE.sub("", slug or "")
    q = (question or "").lower()
    text = f"{q} {s.replace('-', ' ').lower()}"

    if _TEMP_SLUG_RE.match(s) or "temperature in" in q:
        return "weather_temp"

    m = _DATE_RE.search(s)
    if m:
        head = s[: m.start()]
        code = head.split("-")[0] if head else ""
        if code and re.fullmatch(r"[a-z][a-z0-9]{1,9}", code):
            return f"lg_{code}"

    for fam, kws in _FAMILY_KEYWORDS:
        if any(k in text for k in kws):
            return fam

    if category and category != "other":
        return category
    return "misc"


# ---------------------------------------------------------------------------
# Frame-level assignment
# ---------------------------------------------------------------------------

NICHE_COLUMNS = ("niche_family", "niche_form", "niche_l1", "niche_l2")


def assign_niches(markets: pd.DataFrame,
                  slug_col: str = "slug",
                  question_col: str = "question",
                  category_col: str = "category") -> pd.DataFrame:
    """Attach `niche_family`, `niche_form`, `niche_l1`, `niche_l2` to a frame.

    Labelling is computed once per distinct (slug, question, category) triple and
    mapped back, so a 4M-row bet tape costs one pass over its ~380k markets rather
    than 4M regex evaluations.

    The hierarchy the baseline pools along is
        niche_l2 -> niche_l1 -> category -> real-world -> global
    where
        niche_l1 = family
        niche_l2 = family | form
    """
    out = markets.copy()
    if out.empty:
        for c in NICHE_COLUMNS:
            out[c] = pd.Series(dtype=object)
        return out

    cats = out[category_col] if category_col in out.columns else pd.Series(
        ["other"] * len(out), index=out.index)
    keys = pd.DataFrame({
        "slug": out[slug_col].astype(object),
        "question": out[question_col].astype(object),
        "category": cats.astype(object),
    })
    uniq = keys.drop_duplicates()
    fam = [market_family(s, q, c) for s, q, c in
           zip(uniq["slug"], uniq["question"], uniq["category"])]
    frm = [bet_form(s, q) for s, q in zip(uniq["slug"], uniq["question"])]
    uniq = uniq.assign(niche_family=fam, niche_form=frm)

    merged = keys.merge(uniq, on=["slug", "question", "category"], how="left")
    out["niche_family"] = merged["niche_family"].to_numpy()
    out["niche_form"] = merged["niche_form"].to_numpy()
    out["niche_l1"] = out["niche_family"]
    out["niche_l2"] = out["niche_family"].astype(str) + "|" + out["niche_form"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Narrative: one level ABOVE the event complex
# ---------------------------------------------------------------------------
#
# The complex unit catches a rolling-deadline ladder. It does NOT catch a wallet
# betting CORRELATED complexes: "Iran closes Hormuz" + "Israel strikes Iran" +
# "Hormuz shipping falls" are three distinct complexes but one geopolitical bet.
# That is the same confound one level up, and this map is what measures it.
#
# DIAGNOSTIC ONLY — additive metadata and a flag. It never gates `edge_persisted`
# and never removes a row. Retrospective analysis cannot close this regress
# (there is always a level above); the forward test is what breaks it.
#
# The crux is `commodity_crude`. Crude spikes on Hormuz risk, so a crude
# specialist may be making the same bet as an Iran specialist — but crude also
# trades on OPEC, demand and inventories, which have nothing to do with the Gulf.
# Rather than resolve that by fiat, BOTH readings are computed and reported:
#   strict — commodity_crude/energy stand alone
#   broad  — commodity_crude/energy fold into mideast_escalation
# and the sensitivity between them is part of the finding.

_NARRATIVE_BASE: dict[str, tuple[str, ...]] = {
    "mideast_escalation": ("geo_iran", "geo_israel", "hormuz_shipping"),
    "ukraine_war": ("geo_ukraine",),
    "great_power": ("geo_asia", "geo_other"),
    "us_domestic": ("us_politics", "us_shutdown"),
    "intl_politics": ("intl_politics",),
    "macro": ("macro_rates", "commodity_metal"),
    "energy_commodities": ("commodity_crude", "commodity_energy"),
    "tech_crypto": ("ai_tech", "crypto_event", "crypto_airdrop"),
    "culture": ("awards_culture", "culture", "conclave"),
    "weather": ("weather_temp", "weather_storm", "disaster"),
    "corporate": ("corporate",),
    "sports": ("golf", "chess", "combat_sports", "esports"),
}

# Families folded into mideast_escalation under the BROAD reading only.
_MIDEAST_LINKED = ("commodity_crude", "commodity_energy")

_FAMILY_TO_NARRATIVE = {fam: narr for narr, fams in _NARRATIVE_BASE.items()
                        for fam in fams}


def narrative(family: str | None, mode: str = "strict") -> str:
    """Map an event-complex family to its higher-level narrative.

    `mode="broad"` additionally folds the energy complex into
    `mideast_escalation` — see the module note on why this is reported both ways
    rather than decided. Sports leagues (`lg_*`) and residual `sports_*`
    categories collapse to `sports`; anything unmapped becomes `other`, which is
    a real bucket, not a drop."""
    if not family:
        return "other"
    f = str(family)
    if mode == "broad" and f in _MIDEAST_LINKED:
        return "mideast_escalation"
    if f in _FAMILY_TO_NARRATIVE:
        return _FAMILY_TO_NARRATIVE[f]
    if f.startswith("lg_") or f.startswith("sports"):
        return "sports"
    return "other"


def assign_narratives(bets: pd.DataFrame, family_col: str = "niche_l1") -> pd.DataFrame:
    """Attach `narrative_strict` and `narrative_broad` to a frame carrying the
    event-complex family. Additive; nothing is dropped."""
    out = bets.copy()
    if out.empty:
        out["narrative_strict"] = pd.Series(dtype=object)
        out["narrative_broad"] = pd.Series(dtype=object)
        return out
    fams = out[family_col].astype(object)
    uniq = pd.unique(fams)
    strict = {f: narrative(f, "strict") for f in uniq}
    broad = {f: narrative(f, "broad") for f in uniq}
    out["narrative_strict"] = fams.map(strict).to_numpy()
    out["narrative_broad"] = fams.map(broad).to_numpy()
    return out


def effective_n(labels) -> float:
    """Effective number of distinct labels = 1 / HHI of their shares. Used for
    both complex breadth and narrative breadth."""
    s = pd.Series(list(labels)).value_counts(normalize=True).to_numpy()
    return float(1.0 / np.sum(s ** 2)) if s.size else 0.0


def narrative_spread(resid: pd.DataFrame, wallets, oos_split: float = 0.5,
                     single_narrative_threshold: float = 0.80) -> pd.DataFrame:
    """Per-wallet narrative spread over the HELD-OUT half, under both maps.

    `single_narrative_flag` marks a wallet whose edge collapses to one narrative
    even across distinct event complexes. Metadata only — it never gates."""
    from src.validate import split_in_sample_out_of_sample

    wallets = set(wallets)
    rows = []
    for w, g in resid[resid["wallet"].isin(wallets)].groupby("wallet", sort=False):
        _, oos = split_in_sample_out_of_sample(g, oos_split)
        if oos.empty:
            continue
        rec = {"wallet": w, "out_n": len(oos),
               "n_complexes": int(pd.unique(oos["niche_l1"]).size),
               "eff_complexes": effective_n(oos["niche_l1"])}
        for mode in ("strict", "broad"):
            col = f"narrative_{mode}"
            vc = oos[col].value_counts()
            rec[f"n_narratives_{mode}"] = int(vc.size)
            rec[f"eff_narratives_{mode}"] = effective_n(oos[col])
            rec[f"top_narrative_{mode}"] = str(vc.index[0])
            rec[f"top_narrative_share_{mode}"] = float(vc.iloc[0] / len(oos))
        rec["single_narrative_flag"] = bool(
            rec["top_narrative_share_broad"] >= single_narrative_threshold)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("top_narrative_share_broad", ascending=False)


def niche_summary(bets: pd.DataFrame, level: str = "niche_l2") -> pd.DataFrame:
    """Population diagnostic: bets, distinct markets and distinct wallets per
    niche. Reported so a reviewer can see bin populations before trusting any
    baseline fitted on them (the thin-bin false-negative trap). Never a finding."""
    if bets.empty or level not in bets.columns:
        return pd.DataFrame()
    agg = {"n_bets": (level, "size")}
    if "market_id" in bets.columns:
        agg["n_markets"] = ("market_id", "nunique")
    if "wallet" in bets.columns:
        agg["n_wallets"] = ("wallet", "nunique")
    tbl = bets.groupby(level).agg(**agg).reset_index()
    tbl["share"] = tbl["n_bets"] / len(bets)
    return tbl.sort_values("n_bets", ascending=False).reset_index(drop=True)
