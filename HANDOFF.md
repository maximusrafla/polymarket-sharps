# HANDOFF — Polymarket Sharps

Written for a future session starting cold, with no memory of prior chats. Read
this and `CLAUDE.md` before touching anything. `DECISIONS.md` has the data-source
and ingest-mechanics reasoning; this file covers *why the analysis is shaped the
way it is* and *what to be suspicious of*.

> ### ⭐ Read `docs/polymarket_mechanics.md` before trusting any timing number
>
> Ground-truth reference on how Polymarket actually works, built 2026-07-26 from
> primary sources and verified against the live API and Polygon mainnet. It exists
> because much of this repo's analysis rests on mechanics that were never checked.
> It has a section **"Assumptions this repo has made that are WRONG"** (16 entries).
> The two that matter most for the measurements here:
>
> - **The tape's `price` is a post-slippage VWAP across 2–8+ fills**, not a book
>   level — 34% of prints sit off the tick grid because of it. One `/trades` row is
>   one taker's aggregated match, not one trade. Every feature in this repo is
>   computed from that tape, so this is a statement about what the input *is*.
> - **Taker fees are real and were zero in everything measured here.**
>   `fee = shares × k × p × (1−p)` with **k = 0.04 politics / 0.05 sports / 0.07
>   crypto** (0 only for geopolitics), live since 2026-01-05 and confirmed on-chain
>   to 7 s.f. Every edge in this repo is a **gross** number from a zero-fee era, and
>   the charge is the same order of magnitude as the edges being measured. That is a
>   statement about whether the effect is reachable, not about a strategy.

> ### 🧭 Read `docs/FRAGILITY_INDEX.md` before citing any conclusion in this file
>
> A confidence map, not a summary: every load-bearing claim graded A/B/C/D by how
> much it would take to overturn it, with the cheapest falsifier attached. It exists
> because **this project's characteristic error is conclusions hardening into
> instructions** — a `do NOT re-attempt` note once fossilised a mis-scoped threshold
> and cost a real 12-wallet cohort, and two more instances surfaced on 2026-07-30
> from a fresh pass rather than new data. Grade **C** is where the errors are: claims
> that move when a defensible alternative null, threshold or cluster unit is chosen.
> Grade **D** is three load-bearing negatives with no committed script.

---

## What this project is (and why it exists)

An **identification engine** for Polymarket. It ingests public on-chain trade
history and ranks every wallet by how *genuinely sharp* it is — how consistently
it earns positive edge versus the market price at the moment it entered, with that
edge shown to survive out-of-sample.

It exists to answer one question: *which wallets are worth watching, because their
skill is real and repeatable rather than a hot streak?* The output is a ranked
table (`data/processed/ranked_wallets.parquet`) and a human-readable summary
(`data/processed/report.md`), plus an alert-only monitor (`src/watch.py`).

**Scope is analysis only.** This repo measures; it does not trade, mirror wallets
or place orders, and it holds no keys. Every fetch it makes is a public,
unauthenticated GET. The question it exists to answer is whether the wallets it
ranks are really skilled and whether that skill is reachable by anyone watching
from outside — not what to do about it.

Pipeline stages, in order:
`ingest.py` → `features.py` → `validate.py` → `rank.py` → `report.py`, with
`watch.py` layered on top of the ranked output.

---

## Why raw win rate and raw profit are traps

Two "obvious" metrics both measure the wrong thing:

- **Raw win rate is inflated by favorite-betting.** A wallet that only ever backs
  heavy favorites at 0.90 will win ~90% of the time and look brilliant, while
  earning essentially nothing per bet (and losing badly on the 10% it drops).
  Win rate rewards picking likely outcomes, not *mispriced* ones. Skill is buying
  something that pays more than you paid for it — which a favorite at fair odds is
  not.
- **Raw profit is inflated by size.** A wallet that bets huge stakes on coin-flip
  edges racks up big dollar totals through volume and bankroll, not skill. Profit
  conflates "how much they wagered" with "how good each decision was."

### Why edge-over-entry-price is the core metric

For each resolved bet we compute **`resolved_value − entry_price`**: the realized
outcome (1 if the backed side won, 0 if it lost) minus the price paid at entry.
This is the per-bet edge in probability/price units. A wallet that repeatedly pays
0.60 for outcomes that resolve YES is capturing +0.40 of edge per correct call and
is genuinely finding mispricing; a favorite-better paying 0.90 for a 0.90-likely
outcome averages ~0 edge no matter how often it "wins."

Averaged over many bets, **mean edge is dimensionless in stake and neutral to how
likely the pick was** — it isolates *"did they pay less than the thing turned out
to be worth?"*, which is exactly the skill we want and exactly what win rate and
profit hide.

> ⚠️ **Refinement (2026-07-18, now the actual metric):** raw `resolved_value −
> entry_price` turned out to be dominated by a structural favorite-longshot base
> rate (~62% of bets positive by default), not skill. Ranking and validation now
> key on **skill (residual) edge** = `outcome − E[outcome | entry_price]` — how
> much the wallet beat the price it actually paid. Raw edge is still computed and
> reported alongside. See DECISIONS.md "Favorite-longshot residualization".

Supporting metrics, all secondary to edge:
- **Copy window** — gap between entry price and estimated fair value; how much room
  a follower still has. A first-class ranking variable (`w_copy_window`).
- **Earliness** — did price drift toward their side *after* they entered (they were
  early) vs. before (they chased)?
- **Sample size** — resolved-bet count; small samples are shrunk hard toward zero.
- **Breadth** — distinct markets, so one lucky event can't dominate.
- **Time-consistency** — does edge hold across chronological buckets or is it one
  spike?

---

## Why out-of-sample validation is non-negotiable

With thousands of wallets, **some will sit in the "sharp" corner by pure chance.**
If you rank on in-sample edge alone, you are guaranteed to crown the luckiest noise,
not the most skilled wallets — the tail of a random distribution looks identical to
skill until you ask it to repeat.

So `validate.py` does the anti-self-deception step: for each wallet, split its
resolved bets **chronologically** — first half (`oos_split = 0.5`) is the
select/fit half, second half is held out. A wallet is a **candidate** if its
in-sample **skill** edge is positive; it **persists** only if its held-out skill
edge is positive **and statistically significantly greater than 0** — the
market-block bootstrap p (`out_of_sample_cluster_p`) must clear Project 1's own
scoped `scoring.project1.oos_significance_alpha` (**0.005** since 2026-07-26;
tightened from 0.05 as a multiplicity correction over the 691-candidate search —
see "Tightened significance threshold" below), over at least `min_bets_per_half`
bets. `report.py`
prints in- and out-of-sample edge side by side so the decay is visible, and this
runs *before* any ranking is emitted. Wallets whose edge does not persist are **not
dropped** (nothing ever is — see below) but carry a reliability penalty
(`non_persisted_penalty = 0.3`) in the score.

> ✅ **This was the original weak spot, now fixed and battle-tested.** The first
> version used a *sign-only* test on tiny samples, which re-measured the
> favorite-longshot base rate and produced an implausible ~73% persistence. It was
> replaced with the skill-edge + significance test above (persistence → ~27%; a
> shuffled-outcome null persists only ~5%). The **full deep backfill (2026-07-19)**
> then proved it out: deepening the top-100 to ~5,000 bets/half collapsed the
> thin-sample "sharp" edges to noise (median held-out skill 0.093 → 0.0007) and
> left **21 persisters, 16 of them deep and rock-solid** (p down to 5.8e-26). See
> "Full deep backfill" below and DECISIONS.md "Out-of-sample validation".

---

## Why flags are metadata-only and never affect score or rank

`manufactured_record_flag` and `pattern_flag` are **additive columns, nothing
more.** Per CLAUDE.md's ironclad rule: **every wallet is ingested, scored, and
ranked — nothing is ever dropped, soft-deleted, down-weighted, or filtered out for
any reason.**

The reasoning:
- Flags encode *suspicion*, and suspicion is fallible. "Trades against co-funded
  wallets" can be wash-trading, or can be an artifact of how we linked addresses.
  Baking a guess into the score corrupts the ranking with our own error and makes it
  impossible for the user to see what the wallet would look like un-penalized.
- We deliberately **do not infer or record intent** — no "insider" labels asserted
  as fact. Flags are descriptive notes (`manufactured_record_flag`,
  `pattern_flag = late_concentrated_entry`, etc.), not verdicts.
- The design goal is a **complete dataset the user filters themselves.** The engine
  collects everything and hides nothing; the user sorts/filters on flag columns
  according to their own risk tolerance.

**Invariant to preserve:** flags MUST NOT feed the score, alter rank order, or
remove rows. If you find yourself wanting a flag to change a number, don't — surface
it as a column and let the user decide.

---

## `watch.py` — the alert-only monitor

`watch.py` is a real-time monitor that loads the top-N ranked wallets, polls public
Polymarket data on an interval (default 60s), and — when a watched wallet takes a
new position whose **current** price is still inside that wallet's profitable
copy-window — emits an alert (stdout + `data/alerts.log` + optional webhook), once
per position.

**It is read-only** — it observes and notifies, no order placement or key handling.
Its practical value is as an instrument: it is the only part of the repo that sees
a signal at the latency a watcher would actually see it. NB: the edge-decay
analysis found the real-world copy-window closes in under a minute, so a 60s poll
is already slower than the window it is measuring.

---

## 🚩 START HERE — session of 2026-07-26/27, and the ONE task queued next

**Owner's decision, taken at the end of the session: DROP micro-crypto entirely. Build a
deep REAL-WORLD wallet dataset instead.** Rationale below; it is the highest-value
unstarted work in the project and everything else is either finished or running itself.

### ✅ STATUS 2026-07-27: THE DEEPENING RAN AND IS DONE — screens are the next step

Completed **2026-07-27T13:29:10Z**, no warnings, no fetch errors. Fetch 06:43→09:05
(2 h 22 m, 2,117/2,117 wallets); resolve 09:05→13:29 (4 h 23 m, 306,525 markets pulled
from CLOB). The disk guard never fired: **0.52 GB used against a 1.6 GB cap**, 4.2 GB
free.

| | |
|---|---|
| wallets deepened | **2,117** (+383 reused from slow/sports = 2,500 drawn) |
| trades | **7,052,626** across 88 shards, 410,523 markets |
| resolved BUY bets (the scoring unit) | **3,842,708** |
| **REAL-WORLD** resolved bets | **3,385,372 — 88.1%** of the tape |
| micro-crypto | 457,336 — 11.9% |
| tape span | 2023-07-27 → 2026-07-27 |

**Wallets by real-world resolved bets:** ≥100 → **1,953**; ≥500 → **992**; ≥1000 → **758**.
For scale, the shared ledger's *entire* real-world side had **98** wallets with 1000+ bets.
That is the collection artifact, closed: **7.7× more deep real-world wallets than the whole
ledger had**, for 0.52 GB.

**How the split was measured, and its one caveat.** `python -m src.realworld_deepen profile`
classifies markets with `discover.classify_market` on slug/question — the same classifier
that built the discovery corpus, no network needed. The `market_meta` sidecar is **not**
usable for this: it covers only 116k of these 410k markets (294,529 absent, 32,476 NaN
lifespan), which would park 66% of the tape as "unknown". Deriving lifespan from our own
tape would be *actively wrong* — we hold only our wallets' fills, so a market touched once
looks zero-second and reads as micro. **Caveat: 48.8% of the tape lands in the classifier's
`other` fallback.** Sampling it shows geopolitics, golf, esports and league codes the
keyword rules don't know (`fl1`, `itc`, `fifwc`, `lol`) — a *granularity* gap, not a micro
leak, so the 88.1% figure holds but per-category splits below `other` do not.
Artifacts: `wallet_profile.parquet`, `market_category.parquet` (both in
`data/interim/realworld/`, gitignored).

**NEXT: run the screens on this population** — the owner's ROI screen, the 2-D screen, and
P1-style validation — reporting the raw sample statistic **and** the population-reweighted
one (`pool_census.parquet` carries the inclusion probabilities). This is the first time
those screens face a real-world sample that is both deep and unbiased.

### How it was built (design record — read before touching it)

`src/realworld_deepen.py` (commit 8601e09), `scripts/run_realworld_deepen.sh`, 18 tests
(579 green). Sample **drawn and frozen** at seed 20260727; the fetch is running detached
(`~/.pmrun/rw_deepen.log`). Read `DECISIONS.md` → *"Real-world deepening is a PROBABILITY
SAMPLE, not a shortlist"* before touching it. The three things that make it different
from the slow/sports arms:

- **Performance-blind selection.** The sampler sees discovery bet count and market count
  and nothing else — never edge, profit or win rate (asserted by test). So the owner's
  ROI screen and 2-D screen, when re-run on this population, are being **tested** rather
  than re-measured.
- **A probability sample with known weights**, not a top-N slice: 6,802-wallet frame,
  upper three strata censused at p=1 (397 / 342 / 1,172), the 20–49 band sampled 589 of
  4,891 at **p=0.1204**, total **2,500 drawn**. `pool_census.parquet` keeps the whole
  frame with each wallet's stratum + inclusion probability, so results reweight to the
  population (Horvitz–Thompson, pinned by test). This is the *representative census* the
  red-team audit listed as outstanding process debt.
- **383 drawn wallets are already deep** from the slow/sports arms → flagged `prior_arm`,
  reused not re-fetched, **2,117 to fetch**. The draw happens *before* the on-disk check,
  so inclusion probability is untouched.

**Disk budget, enforced in code:** stops cleanly (cursors saved, resumable) if free space
falls below **2.5 GB** or the dataset reaches **1.6 GB**. Estimate ~1.0–1.4 GB against
4.82 GB free. Trades are **append-only shards**, not one rewritten parquet — the
slow/sports rewrite pattern would have meant ~85 rewrites of a ~1 GB file on a box that
has been OOM-killed twice. Isolated to `data/interim/realworld/`; the shared ledger is
never written (asserted by test).

**Resume/inspect:** `.venv/bin/python -m src.realworld_deepen status`, and re-run
`./scripts/run_realworld_deepen.sh` — every stage is idempotent. **Do not re-run `plan`
with a different seed**: the draw is the pre-registration.

**What remains after the fetch:** resolve the markets (long, drains across re-runs), then
re-run the screens — the owner's ROI screen, the 2-D screen, and P1-style validation — on
this population, reporting both the raw sample statistic and the population-reweighted
one.

### THE TASK, as originally queued

**Deepen real-world wallets from the existing discovery corpus.** The insight that
produced it: *this repo's real-world sample is small because of how it collected data,
not because real-world traders are rare.*

- `src/ingest.py` is a **global-firehose poller**, and **82% of that firehose is 5-minute
  crypto** — so real-world traders barely register. The ledger holds only **3,192
  real-world wallets, 98 with 1000+ bets**.
- But `data/interim/discovery/discovery_trades.parquet` — built **market-first** over just
  **585 markets (0.6% of the 101,591 real-world markets in the ledger)** — already holds
  **542,397 wallets, of which 10,791 have ≥20 bets**, 982 have ≥100, 238 have ≥500.
  **3.4× more qualifying wallets than the entire ledger's real-world side**, sitting
  unused on disk.
- So step 1 is **not** more discovery. It is **deepening** wallets already discovered
  (`/trades?user=`, the `src/sports_deepen.py` / `src/slow_deepen.py` pattern), which is
  what turns a 20-bet wallet into one that can actually be validated.
- **Disk budget is the binding constraint: ~4.9 GB free, 81% used.** ~0.5 MB per deepened
  wallet (293 sports wallets → 2.02M trades → 163 MB), so **~2,000–3,000 wallets fits**.
  Budget it explicitly and stop before the disk gets tight. Do NOT delete the crypto
  ledger to make room — it frees only ~290 MB and would destroy reproducibility of the
  certified 26 (84% of their bets are crypto). The space is elsewhere on the box and
  is not the pipeline's to reclaim.
- Then re-run the screens on that population — including the owner's ROI screen and 2-D
  screen, which have only ever been tested on a thin real-world slice.

### What was settled this session (all committed and pushed)

| finding | commit |
|---|---|
| Red-team audit reopened the sports arm; a mis-scoped 10¢ floor had killed a real 12-wallet cohort → frozen as `s_sig2c` | 7d008b8, af78eea |
| Certified set's true FDR measured for the first time: **80.6% / 55%**, worse than the stale "~50%" | 6997562 |
| α tightened 0.05 → **0.005** (scoped to P1): certified set **41 → 26**, FDR **55% → 20%** | 156105c |
| Metric B: latency copying **NO-GO**; selection copying untestable on this data | 8aaa7cd |
| **Polymarket mechanics ground truth** — `docs/polymarket_mechanics.md` | 363f6df |
| Screening surface: the owner's ROI screen and 2-D screen **both fail**; the one screen that "worked" was a crypto-frequency artifact | 0642b98, 00c5b5d |

**One correction with teeth: "micro-crypto is un-copyable" is a claim about human
latency, not about the data.** Something polling every few seconds could act inside a
5-minute market, so the objection is narrower than it had been written. Crypto was
dropped anyway — a latency race against professional bots plus a prior negative test —
but **the deep-ledger micro copyability question is genuinely untested** (the only test
was 2026-07-18 on 20,305 bets, 0.6% of today's 3.3M) if anyone wants to revisit it.

### What is running unattended

- **Forward scorers** — `forward_score.sh`, every 2 days: slow forecasters, sports,
  `s_sig2c`. Read headline tier + CI/p, never the raw early numbers.

### The through-line, stated plainly

Every method tried — copying trades, copying timing, screening on past returns, screening
on win rate, identifying mispricing directly — returns the same shape: **whatever skill
is there sits where an outside observer cannot reach it**, either inside 5-minute markets
that resolve before a human could act or in the choice of market rather than the moment
of entry. The **real-world deepening above is the first idea in a long time whose
objection is "we have not collected the data yet" rather than "we tested it and it
failed."** That is why it is queued.

## Modeling defaults chosen, and their tradeoffs

All live in `config/config.yaml`. Current values and *why*, so you can revisit them
deliberately rather than cargo-culting:

| Default | Value | Rationale | Tradeoff / risk |
|---|---|---|---|
| `oos_split` | 0.5 | Even split maximizes power in the held-out half; simple and unbiased. | Wallets with few bets get tiny halves → noisy or NaN out-of-sample edge. |
| `min_sample_size` | 30 | Shrinkage scale (`k`); below this, edge is pulled hard toward 0. | Genuinely sharp low-volume wallets are under-credited (accepted: better than crowning noise). |
| `MIN_BETS_PER_HALF` (`validate.py`) | see source | A half with too few bets returns NaN edge, so it can't be a candidate. | Excludes short-history wallets from "persisted" — conservative, intended. |
| `copy_window_hours` | 24 | Forward window to estimate fair value after entry. | Too long → captures post-resolution drift / unrelated news; too short → noisy fair-value estimate. |
| `earliness_window_hours` | 6 | Shorter horizon isolates the "was price already moving their way" signal. | Sensitive to thin-liquidity price noise in the first hours. |
| `time_consistency_buckets` | 4 | Chronological buckets to check edge isn't one spike. | Coarse for short histories; a wallet with <4 buckets' worth of bets is barely tested. |
| Ranking weights | `w_edge=1.0`, `w_copy_window=0.5`, `w_earliness=0.25`, `w_time_consistency=0.1` | Edge dominates; copy window is the second-most followable signal; earliness and consistency are tie-breakers. | Hand-set, not fit to any objective. No ground-truth "future returns" target exists yet to tune against. |
| `breadth_full_credit` | 10 | Distinct markets at which breadth down-weight reaches full credit. | A wallet sharp in one narrow category is penalized even if legitimately specialized. |
| `non_persisted_penalty` | 0.3 | Reliability multiplier when edge didn't persist — never 0, so the wallet stays ranked. | A soft penalty; a non-persisting wallet can still out-rank a persisting one on raw edge. |
| `manufactured_flag_counterparty_share` / `_min_markets` | 0.4 / 5 | ≥40% notional vs one counterparty across ≥5 markets flags likely self-dealing. | Heuristic; both false-positives (market makers) and false-negatives (multi-wallet rings) are possible — hence flag-only. |

**Meta-tradeoff:** the whole score is a hand-weighted linear combination validated
by a sign test. It is defensible and interpretable, but it has **not** been
calibrated against any out-of-time measure of "did following this wallet actually
pay." Treat the exact ordering as a hypothesis, not a result.

---

## Open Questions (start here next session)

### 🔴 READ FIRST — independent RED-TEAM AUDIT of the REAL-WORLD COPY ARM (2026-07-29)

> **`docs/redteam_realworld_copy_2026-07-29.md`.** A fresh adversarial session
> re-derived every load-bearing number in the copy arm from primary data, and built
> the null the analysis never had (`scripts/audit_copy_null.py`, committed — the
> headline "11 of 37" previously had **no committed script at all**).
>
> **Bottom line: the POOLED copy edge is real and stronger than claimed; the
> PER-WALLET ranking is not supported.**
>
> - **Survives everything:** pooled follower edge **+1.98¢ [+1.16, +2.82]** at a
>   2-minute lag; **+2.42¢** on held-out bets only (the copy sim had been scoring the
>   in-sample half the wallets were *selected* on — checked, and it goes the
>   favourable way); **+2.25¢** with the baseline refit to exclude these wallets
>   (the "circularity" is conservative, as in the 2026-07-26 audit's B6); **+1.13¢**
>   at 1.5× the spread proxy; and **+1.67¢ [+1.45, +1.91]** over an *executable*
>   post-signal random anchor. **First copy thesis in this repo to survive a
>   wallet-agnostic placebo.** It also kills "selection not timing" for this cohort:
>   taking their market/side at an arbitrary later moment earns +0.31¢ [−0.44, +1.16].
> - **THE CLEAN TIMING TEST (AMENDMENT B, `scripts/audit_copy_timing.py`): PASSES.**
>   Matching the control on *relative position in the market's life* (donor position
>   from another bet by the same wallet — same market, same side, same typical
>   earliness) removes the confound. **real − donor = +0.83¢ [+0.47, +1.17],
>   p<0.0001**; +0.72¢ at a wider window. So ~half the +1.67¢ was position and half is
>   genuine moment-specific information. Decomposition on the paired subset:
>   **~0.8¢ moment-specific, ~2.4¢ market/side + typical stage.** Withdrawn caveat:
>   the "earlier is better gradient" does NOT exist in residual units (edge by
>   position decile is noisy and non-monotone) — the pre/post-entry gaps are local
>   drift, not lifecycle.
> - **Fails:** *which* wallets. Under a **cell permutation that preserves market
>   clustering**, per-wallet edge dispersion is 3.65¢ real vs 3.22¢ null,
>   **P = 0.230** — the copyable/uncopyable split is a random hand of markets. Only
>   **3** wallets are copyable in both halves of their own history; the count moves
>   11↔12 on RNG seed alone. ⇒ the per-wallet copyable/uncopyable label is not a
>   finding; only the pooled number is.
> - **WRONG:** "4 of 72 *uncertified* are copyable, vs 30% for certified" — 9 of the
>   72 ARE certified and 2 of the 4 copyable ones are certified; like-for-like the
>   uncertified score *better* (+1.47¢ vs +1.35¢).
> - **OVERSTATED:** "78% edge retained" is survivor-conditioned (all 37: **49%**;
>   bet-weighted **39%**). **FDR 17.3%** is the unconditional
>   number and is right, but the D≥1 criterion used to prefer it is mis-scoped (its
>   source scopes design effects to *per-wallet* claims); B's message — 40 of 58
>   certifications survive with market footprint held fixed — was never carried into
>   the copy conclusion.
> - **AMENDMENT 1 is legitimate:** all 16 pre-amendment observations came from
>   wallets it DEMOTED (11 from `0x1ee9a5fc09`), none from the five promoted. But the
>   manifest contradicts itself — `hypothesis.H4` and `strata.note` still describe the
>   superseded two-wallet `rw2_copyable` as THE HEADLINE, `amendments: []` is empty,
>   and `scripts/crontab.example` still says `rw2_copyable`. Fix the strings only.
> - **Verdict: a pooled copy edge is measurable in this cohort; the claim that
>   particular wallets are the copyable ones is not.** Everything downstream of the
>   per-wallet selection in this arm should be read as unsupported.

### 🔴 READ FIRST — independent RED-TEAM AUDIT (2026-07-26, commit 7d008b8) reopened sports

> **`docs/redteam_audit_2026-07-26.md`.** A fresh adversarial session re-derived every
> load-bearing number in this repo from primary data, mandated to assume the conclusions
> were wrong in *both* directions. **Its bottom line: "there is no actionable edge" is
> ~80% justified and 20% premature, and the premature part is one concrete, cheap,
> currently-testable thing.** Read it before acting on any NO-GO below — several of
> those notes say "do NOT re-attempt", and that instruction is what fossilised the error.
>
> **The finding, independently re-derived by the coordinator and reproducing exactly:**
> the sports arm's `scoring.sports.min_skill_edge = 0.10` floor was **mis-transposed**
> from the small-n slow-forecaster arm onto a large-n universe (1.1M bets / 40,256
> events). `config/config.yaml:58` documents it in exactly those words — *"same scoped
> rationale as the slow arm's"* — which is the bug in the config's own handwriting.
> Project 1's logic puts a large-n universe at **2¢**, so the floor is wrong by 5×.
> Worse, applied to *held-out* edge it is a **winner's-curse selector**: at fixed true
> skill it picks the most upward-noisy record. It did exactly that.
>
> | | certified `s_core` survivor | the excluded cohort |
> |---|---|---|
> | in-sample → held-out skill | **+0.1¢ → +18.8¢** (167×) | stationary, in ≈ out |
> | held-out events | 182 | up to **947** |
> | BH-FDR across all 174 candidates | **fails** | **6 of 12 pass** |
>
> **12 wallets clear every other gate** (candidacy, event-clustered bootstrap
> significance, market count, event-unit concentration) at ≥2¢ — and had **no tier that
> could read them**: `s_wide` is 10¢-gated, `s_all` dilutes them among 162 unvetted
> names. One of them, `0xdbdd45…`, was **independently in Project 1's certified 41**
> (then rank 23) — cross-confirmation across two separate pipelines and datasets.
> ⚠️ **2026-07-26:** its Project 1 cluster p is 0.024, so tightening α to 0.005 demoted it
> (now rank 65, un-flagged but fully in the dataset). The cross-confirmation is weaker than
> it read: both arms saw it at p<0.05, neither at p<0.005.
>
> **The justification for reopening is floor-independent**, which is what makes this a
> unit correction rather than threshold-hunting: among the 174 candidates, **37 have
> event-clustered p<0.05 against ~8.7 expected** under a global null and **26 have
> p<0.01 against ~1.7**; BH across all 174 yields **27 survivors at q=0.10**. The
> validation doc's claim that the survivor "would not survive" 174-wide FDR
> mis-applied BH (it checked only the rank-1 threshold; BH is a step-up procedure) —
> notably an error that made the evidence look *weaker* than it is.
>
> **The audit's other verdicts, in brief.** SOUND: the slow 52→46→30→12 chain (and its
> cut cohorts were retained as t30/t52 forward tiers, so nothing was destroyed);
> event-unit clustering; the nulls' design-effect discipline; the FLB census selection
> diagnosis ("the single best analysis in the repo"); the insurance-blindness caveat.
> **Project 1's 41 verified sound** (the certified set is **26** since α was tightened on
> 2026-07-26; the audit's verification stands for the wallets it covered) — funnel, edges
> and p's reproduce, 0/41 flip across
> 5 alternative bootstrap seeds, parameter-stable, and the residualization is
> **conservative**: the baseline is fit on a tape that is 98.1% deepened/selected
> wallets, so refitting on the organic tail *raises* the 41's mean residual from
> +3.19¢ to +4.28¢. It is **understating** certified skill by ~1¢.
> **OVERSTATED, not wrong:** "cannot distinguish from fair" hardened into "fair-priced /
> fully closed" (p=0.33–0.47 on 4 tail firings is a *no-power* test); one 22-set
> point-in-time snapshot became "arb dead"; the copy-decay placebo tests *timing* but
> the headline extends it to *selection*, which it cannot test. **Stale number:** the
> "~50% FDR" on the 41 was measured against the **pre-hardening** gate (118 vs null 59).
> ✅ **Re-measured 2026-07-26** (`docs/persistence_fdr_hardened.md`): the prediction that
> it would come back better was **wrong** — like-for-like it is **80.6%**, and **55.0%**
> under a stricter null. "~50%" was an under-statement, not an upper bound. **Process gap:** the final three
> verdicts (weather gate, category sweep, live arb probe) have **no committed scripts**
> anywhere — unreproducible coordinator one-shots in a repo whose every other verdict
> ships a script.

### ✅ ACTED ON — `s_sig2c`, a second pre-registered sports arm (2026-07-26, commit af78eea)

> **`src/sports_sig2c.py`, `python -m src.sports_sig2c {preview,freeze,score}`.**
> Frozen **2026-07-26T06:03:11Z with `forward_observations_at_freeze = 0`.**
> Artifacts: `sports_sig2c_freeze_manifest.json`, `sports_sig2c_frozen_set.parquet`,
> `sports_sig2c_forward_scoreboard.{md,parquet}`. Now on the 2-day cron alongside the
> other two scorers. 403 tests green (13 new).
>
> **The parent `sports_freeze_manifest.json` is NOT amended** — it has forward
> observations against it and is correctly immutable. This is a separate artifact with
> its own freeze timestamp, exactly as the sports freeze was legitimately created beside
> the forecaster freeze. Verified byte-identical (md5) across the freeze. The parent's
> **baseline is reused verbatim**, never refit, so the two arms stay comparable and no
> forward number can drift.
>
> | tier | n | rule |
> |---|---:|---|
> | **`s2_core`** (headline) | **10** | ≥2¢ + p<0.05 under **all five** saved baseline variants |
> | `s2_twelve` | 12 | the audit's cohort — standard baseline only |
> | `s2_slow` | 3 | ≤20 held-out bets/day — **the only copy-relevant tier** |
>
> **New evidence produced here, not in the audit — baseline robustness.** The confound
> that cut the forecasters 52→46 was baseline granularity, and the parent arm only ever
> ran that check for its *single* survivor. Run across all 12 and all five saved
> variants (league-only/20, league|form/20 and /40, sub-league|form/20 and /40):
> **10 of 12 hold ≥2¢ AND p<0.05 in every variant** (`0xdbdd45…` moves 0.0545–0.0561
> with p=0.0005 in all five). Two are fragile — `0x90448cec…` (p→0.057) and
> `0x09fe78c8…` (edge→1.7¢, p→0.070) — and are demoted out of the headline. So the
> headline tier is a **stricter** standard than the parent arm applied, not a looser one.
>
> **`s2_slow` is the tier that matters for the copy thesis** and it is only three
> wallets: `0x7e3a1f95…` (3.1 held-out bets/day, 107-day span, +5.7¢, eff. breadth 83),
> `0x90448cec…` (5.8/day, +2.5¢, but baseline-fragile), `0x97df146f…` (7.3/day, 130-day
> span, +2.8¢, eff. breadth 39). Held-out cadence splits the twelve across a **~7× gap
> with nothing inside it** (3.1/5.8/7.3 then 53/54/126/217/225/324/337/1185/1745), so
> any threshold in that gap selects the same three. Everything else in the cohort is
> in-play high-frequency and is *not* copyable — which is the correct, narrower version
> of the "sports = uncopyable in-play HFT" closure.
>
> ⚠️ **Honesty conditions, recorded in the manifest's own `known_limits`:** the 2¢ floor
> was chosen *after* the parent table existed. The mitigations — 2¢ is the repo's
> pre-existing global floor, the counterfactual is smooth (12/4/1/1 at 2/5/10/15¢, no
> cliff), and the population excess is floor-independent — are stated, not assumed.
> **The forward number is the arbiter; the retrospective number is not evidence for it.**
> Also on record: the parent's aggregate `s_all` forward reading had been seen before
> this freeze, which is *why* the freeze timestamp is now rather than the parent's, and
> why **no per-wallet decomposition of that aggregate was computed** — membership must
> not be conditionable on forward outcomes.

### 📊 First real forward reading — sports, 2026-07-26 (commit af78eea)

> `s_core` **0 events**, `s_wide` **0 events** — the pre-registered headline is
> **unreadable**, and no conclusion about the frozen sports thesis is available yet.
> `s_all` (174 wallets, explicitly unvetted, "expected to be mostly noise"):
> 7,604 bets / 38 wallets / 58 events / **18.8 effective events** →
> **+3.13¢ event-weighted, 95% CI [−0.38¢, +6.61¢], one-sided p=0.038.**
>
> **Read that as a faint positive lean, not a result:** the CI includes zero, it is the
> least-vetted tier, and 18.8 effective events is thin. The alert in `forward_score.sh`
> fires on *any* resolved post-freeze bet in *any* tier — always read the headline
> tier's events and CI before concluding anything.

### 🟠 Metric B / copyability of `s2_slow` — LATENCY copying NO-GO, SELECTION copying still open (2026-07-26, commit 8aaa7cd)

> **`docs/project3_sports_metric_b.md`**; reproduce with
> `flock data/interim/.analysis.lock -c 'PYTHONPATH=. .venv/bin/python scripts/audit_sports_copyability.py'`
> (`--no-ledger --no-live` for the offline part). 445 tests green.
>
> **NO-GO for latency copying** — "see one of these wallets enter, then enter yourself."
> No measurable advantage at any latency from 30 s to 3 days. **Every positive follower
> number is matched or beaten by a wallet-agnostic placebo:** at Δ=1h the follower earns
> +2.8¢ bet-weighted / +4.4¢ event-weighted (CI [+1.6, +4.1]) — but `real − random` =
> **−0.42¢** and `real − pre-entry` = **−0.34¢**, and on discordant pairs the pre-entry
> anchor beats copying by **−6.6¢ / −4.5¢ / −2.1¢**. Same result as every prior copy
> thesis in this repo, now confirmed on the one cohort slow enough to have had a chance.
>
> **INCONCLUSIVE — deliberately not rounded to a NO-GO — for SELECTION copying:** "take
> their market and side, refuse the chase." It is **not tested and cannot be tested on
> this data**, because the placebo that kills the chase *is itself a selection strategy*.
> This is the same gap the audit named in verdict A4: the placebo attributes **timing**,
> not **selection**.
>
> ⚠️ **A correction to the brief, and it matters beyond this task.** The worker was told
> "coverage is the single most important number." **It is not** — it is the most important
> number about *our tape*, not about the world. Coverage is 30.4% pooled (9.5% / 65.6% /
> 12.9% per wallet), but the live book probe found **continuous two-sided liquidity,
> median 1.0¢ spread and median 4,126 shares at best ask** in exactly these markets. A
> real follower is not restricted to moments when another *tracked* wallet printed, so
> low coverage is a **measurement artifact of a selected-wallet backfill and a weak basis
> for a NO-GO**. What carries the verdict is the **placebo comparison**, which is
> tape-limited identically for real and placebo anchors and therefore is not confounded
> by the sparsity. **Apply this correction to the earlier edge-decay conclusions too** —
> the "only 25% of real-world bets are copyable at all" line has been doing more work in
> this repo's reasoning than it can bear.
>
> **Other findings worth keeping:** the decay curve is *flat* (77–90% retention from 30 s
> to 3 days), which is the tell that the cohort is being measured rather than the wallet's
> timing; at realistic bounded fill windows coverage never exceeds 4.2%, so the bounded
> placebo that killed the last copy thesis **is not runnable here** (0–9 paired bets) and
> had to be moved onto a next-print anchor; the pre-entry placebo degenerates on a sparse
> slow tape (identical fill in 41–99% of pairs) — the discordant-pairs restriction is the
> fix. `0x90448cec…` dominates every pooled statistic and enters at a mean price of
> **0.959** (insurance-shaped, negative copy window) — read the per-wallet tables, never
> the pooled row. The whole testable cohort is worth **~5 effective resolution events**.
>
> **One positive cell exists and is NOT believed:** `0x97df…` at Δ=24h, `real − random` =
> +0.0081 [+0.001, +0.017]. Its **design effect is 0.17** — the documented over-constraint
> pathology — it is 1 of 12 uncorrected cells, and it does not replicate at that wallet's
> other three latencies. Logged, not believed.
>
> **What this leaves live.** The forward scoreboard (`s_sig2c`) still grades
> *identification* and is unaffected. The open strategy is the audit's "take the
> selection, refuse the chase" — enter the wallet's market/side on a resting limit rather
> than chasing the print. Note the honest complication before anyone gets excited: random
> -anchor entry in these wallets' chosen markets is itself **positive** (~+3.2¢), so if
> that survives, the next question is whether the signal needs the *wallet* at all or
> lives in the *market* — which is this repo's own "not followable alpha" test, one level
> up. If it lives in the market, it is still tradeable, but you do not need to copy anyone.

### 🧭 COORDINATOR SYNTHESIS (2026-07-26) — two independent runs converged on the same thing

> Neither worker could see this, because each saw only its own task. Metric B
> (`docs/project3_sports_metric_b.md`, sports, 3 slow wallets) and the FDR re-run
> (`docs/persistence_fdr_hardened.md`, Project 1, the micro-crypto 41) used **different
> methods on different cohorts in different market classes** and landed on the same
> conclusion:
>
> **The edge is in WHICH MARKETS these wallets were in, not in HOW they traded inside
> them.**
>
> - **Metric B:** copying the wallet's *timing* adds nothing — every follower number is
>   matched or beaten by a wallet-agnostic anchor. But entering the wallet's chosen
>   market/side at a **random** moment is itself **positive (~+3.2¢)**. The value is in
>   the market/side pair, not the moment of entry.
> - **FDR:** under null B — which holds each wallet's **market footprint fixed** and
>   permutes only who traded inside it — tightening α does **not** reduce the FDR at all
>   (81→67→70→83%), whereas under B2, which also randomizes footprint, it falls cleanly
>   (55→32→23→10%). A null that reproduces the signal when it preserves footprint is
>   saying the footprint *is* the signal.
>
> - **THIRD, INDEPENDENT CONFIRMATION (α re-run, 156105c/15f27d1):** tightening α from
>   0.05 → 0.005 drops the strict-null FDR 55% → 20%, but under the **footprint-preserving**
>   null the FDR does **not** fall at any threshold (81→67→70→70→83%). **No significance
>   threshold can fix this** — thresholds test *whether* an edge is real, not *where it
>   lives*. Three different runs, three different methods, same answer.
>
> **Why this matters more than either result alone.** It reframes the project's central
> question. "Can I copy a sharp trader?" is answered — no, repeatedly, at every latency
> and in every cohort. The live question is now **"can I identify the mispriced markets
> they were in, prospectively?"** That is a different problem with a different failure
> mode, and this repo has never posed it directly.
>
> ⚠️ **The catch, and it is the repo's own standard.** If an edge is structural in a
> market then *everyone* in that market earns it, which fails the "not followable alpha"
> test recorded in the favorite-longshot section — a copier captures it by trading the
> market directly, no wallet required. **But picking those markets is still a skill**, and
> "the wallets are a market-discovery signal" is a genuinely different, untested thesis
> from "the wallets are traders worth copying". Before anyone invests in it, note that
> Project 4 already tested trading structural mispricing *directly* and returned NO-GO —
> the unexplored middle is whether these wallets' **footprints** locate mispricing that a
> census-level scan misses. Treat as a hypothesis with one supporting observation from
> each of two runs, not as a finding.

### 📋 The re-open list (audit §2E, ranked by expected value) — #1 is DONE, #2 is next

| # | Item | Cost | Status |
|---|---|---|---|
| 1 | Pre-register the sports 2¢ cohort as its own frozen tier | ~1 h | ✅ **DONE** — `s_sig2c`, commit af78eea |
| 2 | **Metric B / copyability for `s2_slow`** | ~1 d | ✅ **DONE** — commit 8aaa7cd. **NO-GO for latency copying; INCONCLUSIVE (open) for selection copying.** See below |
| 3 | Re-run the cluster null through the **hardened** P1 gate → true FDR of the 41 | ~1 d | ✅ **DONE 2026-07-26** — `scripts/audit_persistence_fdr.py`, `docs/persistence_fdr_hardened.md`. Came back **UNfavourably**: 80.6% like-for-like / 55.0% strict, vs the stale 50%. Spawned item #9, now also done: α tightened to 0.005, set 41 → 26 @ ~20% FDR |
| 4 | **Representative (random condition-id) census** ingest → venue-wide FLB/value read | 1–2 nights | Open. Converts "closed on top-volume markets" into "closed", or finds the tail |
| 5 | News-lag with a free external clock (GDELT/RSS) — "blocked" was a choice, not a hard block | days | Open, genuinely unexplored |
| 6 | Weather specialist market-first ingest | a night | Low for copying (liquidity wall stands); mild for value |
| 7 | SELL-side / exit-copying (SELLs are ingested but never scored) | days | Open, acknowledged scope cut |
| 8 | Standing arb monitor for transients; the 2 tail-lens wallets (ranks 497/320) | small | Low each, listed for completeness |
| 9 | **Tighten `scoring.oos_significance_alpha`** (pipeline commit + test + re-rank) | ~half a day | ✅ **DONE 2026-07-26.** Shipped as a SCOPED key, `scoring.project1.oos_significance_alpha = 0.005` (`validate.certification_alpha`); the global stays 0.05 for the slow/sports arms, whose frozen forward tests depend on it. Certified set **41 → 26** (24 copyable), re-verified at a measured **20% FDR**. See "Tightened significance threshold" in DECISIONS.md and the certified-set section below |

**Process debt the audit flagged, worth clearing:** the weather gate, the category sweep
and the live arb probe have **no committed scripts or artifacts** — they are
coordinator-chat one-shots in a repo whose every other verdict ships a reproduction
recipe. Any of those three verdicts is currently unreproducible.

### 🛑 Project 4 — does the structural mispricing exist at all? = NO (2026-07-26, commit 6f7d602)

Full write-up `docs/project4_value_gate.md`. After the copy thesis was exhausted, this
asked a prior question: forget the wallets — is the favorite-longshot mispricing this repo
had been quoting actually present in the market census? **No.**
(1) The favorite-longshot "edge" in the mid-bands is largely **our own wallet
selection** — on the unselected market census it's ~0/negative (0.6–0.8 band = −3.9¢);
only heavy favorites (0.90–1.00) carry a real census tilt. (2) That surviving
[0.98,1.00) edge is an **insurance payoff** — collect small premiums until a rare tail
fires; under a proper fair-price null p=0.18–0.47 (not significant), and a bootstrap can't
resample a tail that never fired.

> ★ **METHODOLOGICAL CAVEAT FOR THE WHOLE REPO:** split-half persistence / event bootstrap
> is **blind to insurance-shaped payoffs** — both halves under-observe the rare tail, so
> their agreement is automatic (the [0.98,1.00) band passed OOS split-half yet is
> fair-priced). Every persisted result here rests on split-half. So **any wallet/strategy
> whose edge concentrates in extreme-favorite (or extreme-longshot-short) bands must be
> re-read as possible untriggered tail risk**, including forward-test survivors — validate
> those with a fair-price null (one draw per market at its price), never split-half alone.

> 🔴 **AUDIT CORRECTIONS (2026-07-26) — the core is sound, the wording overreached.**
> The census diagnosis reproduces to four decimals and is called out as the best
> analysis in the repo. But: (i) the insurance-band kill rests on **p=0.33–0.47 over
> 179 events with 4 tail firings**, which is a *no-power* test — "cannot distinguish
> from fair" was hardened into "fair-priced / FULLY closed"; (ii) the census markets
> have **median $7.1M volume (min $51k)** and **zero weather rows**, so "no structural
> mispricing" is demonstrated only on the venue's most liquid core, not venue-wide;
> (iii) the arb NO-GO is **one point-in-time snapshot** of 22 sets — a reasonable
> prior-confirming spot check, not "arb dead". None of this reverses the NO-GO; it
> lowers the confidence from *established absent* to *not established present*.
> Also a process gap: **no committed script exists** for the live arb probe.

**Cross-market internal consistency — measured live 2026-07-26** (read-only CLOB `/book`
probe). Across 22 complete exhaustive outcome sets (12 binary + 10 neg-risk, 5–28 legs),
Σ(best ask) ≥ 1.00 on 21 of 22 (binary median 1.0015) — i.e. the book is internally
consistent, and the retrospective "sums ≠ 1" readings are **confirmed to be artifacts** of
incomplete volume-selected subsets. The venue is competed to efficiency on this axis, as
the prior predicted. News-lag (#3) remains untested; it needs an external news clock.
**Net: the search for a reachable mispricing, by wallet or by market, is mapped and comes
back empty.**

### ❄️ Project 3 — FROZEN 2026-07-25 for the forward test (commit 3123d9c). Retrospective work is DONE.

> **READ `docs/project3_freeze.md` FIRST.** Everything below is the history it
> supersedes. There is nothing further to re-analyse retrospectively on Project 3;
> the next real information arrives on **calendar time**.
>
> **The standard pipeline now uses the EVENT COMPLEX as its cluster unit** and reads
> the concentration floors in complex units alongside the market-unit ones (no gate
> removed, nothing dropped). Thresholds are transposed UNCHANGED — correcting the
> unit, not re-tuning the number; ≥2 would have given 19 instead of 12, which is
> the "relax until the table fills" failure. `python -m src.slow_validate` →
> **12 persisted**. Progression: 52 (5c) → 46 (finer baseline) → 30 (complex
> clusters) → **12** (complex-unit gates). Median top-complex share 0.61 → 0.37.
> Single-ladder names fall out BY RULE (one cluster → NaN p → not significant).
>
> **Narrative honesty check.** Above the complex sits a frozen narrative map
> (reported strict AND broad, since crude spikes on Hormuz risk but also trades on
> OPEC/demand). PRIMARY (12): 4 dominant narratives, **3.13 effective independent**,
> median top-narrative share 0.51, 1 single-narrative. SECONDARY (30): 8 dominant
> but only **2.96 effective**, median 0.73, **13 of 30 single-narrative**. Both
> cohorts carry ~the same ~3 effective narratives — the wider set is bigger without
> being broader, which independently vindicates headlining the 12. Metadata + flag
> only; never gates.
>
> **FROZEN: 2026-07-25T06:32:34Z, commit dcae30d5, partition v1.0.0.**
> `data/processed/slow_frozen_set.parquet` + `slow_freeze_manifest.json` (git-tracked)
> carry both cohorts, every gate parameter, the verbatim scoring rule, the fitted
> pre-freeze baseline, and the known limits. **primary (12) is the pre-registered
> headline**; secondary (30, inclusive) is captured so its data isn't wasted but
> re-admits the concentration the gates removed. **Cohorts are NOT re-frozen in
> light of forward outcomes** — `freeze` refuses to overwrite without `--force`.
>
> **WIDENED 2026-07-25 (step 8, commit 5abd782 — `docs/project3_forward_tiers.md`).**
> The forward test now runs over **five pre-registered tiers** sharing the SAME
> freeze cutoff, scored as CROWDS in aggregate (3 bets can't grade a wallet; 3×N
> can grade a cohort). **t12 remains the headline** and is never displaced by a
> wider tier's forward number.
>
> | tier | n | eff. narratives | pre-freeze bets |
> |---|---:|---:|---:|
> | **t12** (headline) | 12 | **3.13** | 3,514 |
> | t30 | 30 | 2.96 | 14,027 |
> | t52 | 52 | 2.24 | 15,542 |
> | t87 (no significance test) | 87 | 2.68 | 26,215 |
> | t699 (least vetted, volume) | 699 | 3.47 | 686,940 |
>
> **Two measured facts, recorded before any forward data existed.** (a) The tiers
> are NOT a nested ladder — legacy t699⊃t87⊃t52 and standard t30⊃t12 use different
> baselines, so t30 and t12 are not subsets of t52. (b) **Widening buys VOLUME, not
> narrative breadth**: t699 has 58× the wallets and 195× the bets of t12 for only
> 3.47 vs 3.13 effective narratives, and t87/t52 are actually NARROWER than t12.
> Forward bet counts will grow far faster than forward evidence — every bet count
> in the scoreboard is printed next to `effective_events` for that reason.
>
> **Statistic:** event-weighted mean (mean over event complexes of the within-complex
> mean), bootstrap resampling COMPLEXES not bets, undefined below 2 events. Tests
> pin that 100× the bets in the SAME events does not narrow the CI. Plus a
> resolution-speed split (≤14d vs longer) and a **wallet-agnostic placebo**: the
> same statistic over non-cohort wallets in the same markets, and a count-matched
> random-crowd percentile. Beating the baseline is the weak test; beating the room
> is the real one.
>
> **Operating from here:** `python -m src.slow_forward score` — weekly is ample.
> Scoreboard: `data/processed/slow_forward_scoreboard.md` (+ .parquet), live from
> day one so "awaiting data" is a visible state. Read-only, public GETs,
> no keys.
>
> **Status:** 699/699 wallets fetched, 647 post-freeze trades captured, **0 resolved
> slow bets so far**. `--amend` now REFUSES (647 observations exist), so the tier
> widening is permanently on record as having happened at 0 observations.
> **Out of scope, deliberately:** deepening new wallets. Scaling the crowd is a
> separate track gated on the early aggregate read.
>
> **Accepted limits, on the record BEFORE any forward data exists:** ~3 effective
> narratives means far fewer independent pieces of evidence than n suggests; the
> primary is mideast-weighted (5 of 12), so a quiet Mideast may leave it with
> nothing to score for months (this is why the secondary is frozen too); slow
> markets take weeks to resolve; ~50% aggregate FDR still applies to individual
> names; and this is identification only — **copyability (metric B) remains
> untested**.

### ⛔ Forecaster BROADENING is a NO-GO — do not re-attempt (coordinator, 2026-07-25)

> The obvious way to grow the forward test is "find more forecasters". **It is
> dead, and the reason is structural, not a data gap.** Beyond
> `mideast_escalation` and sports, the forecaster universe on this venue is a
> handful of *correlated ladders* — the same rolling-deadline generators re-keyed
> ("US strikes Iran by Feb 28 / Mar 1 / Mar 15 …"), which is why every widened
> tier came out NARROWER in effective narratives than the 12 it was meant to
> broaden (t87 → 2.68, t52 → 2.24, vs t12's 3.13, on 7× and 4× the wallets).
> Adding wallets adds bets to the same ~3 events. **The forecaster universe is
> narrow by nature; broadening the forecaster set cannot fix it.** Sports is a
> genuinely different population (thousands of independent games), which is why
> the arm below was opened instead. If a future session feels the urge to widen
> the forecaster crowd again: the measurement was already done, it came back
> negative, and the answer is a different population, not more of this one.

### 🟢 Project 3 SPORTS arm — gate re-run at the corrected unit, deepening under way (2026-07-25)

> 🔴 **PARTIALLY SUPERSEDED 2026-07-26 — do not read this section alone.** The
> "174 candidates → 4 → **1 persisted**" funnel below is an artifact of a
> **mis-transposed 10¢ magnitude floor**; 12 wallets clear every *other* gate at ≥2¢
> and are now frozen as the separate `s_sig2c` arm. Its caveat (c) — "sports = the
> least copyable kind of edge" — was drawn from the *one wallet the floor let through*,
> which is in-play HFT at 337 held-out bets/day; three of the excluded twelve trade at
> **3–7 bets/day**. Caveat (b) ("would not survive 174-wide BH-FDR") **mis-applies BH**:
> the correct step-up gives **27 survivors across all 174**, which this survivor fails
> and 6 of the excluded 12 pass. The GO, the event-unit correction and the population
> statistics in this section all stand — the *per-wallet funnel outcome* does not.
> See the RED-TEAM AUDIT and `s_sig2c` sections at the top of Open Questions.

> **Read `docs/project3_sports_event_unit.md`.** The gate opened this arm on 8.4×
> excess skill dispersion and ρ=+0.48 split-half persistence, "cluster-robust at
> GAME level, 284 independent games".
>
> **The 284 markets are ~25 events.** 117 are season outrights (54 NBA team
> markets settling on ONE Finals, 40 MLB on ONE World Series, 20 NHL on ONE
> Stanley Cup), 42 are season props, and 83 are derivative lines on THREE matches.
> 39% of findable wallets sit in one event, 81% in ≤2. So `market_id` is the wrong
> cluster unit here — and `niche_l1` (league) is wrong the other way, because the
> ledger holds **25,390 distinct coded games across 193 league codes** and
> league-clustering would destroy exactly that breadth. `src/sports_events.py`
> supplies the unit in between, the **resolution event**, from slug syntax only.
>
> **Re-run at the corrected unit — the population claim SURVIVES.** 200
> cluster-preserving shuffles: dispersion excess **8.32×** (vs 7.29× at market
> level), split-half ρ +0.493 against a null that centres at **−0.004** instead of
> +0.072. Correcting the unit removed a small upward bias in the *null*, not the
> signal. Design effects 1.80 / 2.46 median — the unit does real work and does not
> reproduce the over-constraint trap (DE 0.17) that made permute-within-market
> unusable in the black-swan tail.
>
> **What the unit error DOES invalidate is per-wallet inference** — a bootstrap
> over "20 markets" that are one championship, and concentration gates counting
> those 20 as breadth. That is exactly where validation happens, so the correction
> is load-bearing for the validation and not for the GO.
>
> **New statistic, and the one to carry forward: BETWEEN-event persistence** —
> events split, not bets, since split-half cannot separate skill from one lucky
> event when 81% of wallets span ≤2. **ρ=+0.180 over 548 wallets** vs a null of
> −0.044±0.048, p=0.0196 (floor). Positive and null-decisive, but far weaker than
> the +0.493 headline: most of that headline is *within*-event.
>
> **Deepening (293 wallets, three arms).** Two arms reproduce the gate's shortlist
> almost exactly (core ≥10¢: 26 vs its 29; default ≥2¢: 136 vs its 135). The third
> is new and is the point: 212 wallets spanning ≥3 events with positive
> *event-weighted* skill — the only arm that cannot be one lucky tournament.
> Isolated to `data/interim/sports/`; the shared ledger is never opened for writing.
>
> **Validation** (`python -m src.sports_validate`) clusters EVENTS, gates
> concentration in event units, and is scoped to `scoring.sports.*` — the global 2¢
> floor is untouched, proved by a test that runs Project 1's
> `compute_oos_validation` with and without an extreme sports block and asserts the
> frames are bit-identical. Speed buckets are deliberately not applied: fast
> resolution is the feature here, and it is why **this arm's forward test reads in
> days-to-weeks rather than the forecasters' months**.
>
> **VALIDATED + FROZEN 2026-07-25T23:25:48Z (commit 8ce1f0b) — read
> `docs/project3_sports_validation.md`.** 293 wallets deepened (2.02M trades);
> the deep sports universe is **1,135,069 resolved BUY bets across 105,331
> markets → 40,256 RESOLUTION EVENTS**, against ~25 in the entire screen corpus.
> That gap is the arm's whole justification.
>
> **174 candidates → 4 clearing magnitude+breadth → 1 persisted.** Median
> candidate held-out edge +0.0051: the shortlist was overwhelmingly selection
> noise, exactly what the firewall exists to reveal. The survivor
> `0x8ade1d07fd…` holds **+0.1876 over 3,962 held-out bets in 182 independent
> events** (effective breadth 25.2, top event 11%, cluster-p 0.0205), spread
> across LoL/NBA/Dota2/NHL rather than one league.
>
> **The mandatory finer-baseline re-check PASSES.** Adding the line SIDE and
> doubling the price bins moves its edge by **<0.0003** — three orders of
> magnitude below the edge itself, versus ~1¢ and 6 of 52 killed on the
> forecaster side. The COARSE league-only baseline yields **zero** survivors, so
> the edge is masked by a coarse fit rather than manufactured by one.
>
> **Frozen as a new pre-registered tier** (own artifact — the forecaster freeze
> has 647 observations against it and rightly refuses amendment): s_core 1 wallet
> / 335 pre-freeze events, s_wide 4 / 614, s_all 174 / **31,651 events across 212
> leagues**. Where the 699-wallet slow tier carries ~3.5 effective narratives.
> `python -m src.sports_forward score` — **expect a readable sample in DAYS**.
>
> **⚠️ Three caveats recorded BEFORE any forward data (docs §6):** (a) the
> survivor's edge is entirely in a **13-day** held-out window (in-sample
> +0.0011), so "persists" means two weeks, not months; (b) BH-FDR ran over the 4
> economically-gated candidates, and across all 174 it would **not** survive —
> the selection burden is 174-wide and it sits near the line; (c) ~300 bets/day,
> mid-price, 11 leagues at once looks like **high-frequency in-play trading** —
> real skill perhaps, but the least copyable kind, since a copier arrives after
> the price has moved.
>
> **Next after this arm: metric B / copyability for sports.** It is the priority,
> and sports is the one place it can be answered quickly — and caveat (c) makes
> it the question that decides whether this survivor means anything.

### 🟡 Project 3 slow-market pivot — baseline confound CLOSED (2026-07-25, commit 0301985); a BIGGER one found underneath

> **STEP 6 RE-CHECK RESULT — read `docs/project3_finer_baseline.md` first; the
> section below is the 5c state it revises.**
>
> **The coarse-baseline confound is REAL but SMALL — the re-check PASSES.** Refitting
> at niche granularity (frozen syntax-derived family × form, hierarchical partial
> pooling, leave-one-wallet-out) costs the 52 a **median 1.0¢** of held-out edge and
> leaves **46 persisted** (43 of the original 52 + 3 new). The `+0.40…+0.54` band
> survives **8/8**. The direct copier test — each niche's mean residual over **all**
> participants, read off the unselected discovery corpus — puts **every** niche within
> **±3.5¢** (`geo_iran` **+1.0¢** over 128,433 bets / 46,783 wallets). Median
> niche-explained fraction **+6.6%**; 45 of 52 still beat their own niche by ≥10¢.
> Niche miscalibration does **not** explain the 5c positive.
>
> **But a larger, previously unmeasured confound surfaced: rolling-deadline ladders.**
> `geo_iran` is **11.5% of the whole slow universe** and takes the form "US strikes
> Iran by Jan 31 / Feb 27 / Feb 28 / Mar 1 / Mar 15 / Mar 31 / Jun 30…" — dozens of
> distinct markets with distinct resolution days but **one underlying event**. Every
> existing guard (`eff_breadth` up to 60, `entry_days`, the §3.3 resolution-day guard,
> the market-block bootstrap) counts them as independent. Resampling **event
> complexes** instead costs **18 of the 52** — three times the damage the baseline
> question did. **5 of the top 8 by edge, including the +0.54 name, have their entire
> held-out record inside one complex** (complex-bootstrap p = NaN: one event, one
> observation). Under both controls, **30 survive** (23 of the original 52 + 7).
>
> **So: metric B / the forward freeze are still NOT earned — but for a new reason.**
> The defensible freeze set is **30, not 52**. Recommended before step 7–8: adopt the
> event-complex block as the slow path's standard cluster unit, and re-read the
> concentration gates in complex units (`eff_breadth ≥ 3` over markets ≠ over
> complexes). Also: the brief's premise that the CLOB capture persisted Gamma `tags`
> is **false** — the sidecar column is filled for **16 of 348,657** markets; the
> partition is slug-syntax-derived instead (and better, since it encodes bet *form*).
>
> Reproduce: `python -m src.slow_validate --baseline niche_form --cluster niche_l1`
> and `python scripts/audit_slow_niche.py --level niche_l1 --save`. Project 1
> re-verified untouched (zero diff in validate/features/rank/report/ingest/backfill/
> forecaster_metrics/config; `data/processed` unmodified); 288 tests green.

> Full write-up of the 5c state: `docs/project3_slow_verdict.md`. Reproduce:
> `python scripts/resolve_slow_deep.py && python -m src.slow_validate --fdr-q 0.10`.
> Steps 1→2→4→5 of `docs/project3_slow_markets.md` are built; Project 1 verified
> bit-identical, 253 tests green.

**What the chain returned.** Deepening the broad discovery population out of the
adversely-selected micro ledger (the thing §4 stage 1 said a flat gradient could not
rule out) and validating on fresh deep data: on 1,397,864 slow-universe bets / 1,295
wallets / 55,160 markets, **699 candidates → 87 clear the 10¢+markets+concentration
gates → 52 `edge_persisted` / 55 BH-FDR survivors @ q=0.10**, held-out skill edges
**+0.14 to +0.54**, cluster-bootstrap p 0.0005–0.043, eff_breadth up to 60. Aggregate
edge is **horizon-robust** (+0.121 all / +0.127 at H≥6h / +0.124 at H≥24h; median entry
5.5 days out) — so it is NOT near-resolution scalping. This is the strongest-*controlled*
result the repo has produced (disjoint screen/validate data is a firewall stronger than
Project 1's half-split), and it clears every control that killed the prior five theses.

**Why it is NOT closed — the one confound the whole control battery misses (coordinator
assessment, elevate this above the verdict doc's footnote framing).** Skill edge =
`outcome − E[outcome | price, category]`, and **67% of slow bets classify as `"other"`**.
If a *sub-niche* inside that coarse bucket is structurally mispriced, **every** buyer in
it shows a large positive "skill edge" whether skilled or not — and that artifact
**survives all five controls**: disjoint data (the mispricing is in both halves), the
cluster null (genuinely different markets/resolution events), the concentration guards
(genuinely broad — 60 markets), and the horizon check (miscalibration is horizon-flat.)
This is **the favorite-longshot problem one level up** — the exact structural edge
skill-edge exists to strip, but at the *category* level, left un-stripped by a coarse
baseline. A +54¢ held-out edge is far more plausibly niche miscalibration than genuine
54¢/bet forecasting. **Copy implication:** if the edge lives in the *niche* not the
*wallet*, a copier captures it by trading the niche directly regardless of whom they
follow — i.e. it fails the repo's own "not followable alpha" test (see the
favorite-longshot section below). Not yet established either way.

**Correction to the verdict doc:** it claims "the forward test separates skill from
persistent miscalibration." **It does not** — a persistently mispriced niche pays the
wallet *and* a copier going forward, so the forward test reads positive either way. The
forward test separates real-persistent from luck; only a **finer baseline** separates
wallet-skill from niche-mispricing.

**Recommended next step — BEFORE metric B / the forward freeze (steps 7–8).** Refit the
price baseline at **sub-category or category×price** granularity and re-validate the 52.
Cheap, read-only, decisive: survive it → the result is dramatically stronger and steps
7–8 are earned; collapse → it was category miscalibration, caught cheaply instead of
after building copyability machinery on a mirage (the §1.5 mistake). Other honest limits
from the verdict doc still bind: identification-not-copyability (metric B untested),
retrospective-not-forward, ~50% aggregate FDR (individual names provisional), 86% of the
shortlist deepened.

> ✅ **DONE 2026-07-25 (commit 0301985).** The refit was run and the confound did
> **not** survive contact with the data — it is worth ~1¢ of a 10–54¢ edge. See the
> step-6 block at the top of this section and `docs/project3_finer_baseline.md`. The
> replacement blocker is event-complex clustering, not baseline granularity. The
> other honest limits listed above all still bind unchanged.

**Coordinator open flags carried from the worker (small):** (1) §2.2 sort-for-pruning
vs §5.2 P1-bit-identity conflict — resolved by *not* sorting the canonical ledger (now
in DECISIONS.md); (2) baseline scope (slow-only vs all-real-world-in-category) is
pluggable — the finer-baseline re-check above is where this gets decided; (3)
resolution-day guard uses a last-trade proxy, real fix is Gamma `closedTime` (§3.3); (4)
the H≥6h gate isn't formally in `slow_validate` (post-hoc it doesn't move the number —
add for completeness).

### ✅ RE-CHECKED 2026-07-23 — the persisted set SURVIVES a cluster-aware null (but is ~50% noise)
### ⚠️ ITS FDR NUMBER WAS RE-MEASURED 2026-07-26 — see the certified-set section below and `docs/persistence_fdr_hardened.md`. Everything in THIS block describes the pre-hardening 118, not the live set (41 at α=0.05, **26** since α was tightened to 0.005).

> `scripts/audit_persistence_cluster.py` (read-only) · full write-up in
> `docs/persistence_cluster_recheck.md`. This closes the last unchecked positive
> result flagged after the bet-level null was found too permissive — and unlike the
> black-swan tail finding (withdrawn, see below), **this one holds.**
>
> - **Aggregate: real 118 persisted of 691 candidates (17.1%) vs a cluster-preserving
>   null mean of 59.0 (8.1%), max 81 over 200 shuffles → P(null≥real)=0.000.** The
>   cluster null raises the chance floor from the historic ~5% (bet-level null
>   reproduced here at 5.4%) to 8.1%, but the signal clears it decisively.
> - **…with an implied ~50% FDR** *(for the 118 — the pre-hardening set)*. The null
>   manufactures 59 of the 118 by chance. **Superseded for the live 41: re-measured
>   2026-07-26 at 80.6% under this same bracket / 55.0% under a stricter one** — see
>   `docs/persistence_fdr_hardened.md`.
> - **Per wallet: 118 → 83 survive a market-block bootstrap → 42 have ≥30 held-out
>   markets (the defensible set) → 16 on a strict reading.** 48 of 118 have too few
>   held-out markets to be certified either way.
> - **`validate.py`'s t-test is miscalibrated: median design effect 1.60, 43% of
>   persisters >2, max 230.** These wallets are NOT sparse per market (median 2.63
>   held-out bets/market, max 70.4) — `high_frequency_micro_market` means many bets
>   into the *same* market, all settled by one resolution. Worst case
>   `0xcb016f2b41…`: 3,098 held-out bets in 44 markets, t-test p=2e-15 →
>   cluster-robust p=**0.31**.
> - **Two pipeline fixes fall out — ✅ BOTH SHIPPED 2026-07-23** (in `src/validate.py`,
>   27 tests green, DECISIONS.md "Cluster-count gate"):
>   1. **Stable split** — `split_in_sample_out_of_sample` now sorts with
>      `kind="mergesort"` (was default unstable quicksort; **400 wallets** had halves
>      decided by sort internals, 1 flipping `edge_persisted`).
>   2. **Cluster-count floor** — `edge_persisted` now also requires
>      `out_of_sample_markets ≥ scoring.min_oos_markets` (default 30); new columns
>      `edge_markets_ok` + `out_of_sample_markets`. On today's ledger the persisted set
>      goes **118 → 70** (the 48 with <30 held-out markets removed). This was backlog
>      #4's guard, now non-hypothetical (thin wallets had entered the set).
>   3. **Cluster-robust significance ✅ SHIPPED 2026-07-23** (the follow-up, now done):
>      `edge_significant` is gated by a market-block bootstrap
>      (`validate._cluster_bootstrap_p`, `scoring.oos_bootstrap_resamples` default 2000),
>      not the per-bet t-test. New column `out_of_sample_cluster_p` (the gate); the
>      t-test `out_of_sample_residual_p` is still reported for contrast. Deterministic
>      (per-wallet seed). This closes the 70→42 gap: **live persisted set is now 41**,
>      matching the audit's defensible set. Metric D still uses the t-test (additive,
>      never certifies). DECISIONS.md "Cluster-robust significance".
>   - NB: the documented "37 persisted / min oos n=103" baseline elsewhere in this file
>     is stale — the live funnel is now 691 candidates → 146 cluster-significant → **41
>     persisted** (all ≥30 held-out markets, cluster p < 0.05).
> - The 0.02 magnitude floor is weaker than it reads: 118/118 clear it on the point
>   estimate, only 52 at the bootstrap's 5th percentile.
> - ⚠️ **SUPERSEDED 2026-07-26:** the funnel above ends at 41 because α was 0.05. Project 1's
>   certification α is now the scoped `scoring.project1.oos_significance_alpha = 0.005`, and the
>   live funnel is **691 candidates → 98 cluster-significant → 26 persisted**.

### 📋 The final certified set — 26 wallets (2026-07-26, α tightened to 0.005)

**This set was 41 until 2026-07-26.** It is now 26, deliberately: `docs/persistence_fdr_hardened.md`
measured the 41's false-discovery rate at **55%** (strict null) and traced it to the gate's own
α=0.05 applied across a **691-candidate** search. Project 1's certification threshold was
tightened to a **scoped** `scoring.project1.oos_significance_alpha = 0.005` (the global key stays
0.05 for the slow/sports arms, whose forward tests are frozen against it), and the measured FDR at
the new operating point is **20.1% ± 0.6%** (re-measured, 200 shuffles) — the same set-level
claim at roughly a third of the error rate. Under the *within-market* bracket the FDR ratio
stays ~70%, but that bracket's P(null≥real) improved from a marginal 0.045 to 0.000. See DECISIONS.md "Tightened significance threshold" for why 0.005 and not 0.001. Nothing was
removed from the dataset: all 23,956 wallets keep their rows, metrics and rank positions.

This is the repo's current certified output: `edge_persisted == True` in
`data/processed/ranked_wallets.parquet`, i.e. positive in-sample skill edge AND
**cluster-robust** held-out significance (`out_of_sample_cluster_p < 0.005`) AND held-out
skill edge ≥ 0.02 AND ≥ 30 distinct held-out markets. Regenerate with
`python -m src.validate && python -m src.rank`. It moves as the ledger grows — re-derive,
don't treat this list as frozen.

> ✅ **AUDIT-VERIFIED 2026-07-26** (on the 41 this set is drawn from). Independently
> reimplemented: the funnel (23,956 wallets → 691 candidates → 41 at α=0.05 → **26** at
> α=0.005), every wallet's edge and cluster-p reproduce; **0 of 41 flipped significance
> across 5 alternative bootstrap seeds**; the set is parameter-stable (markets floor
> 20/30/50/100 → 43/41/40/36; magnitude 1/2/3/5¢ → 63/41/23/16 — a smooth gradient, no
> cliff). The insurance caveat does **not** bite it (45% of held-out entries mid-book, only
> 7% above 0.98). And the residualization is **conservative, not generous**:
> `fit_price_baseline` is fit on a tape that is **98.1% deepened (sharpness-selected)
> wallets**, so E[outcome|price] sits above true market calibration — refit on the
> un-deepened organic tail, the mean residual **rises +3.19¢ → +4.28¢**. The engine is
> understating certified skill by ~1¢.
>
> ✅ **THE FDR IS MEASURED — and it is what drove the α change.**
> `scripts/audit_persistence_fdr.py`, write-up `docs/persistence_fdr_hardened.md`. 200
> shuffles, `--boot 2000` in **both** arms, with the harness first asserted equal to the
> shipped `wallet_validated.parquet` (0 mismatches on all four flags).
> - **At the old α=0.05 (41 wallets):** implied FDR **80.6% ± 0.7%** like-for-like with the
>   2026-07-23 within-market bracket (null mean 33.06, max 48, P(null≥real)=0.045), and
>   **55.0% ± 0.8%** under the stricter cell-permutation null (null 22.55, P=0.000).
> - **At the shipped α=0.005 (26 wallets) — RE-MEASURED after the change landed, 200
>   shuffles per bracket:** strict cell-permutation null **5.21 ± 0.16** → measured FDR
>   **20.1% ± 0.6%**, P(null≥26)=0.000; within-market bracket 18.27 ± 0.19 → **70.3%**,
>   P=0.000. **The bracket that was marginal is no longer marginal:** at α=0.05 the
>   within-market null exceeded the real count in 4.5% of shuffles (max 48 > 41); at
>   α=0.005 it never reaches it (max 25 < 26). Tightening did not improve *that* bracket's
>   FDR ratio (70%, as the curve predicted) but it did make the count decisive under both
>   nulls. A null replicate's persisters now contain on average 2.22 of the certified 26
>   (B) / 0.20 (B2).
>   The full curve: 0.05 → 41 @ 55%; 0.0171 (BH q=0.10) → 36 @ 32%; 0.0073 (BH q=0.05) →
>   28 @ 23%; **0.005 → 26 @ 20%**; 0.001 → 17 @ 10%. Under the strict null the FDR falls
>   monotonically as α tightens — the gate discards far more noise than signal at every
>   step. Under the *within-market* bracket it does **not** fall (81→67→70→83%), because
>   that null holds each wallet's market footprint fixed and so contains a real component:
>   the cleanest evidence in the repo that a large part of this set's edge is *which markets
>   it was in* rather than *how it traded inside them*.
> - **Not knife-edge:** the 78 candidates with a defined cluster p have a gap between
>   0.0035 and 0.0060, so the certified set is **26 for any α in (0.0035, 0.0060)** — the
>   count does not turn on the third decimal (α=0.003 → 25, 0.0073 → 28, 0.01 → 31).
> - Every FDR here assumes π₀=1 (nobody has skill), so all of them are **upper bounds**.
> - **Not a freeze artifact** (checked): under the within-market null the certified wallets
>   keep a median of only ~9–18% of their own bets, and a null replicate's persisters
>   contain on average ~3.6 of the real ones.
> - Stability: at 50,000 resamples the shipped 2,000-resample p's move only 0.00250 →
>   0.00220 (median). The 2,000 setting is not flattering anyone — but it *is* why α was
>   not taken to 0.001: below that only two attainable p-values exist at 2,000 resamples.

**Exactly what moved (diffed artifact-to-artifact, old vs new `ranked_wallets.parquet`):**
23,956 rows before and after with an identical wallet set; `out_of_sample_residual_edge`,
`out_of_sample_cluster_p`, `copy_window`, `in_sample_residual_edge`,
`out_of_sample_residual_p`, `edge_magnitude_ok`, `edge_markets_ok`,
`manufactured_record_flag`, `pattern_flag`, `regime_flag`, `regime_watch` and
`persisted_recent` are **bit-identical**; `edge_persisted` moved 41 → 26; and **exactly 15
wallets' scores changed** — the 15 demoted ones, via the `non_persisted_penalty`
multiplier. No other wallet's *score* moved; 376 wallets' rank *numbers* shift only because
the 15 slid down past them (max displacement 16 places for anyone else).

**Set-level:** 26 persisted, **24 copyable** (positive copy-window); the other 2 realize
edge at/near resolution (unfollowable). Copy-actionable `(edge_persisted OR
persisted_recent) AND copyable` = 27. Held-out markets: min 59, median 1,990, max 4,750.
Held-out skill edge: min +0.020, median +0.034, max +0.169. Held-out n: min 63, median
4,854. Cluster p: all ≤ 0.0035. Flags: 23 `high_frequency_micro_market`, 3
`late_concentrated_entry`, **0 manufactured-record**. Metric D on the same run:
`regime_watch` 53, `persisted_recent` 5 — both **unchanged** by the α move (metric D
deliberately stays on the global 0.05; it is additive and never certifies).

**Caveats that still bind:** (1) ~1 in 5 of these 26 is still what chance manufactures at
this α, so quote it as "26 certified, ~5 of them chance" — tighter than the old "41
certified, ~23 of them chance", but not zero; (2) 23/26 are micro-crypto specialists, so
this is a micro-regime set, not a slow-market one; (3) the within-market null says a large
share of the edge is market *selection*, not within-market trading — which a copier
inherits only if they can enter the same markets; (4) certified ≠ copyable — the
copy-window and edge-decay measurements below are what speak to that, and they are
negative.

**The per-wallet table is deliberately not published here.** The certified set is a list
of identifiable public addresses carrying this project's own descriptive pattern flags,
and naming them adds nothing to the method. Regenerate it locally — `python -m src.validate
&& python -m src.rank`, then filter `edge_persisted` in `data/processed/ranked_wallets.parquet`
— and read `report.md` for the per-wallet detail. Rank order there is the overall `score`,
which also weights copy-window and breadth, so two non-copyable persisters sit higher than
copyable ones.

**The 15 wallets the tightening demoted — NOT judged fake.** These cleared every gate at
α=0.05 and their cluster p now sits in [0.005, 0.05). They are the wallets whose held-out
edge can no longer be distinguished from chance *given how wide the search was* (691
candidates); the strict null says roughly two thirds of a set this size at α=0.05 is noise,
but it cannot say which. Per CLAUDE.md they remain fully ingested, scored and ranked, with
every metric intact — they simply carry the `non_persisted_penalty` reliability multiplier
and are recoverable at any time by sorting on `out_of_sample_cluster_p`. If the ledger
deepens they can re-certify on their own evidence.

### ✅ RESOLVED — the persistence audit was run (2026-07-18). The number is mostly an artifact.

> Reproduce with `PYTHONPATH=. python scripts/audit_persistence.py` (read-only
> against the ledger). Numbers below are from the 7,014-wallet ledger on disk;
> the earlier 4,629-wallet run (299/407 ≈ 73%) shows the identical phenomenon.

**Verdict: it is NOT classic look-ahead leakage. It is the second branch —
the validation is far weaker than it looks — and the ranking as it stands mostly
measures "habitually buys favorites," not forecasting skill.**

What the audit found:

1. **Real persistence ≈ 68.8%** (575/836 candidates on current data).
2. **Shuffled-outcomes null test (the decisive one): 61.9%.** Permuting the
   resolution labels destroys all skill, yet persistence barely moves — the real
   rate is only **~7 points above pure noise**. Whatever the validation is
   rewarding, ~90% of it survives with the outcomes randomized.
3. **Why the null floor is ~62%, not 50%:** the market has a structural buyer's
   edge — mean `resolved_value` 0.622 > mean `entry_price` 0.604, so **62.2% of
   all individual bets have positive edge by default.** A sign test ("mean edge
   > 0 in both halves") on tiny samples just re-measures that base rate.
4. **Favorite-longshot bias is the engine of it.** Mean edge is strongly
   price-dependent: buying at 0.6–0.8 earns **+0.139/bet**, buying at 0.2–0.4
   **loses −0.179/bet**. Habitual price preference is a persistent per-wallet
   trait, so a favorite-buyer looks "sharp" in both halves with zero skill. This
   is *not followable alpha* — a copier buying at the same 0.7 price gets the same
   structural edge regardless of which wallet it copied.
5. **Neutralize favorite-longshot** (subtract each price bucket's mean edge, so a
   wallet must beat the free edge available at the prices it trades) and real
   persistence falls to **58.9%** vs a residual null of **52.7%** → the genuine
   out-of-sample skill signal is **≈ 6 points of excess persistence**. Real skill
   exists, but it's a small minority of the "persisting" wallets, not 73% of them.
6. **The split itself is clean — not leaking.** Median share of a wallet's
   held-out markets that also appear in its in-sample half is **0%**: the
   chronological per-wallet split does not leak outcomes across the boundary, and
   `edge_persisted` never touches the forward-looking copy-window/earliness
   quantities (it uses only `resolved_value − entry_price`). So the copy-window
   look-ahead worry from the first draft is real for the *ranking score* but is
   **not** the cause of the persistence number.
7. **Samples are tiny.** Median candidate wallet: **7 resolved bets across 5
   markets**, and `MIN_BETS_PER_HALF = 2`. A 2-bet half's mean-sign is a coin flip.

### ✅ FIXED — the three prescribed changes landed the same day (2026-07-18).

The first three fixes below are **implemented and verified end-to-end** (validate
→ rank → report re-run on the real ledger; `tests/` updated, 79 passing). The
fourth (copy-window forward-price audit) remains open.

- **[DONE] Score skill (residual) edge, not raw edge.** `features.fit_price_baseline`
  fits the market-wide calibration curve `E[resolved_value | entry_price]` over
  quantile bins (`scoring.price_baseline_bins`); `residual_edge_per_bet` /
  `compute_skill_edge` subtract it. `validate.py` selects/validates on skill edge
  and `rank.py`'s `w_edge` term now uses `out_of_sample_residual_edge`. Raw edge is
  still computed and reported alongside (per CLAUDE.md).
- **[DONE] Significance test replaces the sign test.** `validate._oos_significance`
  runs a one-sided t-test (`scipy.stats.ttest_1samp`, `alternative="greater"`); a
  wallet persists only if held-out skill edge is positive **and** its p clears the
  certification alpha. Zero-variance/degenerate halves → NaN p → not significant
  (conservative). **Twice superseded since:** the gating statistic is now the
  market-block bootstrap p, not the t-test (2026-07-23), and the threshold is now
  Project 1's scoped `scoring.project1.oos_significance_alpha = 0.005`, not the
  global 0.05 (2026-07-26). The t-test still gates metric D, which is additive.
- **[DONE] `min_bets_per_half` raised to 10** (`scoring.min_bets_per_half`, was a
  hardcoded 2). Below it, edges gate to NaN and the wallet is down-weighted, never
  a candidate. ~30 total resolved bets required — consistent with `min_sample_size`.
- **[DONE] Forward-price leakage audit + fix.** The audit found `copy_window`
  correlated **~0.86 with the realized outcome** (a forward price approaches the
  0/1 answer as it nears resolution) and that the old `resolved_value` fallback
  injected the outcome directly (~2.4% of bets). Fixed in `features.forward_price`
  / `compute_forward_drift`: (1) no resolved_value fallback, (2) strictly
  window-bounded (no unbounded "nearest trade" reach), (3) a resolution guard
  (`scoring.fair_value_resolution_guard`=0.2) drops trades in the final 20% of a
  token's observed lifespan; no qualifying trade → NaN, never the outcome. Outcome
  correlation fell **0.86 → 0.56**. Separately, the audit showed `copy_window` and
  `earliness` are **numerically identical for 100% of wallets** (both forward
  windows capture the same clustered sub-6h trades), so scoring both double-counted
  one signal — `rank.py` now scores **copy_window only** (`w_earliness`=0, earliness
  still computed/reported). See DECISIONS.md "Forward-price leakage fix". Affects
  rank order, not the persistence number.

**Result after the fixes (verify with `scripts/audit_persistence.py` for the raw
diagnostic, or re-run `python -m src.validate`):**

- Out-of-sample persistence fell from **68.8% → 27.3%** (30 of 110 candidates).
- The **shuffled-outcomes null through the *new* validator persists only ~5%**
  (~6 wallets) — so the surviving signal is now **~5× the null**, versus ~1.1×
  before. The validator finally discriminates skill from the base rate.
- The 30 surviving wallets have positive, statistically-significant held-out skill
  edge over ≥10-bet halves; the ranked table is now skill-led (skill edge is the
  dominant score term), not favorite-longshot-led.

**Caveat for whoever reads this next:** 30 wallets is a small, honest set. Treat it
as a real-but-thin signal, and remember the ingest window limitation below (sample
sizes are what the poller *observed*, not full histories). Both the validation
(skill edge + significance) and the copy-window forward-price leakage are now fixed;
the main remaining correctness lever is the **unfit ranking weights** (below) — the
score combines skill edge, copy_window, and time-consistency with hand-set weights
that have never been tuned against a realized-forward-return target.

> ⚠️ **SUPERSEDED (2026-07-19).** The "30 wallets / 27.3%" numbers above were
> measured on thin, poller-observed samples. A full deep backfill has since run —
> see the new baseline immediately below. The 30-wallet set was, as suspected,
> mostly thin-sample noise: only 7 of it survived deepening.

### ✅ DONE — Top-500 deep backfill + rebuild (2026-07-20): the current baseline

Widened `backfill.top_n_wallets` **100 → 500** and looped `src/backfill.py` to
convergence over the top-500 wallets' full `/trades?user=` histories, then
features→validate→rank→report re-run. **This supersedes the 2026-07-19 top-100
baseline below.**

**Ledger now:** **3,045,424 bets, 2,990,739 resolved across 311,506 markets, 500
wallets** — every ranked wallet is now deeply sampled (the old top-100 baseline left
ranks 100–500 thin). Convergence = the never-checked backlog drained to ~5,500
genuinely-open markets (drain deltas fell to <90/run, then net-negative → open, not a
failure). Disk ~315 MB.

**New funnel baseline (over the 500 deepened wallets):**
`246 candidates (in-sample skill edge > 0) → 78 edge_significant → 59 edge_magnitude_ok
→ 36 edge_persisted (both gates) → 33 persisted ∩ copyable`. Up from the top-100
baseline's `21 significant → 14 persisted → 11 copyable`. **Forward validity tripled:
Spearman +0.105 → +0.34 (p≈0)** — deeper per-wallet samples make in-sample skill edge
predict held-out skill far better, exactly as the weight-tuning analysis predicted.

**Backlog #4 (thin-interloper handling) is resolved by this.** The old top-100 baseline
had 5 thin/breadth-1–2 "mirage" persisters (ranks 2,39,41,87,108). After deepening
ranks 100–500, **0 persisters have breadth ≤ 2, 0 have < 50 held-out bets** (min
out-of-sample n = 103, median 3,876; 29 of 36 are deep at ≥500 bets/half). The mirages
either gained real samples and stood, or collapsed out of the persisted set — no thin
interlopers remain. (The complementary idea — deepen on *reached-persisted-set* rather
than top-N-by-rank, or add an explicit breadth floor to the flag — is now moot for the
current set but kept as a guard for future cycles.) Profile unchanged: 28/36 persisters
carry `pattern_flag=high_frequency_micro_market`, 6 `late_concentrated_entry`, 0
manufactured-record flag.

**Two incidents worth knowing (both fixed):**
1. **Ledger corruption + atomicity fix.** `save_ledger` wrote parquet *in place*
   (`df.to_parquet(path)`), so a writer killed mid-write left a footer-less, unreadable
   file — which is exactly what happened when a backfill was `pkill`-ed during its save,
   destroying the only ledger copy. Recovered because `data/raw/resolutions.parquet`
   (saved *before* the ledger each run) survived with ~109k resolved markets, so the
   rebuild re-fetched only trades and reused cached resolutions. **Fix:** added
   `common.atomic_to_parquet` / `atomic_write_json` (temp file + `os.replace`); the
   ledger, resolutions cache, and both cursor writers now write atomically. A kill/
   teardown mid-write can no longer corrupt them. **Rule: never kill a writer mid-save;
   the pipeline only *reads* the ledger, so killing pipeline stages is always safe.**
2. **Parallel resolution fetch.** `ingest.update_resolutions` now fetches with a small
   thread pool (`ingest.resolution_fetch_workers`, default 5). The CLOB endpoint
   rate-limits *aggregate* throughput to ~20–22 req/s regardless of concurrency (8+
   workers just draws 429s), so 5 is polite and near-optimal (~2.5× serial), backed by
   the session's existing 429-retry. Output is byte-identical to serial (verified).

**Peak RSS on the re-rank was ~3.0 GB** (features stage) on the 3.7 GB box — it fit, but
this is the memory ceiling; the `manufactured_record_flag` tx-hash loop is the next
thing to watch/vectorize if the ledger grows much past 3M bets.

**Caveat — the ledger now holds only the 500 deepened wallets.** The corruption cost the
old ~6,500-wallet shallow tail (global-feed discoveries); it re-accrues via forward
`ingest.py` polling over the coming days. So the funnel's *candidate* denominator (246)
is not apples-to-apples with the old 7,014-wallet universe — but the *persisted set* is
directly comparable and strictly stronger (larger, deeper, no interlopers).

> **Update (2026-07-20, later — tail re-accrual started).** A clean manual
> `ingest → backfill → re-rank` ran after the OOM/cron hardening below. One ingest
> poll already re-grew the universe **500 → 2,547 wallets** (ledger 3,045,424 →
> 3,117,791 bets; backfill added 68,126 rows via the new batched path). Re-rank over
> the 2,547: **247 candidates → 78 significant → 62 magnitude-ok → 37 persisted
> (15.0%), 34 copyable.** The persisted set barely moved (36→37 / 33→34) even though
> ~2,000 shallow-tail wallets were added — the freshly-surfaced thin wallets aren't
> deep enough to persist, so no interlopers flooded in (as intended). The tail will
> keep growing as ingest runs; re-check that thin wallets stay out of the persisted
> set each cycle. Backlog #4 guards (deepen-on-reached-persisted-set / breadth floor)
> remain the fallback if a thin wallet ever persists off the shallow feed.

### ✅ DONE — Full deep backfill + re-rank (2026-07-19): the prior (top-100) baseline

`src/backfill.py` was looped to convergence over the top-100 wallets' full
`/trades?user=` histories, then features→validate→rank→report re-run. **Superseded by
the top-500 rebuild above (2026-07-20);** the numbers here are historical.

**What the backfill did to the data:**
- Ledger grew **36,456 → 832,603 bets** (807,672 resolved across 70,961 markets).
  The deepened top-100 wallets went from a median ~7 resolved bets to **~5,000
  resolved bets per out-of-sample half** (capped by the ~10k per-user offset ceiling).
- Convergence = the never-checked market backlog drained to ~0; the ~2.6k markets
  left unresolved are genuinely still-open, not a failure (the nightly cron re-checks
  them). Took ~13 drain runs (temporarily raised `max_resolution_fetches_per_run` to
  5000 for the drain, now **restored to 2000**).
- **Two `backfill.py` bugs were found and fixed en route** (both silently blocked
  convergence): (1) only markets from *this run's new trades* were queued for
  resolution, so the ~66k markets pulled in the first deep run were orphaned —
  pulled but never resolved, and never re-queued on later (incremental) runs;
  (2) a `408 Request Timeout` at deep offsets (~9000) made `fetch_user_trades` raise,
  and the caller discarded the whole wallet, so 7 high-volume wallets got **zero**
  backfill and re-triggered the 408 every run. Now: all ledger markets are re-queued
  each run (drains deterministically at the per-run cap), and a 408 keeps the
  already-fetched newest ~9k trades (which is what matters — pages are newest-first).

**The re-rank result (this is the honest signal):**
> ⚠️ **UPDATED (2026-07-19, later same day).** The "21 persist" below is the
> *significance-only* count. An economic-magnitude gate (`min_skill_edge`=0.02) and
> a `copyable` flag were added afterward (see "Backlog" items 2–3, now checked off):
> **`edge_persisted` now requires significance AND ≥2¢ magnitude → 14 persist, 11 of
> them copyable.** The 21 is still the right number for "significant at α=0.05"
> (now the `edge_significant` column).
- **21 wallets persist** out-of-sample (significant at α=0.05), out of 104 with
  positive in-sample skill edge (20.2% survival). Was a provisional 30.
- **Only 7 of the old provisional top-30 survived.** For the 25 old wallets now
  deepened to ≥500 bets/half, **median held-out skill edge collapsed 0.093 → 0.0007**
  — their apparent edge was almost entirely thin-sample noise. Textbook cases: old #8
  went **0.457 edge on 10 bets → 0.004 on 5,015 bets**; old #3 went **0.161 on 14
  bets → −0.056 on 243 bets** (now rank 6,979).
- **16 of the 21 survivors are deep (≥500 bets/half) and genuinely stationary:**
  median in-sample skill edge **0.0230** vs out-of-sample **0.0233**, median |decay|
  **0.009**, p-values down to **5.8e-26**. This is the anti-self-deception validation
  finally working at scale — small (~1–7¢/bet) but rock-solid edges.
- **New #1 = old #10** (`0x0e7bcc59…`): +0.066 skill edge on 4,390 bets/half, p=5.8e-26.
- **Profile:** 18 of the 21 carry `pattern_flag=high_frequency_micro_market` — the
  wallets that survive rigorous validation are high-frequency micro-market traders.
  (Descriptive metadata only; never touches rank.) 0 carry the manufactured-record flag.

**⚠️ Two issues this surfaced — read before trusting the top of the table blindly:**

1. **Thin-sample interlopers persist because backfill only deepens the *current*
   top-100.** 5 of the 21 persisted wallets still have <500 bets/half (ranks 2, 39,
   41, 87, 108) — they surfaced from the shallow global feed and have **not** been
   deepened. Some are fragile mirages: rank 41 is 0.196 edge on **10 bets, breadth 2**;
   rank 87 is breadth **1** (a single market). These are exactly the shape that
   collapsed for the old top-30 — expect most to evaporate when the *next* backfill
   cycle deepens them (it re-selects top-100 by current rank, so they'll be included).
   **Trust the deep (n≥500) persisters; treat the thin ones as provisional.** A cleaner
   fix: deepen any wallet that reaches the persisted set, not just top-100 by rank.
2. **`copy_window` ≠ skill edge — several top persisters are NOT followable.** Ranks
   15/16/17/108 have real, significant skill edge but *negative* copy_window (rank 108:
   −0.19): their edge realizes at resolution, not within the 24h forward window, so a
   follower entering after them has no room. So the two questions are different
   questions: rank by copy_window, not skill edge, when what you are measuring is
   followability. Skill and followability are only weakly related here.

**Performance regression (logged as its own task — do this before the next big re-rank):**
the re-rank took **~32 min** because `features.compute_forward_drift` masks the full
per-token trade array once per bet — **O(bets×trades) per token**, called twice
(copy_window + earliness) — and `validate.py` (line ~147) re-runs the *entire* feature
computation, a second full quadratic pass plus a redundant recompute of the already-cached
`wallet_features.parquet`. Fine at 36k bets, painful at 832k, and paid on **every** nightly
run now. Fix: `np.searchsorted` on per-token sorted timestamps (the window+cutoff is a
contiguous slice → non-quadratic), keeping output identical (verify on a sample); and have
`validate` load the cached features instead of recomputing. Peak RSS ~2 GB — worth watching
on the Chromebook.

### Other open questions (lower priority)

- **[DONE] Ranking weights — validated against a forward target (2026-07-18).**
  Built the leakage-free forward objective (`scripts/tune_weights.py`): predict each
  wallet's **held-out skill edge** from its **first-half** features. Result: in-sample
  skill edge really does predict held-out skill (Spearman +0.105, p=0.026 pooled;
  stronger with more bets/half), so the ranking has modest-but-real forward validity —
  but **no weight scheme beats another beyond bootstrap noise** (95% CI on the
  difference spans 0). Kept the current weights (skill-dominant, within noise of
  optimal); precision-tuning ~170 noisy wallets would overfit. Added a "Forward
  validity" line to `report.py` so rank order is presented as a coarse skill filter.
  **The real ceiling is per-wallet sample size, not the weights** → the next lever is
  the ingest limitation below, not more weight tuning. See DECISIONS.md "Ranking
  weights". Re-run `tune_weights.py` as data grows; re-tune only if its CI excludes 0.
- **[DONE 2026-07-19] Per-wallet backfill — the sample-size lever.**
  `src/backfill.py` deepens the top-ranked wallets by pulling each one's own
  `/trades?user=` history (the `user=` filter works even though `after`/`before`
  don't). **The full run is done — see "Full deep backfill + re-rank" above for the
  new baseline.** The top-3 smoke test's warning held at scale: only 7 of the old
  provisional top-30 survived deepening; the rest were thin-sample noise. Runs nightly
  before the rest of `run.sh` (`backfill.*` config). See DECISIONS.md "Per-wallet
  backfill". Remaining gap: it only deepens the current top-100, so freshly-surfaced
  thin-sample wallets ride into the top ranks for one cycle (see issue #1 above).
- **Non-stationary wallets (surfaced by the backfill).** The old provisional wallet #1 was
  skill-negative in its first half and strongly skill-positive in its second (p≈0).
  The chronological select-then-validate split treats a wallet that *became* sharp
  as "not a candidate." That's conservative-correct for luck, but a wallet with a
  genuine regime change is mishandled. Worth revisiting the split design once deep
  samples are the norm. **✅ NOW DESIGNED (2026-07-20) as metric D** — deep samples are
  now the norm and 13 such became-sharp wallets (8 deep, p to ~1e-24) were measured; see
  the "Non-stationary wallets (metric D)" backlog item below and DECISIONS.md "Metric D".
- **Ingest still only discovers via a rolling window.** The `/trades` global feed
  caps at ~10k most-recent trades platform-wide (see `DECISIONS.md`); backfill fixes
  per-wallet *depth* but discovery of *new* wallets still depends on frequent polling.
  Confirm the cron cadence overlaps polls so no trades fall in the gap.
- **Counterparty/self-dealing linkage is heuristic.** The manufactured-record flag
  keys on notional share vs. a single counterparty; multi-wallet rings and legitimate
  market-makers are both mis-served. Flag-only by design, but worth revisiting the
  heuristic if it's noisy.

### ✅ DONE — Edge-left-at-detection analysis (2026-07-18)

**Script:** `scripts/audit_edge_decay.py` (read-only, writes nothing;
`PYTHONPATH=. python scripts/audit_edge_decay.py`). Follower edge =
`resolved_value − price_at(entry_ts + Δ)`, where `price_at` is the size-weighted
price of *other* wallets' trades in the bounded window `(entry+Δ, entry+Δ+bandwidth]`,
reusing `features._forward_price_arrays` (the numpy core of `forward_price`) with the
exact copy_window leakage discipline: bounded window, resolution guard, **no
`resolved_value` fallback**. Δ=0 is the wallet's own edge (the ceiling). Every decay
number is **same-cohort** (own edge recomputed on the identical bets measurable at that
Δ, so cohort selection can't masquerade as decay), with a **shuffled-outcome null**, a
**skill-neutralized** follower edge (`resolved − E[outcome|price_at]`, favorite-longshot
removed) and **bootstrap 95% CIs**. Stratified by horizon (micro crypto up/down vs.
real-world) and entry-price band.

**Verdict: for micro-markets there is ~nothing left to copy by the time a watcher could
see it; for real-world markets a thin, genuine edge survives only at sub-minute latency
and cannot be measured past ~15 min on this ledger.** Anything an observer could act on
at human latency is unmeasurable here, which is a limitation of the data as much as a
finding — see the deep-ledger re-run below, which had the coverage to settle it.

Findings (bandwidth 60s = `watch.py`'s poll interval; same-cohort):
- **Hard ceiling first.** The ledger spans only **~1.4 h** of wall-clock (rolling-window
  poll of a fast feed, see DECISIONS.md). No token has trades an hour+ after any entry,
  so **Δ ≥ 1h is 0% coverage — unmeasurable by construction, not a result.** Everything
  below is sub-15-min, and coverage already thins to ~1–6% by 15 min.
- **Micro-crypto** (20,305 bets ≈ all of the data). Own-edge ceiling only **+1.7¢/bet**.
  Just **54%** of positions have *any* follow-on other-wallet liquidity before the guard
  — ~half are un-copyable at any latency (nobody trades after them). At 30–60s the raw
  follower edge is ~**+3¢** (retain ~60%) but it **does not clear the shuffled-outcome
  base-rate floor** (excess over null ≈ 0 / slightly negative); the skill residual is at
  most ~**+2.5¢** — inside the measurement's own uncertainty. Micro edge is
  structural, or gone by the time an observer could see it.
- **Real-world / sports** (980 bets). **75%** have follow-on liquidity. At ~30s latency
  raw follower edge **+4.4¢**, skill residual **+3.1¢ (95% CI +0.6…+5.7¢, excludes 0)**,
  with the shuffled-outcome null strongly *negative* — a genuine, outcome-correlated,
  favorite-longshot-neutral edge. But it **decays into noise fast**: by ~1m the skill CI
  already spans 0 (+0.3¢ [−1.9,+2.5]). Edge concentrates in the 0.4–0.8 entry band.
- Bandwidth sensitivity (30/60/300s) and the price-band tables are in the script output;
  the qualitative story is robust to bandwidth.

Caveats / re-run conditions:
- Cohorts are small and selection-prone (real-world n in the hundreds; each larger-Δ
  cohort is a biased longer-lived subset — hence the same-cohort own-edge and the CIs).
- Run across **all** resolved bets, not just validated wallets (a decay curve needs
  volume, not per-wallet depth). **Re-run on backfilled + validated wallets** once deep
  samples exist, and after a **multi-day ingest** so the hour-to-day latencies a human
  actually operates at become measurable (0% coverage today).
- Implication: whatever is left at real detection latency is small and confined to
  real-world markets, and a 1.4 h ledger cannot resolve it. A forward measurement at
  real detection latency is the only unbiased instrument, because it is unbounded by
  the ledger's span.

### ✅ DONE — Long-latency edge-decay re-run, Δ=1h primary (2026-07-22)

**Script:** `scripts/audit_edge_decay_long.py` (read-only, writes nothing;
`PYTHONPATH=. .venv/bin/python scripts/audit_edge_decay_long.py [--verify]`). Run on
the **Chromebook**, not the VM (heavy analysis there starves the 5-min cron ingest).
Ledger: 4,695,081 trades / 4,070,091 resolved BUY bets / 742-day span — the 1.4 h
ceiling that made Δ ≥ 1h unmeasurable in the 2026-07-18 run is gone.

Same discipline as the original (same-cohort own-edge baseline, shuffled-outcome null,
skill-neutralized follower edge, bootstrap CIs, price-band strata, exact copy_window
leakage guard). Three things are new, each forced by what changed:
1. **Vectorized** follower price (composite-key + prefix-sum, the form
   `features._forward_drift_from_arrays` uses) — the original's per-bet
   `_forward_price_arrays` loop is hopeless at 4.1M bets. `--verify` checks it against
   the scalar production core on a random sample: **300 bets, 0 mismatches**.
2. **Scoped to real-world.** Micro-crypto is reported as closed, not re-litigated.
3. **Concentration guard + cohort-selection placebos** (below).

**VERDICT: there is no copyable edge at any latency — not at 1h, and not at 30s
either. Following a validated wallet is measurably WORSE than entering the same
market at an arbitrary moment. That is the central negative result of this project,
and it also withdraws the thin sub-minute edge the 2026-07-18 run reported. Only a
forward measurement could still revive the copy thesis.**

**1. The copyability ceiling collapsed on the deep ledger.** Only **25%** of the
734,208 real-world resolved BUY bets have *any* other-wallet trade after entry and
before the resolution guard (it was 75% on the tiny 1.4h ledger). **Three quarters of
real-world positions are un-copyable at any latency** — there is no later print to
enter against.

**2. The Δ=1h decay curve looks fantastic, and that is the tell.** Same-cohort,
bandwidth 60s, real-world:

| Δ | cov | n | own(coh) | follow_raw | skill_edge [iid 95% CI] | null_raw | excess | retain |
|---|---|---|---|---|---|---|---|---|
| 0 | 100% | 734,208 | +0.005 | +0.005 | +0.005 [+0.004,+0.006] | — | — | 100% |
| 30s | 6.04% | 44,328 | +0.070 | +0.060 | +0.052 [+0.048,+0.056] | +0.065 | −0.005 | 86% |
| 1m | 5.55% | 40,731 | +0.071 | +0.059 | +0.051 [+0.047,+0.055] | +0.068 | −0.009 | 83% |
| 5m | 4.42% | 32,472 | +0.089 | +0.068 | +0.060 [+0.055,+0.064] | +0.064 | +0.004 | 76% |
| 15m | 3.15% | 23,153 | +0.117 | +0.083 | +0.075 [+0.070,+0.080] | +0.055 | +0.027 | 70% |
| **1h** | **1.17%** | **8,619** | **+0.179** | **+0.129** | **+0.123 [+0.113,+0.132]** | +0.053 | +0.076 | 72% |
| 6h | 0.04% | 286 | +0.031 | −0.011 | −0.017 [−0.058,+0.024] | −0.088 | +0.076 | −36% |
| 24h | 0.01% | 103 | −0.103 | −0.113 | −0.119 [−0.173,−0.066] | −0.052 | −0.061 | 109% |

Edge **increases** with latency (+5.2¢ at 30s → +12.3¢ at 1h) and clears the shuffled
null with p=0.00. Decay curves do not go up. What actually rises is the *cohort's own
edge* (+0.5¢ full sample → +17.9¢ for the 1.17% measurable at 1h): the surviving
subset is selected on "somebody else traded this token exactly an hour later", a
condition about the market, not the wallet. Measured 1h coverage 1.17% vs the
1.33% probe upper bound — consistent (the probe counted own trades and no guard).

**3. The placebos kill it.** Identical cohort, only the entry anchor moves — and both
wallet-agnostic anchors BEAT copying the wallet. Paired on the bets where all three
anchors exist (same bets, same outcomes, so the difference is purely the anchor):

| Δ | paired n | real (entry+Δ) | random-anchor | pre-entry (−Δ) | real − random [family CI] | real − pre-entry [family CI] |
|---|---|---|---|---|---|---|
| 30s | 8,477 | +0.056 | +0.069 | +0.066 | **−0.013 [−0.021,−0.004]** | **−0.011 [−0.014,−0.007]** |
| 15m | 1,853 | +0.102 | +0.137 | +0.168 | **−0.036 [−0.060,−0.009]** | **−0.066 [−0.104,−0.028]** |
| 1h | 177 | +0.384 | +0.451 | +0.504 | −0.067 [−0.127,+0.021] | **−0.120 [−0.163,−0.054]** |

- *random-anchor* = price the same token at a uniformly random time in its observed
  pre-guard life — "buy this market whenever", not conditioned on the wallet at all.
- *pre-entry* = price at entry **minus** Δ — a follower who acted an hour *before* the
  wallet and therefore could not have been copying it.

Every paired difference is **negative**, and 5 of 6 CIs exclude 0. The wallet's entry
time carries no exploitable information at these latencies; the big numbers belong to
the mispriced-market subset, which you capture better by ignoring the wallet.

**4. Concentration guard (the World Cup trap) — applied, and it is not the main
problem here.** At Δ=1h the cohort is broader than feared: 1,561 markets / 877
slug-families / 57 decision-days, Kish effective breadth 169.9 markets / 94.4 families
/ **7.5 days**, largest family `highest-temperature-in` at 6% of bets. Cluster
bootstrap by market/family/day still excludes 0 ([+0.066,+0.189] / [+0.066,+0.199] /
[+0.057,+0.166]) and leave-one-family-out only moves +0.123 → +0.092. So the 1h result
is *not* a single-event artifact — it is a **selection** artifact, which is why the
placebos, not the concentration guard, are what caught it. The guard does bite at the
tail: **Δ=6h** (n=286, Kish 5.0 families, one family = 43% of the cohort) and **Δ=24h**
(n=103, Kish 5.1 families / 3.3 days, `fifwc-esp-arg` = 35%) are too concentrated and
too small to conclude anything from; both are negative anyway. Effective *day* breadth
(7.5 of 57) is the weakest axis even at 1h — a caveat on all three CIs.

**5. Micro-crypto: CLOSED, not re-litigated.** 3,335,883 resolved micro bets, and its
coverage decay is structural — 5-minute BTC/ETH/HYPE "up or down" markets are
*resolved* an hour after entry, so there is no tape to price a 1h follower against.
Longer ingest cannot fix that. Not analysed here by design.

**6. This revises the 2026-07-18 sub-minute finding.** That run reported a genuine
+3.1¢ skill residual at ~30s on n=980 real-world bets. On the deep ledger the
comparable number is +5.2¢ on n=44,328 — but it now **fails a control the original run
never had**: both wallet-agnostic placebos beat it (paired −1.3¢ and −1.1¢, CIs exclude
0). The old result was not wrong given its data; it was under-controlled. Treat
"thin genuine sub-minute edge" as **withdrawn**.

Implications:
- **Copying a wallet's trades: dead** on this evidence, at every latency measured.
- **The thin sub-minute edge: withdrawn.** It does not survive the placebo.
- **A forward measurement is the only instrument left that could revive the
  thesis.** It reads a real detected price at real
  latency, going forward, unbounded by the 25%-liquidity / 1%-coverage selection that
  poisons every retrospective cohort here. Do not spend more effort on retrospective
  copy-window analysis; the ledger cannot answer it.
- Caveat worth stating plainly: this ledger is a *backfill of selected wallets*, so the
  "market tape" a follower prices against is itself made of other tracked wallets'
  trades, not the full order book. That biases the follower price toward those wallets'
  entries. It is a further reason the forward test — which sees the real book — is the
  honest arbiter.

### The two immediate next tasks (each its own session)

1. **[DONE — see "Edge-left-at-detection analysis" above]** ~~Edge-left-at-detection
   analysis~~. Answered: micro ≈ 0 copyable edge even at 30s; real-world retains a thin
   (~+3¢ skill, sub-minute) edge; Δ ≥ 1h unmeasurable on the current 1.4h ledger. Re-run
   `scripts/audit_edge_decay.py` on backfilled validated wallets after a multi-day ingest.
2. **[DONE 2026-07-19] Full deep backfill.** Ran to convergence; re-ranked. The sharp
   set shrank to a trustworthy **21** (16 of them deep, stationary, real p-values on
   ~5,000 bets/half). See "Full deep backfill + re-rank" above. **New immediate tasks
   it spawned:**
   - **Fix the `compute_forward_drift` O(bets×trades) quadratic** (and validate's
     redundant recompute) before the next big re-rank — ~32 min/run today. See the
     performance-regression note above.
   - **Deepen the persisted-set interlopers.** 5 persisters are still thin (<500
     bets/half); the next nightly backfill will pick them up since they're now top-100.
     Confirm they collapse-or-survive, and consider deepening on "reached persisted set"
     rather than "top-100 by rank."
   - **Re-run `scripts/audit_edge_decay.py`** on the deepened validated wallets (the
     copyable-edge question) once a multi-day ingest gives >1.4h of forward coverage.

---

## Backlog — everything next (consolidated 2026-07-19)

Single source of truth for what's left, after the deep backfill closed Project 1.
Roughly ordered by leverage; each is its own session.

**Infrastructure (do first — cheap, unblocks the rest):**
- [x] **`searchsorted` perf fix** for `features.compute_forward_drift` — DONE
  (2026-07-19). Replaced the O(bets×trades)/token boolean-mask-per-bet loop (the
  57k tiny tokens dominated) with a fully vectorized computation: integer
  composite keys (`token*BIG+ts` for the fair-value window; `(token,wallet)*BIG+ts`
  for the excluded own-wallet trades) + prefix-sum range differences, no Python
  per-token/per-bet loop. **Both drift passes over the full ledger: ~325s → ~12s
  (~26×).** Numerically equivalent, not bit-identical (sums accumulate in
  timestamp order, not original row order → ~1e-8 diff): verified no top-137 rank
  change, all persisters identical, copy_window sign flips only at exact 0
  (~1e-16). Also: **`validate.py` no longer recomputes the whole feature set** (it
  loads the `wallet_features.parquet` features.py just wrote) → validate ~11 min →
  ~20s. Net re-rank ~32 min → ~6 min. **New bottleneck: `compute_manufactured_record_flag`
  (~5 min — a Python loop over ~800k `tx_hash` groups); vectorize it next.**
- [x] **Memory-safe backfill batching** — DONE (2026-07-20). `run_backfill` now
  fetches+folds+frees wallets in batches of `backfill.wallet_batch_size` (25) so a
  **from-empty** top-500 run can't OOM the 3.7 GB box (the old single `all_new`
  list would hold millions of raw trade dicts at once). Fold cadence (per batch,
  bounds memory) is decoupled from save cadence (`backfill.checkpoint_min_rows`,
  100k) so a caught-up nightly run still saves **once at the end** — no per-batch
  I/O regression (verified: a real run folded 68 126 rows / 20 batches, 0 mid-run
  checkpoints). See DECISIONS.md "Memory-safe wallet batching". Residual ceiling is
  the ~2 GB resident ledger itself (same as `features`), not the raw dicts.
- [x] **Cron schedule + writer lock** — DONE (2026-07-20). Split into
  `scripts/ingest.sh` (every ~5 min) and `scripts/recompute.sh` (nightly backfill +
  re-rank), both serialized on a shared `flock` (`data/interim/.ledger.lock`) so the
  two ledger writers never overlap: ingest uses `-n` and skips a tick if backfill
  holds the lock; the read-only re-rank stages run unlocked. `run.sh` routes its
  writers through the same lock. `scripts/crontab.example` documents the cadence;
  see DECISIONS.md "Scheduling". **ENABLED 2026-07-20** (`crontab scripts/crontab.example`;
  cron daemon live): ingest every 5 min, recompute nightly at 03:17, both behind the
  writer lock. **First unattended 03:17 recompute is 2026-07-21 — glance at
  `data/cron.log` that morning to confirm it ran clean.** Scheduling choice: plain cron
  is correct here — do NOT add anacron. anacron can't do sub-day intervals (so it's
  inapplicable to the 5-min ingest), and catch-up is a no-op for the rolling 10k-row feed
  (missed trades have already fallen out of the window). The recompute is idempotent and
  not time-critical, so a night skipped by a suspended box self-heals on the next run.
  ONLY if `cron.log` shows the box slept through 03:17 and a guaranteed daily refresh is
  wanted, switch *just the recompute* to a `systemd` timer with `Persistent=true` (wake-
  aware, cleaner than anacron) — not needed otherwise.
- [x] **`features`/`validate` memory optimization** — DONE (2026-07-22). Cut the
  `features` peak **5.4 GB → 2.8 GB** and compute **440 s → 105 s** at 4.7 M rows,
  all **byte-identical** (full-ledger baseline diff + sample harness + 183 tests):
  (1) `manufactured_record_flag` `duplicated(keep=False)` pre-filter — the ~5-min
  Python loop was over 4.7 M *singleton* tx_hash groups (tx_hash is unique per row),
  all skipped, so restricting to ≥2-row tx_hashes is exact and near-instant; this is
  the "vectorize manufactured_record_flag" item, resolved; (2) `load_ledger(columns=)`
  projection (drops ~480 MB of unused `outcome`/`question`/`slug`); (3) categorical
  `market_id`/`token_id`/`side` (ledger 2.0 GB → 883 MB, 3.5× faster); (4) one-pass
  `compute_forward_drift_multi` (both windows share the sort/prefix machinery). See
  DECISIONS.md "features/validate memory footprint".
- [x] **Streaming rewrite — the free e2-micro now runs the whole pipeline** — DONE
  (2026-07-22). The 2.8 GB peak above still swapped ~1.8 GB on the VM's 1 GB RAM, and
  the free tier's *standard* disk has such low IOPS that this measured **98.5 %
  I/O-wait — 58 min elapsed for 54 s of CPU** (≈2 h for `features` alone). Two
  findings drove the fix: **glibc never returns freed memory to the OS** (so freeing a
  column can't lower peak RSS — only never allocating it can), and peak is set by the
  **largest momentary allocation**. So `features` became
  `compute_wallet_features_streaming`: four column-projected passes, each freeing
  before the next; the drift pass reads Arrow straight into numpy (no DataFrames) and
  batches the bet side; `manufactured_record_flag` short-circuits by asking Arrow
  whether any tx_hash repeats at all. `ingest` became `stream_merge_ledger` (batch
  read → resolution refresh → drop superseded → temp file → atomic replace), and the
  ledger codec moved gzip → **zstd** (full rewrite 192.7 s → 20.8 s, file 395 → 354 MB).
  **On the VM: ingest 767 MB / 5.9 min, features 773 MB / 26 min, validate+rank+report
  28 min — all fitting in 1 GB RAM.** Every output byte-identical (full 4.7M-row
  baseline diff, worst float delta 0.0; 183 tests). **Cron is ENABLED on the VM**
  (ingest */5, recompute 03:17); the recompute holds the writer lock for its whole
  duration so ingest can't run concurrently and blow RAM. Chromebook cron stays off.
  See DECISIONS.md "features/validate memory footprint".
  **Next lever:** `validate` (27.5 min) has *not* had the streaming treatment — it
  still loads the ledger and runs a per-wallet Python loop; the same shape would cut
  the nightly from ~54 to ~35 min.
- [ ] **🔴 `backfill` does NOT fit the e2-micro — it starved ingest for 2.5 h (2026-07-23). VM CRON IS PAUSED.**
  The first unattended nightly (03:17 UTC) hung in `src/backfill.py`: **2 h 35 m elapsed,
  570 MB RSS, 4 GB swap in use, 91–96 % I/O-wait** — the swap-on-slow-disk death spiral
  the migration documented (the free tier's *standard* disk has too few IOPS for swap to
  be viable). It never reached `features`. Because the recompute holds the writer lock
  for its whole run, **30 consecutive ingest ticks SKIPPED and ~2.5 h of trades were
  permanently lost** — the bug just fixed, re-entering through a different door.
  **Why it was missed:** the streaming rewrite profiled `ingest` (767 MB) and `features`
  (773 MB) and flagged `validate` as next. **`backfill` was never profiled on the VM at
  all** — its 2026-07-20 memory batching was sized for the Chromebook's 3.7 GB, not 1 GB.
  **Resolution:** chain killed by PID (parents first so it could not advance a stage);
  ledger verified intact and readable at **4,952,808 rows**, 0 stray `.tmp`, delta dir
  empty, lock released, swap 4.2 GB → 84 MB. Atomic writes did their job — killing a
  writer is safe now, which is exactly what they were added for.
  **Both VM cron jobs are PAUSED** (`crontab` commented; backup at
  `~/.pmrun/crontab.backup.20260723T060025Z`) pending the Project 3 decision — see
  `docs/project3_slow_markets.md`. **Safe to pause because slow markets do not scroll:**
  the 5-min cadence, the writer lock and the delta parts all exist to service a rolling
  10k-row global window that empties every ~8 min, which is a property of *fast* markets.
  Slow markets are collected retrospectively via `/trades?market=` (`discover.py`)
  whenever we like. Before re-enabling: give `backfill` the streaming treatment, drop it
  from the nightly, or let the Project 3 rescope (~6× smaller) dissolve it.
- [x] **🔴 ingest cadence + pager — FIXED 2026-07-23** (found the same day). All three
  fixes shipped in `src/ingest.py` (+ `scripts/`): (1) **incremental delta write** —
  ingest appends new rows to `data/interim/ledger_delta/part_*.parquet` instead of
  rewriting the 4.7M-row ledger; `python -m src.ingest --fold-delta` folds them in as
  the first step of the nightly recompute, in memory-bounded passes
  (`ingest.delta_fold_rows_per_pass`, default 300k rows). (2) **pager sweeps all
  pages** and filters by timestamp, never breaking on a stale row, with within-sweep
  key dedup. (3) SKIP line names the real lock holder. Verified **byte-identical on
  the real 4,695,081-row ledger** (old inline merge vs delta round-trip + fold: same
  sha256) and 190 tests green (183 + 7 new). **Deployed + verified on the VM the same
  day: ingest 5.4 min → ~48 s, no more alternating SKIPs, and consecutive
  `data/raw/trades/batch_*.parquet` spans went from +2.5/+1.6 min gaps to +0.5 min /
  exact overlap** (a control sweep shows the feed's own quiet slots run to 52-54 s, so
  sub-minute holes are burstiness, not loss). A manual fold of 21,907 pending delta
  rows into the 4.74M-row ledger took 3m47s and landed exactly +21,907 rows. Residual
  accepted gap (late-*arriving* older rows) documented in DECISIONS.md.
  ⚠️ **The VM has no GitHub credentials** (Stage 4 item 5 of the migration was never
  done), so deploys go: `git bundle create /tmp/pm.bundle master` → `gcloud compute scp`
  → `git pull --ff-only /tmp/pm.bundle master` on the box. Its working tree also carries
  locally-regenerated `data/processed/*` — back those up before any `git checkout --`.
  <details><summary>original diagnosis</summary>

  First steady-state check after the migration. Consecutive polls **gap** instead of
  overlapping (+2.5 min and +1.6 min between the retained raw batches) → ≈20% of
  platform trades never ingested, unrecoverably. Two independent causes, full evidence
  in DECISIONS.md "`/trades` pages are NOT time-ordered":
  1. **Cadence.** Feed depth ~8.3 min, but the effective poll period is ~10 min: one
     ingest takes 5-8 min and still holds the writer lock when the next `*/5` tick
     fires, so that tick SKIPs. The 5-8 min is **not** I/O (full 4.7M-row ledger
     read+rewrite benchmarks at 27 s / 705 MB on the VM) — it is the per-batch
     resolutions merge + object-dtype `MultiIndex.isin` inside `stream_merge_ledger`,
     ×24 batches, plus ≤500 resolution GETs. **Fix:** the "v1 incremental ledger write"
     from the VM migration notes — ingest appends new rows to a small delta parquet,
     the nightly recompute folds it into the ledger. Ingest → ~1 min, so a true 5-min
     cadence sits comfortably inside the 8.3-min window.
  2. **Pager.** The feed is **not time-ordered across pages** (12/19 boundaries are
     inversions, up to ~4 min), yet `fetch_new_trades` breaks on the first trade older
     than the cursor floor — so it can stop with newer unseen trades at deeper offsets.
     Latent today (the floor is old enough that all 20 pages clear it); becomes the
     dominant leak once #1 is fixed and the floor gets recent. **Fix:** sweep all 20
     pages, filter by timestamp, never break early.
  3. Cosmetic: the SKIP line says "backfill running" when the lock holder is almost
     always the previous *ingest* — which is what made this look healthy.
  This is a ledger-writer change: verify against the existing byte-identity harness,
  and confirm afterwards by checking that consecutive `data/raw/trades/batch_*.parquet`
  spans **overlap**.
  </details>

**Project 1 — make the ranking *actionable* (the deep backfill proved these are needed):**
- [x] **Economic-magnitude gate (metric C)** — DONE (2026-07-19). `scoring.min_skill_edge`
  (default 0.02) is a magnitude floor the held-out skill edge must clear
  *independent of* significance; `validate.py` reports `edge_significant` and
  `edge_magnitude_ok` as separate columns, `edge_persisted = both`. Floor is above
  1¢ on purpose. **Effect: 21 significant → 14 persisted;** the 7 dropped are the
  sub-2¢ edges (one at p=2.3e-6 over 5,062 bets but only 0.5¢). Gates the flag/score
  only — nobody is dropped from the dataset.
- [x] **Copyability gate/score (metric B)** — DONE (2026-07-19). `rank.py` adds a
  `copyable` column = `copy_window > ranking.copyable_window_floor` (default 0.0,
  strictly positive) — a first-class sortable filter, **kept separate from skill and
  NOT fed into the score**. `report.py` surfaces it per-wallet and as an "Actionable
  set" summary. **Effect: of the 14 persisters, 11 are copyable** (all of the top 10,
  deep n≥768); 3 are sharp-but-unfollowable (rank 108 copy_window −0.19, rank 87 NaN,
  rank 35 ~0). NB: `0x32ed517a` (old "most copyable" deep persister) is now dropped by
  the *magnitude* gate (1.35¢ skill) though its copy_window is +0.12 — the two metrics
  are genuinely orthogonal, which is why they're reported separately.
- [x] **Thin-interloper handling** — DONE (2026-07-20) by widening the deep backfill to
  top-500 (`backfill.top_n_wallets` 100→500). The 5 old thin/breadth-1–2 mirages
  (ranks 2,39,41,87,108) got deepened along with all of ranks 100–500; in the new
  baseline **0 persisters have breadth ≤ 2 and 0 have < 50 held-out bets** (min oos_n
  103, median 3,876) — the mirages either gained real samples and stood or dropped out.
  See "Top-500 deep backfill + rebuild (2026-07-20)" above. The complementary guards
  (deepen on *reached-persisted-set*, or an explicit breadth floor on `edge_persisted`)
  are now moot for the current set but worth adding if a future cycle re-surfaces thin
  persisters from the shallow feed.
- [x] **Non-stationary wallets (metric D) — IMPLEMENTED 2026-07-20** (designed same
  day; shipped in `src/validate.py` + surfaced in `rank.py`/`report.py`).
  The `in_sample_residual_edge>0` candidate gate (anti-leakage — kept) silently
  excludes wallets that *became* sharp. Measured on the top-500 ledger (from
  `wallet_validated.parquet`): of 246 candidates → 36 persisted, but **13 are
  "became-sharp"** (recent half significant + ≥2¢, early residual ≤0), **8 deep
  (≥500-bet recent half, p to ~1e-24)** — genuine regime changes buried at
  `edge_persisted=False`, though they're the "sharp now" copy targets (e.g.
  `0x5d634050ad`: in-sample −0.024 → out-of-sample +0.068 over ~3,915 bets, p=9.4e-24).
  210 candidates are the symmetric "decaying" case. **You cannot just relax the gate**
  (selecting+validating on the recent half re-introduces leakage). Design = detection +
  honest routing, all additive, score untouched (full rationale in DECISIONS.md "Metric
  D"). Shipped as three additive columns on `wallet_validated.parquet`, none feeding the
  score or `edge_persisted`: **D1** `regime_flag` enum
  (improving/decaying/stable/insufficient, + `improving_confirmed` when D3 fires) —
  directional only when a Welch half-difference AND a Spearman time-trend agree at
  `scoring.regime_trend_alpha`; **D2** became-sharp → `regime_watch=True` (a sortable
  watchlist for the forward test), not persisted; **D3** within-recent-regime
  sub-split (recent half ≥ `scoring.regime_min_bets`, both recent sub-halves
  significant+material) → `persisted_recent=True` while `edge_persisted` stays False, so
  the copy-actionable set is `(edge_persisted OR persisted_recent) AND copyable`
  (surfaced in `report.py`). Config knobs `scoring.regime_min_bets` (default 50) +
  `scoring.regime_trend_alpha` (default 0.05). Tests: `tests/test_validate.py` +
  `tests/test_report.py` (hand-built fixtures, regime known by construction).
  Reproduce the evidence: `PYTHONPATH=. python scripts/audit_nonstationary.py`
  (`--deep` for trend + D3 counts).

**Project 2 — find copyable forecasters (the main product direction; spec +
Gamma-validated in `docs/project2_forecaster_discovery.md`):**
- [x] **Build `src/discover.py` + targeted real-world ingest** — DONE (7fcba83 groundwork
  + 2c2eb59). Market-first via Gamma volume-ranked enumeration + `/trades?market=`,
  `--closed-only` sweep, `--resolve` (own discovery_resolutions cache), incremental
  checkpointing.
- [x] **Live open-market capture** — DONE (2cfd035). `discover.py --live` polls open
  real-world markets to catch early entries before they scroll past the 10k cap;
  `--events` pulls the full mid-volume tape tail. Both §1.4 prongs built + tested.
- [~] **A/B/C/D metric stack, per category** — A (per-category favorite-longshot baseline
  + OOS-validated skill) and C (magnitude floor) DONE in `forecaster_metrics.py`; **metric
  B (copyability, Δ-window token arrays) DEFERRED** — its ~2.9 GB peak can't coexist with
  cron ingest on the 3.7 GB box (OOM'd twice), needs a bigger box or a no-ingest window;
  D (recency/regime) not yet ported to Project 2.
- [x] **Per-category dashboard output** — DONE (2c2eb59). `rank_forecasters.py` emits a
  sortable per-category markdown + self-contained click-to-sort HTML, ranked on copyable
  skill edge, never profit.
- [x] **Shuffled-outcome null + FDR correction (§1.5)** — DONE (2026-07-21, commit
  1c3ffa9; `scripts/audit_forecaster_null.py`, lifts `audit_blackswan.py`'s shuffle-null
  posture + `bh_reject`; 183 tests green). **VERDICT: selection noise — 0 credible
  copyable forecasters survive; winner list stays withheld; metric B NOT run** (it would
  multiply onto a mirage). The null shuffles outcomes within (category, 1¢-price) over all
  1.69M resolved BUY bets — preserves each category's favorite-longshot base rate and each
  wallet's price mix, destroys only the wallet↔outcome (skill) link; 500 shuffles re-score
  metric A against a fixed baseline (vectorized scorer reproduces the production gate
  exactly: 1,204 significant / 595 persisted). Results: **count-null estimated gate FDR
  ~40%** (595 real vs 235 null persisted; 318 vs 127 winnerA) — P(null≥real)=0.000 (the
  aggregate beats chance, but ~40% of the set is manufactured); **pooled-null BH-FDR
  q=0.05 → 6 survive, 2 clear economic gates**; analytic BH (H0 mean=0) inflates to 932.
  **Why 0 not 2:** both FDR survivors had ~50¢ "edges" (real skill is 1–7¢) — inspection
  showed both are **concentrated single-event artifacts** the exchangeable shuffle can't
  detect (one Sunday's 5 NFL games fill-inflated to n=60; 3 props of one Fed presser, n=27
  all YES). A principled **concentration guard** (`eff_breadth = 1/HHI` of per-market fills
  ≥3 AND distinct decision-days ≥3) rejects both. Bug fixed en route: degenerate
  near-zero-variance cells produced a finite-but-blown-up t (~1e16) that leaked through the
  empirical-p FDR path and polluted the pooled null — now NaN'd consistently with the
  analytic gate. Dashboard leads with a red verdict banner + a `WIN·NULL` column (0 rows);
  new §1.5 in `docs/project2_section14_findings.md`. **Residual limitation (next guard):**
  the concentration guard misses *correlated markets sharing one resolution event* in the
  general case — the discovery resolutions table has no timestamp, so resolution-time
  clustering is the next guard. **The forward test remains the real copyability
  arbiter.** Metric B (copyability) stays deferred — only matters for winners that survive
  the null, and none did.

**New metric family — black-swan / tail-edge detector (2026-07-19):**
- [ ] The current mean-edge + variance-based significance machinery **structurally
  misses longshot/tail-mispricing wallets** — their +EV lives in fat-tail variance, so
  mean edge reads ~negative and the t-test can't separate tail-skill from luck without
  enormous samples. Needs a *different* metric family: tail calibration (longshot
  hit-rate vs. implied probability), total-return / Kelly-growth, variance-aware tests.
  Least copyable of all → watchlist, not auto-copy.

  > ✅ **INVESTIGATED (2026-07-19) — `scripts/audit_blackswan.py` (read-only, writes
  > nothing; `PYTHONPATH=. python scripts/audit_blackswan.py`).** A separate lens on
  > entry≤0.15 bets: tail hit-rate vs. the market-wide base rate `E[outcome|price]`
  > (tail skill), realized ROI + Kelly log-growth `D(q‖p)` (growth, not mean), an
  > **exact Poisson-binomial** hit-rate test (the correct heterogeneous-Bernoulli null;
  > verified vs. Monte-Carlo, and it diverges ~10× from a normal/t approx in the skewed
  > tail — which is *why* the pipeline's t-test misses these), **bootstrap** CIs for the
  > fat-tailed ROI/residual, BH-FDR across wallets, and a **shuffle-within-price-bucket
  > null** as the arbiter. All leakage-free (uses only entry_price + resolved_value on
  > resolved bets; no forward price).
  >
  > ### ⛔ WITHDRAWN 2026-07-23 — re-arbitrated against a cluster-preserving null.
  > See `docs/blackswan_cluster_null.md` and `scripts/audit_blackswan_cluster.py`.
  > The finding below rests on a **bet-level** shuffled-outcome null, the one Project 3
  > stage 1 (§10.1) measured as too permissive for wallet-level statistics. Re-run on
  > today's ledger (498 tested tail wallets at thr=0.15, vs 85 then):
  > - **Winner count REVERSES.** Real 78 winners vs a cluster-preserving null mean of
  >   **86.6** — P(null ≥ real) = **0.99** (bet-level null: 18.4, p=0.000). Same
  >   direction at thr=0.10 (0.98) and 0.20 (1.00). The procedure invents *more*
  >   winners on permuted data than the real data contains.
  > - **The +0.218 persistence DIES.** Reproduced at +0.214, but the cluster-preserving
  >   null centres at **+0.244** (p=0.92) — split-half persistence is a property of
  >   which markets a wallet is in, not tail skill.
  > - **Per-wallet: 46 → 8.** Under a market-block bootstrap (the cluster-robust
  >   bracket; B is over-constrained here, design effect 0.17), 8 wallets survive
  >   BH-FDR and the §1.5 concentration guard — but **6 of the 8 are already
  >   `edge_persisted` at ranks 1/2/12/13/17/20**. The "6 wallets the pipeline misses"
  >   payoff shrinks to **2**: `0x16c91ea1c8…` (rank 497) and `0xee65685de4…` (rank 320).
  > - **Methodological, and repo-wide:** §6/§10.1's "make `permute_wallets_within_market`
  >   standard for every wallet-level statistic" needs a qualification. In the sparse
  >   tail (median 2 tail bets/market) it is *more* constrained than independence
  >   (design effect 0.17 vs 1.45 for the bootstrap) and manufactures its own per-wallet
  >   false positives. **Measure the design effect before trusting any null** — that is
  >   the rule to adopt, not one specific permutation.
  >
  > The original 2026-07-19 text is kept below for the record.
  >
  > **Finding — tail-edge wallets DO exist here, and the premise holds.** At thr=0.15,
  > 85 wallets have ≥20 tail bets; **19** clear (resid_vs_base>0 & PB p<0.05), **11**
  > survive BH-FDR — decisively above the null (mean 3.1, 95th pct 6, max 9 over 300
  > shuffles; P(null≥19)=0.000). Split-half tail residual persists out-of-sample at
  > Spearman **+0.218** (> the pipeline's +0.105 mean-edge forward validity).
  > **The payoff: 6 of the 11 FDR survivors are NOT flagged `edge_persisted` by the
  > mean-edge pipeline** — buried as deep as **rank 299** (`0xee65685de4…`: buys 1.6%-
  > base extreme longshots that hit 3.4%, **+103% ROI**, p=1.3e-6) and rank 229
  > (`0x77bcf759a5…`: n=23, hit 39% vs 9% base, **+234% ROI**). Their per-bet MEAN edge
  > is tiny, so the t-test sinks them; their tail hit-rate is significant. The other 5
  > survivors are already deep persisters (ranks 1,2,4,5,6) — re-confirmed from a
  > variance-aware angle.
  >
  > **Caveats (stay skeptical):** (1) the classic favorite-longshot bias is **largely
  > absent in this micro-crypto tail** — §0 shows it ~calibrated (small mixed-sign
  > residuals), so "beat base rate" ≈ "beat price" here; the strong-signal framing
  > (longshots overpriced) would matter more in a real-world/sports book (→ Project 2).
  > (2) The set is null-decisive, but individual small-n survivors (n=23–27) are exactly
  > where the ~3 expected false positives hide — treat them as provisional. (3) ROI can
  > be positive with a below-base hit-rate via a couple of big size-weighted wins
  > (§4, e.g. `0xe9076a87c5…` +$33k, hit-rate insignificant) — that's variance/sizing,
  > not certified tail skill. **Tail edge is the least copyable signal (payoff = a rare
  > event you can't reliably enter alongside) → watchlist, never auto-copy.**
  >
  > **Next if pursued:** fold a tail-skill column + copyability-aware watchlist into the
  > Project 2 per-category stack (a per-category FL baseline is the missing ingredient
  > to make "beat base rate" a strong signal); re-run after real-world ingest deepens
  > the non-crypto tail. Design-only for now — no pipeline edits made.

**Ongoing / data:**
- [x] **Multi-day continuous ingest** — DONE, and the "1.4h ledger" blocker is
  **obsolete**. The deep per-wallet backfills pulled full `/trades?user=` histories,
  so the ledger now holds **2,740 dense hours** (≥100 trades/hr) spanning
  2025-06-02 → 2026-07-23, including a **contiguous 1,268-hour (53-day) stretch**
  (2026-05-29 → 2026-07-21). Wall-clock span is no longer the constraint.
- [x] **🔵 Re-run `audit_edge_decay.py` — DONE 2026-07-22, see "Long-latency edge-decay
  re-run" below. Verdict: no copyable edge at ANY latency — following a wallet is
  strictly WORSE than entering the same market at an arbitrary time.** The original
  scoping notes are kept below for the record.
  Δ-coverage measured 2026-07-23 (read-only probe over 4.12M resolved BUY bets;
  "coverage" = share of bets with **any** same-token trade in `(t+Δ, t+Δ+60s]`, an
  **upper bound** — it counts the wallet's own trades and applies no resolution guard):

  | Δ | 30s | 1m | 5m | 15m | 1h | 6h | 24h |
  |---|---|---|---|---|---|---|---|
  | coverage | 57.4% | 49.3% | 11.0% | 3.50% | **1.33%** | 0.08% | 0.03% |

  The aggregate collapse looks fatal but is **misleading — the composition is what
  matters.** The surviving slice is not a thin random remnant, it is precisely the
  real-world cohort where the original audit found the only genuine edge:

  | Δ | bets | distinct tokens | real-world | micro-crypto |
  |---|---|---|---|---|
  | 1h | **54,774** | **6,671** | 53,897 (98.4%) | 877 |
  | 6h | 3,436 | 948 | 3,389 (98.6%) | 47 |
  | 24h | 1,291 | 326 | 1,291 (100%) | 0 |

  So Δ=1h went from **0% coverage (unmeasurable by construction)** to a 54,774-bet,
  6,671-token real-world sample — versus the n=980 sub-15-min cohort the original
  real-world finding (+3.1¢ skill at 30s) rested on. **Δ=1h is the run to do.** 6h is
  thinner but plausible; 24h is marginal and concentrated (top tokens are 2026 World
  Cup football markets — watch for correlated-single-event artifacts, the same trap
  that killed Project 2 §1.5).

  **Why the micro-crypto horizon question is permanently closed:** its coverage decay
  is *structural*, not a data limitation. Polymarket volume is dominated by
  5-minute-cycle BTC/ETH/HYPE "up or down" markets — an hour after entry the token is
  resolved and there is no tape to price against. Ingesting longer cannot fix it. Only
  877 micro-crypto bets have 1h follow-on tape, against 53,897 real-world.

  **Run it on the CHROMEBOOK, not the VM.** The Chromebook is idle with cron off,
  3.8 GB RAM, and holds a 4,695,081-row ledger copy (98.6% of the VM's). Read-only
  probes peaking 743–751 MB on the 1 GB VM starved the 5-min ingest — it degraded from
  ~48 s to 5–7 min per tick and only recovered when the probe was killed. Do not run
  heavy analysis on the VM while cron ingest is live.
- [ ] **Re-check weights** (`scripts/tune_weights.py`) as data grows; adopt new weights
  only if its bootstrap CI excludes 0.
