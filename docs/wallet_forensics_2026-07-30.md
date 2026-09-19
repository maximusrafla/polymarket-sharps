# Per-wallet forensics — the distributional view, and what it changed

**Owner's objection, 2026-07-30**, and it was correct: every conclusion in this
repo is a **mean in cents with a CI**. A mean of +2¢ can be 500 quiet +2¢ bets, one
10¢ longshot that hit, or strongly positive in one price band and negative in
another. No artifact in this project could tell those apart. This is the shape
layer: `scripts/wallet_forensics.py` (profiles), `scripts/forensics_checks.py`
(three adversarial checks), `scripts/forensics_persistence.py` (event-disjoint
split-halves), `scripts/forensics_export.py` + `docs/forensics_dashboard.template.html`
(the dashboard).

**Universe:** 3,385,372 resolved real-world BUY bets, 2,094 performance-blind
deepened wallets, 166,230 resolution events, 2023-07-27 → 2026-07-27. Scored
population: **1,628 wallets** with ≥100 bets and ≥30 events. Every mean and CI is
event-weighted with a cluster-robust SE; sample size is Kish effective events.

## The headline: the persistence is real, and it is almost entirely on the DOWNSIDE

Chronological split-half, **every event appearing in both halves removed from
both** (the streak analysis' own 65–80% same-event leakage is the reason this is
not optional):

| | ρ | n |
|---|---:|---:|
| raw split-half | +0.252 | 1,373 |
| **event-disjoint** | **+0.185** | **1,174** |
| permuted null (H2 shuffled across wallets) | +0.001 [−0.054, +0.056] | — |

**P < 0.0001.** Wallet-level skill persists. But sorting on the first half and
reading the second half shows *where* that information lives:

| first-half quintile | n | H1 | **H2 (the exam)** | mean price |
|---|---:|---:|---:|---:|
| 1 worst | 235 | −4.87¢ | **−1.15¢**  (t=−4.79, p=3×10⁻⁶) | 0.47 |
| 2 | 235 | −1.32¢ | −0.58¢ | 0.61 |
| 3 | 234 | +0.08¢ | −0.16¢ | 0.64 |
| 4 | 235 | +1.27¢ | −0.02¢ | 0.70 |
| 5 best | 235 | +4.49¢ | **+0.22¢**  (t=0.74, **p=0.46**) | 0.56 |

- **Picking good wallets buys nothing measurable.** +0.22¢ is indistinguishable
  from zero *before* the ~1.20¢ taker fee. Selecting harder is worse: the top
  **decile** keeps **1%** of its first-half edge (+6.20¢ → +0.04¢).
- **Picking bad wallets works, decisively.** −1.15¢ at p=3×10⁻⁶, **5.1×** the
  magnitude of the positive side. It survives price stratification — worst-quintile
  H2 by price tercile: **−1.73¢ / −1.11¢ / −0.12¢** (low/mid/high) — so it is not
  "longshot buyers lose".

**This is a genuinely new direction** — every arm in this project has searched for
wallets to *follow*, and the reliable per-wallet signal is the opposite one. It was
costed immediately, and it does not survive. See the next section.

### 🛑 The fade, costed: NO-GO on spread (`scripts/fade_spread_cost.py`)

Fading means: when the wallet buys outcome X, you buy NOT-X. The gross estimate
credits the fader at exactly (1 − p), which is wrong in a knowable direction:

```
ask(X)  = p            (what the wallet paid)
bid(X)  = p − s        (s = bid-ask spread)
ask(¬X) = 1 − bid(X) = 1 − p + s
```

**The fader pays the FULL spread**, not half and not none — crossing to the other
side of the book is the entire mechanic of the trade.

Measured on **321 live CLOB books**, quota-stratified 50–83 per price band (the
first pass was unstratified and landed 87% in 0–20¢, thin exactly where the
cohort's weight sits — the stratified number came back *worse*, 0.86¢ vs 0.72¢):

| price band | n books | median spread |
|---|---:|---:|
| 0–20¢ | 62 | 0.10¢ |
| 20–40¢ | 83 | 2.00¢ |
| 40–60¢ | 66 | 1.00¢ |
| 60–80¢ | 50 | 1.00¢ |
| 80–100¢ | 60 | 1.00¢ |

Cohort-weighted (matched to the bottom decile's own barbell price mix — 21.3% at
0–10¢, 16.9% at 90–100¢): **0.86¢**. Break-even was **0.47¢**.

| weighting / cut | gross | fee | spread | **net** |
|---|---:|---:|---:|---:|
| bet-equal / decile | +1.41¢ | 0.94¢ | 0.86¢ | **−0.39¢** |
| notional-wtd / decile | +1.45¢ | 0.96¢ | 0.86¢ | **−0.37¢** |
| share-wtd / quintile | +1.01¢ | 0.92¢ | 0.86¢ | **−0.77¢** |

**Every cell is negative.** The fade clears the *fee* and dies on the *spread* —
which is exactly where it was flagged as most likely to die, so this is a
pre-stated falsifier firing, not a surprise.

**The one variant this does not test** is an entry that never crosses the spread.
It cannot be tested here: such an entry fills selectively, and a fill hours later
is not the same observation. That needs a fill model, not another backtest.

**What survives:** the *identification* result stands untouched. Bad wallets are
predictable (−1.15¢, p=3×10⁻⁶, 5.1× the positive side, robust to weighting and
stable across selection rules at 89% cohort overlap). It is the *monetisation* that
fails. That distinction matters: an edge you can measure but not capture is still a
true fact about the venue.

**Cohort stability**, computed for the (unbuilt) pre-registration and worth keeping:
the bottom decile is 89% identical across all three weighting rules (111/118
bet-equal ∩ share-wtd, 109/118 ∩ notional-wtd, 105 in all three), and the
rule-invariant core of 105 still reads −1.25¢. The selection is not fragile; only
the economics are.

## The other results

**1. The core metric is sound — I went looking for a bug and did not find one.**
`features.fit_price_baseline` uses 20 quantile bins against 467,527 distinct entry
prices, so within-bin price slope could manufacture per-wallet residual. Refit at
200 and 1,000 bins: top-100 cohort edge **+4.855¢ → +4.868¢ → +4.831¢**, and
**97/100** of the same wallets survive. Binning is doing no work. *(An exact-price
baseline is NOT the control — ~7 bets per distinct price absorbs real skill along
with any artifact; it shrinks wallet-residual sd 3.23→2.52 mostly by overfitting.)*

**2. Wallets bet BIGGER on their WORSE bets.** Notional-weighted residual minus
equal-weighted: median **−0.112¢**, 57.4% of wallets negative, Wilcoxon
**p=8.8×10⁻⁵**, same sign in all three price terciles. `size` is on the shards and
no analysis in this repo had ever used it. **Consequence: every headline number
here is bet-weighted and therefore flatters these wallets** — their real dollar
performance is worse than their measured edge, and copying them in proportion to
their stake underperforms the published figures.

### Follow-up: the same numbers, weighted by money actually staked

`scripts/forensics_dollar_weighted.py`. Since wallets bet bigger on their worse bets,
every bet-equal figure here flatters its subject. Re-run three ways — **bet-equal**
(the repo's convention), **share-weighted** (per share actually held), **notional-
weighted** (per dollar deployed) — with event clustering preserved in all three
(the weight applies *within* an event; events are then averaged equally).

| | bet-equal | share-wtd | notional-wtd |
|---|---:|---:|---:|
| spearman H1~H2 | +0.185 | +0.196 | +0.177 |
| **best** H1 quintile → H2 | +0.22¢ (p=0.46) | +0.47¢ (p=0.11) | **−0.06¢ (p=0.83)** |
| **worst** H1 quintile → H2 | −1.15¢ (p=3e-6) | −1.01¢ (p=4e-5) | **−1.36¢ (p=2.8e-8)** |
| fade, bottom decile, net of fee | **+0.47¢** | **+0.33¢** | **+0.49¢** |

Three readings:

1. **Persistence is weighting-invariant** (ρ = 0.177–0.196). It is not an artifact of
   counting bets equally.
2. **The follow side gets worse on money and the fade side gets better.** Selecting the
   best wallets returns **−0.06¢ on dollars deployed** — on the capital they actually
   committed, the best-selectable cohort earns nothing. The worst quintile is *more*
   negative on dollars (−1.36¢, t=−5.75). Both move in the direction the conviction
   finding predicts, which is a consistency check the conviction result passes.
3. **The fade survives every weighting**, +0.33¢ to +0.49¢ net at the bottom decile. It
   does not depend on the weighting convention, which was the most obvious cheap way for
   it to be an artifact.

One subtlety worth recording: share-weighting *helps* the best quintile (+0.47¢) while
notional-weighting *hurts* it (−0.06¢). Notional is shares × price, so it upweights
favourites. Reading: sizing up in **share count** is mildly informative; sizing up in
**dollars** is not, because the dollars skew to favourites.

**Not re-weighted:** the pooled real-world copy edge (+1.98¢) and the certified-26 edge
live in different pipelines on the ledger, not this tape. They are bet-weighted and, by
this result, are very likely overstated on a dollars basis. Untested.

**3. Fragility is a mid-table problem, not a top-table one.** Removing each
wallet's single best resolution event: the median positive-edge wallet keeps
**79%**, and **12% flip negative** on that one deletion. But among the **top 50 by
edge, 0/50 flip** and the median keeps **89.6%**. The owner's "one crazy 10¢ win"
worry is real in the middle of the table and largely absent at the top.

**4. ⚠️ METHOD WARNING: the dispersion-vs-null design this repo leans on is
fragile.** "Can you rank wallets?" has been answered with per-wallet edge
dispersion against a cluster-preserving null. That statistic is extremely
sensitive to null construction:

| null | real | null | P |
|---|---:|---:|---:|
| free permutation of (wallet, event) cells | 2.99¢ | 4.37¢ | 1.000 |
| **stratified by cell bet-count** | 2.56¢ | 2.67¢ | 0.957 |

The first is **broken** — it shatters each wallet's bets-per-event structure, which
ranges **1.05 to 713** across this population, inflating the null's variance. The
second is defensible and still returns real *narrower* than null, which the
persistence decomposition says it should not be (ρ=0.185 implies noise sd ≈2.31¢,
not 2.67¢) — so it remains somewhat over-dispersed. **The split-half test needs no
synthetic null at all** (shuffling which H2 pairs with which H1 is exact) and is
better powered. Where the two disagree, trust the split-half. This does not
overturn the 2026-07-29 audit's `P=0.23` — the two agree that you cannot rank on
*level*; the split-half adds that the ranking nevertheless carries information,
concentrated at the bottom.

**5. Open, unexplained.** After the favourite–longshot curve is removed,
wallets that habitually buy favourites *still* show higher residual edge:
`corr(wallet mean price, residual) = +0.21`, stable at 20 / 200 / 1,000 bins, so
not a binning artifact. Either the residualization leaves something behind at high
prices, or favourite-buyers are genuinely better. Not resolved.

## Artifacts

- `data/interim/forensics/wallet_profile.parquet` — 2,095 wallets × 28 shape
  statistics (event-weighted edge + cluster SE, z (precision-weighted), median /
  trimmed mean / skew, drop-best-event, frac events positive, conviction, halves)
- `wallet_bands.parquet` / `wallet_categories.parquet` / `wallet_years.parquet` —
  per-wallet breakdowns with event-weighted means and SEs
- `persistence.parquet` — raw AND event-disjoint halves per wallet
- `population.json`, `checks.json` — baseline curve, nulls, seeds
- `data/processed/forensics_dashboard.json` → the dashboard

**Reproduce:** `flock data/interim/.analysis.lock -c 'PYTHONPATH=. .venv/bin/python
scripts/wallet_forensics.py'`, then `forensics_checks.py`, `forensics_persistence.py`,
`forensics_export.py`, then `build_forensics_dashboard.py <out.html>`. Seed 20260730
throughout. All read-only; nothing touches the shared ledger.

## What this does NOT change

The copy thesis is still negative. Nothing here makes any wallet followable — it
makes the *reason* sharper: it is not that the ranking is pure noise (it is not),
it is that the informative half of the ranking is the half you cannot profit from
by copying. A forward reading on data the selection never saw remains the only
arbiter left.
