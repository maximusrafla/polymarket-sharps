# Project 2 — First Ranked Output (copyable forecasters, per category)

**Date:** 2026-07-20. **Status:** first end-to-end run of the Project-2 stack —
market-first discovery → resolution enrichment → per-(wallet, category) A/B/C
scoring → per-category dashboard. Read `docs/project2_forecaster_discovery.md`
(spec) and `docs/project2_verify_gate_findings.md` (API verify gate) first; this
doc records what the first real run produced and the choices made to get there.

**Read-only / analysis-only, and fully isolated.** Everything here writes ONLY to
`data/interim/discovery/` (gitignored/local). The shared `bet_ledger.parquet`,
`backfill_cursors.json`, and `data/raw/resolutions.parquet` were never written —
the shared resolutions cache was read **read-only**, and resolutions we fetched
ourselves went to our own `discovery_resolutions.parquet`.

---

## What was built

| Module | Role |
|---|---|
| `src/discover.py` (extended) | `--closed-only` sweep (resolved markets → scoreable bets); `--resolve` resolution-enrich (shared cache read-only → `discovery_resolutions.parquet`); incremental **checkpointing** (tape+cursors every N markets) so a long pull is interruption-safe/resumable |
| `src/forecaster_metrics.py` (new) | per-(wallet, category) A/B/C stack → `forecasters.parquet` |
| `src/rank_forecasters.py` (new) | per-category sortable dashboard (markdown + self-contained HTML) |

Tests: `tests/test_forecaster_metrics.py`, `tests/test_rank_forecasters.py`, and
new resolution-enrichment cases in `tests/test_discover.py` — planted fixtures
where the correct skill edge, copyability, magnitude, and winner/uncopyable
split are known by construction.

## The metric stack (per (wallet, category) cell, + a per-wallet `all_real_world` aggregate)

- **A — is the edge real?** Favorite-longshot-neutralized **skill** edge
  (`resolved − E[outcome | entry_price]`), selected in-sample and validated
  out-of-sample with the same chronological split + one-sided significance test
  as Project 1 (`src/validate.py`). The one correctness change (spec §2.0): a
  **per-category price baseline** — `fit_category_baselines` fits `E[outcome|price]`
  per category where sample allows and falls back sub-category → real-world-wide →
  global, so a 0.75 NFL favorite isn't residualized against a crypto-dominated
  curve. `edge_persisted` = significant **and** clears the skill-magnitude floor.
- **B — copyability.** Reusing `features._forward_price_arrays` with the exact
  leakage discipline (bounded window, resolution guard, **no** `resolved_value`
  fallback), for each resolved BUY bet: the price a follower entering Δ∈{30,60,300}s
  late would pay, and the FL-neutral edge they'd keep. Two components kept separate:
  **reachability** (is there any follow-on liquidity to fill against?) and
  **retained copyable skill edge**.
- **C — economic magnitude.** An independent floor on the copyable **skill** edge
  in cents — significance is not magnitude, and a 1¢ mean certified at p≈1e-26 is
  not a finding about the world. Units are residual-skill cents, **never** $ profit
  or return %.
- A **winner** clears A ∩ B ∩ C. Every cell stays in the table (nothing dropped);
  gates are additive boolean columns and the default sort is copyable skill edge —
  never profit/returns/win-rate.

## Run parameters + performance choices

- **Discovery:** volume-ranked Gamma enumeration to the ~2400 offset ceiling →
  **2,100 closed real-world markets** (zero micro-crypto — the discovery gap is
  solved). Pulled the **top 120 by volume** (`--closed-only`, checkpoint every 5).
  Every one hit the ~10.5k-trade offset cap (~14s/market) → **1,244,854 trades,
  120 markets, 331,726 wallets**. Category mix: soccer, politics, econ_macro, NBA,
  NFL, crypto_event, plus "other".
  - *Caveat (spec §1.2):* the biggest markets' retrievable tape is the most-recent
    ~10.5k trades, which clusters **near resolution** — so copyability (B) is
    understated for the mega markets (their early-entry tape scrolled past the cap).
    Mid/low-volume markets and live open-market capture (§1.4) are the fix; deferred.
- **Resolution:** of 120 markets, 61 were already in the shared cache (read-only),
  59 fetched from CLOB (≤5 workers, polite) → all 1,244,854 bets resolved.
- **Scoring performance:** the tape has 278k wallets with a BUY bet but only
  ~21k with ≥5 and ~2k with ≥20 (the depth needed to test persistence). So the
  expensive per-cell OOS + copyability computation runs only for cells ≥
  `min_forecaster_score_bets` (5); all other cells keep cheap vectorized base
  metrics and NaN gates (`scored=False`) — present, sortable, never dropped.
  Copyability follower-prices use `searchsorted`-bounded windows so mega-market
  tokens (~5k trades) don't cost O(T) per bet.

## Results

The pipeline runs end-to-end and produces a ranked per-category output. **The
headline finding is a data caveat, not a wallet:** the top-120-by-volume tapes are
so biased toward the near-resolution slice that metric A is not yet a trustworthy
skill measure — confirming spec §1.2 empirically.

**Funnel (first run, `--min-score-bets 10`):** 338,646 (wallet, category) cells over
278,257 wallets → 226 persist (A skill: significant + magnitude) → 125 cleared A∩B∩C
*before* a breadth floor. `all_real_world`, `sports_soccer`, `other`, and `politics`
carried most winners.

**Why the raw winners are not trustworthy (the near-resolution bias):**
- Winner **skill edges were +0.14 to +0.94** — 10–50× genuine forecasting skill
  (Project 1's validated wallets are ~1–7¢). That magnitude alone is a red flag.
- **39 of 125 winners had a 100% win rate** (never lost a bet); the median winner
  win-rate was **87%** vs. the ~50–62% a real forecaster shows.
- **42 of 125 had breadth ≤ 2** (18 were single-market) — concentrated one-offs, the
  exact "mirage" HANDOFF flagged.
- Root cause: `/trades?market=` returns only a mega market's most-recent ~10.5k
  trades, which cluster near resolution. Scoring those late entries as if they were
  *forecasts* rewards "bought the winning side in the endgame," not skill. Metric A
  has no lifespan guard (and can't have a meaningful one when the whole retrievable
  window is already the endgame).

**Guardrails added in response (this run):**
- A **`win_rate` diagnostic column** — a rate near 1.0 paired with a large skill edge
  flags the near-resolution artifact; surfaced in the dashboard.
- A **breadth floor on the winner gate** (`min_winner_breadth` = 3): a copyable
  *forecaster* needs a multi-market record, which removes the single-market mirages.

**After the breadth floor (≥3 markets):** **83 winners** remain (down from 125). Their
`win_rate` median is still **0.80** (16 are 100%-win) and their skill-edge median is
**0.129** (~13¢) — so even the diversified survivors carry near-resolution inflation.
The absolute numbers are **provisional**: read the ranked table as a **methodology
demonstrator + a data-quality verdict**, not a vetted copy list. `win_rate` and `breadth`
are dashboard columns so a user can filter the worst artifacts themselves.

**Winners by category (funnel):**

```
      category   cells  wallets  skill>0  persist(A+C)  reach(B)  winners(A+B+C)
all_real_world   28610    28610     2909           82       745            31
 sports_soccer   86524    86524     6987           62       605            30
         other   52971    52971     7787           48      1366            15
      politics   63912    63912      657           15      1566             7
  crypto_event    3384     3384      137            0       196             0
    econ_macro   53691    53691     1554            4       467             0
    sports_nba   33185    33185      286           15        32             0
    sports_nfl   16369    16369       47            0        35             0
```

The most *credible-looking* survivors are the deep, diversified ones (e.g. a politics
wallet with 434 bets across 11 markets, 77% win-rate, held-out skill +0.22 at p≈8e-23) —
still bias-inflated, but the shape a real forecaster would have. NBA/NFL/econ/crypto_event
produced no winners (too few reachable cells — copyability liquidity is thin there in this
sample).

**The fix is data-side, not scoring-side** (spec §1.2/§1.4): pull the **mid/low-volume
long tail** (fuller tapes, early entries intact) and run **live open-market capture** to
record early entries before they scroll past the 10.5k cap. Both are already-specced
follow-ons; this run is what makes their necessity concrete.

## Honest caveats / what's next

- **Sample depth is the binding constraint** (spec §2.z). Real-world forecasters
  bet infrequently; per-(wallet, category) depth caps how many can be validated.
  The `all_real_world` aggregate pools a wallet's categories to recover depth.
- **Near-resolution bias on mega markets** understates copyability (B) — needs the
  mid-volume long tail + live open-market capture.
- **Config knobs are module constants** for now (copyability Δ/bandwidth/reachability,
  magnitude cents floor, `min_forecaster_score_bets`), mirroring the spec's §2.y
  sketch — the shared `config.yaml` has a concurrent workstream, so these are kept
  local (as `discover.py` already does) and promoted to config later.
- **Metric D (recency / regime-change)** is deferred; the table carries
  `bets_per_month` and `skill_edge_last20` as lightweight form signals only.
- **Not merged into the main ledger.** Discovery is a separate dataset by design
  (the shared ledger is owned by another workstream). Merging is a documented
  follow-on.
