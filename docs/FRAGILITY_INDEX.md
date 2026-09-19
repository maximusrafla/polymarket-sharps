# Fragility index — what this project has concluded, and how much weight each conclusion holds

**Purpose.** This repo carries ~30 analysis documents, a 2,100-line HANDOFF and a 90 KB
DECISIONS log. A new reader — human or model — inherits all of it as settled fact, and
several conclusions are phrased as *instructions* ("do NOT re-attempt"). **That has
already caused one documented failure** (see §5). This index exists so a fresh reader can
tell in one pass which conclusions are load-bearing, which are one method-choice away from
flipping, and which cannot be reproduced at all.

It is deliberately *not* a summary of findings — `HANDOFF.md` does that. It is a
confidence map.

## The grades

| grade | meaning | how to treat it |
|---|---|---|
| **A** | Structural. Arithmetic, ground truth, or multiple independent methods agreeing. Overturning it needs new data or a found error. | Build on it. |
| **B** | Solid, single design. Well executed, controls passed, but one design and no independent replication. | Trust; re-derive before relying on it. |
| **C** | **Method-sensitive.** The conclusion moves when a defensible alternative null / threshold / cluster unit is chosen. Some have already flipped once. | **Re-open before citing.** These are where the errors are. |
| **D** | Unreproducible. No committed script or artifact. | Treat as an unverified assertion. |

A second axis matters as much as the grade: **what would flip it.** A claim with a cheap,
known falsifier is safer than a grade-B claim nobody can test.

---

## 1. Grade A — the foundations

| claim | rests on | notes |
|---|---|---|
| Raw win rate and raw profit are traps; edge must be residualized on entry price | Arithmetic + the favorite-longshot curve measured on 3.4M bets | **Re-derived 2026-07-30.** Refit at 20 / 200 / 1,000 bins the top-100 cohort moves +4.855 → +4.868 → +4.831¢, 97/100 same wallets. The 20-bin choice is not doing work. |
| The unit of evidence is the resolution **event**, not the bet | Two separate unit errors caught and corrected (sports "284 games" ≈ 25 events; slow-market rolling-deadline ladders) | Every CI in the repo is event-clustered because of these. Getting this wrong has been the single most common error mode. |
| Fees are real: k = 0.07 crypto / 0.05 sports / 0.04 politics-finance-tech, taker-only | Verified on-chain to 7 s.f. | ~1.2–1.75¢/share at mid prices. This is the number that has killed nearly every candidate. |
| Split-half persistence is **blind to insurance-shaped payoffs** | Project 4: the [0.98,1.00) band passed OOS split-half yet is fair-priced | Both halves under-observe a rare tail, so agreement is automatic. Any edge concentrated at price extremes needs a fair-price null instead. |
| Same-event leakage inflates any "next bet / other half" statistic | Streak test: 65–80% of "next bets" shared the streak's event. Forensics: raw split-half ρ 0.252 → 0.185 disjoint | **Check this first** on any new paired design. It has bitten twice. |

## 2. Grade B — solid, single design

| claim | rests on | cheapest falsifier |
|---|---|---|
| Copying **timing** is a NO-GO at every latency (30s → 3 days) and in every cohort | Wallet-agnostic placebos beating or matching the follower in each run | A cohort where the placebo is genuinely unavailable |
| Pooled real-world copy edge +1.98¢ [+1.16, +2.82] survives five controls incl. the first placebo any copy thesis here passed | `docs/redteam_realworld_copy_2026-07-29.md` | A forward reading on data the selection never saw |
| Clean timing test passes: real − donor = +0.83¢, p<0.0001 | `scripts/audit_copy_timing.py` | — |
| Following the certified 26 forward is **decisively negative** against a wallet-agnostic placebo (459 eff. events) | Actual forward data — the strongest evidence class in the repo | — |
| Forecaster broadening is structurally dead (universe is correlated ladders) | Widened tiers came out *narrower* in effective narratives (t87 2.68, t52 2.24 vs t12 3.13) | — |
| Streak-following: raw rule is an illusion; clean residual +1.8¢, decaying to ≈ fees by 2026 | `scripts/streak_follow.py` + attribution | Resolution timestamps (not collected) would make it testable as implemented |
| **Wallet skill genuinely persists**: event-disjoint split-half ρ = +0.185, exact permutation null, P<0.0001, n=1,174 | `scripts/forensics_checks.py` (2026-07-30) | — |
| **Persistence is concentrated on the downside**: worst H1 quintile → −1.15¢ H2 (p=3×10⁻⁶); best → +0.22¢ (p=0.46) | same; robust to bet/share/notional weighting; cohort 89% stable across selection rules | — |
| **Fading that cohort is a NO-GO on spread** (net −0.37¢ to −0.77¢) | `scripts/fade_spread_cost.py`, 321 quota-stratified live books, cohort-weighted spread 0.86¢ vs 0.47¢ break-even | A maker (resting-limit) variant, which needs a fill model rather than a backtest |
| Wallets bet **bigger on worse bets** (notional- minus equal-weighted, p=8.8×10⁻⁵, all price terciles) | `wallet_profile.parquet` | — |
| Value-betting on structural mispricing = NO-GO | Census read; FLB "edge" is our own wallet selection | ⚠️ audit downgraded this from *established absent* to **not established present** — see §3 |

## 3. Grade C — method-sensitive. Re-open before citing.

These are the ones to distrust. Each has a documented sensitivity.

| claim | the sensitivity | status |
|---|---|---|
| **"You cannot rank wallets" (per-wallet dispersion vs null, P=0.23)** | **Demonstrated fragile 2026-07-30.** A free (wallet,event) cell permutation returns real dispersion *narrower* than null (P=1.000 — a broken null; it shatters bets-per-event structure spanning 1.05→713). Stratified by cell size → P=0.957, still over-dispersed relative to what the persistence decomposition implies. | The conclusion "can't rank on **level**" survives. But the split-half test — which needs **no synthetic null**, since shuffling the H1↔H2 pairing is exact — says the ranking *does* carry information. **Prefer the split-half design.** |
| **FDR of the certified set** | Has been quoted as ~50%, then 80.6% like-for-like, then 55% strict, then 20% after α → 0.005. The number moves with null construction and threshold, not with evidence. | Quote the null alongside the number or don't quote it. |
| **α = 0.005 (Project 1 certification)** | Chosen after seeing the FDR table. Scoped to P1; global stays 0.05 for the frozen slow/sports arms. | Legitimate but post-hoc; the forward tests are the arbiter. |
| **The sports 2¢ floor / `s_sig2c` cohort** | The 2¢ floor was chosen *after* the parent table existed. Mitigations are stated (smooth counterfactual, floor-independent population excess), not assumed. | Manifest's own `known_limits` says so. Forward number is the arbiter. |
| **Cross-market arb = NO-GO** | **One point-in-time snapshot** of 22 outcome sets. | Reasonable prior-confirming check; **not** "arb is dead". No committed script (§4). |
| **"Only ~25% of real-world bets are copyable at all"** | Already corrected in-repo: it is a **measurement artifact of a selected-wallet backfill**, not a property of the venue. Live book probe found continuous two-sided liquidity, 1.0¢ median spread. | Was "doing more work in this repo's reasoning than it can bear." Do not reuse. |
| **"The edge is in WHICH MARKETS, not HOW they traded"** | Two independent runs converged, which is genuine support — but it is **one supporting observation each**, explicitly logged as "a hypothesis, not a finding". | Untested directly. |
| **corr(wallet mean price, residual) = +0.21** | Stable at 20/200/1,000 bins so *not* a binning artifact, but unexplained. Either the residualization leaves something at high prices or favourite-buyers are genuinely better. | **Open.** Nobody has looked. |

## 4. Grade D — unreproducible

Flagged by the 2026-07-26 audit as process debt, still outstanding. Every other verdict in
this repo ships a script; these three do not:

- **Weather gate** (NO-GO) — coordinator one-shot, no committed script or artifact
- **Category sweep** ("category space exhausted") — same
- **Live arb probe** — same

These are load-bearing negatives that closed whole search directions. They may well be
right. They cannot currently be checked.

## 5. The failure mode this index exists to prevent

**A "do NOT re-attempt" note fossilized an error, and it cost a real cohort.**

`scoring.sports.min_skill_edge = 0.10` was mis-transposed from the small-n slow-forecaster
arm onto a large-n universe (1.1M bets / 40,256 events). Applied to *held-out* edge it is a
**winner's-curse selector** — at fixed true skill it picks the most upward-noisy record,
and it did exactly that. The config documented the bug in its own handwriting
(*"same scoped rationale as the slow arm's"*). 12 wallets clearing every other gate at ≥2¢
had no tier that could read them. The note saying not to look is what kept it there.

Two more instances surfaced on 2026-07-30, both from a fresh pass rather than new data:

- `size` was on the shards from the start and **no analysis in the project had ever used
  it** — the conviction finding (wallets bet bigger on worse bets) was sitting in plain
  sight, and it means every bet-weighted headline here flatters its subject.
- The "can't rank wallets" conclusion was resting on a design I could break in two tries.

**The pattern is consistent: this project's characteristic error is conclusions hardening
into instructions.** The mitigation is not to distrust the negatives — most are correct and
hard-won — but to keep the grade and the falsifier attached to each one.

## 6. If you are picking up this project cold

1. Read `HANDOFF.md`'s START HERE, then this file. Everything else is depth-on-demand.
2. The only live evidence is what the **frozen cohorts** score on bets placed after
   their freeze. Retrospective work is largely exhausted; the copy thesis is a
   thorough, multi-front negative.
3. Before re-testing anything in §3 or §4, note that it is cheap to re-open and the repo
   has been wrong in *both* directions — a premature NO-GO cost 12 wallets, and premature
   enthusiasm has cost more sessions than that.
4. **Do not trust a mean without its event count.** Kish effective events is beside every
   number in this repo for a reason.
5. Grade any *new* claim you add here, with its falsifier. An ungraded conclusion is how
   this file becomes the thing it was written to prevent.

---

*Maintained by hand. Last pass 2026-07-30 (commit-adjacent to the forensics work). Grades
in §1 marked "re-derived" were checked directly in that session; the rest are graded from
the documentary record and the audits, not independently re-computed.*
