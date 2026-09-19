# Conditional analysis over the 74 — and the artifact it caught in the funnel's own universe rule

**Task** (queued by `docs/wallet_funnel_2026-07-29.md`): across the 74 structural
survivors, does copyable edge concentrate by category, entry-price band, or
time-to-close? Cohort-level only. **Script:** `scripts/funnel_conditional.py`
(pre-registered design in its docstring; registered run + `--check`).

**Answer: no. The apparent conditional structure is an artifact of the
funnel's own bet-universe rule, and once that rule is removed no
wallet-conditional pattern survives.** The one real regularity left is the
project's recurring one: the cohort's raw advantage lives in *which events it
is in*, not in beating anyone within those events.

---

## 1. The registered run (universe rule exactly as funnel F1: `tape_t_max − entry ≥ 24 h`)

Universe: 521,292 copyable resolved real-world BUYs (of 3.39M total; 2.07M have
no `tape_t_max` at all — the funnel chain sees ~15–40% of the tape). Cohort =
63,927 bets / 5,717 events; placebo = the other 1,640 deepened wallets' bets in
the same universe. Base rates first: cohort event-weighted −0.17¢ [−0.97, +0.66];
placebo −0.77¢ [−1.40, −0.16].

53 cells (category, price band, horizon marginals + category × coarse band).
8 flagged on H1 (eff ≥ 30 & CI > 0); **5 passed the H2 exam** — and every one
of them was a longshot cell, mirrored by the favorite cells being symmetric
*negative*:

| cell | H1 | H2 | placebo |
|---|---:|---:|---:|
| price [0.0, 0.1) | +4.5¢ | +5.2¢ | +3.5¢ |
| price [0.1, 0.2) | +5.2¢ | +7.1¢ | +4.1¢ |
| price [0.3, 0.4) | +7.3¢ | +6.1¢ | +2.7¢ |
| other × long(<0.2) | +5.8¢ | +7.1¢ | +3.5¢ |
| culture × long(<0.2) | +6.8¢ | +5.2¢ | +5.8¢ |
| …while price [0.8, 0.9) | −4.8¢ | −4.9¢ | −4.8¢ |
| …and price [0.9, 1.0) | −4.0¢ | −3.6¢ | −3.4¢ |

Paired on shared events, 4 of the 5 passing cells read **market-level** (cohort
≈ placebo); one (`culture × long`) read weakly wallet-conditional at +0.83¢
[+0.08, +1.78] — 1 of 5 contrasts tested, not believed.

## 2. The artifact — the universe rule is forward-looking

`tape_t_max` is the **last observed trade** in the market. Requiring
`tape_t_max − entry ≥ 24 h` therefore conditions every bet on **post-entry
market activity**. A longshot that collapses to a near-certain loser can go
quiet (bet excluded); a longshot that surges toward winning keeps trading (bet
included). Favorites, symmetrically: the ones that cruise to a win go quiet
(excluded), the contested ones stay (included). Prediction: the rule
manufactures longshot-positive / favorite-negative raw edge **for cohort and
placebo alike** — which is exactly the table above. This is the same artifact
family as the 2026-07-22 edge-decay "coverage" trap (conditioning on a later
print selects mispriced markets).

**Check** (`--check`): the outcome-independent close variant (Gamma scheduled
close) is *unrunnable on this tape* — `market_meta`'s Gamma close fields cover
121 of 3.39M bets. The no-filter variant is decisive:

| band | cohort, filtered | cohort, NO filter | placebo, filtered | placebo, NO filter |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | +4.5 / +5.2¢ | **−0.0¢** [−0.35, +0.26] | +3.5¢ | −0.2¢ |
| [0.1, 0.2) | +5.2 / +7.1¢ | **+0.3¢** [−0.34, +1.07] | +4.1¢ | −1.3¢ |
| [0.7, 0.8) | −3.4 / −3.5¢ | **+2.8¢** [+1.92, +3.69] | −2.1¢ | +2.0¢ |
| [0.8, 0.9) | −4.8 / −4.9¢ | **+2.3¢** [+1.76, +2.92] | −4.8¢ | +1.4¢ |

The gradient does not shrink — it **inverts**. The conditional structure was
the filter, not the market. All five passing cells are withdrawn.

## 3. What is actually there, once the artifact is removed

On all 3.39M unfiltered resolved real-world BUYs: cohort +0.87¢ event-weighted
[+0.65, +1.09] (340,997 bets / 52,629 events); placebo +0.20¢ [+0.09, +0.31].
But **paired within the 41,551 shared events the contrast is +0.19¢
[−0.10, +0.48]** — no per-event advantage. The cohort's raw excess is
composition: it sits in mid-price bands and events where raw edge was positive
in this sample. That is the footprint result again — fifth independent
appearance (FDR null B, metric B, s_all-vs-s2_core forward, the copy audit's
+2.4¢ position component, now this).

## 4. Collateral: what this does to the funnel

- **F2/F3 counted "copyable bets/markets/events" under the same forward-looking
  rule**, and the **performance stage** (ROI/edge halves, the 65% base rate,
  the winner's-curse table, the six provisional wallets) was computed entirely
  inside the artifact universe. Those numbers should not be trusted until
  re-derived on an outcome-independent close time. The funnel's *structural
  logic* — power arithmetic, the F3 correction, calibration-case discipline —
  stands; its *bet universe* does not.
- **F1 (wallet-level median ttc)** is less poisoned (a median over a wallet's
  bets is far blunter than per-bet selection), and its conservative direction
  ("errs toward calling wallets fast") still holds; but the same re-derivation
  should re-check it.
- **Forward scoring is untouched** — it reads Gamma `endDate` live per signal
  (55/56 signals carry `gamma_endDate` as the horizon source), so forward data
  stays clean. The retrospective pooled +2¢ copy result is also mostly
  insulated: its load-bearing number is a *contrast* against placebos that
  share the same fill-conditioning, though its absolute level shares this
  concern class.

## 5. The fix, if the conditional question is worth asking again

Backfill scheduled close times from Gamma for the tape's markets (Gamma
`/markets` pages ~500/request; the whole 410k-market set is ~1–2 hours of
polite GETs), then re-run: (a) funnel F1–F3 counts, (b) the performance stage,
(c) this conditional analysis, all on `gamma_end − entry ≥ 24 h`. Until then,
no retrospective "copyable universe" claim from this tape should be quoted.

## 6. Traps updated (append to the funnel doc's list)

6. **A time-to-close measured from the tape is a forward-looking universe
   rule.** `tape_t_max` encodes post-entry activity, and conditioning on it
   manufactures a longshot-positive / favorite-negative gradient for every
   buyer. Any per-bet filter must use a *scheduled* close time, never the last
   trade. (Wallet-level medians are blunter but re-check them too.)
