# RED-TEAM BRIEF — the real-world copy arm (2026-07-29)

**Paste this into a fresh max-effort session.** It is written by the analyst whose
work you are auditing. Treat it as a map of where the bodies are *probably*
buried, not as a bound on where to dig.

---

## YOUR MANDATE

Assume every conclusion below is **wrong in both directions** — that the copy edge
is fake, *and* that a real edge was destroyed by an over-correction. Re-derive
every load-bearing number from primary data. Repo discipline: `CLAUDE.md`,
`HANDOFF.md`, `DECISIONS.md`, and the precedent set by
`docs/redteam_audit_2026-07-26.md` (a prior audit of this repo, which found the
then-conclusions "~80% justified and 20% premature" — that is the standard).

Python is `.venv/bin/python`. 3 GB RAM box, OOM-killed repeatedly — stream, project
columns, dictionary-encode strings. Read-only public GETs only. **Do not touch the
running forward scorers** — they are scoring pre-registered cohorts.

---

## WHAT WAS BUILT (chronological, with commits)

1. **`src/realworld_deepen.py`** (8601e09, 382c7bb) — the thin real-world sample was
   a *collection artifact*: `ingest.py` is a global firehose that is ~82% 5-minute
   crypto. Deepened a **stratified, performance-blind probability sample** of 2,500
   discovery wallets → 2,117 fetched, 7.05M trades, **3.39M real-world resolved
   BUY bets**, 88.1% real-world.
2. **`src/realworld_validate.py`** (f56fca3, 0954022) — Project 1 gate, unchanged, on
   that population: **58 certified** of 2,475.
3. **FDR** (`scripts/audit_persistence_fdr.py --tape realworld`, 24714aa) — measured
   **17.3% ± 0.4%** under null B2 → ~48 of the 58 real. Real arm reproduced the
   shipped artifact bit-for-bit before any null was scored.
4. **`src/copy_sim.py`** (4448ba5, ecdcc7c, 2cad18f, 35d6901) — can a follower 2
   minutes behind capture it?
5. **The forward cohort** (bf49587, 655f74c) — 9 wallets, pre-registered, scored on
   bets placed after the freeze.

## CURRENT HEADLINE CLAIMS (all suspect)

- **11 of 37** certified wallets are copyable at a 2-minute lag, median **78%** of
  the wallet's own edge retained. Best ~**+5–6¢/share**.
- **4 of 72** high-volume *uncertified* wallets are copyable (hit rate 5.6% vs 30%
  for certified — used to argue certification "does real work").
- **Structural claim:** high-volume wallets are systematically uncopyable because
  they trade liquid fast markets.

---

## MY OWN ERRORS — five in one session, so start here

Each was caught late; each was larger than the effect being measured. **Three of
the first four moved the answer toward "copying works."**

| # | error | size | how it was caught |
|---|---|---|---|
| 1 | Copy-sim null shuffled outcomes **within the certified wallets' own bets**, absorbing the skill it tested for | turned a real effect into "+0.0003¢, dead" | it reported **exactly 0.000 alpha at Δ=0**, which is impossible |
| 2 | Charged a **1¢ adverse tick**; 97% of these markets quote **0.001** | ~10× overcharge | checked tick_size in the book |
| 3 | Zeroed fees across the whole **`other`** bucket (48.8% of tape) by extrapolating from verified-zero-fee geopolitics | inflated edge | noticed the extrapolation was unverified |
| 4 | **Credited the follower at the book MIDPOINT while charging the wallet its post-slippage taker VWAP** | **~+3.13¢/share of unearned edge**; copyable set read 38 instead of 11 | **14 of 46 wallets showed the follower beating the wallet on identical bets** — impossible |
| 5 | Proposed daily **mark-to-market** on open positions to shorten the wait | rejected before running | it is circular (credits the wallet for its own price impact), re-measures the already-known post-trade drift, and a mark is meaningless in a book this thin |

**Error 4 is the template: the tell was in my own output and I did not check it.**
An impossibility assertion (`follower_net > own_net` must never happen) now exists,
but only for that one case. **Look for the other impossibilities nobody asserted.**

---

## WHERE I THINK THE REMAINING SOFT SPOTS ARE

Ranked by how much they'd change the answer.

1. **The population baseline may be circular.** After killing null #1, I compare the
   follower against `E[outcome | price]` fitted on the *same 3.39M-bet tape* that
   contains these wallets' bets. Contamination is small (46 of 2,094 wallets) but
   unquantified. **There is currently NO null in the copy analysis at all** — only a
   baseline. That is the single biggest hole.
2. **Per-wallet verdicts are unstable.** `0x5415c298dde8` read *not copyable*
   (CI [−0.50, +7.86]) in the 46-wallet run and *copyable* in the 72-wallet run.
   Different windows/guards, so not the same test — but it means individual
   classifications are shakier than their CIs imply. Quantify that instability.
3. **Selection on the copyable set.** 11 of 37 were selected *after* seeing results,
   then their CIs were reported as if pre-specified. No multiplicity correction was
   applied to the per-wallet CIs. How many of the 11 survive BH-FDR?
4. **The 78% "edge retained" figure** is a median over survivors — i.e. conditioned
   on having survived. Almost certainly optimistic.
5. **`0xa8638d8d7a00` is in the copyable set and is `micro_crypto`**, a category the
   owner explicitly dropped. Its presence suggests the real-world filter leaks.
6. **The spread proxy** `max(entry_price − mid, tick)` assumes the follower faces the
   same spread the wallet did, 2 minutes later. Unvalidated. Direction of bias unknown.
7. **The live-book probe** samples books for markets that are still open —
   survivorship toward longer-lived, more liquid markets — and it was compared
   against a **stale 2.02¢** constant in the printed output.
8. **API truncation:** `/trades?user=` caps at ~10,000 rows. 16% of deepened wallets
   (and 15–20% of certified) are truncated to their most recent 10k trades →
   recent-performance bias.
9. **Fees are historically wrong by construction.** Polymarket was genuinely zero-fee
   before 2026-01-05, and most of this tape predates that. Net figures apply today's
   fees retroactively — correct for decision-making, but these wallets never
   optimised against fees.
10. **The pre-registered cohort was re-frozen once** (AMENDMENT 1) with 16
    unresolved observations already recorded. I argued this was legitimate because
    none had *resolved*. Audit that argument.

## SPECIFIC THINGS TO RE-DERIVE

- The **17.3% FDR**. Null B2 was chosen over null B (69.2%) on a pre-registered
  D ≥ 1 criterion — but the criterion selected the *favourable* number. Is the
  criterion sound, or is the honest range 17–69%?
- **Build a real null for the copy analysis.** Nothing currently plays that role.
- Re-run the copy sim with the spread proxy **swept**, not fixed.
- Whether "high-volume wallets are uncopyable" is causal or an artifact of the
  uncertified pool being mostly noise to begin with.

## WHAT SURVIVED SCRUTINY SO FAR (attack these too, but they are the sturdier parts)

- The deepening's **stratified performance-blind design** — inclusion probabilities
  are known and persisted; the draw provably cannot see performance (test-enforced).
- The **FDR machinery's self-check**: the vectorised replica reproduces the shipped
  artifact bit-for-bit, over a different code path, before any null is scored.
- **Drift is front-loaded** — 1.80¢ of ~2.5¢ total lands inside 60 seconds. This
  closes the "faster bot" hypothesis and is robust to every cost assumption because
  it is measured on prices, not P&L.
- The **token-index cache-safety fix** (a rebuild would have silently re-pointed
  3.5M cached price points at the wrong tokens).

## DELIVERABLE

`docs/redteam_realworld_copy_2026-07-29.md`: for each headline claim —
**CONFIRMED / OVERSTATED / WRONG**, with the re-derived number. Then a single
verdict on the question that actually matters: **is there a copyable edge here at
all?** A clean negative is a valuable result; this repo has a long history of them
and they have all been worth having.
