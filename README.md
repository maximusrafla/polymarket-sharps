# polymarket-sharps

A measurement study on Polymarket. Every trade on the venue is public, so you can
rank wallets by how well they have done and pull out the ones that look sharp.
This repo asks whether that ranking means anything:

1. Is a wallet that looks sharp actually sharp, or is it the top of a noise
   distribution with thousands of draws in it?
2. If some of them are sharp, can an outside observer who sees their trades
   capture any part of it?

**The short answer is no to the second question, and a heavily qualified yes to
the first.** A small set of wallets does hold a real, out-of-sample, statistically
significant edge over the price they paid. Following them does not work at any
latency this data can measure, and most of what the certified set has is the
choice of market rather than the trading inside it. The copy-a-sharp-trader thesis
this project was built to test did not survive its own validation.

**Scope:** analysis and identification only. This repo does not trade, mirror
wallets, or execute transactions. It holds no keys, signs nothing, and reads
public unauthenticated endpoints.

`src/paper_trader.py` and `src/paper_rw.py` are a fill simulator, not an
execution path. They model what a fill would have cost by walking the public
order book and applying the venue's taker fee, and the audits that price the
findings net of costs import them for exactly that. Their own tests assert that
no key-handling symbol exists in either module.

## What it measures

The pipeline is `ingest` then `features` then `validate` then `rank` then
`report`, with the analysis arms layered on top of the same ledger.

**Ingest.** The public `/trades` feed has no working time filter and hard caps its
offset at 10,000 rows, which at current volume is 5 to 10 minutes of platform
activity. So ingest is a rolling window poller that has to run often enough for
consecutive polls to overlap, and `backfill` deepens individual wallets through
`/trades?user=`, which does filter. Resolutions come from the CLOB, one market at
a time, because the Gamma query filters were unreliable in testing. This is
written up in `DECISIONS.md`.

**The metric.** Raw win rate rewards buying favourites and raw profit rewards
betting big, so neither measures skill. The unit here is per bet
`resolved_value - entry_price`, the amount by which the outcome beat the price
paid. That turned out not to be enough either: the venue has a structural
favourite-longshot tilt that makes about 62% of all bets positive by default, and
a wallet's habitual price band is a persistent trait that survives any
out-of-sample split on its own. So the scored quantity is **skill (residual)
edge**, `outcome - E[outcome | entry_price]`, with the calibration curve fit
market-wide over quantile price bins. Raw edge is still computed and reported
beside it.

**Validation.** Each wallet's resolved bets are split chronologically. The first
half selects, the second half is held out, and a wallet is certified only if its
held-out skill edge is positive, clears an economic magnitude floor, spans enough
distinct held-out markets, and is significant under a market-block bootstrap.
Three separate corrections were forced by measurement rather than by taste:

- A sign test on tiny samples was re-measuring the base rate, not skill. A
  shuffled-outcome null reproduced almost all of the original 69% persistence.
- Bets inside one market settle on one event, so the per-bet t test overstates
  the sample. Measured median design effect on the persisted set was 1.60, with a
  maximum of 230. Significance moved to a bootstrap that resamples whole markets.
- Applying alpha = 0.05 per wallet across a 691 candidate search is a multiplicity
  problem. A permutation measured the resulting false discovery rate at 55%. The
  threshold moved to 0.005, roughly the calibrated form of Benjamini-Hochberg at
  q = 0.05 on this data, and the certified set went from 41 wallets to 26 at a
  measured 20% FDR. Every FDR figure here assumes nobody has skill, so they are
  upper bounds.

**Copyability.** Skill and followability are different quantities and are kept in
different columns. `copy_window` is the gap between a wallet's entry price and the
price the rest of the market converged to shortly after, with the leakage guards
described in `DECISIONS.md`, and `copyable` is a separate sortable flag that never
feeds the score.

## What came back

- **Thin samples were most of the original signal.** Deepening the top ranked
  wallets to roughly 5,000 bets per half collapsed their apparent edge: median
  held-out skill edge fell from +0.093 to +0.0007, and only 7 of a provisional
  top 30 survived.
- **A real certified set exists and is small.** 26 wallets out of 23,956, held-out
  skill edge from +2 to +17 cents per bet, bootstrap p below 0.005, at least 30
  distinct held-out markets each. About 1 in 5 of them is what chance manufactures
  at that threshold. 23 of the 26 trade five-minute crypto markets, so this is a
  micro-market result, not a forecasting one.
- **Most of that edge is market selection, not trading.** Under a null that holds
  each wallet's market footprint fixed and permutes only who traded inside it, the
  false discovery rate does not fall at any significance threshold. Under a null
  that also randomises the footprint, it falls cleanly. Three independent runs
  reached the same conclusion by different routes.
- **Copying does not work.** Following a validated wallet was measured against
  wallet-agnostic anchors on the same bets, so only the entry moment differs.
  Entering the same market at a uniformly random time, or an hour before the
  wallet acted, both beat copying it. Latency copying was separately tested from
  30 seconds to 3 days on the one cohort slow enough to have had a chance, and
  every positive follower number was matched or beaten by the placebo.
- **The per-wallet copyable label is noise.** Under a permutation that preserves
  market clustering, dispersion of per-wallet copy edge was 3.65 cents real
  against 3.22 cents null, P = 0.23, and the count of copyable wallets moved by
  one on the random seed alone.
- **Simple screens fail.** Screening on realised return, on win rate, and on a
  two-dimensional combination of the two were all tested against out-of-sample
  return on held-out windows. All fail. The one screen that appeared to work was
  an artifact of how often crypto markets resolve.
- **Streak following fails the same way.** A three-to-four-win streak rule looks
  strong raw and is an illusion, for the reason the literature already gives:
  streak winners switch to safer odds.

`HANDOFF.md` is the full record, including the results that were withdrawn after
a better control was built, and `docs/FRAGILITY_INDEX.md` grades every
load-bearing claim by how much it would take to overturn it.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml

python -m src.ingest              # poll the public trade feed into data/raw
python -m src.ingest --fold-delta # fold the delta parts into the bet ledger
python -m src.backfill            # deepen per-wallet history for ranked wallets
python -m src.features            # per-wallet metrics
python -m src.validate            # the out-of-sample split, required
python -m src.rank                # ranked table into data/processed
python -m src.report              # readable summary
```

`./run.sh` runs all of it in order. `scripts/crontab.example` documents the
schedule the pipeline was actually run on, which splits the five-minute ingest
from the nightly recompute.

Tests are hand-built fixtures where the correct answer is known by construction,
and no test makes a network call:

```bash
pytest
```

Everything under `data/` is generated and is not committed, including the ranked
table and the per-wallet report. They name individual public wallet addresses
next to this project's own descriptive pattern flags, which is a judgement about
identifiable people that the repo has no reason to publish. Run the pipeline to
rebuild them.

The analyses that are not part of the nightly pipeline live in `scripts/` and
each writes up its result in `docs/`. They are read-only and every one of them
names the script that reproduces it.

## Layout

```
src/          the pipeline and the analysis arms
scripts/      one-shot audits, each paired with a doc in docs/
docs/         the write-up for each analysis, including the negative ones
tests/        fixtures with known answers, no network
config/       config.example.yaml, copy to config.yaml
CLAUDE.md     the original build spec
DECISIONS.md  every modelling choice and the measurement that forced it
HANDOFF.md    the running record, including what was withdrawn and why
```

## Known limits

Sample sizes are what the poller observed, not a wallet's complete history. The
bet ledger is a backfill of selected wallets, so the tape a follower is priced
against is itself made of tracked wallets' trades rather than the full book.
Split-half persistence is structurally blind to insurance-shaped payoffs, where
the edge is a rare tail that neither half observes. Sell-side fills are ingested
but never scored, because attributing a sell's entry needs position lineage this
pass does not reconstruct. Three verdicts in the record have no committed script
and are marked unreproducible in `docs/FRAGILITY_INDEX.md`.
