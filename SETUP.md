# SETUP.md — one-time setup & how to run this hands-off

You do the steps in **Part 1** once. After that the project runs itself; **Part 2**
explains the two ways it "churns" and which budget each one spends.

> Command names below match Claude Code v2.1.x (mid-2026). These change. Type `/`
> in Claude Code to see what your build exposes, and treat the official Commands
> page as the source of truth if anything here is stale.

---

## Part 1 — one-time setup (do once, on the always-on box)

1. Put this repo in a **private** GitHub repo. `git init`, commit, push.
2. On the Debian Chromebook: install Claude Code
   `curl -fsSL https://claude.ai/install.sh | bash`, then `claude auth login`.
3. `cd` into the repo. Create the venv and install deps:
   `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
4. `cp config/config.example.yaml config/config.yaml` (config.yaml is gitignored).

That's the whole manual part. Everything below is how it operates.

---

## Part 2 — the two phases (this is the answer to "/loop vs /goal, effort, model")

There are two separate activities, and they are NOT the same mode.

### Phase A — BUILD the pipeline (interactive, one-time, agent-driven)
This is where `/goal`, `/loop`, and effort/model choices belong. You run Claude
Code interactively and let it build until the tests pass.

- **Model:** default is **Sonnet 5** — leave it there for most of the build; it
  handles pipeline/boilerplate fine and is cheap. Switch to Opus only for the two
  genuinely hard parts (the edge/fair-value math and the out-of-sample validation):
  `/model opus` then `/effort xhigh`, and switch back with `/model sonnet` after.
- **Autonomy:** use `/goal` to set the target and `/loop` to let it self-correct
  until a validator (the test suite) passes. Pair them: `/loop` runs the build/fix
  cycle, `/goal` holds the objective, and it keeps going until tests are green.
- **Do it in stages, not all at once.** Point it at CLAUDE.md and have it implement
  `ingest` first, stop, you eyeball the pulled data, then continue. Ingestion is
  the one place a wrong turn quietly poisons everything downstream.

### Phase B — CHURN (recurring, hands-off)
Once built, "churning" means two different things — keep them separate:

1. **The data run itself (free).** `run.sh` is plain Python. It re-pulls new
   trades, recomputes metrics, re-ranks. It uses **zero model tokens**.
   Important: the public trade feed has no time filter and hard-caps its
   offset at 10,000 rows, which at current volume is only ~5-10 minutes of
   activity (see DECISIONS.md) — so `src.ingest` needs to run every few
   minutes on its own, separate from the heavier nightly full pipeline
   (`run.sh` = ingest + backfill + features/validate/rank/report):
   ```
   crontab -e
   */5 * * * * cd ~/polymarket-sharps && ./scripts/ingest.sh
   17 3 * * * cd ~/polymarket-sharps && ./scripts/recompute.sh
   ```
   (`scripts/crontab.example` is the canonical version: both wrappers serialize
   on the shared ledger writer lock, the 5-min ingest only appends small delta
   parts so it stays ~a minute, and the nightly recompute folds them into the
   ledger before re-ranking.)
   This is the part that runs 24/7 on the Chromebook and costs nothing. The
   nightly `run.sh` also backfills the top-ranked wallets' full per-user history
   (`src.backfill`), the main lever on ranking quality — see DECISIONS.md.

2. **Optional agent maintenance (separate budget).** If you want Claude Code to
   *also* periodically self-heal (e.g. ingestion broke because the data source
   changed) or improve the model, run it **headless** on a schedule:
   ```
   0 5 * * 0 cd ~/polymarket-sharps && \
     CLAUDE_CODE_EFFORT_LEVEL=auto \
     claude -p --model sonnet "Read CLAUDE.md. Run the test suite. If anything
     fails or ingestion returned no new rows this week, diagnose and fix, commit
     with a clear message, and summarize what changed." \
     >> data/agent.log 2>&1
   ```
   **Budget:** headless `claude -p` bills against your **separate monthly Agent SDK
   credit** (since 2026-06-15), NOT your interactive weekly limit. So this does not
   compete with your chat usage. Confirm the split on your plan.

---

## Straight answers to the specific questions

- **`/loop` or `/goal`?** Both, together, but only in **Phase A (interactive build)**.
  `/goal` sets the objective, `/loop` runs the self-correcting cycle until the
  tests pass. They are **not available in headless `-p`** — there you put the same
  intent in plain language in the prompt (as in the Phase B example).
- **Effort level?** For the build, `xhigh` on the hard math (Opus), default `high`
  elsewhere. For unattended headless runs, `CLAUDE_CODE_EFFORT_LEVEL=auto` lets it
  self-scale thinking per task — this is the closest thing to "self-adjusting."
- **Model selection, auto-switching?** Be aware: **effort** can be `auto`, but
  **model does not auto-switch by task difficulty**. You pick it. Practical setup:
  pin **Sonnet 5** as the default (cheap, capable) and only reach for Opus manually
  on the hard steps during the build. There is no reliable hands-off "use Opus only
  when it's hard" toggle — so don't architect around one.
- **Weekly limit drain?** The recurring churn (Phase B.1) is free Python. The only
  token spend is the one-time build and any optional headless maintenance, and that
  maintenance is on the separate Agent SDK credit. Your interactive weekly limit is
  largely untouched by the running system.
