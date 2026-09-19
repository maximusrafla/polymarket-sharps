"""Project 3 step 7b — narrative-concentration diagnostic.

THE QUESTION
------------
Step 7a made the EVENT COMPLEX the cluster unit, which stops a rolling-deadline
ladder counting as dozens of independent markets. It does NOT stop a wallet
betting CORRELATED complexes: "Iran closes Hormuz" + "Israel strikes Iran" +
"Hormuz shipping falls" are three distinct complexes but one geopolitical bet.
That is the same confound one level up.

So, before freezing: **how many genuinely independent narratives does the frozen
set actually represent?** A cohort of 12 wallets that are all one Mideast bet is
not 12 independent pieces of evidence, and the forward test inherits that.

WHAT THIS IS AND IS NOT
-----------------------
Reported as additive metadata and a flag. It NEVER gates `edge_persisted` and
never removes a row. The regress does not terminate retrospectively — there is
always a level above — so the honest move is to measure how broad the set really
is, say so plainly, and let the forward test break the regress.

The `commodity_crude` judgement (crude spikes on Hormuz risk, but also trades on
OPEC/demand/inventories) is reported BOTH ways — strict and broad — because it
materially moves the answer and picking one by fiat would hide that.

READ-ONLY: stdout plus one optional interim parquet.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/agent47/polymarket-sharps")

from src.common import INTERIM_DIR, atomic_to_parquet  # noqa: E402
from src.slow_niche import (  # noqa: E402
    NARRATIVE_MAP_VERSION,
    assign_narratives,
    effective_n as eff_n,
    narrative_spread as per_wallet_spread,
)

OUT_PATH = INTERIM_DIR / "slow_deepening" / "narrative_concentration.parquet"


def set_level_summary(spread: pd.DataFrame, label: str) -> None:
    """The number the coordinator asked for: how many independent narratives does
    the SET represent, counting each wallet by its dominant narrative."""
    print(f"\n[{label}] set-level independence — each wallet counted by its "
          f"dominant narrative")
    for mode in ("strict", "broad"):
        col = f"top_narrative_{mode}"
        vc = spread[col].value_counts()
        eff = eff_n(spread[col])
        print(f"  {mode:<7} distinct narratives={vc.size:<3} "
              f"EFFECTIVE independent narratives={eff:.2f}  (of {len(spread)} wallets)")
        for n, k in vc.items():
            print(f"            {k:>3}  {n}")
    flagged = int(spread["single_narrative_flag"].sum())
    print(f"  single-narrative wallets (>=80% one narrative, broad map): "
          f"{flagged} of {len(spread)}")
    print(f"  median top-narrative share: strict "
          f"{spread['top_narrative_share_strict'].median():.2f}  broad "
          f"{spread['top_narrative_share_broad'].median():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import src.slow_validate as sv

    print(f"[narrative map] version {NARRATIVE_MAP_VERSION}")
    bets = sv.load_deep_slow_bets_niched()
    resid, _ = sv.residualize_for(bets, sv.STANDARD_BASELINE)
    resid = assign_narratives(resid)

    primary_t = sv.validate_slow(resid, baselines={}, pre_residualized=True,
                                 cluster_col=sv.STANDARD_CLUSTER, complex_gates=True)
    secondary_t = sv.validate_slow(resid, baselines={}, pre_residualized=True,
                                   cluster_col=sv.STANDARD_CLUSTER, complex_gates=False)
    primary = set(primary_t.query("edge_persisted")["wallet"])
    secondary = set(secondary_t.query("edge_persisted")["wallet"])
    print(f"[cohorts] primary (complex-unit gates)={len(primary)}  "
          f"secondary (complex clusters only)={len(secondary)}")

    out = []
    for label, cohort in (("PRIMARY", primary), ("SECONDARY", secondary)):
        spread = per_wallet_spread(resid, cohort)
        print(f"\n[{label}] per-wallet narrative spread ({len(spread)} wallets)")
        show = spread.copy()
        show["wallet"] = show["wallet"].str[:12]
        with pd.option_context("display.width", 240, "display.max_rows", None):
            print(show[["wallet", "out_n", "n_complexes", "eff_complexes",
                        "n_narratives_broad", "eff_narratives_broad",
                        "top_narrative_broad", "top_narrative_share_broad",
                        "top_narrative_share_strict", "single_narrative_flag"]]
                  .round(3).to_string(index=False))
        set_level_summary(spread, label)
        out.append(spread.assign(cohort=label))

    if args.save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(pd.concat(out, ignore_index=True), OUT_PATH, compression="gzip")
        print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
