"""Event-weighted aggregate scoring for the forward test — Project 3 step 8b.

WHY AGGREGATE, AND WHY EVENT-WEIGHTED
-------------------------------------
Three post-freeze bets cannot grade a wallet. 3xN bets CAN grade a crowd, because
wallets selected by luck wash to zero forward while a real crowd edge survives.
So the unit of inference is the TIER, never the individual wallet.

Within a tier, the naive aggregate — the mean residual over all bets — is wrong
for this data, because bets are not independent draws. The slow tape is dominated
by rolling-deadline ladders: one hyperactive event complex can contribute
thousands of bets and simply become the answer. The headline statistic is
therefore the EVENT-WEIGHTED mean:

    edge_event_weighted = mean over event complexes of (mean residual in complex)

so every event counts once regardless of how many bets it generated. The
bet-weighted mean is reported next to it; a large gap between them is itself
diagnostic that one event is dominating.

Uncertainty resamples EVENTS, not bets (the repo-wide cluster-preserving
standard, applied at the unit step 7 established as correct for this path). With
few events the CI is wide and that is the honest answer — which is exactly why
`effective_events` is reported beside every bet count. A tier with 50,000 bets
across 3 complexes has the evidentiary weight of roughly 3 observations, and the
scoreboard must never let that read as 50,000.

READ-ONLY / ANALYSIS-ONLY: pure functions over an already-scored frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP = 2000
DEFAULT_CI = 0.95


def effective_events(events) -> float:
    """1 / HHI over per-event bet shares — the number of events the sample
    effectively contains, as opposed to the number it nominally contains."""
    s = pd.Series(list(events)).value_counts(normalize=True).to_numpy()
    return float(1.0 / np.sum(s ** 2)) if s.size else 0.0


def event_weighted_mean(residuals: np.ndarray, events: np.ndarray) -> float:
    """Mean over events of the within-event mean residual."""
    if residuals.size == 0:
        return float("nan")
    df = pd.DataFrame({"r": residuals, "e": events})
    return float(df.groupby("e")["r"].mean().mean())


def event_block_bootstrap(residuals: np.ndarray, events: np.ndarray,
                          rng: np.random.Generator,
                          n_boot: int = DEFAULT_BOOTSTRAP,
                          ci: float = DEFAULT_CI) -> dict:
    """Resample EVENT COMPLEXES with replacement (bets within an event travel
    together, since they share one resolution) and recompute the event-weighted
    mean each time.

    Returns the CI, a one-sided p for "edge <= 0", and the number of events. With
    fewer than 2 distinct events nothing is estimable and the p is NaN — the same
    by-rule refusal the retrospective path uses, not a silent 0."""
    out = {"n_events": 0, "ci_low": float("nan"), "ci_high": float("nan"),
           "p_one_sided": float("nan"), "boot_se": float("nan")}
    if residuals.size == 0:
        return out

    df = pd.DataFrame({"r": residuals, "e": events})
    groups = [g["r"].to_numpy() for _, g in df.groupby("e", sort=True)]
    k = len(groups)
    out["n_events"] = k
    if k < 2:
        return out

    means = np.array([g.mean() for g in groups], dtype=float)
    obs = float(means.mean())
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = means[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    out["ci_low"] = float(np.quantile(boot, lo))
    out["ci_high"] = float(np.quantile(boot, 1.0 - lo))
    out["boot_se"] = float(np.std(boot, ddof=1))
    # one-sided: how often does the resampled crowd edge fail to exceed zero,
    # centred on the observed estimate (the standard bootstrap-p construction)
    out["p_one_sided"] = float(np.mean((boot - obs) >= obs))
    return out


def score_group(bets: pd.DataFrame, event_col: str = "niche_l1",
                residual_col: str = "residual_skill",
                seed: int = 0, n_boot: int = DEFAULT_BOOTSTRAP) -> dict:
    """The full aggregate readout for one group of post-freeze bets."""
    empty = {"n_bets": 0, "n_wallets": 0, "n_events": 0, "effective_events": 0.0,
             "edge_event_weighted": float("nan"), "edge_bet_weighted": float("nan"),
             "ci_low": float("nan"), "ci_high": float("nan"),
             "p_one_sided": float("nan")}
    if bets.empty or residual_col not in bets.columns or event_col not in bets.columns:
        return empty
    r_all = bets[residual_col].to_numpy(dtype=float)
    ok = ~np.isnan(r_all)
    r = r_all[ok]
    e = bets[event_col].to_numpy()[ok]
    if r.size == 0:
        return empty
    rng = np.random.default_rng(seed)
    boot = event_block_bootstrap(r, e, rng, n_boot=n_boot)
    return {
        "n_bets": int(r.size),
        # positional, NOT .loc — callers routinely pass filtered frames whose
        # index is no longer 0..n-1
        "n_wallets": int(bets["wallet"].to_numpy()[ok].size and
                         pd.unique(bets["wallet"].to_numpy()[ok]).size)
        if "wallet" in bets.columns else 0,
        "n_events": int(boot["n_events"]),
        "effective_events": effective_events(e),
        "edge_event_weighted": event_weighted_mean(r, e),
        "edge_bet_weighted": float(np.mean(r)),
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "p_one_sided": boot["p_one_sided"],
    }


def resolution_speed_stratum(bets: pd.DataFrame, split_days: int) -> pd.Series:
    """`fast` when a bet resolved within `split_days` of ENTRY, else `slow`.

    Keyed on the market's own speed rather than on time since the freeze, so a
    bet's stratum never changes between scoring runs."""
    days = (bets["resolution_ts"].to_numpy(dtype=float)
            - bets["timestamp"].to_numpy(dtype=float)) / 86400.0
    return pd.Series(np.where(days <= split_days, "fast", "slow"), index=bets.index)


def score_tier(bets: pd.DataFrame, tier: str, split_days: int,
               event_col: str = "niche_l1", n_boot: int = DEFAULT_BOOTSTRAP) -> list[dict]:
    """Aggregate readout for a tier: overall, then each resolution-speed stratum,
    then the finer (complex x resolution-week) event unit as a secondary view.

    The coarse complex is the HEADLINE event unit; the finer one is reported so a
    reader can see how much of the uncertainty is the unit choice, never so that
    significance can be harvested from the looser definition."""
    seed = abs(hash(tier)) % (2 ** 32)
    rows = []
    base = dict(tier=tier, event_unit=event_col)

    rows.append({**base, "stratum": "all",
                 **score_group(bets, event_col, seed=seed, n_boot=n_boot)})
    if "resolution_ts" in bets.columns:
        strat = resolution_speed_stratum(bets, split_days)
        for name in ("fast", "slow"):
            sub = bets[strat == name]
            if len(sub):
                rows.append({**base, "stratum": f"{name} (<={split_days}d)"
                             if name == "fast" else f"{name} (>{split_days}d)",
                             **score_group(sub, event_col, seed=seed + 1, n_boot=n_boot)})
    return rows


def add_fine_event_column(bets: pd.DataFrame, event_col: str = "niche_l1") -> pd.DataFrame:
    """`event_fine` = (complex, ISO resolution week). The same complex resolving in
    March and in June carries genuinely different information, so this is a
    defensible secondary unit — reported alongside, never as the headline."""
    out = bets.copy()
    wk = pd.to_datetime(out["resolution_ts"], unit="s").dt.strftime("%G-W%V")
    out["event_fine"] = out[event_col].astype(str) + "@" + wk
    return out
