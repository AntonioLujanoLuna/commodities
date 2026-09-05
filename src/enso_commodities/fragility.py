from __future__ import annotations

import pandas as pd


def summarize_candidate_fragility(
    scenarios: pd.DataFrame,
    full_results: pd.DataFrame,
) -> pd.DataFrame:
    required_scenarios = {
        "commodity",
        "deleted_episode_id",
        "mean_return",
        "sign_agreement",
        "stable_direction",
        "bootstrap_bh_q_value",
        "reject_bootstrap_fdr",
    }
    required_full = {"commodity", "macro_mean_return", "passes_macro_gates"}
    if not required_scenarios.issubset(scenarios.columns):
        raise ValueError(
            f"Fragility scenarios are missing columns: {sorted(required_scenarios - set(scenarios))}"
        )
    if not required_full.issubset(full_results.columns):
        raise ValueError(
            f"Full macro results are missing columns: {sorted(required_full - set(full_results))}"
        )
    if scenarios.duplicated(["commodity", "deleted_episode_id"]).any():
        raise ValueError("Fragility scenario keys must be unique")

    full = full_results.loc[:, ["commodity", "macro_mean_return", "passes_macro_gates"]].copy()
    merged = scenarios.merge(full, on="commodity", how="left", validate="many_to_one")
    merged["absolute_change_from_full"] = (
        merged["mean_return"].sub(merged["macro_mean_return"]).abs()
    )
    rows: list[dict[str, object]] = []
    for commodity, group in merged.groupby("commodity", sort=True, observed=True):
        influential = group.loc[group["absolute_change_from_full"].idxmax()]
        deletions = len(group)
        rejections = int(group["reject_bootstrap_fdr"].sum())
        sign_agreement_all = bool(group["sign_agreement"].all())
        stable_direction_all = bool(group["stable_direction"].all())
        full_passes = bool(group["passes_macro_gates"].iloc[0])
        rows.append(
            {
                "commodity": commodity,
                "full_macro_mean_return": float(group["macro_mean_return"].iloc[0]),
                "deletions_tested": deletions,
                "loo_min_mean_return": float(group["mean_return"].min()),
                "loo_max_mean_return": float(group["mean_return"].max()),
                "loo_max_abs_change_from_full": float(group["absolute_change_from_full"].max()),
                "most_influential_episode_id": influential["deleted_episode_id"],
                "loo_sign_agreement_all": sign_agreement_all,
                "loo_stable_direction_all": stable_direction_all,
                "loo_bootstrap_fdr_rejections": rejections,
                "loo_worst_bootstrap_q_value": float(group["bootstrap_bh_q_value"].max()),
                "full_passes_macro_gates": full_passes,
                "survives_all_deletions": (
                    full_passes
                    and rejections == deletions
                    and sign_agreement_all
                    and stable_direction_all
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_control_fragility(
    scenarios: pd.DataFrame,
    full_results: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "commodity",
        "deleted_episode_id",
        "mean_return",
        "bootstrap_p_value",
        "stable_direction",
    }
    if not required.issubset(scenarios.columns):
        raise ValueError(
            f"Control scenarios are missing columns: {sorted(required - set(scenarios))}"
        )
    full = full_results.loc[:, ["commodity", "macro_mean_return"]]
    merged = scenarios.merge(full, on="commodity", how="left", validate="many_to_one")
    merged["absolute_change_from_full"] = (
        merged["mean_return"].sub(merged["macro_mean_return"]).abs()
    )
    rows: list[dict[str, object]] = []
    for commodity, group in merged.groupby("commodity", sort=True, observed=True):
        influential = group.loc[group["absolute_change_from_full"].idxmax()]
        deletions = len(group)
        raw_rejections = int(group["bootstrap_p_value"].lt(0.05).sum())
        rows.append(
            {
                "commodity": commodity,
                "full_macro_mean_return": float(group["macro_mean_return"].iloc[0]),
                "deletions_tested": deletions,
                "loo_min_mean_return": float(group["mean_return"].min()),
                "loo_max_mean_return": float(group["mean_return"].max()),
                "loo_max_abs_change_from_full": float(group["absolute_change_from_full"].max()),
                "most_influential_episode_id": influential["deleted_episode_id"],
                "loo_stable_direction_all": bool(group["stable_direction"].all()),
                "loo_raw_bootstrap_rejections": raw_rejections,
                "rejects_after_every_deletion": raw_rejections == deletions,
            }
        )
    return pd.DataFrame.from_records(rows)
