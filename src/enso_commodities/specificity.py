from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .statistics import benjamini_hochberg


@dataclass(frozen=True)
class ContrastOutput:
    results: pd.DataFrame
    replicates: pd.DataFrame


def randomization_direction_contrast(
    data: pd.DataFrame, *, value_column: str, replicates: int, seed: int
) -> ContrastOutput:
    required = {"episode_id", "direction", "commodity", value_column}
    if not required.issubset(data.columns):
        raise ValueError(f"Contrast data is missing columns: {sorted(required - set(data))}")
    if data.duplicated(["episode_id", "commodity"]).any():
        raise ValueError("Contrast data has duplicate episode/commodity keys")
    if set(data["direction"].dropna().unique()) != {"warm", "cold"}:
        raise ValueError("Contrast requires warm and cold observations")
    clean = data.loc[data[value_column].notna()].copy()
    episode_direction = clean.loc[:, ["episode_id", "direction"]].drop_duplicates()
    if episode_direction["episode_id"].duplicated().any():
        raise ValueError("Each episode must have one direction")
    episode_ids = sorted(episode_direction["episode_id"].astype(str))
    warm_count = int(episode_direction["direction"].eq("warm").sum())
    values = {
        str(commodity): dict(zip(group["episode_id"], group[value_column].astype(float), strict=True))
        for commodity, group in clean.groupby("commodity", sort=True, observed=True)
    }
    observed: dict[str, float] = {}
    for commodity, group in clean.groupby("commodity", sort=True, observed=True):
        means = group.groupby("direction", observed=True)[value_column].mean()
        observed[str(commodity)] = float(means["warm"] - means["cold"])

    rng = random.Random(seed)
    extreme = dict.fromkeys(values, 0)
    valid = dict.fromkeys(values, 0)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        assigned_warm = set(rng.sample(episode_ids, k=warm_count))
        for commodity, lookup in values.items():
            warm = [value for episode, value in lookup.items() if episode in assigned_warm]
            cold = [value for episode, value in lookup.items() if episode not in assigned_warm]
            if not warm or not cold:
                continue
            difference = sum(warm) / len(warm) - sum(cold) / len(cold)
            valid[commodity] += 1
            if abs(difference) >= abs(observed[commodity]):
                extreme[commodity] += 1
            rows.append(
                {
                    "replicate": replicate,
                    "commodity": commodity,
                    "warm_observations": len(warm),
                    "cold_observations": len(cold),
                    "randomized_mean_difference": difference,
                }
            )
    result_rows = []
    for commodity, group in clean.groupby("commodity", sort=True, observed=True):
        means = group.groupby("direction", observed=True)[value_column].mean()
        result_rows.append(
            {
                "commodity": commodity,
                "warm_episodes": int(group["direction"].eq("warm").sum()),
                "cold_episodes": int(group["direction"].eq("cold").sum()),
                "warm_mean_return": float(means["warm"]),
                "cold_mean_return": float(means["cold"]),
                "warm_minus_cold": observed[str(commodity)],
                "opposite_mean_directions": bool(means["warm"] * means["cold"] < 0),
                "valid_randomization_replicates": valid[str(commodity)],
                "contrast_p_value": (extreme[str(commodity)] + 1) / (valid[str(commodity)] + 1),
            }
        )
    return ContrastOutput(pd.DataFrame(result_rows), pd.DataFrame(rows))


def add_candidate_fdr(results: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    output = results.copy()
    output["contrast_bh_q_value"] = benjamini_hochberg(output["contrast_p_value"])
    output["reject_direction_contrast_fdr"] = output["contrast_bh_q_value"].le(alpha)
    return output


def control_episode_influence(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in endpoints.groupby(
        ["index_definition", "anchor_type", "direction", "commodity"], observed=True
    ):
        clean = group.loc[group["value"].notna()]
        if len(clean) < 2:
            continue
        full_mean = float(clean["value"].mean())
        for row in clean.itertuples(index=False):
            leave_one_out = float((clean["value"].sum() - row.value) / (len(clean) - 1))
            rows.append(
                {
                    "index_definition": keys[0],
                    "anchor_type": keys[1],
                    "direction": keys[2],
                    "commodity": keys[3],
                    "episode_id": row.episode_id,
                    "anchor_date": row.anchor_date,
                    "episode_return": row.value,
                    "full_mean_return": full_mean,
                    "leave_one_out_mean_return": leave_one_out,
                    "mean_change_when_deleted": leave_one_out - full_mean,
                }
            )
    return pd.DataFrame(rows)
