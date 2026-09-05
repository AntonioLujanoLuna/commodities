from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BootstrapOutput:
    results: pd.DataFrame
    replicates: pd.DataFrame


def salted_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(float("nan"), index=p_values.index, dtype="float64")
    if valid.empty:
        return result
    ordered = valid.sort_values(kind="mergesort")
    count = len(ordered)
    raw_adjusted = [float(value) * count / rank for rank, value in enumerate(ordered, start=1)]
    monotone = raw_adjusted.copy()
    for position in range(count - 2, -1, -1):
        monotone[position] = min(monotone[position], monotone[position + 1])
    result.loc[ordered.index] = [min(value, 1.0) for value in monotone]
    return result


def bootstrap_episode_means(
    data: pd.DataFrame,
    *,
    value_column: str,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> BootstrapOutput:
    required = {"episode_id", "commodity", value_column}
    if not required.issubset(data.columns):
        raise ValueError(f"Inference data is missing columns: {sorted(required - set(data))}")
    if data.duplicated(["episode_id", "commodity"]).any():
        raise ValueError("Inference data has duplicate episode/commodity keys")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must fall between zero and one")

    clean = data.loc[data[value_column].notna(), ["episode_id", "commodity", value_column]].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="raise")
    episode_ids = sorted(clean["episode_id"].unique())
    commodities = sorted(clean["commodity"].unique())
    if not episode_ids or not commodities:
        raise ValueError("Inference data has no valid observations")
    values: dict[str, dict[str, float]] = {}
    observed_means: dict[str, float] = {}
    for commodity, group in clean.groupby("commodity", sort=True, observed=True):
        commodity_values = dict(
            zip(group["episode_id"], group[value_column].astype(float), strict=True)
        )
        values[str(commodity)] = commodity_values
        observed_means[str(commodity)] = sum(commodity_values.values()) / len(commodity_values)

    rng = random.Random(seed)
    replicate_rows: list[dict[str, object]] = []
    null_extreme_counts = dict.fromkeys(commodities, 0)
    for replicate in range(replicates):
        draw = rng.choices(episode_ids, k=len(episode_ids))
        for commodity in commodities:
            sampled = [
                values[commodity][episode] for episode in draw if episode in values[commodity]
            ]
            if not sampled:
                continue
            bootstrap_mean = sum(sampled) / len(sampled)
            null_mean = sum(value - observed_means[commodity] for value in sampled) / len(sampled)
            if abs(null_mean) >= abs(observed_means[commodity]):
                null_extreme_counts[commodity] += 1
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "commodity": commodity,
                    "sampled_observations": len(sampled),
                    "bootstrap_mean": bootstrap_mean,
                    "null_bootstrap_mean": null_mean,
                }
            )
    replicate_table = pd.DataFrame.from_records(replicate_rows)
    alpha = 1 - confidence_level
    result_rows: list[dict[str, object]] = []
    for commodity in commodities:
        sample = clean.loc[clean["commodity"].eq(commodity), value_column]
        bootstrap_sample = replicate_table.loc[
            replicate_table["commodity"].eq(commodity), "bootstrap_mean"
        ]
        mean = float(sample.mean())
        median = float(sample.median())
        positive_share = float(sample.gt(0).mean())
        direction = "positive" if mean > 0 else "negative" if mean < 0 else "zero"
        sign_agreement = (mean > 0 and median > 0 and positive_share > 0.5) or (
            mean < 0 and median < 0 and positive_share < 0.5
        )
        result_rows.append(
            {
                "commodity": commodity,
                "episodes": len(sample),
                "draw_universe_episodes": len(episode_ids),
                "mean_return": mean,
                "median_return": median,
                "positive_share": positive_share,
                "direction": direction,
                "sign_agreement": sign_agreement,
                "confidence_level": confidence_level,
                "ci_lower": float(bootstrap_sample.quantile(alpha / 2)),
                "ci_upper": float(bootstrap_sample.quantile(1 - alpha / 2)),
                "bootstrap_p_value": (null_extreme_counts[commodity] + 1) / (replicates + 1),
            }
        )
    return BootstrapOutput(
        results=pd.DataFrame.from_records(result_rows),
        replicates=replicate_table,
    )
