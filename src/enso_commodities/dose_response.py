"""Episode-amplitude dose-response diagnostic for the frozen event endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import project_root
from .statistics import benjamini_hochberg


@dataclass(frozen=True)
class DoseResponseSpec:
    config: dict[str, Any]
    outcome: str
    minimum_valid_episodes: int
    replicates: int
    random_seed: int
    fdr_alpha: float
    fwer_alpha: float


def load_dose_response_config(path: Path | None = None) -> DoseResponseSpec:
    config_path = path or project_root() / "config" / "dose_response.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["provenance"] != {
        "version": 1,
        "authored_on": date(2026, 9, 6),
        "inference_scope": "exploratory_secondary",
    }:
        raise ValueError("Dose-response provenance is not the frozen secondary contract")
    design = raw["design"]
    expected = {
        "exposure": "episode_peak_roni",
        "exposure_scaling": "z_score_across_warm_episodes",
        "outcome": "seasonal_adjusted_log_return",
        "estimator": "ordinary_least_squares_hc3",
        "alternative": "two_sided",
        "minimum_valid_episodes": 10,
    }
    if design != expected:
        raise ValueError("Dose-response design differs from the frozen secondary contract")
    inference = raw["inference"]
    if inference["permutation_unit"] != "whole_episode_amplitude_labels":
        raise ValueError("Dose response must permute whole-episode amplitude labels")
    replicates = int(inference["replicates"])
    if replicates < 99:
        raise ValueError("Dose-response inference requires at least 99 permutations")
    return DoseResponseSpec(
        config=raw,
        outcome=str(design["outcome"]),
        minimum_valid_episodes=int(design["minimum_valid_episodes"]),
        replicates=replicates,
        random_seed=int(inference["random_seed"]),
        fdr_alpha=float(inference["fdr_alpha"]),
        fwer_alpha=float(inference["fwer_alpha"]),
    )


def prepare_dose_panel(
    outcomes: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    outcome_column: str,
) -> pd.DataFrame:
    required_outcomes = {"episode_id", "commodity", "group", "role", outcome_column}
    missing = required_outcomes - set(outcomes.columns)
    if missing:
        raise ValueError(f"Dose-response outcomes are missing columns: {sorted(missing)}")
    required_episodes = {"episode_id", "index_name", "direction", "peak_value"}
    if required_episodes - set(episodes.columns):
        raise ValueError("Episode table lacks the peak-amplitude contract")
    if episodes["episode_id"].duplicated().any():
        raise ValueError("Episode amplitudes contain duplicate episode IDs")
    episode_frame = episodes.loc[
        episodes["index_name"].eq("roni") & episodes["direction"].eq("warm"),
        ["episode_id", "peak_value"],
    ].copy()
    if episode_frame.empty or episode_frame["peak_value"].isna().any():
        raise ValueError("Warm RONI episodes require complete peak amplitudes")
    scale = float(episode_frame["peak_value"].std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Episode peak amplitudes have no variation")
    episode_frame["peak_amplitude_z"] = (
        episode_frame["peak_value"] - episode_frame["peak_value"].mean()
    ) / scale
    panel = outcomes.merge(episode_frame, on="episode_id", how="inner", validate="many_to_one")
    if panel.duplicated(["episode_id", "commodity"]).any():
        raise ValueError("Dose-response panel has duplicate episode/commodity keys")
    return panel


def hc3_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, HC3 standard error, and t statistic for y ~ 1 + x."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype="float64")
    y = np.asarray(y[valid], dtype="float64")
    if len(y) < 4 or np.ptp(x) <= 0:
        return float("nan"), float("nan"), float("nan")
    design = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    leverage = np.sum((design @ inverse) * design, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-12, None)
    meat = (design * adjusted[:, None]).T @ (design * adjusted[:, None])
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = float(beta[1] / standard_error) if standard_error > 0 else float("nan")
    return float(beta[1]), standard_error, statistic


def amplitude_permutation_inference(
    panel: pd.DataFrame,
    *,
    outcome_column: str,
    minimum_valid_episodes: int,
    replicates: int,
    seed: int,
    fdr_alpha: float,
    fwer_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute one shared episode-amplitude mapping across every commodity."""
    episodes = sorted(panel["episode_id"].unique())
    amplitudes = (
        panel.loc[:, ["episode_id", "peak_amplitude_z"]]
        .drop_duplicates()
        .set_index("episode_id")
        .loc[episodes, "peak_amplitude_z"]
        .to_numpy(dtype="float64")
    )
    metadata = (
        panel.loc[:, ["commodity", "group", "role"]]
        .drop_duplicates()
        .sort_values("commodity", ignore_index=True)
    )
    if metadata["commodity"].duplicated().any():
        raise ValueError("Each commodity must have one group and role")
    outcome_matrix = panel.pivot(index="episode_id", columns="commodity", values=outcome_column)
    outcome_matrix = outcome_matrix.reindex(index=episodes, columns=metadata["commodity"])

    observed_rows: list[dict[str, Any]] = []
    observed_t = np.full(len(metadata), np.nan)
    for position, commodity in enumerate(metadata["commodity"]):
        y = outcome_matrix[commodity].to_numpy(dtype="float64")
        valid_count = int(np.isfinite(y).sum())
        slope, standard_error, statistic = hc3_slope(amplitudes, y)
        if valid_count < minimum_valid_episodes:
            slope = standard_error = statistic = float("nan")
        observed_t[position] = statistic
        observed_rows.append(
            {
                "commodity": commodity,
                "group": metadata.loc[position, "group"],
                "role": metadata.loc[position, "role"],
                "valid_episode_count": valid_count,
                "slope_per_peak_roni_sd": slope,
                "hc3_standard_error": standard_error,
                "t_statistic": statistic,
            }
        )

    generator = np.random.default_rng(seed)
    permutation_t = np.full((replicates, len(metadata)), np.nan)
    for replicate in range(replicates):
        permuted = generator.permutation(amplitudes)
        for position, commodity in enumerate(metadata["commodity"]):
            y = outcome_matrix[commodity].to_numpy(dtype="float64")
            if np.isfinite(y).sum() >= minimum_valid_episodes:
                permutation_t[replicate, position] = hc3_slope(permuted, y)[2]

    results = pd.DataFrame.from_records(observed_rows)
    p_values = []
    for position, statistic in enumerate(observed_t):
        null = permutation_t[:, position]
        valid = np.isfinite(null)
        p_values.append(
            (1 + int(np.sum(np.abs(null[valid]) >= abs(statistic)))) / (1 + int(valid.sum()))
            if np.isfinite(statistic) and valid.any()
            else float("nan")
        )
    results["permutation_p_value"] = p_values
    results["bh_q_value"] = np.nan
    candidates = results["role"].eq("mechanism_candidate")
    results.loc[candidates, "bh_q_value"] = benjamini_hochberg(
        results.loc[candidates].set_index("commodity")["permutation_p_value"]
    ).to_numpy()
    results["reject_fdr"] = results["bh_q_value"].le(fdr_alpha).fillna(False)

    candidate_positions = np.flatnonzero(candidates.to_numpy())
    maximum_t = np.nanmax(np.abs(permutation_t[:, candidate_positions]), axis=1)
    results["max_t_fwer_p_value"] = np.nan
    for candidate_position in candidate_positions:
        position = int(candidate_position)
        statistic = observed_t[position]
        if np.isfinite(statistic):
            results.loc[position, "max_t_fwer_p_value"] = (
                1 + int(np.sum(maximum_t >= abs(statistic)))
            ) / (1 + len(maximum_t))
    results["reject_fwer"] = results["max_t_fwer_p_value"].le(fwer_alpha).fillna(False)

    replicate_frame = pd.DataFrame(
        permutation_t,
        columns=metadata["commodity"],
    )
    replicate_frame.insert(0, "replicate", np.arange(replicates))
    replicate_frame = replicate_frame.melt(
        id_vars="replicate", var_name="commodity", value_name="permuted_t_statistic"
    )
    return results, replicate_frame
