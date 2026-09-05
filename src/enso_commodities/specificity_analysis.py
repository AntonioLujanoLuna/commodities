from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adjustments import add_adjusted_event_paths
from .config import project_root
from .enso import construct_cold_episodes
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .returns import build_event_return_paths
from .specificity import (
    add_candidate_fdr,
    control_episode_influence,
    randomization_direction_contrast,
)
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("comparison") != "warm_minus_cold":
        raise ValueError("Specificity comparison must be warm_minus_cold")
    if config.get("fdr_family") != "mechanism_candidates_only":
        raise ValueError("Specificity FDR must be confined to mechanism candidates")
    if float(config["cold_threshold"]) >= 0:
        raise ValueError("Specificity cold_threshold must be negative")
    return config


def _real_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError("Specificity diagnostics require real-data inputs")
    return summary


def _family_results(
    endpoints: pd.DataFrame,
    commodities: set[str],
    *,
    index_definition: str,
    anchor_type: str,
    config: dict[str, Any],
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = endpoints.loc[endpoints["commodity"].isin(commodities)].copy()
    contrast = randomization_direction_contrast(
        sample,
        value_column="value",
        replicates=int(config["randomization_replicates"]),
        seed=salted_seed(
            int(config["random_seed"]),
            f"specificity_contrast:{family}:{index_definition}:{anchor_type}",
        ),
    )
    cold = sample.loc[sample["direction"].eq("cold")]
    cold_bootstrap = bootstrap_episode_means(
        cold,
        value_column="value",
        replicates=int(config["randomization_replicates"]),
        confidence_level=float(config["confidence_level"]),
        seed=salted_seed(
            int(config["random_seed"]),
            f"specificity_cold:{family}:{index_definition}:{anchor_type}",
        ),
    ).results.rename(
        columns={
            "mean_return": "cold_bootstrap_mean_return",
            "bootstrap_p_value": "cold_bootstrap_p_value",
            "direction": "cold_direction",
            "sign_agreement": "cold_sign_agreement",
        }
    )
    keep = [
        "commodity",
        "cold_bootstrap_mean_return",
        "cold_bootstrap_p_value",
        "cold_direction",
        "cold_sign_agreement",
    ]
    result = contrast.results.merge(cold_bootstrap[keep], on="commodity", validate="one_to_one")
    if family == "candidates":
        result = add_candidate_fdr(result, alpha=float(config["fdr_alpha"]))
        result["cold_bootstrap_bh_q_value"] = benjamini_hochberg(
            result["cold_bootstrap_p_value"]
        )
        result["reject_cold_bootstrap_fdr"] = result["cold_bootstrap_bh_q_value"].le(
            float(config["fdr_alpha"])
        )
    else:
        result["reject_direction_contrast_raw"] = result["contrast_p_value"].le(
            float(config["fdr_alpha"])
        )
        result["reject_cold_bootstrap_raw"] = result["cold_bootstrap_p_value"].le(
            float(config["fdr_alpha"])
        )
    result.insert(0, "anchor_type", anchor_type)
    result.insert(0, "index_definition", index_definition)
    replicates = contrast.replicates
    replicates.insert(0, "family", family)
    replicates.insert(0, "anchor_type", anchor_type)
    replicates.insert(0, "index_definition", index_definition)
    return result, replicates


def run_specificity_diagnostics(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    specificity_config_path: Path | None = None,
) -> Path:
    root = project_root()
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or root / "tables") / snapshot.name
    config_path = specificity_config_path or root / "config" / "specificity.yaml"
    config = _load_config(config_path)
    robustness_path = output_dir / "robustness_summary.json"
    receipt = _real_summary(robustness_path)
    output_hashes = receipt.get("output_hashes")
    if not isinstance(output_hashes, dict):
        raise ValueError("Robustness output-hash receipt is missing")
    verify_hashes(output_dir, output_hashes)
    monthly_name = "commodity_returns_monthly.parquet"
    expected_monthly_hash = receipt.get("input_hashes", {}).get(monthly_name)
    if not isinstance(expected_monthly_hash, str):
        raise ValueError("Robustness monthly-return input hash is missing")
    verify_hashes(output_dir, {monthly_name: expected_monthly_hash})

    enso = pd.read_csv(snapshot / "enso_monthly.csv", parse_dates=["date"])
    monthly = pd.read_parquet(output_dir / monthly_name)
    adjusted = pd.read_parquet(output_dir / "robustness_macro_adjusted_monthly.parquet")
    warm_paths = pd.read_parquet(output_dir / "robustness_event_paths.parquet")
    candidate_prior = pd.read_csv(output_dir / "robustness_candidate_results.csv")
    control_prior = pd.read_csv(output_dir / "robustness_control_results.csv")
    candidates = set(candidate_prior["commodity"].unique())
    controls = set(control_prior["commodity"].unique())

    episode_outputs: list[pd.DataFrame] = []
    endpoint_outputs: list[pd.DataFrame] = []
    candidate_outputs: list[pd.DataFrame] = []
    control_outputs: list[pd.DataFrame] = []
    replicate_outputs: list[pd.DataFrame] = []
    for index_definition in config["index_definitions"]:
        cold_episodes = construct_cold_episodes(
            enso,
            index_name=str(index_definition),
            threshold=float(config["cold_threshold"]),
            minimum_duration_months=int(config["minimum_duration_months"]),
            observable_delay_after_center_months=int(
                config["observable_delay_after_center_months"]
            ),
        )
        stored_episodes = cold_episodes.copy()
        stored_episodes.insert(0, "index_definition", index_definition)
        episode_outputs.append(stored_episodes)
        cold_paths = build_event_return_paths(
            monthly,
            cold_episodes,
            first_relative_month=-12,
            last_relative_month=24,
            base_relative_month=-1,
            anchor_types=tuple(config["anchor_types"]),
        )
        index_adjusted = adjusted.loc[adjusted["index_definition"].eq(index_definition)]
        cold_paths = add_adjusted_event_paths(
            cold_paths,
            index_adjusted,
            base_relative_month=-1,
            monthly_columns=["macro_adjusted_log_return"],
        )
        for anchor_type in config["anchor_types"]:
            warm = warm_paths.loc[
                warm_paths["index_definition"].eq(index_definition)
                & warm_paths["anchor_type"].eq(anchor_type)
                & warm_paths["relative_month"].eq(int(config["horizon_months"])),
                ["episode_id", "anchor_date", "commodity", "macro_adjusted_cumulative_return"],
            ].copy()
            warm["direction"] = "warm"
            cold = cold_paths.loc[
                cold_paths["anchor_type"].eq(anchor_type)
                & cold_paths["relative_month"].eq(int(config["horizon_months"])),
                ["episode_id", "anchor_date", "commodity", "macro_adjusted_cumulative_return"],
            ].copy()
            cold["direction"] = "cold"
            endpoints = pd.concat([warm, cold], ignore_index=True).rename(
                columns={"macro_adjusted_cumulative_return": "value"}
            )
            endpoints.insert(0, "anchor_type", anchor_type)
            endpoints.insert(0, "index_definition", index_definition)
            endpoint_outputs.append(endpoints)
            for family, names, sink in [
                ("candidates", candidates, candidate_outputs),
                ("controls", controls, control_outputs),
            ]:
                result, replicates = _family_results(
                    endpoints,
                    names,
                    index_definition=str(index_definition),
                    anchor_type=str(anchor_type),
                    config=config,
                    family=family,
                )
                sink.append(result)
                replicate_outputs.append(replicates)

    endpoint_table = pd.concat(endpoint_outputs, ignore_index=True)
    candidate_table = pd.concat(candidate_outputs, ignore_index=True)
    control_table = pd.concat(control_outputs, ignore_index=True)
    influence = control_episode_influence(endpoint_table.loc[endpoint_table["commodity"].isin(controls)])
    paths = {
        "episodes": output_dir / "specificity_cold_episodes.csv",
        "endpoints": output_dir / "specificity_event_endpoints.parquet",
        "candidates": output_dir / "specificity_candidate_results.csv",
        "controls": output_dir / "specificity_control_results.csv",
        "replicates": output_dir / "specificity_contrast_replicates.parquet",
        "influence": output_dir / "specificity_control_episode_influence.csv",
    }
    pd.concat(episode_outputs, ignore_index=True).to_csv(paths["episodes"], index=False, date_format="%Y-%m-%d")
    endpoint_table.to_parquet(paths["endpoints"], index=False)
    candidate_table.to_csv(paths["candidates"], index=False)
    control_table.to_csv(paths["controls"], index=False)
    pd.concat(replicate_outputs, ignore_index=True).to_parquet(paths["replicates"], index=False)
    influence.to_csv(paths["influence"], index=False, date_format="%Y-%m-%d")
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "scope": "exploratory_falsification_not_primary_design",
        "contract": config,
        "diagnostics": {
            "cold_episode_counts": pd.concat(episode_outputs).groupby("index_definition").size().to_dict(),
            "candidate_rows": len(candidate_table),
            "control_rows": len(control_table),
            "controls_with_direction_specific_contrast_cells": int(control_table["reject_direction_contrast_raw"].sum()),
        },
        "input_hashes": {
            robustness_path.name: sha256_file(robustness_path),
            config_path.name: sha256_file(config_path),
            "enso_monthly.csv": sha256_file(snapshot / "enso_monthly.csv"),
            monthly_name: sha256_file(output_dir / monthly_name),
            **{name: sha256_file(output_dir / name) for name in [
                "robustness_macro_adjusted_monthly.parquet",
                "robustness_event_paths.parquet",
                "robustness_candidate_results.csv",
                "robustness_control_results.csv",
            ]},
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
    }
    write_json_atomic(output_dir / "specificity_summary.json", summary)
    return output_dir
