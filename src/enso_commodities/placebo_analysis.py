from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_research_config, project_root
from .placebo import (
    build_placebo_endpoints,
    draw_calendar_matched_anchors,
    eligible_neutral_anchors,
    evaluate_placebo_means,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import benjamini_hochberg, salted_seed
from .universe import load_commodity_registry


def _read_real_summary(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs")
    return summary


def _verify_endpoint_reconstruction(
    reconstructed: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    return_column: str,
) -> float:
    expected = sample.loc[
        sample[return_column].notna(), ["anchor_date", "commodity", return_column]
    ].copy()
    comparison = expected.merge(
        reconstructed.loc[:, ["anchor_date", "commodity", "placebo_return"]],
        on=["anchor_date", "commodity"],
        how="left",
        validate="one_to_one",
    )
    if comparison["placebo_return"].isna().any():
        raise ValueError("Could not reconstruct every observed endpoint from monthly returns")
    difference = comparison[return_column].sub(comparison["placebo_return"]).abs()
    maximum = float(difference.max()) if len(difference) else 0.0
    if maximum > 1e-12:
        raise ValueError(f"Observed endpoint reconstruction differs by {maximum}")
    return maximum


def run_neutral_date_placebo(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    config_path = research_config_path or project_root() / "config" / "research.yaml"
    config = load_research_config(config_path)
    registry = load_commodity_registry(registry_path)
    spec = registry.inference

    processed_summary_path = snapshot / "summary.json"
    adjusted_summary_path = output_dir / "adjusted_event_summary.json"
    inference_summary_path = output_dir / "inference_summary.json"
    processed_summary = _read_real_summary(processed_summary_path, "Neutral-date placebo")
    adjusted_summary = _read_real_summary(adjusted_summary_path, "Neutral-date placebo")
    inference_summary = _read_real_summary(inference_summary_path, "Neutral-date placebo")

    processed_hashes = processed_summary.get("output_hashes")
    adjusted_hashes = adjusted_summary.get("output_hashes")
    inference_hashes = inference_summary.get("output_hashes")
    if not isinstance(processed_hashes, dict):
        raise ValueError("The processed-data output-hash receipt is missing")
    if not isinstance(adjusted_hashes, dict):
        raise ValueError("The adjusted-return output-hash receipt is missing")
    if not isinstance(inference_hashes, dict):
        raise ValueError("A required upstream output-hash receipt is missing")
    enso_name = "enso_monthly.csv"
    monthly_name = "commodity_returns_adjusted_monthly.parquet"
    primary_name = "primary_inference_results.csv"
    control_name = "negative_control_inference.csv"
    verify_hashes(snapshot, {enso_name: processed_hashes[enso_name]})
    verify_hashes(output_dir, {monthly_name: adjusted_hashes[monthly_name]})
    verify_hashes(
        output_dir,
        {
            primary_name: inference_hashes[primary_name],
            control_name: inference_hashes[control_name],
        },
    )
    if inference_summary.get("input_hashes", {}).get("research.yaml") != sha256_file(config_path):
        raise ValueError("Inference results were not built with the current research configuration")

    episodes_path = output_dir / "enso_episodes.csv"
    primary_sample_path = output_dir / "primary_inference_family.parquet"
    control_sample_path = output_dir / "negative_control_sample.parquet"
    enso = pd.read_csv(snapshot / enso_name, parse_dates=["date"])
    episodes = pd.read_csv(episodes_path, parse_dates=["onset_date"])
    monthly = pd.read_parquet(output_dir / monthly_name)
    primary_observed = pd.read_csv(output_dir / primary_name)
    control_observed = pd.read_csv(output_dir / control_name)
    primary_sample = pd.read_parquet(primary_sample_path)
    control_sample = pd.read_parquet(control_sample_path)

    real_episode_ids = sorted(primary_sample["episode_id"].unique())
    real_episodes = episodes.loc[episodes["episode_id"].isin(real_episode_ids)].copy()
    if len(real_episodes) != len(real_episode_ids):
        raise ValueError("Primary episode identifiers do not match the episode table")
    first_anchor = pd.to_datetime(monthly["date"]).min()
    last_anchor = pd.to_datetime(monthly["date"]).max() - pd.DateOffset(
        months=spec.primary_horizon_months
    )
    eligible = eligible_neutral_anchors(
        enso,
        episodes,
        index_name=config.primary_index,
        neutral_absolute_threshold=config.placebo_neutral_absolute_threshold,
        exclusion_window=config.placebo_actual_event_exclusion_window,
        first_anchor=first_anchor,
        last_anchor=last_anchor,
    )
    commodities = sorted(set(primary_observed["commodity"]) | set(control_observed["commodity"]))
    endpoints = build_placebo_endpoints(
        monthly,
        eligible["date"],
        commodities,
        horizon_months=spec.primary_horizon_months,
    )
    reconstructed = build_placebo_endpoints(
        monthly,
        real_episodes["onset_date"],
        commodities,
        horizon_months=spec.primary_horizon_months,
    )
    reconstruction_difference = max(
        _verify_endpoint_reconstruction(
            reconstructed,
            primary_sample,
            return_column=spec.primary_return,
        ),
        _verify_endpoint_reconstruction(
            reconstructed,
            control_sample,
            return_column=spec.primary_return,
        ),
    )

    primary_draws = draw_calendar_matched_anchors(
        eligible,
        real_episodes["onset_date"],
        replicates=config.placebo_replicates,
        seed=salted_seed(config.random_seed, "primary_neutral_date_placebo"),
    )
    control_draws = draw_calendar_matched_anchors(
        eligible,
        real_episodes["onset_date"],
        replicates=config.placebo_replicates,
        seed=salted_seed(config.random_seed, "negative_control_neutral_date_placebo"),
    )
    primary_placebo = evaluate_placebo_means(
        endpoints.loc[endpoints["commodity"].isin(primary_observed["commodity"])],
        primary_draws,
        primary_observed,
        replicates=config.placebo_replicates,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
        confidence_level=config.confidence_level,
    )
    control_placebo = evaluate_placebo_means(
        endpoints.loc[endpoints["commodity"].isin(control_observed["commodity"])],
        control_draws,
        control_observed,
        replicates=config.placebo_replicates,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
        confidence_level=config.confidence_level,
    )

    primary_results = primary_observed.merge(
        primary_placebo.results,
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    primary_results["placebo_bh_q_value"] = benjamini_hochberg(primary_results["placebo_p_value"])
    primary_results["reject_placebo_fdr"] = primary_results["placebo_bh_q_value"].le(
        config.fdr_alpha
    )
    primary_results["passes_current_gates"] = (
        primary_results["reject_fdr"]
        & primary_results["sign_agreement"]
        & primary_results["reject_placebo_fdr"]
    )
    control_results = control_observed.merge(
        control_placebo.results,
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    control_results["diagnostic_placebo_bh_q_value"] = benjamini_hochberg(
        control_results["placebo_p_value"]
    )
    control_results["reject_placebo_raw"] = control_results["placebo_p_value"].lt(0.05)

    paths = {
        "eligible": output_dir / "placebo_eligible_anchors.csv",
        "endpoints": output_dir / "placebo_endpoints.parquet",
        "primary_draws": output_dir / "primary_placebo_draws.parquet",
        "control_draws": output_dir / "negative_control_placebo_draws.parquet",
        "primary_replicates": output_dir / "primary_placebo_replicates.parquet",
        "control_replicates": output_dir / "negative_control_placebo_replicates.parquet",
        "primary_results": output_dir / "primary_placebo_results.csv",
        "control_results": output_dir / "negative_control_placebo_results.csv",
    }
    eligible.to_csv(paths["eligible"], index=False, date_format="%Y-%m-%d")
    endpoints.to_parquet(paths["endpoints"], index=False)
    primary_draws.to_parquet(paths["primary_draws"], index=False)
    control_draws.to_parquet(paths["control_draws"], index=False)
    primary_placebo.replicates.to_parquet(paths["primary_replicates"], index=False)
    control_placebo.replicates.to_parquet(paths["control_replicates"], index=False)
    primary_results.sort_values("commodity").to_csv(paths["primary_results"], index=False)
    control_results.sort_values("commodity").to_csv(paths["control_results"], index=False)

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "placebo_contract": {
            "actual_event_exclusion_window": list(config.placebo_actual_event_exclusion_window),
            "alternative": "equal_tailed_two_sided",
            "match_anchor_calendar_month": config.placebo_match_anchor_calendar_month,
            "minimum_valid_episodes": spec.minimum_valid_episodes,
            "minimum_valid_replicate_share": config.placebo_minimum_valid_replicate_share,
            "neutral_absolute_threshold": config.placebo_neutral_absolute_threshold,
            "primary_horizon_months": spec.primary_horizon_months,
            "replicates": config.placebo_replicates,
            "sample_without_replacement": config.placebo_sample_without_replacement,
        },
        "diagnostics": {
            "eligible_neutral_anchors": len(eligible),
            "endpoint_reconstruction_max_abs_difference": reconstruction_difference,
            "negative_controls_rejecting_raw_placebo": int(
                control_results["reject_placebo_raw"].sum()
            ),
            "primary_candidates_with_estimated_placebo": int(
                primary_results["placebo_status"].eq("estimated").sum()
            ),
            "primary_candidates_with_insufficient_placebo_replicates": int(
                primary_results["placebo_status"].ne("estimated").sum()
            ),
            "primary_candidates_passing_current_gates": int(
                primary_results["passes_current_gates"].sum()
            ),
            "primary_candidates_rejecting_placebo_fdr": int(
                primary_results["reject_placebo_fdr"].sum()
            ),
            "real_episode_count": len(real_episodes),
        },
        "input_hashes": {
            adjusted_summary_path.name: sha256_file(adjusted_summary_path),
            monthly_name: sha256_file(output_dir / monthly_name),
            control_name: sha256_file(output_dir / control_name),
            enso_name: sha256_file(snapshot / enso_name),
            episodes_path.name: sha256_file(episodes_path),
            inference_summary_path.name: sha256_file(inference_summary_path),
            primary_name: sha256_file(output_dir / primary_name),
            "research.yaml": sha256_file(config_path),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
        "random_seed": config.random_seed,
    }
    write_json_atomic(output_dir / "placebo_summary.json", summary)
    return output_dir
