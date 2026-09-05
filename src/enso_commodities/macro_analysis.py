from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .adjustments import add_adjusted_event_paths
from .config import load_research_config, project_root
from .macro_adjustments import fit_macro_adjusted_returns
from .macro_data import latest_macro_snapshot
from .placebo import (
    build_placebo_endpoints,
    draw_calendar_matched_anchors,
    eligible_neutral_anchors,
    evaluate_placebo_means,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed
from .universe import load_commodity_registry


def _load_real_summary(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs")
    return summary


def _prefix_columns(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return data.rename(
        columns={column: f"{prefix}{column}" for column in data.columns if column != "commodity"}
    )


def _endpoint_reconstruction_error(
    endpoint: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    return_column: str,
) -> float:
    expected = endpoint.loc[
        endpoint[return_column].notna(), ["anchor_date", "commodity", return_column]
    ]
    comparison = expected.merge(
        reconstructed.loc[:, ["anchor_date", "commodity", "placebo_return"]],
        on=["anchor_date", "commodity"],
        how="left",
        validate="one_to_one",
    )
    if comparison["placebo_return"].isna().any():
        raise ValueError("Could not reconstruct every macro-adjusted event endpoint")
    difference = comparison[return_column].sub(comparison["placebo_return"]).abs()
    maximum = float(difference.max()) if len(difference) else 0.0
    if maximum > 1e-12:
        raise ValueError(f"Macro endpoint reconstruction differs by {maximum}")
    return maximum


def run_macro_control_analysis(
    processed_snapshot: Path | None = None,
    macro_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    macro_input = macro_snapshot or latest_macro_snapshot(
        project_root() / "data" / "macro" / "processed"
    )
    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    config_path = research_config_path or project_root() / "config" / "research.yaml"
    config = load_research_config(config_path)
    spec = load_commodity_registry(registry_path).inference

    adjusted_summary_path = output_dir / "adjusted_event_summary.json"
    raw_summary_path = output_dir / "raw_event_summary.json"
    inference_summary_path = output_dir / "inference_summary.json"
    macro_summary_path = macro_input / "summary.json"
    adjusted_summary = _load_real_summary(adjusted_summary_path, "Macro-control analysis")
    raw_summary = _load_real_summary(raw_summary_path, "Macro-control analysis")
    inference_summary = _load_real_summary(inference_summary_path, "Macro-control analysis")
    macro_summary = _load_real_summary(macro_summary_path, "Macro-control analysis")
    adjusted_hashes = adjusted_summary.get("output_hashes")
    raw_hashes = raw_summary.get("output_hashes")
    inference_hashes = inference_summary.get("output_hashes")
    macro_hashes = macro_summary.get("output_hashes")
    for label, hashes in [
        ("adjusted", adjusted_hashes),
        ("raw", raw_hashes),
        ("inference", inference_hashes),
        ("macro", macro_hashes),
    ]:
        if not isinstance(hashes, dict):
            raise ValueError(f"The {label} output-hash receipt is missing")
    assert isinstance(adjusted_hashes, dict)
    assert isinstance(raw_hashes, dict)
    assert isinstance(inference_hashes, dict)
    assert isinstance(macro_hashes, dict)

    monthly_name = "commodity_returns_adjusted_monthly.parquet"
    paths_name = "raw_event_return_paths.parquet"
    primary_name = "primary_inference_results.csv"
    control_name = "negative_control_inference.csv"
    macro_name = "macro_controls_monthly.csv"
    verify_hashes(output_dir, {monthly_name: adjusted_hashes[monthly_name]})
    verify_hashes(output_dir, {paths_name: raw_hashes[paths_name]})
    verify_hashes(
        output_dir,
        {
            primary_name: inference_hashes[primary_name],
            control_name: inference_hashes[control_name],
        },
    )
    verify_hashes(macro_input, {macro_name: macro_hashes[macro_name]})
    if inference_summary.get("input_hashes", {}).get("research.yaml") != sha256_file(config_path):
        raise ValueError("Inference results were not built with the current research configuration")

    episodes_path = output_dir / "enso_episodes.csv"
    enso_path = snapshot / "enso_monthly.csv"
    monthly = pd.read_parquet(output_dir / monthly_name)
    event_paths = pd.read_parquet(output_dir / paths_name)
    episodes = pd.read_csv(episodes_path, parse_dates=["onset_date"])
    enso = pd.read_csv(enso_path, parse_dates=["date"])
    macro = pd.read_csv(macro_input / macro_name, parse_dates=["date"])
    prior_primary = pd.read_csv(output_dir / primary_name)
    prior_controls = pd.read_csv(output_dir / control_name)

    adjusted, models = fit_macro_adjusted_returns(
        monthly,
        macro,
        episodes,
        control_columns=config.macro_control_columns,
        estimation_exclusion_window=config.macro_estimation_exclusion_window,
        minimum_model_observations=config.minimum_macro_model_observations,
    )
    adjusted_paths = add_adjusted_event_paths(
        event_paths,
        adjusted,
        base_relative_month=config.base_relative_month,
        monthly_columns=["macro_adjusted_log_return"],
    )
    endpoint = adjusted_paths.loc[
        adjusted_paths["anchor_type"].eq(spec.primary_anchor)
        & adjusted_paths["relative_month"].eq(spec.primary_horizon_months)
    ].copy()
    candidate_metadata = prior_primary.loc[:, ["commodity", "group"]]
    control_metadata = prior_controls.loc[:, ["commodity", "group"]]
    candidates = endpoint.merge(
        candidate_metadata, on="commodity", how="inner", validate="many_to_one"
    )
    controls = endpoint.merge(control_metadata, on="commodity", how="inner", validate="many_to_one")
    return_column = "macro_adjusted_cumulative_return"

    candidate_bootstrap = bootstrap_episode_means(
        candidates,
        value_column=return_column,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        seed=salted_seed(config.random_seed, "macro_primary_candidates"),
        p_value_method=config.p_value_method,
        minimum_studentized_replicate_share=config.minimum_studentized_replicate_share,
    )
    control_bootstrap = bootstrap_episode_means(
        controls,
        value_column=return_column,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        seed=salted_seed(config.random_seed, "macro_negative_controls"),
        p_value_method=config.p_value_method,
        minimum_studentized_replicate_share=config.minimum_studentized_replicate_share,
    )
    candidate_inference = candidate_bootstrap.results.merge(
        candidate_metadata, on="commodity", how="left", validate="one_to_one"
    )
    candidate_inference["eligible"] = candidate_inference["episodes"].ge(
        spec.minimum_valid_episodes
    )
    candidate_inference.loc[~candidate_inference["eligible"], "bootstrap_p_value"] = 1.0
    candidate_inference["bootstrap_bh_q_value"] = benjamini_hochberg(
        candidate_inference["bootstrap_p_value"]
    )
    candidate_inference["reject_bootstrap_fdr"] = candidate_inference["bootstrap_bh_q_value"].le(
        config.fdr_alpha
    )
    control_inference = control_bootstrap.results.merge(
        control_metadata, on="commodity", how="left", validate="one_to_one"
    )

    real_episode_ids = sorted(
        candidates.loc[candidates[return_column].notna(), "episode_id"].unique()
    )
    real_episodes = episodes.loc[episodes["episode_id"].isin(real_episode_ids)].copy()
    valid_macro_dates = (
        adjusted.loc[adjusted["macro_adjusted_log_return"].notna(), "date"]
        .drop_duplicates()
        .sort_values()
    )
    first_anchor = pd.Timestamp(valid_macro_dates.min())
    last_anchor = pd.Timestamp(valid_macro_dates.max()) - pd.DateOffset(
        months=spec.primary_horizon_months
    )
    eligible_anchors = eligible_neutral_anchors(
        enso,
        episodes,
        index_name=config.primary_index,
        neutral_absolute_threshold=config.placebo_neutral_absolute_threshold,
        exclusion_window=config.placebo_actual_event_exclusion_window,
        first_anchor=first_anchor,
        last_anchor=last_anchor,
    )
    all_commodities = sorted(
        set(candidate_inference["commodity"]) | set(control_inference["commodity"])
    )
    reconstructed = build_placebo_endpoints(
        adjusted,
        real_episodes["onset_date"],
        all_commodities,
        horizon_months=spec.primary_horizon_months,
        monthly_return_column="macro_adjusted_log_return",
    )
    reconstruction_error = _endpoint_reconstruction_error(
        endpoint.loc[
            endpoint["episode_id"].isin(real_episode_ids)
            & endpoint["commodity"].isin(all_commodities)
        ],
        reconstructed,
        return_column=return_column,
    )
    placebo_endpoints = build_placebo_endpoints(
        adjusted,
        eligible_anchors["date"],
        all_commodities,
        horizon_months=spec.primary_horizon_months,
        monthly_return_column="macro_adjusted_log_return",
    )
    primary_draws = draw_calendar_matched_anchors(
        eligible_anchors,
        real_episodes["onset_date"],
        replicates=config.placebo_replicates,
        seed=salted_seed(config.random_seed, "macro_primary_neutral_date_placebo"),
    )
    control_draws = draw_calendar_matched_anchors(
        eligible_anchors,
        real_episodes["onset_date"],
        replicates=config.placebo_replicates,
        seed=salted_seed(config.random_seed, "macro_control_neutral_date_placebo"),
    )
    primary_placebo = evaluate_placebo_means(
        placebo_endpoints.loc[placebo_endpoints["commodity"].isin(candidate_metadata["commodity"])],
        primary_draws,
        candidate_inference,
        replicates=config.placebo_replicates,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
        confidence_level=config.confidence_level,
    )
    control_placebo = evaluate_placebo_means(
        placebo_endpoints.loc[placebo_endpoints["commodity"].isin(control_metadata["commodity"])],
        control_draws,
        control_inference,
        replicates=config.placebo_replicates,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
        confidence_level=config.confidence_level,
    )

    primary_results = _prefix_columns(candidate_inference, "macro_").merge(
        _prefix_columns(primary_placebo.results, "macro_"),
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    primary_results["macro_placebo_bh_q_value"] = benjamini_hochberg(
        primary_results["macro_placebo_p_value"]
    )
    primary_results["macro_reject_placebo_fdr"] = primary_results["macro_placebo_bh_q_value"].le(
        config.fdr_alpha
    )
    primary_results["passes_macro_gates"] = (
        primary_results["macro_eligible"]
        & primary_results["macro_sign_agreement"]
        & primary_results["macro_reject_bootstrap_fdr"]
        & primary_results["macro_reject_placebo_fdr"]
    )
    control_results = _prefix_columns(control_inference, "macro_").merge(
        _prefix_columns(control_placebo.results, "macro_"),
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    control_results["macro_reject_bootstrap_raw"] = control_results["macro_bootstrap_p_value"].lt(
        0.05
    )
    control_results["macro_reject_placebo_raw"] = control_results["macro_placebo_p_value"].lt(0.05)

    paths = {
        "monthly": output_dir / "commodity_returns_macro_adjusted_monthly.parquet",
        "models": output_dir / "macro_models.csv",
        "event_paths": output_dir / "macro_adjusted_event_paths.parquet",
        "candidate_sample": output_dir / "macro_candidate_sample.parquet",
        "control_sample": output_dir / "macro_negative_control_sample.parquet",
        "eligible_anchors": output_dir / "macro_placebo_eligible_anchors.csv",
        "placebo_endpoints": output_dir / "macro_placebo_endpoints.parquet",
        "primary_draws": output_dir / "macro_primary_placebo_draws.parquet",
        "control_draws": output_dir / "macro_control_placebo_draws.parquet",
        "primary_bootstrap": output_dir / "macro_primary_bootstrap_replicates.parquet",
        "control_bootstrap": output_dir / "macro_control_bootstrap_replicates.parquet",
        "primary_placebo": output_dir / "macro_primary_placebo_replicates.parquet",
        "control_placebo": output_dir / "macro_control_placebo_replicates.parquet",
        "primary_results": output_dir / "macro_primary_results.csv",
        "control_results": output_dir / "macro_control_results.csv",
    }
    adjusted.to_parquet(paths["monthly"], index=False)
    models.to_csv(paths["models"], index=False)
    adjusted_paths.to_parquet(paths["event_paths"], index=False)
    candidates.to_parquet(paths["candidate_sample"], index=False)
    controls.to_parquet(paths["control_sample"], index=False)
    eligible_anchors.to_csv(paths["eligible_anchors"], index=False, date_format="%Y-%m-%d")
    placebo_endpoints.to_parquet(paths["placebo_endpoints"], index=False)
    primary_draws.to_parquet(paths["primary_draws"], index=False)
    control_draws.to_parquet(paths["control_draws"], index=False)
    candidate_bootstrap.replicates.to_parquet(paths["primary_bootstrap"], index=False)
    control_bootstrap.replicates.to_parquet(paths["control_bootstrap"], index=False)
    primary_placebo.replicates.to_parquet(paths["primary_placebo"], index=False)
    control_placebo.replicates.to_parquet(paths["control_placebo"], index=False)
    primary_results.sort_values("commodity").to_csv(paths["primary_results"], index=False)
    control_results.sort_values("commodity").to_csv(paths["control_results"], index=False)

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "macro_snapshot": macro_input.name,
        "contract": {
            "controls": list(config.macro_control_columns),
            "estimation_exclusion_window": list(config.macro_estimation_exclusion_window),
            "estimator": config.macro_estimator,
            "minimum_model_observations": config.minimum_macro_model_observations,
            "primary_horizon_months": spec.primary_horizon_months,
        },
        "diagnostics": {
            "eligible_neutral_anchors": len(eligible_anchors),
            "endpoint_reconstruction_max_abs_difference": reconstruction_error,
            "macro_models_estimated": int(models["macro_model_status"].eq("estimated").sum()),
            "macro_models_not_estimated": int(models["macro_model_status"].ne("estimated").sum()),
            "negative_controls_rejecting_bootstrap_raw": int(
                control_results["macro_reject_bootstrap_raw"].sum()
            ),
            "negative_controls_rejecting_placebo_raw": int(
                control_results["macro_reject_placebo_raw"].sum()
            ),
            "primary_candidates_passing_macro_gates": int(
                primary_results["passes_macro_gates"].sum()
            ),
            "primary_candidates_rejecting_bootstrap_fdr": int(
                primary_results["macro_reject_bootstrap_fdr"].sum()
            ),
            "primary_candidates_rejecting_placebo_fdr": int(
                primary_results["macro_reject_placebo_fdr"].sum()
            ),
            "primary_candidates_with_estimated_placebo": int(
                primary_results["macro_placebo_status"].eq("estimated").sum()
            ),
            "primary_macro_eligible_candidates": int(primary_results["macro_eligible"].sum()),
            "real_macro_episode_count": len(real_episodes),
        },
        "input_hashes": {
            adjusted_summary_path.name: sha256_file(adjusted_summary_path),
            enso_path.name: sha256_file(enso_path),
            episodes_path.name: sha256_file(episodes_path),
            inference_summary_path.name: sha256_file(inference_summary_path),
            macro_name: sha256_file(macro_input / macro_name),
            macro_summary_path.name: sha256_file(macro_summary_path),
            monthly_name: sha256_file(output_dir / monthly_name),
            paths_name: sha256_file(output_dir / paths_name),
            "research.yaml": sha256_file(config_path),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
        "random_seed": config.random_seed,
    }
    write_json_atomic(output_dir / "macro_analysis_summary.json", summary)
    return output_dir
