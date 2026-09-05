from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .adjustments import add_adjusted_event_paths, adjust_monthly_returns
from .config import ResearchConfig, load_research_config, project_root
from .enso import construct_warm_episodes
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
from .returns import build_event_return_paths
from .robustness import summarize_candidate_robustness, summarize_control_robustness
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed
from .universe import InferenceSpec, load_commodity_registry


@dataclass(frozen=True)
class SpecificationOutput:
    candidate_results: pd.DataFrame
    control_results: pd.DataFrame
    bootstrap_replicates: pd.DataFrame
    placebo_draws: pd.DataFrame
    placebo_replicates: pd.DataFrame


def _real_summary(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    if result.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs")
    return result


def _add_specification_columns(
    data: pd.DataFrame,
    *,
    index_definition: str,
    anchor_type: str,
    family: str | None = None,
) -> pd.DataFrame:
    result = data.copy()
    result.insert(0, "index_definition", index_definition)
    result.insert(1, "anchor_type", anchor_type)
    if family is not None:
        result.insert(2, "family", family)
    return result


def _infer_specification(
    endpoint: pd.DataFrame,
    adjusted_monthly: pd.DataFrame,
    enso: pd.DataFrame,
    episodes: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    control_metadata: pd.DataFrame,
    full_candidate_directions: dict[str, str],
    full_control_directions: dict[str, str],
    *,
    index_definition: str,
    anchor_type: str,
    config: ResearchConfig,
    spec: InferenceSpec,
) -> SpecificationOutput:
    return_column = "macro_adjusted_cumulative_return"
    candidates = endpoint.merge(
        candidate_metadata, on="commodity", how="inner", validate="many_to_one"
    )
    controls = endpoint.merge(control_metadata, on="commodity", how="inner", validate="many_to_one")
    label = f"{index_definition}:{anchor_type}"
    candidate_bootstrap = bootstrap_episode_means(
        candidates,
        value_column=return_column,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        seed=salted_seed(config.random_seed, f"robustness_candidates:{label}"),
        p_value_method=config.p_value_method,
        minimum_studentized_replicate_share=config.minimum_studentized_replicate_share,
    )
    control_bootstrap = bootstrap_episode_means(
        controls,
        value_column=return_column,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        seed=salted_seed(config.random_seed, f"robustness_controls:{label}"),
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
    candidate_inference["stable_direction"] = candidate_inference.apply(
        lambda row: row["direction"] == full_candidate_directions[row["commodity"]], axis=1
    )
    control_inference = control_bootstrap.results.merge(
        control_metadata, on="commodity", how="left", validate="one_to_one"
    )
    control_inference["stable_direction"] = control_inference.apply(
        lambda row: row["direction"] == full_control_directions[row["commodity"]], axis=1
    )

    real_episode_ids = sorted(
        candidates.loc[candidates[return_column].notna(), "episode_id"].unique()
    )
    real_episodes = episodes.loc[episodes["episode_id"].isin(real_episode_ids)].copy()
    anchor_column = "onset_date" if anchor_type == "retrospective" else "observable_date"
    valid_dates = (
        adjusted_monthly.loc[adjusted_monthly["macro_adjusted_log_return"].notna(), "date"]
        .drop_duplicates()
        .sort_values()
    )
    first_anchor = pd.Timestamp(valid_dates.min())
    last_anchor = pd.Timestamp(valid_dates.max()) - pd.DateOffset(
        months=spec.primary_horizon_months
    )
    eligible_anchors = eligible_neutral_anchors(
        enso,
        episodes,
        index_name=index_definition,
        neutral_absolute_threshold=config.placebo_neutral_absolute_threshold,
        exclusion_window=config.placebo_actual_event_exclusion_window,
        first_anchor=first_anchor,
        last_anchor=last_anchor,
    )
    commodities = sorted(set(candidate_metadata["commodity"]) | set(control_metadata["commodity"]))
    placebo_endpoints = build_placebo_endpoints(
        adjusted_monthly,
        eligible_anchors["date"],
        commodities,
        horizon_months=spec.primary_horizon_months,
        monthly_return_column="macro_adjusted_log_return",
    )
    active_anchors = real_episodes[anchor_column]
    candidate_draws = draw_calendar_matched_anchors(
        eligible_anchors,
        active_anchors,
        replicates=config.placebo_replicates,
        seed=salted_seed(config.random_seed, f"robustness_placebo_candidates:{label}"),
        sample_without_replacement=config.robustness_placebo_sample_without_replacement,
    )
    control_draws = draw_calendar_matched_anchors(
        eligible_anchors,
        active_anchors,
        replicates=config.placebo_replicates,
        seed=salted_seed(config.random_seed, f"robustness_placebo_controls:{label}"),
        sample_without_replacement=config.robustness_placebo_sample_without_replacement,
    )
    candidate_placebo = evaluate_placebo_means(
        placebo_endpoints.loc[placebo_endpoints["commodity"].isin(candidate_metadata["commodity"])],
        candidate_draws,
        candidate_inference,
        replicates=config.placebo_replicates,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
        confidence_level=config.confidence_level,
        sample_without_replacement=config.robustness_placebo_sample_without_replacement,
    )
    control_placebo = evaluate_placebo_means(
        placebo_endpoints.loc[placebo_endpoints["commodity"].isin(control_metadata["commodity"])],
        control_draws,
        control_inference,
        replicates=config.placebo_replicates,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
        confidence_level=config.confidence_level,
        sample_without_replacement=config.robustness_placebo_sample_without_replacement,
    )
    candidate_results = candidate_inference.merge(
        candidate_placebo.results,
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    candidate_results["placebo_bh_q_value"] = benjamini_hochberg(
        candidate_results["placebo_p_value"]
    )
    candidate_results["reject_placebo_fdr"] = candidate_results["placebo_bh_q_value"].le(
        config.fdr_alpha
    )
    candidate_results["passes_specification"] = (
        candidate_results["eligible"]
        & candidate_results["stable_direction"]
        & candidate_results["sign_agreement"]
        & candidate_results["reject_bootstrap_fdr"]
        & candidate_results["reject_placebo_fdr"]
    )
    candidate_results["real_episode_count"] = len(real_episodes)
    candidate_results["eligible_neutral_anchors"] = len(eligible_anchors)
    control_results = control_inference.merge(
        control_placebo.results,
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    control_results["reject_bootstrap_raw"] = control_results["bootstrap_p_value"].lt(0.05)
    control_results["reject_placebo_raw"] = control_results["placebo_p_value"].lt(0.05)
    control_results["real_episode_count"] = len(real_episodes)
    control_results["eligible_neutral_anchors"] = len(eligible_anchors)

    bootstrap_replicates = pd.concat(
        [
            _add_specification_columns(
                candidate_bootstrap.replicates,
                index_definition=index_definition,
                anchor_type=anchor_type,
                family="candidates",
            ),
            _add_specification_columns(
                control_bootstrap.replicates,
                index_definition=index_definition,
                anchor_type=anchor_type,
                family="controls",
            ),
        ],
        ignore_index=True,
    )
    placebo_draws = pd.concat(
        [
            _add_specification_columns(
                candidate_draws,
                index_definition=index_definition,
                anchor_type=anchor_type,
                family="candidates",
            ),
            _add_specification_columns(
                control_draws,
                index_definition=index_definition,
                anchor_type=anchor_type,
                family="controls",
            ),
        ],
        ignore_index=True,
    )
    placebo_replicates = pd.concat(
        [
            _add_specification_columns(
                candidate_placebo.replicates,
                index_definition=index_definition,
                anchor_type=anchor_type,
                family="candidates",
            ),
            _add_specification_columns(
                control_placebo.replicates,
                index_definition=index_definition,
                anchor_type=anchor_type,
                family="controls",
            ),
        ],
        ignore_index=True,
    )
    return SpecificationOutput(
        candidate_results=_add_specification_columns(
            candidate_results, index_definition=index_definition, anchor_type=anchor_type
        ),
        control_results=_add_specification_columns(
            control_results, index_definition=index_definition, anchor_type=anchor_type
        ),
        bootstrap_replicates=bootstrap_replicates,
        placebo_draws=placebo_draws,
        placebo_replicates=placebo_replicates,
    )


def run_timing_index_robustness(
    processed_snapshot: Path | None = None,
    macro_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    root = project_root()
    snapshot = processed_snapshot or latest_processed_snapshot()
    macro_input = macro_snapshot or latest_macro_snapshot(root / "data" / "macro" / "processed")
    output_dir = (tables_root or root / "tables") / snapshot.name
    config_path = research_config_path or root / "config" / "research.yaml"
    config = load_research_config(config_path)
    spec = load_commodity_registry(registry_path).inference

    processed_summary_path = snapshot / "summary.json"
    raw_summary_path = output_dir / "raw_event_summary.json"
    macro_summary_path = macro_input / "summary.json"
    fragility_run_path = output_dir / "macro_fragility_run_summary.json"
    processed_summary = _real_summary(processed_summary_path, "Timing/index robustness")
    raw_summary = _real_summary(raw_summary_path, "Timing/index robustness")
    macro_summary = _real_summary(macro_summary_path, "Timing/index robustness")
    fragility_run = _real_summary(fragility_run_path, "Timing/index robustness")
    processed_hashes = processed_summary.get("output_hashes")
    raw_hashes = raw_summary.get("output_hashes")
    macro_hashes = macro_summary.get("output_hashes")
    fragility_hashes = fragility_run.get("output_hashes")
    for label, hashes in [
        ("processed", processed_hashes),
        ("raw", raw_hashes),
        ("macro", macro_hashes),
        ("fragility", fragility_hashes),
    ]:
        if not isinstance(hashes, dict):
            raise ValueError(f"The {label} output-hash receipt is missing")
    assert isinstance(processed_hashes, dict)
    assert isinstance(raw_hashes, dict)
    assert isinstance(macro_hashes, dict)
    assert isinstance(fragility_hashes, dict)
    if fragility_run.get("input_hashes", {}).get("research.yaml") != sha256_file(config_path):
        raise ValueError("Fragility analysis was not built with the current research configuration")

    enso_name = "enso_monthly.csv"
    market_name = "world_bank_indices_monthly.csv"
    monthly_name = "commodity_returns_monthly.parquet"
    macro_name = "macro_controls_monthly.csv"
    fragility_name = "macro_fragility_summary.csv"
    control_fragility_name = "macro_control_fragility_summary.csv"
    verify_hashes(
        snapshot,
        {enso_name: processed_hashes[enso_name], market_name: processed_hashes[market_name]},
    )
    verify_hashes(output_dir, {monthly_name: raw_hashes[monthly_name]})
    verify_hashes(macro_input, {macro_name: macro_hashes[macro_name]})
    verify_hashes(
        output_dir,
        {
            fragility_name: fragility_hashes[fragility_name],
            control_fragility_name: fragility_hashes[control_fragility_name],
        },
    )

    enso = pd.read_csv(snapshot / enso_name, parse_dates=["date"])
    market = pd.read_csv(snapshot / market_name, parse_dates=["date"])
    monthly = pd.read_parquet(output_dir / monthly_name)
    macro = pd.read_csv(macro_input / macro_name, parse_dates=["date"])
    fragility = pd.read_csv(output_dir / fragility_name)
    primary_macro = pd.read_csv(output_dir / "macro_primary_results.csv")
    control_macro = pd.read_csv(output_dir / "macro_control_results.csv")
    candidate_metadata = primary_macro.loc[:, ["commodity", "macro_group"]].rename(
        columns={"macro_group": "group"}
    )
    control_metadata = control_macro.loc[:, ["commodity", "macro_group"]].rename(
        columns={"macro_group": "group"}
    )
    full_candidate_directions = primary_macro.set_index("commodity")["macro_direction"].to_dict()
    full_control_directions = control_macro.set_index("commodity")["macro_direction"].to_dict()

    candidate_results: list[pd.DataFrame] = []
    control_results: list[pd.DataFrame] = []
    bootstrap_replicates: list[pd.DataFrame] = []
    placebo_draws: list[pd.DataFrame] = []
    placebo_replicates: list[pd.DataFrame] = []
    adjusted_monthly_outputs: list[pd.DataFrame] = []
    event_path_outputs: list[pd.DataFrame] = []
    episode_outputs: list[pd.DataFrame] = []
    model_outputs: list[pd.DataFrame] = []
    for index_definition in config.robustness_index_definitions:
        episodes = construct_warm_episodes(
            enso,
            index_name=index_definition,
            threshold=config.warm_threshold,
            minimum_duration_months=config.minimum_duration_months,
            observable_delay_after_center_months=config.observable_delay_after_center_months,
        )
        seasonal_adjusted, _, _, _ = adjust_monthly_returns(
            monthly,
            market,
            episodes,
            market_exclusion_window=config.market_estimation_exclusion_window,
            minimum_seasonal_observations=config.minimum_seasonal_observations,
            minimum_market_model_observations=config.minimum_market_model_observations,
        )
        macro_adjusted, models = fit_macro_adjusted_returns(
            seasonal_adjusted,
            macro,
            episodes,
            control_columns=config.macro_control_columns,
            estimation_exclusion_window=config.macro_estimation_exclusion_window,
            minimum_model_observations=config.minimum_macro_model_observations,
        )
        event_paths = build_event_return_paths(
            monthly,
            episodes,
            first_relative_month=config.first_relative_month,
            last_relative_month=config.last_relative_month,
            base_relative_month=config.base_relative_month,
            anchor_types=config.robustness_anchor_types,
        )
        adjusted_paths = add_adjusted_event_paths(
            event_paths,
            macro_adjusted,
            base_relative_month=config.base_relative_month,
            monthly_columns=["macro_adjusted_log_return"],
        )
        stored_monthly = macro_adjusted.copy()
        stored_monthly.insert(0, "index_definition", index_definition)
        adjusted_monthly_outputs.append(stored_monthly)
        stored_paths = adjusted_paths.copy()
        stored_paths.insert(0, "index_definition", index_definition)
        event_path_outputs.append(stored_paths)
        episodes = episodes.copy()
        episodes.insert(0, "index_definition", index_definition)
        episode_outputs.append(episodes)
        models = models.copy()
        models.insert(0, "index_definition", index_definition)
        model_outputs.append(models)

        for anchor_type in config.robustness_anchor_types:
            endpoint = adjusted_paths.loc[
                adjusted_paths["anchor_type"].eq(anchor_type)
                & adjusted_paths["relative_month"].eq(spec.primary_horizon_months)
            ]
            output = _infer_specification(
                endpoint,
                macro_adjusted,
                enso,
                episodes.drop(columns="index_definition"),
                candidate_metadata,
                control_metadata,
                full_candidate_directions,
                full_control_directions,
                index_definition=index_definition,
                anchor_type=anchor_type,
                config=config,
                spec=spec,
            )
            candidate_results.append(output.candidate_results)
            control_results.append(output.control_results)
            bootstrap_replicates.append(output.bootstrap_replicates)
            placebo_draws.append(output.placebo_draws)
            placebo_replicates.append(output.placebo_replicates)

    candidate_table = pd.concat(candidate_results, ignore_index=True)
    control_table = pd.concat(control_results, ignore_index=True)
    expected_specifications = len(config.robustness_index_definitions) * len(
        config.robustness_anchor_types
    )
    candidate_summary = summarize_candidate_robustness(
        candidate_table, fragility, expected_specifications=expected_specifications
    )
    control_summary = summarize_control_robustness(
        control_table, expected_specifications=expected_specifications
    )
    baseline = candidate_table.loc[
        candidate_table["index_definition"].eq("roni")
        & candidate_table["anchor_type"].eq("retrospective"),
        ["commodity", "mean_return"],
    ].merge(
        primary_macro.loc[:, ["commodity", "macro_mean_return"]],
        on="commodity",
        validate="one_to_one",
    )
    baseline_error = float(baseline["mean_return"].sub(baseline["macro_mean_return"]).abs().max())
    if baseline_error > 1e-12:
        raise ValueError(f"RONI retrospective baseline reconstruction differs by {baseline_error}")

    paths = {
        "episodes": output_dir / "robustness_episodes.csv",
        "models": output_dir / "robustness_macro_models.csv",
        "monthly": output_dir / "robustness_macro_adjusted_monthly.parquet",
        "event_paths": output_dir / "robustness_event_paths.parquet",
        "candidate_results": output_dir / "robustness_candidate_results.csv",
        "control_results": output_dir / "robustness_control_results.csv",
        "candidate_summary": output_dir / "robustness_candidate_summary.csv",
        "control_summary": output_dir / "robustness_control_summary.csv",
        "bootstrap_replicates": output_dir / "robustness_bootstrap_replicates.parquet",
        "placebo_draws": output_dir / "robustness_placebo_draws.parquet",
        "placebo_replicates": output_dir / "robustness_placebo_replicates.parquet",
    }
    pd.concat(episode_outputs, ignore_index=True).to_csv(
        paths["episodes"], index=False, date_format="%Y-%m-%d"
    )
    pd.concat(model_outputs, ignore_index=True).to_csv(paths["models"], index=False)
    pd.concat(adjusted_monthly_outputs, ignore_index=True).to_parquet(paths["monthly"], index=False)
    pd.concat(event_path_outputs, ignore_index=True).to_parquet(paths["event_paths"], index=False)
    candidate_table.sort_values(["index_definition", "anchor_type", "commodity"]).to_csv(
        paths["candidate_results"], index=False
    )
    control_table.sort_values(["index_definition", "anchor_type", "commodity"]).to_csv(
        paths["control_results"], index=False
    )
    candidate_summary.to_csv(paths["candidate_summary"], index=False)
    control_summary.to_csv(paths["control_summary"], index=False)
    pd.concat(bootstrap_replicates, ignore_index=True).to_parquet(
        paths["bootstrap_replicates"], index=False
    )
    pd.concat(placebo_draws, ignore_index=True).to_parquet(paths["placebo_draws"], index=False)
    pd.concat(placebo_replicates, ignore_index=True).to_parquet(
        paths["placebo_replicates"], index=False
    )

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "contract": {
            "anchor_types": list(config.robustness_anchor_types),
            "index_definitions": list(config.robustness_index_definitions),
            "required_gates": [
                "primary_fragility_pass",
                "stable_direction",
                "sign_agreement",
                "bootstrap_fdr",
                "placebo_fdr",
            ],
            "specifications": expected_specifications,
            "placebo_sample_without_replacement": (
                config.robustness_placebo_sample_without_replacement
            ),
        },
        "diagnostics": {
            "baseline_mean_reconstruction_max_abs_difference": baseline_error,
            "candidate_result_rows": len(candidate_table),
            "candidates_passing_all_specifications_and_fragility": int(
                candidate_summary["passes_timing_index_robustness"].sum()
            ),
            "controls_failing_specificity_everywhere": int(
                control_summary["fails_specificity_everywhere"].sum()
            ),
            "control_result_rows": len(control_table),
        },
        "input_hashes": {
            fragility_run_path.name: sha256_file(fragility_run_path),
            fragility_name: sha256_file(output_dir / fragility_name),
            control_fragility_name: sha256_file(output_dir / control_fragility_name),
            "macro_primary_results.csv": sha256_file(output_dir / "macro_primary_results.csv"),
            "macro_control_results.csv": sha256_file(output_dir / "macro_control_results.csv"),
            enso_name: sha256_file(snapshot / enso_name),
            macro_name: sha256_file(macro_input / macro_name),
            market_name: sha256_file(snapshot / market_name),
            monthly_name: sha256_file(output_dir / monthly_name),
            "research.yaml": sha256_file(config_path),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
        "random_seed": config.random_seed,
    }
    write_json_atomic(output_dir / "robustness_summary.json", summary)
    return output_dir
