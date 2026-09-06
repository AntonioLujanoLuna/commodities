"""Exploratory exposure-weighted panel stage.

Runs the two-way fixed-effects design over the index, lag and weighting grid,
and writes the same kind of hash-linked receipt as every other stage. This is a
separate identification strategy, not a revision of the frozen event-study
contract, and nothing here promotes or demotes a commodity in that contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .panel import (
    CONTROL_TERM,
    EXPOSURE_TERM,
    PRIMARY_BLOCK_COLUMN,
    PanelFit,
    PanelSpec,
    attach_resampling_blocks,
    build_exposure_table,
    build_panel,
    fit_exposure_panel,
    load_panel_config,
    panel_regressors,
    permute_candidate_exposure_weights,
    year_block_bootstrap,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import benjamini_hochberg, salted_seed
from .universe import load_commodity_registry


def _load_real_summary(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs: {path}")
    return summary


def run_panel_analysis(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    panel_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    config_path = panel_config_path or project_root() / "config" / "panel.yaml"
    spec = load_panel_config(config_path)
    registry = load_commodity_registry(registry_path)

    adjusted_summary_path = output_dir / "adjusted_event_summary.json"
    processed_summary_path = snapshot / "summary.json"
    adjusted_summary = _load_real_summary(adjusted_summary_path, "Panel analysis")
    processed_summary = _load_real_summary(processed_summary_path, "Panel analysis")
    adjusted_hashes = adjusted_summary.get("output_hashes")
    processed_hashes = processed_summary.get("output_hashes")
    if not isinstance(adjusted_hashes, dict) or not isinstance(processed_hashes, dict):
        raise ValueError("Panel analysis requires adjusted and processed output-hash receipts")
    monthly_name = "commodity_returns_adjusted_monthly.parquet"
    enso_name = "enso_monthly.csv"
    verify_hashes(output_dir, {monthly_name: adjusted_hashes[monthly_name]})
    verify_hashes(snapshot, {enso_name: processed_hashes[enso_name]})

    monthly = pd.read_parquet(output_dir / monthly_name)
    enso = pd.read_csv(snapshot / enso_name, parse_dates=["date"])
    exposure = build_exposure_table(registry, spec)

    results, replicates = _run_specification_grid(monthly, enso, exposure, spec)
    primary_panel = build_panel(
        monthly,
        enso,
        exposure,
        index_name=spec.primary_index,
        lag_months=spec.primary_lag_months,
        outcome_column=spec.outcome_column,
        weighting=spec.primary_weighting,
    )
    primary_fit = fit_exposure_panel(
        primary_panel,
        tolerance=spec.demeaning_tolerance,
        max_iterations=spec.demeaning_max_iterations,
        regressors=panel_regressors(spec.primary_weighting),
    )
    permutation = permute_candidate_exposure_weights(
        primary_panel,
        primary_fit,
        replicates=spec.weight_permutation_replicates,
        seed=salted_seed(spec.random_seed, "panel:exposure_weight_permutation"),
        tolerance=spec.demeaning_tolerance,
        max_iterations=spec.demeaning_max_iterations,
    )
    block_sensitivity = _run_block_sensitivity(primary_panel, primary_fit, spec)

    exposure_path = output_dir / "panel_exposure_weights.csv"
    results_path = output_dir / "panel_specification_results.csv"
    replicates_path = output_dir / "panel_bootstrap_replicates.parquet"
    permutation_path = output_dir / "panel_exposure_weight_permutations.parquet"
    block_sensitivity_path = output_dir / "panel_block_sensitivity.csv"
    exposure.to_csv(exposure_path, index=False)
    results.to_csv(results_path, index=False)
    replicates.to_parquet(replicates_path, index=False)
    permutation.replicates.to_parquet(permutation_path, index=False)
    block_sensitivity.to_csv(block_sensitivity_path, index=False)

    primary = results.loc[
        results["index_definition"].eq(spec.primary_index)
        & results["lag_months"].eq(spec.primary_lag_months)
        & results["weighting"].eq(spec.primary_weighting)
    ]
    primary_exposure = primary.loc[primary["term"].eq(EXPOSURE_TERM)]
    primary_control = primary.loc[primary["term"].eq(CONTROL_TERM)]
    exposure_cells = results.loc[results["term"].eq(EXPOSURE_TERM)]
    control_cells = results.loc[results["term"].eq(CONTROL_TERM)]

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "design": {
            "estimator": "two_way_within_ordinary_least_squares",
            "identification": "cross-sectional exposure interacted with the ENSO index, "
            "with commodity and calendar-month fixed effects",
            "outcome_column": spec.outcome_column,
            "exposure_version": spec.exposure_version,
            "exposure_method": spec.exposure_method,
            "exposure_authored_on": spec.exposure_authored_on.isoformat(),
            "exposure_outcome_blind": spec.exposure_outcome_blind,
            "exposure_inference_scope": spec.exposure_inference_scope,
            "primary_index": spec.primary_index,
            "primary_lag_months": spec.primary_lag_months,
            "primary_weighting": spec.primary_weighting,
            "resampling_block": spec.block,
            "resampling_block_sensitivity": [
                {
                    "name": definition.name,
                    "start_month": definition.start_month,
                    "length_years": definition.length_years,
                }
                for definition in spec.block_sensitivity
            ],
            "bootstrap_replicates": spec.bootstrap_replicates,
            "weight_permutation_replicates": spec.weight_permutation_replicates,
            "random_seed": spec.random_seed,
            "status": spec.exposure_inference_scope.removesuffix("_only"),
        },
        "input_hashes": {
            adjusted_summary_path.name: sha256_file(adjusted_summary_path),
            monthly_name: sha256_file(output_dir / monthly_name),
            enso_name: sha256_file(snapshot / enso_name),
            "commodities.yaml": sha256_file(
                registry_path or project_root() / "config" / "commodities.yaml"
            ),
            "panel.yaml": sha256_file(config_path),
        },
        "output_hashes": {
            block_sensitivity_path.name: sha256_file(block_sensitivity_path),
            exposure_path.name: sha256_file(exposure_path),
            permutation_path.name: sha256_file(permutation_path),
            replicates_path.name: sha256_file(replicates_path),
            results_path.name: sha256_file(results_path),
        },
        "results": {
            "cells": int(results["cell"].nunique()),
            "control_cells_rejecting_raw_5_percent": int(
                control_cells["studentized_p_value"].lt(0.05).sum()
            ),
            "exposure_cells_rejecting_fdr": int(exposure_cells["reject_fdr"].sum()),
            "exposure_cells_with_positive_estimate": int(exposure_cells["estimate"].gt(0).sum()),
            "primary_control_estimate": _scalar(primary_control, "estimate"),
            "primary_control_studentized_p_value": _scalar(primary_control, "studentized_p_value"),
            "primary_exposure_ci_lower": _scalar(primary_exposure, "ci_lower"),
            "primary_exposure_ci_upper": _scalar(primary_exposure, "ci_upper"),
            "primary_exposure_estimate": _scalar(primary_exposure, "estimate"),
            "primary_exposure_studentized_p_value": _scalar(
                primary_exposure, "studentized_p_value"
            ),
            "primary_exposure_weight_mapping_percentile": permutation.mapping_percentile,
            "primary_exposure_weight_permutation_mean": permutation.permutation_mean,
            "primary_exposure_weight_permutation_median": permutation.permutation_median,
            "primary_exposure_weight_permutation_p_value": permutation.p_value,
            "primary_exposure_weight_permutation_valid_replicates": (
                permutation.valid_replicates
            ),
            # The frozen block is shorter than an ENSO episode, so it can only
            # have understated the interval. These say by how much.
            "primary_exposure_block_sensitivity": _block_sensitivity_digest(block_sensitivity),
        },
        "snapshot": snapshot.name,
    }
    write_json_atomic(output_dir / "panel_summary.json", summary)
    return output_dir


def _run_block_sensitivity(
    primary_panel: pd.DataFrame, primary_fit: PanelFit, spec: PanelSpec
) -> pd.DataFrame:
    """Rerun the primary cell's bootstrap under each alternative block.

    Only the resampling changes: the point estimate, the sample and the fixed
    effects are the frozen primary fit, so any movement here is uncertainty that
    the calendar-year block was not capturing.
    """
    panel = attach_resampling_blocks(primary_panel, spec.block_sensitivity)
    regressors = panel_regressors(spec.primary_weighting)
    frames: list[pd.DataFrame] = []
    definitions = [(spec.block, PRIMARY_BLOCK_COLUMN, 1, 12)] + [
        (definition.name, definition.column, definition.length_years, definition.start_month)
        for definition in spec.block_sensitivity
    ]
    for name, column, length_years, start_month in definitions:
        results, _ = year_block_bootstrap(
            panel,
            primary_fit,
            replicates=spec.bootstrap_replicates,
            confidence_level=spec.confidence_level,
            seed=salted_seed(spec.random_seed, f"panel:block:{name}"),
            tolerance=spec.demeaning_tolerance,
            max_iterations=spec.demeaning_max_iterations,
            regressors=regressors,
            block_column=column,
        )
        results.insert(0, "block", name)
        results.insert(1, "block_length_years", length_years)
        results.insert(2, "block_start_month", start_month)
        results.insert(3, "blocks", int(panel[column].nunique()))
        results["is_primary_block"] = name == spec.block
        frames.append(results)
    return pd.concat(frames, ignore_index=True)


def _block_sensitivity_digest(sensitivity: pd.DataFrame) -> dict[str, Any]:
    exposure_rows = sensitivity.loc[sensitivity["term"].eq(EXPOSURE_TERM)]
    return {
        str(row["block"]): {
            "blocks": int(row["blocks"]),
            "ci_lower": _finite(row["ci_lower"]),
            "ci_upper": _finite(row["ci_upper"]),
            "interval_excludes_zero": (
                bool(row["ci_lower"] > 0 or row["ci_upper"] < 0)
                if pd.notna(row["ci_lower"]) and pd.notna(row["ci_upper"])
                else None
            ),
            "studentized_p_value": _finite(row["studentized_p_value"]),
        }
        for row in exposure_rows.to_dict("records")
    }


def _finite(value: Any) -> float | None:
    number = float(value)
    return None if pd.isna(number) else number


def _scalar(frame: pd.DataFrame, column: str) -> float | None:
    if len(frame) != 1:
        return None
    value = float(frame.iloc[0][column])
    return None if pd.isna(value) else value


def _run_specification_grid(
    monthly: pd.DataFrame,
    enso: pd.DataFrame,
    exposure: pd.DataFrame,
    spec: PanelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    replicate_frames: list[pd.DataFrame] = []
    for index_definition in spec.index_definitions:
        for lag_months in spec.reported_lag_months:
            for weighting in spec.weighting_schemes:
                cell = f"{index_definition}:lag{lag_months}:{weighting}"
                panel = build_panel(
                    monthly,
                    enso,
                    exposure,
                    index_name=index_definition,
                    lag_months=lag_months,
                    outcome_column=spec.outcome_column,
                    weighting=weighting,
                )
                if len(panel) < spec.minimum_observations:
                    raise ValueError(
                        f"Panel cell {cell} has {len(panel)} observations, "
                        f"below the configured minimum of {spec.minimum_observations}"
                    )
                regressors = panel_regressors(weighting)
                fit = fit_exposure_panel(
                    panel,
                    tolerance=spec.demeaning_tolerance,
                    max_iterations=spec.demeaning_max_iterations,
                    regressors=regressors,
                )
                results, replicates = year_block_bootstrap(
                    panel,
                    fit,
                    replicates=spec.bootstrap_replicates,
                    confidence_level=spec.confidence_level,
                    seed=salted_seed(spec.random_seed, f"panel:{cell}"),
                    tolerance=spec.demeaning_tolerance,
                    max_iterations=spec.demeaning_max_iterations,
                    regressors=regressors,
                )
                results.insert(0, "cell", cell)
                results.insert(1, "index_definition", index_definition)
                results.insert(2, "lag_months", lag_months)
                results.insert(3, "weighting", weighting)
                results["observations"] = fit.observations
                results["units"] = fit.units
                results["periods"] = fit.periods
                results["within_r_squared"] = fit.within_r_squared
                results["design_condition_number"] = fit.design_condition_number
                replicates.insert(0, "cell", cell)
                result_frames.append(results)
                replicate_frames.append(replicates)

    results = pd.concat(result_frames, ignore_index=True)
    # The exposure term across the grid is one family; the control term is a
    # diagnostic and stays outside it, exactly as in the event-study stages.
    exposure_rows = results["term"].eq(EXPOSURE_TERM)
    results["bh_q_value"] = pd.Series(float("nan"), index=results.index)
    results.loc[exposure_rows, "bh_q_value"] = benjamini_hochberg(
        results.loc[exposure_rows, "studentized_p_value"]
    )
    results["reject_fdr"] = results["bh_q_value"].le(spec.fdr_alpha).fillna(False)
    return results, pd.concat(replicate_frames, ignore_index=True)
