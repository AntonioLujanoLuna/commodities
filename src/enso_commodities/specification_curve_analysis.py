"""Real-data orchestration for the whole-year circular-shift diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .panel import build_exposure_table, load_panel_config
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .specification_curve import (
    fit_curve,
    joint_null_p_value,
    load_curve_config,
    summarize_curve,
)
from .universe import load_commodity_registry


def _load_real(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError(f"Specification curve requires real-data input: {path}")
    return summary


def run_specification_curve(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    panel_config_path: Path | None = None,
    curve_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or project_root() / "tables") / snapshot.name
    panel_path = panel_config_path or project_root() / "config" / "panel.yaml"
    curve_path = curve_config_path or project_root() / "config" / "specification_curve.yaml"
    panel_spec = load_panel_config(panel_path)
    curve_spec = load_curve_config(curve_path)
    registry = load_commodity_registry(registry_path)

    adjusted_summary = _load_real(output_dir / "adjusted_event_summary.json")
    processed_summary = _load_real(snapshot / "summary.json")
    monthly_name = "commodity_returns_adjusted_monthly.parquet"
    enso_name = "enso_monthly.csv"
    verify_hashes(output_dir, {monthly_name: adjusted_summary["output_hashes"][monthly_name]})
    verify_hashes(snapshot, {enso_name: processed_summary["output_hashes"][enso_name]})
    monthly = pd.read_parquet(output_dir / monthly_name)
    enso = pd.read_csv(snapshot / enso_name, parse_dates=["date"])
    external_weights: pd.DataFrame | None = None
    external_summary_path = output_dir / "external_exposure_summary.json"
    if panel_spec.exposure_weights_file:
        external_summary = _load_real(external_summary_path)
        verify_hashes(
            output_dir,
            {
                panel_spec.exposure_weights_file: external_summary["output_hashes"][
                    panel_spec.exposure_weights_file
                ]
            },
        )
        external_weights = pd.read_csv(output_dir / panel_spec.exposure_weights_file)
    exposure = build_exposure_table(registry, panel_spec, external_weights)

    frames = [
        fit_curve(monthly, enso, exposure, panel_spec, shift_years=shift)
        for shift in range(0, curve_spec.maximum_shift_years + 1)
        if shift == 0 or shift >= curve_spec.minimum_shift_years
    ]
    cells = pd.concat(frames, ignore_index=True)
    summaries = summarize_curve(cells, minimum_valid_cell_share=curve_spec.minimum_valid_cell_share)
    cells_path = output_dir / "specification_curve_cells.csv"
    shifts_path = output_dir / "specification_curve_shifts.csv"
    cells.to_csv(cells_path, index=False)
    summaries.to_csv(shifts_path, index=False)

    observed = summaries.loc[summaries["is_observed_timing"]].iloc[0]
    receipt = {
        "data_provenance": "real",
        "contract": {
            "shift_unit": "whole_year",
            "wrap": "circular",
            "minimum_shift_years": curve_spec.minimum_shift_years,
            "maximum_shift_years": curve_spec.maximum_shift_years,
            "null_alignments": int((~summaries["is_observed_timing"]).sum()),
            "cells_per_alignment": int(cells["cell"].nunique()),
            "primary_statistic": "median_absolute_naive_t",
        },
        "input_hashes": {
            "adjusted_event_summary.json": sha256_file(output_dir / "adjusted_event_summary.json"),
            monthly_name: sha256_file(output_dir / monthly_name),
            enso_name: sha256_file(snapshot / enso_name),
            "commodities.yaml": sha256_file(
                registry_path or project_root() / "config" / "commodities.yaml"
            ),
            "panel.yaml": sha256_file(panel_path),
            "specification_curve.yaml": sha256_file(curve_path),
            **(
                {external_summary_path.name: sha256_file(external_summary_path)}
                if panel_spec.exposure_weights_file
                else {}
            ),
        },
        "output_hashes": {
            cells_path.name: sha256_file(cells_path),
            shifts_path.name: sha256_file(shifts_path),
        },
        "results": {
            "observed_median_absolute_naive_t": float(observed["median_absolute_naive_t"]),
            "observed_maximum_absolute_naive_t": float(observed["maximum_absolute_naive_t"]),
            "observed_positive_estimate_share": float(observed["positive_estimate_share"]),
            "joint_timing_p_value": joint_null_p_value(summaries, "median_absolute_naive_t"),
            "maximum_t_timing_p_value": joint_null_p_value(summaries, "maximum_absolute_naive_t"),
            "positive_share_timing_p_value": joint_null_p_value(
                summaries, "positive_estimate_share"
            ),
        },
        "scope": "panel_specification_family_joint_timing_null",
        "snapshot": snapshot.name,
    }
    write_json_atomic(output_dir / "specification_curve_summary.json", receipt)
    return output_dir
