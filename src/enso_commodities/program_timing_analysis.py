"""Receipt-writing orchestration for the v2 event-study program timing null."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_research_config, project_root
from .program_timing_null import (
    fit_program_alignment,
    prepare_program_lookups,
    program_null_p_values,
    summarize_program_alignments,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .validation import load_validation_contract


def run_program_timing_null(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    validation_config_path: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or project_root() / "tables") / snapshot.name
    validation_path = validation_config_path or project_root() / "config" / "validation_v2.yaml"
    research_path = research_config_path or project_root() / "config" / "research.yaml"
    contract = load_validation_contract(validation_path)
    research = load_research_config(research_path)
    adjusted_receipt_path = output_dir / "adjusted_event_summary.json"
    with adjusted_receipt_path.open(encoding="utf-8") as handle:
        adjusted_receipt: dict[str, Any] = json.load(handle)
    monthly_name = "commodity_returns_adjusted_monthly.parquet"
    verify_hashes(output_dir, {monthly_name: adjusted_receipt["output_hashes"][monthly_name]})
    monthly = pd.read_parquet(output_dir / monthly_name)
    enso = pd.read_csv(snapshot / "enso_monthly.csv", parse_dates=["date"])
    lookups = prepare_program_lookups(monthly, contract)
    shifts = [0, *range(contract.minimum_shift_years, contract.maximum_shift_years + 1)]
    frames = [
        fit_program_alignment(
            monthly,
            enso,
            contract,
            shift_years=shift,
            threshold=research.warm_threshold,
            minimum_duration_months=research.minimum_duration_months,
            observable_delay_after_center_months=research.observable_delay_after_center_months,
            lookups=lookups,
        )
        for shift in shifts
    ]
    cells = pd.concat(frames, ignore_index=True)
    summaries = summarize_program_alignments(cells)
    p_values = program_null_p_values(summaries)
    cells_path = output_dir / "program_timing_null_cells.csv"
    shifts_path = output_dir / "program_timing_null_shifts.csv"
    cells.to_csv(cells_path, index=False)
    summaries.to_csv(shifts_path, index=False)
    receipt = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "scope": "locked_v2_selected_event_study_family_retrospective_calibration",
        "selection_is_outcome_informed": True,
        "contract": {
            "commodities": list(contract.commodities),
            "indexes": list(contract.timing_indexes),
            "anchors": list(contract.timing_anchors),
            "horizons_months": list(contract.timing_horizons),
            "outcomes": list(contract.timing_outcomes),
            "cells_per_alignment": int(cells.loc[cells["shift_years"].eq(0)].shape[0]),
            "null_alignments": len(shifts) - 1,
        },
        "results": p_values,
        "input_hashes": {
            adjusted_receipt_path.name: sha256_file(adjusted_receipt_path),
            monthly_name: sha256_file(output_dir / monthly_name),
            "enso_monthly.csv": sha256_file(snapshot / "enso_monthly.csv"),
            validation_path.name: sha256_file(validation_path),
            research_path.name: sha256_file(research_path),
        },
        "output_hashes": {
            cells_path.name: sha256_file(cells_path),
            shifts_path.name: sha256_file(shifts_path),
        },
    }
    write_json_atomic(output_dir / "program_timing_null_summary.json", receipt)
    return output_dir
