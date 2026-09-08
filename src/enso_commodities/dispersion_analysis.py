"""Real-data orchestration for the W1 dispersion endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .dispersion import (
    dispersion_shift_inference,
    load_dispersion_config,
    load_program_register,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


def _commodity_roles(output: Path) -> pd.Series:
    """Map each frozen commodity to its role, from the universe stage's own output."""
    candidates = pd.read_parquet(output / "primary_inference_family.parquet")
    controls = pd.read_parquet(output / "negative_control_sample.parquet")
    if "control_eligible" in controls.columns:
        controls = controls.loc[controls["control_eligible"]]
    roles = {str(name): "mechanism_candidate" for name in candidates["commodity"].unique()}
    for name in controls["commodity"].unique():
        roles.setdefault(str(name), "negative_control")
    return pd.Series(roles, name="role")


def run_dispersion_analysis(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    config_path: Path | None = None,
    program_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output = (tables_root or project_root() / "tables") / snapshot.name
    universe_summary_path = output / "universe_summary.json"
    adjusted_summary_path = output / "adjusted_event_summary.json"
    raw_summary_path = output / "raw_event_summary.json"
    summaries: dict[str, dict[str, Any]] = {}
    for path in (universe_summary_path, adjusted_summary_path, raw_summary_path):
        with path.open(encoding="utf-8") as handle:
            summaries[path.name] = json.load(handle)
    if any(item.get("data_provenance") != "real" for item in summaries.values()):
        raise ValueError("Dispersion analysis requires real-data receipts")

    inputs = {
        "commodity_returns_adjusted_monthly.parquet": summaries["adjusted_event_summary.json"][
            "output_hashes"
        ]["commodity_returns_adjusted_monthly.parquet"],
        "enso_episodes.csv": summaries["raw_event_summary.json"]["output_hashes"][
            "enso_episodes.csv"
        ],
        "primary_inference_family.parquet": summaries["universe_summary.json"]["output_hashes"][
            "primary_inference_family.parquet"
        ],
        "negative_control_sample.parquet": summaries["universe_summary.json"]["output_hashes"][
            "negative_control_sample.parquet"
        ],
    }
    verify_hashes(output, inputs)

    spec = load_dispersion_config(config_path)
    program = load_program_register(program_config_path)
    returns = pd.read_parquet(output / "commodity_returns_adjusted_monthly.parquet")
    episodes = pd.read_csv(output / "enso_episodes.csv", parse_dates=["onset_date"])
    episodes = episodes.loc[
        episodes["index_name"].eq("roni") & episodes["direction"].eq("warm")
    ].copy()
    if episodes.empty:
        raise ValueError("Dispersion analysis found no warm RONI episodes")
    roles = _commodity_roles(output)
    results, null_frame, statistics = dispersion_shift_inference(
        returns.loc[returns["commodity"].isin(roles.index)],
        episodes,
        roles,
        spec=spec,
        program=program,
    )

    results_path = output / "dispersion_results.csv"
    null_path = output / "dispersion_shift_null.parquet"
    results.to_csv(results_path, index=False)
    null_frame.to_parquet(null_path, index=False)

    config_file = config_path or project_root() / "config" / "dispersion.yaml"
    program_file = program_config_path or project_root() / "config" / "findings_v3.yaml"
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "design": {
            **spec.config["design"],
            "inference_scope": spec.config["provenance"]["inference_scope"],
            "workstream": spec.config["provenance"]["workstream"],
            "reference_distribution": spec.config["inference"]["reference_distribution"],
            "warm_episodes": len(episodes),
        },
        "results": statistics,
        "input_hashes": {
            **{name: sha256_file(output / name) for name in inputs},
            "dispersion.yaml": sha256_file(config_file),
            "findings_v3.yaml": sha256_file(program_file),
            **{name: sha256_file(output / name) for name in summaries},
        },
        "output_hashes": {
            results_path.name: sha256_file(results_path),
            null_path.name: sha256_file(null_path),
        },
    }
    write_json_atomic(output / "dispersion_summary.json", summary)
    return output
