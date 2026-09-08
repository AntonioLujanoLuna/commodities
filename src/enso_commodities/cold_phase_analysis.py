"""Real-data orchestration for the W2 cold-phase disruption endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .cold_phase import cold_phase_inference, load_cold_phase_config
from .config import project_root
from .dispersion import load_program_register
from .dispersion_analysis import commodity_roles
from .enso import construct_cold_episodes
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


def run_cold_phase_analysis(
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
    summaries: dict[str, dict[str, Any]] = {}
    for path in (universe_summary_path, adjusted_summary_path):
        with path.open(encoding="utf-8") as handle:
            summaries[path.name] = json.load(handle)
    if any(item.get("data_provenance") != "real" for item in summaries.values()):
        raise ValueError("Cold-phase analysis requires real-data receipts")

    inputs = {
        "commodity_returns_adjusted_monthly.parquet": summaries["adjusted_event_summary.json"][
            "output_hashes"
        ]["commodity_returns_adjusted_monthly.parquet"],
        "primary_inference_family.parquet": summaries["universe_summary.json"]["output_hashes"][
            "primary_inference_family.parquet"
        ],
        "negative_control_sample.parquet": summaries["universe_summary.json"]["output_hashes"][
            "negative_control_sample.parquet"
        ],
    }
    verify_hashes(output, inputs)

    spec = load_cold_phase_config(config_path)
    program = load_program_register(program_config_path)

    # Cold episodes are constructed here rather than read from the raw-event
    # stage, which freezes warm episodes only. The construction rule is the same
    # one, with the sign of the threshold reversed.
    enso_path = snapshot / "enso_monthly.csv"
    enso = pd.read_csv(enso_path, parse_dates=["date"])
    cold_episodes = construct_cold_episodes(
        enso,
        index_name=spec.index,
        threshold=spec.cold_threshold,
        minimum_duration_months=spec.minimum_duration_months,
        observable_delay_after_center_months=spec.observable_delay_after_center_months,
    )
    if cold_episodes.empty:
        raise ValueError("Cold-phase analysis found no cold episodes")

    returns = pd.read_parquet(output / "commodity_returns_adjusted_monthly.parquet")
    roles = commodity_roles(output)
    results, null_frame, statistics = cold_phase_inference(
        returns.loc[returns["commodity"].isin(roles.index)],
        cold_episodes,
        roles,
        spec=spec,
        program=program,
    )

    results_path = output / "cold_phase_results.csv"
    null_path = output / "cold_phase_shift_null.parquet"
    episodes_path = output / "cold_phase_episodes.csv"
    results.to_csv(results_path, index=False)
    null_frame.to_parquet(null_path, index=False)
    cold_episodes.to_csv(episodes_path, index=False, date_format="%Y-%m-%d")

    config_file = config_path or project_root() / "config" / "cold_phase.yaml"
    program_file = program_config_path or project_root() / "config" / "findings_v3.yaml"
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "design": {
            **spec.config["episodes"],
            **spec.config["design"],
            "inference_scope": spec.config["provenance"]["inference_scope"],
            "workstream": spec.config["provenance"]["workstream"],
            "reference_distribution": spec.config["inference"]["reference_distribution"],
            "signed_hypotheses": spec.hypotheses,
            "excluded_with_reason": spec.excluded,
        },
        "results": statistics,
        "input_hashes": {
            **{name: sha256_file(output / name) for name in inputs},
            "enso_monthly.csv": sha256_file(enso_path),
            "cold_phase.yaml": sha256_file(config_file),
            "findings_v3.yaml": sha256_file(program_file),
            **{name: sha256_file(output / name) for name in summaries},
        },
        "output_hashes": {
            results_path.name: sha256_file(results_path),
            null_path.name: sha256_file(null_path),
            episodes_path.name: sha256_file(episodes_path),
        },
    }
    write_json_atomic(output / "cold_phase_summary.json", summary)
    return output
