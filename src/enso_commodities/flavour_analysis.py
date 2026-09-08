"""Real-data orchestration for the W5 episode-flavour diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .dispersion import load_program_register
from .flavour import classify_episode_flavour, flavour_contrast, load_flavour_config
from .flavour_data import latest_flavour_snapshot
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


def run_flavour_analysis(
    processed_snapshot: Path | None = None,
    *,
    flavour_snapshot: Path | None = None,
    tables_root: Path | None = None,
    config_path: Path | None = None,
    program_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output = (tables_root or project_root() / "tables") / snapshot.name
    universe_summary_path = output / "universe_summary.json"
    raw_summary_path = output / "raw_event_summary.json"
    summaries: dict[str, dict[str, Any]] = {}
    for path in (universe_summary_path, raw_summary_path):
        with path.open(encoding="utf-8") as handle:
            summaries[path.name] = json.load(handle)
    if any(item.get("data_provenance") != "real" for item in summaries.values()):
        raise ValueError("Flavour analysis requires real-data receipts")

    index_snapshot = flavour_snapshot or (
        project_root() / "data" / "flavour" / "processed" / latest_flavour_snapshot().name
    )
    index_summary_path = index_snapshot / "summary.json"
    with index_summary_path.open(encoding="utf-8") as handle:
        index_summary: dict[str, Any] = json.load(handle)
    if index_summary.get("data_provenance") != "real":
        raise ValueError("Flavour region indices are not marked as real data")
    verify_hashes(
        index_snapshot,
        {
            "enso_region_indices_monthly.csv": index_summary["output_hashes"][
                "enso_region_indices_monthly.csv"
            ]
        },
    )

    inputs = {
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

    spec = load_flavour_config(config_path)
    program = load_program_register(program_config_path)
    episodes = pd.read_csv(
        output / "enso_episodes.csv", parse_dates=["onset_date", "peak_date", "end_date"]
    )
    episodes = episodes.loc[
        episodes["index_name"].eq("roni") & episodes["direction"].eq("warm")
    ].copy()
    if episodes.empty:
        raise ValueError("Flavour analysis found no warm RONI episodes")
    indices = pd.read_csv(index_snapshot / "enso_region_indices_monthly.csv", parse_dates=["date"])

    flavours = classify_episode_flavour(episodes, indices, spec=spec)
    sensitivity = classify_episode_flavour(
        episodes, indices, spec=spec, reference_month=spec.sensitivity_reference_month
    )

    candidates = pd.read_parquet(output / "primary_inference_family.parquet").assign(
        role="mechanism_candidate"
    )
    controls = pd.read_parquet(output / "negative_control_sample.parquet")
    controls = controls.loc[controls["control_eligible"] & controls[spec.outcome].notna()].assign(
        role="negative_control"
    )
    outcomes = pd.concat([candidates, controls], ignore_index=True)
    results, statistics = flavour_contrast(outcomes, flavours, spec=spec)

    agreement = flavours.merge(
        sensitivity.loc[:, ["episode_id", "flavour"]],
        on="episode_id",
        suffixes=("", "_episode_mean"),
        validate="one_to_one",
    )
    statistics["classification_agreement_share"] = float(
        agreement["flavour"].eq(agreement["flavour_episode_mean"]).mean()
    )
    statistics["program_threshold"] = program.threshold
    statistics["program_primary_tests"] = program.primary_tests
    statistics["clears_program_threshold"] = bool(
        pd.notna(statistics["family_bootstrap_p_value"])
        and statistics["family_bootstrap_p_value"] <= program.threshold
    )
    statistics["status"] = (
        "program_level_finding"
        if statistics["clears_program_threshold"]
        else "exploratory_within_stage_only"
    )

    results_path = output / "flavour_contrast_results.csv"
    labels_path = output / "flavour_episode_labels.csv"
    sensitivity_path = output / "flavour_episode_labels_sensitivity.csv"
    results.to_csv(results_path, index=False)
    flavours.to_csv(labels_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    config_file = config_path or project_root() / "config" / "flavour.yaml"
    program_file = program_config_path or project_root() / "config" / "findings_v3.yaml"
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "flavour_index_snapshot": index_snapshot.name,
        "design": {
            **spec.config["classification"],
            **spec.config["contrast"],
            "inference_scope": spec.config["provenance"]["inference_scope"],
            "workstream": spec.config["provenance"]["workstream"],
        },
        "results": statistics,
        "input_hashes": {
            **{name: sha256_file(output / name) for name in inputs},
            "enso_region_indices_monthly.csv": sha256_file(
                index_snapshot / "enso_region_indices_monthly.csv"
            ),
            "flavour.yaml": sha256_file(config_file),
            "findings_v3.yaml": sha256_file(program_file),
            **{name: sha256_file(output / name) for name in summaries},
        },
        "output_hashes": {
            results_path.name: sha256_file(results_path),
            labels_path.name: sha256_file(labels_path),
            sensitivity_path.name: sha256_file(sensitivity_path),
        },
        "random_seed": spec.random_seed,
    }
    write_json_atomic(output / "flavour_summary.json", summary)
    return output
