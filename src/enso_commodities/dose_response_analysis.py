"""Real-data orchestration for the secondary episode-amplitude diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .dose_response import (
    amplitude_permutation_inference,
    load_dose_response_config,
    prepare_dose_panel,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


def run_dose_response_analysis(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output = (tables_root or project_root() / "tables") / snapshot.name
    universe_summary_path = output / "universe_summary.json"
    raw_summary_path = output / "raw_event_summary.json"
    with universe_summary_path.open(encoding="utf-8") as handle:
        universe_summary: dict[str, Any] = json.load(handle)
    with raw_summary_path.open(encoding="utf-8") as handle:
        raw_summary: dict[str, Any] = json.load(handle)
    if (
        universe_summary.get("data_provenance") != "real"
        or raw_summary.get("data_provenance") != "real"
    ):
        raise ValueError("Dose-response analysis requires real-data receipts")
    inputs = {
        "primary_inference_family.parquet": universe_summary["output_hashes"][
            "primary_inference_family.parquet"
        ],
        "negative_control_sample.parquet": universe_summary["output_hashes"][
            "negative_control_sample.parquet"
        ],
        "enso_episodes.csv": raw_summary["output_hashes"]["enso_episodes.csv"],
    }
    verify_hashes(output, inputs)

    spec = load_dose_response_config(config_path)
    candidates = pd.read_parquet(output / "primary_inference_family.parquet").assign(
        role="mechanism_candidate"
    )
    controls = pd.read_parquet(output / "negative_control_sample.parquet")
    controls = controls.loc[controls["control_eligible"] & controls[spec.outcome].notna()].assign(
        role="negative_control"
    )
    outcomes = pd.concat([candidates, controls], ignore_index=True)
    episodes = pd.read_csv(output / "enso_episodes.csv")
    panel = prepare_dose_panel(outcomes, episodes, outcome_column=spec.outcome)
    results, replicates = amplitude_permutation_inference(
        panel,
        outcome_column=spec.outcome,
        minimum_valid_episodes=spec.minimum_valid_episodes,
        replicates=spec.replicates,
        seed=spec.random_seed,
        fdr_alpha=spec.fdr_alpha,
        fwer_alpha=spec.fwer_alpha,
    )
    results_path = output / "dose_response_results.csv"
    replicates_path = output / "dose_response_permutations.parquet"
    results.to_csv(results_path, index=False)
    replicates.to_parquet(replicates_path, index=False)
    config_file = config_path or project_root() / "config" / "dose_response.yaml"
    candidates_mask = results["role"].eq("mechanism_candidate")
    controls_mask = results["role"].eq("negative_control")
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "design": {
            **spec.config["design"],
            "inference_scope": spec.config["provenance"]["inference_scope"],
            "permutation_replicates": spec.replicates,
            "permutation_unit": spec.config["inference"]["permutation_unit"],
        },
        "results": {
            "candidate_family_size": int(candidates_mask.sum()),
            "candidate_positive_slopes": int(
                results.loc[candidates_mask, "slope_per_peak_roni_sd"].gt(0).sum()
            ),
            "candidate_raw_rejections": int(
                results.loc[candidates_mask, "permutation_p_value"].lt(spec.fdr_alpha).sum()
            ),
            "candidate_fdr_rejections": int(results.loc[candidates_mask, "reject_fdr"].sum()),
            "candidate_fwer_rejections": int(results.loc[candidates_mask, "reject_fwer"].sum()),
            "control_raw_rejections": int(
                results.loc[controls_mask, "permutation_p_value"].lt(spec.fdr_alpha).sum()
            ),
        },
        "input_hashes": {
            **{name: sha256_file(output / name) for name in inputs},
            "dose_response.yaml": sha256_file(config_file),
            universe_summary_path.name: sha256_file(universe_summary_path),
            raw_summary_path.name: sha256_file(raw_summary_path),
        },
        "output_hashes": {
            results_path.name: sha256_file(results_path),
            replicates_path.name: sha256_file(replicates_path),
        },
        "random_seed": spec.random_seed,
    }
    write_json_atomic(output / "dose_response_summary.json", summary)
    return output
