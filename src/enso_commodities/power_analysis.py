"""Minimum-detectable-effect stage for the frozen primary endpoint.

Reads the inference stage's own hash-verified outputs and answers the question
the evidence ladder needs before it can call anything a null: what size of
effect would this design have found, given the episodes it has?

No resampling happens here. The stored studentized replicates already are the
reference distribution, and the alternative is an additive shift that leaves
the standard error alone, so the whole stage is a rearrangement of a bootstrap
the study has already run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_research_config, project_root
from .power import evaluate_power
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import studentized_null_matrix


def _read_real_summary(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs")
    return summary


def _median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if len(numeric) else None


def run_power_analysis(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    config_path = research_config_path or project_root() / "config" / "research.yaml"
    config = load_research_config(config_path)

    inference_summary_path = output_dir / "inference_summary.json"
    inference_summary = _read_real_summary(inference_summary_path, "Power analysis")
    inference_hashes = inference_summary.get("output_hashes")
    if not isinstance(inference_hashes, dict):
        raise ValueError("The inference output-hash receipt is missing")
    names = {
        "primary_results": "primary_inference_results.csv",
        "control_results": "negative_control_inference.csv",
        "primary_replicates": "primary_bootstrap_replicates.parquet",
        "control_replicates": "negative_control_bootstrap_replicates.parquet",
    }
    verify_hashes(output_dir, {name: inference_hashes[name] for name in names.values()})
    if inference_summary.get("input_hashes", {}).get("research.yaml") != sha256_file(config_path):
        raise ValueError("Inference results were not built with the current research configuration")

    primary_results = pd.read_csv(output_dir / names["primary_results"])
    control_results = pd.read_csv(output_dir / names["control_results"])
    primary_nulls = studentized_null_matrix(
        pd.read_parquet(output_dir / names["primary_replicates"])
    )
    control_nulls = studentized_null_matrix(
        pd.read_parquet(output_dir / names["control_replicates"])
    )

    primary_power = evaluate_power(
        primary_results,
        primary_nulls,
        alpha=config.fdr_alpha,
        family_size=len(primary_results),
        target_power=config.power_target_power,
        reference_effects=config.power_reference_effects,
        maximum_effect=config.power_maximum_effect,
    )
    # Controls are diagnostics outside the candidate FDR family, so no family
    # penalty applies to them and the two thresholds coincide by construction.
    control_power = evaluate_power(
        control_results,
        control_nulls,
        alpha=config.fdr_alpha,
        family_size=1,
        target_power=config.power_target_power,
        reference_effects=config.power_reference_effects,
        maximum_effect=config.power_maximum_effect,
    )

    paths = {
        "primary_effects": output_dir / "primary_minimum_detectable_effect.csv",
        "control_effects": output_dir / "negative_control_minimum_detectable_effect.csv",
        "primary_curves": output_dir / "primary_power_curves.csv",
        "control_curves": output_dir / "negative_control_power_curves.csv",
    }
    primary_power.minimum_detectable_effects.to_csv(paths["primary_effects"], index=False)
    control_power.minimum_detectable_effects.to_csv(paths["control_effects"], index=False)
    primary_power.power_curves.to_csv(paths["primary_curves"], index=False)
    control_power.power_curves.to_csv(paths["control_curves"], index=False)

    effects = primary_power.minimum_detectable_effects
    curves = primary_power.power_curves
    estimated = effects["minimum_detectable_effect_family_status"].eq("estimated")
    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "power_contract": {
            "alternative": "additive_shift_on_the_frozen_endpoint",
            "family_alpha_rule": "fdr_alpha divided by the candidate-family size, the level "
            "Benjamini-Hochberg requires when exactly one member rejects",
            "marginal_alpha": config.fdr_alpha,
            "maximum_effect": config.power_maximum_effect,
            "reference_effects": list(config.power_reference_effects),
            "target_power": config.power_target_power,
            "test": "two_sided_studentized_bootstrap",
        },
        "diagnostics": {
            "candidate_family_size": len(primary_results),
            "candidates_below_marginal_mde": int(
                effects["observed_effect_below_marginal_mde"].fillna(False).astype(bool).sum()
            ),
            "candidates_with_unreachable_family_mde": int((~estimated).sum()),
            "median_minimum_detectable_effect_family": _median(
                effects["minimum_detectable_effect_family"]
            ),
            "median_minimum_detectable_effect_marginal": _median(
                effects["minimum_detectable_effect_marginal"]
            ),
            "median_power_at_reference_effect": {
                f"{effect:g}": _median(
                    curves.loc[np.isclose(curves["effect"], effect), "power_family"]
                )
                for effect in config.power_reference_effects
            },
            "negative_control_median_minimum_detectable_effect": _median(
                control_power.minimum_detectable_effects["minimum_detectable_effect_marginal"]
            ),
        },
        "input_hashes": {
            inference_summary_path.name: sha256_file(inference_summary_path),
            **{name: sha256_file(output_dir / name) for name in names.values()},
            "research.yaml": sha256_file(config_path),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
    }
    write_json_atomic(output_dir / "power_summary.json", summary)
    return output_dir
