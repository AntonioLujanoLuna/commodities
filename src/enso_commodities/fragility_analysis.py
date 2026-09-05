from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_research_config, project_root
from .fragility import summarize_candidate_fragility, summarize_control_fragility
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed
from .universe import load_commodity_registry


def run_leave_one_episode_out(
    *,
    tables_snapshot: Path | None = None,
    registry_path: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    root = project_root()
    if tables_snapshot is None:
        candidates = sorted(path for path in (root / "tables").glob("????-??-??") if path.is_dir())
        if not candidates:
            raise FileNotFoundError("No dated analysis tables found")
        output_dir = candidates[-1]
    else:
        output_dir = tables_snapshot
    config_path = research_config_path or root / "config" / "research.yaml"
    config = load_research_config(config_path)
    spec = load_commodity_registry(registry_path).inference

    macro_summary_path = output_dir / "macro_analysis_summary.json"
    with macro_summary_path.open(encoding="utf-8") as handle:
        macro_summary: dict[str, Any] = json.load(handle)
    if macro_summary.get("data_provenance") != "real":
        raise ValueError("Leave-one-out analysis requires real macro-adjusted data")
    output_hashes = macro_summary.get("output_hashes")
    if not isinstance(output_hashes, dict):
        raise ValueError("Macro-analysis output hashes are missing")
    candidate_sample_name = "macro_candidate_sample.parquet"
    control_sample_name = "macro_negative_control_sample.parquet"
    primary_results_name = "macro_primary_results.csv"
    control_results_name = "macro_control_results.csv"
    required = [
        candidate_sample_name,
        control_sample_name,
        primary_results_name,
        control_results_name,
    ]
    verify_hashes(output_dir, {name: output_hashes[name] for name in required})
    if macro_summary.get("input_hashes", {}).get("research.yaml") != sha256_file(config_path):
        raise ValueError("Macro analysis was not built with the current research configuration")

    candidate_sample = pd.read_parquet(output_dir / candidate_sample_name)
    control_sample = pd.read_parquet(output_dir / control_sample_name)
    full_primary = pd.read_csv(output_dir / primary_results_name)
    full_controls = pd.read_csv(output_dir / control_results_name)
    episodes = pd.read_csv(output_dir / "enso_episodes.csv", parse_dates=["onset_date"])
    return_column = "macro_adjusted_cumulative_return"
    episode_ids = sorted(
        candidate_sample.loc[candidate_sample[return_column].notna(), "episode_id"].unique()
    )
    episode_dates = episodes.set_index("episode_id")["onset_date"].to_dict()
    full_directions = full_primary.set_index("commodity")["macro_direction"].to_dict()
    control_directions = full_controls.set_index("commodity")["macro_direction"].to_dict()

    candidate_rows: list[pd.DataFrame] = []
    control_rows: list[pd.DataFrame] = []
    for deleted_episode_id in episode_ids:
        candidate_scenario = candidate_sample.loc[
            candidate_sample["episode_id"].ne(deleted_episode_id)
        ]
        candidate_bootstrap = bootstrap_episode_means(
            candidate_scenario,
            value_column=return_column,
            replicates=config.bootstrap_replicates,
            confidence_level=config.confidence_level,
            seed=salted_seed(config.random_seed, f"macro_loo_candidates:{deleted_episode_id}"),
        ).results
        candidate_bootstrap["eligible"] = candidate_bootstrap["episodes"].ge(
            spec.minimum_valid_episodes
        )
        candidate_bootstrap.loc[~candidate_bootstrap["eligible"], "bootstrap_p_value"] = 1.0
        candidate_bootstrap["bootstrap_bh_q_value"] = benjamini_hochberg(
            candidate_bootstrap["bootstrap_p_value"]
        )
        candidate_bootstrap["reject_bootstrap_fdr"] = candidate_bootstrap[
            "bootstrap_bh_q_value"
        ].le(config.fdr_alpha)
        candidate_bootstrap["stable_direction"] = candidate_bootstrap.apply(
            lambda row: row["direction"] == full_directions[row["commodity"]], axis=1
        )
        candidate_bootstrap.insert(0, "deleted_episode_id", deleted_episode_id)
        candidate_bootstrap.insert(1, "deleted_onset_date", episode_dates[deleted_episode_id])
        candidate_rows.append(candidate_bootstrap)

        control_scenario = control_sample.loc[control_sample["episode_id"].ne(deleted_episode_id)]
        control_bootstrap = bootstrap_episode_means(
            control_scenario,
            value_column=return_column,
            replicates=config.bootstrap_replicates,
            confidence_level=config.confidence_level,
            seed=salted_seed(config.random_seed, f"macro_loo_controls:{deleted_episode_id}"),
        ).results
        control_bootstrap["stable_direction"] = control_bootstrap.apply(
            lambda row: row["direction"] == control_directions[row["commodity"]], axis=1
        )
        control_bootstrap.insert(0, "deleted_episode_id", deleted_episode_id)
        control_bootstrap.insert(1, "deleted_onset_date", episode_dates[deleted_episode_id])
        control_rows.append(control_bootstrap)

    candidate_scenarios = pd.concat(candidate_rows, ignore_index=True)
    control_scenarios = pd.concat(control_rows, ignore_index=True)
    candidate_summary = summarize_candidate_fragility(candidate_scenarios, full_primary)
    control_summary = summarize_control_fragility(control_scenarios, full_controls)

    paths = {
        "candidate_scenarios": output_dir / "macro_leave_one_out_candidates.csv",
        "control_scenarios": output_dir / "macro_leave_one_out_controls.csv",
        "candidate_summary": output_dir / "macro_fragility_summary.csv",
        "control_summary": output_dir / "macro_control_fragility_summary.csv",
    }
    candidate_scenarios.sort_values(["deleted_episode_id", "commodity"]).to_csv(
        paths["candidate_scenarios"], index=False, date_format="%Y-%m-%d"
    )
    control_scenarios.sort_values(["deleted_episode_id", "commodity"]).to_csv(
        paths["control_scenarios"], index=False, date_format="%Y-%m-%d"
    )
    candidate_summary.to_csv(paths["candidate_summary"], index=False)
    control_summary.to_csv(paths["control_summary"], index=False)

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": output_dir.name,
        "contract": {
            "bootstrap_replicates_per_deletion": config.bootstrap_replicates,
            "fdr_alpha": config.fdr_alpha,
            "fdr_family": config.fragility_fdr_family,
            "inference_gate": config.fragility_inference_gate,
            "method": config.fragility_method,
            "require_sign_agreement": config.fragility_require_sign_agreement,
            "require_stable_direction": config.fragility_require_stable_direction,
        },
        "diagnostics": {
            "candidate_scenario_rows": len(candidate_scenarios),
            "controls_rejecting_after_every_deletion": int(
                control_summary["rejects_after_every_deletion"].sum()
            ),
            "episode_deletions": len(episode_ids),
            "primary_candidates_surviving_all_deletions": int(
                candidate_summary["survives_all_deletions"].sum()
            ),
        },
        "input_hashes": {
            macro_summary_path.name: sha256_file(macro_summary_path),
            **{name: sha256_file(output_dir / name) for name in required},
            "research.yaml": sha256_file(config_path),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
        "random_seed": config.random_seed,
    }
    write_json_atomic(output_dir / "macro_fragility_run_summary.json", summary)
    return output_dir
