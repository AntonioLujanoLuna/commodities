from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import project_root
from .financial_adjustments import fit_financial_adjusted_returns
from .financial_data import latest_financial_snapshot
from .macro_data import latest_macro_snapshot
from .placebo import build_placebo_endpoints
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .specificity import add_candidate_fdr, randomization_direction_contrast
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed


def _load_json(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    if result.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs")
    return result


def run_financial_control_analysis(
    processed_snapshot: Path | None = None,
    macro_snapshot: Path | None = None,
    financial_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    root = project_root()
    snapshot = processed_snapshot or latest_processed_snapshot()
    macro_input = macro_snapshot or latest_macro_snapshot(root / "data" / "macro" / "processed")
    financial_input = financial_snapshot or latest_financial_snapshot(root / "data" / "financial" / "processed")
    output_dir = (tables_root or root / "tables") / snapshot.name
    settings_path = config_path or root / "config" / "financial_robustness.yaml"
    with settings_path.open(encoding="utf-8") as handle:
        settings: dict[str, Any] = yaml.safe_load(handle)
    if settings.get("fdr_family") != "mechanism_candidates_only":
        raise ValueError("Financial robustness FDR must use mechanism candidates only")
    robustness_receipt = _load_json(output_dir / "robustness_summary.json", "Financial analysis")
    specificity_receipt = _load_json(output_dir / "specificity_summary.json", "Financial analysis")
    financial_receipt = _load_json(financial_input / "summary.json", "Financial analysis")
    verify_hashes(output_dir, robustness_receipt["output_hashes"])
    verify_hashes(output_dir, specificity_receipt["output_hashes"])
    verify_hashes(financial_input, financial_receipt["output_hashes"])

    monthly = pd.read_parquet(output_dir / "robustness_macro_adjusted_monthly.parquet")
    macro = pd.read_csv(macro_input / "macro_controls_monthly.csv", parse_dates=["date"])
    financial = pd.read_csv(financial_input / "financial_controls_monthly.csv", parse_dates=["date"])
    external = macro.merge(
        financial.drop(columns=["us_cpi"]), on="date", how="outer", validate="one_to_one"
    )
    warm_episodes = pd.read_csv(output_dir / "robustness_episodes.csv", parse_dates=["onset_date", "observable_date"])
    cold_episodes = pd.read_csv(output_dir / "specificity_cold_episodes.csv", parse_dates=["onset_date", "observable_date"])
    candidate_names = set(pd.read_csv(output_dir / "robustness_candidate_results.csv")["commodity"].unique())
    control_names = set(pd.read_csv(output_dir / "robustness_control_results.csv")["commodity"].unique())
    all_names = sorted(candidate_names | control_names)

    adjusted_outputs: list[pd.DataFrame] = []
    model_outputs: list[pd.DataFrame] = []
    endpoint_outputs: list[pd.DataFrame] = []
    result_outputs: list[pd.DataFrame] = []
    bootstrap_outputs: list[pd.DataFrame] = []
    contrast_outputs: list[pd.DataFrame] = []
    contrast_replicates: list[pd.DataFrame] = []
    controls_tuple = tuple(str(value) for value in settings["controls"])
    for index_definition in settings["index_definitions"]:
        index_warm = warm_episodes.loc[warm_episodes["index_definition"].eq(index_definition)]
        index_cold = cold_episodes.loc[cold_episodes["index_definition"].eq(index_definition)]
        index_monthly = monthly.loc[monthly["index_definition"].eq(index_definition)]
        exclusion_values = tuple(int(value) for value in settings["estimation_exclusion_window"])
        if len(exclusion_values) != 2:
            raise ValueError("Financial estimation exclusion window must have two endpoints")
        adjusted, models = fit_financial_adjusted_returns(
            index_monthly,
            external,
            index_warm,
            control_columns=controls_tuple,
            exclusion_window=(exclusion_values[0], exclusion_values[1]),
            minimum_observations=int(settings["minimum_model_observations"]),
        )
        adjusted_outputs.append(adjusted)
        models.insert(0, "index_definition", index_definition)
        model_outputs.append(models)
        for anchor_type in settings["anchor_types"]:
            cell_endpoints: list[pd.DataFrame] = []
            anchor_column = "onset_date" if anchor_type == "retrospective" else "observable_date"
            for direction, episodes in [("warm", index_warm), ("cold", index_cold)]:
                endpoints = build_placebo_endpoints(
                    adjusted,
                    episodes[anchor_column],
                    all_names,
                    horizon_months=int(settings["horizon_months"]),
                    monthly_return_column="financial_adjusted_log_return",
                ).merge(
                    episodes.loc[:, ["episode_id", anchor_column]].rename(columns={anchor_column: "anchor_date"}),
                    on="anchor_date",
                    how="left",
                    validate="many_to_one",
                )
                endpoints = endpoints.rename(columns={"placebo_return": "financial_adjusted_cumulative_return"})
                endpoints["direction"] = direction
                endpoints["anchor_type"] = anchor_type
                endpoints["index_definition"] = index_definition
                endpoint_outputs.append(endpoints)
                cell_endpoints.append(endpoints)
                for family, names in [("candidates", candidate_names), ("controls", control_names)]:
                    sample = endpoints.loc[endpoints["commodity"].isin(names)]
                    bootstrap = bootstrap_episode_means(
                        sample,
                        value_column="financial_adjusted_cumulative_return",
                        replicates=int(settings["bootstrap_replicates"]),
                        confidence_level=float(settings["confidence_level"]),
                        seed=salted_seed(int(settings["random_seed"]), f"financial:{family}:{index_definition}:{anchor_type}:{direction}"),
                    )
                    results = bootstrap.results.rename(columns={"direction": "return_direction"})
                    if family == "candidates":
                        results["bootstrap_bh_q_value"] = benjamini_hochberg(results["bootstrap_p_value"])
                        results["reject_bootstrap_fdr"] = results["bootstrap_bh_q_value"].le(float(settings["fdr_alpha"]))
                    else:
                        results["reject_bootstrap_raw"] = results["bootstrap_p_value"].le(float(settings["fdr_alpha"]))
                    for column, value in [("family", family), ("direction", direction), ("anchor_type", anchor_type), ("index_definition", index_definition)]:
                        results.insert(0, column, value)
                        bootstrap.replicates.insert(0, column, value)
                    result_outputs.append(results)
                    bootstrap_outputs.append(bootstrap.replicates)
            combined = pd.concat(cell_endpoints, ignore_index=True)
            for family, names in [("candidates", candidate_names), ("controls", control_names)]:
                contrast = randomization_direction_contrast(
                    combined.loc[combined["commodity"].isin(names)].rename(columns={"financial_adjusted_cumulative_return": "value"}),
                    value_column="value",
                    replicates=int(settings["bootstrap_replicates"]),
                    seed=salted_seed(int(settings["random_seed"]), f"financial_contrast:{family}:{index_definition}:{anchor_type}"),
                )
                results = contrast.results
                if family == "candidates":
                    results = add_candidate_fdr(results, alpha=float(settings["fdr_alpha"]))
                else:
                    results["reject_direction_contrast_raw"] = results["contrast_p_value"].le(float(settings["fdr_alpha"]))
                for column, value in [("family", family), ("anchor_type", anchor_type), ("index_definition", index_definition)]:
                    results.insert(0, column, value)
                    contrast.replicates.insert(0, column, value)
                contrast_outputs.append(results)
                contrast_replicates.append(contrast.replicates)

    results = pd.concat(result_outputs, ignore_index=True)
    contrasts = pd.concat(contrast_outputs, ignore_index=True)
    paths = {
        "models": output_dir / "financial_control_models.csv",
        "monthly": output_dir / "financial_adjusted_monthly.parquet",
        "endpoints": output_dir / "financial_control_endpoints.parquet",
        "results": output_dir / "financial_control_results.csv",
        "contrasts": output_dir / "financial_direction_contrasts.csv",
        "bootstrap": output_dir / "financial_control_bootstrap_replicates.parquet",
        "contrast_replicates": output_dir / "financial_contrast_replicates.parquet",
    }
    pd.concat(model_outputs).to_csv(paths["models"], index=False)
    pd.concat(adjusted_outputs).to_parquet(paths["monthly"], index=False)
    pd.concat(endpoint_outputs).to_parquet(paths["endpoints"], index=False)
    results.to_csv(paths["results"], index=False)
    contrasts.to_csv(paths["contrasts"], index=False)
    pd.concat(bootstrap_outputs).to_parquet(paths["bootstrap"], index=False)
    pd.concat(contrast_replicates).to_parquet(paths["contrast_replicates"], index=False)
    control_results = results.loc[results["family"].eq("controls")]
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "scope": "exploratory_financial_control_not_primary_design",
        "contract": settings,
        "diagnostics": {
            "models_estimated": int(pd.concat(model_outputs)["financial_model_status"].eq("estimated").sum()),
            "result_rows": len(results),
            "control_raw_rejections": int(control_results["reject_bootstrap_raw"].fillna(False).sum()),
            "control_direction_contrast_raw_rejections": int(contrasts.loc[contrasts["family"].eq("controls"), "reject_direction_contrast_raw"].fillna(False).sum()),
        },
        "input_hashes": {
            "financial_robustness.yaml": sha256_file(settings_path),
            "robustness_summary.json": sha256_file(output_dir / "robustness_summary.json"),
            "specificity_summary.json": sha256_file(output_dir / "specificity_summary.json"),
            "financial_summary.json": sha256_file(financial_input / "summary.json"),
            "macro_controls_monthly.csv": sha256_file(macro_input / "macro_controls_monthly.csv"),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
    }
    write_json_atomic(output_dir / "financial_control_summary.json", summary)
    return output_dir
