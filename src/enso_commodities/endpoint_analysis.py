from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import project_root
from .endpoint_diagnostics import (
    add_endpoint_shape,
    endpoint_shape_summary,
    regime_correlations,
    strict_factor_windows,
)
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result: dict[str, Any] = yaml.safe_load(handle)
    if result.get("fdr_family") != "mechanism_candidates_only":
        raise ValueError("Endpoint diagnostic FDR must use mechanism candidates only")
    return result


def run_endpoint_diagnostics(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    endpoint_config_path: Path | None = None,
) -> Path:
    root = project_root()
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or root / "tables") / snapshot.name
    config_path = endpoint_config_path or root / "config" / "endpoint_diagnostics.yaml"
    config = _config(config_path)
    specificity_path = output_dir / "specificity_summary.json"
    with specificity_path.open(encoding="utf-8") as handle:
        receipt: dict[str, Any] = json.load(handle)
    if receipt.get("data_provenance") != "real":
        raise ValueError("Endpoint diagnostics require real specificity inputs")
    output_hashes = receipt.get("output_hashes")
    if not isinstance(output_hashes, dict):
        raise ValueError("Specificity output-hash receipt is missing")
    verify_hashes(output_dir, output_hashes)

    endpoints = add_endpoint_shape(
        pd.read_parquet(output_dir / "specificity_event_endpoints.parquet")
    )
    adjusted = pd.read_parquet(output_dir / "robustness_macro_adjusted_monthly.parquet")
    candidate_names = set(
        pd.read_csv(output_dir / "specificity_candidate_results.csv")["commodity"].unique()
    )
    control_names = set(
        pd.read_csv(output_dir / "specificity_control_results.csv")["commodity"].unique()
    )
    summary = endpoint_shape_summary(endpoints)
    bootstrap_outputs: list[pd.DataFrame] = []
    result_outputs: list[pd.DataFrame] = []
    for (index_definition, anchor_type, direction), cell in endpoints.groupby(
        ["index_definition", "anchor_type", "direction"], sort=True, observed=True
    ):
        for family, names in [("candidates", candidate_names), ("controls", control_names)]:
            sample = cell.loc[cell["commodity"].isin(names)]
            output = bootstrap_episode_means(
                sample,
                value_column="log_value",
                replicates=int(config["bootstrap_replicates"]),
                confidence_level=float(config["confidence_level"]),
                seed=salted_seed(
                    int(config["random_seed"]),
                    f"endpoint_log:{family}:{index_definition}:{anchor_type}:{direction}",
                ),
                p_value_method=str(config["p_value_method"]),
                minimum_studentized_replicate_share=float(
                    config["minimum_studentized_replicate_share"]
                ),
            )
            results = output.results.rename(
                columns={
                    "mean_return": "bootstrap_mean_log_return",
                    "bootstrap_p_value": "log_mean_bootstrap_p_value",
                    "direction": "log_mean_direction",
                    "sign_agreement": "log_sign_agreement",
                }
            )
            results = results.merge(
                summary.loc[
                    summary["index_definition"].eq(index_definition)
                    & summary["anchor_type"].eq(anchor_type)
                    & summary["direction"].eq(direction)
                    & summary["commodity"].isin(names)
                ],
                on="commodity",
                validate="one_to_one",
            )
            if family == "candidates":
                results["log_mean_bh_q_value"] = benjamini_hochberg(
                    results["log_mean_bootstrap_p_value"]
                )
                results["reject_log_mean_fdr"] = results["log_mean_bh_q_value"].le(
                    float(config["fdr_alpha"])
                )
                results["time_trend_bh_q_value"] = benjamini_hochberg(
                    results["time_spearman_p_value"]
                )
                results["reject_time_trend_fdr"] = results["time_trend_bh_q_value"].le(
                    float(config["fdr_alpha"])
                )
            else:
                results["reject_log_mean_raw"] = results["log_mean_bootstrap_p_value"].le(
                    float(config["fdr_alpha"])
                )
                results["reject_time_trend_raw"] = results["time_spearman_p_value"].le(
                    float(config["fdr_alpha"])
                )
            results.insert(0, "family", family)
            result_outputs.append(results)
            replicates = output.replicates
            replicates.insert(0, "direction", direction)
            replicates.insert(0, "family", family)
            replicates.insert(0, "anchor_type", anchor_type)
            replicates.insert(0, "index_definition", index_definition)
            bootstrap_outputs.append(replicates)

    results = pd.concat(result_outputs, ignore_index=True)
    factor_columns = tuple(str(value) for value in config["factor_columns"])
    factors = strict_factor_windows(
        endpoints,
        adjusted,
        factor_columns=factor_columns,
        horizon_months=int(config["horizon_months"]),
    )
    registered_endpoints = endpoints.loc[
        endpoints["commodity"].isin(candidate_names | control_names)
    ]
    correlations = regime_correlations(
        registered_endpoints, factors, factor_columns=factor_columns
    )
    correlations["family"] = correlations["commodity"].map(
        lambda commodity: "candidates" if commodity in candidate_names else "controls"
    )
    correlations["spearman_bh_q_value"] = float("nan")
    for _, positions in correlations.loc[correlations["family"].eq("candidates")].groupby(
        ["index_definition", "anchor_type", "direction", "factor"], observed=True
    ).groups.items():
        correlations.loc[positions, "spearman_bh_q_value"] = benjamini_hochberg(
            correlations.loc[positions, "spearman_p_value"]
        )
    correlations["reject_regime_correlation_fdr"] = correlations["spearman_bh_q_value"].le(
        float(config["fdr_alpha"])
    )
    correlations["reject_regime_correlation_raw"] = correlations["spearman_p_value"].le(
        float(config["fdr_alpha"])
    )

    paths = {
        "results": output_dir / "endpoint_diagnostic_results.csv",
        "replicates": output_dir / "endpoint_log_bootstrap_replicates.parquet",
        "factors": output_dir / "endpoint_factor_windows.csv",
        "correlations": output_dir / "endpoint_regime_correlations.csv",
    }
    results.to_csv(paths["results"], index=False)
    pd.concat(bootstrap_outputs, ignore_index=True).to_parquet(paths["replicates"], index=False)
    factors.to_csv(paths["factors"], index=False, date_format="%Y-%m-%d")
    correlations.to_csv(paths["correlations"], index=False)
    control_results = results.loc[results["family"].eq("controls")]
    run_summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "scope": "exploratory_endpoint_falsification_not_primary_design",
        "contract": config,
        "diagnostics": {
            "result_rows": len(results),
            "control_log_mean_raw_rejections": int(control_results["reject_log_mean_raw"].sum()),
            "control_time_trend_raw_rejections": int(control_results["reject_time_trend_raw"].sum()),
            "candidate_regime_correlations_fdr": int(correlations["reject_regime_correlation_fdr"].sum()),
            "control_regime_correlations_raw": int(
                correlations.loc[correlations["family"].eq("controls"), "reject_regime_correlation_raw"].sum()
            ),
        },
        "input_hashes": {
            specificity_path.name: sha256_file(specificity_path),
            config_path.name: sha256_file(config_path),
            **{name: sha256_file(output_dir / name) for name in [
                "specificity_event_endpoints.parquet",
                "specificity_candidate_results.csv",
                "specificity_control_results.csv",
                "robustness_macro_adjusted_monthly.parquet",
            ]},
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
    }
    write_json_atomic(output_dir / "endpoint_diagnostics_summary.json", run_summary)
    return output_dir
