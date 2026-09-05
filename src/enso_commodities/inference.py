from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_research_config, project_root
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bootstrap_episode_means,
    salted_seed,
    studentized_null_matrix,
    westfall_young_step_down,
)
from .universe import load_commodity_registry


def _add_multiplicity_columns(
    results: pd.DataFrame,
    *,
    replicates: pd.DataFrame,
    fdr_alpha: float,
    fwer_alpha: float,
    minimum_valid_replicate_share: float,
    prefix: str = "",
) -> pd.DataFrame:
    """Attach three multiplicity verdicts to one bootstrap family.

    Benjamini-Hochberg is the pre-registered gate. Benjamini-Yekutieli repeats
    it without assuming anything about how the commodity tests co-move.
    Westfall-Young reuses the shared episode draws already stored in
    ``replicates`` to build the joint null of the largest statistic, so it
    controls the family-wise error rate under the dependence the resampling
    scheme actually preserves -- a stricter bar than either FDR rule.
    """
    output = results.copy()
    indexed = output.set_index("commodity")
    output[f"{prefix}bh_q_value"] = benjamini_hochberg(indexed["bootstrap_p_value"]).to_numpy()
    output[f"{prefix}by_q_value"] = benjamini_yekutieli(indexed["bootstrap_p_value"]).to_numpy()
    westfall_young = westfall_young_step_down(
        indexed["t_statistic"],
        studentized_null_matrix(replicates),
        minimum_valid_replicate_share=minimum_valid_replicate_share,
    ).set_index("commodity")
    output[f"{prefix}westfall_young_p_value"] = westfall_young.loc[
        indexed.index, "westfall_young_p_value"
    ].to_numpy()
    output[f"{prefix}westfall_young_status"] = westfall_young.loc[
        indexed.index, "westfall_young_status"
    ].to_numpy()
    output[f"{prefix}reject_fdr"] = output[f"{prefix}bh_q_value"].le(fdr_alpha)
    output[f"{prefix}reject_fdr_arbitrary_dependence"] = output[f"{prefix}by_q_value"].le(fdr_alpha)
    output[f"{prefix}reject_fwer"] = (
        output[f"{prefix}westfall_young_p_value"].le(fwer_alpha).fillna(False)
    )
    return output


def run_primary_inference(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    universe_summary_path = output_dir / "universe_summary.json"
    with universe_summary_path.open(encoding="utf-8") as handle:
        universe_summary = json.load(handle)
    if universe_summary.get("data_provenance") != "real":
        raise ValueError("Primary inference requires a real-data universe")
    universe_hashes = universe_summary.get("output_hashes")
    if not isinstance(universe_hashes, dict):
        raise ValueError("Universe output hashes are missing")
    required_inputs = ["primary_inference_family.parquet", "negative_control_sample.parquet"]
    verify_hashes(output_dir, {name: universe_hashes[name] for name in required_inputs})

    config_path = research_config_path or project_root() / "config" / "research.yaml"
    config = load_research_config(config_path)
    registry = load_commodity_registry(registry_path)
    spec = registry.inference
    primary_path = output_dir / "primary_inference_family.parquet"
    controls_path = output_dir / "negative_control_sample.parquet"
    primary = pd.read_parquet(primary_path)
    controls = pd.read_parquet(controls_path)
    primary_bootstrap = bootstrap_episode_means(
        primary,
        value_column=spec.primary_return,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        seed=salted_seed(config.random_seed, "primary_candidates"),
        p_value_method=config.p_value_method,
        minimum_studentized_replicate_share=config.minimum_studentized_replicate_share,
    )
    primary_results = primary_bootstrap.results.merge(
        primary.loc[:, ["commodity", "group"]].drop_duplicates(),
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    primary_results = _add_multiplicity_columns(
        primary_results,
        replicates=primary_bootstrap.replicates,
        fdr_alpha=config.fdr_alpha,
        fwer_alpha=config.fwer_alpha,
        minimum_valid_replicate_share=config.minimum_studentized_replicate_share,
    )

    control_bootstrap = bootstrap_episode_means(
        controls,
        value_column=spec.primary_return,
        replicates=config.bootstrap_replicates,
        confidence_level=config.confidence_level,
        seed=salted_seed(config.random_seed, "negative_controls"),
        p_value_method=config.p_value_method,
        minimum_studentized_replicate_share=config.minimum_studentized_replicate_share,
    )
    control_results = control_bootstrap.results.merge(
        controls.loc[:, ["commodity", "group"]].drop_duplicates(),
        on="commodity",
        how="left",
        validate="one_to_one",
    )
    control_results = _add_multiplicity_columns(
        control_results,
        replicates=control_bootstrap.replicates,
        fdr_alpha=config.fdr_alpha,
        fwer_alpha=config.fwer_alpha,
        minimum_valid_replicate_share=config.minimum_studentized_replicate_share,
        prefix="diagnostic_",
    )

    primary_results_path = output_dir / "primary_inference_results.csv"
    control_results_path = output_dir / "negative_control_inference.csv"
    primary_replicates_path = output_dir / "primary_bootstrap_replicates.parquet"
    control_replicates_path = output_dir / "negative_control_bootstrap_replicates.parquet"
    primary_results.sort_values("commodity").to_csv(primary_results_path, index=False)
    control_results.sort_values("commodity").to_csv(control_results_path, index=False)
    primary_bootstrap.replicates.to_parquet(primary_replicates_path, index=False)
    control_bootstrap.replicates.to_parquet(control_replicates_path, index=False)

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "inference": {
            "alternative": spec.alternative,
            "bootstrap_replicates": config.bootstrap_replicates,
            "confidence_level": config.confidence_level,
            "fdr_alpha": config.fdr_alpha,
            "fdr_family_size": len(primary_results),
            "fwer_alpha": config.fwer_alpha,
            "p_value_method": config.p_value_method,
            "primary_fwer_rejections": int(primary_results["reject_fwer"].sum()),
            "primary_rejections_arbitrary_dependence": int(
                primary_results["reject_fdr_arbitrary_dependence"].sum()
            ),
            "negative_controls_rejecting_fwer": int(
                control_results["diagnostic_reject_fwer"].sum()
            ),
            "negative_controls_with_raw_p_below_0_05": int(
                control_results["bootstrap_p_value"].lt(0.05).sum()
            ),
            "primary_rejections": int(primary_results["reject_fdr"].sum()),
            "sign_agreement_count": int(primary_results["sign_agreement"].sum()),
        },
        "input_hashes": {
            controls_path.name: sha256_file(controls_path),
            primary_path.name: sha256_file(primary_path),
            "research.yaml": sha256_file(config_path),
            universe_summary_path.name: sha256_file(universe_summary_path),
        },
        "output_hashes": {
            control_replicates_path.name: sha256_file(control_replicates_path),
            control_results_path.name: sha256_file(control_results_path),
            primary_replicates_path.name: sha256_file(primary_replicates_path),
            primary_results_path.name: sha256_file(primary_results_path),
        },
        "random_seed": config.random_seed,
        "snapshot": snapshot.name,
    }
    write_json_atomic(output_dir / "inference_summary.json", summary)
    return output_dir
