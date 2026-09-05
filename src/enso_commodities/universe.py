from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_research_config, project_root
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


@dataclass(frozen=True)
class InferenceSpec:
    primary_anchor: str
    primary_horizon_months: int
    primary_return: str
    alternative: str
    minimum_valid_episodes: int
    fdr_family: str


@dataclass(frozen=True)
class CommodityRegistry:
    inference: InferenceSpec
    entries: pd.DataFrame
    expected_counts: dict[str, int]


def load_commodity_registry(path: Path | None = None) -> CommodityRegistry:
    registry_path = path or project_root() / "config" / "commodities.yaml"
    with registry_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    inference = InferenceSpec(**raw["inference"])
    records: list[dict[str, object]] = []
    for item in raw["mechanism_candidates"]:
        records.append(
            {
                "commodity": item["name"],
                "role": "mechanism_candidate",
                "group": item["group"],
                "selection_note": item["pathway"],
            }
        )
    for item in raw["negative_controls"]:
        records.append(
            {
                "commodity": item["name"],
                "role": "negative_control",
                "group": item["group"],
                "selection_note": item["rationale"],
            }
        )
    for item in raw["excluded"]:
        records.append(
            {
                "commodity": item["name"],
                "role": "excluded",
                "group": None,
                "selection_note": item["reason"],
            }
        )
    entries = pd.DataFrame.from_records(records)
    expected_counts = {key: int(value) for key, value in raw["expected_counts"].items()}
    return CommodityRegistry(
        inference=inference,
        entries=entries,
        expected_counts=expected_counts,
    )


def validate_registry(
    registry: CommodityRegistry,
    *,
    source_commodities: set[str],
    allowed_horizons: set[int],
    allowed_anchors: set[str],
) -> None:
    entries = registry.entries
    if entries["commodity"].duplicated().any():
        duplicates = sorted(entries.loc[entries["commodity"].duplicated(), "commodity"])
        raise ValueError(f"Commodity registry contains duplicates: {duplicates}")
    registered = set(entries["commodity"])
    if registered != source_commodities:
        raise ValueError(
            "Commodity registry does not exactly cover the source universe; "
            f"missing={sorted(source_commodities - registered)}, "
            f"unknown={sorted(registered - source_commodities)}"
        )
    role_counts = entries["role"].value_counts().to_dict()
    expected_role_counts = {
        "mechanism_candidate": registry.expected_counts["mechanism_candidates"],
        "negative_control": registry.expected_counts["negative_controls"],
        "excluded": registry.expected_counts["excluded"],
    }
    if role_counts != expected_role_counts:
        raise ValueError(f"Commodity registry count mismatch: {role_counts}")
    if len(entries) != registry.expected_counts["source_total"]:
        raise ValueError("Commodity registry source-total count mismatch")
    if registry.inference.primary_horizon_months not in allowed_horizons:
        raise ValueError("Primary horizon is not among the pre-specified reported horizons")
    if registry.inference.primary_anchor not in allowed_anchors:
        raise ValueError("Primary anchor is not supported")
    if registry.inference.alternative != "two_sided":
        raise ValueError("Only a two-sided primary alternative is supported")
    if registry.inference.fdr_family != "mechanism_candidates_only":
        raise ValueError("FDR family must contain mechanism candidates only")
    if registry.inference.minimum_valid_episodes < 3:
        raise ValueError("minimum_valid_episodes must be at least three")


def build_inference_universe(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    adjusted_summary_path = output_dir / "adjusted_event_summary.json"
    with adjusted_summary_path.open(encoding="utf-8") as handle:
        adjusted_summary = json.load(handle)
    if adjusted_summary.get("data_provenance") != "real":
        raise ValueError("Inference universe requires real-data adjusted returns")
    output_hashes = adjusted_summary.get("output_hashes")
    if not isinstance(output_hashes, dict):
        raise ValueError("Adjusted-event output hashes are missing")
    horizon_filename = "adjusted_event_returns_horizons.parquet"
    verify_hashes(output_dir, {horizon_filename: output_hashes[horizon_filename]})

    config = load_research_config()
    registry = load_commodity_registry(registry_path)
    horizon_data = pd.read_parquet(output_dir / horizon_filename)
    source_commodities = set(horizon_data["commodity"].unique())
    validate_registry(
        registry,
        source_commodities=source_commodities,
        allowed_horizons=set(config.reported_horizons),
        allowed_anchors=set(config.anchor_types),
    )
    spec = registry.inference
    if spec.primary_return not in horizon_data.columns:
        raise ValueError(f"Primary return column is missing: {spec.primary_return}")
    endpoint = horizon_data.loc[
        horizon_data["anchor_type"].eq(spec.primary_anchor)
        & horizon_data["relative_month"].eq(spec.primary_horizon_months)
    ].copy()
    if endpoint.duplicated(["episode_id", "commodity"]).any():
        raise ValueError("Primary endpoint contains duplicate episode/commodity keys")

    registry_table = registry.entries.copy()
    valid_counts = (
        endpoint.groupby("commodity", observed=True)[spec.primary_return]
        .count()
        .rename("valid_episode_count")
    )
    registry_table = registry_table.merge(
        valid_counts, left_on="commodity", right_index=True, how="left", validate="one_to_one"
    )
    registry_table["valid_episode_count"] = (
        registry_table["valid_episode_count"].fillna(0).astype(int)
    )
    registry_table["primary_eligible"] = registry_table["role"].eq(
        "mechanism_candidate"
    ) & registry_table["valid_episode_count"].ge(spec.minimum_valid_episodes)
    registry_table["control_eligible"] = registry_table["role"].eq(
        "negative_control"
    ) & registry_table["valid_episode_count"].ge(spec.minimum_valid_episodes)

    candidates = registry_table.loc[
        registry_table["role"].eq("mechanism_candidate"),
        ["commodity", "group", "selection_note", "valid_episode_count", "primary_eligible"],
    ]
    candidate_sample = endpoint.merge(
        candidates, on="commodity", how="inner", validate="many_to_one"
    )
    inference_family = candidate_sample.loc[
        candidate_sample["primary_eligible"] & candidate_sample[spec.primary_return].notna()
    ].copy()
    controls = registry_table.loc[
        registry_table["role"].eq("negative_control"),
        ["commodity", "group", "selection_note", "valid_episode_count", "control_eligible"],
    ]
    control_sample = endpoint.merge(controls, on="commodity", how="inner", validate="many_to_one")

    registry_output_path = output_dir / "commodity_registry.csv"
    candidate_path = output_dir / "primary_candidate_sample.parquet"
    family_path = output_dir / "primary_inference_family.parquet"
    control_path = output_dir / "negative_control_sample.parquet"
    registry_table.sort_values(["role", "group", "commodity"], na_position="last").to_csv(
        registry_output_path, index=False
    )
    candidate_sample.to_parquet(candidate_path, index=False)
    inference_family.to_parquet(family_path, index=False)
    control_sample.to_parquet(control_path, index=False)

    eligible = registry_table.loc[registry_table["primary_eligible"], "commodity"]
    ineligible = registry_table.loc[
        registry_table["role"].eq("mechanism_candidate") & ~registry_table["primary_eligible"],
        "commodity",
    ]
    summary = {
        "data_provenance": "real",
        "frozen_contract": {
            "alternative": spec.alternative,
            "fdr_family": spec.fdr_family,
            "minimum_valid_episodes": spec.minimum_valid_episodes,
            "primary_anchor": spec.primary_anchor,
            "primary_horizon_months": spec.primary_horizon_months,
            "primary_return": spec.primary_return,
        },
        "input_hashes": {
            adjusted_summary_path.name: sha256_file(adjusted_summary_path),
            horizon_filename: sha256_file(output_dir / horizon_filename),
            "commodities.yaml": sha256_file(
                registry_path or project_root() / "config" / "commodities.yaml"
            ),
        },
        "output_hashes": {
            candidate_path.name: sha256_file(candidate_path),
            control_path.name: sha256_file(control_path),
            family_path.name: sha256_file(family_path),
            registry_output_path.name: sha256_file(registry_output_path),
        },
        "universe": {
            "eligible_mechanism_candidates": len(eligible),
            "excluded_series": int(registry_table["role"].eq("excluded").sum()),
            "ineligible_mechanism_candidates": sorted(ineligible.tolist()),
            "mechanism_candidates": int(registry_table["role"].eq("mechanism_candidate").sum()),
            "negative_controls": int(registry_table["role"].eq("negative_control").sum()),
            "primary_inference_rows": len(inference_family),
            "source_series": len(registry_table),
        },
    }
    write_json_atomic(output_dir / "universe_summary.json", summary)
    return output_dir
