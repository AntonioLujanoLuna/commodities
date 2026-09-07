from __future__ import annotations

import json
from pathlib import Path

import yaml

from enso_commodities.adjusted_events import build_adjusted_event_tables
from enso_commodities.config import project_root
from enso_commodities.inference import run_primary_inference
from enso_commodities.provenance import sha256_file, verify_hashes, write_json_atomic
from enso_commodities.raw_events import build_raw_event_tables
from enso_commodities.synthetic import synthetic_market
from enso_commodities.universe import build_inference_universe


def test_synthetic_snapshot_crosses_the_receipt_writing_pipeline(tmp_path: Path) -> None:
    """Exercise four production stages and every receipt boundary with known fake data."""
    market = synthetic_market(seed=20260906, commodities=4, affected_commodities=1, years=45)
    snapshot = tmp_path / "processed" / "2026-09-06"
    tables_root = tmp_path / "tables"
    snapshot.mkdir(parents=True)
    market.enso.to_csv(snapshot / "enso_monthly.csv", index=False)
    market.prices.to_parquet(snapshot / "commodity_prices_monthly.parquet", index=False)
    market.market_index.to_csv(snapshot / "world_bank_indices_monthly.csv", index=False)
    write_json_atomic(
        snapshot / "summary.json",
        {
            "data_provenance": "real",
            "output_hashes": {
                name: sha256_file(snapshot / name)
                for name in ("enso_monthly.csv", "commodity_prices_monthly.parquet")
            },
        },
    )

    with (project_root() / "config" / "research.yaml").open(encoding="utf-8") as handle:
        research = yaml.safe_load(handle)
    research["inference"]["bootstrap_replicates"] = 999
    research["inference"]["minimum_studentized_replicate_share"] = 0.8
    research["adjustments"]["minimum_market_model_observations"] = 24
    research_path = tmp_path / "research.yaml"
    research_path.write_text(yaml.safe_dump(research, sort_keys=False), encoding="utf-8")

    registry = {
        "inference": {
            "primary_anchor": "retrospective",
            "primary_horizon_months": 12,
            "primary_return": "market_adjusted_cumulative_return",
            "alternative": "two_sided",
            "minimum_valid_episodes": 3,
            "fdr_family": "mechanism_candidates_only",
        },
        "expected_counts": {
            "mechanism_candidates": 2,
            "negative_controls": 1,
            "excluded": 1,
            "source_total": 4,
        },
        "mechanism_candidates": [
            {"name": "SYN-00", "group": "synthetic", "pathway": "planted"},
            {"name": "SYN-01", "group": "synthetic", "pathway": "null"},
        ],
        "negative_controls": [{"name": "SYN-02", "group": "synthetic", "rationale": "control"}],
        "excluded": [{"name": "SYN-03", "reason": "fixture exclusion"}],
    }
    registry_path = tmp_path / "commodities.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")

    output = build_raw_event_tables(
        snapshot, tables_root=tables_root, research_config_path=research_path
    )
    build_adjusted_event_tables(
        snapshot, tables_root=tables_root, research_config_path=research_path
    )
    build_inference_universe(snapshot, tables_root=tables_root, registry_path=registry_path)
    run_primary_inference(
        snapshot,
        tables_root=tables_root,
        registry_path=registry_path,
        research_config_path=research_path,
    )

    with (output / "inference_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["inference"]["bootstrap_replicates"] == 999
    assert summary["inference"]["fdr_family_size"] == 2
    verify_hashes(output, summary["output_hashes"])
