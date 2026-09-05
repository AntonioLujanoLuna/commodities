from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_research_config, project_root
from .enso import construct_warm_episodes
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .returns import build_event_return_paths, compute_monthly_returns


def latest_processed_snapshot(processed_root: Path | None = None) -> Path:
    root = processed_root or project_root() / "data" / "processed"
    candidates = sorted(path for path in root.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No processed real-data snapshots found under {root}")
    return candidates[-1]


def build_raw_event_tables(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    summary_path = snapshot / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        input_summary = json.load(handle)
    if input_summary.get("data_provenance") != "real":
        raise ValueError(f"{summary_path}: only real-data inputs are accepted")
    expected_processed_hashes = input_summary.get("output_hashes")
    if not isinstance(expected_processed_hashes, dict):
        raise ValueError(f"{summary_path}: processed output hashes are missing")
    verify_hashes(
        snapshot,
        {
            name: expected_processed_hashes[name]
            for name in ["enso_monthly.csv", "commodity_prices_monthly.parquet"]
        },
    )

    config = load_research_config(research_config_path)
    enso_path = snapshot / "enso_monthly.csv"
    prices_path = snapshot / "commodity_prices_monthly.parquet"
    enso = pd.read_csv(enso_path, parse_dates=["date"])
    prices = pd.read_parquet(prices_path)
    episodes = construct_warm_episodes(
        enso,
        index_name=config.primary_index,
        threshold=config.warm_threshold,
        minimum_duration_months=config.minimum_duration_months,
        observable_delay_after_center_months=config.observable_delay_after_center_months,
    )
    monthly_returns = compute_monthly_returns(prices)
    event_paths = build_event_return_paths(
        prices,
        episodes,
        first_relative_month=config.first_relative_month,
        last_relative_month=config.last_relative_month,
        base_relative_month=config.base_relative_month,
        anchor_types=config.anchor_types,
    )
    horizons = event_paths.loc[event_paths["relative_month"].isin(config.reported_horizons)].copy()
    coverage = (
        horizons.groupby(["anchor_type", "relative_month"], observed=True)
        .agg(
            possible_returns=("raw_cumulative_return", "size"),
            valid_returns=("raw_cumulative_return", "count"),
        )
        .reset_index()
    )
    coverage["coverage_share"] = coverage["valid_returns"].div(coverage["possible_returns"])

    output_root = tables_root or project_root() / "tables"
    output_dir = output_root / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "enso_episodes.csv"
    monthly_return_path = output_dir / "commodity_returns_monthly.parquet"
    event_path = output_dir / "raw_event_return_paths.parquet"
    horizon_path = output_dir / "raw_event_returns_horizons.parquet"
    coverage_path = output_dir / "raw_event_return_coverage.csv"
    episodes.to_csv(episode_path, index=False, date_format="%Y-%m-%d")
    monthly_returns.to_parquet(monthly_return_path, index=False)
    event_paths.to_parquet(event_path, index=False)
    horizons.to_parquet(horizon_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    output_summary: dict[str, Any] = {
        "anchors": list(config.anchor_types),
        "base_relative_month": config.base_relative_month,
        "data_provenance": "real",
        "episodes": {
            "count": len(episodes),
            "first_onset": episodes["onset_date"].min().date().isoformat(),
            "last_onset": episodes["onset_date"].max().date().isoformat(),
        },
        "event_paths": {
            "episodes_with_any_valid_return": int(
                event_paths.loc[
                    event_paths["raw_cumulative_return"].notna(), "episode_id"
                ].nunique()
            ),
            "first_relative_month": config.first_relative_month,
            "last_relative_month": config.last_relative_month,
            "rows": len(event_paths),
            "valid_returns": int(event_paths["raw_cumulative_return"].notna().sum()),
        },
        "input_hashes": {
            "enso_monthly.csv": sha256_file(enso_path),
            "commodity_prices_monthly.parquet": sha256_file(prices_path),
            "input_summary.json": sha256_file(summary_path),
            "research.yaml": sha256_file(
                research_config_path or project_root() / "config" / "research.yaml"
            ),
        },
        "monthly_returns": {
            "rows": len(monthly_returns),
            "valid_simple_returns": int(monthly_returns["raw_simple_return"].notna().sum()),
        },
        "output_hashes": {
            coverage_path.name: sha256_file(coverage_path),
            episode_path.name: sha256_file(episode_path),
            event_path.name: sha256_file(event_path),
            horizon_path.name: sha256_file(horizon_path),
            monthly_return_path.name: sha256_file(monthly_return_path),
        },
        "primary_index": config.primary_index,
        "reported_horizons": list(config.reported_horizons),
        "snapshot": snapshot.name,
        "warm_threshold": config.warm_threshold,
    }
    write_json_atomic(output_dir / "raw_event_summary.json", output_summary)
    return output_dir
