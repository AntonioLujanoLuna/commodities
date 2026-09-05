from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .adjustments import add_adjusted_event_paths, adjust_monthly_returns
from .config import load_research_config, project_root
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


def build_adjusted_event_tables(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    research_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_root = tables_root or project_root() / "tables"
    raw_event_dir = output_root / snapshot.name
    raw_summary_path = raw_event_dir / "raw_event_summary.json"
    with raw_summary_path.open(encoding="utf-8") as handle:
        raw_summary = json.load(handle)
    if raw_summary.get("data_provenance") != "real":
        raise ValueError(f"{raw_summary_path}: only real-data inputs are accepted")
    expected_raw_hashes = raw_summary.get("output_hashes")
    if not isinstance(expected_raw_hashes, dict):
        raise ValueError(f"{raw_summary_path}: raw-event output hashes are missing")
    verify_hashes(
        raw_event_dir,
        {
            name: expected_raw_hashes[name]
            for name in [
                "commodity_returns_monthly.parquet",
                "enso_episodes.csv",
                "raw_event_return_paths.parquet",
            ]
        },
    )

    config = load_research_config(research_config_path)
    monthly_path = raw_event_dir / "commodity_returns_monthly.parquet"
    paths_path = raw_event_dir / "raw_event_return_paths.parquet"
    episodes_path = raw_event_dir / "enso_episodes.csv"
    market_index_path = snapshot / "world_bank_indices_monthly.csv"
    monthly = pd.read_parquet(monthly_path)
    paths = pd.read_parquet(paths_path)
    episodes = pd.read_csv(episodes_path, parse_dates=["onset_date"])
    market_index = pd.read_csv(market_index_path, parse_dates=["date"])

    adjusted, seasonality, models, factor = adjust_monthly_returns(
        monthly,
        market_index,
        episodes,
        market_exclusion_window=config.market_estimation_exclusion_window,
        minimum_seasonal_observations=config.minimum_seasonal_observations,
        minimum_market_model_observations=config.minimum_market_model_observations,
    )
    adjusted_paths = add_adjusted_event_paths(
        paths,
        adjusted,
        base_relative_month=config.base_relative_month,
    )
    horizons = adjusted_paths.loc[
        adjusted_paths["relative_month"].isin(config.reported_horizons)
    ].copy()
    coverage = (
        horizons.groupby(["anchor_type", "relative_month"], observed=True)
        .agg(
            possible_returns=("raw_cumulative_return", "size"),
            raw_valid_returns=("raw_cumulative_return", "count"),
            seasonal_valid_returns=("seasonal_adjusted_cumulative_return", "count"),
            market_valid_returns=("market_adjusted_cumulative_return", "count"),
        )
        .reset_index()
    )
    for prefix in ["raw", "seasonal", "market"]:
        coverage[f"{prefix}_coverage_share"] = coverage[f"{prefix}_valid_returns"].div(
            coverage["possible_returns"]
        )

    adjusted_path = raw_event_dir / "commodity_returns_adjusted_monthly.parquet"
    seasonality_path = raw_event_dir / "seasonal_estimates.csv"
    models_path = raw_event_dir / "market_models.csv"
    factor_path = raw_event_dir / "market_factor_monthly.csv"
    adjusted_paths_path = raw_event_dir / "adjusted_event_return_paths.parquet"
    horizons_path = raw_event_dir / "adjusted_event_returns_horizons.parquet"
    coverage_path = raw_event_dir / "adjusted_event_return_coverage.csv"
    adjusted.to_parquet(adjusted_path, index=False)
    seasonality.to_csv(seasonality_path, index=False)
    models.to_csv(models_path, index=False)
    factor.to_csv(factor_path, index=False, date_format="%Y-%m-%d")
    adjusted_paths.to_parquet(adjusted_paths_path, index=False)
    horizons.to_parquet(horizons_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    estimated_models = models["market_model_status"].eq("estimated")
    summary: dict[str, Any] = {
        "data_provenance": "real",
        "estimation": {
            "market_excluded_relative_months": list(config.market_estimation_exclusion_window),
            "market_factor": config.market_factor,
            "market_models_estimated": int(estimated_models.sum()),
            "market_models_not_estimated": int((~estimated_models).sum()),
            "minimum_market_model_observations": config.minimum_market_model_observations,
            "minimum_seasonal_observations": config.minimum_seasonal_observations,
            "market_estimation_months": int(factor["is_market_estimation_month"].sum()),
            "seasonal_estimation_excludes": config.seasonal_estimation_excludes,
            "seasonal_estimation_months": int(factor["is_seasonal_estimation_month"].sum()),
        },
        "input_hashes": {
            "commodity_returns_monthly.parquet": sha256_file(monthly_path),
            "enso_episodes.csv": sha256_file(episodes_path),
            "raw_event_return_paths.parquet": sha256_file(paths_path),
            "raw_event_summary.json": sha256_file(raw_summary_path),
            "research.yaml": sha256_file(
                research_config_path or project_root() / "config" / "research.yaml"
            ),
            "world_bank_indices_monthly.csv": sha256_file(market_index_path),
        },
        "returns": {
            "market_adjusted_event_returns": int(
                adjusted_paths["market_adjusted_cumulative_return"].notna().sum()
            ),
            "market_adjusted_monthly_returns": int(
                adjusted["market_adjusted_log_return"].notna().sum()
            ),
            "seasonal_adjusted_event_returns": int(
                adjusted_paths["seasonal_adjusted_cumulative_return"].notna().sum()
            ),
            "seasonal_adjusted_monthly_returns": int(
                adjusted["seasonal_adjusted_log_return"].notna().sum()
            ),
        },
        "output_hashes": {
            adjusted_path.name: sha256_file(adjusted_path),
            adjusted_paths_path.name: sha256_file(adjusted_paths_path),
            coverage_path.name: sha256_file(coverage_path),
            factor_path.name: sha256_file(factor_path),
            horizons_path.name: sha256_file(horizons_path),
            models_path.name: sha256_file(models_path),
            seasonality_path.name: sha256_file(seasonality_path),
        },
        "snapshot": snapshot.name,
    }
    write_json_atomic(raw_event_dir / "adjusted_event_summary.json", summary)
    return raw_event_dir
