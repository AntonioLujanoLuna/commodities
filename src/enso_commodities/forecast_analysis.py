"""Real-data orchestration for the explicitly pseudo-out-of-sample forecast stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .forecasting import expanding_window_forecasts, prepare_forecast_frame, summarize_forecasts
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .validation import load_validation_contract


def run_forecast_analysis(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    validation_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or project_root() / "tables") / snapshot.name
    contract_path = validation_config_path or project_root() / "config" / "validation_v2.yaml"
    contract = load_validation_contract(contract_path)
    adjusted_receipt_path = output_dir / "adjusted_event_summary.json"
    with adjusted_receipt_path.open(encoding="utf-8") as handle:
        adjusted_receipt: dict[str, Any] = json.load(handle)
    if adjusted_receipt.get("data_provenance") != "real":
        raise ValueError("Forecast analysis requires real adjusted returns")
    monthly_name = "commodity_returns_adjusted_monthly.parquet"
    verify_hashes(output_dir, {monthly_name: adjusted_receipt["output_hashes"][monthly_name]})
    monthly = pd.read_parquet(output_dir / monthly_name)
    enso = pd.read_csv(snapshot / "enso_monthly.csv", parse_dates=["date"])
    joined = monthly.merge(enso[["date", "roni"]], on="date", how="left", validate="many_to_one")

    forecast_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for commodity in contract.commodities:
        commodity_data = joined.loc[joined["commodity"].eq(commodity)].copy()
        if commodity_data.empty:
            raise ValueError(f"Validation commodity is absent from monthly returns: {commodity}")
        for horizon in contract.forecast_horizons:
            prepared = prepare_forecast_frame(commodity_data, horizon_months=horizon)
            forecasts = expanding_window_forecasts(
                prepared,
                minimum_training_months=contract.forecast_minimum_training_months,
                origin_frequency_months=contract.forecast_origin_frequency,
            )
            forecasts.insert(0, "horizon_months", horizon)
            forecasts.insert(0, "commodity", commodity)
            forecast_frames.append(forecasts)
            summary = summarize_forecasts(
                forecasts,
                transaction_cost_bps=contract.transaction_cost_bps,
                seed=20260906 + horizon,
            )
            summaries.append({"commodity": commodity, "horizon_months": horizon, **summary.__dict__})

    forecast_table = pd.concat(forecast_frames, ignore_index=True)
    summary_table = pd.DataFrame.from_records(summaries)
    forecast_path = output_dir / "forecast_predictions.csv"
    results_path = output_dir / "forecast_results.csv"
    forecast_table.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")
    summary_table.to_csv(results_path, index=False)
    best = summary_table.sort_values("loss_difference_p_value", kind="mergesort").iloc[0]
    receipt = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "scope": "retrospective_pseudo_out_of_sample_price_index_proxy",
        "genuine_out_of_sample": False,
        "tradability_claim_permitted": False,
        "limitations": [
            "RONI values are final revised series rather than archived real-time vintages",
            "World Bank price indexes are not investable futures returns",
            "transaction costs are a configured proxy and exclude contract roll mechanics",
        ],
        "results": {
            "cells": len(summary_table),
            "cells_with_positive_rmse_improvement": int(summary_table["rmse_improvement"].gt(0).sum()),
            "cells_with_loss_difference_p_below_0_05": int(
                summary_table["loss_difference_p_value"].lt(0.05).sum()
            ),
            "cells_with_positive_net_excess_over_buy_and_hold": int(
                summary_table["net_excess_over_buy_and_hold"].gt(0).sum()
            ),
            "best_loss_test_commodity": str(best["commodity"]),
            "best_loss_test_horizon_months": int(best["horizon_months"]),
            "best_loss_test_p_value": float(best["loss_difference_p_value"]),
            "best_loss_test_rmse_improvement": float(best["rmse_improvement"]),
            "best_loss_test_net_excess_over_buy_and_hold": float(
                best["net_excess_over_buy_and_hold"]
            ),
        },
        "input_hashes": {
            adjusted_receipt_path.name: sha256_file(adjusted_receipt_path),
            monthly_name: sha256_file(output_dir / monthly_name),
            "enso_monthly.csv": sha256_file(snapshot / "enso_monthly.csv"),
            contract_path.name: sha256_file(contract_path),
        },
        "output_hashes": {
            forecast_path.name: sha256_file(forecast_path),
            results_path.name: sha256_file(results_path),
        },
    }
    write_json_atomic(output_dir / "forecast_summary.json", receipt)
    return output_dir
