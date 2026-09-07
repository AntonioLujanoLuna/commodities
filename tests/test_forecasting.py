from __future__ import annotations

import numpy as np
import pandas as pd

from enso_commodities.forecasting import (
    expanding_window_forecasts,
    prepare_forecast_frame,
    summarize_forecasts,
)


def test_expanding_forecast_uses_only_prior_training_rows() -> None:
    generator = np.random.default_rng(44)
    dates = pd.date_range("1980-01-01", periods=300, freq="MS")
    roni = np.sin(np.arange(len(dates)) * 2 * np.pi / 48)
    returns = 0.025 * np.roll(roni, 1) + generator.normal(0, 0.01, len(dates))
    frame = prepare_forecast_frame(
        pd.DataFrame(
            {"date": dates, "market_adjusted_log_return": returns, "roni": roni}
        ),
        horizon_months=3,
    )
    forecasts = expanding_window_forecasts(
        frame, minimum_training_months=120, origin_frequency_months=12
    )
    assert forecasts["training_observations"].is_monotonic_increasing
    assert (forecasts["origin_date"] < forecasts["target_end_date"]).all()
    assert (forecasts["latest_training_target_end_date"] <= forecasts["origin_date"]).all()
    summary = summarize_forecasts(forecasts, transaction_cost_bps=10, seed=8)
    assert summary.observations == len(forecasts)
    assert np.isfinite(summary.net_strategy_return)


def test_future_outcome_change_cannot_change_earlier_prediction() -> None:
    dates = pd.date_range("1980-01-01", periods=220, freq="MS")
    base = pd.DataFrame(
        {
            "date": dates,
            "market_adjusted_log_return": np.sin(np.arange(220) / 7) / 100,
            "roni": np.cos(np.arange(220) / 11),
        }
    )
    first = expanding_window_forecasts(
        prepare_forecast_frame(base, horizon_months=3),
        minimum_training_months=120,
        origin_frequency_months=12,
    )
    changed = base.copy()
    changed.loc[changed.index[-10:], "market_adjusted_log_return"] = 99.0
    second = expanding_window_forecasts(
        prepare_forecast_frame(changed, horizon_months=3),
        minimum_training_months=120,
        origin_frequency_months=12,
    )
    assert first.iloc[0]["enso_prediction"] == second.iloc[0]["enso_prediction"]
