import math

import pandas as pd
import pytest

from enso_commodities.adjustments import (
    add_adjusted_event_paths,
    adjust_monthly_returns,
    build_non_event_mask,
    build_outside_episode_core_mask,
    compute_market_factor,
)


def test_non_event_mask_excludes_configured_window() -> None:
    dates = pd.Series(pd.date_range("1999-11-01", "2000-05-01", freq="MS"))
    episodes = pd.DataFrame(
        {
            "onset_date": pd.to_datetime(["2000-02-01"]),
            "end_date": pd.to_datetime(["2000-03-01"]),
        }
    )

    mask = build_non_event_mask(
        dates,
        episodes,
        first_relative_month=-1,
        last_relative_month=2,
    )

    included = dates[mask].dt.strftime("%Y-%m").tolist()
    assert included == ["1999-11", "1999-12", "2000-05"]
    seasonal_included = dates[build_outside_episode_core_mask(dates, episodes)]
    assert seasonal_included.dt.strftime("%Y-%m").tolist() == [
        "1999-11",
        "1999-12",
        "2000-01",
        "2000-04",
        "2000-05",
    ]


def test_adjustment_recovers_known_market_loading() -> None:
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    market_returns = [None] + [0.01 if date.year % 2 == 0 else -0.01 for date in dates[1:]]
    levels = [100.0]
    for value in market_returns[1:]:
        assert value is not None
        levels.append(levels[-1] * math.exp(value))
    market_index = pd.DataFrame({"date": dates, "world_bank_total_index": levels})
    factor = compute_market_factor(market_index)
    seasonal_effect = pd.Series(dates.month, dtype="float64") / 1000
    monthly = pd.DataFrame(
        {
            "date": dates,
            "commodity": ["Cocoa"] * len(dates),
            "raw_log_return": seasonal_effect + 2 * factor["market_log_return"].fillna(0),
        }
    )
    monthly.loc[0, "raw_log_return"] = pd.NA
    episodes = pd.DataFrame({"onset_date": pd.Series(dtype="datetime64[ns]")})

    adjusted, seasonality, models, _ = adjust_monthly_returns(
        monthly,
        market_index,
        episodes,
        market_exclusion_window=(-12, 24),
        minimum_seasonal_observations=8,
        minimum_market_model_observations=60,
    )

    assert len(seasonality) == 12
    model = models.iloc[0]
    assert model["market_model_status"] == "estimated"
    assert model["market_beta"] == pytest.approx(2.0)
    residuals = adjusted["market_adjusted_log_return"].dropna()
    assert residuals.abs().max() < 1e-12


def test_adjusted_event_paths_stop_after_a_missing_month() -> None:
    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    paths = pd.DataFrame(
        {
            "episode_id": ["event"] * 4,
            "anchor_type": ["retrospective"] * 4,
            "commodity": ["Cocoa"] * 4,
            "relative_month": [-1, 0, 1, 2],
            "date": dates,
            "base_value": [100.0] * 4,
        }
    )
    monthly = pd.DataFrame(
        {
            "date": dates,
            "commodity": ["Cocoa"] * 4,
            "seasonal_adjusted_log_return": [0.0, 0.1, None, 0.2],
            "market_adjusted_log_return": [0.0, 0.05, None, 0.1],
        }
    )

    result = add_adjusted_event_paths(paths, monthly, base_relative_month=-1).set_index(
        "relative_month"
    )

    assert result.loc[-1, "seasonal_adjusted_cumulative_log_return"] == 0
    assert result.loc[0, "seasonal_adjusted_cumulative_log_return"] == pytest.approx(0.1)
    assert pd.isna(result.loc[1, "seasonal_adjusted_cumulative_log_return"])
    assert pd.isna(result.loc[2, "seasonal_adjusted_cumulative_log_return"])
