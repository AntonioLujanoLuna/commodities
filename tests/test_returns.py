import pandas as pd
import pytest

from enso_commodities.returns import build_event_return_paths, compute_monthly_returns


def test_monthly_returns_do_not_bridge_missing_values_or_months() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2000-01-01", "2000-02-01", "2000-03-01", "2000-05-01", "2000-06-01"]
            ),
            "commodity": ["Cocoa"] * 5,
            "unit": ["$/kg"] * 5,
            "value": [100.0, 110.0, None, 121.0, 133.1],
        }
    )

    result = compute_monthly_returns(prices).set_index("date")

    assert result.loc["2000-02-01", "raw_simple_return"] == pytest.approx(0.1)
    assert pd.isna(result.loc["2000-03-01", "raw_simple_return"])
    assert result.loc["2000-03-01", "return_quality_flag"] == "missing_current_price"
    assert pd.isna(result.loc["2000-05-01", "raw_simple_return"])
    assert result.loc["2000-05-01", "return_quality_flag"] == "missing_previous_price"
    assert result.loc["2000-06-01", "raw_simple_return"] == pytest.approx(0.1)


def test_event_paths_use_month_before_each_anchor_as_base() -> None:
    dates = pd.date_range("1999-12-01", "2000-09-01", freq="MS")
    prices = pd.DataFrame(
        {
            "date": dates,
            "commodity": ["Cocoa"] * len(dates),
            "unit": ["$/kg"] * len(dates),
            "value": [100.0 + 10 * index for index in range(len(dates))],
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["roni_warm_2000_01"],
            "onset_date": pd.to_datetime(["2000-01-01"]),
            "observable_date": pd.to_datetime(["2000-07-01"]),
        }
    )

    result = build_event_return_paths(
        prices,
        episodes,
        first_relative_month=-1,
        last_relative_month=2,
        base_relative_month=-1,
        anchor_types=("retrospective", "observable"),
    )
    retrospective = result[result["anchor_type"] == "retrospective"].set_index("relative_month")
    observable = result[result["anchor_type"] == "observable"].set_index("relative_month")

    assert retrospective.loc[-1, "raw_cumulative_return"] == 0
    assert retrospective.loc[0, "raw_cumulative_return"] == pytest.approx(0.1)
    assert observable.loc[-1, "base_value"] == 160.0
    assert observable.loc[0, "raw_cumulative_return"] == pytest.approx(10 / 160)
