import pandas as pd

from enso_commodities.financial_adjustments import fit_financial_adjusted_returns


def test_financial_adjustment_estimates_requested_controls() -> None:
    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    controls = pd.DataFrame({"date": dates, "x": range(12)})
    monthly = pd.DataFrame(
        {
            "date": dates,
            "commodity": "A",
            "seasonal_adjusted_log_return": [1 + 2 * value for value in range(12)],
            "market_seasonal_adjusted_log_return": 0.0,
        }
    )
    adjusted, models = fit_financial_adjusted_returns(
        monthly,
        controls,
        pd.DataFrame({"onset_date": pd.Series(dtype="datetime64[ns]")}),
        control_columns=("x",),
        exclusion_window=(-1, 1),
        minimum_observations=10,
    )
    assert models.loc[0, "financial_model_status"] == "estimated"
    assert adjusted["financial_adjusted_log_return"].abs().max() < 1e-10
