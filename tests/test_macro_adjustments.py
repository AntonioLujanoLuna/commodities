import pandas as pd
import pytest

from enso_commodities.macro_adjustments import fit_macro_adjusted_returns


def test_macro_model_is_fit_only_outside_event_window_and_recovers_coefficients() -> None:
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    rows = []
    macro_rows = []
    for index, date in enumerate(dates):
        market = ((index % 7) - 3) / 100
        neer = ((index % 11) - 5) / 200
        cpi = ((index % 13) - 6) / 300
        activity = ((index % 17) - 8) / 10
        outcome = 0.01 + 2 * market - 3 * neer + 4 * cpi + 0.05 * activity
        rows.append(
            {
                "date": date,
                "commodity": "Cocoa",
                "seasonal_adjusted_log_return": outcome,
                "market_seasonal_adjusted_log_return": market,
            }
        )
        macro_rows.append(
            {
                "date": date,
                "us_neer_log_change": neer,
                "us_cpi_log_change": cpi,
                "global_real_activity_change": activity,
            }
        )
    episodes = pd.DataFrame({"onset_date": [pd.Timestamp("2005-01-01")]})

    adjusted, models = fit_macro_adjusted_returns(
        pd.DataFrame(rows),
        pd.DataFrame(macro_rows),
        episodes,
        control_columns=(
            "market_seasonal_adjusted_log_return",
            "us_neer_log_change",
            "us_cpi_log_change",
            "global_real_activity_change",
        ),
        estimation_exclusion_window=(-12, 24),
        minimum_model_observations=60,
    )

    model = models.iloc[0]
    assert model["macro_model_status"] == "estimated"
    assert model["macro_model_observations"] == 83
    assert model["macro_alpha"] == pytest.approx(0.01)
    assert model["macro_beta_market"] == pytest.approx(2.0)
    assert model["macro_beta_us_neer"] == pytest.approx(-3.0)
    assert model["macro_beta_us_cpi"] == pytest.approx(4.0)
    assert model["macro_beta_global_activity"] == pytest.approx(0.05)
    assert adjusted["macro_adjusted_log_return"].abs().max() < 1e-12
