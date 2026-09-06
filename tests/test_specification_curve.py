from __future__ import annotations

import pandas as pd
import pytest

from enso_commodities.specification_curve import (
    circular_shift_enso,
    joint_null_p_value,
    summarize_curve,
)


def test_whole_year_shift_preserves_dates_and_calendar_phase() -> None:
    dates = pd.date_range("2000-01-01", periods=36, freq="MS")
    enso = pd.DataFrame({"date": dates, "roni": range(36), "oni": range(100, 136)})
    shifted = circular_shift_enso(enso, years=1, columns=("roni", "oni"))
    assert shifted["date"].equals(enso["date"])
    assert shifted.loc[12, "roni"] == 0
    assert shifted.loc[0, "roni"] == 24
    assert shifted.loc[12, "date"].month == enso.loc[0, "date"].month


def test_curve_summary_and_finite_sample_joint_p_value() -> None:
    cells = pd.DataFrame(
        {
            "shift_years": [0, 0, 1, 1, 2, 2],
            "is_observed_timing": [True, True, False, False, False, False],
            "cell": ["a", "b"] * 3,
            "estimate": [2.0, 1.0, 0.5, -0.5, 0.2, 0.1],
            "naive_t": [4.0, 2.0, 1.0, -1.0, 0.4, 0.2],
        }
    )
    summary = summarize_curve(cells, minimum_valid_cell_share=1.0)
    assert summary.loc[summary["is_observed_timing"], "median_absolute_naive_t"].item() == 3
    assert joint_null_p_value(summary, "median_absolute_naive_t") == pytest.approx(1 / 3)
