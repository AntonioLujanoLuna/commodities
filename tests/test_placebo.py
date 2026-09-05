import math

import pandas as pd
import pytest

from enso_commodities.placebo import (
    build_placebo_endpoints,
    draw_calendar_matched_anchors,
    eligible_neutral_anchors,
    evaluate_placebo_means,
)


def test_neutral_anchors_respect_threshold_bounds_and_event_exclusion() -> None:
    dates = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
    index_data = pd.DataFrame({"date": dates, "roni": [0.0] * len(dates)})
    index_data.loc[index_data["date"].eq("2000-03-01"), "roni"] = -0.5
    episodes = pd.DataFrame({"onset_date": [pd.Timestamp("2000-06-01")]})

    result = eligible_neutral_anchors(
        index_data,
        episodes,
        index_name="roni",
        neutral_absolute_threshold=0.5,
        exclusion_window=(-1, 1),
        first_anchor=pd.Timestamp("2000-02-01"),
        last_anchor=pd.Timestamp("2001-11-01"),
    )

    selected = set(result["date"])
    assert pd.Timestamp("2000-03-01") not in selected
    assert (
        not {
            pd.Timestamp("2000-05-01"),
            pd.Timestamp("2000-06-01"),
            pd.Timestamp("2000-07-01"),
        }
        & selected
    )
    assert result["date"].min() == pd.Timestamp("2000-02-01")
    assert result["date"].max() == pd.Timestamp("2001-11-01")


def test_placebo_endpoint_uses_strict_complete_adjusted_path() -> None:
    rows = []
    for commodity in ["A", "B"]:
        for date, value in zip(
            pd.date_range("2000-01-01", periods=3, freq="MS"),
            [0.01, 0.02, 0.03],
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "commodity": commodity,
                    "market_adjusted_log_return": value,
                }
            )
    monthly = pd.DataFrame(rows)
    monthly.loc[
        monthly["commodity"].eq("B") & monthly["date"].eq("2000-02-01"),
        "market_adjusted_log_return",
    ] = float("nan")

    result = build_placebo_endpoints(
        monthly,
        pd.Series([pd.Timestamp("2000-01-01")]),
        ["A", "B"],
        horizon_months=2,
    ).set_index("commodity")

    assert result.loc["A", "placebo_return"] == pytest.approx(math.exp(0.06) - 1)
    assert pd.isna(result.loc["B", "placebo_return"])
    assert result.loc["B", "placebo_quality_flag"] == "incomplete_adjusted_return_path"


def test_calendar_matched_draws_are_deterministic_and_unique() -> None:
    eligible = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["1999-01-01", "2000-01-01", "2001-01-01", "1999-02-01", "2000-02-01"]
            ),
            "calendar_month": [1, 1, 1, 2, 2],
        }
    )
    real_onsets = pd.Series(pd.to_datetime(["1980-01-01", "1981-01-01", "1982-02-01"]))

    first = draw_calendar_matched_anchors(eligible, real_onsets, replicates=20, seed=42)
    second = draw_calendar_matched_anchors(eligible, real_onsets, replicates=20, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["replicate", "anchor_date"]).any()
    counts = first.groupby(["replicate", "matched_calendar_month"]).size().unstack(fill_value=0)
    assert counts[1].eq(2).all()
    assert counts[2].eq(1).all()


def test_calendar_matched_draws_can_use_replacement_for_sparse_cells() -> None:
    eligible = pd.DataFrame({"date": pd.to_datetime(["2000-01-01"]), "calendar_month": [1]})
    real_onsets = pd.Series(pd.to_datetime(["1980-01-01", "1981-01-01"]))

    result = draw_calendar_matched_anchors(
        eligible,
        real_onsets,
        replicates=3,
        seed=42,
        sample_without_replacement=False,
    )

    assert len(result) == 6
    assert result.groupby("replicate").size().eq(2).all()


def test_equal_tailed_placebo_test_detects_extreme_observed_mean() -> None:
    anchors = pd.date_range("1990-01-01", periods=99, freq="MS")
    endpoints = pd.DataFrame(
        {"anchor_date": anchors, "commodity": ["A"] * 99, "placebo_return": [0.1] * 99}
    )
    draws = pd.DataFrame(
        {"replicate": range(99), "anchor_date": anchors, "draw_position": [0] * 99}
    )
    observed = pd.DataFrame({"commodity": ["A"], "mean_return": [1.0]})

    output = evaluate_placebo_means(
        endpoints,
        draws,
        observed,
        replicates=99,
        minimum_valid_episodes=1,
        minimum_valid_replicate_share=1.0,
        confidence_level=0.95,
    )

    result = output.results.iloc[0]
    assert result["placebo_status"] == "estimated"
    assert result["placebo_mean_return"] == pytest.approx(0.1)
    assert result["placebo_p_value"] == pytest.approx(0.02)
