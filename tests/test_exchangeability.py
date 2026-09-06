from __future__ import annotations

import pandas as pd
import pytest

from enso_commodities.exchangeability import assign_era, era_balance


def _anchors(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "calendar_month": dates.month})


def _draws(pool: pd.DatetimeIndex, *, size: int, replicates: int, seed: int) -> pd.DataFrame:
    generator = pd.Series(pool)
    rows = []
    for replicate in range(replicates):
        sampled = generator.sample(n=size, random_state=seed + replicate, replace=False)
        for anchor in sampled:
            rows.append({"replicate": replicate, "anchor_date": anchor})
    return pd.DataFrame.from_records(rows)


def test_assign_era_groups_years_into_blocks() -> None:
    dates = pd.to_datetime(["1961-03-01", "1969-12-01", "1970-01-01", "1985-06-01"])
    eras = assign_era(pd.Series(dates), era_length_years=10, origin_year=1960)
    assert eras.tolist() == [1960, 1960, 1970, 1980]


def test_a_pool_confined_to_the_late_sample_is_detected() -> None:
    """The failure mode the diagnostic exists for.

    Onsets spread across the whole sample; eligible anchors survive only in its
    second half. The placebo then compares event windows against windows drawn
    from a different era, which is exactly the confound that would make a
    financial negative control reject for a non-ENSO reason.
    """
    onsets = pd.Series(pd.date_range("1960-06-01", "2020-06-01", freq="48MS"))
    pool = pd.date_range("1995-01-01", "2020-12-01", freq="MS")
    output = era_balance(
        onsets, _anchors(pool), _draws(pool, size=len(onsets), replicates=400, seed=1)
    )
    statistics = output.statistics.set_index("statistic")
    assert statistics.loc["mean_year", "p_value"] <= 0.01
    assert statistics.loc["mean_year", "observed"] < statistics.loc["mean_year", "placebo_mean"]


def test_a_pool_spanning_the_same_history_is_not_flagged() -> None:
    onsets = pd.Series(pd.date_range("1960-06-01", "2020-06-01", freq="48MS"))
    pool = pd.date_range("1960-01-01", "2020-12-01", freq="MS")
    output = era_balance(
        onsets, _anchors(pool), _draws(pool, size=len(onsets), replicates=400, seed=7)
    )
    statistics = output.statistics.set_index("statistic")
    assert statistics.loc["mean_year", "p_value"] > 0.05


def test_era_composition_accounts_for_every_onset_and_anchor() -> None:
    onsets = pd.Series(pd.date_range("1962-06-01", "2018-06-01", freq="60MS"))
    pool = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    output = era_balance(
        onsets,
        _anchors(pool),
        _draws(pool, size=len(onsets), replicates=50, seed=3),
        era_length_years=20,
    )
    eras = output.eras
    assert eras["onsets"].sum() == len(onsets)
    assert eras["eligible_anchors"].sum() == len(pool)
    assert eras["onset_share"].sum() == pytest.approx(1.0)
    assert eras["drawn_anchor_share"].sum() == pytest.approx(1.0)
    assert (eras["era_length_years"] == 20).all()


def test_missing_columns_are_refused() -> None:
    onsets = pd.Series(pd.to_datetime(["1997-06-01"]))
    pool = pd.date_range("1990-01-01", "1992-12-01", freq="MS")
    with pytest.raises(ValueError, match="replicate and anchor_date"):
        era_balance(onsets, _anchors(pool), pd.DataFrame({"anchor_date": pool}))
    with pytest.raises(ValueError, match="missing date"):
        era_balance(
            onsets,
            pd.DataFrame({"anchor": pool}),
            _draws(pool, size=1, replicates=5, seed=1),
        )
