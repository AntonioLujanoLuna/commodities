from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enso_commodities.adjustments import add_adjusted_event_paths
from enso_commodities.climate_data import parse_psl_index
from enso_commodities.returns import build_event_return_paths
from enso_commodities.surrogate_treatment import (
    endpoint_from_monthly,
    moment_match,
    phase_randomized_surrogate,
)

PSL_FILE = """  1950         1952
 1950    0.100    0.200    0.300    0.400    0.500    0.600    0.700    0.800    0.900    1.000    1.100    1.200
 1951   -0.100   -0.200   -0.300   -0.400   -0.500   -0.600   -0.700   -0.800   -0.900   -1.000   -1.100   -1.200
 1952    0.000    0.000    0.000    0.000    0.000    0.000  -99.990  -99.990  -99.990  -99.990  -99.990  -99.990
  -99.99
  A trailer line that is not data
"""


def _monthly_index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("1950-01-01", periods=periods, freq="MS")


def test_psl_parser_reads_year_rows_and_honours_the_sentinel(tmp_path: Path) -> None:
    path = tmp_path / "psl_test.data"
    path.write_text(PSL_FILE, encoding="utf-8")
    with pytest.raises(ValueError, match="twenty years"):
        parse_psl_index(path, "index")

    long_rows = ["  1950         1979"]
    for year in range(1950, 1980):
        long_rows.append(f" {year}" + "".join(f"{0.01 * month:9.3f}" for month in range(1, 13)))
    long_rows.append("  -99.99")
    long_rows.append("  trailer")
    path.write_text("\n".join(long_rows) + "\n", encoding="utf-8")
    parsed = parse_psl_index(path, "index")
    assert len(parsed) == 30 * 12
    assert parsed["date"].iloc[0] == pd.Timestamp("1950-01-01")
    assert parsed["index"].notna().all()


def test_psl_parser_requires_a_sentinel_line(tmp_path: Path) -> None:
    path = tmp_path / "psl_no_sentinel.data"
    rows = ["  1950         1979"]
    for year in range(1950, 1980):
        rows.append(f" {year}" + "".join(f"{0.01 * month:9.3f}" for month in range(1, 13)))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sentinel"):
        parse_psl_index(path, "index")


def test_moment_matching_is_the_identity_for_the_reference_series() -> None:
    rng = np.random.default_rng(7)
    reference = pd.Series(rng.normal(0.3, 1.4, size=400))
    matched = moment_match(reference, reference)
    assert float((matched - reference).abs().max()) < 1e-10

    scaled = reference * 25.0 + 100.0
    remapped = moment_match(scaled, reference)
    assert float((remapped - reference).abs().max()) < 1e-9


def test_phase_randomization_preserves_the_spectrum_and_moves_the_timing() -> None:
    index = _monthly_index(600)
    trend = np.sin(np.arange(600) / 11.0)
    values = pd.Series(trend + 0.2 * np.arange(600) % 1.0, index=index)
    rng = np.random.default_rng(11)
    surrogate = phase_randomized_surrogate(values, rng=rng, preserve_seasonality=False)

    observed_power = np.abs(np.fft.rfft(values.to_numpy() - values.mean()))
    surrogate_power = np.abs(np.fft.rfft(surrogate.to_numpy() - surrogate.mean()))
    assert np.allclose(observed_power, surrogate_power, atol=1e-8)
    assert surrogate.std() == pytest.approx(values.std(), rel=1e-6)
    assert abs(float(surrogate.corr(values))) < 0.9


def test_seasonal_preserving_surrogates_keep_the_calendar_month_climatology() -> None:
    index = _monthly_index(480)
    rng = np.random.default_rng(3)
    seasonal = np.tile(np.arange(12) / 6.0, 40)
    values = pd.Series(seasonal + rng.normal(0, 0.3, size=480), index=index)
    surrogate = phase_randomized_surrogate(values, rng=rng, preserve_seasonality=True)
    months = pd.Series(index.month, index=index)
    observed = values.groupby(months).mean()
    produced = surrogate.groupby(months).mean()
    assert np.allclose(observed.to_numpy(), produced.to_numpy(), atol=1e-8)


def test_phase_randomization_refuses_a_gapped_series() -> None:
    values = pd.Series([1.0, np.nan, 2.0], index=_monthly_index(3))
    with pytest.raises(ValueError, match="gap-free"):
        phase_randomized_surrogate(values, rng=np.random.default_rng(0), preserve_seasonality=False)


def test_fast_endpoint_matches_the_frozen_event_path_construction() -> None:
    index = _monthly_index(120)
    rng = np.random.default_rng(19)
    frames = []
    for name in ("Alpha", "Beta"):
        frame = pd.DataFrame(
            {
                "date": index,
                "commodity": name,
                "unit": "usd",
                "value": 100.0 + np.arange(120),
                "adjusted_log_return": rng.normal(0.0, 0.05, size=120),
            }
        )
        frames.append(frame)
    monthly = pd.concat(frames, ignore_index=True)
    # A hole in one series must make its endpoint undefined, not merely shorter.
    monthly.loc[
        monthly["commodity"].eq("Beta") & monthly["date"].eq(pd.Timestamp("1953-04-01")),
        "adjusted_log_return",
    ] = np.nan
    episodes = pd.DataFrame(
        {
            "episode_id": ["e1", "e2"],
            "onset_date": [pd.Timestamp("1953-01-01"), pd.Timestamp("1956-06-01")],
        }
    )

    paths = build_event_return_paths(
        monthly,
        episodes.assign(observable_date=episodes["onset_date"]),
        first_relative_month=-12,
        last_relative_month=24,
        base_relative_month=-1,
        anchor_types=("retrospective",),
    )
    frozen = add_adjusted_event_paths(
        paths,
        monthly,
        base_relative_month=-1,
        monthly_columns=["adjusted_log_return"],
    )
    frozen_endpoint = frozen.loc[
        frozen["relative_month"].eq(12),
        ["episode_id", "commodity", "adjusted_cumulative_return"],
    ]
    fast = endpoint_from_monthly(
        monthly,
        episodes,
        anchor_column="onset_date",
        base_relative_month=-1,
        horizon_months=12,
        monthly_return_column="adjusted_log_return",
    )
    merged = frozen_endpoint.merge(
        fast, on=["episode_id", "commodity"], how="left", suffixes=("_frozen", "_fast")
    )
    both = merged.dropna(subset=["adjusted_cumulative_return_frozen"])
    assert not both.empty
    assert (
        float(
            both["adjusted_cumulative_return_frozen"]
            .sub(both["adjusted_cumulative_return_fast"])
            .abs()
            .max()
        )
        < 1e-12
    )
    missing = merged.loc[merged["adjusted_cumulative_return_frozen"].isna()]
    assert missing["adjusted_cumulative_return_fast"].isna().all()
