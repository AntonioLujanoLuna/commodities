from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enso_commodities.dispersion import (
    available_year_shifts,
    build_event_window_mask,
    build_monthly_matrix,
    dispersion_shift_inference,
    load_dispersion_config,
    load_program_register,
)

SPEC = load_dispersion_config()
PROGRAM = load_program_register()


def _dates(years: int) -> pd.DatetimeIndex:
    return pd.date_range("1960-01-01", periods=years * 12, freq="MS")


# ENSO recurs irregularly, at two to seven years. Exactly periodic onsets would
# alias onto themselves under whole-year shifts and rob the null of resolution,
# so the fixtures use irregular gaps as the real index does.
_GAPS_YEARS = (4, 3, 6, 2, 5, 3, 7, 4, 2, 5, 3, 6, 4, 3, 5)


def _episodes(dates: pd.DatetimeIndex) -> pd.DataFrame:
    positions: list[int] = []
    cursor = 6
    for gap in _GAPS_YEARS:
        if cursor >= len(dates) - 30:
            break
        positions.append(cursor)
        cursor += gap * 12
    onsets = [dates[index] for index in positions]
    return pd.DataFrame({"episode_id": [f"e{i}" for i in range(len(onsets))], "onset_date": onsets})


def _panel(dates: pd.DatetimeIndex, series: dict[str, np.ndarray]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"date": dates, "commodity": name, "seasonal_adjusted_log_return": values})
        for name, values in series.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _clustered_volatility(generator: np.random.Generator, months: int) -> np.ndarray:
    """A stationary series whose volatility persists, with no ENSO relationship.

    This is the process the shift null exists to survive: a test that treats
    months as exchangeable finds structure here that has nothing to do with the
    treatment.
    """
    log_variance = np.zeros(months)
    for position in range(1, months):
        log_variance[position] = 0.97 * log_variance[position - 1] + generator.normal(0, 0.3)
    return generator.normal(0, 1, months) * np.exp(log_variance / 2)


def test_event_window_mask_covers_exactly_the_frozen_window() -> None:
    dates = _dates(10)
    onsets = pd.Series([pd.Timestamp("1962-01-01")])
    mask = build_event_window_mask(dates, onsets, first_relative_month=-2, last_relative_month=3)
    assert dates[mask].min() == pd.Timestamp("1961-11-01")
    assert dates[mask].max() == pd.Timestamp("1962-04-01")
    assert int(mask.sum()) == 6


def test_year_shifts_exclude_the_identity_and_respect_the_sample() -> None:
    shifts = available_year_shifts(65 * 12, minimum_shift_years=1, maximum_shift_years=64)
    assert shifts[0] == 1
    assert shifts[-1] == 64
    assert 0 not in shifts
    assert available_year_shifts(20 * 12, minimum_shift_years=1, maximum_shift_years=64)[-1] == 19
    with pytest.raises(ValueError):
        available_year_shifts(12, minimum_shift_years=1, maximum_shift_years=4)


def test_monthly_matrix_materialises_missing_months_as_gaps() -> None:
    dates = pd.to_datetime(["1980-01-01", "1980-02-01", "1980-04-01"])
    frame = pd.DataFrame(
        {
            "date": dates,
            "commodity": ["crop"] * 3,
            "seasonal_adjusted_log_return": [0.1, 0.2, 0.3],
        }
    )
    grid, commodities, values = build_monthly_matrix(
        frame, outcome_column="seasonal_adjusted_log_return"
    )
    assert list(grid) == list(pd.date_range("1980-01-01", "1980-04-01", freq="MS"))
    assert commodities == ["crop"]
    assert np.isnan(values[2, 0])


def test_planted_variance_shift_is_detected_at_the_resolution_floor() -> None:
    generator = np.random.default_rng(11)
    dates = _dates(65)
    episodes = _episodes(dates)
    mask = build_event_window_mask(
        dates,
        episodes["onset_date"],
        first_relative_month=SPEC.window_first_relative_month,
        last_relative_month=SPEC.window_last_relative_month,
    )
    quiet = generator.normal(0, 0.05, len(dates))
    loud = quiet * np.where(mask, 3.0, 1.0)
    panel = _panel(dates, {"crop": loud, "gold": generator.normal(0, 0.05, len(dates))})
    roles = pd.Series({"crop": "mechanism_candidate", "gold": "negative_control"})
    results, null_frame, summary = dispersion_shift_inference(
        panel, episodes, roles, spec=SPEC, program=PROGRAM
    )
    crop = results.set_index("commodity").loc["crop"]
    assert crop["log_variance_ratio"] > 1.0
    assert crop["variance_shift_p_value"] == pytest.approx(summary["resolution_floor"])
    assert bool(crop["statistics_agree"])
    assert summary["negative_controls_rejected"] is False
    assert summary["family_shift_p_value"] == pytest.approx(summary["resolution_floor"])
    assert len(null_frame) == summary["shift_count"] * 2


def test_shift_null_does_not_reject_on_volatility_clustering_alone() -> None:
    """The stage's own falsification: persistent volatility must not fake a result.

    Returns whose variance drifts in long regimes are exactly what would break a
    test that treated months as exchangeable. The circular shift null never
    touches the return series, so its rejection rate here should sit near the
    nominal level rather than at one.
    """
    dates = _dates(65)
    episodes = _episodes(dates)
    rejections = 0
    replications = 120
    for replicate in range(replications):
        generator = np.random.default_rng(1000 + replicate)
        panel = _panel(dates, {"crop": _clustered_volatility(generator, len(dates))})
        roles = pd.Series({"crop": "mechanism_candidate"})
        results, _, _ = dispersion_shift_inference(
            panel, episodes, roles, spec=SPEC, program=PROGRAM
        )
        if float(results.loc[0, "variance_shift_p_value"]) < SPEC.fdr_alpha:
            rejections += 1
    assert rejections / replications < 0.15


def test_exceedance_ratio_separates_a_single_outlier_from_a_wider_distribution() -> None:
    generator = np.random.default_rng(5)
    dates = _dates(65)
    episodes = _episodes(dates)
    mask = build_event_window_mask(
        dates,
        episodes["onset_date"],
        first_relative_month=SPEC.window_first_relative_month,
        last_relative_month=SPEC.window_last_relative_month,
    )
    spike = generator.normal(0, 0.05, len(dates))
    spike[np.flatnonzero(mask)[0]] = 5.0
    panel = _panel(dates, {"crop": spike})
    roles = pd.Series({"crop": "mechanism_candidate"})
    results, _, _ = dispersion_shift_inference(panel, episodes, roles, spec=SPEC, program=PROGRAM)
    row = results.set_index("commodity").loc["crop"]
    # One month lifts the variance ratio while leaving the exceedance frequency
    # flat, which is precisely the disagreement the secondary statistic exists
    # to expose.
    assert row["log_variance_ratio"] > 0
    assert not bool(row["statistics_agree"])


def test_rejecting_negative_controls_voids_the_stage() -> None:
    generator = np.random.default_rng(3)
    dates = _dates(65)
    episodes = _episodes(dates)
    mask = build_event_window_mask(
        dates,
        episodes["onset_date"],
        first_relative_month=SPEC.window_first_relative_month,
        last_relative_month=SPEC.window_last_relative_month,
    )
    base = generator.normal(0, 0.05, len(dates))
    panel = _panel(
        dates,
        {
            "crop": base * np.where(mask, 3.0, 1.0),
            "gold": generator.normal(0, 0.05, len(dates)) * np.where(mask, 3.0, 1.0),
        },
    )
    roles = pd.Series({"crop": "mechanism_candidate", "gold": "negative_control"})
    _, _, summary = dispersion_shift_inference(panel, episodes, roles, spec=SPEC, program=PROGRAM)
    assert summary["negative_controls_rejected"] is True
    assert summary["status"] == "void_negative_controls_rejected"


def test_resolution_floor_is_reported_against_the_program_threshold() -> None:
    generator = np.random.default_rng(7)
    dates = _dates(65)
    episodes = _episodes(dates)
    panel = _panel(dates, {"crop": generator.normal(0, 0.05, len(dates))})
    roles = pd.Series({"crop": "mechanism_candidate"})
    _, _, summary = dispersion_shift_inference(panel, episodes, roles, spec=SPEC, program=PROGRAM)
    assert summary["program_threshold"] == pytest.approx(PROGRAM.program_alpha / 4)
    assert summary["resolution_floor"] == pytest.approx(1 / (summary["shift_count"] + 1))
    # The whole-year shift null cannot reach the program threshold, and the
    # receipt has to say so rather than letting a floor value read as a finding.
    assert summary["resolution_limited"] is True


def test_program_register_totals_only_workstreams_that_produce_results() -> None:
    assert PROGRAM.primary_tests == 4
    assert PROGRAM.threshold == pytest.approx(0.0125)


def test_program_register_rejects_a_workstream_claiming_tests_it_does_not_run(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "findings_v3.yaml").read_text()
    )
    raw["workstreams"]["W4_term_structure"]["primary_tests"] = 1
    path = tmp_path / "findings_v3.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="produces_empirical_result"):
        load_program_register(path)


def test_dispersion_contract_refuses_a_relaxed_falsification(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "dispersion.yaml").read_text()
    )
    raw["falsification"]["void_stage_if_controls_reject"] = False
    path = tmp_path / "dispersion.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="void"):
        load_dispersion_config(path)
