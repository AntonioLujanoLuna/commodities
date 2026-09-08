from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enso_commodities.term_structure import (
    build_curve_slopes,
    curve_slope_response,
    load_term_structure_config,
    monthly_mean_spread,
    term_structure_status,
)

SPEC = load_term_structure_config()


def _futures(
    *,
    months: int = 480,
    backwardation_months: np.ndarray | None = None,
    seed: int = 6,
) -> pd.DataFrame:
    """A contract-level panel with four listed expiries on every session."""
    generator = np.random.default_rng(seed)
    sessions = pd.date_range("1985-01-01", periods=months, freq="MS")
    level = 100 * np.exp(np.cumsum(generator.normal(0, 0.02, months)))
    rows = []
    for index, session in enumerate(sessions):
        tilt = 0.0
        if backwardation_months is not None and backwardation_months[index]:
            tilt = 0.08
        for rank in range(1, 5):
            # Without a tilt the curve is flat in expectation; the tilt lifts the
            # front leg relative to the deferred, which is backwardation.
            slope = tilt * (4 - rank) / 3.0
            rows.append(
                {
                    "date": session,
                    "commodity": "coal",
                    "contract": f"C{rank}",
                    "settlement_price": level[index] * np.exp(slope),
                    "volume": 1000,
                    "open_interest": 5000,
                    "days_to_expiry": 20 + 30 * (rank - 1),
                    "information_date": session,
                }
            )
    return pd.DataFrame(rows)


def _episodes(dates: pd.DatetimeIndex) -> pd.DataFrame:
    gaps = (5, 3, 7, 4, 2, 6, 3, 5, 4, 7)
    positions: list[int] = []
    cursor = 6
    for gap in gaps:
        if cursor >= len(dates) - 24:
            break
        positions.append(cursor)
        cursor += gap * 12
    return pd.DataFrame({"onset_date": [dates[index] for index in positions]})


def test_the_contract_freezes_the_estimand_and_refuses_price_indexes() -> None:
    assert SPEC.front_rank == 1
    assert SPEC.deferred_rank == 4
    assert SPEC.config["requirements"]["price_indexes_are_not_admissible"] is True
    assert SPEC.config["scope"] == "prospective_estimand_licensed_inputs_absent"


def test_without_licensed_inputs_the_stage_reports_no_result() -> None:
    status = term_structure_status(None)
    assert status["inputs_present"] is False
    assert status["empirical_result"] is False
    assert status["primary_tests_contributed"] == 0
    assert term_structure_status(pd.DataFrame())["empirical_result"] is False


def test_present_inputs_require_the_register_to_be_updated_first() -> None:
    status = term_structure_status(_futures())
    assert status["inputs_present"] is True
    assert status["primary_tests_contributed"] == 1
    assert "findings_v3.yaml" in status["reason"]


def test_slopes_come_from_two_legs_of_the_same_session() -> None:
    futures = _futures(months=24)
    slopes = build_curve_slopes(futures, spec=SPEC)
    assert len(slopes) == 24
    assert slopes["log_spread"].abs().max() < 1e-9

    # Drop the deferred leg on one session; that session must produce no slope
    # rather than one built against a different day's settlement.
    dropped = futures.loc[
        ~(futures["date"].eq(futures["date"].iloc[0]) & futures["contract"].eq("C4"))
    ]
    assert len(build_curve_slopes(dropped, spec=SPEC)) == 23


def test_a_backwardated_curve_inside_the_windows_is_detected() -> None:
    futures = _futures()
    sessions = pd.DatetimeIndex(sorted(futures["date"].unique()))
    episodes = _episodes(sessions)
    from enso_commodities.dispersion import build_event_window_mask

    inside = build_event_window_mask(
        sessions,
        episodes["onset_date"],
        first_relative_month=SPEC.window_first_relative_month,
        last_relative_month=SPEC.window_last_relative_month,
    )
    tilted = _futures(backwardation_months=inside)
    spreads = monthly_mean_spread(build_curve_slopes(tilted, spec=SPEC))
    response = curve_slope_response(spreads, episodes, spec=SPEC)
    row = response.set_index("commodity").loc["coal"]
    assert bool(row["eligible"])
    assert row["spread_difference"] > 0
    assert row["shift_p_value"] == pytest.approx(row["resolution_floor"])


def test_a_flat_curve_produces_no_response() -> None:
    futures = _futures()
    sessions = pd.DatetimeIndex(sorted(futures["date"].unique()))
    spreads = monthly_mean_spread(build_curve_slopes(futures, spec=SPEC))
    response = curve_slope_response(spreads, _episodes(sessions), spec=SPEC)
    assert response.loc[0, "shift_p_value"] > 0.05


def test_the_response_refuses_to_run_on_absent_inputs() -> None:
    sessions = pd.date_range("1985-01-01", periods=480, freq="MS")
    with pytest.raises(ValueError, match="inputs are absent"):
        curve_slope_response(
            pd.DataFrame(columns=["commodity", "date", "log_spread"]),
            _episodes(sessions),
            spec=SPEC,
        )


def test_slopes_inherit_the_futures_panel_validation() -> None:
    futures = _futures(months=24)
    futures.loc[0, "information_date"] = futures.loc[0, "date"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="not available on the trading date"):
        build_curve_slopes(futures, spec=SPEC)


def test_contract_rejects_a_deferred_leg_that_is_not_beyond_the_front(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "term_structure.yaml").read_text()
    )
    raw["design"]["deferred_rank"] = 1
    path = tmp_path / "term_structure.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="deferred_rank"):
        load_term_structure_config(path)
