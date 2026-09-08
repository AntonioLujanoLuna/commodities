from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enso_commodities.cold_phase import (
    cold_phase_inference,
    load_cold_phase_config,
    signed_tail_indicators,
)
from enso_commodities.dispersion import build_event_window_mask, load_program_register

SPEC = load_cold_phase_config()
PROGRAM = load_program_register()

COAL = "Coal, Australian"
PALM = "Palm oil"


def _dates(years: int = 65) -> pd.DatetimeIndex:
    return pd.date_range("1960-01-01", periods=years * 12, freq="MS")


# Irregular gaps, as the real index has. Exactly periodic onsets would alias onto
# themselves under a whole-year shift and leave the null with no resolution.
_GAPS_YEARS = (5, 3, 7, 4, 2, 6, 3, 5, 4, 7, 3, 5, 4)


def _cold_episodes(dates: pd.DatetimeIndex) -> pd.DataFrame:
    positions: list[int] = []
    cursor = 6
    for gap in _GAPS_YEARS:
        if cursor >= len(dates) - 24:
            break
        positions.append(cursor)
        cursor += gap * 12
    onsets = [dates[index] for index in positions]
    return pd.DataFrame({"episode_id": [f"c{i}" for i in range(len(onsets))], "onset_date": onsets})


def _panel(dates: pd.DatetimeIndex, series: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({"date": dates, "commodity": name, "seasonal_adjusted_log_return": values})
            for name, values in series.items()
        ],
        ignore_index=True,
    )


def _window(dates: pd.DatetimeIndex, episodes: pd.DataFrame) -> np.ndarray:
    return build_event_window_mask(
        dates,
        episodes["onset_date"],
        first_relative_month=SPEC.window_first_relative_month,
        last_relative_month=SPEC.window_last_relative_month,
    )


def test_registered_directions_read_opposite_tails() -> None:
    values = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])
    upper, _ = signed_tail_indicators(values, np.array([1.0]), tail_quantile=0.75)
    lower, _ = signed_tail_indicators(values, np.array([-1.0]), tail_quantile=0.75)
    assert bool(upper[4, 0]) and not bool(upper[0, 0])
    assert bool(lower[0, 0]) and not bool(lower[4, 0])


def test_contract_freezes_a_reason_for_every_signed_hypothesis() -> None:
    assert SPEC.hypotheses[COAL] == "positive"
    assert SPEC.hypotheses[PALM] == "negative"
    assert all(reason.strip() for reason in SPEC.reasons.values())
    assert "Natural gas, Europe" in SPEC.excluded


def test_planted_cold_phase_disruption_is_detected_in_its_registered_direction() -> None:
    generator = np.random.default_rng(21)
    dates = _dates()
    episodes = _cold_episodes(dates)
    inside = _window(dates, episodes)
    coal = generator.normal(0, 0.05, len(dates))
    # A disruption is a spike, not a level shift: lift the upper tail inside the
    # window rather than the mean everywhere.
    coal[inside] += np.abs(generator.normal(0, 0.18, int(inside.sum())))
    panel = _panel(dates, {COAL: coal, "Gold": generator.normal(0, 0.05, len(dates))})
    roles = pd.Series({COAL: "mechanism_candidate", "Gold": "negative_control"})
    results, null_frame, summary = cold_phase_inference(
        panel, episodes, roles, spec=SPEC, program=PROGRAM
    )
    row = results.set_index("commodity").loc[COAL]
    assert bool(row["in_signed_family"])
    assert row["log_signed_tail_ratio"] > 0
    assert row["shift_p_value"] == pytest.approx(summary["resolution_floor"])
    assert bool(row["statistics_agree"])
    assert summary["negative_controls_rejected"] is False
    assert len(null_frame) == summary["shift_count"] * 2


def test_an_effect_opposite_to_the_registered_sign_does_not_reject() -> None:
    """The point of committing to a direction in advance.

    Palm oil is registered negative -- La Nina is favourable to yields. A cold
    phase that instead raised its price is evidence against the registered
    hypothesis, and a one-sided test must not reward it.
    """
    generator = np.random.default_rng(22)
    dates = _dates()
    episodes = _cold_episodes(dates)
    inside = _window(dates, episodes)
    palm = generator.normal(0, 0.05, len(dates))
    palm[inside] += np.abs(generator.normal(0, 0.18, int(inside.sum())))
    panel = _panel(dates, {PALM: palm})
    roles = pd.Series({PALM: "mechanism_candidate"})
    results, _, _ = cold_phase_inference(panel, episodes, roles, spec=SPEC, program=PROGRAM)
    row = results.set_index("commodity").loc[PALM]
    assert row["registered_direction"] == "negative"
    assert row["log_signed_tail_ratio"] < 0
    assert row["shift_p_value"] > 0.5
    assert not bool(row["reject_fdr"])


def test_controls_are_judged_in_whichever_tail_favours_rejection() -> None:
    generator = np.random.default_rng(23)
    dates = _dates()
    episodes = _cold_episodes(dates)
    inside = _window(dates, episodes)
    # The control's excursion is in the lower tail. A control tested only in the
    # upper tail would pass; the worst-case rule catches it.
    gold = generator.normal(0, 0.05, len(dates))
    gold[inside] -= np.abs(generator.normal(0, 0.18, int(inside.sum())))
    panel = _panel(dates, {COAL: generator.normal(0, 0.05, len(dates)), "Gold": gold})
    roles = pd.Series({COAL: "mechanism_candidate", "Gold": "negative_control"})
    results, _, summary = cold_phase_inference(panel, episodes, roles, spec=SPEC, program=PROGRAM)
    control = results.set_index("commodity").loc["Gold"]
    assert control["shift_p_value"] > 0.5
    assert control["worst_case_shift_p_value"] < SPEC.fdr_alpha
    assert summary["negative_controls_rejected"] is True
    assert summary["status"] == "void_negative_controls_rejected"


def test_unregistered_commodities_stay_out_of_the_signed_family() -> None:
    generator = np.random.default_rng(24)
    dates = _dates()
    episodes = _cold_episodes(dates)
    panel = _panel(
        dates,
        {
            COAL: generator.normal(0, 0.05, len(dates)),
            "Natural gas, Europe": generator.normal(0, 0.05, len(dates)),
        },
    )
    roles = pd.Series({COAL: "mechanism_candidate", "Natural gas, Europe": "mechanism_candidate"})
    results, _, summary = cold_phase_inference(panel, episodes, roles, spec=SPEC, program=PROGRAM)
    excluded = results.set_index("commodity").loc["Natural gas, Europe"]
    assert not bool(excluded["in_signed_family"])
    assert excluded["registered_direction"] == "unregistered"
    assert summary["signed_family_size"] == 1


def test_shift_null_keeps_its_size_under_clustered_volatility() -> None:
    dates = _dates()
    episodes = _cold_episodes(dates)
    rejections = 0
    replications = 100
    for replicate in range(replications):
        generator = np.random.default_rng(4000 + replicate)
        log_variance = np.zeros(len(dates))
        for position in range(1, len(dates)):
            log_variance[position] = 0.97 * log_variance[position - 1] + generator.normal(0, 0.3)
        values = generator.normal(0, 1, len(dates)) * np.exp(log_variance / 2)
        panel = _panel(dates, {COAL: values})
        results, _, _ = cold_phase_inference(
            panel, episodes, pd.Series({COAL: "mechanism_candidate"}), spec=SPEC, program=PROGRAM
        )
        if float(results.loc[0, "shift_p_value"]) < SPEC.fdr_alpha:
            rejections += 1
    assert rejections / replications < 0.15


def test_contract_rejects_a_hypothesis_without_a_reason(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "cold_phase.yaml").read_text()
    )
    raw["hypotheses"][0]["reason"] = "  "
    path = tmp_path / "cold_phase.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stated reason"):
        load_cold_phase_config(path)


def test_contract_rejects_a_commodity_that_is_both_signed_and_excluded(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "cold_phase.yaml").read_text()
    )
    raw["excluded"].append({"commodity": COAL, "reason": "contradictory entry"})
    path = tmp_path / "cold_phase.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="both hypothesised and excluded"):
        load_cold_phase_config(path)


def test_worst_case_control_rule_keeps_its_nominal_size() -> None:
    """The correction added after the rule was found to reject at roughly 2 alpha.

    A control is looked at in both tails, so the smaller of the two one-sided
    p-values is the minimum of two tests. Left uncorrected it voids the stage
    about twice as often as intended, which would make the falsification useless
    by firing on noise.
    """
    dates = _dates()
    episodes = _cold_episodes(dates)
    rejections = 0
    replications = 200
    for replicate in range(replications):
        generator = np.random.default_rng(7000 + replicate)
        panel = _panel(dates, {"Gold": generator.normal(0, 0.05, len(dates))})
        results, _, _ = cold_phase_inference(
            panel, episodes, pd.Series({"Gold": "negative_control"}), spec=SPEC, program=PROGRAM
        )
        if float(results.loc[0, "worst_case_shift_p_value"]) < SPEC.fdr_alpha:
            rejections += 1
    assert rejections / replications < 0.12
