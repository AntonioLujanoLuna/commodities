"""Recovery and calibration tests against data whose answer is known.

The unit tests elsewhere check that each function does what its own docstring
says. These check the two things that matter about the study as a whole: that a
real effect placed in the data comes back out of the analytical path at roughly
the right size, and that when nothing is there the gates reject at close to
their nominal rate. The second is the direct test of the false-positive rate
that the negative-control gate is failing on.

These drive the analytical functions, not the file-IO stages -- snapshot
loading, hash verification and receipt writing stay covered by their own tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enso_commodities.adjustments import add_adjusted_event_paths, adjust_monthly_returns
from enso_commodities.enso import construct_warm_episodes
from enso_commodities.returns import build_event_return_paths, compute_monthly_returns
from enso_commodities.statistics import (
    benjamini_hochberg,
    bootstrap_episode_means,
    salted_seed,
    studentized_null_matrix,
    westfall_young_step_down,
)
from enso_commodities.synthetic import SyntheticMarket, synthetic_endpoint_panel, synthetic_market

HORIZON_MONTHS = 12
BASE_RELATIVE_MONTH = -1
EVENT_WINDOW = (-12, 24)


def _run_analytical_path(market: SyntheticMarket) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    episodes = construct_warm_episodes(
        market.enso,
        index_name="roni",
        threshold=0.5,
        minimum_duration_months=5,
        observable_delay_after_center_months=2,
    )
    monthly = compute_monthly_returns(market.prices)
    paths = build_event_return_paths(
        market.prices,
        episodes,
        first_relative_month=EVENT_WINDOW[0],
        last_relative_month=EVENT_WINDOW[1],
        base_relative_month=BASE_RELATIVE_MONTH,
        anchor_types=("retrospective",),
    )
    adjusted, _, models, _ = adjust_monthly_returns(
        monthly,
        market.market_index,
        episodes,
        market_exclusion_window=EVENT_WINDOW,
        minimum_seasonal_observations=8,
        minimum_market_model_observations=60,
    )
    adjusted_paths = add_adjusted_event_paths(
        paths, adjusted, base_relative_month=BASE_RELATIVE_MONTH
    )
    endpoint = adjusted_paths.loc[adjusted_paths["relative_month"].eq(HORIZON_MONTHS)]
    return endpoint, models, len(episodes)


def _endpoint_inference(endpoint: pd.DataFrame, label: str) -> pd.DataFrame:
    output = bootstrap_episode_means(
        endpoint,
        value_column="market_adjusted_cumulative_return",
        replicates=999,
        confidence_level=0.95,
        seed=salted_seed(20260905, label),
        p_value_method="studentized",
    )
    results = output.results
    results["bh_q_value"] = benjamini_hochberg(
        results.set_index("commodity")["bootstrap_p_value"]
    ).to_numpy()
    return results


def test_pipeline_recovers_a_planted_effect_and_leaves_other_series_flat() -> None:
    market = synthetic_market(seed=20260905, planted_horizon_effect=0.25, idiosyncratic_scale=0.02)
    endpoint, models, episode_count = _run_analytical_path(market)
    assert episode_count == 17

    # The market model has to find the loadings it was given, or the residual
    # this study calls an abnormal return is not one.
    fitted = models.merge(market.truth, on="commodity", validate="one_to_one")
    assert (fitted["market_beta"].astype(float) - fitted["market_loading"]).abs().max() < 0.1
    assert fitted["market_model_status"].eq("estimated").all()

    results = _endpoint_inference(endpoint, "recovery").merge(
        market.truth, on="commodity", validate="one_to_one"
    )
    affected = results.loc[results["is_affected"]]
    unaffected = results.loc[~results["is_affected"]]

    assert (affected["mean_return"] - affected["planted_simple_effect"]).abs().max() < 0.06
    assert affected["bh_q_value"].le(0.05).all()
    assert unaffected["mean_return"].abs().max() < 0.06
    assert not unaffected["bh_q_value"].le(0.05).any()


def test_recovered_effect_is_unbiased_across_independent_synthetic_markets() -> None:
    """A single draw can land high or low; the average must not."""
    errors: list[float] = []
    for offset in range(4):
        market = synthetic_market(seed=20260905 + offset)
        endpoint, _, _ = _run_analytical_path(market)
        results = _endpoint_inference(endpoint, f"bias:{offset}").merge(
            market.truth, on="commodity", validate="one_to_one"
        )
        affected = results.loc[results["is_affected"]]
        errors.extend((affected["mean_return"] - affected["planted_simple_effect"]).tolist())
    assert abs(float(np.mean(errors))) < 0.025


def _complete_null_rejection_rates(trials: int) -> dict[str, float]:
    centered_any = studentized_any = fwer_any = 0
    for trial in range(trials):
        panel = synthetic_endpoint_panel(seed=90000 + trial)
        output = bootstrap_episode_means(
            panel,
            value_column="value",
            replicates=999,
            confidence_level=0.95,
            seed=salted_seed(20260905, f"size:{trial}"),
            p_value_method="studentized",
        )
        indexed = output.results.set_index("commodity")
        centered_any += int(benjamini_hochberg(indexed["centered_p_value"]).le(0.05).any())
        studentized_any += int(benjamini_hochberg(indexed["studentized_p_value"]).le(0.05).any())
        westfall_young = westfall_young_step_down(
            indexed["t_statistic"], studentized_null_matrix(output.replicates)
        )
        fwer_any += int(westfall_young["westfall_young_p_value"].le(0.05).any())
    return {
        "centered": centered_any / trials,
        "studentized": studentized_any / trials,
        "westfall_young": fwer_any / trials,
    }


def test_gates_hold_their_nominal_size_under_the_complete_null() -> None:
    """Under the complete null, any rejection is a false one.

    Every commodity loads on a shared episode-level factor here, so false
    rejections arrive in clusters rather than singly -- which is exactly the
    case the centred percentile test mishandles.
    """
    rates = _complete_null_rejection_rates(trials=100)

    assert rates["studentized"] <= 0.06
    assert rates["westfall_young"] <= 0.06
    # The reason the studentized test is the configured default: the centred
    # one is materially anti-conservative at this many episodes.
    assert rates["centered"] >= 2 * rates["studentized"]


def test_the_studentized_gate_still_detects_a_real_common_effect() -> None:
    """Correct size is worthless if it costs all the power."""
    detected = 0
    trials = 40
    for trial in range(trials):
        panel = synthetic_endpoint_panel(seed=70000 + trial, effect=0.08)
        output = bootstrap_episode_means(
            panel,
            value_column="value",
            replicates=999,
            confidence_level=0.95,
            seed=salted_seed(20260905, f"power:{trial}"),
            p_value_method="studentized",
        )
        indexed = output.results.set_index("commodity")
        detected += int(benjamini_hochberg(indexed["studentized_p_value"]).le(0.05).any())
    assert detected / trials >= 0.5


def test_synthetic_fixtures_are_never_labelled_real() -> None:
    assert synthetic_market(seed=1).data_provenance == "synthetic"
    panel = synthetic_endpoint_panel(seed=1, commodities=2, episodes=5)
    assert set(panel.columns) == {"episode_id", "commodity", "value"}
    assert panel["value"].notna().all()


@pytest.mark.parametrize("effect", [0.0, 0.2])
def test_planted_effect_is_exact_in_log_space(effect: float) -> None:
    market = synthetic_market(seed=7, planted_horizon_effect=effect)
    assert market.truth["planted_log_effect"].max() == pytest.approx(effect)
