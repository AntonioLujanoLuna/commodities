from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enso_commodities.dose_response import (
    amplitude_permutation_inference,
    hc3_slope,
    load_dose_response_config,
    prepare_dose_panel,
)


def test_hc3_slope_recovers_a_linear_dose_response() -> None:
    x = np.linspace(-1, 1, 20)
    slope, standard_error, statistic = hc3_slope(x, 0.4 * x + 0.01 * x**2)
    assert slope == pytest.approx(0.4, abs=1e-12)
    assert standard_error > 0
    assert statistic > 10


def test_prepare_panel_standardizes_one_episode_mapping() -> None:
    episodes = pd.DataFrame(
        {
            "episode_id": ["e1", "e2", "e3"],
            "index_name": ["roni"] * 3,
            "direction": ["warm"] * 3,
            "peak_value": [0.5, 1.0, 1.5],
        }
    )
    outcomes = pd.DataFrame(
        {
            "episode_id": ["e1", "e2", "e3"],
            "commodity": ["crop"] * 3,
            "group": ["grain"] * 3,
            "role": ["mechanism_candidate"] * 3,
            "return": [0.0, 0.1, 0.2],
        }
    )
    panel = prepare_dose_panel(outcomes, episodes, outcome_column="return")
    assert panel["peak_amplitude_z"].mean() == pytest.approx(0.0, abs=1e-12)
    assert panel["peak_amplitude_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_shared_permutation_detects_planted_amplitude_and_is_deterministic() -> None:
    generator = np.random.default_rng(4)
    episodes = [f"e{i}" for i in range(18)]
    amplitude = np.linspace(-1.5, 1.5, len(episodes))
    rows = []
    for commodity, role, effect in (
        ("crop", "mechanism_candidate", 0.5),
        ("other", "mechanism_candidate", 0.0),
        ("gold", "negative_control", 0.0),
    ):
        for episode, x in zip(episodes, amplitude, strict=True):
            rows.append(
                {
                    "episode_id": episode,
                    "commodity": commodity,
                    "group": "test",
                    "role": role,
                    "peak_amplitude_z": x,
                    "return": effect * x + generator.normal(0, 0.08),
                }
            )
    panel = pd.DataFrame(rows)
    kwargs = {
        "outcome_column": "return",
        "minimum_valid_episodes": 10,
        "replicates": 499,
        "seed": 8,
        "fdr_alpha": 0.05,
        "fwer_alpha": 0.05,
    }
    first, first_replicates = amplitude_permutation_inference(panel, **kwargs)
    second, second_replicates = amplitude_permutation_inference(panel, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_replicates, second_replicates)
    planted = first.set_index("commodity").loc["crop"]
    assert planted["slope_per_peak_roni_sd"] > 0.4
    assert planted["reject_fdr"]
    assert planted["reject_fwer"]


def test_shipped_contract_is_secondary_and_uses_whole_episode_permutations() -> None:
    spec = load_dose_response_config()
    assert spec.config["provenance"]["inference_scope"] == "exploratory_secondary"
    assert spec.config["inference"]["permutation_unit"] == "whole_episode_amplitude_labels"
