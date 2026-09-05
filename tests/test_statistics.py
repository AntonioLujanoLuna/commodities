import math

import pandas as pd
import pytest

from enso_commodities.statistics import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bootstrap_episode_means,
    salted_seed,
    westfall_young_step_down,
)


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, 0.002], index=list("abcd"))

    adjusted = benjamini_hochberg(p_values)

    assert adjusted.to_dict() == pytest.approx({"a": 0.02, "b": 0.04, "c": 0.04, "d": 0.008})


def test_bootstrap_is_deterministic_and_shares_episode_draws() -> None:
    rows = []
    for episode, value in enumerate([0.02, 0.04, 0.06, 0.08, 0.10], start=1):
        rows.append({"episode_id": f"e{episode}", "commodity": "A", "value": value})
        rows.append({"episode_id": f"e{episode}", "commodity": "B", "value": 2 * value})
    data = pd.DataFrame(rows)

    first = bootstrap_episode_means(
        data,
        value_column="value",
        replicates=99,
        confidence_level=0.95,
        seed=salted_seed(123, "test"),
    )
    second = bootstrap_episode_means(
        data,
        value_column="value",
        replicates=99,
        confidence_level=0.95,
        seed=salted_seed(123, "test"),
    )

    pd.testing.assert_frame_equal(first.results, second.results)
    pd.testing.assert_frame_equal(first.replicates, second.replicates)
    pivot = first.replicates.pivot(index="replicate", columns="commodity", values="bootstrap_mean")
    assert (pivot["B"] - 2 * pivot["A"]).abs().max() < 1e-12
    assert first.results["sign_agreement"].all()


def test_centered_null_test_detects_constant_nonzero_effect() -> None:
    data = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(10)],
            "commodity": ["Cocoa"] * 10,
            "value": [0.1] * 10,
        }
    )

    output = bootstrap_episode_means(
        data,
        value_column="value",
        replicates=999,
        confidence_level=0.95,
        seed=42,
    )

    result = output.results.iloc[0]
    assert result["bootstrap_p_value"] == pytest.approx(0.001)
    assert result["ci_lower"] == pytest.approx(0.1)
    assert result["ci_upper"] == pytest.approx(0.1)


def test_studentized_test_reports_scale_and_falls_back_when_degenerate() -> None:
    scaled = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(12)],
            "commodity": ["Cocoa"] * 12,
            "value": [0.02, 0.05, -0.01, 0.09, 0.03, 0.04, -0.02, 0.07, 0.01, 0.06, 0.03, 0.02],
        }
    )
    output = bootstrap_episode_means(
        scaled,
        value_column="value",
        replicates=999,
        confidence_level=0.95,
        seed=salted_seed(11, "studentized"),
        p_value_method="studentized",
    )
    row = output.results.iloc[0]
    expected_error = scaled["value"].std(ddof=1) / math.sqrt(len(scaled))
    assert row["standard_error"] == pytest.approx(expected_error)
    assert row["t_statistic"] == pytest.approx(scaled["value"].mean() / expected_error)
    assert row["studentized_status"] == "estimated"
    assert row["bootstrap_p_value_method"] == "studentized"
    assert row["bootstrap_p_value"] == row["studentized_p_value"]
    assert row["centered_p_value"] != row["studentized_p_value"]
    assert row["studentized_ci_lower"] < row["mean_return"] < row["studentized_ci_upper"]
    assert output.replicates["studentized_statistic"].notna().mean() > 0.9

    # A commodity with no observed dispersion has no studentized statistic at
    # all; it must fall back rather than leaving a hole in the family.
    degenerate = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(10)],
            "commodity": ["Gold"] * 10,
            "value": [0.1] * 10,
        }
    )
    fallback = bootstrap_episode_means(
        degenerate,
        value_column="value",
        replicates=999,
        confidence_level=0.95,
        seed=salted_seed(11, "degenerate"),
        p_value_method="studentized",
    ).results.iloc[0]
    assert fallback["studentized_status"] == "degenerate_observed_scale"
    assert fallback["bootstrap_p_value_method"] == "studentized_fallback_centered"
    assert fallback["bootstrap_p_value"] == pytest.approx(0.001)


def test_benjamini_yekutieli_is_benjamini_hochberg_scaled_by_the_harmonic_number() -> None:
    p_values = pd.Series([0.001, 0.01, 0.02, 0.5], index=list("abcd"))

    bh = benjamini_hochberg(p_values)
    by = benjamini_yekutieli(p_values)

    harmonic = 1 + 1 / 2 + 1 / 3 + 1 / 4
    assert by["a"] == pytest.approx(min(bh["a"] * harmonic, 1.0))
    assert (by >= bh).all()


def test_westfall_young_uses_the_joint_null_of_the_largest_statistic() -> None:
    observed = pd.Series({"A": 3.0, "B": 1.0})
    nulls = pd.DataFrame({"A": [0.5, 2.0, 0.1, 4.0], "B": [0.2, 3.5, 0.3, 0.4]})

    adjusted = westfall_young_step_down(observed, nulls).set_index("commodity")

    # Step 1 compares |t_A| = 3 against max(|A|, |B|) per replicate -> 2 of 4.
    assert adjusted.loc["A", "westfall_young_p_value"] == pytest.approx(3 / 5)
    # Step 2 compares |t_B| = 1 against |B| alone -> 1 of 4, then is forced up
    # to the preceding step's value so the sequence stays monotone.
    assert adjusted.loc["B", "westfall_young_p_value"] == pytest.approx(3 / 5)
    assert adjusted.loc["A", "step_down_rank"] == 1
    assert adjusted.loc["B", "step_down_rank"] == 2
    assert set(adjusted["westfall_young_status"]) == {"estimated"}


def test_westfall_young_refuses_a_family_with_too_few_complete_replicates() -> None:
    observed = pd.Series({"A": 3.0, "B": 1.0})
    nulls = pd.DataFrame({"A": [0.5, float("nan"), 0.1, 4.0], "B": [0.2, 3.5, 0.3, 0.4]})

    adjusted = westfall_young_step_down(observed, nulls).set_index("commodity")

    assert set(adjusted["westfall_young_status"]) == {"insufficient_valid_replicates"}
    assert adjusted["westfall_young_p_value"].isna().all()
