import pandas as pd
import pytest

from enso_commodities.statistics import (
    benjamini_hochberg,
    bootstrap_episode_means,
    salted_seed,
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
