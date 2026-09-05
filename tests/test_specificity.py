import pandas as pd

from enso_commodities.specificity import add_candidate_fdr, randomization_direction_contrast


def test_direction_contrast_is_deterministic_and_reports_observed_difference() -> None:
    rows = []
    for direction, values in {"warm": [3.0, 4.0], "cold": [-1.0, -2.0]}.items():
        for index, value in enumerate(values):
            rows.append(
                {"episode_id": f"{direction}{index}", "direction": direction, "commodity": "A", "value": value}
            )
    data = pd.DataFrame(rows)
    first = randomization_direction_contrast(data, value_column="value", replicates=100, seed=7)
    second = randomization_direction_contrast(data, value_column="value", replicates=100, seed=7)
    pd.testing.assert_frame_equal(first.results, second.results)
    pd.testing.assert_frame_equal(first.replicates, second.replicates)
    assert first.results.loc[0, "warm_minus_cold"] == 5.0
    assert bool(first.results.loc[0, "opposite_mean_directions"])


def test_candidate_fdr_is_confined_to_passed_table() -> None:
    result = add_candidate_fdr(
        pd.DataFrame({"commodity": ["A", "B"], "contrast_p_value": [0.01, 0.8]}),
        alpha=0.05,
    )
    assert bool(result.loc[0, "reject_direction_contrast_fdr"])
    assert not bool(result.loc[1, "reject_direction_contrast_fdr"])
