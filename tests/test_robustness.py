import pandas as pd

from enso_commodities.robustness import (
    summarize_candidate_robustness,
    summarize_control_robustness,
)


def test_candidate_requires_every_specification_and_primary_fragility() -> None:
    rows = []
    for commodity in ["A", "B"]:
        for index_definition in ["roni", "oni"]:
            for anchor_type in ["retrospective", "observable"]:
                passes = not (commodity == "B" and anchor_type == "observable")
                rows.append(
                    {
                        "index_definition": index_definition,
                        "anchor_type": anchor_type,
                        "commodity": commodity,
                        "mean_return": 0.2,
                        "bootstrap_bh_q_value": 0.01 if passes else 0.2,
                        "placebo_bh_q_value": 0.02,
                        "passes_specification": passes,
                        "stable_direction": True,
                    }
                )
    fragility = pd.DataFrame({"commodity": ["A", "B"], "survives_all_deletions": [True, True]})

    result = summarize_candidate_robustness(
        pd.DataFrame(rows), fragility, expected_specifications=4
    ).set_index("commodity")

    assert bool(result.loc["A", "passes_timing_index_robustness"])
    assert not bool(result.loc["B", "passes_timing_index_robustness"])
    assert result.loc["B", "specifications_passed"] == 2


def test_control_specificity_failure_must_hold_in_every_specification() -> None:
    rows = []
    for commodity in ["Gold", "Silver"]:
        for index_definition in ["roni", "oni"]:
            for anchor_type in ["retrospective", "observable"]:
                rejected = not (commodity == "Silver" and index_definition == "oni")
                rows.append(
                    {
                        "index_definition": index_definition,
                        "anchor_type": anchor_type,
                        "commodity": commodity,
                        "reject_bootstrap_raw": rejected,
                        "reject_placebo_raw": rejected,
                        "stable_direction": True,
                    }
                )

    result = summarize_control_robustness(pd.DataFrame(rows), expected_specifications=4).set_index(
        "commodity"
    )

    assert bool(result.loc["Gold", "fails_specificity_everywhere"])
    assert not bool(result.loc["Silver", "fails_specificity_everywhere"])
