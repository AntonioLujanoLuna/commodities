import pandas as pd

from enso_commodities.fragility import (
    summarize_candidate_fragility,
    summarize_control_fragility,
)


def test_candidate_fragility_requires_every_deletion_to_pass() -> None:
    scenarios = pd.DataFrame(
        {
            "commodity": ["A", "A", "B", "B"],
            "deleted_episode_id": ["e1", "e2", "e1", "e2"],
            "mean_return": [0.2, 0.1, -0.1, 0.1],
            "sign_agreement": [True, True, True, False],
            "stable_direction": [True, True, True, False],
            "bootstrap_bh_q_value": [0.01, 0.04, 0.02, 0.4],
            "reject_bootstrap_fdr": [True, True, True, False],
        }
    )
    full = pd.DataFrame(
        {
            "commodity": ["A", "B"],
            "macro_mean_return": [0.15, -0.05],
            "passes_macro_gates": [True, True],
        }
    )

    result = summarize_candidate_fragility(scenarios, full).set_index("commodity")

    assert bool(result.loc["A", "survives_all_deletions"])
    assert not bool(result.loc["B", "survives_all_deletions"])
    assert result.loc["A", "loo_worst_bootstrap_q_value"] == 0.04


def test_control_fragility_reports_persistent_false_positive() -> None:
    scenarios = pd.DataFrame(
        {
            "commodity": ["Gold", "Gold", "Silver", "Silver"],
            "deleted_episode_id": ["e1", "e2", "e1", "e2"],
            "mean_return": [0.2, 0.3, 0.1, -0.1],
            "bootstrap_p_value": [0.01, 0.02, 0.01, 0.2],
            "stable_direction": [True, True, True, False],
        }
    )
    full = pd.DataFrame({"commodity": ["Gold", "Silver"], "macro_mean_return": [0.25, 0.05]})

    result = summarize_control_fragility(scenarios, full).set_index("commodity")

    assert bool(result.loc["Gold", "rejects_after_every_deletion"])
    assert not bool(result.loc["Silver", "rejects_after_every_deletion"])
