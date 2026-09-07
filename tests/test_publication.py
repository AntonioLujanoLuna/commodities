from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from enso_commodities.provenance import sha256_file, write_json_atomic
from enso_commodities.publication import build_scorecard, verify_publication_bundle
from enso_commodities.reporting import analysis_source_tree_hash


def _write_scorecard_inputs(root: Path) -> None:
    commodities = ["A", "B"]
    pd.DataFrame(
        {
            "commodity": commodities,
            "group": ["crop", "crop"],
            "mean_return": [0.2, 0.01],
            "studentized_ci_lower": [0.1, -0.2],
            "studentized_ci_upper": [0.3, 0.2],
            "reject_fdr": [True, False],
            "reject_fwer": [True, False],
        }
    ).to_csv(root / "primary_inference_results.csv", index=False)
    pd.DataFrame({"commodity": commodities, "passes_current_gates": [True, False]}).to_csv(
        root / "primary_placebo_results.csv", index=False
    )
    pd.DataFrame(
        {
            "commodity": commodities,
            "minimum_detectable_effect_marginal": [0.1, 0.3],
            "observed_effect_below_marginal_mde": [False, True],
        }
    ).to_csv(root / "primary_minimum_detectable_effect.csv", index=False)
    pd.DataFrame({"commodity": commodities, "passes_macro_gates": [True, False]}).to_csv(
        root / "macro_primary_results.csv", index=False
    )
    pd.DataFrame({"commodity": commodities, "survives_all_deletions": [True, False]}).to_csv(
        root / "macro_fragility_summary.csv", index=False
    )
    pd.DataFrame(
        {"commodity": commodities, "passes_timing_index_robustness": [True, False]}
    ).to_csv(root / "robustness_candidate_summary.csv", index=False)
    pd.DataFrame(
        {
            "commodity": commodities,
            "role": ["mechanism_candidate", "mechanism_candidate"],
            "reject_fdr": [False, False],
        }
    ).to_csv(root / "dose_response_results.csv", index=False)
    pd.DataFrame(
        {
            "commodity": ["A", "A", "B", "B"],
            "reject_direction_contrast_fdr": [False, True, False, False],
        }
    ).to_csv(root / "specificity_candidate_results.csv", index=False)


def test_scorecard_exposes_robust_and_underpowered_states(tmp_path: Path) -> None:
    _write_scorecard_inputs(tmp_path)
    scorecard = build_scorecard(tmp_path).set_index("commodity")
    assert scorecard.loc["A", "evidence_status"] == "robust_historical_association"
    assert scorecard.loc["B", "evidence_status"] == "underpowered"
    assert bool(scorecard.loc["A", "any_warm_cold_contrast_fdr"])
    assert set(scorecard["out_of_sample_status"]) == {"not_tested"}


def test_bundle_verifier_detects_a_published_file_change(tmp_path: Path) -> None:
    _write_scorecard_inputs(tmp_path)
    scorecard_path = tmp_path / "scorecard.csv"
    build_scorecard(tmp_path).to_csv(scorecard_path, index=False, lineterminator="\n")
    published = {path.name: sha256_file(path) for path in tmp_path.glob("*.csv")}
    write_json_atomic(
        tmp_path / "manifest.json",
        {
            "analysis_source_tree_sha256": analysis_source_tree_hash(),
            "data_provenance": "real",
            "published_files": published,
            "snapshot": tmp_path.name,
            "source_receipts": {},
        },
    )
    verify_publication_bundle(tmp_path)
    scorecard_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_publication_bundle(tmp_path)
