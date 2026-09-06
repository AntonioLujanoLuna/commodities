from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from enso_commodities.config import project_root
from enso_commodities.power import (
    bootstrap_critical_value,
    bootstrap_power,
    evaluate_power,
    minimum_detectable_effect,
)
from enso_commodities.power_analysis import run_power_analysis
from enso_commodities.provenance import sha256_file, write_json_atomic
from enso_commodities.statistics import (
    benjamini_hochberg,
    bootstrap_episode_means,
    salted_seed,
    studentized_null_matrix,
)
from enso_commodities.synthetic import synthetic_endpoint_panel

RESEARCH_CONFIG = project_root() / "config" / "research.yaml"


def _standard_null(size: int = 20000, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(size)


def test_power_at_a_zero_effect_is_the_test_size() -> None:
    nulls = _standard_null()
    critical = bootstrap_critical_value(nulls, alpha=0.05)
    power = bootstrap_power(nulls, standard_error=0.05, effect=0.0, critical_value=critical)
    assert power == pytest.approx(0.05, abs=0.005)


def test_power_increases_with_the_size_of_the_shift() -> None:
    nulls = _standard_null()
    critical = bootstrap_critical_value(nulls, alpha=0.05)
    powers = [
        bootstrap_power(nulls, standard_error=0.05, effect=effect, critical_value=critical)
        for effect in (0.0, 0.05, 0.10, 0.25)
    ]
    assert powers == sorted(powers)
    # Five standard errors of shift against a two-sided 5% threshold.
    assert powers[-1] > 0.99


def test_minimum_detectable_effect_delivers_the_target_power() -> None:
    nulls = _standard_null()
    effect, status = minimum_detectable_effect(
        nulls, standard_error=0.05, alpha=0.05, target_power=0.8
    )
    assert status == "estimated"
    critical = bootstrap_critical_value(nulls, alpha=0.05)
    assert bootstrap_power(
        nulls, standard_error=0.05, effect=effect, critical_value=critical
    ) == pytest.approx(0.8, abs=0.01)
    # The textbook normal-theory value for this configuration.
    assert effect == pytest.approx(0.05 * (1.96 + 0.8416), rel=0.02)


def test_a_stricter_threshold_needs_a_larger_effect() -> None:
    nulls = _standard_null()
    marginal, _ = minimum_detectable_effect(
        nulls, standard_error=0.05, alpha=0.05, target_power=0.8
    )
    family, _ = minimum_detectable_effect(
        nulls, standard_error=0.05, alpha=0.05 / 30, target_power=0.8
    )
    assert family > marginal


def test_a_noisier_series_needs_a_larger_effect() -> None:
    nulls = _standard_null()
    quiet, _ = minimum_detectable_effect(nulls, standard_error=0.05, alpha=0.05, target_power=0.8)
    noisy, _ = minimum_detectable_effect(nulls, standard_error=0.20, alpha=0.05, target_power=0.8)
    assert noisy == pytest.approx(4 * quiet, rel=0.02)


def test_an_unreachable_effect_is_reported_rather_than_extrapolated() -> None:
    nulls = _standard_null()
    effect, status = minimum_detectable_effect(
        nulls, standard_error=0.05, alpha=0.05, target_power=0.8, maximum_effect=0.01
    )
    assert status == "unreachable_within_maximum_effect"
    assert np.isnan(effect)


def test_degenerate_inputs_report_a_status_instead_of_a_number() -> None:
    nulls = _standard_null()
    _, no_scale = minimum_detectable_effect(nulls, standard_error=0.0, alpha=0.05, target_power=0.8)
    _, no_replicates = minimum_detectable_effect(
        np.array([np.nan, np.nan]), standard_error=0.05, alpha=0.05, target_power=0.8
    )
    assert no_scale == "degenerate_standard_error"
    assert no_replicates == "no_valid_replicates"


def test_evaluate_power_marks_estimates_the_design_could_not_have_seen() -> None:
    results = pd.DataFrame(
        {
            "commodity": ["QUIET", "LOUD"],
            "episodes": [17, 17],
            "mean_return": [0.02, 0.60],
            "standard_error": [0.05, 0.05],
        }
    )
    nulls = pd.DataFrame({"QUIET": _standard_null(5000), "LOUD": _standard_null(5000, seed=12)})
    output = evaluate_power(
        results,
        nulls,
        alpha=0.05,
        family_size=30,
        target_power=0.8,
        reference_effects=(0.10, 0.30),
    )
    effects = output.minimum_detectable_effects.set_index("commodity")
    # A 2% estimate is far below what a 5% standard error can resolve; a 60%
    # one is not. Only the first is uninformative rather than null.
    assert bool(effects.loc["QUIET", "observed_effect_below_marginal_mde"])
    assert not bool(effects.loc["LOUD", "observed_effect_below_marginal_mde"])
    assert set(output.power_curves["effect"]) == {0.10, 0.30}
    assert (output.power_curves["power_marginal"] >= output.power_curves["power_family"]).all()


def test_a_commodity_outside_the_replicate_matrix_reports_no_replicates() -> None:
    results = pd.DataFrame(
        {
            "commodity": ["MISSING"],
            "episodes": [17],
            "mean_return": [0.02],
            "standard_error": [0.05],
        }
    )
    output = evaluate_power(
        results,
        pd.DataFrame({"OTHER": _standard_null(100)}),
        alpha=0.05,
        family_size=1,
        target_power=0.8,
        reference_effects=(0.10,),
    )
    row = output.minimum_detectable_effects.iloc[0]
    assert row["minimum_detectable_effect_marginal_status"] == "no_valid_replicates"
    assert row["valid_null_replicates"] == 0


def _single_commodity_panel(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": [f"e{position:02d}" for position in range(len(values))],
            "commodity": "SYN",
            "value": values,
        }
    )


def test_the_reported_minimum_detectable_effect_is_the_one_the_gate_delivers() -> None:
    """Plant the estimated minimum detectable effect and count the rejections.

    This is the claim the module rests on: an effect of the reported size,
    added to a sample of the same size and dispersion, is rejected by the
    studentized bootstrap about `target_power` of the time. It is checked
    against the same bootstrap the study runs rather than a normal
    approximation of it.

    The minimum detectable effect is derived from one observed standard error,
    which is itself noisy at seventeen episodes, so the check averages over
    several reference samples. Calibration under heavy tails is a separate
    question and is measured by the complete-null size test.
    """
    generator = np.random.default_rng(20260906)
    dispersion = 0.12
    episodes = 17
    rejections = 0
    trials = 0
    for reference in range(8):
        baseline = bootstrap_episode_means(
            _single_commodity_panel(generator.normal(0.0, dispersion, size=episodes)),
            value_column="value",
            replicates=1500,
            confidence_level=0.95,
            seed=salted_seed(1, f"power:reference:{reference}"),
            p_value_method="studentized",
        )
        nulls = studentized_null_matrix(baseline.replicates)
        effect, status = minimum_detectable_effect(
            nulls["SYN"].to_numpy(dtype="float64"),
            standard_error=float(baseline.results.iloc[0]["standard_error"]),
            alpha=0.05,
            target_power=0.8,
        )
        assert status == "estimated"
        for trial in range(25):
            output = bootstrap_episode_means(
                _single_commodity_panel(generator.normal(effect, dispersion, size=episodes)),
                value_column="value",
                replicates=999,
                confidence_level=0.95,
                seed=salted_seed(2, f"power:trial:{reference}:{trial}"),
                p_value_method="studentized",
            )
            rejections += int(output.results.iloc[0]["studentized_p_value"] <= 0.05)
            trials += 1
    assert 0.65 <= rejections / trials <= 0.92


def test_family_size_widens_what_the_frozen_gate_can_see() -> None:
    """The family bound is the level BH demands when one member rejects."""
    panel = synthetic_endpoint_panel(seed=77, commodities=6, episodes=17)
    output = bootstrap_episode_means(
        panel,
        value_column="value",
        replicates=1500,
        confidence_level=0.95,
        seed=salted_seed(3, "power:family"),
        p_value_method="studentized",
    )
    power = evaluate_power(
        output.results,
        studentized_null_matrix(output.replicates),
        alpha=0.05,
        family_size=6,
        target_power=0.8,
        reference_effects=(0.10,),
    )
    effects = power.minimum_detectable_effects
    estimated = effects["minimum_detectable_effect_family_status"].eq("estimated")
    assert estimated.all()
    assert (
        effects["minimum_detectable_effect_family"] > effects["minimum_detectable_effect_marginal"]
    ).all()
    # And the family threshold is what the reported q-values are compared with.
    assert (
        benjamini_hochberg(output.results.set_index("commodity")["studentized_p_value"])
        .notna()
        .all()
    )


def _write_inference_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out the inference receipt and outputs the power stage reads."""
    snapshot = tmp_path / "data" / "processed" / "2026-09-05"
    tables = tmp_path / "tables" / "2026-09-05"
    snapshot.mkdir(parents=True)
    tables.mkdir(parents=True)

    outputs = {}
    for label, prefix, commodities in (
        ("primary", "primary", 6),
        ("control", "negative_control", 3),
    ):
        bootstrap = bootstrap_episode_means(
            synthetic_endpoint_panel(seed=500 + commodities, commodities=commodities, episodes=17),
            value_column="value",
            replicates=999,
            confidence_level=0.95,
            seed=salted_seed(10, label),
            p_value_method="studentized",
        )
        results_name = (
            "primary_inference_results.csv"
            if label == "primary"
            else "negative_control_inference.csv"
        )
        replicates_name = f"{prefix}_bootstrap_replicates.parquet"
        bootstrap.results.to_csv(tables / results_name, index=False)
        bootstrap.replicates.to_parquet(tables / replicates_name, index=False)
        outputs[results_name] = sha256_file(tables / results_name)
        outputs[replicates_name] = sha256_file(tables / replicates_name)

    write_json_atomic(
        tables / "inference_summary.json",
        {
            "data_provenance": "real",
            "output_hashes": outputs,
            "input_hashes": {"research.yaml": sha256_file(RESEARCH_CONFIG)},
        },
    )
    return snapshot, tmp_path / "tables"


def test_power_stage_writes_a_hash_linked_receipt(tmp_path: Path) -> None:
    snapshot, tables_root = _write_inference_fixture(tmp_path)
    output_dir = run_power_analysis(
        snapshot, tables_root=tables_root, research_config_path=RESEARCH_CONFIG
    )

    effects = pd.read_csv(output_dir / "primary_minimum_detectable_effect.csv")
    curves = pd.read_csv(output_dir / "primary_power_curves.csv")
    with (output_dir / "power_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["data_provenance"] == "real"
    assert summary["diagnostics"]["candidate_family_size"] == len(effects)
    assert set(summary["output_hashes"]) == {
        "negative_control_minimum_detectable_effect.csv",
        "negative_control_power_curves.csv",
        "primary_minimum_detectable_effect.csv",
        "primary_power_curves.csv",
    }
    # The family bound is stricter than the marginal one, so it always asks for
    # a larger effect wherever both are estimable.
    both = effects["minimum_detectable_effect_family_status"].eq("estimated") & effects[
        "minimum_detectable_effect_marginal_status"
    ].eq("estimated")
    assert both.any()
    assert (
        effects.loc[both, "minimum_detectable_effect_family"]
        > effects.loc[both, "minimum_detectable_effect_marginal"]
    ).all()
    assert curves["power_marginal"].between(0.0, 1.0).all()
    # Controls sit outside the candidate FDR family, so they carry no penalty.
    controls = pd.read_csv(output_dir / "negative_control_minimum_detectable_effect.csv")
    assert (controls["family_size"] == 1).all()
    assert controls["minimum_detectable_effect_family"].equals(
        controls["minimum_detectable_effect_marginal"]
    )


def test_power_stage_refuses_a_tampered_inference_output(tmp_path: Path) -> None:
    snapshot, tables_root = _write_inference_fixture(tmp_path)
    results_path = tables_root / "2026-09-05" / "primary_inference_results.csv"
    tampered = pd.read_csv(results_path)
    tampered.loc[0, "mean_return"] = 99.0
    tampered.to_csv(results_path, index=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        run_power_analysis(snapshot, tables_root=tables_root, research_config_path=RESEARCH_CONFIG)
