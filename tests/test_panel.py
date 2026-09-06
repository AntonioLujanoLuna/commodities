"""Tests for the exposure-weighted two-way fixed-effects panel."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enso_commodities.config import project_root
from enso_commodities.panel import (
    CONTROL_TERM,
    EXPOSURE_TERM,
    BlockDefinition,
    assign_resampling_blocks,
    attach_resampling_blocks,
    build_exposure_table,
    build_panel,
    fit_exposure_panel,
    load_panel_config,
    panel_regressors,
    permute_candidate_exposure_weights,
    two_way_within_transform,
    year_block_bootstrap,
)
from enso_commodities.panel_analysis import run_panel_analysis
from enso_commodities.provenance import sha256_file, write_json_atomic
from enso_commodities.synthetic import synthetic_exposure_panel
from enso_commodities.universe import load_commodity_registry

TOLERANCE = 1e-10
MAX_ITERATIONS = 500


SHIPPED_CONFIG = project_root() / "config" / "panel.yaml"


def _config_with(tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    with SHIPPED_CONFIG.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    mutate(raw)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "panel.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _panel_from(fixture, *, lag_months: int = 0, weighting: str = "exposure") -> pd.DataFrame:
    return build_panel(
        fixture.monthly,
        fixture.enso,
        fixture.exposure,
        index_name="roni",
        lag_months=lag_months,
        outcome_column="seasonal_adjusted_log_return",
        weighting=weighting,
    )


def test_shipped_exposure_weights_cover_the_frozen_candidate_registry() -> None:
    spec = load_panel_config(SHIPPED_CONFIG)
    exposure = build_exposure_table(load_commodity_registry(), spec)

    assert len(exposure) == 35
    assert exposure.loc[exposure["is_negative_control"], "exposure_weight"].eq(0.0).all()
    assert exposure.loc[~exposure["is_negative_control"], "exposure_weight"].gt(0.0).all()
    assert exposure.loc[~exposure["is_negative_control"], "uniform_weight"].eq(1.0).all()
    assert exposure["exposure_method"].eq("post_outcome_expert_judgment").all()
    assert exposure["exposure_outcome_blind"].eq(False).all()
    assert exposure["exposure_inference_scope"].eq("exploratory_only").all()
    # Identification of the separate control coefficient needs the candidate
    # weights to vary; constant weights collapse to the uniform scheme.
    assert exposure.loc[~exposure["is_negative_control"], "exposure_weight"].nunique() > 1


def test_exposure_table_refuses_a_registry_it_does_not_cover() -> None:
    spec = load_panel_config(SHIPPED_CONFIG)
    registry = load_commodity_registry()
    registry.entries.loc[registry.entries.index[0], "commodity"] = "Unlisted commodity"

    with pytest.raises(ValueError, match="do not exactly cover"):
        build_exposure_table(registry, spec)


def test_config_refuses_to_drop_the_month_fixed_effects(tmp_path: Path) -> None:
    def drop(raw: dict) -> None:
        raw["specification"]["time_fixed_effects"] = False

    with pytest.raises(ValueError, match="month fixed effects"):
        load_panel_config(_config_with(tmp_path, drop))


def test_config_refuses_a_nonzero_control_exposure(tmp_path: Path) -> None:
    def weight_the_controls(raw: dict) -> None:
        raw["negative_control_weight"] = 0.5

    with pytest.raises(ValueError, match="zero exposure"):
        load_panel_config(_config_with(tmp_path, weight_the_controls))


def test_config_refuses_to_relabel_post_outcome_weights_as_outcome_blind(tmp_path: Path) -> None:
    def mislabel(raw: dict) -> None:
        raw["exposure_provenance"]["outcome_blind"] = True

    with pytest.raises(ValueError, match="cannot be labelled outcome-blind"):
        load_panel_config(_config_with(tmp_path, mislabel))


def test_two_way_transform_reproduces_an_explicit_dummy_regression() -> None:
    """Alternating projections must give the coefficients a dummy design gives."""
    generator = np.random.default_rng(3)
    records = []
    for unit in range(6):
        for period in range(9):
            # Deliberately unbalanced: the transform has to cope with holes.
            if (unit + period) % 7 == 0:
                continue
            records.append(
                {
                    "unit": f"u{unit}",
                    "period": period,
                    "x1": float(generator.standard_normal()),
                    "x2": float(generator.standard_normal()),
                    "y": float(generator.standard_normal()),
                }
            )
    frame = pd.DataFrame.from_records(records)

    unit_codes = pd.factorize(frame["unit"], sort=True)[0]
    time_codes = pd.factorize(frame["period"], sort=True)[0]
    demeaned = two_way_within_transform(
        frame.loc[:, ["y", "x1", "x2"]].to_numpy(dtype="float64"),
        unit_codes,
        time_codes,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )
    within = np.linalg.lstsq(demeaned[:, 1:], demeaned[:, 0], rcond=None)[0]

    dummies = np.column_stack(
        [
            frame["x1"].to_numpy(),
            frame["x2"].to_numpy(),
            np.ones(len(frame)),
            pd.get_dummies(frame["unit"], drop_first=True).to_numpy(dtype="float64"),
            pd.get_dummies(frame["period"], drop_first=True).to_numpy(dtype="float64"),
        ]
    )
    explicit = np.linalg.lstsq(dummies, frame["y"].to_numpy(), rcond=None)[0][:2]

    assert within == pytest.approx(explicit, abs=1e-8)


def test_panel_recovers_a_planted_exposure_response_and_leaves_controls_null() -> None:
    fixture = synthetic_exposure_panel(seed=20260905)
    fit = fit_exposure_panel(
        _panel_from(fixture), tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS
    )

    assert fit.units == 35
    assert fit.periods == 780
    assert fit.coefficients[EXPOSURE_TERM] == pytest.approx(
        fixture.planted_exposure_response, abs=0.0015
    )
    assert abs(fit.coefficients[CONTROL_TERM]) < 0.0015


def test_month_fixed_effects_absorb_a_regime_factor_the_controls_load_on() -> None:
    """The reason for the whole design.

    The fixture plants a global factor that scales with the size of the ENSO
    index and that the control series load on three times as heavily as the
    candidates -- the shape of the precious-metal problem in the real study.
    Pooled without month effects, that factor is read as a control-group
    response to ENSO. Month effects have to remove it.
    """
    fixture = synthetic_exposure_panel(seed=11, planted_exposure_response=0.0)
    panel = _panel_from(fixture)

    pooled_design = np.column_stack(
        [
            np.ones(len(panel)),
            panel[EXPOSURE_TERM].to_numpy(),
            panel[CONTROL_TERM].to_numpy(),
        ]
    )
    pooled = np.linalg.lstsq(pooled_design, panel["outcome"].to_numpy(), rcond=None)[0]
    fit = fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)

    assert abs(pooled[2]) > 4 * abs(fit.coefficients[CONTROL_TERM])
    assert abs(fit.coefficients[CONTROL_TERM]) < 0.0015


def test_year_block_bootstrap_is_deterministic_and_brackets_the_truth() -> None:
    fixture = synthetic_exposure_panel(seed=5)
    panel = _panel_from(fixture)
    fit = fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)

    first, first_replicates = year_block_bootstrap(
        panel,
        fit,
        replicates=200,
        confidence_level=0.95,
        seed=99,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )
    second, second_replicates = year_block_bootstrap(
        panel,
        fit,
        replicates=200,
        confidence_level=0.95,
        seed=99,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_replicates, second_replicates)

    indexed = first.set_index("term")
    exposure = indexed.loc[EXPOSURE_TERM]
    assert exposure["ci_lower"] < fixture.planted_exposure_response < exposure["ci_upper"]
    assert exposure["studentized_p_value"] < 0.05
    # The control term is a diagnostic, and it must not fire.
    assert indexed.loc[CONTROL_TERM, "studentized_p_value"] > 0.05


def test_block_standard_error_exceeds_the_naive_one_under_serial_dependence() -> None:
    """Months are not independent draws; pretending otherwise understates risk."""
    fixture = synthetic_exposure_panel(seed=13)
    panel = _panel_from(fixture)
    fit = fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)

    results = year_block_bootstrap(
        panel,
        fit,
        replicates=200,
        confidence_level=0.95,
        seed=4,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )[0].set_index("term")

    control = results.loc[CONTROL_TERM]
    assert control["block_standard_error"] > control["naive_standard_error"]


def test_weight_permutation_detects_the_planted_candidate_mapping() -> None:
    fixture = synthetic_exposure_panel(seed=23)
    panel = _panel_from(fixture)
    fit = fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)

    first = permute_candidate_exposure_weights(
        panel,
        fit,
        replicates=499,
        seed=17,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )
    second = permute_candidate_exposure_weights(
        panel,
        fit,
        replicates=499,
        seed=17,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
    )

    assert first.observed_estimate == pytest.approx(fixture.planted_exposure_response, abs=0.0015)
    assert first.mapping_percentile > 0.95
    assert first.p_value < 0.05
    assert first.valid_replicates == 499
    pd.testing.assert_frame_equal(first.replicates, second.replicates)


def test_lagging_the_index_shifts_the_signal_forward() -> None:
    fixture = synthetic_exposure_panel(seed=2)
    contemporaneous = _panel_from(fixture, lag_months=0)
    lagged = _panel_from(fixture, lag_months=6)

    merged = contemporaneous.merge(lagged, on=["date", "commodity"], suffixes=("_now", "_lagged"))
    reference = merged.loc[merged["commodity"].eq(merged["commodity"].iloc[0])]
    shifted = reference["enso_index_now"].shift(6).dropna()
    assert reference["enso_index_lagged"].iloc[6:].to_numpy() == pytest.approx(shifted.to_numpy())


def test_uniform_weighting_reduces_to_a_candidate_versus_control_contrast() -> None:
    fixture = synthetic_exposure_panel(seed=8)
    panel = _panel_from(fixture, weighting="uniform")

    assert panel.loc[~panel["is_negative_control"], "weight"].unique().tolist() == [1.0]
    assert panel.loc[panel["is_negative_control"], "weight"].unique().tolist() == [0.0]

    # Under uniform weights the control indicator is one minus the candidate
    # indicator, so only the exposure term survives the fixed effects.
    assert panel_regressors("uniform") == (EXPOSURE_TERM,)
    assert panel_regressors("exposure") == (EXPOSURE_TERM, CONTROL_TERM)
    fit = fit_exposure_panel(
        panel,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        regressors=panel_regressors("uniform"),
    )
    assert set(fit.coefficients) == {EXPOSURE_TERM}


def test_collinear_interactions_are_refused_rather_than_silently_fitted() -> None:
    """Equal weights across candidates is the same degeneracy in disguise."""
    fixture = synthetic_exposure_panel(seed=8)
    exposure = fixture.exposure.copy()
    exposure.loc[~exposure["is_negative_control"], "exposure_weight"] = 1.0
    panel = build_panel(
        fixture.monthly,
        fixture.enso,
        exposure,
        index_name="roni",
        lag_months=0,
        outcome_column="seasonal_adjusted_log_return",
        weighting="exposure",
    )

    with pytest.raises(ValueError, match="rank deficient"):
        fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)


def test_an_enso_year_block_keeps_a_warm_peak_whole() -> None:
    """The reason the alternative blocks exist.

    ENSO peaks in November-January, so a calendar-year boundary cuts through
    the middle of an event. A May-April block keeps the peak of the 1997-98
    event inside one resampling unit.
    """
    dates = pd.Series(pd.date_range("1997-05-01", "1998-05-01", freq="MS"))
    enso_year = assign_resampling_blocks(dates, start_month=5, length_years=1)
    calendar = assign_resampling_blocks(dates, start_month=1, length_years=1)
    assert enso_year.iloc[:-1].nunique() == 1
    assert enso_year.iloc[-1] != enso_year.iloc[0]
    # The calendar year splits the same span at the December/January boundary.
    assert calendar.nunique() == 2
    assert calendar[dates.dt.year.eq(1997)].nunique() == 1


def test_longer_blocks_merge_consecutive_ones() -> None:
    dates = pd.Series(pd.date_range("1960-05-01", "1972-04-01", freq="MS"))
    single = assign_resampling_blocks(dates, start_month=5, length_years=1)
    triple = assign_resampling_blocks(dates, start_month=5, length_years=3)
    assert single.nunique() == 12
    # Longer blocks are cut on the same absolute month index, so a span of
    # twelve one-year blocks covers four or five three-year ones depending on
    # where it starts; what matters is that the shorter blocks nest inside.
    assert triple.nunique() in {4, 5}
    assert single.groupby(triple).nunique().le(3).all()


def test_assign_resampling_blocks_refuses_impossible_definitions() -> None:
    dates = pd.Series(pd.date_range("1990-01-01", periods=12, freq="MS"))
    with pytest.raises(ValueError, match="start_month"):
        assign_resampling_blocks(dates, start_month=13, length_years=1)
    with pytest.raises(ValueError, match="length_years"):
        assign_resampling_blocks(dates, start_month=1, length_years=0)


def _dependent_panel(*, seed: int, years: int = 36, span_months: int = 60) -> pd.DataFrame:
    """A panel whose residual dependence spans five years, not one.

    Two-way fixed effects absorb a shock that hits every commodity equally, so
    the shock here has a commodity-specific loading and survives the transform.
    There is no true ENSO response; all that is left for a block bootstrap to
    measure is the dependence in the residual.
    """
    generator = np.random.default_rng(seed)
    dates = pd.date_range("1960-01-01", periods=years * 12, freq="MS")
    shock_block = np.arange(len(dates)) // span_months
    shocks = generator.standard_normal(shock_block.max() + 1)
    index = generator.standard_normal(len(dates)).cumsum() * 0.05
    rows = []
    for position in range(8):
        control = position >= 6
        weight = 0.0 if control else float([0.25, 0.5, 1.0][position % 3])
        loading = float(generator.uniform(0.5, 2.0))
        outcome = loading * shocks[shock_block] + generator.standard_normal(len(dates)) * 0.05
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "commodity": f"C{position:02d}",
                    "outcome": outcome,
                    EXPOSURE_TERM: weight * index,
                    CONTROL_TERM: float(control) * index,
                    "calendar_year": dates.year,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_a_block_shorter_than_the_dependence_understates_the_standard_error() -> None:
    """Why the frozen calendar-year interval is the optimistic one.

    A block bootstrap only preserves dependence that fits inside a block. When
    the residual dependence runs five years and the block runs one, each
    replicate treats five correlated stretches as independent draws and the
    interval comes out too narrow. Moving the boundary alone does not help;
    only a longer block does.
    """
    panel = attach_resampling_blocks(
        _dependent_panel(seed=5),
        (
            BlockDefinition(name="enso_year", start_month=5, length_years=1),
            BlockDefinition(name="enso_three_year", start_month=5, length_years=3),
        ),
    )
    fit = fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)
    errors = {}
    for column in ("calendar_year", "block_enso_year", "block_enso_three_year"):
        results, _ = year_block_bootstrap(
            panel,
            fit,
            replicates=200,
            confidence_level=0.95,
            seed=99,
            tolerance=TOLERANCE,
            max_iterations=MAX_ITERATIONS,
            block_column=column,
        )
        row = results.loc[results["term"].eq(EXPOSURE_TERM)].iloc[0]
        errors[column] = float(row["block_standard_error"])
    assert errors["block_enso_three_year"] > 1.25 * errors["calendar_year"]
    assert errors["block_enso_three_year"] > 1.25 * errors["block_enso_year"]


def test_the_bootstrap_refuses_a_block_column_the_panel_does_not_have() -> None:
    panel = _dependent_panel(seed=6)
    fit = fit_exposure_panel(panel, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)
    with pytest.raises(ValueError, match="resampling-block column"):
        year_block_bootstrap(
            panel,
            fit,
            replicates=10,
            confidence_level=0.95,
            seed=1,
            tolerance=TOLERANCE,
            max_iterations=MAX_ITERATIONS,
            block_column="block_missing",
        )


def test_config_refuses_a_malformed_block_sensitivity(tmp_path: Path) -> None:
    def bad_month(raw: dict) -> None:
        raw["inference"]["block_sensitivity"] = [
            {"name": "bad", "start_month": 0, "length_years": 1}
        ]

    def duplicate(raw: dict) -> None:
        raw["inference"]["block_sensitivity"] = [
            {"name": "same", "start_month": 5, "length_years": 1},
            {"name": "same", "start_month": 5, "length_years": 2},
        ]

    def shadows_primary(raw: dict) -> None:
        raw["inference"]["block_sensitivity"] = [
            {"name": "calendar_year", "start_month": 5, "length_years": 1}
        ]

    with pytest.raises(ValueError, match="start_month"):
        load_panel_config(_config_with(tmp_path / "a", bad_month))
    with pytest.raises(ValueError, match="unique"):
        load_panel_config(_config_with(tmp_path / "b", duplicate))
    with pytest.raises(ValueError, match="primary block name"):
        load_panel_config(_config_with(tmp_path / "c", shadows_primary))


def _write_stage_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Lay out the snapshot, tables and config a stage run expects."""
    fixture = synthetic_exposure_panel(seed=21, candidates=6, controls=2)
    snapshot = tmp_path / "data" / "processed" / "2026-09-05"
    tables = tmp_path / "tables" / "2026-09-05"
    snapshot.mkdir(parents=True)
    tables.mkdir(parents=True)

    enso_path = snapshot / "enso_monthly.csv"
    fixture.enso.to_csv(enso_path, index=False, date_format="%Y-%m-%d")
    write_json_atomic(
        snapshot / "summary.json",
        {"data_provenance": "real", "output_hashes": {"enso_monthly.csv": sha256_file(enso_path)}},
    )
    monthly_path = tables / "commodity_returns_adjusted_monthly.parquet"
    fixture.monthly.to_parquet(monthly_path, index=False)
    write_json_atomic(
        tables / "adjusted_event_summary.json",
        {
            "data_provenance": "real",
            "output_hashes": {
                "commodity_returns_adjusted_monthly.parquet": sha256_file(monthly_path)
            },
        },
    )

    candidates = fixture.exposure.loc[~fixture.exposure["is_negative_control"], "commodity"]
    controls = fixture.exposure.loc[fixture.exposure["is_negative_control"], "commodity"]
    registry_path = tmp_path / "commodities.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "inference": {
                    "primary_anchor": "retrospective",
                    "primary_horizon_months": 12,
                    "primary_return": "market_adjusted_cumulative_return",
                    "alternative": "two_sided",
                    "minimum_valid_episodes": 10,
                    "fdr_family": "mechanism_candidates_only",
                },
                "expected_counts": {
                    "mechanism_candidates": len(candidates),
                    "negative_controls": len(controls),
                    "excluded": 0,
                    "source_total": len(fixture.exposure),
                },
                "mechanism_candidates": [
                    {"name": name, "group": "synthetic", "pathway": "planted"}
                    for name in candidates
                ],
                "negative_controls": [
                    {"name": name, "group": "precious_metals", "rationale": "planted"}
                    for name in controls
                ],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )

    def shrink(raw: dict) -> None:
        raw["specification"]["index_definitions"] = ["roni"]
        raw["specification"]["reported_lag_months"] = [0]
        raw["specification"]["primary_lag_months"] = 0
        raw["specification"]["weighting_schemes"] = ["exposure"]
        raw["specification"]["minimum_observations"] = 100
        raw["inference"]["bootstrap_replicates"] = 999
        raw["inference"]["weight_permutation_replicates"] = 999
        # Candidates must carry different weights or the exposure and control
        # interactions are collinear once the month effects are swept out.
        raw["exposure_weights"] = {
            name: [0.25, 0.5, 1.0][position % 3] for position, name in enumerate(candidates)
        }

    return snapshot, tmp_path / "tables", registry_path, _config_with(tmp_path, shrink)


def test_stage_run_verifies_hashes_and_writes_a_receipt(tmp_path: Path) -> None:
    snapshot, tables_root, registry_path, config_path = _write_stage_fixture(tmp_path)

    output_dir = run_panel_analysis(
        snapshot,
        tables_root=tables_root,
        registry_path=registry_path,
        panel_config_path=config_path,
    )

    with (output_dir / "panel_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["design"]["status"] == "exploratory"
    assert summary["design"]["exposure_outcome_blind"] is False
    assert summary["design"]["exposure_inference_scope"] == "exploratory_only"
    assert summary["results"]["cells"] == 1
    assert summary["results"]["primary_exposure_estimate"] is not None
    assert summary["results"]["primary_exposure_weight_permutation_p_value"] is not None
    for name in summary["output_hashes"]:
        assert (output_dir / name).is_file()

    results = pd.read_csv(output_dir / "panel_specification_results.csv")
    assert set(results["term"]) == {EXPOSURE_TERM, CONTROL_TERM}
    assert results.loc[results["term"].eq(CONTROL_TERM), "bh_q_value"].isna().all()
    permutations = pd.read_parquet(output_dir / "panel_exposure_weight_permutations.parquet")
    assert len(permutations) == 999

    sensitivity = pd.read_csv(output_dir / "panel_block_sensitivity.csv")
    assert sensitivity["is_primary_block"].sum() == len(sensitivity["term"].unique())
    exposure_rows = sensitivity.loc[sensitivity["term"].eq(EXPOSURE_TERM)].set_index("block")
    assert "calendar_year" in exposure_rows.index
    # The point estimate is the frozen fit; only the resampling changes.
    assert exposure_rows["estimate"].nunique() == 1
    # Longer blocks mean fewer of them.
    assert exposure_rows.loc["enso_three_year", "blocks"] < exposure_rows.loc["enso_year", "blocks"]


def test_stage_run_refuses_a_tampered_input(tmp_path: Path) -> None:
    snapshot, tables_root, registry_path, config_path = _write_stage_fixture(tmp_path)
    enso = pd.read_csv(snapshot / "enso_monthly.csv")
    enso.loc[0, "roni"] = 99.0
    enso.to_csv(snapshot / "enso_monthly.csv", index=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        run_panel_analysis(
            snapshot,
            tables_root=tables_root,
            registry_path=registry_path,
            panel_config_path=config_path,
        )
