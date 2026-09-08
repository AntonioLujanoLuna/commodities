from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enso_commodities.forecast_news import (
    align_returns,
    build_revision_series,
    load_news_config,
    news_regression,
    run_news_inference,
    validate_forecast_archive,
    wild_cluster_bootstrap,
)

SPEC = dataclasses.replace(load_news_config(), replicates=999, minimum_observations=60)
OUTCOME = "seasonal_adjusted_log_return"
MAX_LEAD = 9


def _archive(months: int = 264, seed: int = 1) -> pd.DataFrame:
    """A well-formed issuance archive whose probabilities evolve as news arrives.

    Each target season is forecast from lead nine down to lead zero, and the
    forecast for a fixed target is revised by a fresh increment at each issuance.
    """
    generator = np.random.default_rng(seed)
    targets = pd.date_range("2002-10-01", periods=months, freq="MS")
    rows = []
    for target in targets:
        probability = 0.33
        for lead in range(MAX_LEAD, -1, -1):
            probability = float(np.clip(probability + generator.normal(0, 0.06), 0.02, 0.96))
            issue = target - pd.DateOffset(months=lead)
            remainder = 1.0 - probability
            rows.append(
                {
                    "issue_date": issue,
                    "target_center_date": target,
                    "lead_months": lead,
                    "probability_el_nino": probability,
                    "probability_neutral": remainder * 0.6,
                    "probability_la_nina": remainder * 0.4,
                }
            )
    return pd.DataFrame(rows)


def _returns(
    revisions: pd.DataFrame,
    *,
    beta: float,
    response_offset: int,
    commodities: tuple[str, ...] = ("crop", "gold"),
    noise: float = 0.03,
    seed: int = 4,
) -> pd.DataFrame:
    """Monthly returns that respond to the revision ``response_offset`` months away.

    ``response_offset`` zero makes the response contemporaneous with the
    issuance. A value of one makes the price move a month BEFORE the revision is
    published, which is the anticipation the lead placebo has to catch.
    """
    generator = np.random.default_rng(seed)
    grid = pd.date_range("2001-01-01", "2026-12-01", freq="MS")
    effect = (
        revisions.set_index("issue_month")["revision"]
        .reindex(grid + pd.DateOffset(months=response_offset))
        .to_numpy(dtype="float64")
    )
    effect = np.nan_to_num(effect)
    frames = []
    for name in commodities:
        loading = beta if name == "crop" else 0.0
        frames.append(
            pd.DataFrame(
                {
                    "date": grid,
                    "commodity": name,
                    OUTCOME: loading * effect + generator.normal(0, noise, len(grid)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


ROLES = pd.Series({"crop": "mechanism_candidate", "gold": "negative_control"})


def test_archive_validation_rejects_a_target_before_its_issuance() -> None:
    archive = _archive()
    archive.loc[0, "target_center_date"] = archive.loc[0, "issue_date"] - pd.DateOffset(months=1)
    with pytest.raises(ValueError, match="before its own issuance"):
        validate_forecast_archive(archive)


def test_archive_validation_rejects_probabilities_that_do_not_sum_to_one() -> None:
    archive = _archive()
    archive.loc[0, "probability_neutral"] = archive.loc[0, "probability_neutral"] + 0.5
    with pytest.raises(ValueError, match="sum to one"):
        validate_forecast_archive(archive)


def test_archive_validation_accepts_the_declared_one_point_rounding_tolerance() -> None:
    archive = _archive()
    archive.loc[0, "probability_neutral"] += 0.01
    validate_forecast_archive(archive)


def test_archive_validation_rejects_a_lead_that_contradicts_the_dates() -> None:
    archive = _archive()
    archive.loc[0, "lead_months"] = archive.loc[0, "lead_months"] + 3
    with pytest.raises(ValueError, match="lead_months disagrees"):
        validate_forecast_archive(archive)


def test_archive_validation_rejects_duplicate_issuance_target_pairs() -> None:
    archive = _archive()
    with pytest.raises(ValueError, match="duplicate issuance/target pairs"):
        validate_forecast_archive(pd.concat([archive, archive.head(1)], ignore_index=True))


def test_revisions_difference_successive_issuances_of_the_same_target() -> None:
    archive = _archive()
    revisions = build_revision_series(
        archive, lead_months=6, probability_column="probability_el_nino"
    )
    target = revisions.loc[0, "target_center_date"]
    same_target = archive.loc[archive["target_center_date"].eq(target)].set_index("lead_months")
    expected = same_target.loc[6, "probability_el_nino"] - same_target.loc[7, "probability_el_nino"]
    assert revisions.loc[0, "revision"] == pytest.approx(expected)
    assert not revisions["issue_month"].duplicated().any()


def test_a_gap_in_the_archive_produces_no_revision_across_it() -> None:
    archive = _archive()
    target = archive["target_center_date"].iloc[0]
    dropped = archive.loc[
        ~(archive["target_center_date"].eq(target) & archive["lead_months"].eq(7))
    ]
    revisions = build_revision_series(
        dropped, lead_months=6, probability_column="probability_el_nino"
    )
    # The lead-6 issuance for that target now has an eight-month-old predecessor,
    # so it must be dropped rather than differenced across the hole.
    assert not (revisions["target_center_date"].eq(target)).any()


def test_alignment_reads_the_month_the_relative_offset_names() -> None:
    archive = _archive()
    revisions = build_revision_series(
        archive, lead_months=6, probability_column="probability_el_nino"
    )
    returns = _returns(revisions, beta=0.0, response_offset=0)
    contemporaneous = align_returns(revisions, returns, outcome_column=OUTCOME, relative_month=0)
    ahead = align_returns(revisions, returns, outcome_column=OUTCOME, relative_month=-1)
    first = revisions.loc[0, "issue_month"]
    assert contemporaneous["date"].min() == first
    assert ahead["date"].min() == first - pd.DateOffset(months=1)


def test_a_contemporaneous_news_response_is_recovered_and_the_placebos_are_quiet() -> None:
    archive = _archive()
    revisions = build_revision_series(
        archive, lead_months=6, probability_column="probability_el_nino"
    )
    returns = _returns(revisions, beta=0.9, response_offset=0)
    results, primary, summary = run_news_inference(
        revisions, returns, ROLES, spec=SPEC, program_threshold=0.0125
    )
    crop = primary.set_index("commodity").loc["crop"]
    assert crop["coefficient"] == pytest.approx(0.9, abs=0.15)
    assert crop["wild_bootstrap_p_value"] < 0.01
    assert summary["lead_placebo_rejections"] == 0
    assert summary["control_rejections"] == 0
    assert summary["status"] in {"program_level_finding", "exploratory_within_stage_only"}
    assert set(results["window"]) == {
        "primary",
        "secondary",
        "lag_placebo_2",
        "lag_placebo_3",
        "lead_placebo_-1",
    }


def test_a_price_that_moves_before_the_revision_voids_the_stage() -> None:
    """The falsification that makes the contemporaneous coefficient readable.

    Here the return responds to next month's revision, so the revision series is
    picking up something already in prices. The contemporaneous estimate would
    look fine on its own; the lead placebo is what catches it.
    """
    archive = _archive()
    revisions = build_revision_series(
        archive, lead_months=6, probability_column="probability_el_nino"
    )
    returns = _returns(revisions, beta=0.9, response_offset=1)
    _, _, summary = run_news_inference(
        revisions, returns, ROLES, spec=SPEC, program_threshold=0.0125
    )
    assert summary["lead_placebo_rejections"] > 0
    assert summary["status"] == "void_lead_placebo_rejected"


def test_no_response_to_anything_is_reported_as_no_response() -> None:
    archive = _archive()
    revisions = build_revision_series(
        archive, lead_months=6, probability_column="probability_el_nino"
    )
    returns = _returns(revisions, beta=0.0, response_offset=0)
    _, primary, summary = run_news_inference(
        revisions, returns, ROLES, spec=SPEC, program_threshold=0.0125
    )
    assert summary["candidate_fdr_rejections"] == 0
    assert summary["lead_placebo_rejections"] == 0
    assert summary["status"] == "exploratory_within_stage_only"
    assert primary.set_index("commodity").loc["crop", "wild_bootstrap_p_value"] > 0.05


def test_wild_bootstrap_holds_its_size_under_the_null() -> None:
    archive = _archive()
    revisions = build_revision_series(
        archive, lead_months=6, probability_column="probability_el_nino"
    )
    rejections = 0
    replications = 60
    for replicate in range(replications):
        returns = _returns(revisions, beta=0.0, response_offset=0, seed=500 + replicate)
        panel = align_returns(revisions, returns, outcome_column=OUTCOME, relative_month=0)
        estimates = news_regression(
            panel, outcome_column=OUTCOME, minimum_observations=SPEC.minimum_observations
        )
        null = wild_cluster_bootstrap(
            panel,
            outcome_column=OUTCOME,
            commodities=["crop", "gold"],
            replicates=399,
            seed=SPEC.random_seed + replicate,
            minimum_observations=SPEC.minimum_observations,
        )
        statistic = float(estimates.set_index("commodity").loc["crop", "t_statistic"])
        column = null[:, 0]
        finite = column[np.isfinite(column)]
        p_value = (1 + int(np.sum(np.abs(finite) >= abs(statistic)))) / (1 + finite.size)
        if p_value < 0.05:
            rejections += 1
    assert rejections / replications < 0.15


def test_contract_keeps_the_lead_placebo_and_the_imposed_null(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config" / "forecast_news.yaml"
    for section, key, message in (
        ("placebos", "void_stage_if_lead_placebo_rejects", "lead placebo"),
        ("inference", "impose_null_on_residuals", "impose the null"),
    ):
        raw = yaml.safe_load(source.read_text())
        raw[section][key] = False
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_news_config(path)


def test_the_stage_refuses_to_run_without_a_verified_issuance_archive(tmp_path: Path) -> None:
    """No archive is better than a reconstructed one, and the error has to say why."""
    from enso_commodities.news_analysis import latest_forecast_archive

    with pytest.raises(FileNotFoundError, match="reconstructed or revised"):
        latest_forecast_archive(tmp_path / "absent")
