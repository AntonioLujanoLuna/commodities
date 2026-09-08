from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from enso_commodities.flavour import (
    CENTRAL,
    EASTERN,
    classify_episode_flavour,
    flavour_contrast,
    load_flavour_config,
    standardize_regions,
)
from enso_commodities.power import minimum_detectable_effect

SPEC = dataclasses.replace(load_flavour_config(), replicates=2000)


def _indices(months: int = 780, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    dates = pd.date_range("1960-01-01", periods=months, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            # Niño 4 is deliberately the less variable region, as it is in nature.
            "nino3": generator.normal(0, 1.0, months),
            "nino4": generator.normal(0, 0.4, months),
        }
    )


def _episodes(dates: pd.DatetimeIndex) -> pd.DataFrame:
    onsets = [dates[index] for index in range(6, len(dates) - 30, 48)]
    return pd.DataFrame(
        {
            "episode_id": [f"e{i}" for i in range(len(onsets))],
            "onset_date": onsets,
            "peak_date": [onset + pd.DateOffset(months=6) for onset in onsets],
            "end_date": [onset + pd.DateOffset(months=11) for onset in onsets],
        }
    )


def test_standardization_removes_the_scale_advantage_of_the_wider_region() -> None:
    standardized = standardize_regions(_indices(), eastern_index="nino3", central_index="nino4")
    shared = standardized.loc[standardized["shared_window"]]
    assert shared["nino3_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert shared["nino4_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_standardization_refuses_a_short_shared_window() -> None:
    with pytest.raises(ValueError, match="twenty years"):
        standardize_regions(_indices(months=120), eastern_index="nino3", central_index="nino4")


def test_classification_follows_the_larger_standardized_region() -> None:
    indices = _indices()
    episodes = _episodes(pd.DatetimeIndex(indices["date"]))
    # Force the first episode's peak month to be unambiguously Eastern Pacific
    # and the second unambiguously Central Pacific.
    indices.loc[indices["date"].eq(episodes.loc[0, "peak_date"]), ["nino3", "nino4"]] = [3.0, 0.0]
    indices.loc[indices["date"].eq(episodes.loc[1, "peak_date"]), ["nino3", "nino4"]] = [0.0, 1.2]
    labels = classify_episode_flavour(episodes, indices, spec=SPEC).set_index("episode_id")
    assert labels.loc["e0", "flavour"] == EASTERN
    assert labels.loc["e1", "flavour"] == CENTRAL
    assert labels.loc["e0", "flavour_margin"] > labels.loc["e1", "flavour_margin"]


def test_classification_supports_the_episode_mean_sensitivity() -> None:
    indices = _indices()
    episodes = _episodes(pd.DatetimeIndex(indices["date"]))
    peak_labels = classify_episode_flavour(episodes, indices, spec=SPEC)
    mean_labels = classify_episode_flavour(
        episodes, indices, spec=SPEC, reference_month="episode_mean"
    )
    assert set(peak_labels["reference_month_scope"]) == {"episode_peak_month"}
    assert set(mean_labels["reference_month_scope"]) == {"episode_mean"}
    assert len(mean_labels) == len(episodes)


def _labelled_outcomes(
    contrast: float, *, noise: float = 0.05, seed: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    episode_ids = [f"e{i}" for i in range(16)]
    flavours = pd.DataFrame(
        {
            "episode_id": episode_ids,
            "flavour": [EASTERN] * 8 + [CENTRAL] * 8,
        }
    )
    rows = []
    for episode, flavour in zip(episode_ids, flavours["flavour"], strict=True):
        for commodity, role in (("crop", "mechanism_candidate"), ("gold", "negative_control")):
            shift = contrast if (flavour == EASTERN and commodity == "crop") else 0.0
            rows.append(
                {
                    "episode_id": episode,
                    "commodity": commodity,
                    "role": role,
                    "seasonal_adjusted_log_return": shift + generator.normal(0, noise),
                }
            )
    return pd.DataFrame(rows), flavours


def test_contrast_recovers_a_large_planted_flavour_difference() -> None:
    outcomes, flavours = _labelled_outcomes(contrast=0.6, noise=0.05)
    results, summary = flavour_contrast(outcomes, flavours, spec=SPEC)
    crop = results.set_index("commodity").loc["crop"]
    assert crop["eastern_minus_central"] == pytest.approx(0.6, abs=0.1)
    assert crop["bootstrap_p_value"] < 0.05
    assert not bool(crop["observed_contrast_below_mde"])
    assert crop["interpretation"] == "flavour_contrast_detected"
    assert summary["split_resolvable"] is True


def test_a_realistic_contrast_is_reported_as_underpowered_not_null() -> None:
    """The result this stage most likely produces, and the reason it exists.

    A contrast of ten percent against commodity-return noise is well inside the
    range the physical literature would consider material, and eight episodes a
    side cannot see it. The stage must say "underpowered", not "no difference".
    """
    outcomes, flavours = _labelled_outcomes(contrast=0.10, noise=0.30)
    results, summary = flavour_contrast(outcomes, flavours, spec=SPEC)
    crop = results.set_index("commodity").loc["crop"]
    assert crop["bootstrap_p_value"] > 0.05
    assert bool(crop["observed_contrast_below_mde"])
    assert crop["interpretation"] == "inconclusive_underpowered_subgroup"
    assert summary["candidates_below_their_own_mde"] == 1


def test_splitting_the_sample_costs_power_against_the_pooled_mean() -> None:
    """Splitting seventeen episodes in two raises the detectable effect.

    Both quantities come from the same machinery, so this compares like with
    like: the minimum detectable contrast between two halves against the minimum
    detectable mean on the pooled sample of the same episodes.
    """
    outcomes, flavours = _labelled_outcomes(contrast=0.0, noise=0.30)
    results, _ = flavour_contrast(outcomes, flavours, spec=SPEC)
    contrast_mde = float(results.set_index("commodity").loc["crop", "minimum_detectable_contrast"])

    pooled = outcomes.loc[outcomes["commodity"].eq("crop"), "seasonal_adjusted_log_return"]
    sample = pooled.to_numpy(dtype="float64")
    generator = np.random.default_rng(9)
    pooled_null = []
    for _ in range(SPEC.replicates):
        draw = generator.choice(sample, size=sample.size, replace=True)
        error = draw.std(ddof=1) / np.sqrt(draw.size)
        pooled_null.append((draw.mean() - sample.mean()) / error if error > 0 else np.nan)
    pooled_mde, status = minimum_detectable_effect(
        np.asarray(pooled_null),
        standard_error=float(sample.std(ddof=1) / np.sqrt(sample.size)),
        alpha=SPEC.fdr_alpha,
        target_power=SPEC.target_power,
        maximum_effect=SPEC.maximum_effect,
    )
    assert status == "estimated"
    assert contrast_mde > pooled_mde


def test_contract_requires_the_power_statement(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "flavour.yaml").read_text()
    )
    raw["power"]["require_mde_before_estimate"] = False
    path = tmp_path / "flavour.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum detectable effect"):
        load_flavour_config(path)
