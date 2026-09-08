"""W5: Eastern-Pacific versus Central-Pacific episode flavour.

Eastern-Pacific and Central-Pacific ("Modoki") events have materially different
teleconnection patterns over Southeast Asia, Australia and the Americas -- which
is to say, over the production regions of most of this registry. Pooling them
averages two treatments, and that is a plausible reason the phase-specificity
contrast comes out null.

The classification is cheap and the split is expensive. Seventeen episodes
become two groups of roughly eight, and the minimum detectable effect on a
difference of two small means is larger than on either mean alone. This module
is therefore written so the power statement comes first: ``flavour_contrast``
computes each commodity's minimum detectable contrast before it reports an
estimate, and marks anything below it as underpowered rather than null. The
expected honest output of this stage is a number attached to the sentence "the
flavour split cannot be resolved at this sample size".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import project_root
from .power import minimum_detectable_effect
from .statistics import benjamini_hochberg

EASTERN = "eastern_pacific"
CENTRAL = "central_pacific"


@dataclass(frozen=True)
class FlavourSpec:
    config: dict[str, Any]
    eastern_index: str
    central_index: str
    reference_month: str
    sensitivity_reference_month: str
    minimum_margin: float
    minimum_episodes_per_flavour: int
    outcome: str
    replicates: int
    random_seed: int
    confidence_level: float
    fdr_alpha: float
    target_power: float
    maximum_effect: float


def load_flavour_config(path: Path | None = None) -> FlavourSpec:
    config_path = path or project_root() / "config" / "flavour.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["provenance"] != {
        "version": 1,
        "authored_on": date(2026, 9, 8),
        "inference_scope": "prespecified_underpowered_diagnostic",
        "program_register": "config/findings_v3.yaml",
        "workstream": "W5_flavour",
    }:
        raise ValueError("Flavour provenance is not the frozen W5 contract")
    classification = raw["classification"]
    if classification["standardization"] != "z_score_over_shared_window":
        raise ValueError("Flavour regions must be standardized before comparison")
    if classification["rule"] != "larger_standardized_peak_region":
        raise ValueError("Unsupported flavour classification rule")
    contrast = raw["contrast"]
    if contrast["estimand"] != "eastern_mean_minus_central_mean":
        raise ValueError("Unsupported flavour estimand")
    if contrast["resampling"] != "stratified_shared_draw_episode_bootstrap":
        raise ValueError("Flavour inference requires the stratified shared-draw bootstrap")
    if contrast["fdr_family"] != "mechanism_candidates_only":
        raise ValueError("Flavour FDR family must contain mechanism candidates only")
    power = raw["power"]
    if not power["require_mde_before_estimate"]:
        raise ValueError("W5 must report its minimum detectable effect with every estimate")
    replicates = int(contrast["replicates"])
    if replicates < 999:
        raise ValueError("Flavour bootstrap requires at least 999 replicates")
    return FlavourSpec(
        config=raw,
        eastern_index=str(classification["eastern_index"]),
        central_index=str(classification["central_index"]),
        reference_month=str(classification["reference_month"]),
        sensitivity_reference_month=str(classification["sensitivity_reference_month"]),
        minimum_margin=float(classification["minimum_margin"]),
        minimum_episodes_per_flavour=int(classification["minimum_episodes_per_flavour"]),
        outcome=str(contrast["outcome"]),
        replicates=replicates,
        random_seed=int(contrast["random_seed"]),
        confidence_level=float(contrast["confidence_level"]),
        fdr_alpha=float(contrast["fdr_alpha"]),
        target_power=float(power["target_power"]),
        maximum_effect=float(power["maximum_effect"]),
    )


def standardize_regions(
    indices: pd.DataFrame, *, eastern_index: str, central_index: str
) -> pd.DataFrame:
    """Put both region indices on a common scale over their shared history.

    Niño 4 varies materially less than Niño 3, so comparing raw magnitudes would
    label nearly every episode Eastern Pacific by construction. The z-scores use
    only months where both regions are observed, so the comparison is not moved
    by one series covering a longer period than the other.
    """
    required = {"date", eastern_index, central_index}
    missing = required - set(indices.columns)
    if missing:
        raise ValueError(f"Region indices are missing columns: {sorted(missing)}")
    frame = indices.loc[:, ["date", eastern_index, central_index]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", ignore_index=True)
    if frame["date"].duplicated().any():
        raise ValueError("Region index dates must be unique")
    shared = frame[eastern_index].notna() & frame[central_index].notna()
    if int(shared.sum()) < 240:
        raise ValueError("Region indices share fewer than twenty years of months")
    for column in (eastern_index, central_index):
        values = frame.loc[shared, column]
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{column} has no variation over the shared window")
        frame[f"{column}_z"] = (frame[column] - float(values.mean())) / scale
    frame["shared_window"] = shared
    return frame


def classify_episode_flavour(
    episodes: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    spec: FlavourSpec,
    reference_month: str | None = None,
) -> pd.DataFrame:
    """Label each warm episode Eastern or Central Pacific.

    The rule compares the two standardized region anomalies at the episode's
    reference month and takes the larger. ``minimum_margin`` allows an
    unclassified band; at its frozen value of zero every episode is labelled,
    and the margin is reported so a reader can see how many labels rest on a
    hair's difference.
    """
    required = {"episode_id", "peak_date", "onset_date", "end_date"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"Episodes are missing columns: {sorted(missing)}")
    scope = reference_month or spec.reference_month
    standardized = standardize_regions(
        indices, eastern_index=spec.eastern_index, central_index=spec.central_index
    )
    eastern_column = f"{spec.eastern_index}_z"
    central_column = f"{spec.central_index}_z"
    lookup = standardized.set_index("date")

    records: list[dict[str, Any]] = []
    for _, episode in episodes.iterrows():
        if scope == "episode_peak_month":
            window = lookup.reindex([pd.Timestamp(episode["peak_date"])])
        elif scope == "episode_mean":
            window = lookup.loc[
                pd.Timestamp(episode["onset_date"]) : pd.Timestamp(episode["end_date"])
            ]
        else:
            raise ValueError(f"Unsupported reference month scope: {scope}")
        eastern = float(window[eastern_column].mean())
        central = float(window[central_column].mean())
        margin = eastern - central
        if not np.isfinite(margin):
            flavour = "unclassified_missing_index"
        elif abs(margin) < spec.minimum_margin:
            flavour = "unclassified_within_margin"
        else:
            flavour = EASTERN if margin > 0 else CENTRAL
        records.append(
            {
                "episode_id": episode["episode_id"],
                "reference_month_scope": scope,
                "eastern_standardized": eastern,
                "central_standardized": central,
                "flavour_margin": margin,
                "flavour": flavour,
            }
        )
    return pd.DataFrame.from_records(records)


def _contrast_moments(
    values: np.ndarray, eastern: np.ndarray, central: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Difference of flavour means and its two-sample standard error, per commodity."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    indicator = finite.astype("float64")

    def group(selector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = selector @ indicator
        totals = selector @ filled
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, totals / counts, np.nan)
        centred = np.where(finite, (values - means) ** 2, 0.0)
        squares = selector @ centred
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = np.where(counts > 1, squares / (counts - 1.0), np.nan)
        return counts, means, variance

    eastern_counts, eastern_means, eastern_variance = group(eastern)
    central_counts, central_means, central_variance = group(central)
    with np.errstate(invalid="ignore", divide="ignore"):
        contrast = eastern_means - central_means
        error = np.sqrt(
            np.where(eastern_counts > 0, eastern_variance / eastern_counts, np.nan)
            + np.where(central_counts > 0, central_variance / central_counts, np.nan)
        )
    return contrast, error, eastern_counts, central_counts


def flavour_contrast(
    outcomes: pd.DataFrame,
    flavours: pd.DataFrame,
    *,
    spec: FlavourSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eastern-minus-Central contrast per commodity, power statement first."""
    required = {"episode_id", "commodity", "role", spec.outcome}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Flavour outcomes are missing columns: {sorted(missing)}")
    labelled = outcomes.merge(
        flavours.loc[:, ["episode_id", "flavour"]], on="episode_id", validate="many_to_one"
    )
    labelled = labelled.loc[labelled["flavour"].isin([EASTERN, CENTRAL])]
    if labelled.empty:
        raise ValueError("No episodes carry an Eastern or Central label")

    episodes = sorted(labelled["episode_id"].unique())
    metadata = (
        labelled.loc[:, ["commodity", "role"]]
        .drop_duplicates()
        .sort_values("commodity", ignore_index=True)
    )
    if metadata["commodity"].duplicated().any():
        raise ValueError("Each commodity must have exactly one role")
    matrix = labelled.pivot(index="episode_id", columns="commodity", values=spec.outcome)
    matrix = matrix.reindex(index=episodes, columns=metadata["commodity"])
    values = matrix.to_numpy(dtype="float64")

    episode_flavour = (
        flavours.set_index("episode_id").loc[episodes, "flavour"].to_numpy(dtype=object)
    )
    eastern_positions = np.flatnonzero(episode_flavour == EASTERN)
    central_positions = np.flatnonzero(episode_flavour == CENTRAL)
    eastern_count = len(eastern_positions)
    central_count = len(central_positions)
    resolvable = (
        eastern_count >= spec.minimum_episodes_per_flavour
        and central_count >= spec.minimum_episodes_per_flavour
    )

    eastern_selector = np.zeros((1, len(episodes)))
    eastern_selector[0, eastern_positions] = 1.0
    central_selector = np.zeros((1, len(episodes)))
    central_selector[0, central_positions] = 1.0
    contrast_rows, error_rows, eastern_valid, central_valid = _contrast_moments(
        values, eastern_selector, central_selector
    )
    observed_contrast = contrast_rows[0]
    observed_error = error_rows[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_t = np.where(observed_error > 0, observed_contrast / observed_error, np.nan)

    generator = np.random.default_rng(spec.random_seed)
    null_t = np.full((spec.replicates, len(metadata)), np.nan)
    for replicate in range(spec.replicates):
        eastern_draw = generator.choice(eastern_positions, size=eastern_count, replace=True)
        central_draw = generator.choice(central_positions, size=central_count, replace=True)
        eastern_row = np.zeros((1, len(episodes)))
        central_row = np.zeros((1, len(episodes)))
        np.add.at(eastern_row[0], eastern_draw, 1.0)
        np.add.at(central_row[0], central_draw, 1.0)
        replicate_contrast, replicate_error, _, _ = _contrast_moments(
            values, eastern_row, central_row
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            null_t[replicate] = np.where(
                replicate_error[0] > 0,
                (replicate_contrast[0] - observed_contrast) / replicate_error[0],
                np.nan,
            )

    rows: list[dict[str, Any]] = []
    for position, commodity in enumerate(metadata["commodity"]):
        nulls = null_t[:, position]
        effect, status = minimum_detectable_effect(
            nulls,
            standard_error=float(observed_error[position]),
            alpha=spec.fdr_alpha,
            target_power=spec.target_power,
            maximum_effect=spec.maximum_effect,
        )
        finite_nulls = nulls[np.isfinite(nulls)]
        p_value = (
            float(
                (1 + int(np.sum(np.abs(finite_nulls) >= abs(observed_t[position]))))
                / (1 + finite_nulls.size)
            )
            if finite_nulls.size and np.isfinite(observed_t[position])
            else float("nan")
        )
        below_mde = bool(
            np.isfinite(effect) and abs(float(observed_contrast[position])) < float(effect)
        )
        rows.append(
            {
                "commodity": commodity,
                "role": metadata.loc[position, "role"],
                "eastern_episodes": int(eastern_valid[0, position]),
                "central_episodes": int(central_valid[0, position]),
                # The power statement is deliberately ahead of the estimate in
                # the column order, because the contract requires it be read first.
                "minimum_detectable_contrast": effect,
                "minimum_detectable_contrast_status": status,
                "eastern_minus_central": float(observed_contrast[position]),
                "contrast_standard_error": float(observed_error[position]),
                "contrast_t_statistic": float(observed_t[position]),
                "bootstrap_p_value": p_value,
                "observed_contrast_below_mde": below_mde,
            }
        )

    results = pd.DataFrame.from_records(rows)
    candidates = results["role"].eq("mechanism_candidate")
    results["bh_q_value"] = np.nan
    if candidates.any():
        results.loc[candidates, "bh_q_value"] = benjamini_hochberg(
            results.loc[candidates].set_index("commodity")["bootstrap_p_value"]
        ).to_numpy()
    results["reject_fdr"] = results["bh_q_value"].le(spec.fdr_alpha).fillna(False)
    results["interpretation"] = np.where(
        results["reject_fdr"],
        "flavour_contrast_detected",
        np.where(
            results["observed_contrast_below_mde"],
            spec.config["power"]["underpowered_label"],
            "inconclusive",
        ),
    )

    candidate_t = results.loc[candidates, "contrast_t_statistic"].abs()
    family_statistic = float(candidate_t.max()) if not candidate_t.empty else float("nan")
    family_p = float("nan")
    if np.isfinite(family_statistic) and candidates.any():
        candidate_positions = np.flatnonzero(candidates.to_numpy())
        with np.errstate(invalid="ignore"):
            null_maxima = np.nanmax(np.abs(null_t[:, candidate_positions]), axis=1)
        finite_maxima = null_maxima[np.isfinite(null_maxima)]
        if finite_maxima.size:
            family_p = float(
                (1 + int(np.sum(finite_maxima >= family_statistic))) / (1 + finite_maxima.size)
            )

    summary: dict[str, Any] = {
        "eastern_episodes": eastern_count,
        "central_episodes": central_count,
        "minimum_episodes_per_flavour": spec.minimum_episodes_per_flavour,
        "split_resolvable": resolvable,
        "family_size": int(candidates.sum()),
        "family_maximum_absolute_contrast_t": family_statistic,
        "family_bootstrap_p_value": family_p,
        "candidate_fdr_rejections": int(results.loc[candidates, "reject_fdr"].sum()),
        "candidates_below_their_own_mde": int(
            results.loc[candidates, "observed_contrast_below_mde"].sum()
        ),
        "median_minimum_detectable_contrast": float(
            results.loc[candidates, "minimum_detectable_contrast"].median()
        )
        if candidates.any()
        else float("nan"),
    }
    return results, summary
