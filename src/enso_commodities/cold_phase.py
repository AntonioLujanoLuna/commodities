"""W2: cold-phase disruption endpoint with signed, prespecified hypotheses.

`config/commodities.yaml` records the Australian coal pathway as "rainfall
affects mining and export logistics". That is a La Nina mechanism -- flooded
pits, washed-out rail -- and it raises prices. The frozen design tests it as a
warm-phase mean return at plus twelve months, and reports -0.228. The registry
wrote down a mechanism and then measured something else.

This module tests what the registry wrote down. Three things change, all fixed
in the contract before the run:

* Cold episodes rather than warm ones.
* A direction per commodity, tested one-sided. Committing to a sign in a hashed
  file in advance buys real evidential weight for nothing; a commodity whose
  sign cannot be defended is excluded rather than handed a two-sided test.
* An endpoint that can see a disruption. A flood does not move an annual mean,
  it produces a spike, so the endpoint is the frequency of months in the tail
  the hypothesis points at.

The reference distribution is the one W1 uses, for the same reason: circular
whole-year shifts of the episode mask leave every return series untouched, so
volatility clustering and seasonality cannot manufacture a result.
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
from .dispersion import (
    MONTHS_PER_YEAR,
    ProgramRegister,
    available_year_shifts,
    build_event_window_mask,
    build_monthly_matrix,
)
from .statistics import benjamini_hochberg

POSITIVE = "positive"
NEGATIVE = "negative"


@dataclass(frozen=True)
class ColdPhaseSpec:
    config: dict[str, Any]
    index: str
    cold_threshold: float
    minimum_duration_months: int
    observable_delay_after_center_months: int
    outcome: str
    window_first_relative_month: int
    window_last_relative_month: int
    tail_quantile: float
    minimum_inside_months: int
    minimum_outside_months: int
    minimum_shift_years: int
    maximum_shift_years: int
    fdr_alpha: float
    hypotheses: dict[str, str]
    reasons: dict[str, str]
    excluded: dict[str, str]


def load_cold_phase_config(path: Path | None = None) -> ColdPhaseSpec:
    config_path = path or project_root() / "config" / "cold_phase.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["provenance"] != {
        "version": 1,
        "authored_on": date(2026, 9, 8),
        "inference_scope": "prespecified_secondary_endpoint",
        "program_register": "config/findings_v3.yaml",
        "workstream": "W2_cold_phase",
    }:
        raise ValueError("Cold-phase provenance is not the frozen W2 contract")
    design = raw["design"]
    if design["primary_statistic"] != "log_signed_tail_ratio":
        raise ValueError("W2 primary statistic is the log signed-tail ratio")
    if design["tail_threshold_scope"] != "unconditional_signed_return":
        raise ValueError("Tail thresholds must be unconditional, so the null cannot move them")
    if design["alternative"] != "greater":
        raise ValueError("W2 tests one-sided against its registered direction")
    inference = raw["inference"]
    if inference["reference_distribution"] != "circular_whole_year_shift":
        raise ValueError("W2 requires the circular whole-year shift null")
    falsification = raw["falsification"]
    if falsification["control_direction"] != "worst_case_of_both_tails":
        raise ValueError("W2 controls must be evaluated in their worst-case direction")
    if falsification["control_tail_correction"] != "bonferroni_two_tails":
        raise ValueError("W2 control p-values must be corrected across the two tails")
    if not falsification["void_stage_if_controls_reject"]:
        raise ValueError("W2 must void the stage when its controls reject")
    episodes = raw["episodes"]
    threshold = float(episodes["cold_threshold"])
    if threshold >= 0:
        raise ValueError("cold_threshold must be negative")

    hypotheses: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for entry in raw["hypotheses"]:
        commodity = str(entry["commodity"])
        direction = str(entry["direction"])
        if direction not in {POSITIVE, NEGATIVE}:
            raise ValueError(f"{commodity}: direction must be 'positive' or 'negative'")
        if not str(entry.get("reason", "")).strip():
            raise ValueError(f"{commodity}: a signed hypothesis requires a stated reason")
        if commodity in hypotheses:
            raise ValueError(f"{commodity}: duplicated signed hypothesis")
        hypotheses[commodity] = direction
        reasons[commodity] = str(entry["reason"])
    if not hypotheses:
        raise ValueError("W2 requires at least one signed hypothesis")

    excluded: dict[str, str] = {}
    for entry in raw.get("excluded", []):
        commodity = str(entry["commodity"])
        if not str(entry.get("reason", "")).strip():
            raise ValueError(f"{commodity}: an exclusion requires a stated reason")
        if commodity in hypotheses:
            raise ValueError(f"{commodity}: cannot be both hypothesised and excluded")
        excluded[commodity] = str(entry["reason"])

    return ColdPhaseSpec(
        config=raw,
        index=str(episodes["index"]),
        cold_threshold=threshold,
        minimum_duration_months=int(episodes["minimum_duration_months"]),
        observable_delay_after_center_months=int(episodes["observable_delay_after_center_months"]),
        outcome=str(design["outcome"]),
        window_first_relative_month=int(design["window_first_relative_month"]),
        window_last_relative_month=int(design["window_last_relative_month"]),
        tail_quantile=float(design["tail_quantile"]),
        minimum_inside_months=int(design["minimum_inside_months"]),
        minimum_outside_months=int(design["minimum_outside_months"]),
        minimum_shift_years=int(inference["minimum_shift_years"]),
        maximum_shift_years=int(inference["maximum_shift_years"]),
        fdr_alpha=float(inference["fdr_alpha"]),
        hypotheses=hypotheses,
        reasons=reasons,
        excluded=excluded,
    )


def signed_tail_indicators(
    values: np.ndarray, directions: np.ndarray, *, tail_quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Flag months in the tail each commodity's registered direction points at.

    Thresholds come from the unconditional distribution, so a circular shift of
    the episode mask cannot move them. A positive hypothesis reads the upper
    tail and a negative one the lower tail, which is what makes the subsequent
    one-sided test a test of the hypothesis rather than of its absolute size.
    """
    if not 0.5 < tail_quantile < 1:
        raise ValueError("tail_quantile must fall in (0.5, 1)")
    finite = np.isfinite(values)
    masked = np.where(finite, values, np.nan)
    with np.errstate(invalid="ignore"):
        upper = np.nanquantile(masked, tail_quantile, axis=0)
        lower = np.nanquantile(masked, 1 - tail_quantile, axis=0)
    thresholds = np.where(directions > 0, upper, lower)
    indicators = np.where(
        directions > 0,
        finite & (masked > thresholds),
        finite & (masked < thresholds),
    )
    return indicators, thresholds


def _signed_ratios(
    masks: np.ndarray,
    values: np.ndarray,
    finite: np.ndarray,
    indicators: np.ndarray,
    directions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Tail-frequency ratios and signed mean differences for every mask at once."""
    indicator = finite.astype("float64")
    filled = np.where(finite, values, 0.0)
    tails = np.where(finite, indicators, False).astype("float64")

    inside = masks.astype("float64")
    outside = 1.0 - inside

    inside_counts = inside @ indicator
    outside_counts = outside @ indicator
    inside_tails = inside @ tails
    outside_tails = outside @ tails
    inside_totals = inside @ filled
    outside_totals = outside @ filled

    with np.errstate(invalid="ignore", divide="ignore"):
        inside_rate = (inside_tails + 0.5) / (inside_counts + 1.0)
        outside_rate = (outside_tails + 0.5) / (outside_counts + 1.0)
        tail_ratio = np.where(
            (inside_counts > 0) & (outside_counts > 0),
            np.log(inside_rate / outside_rate),
            np.nan,
        )
        inside_mean = np.where(inside_counts > 0, inside_totals / inside_counts, np.nan)
        outside_mean = np.where(outside_counts > 0, outside_totals / outside_counts, np.nan)
    # Signed so that a positive number always means "consistent with the
    # registered hypothesis", whichever direction that hypothesis points.
    signed_difference = (inside_mean - outside_mean) * directions
    return {
        "inside_months": inside_counts,
        "outside_months": outside_counts,
        "inside_tail_months": inside_tails,
        "log_signed_tail_ratio": tail_ratio,
        "signed_mean_difference": signed_difference,
    }


def cold_phase_inference(
    returns: pd.DataFrame,
    cold_episodes: pd.DataFrame,
    roles: pd.Series,
    *,
    spec: ColdPhaseSpec,
    program: ProgramRegister,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run W2 and return per-commodity results, the shift null, and a summary."""
    dates, commodities, values = build_monthly_matrix(returns, outcome_column=spec.outcome)
    finite = np.isfinite(values)

    # Controls carry no registered direction. They are run through both tails and
    # judged on whichever is more favourable to rejection, which is the hardest
    # version of the falsification for this stage to survive.
    hypothesis_directions = np.array(
        [1.0 if spec.hypotheses.get(name) != NEGATIVE else -1.0 for name in commodities]
    )
    is_hypothesised = np.array([name in spec.hypotheses for name in commodities])
    is_control = np.array([roles.get(name) == "negative_control" for name in commodities])

    observed_mask = build_event_window_mask(
        dates,
        cold_episodes["onset_date"],
        first_relative_month=spec.window_first_relative_month,
        last_relative_month=spec.window_last_relative_month,
    )
    shifts = available_year_shifts(
        len(dates),
        minimum_shift_years=spec.minimum_shift_years,
        maximum_shift_years=spec.maximum_shift_years,
    )
    masks = np.vstack(
        [observed_mask] + [np.roll(observed_mask, shift * MONTHS_PER_YEAR) for shift in shifts]
    )

    by_direction: dict[float, dict[str, np.ndarray]] = {}
    for direction in (1.0, -1.0):
        directions = np.full(len(commodities), direction)
        indicators, _ = signed_tail_indicators(values, directions, tail_quantile=spec.tail_quantile)
        by_direction[direction] = _signed_ratios(masks, values, finite, indicators, directions)

    indicators, thresholds = signed_tail_indicators(
        values, hypothesis_directions, tail_quantile=spec.tail_quantile
    )
    registered = _signed_ratios(masks, values, finite, indicators, hypothesis_directions)

    def shift_p(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
        extreme = np.nansum(null >= observed, axis=0)
        p_values = (1.0 + extreme) / (1.0 + null.shape[0])
        return np.where(np.isfinite(observed), p_values, np.nan)

    eligible = (
        (registered["inside_months"][0] >= spec.minimum_inside_months)
        & (registered["outside_months"][0] >= spec.minimum_outside_months)
        & np.isfinite(registered["log_signed_tail_ratio"][0])
    )
    registered_p = shift_p(
        registered["log_signed_tail_ratio"][0], registered["log_signed_tail_ratio"][1:]
    )
    # Controls are looked at in both tails, so the smaller of the two one-sided
    # p-values is the minimum of two tests and is not a p-value at the nominal
    # level. Doubling it is the Bonferroni correction across the two tails and
    # restores the size; without it a pure-noise control would reject about
    # twice as often as intended and void the stage for no reason.
    worst_case_p = np.minimum(
        2.0
        * np.minimum(
            shift_p(
                by_direction[1.0]["log_signed_tail_ratio"][0],
                by_direction[1.0]["log_signed_tail_ratio"][1:],
            ),
            shift_p(
                by_direction[-1.0]["log_signed_tail_ratio"][0],
                by_direction[-1.0]["log_signed_tail_ratio"][1:],
            ),
        ),
        1.0,
    )

    frame = pd.DataFrame(
        {
            "commodity": commodities,
            "role": [roles.get(name, "unclassified") for name in commodities],
            "registered_direction": [
                spec.hypotheses.get(name, "unregistered") for name in commodities
            ],
            "registered_reason": [spec.reasons.get(name, "") for name in commodities],
            "in_signed_family": is_hypothesised,
            "inside_months": registered["inside_months"][0].astype("int64"),
            "outside_months": registered["outside_months"][0].astype("int64"),
            "inside_tail_months": registered["inside_tail_months"][0].astype("int64"),
            "tail_threshold": thresholds,
            "log_signed_tail_ratio": np.where(
                eligible, registered["log_signed_tail_ratio"][0], np.nan
            ),
            "signed_mean_difference": np.where(
                eligible, registered["signed_mean_difference"][0], np.nan
            ),
            "eligible": eligible,
            "shift_p_value": np.where(eligible, registered_p, np.nan),
            "worst_case_shift_p_value": np.where(eligible, worst_case_p, np.nan),
        }
    )
    # A registered hypothesis is only supported if both statistics point the way
    # the hypothesis does; a tail ratio alone can be one spike.
    frame["statistics_agree"] = frame["log_signed_tail_ratio"].gt(0) & frame[
        "signed_mean_difference"
    ].gt(0)

    family = pd.Series(is_hypothesised & eligible, index=frame.index)
    frame["bh_q_value"] = np.nan
    if family.any():
        frame.loc[family, "bh_q_value"] = benjamini_hochberg(
            frame.loc[family].set_index("commodity")["shift_p_value"]
        ).to_numpy()
    frame["reject_fdr"] = frame["bh_q_value"].le(spec.fdr_alpha).fillna(False)

    controls = pd.Series(is_control & eligible, index=frame.index)
    control_rejections = int(
        frame.loc[controls, "worst_case_shift_p_value"].lt(spec.fdr_alpha).sum()
    )

    null_ratios = registered["log_signed_tail_ratio"][1:]
    centre = np.nanmean(null_ratios, axis=0)
    scale = np.nanstd(null_ratios, axis=0, ddof=1)
    usable = np.isfinite(scale) & (scale > 0)
    standardized_observed = np.where(
        usable, (registered["log_signed_tail_ratio"][0] - centre) / scale, np.nan
    )
    standardized_null = np.where(usable, (null_ratios - centre) / scale, np.nan)
    frame["standardized_log_signed_tail_ratio"] = standardized_observed

    family_positions = np.flatnonzero(family.to_numpy())
    resolution_floor = 1.0 / (len(shifts) + 1)
    family_statistic = float("nan")
    family_p = float("nan")
    if family_positions.size and np.isfinite(standardized_observed[family_positions]).any():
        with np.errstate(invalid="ignore"):
            family_statistic = float(np.nanmax(standardized_observed[family_positions]))
            null_maxima = np.nanmax(standardized_null[:, family_positions], axis=1)
        family_p = float(
            (1 + int(np.sum(null_maxima >= family_statistic))) / (1 + len(null_maxima))
        )

    clears_program = bool(np.isfinite(family_p) and family_p <= program.threshold)
    controls_reject = control_rejections > 0
    if controls_reject:
        status = "void_negative_controls_rejected"
    elif clears_program:
        status = "program_level_finding"
    else:
        status = "exploratory_within_stage_only"

    null_frame = pd.DataFrame(
        {
            "shift_years": np.repeat(shifts, len(commodities)),
            "commodity": commodities * len(shifts),
            "log_signed_tail_ratio": null_ratios.reshape(-1),
        }
    )

    summary: dict[str, Any] = {
        "cold_episodes": len(cold_episodes),
        "shift_count": len(shifts),
        "resolution_floor": resolution_floor,
        "resolution_limited": bool(resolution_floor > program.threshold),
        "program_threshold": program.threshold,
        "program_primary_tests": program.primary_tests,
        "signed_family_size": int(family.sum()),
        "excluded_commodities": len(spec.excluded),
        "family_maximum_standardized_signed_tail_ratio": family_statistic,
        "family_shift_p_value": family_p,
        "family_fdr_rejections": int(frame.loc[family, "reject_fdr"].sum()),
        "family_with_agreeing_statistics": int(frame.loc[family, "statistics_agree"].sum()),
        "control_family_size": int(controls.sum()),
        "control_worst_case_rejections": control_rejections,
        "negative_controls_rejected": controls_reject,
        "clears_program_threshold": clears_program,
        "status": status,
    }
    return frame, null_frame, summary
