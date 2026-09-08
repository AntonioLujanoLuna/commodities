"""W1: dispersion of monthly returns inside ENSO event windows.

The frozen primary design reduces a commodity to one number per episode, so its
sample size is the episode count and nothing else. A dispersion statistic
computed over the same windows uses every commodity-month instead, which is
roughly six hundred observations rather than seventeen. That is the whole
argument for this stage: same data, same windows, an estimand that does not throw
the months away.

The cost is that the null becomes the hard part. Commodity returns cluster in
volatility, so a test that treats months as exchangeable finds variance
differences everywhere -- the negative controls first. The reference distribution
here therefore never touches the return series. It circularly shifts the episode
mask by whole years, which preserves each commodity's seasonality and its
volatility clustering exactly, preserves ENSO's own phase-locking to the
calendar, and varies only the timing of the treatment.
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
from .statistics import benjamini_hochberg

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class ProgramRegister:
    """The frozen program-level multiplicity contract shared by every v3 stage."""

    config: dict[str, Any]
    program_alpha: float
    primary_tests: int

    @property
    def threshold(self) -> float:
        return self.program_alpha / self.primary_tests


@dataclass(frozen=True)
class DispersionSpec:
    config: dict[str, Any]
    outcome: str
    window_first_relative_month: int
    window_last_relative_month: int
    exceedance_quantile: float
    alternative: str
    minimum_inside_months: int
    minimum_outside_months: int
    minimum_shift_years: int
    maximum_shift_years: int
    fdr_alpha: float


def load_program_register(path: Path | None = None) -> ProgramRegister:
    """Read the frozen v3 register and total the primary tests it declares."""
    config_path = path or project_root() / "config" / "findings_v3.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if int(raw["program_version"]) != 3:
        raise ValueError("Program register is not the v3 contract")
    alpha = float(raw["program_alpha"])
    if not 0 < alpha < 1:
        raise ValueError("program_alpha must fall between zero and one")
    workstreams = raw["workstreams"]
    if not isinstance(workstreams, dict) or not workstreams:
        raise ValueError("Program register declares no workstreams")
    total = 0
    for name, entry in workstreams.items():
        tests = int(entry["primary_tests"])
        if tests < 0:
            raise ValueError(f"{name}: primary_tests cannot be negative")
        if bool(entry["produces_empirical_result"]) != (tests > 0):
            raise ValueError(f"{name}: primary_tests disagrees with produces_empirical_result")
        total += tests
    if total < 1:
        raise ValueError("Program register declares no primary tests")
    return ProgramRegister(config=raw, program_alpha=alpha, primary_tests=total)


def load_dispersion_config(path: Path | None = None) -> DispersionSpec:
    config_path = path or project_root() / "config" / "dispersion.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["provenance"] != {
        "version": 1,
        "authored_on": date(2026, 9, 8),
        "inference_scope": "prespecified_secondary_endpoint",
        "program_register": "config/findings_v3.yaml",
        "workstream": "W1_dispersion",
    }:
        raise ValueError("Dispersion provenance is not the frozen W1 contract")
    design = raw["design"]
    if design["primary_statistic"] != "log_variance_ratio":
        raise ValueError("W1 primary statistic is the log variance ratio")
    if design["secondary_statistic"] != "log_exceedance_ratio":
        raise ValueError("W1 secondary statistic is the log exceedance ratio")
    if design["exceedance_threshold_scope"] != "unconditional_absolute_return":
        raise ValueError(
            "Exceedance thresholds must be unconditional, so the null cannot move them"
        )
    if design["continuity_correction"] != "haldane_anscombe":
        raise ValueError("Unsupported continuity correction")
    alternative = str(design["alternative"])
    if alternative not in {"greater", "two_sided"}:
        raise ValueError("alternative must be 'greater' or 'two_sided'")
    inference = raw["inference"]
    if inference["reference_distribution"] != "circular_whole_year_shift":
        raise ValueError("W1 requires the circular whole-year shift null")
    if inference["family_statistic"] != "maximum_standardized":
        raise ValueError("W1 family statistic must be the standardized maximum")
    if inference["fdr_family"] != "mechanism_candidates_only":
        raise ValueError("W1 FDR family must contain mechanism candidates only")
    falsification = raw["falsification"]
    if not falsification["negative_controls_must_not_reject"]:
        raise ValueError("W1 must keep its negative-control falsification enabled")
    if not falsification["void_stage_if_controls_reject"]:
        raise ValueError("W1 must void the stage when its controls reject")
    quantile = float(design["exceedance_quantile"])
    if not 0.5 < quantile < 1:
        raise ValueError("exceedance_quantile must fall in (0.5, 1)")
    first = int(design["window_first_relative_month"])
    last = int(design["window_last_relative_month"])
    if first > last:
        raise ValueError("Dispersion window endpoints are reversed")
    minimum_shift_years = int(inference["minimum_shift_years"])
    maximum_shift_years = int(inference["maximum_shift_years"])
    if minimum_shift_years < 1:
        raise ValueError("minimum_shift_years must be positive")
    if maximum_shift_years < minimum_shift_years:
        raise ValueError("maximum_shift_years must not precede minimum_shift_years")
    return DispersionSpec(
        config=raw,
        outcome=str(design["outcome"]),
        window_first_relative_month=first,
        window_last_relative_month=last,
        exceedance_quantile=quantile,
        alternative=alternative,
        minimum_inside_months=int(design["minimum_inside_months"]),
        minimum_outside_months=int(design["minimum_outside_months"]),
        minimum_shift_years=minimum_shift_years,
        maximum_shift_years=maximum_shift_years,
        fdr_alpha=float(inference["fdr_alpha"]),
    )


def build_monthly_matrix(
    returns: pd.DataFrame, *, outcome_column: str
) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    """Pivot monthly returns onto a gap-free month grid.

    A circular shift is only meaningful on a contiguous calendar, so missing
    months are materialised as NaN rather than silently closing the gap.
    """
    required = {"date", "commodity", outcome_column}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"Monthly returns are missing columns: {sorted(missing)}")
    frame = returns.loc[:, ["date", "commodity", outcome_column]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["date", "commodity"]).any():
        raise ValueError("Monthly returns have duplicate date/commodity keys")
    wide = frame.pivot(index="date", columns="commodity", values=outcome_column)
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    wide = wide.reindex(grid)
    commodities = [str(name) for name in wide.columns]
    return pd.DatetimeIndex(grid), commodities, wide.to_numpy(dtype="float64")


def build_event_window_mask(
    dates: pd.DatetimeIndex,
    onsets: pd.Series,
    *,
    first_relative_month: int,
    last_relative_month: int,
) -> np.ndarray:
    """Flag every month falling inside any episode's event window."""
    if first_relative_month > last_relative_month:
        raise ValueError("Window endpoints are reversed")
    parsed = pd.to_datetime(onsets)
    if parsed.empty:
        raise ValueError("At least one episode onset is required")
    mask = np.zeros(len(dates), dtype=bool)
    for onset in parsed:
        start = onset + pd.DateOffset(months=first_relative_month)
        end = onset + pd.DateOffset(months=last_relative_month)
        mask |= (dates >= start) & (dates <= end)
    return mask


def available_year_shifts(
    month_count: int, *, minimum_shift_years: int, maximum_shift_years: int
) -> list[int]:
    """Whole-year circular shifts that leave the mask genuinely displaced.

    A shift of a whole number of years preserves ENSO's phase-locking to the
    annual cycle; a shift equal to the sample length is the identity and is
    excluded, as is anything that would wrap past it.

    One assumption is worth naming because nothing here enforces it. If episodes
    recurred at a fixed period of ``p`` years, every shift that is a multiple of
    ``p`` would map the mask onto itself, those draws would tie with the observed
    statistic, and the null would lose that fraction of its resolution. Real ENSO
    recurrence is irregular -- two to seven years -- so this does not bite in
    practice, but a treatment that were near-periodic would need a different
    reference distribution rather than this one.
    """
    if month_count < 2 * MONTHS_PER_YEAR:
        raise ValueError("A whole-year shift null needs at least two years of months")
    sample_years = month_count // MONTHS_PER_YEAR
    upper = min(maximum_shift_years, sample_years - 1)
    if upper < minimum_shift_years:
        raise ValueError("The sample is too short for the requested shift range")
    return list(range(minimum_shift_years, upper + 1))


def _ratio_statistics(
    masks: np.ndarray,
    values: np.ndarray,
    finite: np.ndarray,
    exceedances: np.ndarray,
) -> dict[str, np.ndarray]:
    """Inside/outside variance and exceedance ratios for every mask at once.

    ``masks`` is (draws, months); the three matrix products below give the count,
    sum and sum of squares of the observed returns selected by each mask, for
    every commodity, in one pass.
    """
    filled = np.where(finite, values, 0.0)
    squares = np.where(finite, values**2, 0.0)
    indicator = finite.astype("float64")
    exceeded = np.where(finite, exceedances, False).astype("float64")

    inside = masks.astype("float64")
    outside = 1.0 - inside

    def moments(selector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = selector @ indicator
        totals = selector @ filled
        squared = selector @ squares
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, totals / counts, np.nan)
            variance = np.where(counts > 1, (squared - counts * means**2) / (counts - 1.0), np.nan)
        return counts, means, np.clip(variance, 0.0, None)

    inside_counts, _, inside_variance = moments(inside)
    outside_counts, _, outside_variance = moments(outside)
    inside_exceeded = inside @ exceeded
    outside_exceeded = outside @ exceeded

    with np.errstate(invalid="ignore", divide="ignore"):
        variance_ratio = np.where(
            (inside_variance > 0) & (outside_variance > 0),
            np.log(inside_variance / outside_variance),
            np.nan,
        )
        # Haldane-Anscombe: a window that happens to contain no exceedance is a
        # small count, not an infinitely strong result.
        inside_rate = (inside_exceeded + 0.5) / (inside_counts + 1.0)
        outside_rate = (outside_exceeded + 0.5) / (outside_counts + 1.0)
        exceedance_ratio = np.where(
            (inside_counts > 0) & (outside_counts > 0),
            np.log(inside_rate / outside_rate),
            np.nan,
        )
    return {
        "inside_months": inside_counts,
        "outside_months": outside_counts,
        "inside_variance": inside_variance,
        "outside_variance": outside_variance,
        "log_variance_ratio": variance_ratio,
        "inside_exceedance_rate": inside_rate,
        "outside_exceedance_rate": outside_rate,
        "log_exceedance_ratio": exceedance_ratio,
    }


def _shift_p_values(observed: np.ndarray, null: np.ndarray, *, alternative: str) -> np.ndarray:
    """Finite-sample shift p-values, one per commodity."""
    draws = null.shape[0]
    if alternative == "greater":
        extreme = np.nansum(null >= observed, axis=0)
    else:
        centre = np.nanmean(null, axis=0)
        extreme = np.nansum(np.abs(null - centre) >= np.abs(observed - centre), axis=0)
    valid = np.isfinite(null).sum(axis=0)
    p_values = (1.0 + extreme) / (1.0 + draws)
    return np.where(np.isfinite(observed) & (valid > 0), p_values, np.nan)


def _standardize(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express observed and null statistics in units of their own shift-null scale.

    Commodities differ in how variable the ratio is across shifts, so the family
    maximum is only meaningful once each column has been put on a common scale.
    The moments come from the same shift set the maximum is compared against,
    which is mildly optimistic and is recorded as such in the receipt.
    """
    centre = np.nanmean(null, axis=0)
    scale = np.nanstd(null, axis=0, ddof=1)
    usable = np.isfinite(scale) & (scale > 0)
    standardized_observed = np.where(usable, (observed - centre) / scale, np.nan)
    standardized_null = np.where(usable, (null - centre) / scale, np.nan)
    return standardized_observed, standardized_null


def dispersion_shift_inference(
    returns: pd.DataFrame,
    episodes: pd.DataFrame,
    roles: pd.Series,
    *,
    spec: DispersionSpec,
    program: ProgramRegister,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run W1 end to end and return per-commodity results, the null, and a summary."""
    dates, commodities, values = build_monthly_matrix(returns, outcome_column=spec.outcome)
    unknown = sorted(set(roles.index) - set(commodities))
    if unknown:
        raise ValueError(f"Roles name commodities absent from the returns panel: {unknown}")
    finite = np.isfinite(values)
    absolute = np.abs(np.where(finite, values, np.nan))
    with np.errstate(invalid="ignore"):
        thresholds = np.nanquantile(absolute, spec.exceedance_quantile, axis=0)
    exceedances = finite & (absolute > thresholds)

    observed_mask = build_event_window_mask(
        dates,
        episodes["onset_date"],
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
    statistics = _ratio_statistics(masks, values, finite, exceedances)

    eligible = (
        (statistics["inside_months"][0] >= spec.minimum_inside_months)
        & (statistics["outside_months"][0] >= spec.minimum_outside_months)
        & np.isfinite(statistics["log_variance_ratio"][0])
    )

    frame = pd.DataFrame(
        {
            "commodity": commodities,
            "role": [roles.get(name, "unclassified") for name in commodities],
            "inside_months": statistics["inside_months"][0].astype("int64"),
            "outside_months": statistics["outside_months"][0].astype("int64"),
            "exceedance_threshold": thresholds,
            "inside_variance": statistics["inside_variance"][0],
            "outside_variance": statistics["outside_variance"][0],
            "log_variance_ratio": np.where(eligible, statistics["log_variance_ratio"][0], np.nan),
            "log_exceedance_ratio": np.where(
                eligible, statistics["log_exceedance_ratio"][0], np.nan
            ),
            "eligible": eligible,
        }
    )

    null_variance = statistics["log_variance_ratio"][1:]
    null_exceedance = statistics["log_exceedance_ratio"][1:]
    frame["variance_shift_p_value"] = _shift_p_values(
        frame["log_variance_ratio"].to_numpy(), null_variance, alternative=spec.alternative
    )
    frame["exceedance_shift_p_value"] = _shift_p_values(
        frame["log_exceedance_ratio"].to_numpy(), null_exceedance, alternative=spec.alternative
    )
    standardized_observed, standardized_null = _standardize(
        frame["log_variance_ratio"].to_numpy(), null_variance
    )
    frame["standardized_log_variance_ratio"] = standardized_observed
    # The two statistics answer the same question; when they disagree the
    # variance result is one month, and the contract says to read it that way.
    frame["statistics_agree"] = (
        frame["log_variance_ratio"].gt(0) & frame["log_exceedance_ratio"].gt(0)
    ) | (frame["log_variance_ratio"].lt(0) & frame["log_exceedance_ratio"].lt(0))

    candidates = frame["role"].eq("mechanism_candidate") & frame["eligible"]
    frame["bh_q_value"] = np.nan
    if candidates.any():
        frame.loc[candidates, "bh_q_value"] = benjamini_hochberg(
            frame.loc[candidates].set_index("commodity")["variance_shift_p_value"]
        ).to_numpy()
    frame["reject_fdr"] = frame["bh_q_value"].le(spec.fdr_alpha).fillna(False)

    controls = frame["role"].eq("negative_control") & frame["eligible"]
    control_rejections = int(frame.loc[controls, "variance_shift_p_value"].lt(spec.fdr_alpha).sum())

    candidate_positions = np.flatnonzero(candidates.to_numpy())
    resolution_floor = 1.0 / (len(shifts) + 1)
    family_p = float("nan")
    family_statistic = float("nan")
    if candidate_positions.size:
        with np.errstate(invalid="ignore"):
            family_statistic = float(
                np.nanmax(standardized_observed[candidate_positions])
                if np.isfinite(standardized_observed[candidate_positions]).any()
                else np.nan
            )
            null_maxima = np.nanmax(standardized_null[:, candidate_positions], axis=1)
        if np.isfinite(family_statistic):
            family_p = float(
                (1 + int(np.sum(null_maxima >= family_statistic))) / (1 + len(null_maxima))
            )

    clears_program = bool(np.isfinite(family_p) and family_p <= program.threshold)
    resolution_limited = bool(resolution_floor > program.threshold)
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
            "log_variance_ratio": null_variance.reshape(-1),
            "log_exceedance_ratio": null_exceedance.reshape(-1),
        }
    )

    summary: dict[str, Any] = {
        "shift_count": len(shifts),
        "resolution_floor": resolution_floor,
        "resolution_limited": resolution_limited,
        "program_threshold": program.threshold,
        "program_primary_tests": program.primary_tests,
        "family_size": int(candidates.sum()),
        "family_maximum_standardized_log_variance_ratio": family_statistic,
        "family_shift_p_value": family_p,
        "candidate_fdr_rejections": int(frame.loc[candidates, "reject_fdr"].sum()),
        "candidates_with_agreeing_statistics": int(frame.loc[candidates, "statistics_agree"].sum()),
        "control_family_size": int(controls.sum()),
        "control_raw_rejections": control_rejections,
        "negative_controls_rejected": controls_reject,
        "clears_program_threshold": clears_program,
        "status": status,
        "standardization_note": (
            "Shift-null moments are computed from the same shift set the family maximum is "
            "compared against, which is mildly optimistic."
        ),
    }
    return frame, null_frame, summary
