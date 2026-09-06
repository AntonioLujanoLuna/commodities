"""Whole-year circular-shift null for the panel specification curve.

The diagnostic asks whether the actual timing of ENSO produces a stronger,
more consistently signed specification family than arbitrary whole-year
alignments of the exact same ENSO series. Rolling by multiples of twelve keeps
seasonality and serial dependence intact and changes only the alignment with
commodity returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .panel import (
    EXPOSURE_TERM,
    PanelSpec,
    build_panel,
    fit_exposure_panel,
    panel_regressors,
)


@dataclass(frozen=True)
class CurveSpec:
    minimum_shift_years: int
    maximum_shift_years: int
    minimum_valid_cell_share: float


def load_curve_config(path: Path) -> CurveSpec:
    with path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    null = raw["timing_null"]
    curve = raw["curve"]
    if null["shift_unit"] != "whole_year" or null["wrap"] != "circular":
        raise ValueError("The timing null must use circular whole-year shifts")
    if not bool(null["include_observed"]):
        raise ValueError("The specification curve must retain the observed timing")
    if curve["primary_statistic"] != "median_absolute_naive_t":
        raise ValueError("Unsupported primary specification-curve statistic")
    expected_secondary = {"maximum_absolute_naive_t", "positive_estimate_share"}
    if set(curve["secondary_statistics"]) != expected_secondary:
        raise ValueError("Unsupported secondary specification-curve statistics")
    spec = CurveSpec(
        minimum_shift_years=int(null["minimum_shift_years"]),
        maximum_shift_years=int(null["maximum_shift_years"]),
        minimum_valid_cell_share=float(curve["minimum_valid_cell_share"]),
    )
    if spec.minimum_shift_years < 1:
        raise ValueError("minimum_shift_years must be positive")
    if spec.maximum_shift_years < spec.minimum_shift_years:
        raise ValueError("maximum_shift_years precedes minimum_shift_years")
    if not 0 < spec.minimum_valid_cell_share <= 1:
        raise ValueError("minimum_valid_cell_share must fall in (0, 1]")
    return spec


def circular_shift_enso(
    enso: pd.DataFrame, *, years: int, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Roll ENSO values by complete years without changing any calendar date."""
    if years < 0:
        raise ValueError("years must be non-negative")
    required = {"date", *columns}
    if not required.issubset(enso.columns):
        raise ValueError(f"ENSO data is missing columns: {sorted(required - set(enso))}")
    result = enso.copy().sort_values("date", ignore_index=True)
    dates = pd.to_datetime(result["date"])
    if dates.duplicated().any():
        raise ValueError("ENSO dates must be unique")
    if len(dates) < 12:
        raise ValueError("ENSO data must contain at least one year")
    month_steps = 12 * years
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype="float64")
        result[column] = np.roll(values, month_steps)
    result["date"] = dates
    return result


def fit_curve(
    monthly: pd.DataFrame,
    enso: pd.DataFrame,
    exposure: pd.DataFrame,
    panel_spec: PanelSpec,
    *,
    shift_years: int,
) -> pd.DataFrame:
    """Fit every frozen panel cell once for one timing alignment."""
    shifted = circular_shift_enso(enso, years=shift_years, columns=panel_spec.index_definitions)
    rows: list[dict[str, Any]] = []
    for index_definition in panel_spec.index_definitions:
        for lag_months in panel_spec.reported_lag_months:
            for weighting in panel_spec.weighting_schemes:
                cell = f"{index_definition}:lag{lag_months}:{weighting}"
                panel = build_panel(
                    monthly,
                    shifted,
                    exposure,
                    index_name=index_definition,
                    lag_months=lag_months,
                    outcome_column=panel_spec.outcome_column,
                    weighting=weighting,
                )
                if len(panel) < panel_spec.minimum_observations:
                    raise ValueError(f"Panel cell {cell} is below the configured minimum")
                fit = fit_exposure_panel(
                    panel,
                    tolerance=panel_spec.demeaning_tolerance,
                    max_iterations=panel_spec.demeaning_max_iterations,
                    regressors=panel_regressors(weighting),
                )
                estimate = fit.coefficients[EXPOSURE_TERM]
                standard_error = fit.naive_standard_errors[EXPOSURE_TERM]
                naive_t = estimate / standard_error if standard_error > 0 else float("nan")
                rows.append(
                    {
                        "shift_years": shift_years,
                        "is_observed_timing": shift_years == 0,
                        "cell": cell,
                        "index_definition": index_definition,
                        "lag_months": lag_months,
                        "weighting": weighting,
                        "estimate": estimate,
                        "naive_standard_error": standard_error,
                        "naive_t": naive_t,
                        "observations": fit.observations,
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_curve(cells: pd.DataFrame, *, minimum_valid_cell_share: float) -> pd.DataFrame:
    required = {"shift_years", "is_observed_timing", "estimate", "naive_t", "cell"}
    if not required.issubset(cells.columns):
        raise ValueError(f"Curve cells are missing columns: {sorted(required - set(cells))}")
    total_cells = int(cells["cell"].nunique())
    rows: list[dict[str, Any]] = []
    for (shift, observed), group in cells.groupby(
        ["shift_years", "is_observed_timing"], sort=True, observed=True
    ):
        valid = group.loc[np.isfinite(group["naive_t"])]
        valid_share = len(valid) / total_cells if total_cells else 0.0
        if valid_share < minimum_valid_cell_share:
            raise ValueError(f"Shift {shift} has only {valid_share:.1%} valid cells")
        rows.append(
            {
                "shift_years": int(shift),
                "is_observed_timing": bool(observed),
                "valid_cells": len(valid),
                "median_absolute_naive_t": float(valid["naive_t"].abs().median()),
                "maximum_absolute_naive_t": float(valid["naive_t"].abs().max()),
                "positive_estimate_share": float(valid["estimate"].gt(0).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def joint_null_p_value(summaries: pd.DataFrame, statistic: str) -> float:
    observed = summaries.loc[summaries["is_observed_timing"], statistic]
    null = summaries.loc[~summaries["is_observed_timing"], statistic].dropna()
    if len(observed) != 1 or null.empty:
        raise ValueError("Joint null requires one observed curve and at least one shifted curve")
    return float((1 + null.ge(float(observed.iloc[0])).sum()) / (len(null) + 1))
