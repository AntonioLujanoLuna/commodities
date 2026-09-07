"""Whole-year timing null for the locked v2 event-study decision family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .enso import construct_warm_episodes
from .specification_curve import circular_shift_enso, joint_null_p_value
from .validation import ValidationContract


def prepare_program_lookups(
    monthly: pd.DataFrame, contract: ValidationContract
) -> dict[str, dict[tuple[str, pd.Timestamp], float]]:
    selected = monthly.loc[
        monthly["commodity"].isin(contract.commodities),
        ["date", "commodity", *contract.timing_outcomes],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    return {
        outcome: {
            (str(row.commodity), pd.Timestamp(row.date)): float(getattr(row, outcome))
            for row in selected.itertuples(index=False)
            if pd.notna(getattr(row, outcome))
        }
        for outcome in contract.timing_outcomes
    }


def _episode_endpoints(
    lookup: dict[tuple[str, pd.Timestamp], float],
    episodes: pd.DataFrame,
    *,
    commodities: tuple[str, ...],
    anchor: str,
    horizon: int,
) -> pd.DataFrame:
    anchor_column = "onset_date" if anchor == "retrospective" else "observable_date"
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        start = pd.Timestamp(getattr(episode, anchor_column))
        dates = pd.date_range(start, periods=horizon + 1, freq="MS")
        for commodity in commodities:
            values = [lookup.get((commodity, pd.Timestamp(date))) for date in dates]
            endpoint = (
                float(sum(value for value in values if value is not None))
                if all(value is not None and np.isfinite(value) for value in values)
                else float("nan")
            )
            rows.append(
                {"episode_id": episode.episode_id, "commodity": commodity, "endpoint": endpoint}
            )
    return pd.DataFrame.from_records(rows)


def fit_program_alignment(
    monthly: pd.DataFrame,
    enso: pd.DataFrame,
    contract: ValidationContract,
    *,
    shift_years: int,
    threshold: float = 0.5,
    minimum_duration_months: int = 5,
    observable_delay_after_center_months: int = 2,
    minimum_valid_episodes: int = 10,
    lookups: dict[str, dict[tuple[str, pd.Timestamp], float]] | None = None,
) -> pd.DataFrame:
    shifted = circular_shift_enso(enso, years=shift_years, columns=contract.timing_indexes)
    value_lookups = lookups or prepare_program_lookups(monthly, contract)
    rows: list[dict[str, Any]] = []
    for index_name in contract.timing_indexes:
        episodes = construct_warm_episodes(
            shifted,
            index_name=index_name,
            threshold=threshold,
            minimum_duration_months=minimum_duration_months,
            observable_delay_after_center_months=observable_delay_after_center_months,
        )
        for anchor in contract.timing_anchors:
            for horizon in contract.timing_horizons:
                for outcome in contract.timing_outcomes:
                    endpoints = _episode_endpoints(
                        value_lookups[outcome],
                        episodes,
                        commodities=contract.commodities,
                        anchor=anchor,
                        horizon=horizon,
                    )
                    for commodity, group in endpoints.groupby("commodity", observed=True):
                        sample = group["endpoint"].dropna().to_numpy(dtype="float64")
                        standard_error = (
                            float(sample.std(ddof=1) / np.sqrt(len(sample)))
                            if len(sample) >= minimum_valid_episodes
                            else float("nan")
                        )
                        estimate = float(sample.mean()) if len(sample) else float("nan")
                        t_value = estimate / standard_error if standard_error > 0 else float("nan")
                        rows.append(
                            {
                                "shift_years": shift_years,
                                "is_observed_timing": shift_years == 0,
                                "commodity": commodity,
                                "index_definition": index_name,
                                "anchor": anchor,
                                "horizon_months": horizon,
                                "outcome": outcome,
                                "episodes": len(sample),
                                "estimate": estimate,
                                "standard_error": standard_error,
                                "t_statistic": t_value,
                            }
                        )
    return pd.DataFrame.from_records(rows)


def summarize_program_alignments(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_cells = int(cells.loc[cells["shift_years"].eq(0)].shape[0])
    for (shift, observed), group in cells.groupby(
        ["shift_years", "is_observed_timing"], sort=True, observed=True
    ):
        valid = group.loc[np.isfinite(group["t_statistic"])]
        if len(valid) != expected_cells:
            raise ValueError(f"Program timing shift {shift} has {len(valid)}/{expected_cells} valid cells")
        rows.append(
            {
                "shift_years": int(shift),
                "is_observed_timing": bool(observed),
                "valid_cells": len(valid),
                "median_absolute_t": float(valid["t_statistic"].abs().median()),
                "maximum_absolute_t": float(valid["t_statistic"].abs().max()),
                "positive_estimate_share": float(valid["estimate"].gt(0).mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def program_null_p_values(summaries: pd.DataFrame) -> dict[str, float]:
    return {
        "median_absolute_t_p_value": joint_null_p_value(summaries, "median_absolute_t"),
        "maximum_absolute_t_p_value": joint_null_p_value(summaries, "maximum_absolute_t"),
        "positive_estimate_share_p_value": joint_null_p_value(
            summaries, "positive_estimate_share"
        ),
    }
