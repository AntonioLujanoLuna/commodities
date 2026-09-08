"""W4: the ENSO response of the forward curve's shape.

A supply shock reaches the shape of the curve before it reaches the level. The
front contract moves relative to the deferred, the curve tips toward
backwardation, and it can do all of that while the spot index barely moves. The
World Bank series this project runs on are spot indexes and cannot see it, which
makes a plus-twelve-month cumulative return close to the worst available
instrument for a transient supply squeeze.

The estimand has one convenient property: the slope is a ratio of two
settlements from the same session, so it contains no roll. `tradability.py`
refuses to splice returns across a roll without an explicit roll-yield method,
and the slope sidesteps that question rather than answering it.

The licensed contract-level inputs this needs are not in the repository. Nothing
here manufactures them, and `curve_slope_response` is the only function that
produces a number -- it requires a validated futures panel to exist first. The
point of freezing the estimand now is that it is frozen before anyone sees
contract data.
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
from .dispersion import MONTHS_PER_YEAR, available_year_shifts, build_event_window_mask
from .tradability import validate_futures_panel


@dataclass(frozen=True)
class TermStructureSpec:
    config: dict[str, Any]
    front_rank: int
    deferred_rank: int
    minimum_days_to_expiry: int
    window_first_relative_month: int
    window_last_relative_month: int
    minimum_inside_months: int
    minimum_outside_months: int
    minimum_shift_years: int
    maximum_shift_years: int
    fdr_alpha: float


def load_term_structure_config(path: Path | None = None) -> TermStructureSpec:
    config_path = path or project_root() / "config" / "term_structure.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["workstream"] != "W4_term_structure":
        raise ValueError("Not the W4 term-structure contract")
    if raw["provenance"] != {
        "version": 1,
        "authored_on": date(2026, 9, 8),
        "program_register": "config/findings_v3.yaml",
    }:
        raise ValueError("Term-structure provenance is not the frozen W4 contract")
    design = raw["design"]
    if design["estimand"] != "front_minus_deferred_log_spread":
        raise ValueError("Unsupported W4 estimand")
    if not design["require_same_session_legs"]:
        raise ValueError("Both curve legs must come from the same session")
    requirements = raw["requirements"]
    if not requirements["price_indexes_are_not_admissible"]:
        raise ValueError("W4 must refuse price indexes; it needs contract-level settlements")
    if not requirements["no_result_without_inputs"]:
        raise ValueError("W4 must produce no result without its licensed inputs")
    front = int(design["front_rank"])
    deferred = int(design["deferred_rank"])
    if front < 1 or deferred <= front:
        raise ValueError("deferred_rank must sit beyond front_rank")
    return TermStructureSpec(
        config=raw,
        front_rank=front,
        deferred_rank=deferred,
        minimum_days_to_expiry=int(design["minimum_days_to_expiry"]),
        window_first_relative_month=int(design["window_first_relative_month"]),
        window_last_relative_month=int(design["window_last_relative_month"]),
        minimum_inside_months=int(raw["inference"]["minimum_inside_months"]),
        minimum_outside_months=int(raw["inference"]["minimum_outside_months"]),
        minimum_shift_years=int(raw["inference"]["minimum_shift_years"]),
        maximum_shift_years=int(raw["inference"]["maximum_shift_years"]),
        fdr_alpha=float(raw["inference"]["fdr_alpha"]),
    )


def build_curve_slopes(futures: pd.DataFrame, *, spec: TermStructureSpec) -> pd.DataFrame:
    """Daily front-minus-deferred log spread, from same-session settlements only.

    Contracts are ranked by time to expiry on each date. A session that does not
    carry both the front and the deferred leg yields no slope, rather than one
    built from a stale quote.
    """
    validate_futures_panel(futures)
    frame = futures.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["days_to_expiry"].ge(spec.minimum_days_to_expiry)]
    frame = frame.sort_values(["commodity", "date", "days_to_expiry", "contract"])
    frame["rank"] = (
        frame.groupby(["commodity", "date"], observed=True).cumcount().astype("int64") + 1
    )
    legs = frame.loc[frame["rank"].isin([spec.front_rank, spec.deferred_rank])]
    wide = legs.pivot_table(
        index=["commodity", "date"],
        columns="rank",
        values="settlement_price",
        aggfunc="first",
    )
    if spec.front_rank not in wide.columns or spec.deferred_rank not in wide.columns:
        return pd.DataFrame(columns=["commodity", "date", "log_spread"])
    complete = wide.dropna(subset=[spec.front_rank, spec.deferred_rank])
    slopes = complete.reset_index().loc[:, ["commodity", "date"]]
    slopes["log_spread"] = np.log(
        complete[spec.front_rank].to_numpy(dtype="float64")
        / complete[spec.deferred_rank].to_numpy(dtype="float64")
    )
    return slopes.reset_index(drop=True)


def monthly_mean_spread(slopes: pd.DataFrame) -> pd.DataFrame:
    """Average the daily slope within each calendar month."""
    if slopes.empty:
        return pd.DataFrame(columns=["commodity", "date", "log_spread"])
    frame = slopes.copy()
    frame["date"] = pd.to_datetime(frame["date"]).values.astype("datetime64[M]")
    return (
        frame.groupby(["commodity", "date"], observed=True)["log_spread"]
        .mean()
        .reset_index()
        .sort_values(["commodity", "date"], ignore_index=True)
    )


def curve_slope_response(
    monthly_spreads: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    spec: TermStructureSpec,
) -> pd.DataFrame:
    """Inside-minus-outside difference in the curve slope, against the shift null.

    The reference distribution is the one W1 and W2 use: circular whole-year
    shifts of the episode mask, which never touch the spread series.
    """
    if monthly_spreads.empty:
        raise ValueError("W4 has no curve slopes to work with; its inputs are absent")
    rows: list[dict[str, Any]] = []
    for commodity, sample in monthly_spreads.groupby("commodity", observed=True):
        grid = pd.date_range(sample["date"].min(), sample["date"].max(), freq="MS")
        values = (
            sample.set_index("date")["log_spread"].reindex(grid).to_numpy(dtype="float64")
        )
        finite = np.isfinite(values)
        mask = build_event_window_mask(
            pd.DatetimeIndex(grid),
            episodes["onset_date"],
            first_relative_month=spec.window_first_relative_month,
            last_relative_month=spec.window_last_relative_month,
        )
        shifts = available_year_shifts(
            len(grid),
            minimum_shift_years=spec.minimum_shift_years,
            maximum_shift_years=spec.maximum_shift_years,
        )
        masks = np.vstack(
            [mask] + [np.roll(mask, shift * MONTHS_PER_YEAR) for shift in shifts]
        ).astype("float64")
        filled = np.where(finite, values, 0.0)
        indicator = finite.astype("float64")
        inside_counts = masks @ indicator
        outside_counts = (1.0 - masks) @ indicator
        with np.errstate(invalid="ignore", divide="ignore"):
            inside_mean = np.where(inside_counts > 0, (masks @ filled) / inside_counts, np.nan)
            outside_mean = np.where(
                outside_counts > 0, ((1.0 - masks) @ filled) / outside_counts, np.nan
            )
        difference = inside_mean - outside_mean
        eligible = (
            inside_counts[0] >= spec.minimum_inside_months
            and outside_counts[0] >= spec.minimum_outside_months
            and np.isfinite(difference[0])
        )
        null = difference[1:]
        p_value = (
            float((1 + int(np.nansum(null >= difference[0]))) / (1 + len(null)))
            if eligible
            else float("nan")
        )
        rows.append(
            {
                "commodity": str(commodity),
                "inside_months": int(inside_counts[0]),
                "outside_months": int(outside_counts[0]),
                "inside_mean_log_spread": float(inside_mean[0]),
                "outside_mean_log_spread": float(outside_mean[0]),
                "spread_difference": float(difference[0]) if eligible else float("nan"),
                "shift_p_value": p_value,
                "shift_count": len(shifts),
                "resolution_floor": 1.0 / (len(shifts) + 1),
                "eligible": bool(eligible),
            }
        )
    return pd.DataFrame.from_records(rows)


def term_structure_status(futures: pd.DataFrame | None) -> dict[str, Any]:
    """Report what W4 can say, which without licensed inputs is nothing.

    This mirrors the v2 futures gate rather than inventing a new convention: an
    input schema is not empirical evidence, and the receipt has to say so
    structurally rather than in prose a reader might skim past.
    """
    if futures is None or futures.empty:
        return {
            "workstream": "W4_term_structure",
            "inputs_present": False,
            "empirical_result": False,
            "primary_tests_contributed": 0,
            "reason": (
                "Licensed contract-level futures settlements are not available to this "
                "project. The estimand is frozen; no number is reported."
            ),
        }
    validate_futures_panel(futures)
    return {
        "workstream": "W4_term_structure",
        "inputs_present": True,
        "empirical_result": True,
        "primary_tests_contributed": 1,
        "reason": (
            "Licensed inputs are present. config/findings_v3.yaml must be updated to "
            "register this workstream's primary test before any result is read."
        ),
    }
