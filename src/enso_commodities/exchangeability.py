"""Is the neutral-date placebo pool drawn from the same history as the events?

The placebo matches an anchor's calendar month to the real onset calendar-month
distribution and nothing else. Eligible anchors must also satisfy
``|index| < threshold`` and sit outside every real onset's frozen event window,
and with seventeen episodes those windows cover most of the sample. So the
eligible pool is not scattered across history: it is whatever quiet stretches
survive both filters, and those stretches need not be spread across eras the
way the onsets are.

That matters because the negative controls are financial assets. Gold, platinum
and silver carry most of their sixty-five-year variance in two episodes -- the
1970s inflation and the 2001-2011 bull market. If onsets and eligible anchors
sit in systematically different parts of history, the specificity gate compares
event windows drawn from one macro-financial regime with placebo windows drawn
from another, and the controls can reject for a reason that has nothing to do
with ENSO and nothing to do with the pipeline being wrong.

This module does not fix that. It measures whether the condition holds, which
has to happen before anyone argues about the cause. The null it tests against
is the placebo's own draw distribution, so a rejection is a statement about the
placebo design in the exact form the study uses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EraBalanceOutput:
    statistics: pd.DataFrame
    eras: pd.DataFrame


def assign_era(dates: pd.Series, *, era_length_years: int, origin_year: int) -> pd.Series:
    """Label each date with the first year of its era block."""
    if era_length_years < 1:
        raise ValueError("era_length_years must be positive")
    years = pd.to_datetime(dates).dt.year
    return origin_year + ((years - origin_year) // era_length_years) * era_length_years


def _location_statistics(dates: pd.Series) -> dict[str, float]:
    years = pd.to_datetime(dates).dt.year.astype("float64")
    fractional = years + (pd.to_datetime(dates).dt.month.astype("float64") - 1) / 12
    return {
        "mean_year": float(fractional.mean()),
        "median_year": float(fractional.median()),
        "year_dispersion": float(fractional.std(ddof=1)) if len(fractional) > 1 else float("nan"),
    }


def _two_sided_p_value(null_values: np.ndarray, observed: float) -> float:
    finite = null_values[np.isfinite(null_values)]
    if finite.size == 0 or not np.isfinite(observed):
        return float("nan")
    lower = (int(np.sum(finite <= observed)) + 1) / (finite.size + 1)
    upper = (int(np.sum(finite >= observed)) + 1) / (finite.size + 1)
    return min(1.0, 2 * min(lower, upper))


def era_balance(
    onset_dates: pd.Series,
    eligible_anchors: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    era_length_years: int = 10,
    confidence_level: float = 0.95,
) -> EraBalanceOutput:
    """Compare where in history the real onsets and the placebo draws sit.

    ``draws`` is the placebo's own calendar-month-matched draw table, so the
    reference distribution is the one the specificity gate actually samples
    from rather than an idealised version of it. Each replicate contributes one
    anchor set of the same size and calendar-month composition as the real
    onsets; the statistic is where that set sits in time.

    A small p-value on ``mean_year`` says the placebo systematically samples a
    different era than the events, which breaks the exchangeability the gate
    assumes. A small p-value on ``year_dispersion`` says the placebo spreads
    across history differently even when it is centred correctly.
    """
    if "anchor_date" not in draws.columns or "replicate" not in draws.columns:
        raise ValueError("Placebo draws must have replicate and anchor_date columns")
    if "date" not in eligible_anchors.columns:
        raise ValueError("Eligible anchors are missing date")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must fall between zero and one")
    onsets = pd.to_datetime(onset_dates).sort_values()
    if onsets.empty:
        raise ValueError("At least one onset is required")

    observed = _location_statistics(onsets)
    replicate_stats = (
        draws.assign(anchor_date=pd.to_datetime(draws["anchor_date"]))
        .groupby("replicate", sort=True, observed=True)["anchor_date"]
        .apply(lambda group: pd.Series(_location_statistics(group)))
        .unstack()
    )

    alpha = 1 - confidence_level
    statistic_rows: list[dict[str, object]] = []
    for name, value in observed.items():
        nulls = replicate_stats[name].to_numpy(dtype="float64")
        finite = nulls[np.isfinite(nulls)]
        statistic_rows.append(
            {
                "statistic": name,
                "observed": value,
                "placebo_mean": float(finite.mean()) if finite.size else float("nan"),
                "placebo_interval_lower": (
                    float(np.quantile(finite, alpha / 2)) if finite.size else float("nan")
                ),
                "placebo_interval_upper": (
                    float(np.quantile(finite, 1 - alpha / 2)) if finite.size else float("nan")
                ),
                "observed_minus_placebo_mean": (
                    value - float(finite.mean()) if finite.size else float("nan")
                ),
                "p_value": _two_sided_p_value(nulls, value),
                "valid_replicates": int(finite.size),
            }
        )

    origin_year = int(
        min(
            pd.to_datetime(eligible_anchors["date"]).dt.year.min(),
            onsets.dt.year.min(),
        )
    )
    era_kwargs = {"era_length_years": era_length_years, "origin_year": origin_year}
    onset_eras = assign_era(onsets, **era_kwargs).value_counts()
    pool_eras = assign_era(eligible_anchors["date"], **era_kwargs).value_counts()
    draw_eras = assign_era(draws["anchor_date"], **era_kwargs).value_counts()
    replicates = int(draws["replicate"].nunique())
    index = pd.Index(
        sorted(set(onset_eras.index) | set(pool_eras.index) | set(draw_eras.index)),
        name="era_start_year",
    )
    eras = pd.DataFrame(
        {
            "era_start_year": index,
            "era_length_years": era_length_years,
            "onsets": onset_eras.reindex(index, fill_value=0).to_numpy(),
            "eligible_anchors": pool_eras.reindex(index, fill_value=0).to_numpy(),
            "mean_drawn_anchors": draw_eras.reindex(index, fill_value=0).to_numpy() / replicates,
        }
    )
    eras["onset_share"] = eras["onsets"] / len(onsets)
    eras["eligible_anchor_share"] = eras["eligible_anchors"] / len(eligible_anchors)
    eras["drawn_anchor_share"] = eras["mean_drawn_anchors"] / len(onsets)
    eras["onset_minus_drawn_share"] = eras["onset_share"] - eras["drawn_anchor_share"]
    return EraBalanceOutput(
        statistics=pd.DataFrame.from_records(statistic_rows),
        eras=eras,
    )
