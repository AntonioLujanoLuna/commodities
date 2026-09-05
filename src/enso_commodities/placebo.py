from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PlaceboOutput:
    results: pd.DataFrame
    replicates: pd.DataFrame


def eligible_neutral_anchors(
    index_data: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    index_name: str,
    neutral_absolute_threshold: float,
    exclusion_window: tuple[int, int],
    first_anchor: pd.Timestamp,
    last_anchor: pd.Timestamp,
) -> pd.DataFrame:
    required_index = {"date", index_name}
    if not required_index.issubset(index_data.columns):
        raise ValueError(
            f"Index data is missing columns: {sorted(required_index - set(index_data))}"
        )
    if "onset_date" not in episodes.columns:
        raise ValueError("Episodes are missing onset_date")
    if exclusion_window[0] > exclusion_window[1]:
        raise ValueError("Placebo exclusion-window endpoints are reversed")

    anchors = index_data.loc[:, ["date", index_name]].copy()
    anchors["date"] = pd.to_datetime(anchors["date"])
    anchors[index_name] = pd.to_numeric(anchors[index_name], errors="coerce")
    if anchors["date"].duplicated().any():
        raise ValueError("Index dates must be unique")
    anchors = anchors.loc[
        anchors[index_name].abs().lt(neutral_absolute_threshold)
        & anchors["date"].between(first_anchor, last_anchor)
    ].copy()

    blocked = pd.Series(False, index=anchors.index)
    for onset in pd.to_datetime(episodes["onset_date"]):
        blocked |= anchors["date"].between(
            onset + pd.DateOffset(months=exclusion_window[0]),
            onset + pd.DateOffset(months=exclusion_window[1]),
        )
    anchors = anchors.loc[~blocked].sort_values("date", ignore_index=True)
    anchors["calendar_month"] = anchors["date"].dt.month
    return anchors


def build_placebo_endpoints(
    adjusted_monthly: pd.DataFrame,
    anchor_dates: pd.Series,
    commodities: list[str],
    *,
    horizon_months: int,
    monthly_return_column: str = "market_adjusted_log_return",
) -> pd.DataFrame:
    required = {"date", "commodity", monthly_return_column}
    if not required.issubset(adjusted_monthly.columns):
        raise ValueError(
            f"Adjusted returns are missing columns: {sorted(required - set(adjusted_monthly))}"
        )
    if horizon_months < 0:
        raise ValueError("Placebo horizon must be non-negative")
    monthly = adjusted_monthly.loc[
        adjusted_monthly["commodity"].isin(commodities),
        ["date", "commodity", monthly_return_column],
    ].copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    if monthly.duplicated(["date", "commodity"]).any():
        raise ValueError("Adjusted monthly date/commodity keys must be unique")

    anchors = pd.DataFrame({"anchor_date": pd.to_datetime(anchor_dates).drop_duplicates()})
    commodity_table = pd.DataFrame({"commodity": sorted(commodities)})
    relative_months = pd.DataFrame({"relative_month": range(horizon_months + 1)})
    grid = anchors.merge(commodity_table, how="cross").merge(relative_months, how="cross")
    grid["date"] = [
        anchor + pd.DateOffset(months=int(relative_month))
        for anchor, relative_month in zip(grid["anchor_date"], grid["relative_month"], strict=True)
    ]
    grid = grid.merge(monthly, on=["date", "commodity"], how="left", validate="many_to_one")
    grouped = grid.groupby(["anchor_date", "commodity"], sort=True, observed=True)
    endpoints = (
        grouped[monthly_return_column]
        .agg(
            valid_months="count",
            cumulative_log_return="sum",
        )
        .reset_index()
    )
    required_months = horizon_months + 1
    complete = endpoints["valid_months"].eq(required_months)
    endpoints["placebo_return"] = (
        endpoints["cumulative_log_return"].map(math.exp).sub(1).where(complete)
    )
    endpoints["placebo_quality_flag"] = pd.Series(pd.NA, index=endpoints.index, dtype="string")
    endpoints.loc[~complete, "placebo_quality_flag"] = "incomplete_adjusted_return_path"
    return endpoints


def draw_calendar_matched_anchors(
    eligible_anchors: pd.DataFrame,
    real_onset_dates: pd.Series,
    *,
    replicates: int,
    seed: int,
    sample_without_replacement: bool = True,
) -> pd.DataFrame:
    required = {"date", "calendar_month"}
    if not required.issubset(eligible_anchors.columns):
        raise ValueError(
            f"Eligible anchors are missing columns: {sorted(required - set(eligible_anchors))}"
        )
    if replicates < 1:
        raise ValueError("Placebo replicates must be positive")
    onsets = pd.to_datetime(real_onset_dates)
    required_counts = onsets.dt.month.value_counts().sort_index()
    pools = {
        int(month): group["date"].sort_values().tolist()
        for month, group in eligible_anchors.groupby("calendar_month", sort=True, observed=True)
    }
    for month, count in required_counts.items():
        available = len(pools.get(int(month), []))
        if sample_without_replacement and available < int(count):
            raise ValueError(
                f"Calendar month {month} has {available} eligible placebo anchors; {count} required"
            )

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        draw_position = 0
        for month, count in required_counts.items():
            sampled = (
                rng.sample(pools[int(month)], k=int(count))
                if sample_without_replacement
                else rng.choices(pools[int(month)], k=int(count))
            )
            for anchor_date in sampled:
                rows.append(
                    {
                        "replicate": replicate,
                        "draw_position": draw_position,
                        "matched_calendar_month": int(month),
                        "anchor_date": anchor_date,
                    }
                )
                draw_position += 1
    return pd.DataFrame.from_records(rows)


def evaluate_placebo_means(
    endpoints: pd.DataFrame,
    draws: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    replicates: int,
    minimum_valid_episodes: int,
    minimum_valid_replicate_share: float,
    confidence_level: float,
    sample_without_replacement: bool = True,
) -> PlaceboOutput:
    endpoint_required = {"anchor_date", "commodity", "placebo_return"}
    draw_required = {"replicate", "anchor_date"}
    observed_required = {"commodity", "mean_return"}
    if not endpoint_required.issubset(endpoints.columns):
        raise ValueError(
            f"Placebo endpoints are missing columns: {sorted(endpoint_required - set(endpoints))}"
        )
    if not draw_required.issubset(draws.columns):
        raise ValueError(f"Placebo draws are missing columns: {sorted(draw_required - set(draws))}")
    if not observed_required.issubset(observed.columns):
        raise ValueError(
            f"Observed results are missing columns: {sorted(observed_required - set(observed))}"
        )
    if sample_without_replacement and draws.duplicated(["replicate", "anchor_date"]).any():
        raise ValueError("A placebo replicate cannot contain duplicate anchors")
    if draws["replicate"].nunique() != replicates:
        raise ValueError("Placebo draw table does not contain the configured replicate count")

    clean = endpoints.loc[endpoints["placebo_return"].notna()].copy()
    values: dict[str, dict[pd.Timestamp, float]] = {}
    for commodity, group in clean.groupby("commodity", sort=True, observed=True):
        values[str(commodity)] = dict(
            zip(
                pd.to_datetime(group["anchor_date"]),
                group["placebo_return"].astype(float),
                strict=True,
            )
        )
    observed_means = dict(
        zip(observed["commodity"].astype(str), observed["mean_return"].astype(float), strict=True)
    )
    commodities = sorted(observed_means)
    rows: list[dict[str, object]] = []
    for replicate, group in draws.groupby("replicate", sort=True, observed=True):
        selected = pd.to_datetime(group["anchor_date"]).tolist()
        for commodity in commodities:
            sample = [
                values.get(commodity, {})[anchor]
                for anchor in selected
                if anchor in values.get(commodity, {})
            ]
            valid = len(sample) >= minimum_valid_episodes
            rows.append(
                {
                    "replicate": int(replicate),
                    "commodity": commodity,
                    "sampled_observations": len(sample),
                    "eligible_replicate": valid,
                    "placebo_mean_return": sum(sample) / len(sample) if valid else float("nan"),
                }
            )
    replicate_table = pd.DataFrame.from_records(rows)
    alpha = 1 - confidence_level
    required_valid_replicates = math.ceil(replicates * minimum_valid_replicate_share)
    results: list[dict[str, object]] = []
    for commodity in commodities:
        distribution = replicate_table.loc[
            replicate_table["commodity"].eq(commodity) & replicate_table["eligible_replicate"],
            "placebo_mean_return",
        ]
        observed_mean = observed_means[commodity]
        valid_replicates = len(distribution)
        status = (
            "estimated"
            if valid_replicates >= required_valid_replicates
            else "insufficient_replicates"
        )
        if valid_replicates:
            lower_tail = (int(distribution.le(observed_mean).sum()) + 1) / (valid_replicates + 1)
            upper_tail = (int(distribution.ge(observed_mean).sum()) + 1) / (valid_replicates + 1)
            p_value = min(1.0, 2 * min(lower_tail, upper_tail))
            center = float(distribution.mean())
            interval_lower = float(distribution.quantile(alpha / 2))
            interval_upper = float(distribution.quantile(1 - alpha / 2))
        else:
            p_value = 1.0
            center = float("nan")
            interval_lower = float("nan")
            interval_upper = float("nan")
        if status != "estimated":
            p_value = 1.0
        results.append(
            {
                "commodity": commodity,
                "observed_mean_return": observed_mean,
                "placebo_mean_return": center,
                "observed_minus_placebo_mean": observed_mean - center,
                "placebo_interval_lower": interval_lower,
                "placebo_interval_upper": interval_upper,
                "valid_placebo_replicates": valid_replicates,
                "required_valid_placebo_replicates": required_valid_replicates,
                "placebo_status": status,
                "placebo_p_value": p_value,
            }
        )
    return PlaceboOutput(pd.DataFrame.from_records(results), replicate_table)
