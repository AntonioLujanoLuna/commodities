from __future__ import annotations

import math

import pandas as pd


def build_non_event_mask(
    dates: pd.Series,
    episodes: pd.DataFrame,
    *,
    first_relative_month: int,
    last_relative_month: int,
) -> pd.Series:
    parsed_dates = pd.to_datetime(dates)
    excluded = pd.Series(False, index=dates.index)
    for onset in pd.to_datetime(episodes["onset_date"]):
        start = onset + pd.DateOffset(months=first_relative_month)
        end = onset + pd.DateOffset(months=last_relative_month)
        excluded |= parsed_dates.between(start, end)
    return ~excluded


def build_outside_episode_core_mask(dates: pd.Series, episodes: pd.DataFrame) -> pd.Series:
    parsed_dates = pd.to_datetime(dates)
    excluded = pd.Series(False, index=dates.index)
    for _, episode in episodes.iterrows():
        excluded |= parsed_dates.between(
            pd.Timestamp(episode["onset_date"]), pd.Timestamp(episode["end_date"])
        )
    return ~excluded


def compute_market_factor(index_data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "world_bank_total_index"}
    if not required.issubset(index_data.columns):
        raise ValueError(f"Market index is missing columns: {sorted(required - set(index_data))}")
    factor = index_data.loc[:, ["date", "world_bank_total_index"]].copy()
    factor["date"] = pd.to_datetime(factor["date"])
    factor = factor.sort_values("date", ignore_index=True)
    if factor["date"].duplicated().any():
        raise ValueError("Market-index dates must be unique")
    prior_date = factor["date"].shift(1)
    month_number = factor["date"].dt.year * 12 + factor["date"].dt.month
    prior_month_number = prior_date.dt.year * 12 + prior_date.dt.month
    valid = (
        month_number.sub(prior_month_number).eq(1)
        & factor["world_bank_total_index"].notna()
        & factor["world_bank_total_index"].shift(1).notna()
    )
    ratio = factor["world_bank_total_index"].div(factor["world_bank_total_index"].shift(1))
    factor["market_log_return"] = ratio.map(math.log, na_action="ignore").where(valid)
    return factor


def adjust_monthly_returns(
    monthly_returns: pd.DataFrame,
    market_index: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    market_exclusion_window: tuple[int, int],
    minimum_seasonal_observations: int,
    minimum_market_model_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor = compute_market_factor(market_index)
    factor["month_of_year"] = factor["date"].dt.month
    factor["is_seasonal_estimation_month"] = build_outside_episode_core_mask(
        factor["date"], episodes
    )
    factor["is_market_estimation_month"] = build_non_event_mask(
        factor["date"],
        episodes,
        first_relative_month=market_exclusion_window[0],
        last_relative_month=market_exclusion_window[1],
    )
    factor_estimation = factor.loc[
        factor["is_seasonal_estimation_month"] & factor["market_log_return"].notna()
    ]
    factor_seasonality = (
        factor_estimation.groupby("month_of_year", observed=True)["market_log_return"]
        .agg(market_seasonal_mean_log_return="mean", market_seasonal_observations="count")
        .reset_index()
    )
    factor_seasonality.loc[
        factor_seasonality["market_seasonal_observations"].lt(minimum_seasonal_observations),
        "market_seasonal_mean_log_return",
    ] = pd.NA
    factor = factor.merge(
        factor_seasonality, on="month_of_year", how="left", validate="many_to_one"
    )
    factor["market_seasonal_adjusted_log_return"] = factor["market_log_return"].sub(
        factor["market_seasonal_mean_log_return"]
    )

    result = monthly_returns.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["month_of_year"] = result["date"].dt.month
    result = result.merge(
        factor.loc[
            :,
            [
                "date",
                "world_bank_total_index",
                "market_log_return",
                "market_seasonal_mean_log_return",
                "market_seasonal_observations",
                "market_seasonal_adjusted_log_return",
                "is_seasonal_estimation_month",
                "is_market_estimation_month",
            ],
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )
    estimation = result.loc[
        result["is_seasonal_estimation_month"].fillna(False) & result["raw_log_return"].notna()
    ]
    seasonality = (
        estimation.groupby(["commodity", "month_of_year"], observed=True)["raw_log_return"]
        .agg(seasonal_mean_log_return="mean", seasonal_observations="count")
        .reset_index()
    )
    seasonality.loc[
        seasonality["seasonal_observations"].lt(minimum_seasonal_observations),
        "seasonal_mean_log_return",
    ] = pd.NA
    result = result.merge(
        seasonality,
        on=["commodity", "month_of_year"],
        how="left",
        validate="many_to_one",
    )
    result["seasonal_adjusted_log_return"] = result["raw_log_return"].sub(
        result["seasonal_mean_log_return"]
    )
    result["seasonal_adjusted_simple_return"] = result["seasonal_adjusted_log_return"].map(
        math.expm1, na_action="ignore"
    )

    model_records: list[dict[str, object]] = []
    for commodity, group in result.groupby("commodity", sort=True, observed=True):
        sample = group.loc[
            group["is_market_estimation_month"].fillna(False)
            & group["seasonal_adjusted_log_return"].notna()
            & group["market_seasonal_adjusted_log_return"].notna()
        ]
        record: dict[str, object] = {
            "commodity": commodity,
            "market_model_observations": len(sample),
            "market_alpha": pd.NA,
            "market_beta": pd.NA,
            "market_model_r_squared": pd.NA,
            "market_model_status": "insufficient_observations",
        }
        if len(sample) >= minimum_market_model_observations:
            x = sample["market_seasonal_adjusted_log_return"]
            y = sample["seasonal_adjusted_log_return"]
            x_centered = x - x.mean()
            denominator = x_centered.pow(2).sum()
            if denominator > 0:
                beta = x_centered.mul(y - y.mean()).sum() / denominator
                alpha = y.mean() - beta * x.mean()
                residual = y - (alpha + beta * x)
                total_sum_squares = (y - y.mean()).pow(2).sum()
                r_squared = (
                    1 - residual.pow(2).sum() / total_sum_squares
                    if total_sum_squares > 0
                    else pd.NA
                )
                record.update(
                    {
                        "market_alpha": float(alpha),
                        "market_beta": float(beta),
                        "market_model_r_squared": float(r_squared),
                        "market_model_status": "estimated",
                    }
                )
        model_records.append(record)
    models = pd.DataFrame.from_records(model_records)
    result = result.merge(models, on="commodity", how="left", validate="many_to_one")
    predicted = (
        result["market_alpha"]
        + result["market_beta"] * result["market_seasonal_adjusted_log_return"]
    )
    result["market_adjusted_log_return"] = result["seasonal_adjusted_log_return"].sub(predicted)
    result["market_adjusted_simple_return"] = result["market_adjusted_log_return"].map(
        math.expm1, na_action="ignore"
    )
    return result, seasonality, models, factor


def add_adjusted_event_paths(
    event_paths: pd.DataFrame,
    adjusted_monthly_returns: pd.DataFrame,
    *,
    base_relative_month: int,
    monthly_columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_columns = monthly_columns or [
        "seasonal_adjusted_log_return",
        "market_adjusted_log_return",
    ]
    monthly = adjusted_monthly_returns.loc[:, ["date", "commodity", *selected_columns]]
    result = event_paths.merge(
        monthly,
        on=["date", "commodity"],
        how="left",
        validate="many_to_one",
    ).sort_values(["episode_id", "anchor_type", "commodity", "relative_month"], ignore_index=True)
    group_keys = ["episode_id", "anchor_type", "commodity"]
    for source_column in selected_columns:
        prefix = source_column.removesuffix("_log_return")
        cumulative_column = f"{prefix}_cumulative_log_return"
        output = pd.Series(float("nan"), index=result.index)
        for _, positions in result.groupby(group_keys, sort=False, observed=True).groups.items():
            ordered_positions = list(positions)
            group = result.loc[ordered_positions]
            relative_months = group["relative_month"].tolist()
            if base_relative_month not in relative_months or group["base_value"].isna().all():
                continue
            base_position = relative_months.index(base_relative_month)
            output.iloc[ordered_positions[base_position]] = 0.0

            running = 0.0
            valid = True
            for offset in range(base_position + 1, len(ordered_positions)):
                value = group.iloc[offset][source_column]
                if pd.isna(value):
                    valid = False
                if valid:
                    running += float(value)
                    output.iloc[ordered_positions[offset]] = running

            running = 0.0
            valid = True
            for offset in range(base_position - 1, -1, -1):
                value = group.iloc[offset + 1][source_column]
                if pd.isna(value):
                    valid = False
                if valid:
                    running -= float(value)
                    output.iloc[ordered_positions[offset]] = running
        result[cumulative_column] = output
        result[f"{prefix}_cumulative_return"] = result[cumulative_column].map(
            math.expm1, na_action="ignore"
        )
    return result
