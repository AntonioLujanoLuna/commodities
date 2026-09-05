from __future__ import annotations

import math

import pandas as pd


def _month_number(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return dates.dt.year * 12 + dates.dt.month


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "commodity", "value"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Prices are missing columns: {sorted(required - set(prices))}")
    result = prices.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["commodity", "date"], ignore_index=True)
    if result.duplicated(["commodity", "date"]).any():
        raise ValueError("Commodity price keys must be unique")

    grouped = result.groupby("commodity", sort=False, observed=True)
    result["previous_date"] = grouped["date"].shift(1)
    result["previous_value"] = grouped["value"].shift(1)
    current_month = _month_number(result["date"])
    previous_month = _month_number(result["previous_date"])
    consecutive = current_month.sub(previous_month).eq(1)
    valid = consecutive & result["value"].notna() & result["previous_value"].notna()
    ratio = result["value"].div(result["previous_value"])
    result["raw_simple_return"] = ratio.sub(1).where(valid)
    result["raw_log_return"] = ratio.map(math.log, na_action="ignore").where(valid)
    result["return_quality_flag"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[result["value"].isna(), "return_quality_flag"] = "missing_current_price"
    result.loc[result["value"].notna() & result["previous_value"].isna(), "return_quality_flag"] = (
        "missing_previous_price"
    )
    result.loc[
        result["value"].notna() & result["previous_value"].notna() & ~consecutive,
        "return_quality_flag",
    ] = "non_consecutive_month"
    return result


def build_event_return_paths(
    prices: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    first_relative_month: int,
    last_relative_month: int,
    base_relative_month: int,
    anchor_types: tuple[str, ...],
) -> pd.DataFrame:
    if episodes.empty:
        raise ValueError("No qualifying episodes were constructed")
    if not first_relative_month <= base_relative_month <= last_relative_month:
        raise ValueError("base_relative_month must fall inside the event window")
    unsupported = set(anchor_types) - {"retrospective", "observable"}
    if unsupported:
        raise ValueError(f"Unsupported anchor types: {sorted(unsupported)}")

    price_lookup = prices.loc[:, ["date", "commodity", "unit", "value"]].copy()
    price_lookup["date"] = pd.to_datetime(price_lookup["date"])
    if price_lookup.duplicated(["date", "commodity"]).any():
        raise ValueError("Commodity price keys must be unique")
    commodities = price_lookup.loc[:, ["commodity", "unit"]].drop_duplicates()
    if commodities["commodity"].duplicated().any():
        raise ValueError("Each commodity must have a single unit")

    anchors: list[pd.DataFrame] = []
    mapping = {"retrospective": "onset_date", "observable": "observable_date"}
    for anchor_type in anchor_types:
        anchor = episodes.loc[:, ["episode_id", mapping[anchor_type]]].copy()
        anchor = anchor.rename(columns={mapping[anchor_type]: "anchor_date"})
        anchor["anchor_type"] = anchor_type
        anchors.append(anchor)
    anchor_table = pd.concat(anchors, ignore_index=True)

    relative_months = pd.DataFrame(
        {"relative_month": range(first_relative_month, last_relative_month + 1)}
    )
    grid = anchor_table.merge(commodities, how="cross").merge(relative_months, how="cross")
    grid["date"] = [
        anchor + pd.DateOffset(months=int(relative_month))
        for anchor, relative_month in zip(grid["anchor_date"], grid["relative_month"], strict=True)
    ]
    grid = grid.merge(
        price_lookup.rename(columns={"value": "event_value"}).drop(columns="unit"),
        on=["date", "commodity"],
        how="left",
        validate="many_to_one",
    )
    bases = (
        grid.loc[
            grid["relative_month"].eq(base_relative_month),
            ["episode_id", "anchor_type", "commodity", "event_value"],
        ]
        .rename(columns={"event_value": "base_value"})
        .copy()
    )
    grid = grid.merge(
        bases,
        on=["episode_id", "anchor_type", "commodity"],
        how="left",
        validate="many_to_one",
    )
    valid = grid["event_value"].notna() & grid["base_value"].notna()
    ratio = grid["event_value"].div(grid["base_value"])
    grid["raw_cumulative_return"] = ratio.sub(1).where(valid)
    grid["raw_log_cumulative_return"] = ratio.map(math.log, na_action="ignore").where(valid)
    grid["event_quality_flag"] = pd.Series(pd.NA, index=grid.index, dtype="string")
    grid.loc[grid["base_value"].isna(), "event_quality_flag"] = "missing_base_price"
    grid.loc[grid["base_value"].notna() & grid["event_value"].isna(), "event_quality_flag"] = (
        "missing_event_price"
    )
    return grid.sort_values(
        ["episode_id", "anchor_type", "commodity", "relative_month"], ignore_index=True
    )
