from __future__ import annotations

from typing import Any

import pandas as pd


def _month_number(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return dates.dt.year * 12 + dates.dt.month


def _construct_episodes(
    index_data: pd.DataFrame,
    *,
    index_name: str,
    threshold: float,
    minimum_duration_months: int,
    observable_delay_after_center_months: int,
    direction: str,
) -> pd.DataFrame:
    required = {"date", index_name}
    if not required.issubset(index_data.columns):
        raise ValueError(f"Index data is missing columns: {sorted(required - set(index_data))}")
    data = index_data.loc[:, ["date", index_name]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date", ignore_index=True)
    if data["date"].duplicated().any():
        raise ValueError("Index dates must be unique")
    if minimum_duration_months < 1:
        raise ValueError("minimum_duration_months must be positive")

    if direction == "warm":
        qualifying = data[index_name].ge(threshold) & data[index_name].notna()
    elif direction == "cold":
        qualifying = data[index_name].le(threshold) & data[index_name].notna()
    else:
        raise ValueError("direction must be 'warm' or 'cold'")
    month_number = _month_number(data["date"])
    run_break = qualifying.ne(qualifying.shift(fill_value=False)) | month_number.diff().ne(1)
    run_id = run_break.cumsum()
    records: list[dict[str, Any]] = []
    for _, run in data.loc[qualifying].groupby(run_id[qualifying], sort=True):
        if len(run) < minimum_duration_months:
            continue
        onset = run.iloc[0]["date"]
        persistence = run.iloc[minimum_duration_months - 1]["date"]
        peak_position = (
            run[index_name].idxmax() if direction == "warm" else run[index_name].idxmin()
        )
        peak = data.loc[peak_position]
        end = run.iloc[-1]["date"]
        records.append(
            {
                "episode_id": f"{index_name}_{direction}_{onset:%Y_%m}",
                "index_name": index_name,
                "direction": direction,
                "threshold": threshold,
                "minimum_duration_months": minimum_duration_months,
                "onset_date": onset,
                "persistence_date": persistence,
                "observable_date": persistence
                + pd.DateOffset(months=observable_delay_after_center_months),
                "observable_delay_after_center_months": observable_delay_after_center_months,
                "end_date": end,
                "duration_months": len(run),
                "peak_date": peak["date"],
                "peak_value": float(peak[index_name]),
                "end_censored": bool(end == data.iloc[-1]["date"]),
            }
        )
    columns = [
        "episode_id",
        "index_name",
        "direction",
        "threshold",
        "minimum_duration_months",
        "onset_date",
        "persistence_date",
        "observable_date",
        "observable_delay_after_center_months",
        "end_date",
        "duration_months",
        "peak_date",
        "peak_value",
        "end_censored",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def construct_warm_episodes(
    index_data: pd.DataFrame,
    *,
    index_name: str,
    threshold: float,
    minimum_duration_months: int,
    observable_delay_after_center_months: int,
) -> pd.DataFrame:
    return _construct_episodes(
        index_data,
        index_name=index_name,
        threshold=threshold,
        minimum_duration_months=minimum_duration_months,
        observable_delay_after_center_months=observable_delay_after_center_months,
        direction="warm",
    )


def construct_cold_episodes(
    index_data: pd.DataFrame,
    *,
    index_name: str,
    threshold: float,
    minimum_duration_months: int,
    observable_delay_after_center_months: int,
) -> pd.DataFrame:
    if threshold >= 0:
        raise ValueError("cold threshold must be negative")
    return _construct_episodes(
        index_data,
        index_name=index_name,
        threshold=threshold,
        minimum_duration_months=minimum_duration_months,
        observable_delay_after_center_months=observable_delay_after_center_months,
        direction="cold",
    )
