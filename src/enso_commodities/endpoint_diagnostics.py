from __future__ import annotations

import math

import pandas as pd
from scipy.stats import spearmanr


def add_endpoint_shape(endpoints: pd.DataFrame) -> pd.DataFrame:
    required = {"index_definition", "anchor_type", "direction", "commodity", "anchor_date", "value"}
    if not required.issubset(endpoints.columns):
        raise ValueError(f"Endpoint data is missing columns: {sorted(required - set(endpoints))}")
    output = endpoints.copy()
    if output["value"].dropna().le(-1).any():
        raise ValueError("Simple cumulative returns must exceed -1 for log conversion")
    output["log_value"] = output["value"].map(math.log1p, na_action="ignore")
    output["anchor_date"] = pd.to_datetime(output["anchor_date"])
    return output


def endpoint_shape_summary(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["index_definition", "anchor_type", "direction", "commodity"]
    for key, group in endpoints.groupby(keys, sort=True, observed=True):
        clean = group.dropna(subset=["value", "log_value"])
        arithmetic = float(clean["value"].mean())
        mean_log = float(clean["log_value"].mean())
        geometric = math.expm1(mean_log)
        ordinal_dates = clean["anchor_date"].astype("int64").astype(float)
        rho, p_value = spearmanr(ordinal_dates, clean["log_value"])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "episodes": len(clean),
                "arithmetic_mean_return": arithmetic,
                "mean_log_return": mean_log,
                "geometric_equivalent_return": geometric,
                "jensen_gap": arithmetic - geometric,
                "jensen_share_of_arithmetic_mean": (
                    (arithmetic - geometric) / arithmetic if arithmetic != 0 else float("nan")
                ),
                "time_spearman_rho": float(rho),
                "time_spearman_p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def strict_factor_windows(
    anchors: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    horizon_months: int,
) -> pd.DataFrame:
    keys = ["index_definition", "episode_id", "anchor_type", "direction", "anchor_date"]
    unique_anchors = anchors.loc[:, keys].drop_duplicates()
    factor_monthly = monthly.loc[:, ["index_definition", "date", *factor_columns]].drop_duplicates()
    if factor_monthly.duplicated(["index_definition", "date"]).any():
        raise ValueError("Factor values differ across commodity rows")
    factor_monthly["date"] = pd.to_datetime(factor_monthly["date"])
    lookup = factor_monthly.set_index(["index_definition", "date"])
    rows: list[dict[str, object]] = []
    for anchor in unique_anchors.itertuples(index=False):
        dates = [pd.Timestamp(anchor.anchor_date) + pd.DateOffset(months=month) for month in range(horizon_months + 1)]
        record = dict(zip(keys, anchor, strict=True))
        for factor in factor_columns:
            values = [lookup[factor].get((anchor.index_definition, date), float("nan")) for date in dates]
            record[f"cumulative_{factor}"] = float(sum(values)) if pd.Series(values).notna().all() else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def regime_correlations(
    endpoints: pd.DataFrame, factor_windows: pd.DataFrame, *, factor_columns: tuple[str, ...]
) -> pd.DataFrame:
    keys = ["index_definition", "episode_id", "anchor_type", "direction", "anchor_date"]
    merged = endpoints.merge(factor_windows, on=keys, how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for key, group in merged.groupby(
        ["index_definition", "anchor_type", "direction", "commodity"], sort=True, observed=True
    ):
        for factor in factor_columns:
            column = f"cumulative_{factor}"
            clean = group.dropna(subset=["log_value", column])
            rho, p_value = spearmanr(clean["log_value"], clean[column])
            rows.append(
                {
                    "index_definition": key[0],
                    "anchor_type": key[1],
                    "direction": key[2],
                    "commodity": key[3],
                    "factor": factor,
                    "episodes": len(clean),
                    "spearman_rho": float(rho),
                    "spearman_p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)
