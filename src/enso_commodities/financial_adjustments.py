from __future__ import annotations

import re

import pandas as pd
import statsmodels.api as sm

from .adjustments import build_non_event_mask


def fit_financial_adjusted_returns(
    monthly: pd.DataFrame,
    external_controls: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    control_columns: tuple[str, ...],
    exclusion_window: tuple[int, int],
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"date", "commodity", "seasonal_adjusted_log_return", "market_seasonal_adjusted_log_return"}
    if not required.issubset(monthly.columns):
        raise ValueError(f"Monthly data is missing columns: {sorted(required - set(monthly))}")
    external_names = tuple(column for column in control_columns if column != "market_seasonal_adjusted_log_return")
    if not {"date", *external_names}.issubset(external_controls.columns):
        raise ValueError("Financial controls do not contain the requested specification")
    controls = external_controls.loc[:, ["date", *external_names]].copy()
    controls["date"] = pd.to_datetime(controls["date"])
    if controls["date"].duplicated().any():
        raise ValueError("Financial-control dates must be unique")
    base_columns = [column for column in monthly.columns if column not in external_names]
    result = monthly.loc[:, base_columns].copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(controls, on="date", how="left", validate="many_to_one")
    result["is_financial_estimation_month"] = build_non_event_mask(
        result["date"], episodes, first_relative_month=exclusion_window[0], last_relative_month=exclusion_window[1]
    )
    coefficient_names = {column: f"financial_beta_{re.sub('[^a-z0-9]+', '_', column.lower())}" for column in control_columns}
    records: list[dict[str, object]] = []
    for commodity, group in result.groupby("commodity", sort=True, observed=True):
        sample = group.loc[
            group["is_financial_estimation_month"]
            & group["seasonal_adjusted_log_return"].notna()
            & group.loc[:, list(control_columns)].notna().all(axis=1)
        ]
        record: dict[str, object] = {
            "commodity": commodity,
            "financial_model_observations": len(sample),
            "financial_alpha": pd.NA,
            **dict.fromkeys(coefficient_names.values(), pd.NA),
            "financial_model_r_squared": pd.NA,
            "financial_model_condition_number": pd.NA,
            "financial_model_status": "insufficient_observations",
        }
        if len(sample) >= minimum_observations:
            design = sm.add_constant(sample.loc[:, list(control_columns)], has_constant="add")
            fitted = sm.OLS(sample["seasonal_adjusted_log_return"], design).fit()
            if int(fitted.df_model) == len(control_columns):
                record.update(
                    {
                        "financial_alpha": float(fitted.params["const"]),
                        **{name: float(fitted.params[column]) for column, name in coefficient_names.items()},
                        "financial_model_r_squared": float(fitted.rsquared),
                        "financial_model_condition_number": float(fitted.condition_number),
                        "financial_model_status": "estimated",
                    }
                )
            else:
                record["financial_model_status"] = "rank_deficient"
        records.append(record)
    models = pd.DataFrame(records)
    result = result.merge(models, on="commodity", how="left", validate="many_to_one")
    complete = result.loc[:, list(control_columns)].notna().all(axis=1)
    estimated = result["financial_model_status"].eq("estimated") & complete
    predicted = pd.Series(float("nan"), index=result.index)
    predicted.loc[estimated] = result.loc[estimated, "financial_alpha"].astype(float)
    for column, coefficient in coefficient_names.items():
        predicted.loc[estimated] += result.loc[estimated, coefficient].astype(float) * result.loc[estimated, column].astype(float)
    result["financial_predicted_log_return"] = predicted
    result["financial_adjusted_log_return"] = result["seasonal_adjusted_log_return"].sub(predicted)
    return result, models
