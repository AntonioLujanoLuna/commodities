from __future__ import annotations

import math

import pandas as pd
import statsmodels.api as sm

from .adjustments import build_non_event_mask

COEFFICIENT_COLUMNS = {
    "market_seasonal_adjusted_log_return": "macro_beta_market",
    "us_neer_log_change": "macro_beta_us_neer",
    "us_cpi_log_change": "macro_beta_us_cpi",
    "global_real_activity_change": "macro_beta_global_activity",
}


def fit_macro_adjusted_returns(
    adjusted_monthly: pd.DataFrame,
    macro_controls: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    control_columns: tuple[str, ...],
    estimation_exclusion_window: tuple[int, int],
    minimum_model_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_required = {
        "date",
        "commodity",
        "seasonal_adjusted_log_return",
        "market_seasonal_adjusted_log_return",
    }
    macro_required = {"date", *(set(control_columns) - {"market_seasonal_adjusted_log_return"})}
    if not monthly_required.issubset(adjusted_monthly.columns):
        raise ValueError(
            f"Adjusted monthly data is missing columns: {sorted(monthly_required - set(adjusted_monthly))}"
        )
    if not macro_required.issubset(macro_controls.columns):
        raise ValueError(
            f"Macro data is missing columns: {sorted(macro_required - set(macro_controls))}"
        )
    if tuple(control_columns) != tuple(COEFFICIENT_COLUMNS):
        raise ValueError("Macro controls do not match the frozen coefficient specification")

    macro = macro_controls.loc[:, sorted(macro_required)].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    if macro["date"].duplicated().any():
        raise ValueError("Macro-control dates must be unique")
    result = adjusted_monthly.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(macro, on="date", how="left", validate="many_to_one")
    result["is_macro_estimation_month"] = build_non_event_mask(
        result["date"],
        episodes,
        first_relative_month=estimation_exclusion_window[0],
        last_relative_month=estimation_exclusion_window[1],
    )

    records: list[dict[str, object]] = []
    for commodity, group in result.groupby("commodity", sort=True, observed=True):
        complete = group.loc[
            group["is_macro_estimation_month"]
            & group["seasonal_adjusted_log_return"].notna()
            & group.loc[:, list(control_columns)].notna().all(axis=1)
        ]
        record: dict[str, object] = {
            "commodity": commodity,
            "macro_model_observations": len(complete),
            "macro_alpha": pd.NA,
            **dict.fromkeys(COEFFICIENT_COLUMNS.values(), pd.NA),
            "macro_model_r_squared": pd.NA,
            "macro_model_condition_number": pd.NA,
            "macro_model_status": "insufficient_observations",
        }
        if len(complete) >= minimum_model_observations:
            design = sm.add_constant(complete.loc[:, list(control_columns)], has_constant="add")
            fitted = sm.OLS(complete["seasonal_adjusted_log_return"], design).fit()
            if int(fitted.df_model) == len(control_columns):
                record.update(
                    {
                        "macro_alpha": float(fitted.params["const"]),
                        **{
                            output: float(fitted.params[source])
                            for source, output in COEFFICIENT_COLUMNS.items()
                        },
                        "macro_model_r_squared": float(fitted.rsquared),
                        "macro_model_condition_number": float(fitted.condition_number),
                        "macro_model_status": "estimated",
                    }
                )
            else:
                record["macro_model_status"] = "rank_deficient"
        records.append(record)
    models = pd.DataFrame.from_records(records)
    result = result.merge(models, on="commodity", how="left", validate="many_to_one")

    complete_controls = result.loc[:, list(control_columns)].notna().all(axis=1)
    estimated = result["macro_model_status"].eq("estimated") & complete_controls
    predicted = pd.Series(float("nan"), index=result.index)
    predicted.loc[estimated] = result.loc[estimated, "macro_alpha"].astype(float)
    for source, coefficient in COEFFICIENT_COLUMNS.items():
        predicted.loc[estimated] += result.loc[estimated, coefficient].astype(float) * result.loc[
            estimated, source
        ].astype(float)
    result["macro_predicted_log_return"] = predicted
    result["macro_adjusted_log_return"] = result["seasonal_adjusted_log_return"].sub(predicted)
    result["macro_adjusted_simple_return"] = result["macro_adjusted_log_return"].map(
        math.expm1, na_action="ignore"
    )
    return result, models
