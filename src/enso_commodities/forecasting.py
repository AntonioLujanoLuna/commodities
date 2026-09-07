"""Leakage-safe expanding-window forecast comparison and price-index strategy proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .statistics import salted_seed


@dataclass(frozen=True)
class ForecastSummary:
    observations: int
    baseline_rmse: float
    enso_rmse: float
    rmse_improvement: float
    baseline_mae: float
    enso_mae: float
    loss_difference_p_value: float
    gross_strategy_return: float
    net_strategy_return: float
    buy_and_hold_return: float
    net_excess_over_buy_and_hold: float
    turnover: float


def _design(frame: pd.DataFrame, *, include_enso: bool) -> pd.DataFrame:
    numeric = frame.loc[:, ["return_lag_1", "return_lag_12"]].astype("float64")
    months = pd.get_dummies(frame["date"].dt.month, prefix="month", dtype="float64")
    result = pd.concat([numeric.reset_index(drop=True), months.reset_index(drop=True)], axis=1)
    if include_enso:
        result = pd.concat(
            [
                result,
                frame.loc[:, ["roni", "roni_lag_3", "roni_lag_6"]]
                .astype("float64")
                .reset_index(drop=True),
            ],
            axis=1,
        )
    result.insert(0, "constant", 1.0)
    return result


def _predict_ols(train: pd.DataFrame, test: pd.DataFrame, *, include_enso: bool) -> float:
    train_x = _design(train, include_enso=include_enso)
    test_x = _design(test, include_enso=include_enso)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0.0)
    coefficients = np.linalg.lstsq(
        train_x.to_numpy(), train["target_return"].to_numpy(dtype="float64"), rcond=None
    )[0]
    return float(test_x.to_numpy()[0] @ coefficients)


def prepare_forecast_frame(
    monthly: pd.DataFrame,
    *,
    horizon_months: int,
    outcome_column: str = "market_adjusted_log_return",
) -> pd.DataFrame:
    required = {"date", outcome_column, "roni"}
    if not required.issubset(monthly.columns):
        raise ValueError(f"Forecast input is missing columns: {sorted(required - set(monthly))}")
    if horizon_months < 1:
        raise ValueError("horizon_months must be positive")
    data = monthly.loc[:, ["date", outcome_column, "roni"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date", ignore_index=True)
    if data["date"].duplicated().any():
        raise ValueError("Forecast dates must be unique within a commodity")
    returns = pd.to_numeric(data[outcome_column], errors="coerce")
    data["return_lag_1"] = returns.shift(1)
    data["return_lag_12"] = returns.shift(12)
    data["roni_lag_3"] = pd.to_numeric(data["roni"], errors="coerce").shift(3)
    data["roni_lag_6"] = pd.to_numeric(data["roni"], errors="coerce").shift(6)
    # At origin t, the target is the cumulative log return over t+1,...,t+h.
    data["target_return"] = sum(returns.shift(-step) for step in range(1, horizon_months + 1))
    data["target_end_date"] = data["date"].shift(-horizon_months)
    return data.dropna(
        subset=[
            "return_lag_1",
            "return_lag_12",
            "roni",
            "roni_lag_3",
            "roni_lag_6",
            "target_return",
            "target_end_date",
        ]
    ).reset_index(drop=True)


def expanding_window_forecasts(
    data: pd.DataFrame,
    *,
    minimum_training_months: int,
    origin_frequency_months: int,
) -> pd.DataFrame:
    if minimum_training_months < 24:
        raise ValueError("minimum_training_months must be at least 24")
    if origin_frequency_months < 1:
        raise ValueError("origin_frequency_months must be positive")
    if len(data) <= minimum_training_months:
        raise ValueError("Forecast input does not exceed the minimum training window")
    rows: list[dict[str, object]] = []
    first_origin_position: int | None = None
    for position in range(len(data)):
        test = data.iloc[[position]].copy()
        origin_date = pd.Timestamp(test.iloc[0]["date"])
        train = data.loc[
            (data.index < position)
            & pd.to_datetime(data["target_end_date"]).le(origin_date)
        ].copy()
        if len(train) < minimum_training_months:
            continue
        if first_origin_position is None:
            first_origin_position = position
        if (position - first_origin_position) % origin_frequency_months:
            continue
        rows.append(
            {
                "origin_date": test.iloc[0]["date"],
                "target_end_date": test.iloc[0]["target_end_date"],
                "training_observations": len(train),
                "latest_training_target_end_date": train["target_end_date"].max(),
                "actual_return": float(test.iloc[0]["target_return"]),
                "baseline_prediction": _predict_ols(train, test, include_enso=False),
                "enso_prediction": _predict_ols(train, test, include_enso=True),
            }
        )
    return pd.DataFrame.from_records(rows)


def _paired_year_block_p_value(loss_difference: pd.Series, years: pd.Series, *, seed: int) -> float:
    blocks = pd.DataFrame({"loss": loss_difference, "year": years}).groupby("year")["loss"].sum()
    if len(blocks) < 3:
        return float("nan")
    observed = float(blocks.sum())
    generator = np.random.default_rng(salted_seed(seed, "forecast-loss-block-signs"))
    signs = generator.choice((-1.0, 1.0), size=(9999, len(blocks)))
    null = signs @ blocks.to_numpy(dtype="float64")
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1))


def summarize_forecasts(
    forecasts: pd.DataFrame, *, transaction_cost_bps: float, seed: int
) -> ForecastSummary:
    required = {"origin_date", "actual_return", "baseline_prediction", "enso_prediction"}
    if not required.issubset(forecasts.columns):
        raise ValueError(f"Forecast results are missing columns: {sorted(required - set(forecasts))}")
    if forecasts.empty:
        raise ValueError("Forecast results are empty")
    actual = forecasts["actual_return"].to_numpy(dtype="float64")
    baseline_error = actual - forecasts["baseline_prediction"].to_numpy(dtype="float64")
    enso_error = actual - forecasts["enso_prediction"].to_numpy(dtype="float64")
    baseline_loss = baseline_error**2
    enso_loss = enso_error**2
    positions = np.sign(forecasts["enso_prediction"].to_numpy(dtype="float64"))
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    gross = positions * actual
    costs = turnover * transaction_cost_bps / 10_000
    return ForecastSummary(
        observations=len(forecasts),
        baseline_rmse=float(np.sqrt(np.mean(baseline_loss))),
        enso_rmse=float(np.sqrt(np.mean(enso_loss))),
        rmse_improvement=float(np.sqrt(np.mean(baseline_loss)) - np.sqrt(np.mean(enso_loss))),
        baseline_mae=float(np.mean(np.abs(baseline_error))),
        enso_mae=float(np.mean(np.abs(enso_error))),
        loss_difference_p_value=_paired_year_block_p_value(
            baseline_loss - enso_loss,
            pd.to_datetime(forecasts["origin_date"]).dt.year,
            seed=seed,
        ),
        gross_strategy_return=float(gross.sum()),
        net_strategy_return=float((gross - costs).sum()),
        buy_and_hold_return=float(actual.sum()),
        net_excess_over_buy_and_hold=float((gross - costs).sum() - actual.sum()),
        turnover=float(turnover.sum()),
    )
