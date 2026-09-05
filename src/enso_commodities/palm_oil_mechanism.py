from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm
import yaml

from .config import project_root
from .palm_oil_data import latest_palm_snapshot
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import benjamini_hochberg


def _fit_link(
    data: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    covariates: tuple[str, ...] = (),
    hac_lags: int,
) -> dict[str, object]:
    columns = [outcome, exposure, *covariates]
    sample = data.dropna(subset=columns)
    design = sm.add_constant(sample.loc[:, [exposure, *covariates]], has_constant="add")
    fitted = sm.OLS(sample[outcome], design).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {
        "observations": len(sample),
        "coefficient": float(fitted.params[exposure]),
        "standard_error": float(fitted.bse[exposure]),
        "p_value": float(fitted.pvalues[exposure]),
        "r_squared": float(fitted.rsquared),
    }


def _direction(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def run_palm_oil_mechanism(
    processed_snapshot: Path | None = None,
    palm_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    mechanism_config_path: Path | None = None,
) -> Path:
    root = project_root()
    snapshot = processed_snapshot or latest_processed_snapshot()
    palm_input = palm_snapshot or latest_palm_snapshot(root / "data" / "mechanisms" / "palm_oil" / "processed")
    output_dir = (tables_root or root / "tables") / snapshot.name
    config_path = mechanism_config_path or root / "config" / "palm_oil_mechanism.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    with (palm_input / "summary.json").open(encoding="utf-8") as handle:
        input_receipt: dict[str, Any] = json.load(handle)
    if input_receipt.get("data_provenance") != "real":
        raise ValueError("Palm-oil mechanism analysis requires real inputs")
    verify_hashes(palm_input, input_receipt["output_hashes"])
    if input_receipt.get("input_hashes", {}).get("palm_oil_mechanism.yaml") != sha256_file(config_path):
        raise ValueError("Palm-oil dataset was not built with the current mechanism contract")

    weather = pd.read_csv(palm_input / "palm_weather_monthly.csv", parse_dates=["date"])
    production = pd.read_csv(palm_input / "faostat_palm_oil_annual.csv")
    enso = pd.read_csv(snapshot / "enso_monthly.csv", parse_dates=["date"])
    prices = pd.read_parquet(snapshot / "commodity_prices_monthly.parquet")
    baseline_start, baseline_end = (int(value) for value in config["weather_baseline_years"])
    weather["month"] = weather["date"].dt.month
    baseline = (
        weather.loc[weather["date"].dt.year.between(baseline_start, baseline_end)]
        .groupby(["point", "parameter", "month"], observed=True)["value"]
        .mean()
        .rename("climatology")
        .reset_index()
    )
    anomalies = weather.merge(baseline, on=["point", "parameter", "month"], validate="many_to_one")
    anomalies["anomaly"] = anomalies["value"] - anomalies["climatology"]
    country_weather = (
        anomalies.groupby(["date", "country", "parameter"], observed=True)["anomaly"]
        .mean()
        .reset_index()
    )
    fruit_production = production.loc[
        production["item"].eq("Oil palm fruit") & production["element"].eq("Production"),
        ["country", "year", "value"],
    ].rename(columns={"value": "fruit_production_tonnes"})
    fruit_production["weight_year"] = fruit_production["year"] + 1
    fruit_production["lagged_share"] = fruit_production.groupby("weight_year")["fruit_production_tonnes"].transform(lambda values: values / values.sum())
    country_weather["weight_year"] = country_weather["date"].dt.year
    weighted = country_weather.merge(
        fruit_production[["country", "weight_year", "lagged_share"]],
        on=["country", "weight_year"],
        how="left",
        validate="many_to_one",
    )
    weighted["weighted_anomaly"] = weighted["anomaly"] * weighted["lagged_share"]
    regional = weighted.groupby(["date", "parameter"], observed=True)["weighted_anomaly"].sum(min_count=2).unstack("parameter").reset_index()
    regional = regional.rename(columns={"PRECTOTCORR": "precipitation_anomaly_mm", "T2M": "temperature_anomaly_c"})
    regional = regional.merge(enso[["date", str(config["enso_index"])]], on="date", how="left", validate="one_to_one")

    results: list[dict[str, object]] = []
    signs = config["expected_signs"]
    for outcome, link, expected in [
        ("precipitation_anomaly_mm", "enso_to_precipitation", signs["enso_to_precipitation"]),
        ("temperature_anomaly_c", "enso_to_temperature", signs["enso_to_temperature"]),
    ]:
        for lag in config["weather_lags_months"]:
            exposure = f"roni_lag_{lag}m"
            regional[exposure] = regional[str(config["enso_index"])].shift(int(lag))
            record = _fit_link(regional, outcome=outcome, exposure=exposure, hac_lags=12)
            results.append({"link": link, "outcome": outcome, "exposure": exposure, "lag": int(lag), "expected_direction": expected, **record})

    country_annual = country_weather.assign(year=country_weather["date"].dt.year).groupby(
        ["country", "year", "parameter"], observed=True
    )["anomaly"].agg("sum").unstack("parameter").reset_index()
    temperature_annual = country_weather.loc[country_weather["parameter"].eq("T2M")].assign(year=lambda frame: frame["date"].dt.year).groupby(["country", "year"], observed=True)["anomaly"].mean()
    country_annual = country_annual.rename(columns={"PRECTOTCORR": "annual_precipitation_anomaly_mm"}).drop(columns=["T2M"]).merge(
        temperature_annual.rename("annual_temperature_anomaly_c"), on=["country", "year"], validate="one_to_one"
    )
    yield_data = production.loc[
        production["item"].eq("Oil palm fruit") & production["element"].eq("Yield"),
        ["country", "year", "value"],
    ].rename(columns={"value": "yield_kg_per_ha"})
    panel = country_annual.merge(yield_data, on=["country", "year"], validate="one_to_one")
    panel["log_yield"] = panel["yield_kg_per_ha"].map(math.log)
    panel["year_centered"] = panel["year"] - panel["year"].min()
    panel["malaysia"] = panel["country"].eq("Malaysia").astype(int)
    for weather_column, link, expected in [
        ("annual_precipitation_anomaly_mm", "precipitation_to_yield", signs["precipitation_to_yield"]),
        ("annual_temperature_anomaly_c", "temperature_to_yield", signs["temperature_to_yield"]),
    ]:
        for lag in config["production_response_lags_years"]:
            exposure = f"{weather_column}_lag_{lag}y"
            panel[exposure] = panel.groupby("country")[weather_column].shift(int(lag))
            record = _fit_link(panel, outcome="log_yield", exposure=exposure, covariates=("malaysia", "year_centered"), hac_lags=2)
            results.append({"link": link, "outcome": "log_yield", "exposure": exposure, "lag": int(lag), "expected_direction": expected, **record})

    palm_production = production.loc[
        production["item"].eq("Palm oil") & production["element"].eq("Production")
    ].groupby("year", observed=True)["value"].sum().rename("palm_oil_production_tonnes").reset_index()
    palm_production["production_log_growth"] = palm_production["palm_oil_production_tonnes"].map(math.log).diff()
    palm_prices = prices.loc[prices["commodity"].eq("Palm oil")].copy()
    palm_prices["year"] = pd.to_datetime(palm_prices["date"]).dt.year
    annual_price = palm_prices.groupby("year", observed=True)["value"].mean().rename("annual_price").reset_index()
    annual_price["price_log_growth"] = annual_price["annual_price"].map(math.log).diff()
    price_panel = palm_production.merge(annual_price, on="year", how="inner", validate="one_to_one")
    for lag in config["price_response_lags_years"]:
        exposure = f"production_log_growth_lag_{lag}y"
        price_panel[exposure] = price_panel["production_log_growth"].shift(int(lag))
        record = _fit_link(price_panel, outcome="price_log_growth", exposure=exposure, hac_lags=2)
        results.append({"link": "production_to_price", "outcome": "price_log_growth", "exposure": exposure, "lag": int(lag), "expected_direction": signs["production_to_price"], **record})

    result_table = pd.DataFrame(results)
    result_table["observed_direction"] = result_table["coefficient"].map(_direction)
    result_table["sign_matches"] = result_table["observed_direction"].eq(result_table["expected_direction"])
    result_table["bh_q_value"] = float("nan")
    for _, positions in result_table.groupby("link", observed=True).groups.items():
        result_table.loc[positions, "bh_q_value"] = benjamini_hochberg(result_table.loc[positions, "p_value"])
    result_table["passes_link_test"] = result_table["sign_matches"] & result_table["bh_q_value"].le(float(config["fdr_alpha"]))

    paths = {
        "weather": output_dir / "palm_weather_anomalies_monthly.csv",
        "panel": output_dir / "palm_country_year_panel.csv",
        "price_panel": output_dir / "palm_production_price_panel.csv",
        "results": output_dir / "palm_mechanism_results.csv",
    }
    regional.to_csv(paths["weather"], index=False, date_format="%Y-%m-%d")
    panel.to_csv(paths["panel"], index=False)
    price_panel.to_csv(paths["price_panel"], index=False)
    result_table.to_csv(paths["results"], index=False)
    link_passes = result_table.groupby("link")["passes_link_test"].any().to_dict()
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "scope": config["scope"],
        "diagnostics": {"link_passes": link_passes, "complete_mechanism_chain": bool(all(link_passes.values())), "tests": len(result_table)},
        "input_hashes": {
            "palm_oil_mechanism.yaml": sha256_file(config_path),
            "palm_dataset_summary.json": sha256_file(palm_input / "summary.json"),
            "enso_monthly.csv": sha256_file(snapshot / "enso_monthly.csv"),
            "commodity_prices_monthly.parquet": sha256_file(snapshot / "commodity_prices_monthly.parquet"),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
    }
    write_json_atomic(output_dir / "palm_mechanism_summary.json", summary)
    return output_dir
