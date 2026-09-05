from __future__ import annotations

import json
import zipfile
from calendar import monthrange
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import project_root
from .provenance import sha256_file, write_json_atomic

FAOSTAT_NAME = "faostat_crops_livestock_normalized.zip"


def latest_palm_snapshot(root: Path | None = None) -> Path:
    source = root or project_root() / "data" / "mechanisms" / "palm_oil" / "raw"
    candidates = sorted(path for path in source.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No palm-oil snapshots found under {source}")
    return candidates[-1]


def _verify(snapshot: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("data_provenance") != "real":
        raise ValueError("Palm-oil snapshot is not marked as real data")
    receipts = manifest.get("sources")
    if not isinstance(receipts, list):
        raise ValueError("Palm-oil manifest has no source receipts")
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError("Palm-oil manifest contains an invalid receipt")
        path = snapshot / str(receipt["filename"])
        if sha256_file(path) != receipt.get("sha256"):
            raise ValueError(f"Palm-oil raw hash mismatch for {path.name}")


def parse_power_monthly(path: Path, *, point: str, country: str) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_end = lines.index("-END HEADER-")
    except ValueError as error:
        raise ValueError(f"NASA POWER header terminator missing in {path.name}") from error
    raw = pd.read_csv(path, skiprows=header_end + 1)
    required = {"PARAMETER", "YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}
    if not required.issubset(raw.columns):
        raise ValueError(f"NASA POWER columns missing in {path.name}")
    tidy = raw.melt(
        id_vars=["PARAMETER", "YEAR"],
        value_vars=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
        var_name="month_name",
        value_name="value",
    )
    month_numbers = {name: index for index, name in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
    tidy["month"] = tidy["month_name"].map(month_numbers)
    tidy["date"] = pd.to_datetime({"year": tidy["YEAR"], "month": tidy["month"], "day": 1})
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce").replace(-999, pd.NA)
    tidy.loc[tidy["PARAMETER"].eq("PRECTOTCORR"), "value"] *= [
        monthrange(date.year, date.month)[1]
        for date in tidy.loc[tidy["PARAMETER"].eq("PRECTOTCORR"), "date"]
    ]
    tidy["point"] = point
    tidy["country"] = country
    return tidy.rename(columns={"PARAMETER": "parameter"})[
        ["date", "country", "point", "parameter", "value"]
    ].sort_values(["date", "parameter"], ignore_index=True)


def parse_faostat(path: Path, *, countries: set[str], items: set[str]) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    columns = ["Area", "Item", "Element", "Year", "Unit", "Value", "Flag", "Note"]
    with zipfile.ZipFile(path) as archive:
        data_name = next(name for name in archive.namelist() if name.endswith("(Normalized).csv"))
        with archive.open(data_name) as handle:
            for chunk in pd.read_csv(handle, usecols=columns, chunksize=200_000, low_memory=False):
                rows = chunk.loc[chunk["Area"].isin(countries) & chunk["Item"].isin(items)]
                if not rows.empty:
                    selected.append(rows)
    if not selected:
        raise ValueError("FAOSTAT contains no configured palm-oil observations")
    result = pd.concat(selected, ignore_index=True)
    result["Value"] = pd.to_numeric(result["Value"], errors="coerce")
    return result.rename(
        columns={"Area": "country", "Item": "item", "Element": "element", "Year": "year", "Unit": "unit", "Value": "value", "Flag": "flag", "Note": "note"}
    ).sort_values(["country", "item", "element", "year"], ignore_index=True)


def build_palm_oil_dataset(
    snapshot_dir: Path | None = None, *, processed_root: Path | None = None, mechanism_config_path: Path | None = None
) -> Path:
    snapshot = snapshot_dir or latest_palm_snapshot()
    config_path = mechanism_config_path or project_root() / "config" / "palm_oil_mechanism.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    _verify(snapshot, manifest)
    weather = []
    for point in config["weather_points"]:
        weather.append(
            parse_power_monthly(
                snapshot / f"power_{point['name']}.csv", point=str(point["name"]), country=str(point["country"])
            )
        )
    weather_table = pd.concat(weather, ignore_index=True)
    faostat = parse_faostat(
        snapshot / FAOSTAT_NAME,
        countries=set(config["countries"]),
        items=set(config["production_items"]),
    )
    output_dir = (processed_root or project_root() / "data" / "mechanisms" / "palm_oil" / "processed") / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    weather_path = output_dir / "palm_weather_monthly.csv"
    production_path = output_dir / "faostat_palm_oil_annual.csv"
    weather_table.to_csv(weather_path, index=False, date_format="%Y-%m-%d")
    faostat.to_csv(production_path, index=False)
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "coverage": {
            "weather_first_month": weather_table["date"].min().date().isoformat(),
            "weather_last_month": weather_table["date"].max().date().isoformat(),
            "weather_points": int(weather_table["point"].nunique()),
            "faostat_first_year": int(faostat["year"].min()),
            "faostat_last_year": int(faostat["year"].max()),
            "faostat_rows": len(faostat),
        },
        "transformations": {"PRECTOTCORR": "mm_per_day_times_calendar_days_in_month", "T2M": "monthly_mean_celsius"},
        "input_hashes": {"manifest.json": sha256_file(manifest_path), "palm_oil_mechanism.yaml": sha256_file(config_path)},
        "output_hashes": {weather_path.name: sha256_file(weather_path), production_path.name: sha256_file(production_path)},
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return output_dir
