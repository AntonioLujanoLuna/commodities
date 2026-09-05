from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .provenance import sha256_file, write_json_atomic

REQUIRED_MACRO_FILENAMES = {
    "bis_us_neer_narrow.csv",
    "fred_cpiaucsl.csv",
    "dallas_fed_igrea.xlsx",
}


def latest_macro_snapshot(raw_root: Path | None = None) -> Path:
    root = raw_root or project_root() / "data" / "macro" / "raw"
    candidates = sorted(path for path in root.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No dated macro-data snapshots found under {root}")
    return candidates[-1]


def verify_macro_snapshot(snapshot: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("data_provenance") != "real":
        raise ValueError("Macro snapshot is not marked as real data")
    receipts = manifest.get("sources")
    if not isinstance(receipts, list):
        raise ValueError("Macro source manifest has no receipt list")
    by_filename = {
        receipt.get("filename"): receipt for receipt in receipts if isinstance(receipt, dict)
    }
    missing = REQUIRED_MACRO_FILENAMES - set(by_filename)
    if missing:
        raise ValueError(f"Macro source manifest is missing files: {sorted(missing)}")
    for filename in REQUIRED_MACRO_FILENAMES:
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = by_filename[filename].get("sha256")
        observed = sha256_file(path)
        if expected != observed:
            raise ValueError(
                f"Raw macro hash mismatch for {filename}: expected {expected}, got {observed}"
            )


def _validate_monthly_series(data: pd.DataFrame, value_column: str, source: str) -> pd.DataFrame:
    result = data.loc[:, ["date", value_column]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result[value_column] = pd.to_numeric(result[value_column], errors="coerce")
    result = result.dropna(subset=["date", value_column]).sort_values("date", ignore_index=True)
    if result.empty:
        raise ValueError(f"{source}: no numeric monthly observations")
    if result["date"].duplicated().any():
        raise ValueError(f"{source}: duplicate month keys")
    return result


def parse_bis_neer(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"FREQ", "EER_TYPE", "EER_BASKET", "REF_AREA", "TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(raw.columns):
        raise ValueError(f"BIS NEER columns missing: {sorted(required - set(raw))}")
    dimensions = raw.loc[:, ["FREQ", "EER_TYPE", "EER_BASKET", "REF_AREA"]].drop_duplicates()
    if len(dimensions) != 1 or dimensions.iloc[0].to_dict() != {
        "FREQ": "M",
        "EER_TYPE": "N",
        "EER_BASKET": "N",
        "REF_AREA": "US",
    }:
        raise ValueError("BIS file is not the monthly US narrow nominal EER series")
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["TIME_PERIOD"].astype(str) + "-01", errors="raise"),
            "us_neer_narrow": raw["OBS_VALUE"],
        }
    )
    result = _validate_monthly_series(data, "us_neer_narrow", "BIS NEER")
    if result["us_neer_narrow"].le(0).any():
        raise ValueError("BIS NEER contains non-positive observations")
    return result


def parse_fred_cpi(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"observation_date", "CPIAUCSL"}
    if not required.issubset(raw.columns):
        raise ValueError(f"FRED CPI columns missing: {sorted(required - set(raw))}")
    data = raw.rename(columns={"observation_date": "date", "CPIAUCSL": "us_cpi"})
    result = _validate_monthly_series(data, "us_cpi", "FRED CPI")
    if result["us_cpi"].le(0).any():
        raise ValueError("FRED CPI contains non-positive observations")
    return result


def parse_dallas_global_activity(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Kilian Index", header=None)
    if raw.shape[1] < 2 or str(raw.iloc[0, 0]).strip() != "Date":
        raise ValueError("Dallas Fed global-activity workbook header not found")
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[1:, 0], errors="coerce"),
            "global_real_activity": pd.to_numeric(raw.iloc[1:, 1], errors="coerce"),
        }
    ).dropna()
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    return _validate_monthly_series(data, "global_real_activity", "Dallas Fed activity")


def _add_consecutive_change(
    data: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
    logarithmic: bool,
) -> pd.DataFrame:
    result = data.copy()
    prior_date = result["date"].shift(1)
    month_number = result["date"].dt.year * 12 + result["date"].dt.month
    prior_month_number = prior_date.dt.year * 12 + prior_date.dt.month
    consecutive = month_number.sub(prior_month_number).eq(1)
    if logarithmic:
        ratio = result[source_column].div(result[source_column].shift(1))
        result[output_column] = ratio.map(math.log, na_action="ignore").where(consecutive)
    else:
        result[output_column] = result[source_column].diff().where(consecutive)
    return result


def build_macro_dataset(
    snapshot_dir: Path | None = None,
    *,
    processed_root: Path | None = None,
) -> Path:
    snapshot = snapshot_dir or latest_macro_snapshot()
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    verify_macro_snapshot(snapshot, manifest)

    neer = _add_consecutive_change(
        parse_bis_neer(snapshot / "bis_us_neer_narrow.csv"),
        source_column="us_neer_narrow",
        output_column="us_neer_log_change",
        logarithmic=True,
    )
    cpi = _add_consecutive_change(
        parse_fred_cpi(snapshot / "fred_cpiaucsl.csv"),
        source_column="us_cpi",
        output_column="us_cpi_log_change",
        logarithmic=True,
    )
    activity = _add_consecutive_change(
        parse_dallas_global_activity(snapshot / "dallas_fed_igrea.xlsx"),
        source_column="global_real_activity",
        output_column="global_real_activity_change",
        logarithmic=False,
    )
    controls = neer.merge(cpi, on="date", how="outer", validate="one_to_one").merge(
        activity, on="date", how="outer", validate="one_to_one"
    )
    controls = controls.sort_values("date", ignore_index=True)

    output_root = processed_root or project_root() / "data" / "macro" / "processed"
    output_dir = output_root / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    controls_path = output_dir / "macro_controls_monthly.csv"
    controls.to_csv(controls_path, index=False, date_format="%Y-%m-%d")
    coverage = {}
    for column in [
        "us_neer_log_change",
        "us_cpi_log_change",
        "global_real_activity_change",
    ]:
        valid = controls.loc[controls[column].notna(), "date"]
        coverage[column] = {
            "first_month": valid.min().date().isoformat(),
            "last_month": valid.max().date().isoformat(),
            "months": len(valid),
        }
    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "transformations": {
            "global_real_activity_change": "first_difference_in_index_points",
            "us_cpi_log_change": "consecutive_month_log_change",
            "us_neer_log_change": "consecutive_month_log_change; positive_is_USD_appreciation",
        },
        "coverage": coverage,
        "source_manifest_sha256": sha256_file(manifest_path),
        "output_hashes": {controls_path.name: sha256_file(controls_path)},
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return output_dir
