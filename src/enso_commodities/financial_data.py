from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .macro_data import latest_macro_snapshot
from .provenance import sha256_file, write_json_atomic

REQUIRED_FILES = {"fred_tb3ms.csv", "fred_baa10ym.csv", "fred_nfci.csv"}


def latest_financial_snapshot(root: Path | None = None) -> Path:
    source = root or project_root() / "data" / "financial" / "raw"
    candidates = sorted(path for path in source.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No financial snapshots found under {source}")
    return candidates[-1]


def _verify_manifest(snapshot: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("data_provenance") != "real":
        raise ValueError("Financial snapshot is not marked as real data")
    receipts = manifest.get("sources")
    if not isinstance(receipts, list):
        raise ValueError("Financial source manifest has no receipt list")
    by_name = {item.get("filename"): item for item in receipts if isinstance(item, dict)}
    if REQUIRED_FILES - set(by_name):
        raise ValueError(f"Financial manifest is missing {sorted(REQUIRED_FILES - set(by_name))}")
    for name in REQUIRED_FILES:
        if sha256_file(snapshot / name) != by_name[name].get("sha256"):
            raise ValueError(f"Raw financial hash mismatch for {name}")


def parse_fred(path: Path, series_id: str, output_column: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"observation_date", series_id}
    if not required.issubset(raw.columns):
        raise ValueError(f"FRED {series_id} columns missing: {sorted(required - set(raw))}")
    result = raw.rename(columns={"observation_date": "date", series_id: output_column})[
        ["date", output_column]
    ]
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result[output_column] = pd.to_numeric(result[output_column], errors="coerce")
    result = result.dropna().sort_values("date", ignore_index=True)
    if result.empty or result["date"].duplicated().any():
        raise ValueError(f"FRED {series_id} has invalid date keys")
    return result


def build_financial_dataset(
    snapshot_dir: Path | None = None,
    macro_snapshot: Path | None = None,
    *,
    processed_root: Path | None = None,
) -> Path:
    snapshot = snapshot_dir or latest_financial_snapshot()
    macro_input = macro_snapshot or latest_macro_snapshot(project_root() / "data" / "macro" / "processed")
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    _verify_manifest(snapshot, manifest)
    tbill = parse_fred(snapshot / "fred_tb3ms.csv", "TB3MS", "us_tbill_3m_percent")
    spread = parse_fred(snapshot / "fred_baa10ym.csv", "BAA10YM", "us_baa_treasury_spread_percent")
    nfci_weekly = parse_fred(snapshot / "fred_nfci.csv", "NFCI", "chicago_fed_nfci")
    nfci_weekly["date"] = nfci_weekly["date"].dt.to_period("M").dt.to_timestamp()
    nfci = nfci_weekly.groupby("date", as_index=False)["chicago_fed_nfci"].mean()
    macro_path = macro_input / "macro_controls_monthly.csv"
    macro = pd.read_csv(macro_path, parse_dates=["date"])
    cpi = macro.loc[:, ["date", "us_cpi"]].dropna().sort_values("date")
    prior = cpi["us_cpi"].shift(12)
    cpi["us_cpi_trailing_12m_log_inflation_percent"] = (
        cpi["us_cpi"].div(prior).map(math.log, na_action="ignore") * 100
    )
    panel = tbill.merge(spread, on="date", how="outer", validate="one_to_one")
    panel = panel.merge(nfci, on="date", how="outer", validate="one_to_one")
    panel = panel.merge(cpi, on="date", how="outer", validate="one_to_one")
    panel["us_ex_post_real_short_rate_percent"] = panel["us_tbill_3m_percent"].sub(
        panel["us_cpi_trailing_12m_log_inflation_percent"]
    )
    panel = panel.sort_values("date", ignore_index=True)
    output_dir = (processed_root or project_root() / "data" / "financial" / "processed") / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "financial_controls_monthly.csv"
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    columns = ["us_ex_post_real_short_rate_percent", "us_baa_treasury_spread_percent", "chicago_fed_nfci"]
    coverage = {
        column: {
            "first_month": panel.loc[panel[column].notna(), "date"].min().date().isoformat(),
            "last_month": panel.loc[panel[column].notna(), "date"].max().date().isoformat(),
            "months": int(panel[column].notna().sum()),
        }
        for column in columns
    }
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "transformations": {
            "chicago_fed_nfci": "calendar_month_mean_of_weekly_index",
            "us_baa_treasury_spread_percent": "monthly_level_in_percentage_points",
            "us_ex_post_real_short_rate_percent": "TB3MS_minus_trailing_12m_CPI_log_inflation",
        },
        "coverage": coverage,
        "input_hashes": {"manifest.json": sha256_file(manifest_path), "macro_controls_monthly.csv": sha256_file(macro_path)},
        "output_hashes": {output_path.name: sha256_file(output_path)},
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return output_dir
