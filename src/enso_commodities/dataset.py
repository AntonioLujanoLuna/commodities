from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import project_root
from .parsers import parse_noaa_index, parse_pink_sheet, parse_pink_sheet_total_index
from .provenance import sha256_file, write_json_atomic

REQUIRED_FILENAMES = {
    "RONI.ascii.txt",
    "oni.ascii.txt",
    "CMO-Historical-Data-Monthly.xlsx",
}


def latest_snapshot(raw_root: Path | None = None) -> Path:
    root = raw_root or project_root() / "data" / "raw"
    candidates = sorted(path for path in root.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No dated real-data snapshots found under {root}")
    return candidates[-1]


def verify_snapshot(snapshot: Path, manifest: dict[str, Any]) -> None:
    receipts = manifest.get("sources")
    if not isinstance(receipts, list):
        raise ValueError("Source manifest has no receipt list")
    by_filename = {
        receipt.get("filename"): receipt for receipt in receipts if isinstance(receipt, dict)
    }
    missing = REQUIRED_FILENAMES - set(by_filename)
    if missing:
        raise ValueError(f"Source manifest is missing required files: {sorted(missing)}")
    for filename in REQUIRED_FILENAMES:
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = by_filename[filename].get("sha256")
        observed = sha256_file(path)
        if expected != observed:
            raise ValueError(
                f"Raw snapshot hash mismatch for {filename}: expected {expected}, got {observed}"
            )


def build_real_dataset(
    snapshot_dir: Path | None = None,
    *,
    processed_root: Path | None = None,
) -> Path:
    snapshot = snapshot_dir or latest_snapshot()
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("data_provenance") != "real":
        raise ValueError(f"{manifest_path}: only real-data snapshots are accepted")
    verify_snapshot(snapshot, manifest)

    roni = parse_noaa_index(snapshot / "RONI.ascii.txt", name="roni")
    oni = parse_noaa_index(snapshot / "oni.ascii.txt", name="oni")
    enso = roni.merge(oni, on="date", how="inner", validate="one_to_one")
    prices, diagnostics = parse_pink_sheet(snapshot / "CMO-Historical-Data-Monthly.xlsx")
    market_index = parse_pink_sheet_total_index(snapshot / "CMO-Historical-Data-Monthly.xlsx")
    panel = prices.merge(enso, on="date", how="inner", validate="many_to_one")
    if panel.empty:
        raise ValueError("ENSO and commodity-price sources have no overlapping months")

    output_root = processed_root or project_root() / "data" / "processed"
    output_dir = output_root / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    enso_path = output_dir / "enso_monthly.csv"
    prices_path = output_dir / "commodity_prices_monthly.parquet"
    panel_path = output_dir / "monthly_panel.parquet"
    market_index_path = output_dir / "world_bank_indices_monthly.csv"
    enso.to_csv(enso_path, index=False, date_format="%Y-%m-%d")
    prices.to_parquet(prices_path, index=False)
    panel.to_parquet(panel_path, index=False)
    market_index.to_csv(market_index_path, index=False, date_format="%Y-%m-%d")

    summary: dict[str, Any] = {
        "data_provenance": "real",
        "enso": {
            "first_month": enso["date"].min().date().isoformat(),
            "last_month": enso["date"].max().date().isoformat(),
            "months": len(enso),
        },
        "panel": {
            "commodities": int(panel["commodity"].nunique()),
            "first_month": panel["date"].min().date().isoformat(),
            "last_month": panel["date"].max().date().isoformat(),
            "missing_price_values": int(panel["value"].isna().sum()),
            "rows": len(panel),
        },
        "output_hashes": {
            enso_path.name: sha256_file(enso_path),
            market_index_path.name: sha256_file(market_index_path),
            panel_path.name: sha256_file(panel_path),
            prices_path.name: sha256_file(prices_path),
        },
        "world_bank_total_index": {
            "first_month": market_index["date"].min().date().isoformat(),
            "last_month": market_index["date"].max().date().isoformat(),
            "months": len(market_index),
        },
        "pink_sheet": {
            "commodities": diagnostics.commodities,
            "first_period": diagnostics.first_period,
            "last_period": diagnostics.last_period,
            "mismatch_rows_reported_by_source": diagnostics.mismatch_rows,
            "months": diagnostics.months,
            "non_positive_values_treated_as_missing": diagnostics.non_positive_values,
            "source_updated": diagnostics.source_updated,
        },
        "snapshot": snapshot.name,
        "source_manifest_sha256": sha256_file(manifest_path),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return output_dir
