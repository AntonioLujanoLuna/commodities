"""Niño 3 and Niño 4 region indices used only to classify episode flavour.

These series never enter the frozen primary contract. They label episodes that
the primary design has already constructed from RONI; they are not treatments,
not controls and not regressors.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .climate_data import parse_psl_index
from .config import project_root
from .provenance import sha256_file, write_json_atomic

REQUIRED_FILES = {
    "psl_nino3.data": "nino3",
    "psl_nino4.data": "nino4",
}


def latest_flavour_snapshot(root: Path | None = None) -> Path:
    source = root or project_root() / "data" / "flavour" / "raw"
    candidates = sorted(path for path in source.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No flavour snapshots found under {source}")
    return candidates[-1]


def _verify_manifest(snapshot: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("data_provenance") != "real":
        raise ValueError("Flavour snapshot is not marked as real data")
    receipts = manifest.get("sources")
    if not isinstance(receipts, list):
        raise ValueError("Flavour source manifest has no receipt list")
    by_name = {item.get("filename"): item for item in receipts if isinstance(item, dict)}
    missing = set(REQUIRED_FILES) - set(by_name)
    if missing:
        raise ValueError(f"Flavour manifest is missing {sorted(missing)}")
    for name in REQUIRED_FILES:
        if sha256_file(snapshot / name) != by_name[name].get("sha256"):
            raise ValueError(f"Raw flavour hash mismatch for {name}")


def build_flavour_dataset(
    snapshot_dir: Path | None = None,
    *,
    processed_root: Path | None = None,
) -> Path:
    snapshot = snapshot_dir or latest_flavour_snapshot()
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    _verify_manifest(snapshot, manifest)

    panel: pd.DataFrame | None = None
    for filename, column in REQUIRED_FILES.items():
        parsed = parse_psl_index(snapshot / filename, column)
        panel = (
            parsed
            if panel is None
            else panel.merge(parsed, on="date", how="outer", validate="one_to_one")
        )
    assert panel is not None
    panel = panel.sort_values("date", ignore_index=True)

    output_dir = (
        processed_root or project_root() / "data" / "flavour" / "processed"
    ) / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "enso_region_indices_monthly.csv"
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    shared = panel["nino3"].notna() & panel["nino4"].notna()
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "role": "episode_flavour_classification_only",
        "coverage": {
            column: {
                "first_month": panel.loc[panel[column].notna(), "date"].min().date().isoformat(),
                "last_month": panel.loc[panel[column].notna(), "date"].max().date().isoformat(),
                "months": int(panel[column].notna().sum()),
            }
            for column in REQUIRED_FILES.values()
        },
        "shared_months": int(shared.sum()),
        "input_hashes": {"manifest.json": sha256_file(manifest_path)},
        "output_hashes": {output_path.name: sha256_file(output_path)},
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return output_dir


def download_flavour_snapshot(snapshot_date: date | None = None) -> Path:
    from .download import download_all

    return download_all(
        raw_root=project_root() / "data" / "flavour" / "raw",
        snapshot_date=snapshot_date,
        config_path=project_root() / "config" / "flavour_sources.yaml",
    )
