"""Alternative climate indices used as substitute treatments.

These series exist only so that the frozen event-study machinery can be re-run
with the ENSO index replaced by a different climate index. They are never
controls, never regressors, and never enter the primary inferential contract.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .provenance import sha256_file, write_json_atomic

REQUIRED_FILES = {
    "psl_dmi.data": "dmi",
    "psl_pdo.data": "pdo",
    "psl_nao.data": "nao",
    "psl_amo.data": "amo",
}


def latest_climate_snapshot(root: Path | None = None) -> Path:
    source = root or project_root() / "data" / "climate" / "raw"
    candidates = sorted(path for path in source.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No climate snapshots found under {source}")
    return candidates[-1]


def parse_psl_index(path: Path, column: str) -> pd.DataFrame:
    """Parse the NOAA PSL fixed monthly-index layout.

    The file is a two-year header range, one row per year holding twelve monthly
    values, then a line carrying the missing-value sentinel, then free text. The
    sentinel line is required: silently guessing it would turn a placeholder into
    a real observation.
    """
    text = path.read_text(encoding="utf-8", errors="strict")
    stripped = [line for line in text.splitlines() if line.strip()]
    if not stripped:
        raise ValueError(f"{path.name}: file is empty")
    header = stripped[0].split()
    if len(header) != 2:
        raise ValueError(f"{path.name}: header is not a two-year range")
    first_year, last_year = int(header[0]), int(header[1])
    expected_rows = last_year - first_year + 1
    if expected_rows < 1:
        raise ValueError(f"{path.name}: header year range is empty")
    rows = stripped[1 : 1 + expected_rows]
    if len(rows) != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} year rows, found {len(rows)}")
    if len(stripped) <= 1 + expected_rows:
        raise ValueError(f"{path.name}: missing-value sentinel line not found")
    sentinel_tokens = stripped[1 + expected_rows].split()
    if len(sentinel_tokens) != 1:
        raise ValueError(f"{path.name}: missing-value sentinel line is not a single value")
    sentinel = float(sentinel_tokens[0])

    records: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        tokens = row.split()
        if len(tokens) != 13:
            raise ValueError(f"{path.name}: year row {tokens[:1]} does not hold twelve months")
        year = int(tokens[0])
        if year != first_year + offset:
            raise ValueError(f"{path.name}: year rows are not contiguous at {year}")
        for month, token in enumerate(tokens[1:], start=1):
            value = float(token)
            records.append(
                {
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    column: None if abs(value - sentinel) < 1e-6 else value,
                }
            )
    result = pd.DataFrame.from_records(records)
    result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values("date", ignore_index=True)
    if result["date"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate month keys")
    if result[column].notna().sum() < 240:
        raise ValueError(f"{path.name}: fewer than twenty years of values")
    return result


def _verify_manifest(snapshot: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("data_provenance") != "real":
        raise ValueError("Climate snapshot is not marked as real data")
    receipts = manifest.get("sources")
    if not isinstance(receipts, list):
        raise ValueError("Climate source manifest has no receipt list")
    by_name = {item.get("filename"): item for item in receipts if isinstance(item, dict)}
    missing = set(REQUIRED_FILES) - set(by_name)
    if missing:
        raise ValueError(f"Climate manifest is missing {sorted(missing)}")
    for name in REQUIRED_FILES:
        if sha256_file(snapshot / name) != by_name[name].get("sha256"):
            raise ValueError(f"Raw climate hash mismatch for {name}")


def build_climate_dataset(
    snapshot_dir: Path | None = None,
    *,
    processed_root: Path | None = None,
) -> Path:
    snapshot = snapshot_dir or latest_climate_snapshot()
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
        processed_root or project_root() / "data" / "climate" / "processed"
    ) / snapshot.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "climate_indices_monthly.csv"
    panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    coverage = {
        column: {
            "first_month": panel.loc[panel[column].notna(), "date"].min().date().isoformat(),
            "last_month": panel.loc[panel[column].notna(), "date"].max().date().isoformat(),
            "months": int(panel[column].notna().sum()),
        }
        for column in REQUIRED_FILES.values()
    }
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "role": "substitute_treatment_only",
        "coverage": coverage,
        "input_hashes": {"manifest.json": sha256_file(manifest_path)},
        "output_hashes": {output_path.name: sha256_file(output_path)},
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return output_dir


def download_climate_snapshot(snapshot_date: date | None = None) -> Path:
    from .download import download_all

    return download_all(
        raw_root=project_root() / "data" / "climate" / "raw",
        snapshot_date=snapshot_date,
        config_path=project_root() / "config" / "climate_sources.yaml",
    )
