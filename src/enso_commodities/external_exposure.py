"""Outcome-independent crop exposure from FAO ASIS and SPAM rasters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import Source, project_root
from .download import download_source
from .provenance import sha256_file, write_json_atomic
from .raw_events import latest_processed_snapshot
from .universe import load_commodity_registry


@dataclass(frozen=True)
class ExposureSourceSpec:
    config: dict[str, Any]
    commodity_crop_codes: dict[str, str]
    excluded_candidates: dict[str, str]
    minimum_hotspot_crop_area_share: float


def load_exposure_config(path: Path | None = None) -> ExposureSourceSpec:
    config_path = path or project_root() / "config" / "exposure_v2.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    provenance = raw["provenance"]
    if provenance != {
        "version": 2,
        "method": "external_crop_area_x_el_nino_drought_hotspot",
        "authored_on": date(2026, 9, 6),
        "outcome_blind": True,
        "inference_scope": "retrospective_external_validation",
    }:
        raise ValueError("Exposure v2 provenance is not the frozen outcome-blind contract")
    transformation = raw["transformation"]
    if transformation["crop_area_layer"] != "ALL":
        raise ValueError("Exposure weights must use total physical crop area")
    if transformation["drought_values"] != "positive_difference_only":
        raise ValueError("Exposure weights must use positive drought-risk differences only")
    if transformation["drought_scale_fraction"] != 1.0:
        raise ValueError("FAO ASIS drought values must be read as fractions")
    if transformation["missing_drought_values"] != "zero_non_hotspot":
        raise ValueError("ASIS cells outside the published hotspot layer must map to zero")
    minimum_share = float(transformation["minimum_hotspot_crop_area_share"])
    if not 0 < minimum_share <= 1:
        raise ValueError("minimum_hotspot_crop_area_share must fall in (0, 1]")
    mapping = {str(k): str(v) for k, v in raw["commodity_crop_codes"].items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Each commodity must map to a distinct SPAM crop layer")
    return ExposureSourceSpec(
        config=raw,
        commodity_crop_codes=mapping,
        excluded_candidates={str(k): str(v) for k, v in raw["excluded_candidates"].items()},
        minimum_hotspot_crop_area_share=minimum_share,
    )


def _sources(spec: ExposureSourceSpec) -> list[Source]:
    raw = spec.config["sources"]
    drought = raw["drought_hotspot"]
    sources = [
        Source(
            name="fao_asis_drought_hotspot",
            label=str(drought["label"]),
            url=str(drought["url"]),
            landing_page=str(drought["landing_page"]),
            filename=str(drought["filename"]),
            format="tif",
            minimum_bytes=int(drought["minimum_bytes"]),
        )
    ]
    base = str(raw["crop_area_base_url"]).rstrip("/")
    landing = str(raw["crop_area_landing_page"])
    for code in sorted(set(spec.commodity_crop_codes.values())):
        object_name = f"AFIRM.SPAM2020-PHYSICAL-AREA.{code}.ALL.tif"
        sources.append(
            Source(
                name=f"spam2020_{code.lower()}",
                label=f"SPAM 2020 total physical crop area, {code}",
                url=f"{base}/{object_name}",
                landing_page=landing,
                filename=f"spam2020_{code.lower()}.tif",
                format="tif",
                minimum_bytes=100_000,
            )
        )
    return sources


def download_exposure_data(
    *,
    raw_root: Path | None = None,
    snapshot_date: date | None = None,
    config_path: Path | None = None,
) -> Path:
    spec = load_exposure_config(config_path)
    root = raw_root or project_root() / "data" / "exposure" / "raw"
    snapshot = (snapshot_date or date.today()).isoformat()
    output = root / snapshot
    receipts = [download_source(source, output, timeout_seconds=180) for source in _sources(spec)]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": "real",
        "snapshot": snapshot,
        "sources": receipts,
    }
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        write_json_atomic(manifest_path, manifest)
    return output


def latest_exposure_snapshot(root: Path | None = None) -> Path:
    source = root or project_root() / "data" / "exposure" / "raw"
    candidates = sorted(path for path in source.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No external-exposure snapshots found under {source}")
    return candidates[-1]


def crop_weight(
    crop_area: np.ndarray, drought_difference: np.ndarray
) -> tuple[float, float, float]:
    """Area-weight the published positive El Nino drought-hotspot burden.

    The FAO raster contains only positive hotspot cells; crop area outside the
    published hotspot support contributes zero to the numerator and remains in
    the denominator. Conditioning on hotspot cells would erase the main source
    of cross-crop variation: how much production geography overlaps a hotspot.
    """
    area = np.asarray(crop_area, dtype="float64")
    drought = np.asarray(drought_difference, dtype="float64")
    if area.shape != drought.shape:
        raise ValueError("Crop-area and drought grids must have the same shape")
    positive_area = np.isfinite(area) & (area > 0)
    total_area = float(area[positive_area].sum())
    hotspot = positive_area & np.isfinite(drought) & (drought > 0)
    hotspot_area = float(area[hotspot].sum())
    if total_area <= 0 or hotspot_area <= 0:
        raise ValueError("Crop layer contains no usable physical area")
    risk = np.clip(drought[hotspot], 0.0, None)
    weight = float(np.sum(area[hotspot] * risk) / total_area)
    return weight, hotspot_area / total_area, total_area


def _verify_raw(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    if manifest.get("data_provenance") != "real":
        raise ValueError("External-exposure snapshot is not real data")
    for receipt in manifest.get("sources") or []:
        name = str(receipt["filename"])
        if sha256_file(snapshot / name) != receipt["sha256"]:
            raise ValueError(f"External-exposure raw hash mismatch for {name}")
    return manifest


def build_external_exposure_weights(
    processed_snapshot: Path | None = None,
    *,
    raw_snapshot: Path | None = None,
    tables_root: Path | None = None,
    config_path: Path | None = None,
    registry_path: Path | None = None,
) -> Path:
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as error:
        raise RuntimeError("External exposure weights require the 'spatial' extra") from error

    processed = processed_snapshot or latest_processed_snapshot()
    raw = raw_snapshot or latest_exposure_snapshot()
    output = (tables_root or project_root() / "tables") / processed.name
    output.mkdir(parents=True, exist_ok=True)
    spec = load_exposure_config(config_path)
    registry = load_commodity_registry(registry_path)
    candidates = set(
        registry.entries.loc[registry.entries["role"].eq("mechanism_candidate"), "commodity"]
    )
    configured = set(spec.commodity_crop_codes) | set(spec.excluded_candidates)
    if configured != candidates:
        raise ValueError(
            "Exposure v2 coverage must partition every candidate; "
            f"missing={sorted(candidates - configured)}, unknown={sorted(configured - candidates)}"
        )
    manifest = _verify_raw(raw)
    drought_path = raw / "asis_drought_probability_cropland.tif"
    rows: list[dict[str, Any]] = []
    reference_grid: np.ndarray | None = None
    reference_signature: tuple[Any, ...] | None = None
    with rasterio.open(drought_path) as drought_source:
        for commodity, code in sorted(spec.commodity_crop_codes.items()):
            crop_path = raw / f"spam2020_{code.lower()}.tif"
            with rasterio.open(crop_path) as crop_source:
                signature = (
                    crop_source.height,
                    crop_source.width,
                    crop_source.transform,
                    crop_source.crs,
                )
                if reference_signature is None:
                    reference_signature = signature
                    reference_grid = np.full(
                        (crop_source.height, crop_source.width), np.nan, dtype="float32"
                    )
                    reproject(
                        source=rasterio.band(drought_source, 1),
                        destination=reference_grid,
                        src_transform=drought_source.transform,
                        src_crs=drought_source.crs,
                        src_nodata=drought_source.nodata,
                        dst_transform=crop_source.transform,
                        dst_crs=crop_source.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.average,
                    )
                elif signature != reference_signature:
                    raise ValueError("SPAM crop layers do not share one frozen grid")
                area = crop_source.read(1, masked=True).filled(np.nan)
            assert reference_grid is not None
            weight, hotspot_share, total_area = crop_weight(area, reference_grid)
            if hotspot_share < spec.minimum_hotspot_crop_area_share:
                raise ValueError(
                    f"{commodity} has only {hotspot_share:.1%} crop area in the published "
                    "hotspot support, below the contract"
                )
            rows.append(
                {
                    "commodity": commodity,
                    "crop_code": code,
                    "exposure_weight": weight,
                    "hotspot_crop_area_share": hotspot_share,
                    "physical_crop_area_hectares": total_area,
                }
            )
    weights = pd.DataFrame.from_records(rows).sort_values("commodity", ignore_index=True)
    weights_path = output / "external_exposure_weights.csv"
    weights.to_csv(weights_path, index=False)
    config_file = config_path or project_root() / "config" / "exposure_v2.yaml"
    summary = {
        "data_provenance": "real",
        "provenance": {
            **spec.config["provenance"],
            "authored_on": spec.config["provenance"]["authored_on"].isoformat(),
        },
        "coverage": {
            "included_candidates": len(weights),
            "excluded_candidates": len(spec.excluded_candidates),
            "minimum_hotspot_crop_area_share": float(weights["hotspot_crop_area_share"].min()),
        },
        "input_hashes": {
            "manifest.json": sha256_file(raw / "manifest.json"),
            "exposure_v2.yaml": sha256_file(config_file),
            "commodities.yaml": sha256_file(
                registry_path or project_root() / "config" / "commodities.yaml"
            ),
            **{str(receipt["filename"]): str(receipt["sha256"]) for receipt in manifest["sources"]},
        },
        "output_hashes": {weights_path.name: sha256_file(weights_path)},
        "snapshot": processed.name,
        "source_snapshot": raw.name,
    }
    write_json_atomic(output / "external_exposure_summary.json", summary)
    return output
