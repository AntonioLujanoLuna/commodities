from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import Source, load_sources, project_root
from .provenance import sha256_file, write_json_atomic

USER_AGENT = "enso-commodities/0.1 (+https://github.com/AntonioLujanoLuna/commodities)"


def _validate_download(path: Path, source: Source) -> None:
    size = path.stat().st_size
    if size < source.minimum_bytes:
        raise ValueError(
            f"{source.name}: download is only {size} bytes; expected at least "
            f"{source.minimum_bytes}"
        )
    with path.open("rb") as handle:
        prefix = handle.read(256)
    if source.format == "xlsx" and not prefix.startswith(b"PK"):
        raise ValueError(f"{source.name}: response is not an XLSX/ZIP file")
    if source.format == "zip" and not prefix.startswith(b"PK"):
        raise ValueError(f"{source.name}: response is not a ZIP file")
    if source.format == "noaa_ascii" and b"SEAS" not in prefix:
        raise ValueError(f"{source.name}: NOAA ASCII header not found")
    if source.format == "csv" and prefix.lstrip().lower().startswith((b"<html", b"<!doctype")):
        raise ValueError(f"{source.name}: response is HTML, not CSV")


def download_source(
    source: Source,
    snapshot_dir: Path,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    destination = snapshot_dir / source.filename
    sidecar = destination.with_name(f"{destination.name}.meta.json")

    with requests.get(
        source.url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout_seconds,
        stream=True,
    ) as response:
        response.raise_for_status()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=snapshot_dir, prefix=f".{source.filename}.", suffix=".part"
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            _validate_download(temporary_path, source)
            digest = sha256_file(temporary_path)

            if destination.exists():
                existing_digest = sha256_file(destination)
                if existing_digest != digest:
                    raise FileExistsError(
                        f"Refusing to overwrite immutable snapshot {destination}; "
                        f"existing sha256={existing_digest}, new sha256={digest}"
                    )
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)

            metadata: dict[str, Any] = {
                "bytes": destination.stat().st_size,
                "content_type": response.headers.get("Content-Type"),
                "data_provenance": "real",
                "etag": response.headers.get("ETag"),
                "filename": source.filename,
                "format": source.format,
                "label": source.label,
                "landing_page": source.landing_page,
                "last_modified": response.headers.get("Last-Modified"),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digest,
                "source": source.name,
                "url": source.url,
            }
            if sidecar.exists():
                # A repeat run may verify an existing snapshot, but never rewrites its receipt.
                return metadata
            write_json_atomic(sidecar, metadata)
            return metadata
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def download_all(
    *,
    raw_root: Path | None = None,
    snapshot_date: date | None = None,
    config_path: Path | None = None,
) -> Path:
    root = raw_root or project_root() / "data" / "raw"
    snapshot = (snapshot_date or date.today()).isoformat()
    snapshot_dir = root / snapshot
    receipts = [download_source(source, snapshot_dir) for source in load_sources(config_path)]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": "real",
        "snapshot": snapshot,
        "sources": receipts,
    }
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        write_json_atomic(manifest_path, manifest)
    return snapshot_dir
