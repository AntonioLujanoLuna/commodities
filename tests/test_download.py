from pathlib import Path

import pytest

from enso_commodities.config import Source
from enso_commodities.download import _validate_download


def source(format_name: str, minimum_bytes: int = 1) -> Source:
    return Source(
        name="test",
        label="Test",
        url="https://example.test/data",
        landing_page="https://example.test/",
        filename="data",
        format=format_name,
        minimum_bytes=minimum_bytes,
    )


def test_validate_download_rejects_html_disguised_as_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"<html>error</html>")
    with pytest.raises(ValueError, match="not an XLSX"):
        _validate_download(path, source("xlsx"))


def test_validate_download_rejects_truncated_response(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_bytes(b"SEAS")
    with pytest.raises(ValueError, match="only 4 bytes"):
        _validate_download(path, source("noaa_ascii", minimum_bytes=100))


def test_validate_download_rejects_html_disguised_as_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"<!doctype html><title>error</title>")
    with pytest.raises(ValueError, match="HTML, not CSV"):
        _validate_download(path, source("csv"))
