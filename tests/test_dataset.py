from pathlib import Path

import pytest

from enso_commodities.dataset import verify_snapshot
from enso_commodities.provenance import sha256_file, verify_hashes


def test_verify_snapshot_rejects_changed_raw_file(tmp_path: Path) -> None:
    filenames = ["RONI.ascii.txt", "oni.ascii.txt", "CMO-Historical-Data-Monthly.xlsx"]
    receipts = []
    for filename in filenames:
        path = tmp_path / filename
        path.write_bytes(filename.encode())
        receipts.append({"filename": filename, "sha256": sha256_file(path)})
    manifest = {"sources": receipts}
    (tmp_path / "RONI.ascii.txt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="hash mismatch for RONI"):
        verify_snapshot(tmp_path, manifest)


def test_verify_hashes_accepts_exact_file_and_rejects_change(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("value\n1\n", encoding="utf-8")
    expected = {path.name: sha256_file(path)}
    verify_hashes(tmp_path, expected)

    path.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"hash mismatch for table\.csv"):
        verify_hashes(tmp_path, expected)
