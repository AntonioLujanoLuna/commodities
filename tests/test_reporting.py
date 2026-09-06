from __future__ import annotations

import json
from pathlib import Path

import pytest

from enso_commodities.provenance import sha256_file, write_json_atomic
from enso_commodities.reporting import (
    STAGES,
    apply_sections,
    build_current_results,
    load_stage_summaries,
    render_evidence_table,
    render_sections,
    verify_stage_outputs,
)

REPORT_SKELETON = """# Current results

Hand-written opening that the generator must never touch.

## Run identity

<!-- generated:run-identity start -->

_Not yet generated._

<!-- generated:run-identity end -->

## Evidence summary

<!-- generated:evidence-summary start -->

_Not yet generated._

<!-- generated:evidence-summary end -->

## Stage receipts

<!-- generated:receipts start -->

_Not yet generated._

<!-- generated:receipts end -->

## Bottom line

Hand-written closing that the generator must never touch.
"""


def _write_receipts(tmp_path: Path, *, stages: tuple[str, ...] | None = None) -> tuple[Path, Path]:
    """Lay out a tables directory with a receipt for each named stage."""
    snapshot = tmp_path / "data" / "processed" / "2026-09-05"
    tables = tmp_path / "tables" / "2026-09-05"
    snapshot.mkdir(parents=True)
    tables.mkdir(parents=True)
    wanted = stages or tuple(stage.key for stage in STAGES)

    for stage in STAGES:
        if stage.key not in wanted:
            continue
        artifact = tables / f"{stage.key}_output.csv"
        artifact.write_text(f"stage,{stage.key}\n", encoding="utf-8")
        payload: dict = {
            "data_provenance": "real",
            "snapshot": "2026-09-05",
            "input_hashes": {"research.yaml": "a" * 64},
            "output_hashes": {artifact.name: sha256_file(artifact)},
        }
        for position, metric in enumerate(stage.metrics):
            target = payload
            for key in metric.path[:-1]:
                target = target.setdefault(key, {})
            target[metric.path[-1]] = position + 1
        write_json_atomic(tables / stage.summary_file, payload)
    return snapshot, tmp_path / "tables"


def _report(tmp_path: Path) -> Path:
    path = tmp_path / "current_results.md"
    path.write_text(REPORT_SKELETON, encoding="utf-8")
    return path


def test_generated_blocks_are_filled_and_prose_is_untouched(tmp_path: Path) -> None:
    snapshot, tables_root = _write_receipts(tmp_path)
    report = _report(tmp_path)

    build_current_results(snapshot, tables_root=tables_root, report_path=report)
    text = report.read_text(encoding="utf-8")

    assert "Hand-written opening that the generator must never touch." in text
    assert "Hand-written closing that the generator must never touch." in text
    assert "_Not yet generated._" not in text
    assert "- Data snapshot: `2026-09-05`" in text
    for stage in STAGES:
        assert f"| {stage.title} |" in text
        assert f"`{stage.summary_file}`" in text


def test_a_stage_that_did_not_run_is_named_rather_than_guessed(tmp_path: Path) -> None:
    snapshot, tables_root = _write_receipts(tmp_path, stages=("inference", "placebo"))
    report = _report(tmp_path)

    build_current_results(snapshot, tables_root=tables_root, report_path=report)
    text = report.read_text(encoding="utf-8")

    assert "| Exposure-weighted panel | not run |" in text
    assert "- Stages not run: " in text
    assert "`panel`" in text


def test_the_generator_refuses_a_receipt_whose_artifacts_changed(tmp_path: Path) -> None:
    """A receipt describing artifacts that have since moved is fiction."""
    snapshot, tables_root = _write_receipts(tmp_path)
    (tables_root / "2026-09-05" / "panel_output.csv").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        build_current_results(snapshot, tables_root=tables_root, report_path=_report(tmp_path))


def test_check_mode_fails_on_a_stale_report_and_passes_on_a_fresh_one(tmp_path: Path) -> None:
    snapshot, tables_root = _write_receipts(tmp_path)
    report = _report(tmp_path)

    with pytest.raises(ValueError, match="does not match the run receipts"):
        build_current_results(snapshot, tables_root=tables_root, report_path=report, check=True)

    build_current_results(snapshot, tables_root=tables_root, report_path=report)
    build_current_results(snapshot, tables_root=tables_root, report_path=report, check=True)


def test_rendering_is_deterministic_so_check_mode_means_something(tmp_path: Path) -> None:
    _, tables_root = _write_receipts(tmp_path)
    output_dir = tables_root / "2026-09-05"
    summaries = load_stage_summaries(output_dir)
    assert render_sections(output_dir, summaries) == render_sections(output_dir, summaries)


def test_stages_that_disagree_about_a_configuration_are_refused(tmp_path: Path) -> None:
    snapshot, tables_root = _write_receipts(tmp_path)
    path = tables_root / "2026-09-05" / "panel_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["input_hashes"]["research.yaml"] = "b" * 64
    write_json_atomic(path, summary)

    with pytest.raises(ValueError, match="disagree on the hash"):
        build_current_results(snapshot, tables_root=tables_root, report_path=_report(tmp_path))


def test_a_receipt_missing_a_declared_metric_is_an_error(tmp_path: Path) -> None:
    """The report spec and the stage that feeds it must not drift apart."""
    _, tables_root = _write_receipts(tmp_path)
    path = tables_root / "2026-09-05" / "inference_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["inference"].pop("primary_rejections")
    write_json_atomic(path, summary)

    output_dir = tables_root / "2026-09-05"
    summaries = load_stage_summaries(output_dir)
    with pytest.raises(ValueError, match="primary_rejections"):
        render_evidence_table(summaries)


def test_a_synthetic_receipt_is_never_reported_as_real(tmp_path: Path) -> None:
    _, tables_root = _write_receipts(tmp_path)
    path = tables_root / "2026-09-05" / "placebo_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["data_provenance"] = "synthetic"
    write_json_atomic(path, summary)

    with pytest.raises(ValueError, match="not a real-data receipt"):
        load_stage_summaries(tables_root / "2026-09-05")


def test_an_empty_tables_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No stage receipts"):
        load_stage_summaries(tmp_path / "empty")


def test_a_receipt_without_output_hashes_cannot_be_quoted(tmp_path: Path) -> None:
    _, tables_root = _write_receipts(tmp_path, stages=("inference",))
    path = tables_root / "2026-09-05" / "inference_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.pop("output_hashes")
    write_json_atomic(path, summary)

    output_dir = tables_root / "2026-09-05"
    with pytest.raises(ValueError, match="no output-hash receipt"):
        verify_stage_outputs(output_dir, load_stage_summaries(output_dir))


def test_missing_markers_are_refused_rather_than_appended() -> None:
    with pytest.raises(ValueError, match="missing the 'evidence-summary' generated markers"):
        apply_sections("# Report\n\nNo markers here.\n", {"evidence-summary": "body"})


def test_the_shipped_note_carries_every_generated_marker() -> None:
    from enso_commodities.config import project_root

    text = (project_root() / "reports" / "current_results.md").read_text(encoding="utf-8")
    sections = {"run-identity": "x", "evidence-summary": "y", "receipts": "z"}
    rendered = apply_sections(text, sections)
    assert "x" in rendered and "y" in rendered and "z" in rendered
