"""Generate the numeric part of the current-results note from run receipts.

Roughly sixty lines of prose numbers currently appear twice, in ``README.md``
and in ``reports/current_results.md``, and both copies are typed by hand from
the run artifacts. Two hand-maintained copies of every number is a drift
guarantee; it has not drifted yet only because the project is young.

Every stage already writes a hash-linked JSON receipt. This module reads them
and renders the run identity, the evidence summary and the receipt digests, so
the numbers in the note are produced by the same artifacts a reader would audit
rather than transcribed alongside them. The interpretation prose stays
hand-written: it sits outside the generated markers and is never touched.

Two properties make the output trustworthy:

*Stale is refused.* Before rendering, every output hash each receipt recorded is
re-verified against the file on disk. A receipt that describes artifacts which
have since changed cannot be turned into a report at all.

*Drift is detectable.* The render is a pure function of the receipts -- no
timestamps, no environment capture -- so ``--check`` can re-render and compare.
That turns "the report is out of date" from something a reader might notice
into something the build refuses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import project_root
from .provenance import sha256_file, verify_hashes
from .raw_events import latest_processed_snapshot

MARKER_PREFIX = "<!-- generated:"
MISSING = "not run"


@dataclass(frozen=True)
class Metric:
    """One number pulled out of a receipt by its dotted location."""

    label: str
    path: tuple[str, ...]
    digits: int | None = None

    def render(self, summary: dict[str, Any]) -> str:
        value: Any = summary
        for key in self.path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    f"Receipt is missing {'.'.join(self.path)}; the report spec and the "
                    "stage that writes it have diverged"
                )
            value = value[key]
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float) and self.digits is not None:
            return f"{value:.{self.digits}f}"
        return str(value)


@dataclass(frozen=True)
class StageReport:
    key: str
    title: str
    summary_file: str
    metrics: tuple[Metric, ...]


def _metric(label: str, *path: str, digits: int | None = None) -> Metric:
    return Metric(label=label, path=path, digits=digits)


# The order the pipeline runs in, which is also the order the evidence ladder
# is meant to be read in. A stage whose receipt is absent renders as not run; a
# stage whose receipt lacks a declared key raises, because a silently dropped
# metric is the drift this module exists to prevent.
STAGES: tuple[StageReport, ...] = (
    StageReport(
        key="inference",
        title="Primary event study",
        summary_file="inference_summary.json",
        metrics=(
            _metric("Candidate family size", "inference", "fdr_family_size"),
            _metric("BH rejections", "inference", "primary_rejections"),
            _metric("BY rejections", "inference", "primary_rejections_arbitrary_dependence"),
            _metric("Westfall-Young rejections", "inference", "primary_fwer_rejections"),
            _metric("Sign agreement", "inference", "sign_agreement_count"),
            _metric(
                "Controls with raw p<0.05", "inference", "negative_controls_with_raw_p_below_0_05"
            ),
        ),
    ),
    StageReport(
        key="placebo",
        title="Neutral-date placebo",
        summary_file="placebo_summary.json",
        metrics=(
            _metric("Eligible anchors", "diagnostics", "eligible_neutral_anchors"),
            _metric(
                "Candidates passing gates",
                "diagnostics",
                "primary_candidates_passing_current_gates",
            ),
            _metric(
                "Controls rejecting raw", "diagnostics", "negative_controls_rejecting_raw_placebo"
            ),
            _metric(
                "Onset mean year", "diagnostics", "era_balance", "mean_year", "observed", digits=1
            ),
            _metric(
                "Placebo mean year",
                "diagnostics",
                "era_balance",
                "mean_year",
                "placebo_mean",
                digits=1,
            ),
            _metric(
                "Era-balance p", "diagnostics", "era_balance", "mean_year", "p_value", digits=4
            ),
        ),
    ),
    StageReport(
        key="power",
        title="Minimum detectable effect",
        summary_file="power_summary.json",
        metrics=(
            _metric(
                "Median MDE, marginal",
                "diagnostics",
                "median_minimum_detectable_effect_marginal",
                digits=4,
            ),
            _metric(
                "Median MDE, family bound",
                "diagnostics",
                "median_minimum_detectable_effect_family",
                digits=4,
            ),
            _metric(
                "Candidates below their own MDE", "diagnostics", "candidates_below_marginal_mde"
            ),
            _metric(
                "Candidates with unreachable MDE",
                "diagnostics",
                "candidates_with_unreachable_family_mde",
            ),
        ),
    ),
    StageReport(
        key="macro",
        title="External macro controls",
        summary_file="macro_analysis_summary.json",
        metrics=(
            _metric("Macro episodes", "diagnostics", "real_macro_episode_count"),
            _metric("Models estimated", "diagnostics", "macro_models_estimated"),
            _metric(
                "Candidates passing macro gates",
                "diagnostics",
                "primary_candidates_passing_macro_gates",
            ),
            _metric(
                "Controls rejecting bootstrap",
                "diagnostics",
                "negative_controls_rejecting_bootstrap_raw",
            ),
            _metric(
                "Controls rejecting placebo",
                "diagnostics",
                "negative_controls_rejecting_placebo_raw",
            ),
        ),
    ),
    StageReport(
        key="fragility",
        title="Leave one episode out",
        summary_file="macro_fragility_run_summary.json",
        metrics=(
            _metric("Episode deletions", "diagnostics", "episode_deletions"),
            _metric(
                "Candidates surviving every deletion",
                "diagnostics",
                "primary_candidates_surviving_all_deletions",
            ),
            _metric(
                "Controls rejecting after every deletion",
                "diagnostics",
                "controls_rejecting_after_every_deletion",
            ),
        ),
    ),
    StageReport(
        key="robustness",
        title="Timing and index grid",
        summary_file="robustness_summary.json",
        metrics=(
            _metric(
                "Candidates passing every cell",
                "diagnostics",
                "candidates_passing_all_specifications_and_fragility",
            ),
            _metric(
                "Controls failing specificity everywhere",
                "diagnostics",
                "controls_failing_specificity_everywhere",
            ),
        ),
    ),
    StageReport(
        key="specificity",
        title="Warm versus cold",
        summary_file="specificity_summary.json",
        metrics=(
            _metric("Candidate cells", "diagnostics", "candidate_rows"),
            _metric("Control cells", "diagnostics", "control_rows"),
            _metric(
                "Control direction-specific cells",
                "diagnostics",
                "controls_with_direction_specific_contrast_cells",
            ),
        ),
    ),
    StageReport(
        key="endpoint",
        title="Endpoint diagnostics",
        summary_file="endpoint_diagnostics_summary.json",
        metrics=(
            _metric(
                "Control log-mean rejections", "diagnostics", "control_log_mean_raw_rejections"
            ),
            _metric(
                "Control time-trend rejections", "diagnostics", "control_time_trend_raw_rejections"
            ),
            _metric(
                "Control regime correlations", "diagnostics", "control_regime_correlations_raw"
            ),
        ),
    ),
    StageReport(
        key="financial",
        title="Financial controls",
        summary_file="financial_control_summary.json",
        metrics=(
            _metric("Models estimated", "diagnostics", "models_estimated"),
            _metric("Control raw rejections", "diagnostics", "control_raw_rejections"),
            _metric(
                "Control contrast rejections",
                "diagnostics",
                "control_direction_contrast_raw_rejections",
            ),
        ),
    ),
    StageReport(
        key="palm",
        title="Palm-oil mechanism",
        summary_file="palm_mechanism_summary.json",
        metrics=(
            _metric("Tests", "diagnostics", "tests"),
            _metric("Complete physical chain", "diagnostics", "complete_mechanism_chain"),
        ),
    ),
    StageReport(
        key="panel",
        title="Exposure-weighted panel",
        summary_file="panel_summary.json",
        metrics=(
            _metric("Primary exposure estimate", "results", "primary_exposure_estimate", digits=5),
            _metric(
                "Primary studentized p", "results", "primary_exposure_studentized_p_value", digits=5
            ),
            _metric(
                "Control interaction p", "results", "primary_control_studentized_p_value", digits=5
            ),
            _metric(
                "Weight-mapping permutation p",
                "results",
                "primary_exposure_weight_permutation_p_value",
                digits=5,
            ),
            _metric("Exposure cells passing grid FDR", "results", "exposure_cells_rejecting_fdr"),
        ),
    ),
)


def load_stage_summaries(output_dir: Path) -> dict[str, dict[str, Any] | None]:
    """Read every stage receipt present, refusing anything not real-data."""
    summaries: dict[str, dict[str, Any] | None] = {}
    for stage in STAGES:
        path = output_dir / stage.summary_file
        if not path.is_file():
            summaries[stage.key] = None
            continue
        with path.open(encoding="utf-8") as handle:
            summary: dict[str, Any] = json.load(handle)
        if summary.get("data_provenance") != "real":
            raise ValueError(f"{stage.summary_file} is not a real-data receipt")
        summaries[stage.key] = summary
    if all(summary is None for summary in summaries.values()):
        raise ValueError(f"No stage receipts found under {output_dir}")
    return summaries


def verify_stage_outputs(output_dir: Path, summaries: dict[str, dict[str, Any] | None]) -> None:
    """Re-verify every artifact each receipt claims, before quoting any of it.

    A receipt whose outputs have changed underneath it describes a run that no
    longer exists, and a report built from it would be fiction.
    """
    for stage in STAGES:
        summary = summaries.get(stage.key)
        if summary is None:
            continue
        recorded = summary.get("output_hashes")
        if not isinstance(recorded, dict):
            raise ValueError(f"{stage.summary_file} has no output-hash receipt")
        verify_hashes(output_dir, {str(key): str(value) for key, value in recorded.items()})


def _configuration_hashes(summaries: dict[str, dict[str, Any] | None]) -> dict[str, str]:
    """Every configuration digest the receipts agree on, refusing disagreement."""
    hashes: dict[str, str] = {}
    for stage in STAGES:
        summary = summaries.get(stage.key)
        if summary is None:
            continue
        for name, digest in (summary.get("input_hashes") or {}).items():
            if not str(name).endswith(".yaml"):
                continue
            existing = hashes.get(str(name))
            if existing is not None and existing != str(digest):
                raise ValueError(
                    f"Stages disagree on the hash of {name}; they were not all run "
                    "against the same configuration"
                )
            hashes[str(name)] = str(digest)
    return dict(sorted(hashes.items()))


def render_run_identity(output_dir: Path, summaries: dict[str, dict[str, Any] | None]) -> str:
    lines = [
        f"- Data snapshot: `{output_dir.name}`",
        "- Stage receipts: "
        + ", ".join(f"`{stage.key}`" for stage in STAGES if summaries.get(stage.key) is not None),
    ]
    missing = [stage.key for stage in STAGES if summaries.get(stage.key) is None]
    if missing:
        lines.append("- Stages not run: " + ", ".join(f"`{key}`" for key in missing))
    for name, digest in _configuration_hashes(summaries).items():
        lines.append(f"- `{name}` SHA-256: `{digest}`")
    return "\n".join(lines)


def render_evidence_table(summaries: dict[str, dict[str, Any] | None]) -> str:
    rows = ["| Stage | Result |", "|---|---|"]
    for stage in STAGES:
        summary = summaries.get(stage.key)
        if summary is None:
            rows.append(f"| {stage.title} | {MISSING} |")
            continue
        measured = "; ".join(f"{metric.label} {metric.render(summary)}" for metric in stage.metrics)
        rows.append(f"| {stage.title} | {measured} |")
    return "\n".join(rows)


def render_receipts_table(output_dir: Path, summaries: dict[str, dict[str, Any] | None]) -> str:
    rows = ["| Stage | Receipt | SHA-256 |", "|---|---|---|"]
    for stage in STAGES:
        if summaries.get(stage.key) is None:
            continue
        digest = sha256_file(output_dir / stage.summary_file)
        rows.append(f"| {stage.title} | `{stage.summary_file}` | `{digest}` |")
    return "\n".join(rows)


def render_sections(
    output_dir: Path, summaries: dict[str, dict[str, Any] | None]
) -> dict[str, str]:
    return {
        "run-identity": render_run_identity(output_dir, summaries),
        "evidence-summary": render_evidence_table(summaries),
        "receipts": render_receipts_table(output_dir, summaries),
    }


def apply_sections(text: str, sections: dict[str, str]) -> str:
    """Replace the content between each generated marker, leaving prose alone."""
    result = text
    for name, body in sections.items():
        start = f"{MARKER_PREFIX}{name} start -->"
        end = f"{MARKER_PREFIX}{name} end -->"
        start_position = result.find(start)
        end_position = result.find(end)
        if start_position < 0 or end_position < 0:
            raise ValueError(f"The report is missing the {name!r} generated markers")
        if end_position < start_position:
            raise ValueError(f"The {name!r} generated markers are reversed")
        head = result[: start_position + len(start)]
        tail = result[end_position:]
        result = f"{head}\n\n{body}\n\n{tail}"
    return result


def build_current_results(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    report_path: Path | None = None,
    check: bool = False,
) -> Path:
    """Render the generated blocks of the current-results note.

    With ``check`` the file is not written; a difference raises instead, which
    is what makes a stale report a build failure rather than a reader's problem.
    """
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or project_root() / "tables") / snapshot.name
    path = report_path or project_root() / "reports" / "current_results.md"

    summaries = load_stage_summaries(output_dir)
    verify_stage_outputs(output_dir, summaries)
    text = path.read_text(encoding="utf-8")
    rendered = apply_sections(text, render_sections(output_dir, summaries))
    if check:
        if rendered != text:
            raise ValueError(
                f"{path.name} does not match the run receipts under {output_dir}; "
                "rerun the report generator"
            )
        return path
    if rendered != text:
        path.write_text(rendered, encoding="utf-8")
    return path
