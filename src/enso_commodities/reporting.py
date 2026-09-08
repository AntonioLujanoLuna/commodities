"""Generate the numeric part of the current-results note from run receipts.

Roughly sixty lines of prose numbers currently appear twice, in ``README.md``
and in ``reports/current_results.md``, and both copies are typed by hand from
the run artifacts. Two hand-maintained copies of every number is a drift
guarantee; it has not drifted yet only because the project is young.

Every stage already writes a hash-linked JSON receipt. This module reads them
and renders the run identity, evidence summary, receipt-backed interpretation
and receipt digests, so the claims in the note are produced by the same
artifacts a reader would audit rather than transcribed alongside them. Only
durable framing that does not quote run results stays outside the markers.

Two properties make the output trustworthy:

*Stale is refused.* Before rendering, every output hash each receipt recorded is
re-verified against the file on disk. A receipt that describes artifacts which
have since changed cannot be turned into a report at all.

*Drift is detectable.* The render has no timestamps, fingerprints the source
tree, and captures only stable receipt facts, so ``--check`` can re-render and
compare. That turns "the report is out of date" from something a reader might
notice into something the build refuses.
"""

from __future__ import annotations

import hashlib
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
        key="dose_response",
        title="Episode-amplitude dose response",
        summary_file="dose_response_summary.json",
        metrics=(
            _metric("Candidate family size", "results", "candidate_family_size"),
            _metric("Positive slopes", "results", "candidate_positive_slopes"),
            _metric("Raw p<0.05", "results", "candidate_raw_rejections"),
            _metric("BH rejections", "results", "candidate_fdr_rejections"),
            _metric("Max-t rejections", "results", "candidate_fwer_rejections"),
            _metric("Controls with raw p<0.05", "results", "control_raw_rejections"),
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
        key="surrogate_treatment",
        title="Placebo treatments",
        summary_file="surrogate_treatment_run_summary.json",
        metrics=(
            _metric("Surrogate treatments", "diagnostics", "surrogate_cells"),
            _metric("External climate treatments", "diagnostics", "external_cells"),
            _metric(
                "External max candidates passing gates",
                "diagnostics",
                "external_maximum_candidates_passing_gates",
            ),
            _metric(
                "External max controls rejecting",
                "diagnostics",
                "external_maximum_controls_rejecting_bootstrap_raw",
            ),
            _metric(
                "Mean control rejection rate under surrogates",
                "diagnostics",
                "mean_control_bootstrap_rejection_rate_under_surrogates",
                digits=4,
            ),
            _metric(
                "Candidates-passing surrogate p",
                "diagnostics",
                "candidates_passing_gates_finite_sample_p_value",
                digits=4,
            ),
            _metric(
                "Controls-rejecting-bootstrap surrogate p",
                "diagnostics",
                "controls_rejecting_bootstrap_raw_finite_sample_p_value",
                digits=4,
            ),
            _metric(
                "Controls-rejecting-placebo surrogate p",
                "diagnostics",
                "controls_rejecting_placebo_raw_finite_sample_p_value",
                digits=4,
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
        key="external_exposure",
        title="External physical exposure",
        summary_file="external_exposure_summary.json",
        metrics=(
            _metric("Included crop candidates", "coverage", "included_candidates"),
            _metric("Explicitly excluded candidates", "coverage", "excluded_candidates"),
            _metric(
                "Minimum crop area in hotspot support",
                "coverage",
                "minimum_hotspot_crop_area_share",
                digits=4,
            ),
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
    StageReport(
        key="specification_curve",
        title="Whole-year timing null",
        summary_file="specification_curve_summary.json",
        metrics=(
            _metric("Specification cells", "contract", "cells_per_alignment"),
            _metric("Shifted alignments", "contract", "null_alignments"),
            _metric("Joint timing p", "results", "joint_timing_p_value", digits=4),
            _metric("Maximum-t timing p", "results", "maximum_t_timing_p_value", digits=4),
        ),
    ),
    StageReport(
        key="forecast",
        title="Recursive forecast benchmark",
        summary_file="forecast_summary.json",
        metrics=(
            _metric("Forecast cells", "results", "cells"),
            _metric("Cells improving RMSE", "results", "cells_with_positive_rmse_improvement"),
            _metric(
                "Loss tests rejecting raw",
                "results",
                "cells_with_loss_difference_p_below_0_05",
            ),
            _metric(
                "Proxy cells beating long-only",
                "results",
                "cells_with_positive_net_excess_over_buy_and_hold",
            ),
            _metric("Best cell commodity", "results", "best_loss_test_commodity"),
            _metric("Best cell horizon", "results", "best_loss_test_horizon_months"),
            _metric("Best cell p", "results", "best_loss_test_p_value", digits=4),
            _metric(
                "Best cell RMSE improvement",
                "results",
                "best_loss_test_rmse_improvement",
                digits=4,
            ),
            _metric(
                "Best cell excess over long-only",
                "results",
                "best_loss_test_net_excess_over_buy_and_hold",
                digits=4,
            ),
            _metric("Genuine out of sample", "genuine_out_of_sample"),
            _metric("Tradability claim permitted", "tradability_claim_permitted"),
        ),
    ),
    StageReport(
        key="program_timing_null",
        title="Selected-family timing null",
        summary_file="program_timing_null_summary.json",
        metrics=(
            _metric("Specification cells", "contract", "cells_per_alignment"),
            _metric("Shifted alignments", "contract", "null_alignments"),
            _metric("Median-|t| timing p", "results", "median_absolute_t_p_value", digits=4),
            _metric("Maximum-|t| timing p", "results", "maximum_absolute_t_p_value", digits=4),
            _metric("Outcome-informed selection", "selection_is_outcome_informed"),
        ),
    ),
    StageReport(
        key="dispersion",
        title="W1 dispersion",
        summary_file="dispersion_summary.json",
        metrics=(
            _metric("Warm episodes", "design", "warm_episodes"),
            _metric("Family shift p", "results", "family_shift_p_value", digits=4),
            _metric("Candidate BH rejections", "results", "candidate_fdr_rejections"),
            _metric("Control raw rejections", "results", "control_raw_rejections"),
            _metric("Clears program threshold", "results", "clears_program_threshold"),
        ),
    ),
    StageReport(
        key="cold_phase",
        title="W2 cold-phase disruption",
        summary_file="cold_phase_summary.json",
        metrics=(
            _metric("Cold episodes", "results", "cold_episodes"),
            _metric("Signed family size", "results", "signed_family_size"),
            _metric("Family shift p", "results", "family_shift_p_value", digits=4),
            _metric("Candidate BH rejections", "results", "family_fdr_rejections"),
            _metric(
                "Control worst-case rejections",
                "results",
                "control_worst_case_rejections",
            ),
            _metric("Clears program threshold", "results", "clears_program_threshold"),
        ),
    ),
    StageReport(
        key="flavour",
        title="W5 episode flavour",
        summary_file="flavour_summary.json",
        metrics=(
            _metric("Eastern episodes", "results", "eastern_episodes"),
            _metric("Central episodes", "results", "central_episodes"),
            _metric("Family bootstrap p", "results", "family_bootstrap_p_value", digits=4),
            _metric("Candidate BH rejections", "results", "candidate_fdr_rejections"),
            _metric(
                "Candidates below own MDE",
                "results",
                "candidates_below_their_own_mde",
            ),
            _metric(
                "Classification agreement",
                "results",
                "classification_agreement_share",
                digits=4,
            ),
        ),
    ),
    StageReport(
        key="forecast_news",
        title="W3 forecast news",
        summary_file="forecast_news_summary.json",
        metrics=(
            _metric("Revisions", "results", "revisions"),
            _metric(
                "Family wild-bootstrap p",
                "results",
                "family_wild_bootstrap_p_value",
                digits=4,
            ),
            _metric("Candidate BH rejections", "results", "candidate_fdr_rejections"),
            _metric("Lead-placebo rejections", "results", "lead_placebo_rejections"),
            _metric("Lag-placebo rejections", "results", "lag_placebo_rejections"),
            _metric("Control rejections", "results", "control_rejections"),
            _metric("Status", "results", "status"),
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


def analysis_source_tree_hash() -> str:
    """Fingerprint the analysis implementation without generated outputs."""
    root = project_root()
    paths = [root / "Makefile", root / "pyproject.toml", root / "uv.lock"]
    for directory in (root / "config", root / "scripts", root / "src"):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        # Git normalizes these text inputs, while Windows working trees commonly
        # use CRLF. Hash logical content so the same commit verifies on CI/Linux.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def render_run_context(output_dir: Path, summaries: dict[str, dict[str, Any] | None]) -> str:
    """Render execution facts from the current checkout and frozen receipts."""
    inference = summaries.get("inference") or {}
    panel = summaries.get("panel") or {}
    primary_replicates = (inference.get("inference") or {}).get("bootstrap_replicates", "n/a")
    panel_replicates = (panel.get("design") or {}).get("bootstrap_replicates", "n/a")
    return "\n".join(
        (
            f"- Generated for snapshot: `{output_dir.name}`",
            f"- Analysis source-tree SHA-256: `{analysis_source_tree_hash()}`",
            f"- Primary bootstrap: {primary_replicates} whole-episode draws",
            f"- Panel bootstrap: {panel_replicates} block draws per specification",
        )
    )


def render_panel_result(summaries: dict[str, dict[str, Any] | None]) -> str:
    """Render the panel conclusion from current panel and timing-null receipts."""
    panel = summaries.get("panel")
    if not panel:
        return "_Not run._"
    results = panel.get("results") or {}
    design = panel.get("design") or {}
    required = (
        "primary_exposure_estimate",
        "primary_exposure_ci_lower",
        "primary_exposure_ci_upper",
        "primary_exposure_studentized_p_value",
        "primary_control_studentized_p_value",
        "primary_exposure_weight_mapping_percentile",
        "primary_exposure_weight_permutation_p_value",
        "exposure_cells_rejecting_fdr",
        "control_cells_rejecting_raw_5_percent",
    )
    if any(key not in results for key in required):
        return "The panel ran, but its receipt lacks detailed narrative metrics."

    scope = {
        "retrospective_external_validation": "retrospective external-validation",
        "exploratory_only": "exploratory",
    }.get(str(design.get("exposure_inference_scope")), "unspecified-scope")
    provenance = (
        "outcome-independent external physical weights"
        if design.get("exposure_outcome_blind") is True
        else "post-outcome exposure weights"
    )
    paragraphs = [
        (
            f"The {scope} primary panel uses {provenance}, RONI lagged "
            f"{design.get('primary_lag_months', 'n/a')} months, commodity and calendar-month "
            "fixed effects, and seasonal-adjusted log returns. Its exposure coefficient is "
            f"`{results['primary_exposure_estimate']:.6f}`, with a year-block 95% interval of "
            f"`[{results['primary_exposure_ci_lower']:.6f}, "
            f"{results['primary_exposure_ci_upper']:.6f}]` and studentized "
            f"`p={results['primary_exposure_studentized_p_value']:.5f}`. The corresponding "
            f"negative-control interaction has `p={results['primary_control_studentized_p_value']:.5f}`."
        ),
        (
            "The named exposure assignment is at the "
            f"{100 * results['primary_exposure_weight_mapping_percentile']:.1f}th percentile of "
            "weight shuffles, with a two-sided mapping-permutation "
            f"`p={results['primary_exposure_weight_permutation_p_value']:.5f}`. "
            f"Across the 20-cell panel grid, {results['exposure_cells_rejecting_fdr']} exposure "
            "cells survive BH correction and "
            f"{results['control_cells_rejecting_raw_5_percent']} control cells reject at raw 5%."
        ),
    ]
    curve = summaries.get("specification_curve")
    if curve:
        contract = curve.get("contract") or {}
        timing = curve.get("results") or {}
        if "joint_timing_p_value" in timing:
            paragraphs.append(
                "The panel-family timing null compares the observed curve with "
                f"{contract.get('null_alignments', 'n/a')} circular whole-year ENSO shifts. "
                f"Its joint median-|t| p-value is `{timing['joint_timing_p_value']:.5f}` and "
                f"its maximum-|t| p-value is "
                f"`{timing.get('maximum_t_timing_p_value', float('nan')):.5f}`. The observed "
                "alignment is therefore not unusually strong within this panel specification family."
            )
    return "\n\n".join(paragraphs)


def render_interpretation_table(summaries: dict[str, dict[str, Any] | None]) -> str:
    """Interpret only receipt-backed facts so prose cannot outlive a run."""
    rows = ["| Stage | Interpretation |", "|---|---|"]
    inference = summaries.get("inference")
    if inference:
        values = inference["inference"]
        rows.append(
            "| Primary event study | "
            f"{values['primary_rejections']} candidate associations survive BH and "
            f"{values['primary_fwer_rejections']} survive Westfall-Young; the control failures "
            "prevent a causal reading. |"
        )
    placebo = summaries.get("placebo")
    if placebo:
        values = placebo["diagnostics"]
        p_value = values["era_balance"]["mean_year"]["p_value"]
        rows.append(
            "| Neutral-date placebo | "
            f"{values['primary_candidates_passing_current_gates']} candidates pass the current "
            f"gates; the mean-year imbalance has p={p_value:.4f} and is diagnostic, not a gate. |"
        )
    power = summaries.get("power")
    if power:
        values = power["diagnostics"]
        family_size = values.get("candidate_family_size")
        if family_size is None and inference:
            family_size = inference["inference"]["fdr_family_size"]
        rows.append(
            "| Minimum detectable effect | "
            f"{values['candidates_below_marginal_mde']} of {family_size} "
            "candidates are below their own marginal MDE and are underpowered, not established "
            "nulls. |"
        )
    dose = summaries.get("dose_response")
    if dose:
        values = dose["results"]
        rows.append(
            "| Episode amplitude | "
            f"{values['candidate_fdr_rejections']} candidate slopes survive BH and "
            f"{values['candidate_fwer_rejections']} survive max-t; "
            f"{values['control_raw_rejections']} controls reject at raw 5%. |"
        )
    robustness = summaries.get("robustness")
    if robustness:
        count = robustness["diagnostics"]["candidates_passing_all_specifications_and_fragility"]
        rows.append(
            f"| Timing/index grid | {count} candidates pass every timing/index cell and the "
            "leave-one-episode-out gate. |"
        )
    specificity = summaries.get("specificity")
    if specificity:
        count = specificity["diagnostics"]["controls_with_direction_specific_contrast_cells"]
        rows.append(
            f"| Warm versus cold | {count} control cells show a direction-specific contrast; "
            "the broader candidate pattern remains mostly phase-nonspecific. |"
        )
    financial = summaries.get("financial")
    if financial:
        values = financial["diagnostics"]
        rows.append(
            "| Financial controls | "
            f"{values['control_raw_rejections']} controls still reject in levels, while "
            f"{values['control_direction_contrast_raw_rejections']} reject the warm-minus-cold "
            "contrast. |"
        )
    palm = summaries.get("palm")
    if palm:
        complete = palm["diagnostics"]["complete_mechanism_chain"]
        rows.append(
            f"| Palm-oil mechanism | The prespecified physical chain is "
            f"{'complete' if complete else 'incomplete'}. |"
        )
    external = summaries.get("external_exposure")
    if external:
        values = external["coverage"]
        rows.append(
            "| External physical exposure | "
            f"{values['included_candidates']} crop candidates have outcome-independent weights; "
            f"{values['excluded_candidates']} unsupported candidates are excluded rather than "
            "coded as zero. |"
        )
    panel = summaries.get("panel")
    if panel:
        values = panel["results"]
        sensitivities = values.get("primary_exposure_block_sensitivity", {})
        robust_blocks = bool(sensitivities) and all(
            cell.get("interval_excludes_zero") is True for cell in sensitivities.values()
        )
        rows.append(
            "| Exposure panel | The primary interval "
            f"{'excludes zero under every block sensitivity' if robust_blocks else 'is not robust under every block sensitivity'}, "
            f"but {values['exposure_cells_rejecting_fdr']} exposure cells survive grid FDR. |"
        )
    surrogate = summaries.get("surrogate_treatment")
    if surrogate:
        values = surrogate["diagnostics"]
        rows.append(
            "| Placebo treatments | Under spectrum-matched surrogate treatments the candidate "
            f"gate count is unusual (p={values['candidates_passing_gates_finite_sample_p_value']:.4f}) "
            "and no alternative climate index reproduces it, but the negative-control placebo "
            f"failure is not unusual (p={values['controls_rejecting_placebo_raw_finite_sample_p_value']:.4f}) "
            "and so is not evidence against ENSO specificity. |"
        )
    curve = summaries.get("specification_curve")
    if curve:
        values = curve["results"]
        rows.append(
            "| Whole-year timing null | The observed specification family has joint timing "
            f"p={values['joint_timing_p_value']:.4f} against circular whole-year shifts. |"
        )
    forecast = summaries.get("forecast")
    if forecast:
        values = forecast["results"]
        loss_cells = int(values["cells_with_loss_difference_p_below_0_05"])
        loss_cell_word = "cell" if loss_cells == 1 else "cells"
        rows.append(
            "| Recursive forecast benchmark | "
            f"ENSO improves RMSE in {values['cells_with_positive_rmse_improvement']} of "
            f"{values['cells']} cells, with paired-loss p<0.05 in "
            f"{loss_cells} {loss_cell_word}; final RONI and "
            "non-investable indexes make this pseudo-OOS. |"
        )
        rows.append(
            "| Best forecast cell | "
            f"{values['best_loss_test_commodity']} at {values['best_loss_test_horizon_months']} "
            f"months has paired-loss p={values['best_loss_test_p_value']:.4f} and RMSE improvement "
            f"{values['best_loss_test_rmse_improvement']:.4f}, but its price-index strategy excess "
            f"over long-only is {values['best_loss_test_net_excess_over_buy_and_hold']:.4f}. |"
        )
    program = summaries.get("program_timing_null")
    if program:
        values = program["results"]
        rows.append(
            "| Selected-family timing null | The locked 96-cell selected family has median-|t| "
            f"timing p={values['median_absolute_t_p_value']:.4f}, but selection used the discovery "
            "outcomes and this is retrospective calibration, not independent validation. |"
        )
    dispersion = summaries.get("dispersion")
    if dispersion:
        values = dispersion["results"]
        rows.append(
            "| W1 dispersion | The warm-window dispersion family has shift-null "
            f"p={values['family_shift_p_value']:.4f}; "
            f"{values['candidate_fdr_rejections']} candidates survive BH and "
            f"{values['control_raw_rejections']} controls reject at raw 5%. |"
        )
    cold_phase = summaries.get("cold_phase")
    if cold_phase:
        values = cold_phase["results"]
        rows.append(
            "| W2 cold-phase disruption | The prespecified signed family has shift-null "
            f"p={values['family_shift_p_value']:.4f}; "
            f"{values['family_fdr_rejections']} candidates survive BH and "
            f"{values['control_worst_case_rejections']} controls reject under the corrected "
            "worst-case rule. The shift result does not clear the program threshold. |"
        )
    flavour = summaries.get("flavour")
    if flavour:
        values = flavour["results"]
        rows.append(
            "| W5 episode flavour | The Eastern-minus-Central-Pacific family has bootstrap "
            f"p={values['family_bootstrap_p_value']:.4f}, with "
            f"{values['candidate_fdr_rejections']} BH rejections. All "
            f"{values['candidates_below_their_own_mde']} candidates are below their own "
            "minimum detectable contrast, so the split is unresolved rather than null. |"
        )
    forecast_news = summaries.get("forecast_news")
    if forecast_news:
        values = forecast_news["results"]
        control_count = int(values["control_rejections"])
        control_word = "control rejects" if control_count == 1 else "controls reject"
        rows.append(
            "| W3 forecast news | The six-month revision family has wild-bootstrap "
            f"p={values['family_wild_bootstrap_p_value']:.4f}, but "
            f"{values['lead_placebo_rejections']} lead-placebo cells and "
            f"{control_count} {control_word}. Status is "
            f"`{values['status']}`: contemporaneous coefficients cannot be read as news "
            "responses. |"
        )
    return "\n".join(rows)


def render_bottom_line(summaries: dict[str, dict[str, Any] | None]) -> str:
    power = summaries.get("power") or {}
    robustness = summaries.get("robustness") or {}
    panel = summaries.get("panel") or {}
    underpowered = (power.get("diagnostics") or {}).get("candidates_below_marginal_mde", "n/a")
    survivors = (robustness.get("diagnostics") or {}).get(
        "candidates_passing_all_specifications_and_fragility", "n/a"
    )
    panel_fdr = (panel.get("results") or {}).get("exposure_cells_rejecting_fdr", "n/a")
    forecast = summaries.get("forecast") or {}
    forecast_rejections = (forecast.get("results") or {}).get(
        "cells_with_loss_difference_p_below_0_05", "n/a"
    )
    forecast_cell_word = "cell" if forecast_rejections == 1 else "cells"
    dispersion = summaries.get("dispersion") or {}
    dispersion_p = (dispersion.get("results") or {}).get("family_shift_p_value")
    cold_phase = summaries.get("cold_phase") or {}
    cold_phase_p = (cold_phase.get("results") or {}).get("family_shift_p_value")
    flavour = summaries.get("flavour") or {}
    flavour_p = (flavour.get("results") or {}).get("family_bootstrap_p_value")
    forecast_news = summaries.get("forecast_news") or {}
    forecast_news_status = (forecast_news.get("results") or {}).get("status")
    v3_sentence = ""
    if dispersion_p is not None and cold_phase_p is not None and flavour_p is not None:
        v3_sentence = (
            f" In the v3 endpoints, dispersion has family shift p={dispersion_p:.4f} and the "
            f"signed cold-phase family has p={cold_phase_p:.4f}; neither clears the frozen "
            f"program threshold. The flavour split has family bootstrap p={flavour_p:.4f} "
            "and is underpowered for every candidate."
        )
    if forecast_news_status is not None:
        v3_sentence += (
            f" The forecast-news stage is `{forecast_news_status}` and therefore supplies no "
            "interpretable news-response finding."
        )
    return (
        f"The pipeline leaves {survivors} timing/index-robust historical associations, while "
        f"{underpowered} candidates remain below their own marginal detection threshold. "
        "The phase and negative-control diagnostics still prevent an ENSO-specific causal "
        f"interpretation, and {panel_fdr} panel exposure cells survive correction across the "
        f"specification grid. The retrospective forecast benchmark has paired-loss p<0.05 in "
        f"{forecast_rejections} {forecast_cell_word} and does not yet satisfy the genuine "
        f"out-of-sample gate.{v3_sentence}"
    )


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
        "run-context": render_run_context(output_dir, summaries),
        "evidence-summary": render_evidence_table(summaries),
        "interpretations": render_interpretation_table(summaries),
        "receipts": render_receipts_table(output_dir, summaries),
        "panel-result": render_panel_result(summaries),
        "bottom-line": render_bottom_line(summaries),
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
    verify_outputs: bool = True,
) -> Path:
    """Render the generated blocks of the current-results note.

    With ``check`` the file is not written; a difference raises instead, which
    is what makes a stale report a build failure rather than a reader's problem.
    """
    snapshot = processed_snapshot or latest_processed_snapshot()
    output_dir = (tables_root or project_root() / "tables") / snapshot.name
    path = report_path or project_root() / "reports" / "current_results.md"

    summaries = load_stage_summaries(output_dir)
    if verify_outputs:
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
