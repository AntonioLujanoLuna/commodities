"""Build the compact, auditable publication layer for a completed snapshot.

Large resampling tables remain outside Git. This module publishes the small stage
receipts, the result tables needed to audit the claims, a commodity scorecard and
three figures. A manifest binds those files to the analysis source-tree hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .config import project_root
from .provenance import sha256_file, write_json_atomic
from .raw_events import latest_processed_snapshot
from .reporting import (
    STAGES,
    analysis_source_tree_hash,
    load_stage_summaries,
    verify_stage_outputs,
)

RESULT_FILES = (
    "primary_inference_results.csv",
    "negative_control_inference.csv",
    "primary_placebo_results.csv",
    "negative_control_placebo_results.csv",
    "primary_minimum_detectable_effect.csv",
    "dose_response_results.csv",
    "macro_primary_results.csv",
    "macro_control_results.csv",
    "macro_fragility_summary.csv",
    "macro_control_fragility_summary.csv",
    "robustness_candidate_summary.csv",
    "robustness_control_summary.csv",
    "specificity_candidate_results.csv",
    "specificity_control_results.csv",
    "endpoint_diagnostic_results.csv",
    "surrogate_treatment_summary.csv",
    "surrogate_treatment_gate_null.csv",
    "surrogate_treatment_commodity_rates.csv",
    "financial_control_results.csv",
    "palm_mechanism_results.csv",
    "external_exposure_weights.csv",
    "panel_specification_results.csv",
    "panel_block_sensitivity.csv",
    "specification_curve_shifts.csv",
    "specification_curve_cells.csv",
    "forecast_predictions.csv",
    "forecast_results.csv",
    "program_timing_null_shifts.csv",
    "program_timing_null_cells.csv",
    "dispersion_results.csv",
    "cold_phase_results.csv",
    "cold_phase_episodes.csv",
    "flavour_contrast_results.csv",
    "flavour_episode_labels.csv",
    "flavour_episode_labels_sensitivity.csv",
)


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_scorecard(source_dir: Path) -> pd.DataFrame:
    """Combine the existing gates without inventing a new inferential threshold."""
    primary = _read_csv(source_dir, "primary_inference_results.csv")[
        [
            "commodity",
            "group",
            "mean_return",
            "studentized_ci_lower",
            "studentized_ci_upper",
            "reject_fdr",
            "reject_fwer",
        ]
    ]
    placebo = _read_csv(source_dir, "primary_placebo_results.csv")[
        ["commodity", "passes_current_gates"]
    ]
    power = _read_csv(source_dir, "primary_minimum_detectable_effect.csv")[
        ["commodity", "minimum_detectable_effect_marginal", "observed_effect_below_marginal_mde"]
    ]
    macro = _read_csv(source_dir, "macro_primary_results.csv")[["commodity", "passes_macro_gates"]]
    fragility = _read_csv(source_dir, "macro_fragility_summary.csv")[
        ["commodity", "survives_all_deletions"]
    ]
    robustness = _read_csv(source_dir, "robustness_candidate_summary.csv")[
        ["commodity", "passes_timing_index_robustness"]
    ]
    dose = _read_csv(source_dir, "dose_response_results.csv")
    dose = dose.loc[dose["role"].eq("mechanism_candidate"), ["commodity", "reject_fdr"]].rename(
        columns={"reject_fdr": "amplitude_dose_response_fdr"}
    )
    specificity = _read_csv(source_dir, "specificity_candidate_results.csv")
    specificity = (
        specificity.groupby("commodity", as_index=False)["reject_direction_contrast_fdr"]
        .any()
        .rename(columns={"reject_direction_contrast_fdr": "any_warm_cold_contrast_fdr"})
    )

    scorecard = primary
    for frame in (placebo, power, macro, fragility, robustness, dose, specificity):
        scorecard = scorecard.merge(frame, on="commodity", how="left", validate="one_to_one")

    def label(row: pd.Series) -> str:
        if bool(row["passes_timing_index_robustness"]):
            return "robust_historical_association"
        if bool(row["reject_fwer"]):
            return "familywise_primary_association"
        if bool(row["reject_fdr"]):
            return "primary_association_not_robust"
        if bool(row["observed_effect_below_marginal_mde"]):
            return "inconclusive_underpowered"
        # Without a prespecified equivalence bound, failure to reject is not
        # evidence that the effect is economically negligible.
        return "inconclusive"

    scorecard["evidence_status"] = scorecard.apply(label, axis=1)
    scorecard["mechanism_status"] = np.where(
        scorecard["commodity"].eq("Palm oil"), "tested_incomplete", "not_tested"
    )
    forecast_path = source_dir / "forecast_results.csv"
    scorecard["out_of_sample_status"] = "not_tested"
    if forecast_path.is_file():
        forecast = pd.read_csv(forecast_path)
        forecast_status = (
            forecast.groupby("commodity", observed=True)
            .agg(
                all_horizons_improve=("rmse_improvement", lambda values: bool(values.gt(0).all())),
                any_loss_test_rejects=(
                    "loss_difference_p_value",
                    lambda values: bool(values.lt(0.05).any()),
                ),
            )
            .reset_index()
        )
        forecast_status["out_of_sample_status"] = np.where(
            forecast_status["all_horizons_improve"]
            & forecast_status["any_loss_test_rejects"],
            "pseudo_oos_incremental_value",
            "pseudo_oos_not_established",
        )
        scorecard = scorecard.drop(columns="out_of_sample_status").merge(
            forecast_status[["commodity", "out_of_sample_status"]],
            on="commodity",
            how="left",
            validate="one_to_one",
        )
        scorecard["out_of_sample_status"] = scorecard["out_of_sample_status"].fillna(
            "not_tested"
        )
    return scorecard.sort_values(["evidence_status", "commodity"]).reset_index(drop=True)


def _save_effect_figure(scorecard: pd.DataFrame, path: Path) -> None:
    ordered = scorecard.sort_values("mean_return").reset_index(drop=True)
    y = np.arange(len(ordered))
    palette = {
        "robust_historical_association": "#087f5b",
        "familywise_primary_association": "#1971c2",
        "primary_association_not_robust": "#f08c00",
        "inconclusive_underpowered": "#868e96",
        "inconclusive": "#ced4da",
    }
    colors = [palette[value] for value in ordered["evidence_status"]]
    lower = ordered["mean_return"] - ordered["studentized_ci_lower"]
    upper = ordered["studentized_ci_upper"] - ordered["mean_return"]
    fig, axis = plt.subplots(figsize=(10, 11))
    axis.errorbar(
        ordered["mean_return"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#868e96",
        alpha=0.8,
    )
    axis.scatter(ordered["mean_return"], y, c=colors, s=28, zorder=3)
    axis.axvline(0, color="#343a40", linewidth=1)
    axis.set_yticks(y, ordered["commodity"])
    axis.set_xlabel("Mean month-12 market-adjusted cumulative return")
    axis.set_title("Primary event-study estimates and studentized 95% intervals")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        handles=[
            Patch(color=palette["robust_historical_association"], label="Robust historical"),
            Patch(color=palette["familywise_primary_association"], label="Primary FWER"),
            Patch(color=palette["primary_association_not_robust"], label="Primary BH only"),
            Patch(color=palette["inconclusive_underpowered"], label="Inconclusive: underpowered"),
            Patch(color=palette["inconclusive"], label="Inconclusive"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        metadata={"Title": "Primary event-study estimates", "Creator": "enso-commodities"},
    )
    plt.close(fig)


def _save_gate_figure(scorecard: pd.DataFrame, path: Path) -> None:
    columns = [
        ("reject_fdr", "Primary BH"),
        ("passes_current_gates", "Placebo gates"),
        ("passes_macro_gates", "Macro gates"),
        ("survives_all_deletions", "Leave-one-out"),
        ("passes_timing_index_robustness", "Timing/index"),
        ("amplitude_dose_response_fdr", "Amplitude"),
    ]
    ordered = scorecard.sort_values("commodity").reset_index(drop=True)
    matrix = ordered[[name for name, _ in columns]].fillna(False).astype(int).to_numpy()
    fig, axis = plt.subplots(figsize=(8, 11))
    axis.imshow(
        matrix,
        aspect="auto",
        cmap=matplotlib.colors.ListedColormap(["#f1f3f5", "#087f5b"]),
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(range(len(columns)), [label for _, label in columns], rotation=35, ha="right")
    axis.set_yticks(range(len(ordered)), ordered["commodity"])
    axis.set_title("Evidence gates by commodity (green = passes)")
    axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, metadata={"Title": "Evidence gates", "Creator": "enso-commodities"})
    plt.close(fig)


def _save_panel_figure(source_dir: Path, path: Path) -> None:
    panel = _read_csv(source_dir, "panel_specification_results.csv")
    panel = panel.loc[panel["term"].eq("exposure_x_enso")].sort_values(
        ["index_definition", "lag_months", "weighting"]
    )
    labels = [
        f"{row.index_definition.upper()} lag {row.lag_months}, {row.weighting}"
        for row in panel.itertuples()
    ]
    y = np.arange(len(panel))
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.errorbar(
        panel["estimate"],
        y,
        xerr=np.vstack(
            [panel["estimate"] - panel["ci_lower"], panel["ci_upper"] - panel["estimate"]]
        ),
        fmt="o",
        color="#1971c2",
        ecolor="#74c0fc",
        capsize=2,
    )
    axis.axvline(0, color="#343a40", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Exposure x ENSO coefficient")
    axis.set_title("Exposure-panel specification family")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        metadata={"Title": "Exposure-panel specification family", "Creator": "enso-commodities"},
    )
    plt.close(fig)


def _bundle_readme(snapshot: str) -> str:
    return f"""# Auditable results bundle: {snapshot}

This directory is generated from the locally verified stage receipts for snapshot `{snapshot}`.
It intentionally excludes large bootstrap replicates and raw/licensed data. `manifest.json` binds
each published file to its SHA-256 digest and records the full analysis source-tree fingerprint.

- `scorecard.csv` and `scorecard.md` combine existing gates; they define no new test.
- `figures/` visualizes primary estimates, gate passage, and the panel specification family.
- Stage summary JSON and selected result CSV files are byte-for-byte copies of the run artifacts.

Run `enso-publication --check --bundle reports/artifacts/{snapshot}` to validate this bundle.
"""


def verify_publication_bundle(bundle: Path) -> Path:
    manifest_path = bundle / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    if manifest.get("analysis_source_tree_sha256") != analysis_source_tree_hash():
        raise ValueError("Publication bundle was generated from a different analysis source tree")
    if manifest.get("data_provenance") != "real" or manifest.get("snapshot") != bundle.name:
        raise ValueError("Publication bundle identity or provenance is invalid")
    for name, expected in manifest.get("published_files", {}).items():
        path = bundle / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Publication bundle hash mismatch for {name}")
    for name, expected in manifest.get("source_receipts", {}).items():
        if sha256_file(bundle / name) != expected:
            raise ValueError(f"Published receipt differs from its source digest: {name}")
    expected_scorecard = build_scorecard(bundle).to_csv(index=False, lineterminator="\n")
    if (bundle / "scorecard.csv").read_text(encoding="utf-8") != expected_scorecard:
        raise ValueError("Published scorecard does not match its result tables")
    return bundle


def build_publication_bundle(
    processed_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    output_root: Path | None = None,
    check: bool = False,
    bundle: Path | None = None,
) -> Path:
    if check:
        if bundle is None:
            raise ValueError("--check requires an explicit publication bundle")
        return verify_publication_bundle(bundle)

    snapshot = processed_snapshot or latest_processed_snapshot()
    source_dir = (tables_root or project_root() / "tables") / snapshot.name
    summaries = load_stage_summaries(source_dir)
    verify_stage_outputs(source_dir, summaries)
    destination = (output_root or project_root() / "reports" / "artifacts") / snapshot.name
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)

    copied: list[Path] = []
    for stage in STAGES:
        source = source_dir / stage.summary_file
        if source.is_file():
            target = destination / source.name
            target.write_bytes(source.read_bytes())
            copied.append(target)
    for name in RESULT_FILES:
        source = source_dir / name
        if source.is_file():
            target = destination / name
            target.write_bytes(source.read_bytes())
            copied.append(target)

    scorecard = build_scorecard(destination)
    scorecard_csv = destination / "scorecard.csv"
    scorecard_md = destination / "scorecard.md"
    scorecard.to_csv(scorecard_csv, index=False, lineterminator="\n")
    scorecard_md.write_text(scorecard.to_markdown(index=False) + "\n", encoding="utf-8")
    readme = destination / "README.md"
    readme.write_text(_bundle_readme(snapshot.name), encoding="utf-8", newline="\n")
    copied.extend([scorecard_csv, scorecard_md, readme])

    effect_figure = figures / "primary_effects.png"
    gate_figure = figures / "evidence_gates.png"
    panel_figure = figures / "panel_specifications.png"
    _save_effect_figure(scorecard, effect_figure)
    _save_gate_figure(scorecard, gate_figure)
    _save_panel_figure(destination, panel_figure)
    copied.extend([effect_figure, gate_figure, panel_figure])

    manifest = {
        "analysis_source_tree_sha256": analysis_source_tree_hash(),
        "data_provenance": "real",
        "published_files": {
            path.relative_to(destination).as_posix(): sha256_file(path) for path in sorted(copied)
        },
        "snapshot": snapshot.name,
        "source_receipts": {
            stage.summary_file: sha256_file(source_dir / stage.summary_file)
            for stage in STAGES
            if (source_dir / stage.summary_file).is_file()
        },
    }
    write_json_atomic(destination / "manifest.json", manifest)
    return destination
