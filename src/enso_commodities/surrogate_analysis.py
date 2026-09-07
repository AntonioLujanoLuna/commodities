"""Real-data placebo-treatment stage and hash-linked receipt.

Exploratory. Nothing here promotes or demotes a commodity in the frozen
event-study contract. It measures how often that contract fires when the ENSO
index is replaced by a series that cannot have caused a price move.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .climate_data import latest_climate_snapshot
from .config import ResearchConfig, load_research_config, project_root
from .macro_data import latest_macro_snapshot
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot
from .statistics import salted_seed
from .surrogate_treatment import (
    TREATMENT_COLUMN,
    TreatmentOutcome,
    evaluate_treatment,
    moment_match,
    phase_randomized_surrogate,
)
from .universe import InferenceSpec, load_commodity_registry

GATE_COUNTS = (
    "candidates_passing_gates",
    "candidates_rejecting_bootstrap_fdr",
    "controls_rejecting_bootstrap_raw",
    "controls_rejecting_placebo_raw",
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    contract = config.get("contract", {})
    if contract.get("role") != "placebo_treatment_falsification":
        raise ValueError("Surrogate contract role must be placebo_treatment_falsification")
    if contract.get("changes_frozen_gates") is not False:
        raise ValueError("The placebo-treatment stage may not change a frozen gate")
    if int(config["surrogates"]["replicates"]) < 1:
        raise ValueError("At least one surrogate replicate is required")
    unsupported = set(config["surrogates"]["variants"]) - {
        "phase_randomized",
        "seasonal_preserving",
    }
    if unsupported:
        raise ValueError(f"Unsupported surrogate variants: {sorted(unsupported)}")
    return config


def _real_summary(path: Path, stage: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary: dict[str, Any] = json.load(handle)
    if summary.get("data_provenance") != "real":
        raise ValueError(f"{stage} requires real-data inputs")
    return summary


def _gate_counts(outcome: TreatmentOutcome) -> dict[str, int]:
    if outcome.candidates.empty:
        return dict.fromkeys(GATE_COUNTS, 0)
    return {
        "candidates_passing_gates": int(outcome.candidates["passes_gates"].sum()),
        "candidates_rejecting_bootstrap_fdr": int(outcome.candidates["reject_bootstrap_fdr"].sum()),
        "controls_rejecting_bootstrap_raw": int(outcome.controls["reject_bootstrap_raw"].sum()),
        "controls_rejecting_placebo_raw": int(outcome.controls["reject_placebo_raw"].sum()),
    }


def _tidy(outcome: TreatmentOutcome, *, treatment: str, family_of: dict[str, str]) -> pd.DataFrame:
    frames = []
    for family, frame in (("candidates", outcome.candidates), ("controls", outcome.controls)):
        if frame.empty:
            continue
        tidy = frame.copy()
        tidy.insert(0, "treatment", treatment)
        tidy.insert(1, "family", family)
        if family == "controls":
            tidy["passes_gates"] = False
            tidy["reject_placebo_fdr"] = False
            tidy["bootstrap_bh_q_value"] = np.nan
            tidy["placebo_bh_q_value"] = np.nan
        else:
            tidy["reject_bootstrap_raw"] = tidy["bootstrap_p_value"].lt(0.05)
            tidy["reject_placebo_raw"] = tidy["placebo_p_value"].lt(0.05)
        frames.append(tidy)
    result = pd.concat(frames, ignore_index=True)
    result["registry_family"] = result["commodity"].map(family_of)
    return result


def _finite_sample_p(observed: float, null_values: pd.Series) -> float:
    valid = null_values.dropna()
    return float((1 + int((valid >= observed).sum())) / (1 + len(valid)))


def _treatment_frame(dates: pd.Series, values: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dates.to_numpy(), TREATMENT_COLUMN: values.to_numpy()})
    return frame.loc[frame[TREATMENT_COLUMN].notna()].reset_index(drop=True)


def _evaluate(
    treatment: pd.DataFrame,
    *,
    label: str,
    inputs: dict[str, pd.DataFrame],
    candidates: list[str],
    controls: list[str],
    config: ResearchConfig,
    spec: InferenceSpec,
    stage_config: dict[str, Any],
) -> TreatmentOutcome:
    inference = stage_config["inference"]
    return evaluate_treatment(
        treatment,
        monthly=inputs["monthly"],
        market=inputs["market"],
        macro=inputs["macro"],
        candidate_commodities=candidates,
        control_commodities=controls,
        config=config,
        spec=spec,
        label=label,
        bootstrap_replicates=int(inference["bootstrap_replicates"]),
        placebo_replicates=int(inference["placebo_replicates"]),
        run_placebo=True,
    )


def run_surrogate_treatments(
    processed_snapshot: Path | None = None,
    macro_snapshot: Path | None = None,
    climate_snapshot: Path | None = None,
    *,
    tables_root: Path | None = None,
    registry_path: Path | None = None,
    research_config_path: Path | None = None,
    stage_config_path: Path | None = None,
) -> Path:
    root = project_root()
    snapshot = processed_snapshot or latest_processed_snapshot()
    macro_input = macro_snapshot or latest_macro_snapshot(root / "data" / "macro" / "processed")
    climate_input = climate_snapshot or latest_climate_snapshot(
        root / "data" / "climate" / "processed"
    )
    output_dir = (tables_root or root / "tables") / snapshot.name
    research_path = research_config_path or root / "config" / "research.yaml"
    stage_path = stage_config_path or root / "config" / "surrogate_treatment.yaml"
    config = load_research_config(research_path)
    spec = load_commodity_registry(registry_path).inference
    stage_config = _load_config(stage_path)

    processed_summary = _real_summary(snapshot / "summary.json", "Placebo treatments")
    raw_summary = _real_summary(output_dir / "raw_event_summary.json", "Placebo treatments")
    macro_summary = _real_summary(macro_input / "summary.json", "Placebo treatments")
    climate_summary = _real_summary(climate_input / "summary.json", "Placebo treatments")
    enso_name = "enso_monthly.csv"
    market_name = "world_bank_indices_monthly.csv"
    monthly_name = "commodity_returns_monthly.parquet"
    macro_name = "macro_controls_monthly.csv"
    climate_name = "climate_indices_monthly.csv"
    verify_hashes(
        snapshot,
        {
            enso_name: processed_summary["output_hashes"][enso_name],
            market_name: processed_summary["output_hashes"][market_name],
        },
    )
    verify_hashes(output_dir, {monthly_name: raw_summary["output_hashes"][monthly_name]})
    verify_hashes(macro_input, {macro_name: macro_summary["output_hashes"][macro_name]})
    verify_hashes(climate_input, {climate_name: climate_summary["output_hashes"][climate_name]})

    enso = pd.read_csv(snapshot / enso_name, parse_dates=["date"])
    inputs = {
        "monthly": pd.read_parquet(output_dir / monthly_name),
        "market": pd.read_csv(snapshot / market_name, parse_dates=["date"]),
        "macro": pd.read_csv(macro_input / macro_name, parse_dates=["date"]),
    }
    climate = pd.read_csv(climate_input / climate_name, parse_dates=["date"])
    primary_macro = pd.read_csv(output_dir / "macro_primary_results.csv")
    control_macro = pd.read_csv(output_dir / "macro_control_results.csv")
    candidates = sorted(primary_macro["commodity"].astype(str))
    controls = sorted(control_macro["commodity"].astype(str))
    family_of = dict.fromkeys(candidates, "mechanism_candidate")
    family_of.update(dict.fromkeys(controls, "negative_control"))

    reference_index = str(stage_config["contract"]["reference_treatment"])
    grid = enso.loc[:, ["date", reference_index]].copy()
    grid = grid.loc[grid[reference_index].notna()].reset_index(drop=True)
    reference = grid[reference_index]

    tidy_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    def record(outcome: TreatmentOutcome, *, treatment: str, kind: str, correlation: float) -> None:
        if outcome.candidates.empty:
            summaries.append(
                {
                    "treatment": treatment,
                    "kind": kind,
                    "episodes": 0,
                    "correlation_with_primary_index": correlation,
                    **dict.fromkeys(GATE_COUNTS, 0),
                }
            )
            return
        tidy_frames.append(_tidy(outcome, treatment=treatment, family_of=family_of))
        summaries.append(
            {
                "treatment": treatment,
                "kind": kind,
                "episodes": len(outcome.episodes),
                "correlation_with_primary_index": correlation,
                **_gate_counts(outcome),
            }
        )

    # Reference cell: the substitute-treatment code path fed the primary index.
    reference_frame = _treatment_frame(grid["date"], reference)
    reference_outcome = _evaluate(
        reference_frame,
        label=f"reference:{reference_index}",
        inputs=inputs,
        candidates=candidates,
        controls=controls,
        config=config,
        spec=spec,
        stage_config=stage_config,
    )
    record(reference_outcome, treatment=reference_index, kind="reference", correlation=1.0)
    reconstruction = reference_outcome.candidates.loc[:, ["commodity", "mean_return"]].merge(
        primary_macro.loc[:, ["commodity", "macro_mean_return"]], on="commodity", how="inner"
    )
    reconstruction_error = float(
        reconstruction["mean_return"].sub(reconstruction["macro_mean_return"]).abs().max()
    )
    if reconstruction_error > 1e-10:
        raise ValueError(
            f"Reference treatment does not reproduce the frozen estimates: {reconstruction_error}"
        )

    for name in stage_config["external_treatments"]:
        if name not in climate.columns:
            raise ValueError(f"Climate index {name!r} is not in the processed climate panel")
        aligned = grid.merge(
            climate.loc[:, ["date", name]], on="date", how="left", validate="one_to_one"
        )
        matched = moment_match(aligned[name], reference)
        correlation = float(aligned[name].corr(reference))
        outcome = _evaluate(
            _treatment_frame(aligned["date"], matched),
            label=f"external:{name}",
            inputs=inputs,
            candidates=candidates,
            controls=controls,
            config=config,
            spec=spec,
            stage_config=stage_config,
        )
        record(outcome, treatment=name, kind="external_climate_index", correlation=correlation)

    surrogate_config = stage_config["surrogates"]
    replicates = int(surrogate_config["replicates"])
    for variant in surrogate_config["variants"]:
        rng = np.random.default_rng(salted_seed(config.random_seed, f"surrogate:{variant}"))
        for replicate in range(replicates):
            label = f"{variant}_{replicate:04d}"
            values = phase_randomized_surrogate(
                reference,
                rng=rng,
                preserve_seasonality=variant == "seasonal_preserving",
            )
            outcome = _evaluate(
                _treatment_frame(grid["date"], values),
                label=label,
                inputs=inputs,
                candidates=candidates,
                controls=controls,
                config=config,
                spec=spec,
                stage_config=stage_config,
            )
            record(
                outcome,
                treatment=label,
                kind=variant,
                correlation=float(values.corr(reference)),
            )

    results = pd.concat(tidy_frames, ignore_index=True)
    summary_table = pd.DataFrame.from_records(summaries)
    surrogate_kinds = set(surrogate_config["variants"])
    surrogate_summary = summary_table.loc[summary_table["kind"].isin(surrogate_kinds)]
    observed = summary_table.loc[summary_table["kind"].eq("reference")].iloc[0]

    gate_null = pd.DataFrame.from_records(
        [
            {
                "statistic": statistic,
                "observed": int(observed[statistic]),
                "surrogate_mean": float(surrogate_summary[statistic].mean()),
                "surrogate_median": float(surrogate_summary[statistic].median()),
                "surrogate_maximum": int(surrogate_summary[statistic].max()),
                "surrogates_at_least_observed": int(
                    (surrogate_summary[statistic] >= int(observed[statistic])).sum()
                ),
                "surrogates": len(surrogate_summary),
                "finite_sample_p_value": _finite_sample_p(
                    float(observed[statistic]), surrogate_summary[statistic]
                ),
            }
            for statistic in GATE_COUNTS
        ]
    )

    surrogate_results = results.loc[results["treatment"].isin(surrogate_summary["treatment"])]
    reference_results = results.loc[results["treatment"].eq(reference_index)]
    rate_records: list[dict[str, Any]] = []
    for commodity, group in surrogate_results.groupby("commodity", sort=True, observed=True):
        real = reference_results.loc[reference_results["commodity"].eq(commodity)]
        real_row = real.iloc[0]
        for gate in ("reject_bootstrap_raw", "reject_placebo_raw", "passes_gates"):
            rate_records.append(
                {
                    "commodity": commodity,
                    "registry_family": str(real_row["registry_family"]),
                    "gate": gate,
                    "observed_under_primary_index": bool(real_row[gate]),
                    "surrogate_rate": float(group[gate].mean()),
                    "surrogates": len(group),
                    "finite_sample_p_value": _finite_sample_p(
                        float(bool(real_row[gate])), group[gate].astype(float)
                    ),
                }
            )
    commodity_rates = pd.DataFrame.from_records(rate_records)

    external_summary = summary_table.loc[summary_table["kind"].eq("external_climate_index")]
    paths = {
        "results": output_dir / "surrogate_treatment_results.csv",
        "summary": output_dir / "surrogate_treatment_summary.csv",
        "gate_null": output_dir / "surrogate_treatment_gate_null.csv",
        "commodity_rates": output_dir / "surrogate_treatment_commodity_rates.csv",
    }
    results.sort_values(["treatment", "family", "commodity"]).to_csv(paths["results"], index=False)
    summary_table.to_csv(paths["summary"], index=False)
    gate_null.to_csv(paths["gate_null"], index=False)
    commodity_rates.to_csv(paths["commodity_rates"], index=False)

    control_rates = commodity_rates.loc[commodity_rates["registry_family"].eq("negative_control")]
    control_bootstrap_rates = control_rates.loc[control_rates["gate"].eq("reject_bootstrap_raw")]
    control_placebo_rates = control_rates.loc[control_rates["gate"].eq("reject_placebo_raw")]
    summary: dict[str, Any] = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "status": "exploratory",
        "contract": {
            "role": stage_config["contract"]["role"],
            "changes_frozen_gates": False,
            "reference_treatment": reference_index,
            "external_treatments": list(stage_config["external_treatments"]),
            "surrogate_variants": list(surrogate_config["variants"]),
            "surrogate_replicates": replicates,
            "bootstrap_replicates": int(stage_config["inference"]["bootstrap_replicates"]),
            "placebo_replicates": int(stage_config["inference"]["placebo_replicates"]),
            "gates": [
                "eligible",
                "sign_agreement",
                "bootstrap_fdr",
                "placebo_fdr",
            ],
        },
        "diagnostics": {
            "reference_mean_reconstruction_max_abs_difference": reconstruction_error,
            "reference_episodes": int(observed["episodes"]),
            "surrogate_cells": len(surrogate_summary),
            "surrogate_median_episodes": float(surrogate_summary["episodes"].median()),
            "external_cells": len(external_summary),
            "external_maximum_candidates_passing_gates": int(
                external_summary["candidates_passing_gates"].max()
            ),
            "external_maximum_controls_rejecting_bootstrap_raw": int(
                external_summary["controls_rejecting_bootstrap_raw"].max()
            ),
            "mean_control_bootstrap_rejection_rate_under_surrogates": float(
                control_bootstrap_rates["surrogate_rate"].mean()
            ),
            "mean_control_placebo_rejection_rate_under_surrogates": float(
                control_placebo_rates["surrogate_rate"].mean()
            ),
            "maximum_control_placebo_rejection_rate_under_surrogates": float(
                control_placebo_rates["surrogate_rate"].max()
            ),
            **{
                f"{row.statistic}_finite_sample_p_value": float(row.finite_sample_p_value)
                for row in gate_null.itertuples()
            },
        },
        "input_hashes": {
            enso_name: sha256_file(snapshot / enso_name),
            market_name: sha256_file(snapshot / market_name),
            monthly_name: sha256_file(output_dir / monthly_name),
            macro_name: sha256_file(macro_input / macro_name),
            climate_name: sha256_file(climate_input / climate_name),
            "macro_primary_results.csv": sha256_file(output_dir / "macro_primary_results.csv"),
            "macro_control_results.csv": sha256_file(output_dir / "macro_control_results.csv"),
            "research.yaml": sha256_file(research_path),
            "surrogate_treatment.yaml": sha256_file(stage_path),
        },
        "output_hashes": {path.name: sha256_file(path) for path in paths.values()},
        "random_seed": config.random_seed,
    }
    write_json_atomic(output_dir / "surrogate_treatment_run_summary.json", summary)
    return output_dir
