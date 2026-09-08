"""Real-data orchestration for the W3 forecast-revision news study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_root
from .dispersion import load_program_register
from .dispersion_analysis import commodity_roles
from .forecast_news import build_revision_series, load_news_config, run_news_inference
from .forecast_news_data import ARCHIVE_FILE
from .provenance import sha256_file, verify_hashes, write_json_atomic
from .raw_events import latest_processed_snapshot


def latest_forecast_archive(root: Path | None = None) -> Path:
    """Locate the newest processed issuance archive, or explain why there is none.

    The stage cannot fall back to a current forecast file or a reconstructed
    series: the design rests on each probability having been available on its
    stated date, so a back-filled one would make every coefficient a look-ahead
    artifact. Failing loudly here is the point.
    """
    source = root or project_root() / "data" / "forecast_news" / "processed"
    candidates = (
        sorted(path for path in source.glob("????-??-??") if path.is_dir())
        if source.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(
            f"No archived ENSO forecast issuances under {source}. "
            "run the verified forecast-news download and build stages first. W3 will not run "
            "on a reconstructed or revised probability series."
        )
    return candidates[-1]


def run_news_analysis(
    processed_snapshot: Path | None = None,
    *,
    archive_snapshot: Path | None = None,
    tables_root: Path | None = None,
    config_path: Path | None = None,
    program_config_path: Path | None = None,
) -> Path:
    snapshot = processed_snapshot or latest_processed_snapshot()
    output = (tables_root or project_root() / "tables") / snapshot.name
    universe_summary_path = output / "universe_summary.json"
    adjusted_summary_path = output / "adjusted_event_summary.json"
    summaries: dict[str, dict[str, Any]] = {}
    for path in (universe_summary_path, adjusted_summary_path):
        with path.open(encoding="utf-8") as handle:
            summaries[path.name] = json.load(handle)
    if any(item.get("data_provenance") != "real" for item in summaries.values()):
        raise ValueError("News analysis requires real-data receipts")

    archive_dir = archive_snapshot or latest_forecast_archive()
    archive_summary_path = archive_dir / "summary.json"
    with archive_summary_path.open(encoding="utf-8") as handle:
        archive_summary: dict[str, Any] = json.load(handle)
    if archive_summary.get("data_provenance") != "real":
        raise ValueError("Forecast archive is not marked as real data")
    if not archive_summary.get("issuance_dates_as_published"):
        raise ValueError(
            "Forecast archive does not certify its issuance dates as published; W3 refuses "
            "to treat a revised probability as information available on its stated date"
        )
    verify_hashes(archive_dir, {ARCHIVE_FILE: archive_summary["output_hashes"][ARCHIVE_FILE]})

    inputs = {
        "commodity_returns_adjusted_monthly.parquet": summaries["adjusted_event_summary.json"][
            "output_hashes"
        ]["commodity_returns_adjusted_monthly.parquet"],
        "primary_inference_family.parquet": summaries["universe_summary.json"]["output_hashes"][
            "primary_inference_family.parquet"
        ],
        "negative_control_sample.parquet": summaries["universe_summary.json"]["output_hashes"][
            "negative_control_sample.parquet"
        ],
    }
    verify_hashes(output, inputs)

    spec = load_news_config(config_path)
    program = load_program_register(program_config_path)
    archive = pd.read_csv(
        archive_dir / ARCHIVE_FILE, parse_dates=["issue_date", "target_center_date"]
    )
    revisions = build_revision_series(
        archive,
        lead_months=spec.primary_lead_months,
        probability_column=spec.source_probability,
    )
    if len(revisions) < spec.minimum_issuances:
        raise ValueError(
            f"W3 requires at least {spec.minimum_issuances} monthly revisions; "
            f"the archive yields {len(revisions)}"
        )

    returns = pd.read_parquet(output / "commodity_returns_adjusted_monthly.parquet")
    roles = commodity_roles(output)
    # The primary-window frame is merged back into `results` with its q-values, so
    # the separate copy is not written out again.
    results, _primary, statistics = run_news_inference(
        revisions,
        returns,
        roles,
        spec=spec,
        program_threshold=program.threshold,
    )

    sensitivity_rows: list[pd.DataFrame] = []
    sensitivity_revision_counts: dict[str, int] = {}
    sensitivity_leads_run: list[int] = []
    for lead in spec.sensitivity_lead_months:
        alternative = build_revision_series(
            archive, lead_months=lead, probability_column=spec.source_probability
        )
        sensitivity_revision_counts[str(lead)] = len(alternative)
        if len(alternative) < spec.minimum_issuances:
            continue
        sensitivity_leads_run.append(lead)
        _, lead_primary, _ = run_news_inference(
            alternative, returns, roles, spec=spec, program_threshold=program.threshold
        )
        sensitivity_rows.append(lead_primary.assign(lead_months=lead))
    sensitivity = (
        pd.concat(sensitivity_rows, ignore_index=True) if sensitivity_rows else pd.DataFrame()
    )
    statistics["sensitivity_revision_counts"] = sensitivity_revision_counts
    statistics["sensitivity_leads_run"] = sensitivity_leads_run
    statistics["sensitivity_leads_unavailable"] = [
        lead for lead in spec.sensitivity_lead_months if lead not in sensitivity_leads_run
    ]

    results_path = output / "forecast_news_results.csv"
    revisions_path = output / "forecast_news_revisions.csv"
    sensitivity_path = output / "forecast_news_lead_sensitivity.csv"
    results.to_csv(results_path, index=False)
    revisions.to_csv(revisions_path, index=False, date_format="%Y-%m-%d")
    sensitivity.to_csv(sensitivity_path, index=False)

    config_file = config_path or project_root() / "config" / "forecast_news.yaml"
    program_file = program_config_path or project_root() / "config" / "findings_v3.yaml"
    source_config_file = project_root() / "config" / "forecast_news_sources.yaml"
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "forecast_archive_snapshot": archive_dir.name,
        "design": {
            **spec.config["treatment"],
            **spec.config["design"],
            "placebos": spec.config["placebos"],
            "inference_scope": spec.config["provenance"]["inference_scope"],
            "workstream": spec.config["provenance"]["workstream"],
            "reference_distribution": spec.config["inference"]["reference_distribution"],
        },
        "results": statistics,
        "input_hashes": {
            **{name: sha256_file(output / name) for name in inputs},
            ARCHIVE_FILE: sha256_file(archive_dir / ARCHIVE_FILE),
            "forecast_archive_summary.json": sha256_file(archive_summary_path),
            "forecast_news.yaml": sha256_file(config_file),
            "forecast_news_sources.yaml": sha256_file(source_config_file),
            "findings_v3.yaml": sha256_file(program_file),
            **{name: sha256_file(output / name) for name in summaries},
        },
        "output_hashes": {
            results_path.name: sha256_file(results_path),
            revisions_path.name: sha256_file(revisions_path),
            sensitivity_path.name: sha256_file(sensitivity_path),
        },
        "random_seed": spec.random_seed,
    }
    write_json_atomic(output / "forecast_news_summary.json", summary)
    return output
