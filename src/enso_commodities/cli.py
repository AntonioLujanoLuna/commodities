from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adjusted_events import build_adjusted_event_tables
from .cold_phase_analysis import run_cold_phase_analysis
from .config import project_root
from .dataset import build_real_dataset
from .dispersion_analysis import run_dispersion_analysis
from .disruption_sources import audit_disruption_source_readiness
from .dose_response_analysis import run_dose_response_analysis
from .download import download_all
from .endpoint_analysis import run_endpoint_diagnostics
from .external_exposure import build_external_exposure_weights, download_exposure_data
from .financial_analysis import run_financial_control_analysis
from .financial_data import build_financial_dataset
from .flavour_analysis import run_flavour_analysis
from .flavour_data import build_flavour_dataset, download_flavour_snapshot
from .forecast_analysis import run_forecast_analysis
from .forecast_news_data import build_forecast_archive, download_forecast_archive
from .fragility_analysis import run_leave_one_episode_out
from .inference import run_primary_inference
from .macro_analysis import run_macro_control_analysis
from .macro_data import build_macro_dataset
from .news_analysis import run_news_analysis
from .palm_oil_data import build_palm_oil_dataset
from .palm_oil_mechanism import run_palm_oil_mechanism
from .panel_analysis import run_panel_analysis
from .placebo_analysis import run_neutral_date_placebo
from .power_analysis import run_power_analysis
from .program_timing_analysis import run_program_timing_null
from .publication import build_publication_bundle, verify_publication_bundle
from .raw_events import build_raw_event_tables
from .reporting import build_current_results
from .robustness_analysis import run_timing_index_robustness
from .specification_curve_analysis import run_specification_curve
from .specificity_analysis import run_specificity_diagnostics
from .universe import build_inference_universe


def download() -> None:
    parser = argparse.ArgumentParser(description="Download a dated real-data snapshot.")
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    snapshot = download_all(raw_root=args.raw_root)
    print(snapshot)


def build() -> None:
    parser = argparse.ArgumentParser(description="Build the real monthly ENSO/commodity panel.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--processed-root", type=Path)
    args = parser.parse_args()
    output = build_real_dataset(args.snapshot, processed_root=args.processed_root)
    print(output)


def analyse() -> None:
    parser = argparse.ArgumentParser(description="Build implemented ENSO event-return tables.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--research-config", type=Path)
    args = parser.parse_args()
    build_raw_event_tables(
        args.processed_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    output = build_adjusted_event_tables(
        args.processed_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    output = build_inference_universe(
        args.processed_snapshot,
        tables_root=args.tables_root,
    )
    output = run_primary_inference(
        args.processed_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    output = run_neutral_date_placebo(
        args.processed_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    print(output)


def figures() -> None:
    parser = argparse.ArgumentParser(description="Build the scorecard and publication figures.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(
        build_publication_bundle(
            processed_snapshot=args.processed_snapshot,
            tables_root=args.tables_root,
            output_root=args.output_root,
        )
    )


def macro_build() -> None:
    parser = argparse.ArgumentParser(description="Build the real monthly macro-control panel.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--processed-root", type=Path)
    args = parser.parse_args()
    output = build_macro_dataset(args.snapshot, processed_root=args.processed_root)
    print(output)


def macro_analyse() -> None:
    parser = argparse.ArgumentParser(description="Run external macro-control inference.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--macro-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--research-config", type=Path)
    args = parser.parse_args()
    output = run_macro_control_analysis(
        args.processed_snapshot,
        args.macro_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    print(output)


def fragility() -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-episode-out fragility inference.")
    parser.add_argument("--tables-snapshot", type=Path)
    parser.add_argument("--research-config", type=Path)
    args = parser.parse_args()
    output = run_leave_one_episode_out(
        tables_snapshot=args.tables_snapshot,
        research_config_path=args.research_config,
    )
    print(output)


def robustness() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RONI/ONI and retrospective/observable robustness grid."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--macro-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--research-config", type=Path)
    args = parser.parse_args()
    output = run_timing_index_robustness(
        processed_snapshot=args.processed_snapshot,
        macro_snapshot=args.macro_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    print(output)


def specificity() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory warm-versus-cold diagnostics.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--specificity-config", type=Path)
    args = parser.parse_args()
    output = run_specificity_diagnostics(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        specificity_config_path=args.specificity_config,
    )
    print(output)


def endpoint_diagnostics() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory endpoint diagnostics.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--endpoint-config", type=Path)
    args = parser.parse_args()
    output = run_endpoint_diagnostics(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        endpoint_config_path=args.endpoint_config,
    )
    print(output)


def financial_download() -> None:
    parser = argparse.ArgumentParser(description="Download a dated real financial-data snapshot.")
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    output = download_all(
        raw_root=args.raw_root or project_root() / "data" / "financial" / "raw",
        config_path=project_root() / "config" / "financial_sources.yaml",
    )
    print(output)


def financial_build() -> None:
    parser = argparse.ArgumentParser(description="Build the real monthly financial-control panel.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--macro-snapshot", type=Path)
    parser.add_argument("--processed-root", type=Path)
    args = parser.parse_args()
    output = build_financial_dataset(
        args.snapshot, args.macro_snapshot, processed_root=args.processed_root
    )
    print(output)


def financial_analyse() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory financial-control inference.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--macro-snapshot", type=Path)
    parser.add_argument("--financial-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = run_financial_control_analysis(
        args.processed_snapshot,
        args.macro_snapshot,
        args.financial_snapshot,
        tables_root=args.tables_root,
        config_path=args.config,
    )
    print(output)


def palm_download() -> None:
    parser = argparse.ArgumentParser(description="Download the real palm-oil mechanism inputs.")
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    output = download_all(
        raw_root=args.raw_root or project_root() / "data" / "mechanisms" / "palm_oil" / "raw",
        config_path=project_root() / "config" / "palm_oil_sources.yaml",
    )
    print(output)


def palm_build() -> None:
    parser = argparse.ArgumentParser(description="Build the real palm-oil mechanism panel.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = build_palm_oil_dataset(
        args.snapshot,
        processed_root=args.processed_root,
        mechanism_config_path=args.config,
    )
    print(output)


def palm_analyse() -> None:
    parser = argparse.ArgumentParser(description="Run the exploratory palm-oil mechanism chain.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--palm-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = run_palm_oil_mechanism(
        args.processed_snapshot,
        args.palm_snapshot,
        tables_root=args.tables_root,
        mechanism_config_path=args.config,
    )
    print(output)


def report() -> None:
    parser = argparse.ArgumentParser(
        description="Render the generated blocks of the current-results note from run receipts."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--published-bundle",
        type=Path,
        help="Use a verified compact publication bundle when full local tables are unavailable.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the note does not already match the receipts, instead of rewriting it.",
    )
    args = parser.parse_args()
    if args.published_bundle:
        verify_publication_bundle(args.published_bundle)
        args.processed_snapshot = args.published_bundle
        args.tables_root = args.published_bundle.parent
    output = build_current_results(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        report_path=args.report,
        check=args.check,
        verify_outputs=not bool(args.published_bundle),
    )
    print(output)


def publication() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the compact auditable results bundle."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(
        build_publication_bundle(
            processed_snapshot=args.processed_snapshot,
            tables_root=args.tables_root,
            output_root=args.output_root,
            check=args.check,
            bundle=args.bundle,
        )
    )


def power() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the minimum detectable effect at the frozen endpoint."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--research-config", type=Path)
    args = parser.parse_args()
    output = run_power_analysis(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        research_config_path=args.research_config,
    )
    print(output)


def dose_response() -> None:
    parser = argparse.ArgumentParser(
        description="Run the secondary warm-episode amplitude dose-response diagnostic."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = run_dose_response_analysis(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        config_path=args.config,
    )
    print(output)


def panel() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exploratory exposure-weighted fixed-effects panel."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    output = run_panel_analysis(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        registry_path=args.registry,
        panel_config_path=args.config,
    )
    print(output)


def specification_curve() -> None:
    parser = argparse.ArgumentParser(
        description="Run the whole-year circular-shift panel specification curve."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--panel-config", type=Path)
    parser.add_argument("--curve-config", type=Path)
    args = parser.parse_args()
    output = run_specification_curve(
        processed_snapshot=args.processed_snapshot,
        tables_root=args.tables_root,
        registry_path=args.registry,
        panel_config_path=args.panel_config,
        curve_config_path=args.curve_config,
    )
    print(output)


def forecast() -> None:
    parser = argparse.ArgumentParser(
        description="Run the expanding-window pseudo-out-of-sample forecast benchmark."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--validation-config", type=Path)
    args = parser.parse_args()
    print(
        run_forecast_analysis(
            processed_snapshot=args.processed_snapshot,
            tables_root=args.tables_root,
            validation_config_path=args.validation_config,
        )
    )


def program_timing_null() -> None:
    parser = argparse.ArgumentParser(
        description="Run the circular whole-year null for the locked v2 event-study family."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument("--research-config", type=Path)
    args = parser.parse_args()
    print(
        run_program_timing_null(
            processed_snapshot=args.processed_snapshot,
            tables_root=args.tables_root,
            validation_config_path=args.validation_config,
            research_config_path=args.research_config,
        )
    )


def exposure_download() -> None:
    parser = argparse.ArgumentParser(description="Download external physical-exposure rasters.")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(download_exposure_data(raw_root=args.raw_root, config_path=args.config))


def exposure_build() -> None:
    parser = argparse.ArgumentParser(description="Build outcome-independent exposure weights.")
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--raw-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    print(
        build_external_exposure_weights(
            processed_snapshot=args.processed_snapshot,
            raw_snapshot=args.raw_snapshot,
            tables_root=args.tables_root,
            config_path=args.config,
            registry_path=args.registry,
        )
    )


def dispersion() -> None:
    parser = argparse.ArgumentParser(
        description="Run the W1 dispersion endpoint against the circular whole-year shift null."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--program-config", type=Path)
    args = parser.parse_args()
    print(
        run_dispersion_analysis(
            args.processed_snapshot,
            tables_root=args.tables_root,
            config_path=args.config,
            program_config_path=args.program_config,
        )
    )


def cold_phase() -> None:
    parser = argparse.ArgumentParser(
        description="Run the W2 cold-phase disruption endpoint with signed hypotheses."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--program-config", type=Path)
    args = parser.parse_args()
    print(
        run_cold_phase_analysis(
            args.processed_snapshot,
            tables_root=args.tables_root,
            config_path=args.config,
            program_config_path=args.program_config,
        )
    )


def forecast_news() -> None:
    parser = argparse.ArgumentParser(
        description="Run the W3 forecast-revision news study and its lead and lag placebos."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--archive-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--program-config", type=Path)
    args = parser.parse_args()
    print(
        run_news_analysis(
            args.processed_snapshot,
            archive_snapshot=args.archive_snapshot,
            tables_root=args.tables_root,
            config_path=args.config,
            program_config_path=args.program_config,
        )
    )


def disruption_readiness() -> None:
    parser = argparse.ArgumentParser(
        description="Audit public-source readiness for the W2 physical disruption chain."
    )
    parser.add_argument("--source-config", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_disruption_source_readiness(args.source_config), indent=2))


def forecast_news_download() -> None:
    parser = argparse.ArgumentParser(
        description="Download the verified issue-dated CPC/IRI ENSO probability archive."
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--source-config", type=Path)
    args = parser.parse_args()
    print(download_forecast_archive(raw_root=args.raw_root, config_path=args.source_config))


def forecast_news_build() -> None:
    parser = argparse.ArgumentParser(
        description="Build the strict row-level CPC/IRI forecast-probability archive."
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--source-config", type=Path)
    args = parser.parse_args()
    print(
        build_forecast_archive(
            args.snapshot,
            processed_root=args.processed_root,
            config_path=args.source_config,
        )
    )


def flavour_download() -> None:
    parser = argparse.ArgumentParser(description="Download Nino 3 and Nino 4 region indices.")
    parser.parse_args()
    print(download_flavour_snapshot())


def flavour_build() -> None:
    parser = argparse.ArgumentParser(description="Build the monthly Nino 3/Nino 4 panel.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--processed-root", type=Path)
    args = parser.parse_args()
    print(build_flavour_dataset(args.snapshot, processed_root=args.processed_root))


def flavour() -> None:
    parser = argparse.ArgumentParser(
        description="Run the W5 Eastern/Central Pacific flavour diagnostic."
    )
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument("--flavour-snapshot", type=Path)
    parser.add_argument("--tables-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--program-config", type=Path)
    args = parser.parse_args()
    print(
        run_flavour_analysis(
            args.processed_snapshot,
            flavour_snapshot=args.flavour_snapshot,
            tables_root=args.tables_root,
            config_path=args.config,
            program_config_path=args.program_config,
        )
    )
