from __future__ import annotations

import argparse
from pathlib import Path

from .adjusted_events import build_adjusted_event_tables
from .config import project_root
from .dataset import build_real_dataset
from .download import download_all
from .endpoint_analysis import run_endpoint_diagnostics
from .financial_analysis import run_financial_control_analysis
from .financial_data import build_financial_dataset
from .fragility_analysis import run_leave_one_episode_out
from .inference import run_primary_inference
from .macro_analysis import run_macro_control_analysis
from .macro_data import build_macro_dataset
from .palm_oil_data import build_palm_oil_dataset
from .palm_oil_mechanism import run_palm_oil_mechanism
from .placebo_analysis import run_neutral_date_placebo
from .raw_events import build_raw_event_tables
from .robustness_analysis import run_timing_index_robustness
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
    raise SystemExit("The figure stage is not implemented yet.")


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
