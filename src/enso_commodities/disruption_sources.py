"""Readiness audit for public inputs to the W2 disruption chain.

The audit is intentionally separate from acquisition. It makes source failures
executable without turning a current historical table into a vintage archive or
weakening the frozen cluster requirement to fit the data that happened to be
available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import project_root
from .disruption_chain import DisruptionContract, load_disruption_contract


def load_disruption_source_audit(path: Path | None = None) -> dict[str, Any]:
    config_path = path or project_root() / "config" / "cold_phase_mechanism_sources.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw.get("scope") != "verified_public_source_readiness_audit":
        raise ValueError("Not the W2 public-source readiness audit")
    if raw.get("cluster_unit") != "calendar_year":
        raise ValueError("W2 source readiness must count calendar-year clusters")
    sources = raw.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("W2 source readiness audit has no candidate sources")
    return raw


def _year_clusters(source: dict[str, Any]) -> int:
    first = pd.Period(str(source["first_observation"]), freq="M")
    last = pd.Period(str(source["last_observation"]), freq="M")
    if last < first:
        raise ValueError("Source coverage ends before it begins")
    return last.year - first.year + 1


def _base_eligible(source: dict[str, Any], *, role: str) -> bool:
    return bool(
        source.get("role") == role
        and source.get("frequency") == "monthly"
        and source.get("external") is True
        and source.get("machine_readable") is True
        and source.get("acquisition_reproducible") is True
    )


def audit_disruption_source_readiness(
    path: Path | None = None,
    *,
    contract: DisruptionContract | None = None,
) -> dict[str, Any]:
    """Return link-level readiness and blockers; never estimate a partial chain."""
    raw = load_disruption_source_audit(path)
    frozen = contract or load_disruption_contract()
    sources: dict[str, dict[str, Any]] = raw["sources"]

    diagnostics: dict[str, dict[str, Any]] = {}
    for name, source in sources.items():
        diagnostic: dict[str, Any] = {
            "role": source.get("role"),
            "acquisition_reproducible": source.get("acquisition_reproducible") is True,
        }
        if source.get("role") in {"rainfall", "throughput"}:
            diagnostic["calendar_year_clusters"] = _year_clusters(source)
        if source.get("role") == "throughput":
            diagnostic["historical_publication_dates"] = (
                source.get("historical_publication_dates") is True
            )
        diagnostics[name] = diagnostic

    rainfall_ready = [
        name
        for name, source in sources.items()
        if _base_eligible(source, role="rainfall")
        and source.get("region_definition_frozen") is True
        and _year_clusters(source) >= frozen.minimum_clusters
    ]
    throughput_ready = [
        name
        for name, source in sources.items()
        if _base_eligible(source, role="throughput")
        and _year_clusters(source) >= frozen.minimum_clusters
    ]
    vintage_throughput_ready = [
        name
        for name in throughput_ready
        if sources[name].get("historical_publication_dates") is True
    ]
    price_ready = [name for name, source in sources.items() if _base_eligible(source, role="price")]

    links = {
        "cold_phase_to_regional_rainfall": bool(rainfall_ready),
        "rainfall_to_throughput": bool(rainfall_ready and throughput_ready),
        "throughput_shortfall_to_price": bool(vintage_throughput_ready and price_ready),
    }
    blockers: list[str] = []
    if not rainfall_ready:
        blockers.append("no reproducible monthly rainfall source has a frozen regional extraction")
    if not throughput_ready:
        blockers.append(
            "no reproducible monthly throughput source reaches the frozen "
            f"{frozen.minimum_clusters} calendar-year clusters"
        )
    if not vintage_throughput_ready:
        blockers.append("no eligible throughput source preserves historical publication dates")
    if not price_ready:
        blockers.append("no reproducible monthly price source is registered")

    complete = all(links.values())
    return {
        "workstream": "W2_cold_phase_disruption_chain",
        "verified_on": str(raw["verified_on"]),
        "status": "ready_to_acquire" if complete else "blocked_source_contract",
        "minimum_calendar_year_clusters": frozen.minimum_clusters,
        "complete_chain_ready": complete,
        "links_ready": links,
        "eligible_sources": {
            "rainfall": rainfall_ready,
            "throughput": throughput_ready,
            "vintage_throughput": vintage_throughput_ready,
            "price": price_ready,
        },
        "source_diagnostics": diagnostics,
        "blockers": blockers,
    }
