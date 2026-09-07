"""Frozen v2 validation contract and conservative evidence labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import project_root


@dataclass(frozen=True)
class ValidationContract:
    version: int
    discovery_snapshot: str
    commodities: tuple[str, ...]
    forecast_horizons: tuple[int, ...]
    forecast_origin_frequency: int
    forecast_minimum_training_months: int
    transaction_cost_bps: float
    timing_indexes: tuple[str, ...]
    timing_anchors: tuple[str, ...]
    timing_horizons: tuple[int, ...]
    timing_outcomes: tuple[str, ...]
    minimum_shift_years: int
    maximum_shift_years: int
    equivalence_bounds: dict[str, float]


def load_validation_contract(path: Path | None = None) -> ValidationContract:
    contract_path = path or project_root() / "config" / "validation_v2.yaml"
    with contract_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    forecast = raw["forecast"]
    timing = raw["program_timing_null"]
    interpretation = raw["interpretation"]
    commodities = tuple(str(value) for value in raw["commodities"])
    contract = ValidationContract(
        version=int(raw["contract_version"]),
        discovery_snapshot=str(raw["frozen_after_discovery_snapshot"]),
        commodities=commodities,
        forecast_horizons=tuple(int(value) for value in forecast["horizons_months"]),
        forecast_origin_frequency=int(forecast["origin_frequency_months"]),
        forecast_minimum_training_months=int(forecast["minimum_training_months"]),
        transaction_cost_bps=float(forecast["proxy_transaction_cost_bps"]),
        timing_indexes=tuple(str(value) for value in timing["indexes"]),
        timing_anchors=tuple(str(value) for value in timing["anchors"]),
        timing_horizons=tuple(int(value) for value in timing["horizons_months"]),
        timing_outcomes=tuple(str(value) for value in timing["outcomes"]),
        minimum_shift_years=int(timing["minimum_shift_years"]),
        maximum_shift_years=int(timing["maximum_shift_years"]),
        equivalence_bounds={
            str(name): float(value)
            for name, value in interpretation["equivalence_bounds_absolute_return"].items()
        },
    )
    if contract.version != 2:
        raise ValueError("validation contract_version must be 2")
    if len(contract.commodities) != len(set(contract.commodities)) or not contract.commodities:
        raise ValueError("validation commodities must be unique and non-empty")
    if set(contract.equivalence_bounds) != set(contract.commodities):
        raise ValueError("every validation commodity must have exactly one equivalence bound")
    if any(value <= 0 for value in contract.equivalence_bounds.values()):
        raise ValueError("equivalence bounds must be positive")
    if not set(contract.timing_indexes).issubset({"roni", "oni"}):
        raise ValueError("timing indexes must be RONI or ONI")
    if not set(contract.timing_anchors).issubset({"retrospective", "observable"}):
        raise ValueError("unsupported timing anchor")
    allowed_outcomes = {"seasonal_adjusted_log_return", "market_adjusted_log_return"}
    if not set(contract.timing_outcomes).issubset(allowed_outcomes):
        raise ValueError("unsupported timing-null outcome")
    if contract.minimum_shift_years < 1 or contract.maximum_shift_years < contract.minimum_shift_years:
        raise ValueError("invalid timing-null shift range")
    if contract.forecast_minimum_training_months < 24:
        raise ValueError("forecast minimum training window is too short")
    return contract


def evidence_label(
    *,
    robust_association: bool,
    confidence_lower: float,
    confidence_upper: float,
    equivalence_bound: float,
    below_mde: bool,
) -> str:
    """Separate evidence of absence from a design that could not detect the effect."""
    if robust_association:
        return "robust_historical_association"
    if confidence_lower > -equivalence_bound and confidence_upper < equivalence_bound:
        return "evidence_against_material_effect"
    if below_mde:
        return "inconclusive_underpowered"
    return "inconclusive"
