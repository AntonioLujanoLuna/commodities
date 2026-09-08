"""Strict interface for the W2 cold-phase supply-disruption chain.

The palm-oil chain failed at the supply-to-price link, and it failed there partly
because that link had a handful of annual observations behind it. The chain this
module estimates is the same shape and a far better-measured one: La Nina raises
eastern Australian rainfall, rainfall cuts pit output and rail and port loading,
and a throughput shortfall raises the price. The intended intermediates are
monthly and publicly recorded, but source coverage and release vintages still
have to pass the separate readiness audit.

Those inputs are not in this repository. This module therefore validates and
estimates; it never imputes, and it refuses to report a link with no data behind
it. `config/cold_phase_mechanism.yaml` freezes the schema so the estimand is
fixed before anyone goes looking for the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from .config import project_root
from .statistics import benjamini_hochberg


@dataclass(frozen=True)
class DisruptionInputs:
    rainfall: pd.DataFrame
    throughput: pd.DataFrame
    shipments: pd.DataFrame


@dataclass(frozen=True)
class DisruptionContract:
    config: dict[str, Any]
    commodities: tuple[str, ...]
    required_columns: dict[str, set[str]]
    expected_directions: dict[str, str]
    minimum_clusters: int
    fdr_alpha: float


def load_disruption_contract(path: Path | None = None) -> DisruptionContract:
    config_path = path or project_root() / "config" / "cold_phase_mechanism.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["mechanism"] != "cold_phase_supply_disruption":
        raise ValueError("Not the cold-phase disruption contract")
    requirements = raw["requirements"]
    if not requirements["rainfall_source_must_be_external"]:
        raise ValueError("Rainfall inputs must come from an external source")
    if not requirements["throughput_must_precede_measured_price"]:
        raise ValueError("Throughput information must precede the price it is matched to")
    if not raw["inference"]["require_all_links_for_complete_chain"]:
        raise ValueError("A partial chain is never reported as partial support")
    return DisruptionContract(
        config=raw,
        commodities=tuple(str(name) for name in raw["commodities"]),
        required_columns={
            name: set(table["required_columns"]) for name, table in raw["tables"].items()
        },
        expected_directions={
            str(link["name"]): str(link["expected_direction"]) for link in raw["links"]
        },
        minimum_clusters=int(requirements["minimum_clusters_per_link"]),
        fdr_alpha=float(raw["inference"]["fdr_alpha"]),
    )


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} input is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} input is empty")


def validate_disruption_inputs(inputs: DisruptionInputs, contract: DisruptionContract) -> None:
    """Reject anything the contract does not allow, before a coefficient exists."""
    _require(inputs.rainfall, contract.required_columns["rainfall"], "rainfall")
    _require(inputs.throughput, contract.required_columns["throughput"], "throughput")
    _require(inputs.shipments, contract.required_columns["shipments"], "shipments")

    shipments = inputs.shipments.copy()
    shipments["information_date"] = pd.to_datetime(shipments["information_date"])
    shipments["price_date"] = pd.to_datetime(shipments["price_date"])
    if shipments["information_date"].gt(shipments["price_date"]).any():
        raise ValueError("shipments contain throughput information dated after the measured price")

    shares = pd.to_numeric(inputs.throughput["capacity_share"], errors="coerce")
    if shares.isna().any() or shares.lt(0).any() or shares.gt(1).any():
        raise ValueError("capacity_share must fall in [0, 1]")

    for label, frame in (
        ("rainfall", inputs.rainfall),
        ("throughput", inputs.throughput),
        ("shipments", inputs.shipments),
    ):
        counts = frame.groupby("commodity", observed=True)["cluster"].nunique()
        if counts.lt(contract.minimum_clusters).any():
            failed = counts.loc[counts.lt(contract.minimum_clusters)].index.tolist()
            raise ValueError(
                f"{label} input has fewer than {contract.minimum_clusters} clusters for {failed}"
            )


def _cluster_fit(
    data: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    controls: tuple[str, ...],
    cluster: str,
) -> dict[str, float | int]:
    columns = [outcome, exposure, cluster, *controls]
    sample = data.dropna(subset=columns).copy()
    design = sm.add_constant(sample.loc[:, [exposure, *controls]], has_constant="add")
    fitted = sm.OLS(sample[outcome].astype(float), design.astype(float)).fit(
        cov_type="cluster", cov_kwds={"groups": sample[cluster]}
    )
    return {
        "observations": len(sample),
        "clusters": int(sample[cluster].nunique()),
        "coefficient": float(fitted.params[exposure]),
        "standard_error": float(fitted.bse[exposure]),
        "p_value": float(fitted.pvalues[exposure]),
    }


def fit_disruption_chain(inputs: DisruptionInputs, contract: DisruptionContract) -> pd.DataFrame:
    """Estimate the three registered links, each in its registered direction."""
    validate_disruption_inputs(inputs, contract)
    commodities = sorted(
        set(inputs.rainfall["commodity"])
        & set(inputs.throughput["commodity"])
        & set(inputs.shipments["commodity"])
    )
    if not commodities:
        raise ValueError("disruption inputs have no commodity in common")

    rows: list[dict[str, Any]] = []
    for commodity in commodities:
        rainfall = inputs.rainfall.loc[inputs.rainfall["commodity"].eq(commodity)].copy()
        rainfall["date"] = pd.to_datetime(rainfall["date"])
        rainfall["calendar_month"] = rainfall["date"].dt.month
        rainfall = pd.get_dummies(
            rainfall, columns=["region", "calendar_month"], drop_first=True, dtype=float
        )
        rainfall_controls = tuple(
            column for column in rainfall if column.startswith(("region_", "calendar_month_"))
        )
        rows.append(
            {
                "commodity": commodity,
                "link": "cold_phase_to_regional_rainfall",
                "expected_direction": contract.expected_directions[
                    "cold_phase_to_regional_rainfall"
                ],
                **_cluster_fit(
                    rainfall,
                    outcome="rainfall_anomaly",
                    exposure="enso",
                    controls=rainfall_controls,
                    cluster="cluster",
                ),
            }
        )

        throughput = inputs.throughput.loc[inputs.throughput["commodity"].eq(commodity)].copy()
        throughput["date"] = pd.to_datetime(throughput["date"])
        throughput["calendar_month"] = throughput["date"].dt.month
        throughput["weighted_rainfall_shock"] = (
            throughput["rainfall_shock"] * throughput["capacity_share"]
        )
        throughput = pd.get_dummies(
            throughput, columns=["region", "calendar_month"], drop_first=True, dtype=float
        )
        throughput_controls = tuple(
            column for column in throughput if column.startswith(("region_", "calendar_month_"))
        )
        rows.append(
            {
                "commodity": commodity,
                "link": "rainfall_to_throughput",
                "expected_direction": contract.expected_directions["rainfall_to_throughput"],
                **_cluster_fit(
                    throughput,
                    outcome="log_throughput",
                    exposure="weighted_rainfall_shock",
                    controls=throughput_controls,
                    cluster="cluster",
                ),
            }
        )

        shipments = inputs.shipments.loc[inputs.shipments["commodity"].eq(commodity)].copy()
        rows.append(
            {
                "commodity": commodity,
                "link": "throughput_shortfall_to_price",
                "expected_direction": contract.expected_directions["throughput_shortfall_to_price"],
                **_cluster_fit(
                    shipments,
                    outcome="price_return",
                    exposure="throughput_surprise",
                    controls=(),
                    cluster="cluster",
                ),
            }
        )

    results = pd.DataFrame.from_records(rows)
    results["q_value"] = benjamini_hochberg(results["p_value"])
    results["direction_matches"] = np.where(
        results["expected_direction"].eq("negative"),
        results["coefficient"].lt(0),
        np.where(
            results["expected_direction"].eq("positive"),
            results["coefficient"].gt(0),
            results["coefficient"].ne(0),
        ),
    )
    results["passes_link"] = results["direction_matches"] & results["q_value"].le(
        contract.fdr_alpha
    )
    return results


def chain_is_complete(results: pd.DataFrame, contract: DisruptionContract) -> bool:
    """Every registered link must pass, for every commodity, or the chain is incomplete."""
    if results.empty:
        return False
    expected = set(contract.expected_directions)
    for _, group in results.groupby("commodity", observed=True):
        if set(group["link"]) != expected or not bool(group["passes_link"].all()):
            return False
    return True
