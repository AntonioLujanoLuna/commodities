from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enso_commodities.disruption_chain import (
    DisruptionInputs,
    chain_is_complete,
    fit_disruption_chain,
    load_disruption_contract,
    validate_disruption_inputs,
)

CONTRACT = load_disruption_contract()
COAL = "Coal, Australian"


def _inputs(
    *,
    rainfall_effect: float = 0.8,
    throughput_effect: float = -0.6,
    price_effect: float = -0.5,
    seed: int = 3,
) -> DisruptionInputs:
    generator = np.random.default_rng(seed)
    months = pd.date_range("1990-01-01", periods=360, freq="MS")
    regions = ["bowen_basin", "hunter_valley"]

    rainfall_rows = []
    for region in regions:
        enso = generator.normal(0, 1, len(months))
        anomaly = rainfall_effect * enso + generator.normal(0, 0.5, len(months))
        rainfall_rows.append(
            pd.DataFrame(
                {
                    "commodity": COAL,
                    "region": region,
                    "date": months,
                    "enso": enso,
                    "rainfall_anomaly": anomaly,
                    "cluster": months.year,
                }
            )
        )
    rainfall = pd.concat(rainfall_rows, ignore_index=True)

    throughput_rows = []
    for index, region in enumerate(regions):
        shock = generator.normal(0, 1, len(months))
        share = 0.6 if index == 0 else 0.4
        log_throughput = throughput_effect * shock * share + generator.normal(0, 0.3, len(months))
        throughput_rows.append(
            pd.DataFrame(
                {
                    "commodity": COAL,
                    "region": region,
                    "date": months,
                    "log_throughput": log_throughput,
                    "rainfall_shock": shock,
                    "capacity_share": share,
                    "cluster": months.year,
                }
            )
        )
    throughput = pd.concat(throughput_rows, ignore_index=True)

    surprise = generator.normal(0, 1, len(months))
    shipments = pd.DataFrame(
        {
            "commodity": COAL,
            "information_date": months,
            "price_date": months + pd.DateOffset(days=20),
            "throughput_surprise": surprise,
            "price_return": price_effect * surprise + generator.normal(0, 0.3, len(months)),
            "cluster": months.year,
        }
    )
    return DisruptionInputs(rainfall=rainfall, throughput=throughput, shipments=shipments)


def test_contract_registers_three_signed_links() -> None:
    assert CONTRACT.expected_directions == {
        "cold_phase_to_regional_rainfall": "positive",
        "rainfall_to_throughput": "negative",
        "throughput_shortfall_to_price": "negative",
    }
    assert CONTRACT.commodities == (COAL,)


def test_validation_rejects_information_dated_after_the_price() -> None:
    inputs = _inputs()
    shipments = inputs.shipments.copy()
    shipments.loc[0, "information_date"] = shipments.loc[0, "price_date"] + pd.DateOffset(days=1)
    broken = DisruptionInputs(inputs.rainfall, inputs.throughput, shipments)
    with pytest.raises(ValueError, match="dated after the measured price"):
        validate_disruption_inputs(broken, CONTRACT)


def test_validation_rejects_capacity_shares_outside_the_unit_interval() -> None:
    inputs = _inputs()
    throughput = inputs.throughput.copy()
    throughput.loc[0, "capacity_share"] = 1.4
    broken = DisruptionInputs(inputs.rainfall, throughput, inputs.shipments)
    with pytest.raises(ValueError, match="capacity_share"):
        validate_disruption_inputs(broken, CONTRACT)


def test_validation_rejects_a_missing_column() -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="rainfall input is missing columns"):
        validate_disruption_inputs(
            DisruptionInputs(
                inputs.rainfall.drop(columns=["rainfall_anomaly"]),
                inputs.throughput,
                inputs.shipments,
            ),
            CONTRACT,
        )


def test_a_complete_chain_passes_every_registered_link() -> None:
    results = fit_disruption_chain(_inputs(), CONTRACT)
    assert set(results["link"]) == set(CONTRACT.expected_directions)
    assert bool(results["passes_link"].all())
    assert chain_is_complete(results, CONTRACT)


def test_a_link_with_the_wrong_sign_breaks_the_chain() -> None:
    """A partial chain is reported incomplete, never as partial support."""
    results = fit_disruption_chain(_inputs(throughput_effect=0.6), CONTRACT)
    broken = results.set_index("link").loc["rainfall_to_throughput"]
    assert not bool(broken["direction_matches"])
    assert not bool(broken["passes_link"])
    assert not chain_is_complete(results, CONTRACT)


def test_an_absent_link_is_not_a_complete_chain() -> None:
    results = fit_disruption_chain(_inputs(), CONTRACT)
    assert not chain_is_complete(
        results.loc[results["link"].ne("throughput_shortfall_to_price")], CONTRACT
    )
    assert not chain_is_complete(results.iloc[0:0], CONTRACT)
