from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enso_commodities.mechanism_validation import (
    MechanismInputs,
    fit_mechanism_chain,
    validate_mechanism_inputs,
)


def _inputs() -> MechanismInputs:
    generator = np.random.default_rng(12)
    weather_rows = []
    yield_rows = []
    revision_rows = []
    for year in range(2000, 2015):
        for region_index, region in enumerate(("north", "south")):
            for month in range(1, 13):
                enso = np.sin((year - 2000) + month / 12)
                active = int(month in {3, 4, 5, 6})
                weather_rows.append(
                    {
                        "commodity": "Palm oil",
                        "region": region,
                        "date": f"{year}-{month:02d}-01",
                        "enso": enso,
                        "weather_anomaly": 0.7 * enso * active + generator.normal(0, 0.2),
                        "active_season": active,
                        "cluster": year,
                    }
                )
            shock = generator.normal()
            yield_rows.append(
                {
                    "commodity": "Palm oil",
                    "region": region,
                    "harvest_year": year,
                    "yield_surprise": 0.3 * shock + generator.normal(0, 0.1),
                    "weather_shock": shock,
                    "production_share": 0.6 if region_index == 0 else 0.4,
                    "cluster": year,
                }
            )
        revision = generator.normal()
        revision_rows.append(
            {
                "commodity": "Palm oil",
                "information_date": f"{year}-06-01",
                "price_date": f"{year}-06-02",
                "supply_revision": revision,
                "price_return": -0.2 * revision + generator.normal(0, 0.05),
                "cluster": year,
            }
        )
    return MechanismInputs(
        weather=pd.DataFrame(weather_rows),
        yields=pd.DataFrame(yield_rows),
        supply_revisions=pd.DataFrame(revision_rows),
    )


def test_mechanism_chain_fits_three_timestamp_safe_links() -> None:
    results = fit_mechanism_chain(_inputs())
    assert set(results["link"]) == {
        "enso_to_local_weather",
        "local_weather_to_yield_surprise",
        "supply_revision_to_price",
    }
    assert results["clusters"].min() >= 10


def test_mechanism_rejects_future_information() -> None:
    inputs = _inputs()
    inputs.supply_revisions.loc[0, "information_date"] = "2030-01-01"
    with pytest.raises(ValueError, match="after the measured price"):
        validate_mechanism_inputs(inputs)

