import math

import pandas as pd
import pytest

from enso_commodities.endpoint_diagnostics import add_endpoint_shape, endpoint_shape_summary


def test_endpoint_shape_exposes_jensen_gap() -> None:
    endpoints = pd.DataFrame(
        {
            "index_definition": ["roni", "roni"],
            "anchor_type": ["retrospective", "retrospective"],
            "direction": ["warm", "warm"],
            "commodity": ["Gold", "Gold"],
            "anchor_date": ["2000-01-01", "2001-01-01"],
            "value": [1.0, -0.5],
        }
    )
    shaped = add_endpoint_shape(endpoints)
    summary = endpoint_shape_summary(shaped).iloc[0]
    assert shaped["log_value"].tolist() == pytest.approx([math.log(2), math.log(0.5)])
    assert summary["arithmetic_mean_return"] == pytest.approx(0.25)
    assert summary["geometric_equivalent_return"] == pytest.approx(0.0)
    assert summary["jensen_gap"] == pytest.approx(0.25)
