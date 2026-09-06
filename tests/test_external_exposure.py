from __future__ import annotations

import numpy as np
import pytest

from enso_commodities.external_exposure import crop_weight, load_exposure_config


def test_crop_weight_uses_area_and_positive_drought_differences_only() -> None:
    area = np.array([[1.0, 3.0], [6.0, np.nan]])
    drought = np.array([[0.2, -0.1], [0.5, 1.0]])
    weight, coverage, total = crop_weight(area, drought)
    assert weight == pytest.approx((1 * 0.2 + 6 * 0.5) / 10)
    assert coverage == pytest.approx(0.7)
    assert total == 10.0


def test_shipped_external_contract_partitions_candidates() -> None:
    spec = load_exposure_config()
    assert len(spec.commodity_crop_codes) == 21
    assert len(spec.excluded_candidates) == 11
    assert set(spec.commodity_crop_codes).isdisjoint(spec.excluded_candidates)
