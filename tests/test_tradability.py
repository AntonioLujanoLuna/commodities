from __future__ import annotations

import pandas as pd
import pytest

from enso_commodities.tradability import continuous_front_contract_returns, validate_futures_panel


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03", "2025-01-03"],
            "commodity": ["Palm oil"] * 3,
            "contract": ["H25", "H25", "K25"],
            "settlement_price": [100.0, 101.0, 102.0],
            "volume": [100, 90, 80],
            "open_interest": [200, 190, 180],
            "days_to_expiry": [20, 19, 70],
            "information_date": ["2025-01-02", "2025-01-03", "2025-01-03"],
        }
    )


def test_front_contract_returns_keep_rolls_explicit() -> None:
    result = continuous_front_contract_returns(_panel(), roll_days=10)
    assert len(result) == 2
    assert result.iloc[1]["log_return"] > 0


def test_futures_contract_rejects_lookahead() -> None:
    data = _panel()
    data.loc[0, "information_date"] = "2025-02-01"
    with pytest.raises(ValueError, match="not available"):
        validate_futures_panel(data)
