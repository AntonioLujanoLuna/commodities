"""Strict adapter for licensed futures data; price indexes never pass this gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_FUTURES_COLUMNS = {
    "date",
    "commodity",
    "contract",
    "settlement_price",
    "volume",
    "open_interest",
    "days_to_expiry",
    "information_date",
}


def validate_futures_panel(data: pd.DataFrame) -> None:
    missing = REQUIRED_FUTURES_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Futures panel is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Futures panel is empty")
    dates = pd.to_datetime(data["date"])
    information = pd.to_datetime(data["information_date"])
    if information.gt(dates).any():
        raise ValueError("Futures panel contains information not available on the trading date")
    prices = pd.to_numeric(data["settlement_price"], errors="coerce")
    if prices.isna().any() or prices.le(0).any():
        raise ValueError("settlement prices must be positive and complete")
    if data.duplicated(["date", "commodity", "contract"]).any():
        raise ValueError("Futures panel contains duplicate contract observations")


def continuous_front_contract_returns(data: pd.DataFrame, *, roll_days: int = 10) -> pd.DataFrame:
    """Build an auditable front-contract return series and expose every roll."""
    validate_futures_panel(data)
    if roll_days < 1:
        raise ValueError("roll_days must be positive")
    source = data.copy()
    source["date"] = pd.to_datetime(source["date"])
    source = source.loc[source["days_to_expiry"].ge(roll_days)].sort_values(
        ["commodity", "date", "days_to_expiry", "contract"]
    )
    selected = source.groupby(["commodity", "date"], observed=True).head(1).copy()
    selected = selected.sort_values(["commodity", "date"], ignore_index=True)
    grouped = selected.groupby("commodity", observed=True)
    selected["previous_contract"] = grouped["contract"].shift(1)
    selected["previous_price"] = grouped["settlement_price"].shift(1)
    selected["is_roll"] = selected["contract"].ne(selected["previous_contract"])
    selected["log_return"] = np.log(selected["settlement_price"] / selected["previous_price"])
    # Returns that cross contracts are not spliced: they require an explicit roll-yield calculation.
    selected.loc[selected["is_roll"], "log_return"] = np.nan
    return selected
