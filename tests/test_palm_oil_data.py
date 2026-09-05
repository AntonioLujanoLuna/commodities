from pathlib import Path

import pytest

from enso_commodities.palm_oil_data import parse_power_monthly


def test_parse_power_monthly_converts_daily_precipitation_to_monthly_total(
    tmp_path: Path,
) -> None:
    path = tmp_path / "power.csv"
    path.write_text(
        "-BEGIN HEADER-\nmetadata\n-END HEADER-\n"
        "PARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN\n"
        "PRECTOTCORR,2000,1,1,1,1,1,1,1,1,1,1,1,1,1\n"
        "T2M,2000,25,25,25,25,25,25,25,25,25,25,25,25,25\n",
        encoding="utf-8",
    )
    result = parse_power_monthly(path, point="test", country="Malaysia")
    january_rain = result.loc[
        result["parameter"].eq("PRECTOTCORR") & result["date"].dt.month.eq(1), "value"
    ].iloc[0]
    february_rain = result.loc[
        result["parameter"].eq("PRECTOTCORR") & result["date"].dt.month.eq(2), "value"
    ].iloc[0]
    assert january_rain == pytest.approx(31.0)
    assert february_rain == pytest.approx(29.0)
