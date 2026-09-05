from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from enso_commodities.parsers import (
    parse_noaa_index,
    parse_pink_sheet,
    parse_pink_sheet_total_index,
)


def test_parse_noaa_roni_maps_seasons_to_center_month(tmp_path: Path) -> None:
    path = tmp_path / "roni.txt"
    rows = ["SEAS YR ANOM"]
    seasons = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    for year in range(1950, 2020):
        rows.extend(f"{season} {year} 0.25" for season in seasons)
    path.write_text("\n".join(rows), encoding="utf-8")

    parsed = parse_noaa_index(path, name="roni")

    assert len(parsed) == 840
    assert parsed.iloc[0]["date"] == pd.Timestamp("1950-01-01")
    assert parsed.iloc[11]["date"] == pd.Timestamp("1950-12-01")
    assert parsed["roni"].eq(0.25).all()


def test_parse_noaa_rejects_unknown_season(tmp_path: Path) -> None:
    path = tmp_path / "roni.txt"
    path.write_text("SEAS YR ANOM\nBAD 2000 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid seasons"):
        parse_noaa_index(path, name="roni")


def test_parse_pink_sheet_extracts_tidy_observations(tmp_path: Path) -> None:
    path = tmp_path / "pink.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Monthly Prices"
    sheet.append(["World Bank Commodity Price Data"])
    sheet.append(["monthly prices"])
    sheet.append(["nominal US dollars"])
    sheet.append(["Updated on January 01, 2026"])
    names = [None] + [f"Commodity {index}" for index in range(1, 51)]
    units = [None] + ["($/mt)"] * 50
    sheet.append(names)
    sheet.append(units)
    for offset, period in enumerate(pd.period_range("1960-01", periods=720, freq="M")):
        sheet.append(
            [f"{period.year}M{period.month:02d}"]
            + [float(offset + index + 1) for index in range(50)]
        )
    sheet["B7"] = 0
    indices = workbook.create_sheet("Monthly Indices")
    indices.append(["World Bank Commodity Price Data"])
    indices.append(["monthly indices"])
    indices.append(["nominal US dollars"])
    indices.append(["Updated on January 01, 2026"])
    indices.append([None])
    indices.append([None, "Total Index"])
    indices.append([None])
    indices.append([None])
    indices.append([None])
    for offset, period in enumerate(pd.period_range("1960-01", periods=720, freq="M")):
        indices.append([f"{period.year}M{period.month:02d}", 10.0 + offset])
    workbook.save(path)

    parsed, diagnostics = parse_pink_sheet(path)

    assert len(parsed) == 720 * 50
    assert diagnostics.months == 720
    assert diagnostics.commodities == 50
    assert diagnostics.first_period == "1960M01"
    assert diagnostics.non_positive_values == 1
    flagged = parsed.loc[parsed["quality_flag"] == "source_non_positive"].iloc[0]
    assert flagged["source_value"] == 0
    assert pd.isna(flagged["value"])

    market_index = parse_pink_sheet_total_index(path)
    assert len(market_index) == 720
    assert market_index.iloc[0]["date"] == pd.Timestamp("1960-01-01")
    assert market_index.iloc[-1]["world_bank_total_index"] == 729.0
