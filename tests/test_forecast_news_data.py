from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from enso_commodities.forecast_news import validate_forecast_archive
from enso_commodities.forecast_news_data import (
    load_forecast_archive_source,
    parse_consensus_page,
)


def _page(
    *,
    published: str = "January 9, 2014",
    numeric: bool = False,
    seasons: tuple[str, ...] | None = None,
) -> bytes:
    values = (
        [("DJF", "2", "96", "2"), ("JFM", "4", "92", "4")]
        if numeric
        else [("DJF 2014", "~0%", "99.5%", "<1%"), ("JFM 2014", "4%", "92%", "4%")]
    )
    values.extend(
        [
            ("FMA", "5%", "85%", "10%"),
            ("MAM", "6%", "74%", "20%"),
            ("AMJ", "7%", "63%", "30%"),
            ("MJJ", "7%", "55%", "38%"),
            ("JJA", "7%", "51%", "42%"),
            ("JAS", "7%", "47%", "46%"),
            ("ASO", "8%", "45%", "47%"),
        ]
    )
    if seasons is not None:
        values = [
            (season, cold, neutral, warm)
            for season, (_, cold, neutral, warm) in zip(seasons, values, strict=True)
        ]
    rows = "".join(
        f"<tr><td>{season}</td><td>{cold}</td><td>{neutral}</td><td>{warm}</td></tr>"
        for season, cold, neutral, warm in values
    )
    return f"""
    <html><body>
      <h4>Published: {published}</h4>
      <h3>CPC/IRI Early-Month Consensus ENSO Forecast Probabilities</h3>
      <table>
        <tr><th>Season</th><th>La Nina</th><th>Neutral</th><th>El Nino</th></tr>
        {rows}
      </table>
    </body></html>
    """.encode()


def test_source_freezes_a_consecutive_machine_readable_range() -> None:
    source = load_forecast_archive_source()
    months = source.issue_months()
    assert str(months[0]) == "2014-01"
    assert str(months[-1]) == "2025-04"
    assert len(months) == source.expected_issue_count == 136
    assert source.url(months[0]).endswith("/2014-january-quick-look/")


def test_consensus_parser_uses_explicit_publication_date_and_target_order() -> None:
    parsed = parse_consensus_page(_page(), expected_issue_month=pd.Period("2014-01", freq="M"))
    assert parsed["issue_date"].nunique() == 1
    assert parsed.loc[0, "issue_date"] == pd.Timestamp("2014-01-09")
    assert parsed["target_center_date"].tolist() == list(
        pd.date_range("2014-01-01", periods=9, freq="MS")
    )
    assert parsed["lead_months"].tolist() == list(range(9))
    assert parsed.loc[0, "probability_el_nino"] == pytest.approx(0.005)
    validate_forecast_archive(parsed)


def test_consensus_parser_accepts_later_numeric_tables_without_inventing_a_year() -> None:
    parsed = parse_consensus_page(
        _page(published="January 11, 2024", numeric=True),
        expected_issue_month=pd.Period("2024-01", freq="M"),
    )
    assert parsed.loc[0, "target_center_date"] == pd.Timestamp("2024-01-01")
    assert parsed.loc[0, "probability_neutral"] == pytest.approx(0.96)


def test_consensus_parser_accepts_an_abbreviated_published_month() -> None:
    parsed = parse_consensus_page(
        _page(published="Aug 10, 2017", seasons=("JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA")),
        expected_issue_month=pd.Period("2017-08", freq="M"),
    )
    assert parsed.loc[0, "issue_date"] == pd.Timestamp("2017-08-10")


def test_consensus_parser_normalizes_the_archived_february_2018_typo() -> None:
    parsed = parse_consensus_page(
        _page(published="Februrary 08, 2018"),
        expected_issue_month=pd.Period("2018-02", freq="M"),
    )
    assert parsed.loc[0, "issue_date"] == pd.Timestamp("2018-02-08")


def test_consensus_parser_drops_a_just_completed_season() -> None:
    page = _page(
        published="September 8, 2016",
        seasons=("JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM"),
    )
    parsed = parse_consensus_page(
        page, expected_issue_month=pd.Period("2016-09", freq="M")
    )
    assert len(parsed) == 8
    assert parsed["target_center_date"].min() == pd.Timestamp("2016-09-01")
    assert parsed["lead_months"].tolist() == list(range(8))


def test_consensus_parser_refuses_a_page_without_a_machine_readable_table() -> None:
    page = b"<html><body><h4>Published: May 8, 2025</h4><img src='forecast.png'></body></html>"
    with pytest.raises(ValueError, match="found 0"):
        parse_consensus_page(page, expected_issue_month=pd.Period("2025-05", freq="M"))


def test_consensus_parser_refuses_a_mismatched_publication_month() -> None:
    with pytest.raises(ValueError, match="does not match requested"):
        parse_consensus_page(_page(), expected_issue_month=pd.Period("2014-02", freq="M"))


def test_consensus_parser_uses_an_explicit_page_date_when_the_section_year_is_stale() -> None:
    page = _page(published="January 9, 2019").decode().replace(
        "<html><body>",
        "<html><body><h2>2020 January Quick Look</h2><h4>Published: January 21, 2020</h4>",
    )
    parsed = parse_consensus_page(
        page.encode(), expected_issue_month=pd.Period("2020-01", freq="M")
    )
    assert parsed.loc[0, "issue_date"] == pd.Timestamp("2020-01-21")
    assert set(parsed["publication_date_source"]) == {"quick_look_page_fallback"}


def test_consensus_parser_accepts_the_2021_official_heading() -> None:
    page = _page(published="February 11, 2021").decode().replace(
        "CPC/IRI Early-Month Consensus ENSO Forecast Probabilities",
        "CPC/IRI Official Probabilistic ENSO Forecasts",
    )
    parsed = parse_consensus_page(
        page.encode(), expected_issue_month=pd.Period("2021-02", freq="M")
    )
    assert len(parsed) == 8


def test_consensus_parser_accepts_the_later_identified_table_without_a_heading() -> None:
    page = (
        _page(published="September 09, 2021", seasons=("JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM"))
        .decode()
        .replace("<h3>CPC/IRI Early-Month Consensus ENSO Forecast Probabilities</h3>", "")
        .replace("<table>", '<table class="ensoprob">')
    )
    parsed = parse_consensus_page(
        page.encode(), expected_issue_month=pd.Period("2021-09", freq="M")
    )
    assert len(parsed) == 8


def test_source_range_and_count_cannot_drift_apart(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config" / "forecast_news_sources.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["source"]["expected_issue_count"] = 135
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="range and expected count disagree"):
        load_forecast_archive_source(path)
