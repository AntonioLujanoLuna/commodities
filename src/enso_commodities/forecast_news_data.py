"""Acquire the issue-dated CPC/IRI probabilistic ENSO forecast archive.

The source is a sequence of official IRI Quick Look pages, not a reconstructed
forecast series. Each raw HTML page is retained and hashed. The parser reads the
explicit publication date and the CPC/IRI early-month consensus table; it never
derives probabilities from a chart or from the model plume.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from lxml import html

from .config import project_root
from .download import USER_AGENT
from .forecast_news import validate_forecast_archive
from .provenance import sha256_file, write_json_atomic

ARCHIVE_FILE = "forecast_probabilities.csv"
PUBLISHED_PATTERN = re.compile(r"Published:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})")
SEASON_CENTRE_MONTH = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}


@dataclass(frozen=True)
class ForecastArchiveSource:
    config: dict[str, Any]
    name: str
    label: str
    landing_page: str
    url_template: str
    first_issue_month: pd.Period
    last_issue_month: pd.Period
    expected_issue_count: int

    def issue_months(self) -> pd.PeriodIndex:
        return pd.period_range(self.first_issue_month, self.last_issue_month, freq="M")

    def url(self, issue_month: pd.Period) -> str:
        return self.url_template.format(
            year=issue_month.year,
            month=issue_month.month,
            month_name=issue_month.start_time.strftime("%B").lower(),
        )


def load_forecast_archive_source(path: Path | None = None) -> ForecastArchiveSource:
    config_path = path or project_root() / "config" / "forecast_news_sources.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    if config.get("scope") != "verified_public_monthly_html_archive":
        raise ValueError("Forecast-news source is not a verified public issuance archive")
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValueError("Forecast-news source configuration has no source block")
    if source.get("table_identity") != "official_cpc_iri_early_month_consensus":
        raise ValueError("Forecast-news source is not the frozen CPC/IRI consensus table")
    first = pd.Period(str(source["first_issue_month"]), freq="M")
    last = pd.Period(str(source["last_issue_month"]), freq="M")
    count = int(source["expected_issue_count"])
    if last < first or len(pd.period_range(first, last, freq="M")) != count:
        raise ValueError("Forecast-news issue range and expected count disagree")
    return ForecastArchiveSource(
        config=config,
        name=str(source["name"]),
        label=str(source["label"]),
        landing_page=str(source["landing_page"]),
        url_template=str(source["url_template"]),
        first_issue_month=first,
        last_issue_month=last,
        expected_issue_count=count,
    )


def _normalise(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _probability(value: str) -> float:
    token = _normalise(value).replace("%", "").replace("~", "")
    if not token:
        raise ValueError("Forecast probability is empty")
    if token.startswith("<"):
        return float(token[1:]) / 200.0
    if token.startswith(">"):
        return (float(token[1:]) + 100.0) / 200.0
    numeric = float(token)
    if numeric < 0 or numeric > 100:
        raise ValueError(f"Forecast probability is outside 0-100: {value!r}")
    return numeric / 100.0


def _target_centre(issue_date: pd.Timestamp, season: str) -> pd.Timestamp:
    code = _normalise(season).upper()[:3]
    if code not in SEASON_CENTRE_MONTH:
        raise ValueError(f"Unknown forecast season {season!r}")
    month = SEASON_CENTRE_MONTH[code]
    previous_month = 12 if issue_date.month == 1 else issue_date.month - 1
    if month == previous_month:
        year = issue_date.year - int(issue_date.month == 1)
    else:
        year = issue_date.year + int(month < issue_date.month)
    return pd.Timestamp(year=year, month=month, day=1)


def _published_date(value: str) -> pd.Timestamp:
    # Literal typo on the archived February 2018 Quick Look page. The requested
    # page, numeric day and year are unambiguous; preserve the HTML and normalize
    # only this observed spelling rather than accepting arbitrary fuzzy dates.
    value = value.replace("Februrary", "February")
    for date_format in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return pd.Timestamp(datetime.strptime(value, date_format).date())
        except ValueError:
            continue
    raise ValueError(f"CPC forecast publication date is malformed: {value!r}")


def _issue_date(
    document: Any, anchor: Any, expected_issue_month: pd.Period
) -> tuple[pd.Timestamp, str]:
    preceding = anchor.xpath("preceding::h4[1]")
    if not preceding:
        raise ValueError("CPC forecast table has no explicit publication date")
    match = PUBLISHED_PATTERN.search(_normalise(preceding[0].text_content()))
    if not match:
        raise ValueError("CPC forecast publication date is malformed")
    section_date = _published_date(match.group(1))
    if section_date.to_period("M") == expected_issue_month:
        return section_date, "cpc_section"

    # Some migrated pages carry a stale year in the CPC section. The page-level
    # Quick Look date is an explicit conservative availability date: the table
    # was publicly present no later than that date. This fallback never derives
    # a day from the URL and still requires the independently printed month to
    # match the requested archive page.
    for page_heading in document.xpath("//h2"):
        if "quick look" not in _normalise(page_heading.text_content()).lower():
            continue
        page_dates = page_heading.xpath("following::h4[1]")
        if not page_dates:
            continue
        page_match = PUBLISHED_PATTERN.search(_normalise(page_dates[0].text_content()))
        if page_match:
            page_date = _published_date(page_match.group(1))
            if page_date.to_period("M") == expected_issue_month:
                return page_date, "quick_look_page_fallback"
    raise ValueError(
        f"Published month {section_date:%Y-%m} does not match requested {expected_issue_month}"
    )


def parse_consensus_page(content: bytes, *, expected_issue_month: pd.Period) -> pd.DataFrame:
    """Parse one archived page and bind it to its explicit publication month."""
    document = html.fromstring(content)
    candidates = []
    for heading in document.xpath("//h3"):
        title = _normalise(heading.text_content()).lower()
        identifies_table = (
            "forecast probabilit" in title or "probabilistic enso forecast" in title
        )
        if "cpc" in title and identifies_table:
            candidates.append(heading)
    tables: list[Any]
    if len(candidates) == 1:
        tables = candidates[0].xpath("following::table[1]")
    elif not candidates:
        tables = document.xpath(
            "//table[contains(concat(' ', normalize-space(@class), ' '), ' ensoprob ')]"
        )
    else:
        raise ValueError(
            f"Expected one CPC forecast-probability table, found {len(candidates)}"
        )
    if len(tables) != 1:
        raise ValueError(f"Expected one CPC forecast-probability table, found {len(tables)}")
    issue_date, date_source = _issue_date(document, tables[0], expected_issue_month)
    rows: list[dict[str, Any]] = []
    for row in tables[0].xpath(".//tr"):
        cells = [_normalise(cell.text_content()) for cell in row.xpath("./th|./td")]
        if not cells or cells[0].lower() == "season":
            continue
        if len(cells) != 4:
            raise ValueError(f"CPC forecast row does not have four cells: {cells}")
        target = _target_centre(issue_date, cells[0])
        rows.append(
            {
                "issue_date": issue_date,
                "target_center_date": target,
                "lead_months": (target.year - issue_date.year) * 12
                + target.month
                - issue_date.month,
                "probability_el_nino": _probability(cells[3]),
                "probability_neutral": _probability(cells[2]),
                "probability_la_nina": _probability(cells[1]),
                "source_season_label": cells[0],
                "publication_date_source": date_source,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    # A few official tables retain the just-completed overlapping season as
    # their first row. It was already partly observed when the forecast was
    # issued and the frozen archive contract forbids targets before issuance,
    # so retain the source label in the raw HTML but exclude that row here.
    frame = frame.loc[frame["lead_months"].ge(0)].reset_index(drop=True)
    if len(frame) not in {8, 9}:
        raise ValueError(f"CPC forecast table yields {len(frame)} forward seasons; expected 8 or 9")
    if frame["lead_months"].tolist() != list(range(len(frame))):
        raise ValueError("CPC forecast seasons are not consecutive forward target centres")
    validate_forecast_archive(frame)
    return frame


def _raw_root() -> Path:
    return project_root() / "data" / "forecast_news" / "raw"


def latest_raw_forecast_snapshot(root: Path | None = None) -> Path:
    source = root or _raw_root()
    candidates = sorted(path for path in source.glob("????-??-??") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No raw forecast-news snapshots found under {source}")
    return candidates[-1]


def download_forecast_archive(
    snapshot_date: date | None = None,
    *,
    raw_root: Path | None = None,
    config_path: Path | None = None,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download and hash every issue page in the frozen consecutive range."""
    source = load_forecast_archive_source(config_path)
    snapshot = (snapshot_date or date.today()).isoformat()
    destination = (raw_root or _raw_root()) / snapshot
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            existing: dict[str, Any] = yaml.safe_load(handle)
        config_file = config_path or project_root() / "config" / "forecast_news_sources.yaml"
        if existing.get("source_config_sha256") != sha256_file(config_file):
            raise ValueError("Existing forecast snapshot uses a different source contract")
        pages = existing.get("pages")
        if not isinstance(pages, list) or len(pages) != source.expected_issue_count:
            raise ValueError("Existing forecast snapshot has an incomplete manifest")
        for item, issue_month in zip(pages, source.issue_months(), strict=True):
            path = destination / str(item["filename"])
            if str(item.get("issue_month")) != str(issue_month) or sha256_file(path) != item.get(
                "sha256"
            ):
                raise ValueError(f"Existing forecast snapshot failed verification at {issue_month}")
            parse_consensus_page(path.read_bytes(), expected_issue_month=issue_month)
        return destination
    receipts: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    for issue_month in source.issue_months():
        url = source.url(issue_month)
        response = session.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        content = response.content
        parse_consensus_page(content, expected_issue_month=issue_month)
        filename = f"{issue_month}.html"
        path = destination / filename
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination, prefix=f".{filename}.", suffix=".part"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
            digest = sha256_file(temporary)
            if path.exists():
                if sha256_file(path) != digest:
                    raise FileExistsError(f"Refusing to overwrite changed issuance page {path}")
                temporary.unlink()
            else:
                os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        receipts.append(
            {
                "bytes": path.stat().st_size,
                "content_type": response.headers.get("Content-Type"),
                "etag": response.headers.get("ETag"),
                "filename": filename,
                "issue_month": str(issue_month),
                "last_modified": response.headers.get("Last-Modified"),
                "sha256": digest,
                "url": url,
            }
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": "real",
        "issuance_dates_as_published": True,
        "snapshot": snapshot,
        "source": source.name,
        "source_config_sha256": sha256_file(
            config_path or project_root() / "config" / "forecast_news_sources.yaml"
        ),
        "pages": receipts,
    }
    write_json_atomic(manifest_path, manifest)
    return destination


def build_forecast_archive(
    snapshot_dir: Path | None = None,
    *,
    processed_root: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    """Verify raw issue pages and build the strict row-level probability archive."""
    source = load_forecast_archive_source(config_path)
    snapshot = snapshot_dir or latest_raw_forecast_snapshot()
    manifest_path = snapshot / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: dict[str, Any] = yaml.safe_load(handle)
    if manifest.get("data_provenance") != "real" or not manifest.get(
        "issuance_dates_as_published"
    ):
        raise ValueError("Raw forecast archive does not certify published issuance dates")
    source_config = config_path or project_root() / "config" / "forecast_news_sources.yaml"
    if manifest.get("source_config_sha256") != sha256_file(source_config):
        raise ValueError("Raw forecast archive was acquired under a different source contract")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or len(pages) != source.expected_issue_count:
        raise ValueError("Raw forecast archive does not contain the frozen issue range")
    by_month = {str(item.get("issue_month")): item for item in pages}
    expected_months = [str(month) for month in source.issue_months()]
    if list(by_month) != expected_months:
        raise ValueError("Raw forecast issue months are missing, duplicated, or out of order")

    frames: list[pd.DataFrame] = []
    for issue_month in source.issue_months():
        receipt = by_month[str(issue_month)]
        path = snapshot / str(receipt["filename"])
        if sha256_file(path) != receipt.get("sha256"):
            raise ValueError(f"Raw forecast hash mismatch for {path.name}")
        frames.append(parse_consensus_page(path.read_bytes(), expected_issue_month=issue_month))
    archive = pd.concat(frames, ignore_index=True)
    validate_forecast_archive(archive)

    output = (
        processed_root or project_root() / "data" / "forecast_news" / "processed"
    ) / snapshot.name
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / ARCHIVE_FILE
    archive.to_csv(archive_path, index=False, date_format="%Y-%m-%d")
    summary = {
        "data_provenance": "real",
        "snapshot": snapshot.name,
        "source": source.name,
        "source_label": source.label,
        "source_landing_page": source.landing_page,
        "issuance_dates_as_published": True,
        "coverage": {
            "first_issue_date": archive["issue_date"].min().date().isoformat(),
            "last_issue_date": archive["issue_date"].max().date().isoformat(),
            "issue_months": int(archive["issue_date"].dt.to_period("M").nunique()),
            "forecast_rows": len(archive),
            "publication_date_sources": {
                str(key): int(value)
                for key, value in archive.drop_duplicates("issue_date")[
                    "publication_date_source"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "input_hashes": {
            "forecast_news_sources.yaml": sha256_file(source_config),
            "manifest.json": sha256_file(manifest_path),
        },
        "output_hashes": {ARCHIVE_FILE: sha256_file(archive_path)},
    }
    write_json_atomic(output / "summary.json", summary)
    return output
