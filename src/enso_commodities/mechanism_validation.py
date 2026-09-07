"""Regional crop-calendar mechanism estimators with strict provenance-shaped inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .statistics import benjamini_hochberg


@dataclass(frozen=True)
class MechanismInputs:
    weather: pd.DataFrame
    yields: pd.DataFrame
    supply_revisions: pd.DataFrame


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} input is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} input is empty")


def validate_mechanism_inputs(inputs: MechanismInputs, *, minimum_clusters: int = 10) -> None:
    _require(
        inputs.weather,
        {"commodity", "region", "date", "enso", "weather_anomaly", "active_season", "cluster"},
        "weather",
    )
    _require(
        inputs.yields,
        {
            "commodity",
            "region",
            "harvest_year",
            "yield_surprise",
            "weather_shock",
            "production_share",
            "cluster",
        },
        "yield",
    )
    _require(
        inputs.supply_revisions,
        {"commodity", "information_date", "price_date", "supply_revision", "price_return", "cluster"},
        "supply revision",
    )
    weather = inputs.weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    revisions = inputs.supply_revisions.copy()
    revisions["information_date"] = pd.to_datetime(revisions["information_date"])
    revisions["price_date"] = pd.to_datetime(revisions["price_date"])
    if revisions["information_date"].gt(revisions["price_date"]).any():
        raise ValueError("supply revisions contain information dated after the measured price")
    if not weather["active_season"].isin([0, 1, False, True]).all():
        raise ValueError("active_season must be binary")
    shares = pd.to_numeric(inputs.yields["production_share"], errors="coerce")
    if shares.isna().any() or shares.lt(0).any() or shares.gt(1).any():
        raise ValueError("production_share must fall in [0, 1]")
    for label, frame in (
        ("weather", inputs.weather),
        ("yield", inputs.yields),
        ("supply revision", inputs.supply_revisions),
    ):
        counts = frame.groupby("commodity", observed=True)["cluster"].nunique()
        if counts.lt(minimum_clusters).any():
            failed = counts.loc[counts.lt(minimum_clusters)].index.tolist()
            raise ValueError(f"{label} input has fewer than {minimum_clusters} clusters for {failed}")


def _cluster_fit(
    data: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    controls: tuple[str, ...],
    cluster: str,
) -> dict[str, float | int]:
    columns = [outcome, exposure, cluster, *controls]
    sample = data.dropna(subset=columns).copy()
    design = sm.add_constant(sample.loc[:, [exposure, *controls]], has_constant="add")
    fitted = sm.OLS(sample[outcome].astype(float), design.astype(float)).fit(
        cov_type="cluster", cov_kwds={"groups": sample[cluster]}
    )
    return {
        "observations": len(sample),
        "clusters": int(sample[cluster].nunique()),
        "coefficient": float(fitted.params[exposure]),
        "standard_error": float(fitted.bse[exposure]),
        "p_value": float(fitted.pvalues[exposure]),
    }


def fit_mechanism_chain(inputs: MechanismInputs, *, minimum_clusters: int = 10) -> pd.DataFrame:
    validate_mechanism_inputs(inputs, minimum_clusters=minimum_clusters)
    rows: list[dict[str, object]] = []
    commodities = sorted(
        set(inputs.weather["commodity"])
        & set(inputs.yields["commodity"])
        & set(inputs.supply_revisions["commodity"])
    )
    if not commodities:
        raise ValueError("mechanism inputs have no commodity in common")
    for commodity in commodities:
        weather = inputs.weather.loc[inputs.weather["commodity"].eq(commodity)].copy()
        weather["date"] = pd.to_datetime(weather["date"])
        weather["enso_active"] = weather["enso"] * weather["active_season"].astype(float)
        weather["calendar_month"] = weather["date"].dt.month
        weather = pd.get_dummies(
            weather, columns=["region", "calendar_month"], drop_first=True, dtype=float
        )
        # Calendar-month effects absorb the active-season main effect because
        # the crop calendar is fixed before outcomes are observed.
        weather_controls = tuple(
            ["enso"]
            + [column for column in weather if column.startswith(("region_", "calendar_month_"))]
        )
        rows.append(
            {
                "commodity": commodity,
                "link": "enso_to_local_weather",
                "expected_direction": "nonzero",
                **_cluster_fit(
                    weather,
                    outcome="weather_anomaly",
                    exposure="enso_active",
                    controls=weather_controls,
                    cluster="cluster",
                ),
            }
        )

        yields = inputs.yields.loc[inputs.yields["commodity"].eq(commodity)].copy()
        yields["weighted_weather_shock"] = yields["weather_shock"] * yields["production_share"]
        yields["year_trend"] = yields["harvest_year"] - yields["harvest_year"].min()
        yields = pd.get_dummies(yields, columns=["region"], drop_first=True, dtype=float)
        yield_controls = tuple(
            ["year_trend"] + [column for column in yields if column.startswith("region_")]
        )
        rows.append(
            {
                "commodity": commodity,
                "link": "local_weather_to_yield_surprise",
                "expected_direction": "nonzero",
                **_cluster_fit(
                    yields,
                    outcome="yield_surprise",
                    exposure="weighted_weather_shock",
                    controls=yield_controls,
                    cluster="cluster",
                ),
            }
        )

        revisions = inputs.supply_revisions.loc[
            inputs.supply_revisions["commodity"].eq(commodity)
        ].copy()
        rows.append(
            {
                "commodity": commodity,
                "link": "supply_revision_to_price",
                "expected_direction": "negative",
                **_cluster_fit(
                    revisions,
                    outcome="price_return",
                    exposure="supply_revision",
                    controls=(),
                    cluster="cluster",
                ),
            }
        )
    results = pd.DataFrame.from_records(rows)
    results["q_value"] = benjamini_hochberg(results["p_value"])
    results["direction_matches"] = np.where(
        results["expected_direction"].eq("negative"),
        results["coefficient"].lt(0),
        results["coefficient"].ne(0),
    )
    results["passes_link"] = results["direction_matches"] & results["q_value"].le(0.05)
    return results
