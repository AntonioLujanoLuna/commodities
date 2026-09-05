"""Synthetic fixtures for recovery and calibration tests.

Nothing here may be used to produce research output. The production downloader
and every processed stage refuse manifests that are not ``data_provenance:
real``; these generators exist so the pipeline can be pointed at data whose
answer is known in advance.

Two questions need different fixtures:

``synthetic_market`` plants a known abnormal return on known commodities and
emits the raw inputs the analytical path consumes, so a test can assert that
the pipeline recovers the planted number and leaves the unaffected series flat.

``synthetic_endpoint_panel`` skips the pipeline and emits an episode-by-commodity
endpoint panel with a shared factor, which is what the multiple-testing layer
sees. Repeated draws from it measure the empirical size of the inference gates
under the complete null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

SYNTHETIC_PROVENANCE = "synthetic"


@dataclass(frozen=True)
class SyntheticMarket:
    enso: pd.DataFrame
    prices: pd.DataFrame
    market_index: pd.DataFrame
    truth: pd.DataFrame

    @property
    def data_provenance(self) -> str:
        return SYNTHETIC_PROVENANCE


def _monthly_dates(start_year: int, months: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{start_year}-01-01", periods=months, freq="MS")


def _autoregressive_index(
    generator: np.random.Generator, months: int, *, persistence: float, scale: float
) -> np.ndarray:
    innovations = generator.standard_normal(months) * scale
    series = np.zeros(months, dtype="float64")
    for position in range(1, months):
        series[position] = persistence * series[position - 1] + innovations[position]
    return series


def synthetic_market(
    *,
    seed: int,
    commodities: int = 12,
    affected_commodities: int = 4,
    planted_horizon_effect: float = 0.12,
    horizon_months: int = 12,
    base_relative_month: int = -1,
    years: int = 65,
    start_year: int = 1960,
    persistence: float = 0.80,
    index_scale: float = 0.50,
    idiosyncratic_scale: float = 0.035,
    market_scale: float = 0.03,
    seasonal_scale: float = 0.02,
) -> SyntheticMarket:
    """Generate prices carrying a known cumulative abnormal return.

    The planted effect is spread evenly in log space across the endpoint window
    -- relative months ``base_relative_month + 1`` through ``horizon_months`` --
    so a commodity's true cumulative log abnormal return at the horizon is
    exactly ``planted_horizon_effect``, whatever the episode calendar turns out
    to be. Everything else in the series (seasonality, a common market factor
    with heterogeneous loadings, and idiosyncratic noise) is structure the
    adjustment stages are supposed to remove.
    """
    if not 0 <= affected_commodities <= commodities:
        raise ValueError("affected_commodities must fall between zero and commodities")
    if horizon_months <= base_relative_month:
        raise ValueError("horizon_months must exceed base_relative_month")

    generator = np.random.default_rng(seed)
    months = years * 12
    dates = _monthly_dates(start_year, months)
    index_values = _autoregressive_index(
        generator, months, persistence=persistence, scale=index_scale
    )
    enso = pd.DataFrame(
        {
            "date": dates,
            "roni": index_values,
            "oni": index_values + generator.standard_normal(months) * 0.05,
        }
    )

    market_log_return = generator.standard_normal(months) * market_scale
    market_level = 100.0 * np.exp(np.cumsum(market_log_return))
    market_index = pd.DataFrame({"date": dates, "world_bank_total_index": market_level})

    names = [f"SYN-{position:02d}" for position in range(commodities)]
    affected = set(names[:affected_commodities])
    window_months = horizon_months - base_relative_month
    per_month_effect = planted_horizon_effect / window_months

    # Any month that sits inside some warm episode's endpoint window carries the
    # planted effect. Episodes are recovered by the pipeline, not passed to it,
    # so the mask is built from the same threshold-and-persistence rule.
    in_window = _endpoint_window_mask(
        index_values,
        threshold=0.5,
        minimum_duration_months=5,
        first_offset=base_relative_month + 1,
        last_offset=horizon_months,
    )

    price_frames: list[pd.DataFrame] = []
    truth_rows: list[dict[str, object]] = []
    for name in names:
        loading = float(generator.uniform(0.5, 1.5))
        seasonal = generator.standard_normal(12) * seasonal_scale
        seasonal -= seasonal.mean()
        month_of_year = dates.month.to_numpy() - 1
        effect = per_month_effect if name in affected else 0.0
        log_return = (
            seasonal[month_of_year]
            + loading * market_log_return
            + generator.standard_normal(months) * idiosyncratic_scale
            + effect * in_window
        )
        log_return[0] = 0.0
        price_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "commodity": name,
                    "unit": "index",
                    "value": 100.0 * np.exp(np.cumsum(log_return)),
                }
            )
        )
        truth_rows.append(
            {
                "commodity": name,
                "market_loading": loading,
                "planted_log_effect": effect * window_months,
                "planted_simple_effect": math.expm1(effect * window_months),
                "is_affected": name in affected,
            }
        )
    return SyntheticMarket(
        enso=enso,
        prices=pd.concat(price_frames, ignore_index=True),
        market_index=market_index,
        truth=pd.DataFrame.from_records(truth_rows),
    )


def _endpoint_window_mask(
    index_values: np.ndarray,
    *,
    threshold: float,
    minimum_duration_months: int,
    first_offset: int,
    last_offset: int,
) -> np.ndarray:
    qualifying = index_values >= threshold
    mask = np.zeros(len(index_values), dtype="float64")
    position = 0
    while position < len(qualifying):
        if not qualifying[position]:
            position += 1
            continue
        end = position
        while end < len(qualifying) and qualifying[end]:
            end += 1
        if end - position >= minimum_duration_months:
            start = max(position + first_offset, 0)
            stop = min(position + last_offset + 1, len(mask))
            mask[start:stop] = 1.0
        position = end
    return mask


def synthetic_endpoint_panel(
    *,
    seed: int,
    episodes: int = 17,
    commodities: int = 30,
    effect: float = 0.0,
    factor_scale: float = 0.10,
    idiosyncratic_scale: float = 0.12,
    tail_degrees_of_freedom: int = 4,
) -> pd.DataFrame:
    """Emit one episode-by-commodity endpoint panel with a shared factor.

    Every commodity loads on the same episode-level factor, which is the
    dependence the shared bootstrap draw is designed to preserve, and the
    idiosyncratic shocks are heavy-tailed like commodity returns. With
    ``effect=0`` the complete null holds, so the rejection rate over repeated
    draws is a direct measurement of a gate's size.
    """
    generator = np.random.default_rng(seed)
    common = generator.standard_normal(episodes) * factor_scale
    variance_correction = math.sqrt(
        tail_degrees_of_freedom / (tail_degrees_of_freedom - 2)
        if tail_degrees_of_freedom > 2
        else 1.0
    )
    rows: list[dict[str, object]] = []
    for position in range(commodities):
        loading = float(generator.uniform(0.6, 1.2))
        idiosyncratic = (
            generator.standard_t(tail_degrees_of_freedom, size=episodes)
            * idiosyncratic_scale
            / variance_correction
        )
        values = effect + loading * common + idiosyncratic
        for episode in range(episodes):
            rows.append(
                {
                    "episode_id": f"synthetic_{episode:02d}",
                    "commodity": f"SYN-{position:02d}",
                    "value": float(values[episode]),
                }
            )
    return pd.DataFrame.from_records(rows)


@dataclass(frozen=True)
class SyntheticExposurePanel:
    monthly: pd.DataFrame
    enso: pd.DataFrame
    exposure: pd.DataFrame
    planted_exposure_response: float

    @property
    def data_provenance(self) -> str:
        return SYNTHETIC_PROVENANCE


def synthetic_exposure_panel(
    *,
    seed: int,
    planted_exposure_response: float = 0.004,
    candidates: int = 32,
    controls: int = 3,
    years: int = 65,
    start_year: int = 1960,
    regime_scale: float = 0.02,
    regime_enso_coupling: float = 0.01,
    control_regime_loading: float = 3.0,
    idiosyncratic_scale: float = 0.03,
    persistence: float = 0.80,
    index_scale: float = 0.50,
) -> SyntheticExposurePanel:
    """Generate a monthly panel with a known exposure-weighted ENSO response.

    The generator also plants the confound this design exists to defeat: a
    global regime factor whose size tracks the magnitude of the ENSO index, and
    which the control series load on far more heavily than the candidates. A
    design without month fixed effects reads that factor as a control-group
    response to ENSO; one with them should not.
    """
    generator = np.random.default_rng(seed)
    months = years * 12
    dates = _monthly_dates(start_year, months)
    index_values = _autoregressive_index(
        generator, months, persistence=persistence, scale=index_scale
    )
    enso = pd.DataFrame(
        {
            "date": dates,
            "roni": index_values,
            "oni": index_values + generator.standard_normal(months) * 0.05,
        }
    )
    regime = (
        np.abs(index_values) * regime_enso_coupling
        + generator.standard_normal(months) * regime_scale
    )

    monthly_frames: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, object]] = []
    names = [f"SYN-{position:02d}" for position in range(candidates)] + [
        f"CONTROL-{position:02d}" for position in range(controls)
    ]
    for name in names:
        is_control = name.startswith("CONTROL")
        weight = 0.0 if is_control else float(generator.choice([0.25, 0.5, 1.0]))
        loading = control_regime_loading if is_control else 1.0
        outcome = (
            loading * regime
            + planted_exposure_response * weight * index_values
            + generator.standard_normal(months) * idiosyncratic_scale
        )
        monthly_frames.append(
            pd.DataFrame(
                {"date": dates, "commodity": name, "seasonal_adjusted_log_return": outcome}
            )
        )
        exposure_rows.append(
            {
                "commodity": name,
                "role": "negative_control" if is_control else "mechanism_candidate",
                "group": "precious_metals" if is_control else "synthetic",
                "exposure_weight": weight,
                "uniform_weight": 0.0 if is_control else 1.0,
                "is_negative_control": is_control,
            }
        )
    return SyntheticExposurePanel(
        monthly=pd.concat(monthly_frames, ignore_index=True),
        enso=enso,
        exposure=pd.DataFrame.from_records(exposure_rows),
        planted_exposure_response=planted_exposure_response,
    )
