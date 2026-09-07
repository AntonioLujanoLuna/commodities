"""Placebo *treatments* for the frozen event study.

The neutral-date placebo keeps the treatment and moves the dates. This stage does
the opposite: it keeps the dates and replaces the treatment. Every stage of the
frozen pipeline -- episode construction, seasonal and market adjustment, the
external-macro model, the +12 endpoint, the studentized whole-episode bootstrap,
candidate-family Benjamini-Hochberg and the calendar-matched neutral-date placebo
-- is re-run unchanged with the ENSO index swapped for a substitute series.

Two families of substitute treatment are used.

*External climate indices* (IOD, PDO, NAO, AMO) are real geophysical series with
their own persistence and their own episodes. If a commodity rejects under RONI
and also under a climate index with no plausible pathway to its supply, the
rejection is a property of the machinery rather than of ENSO.

*Phase-randomized surrogates* of RONI itself preserve the power spectrum, and so
the autocorrelation, while destroying the alignment to actual history. Repeating
the whole gate stack over many surrogates gives the finite-sample distribution of
every gate count under a treatment that cannot have caused anything, which is the
direct measurement of the false-positive rate the negative-control gate is
failing on.

A substitute treatment is moment-matched to the primary index over the shared
analysis window before the frozen threshold is applied, so an episode means the
same number of standard deviations for every treatment. The map is affine and is
the identity for the primary index itself, which is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adjustments import adjust_monthly_returns
from .config import ResearchConfig
from .enso import construct_warm_episodes
from .macro_adjustments import fit_macro_adjusted_returns
from .placebo import (
    build_placebo_endpoints,
    draw_calendar_matched_anchors,
    eligible_neutral_anchors,
    evaluate_placebo_means,
)
from .statistics import benjamini_hochberg, bootstrap_episode_means, salted_seed
from .universe import InferenceSpec

TREATMENT_COLUMN = "treatment"


def moment_match(
    values: pd.Series,
    reference: pd.Series,
    *,
    window: pd.Series | None = None,
) -> pd.Series:
    """Rescale values to the mean and standard deviation of a reference series.

    Both series are summarised over the months where the window mask is true and
    both are observed, so a substitute index with a different unit still meets the
    frozen threshold at the same number of standard deviations. The map is affine,
    so it changes no ordering, no episode shape and no persistence.
    """
    mask = values.notna() & reference.notna()
    if window is not None:
        mask &= window
    if int(mask.sum()) < 120:
        raise ValueError("A substitute treatment needs at least ten years of shared months")
    source = values.loc[mask]
    target = reference.loc[mask]
    source_scale = float(source.std(ddof=1))
    if not np.isfinite(source_scale) or source_scale <= 0:
        raise ValueError("A substitute treatment must vary over the shared window")
    scale = float(target.std(ddof=1)) / source_scale
    return (values - float(source.mean())) * scale + float(target.mean())


def phase_randomized_surrogate(
    values: pd.Series,
    *,
    rng: np.random.Generator,
    preserve_seasonality: bool,
) -> pd.Series:
    """Draw a surrogate with the same power spectrum and no real-world timing.

    The Fourier amplitudes of the observed series are kept and its phases are
    replaced by independent uniform draws, which preserves the autocovariance
    function while leaving everything about when the series was high or low to
    chance. With preserve_seasonality the calendar-month climatology is removed
    first and added back afterwards, so the surrogate also keeps the annual cycle
    the seasonal adjustment stage is built around.
    """
    if values.isna().any():
        raise ValueError("Phase randomization requires a gap-free series")
    series = values.astype(float)
    index = values.index
    climatology = pd.Series(0.0, index=index)
    if preserve_seasonality:
        months = pd.to_datetime(pd.Series(index, index=index)).dt.month
        climatology = months.map(series.groupby(months).mean()).astype(float)
    anomaly = series - climatology
    level = float(anomaly.mean())
    centered = anomaly.to_numpy() - level
    spectrum = np.fft.rfft(centered)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[0])
    phases[0] = 0.0
    if centered.size % 2 == 0:
        phases[-1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    surrogate = np.fft.irfft(randomized, n=centered.size) + level
    return pd.Series(surrogate, index=index) + climatology


def endpoint_from_monthly(
    adjusted_monthly: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    anchor_column: str,
    base_relative_month: int,
    horizon_months: int,
    monthly_return_column: str,
    price_column: str = "value",
) -> pd.DataFrame:
    """Build only the frozen horizon endpoint, without the full event path.

    This is the same quantity the event-path builder produces at the frozen
    horizon: the sum of adjusted monthly returns from the month after the base
    month through the horizon, defined only when the base price exists and no
    month in between is missing, reported in cumulative log and simple form under
    the same names the frozen path uses. It exists because the surrogate grid
    evaluates the endpoint thousands of times and the full -12/+24 path is never
    used; the stage asserts it reproduces the frozen construction on the real
    index.
    """
    if horizon_months <= base_relative_month:
        raise ValueError("The horizon must sit after the base month")
    prefix = monthly_return_column.removesuffix("_log_return")
    log_column = f"{prefix}_cumulative_log_return"
    simple_column = f"{prefix}_cumulative_return"
    monthly = adjusted_monthly.loc[
        :, ["date", "commodity", price_column, monthly_return_column]
    ].copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    offsets = list(range(base_relative_month + 1, horizon_months + 1))
    anchors = episodes.loc[:, ["episode_id", anchor_column]].copy()
    anchors["anchor_date"] = pd.to_datetime(anchors[anchor_column])

    base = anchors.copy()
    base["date"] = base["anchor_date"] + pd.DateOffset(months=base_relative_month)
    base = base.merge(
        monthly.loc[:, ["date", "commodity", price_column]].rename(
            columns={price_column: "base_value"}
        ),
        on="date",
        how="left",
    )
    base = base.loc[base["base_value"].notna(), ["episode_id", "commodity", "base_value"]]

    frames = []
    for offset in offsets:
        step = anchors.copy()
        step["date"] = step["anchor_date"] + pd.DateOffset(months=offset)
        frames.append(step)
    grid = pd.concat(frames, ignore_index=True).merge(
        monthly.loc[:, ["date", "commodity", monthly_return_column]],
        on="date",
        how="left",
    )
    grid = grid.dropna(subset=["commodity"])
    counts = (
        grid.groupby(["episode_id", "commodity"], sort=False, observed=True)[monthly_return_column]
        .agg(["sum", "count"])
        .reset_index()
    )
    counts[log_column] = counts["sum"].where(counts["count"].eq(len(offsets)))
    counts[simple_column] = np.expm1(counts[log_column])
    result = counts.loc[:, ["episode_id", "commodity", log_column, simple_column]].merge(
        base, on=["episode_id", "commodity"], how="inner", validate="one_to_one"
    )
    return result.drop(columns="base_value").sort_values(
        ["episode_id", "commodity"], ignore_index=True
    )


@dataclass(frozen=True)
class TreatmentOutcome:
    candidates: pd.DataFrame
    controls: pd.DataFrame
    episodes: pd.DataFrame
    adjusted_monthly: pd.DataFrame


def evaluate_treatment(
    treatment: pd.DataFrame,
    *,
    monthly: pd.DataFrame,
    market: pd.DataFrame,
    macro: pd.DataFrame,
    candidate_commodities: list[str],
    control_commodities: list[str],
    config: ResearchConfig,
    spec: InferenceSpec,
    label: str,
    bootstrap_replicates: int,
    placebo_replicates: int,
    run_placebo: bool,
) -> TreatmentOutcome:
    """Run the frozen gate stack once with a substitute treatment standing in."""
    monthly_return_column = "macro_adjusted_log_return"
    return_column = "macro_adjusted_cumulative_return"
    episodes = construct_warm_episodes(
        treatment,
        index_name=TREATMENT_COLUMN,
        threshold=config.warm_threshold,
        minimum_duration_months=config.minimum_duration_months,
        observable_delay_after_center_months=config.observable_delay_after_center_months,
    )
    empty = pd.DataFrame()
    if episodes.empty:
        return TreatmentOutcome(empty, empty, episodes, empty)
    seasonal_adjusted, _, _, _ = adjust_monthly_returns(
        monthly,
        market,
        episodes,
        market_exclusion_window=config.market_estimation_exclusion_window,
        minimum_seasonal_observations=config.minimum_seasonal_observations,
        minimum_market_model_observations=config.minimum_market_model_observations,
    )
    macro_adjusted, _ = fit_macro_adjusted_returns(
        seasonal_adjusted,
        macro,
        episodes,
        control_columns=config.macro_control_columns,
        estimation_exclusion_window=config.macro_estimation_exclusion_window,
        minimum_model_observations=config.minimum_macro_model_observations,
    )
    endpoint = endpoint_from_monthly(
        macro_adjusted,
        episodes,
        anchor_column="onset_date",
        base_relative_month=config.base_relative_month,
        horizon_months=spec.primary_horizon_months,
        monthly_return_column=monthly_return_column,
    )

    results: dict[str, pd.DataFrame] = {}
    for family, commodities in (
        ("candidates", candidate_commodities),
        ("controls", control_commodities),
    ):
        subset = endpoint.loc[endpoint["commodity"].isin(commodities)]
        bootstrap = bootstrap_episode_means(
            subset,
            value_column=return_column,
            replicates=bootstrap_replicates,
            confidence_level=config.confidence_level,
            seed=salted_seed(config.random_seed, f"surrogate_{family}:{label}"),
            p_value_method=config.p_value_method,
            minimum_studentized_replicate_share=config.minimum_studentized_replicate_share,
        )
        inference = bootstrap.results.copy()
        inference["eligible"] = inference["episodes"].ge(spec.minimum_valid_episodes)
        inference.loc[~inference["eligible"], "bootstrap_p_value"] = 1.0
        results[family] = inference
    candidates = results["candidates"]
    candidates["bootstrap_bh_q_value"] = benjamini_hochberg(candidates["bootstrap_p_value"])
    candidates["reject_bootstrap_fdr"] = candidates["bootstrap_bh_q_value"].le(config.fdr_alpha)
    controls = results["controls"]
    controls["reject_bootstrap_raw"] = controls["bootstrap_p_value"].lt(0.05)

    if not run_placebo:
        candidates["placebo_p_value"] = np.nan
        controls["placebo_p_value"] = np.nan
        candidates["reject_placebo_fdr"] = False
        controls["reject_placebo_raw"] = False
    else:
        valid_dates = (
            macro_adjusted.loc[macro_adjusted[monthly_return_column].notna(), "date"]
            .drop_duplicates()
            .sort_values()
        )
        anchors = eligible_neutral_anchors(
            treatment,
            episodes,
            index_name=TREATMENT_COLUMN,
            neutral_absolute_threshold=config.placebo_neutral_absolute_threshold,
            exclusion_window=config.placebo_actual_event_exclusion_window,
            first_anchor=pd.Timestamp(valid_dates.min()),
            last_anchor=pd.Timestamp(valid_dates.max())
            - pd.DateOffset(months=spec.primary_horizon_months),
        )
        commodities = sorted(set(candidate_commodities) | set(control_commodities))
        placebo_endpoints = build_placebo_endpoints(
            macro_adjusted,
            anchors["date"],
            commodities,
            horizon_months=spec.primary_horizon_months,
            monthly_return_column=monthly_return_column,
        )
        evaluated: dict[str, pd.DataFrame] = {}
        for family, frame, members in (
            ("candidates", candidates, candidate_commodities),
            ("controls", controls, control_commodities),
        ):
            draws = draw_calendar_matched_anchors(
                anchors,
                episodes["onset_date"],
                replicates=placebo_replicates,
                seed=salted_seed(config.random_seed, f"surrogate_placebo_{family}:{label}"),
                sample_without_replacement=False,
            )
            placebo = evaluate_placebo_means(
                placebo_endpoints.loc[placebo_endpoints["commodity"].isin(members)],
                draws,
                frame,
                replicates=placebo_replicates,
                minimum_valid_episodes=spec.minimum_valid_episodes,
                minimum_valid_replicate_share=config.placebo_minimum_valid_replicate_share,
                confidence_level=config.confidence_level,
                sample_without_replacement=False,
            )
            evaluated[family] = frame.merge(
                placebo.results.loc[:, ["commodity", "placebo_p_value"]],
                on="commodity",
                how="left",
                validate="one_to_one",
            )
        candidates = evaluated["candidates"]
        controls = evaluated["controls"]
        candidates["placebo_bh_q_value"] = benjamini_hochberg(candidates["placebo_p_value"])
        candidates["reject_placebo_fdr"] = candidates["placebo_bh_q_value"].le(config.fdr_alpha)
        controls["reject_placebo_raw"] = controls["placebo_p_value"].lt(0.05)

    candidates["passes_gates"] = (
        candidates["eligible"]
        & candidates["sign_agreement"]
        & candidates["reject_bootstrap_fdr"]
        & candidates["reject_placebo_fdr"]
    )
    return TreatmentOutcome(candidates, controls, episodes, macro_adjusted)
