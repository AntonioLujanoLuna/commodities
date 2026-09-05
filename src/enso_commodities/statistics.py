from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_CHUNK = 2000
P_VALUE_METHODS = ("centered", "studentized")


@dataclass(frozen=True)
class BootstrapOutput:
    results: pd.DataFrame
    replicates: pd.DataFrame


def salted_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Step-up FDR adjustment, valid under independence or positive dependence."""
    return _step_up_adjustment(p_values, penalty=1.0)


def benjamini_yekutieli(p_values: pd.Series) -> pd.Series:
    """Step-up FDR adjustment valid under arbitrary dependence.

    Benjamini-Hochberg inflated by the harmonic number of the family size. This
    is the conservative counterpart to the resampling procedure below: it makes
    no assumption about how the commodity tests co-move, at the cost of power.
    """
    valid_count = int(pd.to_numeric(p_values, errors="coerce").notna().sum())
    if valid_count == 0:
        return pd.Series(float("nan"), index=p_values.index, dtype="float64")
    harmonic = float(np.sum(1.0 / np.arange(1, valid_count + 1)))
    return _step_up_adjustment(p_values, penalty=harmonic)


def _step_up_adjustment(p_values: pd.Series, *, penalty: float) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(float("nan"), index=p_values.index, dtype="float64")
    if valid.empty:
        return result
    ordered = valid.sort_values(kind="mergesort")
    count = len(ordered)
    raw_adjusted = [
        float(value) * penalty * count / rank for rank, value in enumerate(ordered, start=1)
    ]
    monotone = raw_adjusted.copy()
    for position in range(count - 2, -1, -1):
        monotone[position] = min(monotone[position], monotone[position + 1])
    result.loc[ordered.index] = [min(value, 1.0) for value in monotone]
    return result


def westfall_young_step_down(
    observed_statistics: pd.Series,
    null_statistics: pd.DataFrame,
    *,
    minimum_valid_replicate_share: float = 0.9,
) -> pd.DataFrame:
    """Step-down max-T resampling adjustment (Westfall-Young).

    ``null_statistics`` must be indexed by replicate and have one column per
    family member, holding statistics drawn under the complete null from the
    *same* resampling draw. Because a replicate is one coherent alternative
    history shared across commodities, the column-wise maximum reproduces the
    joint null of the largest statistic, so the adjusted p-values control the
    family-wise error rate under subset pivotality without assuming any
    particular dependence structure.

    Note this is FWER control, which is strictly stronger -- and therefore
    strictly less powerful -- than the Benjamini-Hochberg false-discovery rate
    reported alongside it.
    """
    if not 0 < minimum_valid_replicate_share <= 1:
        raise ValueError("minimum_valid_replicate_share must fall in (0, 1]")
    observed = pd.to_numeric(observed_statistics, errors="coerce")
    family = [str(name) for name in observed.dropna().index if name in null_statistics.columns]
    columns = ["statistic", "step_down_rank", "westfall_young_p_value", "westfall_young_status"]
    if not family:
        return pd.DataFrame(
            {
                "commodity": list(observed.index),
                "statistic": observed.to_numpy(dtype="float64"),
                "step_down_rank": pd.NA,
                "westfall_young_p_value": float("nan"),
                "westfall_young_status": "no_eligible_family_members",
            }
        )

    nulls = null_statistics.loc[:, family].apply(pd.to_numeric, errors="coerce")
    total_replicates = len(nulls)
    complete = nulls.notna().all(axis=1)
    usable = nulls.loc[complete]
    valid_share = len(usable) / total_replicates if total_replicates else 0.0
    ordered_family = observed.loc[family].abs().sort_values(ascending=False, kind="mergesort").index
    frame = pd.DataFrame(index=pd.Index(observed.index, name="commodity"), columns=columns)
    frame["statistic"] = observed.astype("float64")
    frame["westfall_young_p_value"] = float("nan")
    frame["westfall_young_status"] = "outside_family"
    frame.loc[family, "westfall_young_status"] = "estimated"
    frame.loc[ordered_family, "step_down_rank"] = range(1, len(ordered_family) + 1)

    if total_replicates == 0 or valid_share < minimum_valid_replicate_share:
        frame.loc[family, "westfall_young_status"] = "insufficient_valid_replicates"
        return frame.reset_index()

    absolute_nulls = usable.loc[:, ordered_family].abs().to_numpy(dtype="float64")
    # Successive maxima over the still-untested tail, evaluated per replicate.
    tail_maxima = np.maximum.accumulate(absolute_nulls[:, ::-1], axis=1)[:, ::-1]
    absolute_observed = observed.loc[ordered_family].abs().to_numpy(dtype="float64")
    exceedances = (tail_maxima >= absolute_observed).sum(axis=0)
    raw = (exceedances + 1) / (len(usable) + 1)
    frame.loc[ordered_family, "westfall_young_p_value"] = np.maximum.accumulate(raw)
    return frame.reset_index()


def _episode_commodity_matrix(
    data: pd.DataFrame, value_column: str
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    episode_ids = sorted(str(value) for value in data["episode_id"].unique())
    commodities = sorted(str(value) for value in data["commodity"].unique())
    episode_position = {episode: index for index, episode in enumerate(episode_ids)}
    commodity_position = {commodity: index for index, commodity in enumerate(commodities)}
    values = np.full((len(episode_ids), len(commodities)), np.nan, dtype="float64")
    rows = data["episode_id"].astype(str).map(episode_position).to_numpy(dtype="int64")
    columns = data["commodity"].astype(str).map(commodity_position).to_numpy(dtype="int64")
    values[rows, columns] = data[value_column].to_numpy(dtype="float64")
    mask = np.isfinite(values)
    return episode_ids, commodities, values, mask


def bootstrap_episode_means(
    data: pd.DataFrame,
    *,
    value_column: str,
    replicates: int,
    confidence_level: float,
    seed: int,
    p_value_method: str = "centered",
    minimum_studentized_replicate_share: float = 0.9,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK,
) -> BootstrapOutput:
    """Resample whole episodes and summarise each commodity's mean endpoint.

    One draw of episodes is shared across every commodity, so a replicate is a
    single coherent alternative history. Two tests are reported from the same
    draws: a centred percentile test on the mean, and a studentized
    (bootstrap-t) test that divides each replicate mean by that replicate's own
    standard error. With this few episodes the studentized version is the
    better-calibrated of the two, because it adapts to the heavy-tailed and
    heteroskedastic scale of commodity returns rather than assuming the
    observed scale is the right one.

    ``p_value_method`` selects which of the two fills the canonical
    ``bootstrap_p_value`` column that the downstream gates read. Both are always
    reported under their own names, and a commodity whose studentized statistic
    is undefined falls back to the centred test rather than dropping out of the
    family with a missing p-value.
    """
    required = {"episode_id", "commodity", value_column}
    if not required.issubset(data.columns):
        raise ValueError(f"Inference data is missing columns: {sorted(required - set(data))}")
    if data.duplicated(["episode_id", "commodity"]).any():
        raise ValueError("Inference data has duplicate episode/commodity keys")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must fall between zero and one")
    if p_value_method not in P_VALUE_METHODS:
        raise ValueError(f"p_value_method must be one of {sorted(P_VALUE_METHODS)}")
    if not 0 < minimum_studentized_replicate_share <= 1:
        raise ValueError("minimum_studentized_replicate_share must fall in (0, 1]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    clean = data.loc[data[value_column].notna(), ["episode_id", "commodity", value_column]].copy()
    clean[value_column] = pd.to_numeric(clean[value_column], errors="raise")
    if clean.empty:
        raise ValueError("Inference data has no valid observations")
    episode_ids, commodities, values, mask = _episode_commodity_matrix(clean, value_column)
    if not episode_ids or not commodities:
        raise ValueError("Inference data has no valid observations")

    counts = mask.sum(axis=0).astype("float64")
    observed_means = np.where(counts > 0, np.nansum(np.where(mask, values, 0.0), axis=0), np.nan)
    observed_means = np.divide(
        observed_means, counts, out=np.full_like(observed_means, np.nan), where=counts > 0
    )
    centered = np.where(mask, values - observed_means, 0.0)
    squared = centered**2
    indicator = mask.astype("float64")
    observed_variance = np.divide(
        squared.sum(axis=0),
        counts - 1,
        out=np.full_like(counts, np.nan),
        where=counts > 1,
    )
    observed_error = np.sqrt(
        np.divide(observed_variance, counts, out=np.full_like(counts, np.nan), where=counts > 0)
    )
    observed_t = np.divide(
        observed_means,
        observed_error,
        out=np.full_like(observed_means, np.nan),
        where=np.isfinite(observed_error) & (observed_error > 0),
    )

    generator = np.random.default_rng(seed)
    episode_count = len(episode_ids)
    replicate_frames: list[pd.DataFrame] = []
    centered_extreme = np.zeros(len(commodities), dtype="int64")
    studentized_extreme = np.zeros(len(commodities), dtype="int64")
    studentized_valid = np.zeros(len(commodities), dtype="int64")
    bootstrap_means_by_commodity: list[list[np.ndarray]] = [[] for _ in commodities]
    studentized_by_commodity: list[list[np.ndarray]] = [[] for _ in commodities]
    absolute_observed_mean = np.abs(observed_means)
    absolute_observed_t = np.abs(observed_t)

    drawn = 0
    while drawn < replicates:
        block = min(chunk_size, replicates - drawn)
        draw = generator.integers(0, episode_count, size=(block, episode_count))
        sampled_counts = indicator[draw].sum(axis=1)
        sampled_sums = centered[draw].sum(axis=1)
        sampled_squares = squared[draw].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            null_means = np.where(sampled_counts > 0, sampled_sums / sampled_counts, np.nan)
            replicate_variance = np.where(
                sampled_counts > 1,
                (sampled_squares - sampled_counts * null_means**2) / (sampled_counts - 1),
                np.nan,
            )
            replicate_error = np.sqrt(
                np.where(sampled_counts > 0, replicate_variance / sampled_counts, np.nan)
            )
            studentized = np.where(replicate_error > 0, null_means / replicate_error, np.nan)
        bootstrap_means = null_means + observed_means

        centered_extreme += np.nansum(np.abs(null_means) >= absolute_observed_mean, axis=0).astype(
            "int64"
        )
        finite_studentized = np.isfinite(studentized) & np.isfinite(absolute_observed_t)
        studentized_valid += finite_studentized.sum(axis=0).astype("int64")
        studentized_extreme += (
            (finite_studentized & (np.abs(studentized) >= absolute_observed_t))
            .sum(axis=0)
            .astype("int64")
        )

        present = sampled_counts > 0
        replicate_index, commodity_index = np.nonzero(present)
        replicate_frames.append(
            pd.DataFrame(
                {
                    "replicate": (drawn + replicate_index).astype("int64"),
                    "commodity": [commodities[position] for position in commodity_index],
                    "sampled_observations": sampled_counts[present].astype("int64"),
                    "bootstrap_mean": bootstrap_means[present],
                    "null_bootstrap_mean": null_means[present],
                    "bootstrap_standard_error": replicate_error[present],
                    "studentized_statistic": studentized[present],
                }
            )
        )
        for position in range(len(commodities)):
            bootstrap_means_by_commodity[position].append(bootstrap_means[:, position])
            studentized_by_commodity[position].append(studentized[:, position])
        drawn += block

    replicate_table = pd.concat(replicate_frames, ignore_index=True)
    replicate_table = replicate_table.sort_values(
        ["replicate", "commodity"], ignore_index=True, kind="mergesort"
    )

    alpha = 1 - confidence_level
    minimum_valid = minimum_studentized_replicate_share * replicates
    result_rows: list[dict[str, object]] = []
    for position, commodity in enumerate(commodities):
        column = values[:, position]
        sample = column[mask[:, position]]
        mean = float(observed_means[position])
        median = float(np.median(sample))
        positive_share = float(np.mean(sample > 0))
        direction = "positive" if mean > 0 else "negative" if mean < 0 else "zero"
        sign_agreement = (mean > 0 and median > 0 and positive_share > 0.5) or (
            mean < 0 and median < 0 and positive_share < 0.5
        )
        bootstrap_sample = np.concatenate(bootstrap_means_by_commodity[position])
        bootstrap_sample = bootstrap_sample[np.isfinite(bootstrap_sample)]
        studentized_sample = np.concatenate(studentized_by_commodity[position])
        studentized_sample = studentized_sample[np.isfinite(studentized_sample)]
        valid_studentized = int(studentized_valid[position])

        studentized_p: float = float("nan")
        studentized_lower: float = float("nan")
        studentized_upper: float = float("nan")
        if not np.isfinite(observed_t[position]):
            studentized_status = "degenerate_observed_scale"
        elif valid_studentized < minimum_valid or studentized_sample.size == 0:
            studentized_status = "insufficient_valid_replicates"
        else:
            studentized_status = "estimated"
            studentized_p = (int(studentized_extreme[position]) + 1) / (valid_studentized + 1)
            upper_quantile = float(np.quantile(studentized_sample, 1 - alpha / 2))
            lower_quantile = float(np.quantile(studentized_sample, alpha / 2))
            studentized_lower = mean - upper_quantile * float(observed_error[position])
            studentized_upper = mean - lower_quantile * float(observed_error[position])

        centered_p = (int(centered_extreme[position]) + 1) / (replicates + 1)
        if p_value_method == "centered":
            selected_p, selected_method = centered_p, "centered"
        elif studentized_status == "estimated":
            selected_p, selected_method = studentized_p, "studentized"
        else:
            selected_p, selected_method = centered_p, "studentized_fallback_centered"

        result_rows.append(
            {
                "commodity": commodity,
                "episodes": int(counts[position]),
                "draw_universe_episodes": len(episode_ids),
                "mean_return": mean,
                "median_return": median,
                "positive_share": positive_share,
                "direction": direction,
                "sign_agreement": sign_agreement,
                "standard_error": float(observed_error[position]),
                "t_statistic": float(observed_t[position]),
                "confidence_level": confidence_level,
                "ci_lower": float(np.quantile(bootstrap_sample, alpha / 2)),
                "ci_upper": float(np.quantile(bootstrap_sample, 1 - alpha / 2)),
                "bootstrap_p_value": selected_p,
                "bootstrap_p_value_method": selected_method,
                "centered_p_value": centered_p,
                "studentized_p_value": studentized_p,
                "studentized_ci_lower": studentized_lower,
                "studentized_ci_upper": studentized_upper,
                "studentized_valid_replicates": valid_studentized,
                "studentized_status": studentized_status,
            }
        )
    return BootstrapOutput(
        results=pd.DataFrame.from_records(result_rows),
        replicates=replicate_table,
    )


def studentized_null_matrix(replicates: pd.DataFrame) -> pd.DataFrame:
    """Pivot stored replicates into the replicate-by-commodity joint null."""
    required = {"replicate", "commodity", "studentized_statistic"}
    if not required.issubset(replicates.columns):
        raise ValueError(f"Replicates are missing columns: {sorted(required - set(replicates))}")
    return replicates.pivot(index="replicate", columns="commodity", values="studentized_statistic")
