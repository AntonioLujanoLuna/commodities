"""Minimum detectable effect for the frozen endpoint.

The evidence ladder's fourth bucket, NO MATERIAL ENSO EFFECT, asserts a
negative. Nothing else in this repository estimates what effect the design
could have detected, so a commodity currently lands in that bucket whether it
has no effect or whether seventeen episodes were never enough to see one. Those
are different statements and only one of them is a finding.

The test whose power matters is the one the gates read: a two-sided studentized
bootstrap on the episode-level endpoint. Its power is recoverable from
artifacts the inference stage already writes, with no new resampling, because
the alternative here is an additive location shift.

Under a shift of ``effect`` applied to every episode of one commodity, the
sample mean moves by exactly ``effect`` and the sample variance -- and therefore
the standard error -- does not move at all. So the observed statistic under the
alternative is the null statistic plus ``effect / standard_error``, and the
stored null replicates are a valid reference distribution for it. That makes
this an exact rearrangement of the bootstrap the study already ran rather than
a normal-theory approximation of it.

Two thresholds are reported, because the gate is not a single test. The
marginal threshold uses ``fdr_alpha`` directly and describes a commodity tested
on its own. The family bound uses ``fdr_alpha / family_size``, the level
Benjamini-Hochberg demands when exactly one member of the family rejects, and
is therefore the pessimistic end of the range the actual FDR gate can require.
The truth for any given run sits between them.

Read the result as an order of magnitude, not a threshold. It is a function of
one observed standard error, and a standard error estimated from seventeen
episodes carries roughly a fifth of its own size in sampling noise; repeated
draws from an identical process move the reported minimum detectable effect by
a factor of two. That is precise enough for its purpose, which is to separate
"no effect" from "no power to see one", and nowhere near precise enough to
support a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MAXIMUM_EFFECT = 5.0
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_BISECTIONS = 60


@dataclass(frozen=True)
class PowerOutput:
    minimum_detectable_effects: pd.DataFrame
    power_curves: pd.DataFrame


def bootstrap_critical_value(null_statistics: np.ndarray, *, alpha: float) -> float:
    """Two-sided critical value of the studentized bootstrap null."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must fall between zero and one")
    finite = null_statistics[np.isfinite(null_statistics)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(np.abs(finite), 1 - alpha))


def bootstrap_power(
    null_statistics: np.ndarray,
    *,
    standard_error: float,
    effect: float,
    critical_value: float,
) -> float:
    """Rejection probability of the two-sided studentized test at ``effect``.

    The shift is additive on the endpoint scale, so it leaves the standard error
    unchanged and enters the statistic as ``effect / standard_error``.
    """
    finite = null_statistics[np.isfinite(null_statistics)]
    if finite.size == 0 or not np.isfinite(critical_value):
        return float("nan")
    if not np.isfinite(standard_error) or standard_error <= 0:
        return float("nan")
    shifted = np.abs(finite + effect / standard_error)
    return float(np.mean(shifted >= critical_value))


def minimum_detectable_effect(
    null_statistics: np.ndarray,
    *,
    standard_error: float,
    alpha: float,
    target_power: float,
    maximum_effect: float = DEFAULT_MAXIMUM_EFFECT,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, str]:
    """Smallest positive shift the test rejects with probability ``target_power``.

    Returns the effect and a status. Power is increasing in the size of the
    shift up to Monte Carlo noise, so this bisects; a design that cannot reach
    the target anywhere below ``maximum_effect`` reports that rather than a
    number, because an extrapolated minimum detectable effect is worse than an
    admitted ceiling.
    """
    if not 0 < target_power < 1:
        raise ValueError("target_power must fall between zero and one")
    if maximum_effect <= 0:
        raise ValueError("maximum_effect must be positive")
    critical = bootstrap_critical_value(null_statistics, alpha=alpha)
    if not np.isfinite(critical):
        return float("nan"), "no_valid_replicates"
    if not np.isfinite(standard_error) or standard_error <= 0:
        return float("nan"), "degenerate_standard_error"

    def power_at(effect: float) -> float:
        return bootstrap_power(
            null_statistics,
            standard_error=standard_error,
            effect=effect,
            critical_value=critical,
        )

    if power_at(maximum_effect) < target_power:
        return float("nan"), "unreachable_within_maximum_effect"

    lower, upper = 0.0, float(maximum_effect)
    for _ in range(DEFAULT_MAX_BISECTIONS):
        if upper - lower <= tolerance:
            break
        middle = 0.5 * (lower + upper)
        if power_at(middle) >= target_power:
            upper = middle
        else:
            lower = middle
    return float(upper), "estimated"


def evaluate_power(
    results: pd.DataFrame,
    null_matrix: pd.DataFrame,
    *,
    alpha: float,
    family_size: int,
    target_power: float,
    reference_effects: tuple[float, ...],
    maximum_effect: float = DEFAULT_MAXIMUM_EFFECT,
) -> PowerOutput:
    """Minimum detectable effects and power curves for one inference family.

    ``results`` is a bootstrap result table and ``null_matrix`` its stored
    replicate-by-commodity studentized null, which is exactly what
    ``statistics.studentized_null_matrix`` produces from the same run.
    """
    required = {"commodity", "mean_return", "standard_error", "episodes"}
    if not required.issubset(results.columns):
        raise ValueError(f"Results are missing columns: {sorted(required - set(results))}")
    if family_size < 1:
        raise ValueError("family_size must be positive")
    if any(effect <= 0 for effect in reference_effects):
        raise ValueError("reference_effects must be positive")
    family_alpha = alpha / family_size

    effect_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for record in results.sort_values("commodity").to_dict("records"):
        commodity = str(record["commodity"])
        standard_error = float(record["standard_error"])
        nulls = (
            null_matrix[commodity].to_numpy(dtype="float64")
            if commodity in null_matrix.columns
            else np.empty(0, dtype="float64")
        )
        valid = int(np.isfinite(nulls).sum())
        marginal_critical = bootstrap_critical_value(nulls, alpha=alpha)
        family_critical = bootstrap_critical_value(nulls, alpha=family_alpha)
        marginal_effect, marginal_status = minimum_detectable_effect(
            nulls,
            standard_error=standard_error,
            alpha=alpha,
            target_power=target_power,
            maximum_effect=maximum_effect,
        )
        family_effect, family_status = minimum_detectable_effect(
            nulls,
            standard_error=standard_error,
            alpha=family_alpha,
            target_power=target_power,
            maximum_effect=maximum_effect,
        )
        observed = float(record["mean_return"])
        effect_rows.append(
            {
                "commodity": commodity,
                "episodes": int(record["episodes"]),
                "observed_mean_return": observed,
                "standard_error": standard_error,
                "valid_null_replicates": valid,
                "marginal_alpha": alpha,
                "family_alpha": family_alpha,
                "family_size": family_size,
                "target_power": target_power,
                "marginal_critical_value": marginal_critical,
                "family_critical_value": family_critical,
                "minimum_detectable_effect_marginal": marginal_effect,
                "minimum_detectable_effect_marginal_status": marginal_status,
                "minimum_detectable_effect_family": family_effect,
                "minimum_detectable_effect_family_status": family_status,
                # A commodity whose observed estimate is far below what the
                # design could have seen is uninformative, not null.
                "observed_effect_below_marginal_mde": (
                    bool(abs(observed) < marginal_effect)
                    if np.isfinite(marginal_effect)
                    else pd.NA
                ),
            }
        )
        for effect in reference_effects:
            curve_rows.append(
                {
                    "commodity": commodity,
                    "effect": float(effect),
                    "power_marginal": bootstrap_power(
                        nulls,
                        standard_error=standard_error,
                        effect=float(effect),
                        critical_value=marginal_critical,
                    ),
                    "power_family": bootstrap_power(
                        nulls,
                        standard_error=standard_error,
                        effect=float(effect),
                        critical_value=family_critical,
                    ),
                }
            )
    return PowerOutput(
        minimum_detectable_effects=pd.DataFrame.from_records(effect_rows),
        power_curves=pd.DataFrame.from_records(curve_rows),
    )
