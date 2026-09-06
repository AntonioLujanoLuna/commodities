"""Exposure-weighted panel estimator with commodity and month fixed effects.

The event-study design measures an abnormal return against a baseline fitted on
the months that no warm episode window covers. That baseline is itself selected
on the ENSO state, and the negative controls reject under it. This module takes
the opposite approach: put a fixed effect on every calendar month, so any global
shock -- dollar, inflation, risk appetite, whatever moves gold -- is absorbed
whether or not anyone thought to control for it, and identify the ENSO response
from cross-sectional differences in physical exposure instead.

    y[c,t] = a[c] + d[t] + b1 * (w[c] * ENSO[t-L]) + b2 * (control[c] * ENSO[t-L]) + e[c,t]

``w[c]`` is the pre-specified exposure weight and is zero for the negative
controls, so ``b2`` is the controls' own ENSO response measured against the same
month effects. A design that is working reports ``b2`` indistinguishable from
zero. ``ENSO[t-L]`` alone is collinear with the month effects and ``w[c]`` alone
with the commodity effects; only the interactions survive, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import project_root
from .universe import CommodityRegistry

ALLOWED_OUTCOMES = ("seasonal_adjusted_log_return", "raw_log_return")
ALLOWED_WEIGHTINGS = ("exposure", "uniform")
EXPOSURE_TERM = "exposure_x_enso"
CONTROL_TERM = "control_x_enso"
REGRESSORS = (EXPOSURE_TERM, CONTROL_TERM)
PRIMARY_BLOCK_COLUMN = "calendar_year"


def panel_regressors(weighting: str) -> tuple[str, ...]:
    """Terms that are separately identified under a given weighting scheme.

    Under ``uniform`` weighting the exposure indicator is one minus the control
    indicator, so the two interactions differ only by the ENSO level -- which
    the month effects have already absorbed. They are then exactly collinear,
    and the control term is not a separate coefficient at all: the exposure
    coefficient already *is* the candidate-minus-control contrast. Under
    ``exposure`` weighting the candidates carry different weights, so both terms
    are identified and the control response can be read on its own.
    """
    if weighting not in ALLOWED_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(ALLOWED_WEIGHTINGS)}")
    return REGRESSORS if weighting == "exposure" else (EXPOSURE_TERM,)


@dataclass(frozen=True)
class BlockDefinition:
    """One alternative resampling block for the year-block bootstrap.

    A block bootstrap only preserves dependence that fits inside a block. The
    object this design has to keep intact is an ENSO episode and its price
    response: episodes run at least five months and are read at lags of up to
    twelve, so a run of elevated regressor comfortably exceeds twelve months and
    therefore always straddles a January boundary. ENSO is also phase-locked to
    the annual cycle and peaks in NDJ, so that boundary cuts the peak of very
    nearly every event rather than falling at a random point.

    ``start_month`` moves the boundary -- May starts the conventional ENSO year
    and keeps each NDJ peak whole -- and ``length_years`` makes the block longer
    than the episode it contains. Blocks shorter than the dependence understate
    the standard error, so these variants can only widen the interval.
    """

    name: str
    start_month: int
    length_years: int

    @property
    def column(self) -> str:
        return f"block_{self.name}"


@dataclass(frozen=True)
class PanelSpec:
    exposure_version: int
    exposure_method: str
    exposure_authored_on: date
    exposure_outcome_blind: bool
    exposure_inference_scope: str
    outcome_column: str
    unit_fixed_effects: bool
    time_fixed_effects: bool
    index_definitions: tuple[str, ...]
    primary_index: str
    primary_lag_months: int
    reported_lag_months: tuple[int, ...]
    weighting_schemes: tuple[str, ...]
    primary_weighting: str
    minimum_observations: int
    roles: tuple[str, ...]
    block: str
    block_sensitivity: tuple[BlockDefinition, ...]
    bootstrap_replicates: int
    weight_permutation_replicates: int
    confidence_level: float
    fdr_alpha: float
    random_seed: int
    demeaning_tolerance: float
    demeaning_max_iterations: int
    exposure_weights: dict[str, float]
    negative_control_weight: float


@dataclass(frozen=True)
class PanelFit:
    coefficients: dict[str, float]
    naive_standard_errors: dict[str, float]
    observations: int
    units: int
    periods: int
    within_r_squared: float
    residual_degrees_of_freedom: int
    design_condition_number: float


@dataclass(frozen=True)
class ExposurePermutationOutput:
    observed_estimate: float
    permutation_mean: float
    permutation_median: float
    mapping_percentile: float
    p_value: float
    valid_replicates: int
    replicates: pd.DataFrame


def load_panel_config(path: Path | None = None) -> PanelSpec:
    config_path = path or project_root() / "config" / "panel.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    specification = raw["specification"]
    inference = raw["inference"]
    provenance = raw["exposure_provenance"]
    spec = PanelSpec(
        exposure_version=int(provenance["version"]),
        exposure_method=str(provenance["method"]),
        exposure_authored_on=date.fromisoformat(str(provenance["authored_on"])),
        exposure_outcome_blind=bool(provenance["outcome_blind"]),
        exposure_inference_scope=str(provenance["inference_scope"]),
        outcome_column=str(specification["outcome_column"]),
        unit_fixed_effects=bool(specification["unit_fixed_effects"]),
        time_fixed_effects=bool(specification["time_fixed_effects"]),
        index_definitions=tuple(str(value) for value in specification["index_definitions"]),
        primary_index=str(specification["primary_index"]),
        primary_lag_months=int(specification["primary_lag_months"]),
        reported_lag_months=tuple(int(value) for value in specification["reported_lag_months"]),
        weighting_schemes=tuple(str(value) for value in specification["weighting_schemes"]),
        primary_weighting=str(specification["primary_weighting"]),
        minimum_observations=int(specification["minimum_observations"]),
        roles=tuple(str(value) for value in specification["roles"]),
        block=str(inference["block"]),
        block_sensitivity=tuple(
            BlockDefinition(
                name=str(entry["name"]),
                start_month=int(entry["start_month"]),
                length_years=int(entry["length_years"]),
            )
            for entry in inference.get("block_sensitivity") or ()
        ),
        bootstrap_replicates=int(inference["bootstrap_replicates"]),
        weight_permutation_replicates=int(inference["weight_permutation_replicates"]),
        confidence_level=float(inference["confidence_level"]),
        fdr_alpha=float(inference["fdr_alpha"]),
        random_seed=int(inference["random_seed"]),
        demeaning_tolerance=float(inference["demeaning_tolerance"]),
        demeaning_max_iterations=int(inference["demeaning_max_iterations"]),
        exposure_weights={str(key): float(value) for key, value in raw["exposure_weights"].items()},
        negative_control_weight=float(raw["negative_control_weight"]),
    )
    _validate_panel_spec(spec)
    return spec


def _validate_panel_spec(spec: PanelSpec) -> None:
    if spec.exposure_version < 1:
        raise ValueError("exposure provenance version must be positive")
    if spec.exposure_method != "post_outcome_expert_judgment":
        raise ValueError("Unsupported exposure-weight provenance method")
    if spec.exposure_outcome_blind:
        raise ValueError("Post-outcome expert-judgment weights cannot be labelled outcome-blind")
    if spec.exposure_inference_scope != "exploratory_only":
        raise ValueError("Post-outcome exposure weights must remain exploratory_only")
    if spec.outcome_column not in ALLOWED_OUTCOMES:
        raise ValueError(f"outcome_column must be one of {sorted(ALLOWED_OUTCOMES)}")
    if not spec.time_fixed_effects:
        raise ValueError("The panel design requires month fixed effects; that is its whole point")
    if not spec.unit_fixed_effects:
        raise ValueError("The panel design requires commodity fixed effects")
    if spec.primary_index not in spec.index_definitions:
        raise ValueError("primary_index must appear in index_definitions")
    if spec.primary_lag_months not in spec.reported_lag_months:
        raise ValueError("primary_lag_months must appear in reported_lag_months")
    if any(lag < 0 for lag in spec.reported_lag_months):
        raise ValueError("reported_lag_months must be non-negative")
    if set(spec.weighting_schemes) - set(ALLOWED_WEIGHTINGS):
        raise ValueError(f"weighting_schemes must fall in {sorted(ALLOWED_WEIGHTINGS)}")
    if spec.primary_weighting not in spec.weighting_schemes:
        raise ValueError("primary_weighting must appear in weighting_schemes")
    if spec.roles != ("mechanism_candidate", "negative_control"):
        raise ValueError("The panel estimation sample must be candidates plus negative controls")
    if spec.block != "calendar_year":
        raise ValueError("The primary resampling block is frozen as calendar_year")
    names = [definition.name for definition in spec.block_sensitivity]
    if len(set(names)) != len(names):
        raise ValueError("Block-sensitivity names must be unique")
    if spec.block in names:
        raise ValueError("A block-sensitivity variant cannot reuse the primary block name")
    for definition in spec.block_sensitivity:
        if not 1 <= definition.start_month <= 12:
            raise ValueError(f"Block {definition.name!r} has an out-of-range start_month")
        if definition.length_years < 1:
            raise ValueError(f"Block {definition.name!r} must span at least one year")
    if spec.bootstrap_replicates < 999:
        raise ValueError("bootstrap_replicates must be at least 999")
    if spec.weight_permutation_replicates < 999:
        raise ValueError("weight_permutation_replicates must be at least 999")
    if not 0 < spec.confidence_level < 1:
        raise ValueError("confidence_level must fall between zero and one")
    if not 0 < spec.fdr_alpha < 1:
        raise ValueError("fdr_alpha must fall between zero and one")
    if spec.negative_control_weight != 0.0:
        raise ValueError("Negative controls must carry zero exposure by construction")
    if any(weight < 0 for weight in spec.exposure_weights.values()):
        raise ValueError("Exposure weights must be non-negative")
    if not any(weight > 0 for weight in spec.exposure_weights.values()):
        raise ValueError("At least one mechanism candidate must carry positive exposure")


def build_exposure_table(registry: CommodityRegistry, spec: PanelSpec) -> pd.DataFrame:
    """Attach exposure weights to the registry, refusing any gap or surplus."""
    entries = registry.entries
    sample = entries.loc[entries["role"].isin(spec.roles), ["commodity", "role", "group"]].copy()
    candidates = set(sample.loc[sample["role"].eq("mechanism_candidate"), "commodity"])
    weighted = set(spec.exposure_weights)
    if candidates != weighted:
        raise ValueError(
            "Exposure weights do not exactly cover the mechanism candidates; "
            f"missing={sorted(candidates - weighted)}, unknown={sorted(weighted - candidates)}"
        )
    is_control = sample["role"].eq("negative_control")
    sample["exposure_weight"] = sample["commodity"].map(spec.exposure_weights)
    sample.loc[is_control, "exposure_weight"] = spec.negative_control_weight
    sample["uniform_weight"] = np.where(is_control, spec.negative_control_weight, 1.0)
    sample["is_negative_control"] = is_control
    sample["exposure_version"] = spec.exposure_version
    sample["exposure_method"] = spec.exposure_method
    sample["exposure_authored_on"] = spec.exposure_authored_on.isoformat()
    sample["exposure_outcome_blind"] = spec.exposure_outcome_blind
    sample["exposure_inference_scope"] = spec.exposure_inference_scope
    return sample.sort_values("commodity", ignore_index=True)


def assign_resampling_blocks(
    dates: pd.Series, *, start_month: int, length_years: int
) -> pd.Series:
    """Group months into consecutive resampling blocks of a given length.

    Blocks are cut on an absolute month index, so ``start_month=5`` and
    ``length_years=1`` reproduce the conventional May-April ENSO year, and
    longer blocks simply merge consecutive ones. The label is the block's index,
    which is all the bootstrap needs.
    """
    if not 1 <= start_month <= 12:
        raise ValueError("start_month must fall between one and twelve")
    if length_years < 1:
        raise ValueError("length_years must be positive")
    stamps = pd.to_datetime(dates)
    months = stamps.dt.year * 12 + (stamps.dt.month - 1) - (start_month - 1)
    return (months // (12 * length_years)).astype("int64")


def attach_resampling_blocks(
    panel: pd.DataFrame, definitions: tuple[BlockDefinition, ...]
) -> pd.DataFrame:
    """Add one block-label column per alternative block definition."""
    result = panel.copy()
    for definition in definitions:
        result[definition.column] = assign_resampling_blocks(
            result["date"],
            start_month=definition.start_month,
            length_years=definition.length_years,
        )
    return result


def build_panel(
    monthly: pd.DataFrame,
    enso: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    index_name: str,
    lag_months: int,
    outcome_column: str,
    weighting: str,
) -> pd.DataFrame:
    """Assemble the estimation panel for one index, lag and weighting cell."""
    if weighting not in ALLOWED_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(ALLOWED_WEIGHTINGS)}")
    if lag_months < 0:
        raise ValueError("lag_months must be non-negative")
    monthly_required = {"date", "commodity", outcome_column}
    if not monthly_required.issubset(monthly.columns):
        raise ValueError(
            f"Monthly data is missing columns: {sorted(monthly_required - set(monthly))}"
        )
    if not {"date", index_name}.issubset(enso.columns):
        raise ValueError(
            f"ENSO data is missing columns: {sorted({'date', index_name} - set(enso))}"
        )

    signal = enso.loc[:, ["date", index_name]].copy()
    signal["date"] = pd.to_datetime(signal["date"])
    if signal["date"].duplicated().any():
        raise ValueError("ENSO dates must be unique")
    signal = signal.sort_values("date", ignore_index=True)
    # Shift the index forward so month t carries the state from t - lag_months.
    signal["date"] = signal["date"] + pd.DateOffset(months=lag_months)
    signal = signal.rename(columns={index_name: "enso_index"})

    panel = monthly.loc[:, ["date", "commodity", outcome_column]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if panel.duplicated(["date", "commodity"]).any():
        raise ValueError("Monthly date/commodity keys must be unique")
    weight_column = "exposure_weight" if weighting == "exposure" else "uniform_weight"
    panel = panel.merge(
        exposure.loc[:, ["commodity", "role", "group", weight_column, "is_negative_control"]],
        on="commodity",
        how="inner",
        validate="many_to_one",
    ).merge(signal, on="date", how="left", validate="many_to_one")
    panel = panel.rename(columns={weight_column: "weight", outcome_column: "outcome"})
    panel[EXPOSURE_TERM] = panel["weight"] * panel["enso_index"]
    panel[CONTROL_TERM] = panel["is_negative_control"].astype(float) * panel["enso_index"]
    panel["calendar_year"] = panel["date"].dt.year
    complete = panel["outcome"].notna() & panel["enso_index"].notna()
    return panel.loc[complete].sort_values(["commodity", "date"], ignore_index=True)


def two_way_within_transform(
    values: np.ndarray,
    unit_codes: np.ndarray,
    time_codes: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    """Sweep out both fixed effects by alternating projections.

    Demeaning by commodity and by month in turn converges to the two-way within
    transform, which is the same estimate a full dummy regression gives without
    ever building the several-hundred-column design matrix.
    """
    residual = np.array(values, dtype="float64", copy=True)
    unit_count = int(unit_codes.max()) + 1 if unit_codes.size else 0
    time_count = int(time_codes.max()) + 1 if time_codes.size else 0
    unit_sizes = np.bincount(unit_codes, minlength=unit_count).astype("float64")
    time_sizes = np.bincount(time_codes, minlength=time_count).astype("float64")
    for _ in range(max_iterations):
        largest_change = 0.0
        for codes, sizes, count in (
            (unit_codes, unit_sizes, unit_count),
            (time_codes, time_sizes, time_count),
        ):
            for column in range(residual.shape[1]):
                totals = np.bincount(codes, weights=residual[:, column], minlength=count)
                shift = (totals / sizes)[codes]
                residual[:, column] -= shift
                largest_change = max(largest_change, float(np.abs(shift).max(initial=0.0)))
        if largest_change < tolerance:
            break
    return residual


def _fit_demeaned(
    outcome: np.ndarray, design: np.ndarray, *, units: int, periods: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "The demeaned design is rank deficient: the interaction terms are "
            "collinear once the fixed effects are swept out"
        )
    gram = design.T @ design
    coefficients = np.linalg.solve(gram, design.T @ outcome)
    residual = outcome - design @ coefficients
    residual_sum_squares = float(residual @ residual)
    total_sum_squares = float(outcome @ outcome)
    within_r_squared = (
        1 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else float("nan")
    )
    degrees_of_freedom = len(outcome) - design.shape[1] - (units - 1) - (periods - 1) - 1
    variance = (
        residual_sum_squares / degrees_of_freedom * np.linalg.inv(gram)
        if degrees_of_freedom > 0
        else np.full((design.shape[1], design.shape[1]), np.nan)
    )
    return coefficients, np.sqrt(np.diag(variance)), within_r_squared, degrees_of_freedom


def fit_exposure_panel(
    panel: pd.DataFrame,
    *,
    tolerance: float,
    max_iterations: int,
    regressors: tuple[str, ...] = REGRESSORS,
) -> PanelFit:
    if panel.empty:
        raise ValueError("The estimation panel is empty")
    if not set(regressors).issubset(panel.columns):
        raise ValueError(f"Panel is missing regressors: {sorted(set(regressors) - set(panel))}")
    unit_codes = pd.factorize(panel["commodity"], sort=True)[0]
    time_codes = pd.factorize(panel["date"], sort=True)[0]
    columns = ["outcome", *regressors]
    demeaned = two_way_within_transform(
        panel.loc[:, columns].to_numpy(dtype="float64"),
        unit_codes,
        time_codes,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    units = int(unit_codes.max()) + 1
    periods = int(time_codes.max()) + 1
    coefficients, errors, within_r_squared, degrees_of_freedom = _fit_demeaned(
        demeaned[:, 0], demeaned[:, 1:], units=units, periods=periods
    )
    return PanelFit(
        coefficients=dict(zip(regressors, coefficients.tolist(), strict=True)),
        naive_standard_errors=dict(zip(regressors, errors.tolist(), strict=True)),
        observations=len(panel),
        units=units,
        periods=periods,
        within_r_squared=within_r_squared,
        residual_degrees_of_freedom=degrees_of_freedom,
        design_condition_number=float(np.linalg.cond(demeaned[:, 1:])),
    )


def year_block_bootstrap(
    panel: pd.DataFrame,
    fit: PanelFit,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    tolerance: float,
    max_iterations: int,
    regressors: tuple[str, ...] = REGRESSORS,
    block_column: str = PRIMARY_BLOCK_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample whole blocks of the cross-section with replacement.

    ENSO is one time series, so months are not independent draws and clustering
    on the cross-section would understate the uncertainty badly. Resampling
    whole blocks keeps each block's cross-sectional correlation and its own
    within-block serial correlation intact. A block drawn twice is relabelled so
    the two copies carry separate month effects, which keeps the replicate a
    coherent alternative history rather than a panel with duplicate periods.

    ``block_column`` selects which grouping to resample. The frozen primary is
    the calendar year; ``BlockDefinition`` explains why the alternatives exist
    and why they can only widen the interval.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must fall between zero and one")
    if block_column not in panel.columns:
        raise ValueError(f"Panel is missing the resampling-block column {block_column!r}")

    years = np.sort(panel[block_column].unique())
    rows_by_year = [np.flatnonzero(panel[block_column].to_numpy() == year) for year in years]
    unit_codes_all = pd.factorize(panel["commodity"], sort=True)[0]
    time_codes_all = pd.factorize(panel["date"], sort=True)[0]
    period_count = int(time_codes_all.max()) + 1
    values_all = panel.loc[:, ["outcome", *regressors]].to_numpy(dtype="float64")
    observed = np.array([fit.coefficients[name] for name in regressors])
    observed_errors = np.array([fit.naive_standard_errors[name] for name in regressors])

    generator = np.random.default_rng(seed)
    replicate_rows: list[dict[str, object]] = []
    estimates = np.full((replicates, len(regressors)), np.nan)
    studentized = np.full((replicates, len(regressors)), np.nan)
    for replicate in range(replicates):
        draw = generator.integers(0, len(years), size=len(years))
        selected = [rows_by_year[position] for position in draw]
        index = np.concatenate(selected)
        offsets = np.concatenate([np.full(len(rows), block) for block, rows in enumerate(selected)])
        unit_codes = pd.factorize(unit_codes_all[index], sort=True)[0]
        time_codes = pd.factorize(offsets * period_count + time_codes_all[index], sort=True)[0]
        demeaned = two_way_within_transform(
            values_all[index],
            unit_codes,
            time_codes,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        try:
            coefficients, errors, _, _ = _fit_demeaned(
                demeaned[:, 0],
                demeaned[:, 1:],
                units=int(unit_codes.max()) + 1,
                periods=int(time_codes.max()) + 1,
            )
        except np.linalg.LinAlgError:
            continue
        estimates[replicate] = coefficients
        with np.errstate(invalid="ignore", divide="ignore"):
            studentized[replicate] = np.where(
                errors > 0, (coefficients - observed) / errors, np.nan
            )
        replicate_rows.append(
            {
                "replicate": replicate,
                **{
                    f"{name}_estimate": float(value)
                    for name, value in zip(regressors, coefficients, strict=True)
                },
            }
        )

    alpha = 1 - confidence_level
    result_rows: list[dict[str, object]] = []
    for position, name in enumerate(regressors):
        column = estimates[:, position]
        column = column[np.isfinite(column)]
        studentized_column = studentized[:, position]
        studentized_column = studentized_column[np.isfinite(studentized_column)]
        centered = column - observed[position]
        observed_t = (
            observed[position] / observed_errors[position]
            if observed_errors[position] > 0
            else float("nan")
        )
        studentized_p = float("nan")
        if studentized_column.size and np.isfinite(observed_t):
            studentized_p = (int(np.sum(np.abs(studentized_column) >= abs(observed_t))) + 1) / (
                studentized_column.size + 1
            )
        result_rows.append(
            {
                "term": name,
                "estimate": float(observed[position]),
                "naive_standard_error": float(observed_errors[position]),
                "block_standard_error": float(column.std(ddof=1))
                if column.size > 1
                else float("nan"),
                "ci_lower": float(np.quantile(column, alpha / 2)) if column.size else float("nan"),
                "ci_upper": float(np.quantile(column, 1 - alpha / 2))
                if column.size
                else float("nan"),
                "block_bootstrap_p_value": (
                    (int(np.sum(np.abs(centered) >= abs(observed[position]))) + 1)
                    / (column.size + 1)
                    if column.size
                    else float("nan")
                ),
                "studentized_p_value": studentized_p,
                "valid_replicates": int(column.size),
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame.from_records(result_rows), pd.DataFrame.from_records(replicate_rows)


def permute_candidate_exposure_weights(
    panel: pd.DataFrame,
    fit: PanelFit,
    *,
    replicates: int,
    seed: int,
    tolerance: float,
    max_iterations: int,
) -> ExposurePermutationOutput:
    """Test whether the candidate-to-weight mapping matters.

    Each replicate shuffles the observed exposure weights across mechanism
    candidates while leaving negative controls fixed at zero. The multiset of
    weights, candidate/control composition, ENSO path, outcomes and fixed
    effects are therefore unchanged. Only the assignment of physical exposure
    to named candidate commodities moves.

    This is a falsification of the hand-authored mapping, not a cure for its
    post-outcome construction. A small p-value says the observed assignment is
    unusual among arbitrary assignments of the same weights; it does not make
    those weights prospectively specified.
    """
    required = {
        "commodity",
        "date",
        "outcome",
        "enso_index",
        "weight",
        "is_negative_control",
        CONTROL_TERM,
    }
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel is missing permutation columns: {sorted(required - set(panel))}")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if EXPOSURE_TERM not in fit.coefficients or CONTROL_TERM not in fit.coefficients:
        raise ValueError("Exposure permutation requires the two-term exposure specification")

    metadata = panel.loc[:, ["commodity", "weight", "is_negative_control"]].drop_duplicates()
    if metadata["commodity"].duplicated().any():
        raise ValueError("Each commodity must have one fixed exposure weight and role")
    controls = metadata.loc[metadata["is_negative_control"]]
    candidates = metadata.loc[~metadata["is_negative_control"]]
    if not controls["weight"].eq(0.0).all():
        raise ValueError("Negative controls must remain fixed at zero during permutation")
    if candidates.empty or candidates["weight"].nunique() < 2:
        raise ValueError("Candidate exposure weights must vary for a mapping permutation")

    unit_codes, unit_names = pd.factorize(panel["commodity"], sort=True)
    time_codes = pd.factorize(panel["date"], sort=True)[0]
    unit_metadata = metadata.set_index("commodity").loc[unit_names]
    candidate_positions = np.flatnonzero(~unit_metadata["is_negative_control"].to_numpy(dtype=bool))
    candidate_weights = unit_metadata.iloc[candidate_positions]["weight"].to_numpy(dtype="float64")

    fixed = two_way_within_transform(
        panel.loc[:, ["outcome", CONTROL_TERM]].to_numpy(dtype="float64"),
        unit_codes,
        time_codes,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    enso = panel["enso_index"].to_numpy(dtype="float64")
    generator = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan, dtype="float64")
    rows: list[dict[str, object]] = []
    units = len(unit_names)
    periods = int(time_codes.max()) + 1

    for replicate in range(replicates):
        unit_weights = np.zeros(units, dtype="float64")
        unit_weights[candidate_positions] = generator.permutation(candidate_weights)
        interaction = unit_weights[unit_codes] * enso
        transformed_interaction = two_way_within_transform(
            interaction[:, None],
            unit_codes,
            time_codes,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )[:, 0]
        design = np.column_stack([transformed_interaction, fixed[:, 1]])
        try:
            coefficients, _, _, _ = _fit_demeaned(
                fixed[:, 0], design, units=units, periods=periods
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        estimate = float(coefficients[0])
        estimates[replicate] = estimate
        rows.append({"replicate": replicate, "exposure_estimate": estimate})

    valid = estimates[np.isfinite(estimates)]
    if not valid.size:
        raise ValueError("No exposure-weight permutation produced an estimable design")
    observed = float(fit.coefficients[EXPOSURE_TERM])
    lower_tail = (int(np.sum(valid <= observed)) + 1) / (len(valid) + 1)
    upper_tail = (int(np.sum(valid >= observed)) + 1) / (len(valid) + 1)
    return ExposurePermutationOutput(
        observed_estimate=observed,
        permutation_mean=float(valid.mean()),
        permutation_median=float(np.median(valid)),
        mapping_percentile=lower_tail,
        p_value=min(1.0, 2 * min(lower_tail, upper_tail)),
        valid_replicates=len(valid),
        replicates=pd.DataFrame.from_records(rows),
    )
