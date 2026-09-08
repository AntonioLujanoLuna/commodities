"""W3: commodity price responses to ENSO forecast revisions rather than onsets.

The whole preceding program dates an onset and measures what followed. But ENSO
is forecast six to nine months ahead and those forecasts are public, so an onset
declaration is not news -- it confirms something the market has had months to
price. An efficient market with a real and material ENSO effect would therefore
produce exactly the null event-study results this project already has.

Nothing else in the repository separates that explanation from "there is no
effect". This module does, by making the treatment the *revision*: the change
between consecutive monthly issuances in the forecast probability of El Nino
for the same target season. Revisions arrive monthly rather than seventeen times
in sixty-five years, and they are news by construction.

Two properties of the design are worth stating because they are what make it
readable:

* The revision compares successive issuances for a FIXED target season. A
  difference taken across leads would confound the news with the passage of
  time, since a forecast naturally sharpens as its target approaches.
* The lead placebo -- a response to next month's revision -- must be zero. A
  non-zero one means the revision series is picking up something already in
  prices, and the contemporaneous coefficient is then not a news response at
  all. The contract voids the stage on it.
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
from .statistics import benjamini_hochberg

REQUIRED_ARCHIVE_COLUMNS = {
    "issue_date",
    "target_center_date",
    "lead_months",
    "probability_el_nino",
    "probability_neutral",
    "probability_la_nina",
}


@dataclass(frozen=True)
class NewsSpec:
    config: dict[str, Any]
    source_probability: str
    primary_lead_months: int
    sensitivity_lead_months: tuple[int, ...]
    minimum_issuances: int
    outcome: str
    primary_relative_month: int
    secondary_relative_month: int
    lag_relative_months: tuple[int, ...]
    lead_relative_months: tuple[int, ...]
    minimum_observations: int
    replicates: int
    random_seed: int
    fdr_alpha: float


def load_news_config(path: Path | None = None) -> NewsSpec:
    config_path = path or project_root() / "config" / "forecast_news.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    if raw["provenance"] != {
        "version": 1,
        "authored_on": date(2026, 9, 8),
        "inference_scope": "prespecified_primary_endpoint",
        "program_register": "config/findings_v3.yaml",
        "workstream": "W3_forecast_news",
    }:
        raise ValueError("Forecast-news provenance is not the frozen W3 contract")
    treatment = raw["treatment"]
    if treatment["revision_basis"] != "same_target_successive_issuances":
        raise ValueError("A revision must compare successive issuances for the same target")
    if not treatment["require_consecutive_monthly_issuances"]:
        raise ValueError("W3 revisions require consecutive monthly issuances")
    placebos = raw["placebos"]
    if not placebos["lead_placebo_must_not_reject"]:
        raise ValueError("W3 must keep its lead placebo enabled")
    if not placebos["void_stage_if_lead_placebo_rejects"]:
        raise ValueError("W3 must void the stage when the lead placebo rejects")
    if not placebos["lead_relative_months"]:
        raise ValueError("W3 requires at least one lead placebo month")
    inference = raw["inference"]
    if inference["reference_distribution"] != "cluster_robust_wild_bootstrap":
        raise ValueError("W3 requires the cluster-robust wild bootstrap")
    if not inference["impose_null_on_residuals"]:
        raise ValueError("The wild bootstrap must impose the null on the residuals")
    falsification = raw["falsification"]
    if not falsification["void_stage_if_controls_reject"]:
        raise ValueError("W3 must void the stage when its negative controls reject")
    design = raw["design"]
    if design["covariance"] != "cluster_by_calendar_year":
        raise ValueError("Unsupported covariance for W3")
    return NewsSpec(
        config=raw,
        source_probability=str(treatment["source_probability"]),
        primary_lead_months=int(treatment["primary_lead_months"]),
        sensitivity_lead_months=tuple(int(v) for v in treatment["sensitivity_lead_months"]),
        minimum_issuances=int(treatment["minimum_issuances"]),
        outcome=str(design["outcome"]),
        primary_relative_month=int(design["primary_relative_month"]),
        secondary_relative_month=int(design["secondary_relative_month"]),
        lag_relative_months=tuple(int(v) for v in placebos["lag_relative_months"]),
        lead_relative_months=tuple(int(v) for v in placebos["lead_relative_months"]),
        minimum_observations=int(design["minimum_observations"]),
        replicates=int(inference["replicates"]),
        random_seed=int(inference["random_seed"]),
        fdr_alpha=float(inference["fdr_alpha"]),
    )


def validate_forecast_archive(archive: pd.DataFrame, *, tolerance: float = 0.01) -> None:
    """Reject an archive that could smuggle hindsight into the treatment.

    The design rests entirely on each probability having been available on its
    stated issuance date. A back-filled or revised probability would make every
    coefficient downstream a look-ahead artifact, so the checks here are
    deliberately unforgiving and there is no path that repairs a bad row.
    """
    missing = REQUIRED_ARCHIVE_COLUMNS - set(archive.columns)
    if missing:
        raise ValueError(f"Forecast archive is missing columns: {sorted(missing)}")
    if archive.empty:
        raise ValueError("Forecast archive is empty")
    frame = archive.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    frame["target_center_date"] = pd.to_datetime(frame["target_center_date"])
    issue_months = frame["issue_date"].dt.to_period("M")
    target_months = frame["target_center_date"].dt.to_period("M")
    if target_months.lt(issue_months).any():
        raise ValueError("Forecast archive targets a season before its own issuance")
    if frame.duplicated(["issue_date", "target_center_date"]).any():
        raise ValueError("Forecast archive has duplicate issuance/target pairs")
    probabilities = frame.loc[
        :, ["probability_el_nino", "probability_neutral", "probability_la_nina"]
    ].apply(pd.to_numeric, errors="coerce")
    if probabilities.isna().any().any():
        raise ValueError("Forecast probabilities must be complete and numeric")
    if probabilities.lt(0).any().any() or probabilities.gt(1).any().any():
        raise ValueError("Forecast probabilities must fall in [0, 1]")
    probability_sums = probabilities.sum(axis=1).to_numpy(dtype="float64")
    rounding_tolerance = tolerance + 10 * np.finfo(float).eps
    if (~np.isclose(probability_sums, 1.0, rtol=0.0, atol=rounding_tolerance)).any():
        raise ValueError("Forecast probabilities must sum to one")
    implied = (
        (frame["target_center_date"].dt.year - frame["issue_date"].dt.year) * 12
        + frame["target_center_date"].dt.month
        - frame["issue_date"].dt.month
    )
    if implied.ne(pd.to_numeric(frame["lead_months"])).any():
        raise ValueError("lead_months disagrees with the issuance and target dates")


def build_revision_series(
    archive: pd.DataFrame, *, lead_months: int, probability_column: str
) -> pd.DataFrame:
    """Month-on-month revisions to the probability of a fixed target season.

    For each target season the probabilities are ordered by issuance and
    differenced. A revision is kept only when the previous issuance is exactly
    one month earlier, so a gap in the archive produces no observation rather
    than a spurious jump spanning it.
    """
    validate_forecast_archive(archive)
    frame = archive.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    frame["target_center_date"] = pd.to_datetime(frame["target_center_date"])
    frame = frame.sort_values(["target_center_date", "issue_date"], ignore_index=True)
    grouped = frame.groupby("target_center_date", observed=True)
    frame["previous_probability"] = grouped[probability_column].shift(1)
    frame["previous_issue_date"] = grouped["issue_date"].shift(1)
    months_since = (
        (frame["issue_date"].dt.year - frame["previous_issue_date"].dt.year) * 12
        + frame["issue_date"].dt.month
        - frame["previous_issue_date"].dt.month
    )
    frame["revision"] = np.where(
        months_since.eq(1), frame[probability_column] - frame["previous_probability"], np.nan
    )
    selected = frame.loc[
        pd.to_numeric(frame["lead_months"]).eq(lead_months) & frame["revision"].notna(),
        ["issue_date", "target_center_date", "lead_months", probability_column, "revision"],
    ].copy()
    selected["issue_month"] = selected["issue_date"].values.astype("datetime64[M]")
    selected = selected.sort_values("issue_date", ignore_index=True)
    if selected["issue_month"].duplicated().any():
        raise ValueError("More than one revision at this lead falls in the same issuance month")
    return selected


def align_returns(
    revisions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    outcome_column: str,
    relative_month: int,
) -> pd.DataFrame:
    """Attach each commodity's return at ``relative_month`` to every revision.

    Relative month zero is the month of issuance. Negative values look before it,
    which is what the lead placebo needs.
    """
    required = {"date", "commodity", outcome_column}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"Monthly returns are missing columns: {sorted(missing)}")
    prices = returns.loc[:, ["date", "commodity", outcome_column]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    aligned = revisions.loc[:, ["issue_month", "revision"]].copy()
    aligned["date"] = aligned["issue_month"] + pd.DateOffset(months=relative_month)
    merged = aligned.merge(prices, on="date", how="inner")
    merged["calendar_month"] = merged["date"].dt.month
    merged["cluster"] = merged["date"].dt.year
    merged["relative_month"] = relative_month
    return merged.loc[merged[outcome_column].notna()].reset_index(drop=True)


def _design_matrix(sample: pd.DataFrame) -> np.ndarray:
    """Intercept, revision, and calendar-month effects."""
    months = pd.get_dummies(sample["calendar_month"], prefix="m", drop_first=True, dtype=float)
    return np.column_stack(
        [
            np.ones(len(sample)),
            sample["revision"].to_numpy(dtype="float64"),
            months.to_numpy(dtype="float64"),
        ]
    )


def _cluster_codes(clusters: np.ndarray) -> tuple[np.ndarray, int]:
    unique, codes = np.unique(clusters, return_inverse=True)
    return codes.astype("int64"), len(unique)


def cluster_robust_slopes(
    design: np.ndarray,
    outcomes: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope on the revision and its cluster-robust t, for many outcome vectors.

    ``outcomes`` is (observations, draws); one column is the ordinary single
    regression. The many-column form exists so the wild bootstrap runs as linear
    algebra rather than as one refit per replicate, and both paths share this
    code so the observed statistic and its null cannot drift apart.

    Only the [1, 1] element of the sandwich is needed, and with ``a`` the second
    row of the inverse Gram matrix that element is the sum over clusters of
    ``(a . s_g)**2``, where ``s_g`` is the cluster's score. That identity is what
    makes the vectorised form cheap.
    """
    if outcomes.ndim != 2:
        raise ValueError("outcomes must be two-dimensional")
    draws = outcomes.shape[1]
    nan = np.full(draws, np.nan)
    gram = design.T @ design
    try:
        inverse = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return nan, nan, nan
    beta = inverse @ (design.T @ outcomes)
    residual = outcomes - design @ beta
    scores = np.zeros((cluster_count, design.shape[1], draws))
    np.add.at(scores, cluster_codes, design[:, :, None] * residual[:, None, :])
    projected = np.tensordot(inverse[1], scores, axes=([0], [1]))
    variance = np.sum(projected**2, axis=0)
    if cluster_count > 1:
        # Standard finite-cluster correction. With a few dozen years of clusters
        # it matters, and the wild bootstrap is what actually carries the
        # inference either way.
        variance = variance * cluster_count / (cluster_count - 1)
    slopes = beta[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        errors = np.where(np.isfinite(variance) & (variance > 0), np.sqrt(variance), np.nan)
        statistics = np.where(np.isfinite(errors) & (errors > 0), slopes / errors, np.nan)
    return slopes, errors, statistics


def _cluster_robust_t(
    design: np.ndarray, outcome: np.ndarray, clusters: np.ndarray
) -> tuple[float, float, float]:
    """OLS slope on the revision with a cluster-robust standard error."""
    codes, count = _cluster_codes(clusters)
    slopes, errors, statistics = cluster_robust_slopes(design, outcome.reshape(-1, 1), codes, count)
    return float(slopes[0]), float(errors[0]), float(statistics[0])


def news_regression(
    panel: pd.DataFrame, *, outcome_column: str, minimum_observations: int
) -> pd.DataFrame:
    """Per-commodity response of the return to the revision."""
    rows: list[dict[str, Any]] = []
    for commodity, sample in panel.groupby("commodity", observed=True):
        sample = sample.dropna(subset=[outcome_column, "revision"])
        if len(sample) < minimum_observations or sample["revision"].nunique() < 2:
            rows.append(
                {
                    "commodity": str(commodity),
                    "observations": len(sample),
                    "clusters": int(sample["cluster"].nunique()),
                    "coefficient": float("nan"),
                    "standard_error": float("nan"),
                    "t_statistic": float("nan"),
                    "status": "insufficient_observations",
                }
            )
            continue
        design = _design_matrix(sample)
        outcome = sample[outcome_column].to_numpy(dtype="float64")
        clusters = sample["cluster"].to_numpy()
        coefficient, error, statistic = _cluster_robust_t(design, outcome, clusters)
        rows.append(
            {
                "commodity": str(commodity),
                "observations": len(sample),
                "clusters": int(sample["cluster"].nunique()),
                "coefficient": coefficient,
                "standard_error": error,
                "t_statistic": statistic,
                "status": "estimated" if np.isfinite(statistic) else "degenerate_scale",
            }
        )
    return pd.DataFrame.from_records(rows)


def wild_cluster_bootstrap(
    panel: pd.DataFrame,
    *,
    outcome_column: str,
    commodities: list[str],
    replicates: int,
    seed: int,
    minimum_observations: int,
) -> np.ndarray:
    """Joint null of the per-commodity t statistics under the imposed null.

    Rademacher weights are drawn once per replicate at the cluster level and
    reused across every commodity, so a replicate remains one coherent
    alternative history rather than an independent redraw per series. That is
    what makes the column-wise maximum a valid family-wise reference.
    """
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for commodity in commodities:
        sample = panel.loc[panel["commodity"].eq(commodity)].dropna(
            subset=[outcome_column, "revision"]
        )
        if len(sample) < minimum_observations or sample["revision"].nunique() < 2:
            continue
        design = _design_matrix(sample)
        # The restricted design drops the revision, which imposes the null on the
        # residuals that are then reweighted.
        restricted = np.delete(design, 1, axis=1)
        outcome = sample[outcome_column].to_numpy(dtype="float64")
        try:
            restricted_beta = np.linalg.solve(restricted.T @ restricted, restricted.T @ outcome)
        except np.linalg.LinAlgError:
            continue
        prepared[commodity] = {
            "design": design,
            "fitted": restricted @ restricted_beta,
            "residual": outcome - restricted @ restricted_beta,
            "clusters": sample["cluster"].to_numpy(),
        }
    if not prepared:
        return np.empty((0, len(commodities)))

    # One Rademacher draw per cluster per replicate, shared across commodities.
    all_clusters = np.unique(np.concatenate([item["clusters"] for item in prepared.values()]))
    lookup = {cluster: position for position, cluster in enumerate(all_clusters)}
    generator = np.random.default_rng(seed)
    weights = generator.choice([-1.0, 1.0], size=(len(all_clusters), replicates))

    null = np.full((replicates, len(commodities)), np.nan)
    for position, commodity in enumerate(commodities):
        item = prepared.get(commodity)
        if item is None:
            continue
        rows = np.array([lookup[cluster] for cluster in item["clusters"]], dtype="int64")
        synthetic = item["fitted"][:, None] + weights[rows] * item["residual"][:, None]
        codes, count = _cluster_codes(item["clusters"])
        null[:, position] = cluster_robust_slopes(item["design"], synthetic, codes, count)[2]
    return null


def run_news_inference(
    revisions: pd.DataFrame,
    returns: pd.DataFrame,
    roles: pd.Series,
    *,
    spec: NewsSpec,
    program_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Estimate the contemporaneous response and every registered placebo."""
    windows: dict[str, int] = {"primary": spec.primary_relative_month}
    windows["secondary"] = spec.secondary_relative_month
    for month in spec.lag_relative_months:
        windows[f"lag_placebo_{month}"] = month
    for month in spec.lead_relative_months:
        windows[f"lead_placebo_{month}"] = month

    frames: list[pd.DataFrame] = []
    primary: pd.DataFrame | None = None
    primary_null = np.empty((0, 0))
    commodities = sorted(str(name) for name in roles.index)
    for label, relative_month in windows.items():
        panel = align_returns(
            revisions,
            returns.loc[returns["commodity"].isin(roles.index)],
            outcome_column=spec.outcome,
            relative_month=relative_month,
        )
        estimates = news_regression(
            panel, outcome_column=spec.outcome, minimum_observations=spec.minimum_observations
        )
        null = wild_cluster_bootstrap(
            panel,
            outcome_column=spec.outcome,
            commodities=commodities,
            replicates=spec.replicates,
            seed=spec.random_seed,
            minimum_observations=spec.minimum_observations,
        )
        p_values: list[float] = []
        for _, row in estimates.iterrows():
            position = commodities.index(str(row["commodity"]))
            column = null[:, position] if null.size else np.empty(0)
            finite = column[np.isfinite(column)]
            statistic = float(row["t_statistic"])
            p_values.append(
                (1 + int(np.sum(np.abs(finite) >= abs(statistic)))) / (1 + finite.size)
                if finite.size and np.isfinite(statistic)
                else float("nan")
            )
        estimates["wild_bootstrap_p_value"] = p_values
        estimates["role"] = [roles.get(name, "unclassified") for name in estimates["commodity"]]
        estimates["window"] = label
        estimates["relative_month"] = relative_month
        frames.append(estimates)
        if label == "primary":
            primary = estimates
            primary_null = null

    results = pd.concat(frames, ignore_index=True)
    assert primary is not None

    candidates = primary["role"].eq("mechanism_candidate")
    primary = primary.copy()
    primary["bh_q_value"] = np.nan
    if candidates.any():
        primary.loc[candidates, "bh_q_value"] = benjamini_hochberg(
            primary.loc[candidates].set_index("commodity")["wild_bootstrap_p_value"]
        ).to_numpy()
    primary["reject_fdr"] = primary["bh_q_value"].le(spec.fdr_alpha).fillna(False)
    results = results.merge(
        primary.loc[:, ["commodity", "window", "bh_q_value", "reject_fdr"]],
        on=["commodity", "window"],
        how="left",
    )

    family_statistic = float("nan")
    family_p = float("nan")
    if candidates.any() and primary_null.size:
        candidate_names = primary.loc[candidates, "commodity"].tolist()
        positions = [commodities.index(name) for name in candidate_names]
        observed = primary.loc[candidates, "t_statistic"].abs()
        if observed.notna().any():
            family_statistic = float(observed.max())
            with np.errstate(invalid="ignore"):
                maxima = np.nanmax(np.abs(primary_null[:, positions]), axis=1)
            finite = maxima[np.isfinite(maxima)]
            if finite.size:
                family_p = float((1 + int(np.sum(finite >= family_statistic))) / (1 + finite.size))

    def rejections(window_prefixes: tuple[str, ...], role: str | None = None) -> int:
        rows = results.loc[results["window"].str.startswith(window_prefixes)]
        if role is not None:
            rows = rows.loc[rows["role"].eq(role)]
        return int(rows["wild_bootstrap_p_value"].lt(spec.fdr_alpha).sum())

    lead_rejections = rejections(("lead_placebo_",), "mechanism_candidate")
    lag_rejections = rejections(("lag_placebo_",), "mechanism_candidate")
    control_rejections = int(
        results.loc[
            results["window"].eq("primary") & results["role"].eq("negative_control"),
            "wild_bootstrap_p_value",
        ]
        .lt(spec.fdr_alpha)
        .sum()
    )

    clears_program = bool(np.isfinite(family_p) and family_p <= program_threshold)
    if lead_rejections:
        status = "void_lead_placebo_rejected"
    elif control_rejections:
        status = "void_negative_controls_rejected"
    elif clears_program:
        status = "program_level_finding"
    else:
        status = "exploratory_within_stage_only"

    summary: dict[str, Any] = {
        "revisions": len(revisions),
        "primary_lead_months": spec.primary_lead_months,
        "family_size": int(candidates.sum()),
        "family_maximum_absolute_t": family_statistic,
        "family_wild_bootstrap_p_value": family_p,
        "candidate_fdr_rejections": int(primary.loc[candidates, "reject_fdr"].sum()),
        "lead_placebo_rejections": lead_rejections,
        "lag_placebo_rejections": lag_rejections,
        "control_rejections": control_rejections,
        "program_threshold": program_threshold,
        "clears_program_threshold": clears_program,
        "status": status,
        "interpretation": (
            "A response to revisions with none to realizations is a market-efficiency "
            "finding; no response to either is evidence against a material effect that "
            "the anticipation defence cannot explain away."
        ),
    }
    return results, primary, summary
