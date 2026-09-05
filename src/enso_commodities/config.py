from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Source:
    name: str
    label: str
    url: str
    landing_page: str
    filename: str
    format: str
    minimum_bytes: int


@dataclass(frozen=True)
class ResearchConfig:
    primary_index: str
    warm_threshold: float
    minimum_duration_months: int
    observable_delay_after_center_months: int
    base_relative_month: int
    first_relative_month: int
    last_relative_month: int
    reported_horizons: tuple[int, ...]
    anchor_types: tuple[str, ...]
    seasonal_estimation_excludes: str
    market_estimation_exclusion_window: tuple[int, int]
    minimum_seasonal_observations: int
    minimum_market_model_observations: int
    market_factor: str
    macro_control_columns: tuple[str, ...]
    macro_estimation_exclusion_window: tuple[int, int]
    minimum_macro_model_observations: int
    macro_estimator: str
    bootstrap_replicates: int
    confidence_level: float
    fdr_alpha: float
    random_seed: int
    placebo_replicates: int
    placebo_neutral_absolute_threshold: float
    placebo_actual_event_exclusion_window: tuple[int, int]
    placebo_match_anchor_calendar_month: bool
    placebo_sample_without_replacement: bool
    placebo_minimum_valid_replicate_share: float
    fragility_method: str
    fragility_inference_gate: str
    fragility_require_sign_agreement: bool
    fragility_require_stable_direction: bool
    fragility_fdr_family: str
    robustness_index_definitions: tuple[str, ...]
    robustness_anchor_types: tuple[str, ...]
    robustness_require_direction_stability: bool
    robustness_require_sign_agreement: bool
    robustness_require_bootstrap_fdr: bool
    robustness_require_placebo_fdr: bool
    robustness_require_primary_fragility_pass: bool
    robustness_placebo_sample_without_replacement: bool


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_sources(path: Path | None = None) -> list[Source]:
    config_path = path or project_root() / "config" / "sources.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), dict):
        raise ValueError(f"Invalid source configuration: {config_path}")

    sources: list[Source] = []
    for name, values in raw["sources"].items():
        if not isinstance(values, dict):
            raise ValueError(f"Invalid configuration for source {name!r}")
        sources.append(Source(name=name, **values))
    return sources


def load_research_config(path: Path | None = None) -> ResearchConfig:
    config_path = path or project_root() / "config" / "research.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    episodes = raw.get("episodes", {})
    returns = raw.get("raw_event_returns", {})
    exclusion_values = tuple(
        int(value) for value in raw["adjustments"]["market_estimation_exclusion_window"]
    )
    if len(exclusion_values) != 2:
        raise ValueError("market_estimation_exclusion_window must have two endpoints")
    placebo = raw["placebo"]
    placebo_exclusion_values = tuple(
        int(value) for value in placebo["actual_event_exclusion_window"]
    )
    if len(placebo_exclusion_values) != 2:
        raise ValueError("actual_event_exclusion_window must have two endpoints")
    macro = raw["macro_adjustments"]
    macro_exclusion_values = tuple(int(value) for value in macro["estimation_exclusion_window"])
    if len(macro_exclusion_values) != 2:
        raise ValueError("macro estimation_exclusion_window must have two endpoints")
    config = ResearchConfig(
        primary_index=str(episodes["primary_index"]),
        warm_threshold=float(episodes["warm_threshold"]),
        minimum_duration_months=int(episodes["minimum_duration_months"]),
        observable_delay_after_center_months=int(episodes["observable_delay_after_center_months"]),
        base_relative_month=int(returns["base_relative_month"]),
        first_relative_month=int(returns["first_relative_month"]),
        last_relative_month=int(returns["last_relative_month"]),
        reported_horizons=tuple(int(value) for value in returns["reported_horizons"]),
        anchor_types=tuple(str(value) for value in returns["anchor_types"]),
        seasonal_estimation_excludes=str(raw["adjustments"]["seasonal_estimation_excludes"]),
        market_estimation_exclusion_window=(exclusion_values[0], exclusion_values[1]),
        minimum_seasonal_observations=int(raw["adjustments"]["minimum_seasonal_observations"]),
        minimum_market_model_observations=int(
            raw["adjustments"]["minimum_market_model_observations"]
        ),
        market_factor=str(raw["adjustments"]["market_factor"]),
        macro_control_columns=tuple(str(value) for value in macro["controls"]),
        macro_estimation_exclusion_window=(
            macro_exclusion_values[0],
            macro_exclusion_values[1],
        ),
        minimum_macro_model_observations=int(macro["minimum_model_observations"]),
        macro_estimator=str(macro["estimator"]),
        bootstrap_replicates=int(raw["inference"]["bootstrap_replicates"]),
        confidence_level=float(raw["inference"]["confidence_level"]),
        fdr_alpha=float(raw["inference"]["fdr_alpha"]),
        random_seed=int(raw["inference"]["random_seed"]),
        placebo_replicates=int(placebo["replicates"]),
        placebo_neutral_absolute_threshold=float(placebo["neutral_absolute_threshold"]),
        placebo_actual_event_exclusion_window=(
            placebo_exclusion_values[0],
            placebo_exclusion_values[1],
        ),
        placebo_match_anchor_calendar_month=bool(placebo["match_anchor_calendar_month"]),
        placebo_sample_without_replacement=bool(placebo["sample_without_replacement"]),
        placebo_minimum_valid_replicate_share=float(placebo["minimum_valid_replicate_share"]),
        fragility_method=str(raw["fragility"]["method"]),
        fragility_inference_gate=str(raw["fragility"]["inference_gate"]),
        fragility_require_sign_agreement=bool(raw["fragility"]["require_sign_agreement"]),
        fragility_require_stable_direction=bool(raw["fragility"]["require_stable_direction"]),
        fragility_fdr_family=str(raw["fragility"]["fdr_family"]),
        robustness_index_definitions=tuple(
            str(value) for value in raw["robustness"]["index_definitions"]
        ),
        robustness_anchor_types=tuple(str(value) for value in raw["robustness"]["anchor_types"]),
        robustness_require_direction_stability=bool(
            raw["robustness"]["require_direction_stability"]
        ),
        robustness_require_sign_agreement=bool(raw["robustness"]["require_sign_agreement"]),
        robustness_require_bootstrap_fdr=bool(raw["robustness"]["require_bootstrap_fdr"]),
        robustness_require_placebo_fdr=bool(raw["robustness"]["require_placebo_fdr"]),
        robustness_require_primary_fragility_pass=bool(
            raw["robustness"]["require_primary_fragility_pass"]
        ),
        robustness_placebo_sample_without_replacement=bool(
            raw["robustness"]["placebo_sample_without_replacement"]
        ),
    )
    if config.primary_index not in {"roni", "oni"}:
        raise ValueError("primary_index must be 'roni' or 'oni'")
    if config.minimum_duration_months < 1:
        raise ValueError("minimum_duration_months must be positive")
    if not config.first_relative_month <= config.base_relative_month <= config.last_relative_month:
        raise ValueError("base_relative_month must fall inside the event window")
    if not set(config.anchor_types).issubset({"retrospective", "observable"}):
        raise ValueError("anchor_types contains an unsupported value")
    if config.seasonal_estimation_excludes != "qualifying_episode_months":
        raise ValueError("Unsupported seasonal_estimation_excludes value")
    if config.market_estimation_exclusion_window[0] > config.market_estimation_exclusion_window[1]:
        raise ValueError("market_estimation_exclusion_window endpoints are reversed")
    if config.minimum_seasonal_observations < 1:
        raise ValueError("minimum_seasonal_observations must be positive")
    if config.minimum_market_model_observations < 3:
        raise ValueError("minimum_market_model_observations must be at least three")
    if config.market_factor != "world_bank_total_index":
        raise ValueError("Unsupported market_factor")
    expected_macro_controls = (
        "market_seasonal_adjusted_log_return",
        "us_neer_log_change",
        "us_cpi_log_change",
        "global_real_activity_change",
    )
    if config.macro_control_columns != expected_macro_controls:
        raise ValueError("Unsupported macro-control specification")
    if config.macro_estimation_exclusion_window[0] > config.macro_estimation_exclusion_window[1]:
        raise ValueError("macro estimation-exclusion endpoints are reversed")
    if config.minimum_macro_model_observations < 5:
        raise ValueError("minimum_macro_model_observations must be at least five")
    if config.macro_estimator != "ordinary_least_squares_with_intercept":
        raise ValueError("Unsupported macro estimator")
    if config.bootstrap_replicates < 999:
        raise ValueError("bootstrap_replicates must be at least 999")
    if not 0 < config.confidence_level < 1:
        raise ValueError("confidence_level must fall between zero and one")
    if not 0 < config.fdr_alpha < 1:
        raise ValueError("fdr_alpha must fall between zero and one")
    if config.placebo_replicates < 999:
        raise ValueError("placebo replicates must be at least 999")
    if config.placebo_neutral_absolute_threshold <= 0:
        raise ValueError("placebo neutral_absolute_threshold must be positive")
    if (
        config.placebo_actual_event_exclusion_window[0]
        > config.placebo_actual_event_exclusion_window[1]
    ):
        raise ValueError("placebo actual_event_exclusion_window endpoints are reversed")
    if not config.placebo_match_anchor_calendar_month:
        raise ValueError("placebo anchors must match the real onset calendar-month distribution")
    if not config.placebo_sample_without_replacement:
        raise ValueError("placebo anchors must be sampled without replacement within a replicate")
    if not 0 < config.placebo_minimum_valid_replicate_share <= 1:
        raise ValueError("placebo minimum_valid_replicate_share must fall in (0, 1]")
    if config.fragility_method != "leave_one_episode_out":
        raise ValueError("Unsupported fragility method")
    if config.fragility_inference_gate != "bootstrap_fdr":
        raise ValueError("Unsupported fragility inference gate")
    if not config.fragility_require_sign_agreement:
        raise ValueError("Fragility gate must require sign agreement")
    if not config.fragility_require_stable_direction:
        raise ValueError("Fragility gate must require stable direction")
    if config.fragility_fdr_family != "mechanism_candidates_only":
        raise ValueError("Fragility FDR family must contain mechanism candidates only")
    if config.robustness_index_definitions != ("roni", "oni"):
        raise ValueError("Robustness indices must be RONI and ONI")
    if config.robustness_anchor_types != ("retrospective", "observable"):
        raise ValueError("Robustness anchors must be retrospective and observable")
    robustness_requirements = [
        config.robustness_require_direction_stability,
        config.robustness_require_sign_agreement,
        config.robustness_require_bootstrap_fdr,
        config.robustness_require_placebo_fdr,
        config.robustness_require_primary_fragility_pass,
    ]
    if not all(robustness_requirements):
        raise ValueError("All frozen robustness gates must remain enabled")
    if config.robustness_placebo_sample_without_replacement:
        raise ValueError("Robustness placebos must use replacement for sparse calendar cells")
    return config
