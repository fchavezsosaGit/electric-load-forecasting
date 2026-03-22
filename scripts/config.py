"""Shared pipeline configuration loaded from TOML files.

This module is the single source of truth for paths, supported resolutions,
feature settings, schemas, model feature sets, and modeling runtime policy.
"""

from __future__ import annotations

import math
import os
import tomllib
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"


class SchemaSpec(TypedDict):
    """Schema definition for each pipeline layer."""

    columns: list[str]
    allowed_day_class: list[str]
    required_not_null: list[str]


class DatasetConfig(TypedDict):
    """Central dataset identity used to namespace artifacts."""

    load_type: str
    artifact_namespace: str


class EDAConfig(TypedDict):
    """Centralized notebook and EDA analysis defaults."""

    zscore_threshold: float
    histogram_bins: int
    figure_size: tuple[int, int]
    figure_size_wide: tuple[int, int]
    figure_size_compact: tuple[int, int]
    figure_size_grid: tuple[int, int]
    figure_size_correlation: tuple[int, int]
    day_class_colors: dict[str, str]
    seaborn_style: str
    physical_load_max_watts: float
    physical_load_min_watts: float
    correlation_high_threshold: float
    top_correlations_count: int
    zero_run_threshold_seconds: int
    overlay_resample_frequency: str
    legend_max_labels: int
    percentiles: list[float]
    distribution_features: list[str]


class MultiresConfig(TypedDict):
    """Centralized multi-resolution experiment defaults."""

    enabled: bool
    mode: str
    comparison_mode: str
    resolutions: list[str]
    horizons_minutes: list[int]
    matched_strategies: list[str]
    native_step_enabled: bool
    write_latest: bool


class MultiresSelectionConfig(TypedDict):
    """Selection gates for multi-resolution winner choice."""

    min_eval_coverage: float
    max_fold_std_mae_ratio: float
    min_practical_mae_gain_pct: float
    min_practical_rmse_gain_pct: float
    max_candidate_runtime_minutes: int
    pareto_enabled: bool


class MultiresBaselineConfig(TypedDict):
    """Baseline toggles for multiresolution comparison and rollout runs."""

    include_persistence: bool
    include_previous_day: bool
    include_avg_workday: bool
    include_anchored_workday: bool
    include_hybrid_workday: bool


class MultiresHybridConfig(TypedDict):
    """Config for persistence/profile hybrid baselines."""

    persistence_weight_start: float
    persistence_weight_end: float
    curve: str


class MultiresModeProfile(TypedDict):
    """Per-mode execution scope for Stage-6 multiresolution runs."""

    n_folds: int
    val_window_days: int
    origins_per_fold: int
    resolutions: list[str]
    horizons_minutes: list[int]
    feature_sets: list[str]
    model_labels: list[str]


class MultiresRuntimeConfig(TypedDict):
    """Runtime and caching configuration for multi-resolution runs."""

    enable_cache: bool
    cache_mode: str
    full_runtime_warning_minutes: int
    smoke_origins_per_fold: int
    candidate_origins_per_fold: int
    full_origins_per_fold: int


class MultiresRolloutConfig(TypedDict):
    """Recursive rollout defaults bound to the multiresolution stack."""

    enabled: bool
    selected_resolution: str
    feature_set: str
    model_label: str
    horizon_minutes: int
    strategy: str
    origins_per_run: int
    origin_policy: str
    selection_target: str


class MultiresRolloutChallengerConfig(TypedDict):
    """Config for scheduled Stage-7 challenger sweeps."""

    enabled: bool
    max_candidates: int
    parallel_workers: int
    include_rollout_registry: bool
    include_stage6_registry: bool
    include_horizon_policy_candidates: bool
    include_config_default: bool
    origin_policies: list[str]
    policy_resolutions: list[str]
    recommendation_origin_scope: str


class MultiresRolloutLearnedBlendConfig(TypedDict):
    """Config for learned Stage-7 rollout blend candidates."""

    enabled: bool
    include_persistence_to_raw: bool
    include_persistence_to_residual: bool
    include_raw_to_residual: bool
    include_hybrid_phase_gate: bool
    persistence_weight_start: float
    persistence_weight_end: float
    raw_weight_start: float
    raw_weight_end: float
    hybrid_phase_gate_aligned_weight: float
    hybrid_phase_gate_non_aligned_weight: float
    hybrid_phase_gate_bucket_weights: dict[int, float]
    refinement_enabled: bool
    refinement_step: float
    refinement_neighbors: int
    max_weights_per_family: int
    curve: str


class MultiresRolloutPolicyCandidateConfig(TypedDict):
    """Config for derived Stage-7 rollout policy candidates."""

    enabled: bool
    max_horizon_minutes: int
    selection_targets: list[str]


class MultiresRolloutSweepPolicyConfig(TypedDict):
    """Config for cross-candidate Stage-7 sweep policy synthesis."""

    enabled: bool
    min_horizon_minutes: int
    max_horizon_minutes: int
    origin_policies: list[str]
    selection_targets: list[str]
    min_source_candidates: int
    max_source_candidates: int


class MultiresHorizonCurveConfig(TypedDict):
    """Config for systematic H5 horizon-characterization runs."""

    enabled: bool
    horizons_minutes: list[int]
    origins_per_run: int
    origin_policy: str
    selection_target: str
    max_candidates: int
    include_stage5_anchor: bool
    reuse_existing_sweeps: bool
    write_latest: bool


class ForecastControlConfig(TypedDict):
    """Config for Stage-10 forecast/control loop backtesting."""

    enabled: bool
    day_ahead_horizon_minutes: int
    hourly_horizon_minutes: int
    phase_horizon_minutes: int
    nowcast_horizon_minutes: int
    day_ahead_refresh_enabled: bool
    day_ahead_refresh_interval_minutes: int
    day_ahead_refresh_candidate_label: str
    day_ahead_refresh_lookback_minutes: int
    day_ahead_refresh_residual_drift_mae_pct_threshold: float
    day_ahead_refresh_transition_mae_pct_threshold: float
    day_ahead_refresh_activity_ratio_shift_threshold: float
    day_ahead_refresh_threshold_quantiles: list[float]
    day_ahead_refresh_min_trigger_rate: float
    day_ahead_refresh_max_trigger_rate: float
    day_ahead_refresh_min_profile_gain_fraction_vs_unconditional: float
    day_ahead_refresh_min_lock_gain_fraction_vs_unconditional: float
    day_ahead_refresh_trigger_mode: str
    day_ahead_refresh_candidate_trigger_modes: list[str]
    actual_resolution: str
    lock_interval_minutes: int
    cycle_origin_hour: int
    cycle_origin_minute: int
    cycle_origin_stride_minutes: int
    rolling_benchmark_enabled: bool
    rolling_benchmark_origin_stride_minutes: int
    rolling_benchmark_max_cycles: int
    rolling_benchmark_bootstrap_samples: int
    rolling_benchmark_confidence_level: float
    calibration_splits: list[str]
    evaluation_splits: list[str]
    max_cycles: int
    optimize_replayed_candidates: bool
    control_promotion_scope: str
    control_promotion_guard_enabled: bool
    control_promotion_guard_max_next_lock_regress_pct: float
    control_promotion_guard_max_peak_value_regress_pct: float
    control_promotion_guard_max_peak_miss_regress: float
    allow_baseline_candidates: bool
    candidate_pool_size: int
    candidate_benchmark_origin_cap: int
    phase_candidate_benchmark_origin_cap: int
    phase_candidate_evaluation_origin_cap: int
    phase_control_candidate_pool_size: int
    phase_control_prior_run_limit: int
    phase_control_min_prior_support_runs: int
    phase_control_max_supplemental_contexts_per_resolution: int
    phase_control_exploration_slots: int
    phase_control_origin_cap: int
    phase_stack_native_learned_top_candidates_per_pool: int
    phase_stack_native_baseline_top_candidates_per_pool: int
    phase_stack_blend_weights: list[float]
    phase_stack_blend_parent_top_candidates: int
    phase_stack_bucket_policy_enabled: bool
    phase_stack_bucket_granularity_minutes: int
    phase_stack_baseline_control_blend_enabled: bool
    phase_stack_baseline_control_top_candidates: int
    phase_stack_baseline_control_blend_weights: list[float]
    phase_stack_baseline_control_bucket_blend_enabled: bool
    phase_stack_baseline_control_bucket_size_minutes: int
    phase_stack_guard_enabled: bool
    phase_stack_guard_min_lock_gain_pct: float
    phase_stack_guard_require_rolling_support: bool
    phase_stack_guard_rolling_scope: str
    phase_stack_guard_min_rolling_lock_gain_pct: float
    phase_stack_guard_max_rolling_next_lock_regress_pct: float
    phase_stack_guard_max_rolling_profile_degrade_pct: float
    phase_stack_guard_max_rolling_peak_value_regress_pct: float
    phase_stack_guard_min_rolling_peak_hit_gain: float
    phase_stack_guard_max_rolling_optimizer_regress_pct: float
    phase_stack_guard_max_next_lock_regress_pct: float
    phase_stack_guard_max_profile_degrade_pct: float
    phase_stack_guard_max_peak_value_regress_pct: float
    phase_stack_guard_min_peak_hit_gain: float
    phase_stack_guard_max_optimizer_regress_pct: float
    benchmark_expanded_candidate_pool_size: int
    benchmark_expanded_pool_layers: list[str]
    benchmark_full_origin_layers: list[str]
    nowcast_candidate_pool_size: int
    nowcast_control_blend_enabled: bool
    nowcast_control_blend_weights: list[float]
    nowcast_control_bucket_blend_enabled: bool
    nowcast_control_bucket_size_minutes: int
    nowcast_control_bucket_blend_weights: list[float]
    nowcast_advisory_evidence_enabled: bool
    nowcast_advisory_tie_tolerance: float
    nowcast_dynamic_overlay_enabled: bool
    nowcast_dynamic_overlay_enforce: bool
    nowcast_dynamic_overlay_profile_active_threshold: float
    nowcast_dynamic_overlay_high_ramp_fraction_threshold: float
    nowcast_dynamic_overlay_allow_next_lock: bool
    nowcast_dynamic_overlay_allow_predicted_peak: bool
    nowcast_soft_overlay_shadow_enabled: bool
    nowcast_soft_overlay_supported_weights: list[float]
    nowcast_soft_overlay_background_weights: list[float]
    nowcast_soft_overlay_max_next_lock_regress_pct: float
    nowcast_soft_overlay_max_peak_hit_regress: float
    day_ahead_selection_metric: str
    hourly_selection_metric: str
    phase_selection_metric: str
    phase_stack_selection_metric: str
    nowcast_selection_metric: str
    optimizer_selection_next_lock_weight: float
    optimizer_selection_lock_weight: float
    optimizer_selection_peak_value_weight: float
    optimizer_selection_peak_miss_weight: float
    optimizer_delivery_min_lead_specific_samples: int
    optimizer_delivery_next_lock_min_samples: int
    optimizer_delivery_next_lock_scaled_enabled: bool
    optimizer_delivery_next_lock_scale_floor_quantile: float
    optimizer_delivery_next_lock_scale_floor_min_load: float
    optimizer_delivery_predicted_peak_min_samples: int
    optimizer_delivery_predicted_peak_lead_min_samples: int
    optimizer_delivery_confidence_full_support_n: int
    optimizer_delivery_confidence_band_width_scale_pct: float
    optimizer_delivery_nowcast_cadence_minutes: int
    optimizer_delivery_phase_cadence_minutes: int
    optimizer_delivery_hourly_cadence_minutes: int
    optimizer_delivery_day_ahead_cadence_minutes: int
    optimizer_delivery_nowcast_stale_threshold_minutes: int
    optimizer_delivery_phase_stale_threshold_minutes: int
    optimizer_delivery_hourly_stale_threshold_minutes: int
    optimizer_delivery_day_ahead_stale_threshold_minutes: int
    optimizer_delivery_confidence_layer_multipliers: dict[str, float]
    optimizer_delivery_confidence_quantile_source_multipliers: dict[str, float]
    replay_cache_enabled: bool
    replay_cache_dirname: str
    write_latest: bool


class PerformanceRampConfig(TypedDict):
    """Config for derived ramp-feature generation."""

    quantile: float


class PerformanceBlendSearchConfig(TypedDict):
    """Centralized search space for Stage-5 blend policy selection."""

    enabled: bool
    base_window: int
    base_sharpness: float
    min_weight: float
    max_weight: float
    window_multipliers: list[float]
    sharpness_multipliers: list[float]
    bucket_enabled: bool
    bucket_size_minutes: int
    bucket_cycle_minutes: int
    bucket_candidate_weights: list[float]


ParallelBackend = Literal["threading", "loky", "sequential"]


class ModelingParallelConfig(TypedDict):
    """Shared job execution controls for modeling-heavy stages."""

    enabled: bool
    backend: ParallelBackend
    max_workers: int
    batch_size: int
    pre_dispatch: str
    min_tasks: int
    inner_threads_per_worker: int


class ModelingStageParallelConfig(TypedDict):
    """Per-stage toggle for shared modeling parallelism."""

    enabled: bool
    max_workers: int
    inner_threads_per_worker: int
    high_capacity_host_only: bool


class FourierCycleSpec(TypedDict):
    """Config contract for a cyclical Fourier feature block."""

    source: str
    period: int
    prefix: str


class FeatureConfigSpec(TypedDict):
    """Centralized feature-engineering configuration."""

    lag_periods: list[int]
    rolling_periods: list[int]
    slope_periods: list[int]
    lag_minutes: list[int]
    rolling_minutes: list[int]
    slope_minutes: list[int]
    profile_activity_threshold: float
    fourier_cycles: list[FourierCycleSpec]


class PerformanceHgbSearchConfig(TypedDict):
    """Centralized adaptive HGB screening configuration."""

    enabled: bool
    screen_folds: int
    min_candidates: int
    max_candidates: int
    learning_rates: list[float]
    max_depths: list[int]
    min_samples_leaf: list[int]
    l2_regularization: list[float]
    max_iters: list[int]


class PerformanceHorizonPolicyConfig(TypedDict):
    """Feature/model policy selected by forecast horizon."""

    max_horizon_minutes: int
    feature_sets: list[str]
    model_labels: list[str]
    allow_residual: bool
    allow_blend: bool
    rollout_residual_baseline: str
    rollout_residual_candidates: list[str]
    rollout_origin_policy: str
    rollout_selection_target: str


class PerformanceQuickProfileConfig(TypedDict):
    """Evidence-dense Stage-5 quick profile for one horizon class."""

    feature_sets: list[str]
    model_labels: list[str]
    n_folds: int
    val_window_days: int


class PerformanceEvaluationConfig(TypedDict):
    """Centralized evaluation segmentation configuration."""

    segment_columns: list[str]
    classical_benchmarks: list[str]
    supplemental_surface_splits: list[str]
    supplemental_load_band_quantile: float
    supplemental_ramp_band_quantile: float
    bootstrap_samples: int
    bootstrap_confidence_level: float
    bootstrap_min_block_minutes: int
    bootstrap_max_block_minutes: int
    bootstrap_consecutive_insignificant: int
    importance_repeats: int
    importance_max_features: int
    importance_random_state: int


EDAResolutionMode = Literal["all", "default", "custom"]


def _resolve_config_dir() -> Path:
    """Resolve the configuration directory.

    Set `ELF_CONFIG_DIR` to override the default `PROJECT_ROOT/config` location.
    """
    env_dir = os.getenv("ELF_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return PROJECT_ROOT / "config"


CONFIG_DIR: Final[Path] = _resolve_config_dir()


def _load_toml_file(filename: str) -> dict[str, object]:
    """Load a TOML document from the active config directory."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required config file: {path}. "
            f"Expected config files in {CONFIG_DIR}."
        )
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_path(value: str) -> Path:
    """Resolve a config path value relative to project root when needed."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _as_int_pair(name: str, value: object) -> tuple[int, int]:
    """Validate and normalize a two-element integer list/tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list/tuple. Got: {value!r}")
    first, second = value
    if not isinstance(first, int) or not isinstance(second, int):
        raise ValueError(f"{name} values must be integers. Got: {value!r}")
    return (first, second)


def _as_float(name: str, value: object) -> float:
    """Validate and normalize a numeric scalar into float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric. Got: {value!r}")
    return float(value)


def _as_int(name: str, value: object) -> int:
    """Validate and normalize an integer scalar."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer. Got: {value!r}")
    return value


def _as_bool(name: str, value: object) -> bool:
    """Validate and normalize a boolean scalar."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean. Got: {value!r}")
    return value


def _as_str(name: str, value: object) -> str:
    """Validate and normalize a string scalar."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string. Got: {value!r}")
    return value


def _as_float_list(name: str, value: object) -> list[float]:
    """Validate and normalize a list of numeric values into floats."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list. Got: {value!r}")
    return [_as_float(f"{name}[{idx}]", item) for idx, item in enumerate(value)]


def _as_str_list(name: str, value: object) -> list[str]:
    """Validate and normalize a list of strings."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list. Got: {value!r}")
    output: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{idx}] must be a string. Got: {item!r}")
        output.append(item)
    return output


def _as_str_dict(name: str, value: object) -> dict[str, str]:
    """Validate and normalize a mapping of string keys and values."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping. Got: {value!r}")
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} key must be a string. Got: {key!r}")
        if not isinstance(item, str):
            raise ValueError(f"{name}['{key}'] must be a string. Got: {item!r}")
        output[key] = item
    return output


def _as_float_dict(name: str, value: object) -> dict[str, float]:
    """Validate and normalize a mapping of string keys and numeric values."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping. Got: {value!r}")
    output: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} key must be a string. Got: {key!r}")
        output[key] = _as_float(f"{name}['{key}']", item)
    return output


def _normalize_artifact_namespace(value: str) -> str:
    """Normalize a dataset/load-type label into a filesystem-safe artifact namespace."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    if not normalized:
        raise ValueError(f"Artifact namespace is empty after normalization. Got: {value!r}")
    return normalized


def _as_fourier_cycles(name: str, value: object) -> list[FourierCycleSpec]:
    """Validate and normalize configured Fourier cycle specs."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list. Got: {value!r}")

    cycles: list[FourierCycleSpec] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{idx}] must be a mapping. Got: {item!r}")
        source = _as_str(f"{name}[{idx}].source", item.get("source"))
        period = _as_int(f"{name}[{idx}].period", item.get("period"))
        prefix = _as_str(f"{name}[{idx}].prefix", item.get("prefix"))
        cycles.append({"source": source, "period": period, "prefix": prefix})
    return cycles


_PIPELINE_TOML: Final[dict[str, object]] = _load_toml_file("pipeline.toml")
_EDA_TOML: Final[dict[str, object]] = _load_toml_file("eda.toml")
_MULTIRES_TOML: Final[dict[str, object]] = _load_toml_file("multires.toml")
_MODELING_TOML: Final[dict[str, object]] = _load_toml_file("modeling.toml")

_dataset = cast(dict[str, object], _PIPELINE_TOML["dataset"])
DATASET: Final[DatasetConfig] = {
    "load_type": _as_str("dataset.load_type", _dataset["load_type"]),
    "artifact_namespace": _normalize_artifact_namespace(
        _as_str("dataset.load_type", _dataset["load_type"])
    ),
}

_paths = cast(dict[str, str], _PIPELINE_TOML["paths"])
PATHS: Final[dict[str, Path]] = {
    "raw_mat": _resolve_path(_paths["raw_mat"]),
    "bronze_file": _resolve_path(_paths["bronze_file"]),
    "silver_dir": _resolve_path(_paths["silver_dir"]),
    "gold_dir": _resolve_path(_paths["gold_dir"]),
    "model_dir": _resolve_path(_paths["model_dir"]),
    "logs_dir": _resolve_path(_paths["logs_dir"]),
    "outputs_modeling_dir": _resolve_path(_paths["outputs_modeling_dir"]),
    "outputs_performance_dir": _resolve_path(_paths["outputs_performance_dir"]),
}

_multires_paths = cast(dict[str, str], _MULTIRES_TOML["paths"])
PATHS.update(
    {
        "outputs_multires_dir": _resolve_path(_multires_paths["outputs_multires_dir"]),
        "outputs_rollout_dir": _resolve_path(_multires_paths["outputs_rollout_dir"]),
        "outputs_horizon_curve_dir": _resolve_path(_multires_paths["outputs_horizon_curve_dir"]),
        "outputs_forecast_control_dir": _resolve_path(_multires_paths["outputs_forecast_control_dir"]),
    }
)


def scoped_output_path(base_path: Path) -> Path:
    """Return the dataset-scoped artifact directory for a stage output root."""
    return base_path / DATASET["artifact_namespace"]


def output_path_candidates(base_path: Path) -> tuple[Path, ...]:
    """Return preferred dataset-scoped output roots plus the legacy root for fallback reads."""
    scoped = scoped_output_path(base_path)
    if scoped == base_path:
        return (base_path,)
    return (scoped, base_path)


def preferred_output_path(base_path: Path) -> Path:
    """Return the best existing output root, preferring scoped outputs when present."""
    for candidate in output_path_candidates(base_path):
        if candidate.exists():
            return candidate
    return scoped_output_path(base_path)

_raw_contract = cast(dict[str, object], _PIPELINE_TOML["raw_contract"])
SECONDS_PER_DAY: Final[int] = _as_int(
    "raw_contract.seconds_per_day", _raw_contract["seconds_per_day"]
)
MATLAB_REQUIRED_KEYS: Final[tuple[str, ...]] = tuple(
    _as_str_list("raw_contract.required_keys", _raw_contract["required_keys"])
)

_resolutions = cast(dict[str, object], _PIPELINE_TOML["resolutions"])
SUPPORTED_RESOLUTIONS: Final[tuple[str, ...]] = tuple(cast(list[str], _resolutions["supported"]))
DEFAULT_RESOLUTIONS: Final[tuple[str, ...]] = tuple(cast(list[str], _resolutions["defaults"]))
RESOLUTION_ALIASES: Final[dict[str, str]] = dict(cast(dict[str, str], _resolutions["aliases"]))
RESOLUTION_TO_SUFFIX: Final[dict[str, str]] = dict(cast(dict[str, str], _resolutions["suffixes"]))

# Legacy compatibility name.
RESOLUTIONS: Final[tuple[str, ...]] = SUPPORTED_RESOLUTIONS

_features = cast(dict[str, object], _PIPELINE_TOML["features"])
FEATURE_CONFIG: Final[FeatureConfigSpec] = {
    "lag_periods": [_as_int("features.lag_periods", value) for value in cast(list[int], _features["lag_periods"])],
    "rolling_periods": [
        _as_int("features.rolling_periods", value)
        for value in cast(list[int], _features["rolling_periods"])
    ],
    "slope_periods": [
        _as_int("features.slope_periods", value)
        for value in cast(list[int], _features["slope_periods"])
    ],
    "lag_minutes": [
        _as_int("features.lag_minutes", value) for value in cast(list[int], _features["lag_minutes"])
    ],
    "rolling_minutes": [
        _as_int("features.rolling_minutes", value)
        for value in cast(list[int], _features["rolling_minutes"])
    ],
    "slope_minutes": [
        _as_int("features.slope_minutes", value)
        for value in cast(list[int], _features["slope_minutes"])
    ],
    "profile_activity_threshold": _as_float(
        "features.profile_activity_threshold",
        _features["profile_activity_threshold"],
    ),
    "fourier_cycles": _as_fourier_cycles(
        "features.fourier_cycles",
        _features.get("fourier_cycles", []),
    ),
}

_day_class = cast(dict[str, object], _PIPELINE_TOML["day_class"])
DAY_CLASS_MAP: Final[dict[str, int]] = {
    key: int(value) for key, value in cast(dict[str, int], _day_class["mapping"]).items()
}
VALID_DAY_CLASSES: Final[set[str]] = set(cast(list[str], _day_class["valid_classes"]))
EXPECTED_DAY_CLASSES: Final[set[str]] = set(VALID_DAY_CLASSES)

FOURIER_COLUMNS: Final[list[str]] = [
    column_name
    for cycle in FEATURE_CONFIG["fourier_cycles"]
    for column_name in (f"{cycle['prefix']}_sin", f"{cycle['prefix']}_cos")
]

BASE_SILVER_COLUMNS: Final[list[str]] = [
    "timestamp",
    "avg_load",
    "day_class",
    "workday",
    "year",
    "quarter",
    "month",
    "day",
    "day_of_week",
    "hour",
    "season",
    "time_of_day",
] + FOURIER_COLUMNS

PHASE_CONTEXT_COLUMNS: Final[list[str]] = [
    "phase_minute_15m",
    "phase_progress_15m",
    "phase_boundary_dist_15m",
    "phase_boundary_flag_15m",
    "phase_sin_15m",
    "phase_cos_15m",
]


def _build_silver_columns() -> list[str]:
    """Build ordered silver/gold schema columns.

    Column ordering convention:
    1) base columns (core, business, temporal),
    2) lag columns,
    3) rolling statistic columns,
    4) delta columns,
    5) slope columns.
    """
    columns = list(BASE_SILVER_COLUMNS)
    columns.extend(PHASE_CONTEXT_COLUMNS)

    for lag in FEATURE_CONFIG["lag_periods"]:
        columns.append(f"lag_{lag}")

    for window in FEATURE_CONFIG["rolling_periods"]:
        columns.extend(
            [
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_max_{window}",
                f"rolling_min_{window}",
            ]
        )

    for lag in FEATURE_CONFIG["lag_periods"]:
        if lag == 1:
            continue
        columns.append(f"delta_{lag}")

    for window in FEATURE_CONFIG["slope_periods"]:
        columns.append(f"slope_{window}")

    for lag in FEATURE_CONFIG["lag_minutes"]:
        columns.append(f"lag_min_{lag}")

    for window in FEATURE_CONFIG["rolling_minutes"]:
        columns.extend(
            [
                f"rolling_mean_min_{window}",
                f"rolling_std_min_{window}",
                f"rolling_max_min_{window}",
                f"rolling_min_min_{window}",
            ]
        )

    for window in FEATURE_CONFIG["slope_minutes"]:
        columns.append(f"slope_min_{window}")

    columns.extend(
        [
            "previous_day_load",
            "avg_workday_baseline",
            "anchored_workday_baseline",
            "profile_residual_lag_1",
            "previous_day_residual",
            "prev_day_workday",
            "next_day_workday",
            "workday_transition",
            "profile_activity_ratio",
            "profile_active_flag",
        ]
    )

    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"Duplicate silver columns generated: {duplicates}")

    return columns


SILVER_COLUMNS: Final[list[str]] = _build_silver_columns()
NON_LAG_SILVER_NOT_NULL_COLUMNS: Final[list[str]] = [
    "timestamp",
    "day_class",
    "workday",
    "year",
    "quarter",
    "month",
    "day",
    "day_of_week",
    "hour",
    "season",
    "time_of_day",
    "prev_day_workday",
    "next_day_workday",
    "workday_transition",
] + FOURIER_COLUMNS
NON_LAG_SILVER_NOT_NULL_COLUMNS.extend(PHASE_CONTEXT_COLUMNS)

SCHEMAS: Final[dict[str, SchemaSpec]] = {
    "bronze": {
        "columns": ["timestamp", "day_class", "load"],
        "allowed_day_class": sorted(VALID_DAY_CLASSES),
        "required_not_null": [],
    },
    "silver": {
        "columns": SILVER_COLUMNS,
        "allowed_day_class": sorted(VALID_DAY_CLASSES),
        "required_not_null": NON_LAG_SILVER_NOT_NULL_COLUMNS,
    },
    "gold": {
        "columns": SILVER_COLUMNS,
        "allowed_day_class": sorted(VALID_DAY_CLASSES),
        "required_not_null": NON_LAG_SILVER_NOT_NULL_COLUMNS + ["avg_load"],
    },
}

_splits = cast(dict[str, list[int]], _PIPELINE_TOML["splits"])
SPLIT_DAY_RANGES: Final[dict[str, tuple[int, int]]] = {
    split: (int(bounds[0]), int(bounds[1]))
    for split, bounds in _splits.items()
}

TARGET_COLUMN: Final[str] = cast(dict[str, str], _PIPELINE_TOML["target"])["column"]
FULL_STABLE_FEATURE_SET_NAME: Final[str] = "full_stable"
FULL_STABLE_LEGACY_FEATURE_SET_NAME: Final[str] = "full_stable_legacy"
REGIME_PROFILE_FEATURE_SET_NAME: Final[str] = "regime_profile"
FULL_STABLE_EXCLUDED_COLUMNS: Final[tuple[str, ...]] = (
    "rolling_mean_240",
    "rolling_std_240",
    "rolling_max_240",
    "rolling_min_240",
    "rolling_mean_1440",
    "rolling_std_1440",
    "rolling_max_1440",
    "rolling_min_1440",
)
FULL_STABLE_LEGACY_COLUMNS: Final[tuple[str, ...]] = (
    "workday",
    "year",
    "quarter",
    "month",
    "day",
    "day_of_week",
    "hour",
    "season",
    "time_of_day",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_5",
    "lag_15",
    "lag_60",
    "lag_1440",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_max_5",
    "rolling_min_5",
    "rolling_mean_15",
    "rolling_std_15",
    "rolling_max_15",
    "rolling_min_15",
    "rolling_mean_60",
    "rolling_std_60",
    "rolling_max_60",
    "rolling_min_60",
    "delta_5",
    "delta_15",
    "delta_60",
    "delta_1440",
    "slope_5",
    "slope_15",
    "slope_60",
)
_quality_thresholds = cast(dict[str, object], _PIPELINE_TOML["quality_thresholds"])
RAW_MAX_NAN_PCT: Final[float] = _as_float(
    "quality_thresholds.raw_max_nan_pct",
    _quality_thresholds["raw_max_nan_pct"],
)
RAW_MAX_OUT_OF_RANGE_PCT: Final[float] = _as_float(
    "quality_thresholds.raw_max_out_of_range_pct",
    _quality_thresholds["raw_max_out_of_range_pct"],
)
SILVER_NAN_DROP_WARN_PCT: Final[float] = _as_float(
    "quality_thresholds.silver_nan_drop_warn_pct",
    _quality_thresholds["silver_nan_drop_warn_pct"],
)
SILVER_NAN_DROP_FAIL_PCT: Final[float] = _as_float(
    "quality_thresholds.silver_nan_drop_fail_pct",
    _quality_thresholds["silver_nan_drop_fail_pct"],
)
GOLD_MIN_RETENTION_PCT: Final[float] = _as_float(
    "quality_thresholds.gold_min_retention_pct",
    _quality_thresholds["gold_min_retention_pct"],
)
MODEL_MIN_SPLIT_ROWS: Final[int] = _as_int(
    "quality_thresholds.model_min_split_rows",
    _quality_thresholds["model_min_split_rows"],
)

_feature_sets_toml = cast(dict[str, dict[str, list[str]]], _PIPELINE_TOML["feature_sets"])
FEATURE_SETS: Final[dict[str, list[str]]] = {
    name: list(payload["columns"]) for name, payload in _feature_sets_toml.items()
}
FEATURE_SETS["full"] = [
    col for col in SILVER_COLUMNS if col not in {"timestamp", "day_class", TARGET_COLUMN}
]
FEATURE_SETS[FULL_STABLE_FEATURE_SET_NAME] = [
    column
    for column in FEATURE_SETS["full"]
    if column not in set(FULL_STABLE_EXCLUDED_COLUMNS)
]
FEATURE_SETS[FULL_STABLE_LEGACY_FEATURE_SET_NAME] = list(FULL_STABLE_LEGACY_COLUMNS)

_visualization = cast(dict[str, object], _EDA_TOML["visualization"])
_analysis = cast(dict[str, object], _EDA_TOML["analysis"])
_physical_range = cast(dict[str, object], _analysis["physical_range"])
_distribution_features = cast(dict[str, object], _analysis["distribution_features"])

EDA_CONFIG: Final[EDAConfig] = {
    "zscore_threshold": _as_float("analysis.zscore_threshold", _analysis["zscore_threshold"]),
    "histogram_bins": _as_int("analysis.histogram_bins", _analysis["histogram_bins"]),
    "figure_size": _as_int_pair("visualization.figure_size", _visualization["figure_size"]),
    "figure_size_wide": _as_int_pair(
        "visualization.figure_size_wide", _visualization["figure_size_wide"]
    ),
    "figure_size_compact": _as_int_pair(
        "visualization.figure_size_compact", _visualization["figure_size_compact"]
    ),
    "figure_size_grid": _as_int_pair(
        "visualization.figure_size_grid", _visualization["figure_size_grid"]
    ),
    "figure_size_correlation": _as_int_pair(
        "visualization.figure_size_correlation", _visualization["figure_size_correlation"]
    ),
    "day_class_colors": _as_str_dict(
        "visualization.day_class_colors", _visualization["day_class_colors"]
    ),
    "seaborn_style": _as_str("visualization.seaborn_style", _visualization["seaborn_style"]),
    "physical_load_max_watts": _as_float(
        "analysis.physical_range.load_max_watts", _physical_range["load_max_watts"]
    ),
    "physical_load_min_watts": _as_float(
        "analysis.physical_range.load_min_watts", _physical_range["load_min_watts"]
    ),
    "correlation_high_threshold": _as_float(
        "analysis.correlation_high_threshold", _analysis["correlation_high_threshold"]
    ),
    "top_correlations_count": _as_int(
        "analysis.top_correlations_count", _analysis["top_correlations_count"]
    ),
    "zero_run_threshold_seconds": _as_int(
        "analysis.zero_run_threshold_seconds", _analysis["zero_run_threshold_seconds"]
    ),
    "overlay_resample_frequency": _as_str(
        "analysis.overlay_resample_frequency", _analysis["overlay_resample_frequency"]
    ),
    "legend_max_labels": _as_int("analysis.legend_max_labels", _analysis["legend_max_labels"]),
    "percentiles": _as_float_list("analysis.percentiles", _analysis["percentiles"]),
    "distribution_features": _as_str_list(
        "analysis.distribution_features.columns", _distribution_features["columns"]
    ),
}

EDA_RESOLUTION_MODES: Final[tuple[EDAResolutionMode, ...]] = ("all", "default", "custom")
_resolution_selection = cast(dict[str, str], _EDA_TOML["resolution_selection"])
EDA_DEFAULT_RESOLUTION_MODE: Final[EDAResolutionMode] = cast(
    EDAResolutionMode, _resolution_selection["default_mode"]
)

_multires = cast(dict[str, object], _MULTIRES_TOML["multires"])
MULTIRES_CONFIG: Final[MultiresConfig] = {
    "enabled": _as_bool("multires.enabled", _multires["enabled"]),
    "mode": _as_str("multires.mode", _multires["mode"]),
    "comparison_mode": _as_str("multires.comparison_mode", _multires["comparison_mode"]),
    "resolutions": _as_str_list("multires.resolutions", _multires["resolutions"]),
    "horizons_minutes": [
        _as_int(f"multires.horizons_minutes[{idx}]", value)
        for idx, value in enumerate(cast(list[object], _multires["horizons_minutes"]))
    ],
    "matched_strategies": _as_str_list(
        "multires.matched_strategies", _multires["matched_strategies"]
    ),
    "native_step_enabled": _as_bool(
        "multires.native_step_enabled", _multires["native_step_enabled"]
    ),
    "write_latest": _as_bool("multires.write_latest", _multires["write_latest"]),
}

_multires_selection = cast(dict[str, object], _multires["selection"])
MULTIRES_SELECTION: Final[MultiresSelectionConfig] = {
    "min_eval_coverage": _as_float(
        "multires.selection.min_eval_coverage",
        _multires_selection["min_eval_coverage"],
    ),
    "max_fold_std_mae_ratio": _as_float(
        "multires.selection.max_fold_std_mae_ratio",
        _multires_selection["max_fold_std_mae_ratio"],
    ),
    "min_practical_mae_gain_pct": _as_float(
        "multires.selection.min_practical_mae_gain_pct",
        _multires_selection["min_practical_mae_gain_pct"],
    ),
    "min_practical_rmse_gain_pct": _as_float(
        "multires.selection.min_practical_rmse_gain_pct",
        _multires_selection["min_practical_rmse_gain_pct"],
    ),
    "max_candidate_runtime_minutes": _as_int(
        "multires.selection.max_candidate_runtime_minutes",
        _multires_selection["max_candidate_runtime_minutes"],
    ),
    "pareto_enabled": _as_bool(
        "multires.selection.pareto_enabled", _multires_selection["pareto_enabled"]
    ),
}

_multires_baselines = cast(dict[str, object], _multires["baselines"])
MULTIRES_BASELINES: Final[MultiresBaselineConfig] = {
    "include_persistence": _as_bool(
        "multires.baselines.include_persistence",
        _multires_baselines["include_persistence"],
    ),
    "include_previous_day": _as_bool(
        "multires.baselines.include_previous_day",
        _multires_baselines["include_previous_day"],
    ),
    "include_avg_workday": _as_bool(
        "multires.baselines.include_avg_workday",
        _multires_baselines["include_avg_workday"],
    ),
    "include_anchored_workday": _as_bool(
        "multires.baselines.include_anchored_workday",
        _multires_baselines["include_anchored_workday"],
    ),
    "include_hybrid_workday": _as_bool(
        "multires.baselines.include_hybrid_workday",
        _multires_baselines["include_hybrid_workday"],
    ),
}

_multires_hybrid = cast(dict[str, object], _multires["hybrid"])
MULTIRES_HYBRID: Final[MultiresHybridConfig] = {
    "persistence_weight_start": _as_float(
        "multires.hybrid.persistence_weight_start",
        _multires_hybrid["persistence_weight_start"],
    ),
    "persistence_weight_end": _as_float(
        "multires.hybrid.persistence_weight_end",
        _multires_hybrid["persistence_weight_end"],
    ),
    "curve": _as_str("multires.hybrid.curve", _multires_hybrid["curve"]),
}


def _default_multires_profile_origins(name: str) -> int:
    """Resolve legacy per-profile origin defaults from the runtime section."""
    name_key = name.lower().strip()
    if name_key == "smoke":
        return int(_multires["runtime"]["smoke_origins_per_fold"])
    if name_key == "candidate":
        return int(_multires["runtime"]["candidate_origins_per_fold"])
    if name_key == "full":
        return int(_multires["runtime"]["full_origins_per_fold"])
    return int(_multires["runtime"]["candidate_origins_per_fold"])

_multires_profiles = cast(dict[str, object], _multires["profiles"])
MULTIRES_PROFILES: Final[dict[str, MultiresModeProfile]] = {
    name: {
        "n_folds": _as_int(f"multires.profiles.{name}.n_folds", cast(dict[str, object], payload)["n_folds"]),
        "val_window_days": _as_int(
            f"multires.profiles.{name}.val_window_days",
            cast(dict[str, object], payload)["val_window_days"],
        ),
        "origins_per_fold": _as_int(
            f"multires.profiles.{name}.origins_per_fold",
            cast(dict[str, object], payload).get("origins_per_fold", _default_multires_profile_origins(name)),
        ),
        "resolutions": _as_str_list(
            f"multires.profiles.{name}.resolutions",
            cast(dict[str, object], payload)["resolutions"],
        ),
        "horizons_minutes": [
            _as_int(f"multires.profiles.{name}.horizons_minutes[{idx}]", value)
            for idx, value in enumerate(
                cast(list[object], cast(dict[str, object], payload)["horizons_minutes"])
            )
        ],
        "feature_sets": _as_str_list(
            f"multires.profiles.{name}.feature_sets",
            cast(dict[str, object], payload)["feature_sets"],
        ),
        "model_labels": _as_str_list(
            f"multires.profiles.{name}.model_labels",
            cast(dict[str, object], payload)["model_labels"],
        ),
    }
    for name, payload in _multires_profiles.items()
}

_multires_runtime = cast(dict[str, object], _multires["runtime"])
MULTIRES_RUNTIME: Final[MultiresRuntimeConfig] = {
    "enable_cache": _as_bool(
        "multires.runtime.enable_cache", _multires_runtime["enable_cache"]
    ),
    "cache_mode": _as_str("multires.runtime.cache_mode", _multires_runtime["cache_mode"]),
    "full_runtime_warning_minutes": _as_int(
        "multires.runtime.full_runtime_warning_minutes",
        _multires_runtime["full_runtime_warning_minutes"],
    ),
    "smoke_origins_per_fold": _as_int(
        "multires.runtime.smoke_origins_per_fold",
        _multires_runtime["smoke_origins_per_fold"],
    ),
    "candidate_origins_per_fold": _as_int(
        "multires.runtime.candidate_origins_per_fold",
        _multires_runtime["candidate_origins_per_fold"],
    ),
    "full_origins_per_fold": _as_int(
        "multires.runtime.full_origins_per_fold",
        _multires_runtime["full_origins_per_fold"],
    ),
}

_multires_rollout = cast(dict[str, object], _multires["rollout"])
MULTIRES_ROLLOUT: Final[MultiresRolloutConfig] = {
    "enabled": _as_bool("multires.rollout.enabled", _multires_rollout["enabled"]),
    "selected_resolution": _as_str(
        "multires.rollout.selected_resolution",
        _multires_rollout["selected_resolution"],
    ),
    "feature_set": _as_str("multires.rollout.feature_set", _multires_rollout["feature_set"]),
    "model_label": _as_str("multires.rollout.model_label", _multires_rollout["model_label"]),
    "horizon_minutes": _as_int(
        "multires.rollout.horizon_minutes", _multires_rollout["horizon_minutes"]
    ),
    "strategy": _as_str("multires.rollout.strategy", _multires_rollout["strategy"]),
    "origins_per_run": _as_int(
        "multires.rollout.origins_per_run", _multires_rollout["origins_per_run"]
    ),
    "origin_policy": _as_str(
        "multires.rollout.origin_policy", _multires_rollout["origin_policy"]
    ),
    "selection_target": _as_str(
        "multires.rollout.selection_target", _multires_rollout["selection_target"]
    ),
}

_multires_rollout_challengers = cast(dict[str, object], _multires["rollout_challengers"])
MULTIRES_ROLLOUT_CHALLENGERS: Final[MultiresRolloutChallengerConfig] = {
    "enabled": _as_bool(
        "multires.rollout_challengers.enabled",
        _multires_rollout_challengers["enabled"],
    ),
    "max_candidates": _as_int(
        "multires.rollout_challengers.max_candidates",
        _multires_rollout_challengers["max_candidates"],
    ),
    "parallel_workers": _as_int(
        "multires.rollout_challengers.parallel_workers",
        _multires_rollout_challengers["parallel_workers"],
    ),
    "include_rollout_registry": _as_bool(
        "multires.rollout_challengers.include_rollout_registry",
        _multires_rollout_challengers["include_rollout_registry"],
    ),
    "include_stage6_registry": _as_bool(
        "multires.rollout_challengers.include_stage6_registry",
        _multires_rollout_challengers["include_stage6_registry"],
    ),
    "include_horizon_policy_candidates": _as_bool(
        "multires.rollout_challengers.include_horizon_policy_candidates",
        _multires_rollout_challengers["include_horizon_policy_candidates"],
    ),
    "include_config_default": _as_bool(
        "multires.rollout_challengers.include_config_default",
        _multires_rollout_challengers["include_config_default"],
    ),
    "origin_policies": _as_str_list(
        "multires.rollout_challengers.origin_policies",
        _multires_rollout_challengers["origin_policies"],
    ),
    "policy_resolutions": _as_str_list(
        "multires.rollout_challengers.policy_resolutions",
        _multires_rollout_challengers["policy_resolutions"],
    ),
    "recommendation_origin_scope": _as_str(
        "multires.rollout_challengers.recommendation_origin_scope",
        _multires_rollout_challengers["recommendation_origin_scope"],
    ),
}

_multires_rollout_learned_blends = cast(dict[str, object], _multires["rollout_learned_blends"])
_multires_hybrid_phase_bucket_weights = cast(
    dict[str, object],
    _multires_rollout_learned_blends.get("hybrid_phase_gate_bucket_weights", {}),
)
MULTIRES_ROLLOUT_LEARNED_BLENDS: Final[MultiresRolloutLearnedBlendConfig] = {
    "enabled": _as_bool(
        "multires.rollout_learned_blends.enabled",
        _multires_rollout_learned_blends["enabled"],
    ),
    "include_persistence_to_raw": _as_bool(
        "multires.rollout_learned_blends.include_persistence_to_raw",
        _multires_rollout_learned_blends["include_persistence_to_raw"],
    ),
    "include_persistence_to_residual": _as_bool(
        "multires.rollout_learned_blends.include_persistence_to_residual",
        _multires_rollout_learned_blends["include_persistence_to_residual"],
    ),
    "include_raw_to_residual": _as_bool(
        "multires.rollout_learned_blends.include_raw_to_residual",
        _multires_rollout_learned_blends["include_raw_to_residual"],
    ),
    "include_hybrid_phase_gate": _as_bool(
        "multires.rollout_learned_blends.include_hybrid_phase_gate",
        _multires_rollout_learned_blends["include_hybrid_phase_gate"],
    ),
    "persistence_weight_start": _as_float(
        "multires.rollout_learned_blends.persistence_weight_start",
        _multires_rollout_learned_blends["persistence_weight_start"],
    ),
    "persistence_weight_end": _as_float(
        "multires.rollout_learned_blends.persistence_weight_end",
        _multires_rollout_learned_blends["persistence_weight_end"],
    ),
    "raw_weight_start": _as_float(
        "multires.rollout_learned_blends.raw_weight_start",
        _multires_rollout_learned_blends["raw_weight_start"],
    ),
    "raw_weight_end": _as_float(
        "multires.rollout_learned_blends.raw_weight_end",
        _multires_rollout_learned_blends["raw_weight_end"],
    ),
    "hybrid_phase_gate_aligned_weight": _as_float(
        "multires.rollout_learned_blends.hybrid_phase_gate_aligned_weight",
        _multires_rollout_learned_blends["hybrid_phase_gate_aligned_weight"],
    ),
    "hybrid_phase_gate_non_aligned_weight": _as_float(
        "multires.rollout_learned_blends.hybrid_phase_gate_non_aligned_weight",
        _multires_rollout_learned_blends["hybrid_phase_gate_non_aligned_weight"],
    ),
    "hybrid_phase_gate_bucket_weights": {
        int(_as_str(
            f"multires.rollout_learned_blends.hybrid_phase_gate_bucket_weights.{key}",
            key,
        )): _as_float(
            f"multires.rollout_learned_blends.hybrid_phase_gate_bucket_weights.{key}",
            value,
        )
        for key, value in _multires_hybrid_phase_bucket_weights.items()
    },
    "refinement_enabled": _as_bool(
        "multires.rollout_learned_blends.refinement_enabled",
        _multires_rollout_learned_blends["refinement_enabled"],
    ),
    "refinement_step": _as_float(
        "multires.rollout_learned_blends.refinement_step",
        _multires_rollout_learned_blends["refinement_step"],
    ),
    "refinement_neighbors": _as_int(
        "multires.rollout_learned_blends.refinement_neighbors",
        _multires_rollout_learned_blends["refinement_neighbors"],
    ),
    "max_weights_per_family": _as_int(
        "multires.rollout_learned_blends.max_weights_per_family",
        _multires_rollout_learned_blends["max_weights_per_family"],
    ),
    "curve": _as_str(
        "multires.rollout_learned_blends.curve",
        _multires_rollout_learned_blends["curve"],
    ),
}

_multires_rollout_policy_candidates = cast(
    dict[str, object],
    _multires.get("rollout_policy_candidates", {}),
)
MULTIRES_ROLLOUT_POLICY_CANDIDATES: Final[MultiresRolloutPolicyCandidateConfig] = {
    "enabled": _as_bool(
        "multires.rollout_policy_candidates.enabled",
        _multires_rollout_policy_candidates.get("enabled", True),
    ),
    "max_horizon_minutes": _as_int(
        "multires.rollout_policy_candidates.max_horizon_minutes",
        _multires_rollout_policy_candidates.get("max_horizon_minutes", 15),
    ),
    "selection_targets": _as_str_list(
        "multires.rollout_policy_candidates.selection_targets",
        _multires_rollout_policy_candidates.get(
            "selection_targets",
            ["endpoint_mae", "path_mae", "phase_mean_mae"],
        ),
    ),
}

_multires_rollout_sweep_policies = cast(
    dict[str, object],
    _multires.get("rollout_sweep_policies", {}),
)
MULTIRES_ROLLOUT_SWEEP_POLICIES: Final[MultiresRolloutSweepPolicyConfig] = {
    "enabled": _as_bool(
        "multires.rollout_sweep_policies.enabled",
        _multires_rollout_sweep_policies.get("enabled", True),
    ),
    "min_horizon_minutes": _as_int(
        "multires.rollout_sweep_policies.min_horizon_minutes",
        _multires_rollout_sweep_policies.get("min_horizon_minutes", 30),
    ),
    "max_horizon_minutes": _as_int(
        "multires.rollout_sweep_policies.max_horizon_minutes",
        _multires_rollout_sweep_policies.get("max_horizon_minutes", 120),
    ),
    "origin_policies": _as_str_list(
        "multires.rollout_sweep_policies.origin_policies",
        _multires_rollout_sweep_policies.get("origin_policies", ["phase_balanced"]),
    ),
    "selection_targets": _as_str_list(
        "multires.rollout_sweep_policies.selection_targets",
        _multires_rollout_sweep_policies.get(
            "selection_targets",
            ["next_lock_mae", "path_mae", "profile_shape_mae"],
        ),
    ),
    "min_source_candidates": _as_int(
        "multires.rollout_sweep_policies.min_source_candidates",
        _multires_rollout_sweep_policies.get("min_source_candidates", 2),
    ),
    "max_source_candidates": _as_int(
        "multires.rollout_sweep_policies.max_source_candidates",
        _multires_rollout_sweep_policies.get("max_source_candidates", 3),
    ),
}

_multires_horizon_curve = cast(dict[str, object], _multires["horizon_curve"])
MULTIRES_HORIZON_CURVE: Final[MultiresHorizonCurveConfig] = {
    "enabled": _as_bool("multires.horizon_curve.enabled", _multires_horizon_curve["enabled"]),
    "horizons_minutes": [
        _as_int(f"multires.horizon_curve.horizons_minutes[{idx}]", value)
        for idx, value in enumerate(cast(list[object], _multires_horizon_curve["horizons_minutes"]))
    ],
    "origins_per_run": _as_int(
        "multires.horizon_curve.origins_per_run",
        _multires_horizon_curve["origins_per_run"],
    ),
    "origin_policy": _as_str(
        "multires.horizon_curve.origin_policy",
        _multires_horizon_curve["origin_policy"],
    ),
    "selection_target": _as_str(
        "multires.horizon_curve.selection_target",
        _multires_horizon_curve["selection_target"],
    ),
    "max_candidates": _as_int(
        "multires.horizon_curve.max_candidates",
        _multires_horizon_curve["max_candidates"],
    ),
    "include_stage5_anchor": _as_bool(
        "multires.horizon_curve.include_stage5_anchor",
        _multires_horizon_curve["include_stage5_anchor"],
    ),
    "reuse_existing_sweeps": _as_bool(
        "multires.horizon_curve.reuse_existing_sweeps",
        _multires_horizon_curve["reuse_existing_sweeps"],
    ),
    "write_latest": _as_bool(
        "multires.horizon_curve.write_latest",
        _multires_horizon_curve["write_latest"],
    ),
}

_multires_forecast_control = cast(dict[str, object], _multires.get("forecast_control", {}))
_optimizer_delivery_confidence_layer_defaults: Final[dict[str, float]] = {
    "nowcast": 1.0,
    "phase": 0.9,
    "hourly": 0.8,
    "day_ahead": 0.7,
}
_optimizer_delivery_confidence_quantile_source_defaults: Final[dict[str, float]] = {
    "lead_interval": 1.0,
    "predicted_peak_lead_interval": 0.98,
    "predicted_peak_global": 0.95,
    "next_lock_global": 0.92,
    "layer_global_fallback": 0.85,
    "layer_global": 0.85,
    "unavailable": 0.35,
}
MULTIRES_FORECAST_CONTROL: Final[ForecastControlConfig] = {
    "enabled": _as_bool(
        "multires.forecast_control.enabled",
        _multires_forecast_control.get("enabled", True),
    ),
    "day_ahead_horizon_minutes": _as_int(
        "multires.forecast_control.day_ahead_horizon_minutes",
        _multires_forecast_control.get("day_ahead_horizon_minutes", 1440),
    ),
    "hourly_horizon_minutes": _as_int(
        "multires.forecast_control.hourly_horizon_minutes",
        _multires_forecast_control.get("hourly_horizon_minutes", 60),
    ),
    "phase_horizon_minutes": _as_int(
        "multires.forecast_control.phase_horizon_minutes",
        _multires_forecast_control.get("phase_horizon_minutes", 15),
    ),
    "nowcast_horizon_minutes": _as_int(
        "multires.forecast_control.nowcast_horizon_minutes",
        _multires_forecast_control.get("nowcast_horizon_minutes", 1),
    ),
    "day_ahead_refresh_enabled": _as_bool(
        "multires.forecast_control.day_ahead_refresh_enabled",
        _multires_forecast_control.get("day_ahead_refresh_enabled", True),
    ),
    "day_ahead_refresh_interval_minutes": _as_int(
        "multires.forecast_control.day_ahead_refresh_interval_minutes",
        _multires_forecast_control.get("day_ahead_refresh_interval_minutes", 60),
    ),
    "day_ahead_refresh_candidate_label": _as_str(
        "multires.forecast_control.day_ahead_refresh_candidate_label",
        _multires_forecast_control.get("day_ahead_refresh_candidate_label", "auto"),
    ),
    "day_ahead_refresh_lookback_minutes": _as_int(
        "multires.forecast_control.day_ahead_refresh_lookback_minutes",
        _multires_forecast_control.get("day_ahead_refresh_lookback_minutes", 120),
    ),
    "day_ahead_refresh_residual_drift_mae_pct_threshold": _as_float(
        "multires.forecast_control.day_ahead_refresh_residual_drift_mae_pct_threshold",
        _multires_forecast_control.get("day_ahead_refresh_residual_drift_mae_pct_threshold", 12.0),
    ),
    "day_ahead_refresh_transition_mae_pct_threshold": _as_float(
        "multires.forecast_control.day_ahead_refresh_transition_mae_pct_threshold",
        _multires_forecast_control.get("day_ahead_refresh_transition_mae_pct_threshold", 10.0),
    ),
    "day_ahead_refresh_activity_ratio_shift_threshold": _as_float(
        "multires.forecast_control.day_ahead_refresh_activity_ratio_shift_threshold",
        _multires_forecast_control.get("day_ahead_refresh_activity_ratio_shift_threshold", 0.18),
    ),
    "day_ahead_refresh_threshold_quantiles": _as_float_list(
        "multires.forecast_control.day_ahead_refresh_threshold_quantiles",
        _multires_forecast_control.get(
            "day_ahead_refresh_threshold_quantiles",
            [0.5, 0.65, 0.8, 0.9],
        ),
    ),
    "day_ahead_refresh_min_trigger_rate": _as_float(
        "multires.forecast_control.day_ahead_refresh_min_trigger_rate",
        _multires_forecast_control.get("day_ahead_refresh_min_trigger_rate", 0.10),
    ),
    "day_ahead_refresh_max_trigger_rate": _as_float(
        "multires.forecast_control.day_ahead_refresh_max_trigger_rate",
        _multires_forecast_control.get("day_ahead_refresh_max_trigger_rate", 0.75),
    ),
    "day_ahead_refresh_min_profile_gain_fraction_vs_unconditional": _as_float(
        "multires.forecast_control.day_ahead_refresh_min_profile_gain_fraction_vs_unconditional",
        _multires_forecast_control.get(
            "day_ahead_refresh_min_profile_gain_fraction_vs_unconditional",
            0.60,
        ),
    ),
    "day_ahead_refresh_min_lock_gain_fraction_vs_unconditional": _as_float(
        "multires.forecast_control.day_ahead_refresh_min_lock_gain_fraction_vs_unconditional",
        _multires_forecast_control.get(
            "day_ahead_refresh_min_lock_gain_fraction_vs_unconditional",
            0.50,
        ),
    ),
    "day_ahead_refresh_trigger_mode": _as_str(
        "multires.forecast_control.day_ahead_refresh_trigger_mode",
        _multires_forecast_control.get("day_ahead_refresh_trigger_mode", "any"),
    ),
    "day_ahead_refresh_candidate_trigger_modes": _as_str_list(
        "multires.forecast_control.day_ahead_refresh_candidate_trigger_modes",
        _multires_forecast_control.get(
            "day_ahead_refresh_candidate_trigger_modes",
            [
                "any",
                "residual_or_activity",
                "residual_or_activity_active_band",
                "residual_or_activity_active_or_transition",
                "residual_only",
                "activity_only",
                "activity_active_band",
                "residual_and_activity",
                "two_of_three",
            ],
        ),
    ),
    "actual_resolution": _as_str(
        "multires.forecast_control.actual_resolution",
        _multires_forecast_control.get("actual_resolution", "1min"),
    ),
    "lock_interval_minutes": _as_int(
        "multires.forecast_control.lock_interval_minutes",
        _multires_forecast_control.get("lock_interval_minutes", 15),
    ),
    "cycle_origin_hour": _as_int(
        "multires.forecast_control.cycle_origin_hour",
        _multires_forecast_control.get("cycle_origin_hour", 0),
    ),
    "cycle_origin_minute": _as_int(
        "multires.forecast_control.cycle_origin_minute",
        _multires_forecast_control.get("cycle_origin_minute", 0),
    ),
    "cycle_origin_stride_minutes": _as_int(
        "multires.forecast_control.cycle_origin_stride_minutes",
        _multires_forecast_control.get("cycle_origin_stride_minutes", 360),
    ),
    "rolling_benchmark_enabled": _as_bool(
        "multires.forecast_control.rolling_benchmark_enabled",
        _multires_forecast_control.get("rolling_benchmark_enabled", True),
    ),
    "rolling_benchmark_origin_stride_minutes": _as_int(
        "multires.forecast_control.rolling_benchmark_origin_stride_minutes",
        _multires_forecast_control.get("rolling_benchmark_origin_stride_minutes", 180),
    ),
    "rolling_benchmark_max_cycles": _as_int(
        "multires.forecast_control.rolling_benchmark_max_cycles",
        _multires_forecast_control.get("rolling_benchmark_max_cycles", 0),
    ),
    "rolling_benchmark_bootstrap_samples": _as_int(
        "multires.forecast_control.rolling_benchmark_bootstrap_samples",
        _multires_forecast_control.get("rolling_benchmark_bootstrap_samples", 400),
    ),
    "rolling_benchmark_confidence_level": _as_float(
        "multires.forecast_control.rolling_benchmark_confidence_level",
        _multires_forecast_control.get("rolling_benchmark_confidence_level", 0.95),
    ),
    "calibration_splits": _as_str_list(
        "multires.forecast_control.calibration_splits",
        _multires_forecast_control.get("calibration_splits", ["validate"]),
    ),
    "evaluation_splits": _as_str_list(
        "multires.forecast_control.evaluation_splits",
        _multires_forecast_control.get("evaluation_splits", ["test"]),
    ),
    "max_cycles": _as_int(
        "multires.forecast_control.max_cycles",
        _multires_forecast_control.get("max_cycles", 8),
    ),
    "optimize_replayed_candidates": _as_bool(
        "multires.forecast_control.optimize_replayed_candidates",
        _multires_forecast_control.get("optimize_replayed_candidates", True),
    ),
    "control_promotion_scope": _as_str(
        "multires.forecast_control.control_promotion_scope",
        _multires_forecast_control.get("control_promotion_scope", "held_out_evaluation"),
    ),
    "control_promotion_guard_enabled": _as_bool(
        "multires.forecast_control.control_promotion_guard_enabled",
        _multires_forecast_control.get("control_promotion_guard_enabled", True),
    ),
    "control_promotion_guard_max_next_lock_regress_pct": _as_float(
        "multires.forecast_control.control_promotion_guard_max_next_lock_regress_pct",
        _multires_forecast_control.get("control_promotion_guard_max_next_lock_regress_pct", 0.0),
    ),
    "control_promotion_guard_max_peak_value_regress_pct": _as_float(
        "multires.forecast_control.control_promotion_guard_max_peak_value_regress_pct",
        _multires_forecast_control.get("control_promotion_guard_max_peak_value_regress_pct", 0.0),
    ),
    "control_promotion_guard_max_peak_miss_regress": _as_float(
        "multires.forecast_control.control_promotion_guard_max_peak_miss_regress",
        _multires_forecast_control.get("control_promotion_guard_max_peak_miss_regress", 0.0),
    ),
    "allow_baseline_candidates": _as_bool(
        "multires.forecast_control.allow_baseline_candidates",
        _multires_forecast_control.get("allow_baseline_candidates", True),
    ),
    "candidate_pool_size": _as_int(
        "multires.forecast_control.candidate_pool_size",
        _multires_forecast_control.get("candidate_pool_size", 4),
    ),
    "candidate_benchmark_origin_cap": _as_int(
        "multires.forecast_control.candidate_benchmark_origin_cap",
        _multires_forecast_control.get("candidate_benchmark_origin_cap", 8),
    ),
    "phase_candidate_benchmark_origin_cap": _as_int(
        "multires.forecast_control.phase_candidate_benchmark_origin_cap",
        _multires_forecast_control.get("phase_candidate_benchmark_origin_cap", 96),
    ),
    "phase_candidate_evaluation_origin_cap": _as_int(
        "multires.forecast_control.phase_candidate_evaluation_origin_cap",
        _multires_forecast_control.get("phase_candidate_evaluation_origin_cap", 96),
    ),
    "phase_control_candidate_pool_size": _as_int(
        "multires.forecast_control.phase_control_candidate_pool_size",
        _multires_forecast_control.get("phase_control_candidate_pool_size", 4),
    ),
    "phase_control_prior_run_limit": _as_int(
        "multires.forecast_control.phase_control_prior_run_limit",
        _multires_forecast_control.get("phase_control_prior_run_limit", 6),
    ),
    "phase_control_min_prior_support_runs": _as_int(
        "multires.forecast_control.phase_control_min_prior_support_runs",
        _multires_forecast_control.get("phase_control_min_prior_support_runs", 2),
    ),
    "phase_control_max_supplemental_contexts_per_resolution": _as_int(
        "multires.forecast_control.phase_control_max_supplemental_contexts_per_resolution",
        _multires_forecast_control.get("phase_control_max_supplemental_contexts_per_resolution", 1),
    ),
    "phase_control_exploration_slots": _as_int(
        "multires.forecast_control.phase_control_exploration_slots",
        _multires_forecast_control.get("phase_control_exploration_slots", 1),
    ),
    "phase_control_origin_cap": _as_int(
        "multires.forecast_control.phase_control_origin_cap",
        _multires_forecast_control.get("phase_control_origin_cap", 96),
    ),
    "phase_stack_native_learned_top_candidates_per_pool": _as_int(
        "multires.forecast_control.phase_stack_native_learned_top_candidates_per_pool",
        _multires_forecast_control.get("phase_stack_native_learned_top_candidates_per_pool", 4),
    ),
    "phase_stack_native_baseline_top_candidates_per_pool": _as_int(
        "multires.forecast_control.phase_stack_native_baseline_top_candidates_per_pool",
        _multires_forecast_control.get("phase_stack_native_baseline_top_candidates_per_pool", 1),
    ),
    "phase_stack_blend_weights": _as_float_list(
        "multires.forecast_control.phase_stack_blend_weights",
        _multires_forecast_control.get(
            "phase_stack_blend_weights",
            [0.15, 0.25, 0.35, 0.50, 0.65, 0.80],
        ),
    ),
    "phase_stack_blend_parent_top_candidates": _as_int(
        "multires.forecast_control.phase_stack_blend_parent_top_candidates",
        _multires_forecast_control.get("phase_stack_blend_parent_top_candidates", 12),
    ),
    "phase_stack_bucket_policy_enabled": _as_bool(
        "multires.forecast_control.phase_stack_bucket_policy_enabled",
        _multires_forecast_control.get("phase_stack_bucket_policy_enabled", True),
    ),
    "phase_stack_bucket_granularity_minutes": _as_int(
        "multires.forecast_control.phase_stack_bucket_granularity_minutes",
        _multires_forecast_control.get("phase_stack_bucket_granularity_minutes", 15),
    ),
    "phase_stack_baseline_control_blend_enabled": _as_bool(
        "multires.forecast_control.phase_stack_baseline_control_blend_enabled",
        _multires_forecast_control.get("phase_stack_baseline_control_blend_enabled", True),
    ),
    "phase_stack_baseline_control_top_candidates": _as_int(
        "multires.forecast_control.phase_stack_baseline_control_top_candidates",
        _multires_forecast_control.get("phase_stack_baseline_control_top_candidates", 4),
    ),
    "phase_stack_baseline_control_blend_weights": _as_float_list(
        "multires.forecast_control.phase_stack_baseline_control_blend_weights",
        _multires_forecast_control.get(
            "phase_stack_baseline_control_blend_weights",
            [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80],
        ),
    ),
    "phase_stack_baseline_control_bucket_blend_enabled": _as_bool(
        "multires.forecast_control.phase_stack_baseline_control_bucket_blend_enabled",
        _multires_forecast_control.get("phase_stack_baseline_control_bucket_blend_enabled", True),
    ),
    "phase_stack_baseline_control_bucket_size_minutes": _as_int(
        "multires.forecast_control.phase_stack_baseline_control_bucket_size_minutes",
        _multires_forecast_control.get("phase_stack_baseline_control_bucket_size_minutes", 5),
    ),
    "phase_stack_guard_enabled": _as_bool(
        "multires.forecast_control.phase_stack_guard_enabled",
        _multires_forecast_control.get("phase_stack_guard_enabled", True),
    ),
    "phase_stack_guard_min_lock_gain_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_min_lock_gain_pct",
        _multires_forecast_control.get("phase_stack_guard_min_lock_gain_pct", 0.005),
    ),
    "phase_stack_guard_require_rolling_support": _as_bool(
        "multires.forecast_control.phase_stack_guard_require_rolling_support",
        _multires_forecast_control.get("phase_stack_guard_require_rolling_support", True),
    ),
    "phase_stack_guard_rolling_scope": _as_str(
        "multires.forecast_control.phase_stack_guard_rolling_scope",
        _multires_forecast_control.get("phase_stack_guard_rolling_scope", "rolling_evaluation"),
    ),
    "phase_stack_guard_min_rolling_lock_gain_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_min_rolling_lock_gain_pct",
        _multires_forecast_control.get("phase_stack_guard_min_rolling_lock_gain_pct", 0.001),
    ),
    "phase_stack_guard_max_rolling_next_lock_regress_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_rolling_next_lock_regress_pct",
        _multires_forecast_control.get("phase_stack_guard_max_rolling_next_lock_regress_pct", 0.0),
    ),
    "phase_stack_guard_max_rolling_profile_degrade_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_rolling_profile_degrade_pct",
        _multires_forecast_control.get("phase_stack_guard_max_rolling_profile_degrade_pct", 0.0),
    ),
    "phase_stack_guard_max_rolling_peak_value_regress_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_rolling_peak_value_regress_pct",
        _multires_forecast_control.get("phase_stack_guard_max_rolling_peak_value_regress_pct", 0.0),
    ),
    "phase_stack_guard_min_rolling_peak_hit_gain": _as_float(
        "multires.forecast_control.phase_stack_guard_min_rolling_peak_hit_gain",
        _multires_forecast_control.get("phase_stack_guard_min_rolling_peak_hit_gain", 0.0),
    ),
    "phase_stack_guard_max_rolling_optimizer_regress_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_rolling_optimizer_regress_pct",
        _multires_forecast_control.get("phase_stack_guard_max_rolling_optimizer_regress_pct", 0.0),
    ),
    "phase_stack_guard_max_next_lock_regress_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_next_lock_regress_pct",
        _multires_forecast_control.get("phase_stack_guard_max_next_lock_regress_pct", 0.0),
    ),
    "phase_stack_guard_max_profile_degrade_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_profile_degrade_pct",
        _multires_forecast_control.get("phase_stack_guard_max_profile_degrade_pct", 0.002),
    ),
    "phase_stack_guard_max_peak_value_regress_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_peak_value_regress_pct",
        _multires_forecast_control.get("phase_stack_guard_max_peak_value_regress_pct", 0.0),
    ),
    "phase_stack_guard_min_peak_hit_gain": _as_float(
        "multires.forecast_control.phase_stack_guard_min_peak_hit_gain",
        _multires_forecast_control.get("phase_stack_guard_min_peak_hit_gain", 0.01),
    ),
    "phase_stack_guard_max_optimizer_regress_pct": _as_float(
        "multires.forecast_control.phase_stack_guard_max_optimizer_regress_pct",
        _multires_forecast_control.get("phase_stack_guard_max_optimizer_regress_pct", 0.05),
    ),
    "benchmark_expanded_candidate_pool_size": _as_int(
        "multires.forecast_control.benchmark_expanded_candidate_pool_size",
        _multires_forecast_control.get("benchmark_expanded_candidate_pool_size", 6),
    ),
    "benchmark_expanded_pool_layers": _as_str_list(
        "multires.forecast_control.benchmark_expanded_pool_layers",
        _multires_forecast_control.get("benchmark_expanded_pool_layers", ["hourly", "phase"]),
    ),
    "benchmark_full_origin_layers": _as_str_list(
        "multires.forecast_control.benchmark_full_origin_layers",
        _multires_forecast_control.get("benchmark_full_origin_layers", ["hourly", "phase"]),
    ),
    "nowcast_candidate_pool_size": _as_int(
        "multires.forecast_control.nowcast_candidate_pool_size",
        _multires_forecast_control.get("nowcast_candidate_pool_size", 6),
    ),
    "nowcast_control_blend_enabled": _as_bool(
        "multires.forecast_control.nowcast_control_blend_enabled",
        _multires_forecast_control.get("nowcast_control_blend_enabled", True),
    ),
    "nowcast_control_blend_weights": _as_float_list(
        "multires.forecast_control.nowcast_control_blend_weights",
        _multires_forecast_control.get(
            "nowcast_control_blend_weights",
            [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0],
        ),
    ),
    "nowcast_control_bucket_blend_enabled": _as_bool(
        "multires.forecast_control.nowcast_control_bucket_blend_enabled",
        _multires_forecast_control.get("nowcast_control_bucket_blend_enabled", True),
    ),
    "nowcast_control_bucket_size_minutes": _as_int(
        "multires.forecast_control.nowcast_control_bucket_size_minutes",
        _multires_forecast_control.get("nowcast_control_bucket_size_minutes", 5),
    ),
    "nowcast_control_bucket_blend_weights": _as_float_list(
        "multires.forecast_control.nowcast_control_bucket_blend_weights",
        _multires_forecast_control.get(
            "nowcast_control_bucket_blend_weights",
            [0.0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
        ),
    ),
    "nowcast_advisory_evidence_enabled": _as_bool(
        "multires.forecast_control.nowcast_advisory_evidence_enabled",
        _multires_forecast_control.get("nowcast_advisory_evidence_enabled", True),
    ),
    "nowcast_advisory_tie_tolerance": _as_float(
        "multires.forecast_control.nowcast_advisory_tie_tolerance",
        _multires_forecast_control.get("nowcast_advisory_tie_tolerance", 0.002),
    ),
    "nowcast_dynamic_overlay_enabled": _as_bool(
        "multires.forecast_control.nowcast_dynamic_overlay_enabled",
        _multires_forecast_control.get("nowcast_dynamic_overlay_enabled", True),
    ),
    "nowcast_dynamic_overlay_enforce": _as_bool(
        "multires.forecast_control.nowcast_dynamic_overlay_enforce",
        _multires_forecast_control.get("nowcast_dynamic_overlay_enforce", False),
    ),
    "nowcast_dynamic_overlay_profile_active_threshold": _as_float(
        "multires.forecast_control.nowcast_dynamic_overlay_profile_active_threshold",
        _multires_forecast_control.get("nowcast_dynamic_overlay_profile_active_threshold", 0.5),
    ),
    "nowcast_dynamic_overlay_high_ramp_fraction_threshold": _as_float(
        "multires.forecast_control.nowcast_dynamic_overlay_high_ramp_fraction_threshold",
        _multires_forecast_control.get("nowcast_dynamic_overlay_high_ramp_fraction_threshold", 0.15),
    ),
    "nowcast_dynamic_overlay_allow_next_lock": _as_bool(
        "multires.forecast_control.nowcast_dynamic_overlay_allow_next_lock",
        _multires_forecast_control.get("nowcast_dynamic_overlay_allow_next_lock", True),
    ),
    "nowcast_dynamic_overlay_allow_predicted_peak": _as_bool(
        "multires.forecast_control.nowcast_dynamic_overlay_allow_predicted_peak",
        _multires_forecast_control.get("nowcast_dynamic_overlay_allow_predicted_peak", True),
    ),
    "nowcast_soft_overlay_shadow_enabled": _as_bool(
        "multires.forecast_control.nowcast_soft_overlay_shadow_enabled",
        _multires_forecast_control.get("nowcast_soft_overlay_shadow_enabled", True),
    ),
    "nowcast_soft_overlay_supported_weights": _as_float_list(
        "multires.forecast_control.nowcast_soft_overlay_supported_weights",
        _multires_forecast_control.get("nowcast_soft_overlay_supported_weights", [0.80, 0.90, 0.95, 1.0]),
    ),
    "nowcast_soft_overlay_background_weights": _as_float_list(
        "multires.forecast_control.nowcast_soft_overlay_background_weights",
        _multires_forecast_control.get(
            "nowcast_soft_overlay_background_weights",
            [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0],
        ),
    ),
    "nowcast_soft_overlay_max_next_lock_regress_pct": _as_float(
        "multires.forecast_control.nowcast_soft_overlay_max_next_lock_regress_pct",
        _multires_forecast_control.get("nowcast_soft_overlay_max_next_lock_regress_pct", 0.0),
    ),
    "nowcast_soft_overlay_max_peak_hit_regress": _as_float(
        "multires.forecast_control.nowcast_soft_overlay_max_peak_hit_regress",
        _multires_forecast_control.get("nowcast_soft_overlay_max_peak_hit_regress", 0.0),
    ),
    "day_ahead_selection_metric": _as_str(
        "multires.forecast_control.day_ahead_selection_metric",
        _multires_forecast_control.get("day_ahead_selection_metric", "profile_shape_mae"),
    ),
    "hourly_selection_metric": _as_str(
        "multires.forecast_control.hourly_selection_metric",
        _multires_forecast_control.get("hourly_selection_metric", "next_lock_mae"),
    ),
    "phase_selection_metric": _as_str(
        "multires.forecast_control.phase_selection_metric",
        _multires_forecast_control.get("phase_selection_metric", "next_lock_mae"),
    ),
    "phase_stack_selection_metric": _as_str(
        "multires.forecast_control.phase_stack_selection_metric",
        _multires_forecast_control.get("phase_stack_selection_metric", "optimizer_score"),
    ),
    "nowcast_selection_metric": _as_str(
        "multires.forecast_control.nowcast_selection_metric",
        _multires_forecast_control.get("nowcast_selection_metric", "lock_mae"),
    ),
    "optimizer_selection_next_lock_weight": _as_float(
        "multires.forecast_control.optimizer_selection_next_lock_weight",
        _multires_forecast_control.get("optimizer_selection_next_lock_weight", 0.40),
    ),
    "optimizer_selection_lock_weight": _as_float(
        "multires.forecast_control.optimizer_selection_lock_weight",
        _multires_forecast_control.get("optimizer_selection_lock_weight", 0.15),
    ),
    "optimizer_selection_peak_value_weight": _as_float(
        "multires.forecast_control.optimizer_selection_peak_value_weight",
        _multires_forecast_control.get("optimizer_selection_peak_value_weight", 0.20),
    ),
    "optimizer_selection_peak_miss_weight": _as_float(
        "multires.forecast_control.optimizer_selection_peak_miss_weight",
        _multires_forecast_control.get("optimizer_selection_peak_miss_weight", 0.25),
    ),
    "optimizer_delivery_min_lead_specific_samples": _as_int(
        "multires.forecast_control.optimizer_delivery_min_lead_specific_samples",
        _multires_forecast_control.get("optimizer_delivery_min_lead_specific_samples", 8),
    ),
    "optimizer_delivery_next_lock_min_samples": _as_int(
        "multires.forecast_control.optimizer_delivery_next_lock_min_samples",
        _multires_forecast_control.get("optimizer_delivery_next_lock_min_samples", 8),
    ),
    "optimizer_delivery_next_lock_scaled_enabled": _as_bool(
        "multires.forecast_control.optimizer_delivery_next_lock_scaled_enabled",
        _multires_forecast_control.get("optimizer_delivery_next_lock_scaled_enabled", True),
    ),
    "optimizer_delivery_next_lock_scale_floor_quantile": _as_float(
        "multires.forecast_control.optimizer_delivery_next_lock_scale_floor_quantile",
        _multires_forecast_control.get("optimizer_delivery_next_lock_scale_floor_quantile", 0.25),
    ),
    "optimizer_delivery_next_lock_scale_floor_min_load": _as_float(
        "multires.forecast_control.optimizer_delivery_next_lock_scale_floor_min_load",
        _multires_forecast_control.get("optimizer_delivery_next_lock_scale_floor_min_load", 250.0),
    ),
    "optimizer_delivery_predicted_peak_min_samples": _as_int(
        "multires.forecast_control.optimizer_delivery_predicted_peak_min_samples",
        _multires_forecast_control.get("optimizer_delivery_predicted_peak_min_samples", 6),
    ),
    "optimizer_delivery_predicted_peak_lead_min_samples": _as_int(
        "multires.forecast_control.optimizer_delivery_predicted_peak_lead_min_samples",
        _multires_forecast_control.get("optimizer_delivery_predicted_peak_lead_min_samples", 4),
    ),
    "optimizer_delivery_confidence_full_support_n": _as_int(
        "multires.forecast_control.optimizer_delivery_confidence_full_support_n",
        _multires_forecast_control.get("optimizer_delivery_confidence_full_support_n", 8),
    ),
    "optimizer_delivery_confidence_band_width_scale_pct": _as_float(
        "multires.forecast_control.optimizer_delivery_confidence_band_width_scale_pct",
        _multires_forecast_control.get("optimizer_delivery_confidence_band_width_scale_pct", 100.0),
    ),
    "optimizer_delivery_nowcast_cadence_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_nowcast_cadence_minutes",
        _multires_forecast_control.get("optimizer_delivery_nowcast_cadence_minutes", 1),
    ),
    "optimizer_delivery_phase_cadence_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_phase_cadence_minutes",
        _multires_forecast_control.get("optimizer_delivery_phase_cadence_minutes", 15),
    ),
    "optimizer_delivery_hourly_cadence_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_hourly_cadence_minutes",
        _multires_forecast_control.get("optimizer_delivery_hourly_cadence_minutes", 60),
    ),
    "optimizer_delivery_day_ahead_cadence_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_day_ahead_cadence_minutes",
        _multires_forecast_control.get("optimizer_delivery_day_ahead_cadence_minutes", 1440),
    ),
    "optimizer_delivery_nowcast_stale_threshold_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_nowcast_stale_threshold_minutes",
        _multires_forecast_control.get("optimizer_delivery_nowcast_stale_threshold_minutes", 5),
    ),
    "optimizer_delivery_phase_stale_threshold_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_phase_stale_threshold_minutes",
        _multires_forecast_control.get("optimizer_delivery_phase_stale_threshold_minutes", 30),
    ),
    "optimizer_delivery_hourly_stale_threshold_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_hourly_stale_threshold_minutes",
        _multires_forecast_control.get("optimizer_delivery_hourly_stale_threshold_minutes", 90),
    ),
    "optimizer_delivery_day_ahead_stale_threshold_minutes": _as_int(
        "multires.forecast_control.optimizer_delivery_day_ahead_stale_threshold_minutes",
        _multires_forecast_control.get("optimizer_delivery_day_ahead_stale_threshold_minutes", 1560),
    ),
    "optimizer_delivery_confidence_layer_multipliers": _as_float_dict(
        "multires.forecast_control.optimizer_delivery_confidence_layer_multipliers",
        _multires_forecast_control.get(
            "optimizer_delivery_confidence_layer_multipliers",
            _optimizer_delivery_confidence_layer_defaults,
        ),
    ),
    "optimizer_delivery_confidence_quantile_source_multipliers": _as_float_dict(
        "multires.forecast_control.optimizer_delivery_confidence_quantile_source_multipliers",
        _multires_forecast_control.get(
            "optimizer_delivery_confidence_quantile_source_multipliers",
            _optimizer_delivery_confidence_quantile_source_defaults,
        ),
    ),
    "replay_cache_enabled": _as_bool(
        "multires.forecast_control.replay_cache_enabled",
        _multires_forecast_control.get("replay_cache_enabled", True),
    ),
    "replay_cache_dirname": _as_str(
        "multires.forecast_control.replay_cache_dirname",
        _multires_forecast_control.get("replay_cache_dirname", "replay_cache"),
    ),
    "write_latest": _as_bool(
        "multires.forecast_control.write_latest",
        _multires_forecast_control.get("write_latest", True),
    ),
}

_modeling_parallel = cast(dict[str, object], _MODELING_TOML["parallel"])
MODELING_PARALLEL: Final[ModelingParallelConfig] = {
    "enabled": _as_bool("parallel.enabled", _modeling_parallel["enabled"]),
    "backend": cast(
        ParallelBackend,
        _as_str("parallel.backend", _modeling_parallel["backend"]),
    ),
    "max_workers": _as_int("parallel.max_workers", _modeling_parallel["max_workers"]),
    "batch_size": _as_int("parallel.batch_size", _modeling_parallel["batch_size"]),
    "pre_dispatch": _as_str("parallel.pre_dispatch", _modeling_parallel["pre_dispatch"]),
    "min_tasks": _as_int("parallel.min_tasks", _modeling_parallel["min_tasks"]),
    "inner_threads_per_worker": _as_int(
        "parallel.inner_threads_per_worker",
        _modeling_parallel["inner_threads_per_worker"],
    ),
}

def _build_modeling_stage_parallel_config(
    stage_name: str,
    stage_config: dict[str, object] | None,
) -> ModelingStageParallelConfig:
    """Resolve one per-stage modeling parallel profile from `config/modeling.toml`."""
    resolved_stage_config = stage_config or {}
    return {
        "enabled": _as_bool(
            f"parallel.{stage_name}.enabled",
            resolved_stage_config.get("enabled", MODELING_PARALLEL["enabled"]),
        ),
        "max_workers": _as_int(
            f"parallel.{stage_name}.max_workers",
            resolved_stage_config.get("max_workers", _modeling_parallel["max_workers"]),
        ),
        "inner_threads_per_worker": _as_int(
            f"parallel.{stage_name}.inner_threads_per_worker",
            resolved_stage_config.get(
                "inner_threads_per_worker",
                _modeling_parallel["inner_threads_per_worker"],
            ),
        ),
        "high_capacity_host_only": _as_bool(
            f"parallel.{stage_name}.high_capacity_host_only",
            resolved_stage_config.get("high_capacity_host_only", False),
        ),
    }


MODELING_STAGE_PARALLEL: Final[dict[str, ModelingStageParallelConfig]] = {
    "performance": _build_modeling_stage_parallel_config(
        "performance",
        cast(dict[str, object] | None, _modeling_parallel.get("performance")),
    ),
    "multires": _build_modeling_stage_parallel_config(
        "multires",
        cast(dict[str, object] | None, _modeling_parallel.get("multires")),
    ),
    "rollout_sweep": _build_modeling_stage_parallel_config(
        "rollout_sweep",
        cast(dict[str, object] | None, _modeling_parallel.get("rollout_sweep")),
    ),
    "forecast_control": _build_modeling_stage_parallel_config(
        "forecast_control",
        cast(dict[str, object] | None, _modeling_parallel.get("forecast_control")),
    ),
}

_modeling_performance = cast(dict[str, object], _MODELING_TOML["performance"])
_modeling_performance_ramp = cast(dict[str, object], _modeling_performance["ramp"])
MODELING_PERFORMANCE_RAMP: Final[PerformanceRampConfig] = {
    "quantile": _as_float(
        "performance.ramp.quantile",
        _modeling_performance_ramp["quantile"],
    ),
}

_modeling_performance_blend_search = cast(
    dict[str, object],
    _modeling_performance["blend_search"],
)
MODELING_PERFORMANCE_BLEND_SEARCH: Final[PerformanceBlendSearchConfig] = {
    "enabled": _as_bool(
        "performance.blend_search.enabled",
        _modeling_performance_blend_search["enabled"],
    ),
    "base_window": _as_int(
        "performance.blend_search.base_window",
        _modeling_performance_blend_search["base_window"],
    ),
    "base_sharpness": _as_float(
        "performance.blend_search.base_sharpness",
        _modeling_performance_blend_search["base_sharpness"],
    ),
    "min_weight": _as_float(
        "performance.blend_search.min_weight",
        _modeling_performance_blend_search["min_weight"],
    ),
    "max_weight": _as_float(
        "performance.blend_search.max_weight",
        _modeling_performance_blend_search["max_weight"],
    ),
    "window_multipliers": _as_float_list(
        "performance.blend_search.window_multipliers",
        _modeling_performance_blend_search["window_multipliers"],
    ),
    "sharpness_multipliers": _as_float_list(
        "performance.blend_search.sharpness_multipliers",
        _modeling_performance_blend_search["sharpness_multipliers"],
    ),
    "bucket_enabled": _as_bool(
        "performance.blend_search.bucket_enabled",
        _modeling_performance_blend_search.get("bucket_enabled", True),
    ),
    "bucket_size_minutes": _as_int(
        "performance.blend_search.bucket_size_minutes",
        _modeling_performance_blend_search.get("bucket_size_minutes", 5),
    ),
    "bucket_cycle_minutes": _as_int(
        "performance.blend_search.bucket_cycle_minutes",
        _modeling_performance_blend_search.get("bucket_cycle_minutes", 15),
    ),
    "bucket_candidate_weights": _as_float_list(
        "performance.blend_search.bucket_candidate_weights",
        _modeling_performance_blend_search.get(
            "bucket_candidate_weights",
            [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        ),
    ),
}

_modeling_performance_hgb_search = cast(
    dict[str, object],
    _modeling_performance["hgb_search"],
)
MODELING_PERFORMANCE_HGB_SEARCH: Final[PerformanceHgbSearchConfig] = {
    "enabled": _as_bool(
        "performance.hgb_search.enabled",
        _modeling_performance_hgb_search["enabled"],
    ),
    "screen_folds": _as_int(
        "performance.hgb_search.screen_folds",
        _modeling_performance_hgb_search["screen_folds"],
    ),
    "min_candidates": _as_int(
        "performance.hgb_search.min_candidates",
        _modeling_performance_hgb_search["min_candidates"],
    ),
    "max_candidates": _as_int(
        "performance.hgb_search.max_candidates",
        _modeling_performance_hgb_search["max_candidates"],
    ),
    "learning_rates": _as_float_list(
        "performance.hgb_search.learning_rates",
        _modeling_performance_hgb_search["learning_rates"],
    ),
    "max_depths": [
        _as_int("performance.hgb_search.max_depths", value)
        for value in cast(list[object], _modeling_performance_hgb_search["max_depths"])
    ],
    "min_samples_leaf": [
        _as_int("performance.hgb_search.min_samples_leaf", value)
        for value in cast(list[object], _modeling_performance_hgb_search["min_samples_leaf"])
    ],
    "l2_regularization": _as_float_list(
        "performance.hgb_search.l2_regularization",
        _modeling_performance_hgb_search["l2_regularization"],
    ),
    "max_iters": [
        _as_int("performance.hgb_search.max_iters", value)
        for value in cast(list[object], _modeling_performance_hgb_search["max_iters"])
    ],
}

_modeling_performance_quick_profiles = cast(
    dict[str, object],
    _modeling_performance["quick_profiles"],
)
MODELING_PERFORMANCE_QUICK_PROFILES: Final[dict[str, PerformanceQuickProfileConfig]] = {
    name: {
        "feature_sets": _as_str_list(
            f"performance.quick_profiles.{name}.feature_sets",
            cast(dict[str, object], payload)["feature_sets"],
        ),
        "model_labels": _as_str_list(
            f"performance.quick_profiles.{name}.model_labels",
            cast(dict[str, object], payload)["model_labels"],
        ),
        "n_folds": _as_int(
            f"performance.quick_profiles.{name}.n_folds",
            cast(dict[str, object], payload)["n_folds"],
        ),
        "val_window_days": _as_int(
            f"performance.quick_profiles.{name}.val_window_days",
            cast(dict[str, object], payload)["val_window_days"],
        ),
    }
    for name, payload in _modeling_performance_quick_profiles.items()
}

_modeling_performance_horizon_policies = cast(
    dict[str, object],
    _modeling_performance["horizon_policies"],
)
MODELING_HORIZON_POLICIES: Final[dict[str, PerformanceHorizonPolicyConfig]] = {
    name: {
        "max_horizon_minutes": _as_int(
            f"performance.horizon_policies.{name}.max_horizon_minutes",
            cast(dict[str, object], payload)["max_horizon_minutes"],
        ),
        "feature_sets": _as_str_list(
            f"performance.horizon_policies.{name}.feature_sets",
            cast(dict[str, object], payload)["feature_sets"],
        ),
        "model_labels": _as_str_list(
            f"performance.horizon_policies.{name}.model_labels",
            cast(dict[str, object], payload)["model_labels"],
        ),
        "allow_residual": _as_bool(
            f"performance.horizon_policies.{name}.allow_residual",
            cast(dict[str, object], payload)["allow_residual"],
        ),
        "allow_blend": _as_bool(
            f"performance.horizon_policies.{name}.allow_blend",
            cast(dict[str, object], payload)["allow_blend"],
        ),
        "rollout_residual_baseline": _as_str(
            f"performance.horizon_policies.{name}.rollout_residual_baseline",
            cast(dict[str, object], payload)["rollout_residual_baseline"],
        ),
        "rollout_residual_candidates": _as_str_list(
            f"performance.horizon_policies.{name}.rollout_residual_candidates",
            cast(dict[str, object], payload)["rollout_residual_candidates"],
        ),
        "rollout_origin_policy": _as_str(
            f"performance.horizon_policies.{name}.rollout_origin_policy",
            cast(dict[str, object], payload)["rollout_origin_policy"],
        ),
        "rollout_selection_target": _as_str(
            f"performance.horizon_policies.{name}.rollout_selection_target",
            cast(dict[str, object], payload)["rollout_selection_target"],
        ),
    }
    for name, payload in _modeling_performance_horizon_policies.items()
}

_modeling_performance_evaluation = cast(dict[str, object], _modeling_performance["evaluation"])
MODELING_PERFORMANCE_EVALUATION: Final[PerformanceEvaluationConfig] = {
    "segment_columns": _as_str_list(
        "performance.evaluation.segment_columns",
        _modeling_performance_evaluation["segment_columns"],
    ),
    "classical_benchmarks": _as_str_list(
        "performance.evaluation.classical_benchmarks",
        _modeling_performance_evaluation.get("classical_benchmarks", ["holt_damped"]),
    ),
    "supplemental_surface_splits": _as_str_list(
        "performance.evaluation.supplemental_surface_splits",
        _modeling_performance_evaluation.get("supplemental_surface_splits", ["validate", "test"]),
    ),
    "supplemental_load_band_quantile": _as_float(
        "performance.evaluation.supplemental_load_band_quantile",
        _modeling_performance_evaluation.get("supplemental_load_band_quantile", 0.90),
    ),
    "supplemental_ramp_band_quantile": _as_float(
        "performance.evaluation.supplemental_ramp_band_quantile",
        _modeling_performance_evaluation.get("supplemental_ramp_band_quantile", 0.90),
    ),
    "bootstrap_samples": _as_int(
        "performance.evaluation.bootstrap_samples",
        _modeling_performance_evaluation.get("bootstrap_samples", 400),
    ),
    "bootstrap_confidence_level": _as_float(
        "performance.evaluation.bootstrap_confidence_level",
        _modeling_performance_evaluation.get("bootstrap_confidence_level", 0.95),
    ),
    "bootstrap_min_block_minutes": _as_int(
        "performance.evaluation.bootstrap_min_block_minutes",
        _modeling_performance_evaluation.get("bootstrap_min_block_minutes", 15),
    ),
    "bootstrap_max_block_minutes": _as_int(
        "performance.evaluation.bootstrap_max_block_minutes",
        _modeling_performance_evaluation.get("bootstrap_max_block_minutes", 360),
    ),
    "bootstrap_consecutive_insignificant": _as_int(
        "performance.evaluation.bootstrap_consecutive_insignificant",
        _modeling_performance_evaluation.get("bootstrap_consecutive_insignificant", 5),
    ),
    "importance_repeats": _as_int(
        "performance.evaluation.importance_repeats",
        _modeling_performance_evaluation.get("importance_repeats", 20),
    ),
    "importance_max_features": _as_int(
        "performance.evaluation.importance_max_features",
        _modeling_performance_evaluation.get("importance_max_features", 15),
    ),
    "importance_random_state": _as_int(
        "performance.evaluation.importance_random_state",
        _modeling_performance_evaluation.get("importance_random_state", 42),
    ),
}


def _validate_positive_integer_list(name: str, values: list[int]) -> None:
    """Validate a non-empty list of strictly positive integers."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"FEATURE_CONFIG['{name}'] must be a non-empty list of integers.")
    if any((not isinstance(value, int)) or value <= 0 for value in values):
        raise ValueError(
            f"FEATURE_CONFIG['{name}'] must contain only positive integers. Got: {values}"
        )


def _validate_fourier_cycles(cycles: list[FourierCycleSpec]) -> None:
    """Validate configured Fourier cycles and derived column names."""
    supported_sources = {"hour", "day_of_week"}
    if not isinstance(cycles, list):
        raise ValueError("FEATURE_CONFIG['fourier_cycles'] must be a list.")

    seen_prefixes: set[str] = set()
    generated_columns: list[str] = []
    for cycle in cycles:
        source = cycle["source"]
        period = cycle["period"]
        prefix = cycle["prefix"]
        if source not in supported_sources:
            raise ValueError(
                "FEATURE_CONFIG['fourier_cycles'] contains unsupported source "
                f"'{source}'. Supported: {sorted(supported_sources)}"
            )
        if period <= 0:
            raise ValueError(
                "FEATURE_CONFIG['fourier_cycles'] periods must be positive. "
                f"Got: {period}"
            )
        if not prefix or not prefix.replace("_", "").isalnum():
            raise ValueError(
                "FEATURE_CONFIG['fourier_cycles'] prefixes must be non-empty "
                "alphanumeric/underscore strings."
            )
        if prefix in seen_prefixes:
            raise ValueError(
                "FEATURE_CONFIG['fourier_cycles'] prefixes must be unique. "
                f"Duplicate: {prefix}"
            )
        seen_prefixes.add(prefix)
        generated_columns.extend([f"{prefix}_sin", f"{prefix}_cos"])

    duplicates = sorted(
        {column_name for column_name in generated_columns if generated_columns.count(column_name) > 1}
    )
    if duplicates:
        raise ValueError(
            "FEATURE_CONFIG['fourier_cycles'] generated duplicate columns: "
            f"{duplicates}"
        )


def _validate_split_ranges_contiguous(split_ranges: dict[str, tuple[int, int]]) -> None:
    """Validate split ranges are positive, non-overlapping, and gap-free."""
    if not split_ranges:
        raise ValueError("SPLIT_DAY_RANGES must not be empty.")

    covered: set[int] = set()
    for split_name, bounds in split_ranges.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                f"SPLIT_DAY_RANGES['{split_name}'] must be a tuple(start_day, end_day)."
            )
        start_day, end_day = bounds
        if not (isinstance(start_day, int) and isinstance(end_day, int)):
            raise ValueError(
                f"SPLIT_DAY_RANGES['{split_name}'] values must be integers. Got: {bounds}"
            )
        if start_day <= 0 or end_day <= 0:
            raise ValueError(
                f"SPLIT_DAY_RANGES['{split_name}'] must be positive day numbers. Got: {bounds}"
            )
        if start_day > end_day:
            raise ValueError(
                f"SPLIT_DAY_RANGES['{split_name}'] has start > end. Got: {bounds}"
            )

        day_values = set(range(start_day, end_day + 1))
        overlap = covered & day_values
        if overlap:
            raise ValueError(
                f"SPLIT_DAY_RANGES has overlapping day assignments in '{split_name}': "
                f"{sorted(overlap)}"
            )
        covered |= day_values

    min_day = min(covered)
    max_day = max(covered)
    expected_days = set(range(min_day, max_day + 1))
    missing_days = sorted(expected_days - covered)
    if missing_days:
        raise ValueError(
            "SPLIT_DAY_RANGES must be contiguous with no gaps. Missing days: "
            f"{missing_days}"
        )


def _canonical_resolution(resolution: str) -> str:
    """Resolve aliases and validate a resolution string."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    if canonical not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
            f"(aliases: {sorted(RESOLUTION_ALIASES)})"
        )
    return canonical


def resolve_resolution_suffix(resolution: str) -> str:
    """Return output file suffix for a resolution string."""
    canonical = _canonical_resolution(resolution)
    return RESOLUTION_TO_SUFFIX[canonical]


def get_silver_path(resolution: str) -> Path:
    """Return silver parquet path for a resolution."""
    suffix = resolve_resolution_suffix(resolution)
    return PATHS["silver_dir"] / f"power_load_{suffix}.parquet"


def get_gold_path(resolution: str) -> Path:
    """Return gold parquet path for a resolution."""
    suffix = resolve_resolution_suffix(resolution)
    return PATHS["gold_dir"] / f"power_load_{suffix}_all_features.parquet"


def resolve_eda_resolutions(
    mode: EDAResolutionMode, custom_list: list[str] | tuple[str, ...] | None = None
) -> list[str]:
    """Resolve notebook resolution mode into a canonical ordered list."""
    if mode == "all":
        return list(SUPPORTED_RESOLUTIONS)
    if mode == "default":
        return list(DEFAULT_RESOLUTIONS)
    if mode != "custom":
        raise ValueError(
            f"Unsupported EDA resolution mode '{mode}'. "
            f"Supported modes: {list(EDA_RESOLUTION_MODES)}"
        )

    if custom_list is None or len(custom_list) == 0:
        raise ValueError("custom_list must be provided when mode='custom'.")

    resolved: list[str] = []
    seen: set[str] = set()
    for entry in custom_list:
        canonical = _canonical_resolution(entry)
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    return resolved


def resolve_horizon_policy(horizon_minutes: int) -> PerformanceHorizonPolicyConfig:
    """Return the first configured modeling policy that covers the requested horizon."""
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive. Got: {horizon_minutes}")
    policies = sorted(
        MODELING_HORIZON_POLICIES.items(),
        key=lambda item: item[1]["max_horizon_minutes"],
    )
    for _, policy in policies:
        if int(horizon_minutes) <= int(policy["max_horizon_minutes"]):
            return policy
    return policies[-1][1]


def resolve_performance_quick_profile(horizon_minutes: int) -> PerformanceQuickProfileConfig:
    """Return the Stage-5 quick profile aligned to the requested forecast horizon."""
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive. Got: {horizon_minutes}")
    if horizon_minutes <= int(MODELING_HORIZON_POLICIES["short"]["max_horizon_minutes"]):
        return MODELING_PERFORMANCE_QUICK_PROFILES["short"]
    if horizon_minutes <= int(MODELING_HORIZON_POLICIES["hourly"]["max_horizon_minutes"]):
        return MODELING_PERFORMANCE_QUICK_PROFILES["hourly"]
    return MODELING_PERFORMANCE_QUICK_PROFILES["day_ahead"]


def resolve_rollout_origin_policy(horizon_minutes: int, configured_origin_policy: str) -> str:
    """Resolve `auto` rollout origin policy from the centralized horizon policy."""
    if str(configured_origin_policy) != "auto":
        return str(configured_origin_policy)
    return str(resolve_horizon_policy(int(horizon_minutes))["rollout_origin_policy"])


def resolve_rollout_selection_target(horizon_minutes: int, configured_selection_target: str) -> str:
    """Resolve `auto` rollout selection target from the centralized horizon policy."""
    if str(configured_selection_target) != "auto":
        return str(configured_selection_target)
    return str(resolve_horizon_policy(int(horizon_minutes))["rollout_selection_target"])


def validate_config() -> None:
    """Validate configuration consistency at runtime.

    This function is intentionally callable by tests and by the pipeline
    orchestrator during `--dry-run`.
    """
    for key in (
        "lag_periods",
        "rolling_periods",
        "slope_periods",
        "lag_minutes",
        "rolling_minutes",
        "slope_minutes",
    ):
        _validate_positive_integer_list(key, FEATURE_CONFIG[key])
    _validate_fourier_cycles(FEATURE_CONFIG["fourier_cycles"])
    if not 0.0 < FEATURE_CONFIG["profile_activity_threshold"] < 1.0:
        raise ValueError(
            "FEATURE_CONFIG['profile_activity_threshold'] must be within (0,1). "
            f"Got: {FEATURE_CONFIG['profile_activity_threshold']}"
        )

    if set(DAY_CLASS_MAP) != EXPECTED_DAY_CLASSES:
        raise ValueError(
            "DAY_CLASS_MAP keys must exactly match expected day classes "
            f"{sorted(EXPECTED_DAY_CLASSES)}. Got: {sorted(DAY_CLASS_MAP)}"
        )
    if len(set(DAY_CLASS_MAP.values())) != len(DAY_CLASS_MAP):
        raise ValueError(f"DAY_CLASS_MAP values must be unique. Got: {DAY_CLASS_MAP}")
    if any(not isinstance(value, int) for value in DAY_CLASS_MAP.values()):
        raise ValueError(f"DAY_CLASS_MAP values must be integers. Got: {DAY_CLASS_MAP}")

    unsupported_defaults = sorted(
        [resolution for resolution in DEFAULT_RESOLUTIONS if resolution not in SUPPORTED_RESOLUTIONS]
    )
    if unsupported_defaults:
        raise ValueError(
            "DEFAULT_RESOLUTIONS contains unsupported values: "
            f"{unsupported_defaults}. Supported: {SUPPORTED_RESOLUTIONS}"
        )

    invalid_alias_targets = {
        alias: target
        for alias, target in RESOLUTION_ALIASES.items()
        if target not in SUPPORTED_RESOLUTIONS
    }
    if invalid_alias_targets:
        raise ValueError(
            "RESOLUTION_ALIASES contains unsupported targets: "
            f"{invalid_alias_targets}. Supported: {SUPPORTED_RESOLUTIONS}"
        )

    if EDA_DEFAULT_RESOLUTION_MODE not in EDA_RESOLUTION_MODES:
        raise ValueError(
            f"EDA default mode must be one of {EDA_RESOLUTION_MODES}. "
            f"Got: {EDA_DEFAULT_RESOLUTION_MODE}"
        )

    if MULTIRES_CONFIG["mode"] not in {"smoke", "candidate", "full"}:
        raise ValueError(
            "MULTIRES_CONFIG['mode'] must be one of {'smoke','candidate','full'}. "
            f"Got: {MULTIRES_CONFIG['mode']}"
        )
    if MULTIRES_CONFIG["comparison_mode"] not in {"native_step", "matched_horizon"}:
        raise ValueError(
            "MULTIRES_CONFIG['comparison_mode'] must be 'native_step' or 'matched_horizon'. "
            f"Got: {MULTIRES_CONFIG['comparison_mode']}"
        )
    if not MULTIRES_CONFIG["horizons_minutes"]:
        raise ValueError("MULTIRES_CONFIG['horizons_minutes'] must not be empty.")
    if any(horizon <= 0 for horizon in MULTIRES_CONFIG["horizons_minutes"]):
        raise ValueError(
            "MULTIRES_CONFIG['horizons_minutes'] must contain only positive integers."
        )
    if not MULTIRES_CONFIG["matched_strategies"]:
        raise ValueError("MULTIRES_CONFIG['matched_strategies'] must not be empty.")
    if any(strategy not in {"recursive", "direct_endpoint"} for strategy in MULTIRES_CONFIG["matched_strategies"]):
        raise ValueError(
            "MULTIRES_CONFIG['matched_strategies'] must contain only {'recursive','direct_endpoint'}. "
            f"Got: {MULTIRES_CONFIG['matched_strategies']}"
        )
    if any(
        resolution not in SUPPORTED_RESOLUTIONS for resolution in MULTIRES_CONFIG["resolutions"]
    ):
        raise ValueError(
            "MULTIRES_CONFIG['resolutions'] contains unsupported values. "
            f"Got: {MULTIRES_CONFIG['resolutions']}"
        )
    if MULTIRES_SELECTION["min_eval_coverage"] < 0.0 or MULTIRES_SELECTION["min_eval_coverage"] > 1.0:
        raise ValueError(
            "MULTIRES_SELECTION['min_eval_coverage'] must be within [0,1]. "
            f"Got: {MULTIRES_SELECTION['min_eval_coverage']}"
        )
    if MULTIRES_SELECTION["max_fold_std_mae_ratio"] < 0.0:
        raise ValueError(
            "MULTIRES_SELECTION['max_fold_std_mae_ratio'] must be non-negative."
        )
    if MULTIRES_SELECTION["min_practical_mae_gain_pct"] < 0.0:
        raise ValueError(
            "MULTIRES_SELECTION['min_practical_mae_gain_pct'] must be non-negative."
        )
    if MULTIRES_SELECTION["min_practical_rmse_gain_pct"] < 0.0:
        raise ValueError(
            "MULTIRES_SELECTION['min_practical_rmse_gain_pct'] must be non-negative."
        )
    if MULTIRES_SELECTION["max_candidate_runtime_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_SELECTION['max_candidate_runtime_minutes'] must be positive."
        )
    if not MULTIRES_BASELINES["include_persistence"]:
        raise ValueError(
            "MULTIRES_BASELINES['include_persistence'] must remain true because "
            "persistence is the required comparison anchor."
        )
    required_profile_names = {"smoke", "candidate", "full"}
    missing_profiles = required_profile_names - set(MULTIRES_PROFILES)
    if missing_profiles:
        raise ValueError(
            "MULTIRES_PROFILES must define at least {'smoke','candidate','full'}. "
            f"Missing: {sorted(missing_profiles)}; got: {sorted(MULTIRES_PROFILES)}"
        )
    for profile_name, profile in MULTIRES_PROFILES.items():
        if profile["n_folds"] <= 0:
            raise ValueError(f"MULTIRES_PROFILES['{profile_name}']['n_folds'] must be positive.")
        if profile["val_window_days"] <= 0:
            raise ValueError(
                f"MULTIRES_PROFILES['{profile_name}']['val_window_days'] must be positive."
            )
        if profile["origins_per_fold"] <= 0:
            raise ValueError(
                f"MULTIRES_PROFILES['{profile_name}']['origins_per_fold'] must be positive."
            )
        if not profile["resolutions"]:
            raise ValueError(f"MULTIRES_PROFILES['{profile_name}']['resolutions'] must not be empty.")
        if any(resolution not in SUPPORTED_RESOLUTIONS for resolution in profile["resolutions"]):
            raise ValueError(
                f"MULTIRES_PROFILES['{profile_name}']['resolutions'] contains unsupported values. "
                f"Got: {profile['resolutions']}"
            )
        if not profile["horizons_minutes"] or any(value <= 0 for value in profile["horizons_minutes"]):
            raise ValueError(
                f"MULTIRES_PROFILES['{profile_name}']['horizons_minutes'] must contain only positive integers."
            )
        if not profile["feature_sets"]:
            raise ValueError(f"MULTIRES_PROFILES['{profile_name}']['feature_sets'] must not be empty.")
        if any(name not in FEATURE_SETS for name in profile["feature_sets"]):
            raise ValueError(
                f"MULTIRES_PROFILES['{profile_name}']['feature_sets'] contains unknown values. "
                f"Got: {profile['feature_sets']}"
            )
        if not profile["model_labels"]:
            raise ValueError(f"MULTIRES_PROFILES['{profile_name}']['model_labels'] must not be empty.")
    if MULTIRES_RUNTIME["full_runtime_warning_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_RUNTIME['full_runtime_warning_minutes'] must be positive."
        )
    for key in ("smoke_origins_per_fold", "candidate_origins_per_fold", "full_origins_per_fold"):
        if MULTIRES_RUNTIME[key] <= 0:
            raise ValueError(f"MULTIRES_RUNTIME['{key}'] must be positive.")
    if MULTIRES_ROLLOUT["strategy"] not in {"recursive"}:
        raise ValueError(
            "MULTIRES_ROLLOUT['strategy'] must currently be 'recursive'. "
            f"Got: {MULTIRES_ROLLOUT['strategy']}"
        )
    if MULTIRES_ROLLOUT["selected_resolution"] not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            "MULTIRES_ROLLOUT['selected_resolution'] contains unsupported value. "
            f"Got: {MULTIRES_ROLLOUT['selected_resolution']}"
        )
    if MULTIRES_ROLLOUT["feature_set"] not in FEATURE_SETS:
        raise ValueError(
            "MULTIRES_ROLLOUT['feature_set'] must match a configured feature set. "
            f"Got: {MULTIRES_ROLLOUT['feature_set']}"
        )
    if MULTIRES_ROLLOUT["horizon_minutes"] <= 0:
        raise ValueError("MULTIRES_ROLLOUT['horizon_minutes'] must be positive.")
    if MULTIRES_ROLLOUT["origins_per_run"] <= 0:
        raise ValueError("MULTIRES_ROLLOUT['origins_per_run'] must be positive.")
    if MULTIRES_ROLLOUT["origin_policy"] not in {
        "uniform",
        "midnight",
        "billing_aligned",
        "phase_balanced",
        "auto",
    }:
        raise ValueError(
            "MULTIRES_ROLLOUT['origin_policy'] must be one of "
            "{'uniform','midnight','billing_aligned','phase_balanced','auto'}. "
            f"Got: {MULTIRES_ROLLOUT['origin_policy']}"
        )
    if MULTIRES_ROLLOUT["selection_target"] not in {"endpoint_mae", "path_mae", "phase_mean_mae", "auto"}:
        raise ValueError(
            "MULTIRES_ROLLOUT['selection_target'] must be one of "
            "{'endpoint_mae','path_mae','phase_mean_mae','auto'}. "
            f"Got: {MULTIRES_ROLLOUT['selection_target']}"
        )
    if MULTIRES_ROLLOUT_CHALLENGERS["max_candidates"] <= 0:
        raise ValueError("MULTIRES_ROLLOUT_CHALLENGERS['max_candidates'] must be positive.")
    if MULTIRES_ROLLOUT_CHALLENGERS["parallel_workers"] <= 0:
        raise ValueError("MULTIRES_ROLLOUT_CHALLENGERS['parallel_workers'] must be positive.")
    if not MULTIRES_ROLLOUT_CHALLENGERS["origin_policies"]:
        raise ValueError("MULTIRES_ROLLOUT_CHALLENGERS['origin_policies'] must not be empty.")
    if any(
        policy not in {"uniform", "midnight", "billing_aligned", "phase_balanced"}
        for policy in MULTIRES_ROLLOUT_CHALLENGERS["origin_policies"]
    ):
        raise ValueError(
            "MULTIRES_ROLLOUT_CHALLENGERS['origin_policies'] must contain only "
            "{'uniform','midnight','billing_aligned','phase_balanced'}."
        )
    if not MULTIRES_ROLLOUT_CHALLENGERS["policy_resolutions"]:
        raise ValueError(
            "MULTIRES_ROLLOUT_CHALLENGERS['policy_resolutions'] must not be empty."
        )
    if any(
        _canonical_resolution(resolution) not in SUPPORTED_RESOLUTIONS
        for resolution in MULTIRES_ROLLOUT_CHALLENGERS["policy_resolutions"]
    ):
        raise ValueError(
            "MULTIRES_ROLLOUT_CHALLENGERS['policy_resolutions'] must contain only "
            "supported resolutions."
        )
    if MULTIRES_ROLLOUT_CHALLENGERS["recommendation_origin_scope"] not in {
        "requested_only",
        "all",
    }:
        raise ValueError(
            "MULTIRES_ROLLOUT_CHALLENGERS['recommendation_origin_scope'] must be one of "
            "{'requested_only','all'}."
        )
    if not MULTIRES_HORIZON_CURVE["horizons_minutes"] or any(
        value <= 0 for value in MULTIRES_HORIZON_CURVE["horizons_minutes"]
    ):
        raise ValueError(
            "MULTIRES_HORIZON_CURVE['horizons_minutes'] must contain only positive integers."
        )
    if MULTIRES_HORIZON_CURVE["origins_per_run"] <= 0:
        raise ValueError("MULTIRES_HORIZON_CURVE['origins_per_run'] must be positive.")
    if MULTIRES_HORIZON_CURVE["origin_policy"] not in {
        "uniform",
        "midnight",
        "billing_aligned",
        "phase_balanced",
        "auto",
    }:
        raise ValueError(
            "MULTIRES_HORIZON_CURVE['origin_policy'] must be one of "
            "{'uniform','midnight','billing_aligned','phase_balanced','auto'}. "
            f"Got: {MULTIRES_HORIZON_CURVE['origin_policy']}"
        )
    if MULTIRES_HORIZON_CURVE["selection_target"] not in {
        "endpoint_mae",
        "path_mae",
        "phase_mean_mae",
        "auto",
    }:
        raise ValueError(
            "MULTIRES_HORIZON_CURVE['selection_target'] must be one of "
            "{'endpoint_mae','path_mae','phase_mean_mae','auto'}. "
            f"Got: {MULTIRES_HORIZON_CURVE['selection_target']}"
        )
    if MULTIRES_HORIZON_CURVE["max_candidates"] <= 0:
        raise ValueError("MULTIRES_HORIZON_CURVE['max_candidates'] must be positive.")
    for key in (
        "day_ahead_horizon_minutes",
        "hourly_horizon_minutes",
        "phase_horizon_minutes",
        "nowcast_horizon_minutes",
        "day_ahead_refresh_interval_minutes",
        "day_ahead_refresh_lookback_minutes",
        "lock_interval_minutes",
    ):
        if MULTIRES_FORECAST_CONTROL[key] <= 0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must be positive.")
    if MULTIRES_FORECAST_CONTROL["actual_resolution"] not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['actual_resolution'] must be a supported resolution. "
            f"Got: {MULTIRES_FORECAST_CONTROL['actual_resolution']}"
        )
    if not 0 <= MULTIRES_FORECAST_CONTROL["cycle_origin_hour"] <= 23:
        raise ValueError("MULTIRES_FORECAST_CONTROL['cycle_origin_hour'] must be within [0,23].")
    if not 0 <= MULTIRES_FORECAST_CONTROL["cycle_origin_minute"] <= 59:
        raise ValueError("MULTIRES_FORECAST_CONTROL['cycle_origin_minute'] must be within [0,59].")
    if MULTIRES_FORECAST_CONTROL["cycle_origin_stride_minutes"] <= 0:
        raise ValueError("MULTIRES_FORECAST_CONTROL['cycle_origin_stride_minutes'] must be positive.")
    if MULTIRES_FORECAST_CONTROL["rolling_benchmark_origin_stride_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['rolling_benchmark_origin_stride_minutes'] must be positive."
        )
    if MULTIRES_FORECAST_CONTROL["max_cycles"] < 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['max_cycles'] must be non-negative "
            "(0 means use the full exact-control scope)."
        )
    if MULTIRES_FORECAST_CONTROL["rolling_benchmark_max_cycles"] < 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['rolling_benchmark_max_cycles'] must be non-negative "
            "(0 means use the full rolling benchmark scope)."
        )
    if MULTIRES_FORECAST_CONTROL["rolling_benchmark_bootstrap_samples"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['rolling_benchmark_bootstrap_samples'] must be positive."
        )
    if not 0.0 < MULTIRES_FORECAST_CONTROL["rolling_benchmark_confidence_level"] < 1.0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['rolling_benchmark_confidence_level'] must lie within (0,1)."
        )
    if MULTIRES_FORECAST_CONTROL["candidate_pool_size"] <= 0:
        raise ValueError("MULTIRES_FORECAST_CONTROL['candidate_pool_size'] must be positive.")
    if MULTIRES_FORECAST_CONTROL["candidate_benchmark_origin_cap"] <= 0:
        raise ValueError("MULTIRES_FORECAST_CONTROL['candidate_benchmark_origin_cap'] must be positive.")
    if MULTIRES_FORECAST_CONTROL["phase_candidate_benchmark_origin_cap"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_candidate_benchmark_origin_cap'] must be positive."
        )
    if MULTIRES_FORECAST_CONTROL["phase_candidate_evaluation_origin_cap"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_candidate_evaluation_origin_cap'] must be positive."
        )
    if MULTIRES_FORECAST_CONTROL["phase_control_candidate_pool_size"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_control_candidate_pool_size'] must be positive."
        )
    if MULTIRES_FORECAST_CONTROL["phase_control_origin_cap"] < 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_control_origin_cap'] must be non-negative "
            "(0 means use the full selected phase replay scope)."
        )
    if MULTIRES_FORECAST_CONTROL["phase_stack_native_learned_top_candidates_per_pool"] < 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_native_learned_top_candidates_per_pool'] "
            "must be non-negative."
        )
    if MULTIRES_FORECAST_CONTROL["phase_stack_native_baseline_top_candidates_per_pool"] < 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_native_baseline_top_candidates_per_pool'] "
            "must be non-negative."
        )
    if (
        MULTIRES_FORECAST_CONTROL["phase_stack_native_learned_top_candidates_per_pool"] <= 0
        and MULTIRES_FORECAST_CONTROL["phase_stack_native_baseline_top_candidates_per_pool"] <= 0
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL must keep at least one native phase candidate per pool "
            "for Stage-10 stack evaluation."
        )
    if not MULTIRES_FORECAST_CONTROL["phase_stack_blend_weights"]:
        raise ValueError("MULTIRES_FORECAST_CONTROL['phase_stack_blend_weights'] must not be empty.")
    if any(value <= 0.0 or value >= 1.0 for value in MULTIRES_FORECAST_CONTROL["phase_stack_blend_weights"]):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_blend_weights'] must lie within (0,1)."
        )
    if MULTIRES_FORECAST_CONTROL["phase_stack_blend_parent_top_candidates"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_blend_parent_top_candidates'] must be positive."
        )
    if MULTIRES_FORECAST_CONTROL["phase_stack_bucket_granularity_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_bucket_granularity_minutes'] must be positive."
        )
    if MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_top_candidates"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_baseline_control_top_candidates'] must be positive."
        )
    if not MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_weights"]:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_baseline_control_blend_weights'] must not be empty."
        )
    if any(
        value <= 0.0 or value >= 1.0
        for value in MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_weights"]
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_baseline_control_blend_weights'] must lie within (0,1)."
        )
    if MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_bucket_size_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_baseline_control_bucket_size_minutes'] must be positive."
        )
    for key in (
        "phase_stack_guard_min_lock_gain_pct",
        "phase_stack_guard_min_rolling_lock_gain_pct",
        "phase_stack_guard_max_rolling_next_lock_regress_pct",
        "phase_stack_guard_max_rolling_profile_degrade_pct",
        "phase_stack_guard_max_rolling_peak_value_regress_pct",
        "phase_stack_guard_min_rolling_peak_hit_gain",
        "phase_stack_guard_max_rolling_optimizer_regress_pct",
        "phase_stack_guard_max_next_lock_regress_pct",
        "phase_stack_guard_max_profile_degrade_pct",
        "phase_stack_guard_max_peak_value_regress_pct",
        "phase_stack_guard_min_peak_hit_gain",
        "phase_stack_guard_max_optimizer_regress_pct",
    ):
        if MULTIRES_FORECAST_CONTROL[key] < 0.0 or MULTIRES_FORECAST_CONTROL[key] > 1.0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must lie within [0,1].")
    if MULTIRES_FORECAST_CONTROL["phase_stack_guard_rolling_scope"] not in {
        "rolling_calibration",
        "rolling_evaluation",
        "rolling_combined",
    }:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['phase_stack_guard_rolling_scope'] must be one of "
            "{'rolling_calibration', 'rolling_evaluation', 'rolling_combined'}."
        )
    if MULTIRES_FORECAST_CONTROL["nowcast_candidate_pool_size"] <= 0:
        raise ValueError("MULTIRES_FORECAST_CONTROL['nowcast_candidate_pool_size'] must be positive.")
    if not MULTIRES_FORECAST_CONTROL["nowcast_control_blend_weights"]:
        raise ValueError("MULTIRES_FORECAST_CONTROL['nowcast_control_blend_weights'] must not be empty.")
    if any(
        value < 0.0 or value > 1.0 for value in MULTIRES_FORECAST_CONTROL["nowcast_control_blend_weights"]
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['nowcast_control_blend_weights'] must lie within [0,1]."
        )
    if MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_size_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['nowcast_control_bucket_size_minutes'] must be positive."
        )
    if not MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_weights"]:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['nowcast_control_bucket_blend_weights'] must not be empty."
        )
    if any(
        value < 0.0 or value > 1.0
        for value in MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_weights"]
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['nowcast_control_bucket_blend_weights'] must lie within [0,1]."
        )
    if (
        MULTIRES_FORECAST_CONTROL["day_ahead_refresh_interval_minutes"]
        > MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"]
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_interval_minutes'] must be <= "
            "'day_ahead_horizon_minutes'."
        )
    for key in (
        "day_ahead_refresh_residual_drift_mae_pct_threshold",
        "day_ahead_refresh_transition_mae_pct_threshold",
        "day_ahead_refresh_activity_ratio_shift_threshold",
    ):
        if MULTIRES_FORECAST_CONTROL[key] < 0.0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must be non-negative.")
    if not MULTIRES_FORECAST_CONTROL["day_ahead_refresh_threshold_quantiles"]:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_threshold_quantiles'] must not be empty."
        )
    if any(
        value <= 0.0 or value >= 1.0
        for value in MULTIRES_FORECAST_CONTROL["day_ahead_refresh_threshold_quantiles"]
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_threshold_quantiles'] must lie within (0,1)."
        )
    valid_refresh_trigger_modes = {
        "any",
        "residual_only",
        "activity_only",
        "activity_active_band",
        "transition_only",
        "residual_or_activity",
        "residual_or_activity_active_band",
        "residual_or_activity_active_or_transition",
        "residual_or_transition",
        "activity_or_transition",
        "residual_and_activity",
        "residual_and_transition",
        "activity_and_transition",
        "two_of_three",
    }
    if MULTIRES_FORECAST_CONTROL["day_ahead_refresh_trigger_mode"] not in valid_refresh_trigger_modes:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_trigger_mode'] references an unsupported mode."
        )
    if not MULTIRES_FORECAST_CONTROL["day_ahead_refresh_candidate_trigger_modes"]:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_candidate_trigger_modes'] must not be empty."
        )
    invalid_trigger_modes = sorted(
        set(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_candidate_trigger_modes"]) - valid_refresh_trigger_modes
    )
    if invalid_trigger_modes:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_candidate_trigger_modes'] "
            f"contains unsupported modes: {invalid_trigger_modes}."
        )
    for key in (
        "day_ahead_refresh_min_trigger_rate",
        "day_ahead_refresh_max_trigger_rate",
        "day_ahead_refresh_min_profile_gain_fraction_vs_unconditional",
        "day_ahead_refresh_min_lock_gain_fraction_vs_unconditional",
    ):
        if MULTIRES_FORECAST_CONTROL[key] < 0.0 or MULTIRES_FORECAST_CONTROL[key] > 1.0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must lie within [0,1].")
    if (
        MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_trigger_rate"]
        > MULTIRES_FORECAST_CONTROL["day_ahead_refresh_max_trigger_rate"]
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['day_ahead_refresh_min_trigger_rate'] must be <= "
            "'day_ahead_refresh_max_trigger_rate'."
        )
    if MULTIRES_FORECAST_CONTROL["control_promotion_scope"] not in {
        "calibration_only",
        "held_out_evaluation",
    }:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['control_promotion_scope'] must be either "
            "'calibration_only' or 'held_out_evaluation'."
        )
    for key in (
        "control_promotion_guard_max_next_lock_regress_pct",
        "control_promotion_guard_max_peak_value_regress_pct",
        "control_promotion_guard_max_peak_miss_regress",
    ):
        if MULTIRES_FORECAST_CONTROL[key] < 0.0 or MULTIRES_FORECAST_CONTROL[key] > 1.0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must lie within [0,1].")
    valid_split_names = set(SPLIT_DAY_RANGES)
    for key in ("calibration_splits", "evaluation_splits"):
        values = MULTIRES_FORECAST_CONTROL[key]
        if not values:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must not be empty.")
        unknown = sorted(set(values) - valid_split_names)
        if unknown:
            raise ValueError(
                f"MULTIRES_FORECAST_CONTROL['{key}'] references unknown splits: {unknown}."
            )
    for key in (
        "day_ahead_selection_metric",
        "hourly_selection_metric",
        "phase_selection_metric",
        "phase_stack_selection_metric",
    ):
        if MULTIRES_FORECAST_CONTROL[key] not in {
            "endpoint_mae",
            "path_mae",
            "phase_mean_mae",
            "next_lock_mae",
            "profile_shape_mae",
            "energy_mae",
            "lock_mae",
            "peak_value_mae",
            "peak_interval_miss_rate",
            "optimizer_score",
        }:
            raise ValueError(
                f"MULTIRES_FORECAST_CONTROL['{key}'] must be one of "
                "{'endpoint_mae','path_mae','phase_mean_mae','next_lock_mae','profile_shape_mae','energy_mae','lock_mae','peak_value_mae','peak_interval_miss_rate','optimizer_score'}."
            )
    if MULTIRES_FORECAST_CONTROL["nowcast_selection_metric"] not in {
        "lock_mae",
        "minute_path_mae",
        "profile_shape_mae",
        "energy_mae",
        "next_lock_mae",
        "peak_value_mae",
        "peak_interval_miss_rate",
        "optimizer_score",
    }:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['nowcast_selection_metric'] must be one of "
            "{'lock_mae','minute_path_mae','profile_shape_mae','energy_mae','next_lock_mae','peak_value_mae','peak_interval_miss_rate','optimizer_score'}."
        )
    selection_weight_keys = (
        "optimizer_selection_next_lock_weight",
        "optimizer_selection_lock_weight",
        "optimizer_selection_peak_value_weight",
        "optimizer_selection_peak_miss_weight",
    )
    for key in selection_weight_keys:
        if MULTIRES_FORECAST_CONTROL[key] < 0.0 or MULTIRES_FORECAST_CONTROL[key] > 1.0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must lie within [0,1].")
    selection_weight_total = sum(float(MULTIRES_FORECAST_CONTROL[key]) for key in selection_weight_keys)
    if not math.isclose(selection_weight_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "The optimizer-selection weights in MULTIRES_FORECAST_CONTROL must sum to 1.0. "
            f"Got: {selection_weight_total}"
        )
    positive_optimizer_keys = (
        "optimizer_delivery_min_lead_specific_samples",
        "optimizer_delivery_next_lock_min_samples",
        "optimizer_delivery_next_lock_scale_floor_min_load",
        "optimizer_delivery_predicted_peak_min_samples",
        "optimizer_delivery_predicted_peak_lead_min_samples",
        "optimizer_delivery_confidence_full_support_n",
        "optimizer_delivery_confidence_band_width_scale_pct",
        "optimizer_delivery_nowcast_cadence_minutes",
        "optimizer_delivery_phase_cadence_minutes",
        "optimizer_delivery_hourly_cadence_minutes",
        "optimizer_delivery_day_ahead_cadence_minutes",
        "optimizer_delivery_nowcast_stale_threshold_minutes",
        "optimizer_delivery_phase_stale_threshold_minutes",
        "optimizer_delivery_hourly_stale_threshold_minutes",
        "optimizer_delivery_day_ahead_stale_threshold_minutes",
    )
    for key in positive_optimizer_keys:
        if MULTIRES_FORECAST_CONTROL[key] <= 0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must be positive.")
    for key in (
        "candidate_pool_size",
        "candidate_benchmark_origin_cap",
        "phase_candidate_benchmark_origin_cap",
        "phase_candidate_evaluation_origin_cap",
        "phase_control_candidate_pool_size",
        "phase_control_prior_run_limit",
        "phase_control_min_prior_support_runs",
        "phase_control_max_supplemental_contexts_per_resolution",
        "phase_control_origin_cap",
        "phase_stack_blend_parent_top_candidates",
        "nowcast_candidate_pool_size",
    ):
        if int(MULTIRES_FORECAST_CONTROL[key]) <= 0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must be positive.")
    if int(MULTIRES_FORECAST_CONTROL["phase_control_exploration_slots"]) < 0:
        raise ValueError("MULTIRES_FORECAST_CONTROL['phase_control_exploration_slots'] must be non-negative.")
    for key in (
        "phase_stack_native_learned_top_candidates_per_pool",
        "phase_stack_native_baseline_top_candidates_per_pool",
    ):
        if int(MULTIRES_FORECAST_CONTROL[key]) < 0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must be non-negative.")
    if (
        int(MULTIRES_FORECAST_CONTROL["phase_stack_native_learned_top_candidates_per_pool"]) <= 0
        and int(MULTIRES_FORECAST_CONTROL["phase_stack_native_baseline_top_candidates_per_pool"]) <= 0
    ):
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL must keep at least one native phase candidate per pool "
            "for Stage-10 stack evaluation."
        )
    next_lock_scale_floor_quantile = float(
        MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scale_floor_quantile"]
    )
    if next_lock_scale_floor_quantile <= 0.0 or next_lock_scale_floor_quantile >= 1.0:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['optimizer_delivery_next_lock_scale_floor_quantile'] "
            "must lie strictly between 0 and 1."
        )
    for key in (
        "nowcast_advisory_tie_tolerance",
        "nowcast_dynamic_overlay_profile_active_threshold",
        "nowcast_dynamic_overlay_high_ramp_fraction_threshold",
        "nowcast_soft_overlay_max_next_lock_regress_pct",
        "nowcast_soft_overlay_max_peak_hit_regress",
    ):
        if MULTIRES_FORECAST_CONTROL[key] < 0.0 or MULTIRES_FORECAST_CONTROL[key] > 1.0:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must lie within [0,1].")
    for key in ("nowcast_soft_overlay_supported_weights", "nowcast_soft_overlay_background_weights"):
        weights = MULTIRES_FORECAST_CONTROL[key]
        if not weights:
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must not be empty.")
        if any(value < 0.0 or value > 1.0 for value in weights):
            raise ValueError(f"MULTIRES_FORECAST_CONTROL['{key}'] must contain only values within [0,1].")
    required_confidence_layer_keys = {"nowcast", "phase", "hourly", "day_ahead"}
    missing_confidence_layer_keys = required_confidence_layer_keys - set(
        MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_layer_multipliers"]
    )
    if missing_confidence_layer_keys:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['optimizer_delivery_confidence_layer_multipliers'] "
            f"is missing required keys: {sorted(missing_confidence_layer_keys)}."
        )
    required_quantile_multiplier_keys = {
        "lead_interval",
        "next_lock_scaled_global",
        "predicted_peak_lead_interval",
        "predicted_peak_global",
        "next_lock_global",
        "layer_global_fallback",
        "layer_global",
        "unavailable",
    }
    missing_quantile_multiplier_keys = required_quantile_multiplier_keys - set(
        MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_quantile_source_multipliers"]
    )
    if missing_quantile_multiplier_keys:
        raise ValueError(
            "MULTIRES_FORECAST_CONTROL['optimizer_delivery_confidence_quantile_source_multipliers'] "
            f"is missing required keys: {sorted(missing_quantile_multiplier_keys)}."
        )
    for config_key in (
        "optimizer_delivery_confidence_layer_multipliers",
        "optimizer_delivery_confidence_quantile_source_multipliers",
    ):
        for multiplier_key, multiplier_value in MULTIRES_FORECAST_CONTROL[config_key].items():
            if multiplier_value <= 0.0 or multiplier_value > 1.0:
                raise ValueError(
                    f"MULTIRES_FORECAST_CONTROL['{config_key}']['{multiplier_key}'] must lie within (0,1]."
                )
    for key in (
        "persistence_weight_start",
        "persistence_weight_end",
        "raw_weight_start",
        "raw_weight_end",
        "hybrid_phase_gate_aligned_weight",
        "hybrid_phase_gate_non_aligned_weight",
    ):
        if MULTIRES_ROLLOUT_LEARNED_BLENDS[key] < 0.0 or MULTIRES_ROLLOUT_LEARNED_BLENDS[key] > 1.0:
            raise ValueError(f"MULTIRES_ROLLOUT_LEARNED_BLENDS['{key}'] must be within [0,1].")
    if MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_step"] <= 0.0:
        raise ValueError("MULTIRES_ROLLOUT_LEARNED_BLENDS['refinement_step'] must be positive.")
    if MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_neighbors"] < 0:
        raise ValueError("MULTIRES_ROLLOUT_LEARNED_BLENDS['refinement_neighbors'] must be >= 0.")
    if MULTIRES_ROLLOUT_LEARNED_BLENDS["max_weights_per_family"] <= 0:
        raise ValueError("MULTIRES_ROLLOUT_LEARNED_BLENDS['max_weights_per_family'] must be positive.")
    if (
        MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_end"]
        > MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_start"]
    ):
        raise ValueError(
            "MULTIRES_ROLLOUT_LEARNED_BLENDS['persistence_weight_end'] must be <= "
            "MULTIRES_ROLLOUT_LEARNED_BLENDS['persistence_weight_start']."
        )
    if (
        MULTIRES_ROLLOUT_LEARNED_BLENDS["raw_weight_end"]
        > MULTIRES_ROLLOUT_LEARNED_BLENDS["raw_weight_start"]
    ):
        raise ValueError(
            "MULTIRES_ROLLOUT_LEARNED_BLENDS['raw_weight_end'] must be <= "
            "MULTIRES_ROLLOUT_LEARNED_BLENDS['raw_weight_start']."
        )
    if (
        MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_non_aligned_weight"]
        < MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_aligned_weight"]
    ):
        raise ValueError(
            "MULTIRES_ROLLOUT_LEARNED_BLENDS['hybrid_phase_gate_non_aligned_weight'] "
            "must be >= 'hybrid_phase_gate_aligned_weight'."
        )
    for bucket, weight in MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"].items():
        if bucket < 0 or bucket >= 15 * 60:
            raise ValueError(
                "MULTIRES_ROLLOUT_LEARNED_BLENDS['hybrid_phase_gate_bucket_weights'] "
                f"contains an invalid phase bucket: {bucket}."
            )
        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                "MULTIRES_ROLLOUT_LEARNED_BLENDS['hybrid_phase_gate_bucket_weights'] "
                f"contains an out-of-range weight for bucket {bucket}: {weight}."
            )
    if MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"] not in {"linear"}:
        raise ValueError(
            "MULTIRES_ROLLOUT_LEARNED_BLENDS['curve'] must currently be 'linear'. "
            f"Got: {MULTIRES_ROLLOUT_LEARNED_BLENDS['curve']}"
        )
    if MULTIRES_ROLLOUT_POLICY_CANDIDATES["max_horizon_minutes"] <= 0:
        raise ValueError(
            "MULTIRES_ROLLOUT_POLICY_CANDIDATES['max_horizon_minutes'] must be positive."
        )
    if not MULTIRES_ROLLOUT_POLICY_CANDIDATES["selection_targets"]:
        raise ValueError(
            "MULTIRES_ROLLOUT_POLICY_CANDIDATES['selection_targets'] must not be empty."
        )
    invalid_rollout_policy_targets = set(MULTIRES_ROLLOUT_POLICY_CANDIDATES["selection_targets"]) - {
        "endpoint_mae",
        "path_mae",
        "phase_mean_mae",
        "next_lock_mae",
    }
    if invalid_rollout_policy_targets:
        raise ValueError(
            "MULTIRES_ROLLOUT_POLICY_CANDIDATES['selection_targets'] contains unsupported values: "
            f"{sorted(invalid_rollout_policy_targets)}"
        )
    for key in ("persistence_weight_start", "persistence_weight_end"):
        if MULTIRES_HYBRID[key] < 0.0 or MULTIRES_HYBRID[key] > 1.0:
            raise ValueError(f"MULTIRES_HYBRID['{key}'] must be within [0,1].")
    if MULTIRES_HYBRID["persistence_weight_end"] > MULTIRES_HYBRID["persistence_weight_start"]:
        raise ValueError(
            "MULTIRES_HYBRID['persistence_weight_end'] must be <= "
            "MULTIRES_HYBRID['persistence_weight_start']."
        )
    if MULTIRES_HYBRID["curve"] not in {"linear"}:
        raise ValueError(
            "MULTIRES_HYBRID['curve'] must currently be 'linear'. "
            f"Got: {MULTIRES_HYBRID['curve']}"
        )
    if MODELING_PARALLEL["backend"] not in {"threading", "loky", "sequential"}:
        raise ValueError(
            "MODELING_PARALLEL['backend'] must be one of "
            "{'threading','loky','sequential'}. "
            f"Got: {MODELING_PARALLEL['backend']}"
        )
    for key in ("max_workers", "batch_size", "min_tasks", "inner_threads_per_worker"):
        if MODELING_PARALLEL[key] <= 0:
            raise ValueError(f"MODELING_PARALLEL['{key}'] must be positive.")
    if not MODELING_PARALLEL["pre_dispatch"].strip():
        raise ValueError("MODELING_PARALLEL['pre_dispatch'] must not be empty.")
    expected_stage_parallel_keys = {"performance", "multires", "rollout_sweep", "forecast_control"}
    if set(MODELING_STAGE_PARALLEL) != expected_stage_parallel_keys:
        raise ValueError(
            "MODELING_STAGE_PARALLEL must define exactly "
            f"{sorted(expected_stage_parallel_keys)}."
        )
    if not 0.0 < MODELING_PERFORMANCE_RAMP["quantile"] < 1.0:
        raise ValueError(
            "MODELING_PERFORMANCE_RAMP['quantile'] must be within (0,1). "
            f"Got: {MODELING_PERFORMANCE_RAMP['quantile']}"
        )
    if MODELING_PERFORMANCE_BLEND_SEARCH["base_window"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['base_window'] must be positive."
        )
    if MODELING_PERFORMANCE_BLEND_SEARCH["base_sharpness"] <= 0.0:
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['base_sharpness'] must be positive."
        )
    for key in ("min_weight", "max_weight"):
        if (
            MODELING_PERFORMANCE_BLEND_SEARCH[key] < 0.0
            or MODELING_PERFORMANCE_BLEND_SEARCH[key] > 1.0
        ):
            raise ValueError(
                f"MODELING_PERFORMANCE_BLEND_SEARCH['{key}'] must be within [0,1]."
            )
    if (
        MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"]
        > MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"]
    ):
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['min_weight'] must be <= "
            "MODELING_PERFORMANCE_BLEND_SEARCH['max_weight']."
        )
    for key in ("window_multipliers", "sharpness_multipliers"):
        if not MODELING_PERFORMANCE_BLEND_SEARCH[key]:
            raise ValueError(
                f"MODELING_PERFORMANCE_BLEND_SEARCH['{key}'] must not be empty."
            )
        if any(value <= 0.0 for value in MODELING_PERFORMANCE_BLEND_SEARCH[key]):
            raise ValueError(
                f"MODELING_PERFORMANCE_BLEND_SEARCH['{key}'] must contain only positive values."
            )
    if MODELING_PERFORMANCE_BLEND_SEARCH["bucket_size_minutes"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['bucket_size_minutes'] must be positive."
        )
    if MODELING_PERFORMANCE_BLEND_SEARCH["bucket_cycle_minutes"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['bucket_cycle_minutes'] must be positive."
        )
    if (
        MODELING_PERFORMANCE_BLEND_SEARCH["bucket_cycle_minutes"]
        < MODELING_PERFORMANCE_BLEND_SEARCH["bucket_size_minutes"]
    ):
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['bucket_cycle_minutes'] must be >= "
            "MODELING_PERFORMANCE_BLEND_SEARCH['bucket_size_minutes']."
        )
    if not MODELING_PERFORMANCE_BLEND_SEARCH["bucket_candidate_weights"]:
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['bucket_candidate_weights'] must not be empty."
        )
    if any(
        value < 0.0 or value > 1.0
        for value in MODELING_PERFORMANCE_BLEND_SEARCH["bucket_candidate_weights"]
    ):
        raise ValueError(
            "MODELING_PERFORMANCE_BLEND_SEARCH['bucket_candidate_weights'] must lie within [0,1]."
        )
    if MODELING_PERFORMANCE_HGB_SEARCH["screen_folds"] <= 0:
        raise ValueError("MODELING_PERFORMANCE_HGB_SEARCH['screen_folds'] must be positive.")
    if MODELING_PERFORMANCE_HGB_SEARCH["min_candidates"] <= 0:
        raise ValueError("MODELING_PERFORMANCE_HGB_SEARCH['min_candidates'] must be positive.")
    if (
        MODELING_PERFORMANCE_HGB_SEARCH["max_candidates"]
        < MODELING_PERFORMANCE_HGB_SEARCH["min_candidates"]
    ):
        raise ValueError(
            "MODELING_PERFORMANCE_HGB_SEARCH['max_candidates'] must be >= "
            "MODELING_PERFORMANCE_HGB_SEARCH['min_candidates']."
        )
    for key in ("learning_rates", "l2_regularization"):
        if not MODELING_PERFORMANCE_HGB_SEARCH[key]:
            raise ValueError(f"MODELING_PERFORMANCE_HGB_SEARCH['{key}'] must not be empty.")
    for key in ("max_depths", "min_samples_leaf", "max_iters"):
        if not MODELING_PERFORMANCE_HGB_SEARCH[key]:
            raise ValueError(f"MODELING_PERFORMANCE_HGB_SEARCH['{key}'] must not be empty.")
    if any(value <= 0.0 for value in MODELING_PERFORMANCE_HGB_SEARCH["learning_rates"]):
        raise ValueError(
            "MODELING_PERFORMANCE_HGB_SEARCH['learning_rates'] must contain only positive values."
        )
    if any(value < 0.0 for value in MODELING_PERFORMANCE_HGB_SEARCH["l2_regularization"]):
        raise ValueError(
            "MODELING_PERFORMANCE_HGB_SEARCH['l2_regularization'] must contain only non-negative values."
        )
    for key in ("max_depths", "min_samples_leaf", "max_iters"):
        if any(value <= 0 for value in MODELING_PERFORMANCE_HGB_SEARCH[key]):
            raise ValueError(
                f"MODELING_PERFORMANCE_HGB_SEARCH['{key}'] must contain only positive integers."
            )
    _required_profiles = {"short", "hourly", "day_ahead"}
    if not _required_profiles.issubset(set(MODELING_PERFORMANCE_QUICK_PROFILES)):
        raise ValueError(
            "MODELING_PERFORMANCE_QUICK_PROFILES must define at least "
            f"{_required_profiles}."
        )
    for profile_name, profile in MODELING_PERFORMANCE_QUICK_PROFILES.items():
        if not profile["feature_sets"]:
            raise ValueError(
                f"MODELING_PERFORMANCE_QUICK_PROFILES['{profile_name}']['feature_sets'] must not be empty."
            )
        if not profile["model_labels"]:
            raise ValueError(
                f"MODELING_PERFORMANCE_QUICK_PROFILES['{profile_name}']['model_labels'] must not be empty."
            )
        for key in ("n_folds", "val_window_days"):
            if profile[key] <= 0:
                raise ValueError(
                    f"MODELING_PERFORMANCE_QUICK_PROFILES['{profile_name}']['{key}'] must be positive."
                )
    if not MODELING_HORIZON_POLICIES:
        raise ValueError("MODELING_HORIZON_POLICIES must not be empty.")
    previous_max = 0
    for policy_name, policy in sorted(
        MODELING_HORIZON_POLICIES.items(),
        key=lambda item: item[1]["max_horizon_minutes"],
    ):
        if policy["max_horizon_minutes"] <= previous_max:
            raise ValueError(
                "MODELING_HORIZON_POLICIES must be strictly increasing by max_horizon_minutes."
            )
        previous_max = int(policy["max_horizon_minutes"])
        if not policy["feature_sets"]:
            raise ValueError(
                f"MODELING_HORIZON_POLICIES['{policy_name}']['feature_sets'] must not be empty."
            )
        if any(name not in FEATURE_SETS and name != "curated_ramp" for name in policy["feature_sets"]):
            raise ValueError(
                f"MODELING_HORIZON_POLICIES['{policy_name}']['feature_sets'] contains unknown values."
            )
        if not policy["model_labels"]:
            raise ValueError(
                f"MODELING_HORIZON_POLICIES['{policy_name}']['model_labels'] must not be empty."
            )
        if policy["rollout_residual_baseline"] not in {
            "avg_workday",
            "anchored_workday",
            "hybrid_workday",
            "persistence",
        }:
            raise ValueError(
                "MODELING_HORIZON_POLICIES rollout_residual_baseline must be one of "
                "{'avg_workday', 'anchored_workday', 'hybrid_workday', 'persistence'}."
            )
        if not policy["rollout_residual_candidates"]:
            raise ValueError(
                "MODELING_HORIZON_POLICIES rollout_residual_candidates must not be empty."
            )
        if any(
            baseline not in {"avg_workday", "anchored_workday", "hybrid_workday", "persistence"}
            for baseline in policy["rollout_residual_candidates"]
        ):
            raise ValueError(
                "MODELING_HORIZON_POLICIES rollout_residual_candidates must contain only "
                "{'avg_workday', 'anchored_workday', 'hybrid_workday', 'persistence'}."
            )
        if policy["rollout_origin_policy"] not in {
            "uniform",
            "midnight",
            "billing_aligned",
            "phase_balanced",
        }:
            raise ValueError(
                "MODELING_HORIZON_POLICIES rollout_origin_policy must be one of "
                "{'uniform', 'midnight', 'billing_aligned', 'phase_balanced'}."
            )
        if policy["rollout_selection_target"] not in {
            "endpoint_mae",
            "path_mae",
            "phase_mean_mae",
            "next_lock_mae",
            "profile_shape_mae",
        }:
            raise ValueError(
                "MODELING_HORIZON_POLICIES rollout_selection_target must be one of "
                "{'endpoint_mae', 'path_mae', 'phase_mean_mae', 'next_lock_mae', 'profile_shape_mae'}."
            )
    if not MODELING_PERFORMANCE_EVALUATION["segment_columns"]:
        raise ValueError("MODELING_PERFORMANCE_EVALUATION['segment_columns'] must not be empty.")
    if not MODELING_PERFORMANCE_EVALUATION["classical_benchmarks"]:
        raise ValueError("MODELING_PERFORMANCE_EVALUATION['classical_benchmarks'] must not be empty.")
    allowed_classical_benchmarks = {"holt_damped", "arima"}
    invalid_classical_benchmarks = sorted(
        set(MODELING_PERFORMANCE_EVALUATION["classical_benchmarks"]) - allowed_classical_benchmarks
    )
    if invalid_classical_benchmarks:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['classical_benchmarks'] contains unsupported "
            f"entries: {invalid_classical_benchmarks}"
        )
    if not MODELING_PERFORMANCE_EVALUATION["supplemental_surface_splits"]:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['supplemental_surface_splits'] must not be empty."
        )
    invalid_supplemental_splits = sorted(
        set(MODELING_PERFORMANCE_EVALUATION["supplemental_surface_splits"]) - set(SPLIT_DAY_RANGES)
    )
    if invalid_supplemental_splits:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['supplemental_surface_splits'] references unknown "
            f"splits: {invalid_supplemental_splits}"
        )
    for key in ("supplemental_load_band_quantile", "supplemental_ramp_band_quantile"):
        if not 0.0 < MODELING_PERFORMANCE_EVALUATION[key] < 1.0:
            raise ValueError(
                f"MODELING_PERFORMANCE_EVALUATION['{key}'] must be within (0,1)."
            )
    if MODELING_PERFORMANCE_EVALUATION["bootstrap_samples"] <= 0:
        raise ValueError("MODELING_PERFORMANCE_EVALUATION['bootstrap_samples'] must be positive.")
    if not 0.0 < MODELING_PERFORMANCE_EVALUATION["bootstrap_confidence_level"] < 1.0:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['bootstrap_confidence_level'] must be within (0,1)."
        )
    if MODELING_PERFORMANCE_EVALUATION["bootstrap_min_block_minutes"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['bootstrap_min_block_minutes'] must be positive."
        )
    if MODELING_PERFORMANCE_EVALUATION["bootstrap_max_block_minutes"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['bootstrap_max_block_minutes'] must be positive."
        )
    if (
        MODELING_PERFORMANCE_EVALUATION["bootstrap_min_block_minutes"]
        > MODELING_PERFORMANCE_EVALUATION["bootstrap_max_block_minutes"]
    ):
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['bootstrap_min_block_minutes'] must be <= "
            "MODELING_PERFORMANCE_EVALUATION['bootstrap_max_block_minutes']."
        )
    if MODELING_PERFORMANCE_EVALUATION["bootstrap_consecutive_insignificant"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['bootstrap_consecutive_insignificant'] must be positive."
        )
    if MODELING_PERFORMANCE_EVALUATION["importance_repeats"] <= 0:
        raise ValueError("MODELING_PERFORMANCE_EVALUATION['importance_repeats'] must be positive.")
    if MODELING_PERFORMANCE_EVALUATION["importance_max_features"] <= 0:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION['importance_max_features'] must be positive."
        )

    gold_columns = set(cast(list[str], SCHEMAS["gold"]["columns"]))
    for feature_set_name, columns in FEATURE_SETS.items():
        missing_columns = sorted(set(columns) - gold_columns)
        if missing_columns:
            raise ValueError(
                f"FEATURE_SETS['{feature_set_name}'] references unknown columns: {missing_columns}"
            )
        if TARGET_COLUMN in columns:
            raise ValueError(
                f"FEATURE_SETS['{feature_set_name}'] must not include target column "
                f"'{TARGET_COLUMN}' to prevent leakage."
            )

    invalid_distribution_features = sorted(
        set(EDA_CONFIG["distribution_features"]) - gold_columns
    )
    if invalid_distribution_features:
        raise ValueError(
            "EDA_CONFIG distribution_features references unknown columns: "
            f"{invalid_distribution_features}"
        )
    invalid_segment_columns = sorted(
        set(MODELING_PERFORMANCE_EVALUATION["segment_columns"]) - gold_columns
    )
    if invalid_segment_columns:
        raise ValueError(
            "MODELING_PERFORMANCE_EVALUATION segment_columns references unknown columns: "
            f"{invalid_segment_columns}"
        )

    if SECONDS_PER_DAY <= 0:
        raise ValueError(f"SECONDS_PER_DAY must be positive. Got: {SECONDS_PER_DAY}")
    if not MATLAB_REQUIRED_KEYS:
        raise ValueError("MATLAB_REQUIRED_KEYS must not be empty.")
    if RAW_MAX_NAN_PCT < 0.0 or RAW_MAX_NAN_PCT > 100.0:
        raise ValueError(
            "RAW_MAX_NAN_PCT must be within [0, 100]. "
            f"Got: {RAW_MAX_NAN_PCT}"
        )
    if RAW_MAX_OUT_OF_RANGE_PCT < 0.0 or RAW_MAX_OUT_OF_RANGE_PCT > 100.0:
        raise ValueError(
            "RAW_MAX_OUT_OF_RANGE_PCT must be within [0, 100]. "
            f"Got: {RAW_MAX_OUT_OF_RANGE_PCT}"
        )
    if SILVER_NAN_DROP_WARN_PCT < 0.0 or SILVER_NAN_DROP_WARN_PCT > 100.0:
        raise ValueError(
            "SILVER_NAN_DROP_WARN_PCT must be within [0, 100]. "
            f"Got: {SILVER_NAN_DROP_WARN_PCT}"
        )
    if SILVER_NAN_DROP_FAIL_PCT < 0.0 or SILVER_NAN_DROP_FAIL_PCT > 100.0:
        raise ValueError(
            "SILVER_NAN_DROP_FAIL_PCT must be within [0, 100]. "
            f"Got: {SILVER_NAN_DROP_FAIL_PCT}"
        )
    if SILVER_NAN_DROP_WARN_PCT > SILVER_NAN_DROP_FAIL_PCT:
        raise ValueError(
            "SILVER_NAN_DROP_WARN_PCT must be <= SILVER_NAN_DROP_FAIL_PCT. "
            f"Got warn={SILVER_NAN_DROP_WARN_PCT}, fail={SILVER_NAN_DROP_FAIL_PCT}"
        )
    if GOLD_MIN_RETENTION_PCT < 0.0 or GOLD_MIN_RETENTION_PCT > 100.0:
        raise ValueError(
            "GOLD_MIN_RETENTION_PCT must be within [0, 100]. "
            f"Got: {GOLD_MIN_RETENTION_PCT}"
        )
    if MODEL_MIN_SPLIT_ROWS <= 0:
        raise ValueError(
            f"MODEL_MIN_SPLIT_ROWS must be positive. Got: {MODEL_MIN_SPLIT_ROWS}"
        )

    _validate_split_ranges_contiguous(SPLIT_DAY_RANGES)


validate_config()
