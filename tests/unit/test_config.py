"""Configuration validation tests for schemas, splits, EDA settings, and TOML loading."""

from __future__ import annotations

import importlib.util
import os
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts.config import (
    DATASET,
    DAY_CLASS_MAP,
    DEFAULT_RESOLUTIONS,
    EDA_CONFIG,
    EDA_DEFAULT_RESOLUTION_MODE,
    EDA_RESOLUTION_MODES,
    FEATURE_CONFIG,
    FEATURE_SETS,
    FULL_STABLE_EXCLUDED_COLUMNS,
    FULL_STABLE_LEGACY_FEATURE_SET_NAME,
    FULL_STABLE_FEATURE_SET_NAME,
    GOLD_MIN_RETENTION_PCT,
    MATLAB_REQUIRED_KEYS,
    MODELING_PARALLEL,
    MODELING_PERFORMANCE_BLEND_SEARCH,
    MODELING_PERFORMANCE_EVALUATION,
    MODELING_PERFORMANCE_HGB_SEARCH,
    MODELING_PERFORMANCE_QUICK_PROFILES,
    MODELING_PERFORMANCE_RAMP,
    MODELING_HORIZON_POLICIES,
    MODELING_STAGE_PARALLEL,
    MODEL_MIN_SPLIT_ROWS,
    MULTIRES_BASELINES,
    MULTIRES_CONFIG,
    MULTIRES_FORECAST_CONTROL,
    MULTIRES_HORIZON_CURVE,
    MULTIRES_HYBRID,
    MULTIRES_PROFILES,
    MULTIRES_ROLLOUT,
    MULTIRES_ROLLOUT_CHALLENGERS,
    MULTIRES_ROLLOUT_LEARNED_BLENDS,
    MULTIRES_ROLLOUT_POLICY_CANDIDATES,
    MULTIRES_ROLLOUT_SWEEP_POLICIES,
    MULTIRES_RUNTIME,
    MULTIRES_SELECTION,
    PATHS,
    RAW_MAX_NAN_PCT,
    RAW_MAX_OUT_OF_RANGE_PCT,
    RESOLUTION_ALIASES,
    RESOLUTION_TO_SUFFIX,
    SECONDS_PER_DAY,
    SILVER_NAN_DROP_FAIL_PCT,
    SILVER_NAN_DROP_WARN_PCT,
    SPLIT_DAY_RANGES,
    SCHEMAS,
    SUPPORTED_RESOLUTIONS,
    TARGET_COLUMN,
    VALID_DAY_CLASSES,
    get_gold_path,
    get_silver_path,
    output_path_candidates,
    preferred_output_path,
    resolve_eda_resolutions,
    resolve_horizon_policy,
    resolve_performance_quick_profile,
    resolve_rollout_origin_policy,
    resolve_rollout_selection_target,
    resolve_resolution_suffix,
    scoped_output_path,
    validate_config,
)


def _load_config_module_from_dir(config_dir: Path, module_name: str):
    """Load scripts/config.py under a custom config directory."""
    script_path = Path("scripts/config.py").resolve()
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")

    old_value = os.environ.get("ELF_CONFIG_DIR")
    os.environ["ELF_CONFIG_DIR"] = str(config_dir)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old_value is None:
            os.environ.pop("ELF_CONFIG_DIR", None)
        else:
            os.environ["ELF_CONFIG_DIR"] = old_value
    return module


def test_pipeline_toml_exists_and_valid():
    """Ensure pipeline TOML exists and parses successfully."""
    path = Path("config/pipeline.toml")
    assert path.exists()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert "paths" in data
    assert "resolutions" in data
    assert "features" in data
    assert "dataset" in data
    assert "raw_contract" in data
    assert "quality_thresholds" in data


def test_eda_toml_exists_and_valid():
    """Ensure EDA TOML exists and parses successfully."""
    path = Path("config/eda.toml")
    assert path.exists()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert "visualization" in data
    assert "analysis" in data
    assert "resolution_selection" in data


def test_multires_toml_exists_and_valid():
    """Ensure multires TOML exists and parses successfully."""
    path = Path("config/multires.toml")
    assert path.exists()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert "paths" in data
    assert "multires" in data
    assert "rollout_challengers" in data["multires"]


def test_modeling_toml_exists_and_valid():
    """Ensure modeling TOML exists and parses successfully."""
    path = Path("config/modeling.toml")
    assert path.exists()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert "parallel" in data
    assert "performance" in data["parallel"]
    assert "multires" in data["parallel"]
    assert "performance" in data
    assert "ramp" in data["performance"]
    assert "blend_search" in data["performance"]
    assert "quick_profiles" in data["performance"]


def test_config_paths_are_path_objects_and_have_valid_parents():
    """Ensure configured paths are Path objects with creatable parents."""
    for path in PATHS.values():
        assert isinstance(path, Path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert path.parent.exists()

    assert PATHS["outputs_modeling_dir"].name == "004_modeling"
    assert PATHS["outputs_performance_dir"].name == "005_performance"
    assert PATHS["outputs_multires_dir"].name == "006_multires"
    assert PATHS["outputs_rollout_dir"].name == "007_rollout"
    assert PATHS["outputs_horizon_curve_dir"].name == "009_horizon_curve"
    assert PATHS["outputs_forecast_control_dir"].name == "010_forecast_control"


def test_dataset_scoped_output_helpers_prefer_namespaced_artifacts(tmp_path):
    """Ensure artifact helpers namespace stage outputs by load type with legacy fallback."""
    base_path = tmp_path / "outputs" / "005_performance"
    scoped_path = scoped_output_path(base_path)
    assert scoped_path == base_path / DATASET["artifact_namespace"]
    assert output_path_candidates(base_path) == (scoped_path, base_path)
    assert preferred_output_path(base_path) == scoped_path

    base_path.mkdir(parents=True, exist_ok=True)
    assert preferred_output_path(base_path) == base_path

    scoped_path.mkdir(parents=True, exist_ok=True)
    assert preferred_output_path(base_path) == scoped_path


def test_resolutions_are_valid_pandas_offsets():
    """Ensure supported and default resolutions map to valid positive timedeltas."""
    for resolution in SUPPORTED_RESOLUTIONS:
        delta = pd.to_timedelta(resolution)
        assert delta.total_seconds() > 0
    for resolution in DEFAULT_RESOLUTIONS:
        assert resolution in SUPPORTED_RESOLUTIONS


def test_schemas_define_non_empty_columns():
    """Ensure each schema defines at least one column."""
    for layer_name in ("bronze", "silver", "gold"):
        columns = SCHEMAS[layer_name]["columns"]
        assert isinstance(columns, list)
        assert len(columns) > 0


def test_feature_sets_do_not_include_target():
    """Ensure feature sets do not include the target column."""
    for name, columns in FEATURE_SETS.items():
        assert TARGET_COLUMN not in columns, f"{name} unexpectedly includes target column"


def test_resolution_aliases_resolve_to_supported_values():
    """Ensure alias resolutions resolve to supported canonical resolutions."""
    for alias, canonical in RESOLUTION_ALIASES.items():
        assert canonical in SUPPORTED_RESOLUTIONS
        assert pd.to_timedelta(alias).total_seconds() == pd.to_timedelta(canonical).total_seconds()


def test_feature_sets_are_subset_of_gold_schema():
    """Ensure every feature set column exists in the gold schema."""
    gold_columns = set(SCHEMAS["gold"]["columns"])
    for name, columns in FEATURE_SETS.items():
        missing = set(columns) - gold_columns
        assert not missing, f"{name} has unknown columns: {sorted(missing)}"


def test_feature_config_values_positive_integers():
    """Ensure feature period lists are strictly positive integers."""
    for key in (
        "lag_periods",
        "rolling_periods",
        "slope_periods",
        "lag_minutes",
        "rolling_minutes",
        "slope_minutes",
    ):
        values = FEATURE_CONFIG[key]
        assert isinstance(values, list)
        assert values
        assert all(isinstance(value, int) and value > 0 for value in values), (
            f"{key} must contain positive integers"
        )
    assert 0.0 < FEATURE_CONFIG["profile_activity_threshold"] < 1.0


def test_forecast_control_config_is_valid_and_supported():
    """Ensure Stage-10 forecast-control defaults are internally consistent."""
    assert MULTIRES_FORECAST_CONTROL["enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"] == 1440
    assert MULTIRES_FORECAST_CONTROL["hourly_horizon_minutes"] == 60
    assert MULTIRES_FORECAST_CONTROL["phase_horizon_minutes"] == 15
    assert MULTIRES_FORECAST_CONTROL["nowcast_horizon_minutes"] == 1
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_interval_minutes"] == 60
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_candidate_label"] == "auto"
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_lookback_minutes"] == 120
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_residual_drift_mae_pct_threshold"] >= 0.0
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_transition_mae_pct_threshold"] >= 0.0
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_activity_ratio_shift_threshold"] >= 0.0
    assert MULTIRES_FORECAST_CONTROL["actual_resolution"] in SUPPORTED_RESOLUTIONS
    assert MULTIRES_FORECAST_CONTROL["lock_interval_minutes"] == 15
    assert 0 <= MULTIRES_FORECAST_CONTROL["cycle_origin_hour"] <= 23
    assert 0 <= MULTIRES_FORECAST_CONTROL["cycle_origin_minute"] <= 59
    assert MULTIRES_FORECAST_CONTROL["cycle_origin_stride_minutes"] > 0
    assert MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["rolling_benchmark_origin_stride_minutes"] > 0
    assert MULTIRES_FORECAST_CONTROL["rolling_benchmark_max_cycles"] >= 0
    assert MULTIRES_FORECAST_CONTROL["rolling_benchmark_bootstrap_samples"] > 0
    assert 0.0 < MULTIRES_FORECAST_CONTROL["rolling_benchmark_confidence_level"] < 1.0
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_threshold_quantiles"]
    assert all(0.0 < value < 1.0 for value in MULTIRES_FORECAST_CONTROL["day_ahead_refresh_threshold_quantiles"])
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_trigger_rate"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["day_ahead_refresh_max_trigger_rate"] <= 1.0
    assert (
        MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_trigger_rate"]
        <= MULTIRES_FORECAST_CONTROL["day_ahead_refresh_max_trigger_rate"]
    )
    assert (
        0.0
        <= MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_profile_gain_fraction_vs_unconditional"]
        <= 1.0
    )
    assert (
        0.0
        <= MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_lock_gain_fraction_vs_unconditional"]
        <= 1.0
    )
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_trigger_mode"] in {
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
    assert MULTIRES_FORECAST_CONTROL["day_ahead_refresh_candidate_trigger_modes"]
    assert set(MULTIRES_FORECAST_CONTROL["calibration_splits"]).issubset(SPLIT_DAY_RANGES)
    assert set(MULTIRES_FORECAST_CONTROL["evaluation_splits"]).issubset(SPLIT_DAY_RANGES)
    assert MULTIRES_FORECAST_CONTROL["max_cycles"] >= 0
    assert MULTIRES_FORECAST_CONTROL["optimize_replayed_candidates"] is True
    assert MULTIRES_FORECAST_CONTROL["control_promotion_scope"] in {
        "calibration_only",
        "held_out_evaluation",
    }
    assert MULTIRES_FORECAST_CONTROL["allow_baseline_candidates"] is True
    assert MULTIRES_FORECAST_CONTROL["candidate_pool_size"] >= 1
    assert MULTIRES_FORECAST_CONTROL["candidate_benchmark_origin_cap"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_candidate_benchmark_origin_cap"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_candidate_evaluation_origin_cap"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_control_candidate_pool_size"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_control_prior_run_limit"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_control_min_prior_support_runs"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_control_max_supplemental_contexts_per_resolution"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_control_exploration_slots"] >= 0
    assert MULTIRES_FORECAST_CONTROL["phase_control_origin_cap"] >= 0
    assert MULTIRES_FORECAST_CONTROL["phase_stack_native_learned_top_candidates_per_pool"] >= 0
    assert MULTIRES_FORECAST_CONTROL["phase_stack_native_baseline_top_candidates_per_pool"] >= 0
    assert (
        MULTIRES_FORECAST_CONTROL["phase_stack_native_learned_top_candidates_per_pool"]
        + MULTIRES_FORECAST_CONTROL["phase_stack_native_baseline_top_candidates_per_pool"]
    ) >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_stack_blend_weights"]
    assert all(0.0 < value < 1.0 for value in MULTIRES_FORECAST_CONTROL["phase_stack_blend_weights"])
    assert MULTIRES_FORECAST_CONTROL["phase_stack_blend_parent_top_candidates"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_stack_bucket_policy_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["phase_stack_bucket_granularity_minutes"] > 0
    assert MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_top_candidates"] >= 1
    assert MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_weights"]
    assert all(
        0.0 < value < 1.0 for value in MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_weights"]
    )
    assert MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_bucket_blend_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_bucket_size_minutes"] > 0
    assert MULTIRES_FORECAST_CONTROL["benchmark_expanded_candidate_pool_size"] >= (
        MULTIRES_FORECAST_CONTROL["candidate_pool_size"]
    )
    assert MULTIRES_FORECAST_CONTROL["nowcast_candidate_pool_size"] >= 1
    assert MULTIRES_FORECAST_CONTROL["nowcast_control_blend_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_control_blend_weights"]
    assert all(0.0 <= value <= 1.0 for value in MULTIRES_FORECAST_CONTROL["nowcast_control_blend_weights"])
    assert MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_size_minutes"] > 0
    assert MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_weights"]
    assert all(
        0.0 <= value <= 1.0 for value in MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_weights"]
    )
    assert MULTIRES_FORECAST_CONTROL["nowcast_advisory_evidence_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_advisory_tie_tolerance"] >= 0.0
    assert MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"] is False
    assert MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_profile_active_threshold"] == pytest.approx(0.50)
    assert MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_high_ramp_fraction_threshold"] == pytest.approx(
        0.15
    )
    assert MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_next_lock"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_predicted_peak"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_shadow_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"]
    assert MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"]
    assert all(0.0 <= value <= 1.0 for value in MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"])
    assert all(0.0 <= value <= 1.0 for value in MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"])
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_max_next_lock_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_max_peak_hit_regress"] <= 1.0
    assert set(MULTIRES_FORECAST_CONTROL["benchmark_expanded_pool_layers"]).issubset(
        {"day_ahead", "hourly", "phase"}
    )
    assert set(MULTIRES_FORECAST_CONTROL["benchmark_full_origin_layers"]).issubset(
        {"day_ahead", "hourly", "phase"}
    )
    assert MULTIRES_FORECAST_CONTROL["day_ahead_selection_metric"] == "profile_shape_mae"
    assert MULTIRES_FORECAST_CONTROL["hourly_selection_metric"] == "optimizer_score"
    assert MULTIRES_FORECAST_CONTROL["phase_selection_metric"] == "optimizer_score"
    assert MULTIRES_FORECAST_CONTROL["phase_stack_selection_metric"] == "optimizer_score"
    assert MULTIRES_FORECAST_CONTROL["nowcast_selection_metric"] == "optimizer_score"
    assert MULTIRES_FORECAST_CONTROL["optimizer_selection_next_lock_weight"] == pytest.approx(0.40)
    assert MULTIRES_FORECAST_CONTROL["optimizer_selection_lock_weight"] == pytest.approx(0.15)
    assert MULTIRES_FORECAST_CONTROL["optimizer_selection_peak_value_weight"] == pytest.approx(0.20)
    assert MULTIRES_FORECAST_CONTROL["optimizer_selection_peak_miss_weight"] == pytest.approx(0.25)
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_min_lead_specific_samples"] == 8
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_min_samples"] == 8
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scaled_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scale_floor_quantile"] == pytest.approx(0.25)
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scale_floor_min_load"] == pytest.approx(250.0)
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_predicted_peak_min_samples"] == 6
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_predicted_peak_lead_min_samples"] == 4
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_full_support_n"] == 8
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_band_width_scale_pct"] == pytest.approx(100.0)
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_nowcast_cadence_minutes"] == 1
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_phase_cadence_minutes"] == 15
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_hourly_cadence_minutes"] == 60
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_day_ahead_cadence_minutes"] == 1440
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_nowcast_stale_threshold_minutes"] == 5
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_phase_stale_threshold_minutes"] == 30
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_hourly_stale_threshold_minutes"] == 90
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_day_ahead_stale_threshold_minutes"] == 1560
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_layer_multipliers"] == {
        "nowcast": pytest.approx(1.0),
        "phase": pytest.approx(0.9),
        "hourly": pytest.approx(0.8),
        "day_ahead": pytest.approx(0.7),
    }
    assert MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_quantile_source_multipliers"] == {
        "lead_interval": pytest.approx(1.0),
        "next_lock_scaled_global": pytest.approx(0.97),
        "predicted_peak_lead_interval": pytest.approx(0.98),
        "predicted_peak_global": pytest.approx(0.95),
        "next_lock_global": pytest.approx(0.92),
        "layer_global_fallback": pytest.approx(0.85),
        "layer_global": pytest.approx(0.85),
        "unavailable": pytest.approx(0.35),
    }
    assert MULTIRES_FORECAST_CONTROL["control_promotion_guard_enabled"] is True
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_next_lock_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_peak_value_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_peak_miss_regress"] <= 1.0
    assert MULTIRES_FORECAST_CONTROL["phase_stack_guard_enabled"] is True
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_lock_gain_pct"] <= 1.0
    assert isinstance(MULTIRES_FORECAST_CONTROL["phase_stack_guard_require_rolling_support"], bool)
    assert MULTIRES_FORECAST_CONTROL["phase_stack_guard_rolling_scope"] in {
        "rolling_calibration",
        "rolling_evaluation",
        "rolling_combined",
    }
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_rolling_lock_gain_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_next_lock_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_profile_degrade_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_peak_value_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_rolling_peak_hit_gain"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_optimizer_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_next_lock_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_profile_degrade_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_peak_value_regress_pct"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_peak_hit_gain"] <= 1.0
    assert 0.0 <= MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_optimizer_regress_pct"] <= 1.0
    assert MULTIRES_FORECAST_CONTROL["replay_cache_enabled"] is True
    assert MULTIRES_FORECAST_CONTROL["replay_cache_dirname"] == "replay_cache"


def test_quick_profiles_are_configured_for_each_horizon_class():
    """Ensure Stage-5 quick mode is driven by centralized horizon-aware profiles."""
    assert {"short", "hourly", "day_ahead"}.issubset(set(MODELING_PERFORMANCE_QUICK_PROFILES))
    assert "minimal_phase_anchor" in MODELING_PERFORMANCE_QUICK_PROFILES["short"]["feature_sets"]
    assert "full_stable" in MODELING_PERFORMANCE_QUICK_PROFILES["short"]["feature_sets"]
    assert FULL_STABLE_LEGACY_FEATURE_SET_NAME in MODELING_PERFORMANCE_QUICK_PROFILES["short"]["feature_sets"]
    assert "hgb-frontier-lr010-l2001" in MODELING_PERFORMANCE_QUICK_PROFILES["short"]["model_labels"]
    assert MODELING_PERFORMANCE_QUICK_PROFILES["short"]["n_folds"] == 2
    assert MODELING_PERFORMANCE_QUICK_PROFILES["short"]["val_window_days"] == 2


def test_resolve_performance_quick_profile_tracks_horizon_class():
    """Map short, hourly, and day-ahead horizons onto the matching quick profiles."""
    assert resolve_performance_quick_profile(1) == MODELING_PERFORMANCE_QUICK_PROFILES["short"]
    assert resolve_performance_quick_profile(60) == MODELING_PERFORMANCE_QUICK_PROFILES["hourly"]
    assert resolve_performance_quick_profile(1440) == MODELING_PERFORMANCE_QUICK_PROFILES["day_ahead"]


def test_fourier_cycle_config_is_complete_and_valid():
    """Ensure Fourier cycle specs are fully defined and internally consistent."""
    cycles = FEATURE_CONFIG["fourier_cycles"]
    assert cycles
    seen_prefixes: set[str] = set()
    for cycle in cycles:
        assert set(cycle) == {"source", "period", "prefix"}
        assert cycle["source"] in {"hour", "day_of_week"}
        assert isinstance(cycle["period"], int) and cycle["period"] > 0
        assert isinstance(cycle["prefix"], str) and cycle["prefix"]
        assert cycle["prefix"] not in seen_prefixes
        seen_prefixes.add(cycle["prefix"])


def test_feature_set_counts_include_fourier_columns():
    """Ensure active feature-set definitions reflect the Fourier expansion."""
    assert len(FEATURE_SETS["minimal"]) == 3
    assert len(FEATURE_SETS["minimal_phase"]) == 9
    assert len(FEATURE_SETS["minimal_phase_anchor"]) == 15
    assert len(FEATURE_SETS["temporal"]) == 14
    assert len(FEATURE_SETS["curated"]) == 15
    assert len(FEATURE_SETS["regime_profile"]) == 32
    assert len(FEATURE_SETS["full"]) == 86
    assert len(FEATURE_SETS[FULL_STABLE_FEATURE_SET_NAME]) == 78
    assert len(FEATURE_SETS[FULL_STABLE_LEGACY_FEATURE_SET_NAME]) == 37
    assert all(
        excluded not in FEATURE_SETS[FULL_STABLE_FEATURE_SET_NAME]
        for excluded in FULL_STABLE_EXCLUDED_COLUMNS
    )
    assert FEATURE_SETS[FULL_STABLE_LEGACY_FEATURE_SET_NAME][0] == "workday"
    assert FEATURE_SETS[FULL_STABLE_LEGACY_FEATURE_SET_NAME][-1] == "slope_60"
    for feature_set_name in (
        "temporal",
        "curated",
        "minimal_phase",
        "regime_profile",
        "full",
        FULL_STABLE_FEATURE_SET_NAME,
        FULL_STABLE_LEGACY_FEATURE_SET_NAME,
    ):
        for column_name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            if feature_set_name in {"minimal_phase", "regime_profile"}:
                continue
            assert column_name in FEATURE_SETS[feature_set_name]
    for column_name in (
        "phase_minute_15m",
        "phase_progress_15m",
        "phase_boundary_dist_15m",
        "phase_boundary_flag_15m",
        "phase_sin_15m",
        "phase_cos_15m",
    ):
        assert column_name in FEATURE_SETS["minimal_phase"]
        assert column_name in FEATURE_SETS["minimal_phase_anchor"]
        assert column_name in FEATURE_SETS["regime_profile"]
    assert "avg_workday_baseline" in FEATURE_SETS["regime_profile"]
    assert "avg_workday_baseline" in FEATURE_SETS["minimal_phase_anchor"]
    assert "anchored_workday_baseline" in FEATURE_SETS["minimal_phase_anchor"]
    assert "anchored_workday_baseline" in FEATURE_SETS["regime_profile"]
    assert "profile_activity_ratio" in FEATURE_SETS["regime_profile"]


def test_raw_and_quality_contract_values():
    """Ensure centralized operational constants are valid and typed."""
    assert isinstance(SECONDS_PER_DAY, int)
    assert SECONDS_PER_DAY > 0
    assert isinstance(MATLAB_REQUIRED_KEYS, tuple)
    assert MATLAB_REQUIRED_KEYS == ("P_data", "day_data", "day_class")
    assert isinstance(RAW_MAX_NAN_PCT, float)
    assert 0.0 <= RAW_MAX_NAN_PCT <= 100.0
    assert isinstance(RAW_MAX_OUT_OF_RANGE_PCT, float)
    assert 0.0 <= RAW_MAX_OUT_OF_RANGE_PCT <= 100.0
    assert isinstance(SILVER_NAN_DROP_WARN_PCT, float)
    assert 0.0 <= SILVER_NAN_DROP_WARN_PCT <= 100.0
    assert isinstance(SILVER_NAN_DROP_FAIL_PCT, float)
    assert SILVER_NAN_DROP_WARN_PCT <= SILVER_NAN_DROP_FAIL_PCT <= 100.0
    assert isinstance(GOLD_MIN_RETENTION_PCT, float)
    assert 0.0 <= GOLD_MIN_RETENTION_PCT <= 100.0
    assert isinstance(MODEL_MIN_SPLIT_ROWS, int)
    assert MODEL_MIN_SPLIT_ROWS > 0


def test_day_class_map_constraints():
    """Ensure day-class mappings are complete, unique, and integer encoded."""
    assert set(DAY_CLASS_MAP) == {"full", "half", "none"}
    assert set(DAY_CLASS_MAP) == VALID_DAY_CLASSES
    assert len(set(DAY_CLASS_MAP.values())) == len(DAY_CLASS_MAP)
    assert all(isinstance(value, int) for value in DAY_CLASS_MAP.values())


def test_split_day_ranges_are_contiguous_and_non_overlapping():
    """Ensure split day ranges fully cover days 1-31 without overlap."""
    all_days: set[int] = set()
    for _, (start_day, end_day) in SPLIT_DAY_RANGES.items():
        assert isinstance(start_day, int)
        assert isinstance(end_day, int)
        assert start_day <= end_day
        split_days = set(range(start_day, end_day + 1))
        assert not (all_days & split_days)
        all_days |= split_days

    assert min(all_days) == 1
    assert max(all_days) == 31
    assert all_days == set(range(1, 32))


def test_validate_config_callable():
    """Ensure runtime configuration validation executes without errors."""
    validate_config()


def test_validate_config_rejects_invalid_rolling_forecast_control_settings(monkeypatch):
    """Rolling benchmark settings should fail fast when they are outside the supported range."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "rolling_benchmark_origin_stride_minutes", 0)

    with pytest.raises(ValueError, match="rolling_benchmark_origin_stride_minutes"):
        validate_config()


def test_validate_config_rejects_invalid_phase_stack_blend_weights(monkeypatch):
    """Phase stack blend weights must stay inside the open interval (0,1)."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_stack_blend_weights", [0.0, 0.5])

    with pytest.raises(ValueError, match="phase_stack_blend_weights"):
        validate_config()


def test_validate_config_rejects_invalid_phase_stack_blend_parent_top_candidates(monkeypatch):
    """The phase-stack blend-parent cap must stay positive so Stage-10 keeps some blend search coverage."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_stack_blend_parent_top_candidates", 0)

    with pytest.raises(ValueError, match="phase_stack_blend_parent_top_candidates"):
        validate_config()


def test_validate_config_rejects_negative_phase_stack_native_learned_top_candidates(monkeypatch):
    """Native phase shortlist learned caps may be zero, but not negative."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_stack_native_learned_top_candidates_per_pool", -1)

    with pytest.raises(ValueError, match="phase_stack_native_learned_top_candidates_per_pool"):
        validate_config()


def test_validate_config_rejects_zero_native_phase_shortlist_capacity(monkeypatch):
    """Stage-10 needs at least one native phase candidate per replay pool to benchmark."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_stack_native_learned_top_candidates_per_pool", 0)
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_stack_native_baseline_top_candidates_per_pool", 0)

    with pytest.raises(ValueError, match="at least one native phase candidate per pool"):
        validate_config()


def test_validate_config_rejects_invalid_phase_control_candidate_pool_size(monkeypatch):
    """Phase replay pool size must stay positive."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_control_candidate_pool_size", 0)

    with pytest.raises(ValueError, match="phase_control_candidate_pool_size"):
        validate_config()


def test_validate_config_rejects_invalid_phase_control_prior_run_limit(monkeypatch):
    """Phase replay prior evidence windows must stay positive."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_control_prior_run_limit", 0)

    with pytest.raises(ValueError, match="phase_control_prior_run_limit"):
        validate_config()


def test_validate_config_rejects_invalid_phase_control_min_prior_support_runs(monkeypatch):
    """Phase replay prior support thresholds must stay positive."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_control_min_prior_support_runs", 0)

    with pytest.raises(ValueError, match="phase_control_min_prior_support_runs"):
        validate_config()


def test_validate_config_rejects_invalid_phase_control_context_cap(monkeypatch):
    """Phase replay diversity caps must stay positive."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_control_max_supplemental_contexts_per_resolution", 0)

    with pytest.raises(ValueError, match="phase_control_max_supplemental_contexts_per_resolution"):
        validate_config()


def test_validate_config_rejects_negative_phase_control_exploration_slots(monkeypatch):
    """Phase replay exploration slots may be zero but not negative."""
    monkeypatch.setitem(MULTIRES_FORECAST_CONTROL, "phase_control_exploration_slots", -1)

    with pytest.raises(ValueError, match="phase_control_exploration_slots"):
        validate_config()


def test_validate_config_rejects_invalid_supplemental_surface_split(monkeypatch):
    """Supplemental Stage-5 surfaces must reference known split names."""
    monkeypatch.setitem(MODELING_PERFORMANCE_EVALUATION, "supplemental_surface_splits", ["validate", "mystery"])

    with pytest.raises(ValueError, match="supplemental_surface_splits"):
        validate_config()


def test_validate_config_rejects_missing_optimizer_confidence_multiplier(monkeypatch):
    """Optimizer confidence policy should fail fast when required multiplier keys are missing."""
    monkeypatch.setitem(
        MULTIRES_FORECAST_CONTROL,
        "optimizer_delivery_confidence_layer_multipliers",
        {"nowcast": 1.0, "phase": 0.9, "hourly": 0.8},
    )

    with pytest.raises(ValueError, match="optimizer_delivery_confidence_layer_multipliers"):
        validate_config()


def test_validate_config_rejects_invalid_phase_stack_baseline_control_blend_weights(monkeypatch):
    """Phase baseline-control stack weights should stay inside the open unit interval."""
    monkeypatch.setitem(
        MULTIRES_FORECAST_CONTROL,
        "phase_stack_baseline_control_blend_weights",
        [0.0, 0.5],
    )

    with pytest.raises(ValueError, match="phase_stack_baseline_control_blend_weights"):
        validate_config()


def test_eda_config_contains_expected_keys_and_types():
    """Ensure EDA_CONFIG includes all centralized notebook parameters."""
    expected_key_types = {
        "zscore_threshold": float,
        "histogram_bins": int,
        "figure_size": tuple,
        "figure_size_wide": tuple,
        "figure_size_compact": tuple,
        "figure_size_grid": tuple,
        "figure_size_correlation": tuple,
        "day_class_colors": dict,
        "seaborn_style": str,
        "physical_load_max_watts": float,
        "physical_load_min_watts": float,
        "correlation_high_threshold": float,
        "top_correlations_count": int,
        "zero_run_threshold_seconds": int,
        "overlay_resample_frequency": str,
        "legend_max_labels": int,
        "percentiles": list,
        "distribution_features": list,
    }

    assert set(expected_key_types).issubset(EDA_CONFIG)
    for key, expected_type in expected_key_types.items():
        assert isinstance(EDA_CONFIG[key], expected_type), f"{key} has unexpected type"


def test_eda_resolution_constants():
    """Ensure resolution mode constants are valid and consistent."""
    assert set(EDA_RESOLUTION_MODES) == {"all", "default", "custom"}
    assert EDA_DEFAULT_RESOLUTION_MODE in EDA_RESOLUTION_MODES


def test_multires_config_exports_are_valid():
    """Ensure multires defaults and contracts are typed and internally coherent."""
    assert MULTIRES_CONFIG["mode"] in {"smoke", "candidate", "full"}
    assert MULTIRES_CONFIG["comparison_mode"] in {"native_step", "matched_horizon"}
    assert set(MULTIRES_CONFIG["resolutions"]).issubset(set(SUPPORTED_RESOLUTIONS))
    assert {"1s", "5s", "10s", "30s"}.issubset(set(MULTIRES_CONFIG["resolutions"]))
    assert MULTIRES_CONFIG["horizons_minutes"]
    assert all(isinstance(value, int) and value > 0 for value in MULTIRES_CONFIG["horizons_minutes"])
    assert MULTIRES_CONFIG["matched_strategies"] == ["recursive", "direct_endpoint"]
    assert 0.0 <= MULTIRES_SELECTION["min_eval_coverage"] <= 1.0
    assert MULTIRES_SELECTION["max_fold_std_mae_ratio"] == 0.20
    assert MULTIRES_SELECTION["max_candidate_runtime_minutes"] > 0
    assert MULTIRES_BASELINES["include_persistence"] is True
    assert {"smoke", "candidate", "full"}.issubset(set(MULTIRES_PROFILES))
    for profile in MULTIRES_PROFILES.values():
        assert profile["n_folds"] > 0
        assert profile["val_window_days"] > 0
        assert profile["origins_per_fold"] > 0
        assert profile["resolutions"]
        assert set(profile["resolutions"]).issubset(set(SUPPORTED_RESOLUTIONS))
        assert profile["horizons_minutes"]
        assert all(value > 0 for value in profile["horizons_minutes"])
        assert set(profile["feature_sets"]).issubset(set(FEATURE_SETS))
        assert profile["model_labels"]
    assert MULTIRES_PROFILES["smoke"]["feature_sets"] == ["minimal", "curated"]
    assert MULTIRES_PROFILES["candidate"]["feature_sets"] == [
        "minimal",
        "curated",
        FULL_STABLE_FEATURE_SET_NAME,
    ]
    assert MULTIRES_PROFILES["full"]["feature_sets"] == [
        "minimal",
        "curated",
        FULL_STABLE_FEATURE_SET_NAME,
    ]
    assert MULTIRES_PROFILES["focus_60m"]["horizons_minutes"] == [60]
    assert MULTIRES_PROFILES["focus_60m"]["feature_sets"] == [
        "minimal",
        "curated",
        FULL_STABLE_FEATURE_SET_NAME,
    ]
    assert MULTIRES_PROFILES["focus_60m"]["resolutions"] == ["30s", "1min", "5min", "10min"]
    assert MULTIRES_PROFILES["focus_60m"]["origins_per_fold"] == 4
    assert "hgb-frontier-lr010-l2001" in MULTIRES_PROFILES["candidate"]["model_labels"]
    assert "hgb-frontier-lr010-leaf100" in MULTIRES_PROFILES["candidate"]["model_labels"]
    assert "hgb-frontier-lr010-depth5-leaf100-l2001" in MULTIRES_PROFILES["full"]["model_labels"]
    assert MULTIRES_BASELINES["include_anchored_workday"] is True
    assert MULTIRES_BASELINES["include_hybrid_workday"] is True
    assert MULTIRES_HYBRID["curve"] == "linear"
    assert MULTIRES_HYBRID["persistence_weight_start"] == 1.0
    assert MULTIRES_HYBRID["persistence_weight_end"] == 0.35
    assert MULTIRES_RUNTIME["smoke_origins_per_fold"] > 0
    assert MULTIRES_RUNTIME["candidate_origins_per_fold"] > 0
    assert MULTIRES_RUNTIME["full_origins_per_fold"] > 0
    assert MULTIRES_ROLLOUT["selected_resolution"] in SUPPORTED_RESOLUTIONS
    assert MULTIRES_ROLLOUT["feature_set"] in FEATURE_SETS
    assert MULTIRES_ROLLOUT["selected_resolution"] == "10min"
    assert MULTIRES_ROLLOUT["feature_set"] == "minimal"
    assert MULTIRES_ROLLOUT["model_label"] == "hgb-balanced"
    assert MULTIRES_ROLLOUT["horizon_minutes"] > 0
    assert MULTIRES_ROLLOUT["origin_policy"] == "auto"
    assert MULTIRES_ROLLOUT["selection_target"] == "auto"
    assert MULTIRES_ROLLOUT_CHALLENGERS["enabled"] is True
    assert MULTIRES_ROLLOUT_CHALLENGERS["max_candidates"] == 8
    assert MULTIRES_ROLLOUT_CHALLENGERS["parallel_workers"] == 2
    assert MULTIRES_ROLLOUT_CHALLENGERS["include_rollout_registry"] is True
    assert MULTIRES_ROLLOUT_CHALLENGERS["include_stage6_registry"] is True
    assert MULTIRES_ROLLOUT_CHALLENGERS["include_horizon_policy_candidates"] is True
    assert MULTIRES_ROLLOUT_CHALLENGERS["include_config_default"] is True
    assert MULTIRES_ROLLOUT_CHALLENGERS["origin_policies"] == [
        "phase_balanced",
        "uniform",
        "billing_aligned",
        "midnight",
    ]
    assert MULTIRES_ROLLOUT_CHALLENGERS["policy_resolutions"] == ["1min", "5min", "10min"]
    assert MULTIRES_ROLLOUT_CHALLENGERS["recommendation_origin_scope"] == "requested_only"
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["enabled"] is True
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["include_persistence_to_raw"] is True
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["include_persistence_to_residual"] is True
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["include_raw_to_residual"] is True
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["include_hybrid_phase_gate"] is True
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_aligned_weight"] == 0.15
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_non_aligned_weight"] == 0.75
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"] == {
        0: 0.15,
        300: 0.90,
        600: 0.60,
    }
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_enabled"] is True
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_step"] == 0.05
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_neighbors"] == 1
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["max_weights_per_family"] == 6
    assert MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"] == "linear"
    assert MULTIRES_ROLLOUT_POLICY_CANDIDATES["enabled"] is True
    assert MULTIRES_ROLLOUT_POLICY_CANDIDATES["max_horizon_minutes"] == 15
    assert MULTIRES_ROLLOUT_POLICY_CANDIDATES["selection_targets"] == [
        "endpoint_mae",
        "path_mae",
        "phase_mean_mae",
        "next_lock_mae",
    ]
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["enabled"] is True
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["min_horizon_minutes"] == 30
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["max_horizon_minutes"] == 120
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["origin_policies"] == ["phase_balanced"]
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["selection_targets"] == [
        "next_lock_mae",
        "path_mae",
        "profile_shape_mae",
    ]
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["min_source_candidates"] == 2
    assert MULTIRES_ROLLOUT_SWEEP_POLICIES["max_source_candidates"] == 3
    assert MULTIRES_HORIZON_CURVE["enabled"] is True
    assert MULTIRES_HORIZON_CURVE["horizons_minutes"] == [1, 15, 30, 60, 120, 240, 360, 720, 1440]
    assert MULTIRES_HORIZON_CURVE["origins_per_run"] == 8
    assert MULTIRES_HORIZON_CURVE["origin_policy"] == "auto"
    assert MULTIRES_HORIZON_CURVE["selection_target"] == "auto"
    assert MULTIRES_HORIZON_CURVE["max_candidates"] == 8
    assert MULTIRES_HORIZON_CURVE["include_stage5_anchor"] is True
    assert MULTIRES_HORIZON_CURVE["reuse_existing_sweeps"] is True


def test_modeling_parallel_exports_are_valid():
    """Ensure shared modeling parallel settings are typed and internally coherent."""
    assert MODELING_PARALLEL["backend"] in {"threading", "loky", "sequential"}
    assert MODELING_PARALLEL["max_workers"] > 0
    assert MODELING_PARALLEL["batch_size"] > 0
    assert MODELING_PARALLEL["min_tasks"] > 0
    assert MODELING_PARALLEL["inner_threads_per_worker"] > 0
    assert MODELING_PARALLEL["pre_dispatch"]
    assert set(MODELING_STAGE_PARALLEL) == {
        "performance",
        "multires",
        "rollout_sweep",
        "forecast_control",
    }
    assert isinstance(MODELING_STAGE_PARALLEL["performance"]["enabled"], bool)
    assert isinstance(MODELING_STAGE_PARALLEL["multires"]["enabled"], bool)
    assert isinstance(MODELING_STAGE_PARALLEL["rollout_sweep"]["enabled"], bool)
    assert isinstance(MODELING_STAGE_PARALLEL["forecast_control"]["enabled"], bool)


def test_modeling_performance_search_exports_are_valid():
    """Ensure Stage-5 centralized search defaults are exported and validated."""
    assert 0.0 < MODELING_PERFORMANCE_RAMP["quantile"] < 1.0
    assert MODELING_PERFORMANCE_BLEND_SEARCH["enabled"] is True
    assert MODELING_PERFORMANCE_BLEND_SEARCH["base_window"] > 0
    assert MODELING_PERFORMANCE_BLEND_SEARCH["base_sharpness"] > 0.0
    assert 0.0 <= MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"] <= 1.0
    assert 0.0 <= MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"] <= 1.0
    assert MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"] <= MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"]
    assert MODELING_PERFORMANCE_BLEND_SEARCH["window_multipliers"]
    assert MODELING_PERFORMANCE_BLEND_SEARCH["sharpness_multipliers"]
    assert MODELING_PERFORMANCE_HGB_SEARCH["enabled"] is True
    assert MODELING_PERFORMANCE_HGB_SEARCH["screen_folds"] > 0
    assert MODELING_PERFORMANCE_HGB_SEARCH["min_candidates"] > 0
    assert MODELING_PERFORMANCE_HGB_SEARCH["max_candidates"] >= MODELING_PERFORMANCE_HGB_SEARCH["min_candidates"]
    assert MODELING_HORIZON_POLICIES
    assert "short" in MODELING_HORIZON_POLICIES
    assert "day_ahead" in MODELING_HORIZON_POLICIES
    assert "minimal_phase_anchor" in MODELING_HORIZON_POLICIES["short"]["feature_sets"]
    assert FULL_STABLE_LEGACY_FEATURE_SET_NAME in MODELING_HORIZON_POLICIES["short"]["feature_sets"]
    assert MODELING_HORIZON_POLICIES["short"]["rollout_residual_candidates"] == [
        "persistence",
        "anchored_workday",
        "avg_workday",
    ]
    assert MODELING_HORIZON_POLICIES["hourly"]["rollout_residual_candidates"] == ["avg_workday"]
    assert MODELING_HORIZON_POLICIES["day_ahead"]["feature_sets"] == [
        "minimal",
        "regime_profile",
        FULL_STABLE_FEATURE_SET_NAME,
    ]
    assert MODELING_HORIZON_POLICIES["day_ahead"]["rollout_residual_candidates"] == [
        "avg_workday",
        "anchored_workday",
        "hybrid_workday",
    ]
    assert MODELING_HORIZON_POLICIES["day_ahead"]["allow_residual"] is True
    assert MODELING_PERFORMANCE_EVALUATION["segment_columns"]
    assert "holt_damped" in MODELING_PERFORMANCE_EVALUATION["classical_benchmarks"]
    assert "arima" in MODELING_PERFORMANCE_EVALUATION["classical_benchmarks"]
    assert MODELING_PERFORMANCE_EVALUATION["supplemental_surface_splits"] == ["validate", "test"]
    assert MODELING_PERFORMANCE_EVALUATION["supplemental_load_band_quantile"] == pytest.approx(0.90)
    assert MODELING_PERFORMANCE_EVALUATION["supplemental_ramp_band_quantile"] == pytest.approx(0.90)
    assert MODELING_PERFORMANCE_EVALUATION["bootstrap_samples"] > 0
    assert 0.0 < MODELING_PERFORMANCE_EVALUATION["bootstrap_confidence_level"] < 1.0
    assert (
        MODELING_PERFORMANCE_EVALUATION["bootstrap_min_block_minutes"]
        <= MODELING_PERFORMANCE_EVALUATION["bootstrap_max_block_minutes"]
    )
    assert MODELING_PERFORMANCE_EVALUATION["importance_repeats"] > 0
    assert MODELING_PERFORMANCE_EVALUATION["importance_max_features"] > 0


def test_resolve_horizon_policy_uses_smallest_covering_bucket():
    """Ensure horizon policies resolve deterministically by requested forecast horizon."""
    assert resolve_horizon_policy(5)["feature_sets"] == MODELING_HORIZON_POLICIES["short"]["feature_sets"]
    assert resolve_horizon_policy(5)["rollout_residual_baseline"] == "persistence"
    assert resolve_horizon_policy(5)["rollout_origin_policy"] == "phase_balanced"
    assert resolve_horizon_policy(5)["rollout_selection_target"] == "next_lock_mae"
    assert resolve_horizon_policy(60)["feature_sets"] == MODELING_HORIZON_POLICIES["hourly"]["feature_sets"]
    assert resolve_horizon_policy(60)["rollout_residual_baseline"] == "avg_workday"
    assert resolve_horizon_policy(60)["rollout_origin_policy"] == "phase_balanced"
    assert resolve_horizon_policy(60)["rollout_selection_target"] == "next_lock_mae"
    assert resolve_horizon_policy(1440)["feature_sets"] == MODELING_HORIZON_POLICIES["day_ahead"]["feature_sets"]
    assert resolve_horizon_policy(1440)["rollout_origin_policy"] == "uniform"
    assert resolve_horizon_policy(1440)["rollout_selection_target"] == "profile_shape_mae"


def test_resolve_rollout_policy_helpers_follow_horizon_policy():
    """Keep auto rollout policy helpers aligned with the centralized horizon policy."""
    assert resolve_rollout_origin_policy(15, "auto") == "phase_balanced"
    assert resolve_rollout_origin_policy(120, "auto") == "phase_balanced"
    assert resolve_rollout_selection_target(15, "auto") == "next_lock_mae"
    assert resolve_rollout_selection_target(240, "auto") == "profile_shape_mae"
    assert resolve_rollout_origin_policy(15, "uniform") == "uniform"
    assert resolve_rollout_selection_target(15, "endpoint_mae") == "endpoint_mae"


def test_resolve_eda_resolutions_all_default_and_custom():
    """Ensure resolution mode resolver returns expected canonical values."""
    assert resolve_eda_resolutions("all") == list(SUPPORTED_RESOLUTIONS)
    assert resolve_eda_resolutions("default") == list(DEFAULT_RESOLUTIONS)
    assert resolve_eda_resolutions("custom", ["5min", "15min"]) == ["5min", "15min"]
    assert resolve_eda_resolutions("custom", ["60s"]) == ["1min"]


def test_resolve_eda_resolutions_invalid_inputs_raise():
    """Ensure invalid mode and custom resolutions raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported EDA resolution mode"):
        resolve_eda_resolutions("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="custom_list must be provided"):
        resolve_eda_resolutions("custom", None)

    with pytest.raises(ValueError, match="Unsupported resolution"):
        resolve_eda_resolutions("custom", ["2min"])


def test_resolution_suffix_and_paths_helpers():
    """Ensure suffix and path helpers resolve defaults correctly."""
    for resolution in DEFAULT_RESOLUTIONS:
        suffix = resolve_resolution_suffix(resolution)
        assert get_silver_path(resolution) == PATHS["silver_dir"] / f"power_load_{suffix}.parquet"
        assert get_gold_path(resolution) == PATHS["gold_dir"] / f"power_load_{suffix}_all_features.parquet"

    assert resolve_resolution_suffix("60s") == "1m"
    with pytest.raises(ValueError, match="Unsupported resolution"):
        resolve_resolution_suffix("2min")


def test_toml_round_trip_consistency_with_module_exports():
    """Ensure TOML values match exported runtime constants."""
    with Path("config/pipeline.toml").open("rb") as handle:
        pipeline = tomllib.load(handle)
    with Path("config/eda.toml").open("rb") as handle:
        eda = tomllib.load(handle)
    with Path("config/multires.toml").open("rb") as handle:
        multires = tomllib.load(handle)
    with Path("config/modeling.toml").open("rb") as handle:
        modeling = tomllib.load(handle)

    assert tuple(pipeline["resolutions"]["supported"]) == SUPPORTED_RESOLUTIONS
    assert tuple(pipeline["resolutions"]["defaults"]) == DEFAULT_RESOLUTIONS
    assert dict(pipeline["resolutions"]["aliases"]) == RESOLUTION_ALIASES
    assert dict(pipeline["resolutions"]["suffixes"]) == RESOLUTION_TO_SUFFIX
    assert pipeline["dataset"]["load_type"] == DATASET["load_type"]

    assert pipeline["target"]["column"] == TARGET_COLUMN
    assert list(pipeline["features"]["fourier_cycles"]) == FEATURE_CONFIG["fourier_cycles"]
    assert tuple(pipeline["splits"]["train"]) == SPLIT_DAY_RANGES["train"]
    assert tuple(pipeline["splits"]["validate"]) == SPLIT_DAY_RANGES["validate"]
    assert tuple(pipeline["splits"]["test"]) == SPLIT_DAY_RANGES["test"]
    assert pipeline["raw_contract"]["seconds_per_day"] == SECONDS_PER_DAY
    assert tuple(pipeline["raw_contract"]["required_keys"]) == MATLAB_REQUIRED_KEYS
    assert pipeline["quality_thresholds"]["raw_max_nan_pct"] == RAW_MAX_NAN_PCT
    assert pipeline["quality_thresholds"]["raw_max_out_of_range_pct"] == RAW_MAX_OUT_OF_RANGE_PCT
    assert pipeline["quality_thresholds"]["silver_nan_drop_warn_pct"] == SILVER_NAN_DROP_WARN_PCT
    assert pipeline["quality_thresholds"]["silver_nan_drop_fail_pct"] == SILVER_NAN_DROP_FAIL_PCT
    assert pipeline["quality_thresholds"]["gold_min_retention_pct"] == GOLD_MIN_RETENTION_PCT
    assert pipeline["quality_thresholds"]["model_min_split_rows"] == MODEL_MIN_SPLIT_ROWS
    assert Path(pipeline["paths"]["outputs_modeling_dir"]).name == PATHS["outputs_modeling_dir"].name
    assert Path(pipeline["paths"]["outputs_performance_dir"]).name == PATHS["outputs_performance_dir"].name
    assert Path(multires["paths"]["outputs_horizon_curve_dir"]).name == PATHS["outputs_horizon_curve_dir"].name

    assert eda["analysis"]["histogram_bins"] == EDA_CONFIG["histogram_bins"]
    assert tuple(eda["visualization"]["figure_size"]) == EDA_CONFIG["figure_size"]
    assert multires["multires"]["mode"] == MULTIRES_CONFIG["mode"]
    assert multires["multires"]["comparison_mode"] == MULTIRES_CONFIG["comparison_mode"]
    assert list(multires["multires"]["resolutions"]) == MULTIRES_CONFIG["resolutions"]
    assert list(multires["multires"]["horizons_minutes"]) == MULTIRES_CONFIG["horizons_minutes"]
    assert multires["multires"]["baselines"]["include_persistence"] == MULTIRES_BASELINES["include_persistence"]
    assert list(multires["multires"]["profiles"]["smoke"]["resolutions"]) == MULTIRES_PROFILES["smoke"]["resolutions"]
    assert multires["multires"]["profiles"]["smoke"]["origins_per_fold"] == MULTIRES_PROFILES["smoke"]["origins_per_fold"]
    assert list(multires["multires"]["profiles"]["candidate"]["resolutions"]) == MULTIRES_PROFILES["candidate"]["resolutions"]
    assert multires["multires"]["profiles"]["candidate"]["origins_per_fold"] == MULTIRES_PROFILES["candidate"]["origins_per_fold"]
    assert list(multires["multires"]["profiles"]["full"]["resolutions"]) == MULTIRES_PROFILES["full"]["resolutions"]
    assert multires["multires"]["profiles"]["full"]["origins_per_fold"] == MULTIRES_PROFILES["full"]["origins_per_fold"]
    assert list(multires["multires"]["profiles"]["focus_60m"]["resolutions"]) == MULTIRES_PROFILES["focus_60m"]["resolutions"]
    assert multires["multires"]["profiles"]["focus_60m"]["origins_per_fold"] == MULTIRES_PROFILES["focus_60m"]["origins_per_fold"]
    assert multires["multires"]["baselines"]["include_anchored_workday"] == MULTIRES_BASELINES["include_anchored_workday"]
    assert multires["multires"]["baselines"]["include_hybrid_workday"] == MULTIRES_BASELINES["include_hybrid_workday"]
    assert multires["multires"]["hybrid"]["curve"] == MULTIRES_HYBRID["curve"]
    assert multires["multires"]["rollout"]["selected_resolution"] == MULTIRES_ROLLOUT["selected_resolution"]
    assert multires["multires"]["rollout"]["origin_policy"] == MULTIRES_ROLLOUT["origin_policy"]
    assert multires["multires"]["rollout"]["selection_target"] == MULTIRES_ROLLOUT["selection_target"]
    assert (
        multires["multires"]["rollout_challengers"]["max_candidates"]
        == MULTIRES_ROLLOUT_CHALLENGERS["max_candidates"]
    )
    assert (
        multires["multires"]["rollout_challengers"]["parallel_workers"]
        == MULTIRES_ROLLOUT_CHALLENGERS["parallel_workers"]
    )
    assert (
        multires["multires"]["rollout_challengers"]["recommendation_origin_scope"]
        == MULTIRES_ROLLOUT_CHALLENGERS["recommendation_origin_scope"]
    )
    assert list(multires["multires"]["rollout_challengers"]["policy_resolutions"]) == MULTIRES_ROLLOUT_CHALLENGERS["policy_resolutions"]
    assert multires["multires"]["rollout_learned_blends"]["curve"] == MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"]
    assert (
        multires["multires"]["rollout_learned_blends"]["include_hybrid_phase_gate"]
        == MULTIRES_ROLLOUT_LEARNED_BLENDS["include_hybrid_phase_gate"]
    )
    assert (
        multires["multires"]["rollout_learned_blends"]["hybrid_phase_gate_aligned_weight"]
        == MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_aligned_weight"]
    )
    assert (
        multires["multires"]["rollout_learned_blends"]["hybrid_phase_gate_non_aligned_weight"]
        == MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_non_aligned_weight"]
    )
    assert dict(multires["multires"]["rollout_learned_blends"]["hybrid_phase_gate_bucket_weights"]) == {
        str(key): value
        for key, value in MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"].items()
    }
    assert (
        multires["multires"]["rollout_policy_candidates"]["enabled"]
        == MULTIRES_ROLLOUT_POLICY_CANDIDATES["enabled"]
    )
    assert (
        multires["multires"]["rollout_policy_candidates"]["max_horizon_minutes"]
        == MULTIRES_ROLLOUT_POLICY_CANDIDATES["max_horizon_minutes"]
    )
    assert list(multires["multires"]["rollout_policy_candidates"]["selection_targets"]) == (
        MULTIRES_ROLLOUT_POLICY_CANDIDATES["selection_targets"]
    )
    assert (
        multires["multires"]["rollout_sweep_policies"]["enabled"]
        == MULTIRES_ROLLOUT_SWEEP_POLICIES["enabled"]
    )
    assert (
        multires["multires"]["rollout_sweep_policies"]["min_horizon_minutes"]
        == MULTIRES_ROLLOUT_SWEEP_POLICIES["min_horizon_minutes"]
    )
    assert (
        multires["multires"]["rollout_sweep_policies"]["max_horizon_minutes"]
        == MULTIRES_ROLLOUT_SWEEP_POLICIES["max_horizon_minutes"]
    )
    assert list(multires["multires"]["rollout_sweep_policies"]["origin_policies"]) == (
        MULTIRES_ROLLOUT_SWEEP_POLICIES["origin_policies"]
    )
    assert list(multires["multires"]["rollout_sweep_policies"]["selection_targets"]) == (
        MULTIRES_ROLLOUT_SWEEP_POLICIES["selection_targets"]
    )
    assert (
        multires["multires"]["rollout_learned_blends"]["refinement_enabled"]
        == MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_enabled"]
    )
    assert (
        multires["multires"]["rollout_learned_blends"]["refinement_step"]
        == MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_step"]
    )
    assert list(multires["multires"]["horizon_curve"]["horizons_minutes"]) == MULTIRES_HORIZON_CURVE["horizons_minutes"]
    assert multires["multires"]["horizon_curve"]["origin_policy"] == MULTIRES_HORIZON_CURVE["origin_policy"]
    assert multires["multires"]["horizon_curve"]["selection_target"] == MULTIRES_HORIZON_CURVE["selection_target"]
    assert multires["multires"]["horizon_curve"]["max_candidates"] == MULTIRES_HORIZON_CURVE["max_candidates"]
    assert (
        multires["multires"]["horizon_curve"]["reuse_existing_sweeps"]
        == MULTIRES_HORIZON_CURVE["reuse_existing_sweeps"]
    )
    assert modeling["parallel"]["backend"] == MODELING_PARALLEL["backend"]
    assert modeling["parallel"]["max_workers"] == MODELING_PARALLEL["max_workers"]
    assert modeling["parallel"]["performance"]["enabled"] == MODELING_STAGE_PARALLEL["performance"]["enabled"]
    assert modeling["parallel"]["performance"]["max_workers"] == MODELING_STAGE_PARALLEL["performance"]["max_workers"]
    assert (
        modeling["parallel"]["performance"]["inner_threads_per_worker"]
        == MODELING_STAGE_PARALLEL["performance"]["inner_threads_per_worker"]
    )
    assert (
        modeling["parallel"]["performance"]["high_capacity_host_only"]
        == MODELING_STAGE_PARALLEL["performance"]["high_capacity_host_only"]
    )
    assert modeling["parallel"]["multires"]["enabled"] == MODELING_STAGE_PARALLEL["multires"]["enabled"]
    assert modeling["parallel"]["multires"]["max_workers"] == MODELING_STAGE_PARALLEL["multires"]["max_workers"]
    assert (
        modeling["parallel"]["multires"]["inner_threads_per_worker"]
        == MODELING_STAGE_PARALLEL["multires"]["inner_threads_per_worker"]
    )
    assert (
        modeling["parallel"]["multires"]["high_capacity_host_only"]
        == MODELING_STAGE_PARALLEL["multires"]["high_capacity_host_only"]
    )
    assert (
        modeling["parallel"]["rollout_sweep"]["enabled"]
        == MODELING_STAGE_PARALLEL["rollout_sweep"]["enabled"]
    )
    assert (
        modeling["parallel"]["rollout_sweep"]["max_workers"]
        == MODELING_STAGE_PARALLEL["rollout_sweep"]["max_workers"]
    )
    assert (
        modeling["parallel"]["rollout_sweep"]["inner_threads_per_worker"]
        == MODELING_STAGE_PARALLEL["rollout_sweep"]["inner_threads_per_worker"]
    )
    assert (
        modeling["parallel"]["rollout_sweep"]["high_capacity_host_only"]
        == MODELING_STAGE_PARALLEL["rollout_sweep"]["high_capacity_host_only"]
    )
    assert (
        modeling["parallel"]["forecast_control"]["enabled"]
        == MODELING_STAGE_PARALLEL["forecast_control"]["enabled"]
    )
    assert (
        modeling["parallel"]["forecast_control"]["max_workers"]
        == MODELING_STAGE_PARALLEL["forecast_control"]["max_workers"]
    )
    assert (
        modeling["parallel"]["forecast_control"]["inner_threads_per_worker"]
        == MODELING_STAGE_PARALLEL["forecast_control"]["inner_threads_per_worker"]
    )
    assert (
        modeling["parallel"]["forecast_control"]["high_capacity_host_only"]
        == MODELING_STAGE_PARALLEL["forecast_control"]["high_capacity_host_only"]
    )
    assert modeling["performance"]["ramp"]["quantile"] == MODELING_PERFORMANCE_RAMP["quantile"]
    assert (
        modeling["performance"]["blend_search"]["base_window"]
        == MODELING_PERFORMANCE_BLEND_SEARCH["base_window"]
    )


def test_loading_from_modified_toml_changes_runtime_values(tmp_path):
    """Ensure config is read from TOML rather than hardcoded values."""
    src_dir = Path("config")
    custom_dir = tmp_path / "config"
    custom_dir.mkdir(parents=True, exist_ok=True)
    (custom_dir / "pipeline.toml").write_text(
        (src_dir / "pipeline.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (custom_dir / "multires.toml").write_text(
        (src_dir / "multires.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (custom_dir / "modeling.toml").write_text(
        (src_dir / "modeling.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    eda_text = (src_dir / "eda.toml").read_text(encoding="utf-8")
    eda_text = eda_text.replace("histogram_bins = 50", "histogram_bins = 77")
    (custom_dir / "eda.toml").write_text(eda_text, encoding="utf-8")

    module = _load_config_module_from_dir(custom_dir, "spec01_config_modified")
    assert module.EDA_CONFIG["histogram_bins"] == 77


def test_missing_toml_file_raises_informative_error(tmp_path):
    """Ensure missing TOML files raise FileNotFoundError with expected path context."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.toml").write_text(
        Path("config/pipeline.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (config_dir / "multires.toml").write_text(
        Path("config/multires.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Missing required config file"):
        _load_config_module_from_dir(config_dir, "spec01_config_missing")
