"""Configuration validation tests for schemas, splits, EDA settings, and TOML loading."""

from __future__ import annotations

import importlib.util
import os
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts.config import (
    DAY_CLASS_MAP,
    DEFAULT_RESOLUTIONS,
    EDA_CONFIG,
    EDA_DEFAULT_RESOLUTION_MODE,
    EDA_RESOLUTION_MODES,
    FEATURE_CONFIG,
    FEATURE_SETS,
    GOLD_MIN_RETENTION_PCT,
    MATLAB_REQUIRED_KEYS,
    MODEL_MIN_SPLIT_ROWS,
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
    resolve_eda_resolutions,
    resolve_resolution_suffix,
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


def test_config_paths_are_path_objects_and_have_valid_parents():
    """Ensure configured paths are Path objects with creatable parents."""
    for path in PATHS.values():
        assert isinstance(path, Path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert path.parent.exists()

    assert PATHS["outputs_modeling_dir"].name == "004_modeling"
    assert PATHS["outputs_performance_dir"].name == "005_performance"


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
    for key, values in FEATURE_CONFIG.items():
        assert isinstance(values, list)
        assert values
        assert all(isinstance(value, int) and value > 0 for value in values), (
            f"{key} must contain positive integers"
        )


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

    assert tuple(pipeline["resolutions"]["supported"]) == SUPPORTED_RESOLUTIONS
    assert tuple(pipeline["resolutions"]["defaults"]) == DEFAULT_RESOLUTIONS
    assert dict(pipeline["resolutions"]["aliases"]) == RESOLUTION_ALIASES
    assert dict(pipeline["resolutions"]["suffixes"]) == RESOLUTION_TO_SUFFIX

    assert pipeline["target"]["column"] == TARGET_COLUMN
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

    assert eda["analysis"]["histogram_bins"] == EDA_CONFIG["histogram_bins"]
    assert tuple(eda["visualization"]["figure_size"]) == EDA_CONFIG["figure_size"]


def test_loading_from_modified_toml_changes_runtime_values(tmp_path):
    """Ensure config is read from TOML rather than hardcoded values."""
    src_dir = Path("config")
    custom_dir = tmp_path / "config"
    custom_dir.mkdir(parents=True, exist_ok=True)
    (custom_dir / "pipeline.toml").write_text(
        (src_dir / "pipeline.toml").read_text(encoding="utf-8"), encoding="utf-8"
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

    with pytest.raises(FileNotFoundError, match="Missing required config file"):
        _load_config_module_from_dir(config_dir, "spec01_config_missing")
