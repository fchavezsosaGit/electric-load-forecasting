"""Shared pipeline configuration loaded from TOML files.

This module is the single source of truth for paths, supported resolutions,
feature settings, schemas, and model feature sets.
"""

from __future__ import annotations

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


_PIPELINE_TOML: Final[dict[str, object]] = _load_toml_file("pipeline.toml")
_EDA_TOML: Final[dict[str, object]] = _load_toml_file("eda.toml")

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

_features = cast(dict[str, list[int]], _PIPELINE_TOML["features"])
FEATURE_CONFIG: Final[dict[str, list[int]]] = {
    "lag_periods": list(_features["lag_periods"]),
    "rolling_periods": list(_features["rolling_periods"]),
    "slope_periods": list(_features["slope_periods"]),
}

_day_class = cast(dict[str, object], _PIPELINE_TOML["day_class"])
DAY_CLASS_MAP: Final[dict[str, int]] = {
    key: int(value) for key, value in cast(dict[str, int], _day_class["mapping"]).items()
}
VALID_DAY_CLASSES: Final[set[str]] = set(cast(list[str], _day_class["valid_classes"]))
EXPECTED_DAY_CLASSES: Final[set[str]] = set(VALID_DAY_CLASSES)

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
]

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
    "minimal": list(_feature_sets_toml["minimal"]["columns"]),
    "temporal": list(_feature_sets_toml["temporal"]["columns"]),
    "curated": list(_feature_sets_toml["curated"]["columns"]),
}
FEATURE_SETS["full"] = [
    col for col in SILVER_COLUMNS if col not in {"timestamp", "day_class", TARGET_COLUMN}
]

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


def _validate_positive_integer_list(name: str, values: list[int]) -> None:
    """Validate a non-empty list of strictly positive integers."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"FEATURE_CONFIG['{name}'] must be a non-empty list of integers.")
    if any((not isinstance(value, int)) or value <= 0 for value in values):
        raise ValueError(
            f"FEATURE_CONFIG['{name}'] must contain only positive integers. Got: {values}"
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


def validate_config() -> None:
    """Validate configuration consistency at runtime.

    This function is intentionally callable by tests and by the pipeline
    orchestrator during `--dry-run`.
    """
    for key in ("lag_periods", "rolling_periods", "slope_periods"):
        _validate_positive_integer_list(key, FEATURE_CONFIG[key])

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
