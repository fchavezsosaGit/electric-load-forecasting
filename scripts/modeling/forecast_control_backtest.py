"""Stage-10 forecast-control backtest for 24h profile plus intraday correction layers.

This stage replays the currently selected day-ahead, hourly, and phase-level
policies on shared control cycles. It exists to answer the operational question
"does the stacked forecast-update policy reduce the next locked-interval error,
or are we only winning isolated offline benchmarks?"
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import logging
import math
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    DATASET,
    FEATURE_CONFIG,
    MODELING_STAGE_PARALLEL,
    MULTIRES_FORECAST_CONTROL,
    MULTIRES_ROLLOUT,
    MODELING_PERFORMANCE_BLEND_SEARCH,
    MODELING_PERFORMANCE_RAMP,
    PATHS,
    SPLIT_DAY_RANGES,
    preferred_output_path,
    resolve_rollout_origin_policy,
    resolve_rollout_selection_target,
    scoped_output_path,
    validate_config,
)
from modeling.common import (  # noqa: E402
    FigureGuideEntry,
    ModelSpec,
    _build_hgb_spec,
    _build_xgb_spec,
    build_model_catalog,
    lead_steps_for_horizon,
    stable_config_hash,
    update_latest_alias,
    validate_png_artifact,
    write_figure_guide,
)
from modeling.model_performance import (  # noqa: E402
    BlendConfig as Stage5BlendConfig,
    BucketBlendConfig as Stage5BucketBlendConfig,
    _apply_blend_policy as _apply_stage5_blend_policy,
    _apply_bucket_blend_policy as _apply_stage5_bucket_blend_policy,
    _augment_with_curated_ramp_features as _augment_stage5_curated_ramp_features,
    _build_feature_sets as _build_stage5_feature_sets,
    _fit_and_align as _fit_stage5_candidate_and_align,
    _load_gold_with_full_grid as _load_stage5_gold_with_full_grid,
    _resolve_feature_set_columns as _resolve_stage5_feature_set_columns,
    _stage5_base_target_mode,
    _stage5_blend_base_policy_kind,
    _stage5_blend_policy_kind,
    read_stage5_holdout_registry,
)
from modeling.multires import build_causal_feature_frame, load_base_gold  # noqa: E402
from modeling.parallel import resolve_parallel_plan  # noqa: E402
from modeling.recursive_rollout import (  # noqa: E402
    _aggregate_rollout_metrics,
    resolve_rollout_selection_context,
    run_rollout_evaluation,
)
from modeling.runtime import runtime_summary  # noqa: E402
from utils import emit_quality_gate, optimal_acf_depth  # noqa: E402

logger = logging.getLogger(__name__)
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUTPUT_ROOT = scoped_output_path(PATHS["outputs_forecast_control_dir"])


def _start_runtime_step() -> tuple[str, float]:
    """Capture one runtime step start point in UTC and monotonic time."""
    return datetime.now(UTC).isoformat(), perf_counter()


def _append_runtime_step(
    runtime_records: list[dict[str, Any]],
    *,
    step: str,
    category: str,
    started_at_utc: str,
    started_perf: float,
    **metadata: Any,
) -> None:
    """Append one completed runtime step record."""
    payload: dict[str, Any] = {
        "step": str(step),
        "category": str(category),
        "started_at_utc": str(started_at_utc),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": round(perf_counter() - float(started_perf), 6),
    }
    for key, value in metadata.items():
        if value is None:
            continue
        payload[str(key)] = value
    runtime_records.append(payload)


def _build_runtime_profile_summary(
    runtime_profile: pd.DataFrame,
    *,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Summarize one persisted Stage-10 runtime profile."""
    if runtime_profile.empty:
        return {
            "step_count": 0,
            "wall_clock_seconds": float(wall_clock_seconds),
            "profiled_seconds": 0.0,
            "longest_step": "",
            "longest_step_seconds": 0.0,
            "replay_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "artifacts_seconds": 0.0,
            "setup_seconds": 0.0,
        }
    runtime_df = runtime_profile.copy()
    runtime_df["duration_seconds"] = pd.to_numeric(runtime_df["duration_seconds"], errors="coerce").fillna(0.0)
    longest_row = runtime_df.sort_values(
        ["duration_seconds", "step"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]

    def _category_seconds(category: str) -> float:
        return float(
            runtime_df.loc[
                runtime_df.get("category", pd.Series(dtype="string")).astype("string").eq(str(category)),
                "duration_seconds",
            ].sum()
        )

    return {
        "step_count": int(len(runtime_df)),
        "wall_clock_seconds": float(wall_clock_seconds),
        "profiled_seconds": float(runtime_df["duration_seconds"].sum()),
        "longest_step": str(longest_row.get("step", "")),
        "longest_step_seconds": float(longest_row.get("duration_seconds", 0.0)),
        "replay_seconds": _category_seconds("replay"),
        "evaluation_seconds": _category_seconds("evaluation"),
        "artifacts_seconds": _category_seconds("artifacts"),
        "setup_seconds": _category_seconds("setup"),
    }
_REPLAY_CACHE_LOCK = threading.Lock()
_LOCAL_CACHE_LOCK = threading.Lock()
_STAGE5_NOWCAST_CONTEXTS_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_STAGE5_PREDICTION_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_REPRESENTABLE_ORIGIN_LOOKUP_CACHE: dict[tuple[str, int], set[pd.Timestamp]] = {}
OPTIMIZER_CONTRACT_VERSION = "1.2"
OPTIMIZER_POLICY_VERSION = "1.1"
OPTIMIZER_LAYER_PRIORITY = ["nowcast", "phase", "hourly", "day_ahead"]


def _configure_logging() -> None:
    """Initialize a basic logger for direct execution of the control backtest."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _optimizer_layer_confidence_multiplier_map() -> dict[str, float]:
    """Return the configured trust weighting for each stacked delivery layer."""
    configured = cast(
        dict[str, float],
        MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_layer_multipliers"],
    )
    return {str(key): float(value) for key, value in configured.items()}


def _optimizer_quantile_source_confidence_multiplier_map() -> dict[str, float]:
    """Return the configured trust weighting for each uncertainty-band source."""
    configured = cast(
        dict[str, float],
        MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_quantile_source_multipliers"],
    )
    return {str(key): float(value) for key, value in configured.items()}


def _optimizer_dynamic_soft_overlay_thresholds() -> dict[str, float]:
    """Return the non-regression thresholds for soft minute-overlay shadow promotion."""
    return {
        "max_next_lock_regress_pct": float(
            MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_max_next_lock_regress_pct"]
        ),
        "max_peak_hit_regress": float(MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_max_peak_hit_regress"]),
    }


def _optimizer_dynamic_soft_overlay_grid() -> list[tuple[float, float]]:
    """Return the configured soft-overlay candidate grid as (supported, background) weights."""
    supported_weights = [
        float(value) for value in cast(list[float], MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"])
    ]
    background_weights = [
        float(value)
        for value in cast(list[float], MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"])
    ]
    unique_pairs: set[tuple[float, float]] = {(1.0, 1.0)}
    for supported_weight in supported_weights:
        for background_weight in background_weights:
            if float(supported_weight) < float(background_weight):
                continue
            unique_pairs.add((round(float(supported_weight), 6), round(float(background_weight), 6)))
    return sorted(unique_pairs, key=lambda pair: (-pair[0], -pair[1]))


def _relative_artifact_path(path: Path) -> str:
    """Render an artifact path relative to the repository root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    """Read a CSV artifact when present, otherwise return an empty frame."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _safe_pct(abs_error_sum: float, actual_abs_sum: float) -> float:
    """Convert absolute error totals into a scale-normalized percentage."""
    if not np.isfinite(abs_error_sum) or not np.isfinite(actual_abs_sum) or actual_abs_sum <= 0.0:
        return float("nan")
    return 100.0 * float(abs_error_sum) / float(actual_abs_sum)


def _optimizer_selection_weights() -> dict[str, float]:
    """Return the configured weights used by the optimizer-facing composite score."""
    return {
        "next_lock": float(MULTIRES_FORECAST_CONTROL["optimizer_selection_next_lock_weight"]),
        "lock": float(MULTIRES_FORECAST_CONTROL["optimizer_selection_lock_weight"]),
        "peak_value": float(MULTIRES_FORECAST_CONTROL["optimizer_selection_peak_value_weight"]),
        "peak_miss": float(MULTIRES_FORECAST_CONTROL["optimizer_selection_peak_miss_weight"]),
    }


def _optimizer_promotion_guard_thresholds() -> dict[str, float]:
    """Return the non-regression tolerances that challengers must clear versus the upstream choice."""
    return {
        "next_lock_regress_pct": float(
            MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_next_lock_regress_pct"]
        ),
        "peak_value_regress_pct": float(
            MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_peak_value_regress_pct"]
        ),
        "peak_miss_regress": float(MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_peak_miss_regress"]),
    }


def _optimizer_layer_cadence_minutes_map() -> dict[str, int]:
    """Return the configured delivery cadence for each stacked layer."""
    return {
        "nowcast": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_nowcast_cadence_minutes"]),
        "phase": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_phase_cadence_minutes"]),
        "hourly": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_hourly_cadence_minutes"]),
        "day_ahead": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_day_ahead_cadence_minutes"]),
    }


def _optimizer_layer_stale_threshold_minutes_map() -> dict[str, int]:
    """Return the configured freshness threshold for each stacked layer."""
    return {
        "nowcast": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_nowcast_stale_threshold_minutes"]),
        "phase": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_phase_stale_threshold_minutes"]),
        "hourly": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_hourly_stale_threshold_minutes"]),
        "day_ahead": int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_day_ahead_stale_threshold_minutes"]),
    }


def _optimizer_metric_value(
    row: pd.Series | dict[str, Any],
    metric_name: str,
    *,
    prefix: str = "",
) -> float:
    """Read one numeric optimizer-facing metric from a row, respecting an optional prefix."""
    return float(row.get(f"{prefix}{metric_name}", float("nan")))


def _apply_optimizer_promotion_guard(
    frame: pd.DataFrame,
    *,
    upstream_label: str,
    metric_prefix: str = "",
) -> pd.DataFrame:
    """Filter challenger rows so promotion remains next-lock/peak-safe against the upstream choice."""
    if frame.empty or not bool(MULTIRES_FORECAST_CONTROL["control_promotion_guard_enabled"]):
        return frame.copy()
    if not str(upstream_label):
        return frame.copy()
    working = frame.copy()
    baseline = working.loc[working["candidate_label"].astype("string").eq(str(upstream_label))].copy()
    if baseline.empty:
        return working
    baseline_row = baseline.iloc[0]
    thresholds = _optimizer_promotion_guard_thresholds()
    baseline_next_lock = _optimizer_metric_value(baseline_row, "next_lock_mae", prefix=metric_prefix)
    baseline_peak_value = _optimizer_metric_value(baseline_row, "peak_value_mae", prefix=metric_prefix)
    baseline_peak_miss = _optimizer_metric_value(
        baseline_row,
        "peak_interval_miss_rate",
        prefix=metric_prefix,
    )

    next_lock_values = (
        pd.to_numeric(working[f"{metric_prefix}next_lock_mae"], errors="coerce")
        if f"{metric_prefix}next_lock_mae" in working.columns
        else pd.Series(float("nan"), index=working.index, dtype=float)
    )
    peak_value_values = (
        pd.to_numeric(working[f"{metric_prefix}peak_value_mae"], errors="coerce")
        if f"{metric_prefix}peak_value_mae" in working.columns
        else pd.Series(float("nan"), index=working.index, dtype=float)
    )
    peak_miss_values = (
        pd.to_numeric(working[f"{metric_prefix}peak_interval_miss_rate"], errors="coerce")
        if f"{metric_prefix}peak_interval_miss_rate" in working.columns
        else pd.Series(float("nan"), index=working.index, dtype=float)
    )
    working["promotion_guard_meets_next_lock_rule"] = True
    if np.isfinite(baseline_next_lock):
        if baseline_next_lock > 0.0:
            next_lock_regress_pct = (
                np.maximum(next_lock_values - baseline_next_lock, 0.0) / float(baseline_next_lock)
            )
        else:
            next_lock_regress_pct = np.maximum(next_lock_values - baseline_next_lock, 0.0)
        working["promotion_guard_meets_next_lock_rule"] = next_lock_regress_pct.le(
            float(thresholds["next_lock_regress_pct"])
        ) | next_lock_values.isna()
    working["promotion_guard_meets_peak_value_rule"] = True
    if np.isfinite(baseline_peak_value):
        if baseline_peak_value > 0.0:
            peak_value_regress_pct = (
                np.maximum(peak_value_values - baseline_peak_value, 0.0) / float(baseline_peak_value)
            )
        else:
            peak_value_regress_pct = np.maximum(peak_value_values - baseline_peak_value, 0.0)
        working["promotion_guard_meets_peak_value_rule"] = peak_value_regress_pct.le(
            float(thresholds["peak_value_regress_pct"])
        ) | peak_value_values.isna()
    working["promotion_guard_meets_peak_miss_rule"] = True
    if np.isfinite(baseline_peak_miss):
        peak_miss_regress = peak_miss_values - float(baseline_peak_miss)
        working["promotion_guard_meets_peak_miss_rule"] = peak_miss_regress.le(
            float(thresholds["peak_miss_regress"])
        ) | peak_miss_values.isna()
    eligible_mask = (
        working["promotion_guard_meets_next_lock_rule"].astype(bool)
        & working["promotion_guard_meets_peak_value_rule"].astype(bool)
        & working["promotion_guard_meets_peak_miss_rule"].astype(bool)
    )
    eligible = working.loc[eligible_mask].copy()
    if not eligible.empty:
        return eligible
    return baseline.copy()


def _optimizer_score_from_components(
    *,
    next_lock_mae_pct: float,
    lock_mae_pct: float,
    peak_value_mae_pct: float,
    peak_interval_miss_rate: float,
) -> float:
    """Collapse optimizer-relevant error signals into one lower-is-better composite."""
    weights = _optimizer_selection_weights()
    score = 0.0
    component_seen = False
    component_values = {
        "next_lock": float(next_lock_mae_pct),
        "lock": float(lock_mae_pct),
        "peak_value": float(peak_value_mae_pct),
        "peak_miss": 100.0 * float(peak_interval_miss_rate),
    }
    for component_name, value in component_values.items():
        if not np.isfinite(value):
            continue
        component_seen = True
        score += float(weights[component_name]) * float(value)
    return float(score) if component_seen else float("nan")


def _optimizer_score_from_row(row: pd.Series | dict[str, Any]) -> float:
    """Recover optimizer_score from a summary-like row, recomputing it when needed."""
    optimizer_score = float(row.get("optimizer_score", float("nan")))
    if np.isfinite(optimizer_score):
        return float(optimizer_score)
    return _optimizer_score_from_components(
        next_lock_mae_pct=float(row.get("next_lock_mae_pct", float("nan"))),
        lock_mae_pct=float(row.get("lock_mae_pct", float("nan"))),
        peak_value_mae_pct=float(row.get("peak_value_mae_pct", float("nan"))),
        peak_interval_miss_rate=float(row.get("peak_interval_miss_rate", float("nan"))),
    )


def _phase_stack_selection_metric() -> str:
    """Return the configured selection metric for stack-applied phase candidates."""
    return str(MULTIRES_FORECAST_CONTROL["phase_stack_selection_metric"])


def _sortable_metric_value(value: Any) -> float:
    """Normalize non-finite metric values so tuple sorting keeps them at the end."""
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("inf")


def _phase_stack_metric_tuple_from_row(
    row: pd.Series | dict[str, Any],
    *,
    selection_metric: str,
    tie_breaker: float = float("nan"),
) -> tuple[float, ...]:
    """Build a deterministic lower-is-better comparison tuple for phase-stack candidates."""
    optimizer_score = _optimizer_score_from_row(row)
    next_lock_mae = _sortable_metric_value(row.get("next_lock_mae", float("nan")))
    peak_miss = _sortable_metric_value(row.get("peak_interval_miss_rate", float("nan")))
    peak_value_mae = _sortable_metric_value(row.get("peak_value_mae", float("nan")))
    lock_mae = _sortable_metric_value(row.get("lock_mae", float("nan")))
    profile_shape_mae = _sortable_metric_value(row.get("profile_shape_mae", float("nan")))
    minute_path_mae = _sortable_metric_value(row.get("minute_path_mae", float("nan")))
    if selection_metric == "optimizer_score":
        return (
            _sortable_metric_value(optimizer_score),
            next_lock_mae,
            peak_miss,
            peak_value_mae,
            lock_mae,
            profile_shape_mae,
            minute_path_mae,
            _sortable_metric_value(tie_breaker),
        )
    if selection_metric == "next_lock_mae":
        return (
            next_lock_mae,
            peak_miss,
            peak_value_mae,
            lock_mae,
            profile_shape_mae,
            minute_path_mae,
            _sortable_metric_value(tie_breaker),
        )
    if selection_metric == "peak_value_mae":
        return (
            peak_value_mae,
            peak_miss,
            next_lock_mae,
            lock_mae,
            profile_shape_mae,
            minute_path_mae,
            _sortable_metric_value(tie_breaker),
        )
    if selection_metric == "peak_interval_miss_rate":
        return (
            peak_miss,
            next_lock_mae,
            peak_value_mae,
            lock_mae,
            profile_shape_mae,
            minute_path_mae,
            _sortable_metric_value(tie_breaker),
        )
    if selection_metric == "profile_shape_mae":
        return (
            profile_shape_mae,
            lock_mae,
            next_lock_mae,
            minute_path_mae,
            peak_miss,
            peak_value_mae,
            _sortable_metric_value(tie_breaker),
        )
    if selection_metric == "minute_path_mae":
        return (
            minute_path_mae,
            lock_mae,
            profile_shape_mae,
            next_lock_mae,
            peak_miss,
            peak_value_mae,
            _sortable_metric_value(tie_breaker),
        )
    return (
        lock_mae,
        next_lock_mae,
        peak_miss,
        peak_value_mae,
        profile_shape_mae,
        minute_path_mae,
        _sortable_metric_value(tie_breaker),
    )


def _phase_stack_metric_sort_columns(metric_name: str) -> list[str]:
    """Return the deterministic benchmark ordering for stack-applied phase candidates."""
    primary = [f"{metric_name}_p50", f"{metric_name}_p90", metric_name]
    if metric_name == "optimizer_score":
        return primary + [
            "next_lock_mae_p50",
            "peak_interval_miss_rate_p50",
            "peak_value_mae_p50",
            "lock_mae_p50",
            "profile_shape_mae_p50",
            "candidate_label",
        ]
    if metric_name == "next_lock_mae":
        return primary + [
            "peak_interval_miss_rate_p50",
            "peak_value_mae_p50",
            "lock_mae_p50",
            "profile_shape_mae_p50",
            "candidate_label",
        ]
    if metric_name == "peak_value_mae":
        return primary + [
            "peak_interval_miss_rate_p50",
            "next_lock_mae_p50",
            "lock_mae_p50",
            "profile_shape_mae_p50",
            "candidate_label",
        ]
    if metric_name == "peak_interval_miss_rate":
        return primary + [
            "next_lock_mae_p50",
            "peak_value_mae_p50",
            "lock_mae_p50",
            "profile_shape_mae_p50",
            "candidate_label",
        ]
    if metric_name == "profile_shape_mae":
        return primary + ["lock_mae_p50", "next_lock_mae_p50", "minute_path_mae_p50", "candidate_label"]
    if metric_name == "minute_path_mae":
        return primary + ["lock_mae_p50", "profile_shape_mae_p50", "next_lock_mae_p50", "candidate_label"]
    return primary + [
        "next_lock_mae_p50",
        "peak_interval_miss_rate_p50",
        "peak_value_mae_p50",
        "profile_shape_mae_p50",
        "candidate_label",
    ]


def _timestamp_minute_bucket(
    timestamp: pd.Timestamp | str,
    *,
    bucket_minutes: int,
    cycle_minutes: int,
) -> int:
    """Map one timestamp onto a fixed-size minute bucket within the requested cycle."""
    minute_value = int(pd.Timestamp(timestamp).minute) % int(cycle_minutes)
    return int((minute_value // int(bucket_minutes)) * int(bucket_minutes))


def _index_minute_buckets(
    index: pd.Index | pd.DatetimeIndex,
    *,
    bucket_minutes: int,
    cycle_minutes: int,
) -> pd.Series:
    """Vectorize minute-bucket assignment for one timestamp index."""
    timestamp_index = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    minute_values = timestamp_index.minute.to_numpy(dtype=int) % int(cycle_minutes)
    bucket_values = (minute_values // int(bucket_minutes)) * int(bucket_minutes)
    return pd.Series(bucket_values.astype(int), index=timestamp_index, dtype="int64")


def _profile_shape_mae(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    """Measure shape error after rescaling the prediction to the actual energy."""
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    actual_sum = float(np.sum(actual_values))
    predicted_sum = float(np.sum(predicted_values))
    if not np.isfinite(predicted_sum) or abs(predicted_sum) <= 1e-9:
        return float("nan"), float("nan")
    scaled = predicted_values * (actual_sum / predicted_sum)
    abs_errors = np.abs(actual_values - scaled)
    mae = float(np.mean(abs_errors))
    pct = _safe_pct(float(np.sum(abs_errors)), float(np.sum(np.abs(actual_values))))
    return mae, pct


def _layer_metrics(actual_minute: pd.Series, predicted_minute: pd.Series) -> dict[str, float]:
    """Compute minute-path, profile-shape, and energy metrics for one layer."""
    aligned = pd.DataFrame({"actual": actual_minute, "predicted": predicted_minute}).dropna()
    if aligned.empty:
        return {
            "minute_path_mae": float("nan"),
            "minute_path_mae_pct": float("nan"),
            "profile_shape_mae": float("nan"),
            "profile_shape_mae_pct": float("nan"),
            "energy_mae": float("nan"),
            "energy_mae_pct": float("nan"),
        }
    errors = aligned["actual"] - aligned["predicted"]
    minute_abs_error_sum = float(np.sum(np.abs(errors.to_numpy(dtype=float))))
    actual_abs_sum = float(np.sum(np.abs(aligned["actual"].to_numpy(dtype=float))))
    profile_mae, profile_pct = _profile_shape_mae(aligned["actual"], aligned["predicted"])
    actual_total = float(aligned["actual"].sum())
    predicted_total = float(aligned["predicted"].sum())
    energy_mae = float(abs(actual_total - predicted_total))
    return {
        "minute_path_mae": float(np.mean(np.abs(errors.to_numpy(dtype=float)))),
        "minute_path_mae_pct": _safe_pct(minute_abs_error_sum, actual_abs_sum),
        "profile_shape_mae": profile_mae,
        "profile_shape_mae_pct": profile_pct,
        "energy_mae": energy_mae,
        "energy_mae_pct": _safe_pct(energy_mae, abs(actual_total)),
    }


def _control_cycle_block_length(values: pd.Series) -> int:
    """Infer a stable moving-block length for cycle-level bootstrap comparisons."""
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = int(clean.shape[0])
    if n_obs <= 1:
        return 1
    max_depth = max(1, min(n_obs // 2, int(math.ceil(math.sqrt(n_obs)))))
    depth = optimal_acf_depth(
        clean.to_numpy(dtype=float),
        min_depth=1,
        max_depth=max_depth,
        consecutive_insignificant=2,
    )
    return max(1, min(int(depth), n_obs))


def _block_bootstrap_cycle_indices(
    *,
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one moving-block bootstrap index vector over ordered control cycles."""
    if n_obs <= 1:
        return np.arange(n_obs, dtype=int)
    block_length = max(1, min(int(block_length), int(n_obs)))
    if block_length == n_obs:
        return np.arange(n_obs, dtype=int)
    max_start = n_obs - block_length + 1
    blocks_needed = int(math.ceil(n_obs / block_length))
    sample_parts: list[np.ndarray] = []
    for _ in range(blocks_needed):
        start = int(rng.integers(0, max_start))
        sample_parts.append(np.arange(start, start + block_length, dtype=int))
    return np.concatenate(sample_parts)[:n_obs]


def _bootstrap_mean_comparison_rows(
    *,
    frame: pd.DataFrame,
    scope_name: str,
    candidate_prefix: str,
    baseline_prefix: str,
    comparison_label: str,
    metric_name: str,
) -> list[dict[str, Any]]:
    """Summarize one paired cycle-level comparison with moving-block bootstrap CIs."""
    candidate_column = f"{candidate_prefix}_{metric_name}"
    baseline_column = f"{baseline_prefix}_{metric_name}"
    if candidate_column not in frame.columns or baseline_column not in frame.columns:
        return []
    paired = frame.loc[:, [candidate_column, baseline_column]].copy()
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
    if paired.empty:
        return []
    candidate_values = paired[candidate_column].to_numpy(dtype=float)
    baseline_values = paired[baseline_column].to_numpy(dtype=float)
    gain_values = baseline_values - candidate_values
    n_eval = int(candidate_values.shape[0])
    block_length = _control_cycle_block_length(pd.Series(gain_values, dtype=float))
    bootstrap_samples = int(MULTIRES_FORECAST_CONTROL["rolling_benchmark_bootstrap_samples"])
    confidence_level = float(MULTIRES_FORECAST_CONTROL["rolling_benchmark_confidence_level"])
    alpha = float((1.0 - confidence_level) / 2.0)
    rng = np.random.default_rng(42)
    candidate_boot: list[float] = []
    baseline_boot: list[float] = []
    gain_boot: list[float] = []
    for _ in range(bootstrap_samples):
        sample_idx = _block_bootstrap_cycle_indices(
            n_obs=n_eval,
            block_length=block_length,
            rng=rng,
        )
        candidate_boot.append(float(np.mean(candidate_values[sample_idx])))
        baseline_boot.append(float(np.mean(baseline_values[sample_idx])))
        gain_boot.append(float(np.mean(gain_values[sample_idx])))
    candidate_ci_low, candidate_ci_high = np.quantile(candidate_boot, [alpha, 1.0 - alpha]).tolist()
    baseline_ci_low, baseline_ci_high = np.quantile(baseline_boot, [alpha, 1.0 - alpha]).tolist()
    gain_ci_low, gain_ci_high = np.quantile(gain_boot, [alpha, 1.0 - alpha]).tolist()
    return [
        {
            "scope": str(scope_name),
            "comparison_label": str(comparison_label),
            "candidate_layer": str(candidate_prefix),
            "baseline_layer": str(baseline_prefix),
            "metric_name": str(metric_name),
            "candidate_metric": float(np.mean(candidate_values)),
            "candidate_metric_ci_low": float(candidate_ci_low),
            "candidate_metric_ci_high": float(candidate_ci_high),
            "baseline_metric": float(np.mean(baseline_values)),
            "baseline_metric_ci_low": float(baseline_ci_low),
            "baseline_metric_ci_high": float(baseline_ci_high),
            "gain_metric": float(np.mean(gain_values)),
            "gain_metric_ci_low": float(gain_ci_low),
            "gain_metric_ci_high": float(gain_ci_high),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_confidence_level": float(confidence_level),
            "bootstrap_block_length_cycles": int(block_length),
            "bootstrap_method": "moving_block_bootstrap",
            "one_sided_p_candidate_lt_baseline": float(np.mean(np.asarray(gain_boot) <= 0.0)),
            "two_sided_p": float(
                min(
                    1.0,
                    2.0
                    * min(
                        float(np.mean(np.asarray(gain_boot) <= 0.0)),
                        float(np.mean(np.asarray(gain_boot) >= 0.0)),
                    ),
                )
            ),
            "gain_ci_excludes_zero": bool(float(gain_ci_low) > 0.0 or float(gain_ci_high) < 0.0),
            "candidate_better_than_baseline": bool(float(np.mean(gain_values)) > 0.0),
            "n_eval": int(n_eval),
        }
    ]


def _control_layer_inference_frame(
    *,
    by_cycle: pd.DataFrame,
    scope_name: str,
) -> pd.DataFrame:
    """Compute paired cycle-level inference for the stacked control-layer gains."""
    if by_cycle.empty:
        return pd.DataFrame()
    comparisons = [
        ("hourly", "day_ahead", "hourly_vs_day_ahead"),
        ("phase", "hourly", "phase_vs_hourly"),
    ]
    if "nowcast_lock_mae" in by_cycle.columns:
        comparisons.append(("nowcast", "phase", "nowcast_vs_phase"))
    rows: list[dict[str, Any]] = []
    for candidate_prefix, baseline_prefix, comparison_label in comparisons:
        for metric_name in ("lock_mae", "next_lock_mae", "profile_shape_mae", "minute_path_mae", "energy_mae"):
            rows.extend(
                _bootstrap_mean_comparison_rows(
                    frame=by_cycle,
                    scope_name=str(scope_name),
                    candidate_prefix=str(candidate_prefix),
                    baseline_prefix=str(baseline_prefix),
                    comparison_label=str(comparison_label),
                    metric_name=str(metric_name),
                )
            )
    return pd.DataFrame(rows)


def _rolling_control_summary_frame(
    *,
    by_cycle: pd.DataFrame,
    scope_name: str,
) -> pd.DataFrame:
    """Aggregate rolling control results into mean/median/tail summaries by layer."""
    if by_cycle.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for layer_name in ("day_ahead", "hourly", "phase", "nowcast"):
        lock_column = f"{layer_name}_lock_mae"
        if lock_column not in by_cycle.columns:
            continue
        row: dict[str, Any] = {
            "scope": str(scope_name),
            "layer": _control_layer_label(layer_name),
            "role": str(layer_name),
            "cycle_n": int(len(by_cycle)),
        }
        for metric_name in ("lock_mae", "next_lock_mae", "profile_shape_mae", "minute_path_mae", "energy_mae"):
            values = pd.to_numeric(by_cycle[f"{layer_name}_{metric_name}"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            row[metric_name] = float(values.mean())
            row[f"{metric_name}_p50"] = float(values.quantile(0.5))
            row[f"{metric_name}_p90"] = float(values.quantile(0.9))
            pct_values = pd.to_numeric(
                by_cycle[f"{layer_name}_{metric_name}_pct"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            row[f"{metric_name}_pct"] = float(pct_values.mean())
            row[f"{metric_name}_pct_p50"] = float(pct_values.quantile(0.5))
            row[f"{metric_name}_pct_p90"] = float(pct_values.quantile(0.9))
        for metric_name in ("peak_value_mae",):
            values = pd.to_numeric(by_cycle[f"{layer_name}_{metric_name}"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            row[metric_name] = float(values.mean())
            row[f"{metric_name}_p50"] = float(values.quantile(0.5))
            row[f"{metric_name}_p90"] = float(values.quantile(0.9))
            pct_values = pd.to_numeric(
                by_cycle[f"{layer_name}_{metric_name}_pct"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            row[f"{metric_name}_pct"] = float(pct_values.mean())
            row[f"{metric_name}_pct_p50"] = float(pct_values.quantile(0.5))
            row[f"{metric_name}_pct_p90"] = float(pct_values.quantile(0.9))
        for metric_name in ("peak_interval_hit", "peak_interval_offset_minutes"):
            values = pd.to_numeric(by_cycle[f"{layer_name}_{metric_name}"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            summary_name = "peak_interval_hit_rate" if metric_name == "peak_interval_hit" else metric_name
            row[summary_name] = float(values.mean())
            row[f"{summary_name}_p50"] = float(values.quantile(0.5))
            row[f"{summary_name}_p90"] = float(values.quantile(0.9))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    baseline_row = summary.loc[summary["role"].eq("day_ahead")].iloc[0]
    for metric_name in ("lock_mae", "next_lock_mae", "profile_shape_mae", "minute_path_mae", "energy_mae"):
        summary[f"{metric_name}_gain_vs_day_ahead"] = float(baseline_row[metric_name]) - summary[metric_name]
        summary[f"{metric_name}_gain_vs_day_ahead_p50"] = (
            float(baseline_row[f"{metric_name}_p50"]) - summary[f"{metric_name}_p50"]
        )
        summary[f"{metric_name}_gain_vs_day_ahead_p90"] = (
            float(baseline_row[f"{metric_name}_p90"]) - summary[f"{metric_name}_p90"]
        )
    return summary


def _load_stage5_nowcast_anchor() -> dict[str, Any]:
    """Load the strongest available 1-minute anchor from Stage-5 holdout evidence."""
    performance_root = preferred_output_path(PATHS["outputs_performance_dir"])
    resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])
    registry = read_stage5_holdout_registry(performance_root)
    if not registry.empty:
        registry = registry.loc[registry["resolution"].astype("string").eq(resolution)].copy()
        if not registry.empty:
            registry = registry.sort_values(
                ["learned_beats_persistence", "learned_mae", "generated_at_utc", "run_id"],
                ascending=[False, True, False, False],
                kind="stable",
            ).reset_index(drop=True)
            row = registry.iloc[0]
            use_learned = bool(row["learned_beats_persistence"])
            anchor: dict[str, Any] = {
                "artifact_path": str(row["holdout_evaluation_artifact"]),
                "candidate_label": (
                    str(row["learned_candidate_label"]) if use_learned else "persistence"
                ),
                "candidate_type": "learned" if use_learned else "baseline",
                "resolution": resolution,
                "feature_set": str(row["learned_feature_set"]) if use_learned else "baseline",
                "model_label": str(row["learned_model_label"]) if use_learned else "persistence",
                "target_mode": str(row["learned_target_mode"]) if use_learned else "baseline",
                "mae": float(row["learned_mae"]) if use_learned else float(row["persistence_mae"]),
                "mae_pct": (
                    float(row["learned_mae_pct"])
                    if use_learned
                    else float(row["persistence_mae_pct"])
                ),
                "beats_persistence": bool(row["learned_beats_persistence"]),
                "reason": (
                    "Stage-5 learned nowcast anchor beat persistence on holdout MAE."
                    if use_learned
                    else "Stage-5 learned nowcast anchor did not beat persistence on holdout MAE; using persistence."
                ),
                "source_run_id": str(row["run_id"]),
            }
            for key in (
                "blend_policy_kind",
                "blend_base_policy_kind",
                "blend_window",
                "blend_sharpness",
                "blend_min_weight",
                "blend_max_weight",
                "blend_bucket_size_minutes",
                "blend_bucket_cycle_minutes",
                "blend_bucket_weights_json",
            ):
                value = row.get(key)
                if pd.notna(value):
                    anchor[key] = (
                        value
                        if key.endswith("_json") or key in {"blend_policy_kind", "blend_base_policy_kind"}
                        else float(value)
                    )
            return anchor

    holdout_path = performance_root / "latest" / "holdout_evaluation.csv"
    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing Stage-5 holdout evaluation artifact: {holdout_path}")
    holdout = pd.read_csv(holdout_path)
    learned = holdout.loc[holdout["candidate_type"].astype("string").ne("baseline")].copy()
    if learned.empty:
        raise RuntimeError(f"No learned Stage-5 holdout candidate found in {holdout_path}")
    learned = learned.sort_values(["mae", "candidate_label"], ascending=[True, True], kind="stable")
    learned_row = learned.iloc[0]
    persistence = holdout.loc[holdout["candidate_label"].astype("string").eq("persistence")].copy()
    if persistence.empty:
        raise RuntimeError(f"Missing persistence row in {holdout_path}")
    persistence_row = persistence.iloc[0]
    learned_mae = float(learned_row["mae"])
    persistence_mae = float(persistence_row["mae"])
    use_learned = learned_mae < persistence_mae
    learned_fields = _stage5_blend_fields_from_manifest(
        performance_root,
        run_id=_stage5_latest_run_id(performance_root),
        feature_set=str(learned_row["feature_set"]),
        model_label=str(learned_row["model_label"]),
        target_mode=str(learned_row["target_mode"]),
    )
    return {
        "artifact_path": _relative_artifact_path(holdout_path),
        "candidate_label": (
            str(learned_row["candidate_label"]) if use_learned else str(persistence_row["candidate_label"])
        ),
        "candidate_type": "learned" if use_learned else "baseline",
        "resolution": str(learned_row["resolution"]) if use_learned else str(persistence_row["resolution"]),
        "feature_set": str(learned_row["feature_set"]) if use_learned else "baseline",
        "model_label": str(learned_row["model_label"]) if use_learned else "persistence",
        "target_mode": str(learned_row["target_mode"]) if use_learned else "baseline",
        "mae": learned_mae if use_learned else persistence_mae,
        "mae_pct": float(learned_row["mae_pct"]) if use_learned else float(persistence_row["mae_pct"]),
        "beats_persistence": bool(use_learned),
        "reason": (
            "Stage-5 learned nowcast anchor beat persistence on holdout MAE."
            if use_learned
            else "Stage-5 learned nowcast anchor did not beat persistence on holdout MAE; using persistence."
        ),
        "source_run_id": _stage5_latest_run_id(performance_root),
        **(learned_fields if use_learned else {}),
    }


def _stage5_latest_run_id(performance_root: Path) -> str:
    """Return the real Stage-5 run id behind the Windows-safe `latest` alias."""
    manifest_path = performance_root / "latest" / "run_manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(manifest.get("run_id", ""))


def _stage5_blend_fields_from_manifest(
    performance_root: Path,
    *,
    run_id: str,
    feature_set: str,
    model_label: str,
    target_mode: str,
) -> dict[str, Any]:
    """Recover the saved Stage-5 blend config for one candidate when available."""
    policy_kind = _stage5_blend_policy_kind(str(target_mode))
    if not policy_kind or not run_id:
        return {}
    manifest_path = performance_root / run_id / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    blend_policy = manifest.get("blend_policy")
    if not isinstance(blend_policy, dict) or not bool(blend_policy.get("enabled")):
        return {}
    candidate = blend_policy.get("candidate")
    if not isinstance(candidate, dict):
        return {}
    if (
        str(candidate.get("feature_set", "")) != str(feature_set)
        or str(candidate.get("model_label", "")) != str(model_label)
        or str(candidate.get("target_mode", "")) != _stage5_base_target_mode(str(target_mode))
    ):
        return {}
    manifest_policy_kind = str(blend_policy.get("policy_kind", "sigmoid")).strip().lower() or "sigmoid"
    if manifest_policy_kind != policy_kind:
        return {}
    if policy_kind == "bucket":
        bucket_weights = blend_policy.get("bucket_weights", {})
        if not isinstance(bucket_weights, dict):
            return {}
        pre_bucket_blend = blend_policy.get("pre_bucket_blend")
        if isinstance(pre_bucket_blend, dict):
            return {
                "blend_policy_kind": "bucket",
                "blend_base_policy_kind": "sigmoid",
                "blend_window": float(pre_bucket_blend["window"]),
                "blend_sharpness": float(pre_bucket_blend["sharpness"]),
                "blend_min_weight": float(pre_bucket_blend["min_weight"]),
                "blend_max_weight": float(pre_bucket_blend["max_weight"]),
                "blend_bucket_size_minutes": float(blend_policy["bucket_size_minutes"]),
                "blend_bucket_cycle_minutes": float(blend_policy.get("cycle_minutes", 15)),
                "blend_bucket_weights_json": json.dumps(bucket_weights, sort_keys=True),
            }
        return {
            "blend_policy_kind": "bucket",
            "blend_base_policy_kind": str(blend_policy.get("base_policy_kind", "raw")),
            "blend_bucket_size_minutes": float(blend_policy["bucket_size_minutes"]),
            "blend_bucket_cycle_minutes": float(blend_policy.get("cycle_minutes", 15)),
            "blend_bucket_weights_json": json.dumps(bucket_weights, sort_keys=True),
        }
    return {
        "blend_policy_kind": "sigmoid",
        "blend_base_policy_kind": "raw",
        "blend_window": float(blend_policy["window"]),
        "blend_sharpness": float(blend_policy["sharpness"]),
        "blend_min_weight": float(blend_policy["min_weight"]),
        "blend_max_weight": float(blend_policy["max_weight"]),
    }


def _load_stage5_blend_finalists(performance_root: Path) -> pd.DataFrame:
    """Load the latest Stage-5 per-family blend finalists when that artifact exists."""
    blend_finalists_path = performance_root / "latest" / "blend_finalists.csv"
    if not blend_finalists_path.exists():
        return pd.DataFrame()
    blend_finalists = pd.read_csv(blend_finalists_path)
    if blend_finalists.empty:
        return blend_finalists
    resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])
    blend_finalists = blend_finalists.loc[
        blend_finalists["resolution"].astype("string").eq(resolution)
    ].copy()
    if blend_finalists.empty:
        return blend_finalists
    return blend_finalists.sort_values(
        [
            "meets_p2_fold_degrade_cap",
            "fold_mean_mae_ratio",
            "fold_std_mae_ratio",
            "max_fold_degrade_pct",
            "feature_set",
            "model_label",
            "target_mode",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _split_mask(frame: pd.DataFrame, split_names: list[str]) -> pd.Series:
    """Return the row mask for the named split windows based on `day_idx`."""
    if not split_names:
        return pd.Series(False, index=frame.index, dtype=bool)
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for split_name in split_names:
        start_day, end_day = SPLIT_DAY_RANGES[str(split_name)]
        mask = mask | frame["day_idx"].between(int(start_day), int(end_day))
    return mask.astype(bool)


def _prior_split_names(split_names: list[str]) -> list[str]:
    """Return ordered splits that finish before the requested evaluation splits begin."""
    if not split_names:
        return []
    earliest_start = min(int(SPLIT_DAY_RANGES[str(name)][0]) for name in split_names)
    ordered = sorted(SPLIT_DAY_RANGES.items(), key=lambda item: (int(item[1][0]), str(item[0])))
    return [str(name) for name, bounds in ordered if int(bounds[1]) < earliest_start]


def _stage5_nowcast_context(
    *,
    train_split_names: list[str],
    evaluation_split_names: list[str],
) -> dict[str, Any]:
    """Build one split-aware Stage-5 context for exact-control minute predictions.

    Stage-10 selects nowcast policies on calibration windows and evaluates them on
    held-out windows, so the minute layer needs separate train/evaluation surfaces
    rather than a single fixed train+holdout split.
    """
    resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])
    gold = _load_stage5_gold_with_full_grid(
        resolution,
        preferred_output_path(PATHS["gold_dir"]),
    )
    gold, _ = _augment_stage5_curated_ramp_features(
        gold,
        ramp_quantile=float(MODELING_PERFORMANCE_RAMP["quantile"]),
    )
    feature_sets = _build_stage5_feature_sets(include_curated_ramp=True, include_full_stable=True)
    model_catalog = build_model_catalog(
        include_hgb_coordinate_search=True,
        include_hgb_frontier=True,
    )
    model_catalog.update(_load_stage5_dynamic_model_catalog())
    train_df = gold.loc[_split_mask(gold, train_split_names)].copy()
    eval_df = gold.loc[_split_mask(gold, evaluation_split_names)].copy()
    if train_df.empty:
        raise RuntimeError(
            "Stage-10 nowcast training surface is empty. "
            f"train_split_names={train_split_names}"
        )
    if eval_df.empty:
        raise RuntimeError(
            "Stage-10 nowcast evaluation surface is empty. "
            f"evaluation_split_names={evaluation_split_names}"
        )
    blend_config = Stage5BlendConfig(
        window=int(MODELING_PERFORMANCE_BLEND_SEARCH["base_window"]),
        sharpness=float(MODELING_PERFORMANCE_BLEND_SEARCH["base_sharpness"]),
        min_weight=float(MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"]),
        max_weight=float(MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"]),
    )
    context_key = stable_config_hash(
        {
            "resolution": resolution,
            "train_split_names": [str(value) for value in train_split_names],
            "evaluation_split_names": [str(value) for value in evaluation_split_names],
            "ramp_quantile": float(MODELING_PERFORMANCE_RAMP["quantile"]),
            "blend_window": int(MODELING_PERFORMANCE_BLEND_SEARCH["base_window"]),
            "blend_sharpness": float(MODELING_PERFORMANCE_BLEND_SEARCH["base_sharpness"]),
            "blend_min_weight": float(MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"]),
            "blend_max_weight": float(MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"]),
        }
    )
    return {
        "context_key": context_key,
        "resolution": resolution,
        "train_df": train_df,
        "eval_df": eval_df,
        "feature_sets": feature_sets,
        "model_catalog": model_catalog,
        "blend_config": blend_config,
    }


def _load_stage5_dynamic_model_catalog() -> dict[str, ModelSpec]:
    """Hydrate adaptive Stage-5 model specs from the latest Stage-5 manifest when present."""
    performance_root = preferred_output_path(PATHS["outputs_performance_dir"])
    manifest_path = performance_root / "latest" / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = manifest.get("models")
    if not isinstance(models, list):
        return {}
    hydrated: dict[str, ModelSpec] = {}
    for item in models:
        if not isinstance(item, dict):
            continue
        model_label = str(item.get("model_label", ""))
        family = str(item.get("family", ""))
        params = item.get("params")
        if not model_label or not isinstance(params, dict):
            continue
        if family == "hgb":
            hydrated[model_label] = _build_hgb_spec(model_label, cast(dict[str, Any], params))
        elif family == "xgb":
            hydrated[model_label] = _build_xgb_spec(model_label, cast(dict[str, Any], params))
    return hydrated


def _stage5_nowcast_contexts() -> dict[str, dict[str, Any]]:
    """Build the calibration and held-out evaluation contexts for Stage-10 nowcasts."""
    calibration_splits = [str(value) for value in MULTIRES_FORECAST_CONTROL["calibration_splits"]]
    evaluation_splits = [str(value) for value in MULTIRES_FORECAST_CONTROL["evaluation_splits"]]
    calibration_train_splits = _prior_split_names(calibration_splits)
    evaluation_train_splits = _prior_split_names(evaluation_splits)
    if not calibration_train_splits:
        calibration_train_splits = list(evaluation_train_splits)
    if not evaluation_train_splits:
        evaluation_train_splits = list(calibration_train_splits)
    cache_key = stable_config_hash(
        {
            "resolution": str(MULTIRES_FORECAST_CONTROL["actual_resolution"]),
            "calibration_train_splits": calibration_train_splits,
            "calibration_splits": calibration_splits,
            "evaluation_train_splits": evaluation_train_splits,
            "evaluation_splits": evaluation_splits,
        }
    )
    with _LOCAL_CACHE_LOCK:
        cached = _STAGE5_NOWCAST_CONTEXTS_CACHE.get(cache_key)
    if cached is not None:
        return {
            scope_name: dict(scope_context)
            for scope_name, scope_context in cached.items()
        }
    contexts = {
        "calibration": _stage5_nowcast_context(
            train_split_names=calibration_train_splits,
            evaluation_split_names=calibration_splits,
        ),
        "evaluation": _stage5_nowcast_context(
            train_split_names=evaluation_train_splits,
            evaluation_split_names=evaluation_splits,
        ),
    }
    with _LOCAL_CACHE_LOCK:
        _STAGE5_NOWCAST_CONTEXTS_CACHE[cache_key] = contexts
    return {
        scope_name: dict(scope_context)
        for scope_name, scope_context in contexts.items()
    }


def _load_stage5_nowcast_candidate_pool(
    *,
    upstream_anchor: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load the exact-control nowcast challenger pool from Stage-5 artifacts.

    The pool always includes persistence, then adds the strongest holdout-backed
    Stage-5 candidates across runs before filling remaining slots from the latest
    validation scoreboard. This keeps exact-control nowcasts from being limited
    to whichever quick run currently owns `latest/`.
    """
    performance_root = preferred_output_path(PATHS["outputs_performance_dir"])
    scoreboard_path = performance_root / "latest" / "selection_scoreboard.csv"
    blend_finalists_path = performance_root / "latest" / "blend_finalists.csv"
    latest_run_id = _stage5_latest_run_id(performance_root)
    learned_limit = int(MULTIRES_FORECAST_CONTROL["nowcast_candidate_pool_size"])
    registry_limit = max(1, int(math.ceil(learned_limit / 2.0)))
    resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])

    pool: list[dict[str, Any]] = [
        {
            "candidate_label": "persistence",
            "candidate_type": "baseline",
            "resolution": resolution,
            "feature_set": "baseline",
            "model_label": "persistence",
            "target_mode": "baseline",
            "artifact_path": _relative_artifact_path(performance_root / "latest" / "holdout_evaluation.csv"),
            "pool_source_type": "stage5_holdout_baseline",
            "pool_source_run_id": latest_run_id,
        }
    ]
    seen_labels = {"persistence"}
    registry = read_stage5_holdout_registry(performance_root)
    if not registry.empty:
        registry = registry.loc[registry["resolution"].astype("string").eq(resolution)].copy()
        registry = registry.sort_values(
            ["learned_beats_persistence", "learned_mae", "generated_at_utc", "run_id"],
            ascending=[False, True, False, False],
            kind="stable",
        )
        for row in registry.itertuples(index=False):
            candidate_label = str(row.learned_candidate_label)
            if not candidate_label or candidate_label in seen_labels:
                continue
            candidate = {
                "candidate_label": candidate_label,
                "candidate_type": "learned",
                "resolution": str(row.resolution),
                "feature_set": str(row.learned_feature_set),
                "model_label": str(row.learned_model_label),
                "target_mode": str(row.learned_target_mode),
                "artifact_path": str(row.holdout_evaluation_artifact),
                "pool_source_type": "stage5_holdout_registry",
                "pool_source_run_id": str(row.run_id),
            }
            for key in (
                "blend_policy_kind",
                "blend_base_policy_kind",
                "blend_window",
                "blend_sharpness",
                "blend_min_weight",
                "blend_max_weight",
                "blend_bucket_size_minutes",
                "blend_bucket_cycle_minutes",
                "blend_bucket_weights_json",
            ):
                value = getattr(row, key, float("nan"))
                if pd.notna(value):
                    candidate[key] = (
                        value
                        if key.endswith("_json") or key in {"blend_policy_kind", "blend_base_policy_kind"}
                        else float(value)
                    )
            pool.append(candidate)
            seen_labels.add(candidate_label)
            if len(pool) - 1 >= registry_limit:
                break
    blend_finalists = _load_stage5_blend_finalists(performance_root)
    if not blend_finalists.empty:
        finalists_parts: list[pd.DataFrame] = []
        for feature_set in blend_finalists["feature_set"].astype("string").dropna().unique().tolist():
            feature_rows = blend_finalists.loc[
                blend_finalists["feature_set"].astype("string").eq(str(feature_set))
            ].head(1)
            if not feature_rows.empty:
                finalists_parts.append(feature_rows)
        finalists_parts.append(blend_finalists)
        diversified_finalists = (
            pd.concat(finalists_parts, ignore_index=True)
            .drop_duplicates(subset=["candidate_label"], keep="first")
            .reset_index(drop=True)
        )
        for row in diversified_finalists.itertuples(index=False):
            candidate_label = str(getattr(row, "candidate_label", ""))
            if not candidate_label or candidate_label in seen_labels:
                continue
            pool.append(
                {
                    "candidate_label": candidate_label,
                    "candidate_type": "learned",
                    "resolution": str(row.resolution),
                    "feature_set": str(row.feature_set),
                    "model_label": str(row.model_label),
                    "target_mode": str(row.target_mode),
                    "artifact_path": _relative_artifact_path(blend_finalists_path),
                    "pool_source_type": "stage5_blend_finalists",
                    "pool_source_run_id": latest_run_id,
                    "blend_policy_kind": str(getattr(row, "selected_blend_policy_kind", "sigmoid")),
                    "blend_base_policy_kind": str(getattr(row, "selected_blend_base_policy_kind", "raw")),
                    **(
                        {
                            "blend_bucket_size_minutes": float(
                                getattr(row, "selected_blend_bucket_size_minutes", float("nan"))
                            ),
                            "blend_bucket_cycle_minutes": float(
                                getattr(row, "selected_blend_bucket_cycle_minutes", float("nan"))
                            ),
                            "blend_bucket_weights_json": str(
                                getattr(row, "selected_blend_bucket_weights_json", "")
                            ),
                            **(
                                {
                                    "blend_window": float(getattr(row, "selected_blend_window", float("nan"))),
                                    "blend_sharpness": float(getattr(row, "selected_blend_sharpness", float("nan"))),
                                    "blend_min_weight": float(getattr(row, "selected_blend_min_weight", float("nan"))),
                                    "blend_max_weight": float(getattr(row, "selected_blend_max_weight", float("nan"))),
                                }
                                if str(getattr(row, "selected_blend_base_policy_kind", "raw")) == "sigmoid"
                                else {}
                            ),
                        }
                        if _stage5_blend_policy_kind(str(row.target_mode)) == "bucket"
                        else {
                            "blend_window": float(row.selected_blend_window),
                            "blend_sharpness": float(row.selected_blend_sharpness),
                            "blend_min_weight": float(row.selected_blend_min_weight),
                            "blend_max_weight": float(row.selected_blend_max_weight),
                        }
                    ),
                }
            )
            seen_labels.add(candidate_label)
            if len(pool) - 1 >= learned_limit:
                break
    if scoreboard_path.exists():
        scoreboard = pd.read_csv(scoreboard_path)
        scoreboard = scoreboard.loc[
            scoreboard["resolution"].astype("string").eq(resolution)
        ].copy()
        if not blend_finalists.empty:
            scoreboard = scoreboard.loc[
                ~scoreboard["target_mode"].astype("string").map(
                    lambda value: bool(_stage5_blend_policy_kind(str(value)))
                )
            ].copy()
        scoreboard = scoreboard.sort_values(
            ["fold_mean_mae_ratio", "mean_coverage", "feature_set", "model_label", "target_mode"],
            ascending=[True, False, True, True, True],
            kind="stable",
        )
        scoreboard_parts: list[pd.DataFrame] = []
        for feature_set in scoreboard["feature_set"].astype("string").dropna().unique().tolist():
            feature_rows = scoreboard.loc[
                scoreboard["feature_set"].astype("string").eq(str(feature_set))
            ].head(1)
            if not feature_rows.empty:
                scoreboard_parts.append(feature_rows)
        scoreboard_parts.append(scoreboard)
        scoreboard_candidates = (
            pd.concat(scoreboard_parts, ignore_index=True)
            .drop_duplicates(subset=["feature_set", "model_label", "target_mode"], keep="first")
        )
        for row in scoreboard_candidates.itertuples(index=False):
            candidate_label = f"{row.feature_set}/{row.model_label}/{row.target_mode}"
            if candidate_label in seen_labels:
                continue
            pool.append(
                {
                    "candidate_label": candidate_label,
                    "candidate_type": "learned",
                    "resolution": str(row.resolution),
                    "feature_set": str(row.feature_set),
                    "model_label": str(row.model_label),
                    "target_mode": str(row.target_mode),
                    "artifact_path": _relative_artifact_path(scoreboard_path),
                    "pool_source_type": "stage5_selection_scoreboard",
                    "pool_source_run_id": latest_run_id,
                    **_stage5_blend_fields_from_manifest(
                        performance_root,
                        run_id=latest_run_id,
                        feature_set=str(row.feature_set),
                        model_label=str(row.model_label),
                        target_mode=str(row.target_mode),
                    ),
                }
            )
            seen_labels.add(candidate_label)
            if len(pool) - 1 >= learned_limit:
                break
    upstream_label = str(upstream_anchor.get("candidate_label", ""))
    if upstream_label and upstream_label != "persistence" and upstream_label not in seen_labels:
        candidate = {
            "candidate_label": upstream_label,
            "candidate_type": str(upstream_anchor.get("candidate_type", "learned")),
            "resolution": str(upstream_anchor.get("resolution", resolution)),
            "feature_set": str(upstream_anchor.get("feature_set", "")),
            "model_label": str(upstream_anchor.get("model_label", "")),
            "target_mode": str(upstream_anchor.get("target_mode", "")),
            "artifact_path": str(upstream_anchor.get("artifact_path", "")),
            "pool_source_type": "stage5_holdout_recommendation",
            "pool_source_run_id": str(upstream_anchor.get("source_run_id", latest_run_id)),
        }
        for key in (
            "blend_policy_kind",
            "blend_base_policy_kind",
            "blend_window",
            "blend_sharpness",
            "blend_min_weight",
            "blend_max_weight",
            "blend_bucket_size_minutes",
            "blend_bucket_cycle_minutes",
            "blend_bucket_weights_json",
        ):
            if key in upstream_anchor:
                candidate[key] = (
                    upstream_anchor[key]
                    if key.endswith("_json") or key in {"blend_policy_kind", "blend_base_policy_kind"}
                    else float(upstream_anchor[key])
        )
        pool.append(candidate)
    return pool


def _stage5_nowcast_base_candidate_label(candidate_label: str) -> str:
    """Strip Stage-10 control wrappers so Stage-5 evidence can match the base minute family."""
    return str(candidate_label).split("|", 1)[0].strip()


def _load_stage5_nowcast_advisory_evidence() -> dict[str, dict[str, Any]]:
    """Load broader Stage-5 minute evidence used only as a near-tie breaker in Stage-10."""
    performance_root = preferred_output_path(PATHS["outputs_performance_dir"])
    latest_dir = performance_root / "latest"
    advisory_path = latest_dir / "supplemental_surface_advisory.json"
    segment_path = latest_dir / "supplemental_surface_segment_evaluation.csv"
    if not advisory_path.exists():
        return {}
    try:
        advisory_payload = json.loads(advisory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    base_candidate_label = _stage5_nowcast_base_candidate_label(str(advisory_payload.get("candidate_label", "")))
    if not base_candidate_label:
        return {}
    segment_frame = _read_csv_if_present(segment_path)

    def _segment_ratio(segment_column: str, segment_value: str) -> float:
        if segment_frame.empty:
            return float("nan")
        matched = segment_frame.loc[
            segment_frame["segment_column"].astype("string").eq(str(segment_column))
            & segment_frame["segment_value"].astype("string").eq(str(segment_value))
        ].copy()
        if matched.empty or "candidate_mae_ratio_to_persistence" not in matched.columns:
            return float("nan")
        return float(pd.to_numeric(matched.iloc[0]["candidate_mae_ratio_to_persistence"], errors="coerce"))

    transition_ratios = [
        _segment_ratio("operating_regime", "transition_only"),
        _segment_ratio("operating_regime", "transition_active"),
    ]
    finite_transition_ratios = [value for value in transition_ratios if np.isfinite(value)]
    return {
        base_candidate_label: {
            "advisory_base_candidate_label": base_candidate_label,
            "advisory_surface_supported": bool(advisory_payload.get("learned_beats_persistence", False)),
            "advisory_supported_regime_count": int(
                advisory_payload.get("learned_supported_operating_regime_count", 0)
            ),
            "advisory_supported_operating_regimes": [
                str(value) for value in advisory_payload.get("learned_supported_operating_regimes", [])
            ],
            "advisory_surface_candidate_mae_ratio_to_persistence": float(
                advisory_payload.get("candidate_mae_ratio_to_persistence", float("nan"))
            ),
            "advisory_transition_only_ratio_to_persistence": float(
                _segment_ratio("operating_regime", "transition_only")
            ),
            "advisory_transition_active_ratio_to_persistence": float(
                _segment_ratio("operating_regime", "transition_active")
            ),
            "advisory_transition_best_ratio_to_persistence": (
                float(min(finite_transition_ratios)) if finite_transition_ratios else float("nan")
            ),
            "advisory_high_ramp_ratio_to_persistence": float(
                _segment_ratio("actual_ramp_band", "high_ramp")
            ),
        }
    }


def _attach_nowcast_advisory_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Decorate nowcast candidates with broader Stage-5 advisory evidence when available."""
    working = frame.copy()
    if working.empty:
        return working
    working["advisory_base_candidate_label"] = (
        working["candidate_label"].astype("string").map(_stage5_nowcast_base_candidate_label).astype("string")
    )
    defaults: dict[str, Any] = {
        "advisory_surface_supported": False,
        "advisory_supported_regime_count": 0,
        "advisory_supported_operating_regimes": "",
        "advisory_surface_candidate_mae_ratio_to_persistence": float("nan"),
        "advisory_transition_only_ratio_to_persistence": float("nan"),
        "advisory_transition_active_ratio_to_persistence": float("nan"),
        "advisory_transition_best_ratio_to_persistence": float("nan"),
        "advisory_high_ramp_ratio_to_persistence": float("nan"),
    }
    for column_name, default_value in defaults.items():
        if column_name not in working.columns:
            working[column_name] = default_value
    evidence_by_candidate = _load_stage5_nowcast_advisory_evidence()
    if not evidence_by_candidate:
        return working
    for base_candidate_label, evidence in evidence_by_candidate.items():
        matched = working["advisory_base_candidate_label"].astype("string").eq(str(base_candidate_label))
        if not bool(matched.any()):
            continue
        for column_name, value in evidence.items():
            if column_name == "advisory_supported_operating_regimes":
                working.loc[matched, column_name] = ",".join(str(item) for item in value)
            else:
                working.loc[matched, column_name] = value
    return working


def _stage5_candidate_predictions(
    *,
    candidate: dict[str, Any],
    context: dict[str, Any],
) -> pd.DataFrame:
    """Generate causal 1-minute predictions for one Stage-5 candidate on holdout days."""
    context_key = str(context.get("context_key", ""))
    cache_key = (
        context_key,
        stable_config_hash(
            {
                "candidate_label": str(candidate.get("candidate_label", "")),
                "candidate": candidate,
            }
        ),
    )
    with _LOCAL_CACHE_LOCK:
        cached = _STAGE5_PREDICTION_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    eval_df = cast(pd.DataFrame, context["eval_df"])
    resolution = str(candidate["resolution"])
    candidate_label = str(candidate["candidate_label"])
    candidate_type = str(candidate["candidate_type"])
    feature_set = str(candidate["feature_set"])
    model_label = str(candidate["model_label"])
    target_mode = str(candidate["target_mode"])
    if candidate_label == "persistence":
        aligned = eval_df.loc[:, ["timestamp", "avg_load", "lag_1"]].dropna(subset=["timestamp", "avg_load", "lag_1"]).copy()
        aligned["predicted_load"] = aligned["lag_1"].astype(float)
        output = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(aligned["timestamp"], errors="raise"),
                "actual_load": aligned["avg_load"].astype(float).to_numpy(dtype=float),
                "predicted_load": aligned["predicted_load"].astype(float).to_numpy(dtype=float),
                "candidate_label": candidate_label,
                "candidate_type": candidate_type,
                "resolution": resolution,
                "feature_set": feature_set,
                "model_label": model_label,
                "target_mode": target_mode,
            }
        )
        with _LOCAL_CACHE_LOCK:
            _STAGE5_PREDICTION_CACHE[cache_key] = output.copy()
        return output

    aligned_result = _fit_stage5_candidate_and_align(
        train_df=cast(pd.DataFrame, context["train_df"]),
        eval_df=eval_df,
        feature_cols=_resolve_stage5_feature_set_columns(
            feature_set,
            feature_sets=cast(dict[str, list[str]], context["feature_sets"]),
        ),
        model_spec=cast(dict[str, Any], context["model_catalog"])[model_label],
        target_mode=_stage5_base_target_mode(target_mode),
    )
    if aligned_result is None:
        return pd.DataFrame()
    aligned, _ = aligned_result
    prediction_series = aligned["y_pred"].astype(float)
    blend_policy_kind = _stage5_blend_policy_kind(target_mode)
    if blend_policy_kind == "sigmoid":
        blend_config = cast(Stage5BlendConfig, context["blend_config"])
        if all(key in candidate for key in ("blend_window", "blend_sharpness", "blend_min_weight", "blend_max_weight")):
            blend_config = Stage5BlendConfig(
                window=int(float(candidate["blend_window"])),
                sharpness=float(candidate["blend_sharpness"]),
                min_weight=float(candidate["blend_min_weight"]),
                max_weight=float(candidate["blend_max_weight"]),
            )
        _, decisions = _apply_stage5_blend_policy(
            aligned=aligned,
            blend_config=blend_config,
            n_eval_total=int(len(eval_df)),
        )
        prediction_series = (
            decisions.set_index("row_index")["blend_pred"].reindex(aligned.index).astype(float)
        )
    elif blend_policy_kind == "bucket":
        bucket_aligned = aligned.loc[:, ["y_true", "y_persist", "y_pred"]].copy()
        if str(candidate.get("blend_base_policy_kind", _stage5_blend_base_policy_kind(target_mode) or "raw")) == "sigmoid":
            if not all(key in candidate for key in ("blend_window", "blend_sharpness", "blend_min_weight", "blend_max_weight")):
                raise ValueError(
                    f"Stage-5 bucket-over-sigmoid candidate is missing persisted blend settings: {candidate_label}"
                )
            pre_bucket_blend_config = Stage5BlendConfig(
                window=int(float(candidate["blend_window"])),
                sharpness=float(candidate["blend_sharpness"]),
                min_weight=float(candidate["blend_min_weight"]),
                max_weight=float(candidate["blend_max_weight"]),
            )
            _, pre_bucket_decisions = _apply_stage5_blend_policy(
                aligned=aligned,
                blend_config=pre_bucket_blend_config,
                n_eval_total=int(len(eval_df)),
            )
            bucket_aligned["y_pred"] = (
                pre_bucket_decisions.set_index("row_index")["blend_pred"].reindex(aligned.index).astype(float).to_numpy()
            )
        if not all(
            key in candidate
            for key in (
                "blend_bucket_size_minutes",
                "blend_bucket_cycle_minutes",
                "blend_bucket_weights_json",
            )
        ):
            raise ValueError(
                f"Stage-5 bucket blend candidate is missing persisted bucket settings: {candidate_label}"
            )
        bucket_config = Stage5BucketBlendConfig(
            bucket_size_minutes=int(float(candidate["blend_bucket_size_minutes"])),
            cycle_minutes=int(float(candidate["blend_bucket_cycle_minutes"])),
            bucket_weights=tuple(
                sorted(
                    (
                        int(bucket),
                        float(weight),
                    )
                    for bucket, weight in json.loads(str(candidate["blend_bucket_weights_json"])).items()
                )
            ),
        )
        _, decisions = _apply_stage5_bucket_blend_policy(
            aligned=bucket_aligned,
            timestamps=pd.to_datetime(eval_df.loc[aligned.index, "timestamp"], errors="raise"),
            bucket_config=bucket_config,
            n_eval_total=int(len(eval_df)),
        )
        prediction_series = (
            decisions.set_index("row_index")["blend_pred"].reindex(aligned.index).astype(float)
        )
    timestamps = pd.to_datetime(eval_df.loc[aligned.index, "timestamp"], errors="raise")
    output = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "actual_load": aligned["y_true"].astype(float).to_numpy(dtype=float),
            "predicted_load": prediction_series.to_numpy(dtype=float),
            "candidate_label": candidate_label,
            "candidate_type": candidate_type,
            "resolution": resolution,
            "feature_set": feature_set,
            "model_label": model_label,
            "target_mode": target_mode,
        }
    )
    with _LOCAL_CACHE_LOCK:
        _STAGE5_PREDICTION_CACHE[cache_key] = output.copy()
    return output


def _origin_schedule_matches(timestamp: pd.Timestamp, *, stride_minutes: int) -> bool:
    """Return whether a timestamp lies on the configured Stage-10 origin schedule."""
    minute_of_day = int(timestamp.hour * 60 + timestamp.minute)
    origin_offset = int(
        MULTIRES_FORECAST_CONTROL["cycle_origin_hour"] * 60
        + MULTIRES_FORECAST_CONTROL["cycle_origin_minute"]
    )
    if minute_of_day < origin_offset:
        return False
    return bool((minute_of_day - origin_offset) % stride_minutes == 0)


def _control_origin_schedule_matches(timestamp: pd.Timestamp) -> bool:
    """Return whether a timestamp lies on the configured exact-control origin schedule."""
    return _origin_schedule_matches(
        pd.Timestamp(timestamp),
        stride_minutes=int(MULTIRES_FORECAST_CONTROL["cycle_origin_stride_minutes"]),
    )


def _cap_control_origin_rows(origin_rows: pd.DataFrame, *, cap: int) -> pd.DataFrame:
    """Evenly cap one split's origin rows without biasing toward the earliest windows."""
    if cap <= 0:
        return origin_rows.reset_index(drop=True)
    if len(origin_rows) <= int(cap):
        return origin_rows.reset_index(drop=True)
    positions = np.linspace(0, len(origin_rows) - 1, num=int(cap), dtype=int)
    return origin_rows.iloc[positions].drop_duplicates(subset=["origin_timestamp"]).reset_index(drop=True)


def _build_control_cycle_origin_catalog(
    base: pd.DataFrame,
    *,
    stride_minutes: int,
    cap: int,
    split_names: list[str],
) -> pd.DataFrame:
    """Build the split-aware catalog of eligible day-ahead control origins for one schedule."""
    horizon_minutes = int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"])
    resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])
    step_seconds = int(pd.to_timedelta(resolution).total_seconds())
    horizon_steps = int((horizon_minutes * 60) // step_seconds)
    rows: list[dict[str, Any]] = []
    for split_name in list(dict.fromkeys(str(value) for value in split_names)):
        start_day, end_day = SPLIT_DAY_RANGES[str(split_name)]
        split_rows: list[dict[str, Any]] = []
        for idx in range(len(base)):
            if idx + horizon_steps >= len(base):
                continue
            origin_timestamp = pd.Timestamp(base.iloc[idx]["timestamp"])
            origin_day_idx = int(base.iloc[idx]["day_idx"])
            end_day_idx = int(base.iloc[idx + horizon_steps]["day_idx"])
            if not _origin_schedule_matches(origin_timestamp, stride_minutes=int(stride_minutes)):
                continue
            if not (start_day <= origin_day_idx <= end_day):
                continue
            if not (start_day <= end_day_idx <= end_day):
                continue
            split_rows.append(
                {
                    "split_name": str(split_name),
                    "origin_timestamp": origin_timestamp,
                    "origin_day_idx": origin_day_idx,
                    "end_day_idx": end_day_idx,
                    "origin_hour": int(origin_timestamp.hour),
                    "origin_minute": int(origin_timestamp.minute),
                    "origin_minute_of_day": int(origin_timestamp.hour * 60 + origin_timestamp.minute),
                }
            )
        if not split_rows:
            continue
        capped = _cap_control_origin_rows(
            pd.DataFrame(split_rows).sort_values("origin_timestamp", kind="stable").reset_index(drop=True),
            cap=int(cap),
        )
        rows.extend(capped.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(
            columns=[
                "split_name",
                "origin_timestamp",
                "origin_day_idx",
                "end_day_idx",
                "origin_hour",
                "origin_minute",
                "origin_minute_of_day",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["split_name", "origin_timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def _control_cycle_origin_catalog(base: pd.DataFrame) -> pd.DataFrame:
    """Build the full split-aware catalog of eligible exact-control origins."""
    relevant_splits = list(
        dict.fromkeys(
            [
                *MULTIRES_FORECAST_CONTROL["calibration_splits"],
                *MULTIRES_FORECAST_CONTROL["evaluation_splits"],
            ]
        )
    )
    return _build_control_cycle_origin_catalog(
        base,
        stride_minutes=int(MULTIRES_FORECAST_CONTROL["cycle_origin_stride_minutes"]),
        cap=int(MULTIRES_FORECAST_CONTROL["max_cycles"]),
        split_names=[str(value) for value in relevant_splits],
    )


def _rolling_control_cycle_origin_catalog(base: pd.DataFrame) -> pd.DataFrame:
    """Build the broader rolling benchmark catalog across more start times."""
    if not bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"]):
        return pd.DataFrame()
    relevant_splits = list(
        dict.fromkeys(
            [
                *MULTIRES_FORECAST_CONTROL["calibration_splits"],
                *MULTIRES_FORECAST_CONTROL["evaluation_splits"],
            ]
        )
    )
    return _build_control_cycle_origin_catalog(
        base,
        stride_minutes=int(MULTIRES_FORECAST_CONTROL["rolling_benchmark_origin_stride_minutes"]),
        cap=int(MULTIRES_FORECAST_CONTROL["rolling_benchmark_max_cycles"]),
        split_names=[str(value) for value in relevant_splits],
    )


def _control_cycle_origins(base: pd.DataFrame) -> list[pd.Timestamp]:
    """Backward-compatible helper returning evaluation origins from the split-aware catalog."""
    catalog = _control_cycle_origin_catalog(base)
    if catalog.empty:
        return []
    evaluation_splits = {
        str(value) for value in MULTIRES_FORECAST_CONTROL["evaluation_splits"]
    }
    evaluation_rows = catalog.loc[catalog["split_name"].astype("string").isin(evaluation_splits)].copy()
    return [pd.Timestamp(value) for value in evaluation_rows["origin_timestamp"].tolist()]


def _resolve_control_origin_sets(base: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:
    """Return the split-aware origin catalog plus calibration and evaluation origin lists."""
    catalog = _control_cycle_origin_catalog(base)
    if catalog.empty:
        return catalog, [], []
    calibration_splits = {
        str(value) for value in MULTIRES_FORECAST_CONTROL["calibration_splits"]
    }
    evaluation_splits = {
        str(value) for value in MULTIRES_FORECAST_CONTROL["evaluation_splits"]
    }
    calibration_rows = catalog.loc[catalog["split_name"].astype("string").isin(calibration_splits)].copy()
    evaluation_rows = catalog.loc[catalog["split_name"].astype("string").isin(evaluation_splits)].copy()
    calibration_origins = [pd.Timestamp(value) for value in calibration_rows["origin_timestamp"].tolist()]
    evaluation_origins = [pd.Timestamp(value) for value in evaluation_rows["origin_timestamp"].tolist()]
    if not calibration_origins:
        calibration_origins = list(evaluation_origins)
    if not evaluation_origins:
        evaluation_origins = list(calibration_origins)
    return catalog, calibration_origins, evaluation_origins


def _resolve_rolling_control_origin_sets(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:
    """Return the broader rolling benchmark catalog plus calibration/evaluation origin lists."""
    catalog = _rolling_control_cycle_origin_catalog(base)
    if catalog.empty:
        return catalog, [], []
    calibration_splits = {str(value) for value in MULTIRES_FORECAST_CONTROL["calibration_splits"]}
    evaluation_splits = {str(value) for value in MULTIRES_FORECAST_CONTROL["evaluation_splits"]}
    calibration_rows = catalog.loc[catalog["split_name"].astype("string").isin(calibration_splits)].copy()
    evaluation_rows = catalog.loc[catalog["split_name"].astype("string").isin(evaluation_splits)].copy()
    calibration_origins = [pd.Timestamp(value) for value in calibration_rows["origin_timestamp"].tolist()]
    evaluation_origins = [pd.Timestamp(value) for value in evaluation_rows["origin_timestamp"].tolist()]
    return catalog, calibration_origins, evaluation_origins


def _minute_index_for_cycle(actual_minute_base: pd.DataFrame, origin_timestamp: pd.Timestamp) -> pd.DatetimeIndex:
    """Return the minute-level index covered by one forecast-control cycle."""
    horizon_minutes = int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"])
    start = pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=1)
    end = pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=horizon_minutes)
    rows = actual_minute_base.loc[
        actual_minute_base["timestamp"].between(start, end, inclusive="both"),
        "timestamp",
    ]
    if rows.empty:
        raise RuntimeError(f"No actual minute grid was available for control cycle {origin_timestamp.isoformat()}.")
    return pd.DatetimeIndex(pd.to_datetime(rows, errors="coerce"))


def _day_ahead_refresh_origins(cycle_origins: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Return the hourly checkpoints used to decide whether the 24h path should be refreshed."""
    if not bool(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_enabled"]):
        return []
    interval_minutes = int(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_interval_minutes"])
    horizon_minutes = int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"])
    if interval_minutes <= 0 or interval_minutes >= horizon_minutes:
        return []
    return sorted(
        {
            pd.Timestamp(cycle_origin) + pd.Timedelta(minutes=offset)
            for cycle_origin in cycle_origins
            for offset in range(interval_minutes, horizon_minutes, interval_minutes)
        }
    )


def _layer_update_origins(
    *,
    cycle_origins: list[pd.Timestamp],
    update_interval_minutes: int,
    cycle_horizon_minutes: int,
) -> list[pd.Timestamp]:
    """Expand cycle starts into the exact origin timestamps used by one control layer."""
    if update_interval_minutes <= 0 or cycle_horizon_minutes <= 0 or not cycle_origins:
        return []
    return sorted(
        {
            pd.Timestamp(cycle_origin) + pd.Timedelta(minutes=offset)
            for cycle_origin in cycle_origins
            for offset in range(0, cycle_horizon_minutes, update_interval_minutes)
        }
    )


def _resolve_day_ahead_refresh_candidate_label(day_ahead_selection: dict[str, Any]) -> str:
    """Resolve the exact Stage-7 candidate label used for day-ahead refresh replays."""
    configured = str(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_candidate_label"]).strip()
    if configured and configured.lower() != "auto":
        return configured
    model_label = str(day_ahead_selection.get("model_label", "")).strip()
    if not model_label:
        raise RuntimeError(
            "Day-ahead refresh candidate is set to auto but the selected Stage-7 context has no model_label."
        )
    return f"{model_label}::hybrid_workday_residual"


def _build_day_ahead_refresh_selection(day_ahead_selection: dict[str, Any]) -> dict[str, Any]:
    """Clone the chosen day-ahead replay context and pin it to the refresh candidate label."""
    refresh_selection = dict(day_ahead_selection)
    refresh_selection["requested_candidate_label"] = _resolve_day_ahead_refresh_candidate_label(
        day_ahead_selection
    )
    return refresh_selection


def _replay_day_ahead_refresh_candidate(
    *,
    temp_root: Path,
    cache_root: Path | None,
    day_ahead: dict[str, Any],
    origin_timestamps: list[pd.Timestamp] | None = None,
    benchmark_origin_timestamps: list[pd.Timestamp] | None = None,
    evaluation_origin_timestamps: list[pd.Timestamp] | None = None,
) -> dict[str, Any] | None:
    """Replay the explicit residual-refresh candidate on calibration and held-out checkpoints."""
    if origin_timestamps is not None:
        benchmark_origin_timestamps = list(origin_timestamps)
        evaluation_origin_timestamps = list(origin_timestamps)
    if benchmark_origin_timestamps is None:
        benchmark_origin_timestamps = list(evaluation_origin_timestamps or [])
    if evaluation_origin_timestamps is None:
        evaluation_origin_timestamps = list(benchmark_origin_timestamps or [])
    if not benchmark_origin_timestamps:
        benchmark_origin_timestamps = list(evaluation_origin_timestamps)
    if not evaluation_origin_timestamps:
        evaluation_origin_timestamps = list(benchmark_origin_timestamps)
    if (
        (not benchmark_origin_timestamps and not evaluation_origin_timestamps)
        or not bool(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_enabled"])
    ):
        return None
    selection = _build_day_ahead_refresh_selection(cast(dict[str, Any], day_ahead["selection"]))
    benchmark_origin_timestamps = _representable_selection_origins(
        selection=selection,
        horizon_minutes=int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"]),
        origin_timestamps=benchmark_origin_timestamps,
    )
    evaluation_origin_timestamps = _representable_selection_origins(
        selection=selection,
        horizon_minutes=int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"]),
        origin_timestamps=evaluation_origin_timestamps,
    )
    if not benchmark_origin_timestamps and not evaluation_origin_timestamps:
        return None
    benchmark_result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_root / "benchmark",
        layer_role="day_ahead_refresh",
        selection=selection,
        horizon_minutes=int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"]),
        origin_policy=str(day_ahead["origin_policy"]),
        selection_target=str(day_ahead["selection_target"]),
        origin_timestamps=benchmark_origin_timestamps,
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=False,
    )
    evaluation_result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_root / "evaluation",
        layer_role="day_ahead_refresh",
        selection=selection,
        horizon_minutes=int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"]),
        origin_policy=str(day_ahead["origin_policy"]),
        selection_target=str(day_ahead["selection_target"]),
        origin_timestamps=evaluation_origin_timestamps,
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=True,
    )
    detail_by_origin = evaluation_result["detail_by_origin"].copy()
    available_labels = sorted(
        {
            str(value)
            for value in detail_by_origin.get("candidate_label", pd.Series(dtype="string"))
            .astype("string")
            .dropna()
            .tolist()
            if str(value)
        }
    )
    requested_label = str(selection["requested_candidate_label"])
    if requested_label not in set(available_labels):
        raise RuntimeError(
            "Day-ahead refresh replay did not materialize the requested candidate label "
            f"{requested_label!r}. Available labels: {available_labels}"
        )
    return {
        "selection": selection,
        "candidate_label": requested_label,
        "benchmark_result": {
            "run_dir": benchmark_result["run_dir"],
            "detail_by_origin": benchmark_result["detail_by_origin"].copy(),
        },
        "result": {
            "run_dir": evaluation_result["run_dir"],
            "detail_by_origin": detail_by_origin,
        },
        "benchmark_replay_cache_status": str(benchmark_result.get("replay_cache_status", "")),
        "selected_replay_cache_status": str(evaluation_result.get("replay_cache_status", "")),
        "selected_replay_cache_artifact": _relative_artifact_path(Path(evaluation_result["run_dir"])),
    }


def _scenario_cycle_metrics(
    *,
    minute_frame: pd.DataFrame,
    prediction_column: str,
    lock_interval_minutes: int,
) -> dict[str, float]:
    """Compute minute-path, lock, profile-shape, and energy metrics for one scenario path."""
    actual_minute = minute_frame.set_index("timestamp")["actual_load"].astype(float)
    predicted_minute = minute_frame.set_index("timestamp")[prediction_column].astype(float)
    metrics = _layer_metrics(actual_minute, predicted_minute)
    working = minute_frame.loc[:, ["timestamp", "actual_load", prediction_column]].copy()
    working["interval_start"] = working["timestamp"].dt.floor(f"{lock_interval_minutes}min")
    interval = (
        working.groupby("interval_start", dropna=False)
        .agg(
            actual_interval_mean=("actual_load", "mean"),
            predicted_interval_mean=(prediction_column, "mean"),
        )
        .reset_index()
        .sort_values("interval_start", kind="stable")
    )
    interval_abs_error = (
        interval["actual_interval_mean"].astype(float) - interval["predicted_interval_mean"].astype(float)
    ).abs()
    interval_actual = interval["actual_interval_mean"].astype(float)
    return {
        "minute_path_mae": float(metrics["minute_path_mae"]),
        "minute_path_mae_pct": float(metrics["minute_path_mae_pct"]),
        "lock_mae": float(interval_abs_error.mean()),
        "lock_mae_pct": _safe_pct(
            float(interval_abs_error.sum()),
            float(np.sum(np.abs(interval_actual.to_numpy(dtype=float)))),
        ),
        "profile_shape_mae": float(metrics["profile_shape_mae"]),
        "profile_shape_mae_pct": float(metrics["profile_shape_mae_pct"]),
        "energy_mae": float(metrics["energy_mae"]),
        "energy_mae_pct": float(metrics["energy_mae_pct"]),
    }


def _actual_activity_ratio(feature_window: pd.DataFrame, actual_window: pd.Series) -> float:
    """Estimate the realized activity ratio by projecting actual load onto the stored workday peak."""
    valid = feature_window.loc[
        feature_window["avg_workday_baseline"].notna() & feature_window["profile_activity_ratio"].notna()
    ].copy()
    if valid.empty:
        return float("nan")
    denominator = valid["profile_activity_ratio"].astype(float).replace(0.0, np.nan)
    peak_estimate = (
        valid["avg_workday_baseline"].astype(float).divide(denominator).replace([np.inf, -np.inf], np.nan)
    )
    peak_level = float(peak_estimate.dropna().mean()) if not peak_estimate.dropna().empty else float("nan")
    if not np.isfinite(peak_level) or abs(peak_level) <= 1e-9:
        return float("nan")
    return float(np.mean(np.abs(actual_window.to_numpy(dtype=float))) / abs(peak_level))


def _default_day_ahead_refresh_thresholds() -> dict[str, float]:
    """Return the configured default trigger thresholds for the day-ahead refresh study."""
    return {
        "residual_drift_mae_pct_threshold": float(
            MULTIRES_FORECAST_CONTROL["day_ahead_refresh_residual_drift_mae_pct_threshold"]
        ),
        "transition_mae_pct_threshold": float(
            MULTIRES_FORECAST_CONTROL["day_ahead_refresh_transition_mae_pct_threshold"]
        ),
        "activity_ratio_shift_threshold": float(
            MULTIRES_FORECAST_CONTROL["day_ahead_refresh_activity_ratio_shift_threshold"]
        ),
        "trigger_mode": str(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_trigger_mode"]),
    }


def _apply_day_ahead_refresh_thresholds(
    *,
    signal_row: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply one threshold policy to the raw refresh signal row."""
    effective_thresholds = dict(_default_day_ahead_refresh_thresholds())
    if thresholds is not None:
        for key in (
            "residual_drift_mae_pct_threshold",
            "transition_mae_pct_threshold",
            "activity_ratio_shift_threshold",
            "trigger_mode",
        ):
            value = thresholds.get(key)
            if pd.notna(value):
                effective_thresholds[str(key)] = (
                    str(value) if str(key) == "trigger_mode" else float(value)
                )
    residual_mae_pct = float(signal_row.get("residual_mae_pct", float("nan")))
    transition_active = bool(signal_row.get("workday_transition_active", False))
    transition_state_mismatch = bool(signal_row.get("transition_state_mismatch", False))
    transition_residual_mae_pct = float(signal_row.get("transition_residual_mae_pct", float("nan")))
    activity_ratio_shift = float(signal_row.get("activity_ratio_shift", float("nan")))
    expected_active_flag = float(signal_row.get("expected_active_flag", float("nan")))
    actual_active_flag = float(signal_row.get("actual_active_flag", float("nan")))
    residual_trigger = bool(
        np.isfinite(residual_mae_pct)
        and residual_mae_pct >= float(effective_thresholds["residual_drift_mae_pct_threshold"])
    )
    transition_trigger = bool(
        transition_active
        and transition_state_mismatch
        and np.isfinite(transition_residual_mae_pct)
        and transition_residual_mae_pct >= float(effective_thresholds["transition_mae_pct_threshold"])
    )
    activity_trigger = bool(
        np.isfinite(activity_ratio_shift)
        and activity_ratio_shift >= float(effective_thresholds["activity_ratio_shift_threshold"])
    )
    active_band = bool(
        (np.isfinite(expected_active_flag) and expected_active_flag >= 0.5)
        or (np.isfinite(actual_active_flag) and actual_active_flag >= 0.5)
    )
    trigger_mode = str(effective_thresholds["trigger_mode"]).strip().lower()
    if trigger_mode == "any":
        refresh_triggered = bool(residual_trigger or transition_trigger or activity_trigger)
    elif trigger_mode == "residual_only":
        refresh_triggered = bool(residual_trigger)
    elif trigger_mode == "activity_only":
        refresh_triggered = bool(activity_trigger)
    elif trigger_mode == "activity_active_band":
        refresh_triggered = bool(activity_trigger and active_band)
    elif trigger_mode == "transition_only":
        refresh_triggered = bool(transition_trigger)
    elif trigger_mode == "residual_or_activity":
        refresh_triggered = bool(residual_trigger or activity_trigger)
    elif trigger_mode == "residual_or_activity_active_band":
        refresh_triggered = bool(residual_trigger or (activity_trigger and active_band))
    elif trigger_mode == "residual_or_activity_active_or_transition":
        refresh_triggered = bool(residual_trigger or (activity_trigger and (active_band or transition_active)))
    elif trigger_mode == "residual_or_transition":
        refresh_triggered = bool(residual_trigger or transition_trigger)
    elif trigger_mode == "activity_or_transition":
        refresh_triggered = bool(activity_trigger or transition_trigger)
    elif trigger_mode == "residual_and_activity":
        refresh_triggered = bool(residual_trigger and activity_trigger)
    elif trigger_mode == "residual_and_transition":
        refresh_triggered = bool(residual_trigger and transition_trigger)
    elif trigger_mode == "activity_and_transition":
        refresh_triggered = bool(activity_trigger and transition_trigger)
    elif trigger_mode == "two_of_three":
        refresh_triggered = bool(sum((residual_trigger, transition_trigger, activity_trigger)) >= 2)
    else:
        raise ValueError(f"Unsupported day-ahead refresh trigger mode: {trigger_mode}")
    reasons = [
        label
        for label, enabled in (
            ("residual_drift", residual_trigger),
            ("workday_transition_mismatch", transition_trigger),
            ("activity_profile_shift", activity_trigger),
        )
        if enabled
    ]
    return {
        **signal_row,
        "residual_drift_trigger": bool(residual_trigger),
        "transition_mismatch_trigger": bool(transition_trigger),
        "activity_profile_shift_trigger": bool(activity_trigger),
        "refresh_triggered": bool(refresh_triggered),
        "active_signal_reasons": ",".join(reasons) if reasons else "none",
        "active_signal_count": int(sum((residual_trigger, transition_trigger, activity_trigger))),
        "trigger_reasons": ",".join(reasons) if refresh_triggered and reasons else "none",
        "residual_drift_mae_pct_threshold": float(
            effective_thresholds["residual_drift_mae_pct_threshold"]
        ),
        "transition_mae_pct_threshold": float(
            effective_thresholds["transition_mae_pct_threshold"]
        ),
        "activity_ratio_shift_threshold": float(
            effective_thresholds["activity_ratio_shift_threshold"]
        ),
        "trigger_mode": str(trigger_mode),
    }


def _day_ahead_refresh_signal_row(
    *,
    cycle_origin_timestamp: pd.Timestamp,
    refresh_origin_timestamp: pd.Timestamp,
    minute_feature_frame: pd.DataFrame,
    frozen_forecast: pd.Series,
) -> dict[str, Any]:
    """Compute the raw residual, transition, and activity signals for one refresh checkpoint."""
    default_output = {
        "cycle_origin_timestamp": pd.Timestamp(cycle_origin_timestamp).isoformat(),
        "refresh_origin_timestamp": pd.Timestamp(refresh_origin_timestamp).isoformat(),
        "lookback_start_timestamp": "",
        "lookback_end_timestamp": pd.Timestamp(refresh_origin_timestamp).isoformat(),
        "lookback_observation_n": 0,
        "residual_mae": float("nan"),
        "residual_mae_pct": float("nan"),
        "workday_transition_active": False,
        "transition_residual_mae_pct": float("nan"),
        "expected_active_flag": float("nan"),
        "actual_active_flag": float("nan"),
        "transition_state_mismatch": False,
        "expected_activity_ratio": float("nan"),
        "actual_activity_ratio": float("nan"),
        "activity_ratio_shift": float("nan"),
        "signal_status": "ok",
    }
    lookback_minutes = int(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_lookback_minutes"])
    window_start = max(
        pd.Timestamp(cycle_origin_timestamp) + pd.Timedelta(minutes=1),
        pd.Timestamp(refresh_origin_timestamp) - pd.Timedelta(minutes=lookback_minutes) + pd.Timedelta(minutes=1),
    )
    window_index = frozen_forecast.index[
        (frozen_forecast.index >= window_start) & (frozen_forecast.index <= pd.Timestamp(refresh_origin_timestamp))
    ]
    default_output["lookback_start_timestamp"] = pd.Timestamp(window_start).isoformat()
    if window_index.empty:
        return {**default_output, "signal_status": "insufficient_history"}
    feature_window = minute_feature_frame.reindex(window_index)
    actual_window = feature_window["avg_load"].astype(float)
    predicted_window = frozen_forecast.reindex(window_index).astype(float)
    valid_mask = actual_window.notna() & predicted_window.notna()
    actual_valid = actual_window.loc[valid_mask]
    predicted_valid = predicted_window.loc[valid_mask]
    if actual_valid.empty or predicted_valid.empty:
        return {**default_output, "signal_status": "missing_actuals"}
    residual_abs_error = (actual_valid - predicted_valid).abs()
    transition_mask = feature_window.loc[valid_mask, "workday_transition"].fillna(0.0).astype(float).ge(0.5)
    transition_actual = actual_valid.loc[transition_mask]
    transition_predicted = predicted_valid.loc[transition_mask]
    transition_residual_abs_error = (transition_actual - transition_predicted).abs()
    expected_activity_ratio = float(
        feature_window.loc[valid_mask, "profile_activity_ratio"].astype(float).dropna().mean()
    )
    if "profile_active_flag" in feature_window.columns:
        expected_active_flag = float(
            feature_window.loc[valid_mask, "profile_active_flag"].astype(float).dropna().mean()
        )
    else:
        expected_active_flag = (
            1.0
            if np.isfinite(expected_activity_ratio)
            and expected_activity_ratio >= float(FEATURE_CONFIG["profile_activity_threshold"])
            else 0.0
        )
    actual_activity_ratio = _actual_activity_ratio(
        feature_window.loc[valid_mask].copy(),
        actual_valid,
    )
    profile_activity_threshold = float(FEATURE_CONFIG["profile_activity_threshold"])
    actual_active_flag = float("nan")
    if np.isfinite(actual_activity_ratio):
        actual_active_flag = 1.0 if actual_activity_ratio >= profile_activity_threshold else 0.0
    transition_state_mismatch = bool(
        np.isfinite(expected_active_flag)
        and np.isfinite(actual_active_flag)
        and abs(actual_active_flag - expected_active_flag) >= 0.5
    )
    activity_ratio_shift = (
        float(abs(actual_activity_ratio - expected_activity_ratio))
        if np.isfinite(actual_activity_ratio) and np.isfinite(expected_activity_ratio)
        else float("nan")
    )
    return {
        **default_output,
        "lookback_observation_n": int(valid_mask.sum()),
        "residual_mae": float(residual_abs_error.mean()),
        "residual_mae_pct": _safe_pct(
            float(residual_abs_error.sum()),
            float(np.sum(np.abs(actual_valid.to_numpy(dtype=float)))),
        ),
        "workday_transition_active": bool(
            feature_window.loc[valid_mask, "workday_transition"].fillna(0.0).astype(float).ge(0.5).any()
        ),
        "transition_residual_mae_pct": _safe_pct(
            float(transition_residual_abs_error.sum()),
            float(np.sum(np.abs(transition_actual.to_numpy(dtype=float)))),
        ),
        "expected_active_flag": expected_active_flag,
        "actual_active_flag": actual_active_flag,
        "transition_state_mismatch": transition_state_mismatch,
        "expected_activity_ratio": expected_activity_ratio,
        "actual_activity_ratio": actual_activity_ratio,
        "activity_ratio_shift": activity_ratio_shift,
        "signal_status": "ok",
    }


def _day_ahead_refresh_decision_row(
    *,
    cycle_origin_timestamp: pd.Timestamp,
    refresh_origin_timestamp: pd.Timestamp,
    minute_feature_frame: pd.DataFrame,
    frozen_forecast: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate whether the frozen 24h path should be refreshed at one intraday checkpoint."""
    signal_row = _day_ahead_refresh_signal_row(
        cycle_origin_timestamp=cycle_origin_timestamp,
        refresh_origin_timestamp=refresh_origin_timestamp,
        minute_feature_frame=minute_feature_frame,
        frozen_forecast=frozen_forecast,
    )
    if str(signal_row.get("signal_status", "")) != "ok":
        return {
            **signal_row,
            "residual_drift_trigger": False,
            "transition_mismatch_trigger": False,
            "activity_profile_shift_trigger": False,
            "refresh_triggered": False,
            "trigger_reasons": str(signal_row.get("signal_status", "none")),
            **_default_day_ahead_refresh_thresholds(),
        }
    return _apply_day_ahead_refresh_thresholds(signal_row=signal_row, thresholds=thresholds)


def _selected_learned_candidate_label(by_origin: pd.DataFrame) -> str:
    """Extract the single learned candidate label represented in a rollout replay."""
    learned = by_origin.loc[by_origin["candidate_type"].astype("string").eq("learned"), "candidate_label"]
    labels = sorted(learned.astype("string").dropna().unique().tolist())
    if len(labels) != 1:
        raise RuntimeError(f"Expected exactly one learned candidate in rollout replay, found: {labels}")
    return str(labels[0])


def _control_layer_selection_metric(layer_role: str) -> str:
    """Map a control-layer role to the metric used for candidate promotion."""
    metric_by_role = {
        "day_ahead": str(MULTIRES_FORECAST_CONTROL["day_ahead_selection_metric"]),
        "hourly": str(MULTIRES_FORECAST_CONTROL["hourly_selection_metric"]),
        "phase": str(MULTIRES_FORECAST_CONTROL["phase_selection_metric"]),
    }
    try:
        return metric_by_role[str(layer_role)]
    except KeyError as exc:
        raise ValueError(f"Unsupported control layer role: {layer_role}") from exc


def _control_candidate_metrics_from_detail(
    detail_by_origin: pd.DataFrame,
    *,
    lock_interval_minutes: int,
) -> pd.DataFrame:
    """Score each replayed candidate on optimizer-facing control metrics using detailed paths."""
    if detail_by_origin.empty:
        return pd.DataFrame()
    required_columns = {
        "candidate_label",
        "origin_timestamp",
        "forecast_timestamp",
        "actual_load",
        "predicted_load",
    }
    if not required_columns.issubset(detail_by_origin.columns):
        return pd.DataFrame()
    working = detail_by_origin.copy()
    working["origin_timestamp"] = pd.to_datetime(working["origin_timestamp"], errors="raise")
    working["forecast_timestamp"] = pd.to_datetime(working["forecast_timestamp"], errors="raise")
    rows: list[dict[str, Any]] = []
    for (candidate_label, origin_timestamp), candidate_frame in working.groupby(
        ["candidate_label", "origin_timestamp"],
        sort=False,
        dropna=False,
    ):
        candidate_frame = candidate_frame.sort_values("forecast_timestamp", kind="stable").copy()
        valid = candidate_frame["actual_load"].notna() & candidate_frame["predicted_load"].notna()
        if not bool(valid.any()):
            continue
        scored = candidate_frame.loc[valid, ["forecast_timestamp", "actual_load", "predicted_load"]].copy()
        abs_error = (scored["predicted_load"].astype(float) - scored["actual_load"].astype(float)).abs()
        scored["interval_start"] = pd.to_datetime(scored["forecast_timestamp"], errors="raise").dt.floor(
            f"{int(lock_interval_minutes)}min"
        )
        interval = (
            scored.groupby("interval_start", dropna=False)
            .agg(
                actual_interval_mean=("actual_load", "mean"),
                predicted_interval_mean=("predicted_load", "mean"),
            )
            .reset_index()
            .sort_values("interval_start", kind="stable")
        )
        interval_abs_error = (
            interval["actual_interval_mean"].astype(float) - interval["predicted_interval_mean"].astype(float)
        ).abs()
        interval_actual = interval["actual_interval_mean"].astype(float)
        next_lock_mae = float(interval_abs_error.iloc[0]) if not interval_abs_error.empty else float("nan")
        next_lock_actual = float(abs(interval_actual.iloc[0])) if not interval_actual.empty else float("nan")
        peak_value_mae = float("nan")
        peak_value_mae_pct = float("nan")
        peak_interval_hit_rate = float("nan")
        peak_interval_offset_minutes = float("nan")
        if not interval.empty:
            actual_peak_idx = interval["actual_interval_mean"].astype(float).idxmax()
            predicted_peak_idx = interval["predicted_interval_mean"].astype(float).idxmax()
            actual_peak_start = pd.Timestamp(interval.loc[actual_peak_idx, "interval_start"])
            predicted_peak_start = pd.Timestamp(interval.loc[predicted_peak_idx, "interval_start"])
            actual_peak_value = float(interval.loc[actual_peak_idx, "actual_interval_mean"])
            predicted_peak_value = float(interval.loc[predicted_peak_idx, "predicted_interval_mean"])
            peak_value_mae = float(abs(predicted_peak_value - actual_peak_value))
            peak_value_mae_pct = _safe_pct(peak_value_mae, abs(actual_peak_value))
            peak_interval_hit_rate = float(actual_peak_start == predicted_peak_start)
            peak_interval_offset_minutes = float(
                abs((predicted_peak_start - actual_peak_start) / pd.Timedelta(minutes=1))
            )
        lock_mae = float(interval_abs_error.mean()) if not interval_abs_error.empty else float("nan")
        lock_mae_pct = _safe_pct(
            float(interval_abs_error.sum()),
            float(np.sum(np.abs(interval_actual.to_numpy(dtype=float)))) if not interval_actual.empty else float("nan"),
        )
        next_lock_mae_pct = _safe_pct(next_lock_mae, next_lock_actual)
        peak_interval_miss_rate = (
            float(1.0 - peak_interval_hit_rate) if np.isfinite(peak_interval_hit_rate) else float("nan")
        )
        rows.append(
            {
                "candidate_label": str(candidate_label),
                "origin_timestamp": pd.Timestamp(origin_timestamp).isoformat(),
                "minute_path_mae": float(abs_error.mean()),
                "minute_path_mae_pct": _safe_pct(
                    float(abs_error.sum()),
                    float(np.sum(np.abs(scored["actual_load"].astype(float).to_numpy(dtype=float)))),
                ),
                "lock_mae": lock_mae,
                "lock_mae_pct": lock_mae_pct,
                "next_lock_mae": next_lock_mae,
                "next_lock_mae_pct": next_lock_mae_pct,
                "peak_value_mae": peak_value_mae,
                "peak_value_mae_pct": peak_value_mae_pct,
                "peak_interval_hit_rate": peak_interval_hit_rate,
                "peak_interval_miss_rate": peak_interval_miss_rate,
                "peak_interval_offset_minutes": peak_interval_offset_minutes,
                "optimizer_score": _optimizer_score_from_components(
                    next_lock_mae_pct=next_lock_mae_pct,
                    lock_mae_pct=lock_mae_pct,
                    peak_value_mae_pct=peak_value_mae_pct,
                    peak_interval_miss_rate=peak_interval_miss_rate,
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_control_candidate_metrics(by_origin: pd.DataFrame) -> pd.DataFrame:
    """Aggregate control-aligned candidate metrics across benchmark origins."""
    metric_columns = [
        "minute_path_mae",
        "minute_path_mae_pct",
        "lock_mae",
        "lock_mae_pct",
        "next_lock_mae",
        "next_lock_mae_pct",
        "peak_value_mae",
        "peak_value_mae_pct",
        "peak_interval_hit_rate",
        "peak_interval_miss_rate",
        "peak_interval_offset_minutes",
        "optimizer_score",
    ]
    available_columns = [column for column in metric_columns if column in by_origin.columns]
    if not available_columns or by_origin.empty:
        return pd.DataFrame(columns=["candidate_label"])
    grouped = (
        by_origin.groupby("candidate_label", dropna=False)[available_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    return grouped


def _overlay_candidate_metrics(base: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """Merge candidate-level metrics while letting the overlay win when it provides a value."""
    if base.empty or overlay.empty:
        return base.copy()
    suffix = "__overlay"
    merged = base.merge(overlay, on="candidate_label", how="left", suffixes=("", suffix))
    overlay_columns = [column for column in overlay.columns if column != "candidate_label"]
    for column in overlay_columns:
        overlay_column = f"{column}{suffix}"
        if overlay_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[overlay_column].combine_first(merged[column])
        else:
            merged[column] = merged[overlay_column]
        merged = merged.drop(columns=[overlay_column])
    return merged


def _ensure_selection_metric_column(
    frame: pd.DataFrame,
    *,
    selection_metric: str,
    selection_metric_value_column: str | None = None,
    fallback_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Backfill a configured selection metric from compatible fallbacks when fixtures are sparse."""
    working = frame.copy()
    if selection_metric in working.columns:
        working[selection_metric] = pd.to_numeric(working[selection_metric], errors="coerce")
        return working
    if selection_metric_value_column and selection_metric_value_column in working.columns:
        working[selection_metric] = pd.to_numeric(working[selection_metric_value_column], errors="coerce")
        return working
    if selection_metric == "optimizer_score":
        optimizer_pct_columns = [
            "next_lock_mae_pct",
            "lock_mae_pct",
            "peak_value_mae_pct",
            "peak_interval_miss_rate",
        ]
        if all(column in working.columns for column in optimizer_pct_columns):
            working[selection_metric] = working.apply(
                lambda row: _optimizer_score_from_components(
                    next_lock_mae_pct=float(row.get("next_lock_mae_pct", float("nan"))),
                    lock_mae_pct=float(row.get("lock_mae_pct", float("nan"))),
                    peak_value_mae_pct=float(row.get("peak_value_mae_pct", float("nan"))),
                    peak_interval_miss_rate=float(row.get("peak_interval_miss_rate", float("nan"))),
                ),
                axis=1,
            )
            return working
    fallback_order = list(fallback_metrics or [])
    if selection_metric not in fallback_order:
        fallback_order.append(selection_metric)
    for column in fallback_order:
        if column in working.columns:
            working[selection_metric] = pd.to_numeric(working[column], errors="coerce")
            return working
    working[selection_metric] = float("inf")
    return working


def _ensure_sort_columns(
    frame: pd.DataFrame,
    *,
    sort_columns: list[str],
    selection_metric: str | None = None,
    selection_metric_value_column: str | None = None,
) -> pd.DataFrame:
    """Populate deterministic sort columns so lean benchmark fixtures still rank safely."""
    working = frame.copy()
    for column in sort_columns:
        if column in working.columns:
            if column != "candidate_label":
                working[column] = pd.to_numeric(working[column], errors="coerce")
            continue
        if column == "candidate_label":
            working[column] = working.index.astype(str)
            continue
        fallback_names: list[str] = []
        if column.startswith("evaluation_"):
            fallback_names.append(column[len("evaluation_") :])
        if column.endswith("_p50") or column.endswith("_p90"):
            base_column = column.rsplit("_", 1)[0]
            fallback_names.append(base_column)
            if base_column.startswith("evaluation_"):
                fallback_names.append(base_column[len("evaluation_") :])
        if selection_metric is not None:
            if column == selection_metric or column == f"evaluation_{selection_metric}":
                if selection_metric_value_column:
                    fallback_names.append(selection_metric_value_column)
            fallback_names.append(selection_metric)
        if selection_metric_value_column:
            fallback_names.append(selection_metric_value_column)
        fallback_names.extend(["selection_metric_value", "evaluation_selection_metric_value"])
        fallback_series: pd.Series | None = None
        seen: set[str] = set()
        for candidate_name in fallback_names:
            if not candidate_name or candidate_name in seen or candidate_name == "candidate_label":
                continue
            seen.add(candidate_name)
            if candidate_name in working.columns:
                fallback_series = pd.to_numeric(working[candidate_name], errors="coerce")
                break
        if fallback_series is None:
            working[column] = float("inf")
        else:
            working[column] = fallback_series
    return working


def _control_layer_sort_columns(metric_name: str) -> list[str]:
    """Return the deterministic sort order used for layer-level candidate ranking."""
    primary = [f"{metric_name}_p50", f"{metric_name}_p90", metric_name]
    if metric_name == "optimizer_score":
        return primary + [
            "next_lock_mae_p50",
            "peak_interval_miss_rate_p50",
            "peak_value_mae_p50",
            "lock_mae_p50",
            "candidate_label",
        ]
    if metric_name == "lock_mae":
        return primary + [
            "next_lock_mae_p50",
            "peak_interval_miss_rate_p50",
            "peak_value_mae_p50",
            "candidate_label",
        ]
    if metric_name == "peak_value_mae":
        return primary + [
            "peak_interval_miss_rate_p50",
            "next_lock_mae_p50",
            "lock_mae_p50",
            "candidate_label",
        ]
    if metric_name == "peak_interval_miss_rate":
        return primary + [
            "next_lock_mae_p50",
            "peak_value_mae_p50",
            "lock_mae_p50",
            "candidate_label",
        ]
    if metric_name == "profile_shape_mae":
        return primary + ["path_mae_p50", "path_mae", "next_lock_mae_p50", "next_lock_mae", "candidate_label"]
    if metric_name == "next_lock_mae":
        return primary + [
            "profile_shape_mae_p50",
            "profile_shape_mae",
            "path_mae_p50",
            "path_mae",
            "candidate_label",
        ]
    if metric_name == "path_mae":
        return primary + [
            "profile_shape_mae_p50",
            "profile_shape_mae",
            "next_lock_mae_p50",
            "next_lock_mae",
            "candidate_label",
        ]
    if metric_name == "phase_mean_mae":
        return primary + ["next_lock_mae_p50", "next_lock_mae", "path_mae_p50", "path_mae", "candidate_label"]
    if metric_name == "endpoint_mae":
        return primary + ["next_lock_mae_p50", "next_lock_mae", "path_mae_p50", "path_mae", "candidate_label"]
    if metric_name == "energy_mae":
        return primary + [
            "profile_shape_mae_p50",
            "profile_shape_mae",
            "path_mae_p50",
            "path_mae",
            "candidate_label",
        ]
    raise ValueError(f"Unsupported control-layer metric: {metric_name}")


def _control_layer_distribution_metrics(by_origin: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate stability across control origins using median and tail metrics."""
    metric_columns = [
        "endpoint_mae",
        "path_mae",
        "phase_mean_mae",
        "minute_path_mae",
        "lock_mae",
        "next_lock_mae",
        "peak_value_mae",
        "peak_interval_miss_rate",
        "peak_interval_hit_rate",
        "peak_interval_offset_minutes",
        "profile_shape_mae",
        "energy_mae",
        "optimizer_score",
    ]
    working = by_origin.copy()
    rows: list[dict[str, Any]] = []
    for candidate_label, candidate_frame in working.groupby("candidate_label", sort=False):
        row: dict[str, Any] = {"candidate_label": str(candidate_label)}
        for metric_name in metric_columns:
            if metric_name not in candidate_frame.columns:
                continue
            values = pd.to_numeric(candidate_frame[metric_name], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric_name}_p50"] = float(values.quantile(0.5))
            row[f"{metric_name}_p90"] = float(values.quantile(0.9))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["candidate_label"])
    return pd.DataFrame(rows)


def _rename_control_layer_evaluation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename one rollout-layer benchmark table so calibration and evaluation can be compared side by side."""
    rename_map = {
        "endpoint_mae": "evaluation_endpoint_mae",
        "endpoint_mae_pct": "evaluation_endpoint_mae_pct",
        "path_mae": "evaluation_path_mae",
        "path_mae_pct": "evaluation_path_mae_pct",
        "phase_mean_mae": "evaluation_phase_mean_mae",
        "phase_mean_mae_pct": "evaluation_phase_mean_mae_pct",
        "minute_path_mae": "evaluation_minute_path_mae",
        "minute_path_mae_pct": "evaluation_minute_path_mae_pct",
        "lock_mae": "evaluation_lock_mae",
        "lock_mae_pct": "evaluation_lock_mae_pct",
        "next_lock_mae": "evaluation_next_lock_mae",
        "next_lock_mae_pct": "evaluation_next_lock_mae_pct",
        "peak_value_mae": "evaluation_peak_value_mae",
        "peak_value_mae_pct": "evaluation_peak_value_mae_pct",
        "peak_interval_hit_rate": "evaluation_peak_interval_hit_rate",
        "peak_interval_miss_rate": "evaluation_peak_interval_miss_rate",
        "peak_interval_offset_minutes": "evaluation_peak_interval_offset_minutes",
        "profile_shape_mae": "evaluation_profile_shape_mae",
        "profile_shape_mae_pct": "evaluation_profile_shape_mae_pct",
        "energy_mae": "evaluation_energy_mae",
        "energy_mae_pct": "evaluation_energy_mae_pct",
        "optimizer_score": "evaluation_optimizer_score",
        "mean_coverage": "evaluation_mean_coverage",
        "origin_n": "evaluation_origin_n",
        "selection_metric_name": "evaluation_selection_metric_name",
        "selection_metric_value": "evaluation_selection_metric_value",
        "selection_metric_pct": "evaluation_selection_metric_pct",
        "stack_bucket_policy_json": "evaluation_stack_bucket_policy_json",
        "stack_bucket_granularity_minutes": "evaluation_stack_bucket_granularity_minutes",
    }
    for metric_name in (
        "endpoint_mae",
        "path_mae",
        "phase_mean_mae",
        "minute_path_mae",
        "lock_mae",
        "next_lock_mae",
        "peak_value_mae",
        "peak_interval_miss_rate",
        "peak_interval_hit_rate",
        "peak_interval_offset_minutes",
        "profile_shape_mae",
        "energy_mae",
        "optimizer_score",
    ):
        rename_map[f"{metric_name}_p50"] = f"evaluation_{metric_name}_p50"
        rename_map[f"{metric_name}_p90"] = f"evaluation_{metric_name}_p90"
    keep_columns = ["candidate_label"] + [column for column in rename_map if column in frame.columns]
    return frame.loc[:, keep_columns].rename(columns=rename_map)


def _select_control_layer_candidate(
    *,
    calibration_benchmark: pd.DataFrame,
    evaluation_benchmark: pd.DataFrame,
    upstream_label: str,
) -> tuple[pd.Series, str]:
    """Choose the candidate that should drive the held-out control replay for one rollout layer."""
    selection_scope = str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"])
    selection_mode = "control_layer_candidate_benchmark"
    selected = calibration_benchmark.iloc[0]
    if (
        selection_scope == "held_out_evaluation"
        and bool(MULTIRES_FORECAST_CONTROL["optimize_replayed_candidates"])
        and not evaluation_benchmark.empty
    ):
        selection_mode = "held_out_control_layer_candidate_benchmark"
        guarded_evaluation = _apply_optimizer_promotion_guard(
            evaluation_benchmark,
            upstream_label=str(upstream_label),
        )
        if len(guarded_evaluation) != len(evaluation_benchmark):
            selection_mode = "held_out_control_layer_candidate_benchmark_guarded"
        return guarded_evaluation.iloc[0], selection_mode
    if not bool(MULTIRES_FORECAST_CONTROL["optimize_replayed_candidates"]):
        selection_mode = "upstream_selection"
        matched = calibration_benchmark.loc[
            calibration_benchmark["candidate_label"].astype("string").eq(str(upstream_label))
        ].copy()
        if not matched.empty:
            selected = matched.iloc[0]
    else:
        guarded_calibration = _apply_optimizer_promotion_guard(
            calibration_benchmark,
            upstream_label=str(upstream_label),
        )
        if len(guarded_calibration) != len(calibration_benchmark):
            selection_mode = "control_layer_candidate_benchmark_guarded"
        selected = guarded_calibration.iloc[0]
    return selected, selection_mode


def _rollout_registry_metric_fields(selection_target: str) -> dict[str, str]:
    """Map a control-layer selection target to the matching rollout-registry fields."""
    if selection_target == "profile_shape_mae":
        return {
            "metric": "learned_profile_shape_mae",
            "beat_baseline": "beats_best_baseline_profile_shape",
            "beat_persistence": "beats_persistence_profile_shape",
        }
    if selection_target == "next_lock_mae":
        return {
            "metric": "learned_next_lock_mae",
            "beat_baseline": "beats_best_baseline_next_lock",
            "beat_persistence": "beats_persistence_next_lock",
        }
    if selection_target == "phase_mean_mae":
        return {
            "metric": "learned_phase_mean_mae",
            "beat_baseline": "beats_best_baseline_phase",
            "beat_persistence": "beats_persistence_phase",
        }
    if selection_target == "path_mae":
        return {
            "metric": "learned_path_mae",
            "beat_baseline": "beats_best_baseline_path",
            "beat_persistence": "beats_persistence_path",
        }
    if selection_target == "endpoint_mae":
        return {
            "metric": "learned_endpoint_mae",
            "beat_baseline": "beats_best_baseline_endpoint",
            "beat_persistence": "beats_persistence_endpoint",
        }
    if selection_target == "energy_mae":
        return {
            "metric": "learned_energy_mae",
            "beat_baseline": "beats_best_baseline_phase",
            "beat_persistence": "beats_persistence_phase",
        }
    raise ValueError(f"Unsupported control-layer registry metric: {selection_target}")


def _control_candidate_signature(selection: dict[str, Any]) -> tuple[str, str, str, str]:
    """Build a stable signature for one replayable control candidate selection."""
    return (
        str(selection.get("resolution", "")),
        str(selection.get("feature_set", "")),
        str(selection.get("model_label", "")),
        str(selection.get("portfolio_candidate_label", "")),
    )


def _qualify_control_candidate_label(selection: dict[str, Any], candidate_label: str) -> str:
    """Prefix a native rollout candidate label with its replay context for control comparisons."""
    resolution = str(selection.get("resolution", "unknown"))
    feature_set = str(selection.get("feature_set", "unknown"))
    return f"{resolution}/{feature_set}/{candidate_label}"


def _native_control_candidate_label(selection: dict[str, Any], qualified_candidate_label: str) -> str:
    """Strip the control-layer replay prefix to recover the native Stage-7 candidate label."""
    prefix = _qualify_control_candidate_label(selection, "")
    if qualified_candidate_label.startswith(prefix):
        return qualified_candidate_label[len(prefix) :]
    return qualified_candidate_label


def _replay_cache_root(output_root: Path) -> Path:
    """Return the stable cache root used to store exact-origin replay snapshots."""
    return output_root / str(MULTIRES_FORECAST_CONTROL["replay_cache_dirname"])


def _replay_cache_key(
    *,
    layer_role: str,
    selection: dict[str, Any],
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    origin_timestamps: list[pd.Timestamp],
    capture_path_details: bool,
    candidate_scope: str,
) -> str:
    """Hash the replay inputs that make one control-layer rollout exactly reusable."""
    payload = {
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "layer_role": str(layer_role),
        "selection": selection,
        "horizon_minutes": int(horizon_minutes),
        "origin_policy": str(origin_policy),
        "selection_target": str(selection_target),
        "origin_timestamps": [pd.Timestamp(value).isoformat() for value in origin_timestamps],
        "capture_path_details": bool(capture_path_details),
        "candidate_scope": str(candidate_scope),
    }
    return stable_config_hash(payload).removeprefix("sha256:")


def _replay_cache_dir(*, cache_root: Path, layer_role: str, cache_key: str) -> Path:
    """Resolve the deterministic cache directory for one exact-origin replay."""
    return cache_root / str(layer_role) / str(cache_key)


def _replay_cache_manifest_path(cache_dir: Path) -> Path:
    """Return the manifest path used to audit one cached exact-origin replay."""
    return cache_dir / "replay_cache_manifest.json"


def _refresh_replay_cache_registry(cache_root: Path) -> None:
    """Write a registry snapshot summarizing all cached control replays."""
    manifest_paths = sorted(cache_root.glob("*/*/replay_cache_manifest.json"))
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "layer_role": str(payload.get("layer_role", "")),
                "cache_key": str(payload.get("cache_key", "")),
                "resolution": str(payload.get("resolution", "")),
                "feature_set": str(payload.get("feature_set", "")),
                "model_label": str(payload.get("model_label", "")),
                "requested_candidate_label": str(payload.get("requested_candidate_label", "")),
                "portfolio_candidate_label": str(payload.get("portfolio_candidate_label", "")),
                "horizon_minutes": int(payload.get("horizon_minutes", 0)),
                "origin_policy": str(payload.get("origin_policy", "")),
                "selection_target": str(payload.get("selection_target", "")),
                "origin_count": int(payload.get("origin_count", 0)),
                "candidate_scope": str(payload.get("candidate_scope", "")),
                "capture_path_details": bool(payload.get("capture_path_details", False)),
                "generated_at_utc": str(payload.get("generated_at_utc", "")),
                "last_accessed_at_utc": str(payload.get("last_accessed_at_utc", "")),
                "cache_dir": str(payload.get("cache_dir", "")),
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        registry = pd.DataFrame(
            columns=[
                "layer_role",
                "cache_key",
                "resolution",
                "feature_set",
                "model_label",
                "requested_candidate_label",
                "portfolio_candidate_label",
                "horizon_minutes",
                "origin_policy",
                "selection_target",
                "origin_count",
                "candidate_scope",
                "capture_path_details",
                "generated_at_utc",
                "last_accessed_at_utc",
                "cache_dir",
            ]
        )
    registry.to_csv(cache_root / "replay_cache_registry.csv", index=False)


def _resolve_cache_dir(cache_dir_value: str) -> Path:
    """Resolve one registry cache-dir value into an absolute path."""
    cache_dir = Path(str(cache_dir_value))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    return cache_dir.resolve()


def _subset_cached_rollout_result(
    cached: dict[str, Any],
    *,
    origin_timestamps: list[pd.Timestamp],
    require_path_details: bool,
) -> dict[str, Any] | None:
    """Filter a covering replay-cache hit down to the requested explicit origins."""
    requested_origin_labels = {
        pd.Timestamp(value).isoformat() for value in origin_timestamps
    }
    selected_origins = cast(pd.DataFrame, cached["selected_origins"]).copy()
    selected_origins = selected_origins.loc[
        selected_origins["origin_timestamp"].astype("string").isin(requested_origin_labels)
    ].reset_index(drop=True)
    if len(selected_origins) != len(requested_origin_labels):
        return None

    by_origin = cast(pd.DataFrame, cached["by_origin"]).copy()
    by_origin = by_origin.loc[
        by_origin["origin_timestamp"].astype("string").isin(requested_origin_labels)
    ].reset_index(drop=True)
    if by_origin.empty:
        return None

    detail_by_origin = cast(pd.DataFrame, cached.get("detail_by_origin", pd.DataFrame())).copy()
    if require_path_details:
        if detail_by_origin.empty:
            return None
        detail_by_origin = detail_by_origin.loc[
            detail_by_origin["origin_timestamp"].astype("string").isin(requested_origin_labels)
        ].reset_index(drop=True)
        detail_origin_labels = set(detail_by_origin["origin_timestamp"].astype("string"))
        if not requested_origin_labels.issubset(detail_origin_labels):
            return None
    elif not detail_by_origin.empty:
        detail_by_origin = detail_by_origin.loc[
            detail_by_origin["origin_timestamp"].astype("string").isin(requested_origin_labels)
        ].reset_index(drop=True)

    subset = dict(cached)
    subset["by_origin"] = by_origin
    subset["selected_origins"] = selected_origins
    subset["detail_by_origin"] = detail_by_origin
    subset["replay_cache_status"] = "subset_hit"
    subset["replay_cache_source_origin_count"] = int(len(cast(pd.DataFrame, cached["selected_origins"])))
    return subset


def _load_covering_cached_rollout_result(
    *,
    cache_root: Path,
    layer_role: str,
    selection: dict[str, Any],
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    origin_timestamps: list[pd.Timestamp],
    require_path_details: bool,
    candidate_scope: str,
) -> dict[str, Any] | None:
    """Reuse a larger exact-origin replay when it fully covers the requested subset.

    Stage-10 often benchmarks a candidate on a wider explicit-origin sample and
    then reruns the same candidate on a smaller selected subset. When the wider
    cache already contains every requested origin, filtering that cached replay
    is faster and preserves the same candidate behavior without recomputation.
    """
    registry_path = cache_root / "replay_cache_registry.csv"
    if not registry_path.exists():
        return None
    registry = _read_csv_if_present(registry_path)
    if registry.empty:
        return None

    requested_candidate_label = str(selection.get("requested_candidate_label", ""))
    portfolio_candidate_label = str(selection.get("portfolio_candidate_label", ""))
    registry = registry.loc[
        registry["layer_role"].astype("string").fillna("").eq(str(layer_role))
        & registry["resolution"].astype("string").fillna("").eq(str(selection.get("resolution", "")))
        & registry["feature_set"].astype("string").fillna("").eq(str(selection.get("feature_set", "")))
        & registry["model_label"].astype("string").fillna("").eq(str(selection.get("model_label", "")))
        & registry["requested_candidate_label"].astype("string").fillna("").eq(requested_candidate_label)
        & registry["portfolio_candidate_label"].astype("string").fillna("").eq(portfolio_candidate_label)
        & registry["origin_policy"].astype("string").fillna("").eq(str(origin_policy))
        & registry["selection_target"].astype("string").fillna("").eq(str(selection_target))
        & registry["candidate_scope"].astype("string").fillna("").eq(str(candidate_scope))
        & registry["horizon_minutes"].astype(int).eq(int(horizon_minutes))
    ].copy()
    if registry.empty:
        return None

    if require_path_details:
        registry = registry.loc[
            registry["capture_path_details"]
            .astype("string")
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .astype(bool)
        ].copy()
        if registry.empty:
            return None

    registry = registry.sort_values(
        ["origin_count", "last_accessed_at_utc", "generated_at_utc"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    for row in registry.itertuples(index=False):
        cache_dir = _resolve_cache_dir(str(getattr(row, "cache_dir", "")))
        cached = _load_cached_rollout_result(
            cache_dir=cache_dir,
            require_path_details=bool(require_path_details),
        )
        if cached is None:
            continue
        subset = _subset_cached_rollout_result(
            cached,
            origin_timestamps=origin_timestamps,
            require_path_details=bool(require_path_details),
        )
        if subset is None:
            continue
        _touch_replay_cache_entry(cache_dir, cache_root)
        return subset
    return None


def _load_cached_rollout_result(
    *,
    cache_dir: Path,
    require_path_details: bool,
) -> dict[str, Any] | None:
    """Load a cached exact-origin replay snapshot when all required artifacts exist."""
    manifest_path = _replay_cache_manifest_path(cache_dir)
    by_origin_path = cache_dir / "recursive_rollout_by_origin.csv"
    selected_origins_path = cache_dir / "selected_origins.csv"
    detail_path = cache_dir / "recursive_rollout_detail_by_origin.csv"
    if not manifest_path.exists() or not by_origin_path.exists() or not selected_origins_path.exists():
        return None
    if require_path_details and not detail_path.exists():
        return None
    try:
        cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    by_origin = _read_csv_if_present(by_origin_path)
    selected_origins = _read_csv_if_present(selected_origins_path)
    detail_by_origin = _read_csv_if_present(detail_path) if detail_path.exists() else pd.DataFrame()
    metrics = _read_csv_if_present(cache_dir / "recursive_rollout_metrics.csv")
    selection_summary = _read_csv_if_present(cache_dir / "rollout_selection_summary.csv")
    rollout_health = _read_csv_if_present(cache_dir / "rollout_health.csv")
    if by_origin.empty or selected_origins.empty:
        return None
    return {
        "run_dir": cache_dir,
        "metrics": metrics,
        "by_origin": by_origin,
        "selected_origins": selected_origins,
        "selection_summary": selection_summary,
        "rollout_health": rollout_health,
        "manifest": cache_manifest.get("source_manifest", {}),
        "selection": cache_manifest.get("selection", {}),
        "detail_by_origin": detail_by_origin,
        "replay_cache_status": "hit",
        "replay_cache_key": str(cache_manifest.get("cache_key", "")),
    }


def _persist_cached_rollout_result(
    *,
    cache_dir: Path,
    cache_key: str,
    cache_root: Path,
    layer_role: str,
    selection: dict[str, Any],
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    origin_timestamps: list[pd.Timestamp],
    capture_path_details: bool,
    candidate_scope: str,
    result: dict[str, Any],
) -> None:
    """Persist the minimal Stage-7 artifacts needed to reuse one exact-origin replay."""
    with _REPLAY_CACHE_LOCK:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        result["by_origin"].to_csv(cache_dir / "recursive_rollout_by_origin.csv", index=False, float_format="%.6f")
        result["selected_origins"].to_csv(cache_dir / "selected_origins.csv", index=False)
        metrics = result.get("metrics", pd.DataFrame())
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            metrics.to_csv(cache_dir / "recursive_rollout_metrics.csv", index=False, float_format="%.6f")
        selection_summary = result.get("selection_summary", pd.DataFrame())
        if isinstance(selection_summary, pd.DataFrame) and not selection_summary.empty:
            selection_summary.to_csv(cache_dir / "rollout_selection_summary.csv", index=False, float_format="%.6f")
        rollout_health = result.get("rollout_health", pd.DataFrame())
        if isinstance(rollout_health, pd.DataFrame) and not rollout_health.empty:
            rollout_health.to_csv(cache_dir / "rollout_health.csv", index=False, float_format="%.6f")
        detail_by_origin = result.get("detail_by_origin", pd.DataFrame())
        if isinstance(detail_by_origin, pd.DataFrame) and not detail_by_origin.empty:
            detail_by_origin.to_csv(
                cache_dir / "recursive_rollout_detail_by_origin.csv",
                index=False,
                float_format="%.6f",
            )
        (cache_dir / "selection_context.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        timestamp = datetime.now(UTC).isoformat()
        cache_manifest = {
            "cache_key": str(cache_key),
            "layer_role": str(layer_role),
            "load_type": DATASET["load_type"],
            "artifact_namespace": DATASET["artifact_namespace"],
            "resolution": str(selection.get("resolution", "")),
            "feature_set": str(selection.get("feature_set", "")),
            "model_label": str(selection.get("model_label", "")),
            "requested_candidate_label": str(selection.get("requested_candidate_label", "")),
            "portfolio_candidate_label": str(selection.get("portfolio_candidate_label", "")),
            "horizon_minutes": int(horizon_minutes),
            "origin_policy": str(origin_policy),
            "selection_target": str(selection_target),
            "origin_count": int(len(origin_timestamps)),
            "candidate_scope": str(candidate_scope),
            "capture_path_details": bool(capture_path_details),
            "generated_at_utc": timestamp,
            "last_accessed_at_utc": timestamp,
            "cache_dir": _relative_artifact_path(cache_dir),
            "selection": selection,
            "source_manifest": result.get("manifest", {}),
        }
        _replay_cache_manifest_path(cache_dir).write_text(
            json.dumps(cache_manifest, indent=2),
            encoding="utf-8",
        )
        _refresh_replay_cache_registry(cache_root)


def _touch_replay_cache_entry(cache_dir: Path, cache_root: Path) -> None:
    """Update the cache manifest access time after a successful cache hit."""
    with _REPLAY_CACHE_LOCK:
        manifest_path = _replay_cache_manifest_path(cache_dir)
        if not manifest_path.exists():
            return
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        payload["last_accessed_at_utc"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _refresh_replay_cache_registry(cache_root)


def _run_cached_rollout_evaluation(
    *,
    cache_root: Path | None,
    temp_output_root: Path,
    layer_role: str,
    selection: dict[str, Any],
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    origin_timestamps: list[pd.Timestamp],
    capture_path_details: bool,
    candidate_scope: str,
    persist_artifacts: bool,
) -> dict[str, Any]:
    """Execute or load one exact-origin rollout replay, using a deterministic cache when enabled."""
    if cache_root is None:
        result = run_rollout_evaluation(
            output_root=temp_output_root,
            selection=selection,
            horizon_minutes=int(horizon_minutes),
            origins=len(origin_timestamps),
            origin_policy=str(origin_policy),
            selection_target=str(selection_target),
            origin_timestamps=origin_timestamps,
            capture_path_details=bool(capture_path_details),
            candidate_scope=str(candidate_scope),
            persist_artifacts=bool(persist_artifacts),
            refresh_root_registry=False,
            refresh_latest_alias=False,
        )
        result["replay_cache_status"] = "disabled"
        return result

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = _replay_cache_key(
        layer_role=str(layer_role),
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=origin_timestamps,
        capture_path_details=bool(capture_path_details),
        candidate_scope=str(candidate_scope),
    )
    cache_dir = _replay_cache_dir(cache_root=cache_root, layer_role=str(layer_role), cache_key=cache_key)
    cached = _load_cached_rollout_result(
        cache_dir=cache_dir,
        require_path_details=bool(capture_path_details),
    )
    if cached is not None:
        _touch_replay_cache_entry(cache_dir, cache_root)
        return cached
    covering_cached = _load_covering_cached_rollout_result(
        cache_root=cache_root,
        layer_role=str(layer_role),
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=origin_timestamps,
        require_path_details=bool(capture_path_details),
        candidate_scope=str(candidate_scope),
    )
    if covering_cached is not None:
        return covering_cached

    result = run_rollout_evaluation(
        output_root=temp_output_root,
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origins=len(origin_timestamps),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=origin_timestamps,
        capture_path_details=bool(capture_path_details),
        candidate_scope=str(candidate_scope),
        persist_artifacts=bool(persist_artifacts),
        refresh_root_registry=False,
        refresh_latest_alias=False,
    )
    _persist_cached_rollout_result(
        cache_dir=cache_dir,
        cache_key=cache_key,
        cache_root=cache_root,
        layer_role=str(layer_role),
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=origin_timestamps,
        capture_path_details=bool(capture_path_details),
        candidate_scope=str(candidate_scope),
        result=result,
    )
    cached = _load_cached_rollout_result(
        cache_dir=cache_dir,
        require_path_details=bool(capture_path_details),
    )
    if cached is None:
        result["run_dir"] = cache_dir
        result["replay_cache_status"] = "miss"
        result["replay_cache_key"] = cache_key
        return result
    cached["replay_cache_status"] = "miss"
    cached["replay_cache_key"] = cache_key
    return cached


def _sample_control_origins(origin_timestamps: list[pd.Timestamp], *, cap: int) -> list[pd.Timestamp]:
    """Subsample replay origins deterministically for candidate-pool benchmarking."""
    if len(origin_timestamps) <= cap:
        return list(origin_timestamps)
    positions = np.linspace(0, len(origin_timestamps) - 1, num=cap, dtype=int)
    selected: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for position in positions.tolist():
        timestamp = pd.Timestamp(origin_timestamps[int(position)])
        if timestamp not in seen:
            seen.add(timestamp)
            selected.append(timestamp)
    return selected


def _sample_control_benchmark_origins(origin_timestamps: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Subsample replay origins using the default candidate-pool cap."""
    return _sample_control_origins(
        origin_timestamps,
        cap=int(MULTIRES_FORECAST_CONTROL["candidate_benchmark_origin_cap"]),
    )


def _is_timestamped_run_dir_name(name: str) -> bool:
    """Return whether one directory name looks like a timestamped run id."""
    text = str(name).strip()
    if len(text) < 11 or not text.endswith("Z") or "T" not in text:
        return False
    prefix, _, suffix = text.partition("T")
    return bool(prefix) and prefix.isdigit() and bool(suffix[:-1]) and suffix[:-1].isdigit()


def _recent_forecast_control_run_dirs(control_root: Path, *, limit: int) -> list[Path]:
    """Return recent completed Stage-10 run directories, newest first."""
    if int(limit) <= 0 or not control_root.exists():
        return []
    cache_dirname = str(MULTIRES_FORECAST_CONTROL["replay_cache_dirname"])
    run_dirs = [
        child
        for child in control_root.iterdir()
        if child.is_dir()
        and child.name not in {"latest", cache_dirname}
        and not child.name.startswith("tmp_")
        and _is_timestamped_run_dir_name(child.name)
    ]
    return sorted(run_dirs, key=lambda path: path.name, reverse=True)[: int(limit)]


def _build_phase_control_prior_table(control_root: Path) -> pd.DataFrame:
    """Summarize recent phase-layer control evidence into selection-level priors."""
    max_runs = int(MULTIRES_FORECAST_CONTROL["phase_control_prior_run_limit"])
    if max_runs <= 0:
        return pd.DataFrame()
    run_rows: list[pd.DataFrame] = []
    for run_dir in _recent_forecast_control_run_dirs(control_root, limit=max_runs):
        benchmark = _read_csv_if_present(run_dir / "control_layer_candidate_benchmarks.csv")
        required_columns = {
            "control_layer",
            "replay_resolution",
            "replay_feature_set",
            "replay_model_label",
            "evaluation_selection_metric_value",
        }
        if benchmark.empty or not required_columns.issubset(benchmark.columns):
            continue
        working = benchmark.loc[
            benchmark["control_layer"].astype("string").eq("phase")
            & benchmark["replay_resolution"].astype("string").ne("")
            & benchmark["replay_feature_set"].astype("string").ne("")
            & benchmark["replay_model_label"].astype("string").ne("")
        ].copy()
        if working.empty:
            continue
        for column in (
            "evaluation_selection_metric_value",
            "evaluation_next_lock_mae",
            "evaluation_peak_value_mae",
            "evaluation_peak_interval_miss_rate",
        ):
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce")
            else:
                working[column] = float("nan")
        working["evaluation_peak_hit_rate"] = 1.0 - working["evaluation_peak_interval_miss_rate"].astype(float)
        working = working.sort_values(
            [
                "evaluation_selection_metric_value",
                "evaluation_next_lock_mae",
                "evaluation_peak_value_mae",
                "evaluation_peak_interval_miss_rate",
                "candidate_label",
            ],
            ascending=[True, True, True, True, True],
            kind="stable",
        )
        best_rows = (
            working.groupby(
                ["replay_resolution", "replay_feature_set", "replay_model_label"],
                dropna=False,
                as_index=False,
            )
            .first()
            .rename(
                columns={
                    "replay_resolution": "resolution",
                    "replay_feature_set": "feature_set",
                    "replay_model_label": "model_label",
                }
            )
        )
        best_rows["source_run_id"] = str(run_dir.name)
        run_rows.append(
            best_rows.loc[
                :,
                [
                    "resolution",
                    "feature_set",
                    "model_label",
                    "evaluation_selection_metric_value",
                    "evaluation_next_lock_mae",
                    "evaluation_peak_value_mae",
                    "evaluation_peak_hit_rate",
                    "source_run_id",
                ],
            ].copy()
        )
    if not run_rows:
        return pd.DataFrame()
    history = pd.concat(run_rows, ignore_index=True)
    return (
        history.groupby(["resolution", "feature_set", "model_label"], dropna=False)
        .agg(
            prior_phase_eval_metric=("evaluation_selection_metric_value", "median"),
            prior_phase_next_lock_mae=("evaluation_next_lock_mae", "median"),
            prior_phase_peak_value_mae=("evaluation_peak_value_mae", "median"),
            prior_phase_peak_hit_rate=("evaluation_peak_hit_rate", "median"),
            prior_phase_support_runs=("source_run_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            [
                "prior_phase_eval_metric",
                "prior_phase_next_lock_mae",
                "prior_phase_peak_value_mae",
                "prior_phase_peak_hit_rate",
                "prior_phase_support_runs",
                "resolution",
                "feature_set",
                "model_label",
            ],
            ascending=[True, True, True, False, False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _control_candidate_pool_size(layer_role: str) -> int:
    """Return the configured control-layer candidate-pool size after layer-specific expansion."""
    if str(layer_role) == "phase":
        return int(MULTIRES_FORECAST_CONTROL["phase_control_candidate_pool_size"])
    base_size = int(MULTIRES_FORECAST_CONTROL["candidate_pool_size"])
    expanded_size = int(MULTIRES_FORECAST_CONTROL["benchmark_expanded_candidate_pool_size"])
    expanded_layers = {
        str(value) for value in MULTIRES_FORECAST_CONTROL["benchmark_expanded_pool_layers"]
    }
    if str(layer_role) in expanded_layers:
        return max(base_size, expanded_size)
    return base_size


def _filter_phase_control_candidate_rows(
    working: pd.DataFrame,
    *,
    max_candidates: int,
) -> pd.DataFrame:
    """Apply low-risk prior-backed pruning before exact phase replay."""
    extra_slots = max(0, int(max_candidates) - 1)
    if working.empty or extra_slots <= 0:
        return working.iloc[0:0].copy()

    candidates = working.copy()
    candidates = candidates.drop_duplicates(
        subset=["resolution", "feature_set", "model_label"],
        keep="first",
    ).reset_index(drop=True)
    supported = candidates.loc[candidates["prior_phase_supported"].astype(bool)].copy()
    unsupported = candidates.loc[~candidates["prior_phase_supported"].astype(bool)].copy()
    if supported.empty:
        return candidates.head(extra_slots).reset_index(drop=True)

    per_resolution_cap = int(
        MULTIRES_FORECAST_CONTROL["phase_control_max_supplemental_contexts_per_resolution"]
    )
    exploration_slots = int(MULTIRES_FORECAST_CONTROL["phase_control_exploration_slots"])
    reserved_exploration_slots = min(exploration_slots, extra_slots) if not unsupported.empty else 0
    supported_limit = max(0, extra_slots - reserved_exploration_slots)

    selected_frames: list[pd.DataFrame] = []
    resolution_counts: dict[str, int] = {}
    if supported_limit > 0:
        supported_rows: list[pd.Series] = []
        for _, row in supported.iterrows():
            resolution = str(row.get("resolution", ""))
            if resolution_counts.get(resolution, 0) >= per_resolution_cap:
                continue
            resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
            supported_rows.append(row.copy())
            if len(supported_rows) >= supported_limit:
                break
        if supported_rows:
            selected_frames.append(pd.DataFrame(supported_rows))

    if reserved_exploration_slots > 0:
        selected_frames.append(unsupported.head(reserved_exploration_slots).copy())

    if not selected_frames:
        return candidates.head(extra_slots).reset_index(drop=True)
    return pd.concat(selected_frames, ignore_index=True).head(extra_slots).reset_index(drop=True)


def _control_pool_replay_max_workers(pool_size: int) -> int:
    """Resolve the worker count for replaying one control candidate pool."""
    if int(pool_size) <= 0:
        return 1
    return int(resolve_parallel_plan("forecast_control", task_count=int(pool_size)).n_jobs)


def _control_candidate_scope_plan(candidate_pool: list[dict[str, Any]]) -> list[str]:
    """Assign each pooled candidate the narrowest safe replay scope.

    The first candidate for a given `(resolution, feature_set)` needs
    `selected_plus_baselines` so the benchmark surface includes the comparable
    baseline family. Subsequent candidates in the same scope can stay on
    `selected_only`, which avoids recomputing the same baseline rows repeatedly.
    """
    scopes: list[str] = []
    baseline_scope_keys: set[tuple[str, str]] = set()
    allow_baselines = bool(MULTIRES_FORECAST_CONTROL["allow_baseline_candidates"])
    for pool_item in candidate_pool:
        selection = cast(dict[str, Any], pool_item["selection"])
        baseline_scope_key = (
            str(selection.get("resolution", "")),
            str(selection.get("feature_set", "")),
        )
        candidate_scope = "selected_only"
        if allow_baselines and baseline_scope_key not in baseline_scope_keys:
            candidate_scope = "selected_plus_baselines"
            baseline_scope_keys.add(baseline_scope_key)
        scopes.append(candidate_scope)
    return scopes


def _replay_control_pool_candidate(
    *,
    cache_root: Path | None,
    temp_root: Path,
    layer_role: str,
    pool_rank: int,
    pool_item: dict[str, Any],
    candidate_scope: str,
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    benchmark_origins: list[pd.Timestamp],
    evaluation_benchmark_origins: list[pd.Timestamp],
) -> dict[str, Any]:
    """Replay one pooled control candidate once across the union of requested origins."""
    selection = cast(dict[str, Any], pool_item["selection"])
    union_origins = sorted(
        {
            pd.Timestamp(value)
            for value in [*benchmark_origins, *evaluation_benchmark_origins]
        }
    )
    capture_path_details = str(layer_role) == "phase"
    union_result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_root / f"candidate_{pool_rank:02d}_union",
        layer_role=str(layer_role),
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=union_origins,
        capture_path_details=bool(capture_path_details),
        candidate_scope=str(candidate_scope),
        persist_artifacts=False,
    )
    benchmark_result = _subset_rollout_result_for_origins(
        union_result,
        origin_timestamps=benchmark_origins,
        require_path_details=False,
    )
    evaluation_result = _subset_rollout_result_for_origins(
        union_result,
        origin_timestamps=evaluation_benchmark_origins,
        require_path_details=bool(capture_path_details),
    )
    return {
        "pool_rank": int(pool_rank),
        "pool_item": pool_item,
        "selection": selection,
        "candidate_scope": str(candidate_scope),
        "benchmark_result": benchmark_result,
        "evaluation_result": evaluation_result,
    }


def _control_benchmark_origins(
    *,
    layer_role: str,
    origin_timestamps: list[pd.Timestamp],
) -> tuple[list[pd.Timestamp], str]:
    """Resolve whether a control layer should benchmark on sampled or full exact origins."""
    full_origin_layers = {
        str(value) for value in MULTIRES_FORECAST_CONTROL["benchmark_full_origin_layers"]
    }
    if str(layer_role) in full_origin_layers:
        return list(origin_timestamps), "full_control_scope"
    if str(layer_role) == "phase":
        return _sample_control_origins(
            origin_timestamps,
            cap=int(MULTIRES_FORECAST_CONTROL["phase_candidate_benchmark_origin_cap"]),
        ), "sampled_phase_cap"
    return _sample_control_benchmark_origins(origin_timestamps), "sampled_cap"


def _control_evaluation_benchmark_origins(
    *,
    layer_role: str,
    origin_timestamps: list[pd.Timestamp],
) -> tuple[list[pd.Timestamp], str]:
    """Resolve the held-out origin subset used to compare challenger candidates before full replay."""
    if str(layer_role) == "phase":
        return _sample_control_origins(
            origin_timestamps,
            cap=int(MULTIRES_FORECAST_CONTROL["phase_candidate_evaluation_origin_cap"]),
        ), "sampled_phase_evaluation_cap"
    return list(origin_timestamps), "full_control_scope"


def _representable_selection_origins(
    *,
    selection: dict[str, Any],
    horizon_minutes: int,
    origin_timestamps: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Filter explicit origins down to the timestamps representable for one replay selection."""
    if not origin_timestamps:
        return []
    resolution = str(selection.get("resolution", ""))
    if not resolution or resolution == "mixed":
        return list(origin_timestamps)
    lookup_key = (str(resolution), int(horizon_minutes))
    with _LOCAL_CACHE_LOCK:
        eligible_lookup = _REPRESENTABLE_ORIGIN_LOOKUP_CACHE.get(lookup_key)
    if eligible_lookup is None:
        base = load_base_gold(resolution).copy()
        base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
        horizon_steps = lead_steps_for_horizon(resolution, int(horizon_minutes))
        eligible_lookup = {
            pd.Timestamp(base.iloc[idx]["timestamp"])
            for idx in range(len(base))
            if pd.notna(base.iloc[idx]["timestamp"]) and idx + horizon_steps < len(base)
        }
        with _LOCAL_CACHE_LOCK:
            _REPRESENTABLE_ORIGIN_LOOKUP_CACHE[lookup_key] = eligible_lookup
    return [
        pd.Timestamp(timestamp)
        for timestamp in origin_timestamps
        if pd.Timestamp(timestamp) in eligible_lookup
    ]


def _subset_rollout_result_for_origins(
    result: dict[str, Any],
    *,
    origin_timestamps: list[pd.Timestamp],
    require_path_details: bool,
) -> dict[str, Any]:
    """Subset an in-memory replay result down to the requested origin timestamps."""
    working = dict(result)
    if "selected_origins" not in working:
        by_origin = cast(pd.DataFrame, working.get("by_origin", pd.DataFrame()))
        if "origin_timestamp" in by_origin.columns:
            selected_origins = (
                by_origin.loc[:, ["origin_timestamp"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            selected_origins = pd.DataFrame(columns=["origin_timestamp"])
        working["selected_origins"] = selected_origins
    if not origin_timestamps:
        empty_result = dict(working)
        empty_result["by_origin"] = pd.DataFrame()
        empty_result["selected_origins"] = pd.DataFrame()
        empty_result["detail_by_origin"] = pd.DataFrame()
        return empty_result
    subset = _subset_cached_rollout_result(
        working,
        origin_timestamps=[pd.Timestamp(value) for value in origin_timestamps],
        require_path_details=bool(require_path_details),
    )
    if subset is None:
        raise RuntimeError(
            "The requested replay origins were not present in the in-memory rollout result. "
            f"origin_count={len(origin_timestamps)}"
        )
    return subset


def _subset_detail_by_origin(
    detail_by_origin: pd.DataFrame,
    *,
    origin_timestamps: list[pd.Timestamp],
) -> pd.DataFrame:
    """Filter detailed replay rows down to one explicit set of origin timestamps."""
    if detail_by_origin.empty or not origin_timestamps:
        return pd.DataFrame(columns=detail_by_origin.columns)
    origin_labels = {pd.Timestamp(value).isoformat() for value in origin_timestamps}
    return detail_by_origin.loc[
        detail_by_origin["origin_timestamp"].astype("string").isin(origin_labels)
    ].reset_index(drop=True)


def _shared_representable_origins(
    *,
    candidate_pool: list[dict[str, Any]],
    horizon_minutes: int,
    origin_timestamps: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Intersect explicit origins across the candidate pool so replays stay comparable."""
    shared = [pd.Timestamp(timestamp) for timestamp in origin_timestamps]
    for pool_item in candidate_pool:
        representable = set(
            _representable_selection_origins(
                selection=cast(dict[str, Any], pool_item["selection"]),
                horizon_minutes=int(horizon_minutes),
                origin_timestamps=shared,
            )
        )
        shared = [timestamp for timestamp in shared if timestamp in representable]
        if not shared:
            break
    return shared


def _build_control_candidate_pool(
    *,
    layer_role: str,
    upstream_selection: dict[str, Any],
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
) -> list[dict[str, Any]]:
    """Build the small replay pool used to benchmark one control layer on shared cycles."""
    pool: list[dict[str, Any]] = [
        {
            "selection": dict(upstream_selection),
            "pool_source_type": "upstream_selection",
            "pool_source_run_id": str(upstream_selection.get("selection_run_id") or ""),
            "pool_reason": "Current upstream Stage-7 selection for this control layer.",
        }
    ]
    max_candidates = _control_candidate_pool_size(str(layer_role))
    if max_candidates <= 1:
        return pool

    rollout_root = preferred_output_path(PATHS["outputs_rollout_dir"])
    registry = _read_csv_if_present(rollout_root / "rollout_registry.csv")
    if registry.empty:
        return pool

    fields = _rollout_registry_metric_fields(str(selection_target))
    metric_column = fields["metric"]
    beat_baseline_column = fields["beat_baseline"]
    beat_persistence_column = fields["beat_persistence"]
    working = registry.copy()
    for column in (metric_column, "learned_origin_n"):
        if column not in working.columns:
            return pool
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["generated_at_utc"] = pd.to_datetime(working.get("generated_at_utc"), errors="coerce", utc=True)
    for column in (beat_baseline_column, beat_persistence_column):
        if column in working.columns:
            working[column] = (
                working[column]
                .astype("string")
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )
        else:
            working[column] = False
    working = working.loc[
        working["horizon_minutes"].astype("Int64").eq(int(horizon_minutes))
        & working["selection_target"].astype("string").eq(str(selection_target))
        & working["origin_policy"].astype("string").eq(str(origin_policy))
        & working["strategy"].astype("string").eq("recursive")
        & working["resolution"].astype("string").ne("mixed")
        & working[metric_column].notna()
    ].copy()
    if working.empty:
        return pool
    sort_columns = [
        beat_baseline_column,
        beat_persistence_column,
        "learned_origin_n",
        metric_column,
        "generated_at_utc",
    ]
    sort_ascending = [False, False, False, True, False]
    if str(layer_role) == "phase":
        forecast_control_root = preferred_output_path(PATHS["outputs_forecast_control_dir"])
        phase_prior = _build_phase_control_prior_table(Path(forecast_control_root))
        if not phase_prior.empty:
            working = working.merge(
                phase_prior,
                on=["resolution", "feature_set", "model_label"],
                how="left",
            )
        if "prior_phase_eval_metric" not in working.columns:
            working["prior_phase_eval_metric"] = float("nan")
        if "prior_phase_support_runs" not in working.columns:
            working["prior_phase_support_runs"] = 0
        min_support_runs = int(MULTIRES_FORECAST_CONTROL["phase_control_min_prior_support_runs"])
        working["prior_phase_supported"] = (
            working.get(
                "prior_phase_eval_metric",
                pd.Series(index=working.index, dtype="float64"),
            ).notna()
            & pd.to_numeric(
                working.get("prior_phase_support_runs", pd.Series(index=working.index, dtype="float64")),
                errors="coerce",
            )
            .fillna(0)
            .ge(min_support_runs)
        )
        sort_columns = [
            "prior_phase_supported",
            "prior_phase_eval_metric",
            beat_baseline_column,
            beat_persistence_column,
            "learned_origin_n",
            metric_column,
            "generated_at_utc",
        ]
        sort_ascending = [False, True, False, False, False, True, False]
    working = working.sort_values(
        sort_columns,
        ascending=sort_ascending,
        kind="stable",
    ).reset_index(drop=True)
    if str(layer_role) == "phase":
        working = _filter_phase_control_candidate_rows(
            working,
            max_candidates=max_candidates,
        )

    seen = {_control_candidate_signature(upstream_selection)}
    for _, row in working.iterrows():
        selection = resolve_rollout_selection_context(
            resolution=str(row["resolution"]),
            feature_set=str(row["feature_set"]),
            model_label=str(row["model_label"]),
            requested_horizon_minutes=int(horizon_minutes),
            requested_origin_policy=str(origin_policy),
            selection_target=str(selection_target),
            selection_run_id=None,
        )
        signature = _control_candidate_signature(selection)
        if signature in seen:
            continue
        seen.add(signature)
        pool.append(
            {
                "selection": selection,
                "pool_source_type": "rollout_registry",
                "pool_source_run_id": str(row.get("run_id", "")),
                "pool_prior_phase_eval_metric": float(row.get("prior_phase_eval_metric", float("nan"))),
                "pool_prior_phase_support_runs": int(
                    pd.to_numeric(row.get("prior_phase_support_runs"), errors="coerce")
                    if pd.notna(row.get("prior_phase_support_runs"))
                    else 0
                ),
                "pool_reason": (
                    "Additional rollout-registry challenger replayed on the exact control cycles "
                    "for cross-run control-layer benchmarking."
                    if str(layer_role) != "phase"
                    else (
                        "Additional rollout-registry challenger prioritized by recent "
                        "Stage-10 phase-control evidence before exact replay."
                        if bool(row.get("prior_phase_supported", False))
                        else "Additional rollout-registry challenger retained as a no-prior "
                        "phase exploration slot before exact replay."
                    )
                ),
            }
        )
        if len(pool) >= max_candidates:
            break
    return pool


def _benchmark_control_layer_candidates(
    by_origin: pd.DataFrame,
    *,
    layer_role: str,
    detail_by_origin: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate one replay table and select the best candidate for a control layer."""
    if by_origin.empty:
        raise RuntimeError(f"Cannot benchmark an empty control-layer replay for {layer_role}.")
    working = by_origin.copy()
    if not bool(MULTIRES_FORECAST_CONTROL["allow_baseline_candidates"]):
        working = working.loc[working["candidate_type"].astype("string").eq("learned")].copy()
    if working.empty:
        raise RuntimeError(f"No eligible rollout candidates remained for control layer {layer_role}.")

    detail_metrics = pd.DataFrame()
    if (
        isinstance(detail_by_origin, pd.DataFrame)
        and not detail_by_origin.empty
        and {"actual_load", "predicted_load"}.issubset(detail_by_origin.columns)
    ):
        detail_metrics = _control_candidate_metrics_from_detail(
            detail_by_origin,
            lock_interval_minutes=int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"]),
        )
    benchmark = _aggregate_rollout_metrics(working).copy()
    benchmark = _overlay_candidate_metrics(benchmark, _control_layer_distribution_metrics(working))
    if not detail_metrics.empty:
        benchmark = _overlay_candidate_metrics(benchmark, _aggregate_control_candidate_metrics(detail_metrics))
        benchmark = _overlay_candidate_metrics(benchmark, _control_layer_distribution_metrics(detail_metrics))
    benchmark["control_layer"] = str(layer_role)
    selection_metric = _control_layer_selection_metric(str(layer_role))
    benchmark = _ensure_selection_metric_column(
        benchmark,
        selection_metric=selection_metric,
        fallback_metrics=[
            "next_lock_mae",
            "lock_mae",
            "profile_shape_mae",
            "path_mae",
            "endpoint_mae",
            "energy_mae",
        ],
    )
    for column in (f"{selection_metric}_p50", f"{selection_metric}_p90"):
        if column not in benchmark.columns:
            benchmark[column] = benchmark[selection_metric].astype(float)
        else:
            benchmark[column] = benchmark[column].fillna(benchmark[selection_metric]).astype(float)
    for metric_name in (
        "profile_shape_mae",
        "path_mae",
        "minute_path_mae",
        "lock_mae",
        "next_lock_mae",
        "peak_value_mae",
        "peak_interval_miss_rate",
        "optimizer_score",
    ):
        if metric_name not in benchmark.columns:
            continue
        for suffix in ("p50", "p90"):
            column = f"{metric_name}_{suffix}"
            if column not in benchmark.columns:
                benchmark[column] = benchmark[metric_name].astype(float)
            else:
                benchmark[column] = benchmark[column].fillna(benchmark[metric_name]).astype(float)
    benchmark["selection_metric_name"] = selection_metric
    benchmark["selection_metric_value"] = benchmark[selection_metric].astype(float)
    metric_pct_column = f"{selection_metric}_pct"
    benchmark["selection_metric_pct"] = (
        benchmark[metric_pct_column].astype(float)
        if metric_pct_column in benchmark.columns
        else float("nan")
    )
    benchmark = _ensure_sort_columns(
        benchmark,
        sort_columns=_control_layer_sort_columns(selection_metric),
        selection_metric=selection_metric,
        selection_metric_value_column="selection_metric_value",
    )
    benchmark = benchmark.sort_values(
        _control_layer_sort_columns(selection_metric),
        ascending=[True] * len(_control_layer_sort_columns(selection_metric)),
        kind="stable",
    ).reset_index(drop=True)
    return benchmark, benchmark.iloc[0]


def _replay_rollout_layer(
    *,
    temp_root: Path,
    cache_root: Path | None,
    layer_role: str,
    horizon_minutes: int,
    origin_timestamps: list[pd.Timestamp] | None = None,
    benchmark_origin_timestamps: list[pd.Timestamp] | None = None,
    evaluation_origin_timestamps: list[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    """Replay a control layer on calibration origins, then evaluate the winner on held-out origins."""
    if origin_timestamps is not None:
        benchmark_origin_timestamps = list(origin_timestamps)
        evaluation_origin_timestamps = list(origin_timestamps)
    if benchmark_origin_timestamps is None:
        benchmark_origin_timestamps = list(evaluation_origin_timestamps or [])
    if evaluation_origin_timestamps is None:
        evaluation_origin_timestamps = list(benchmark_origin_timestamps or [])
    if not benchmark_origin_timestamps:
        benchmark_origin_timestamps = list(evaluation_origin_timestamps)
    if not evaluation_origin_timestamps:
        evaluation_origin_timestamps = list(benchmark_origin_timestamps)
    origin_policy = resolve_rollout_origin_policy(int(horizon_minutes), MULTIRES_ROLLOUT["origin_policy"])
    selection_target = resolve_rollout_selection_target(
        int(horizon_minutes),
        MULTIRES_ROLLOUT["selection_target"],
    )
    upstream_selection = resolve_rollout_selection_context(
        resolution=None,
        feature_set=None,
        model_label=None,
        requested_horizon_minutes=int(horizon_minutes),
        requested_origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        selection_run_id=None,
    )
    candidate_pool = _build_control_candidate_pool(
        layer_role=str(layer_role),
        upstream_selection=upstream_selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
    )
    benchmark_origin_timestamps = _shared_representable_origins(
        candidate_pool=candidate_pool,
        horizon_minutes=int(horizon_minutes),
        origin_timestamps=benchmark_origin_timestamps,
    )
    evaluation_origin_timestamps = _shared_representable_origins(
        candidate_pool=candidate_pool,
        horizon_minutes=int(horizon_minutes),
        origin_timestamps=evaluation_origin_timestamps,
    )
    if not benchmark_origin_timestamps or not evaluation_origin_timestamps:
        raise RuntimeError(
            f"No shared explicit origins remained for control layer {layer_role} after representability filtering."
        )
    benchmark_origins, benchmark_origin_mode = _control_benchmark_origins(
        layer_role=str(layer_role),
        origin_timestamps=benchmark_origin_timestamps,
    )
    evaluation_benchmark_origins, evaluation_benchmark_origin_mode = _control_evaluation_benchmark_origins(
        layer_role=str(layer_role),
        origin_timestamps=evaluation_origin_timestamps,
    )
    benchmark_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    benchmark_detail_frames: list[pd.DataFrame] = []
    evaluation_detail_frames: list[pd.DataFrame] = []
    upstream_label = ""
    benchmark_cache_hits = 0
    benchmark_cache_misses = 0
    evaluation_cache_hits = 0
    evaluation_cache_misses = 0
    scope_plan = _control_candidate_scope_plan(candidate_pool)
    replay_tasks = [
        {
            "pool_rank": int(pool_rank),
            "pool_item": pool_item,
            "candidate_scope": scope_plan[pool_rank - 1],
        }
        for pool_rank, pool_item in enumerate(candidate_pool, start=1)
    ]
    pool_results: list[dict[str, Any]] = []
    max_workers = _control_pool_replay_max_workers(len(replay_tasks))
    if max_workers > 1 and len(replay_tasks) > 1:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="elf_control_replay") as executor:
            future_map = {
                executor.submit(
                    _replay_control_pool_candidate,
                    cache_root=cache_root,
                    temp_root=temp_root,
                    layer_role=str(layer_role),
                    pool_rank=int(task["pool_rank"]),
                    pool_item=cast(dict[str, Any], task["pool_item"]),
                    candidate_scope=str(task["candidate_scope"]),
                    horizon_minutes=int(horizon_minutes),
                    origin_policy=str(origin_policy),
                    selection_target=str(selection_target),
                    benchmark_origins=benchmark_origins,
                    evaluation_benchmark_origins=evaluation_benchmark_origins,
                ): task
                for task in replay_tasks
            }
            for future in as_completed(future_map):
                pool_results.append(future.result())
    else:
        for task in replay_tasks:
            pool_results.append(
                _replay_control_pool_candidate(
                    cache_root=cache_root,
                    temp_root=temp_root,
                    layer_role=str(layer_role),
                    pool_rank=int(task["pool_rank"]),
                    pool_item=cast(dict[str, Any], task["pool_item"]),
                    candidate_scope=str(task["candidate_scope"]),
                    horizon_minutes=int(horizon_minutes),
                    origin_policy=str(origin_policy),
                    selection_target=str(selection_target),
                    benchmark_origins=benchmark_origins,
                    evaluation_benchmark_origins=evaluation_benchmark_origins,
                )
            )
    pool_results.sort(key=lambda payload: int(payload["pool_rank"]))
    for pool_result in pool_results:
        pool_rank = int(pool_result["pool_rank"])
        pool_item = cast(dict[str, Any], pool_result["pool_item"])
        selection = cast(dict[str, Any], pool_result["selection"])
        benchmark_result = cast(dict[str, Any], pool_result["benchmark_result"])
        evaluation_result = cast(dict[str, Any], pool_result["evaluation_result"])
        if str(benchmark_result.get("replay_cache_status", "")) in {"hit", "subset_hit"}:
            benchmark_cache_hits += 1
        elif str(benchmark_result.get("replay_cache_status", "")) == "miss":
            benchmark_cache_misses += 1
        if pool_rank == 1:
            upstream_label = _qualify_control_candidate_label(
                selection,
                _selected_learned_candidate_label(benchmark_result["by_origin"]),
            )
        qualified_benchmark = benchmark_result["by_origin"].copy()
        qualified_benchmark["native_candidate_label"] = (
            qualified_benchmark["candidate_label"].astype("string")
        )
        qualified_benchmark["candidate_label"] = qualified_benchmark["native_candidate_label"].map(
            lambda value: _qualify_control_candidate_label(selection, str(value))
        )
        qualified_benchmark["replay_pool_rank"] = int(pool_rank)
        qualified_benchmark["replay_pool_source_type"] = str(pool_item["pool_source_type"])
        qualified_benchmark["replay_pool_source_run_id"] = str(pool_item["pool_source_run_id"])
        qualified_benchmark["replay_resolution"] = str(selection.get("resolution", ""))
        qualified_benchmark["replay_feature_set"] = str(selection.get("feature_set", ""))
        qualified_benchmark["replay_model_label"] = str(selection.get("model_label", ""))
        qualified_benchmark["replay_pool_prior_phase_eval_metric"] = float(
            pool_item.get("pool_prior_phase_eval_metric", float("nan"))
        )
        qualified_benchmark["replay_pool_prior_phase_support_runs"] = int(
            pool_item.get("pool_prior_phase_support_runs", 0)
        )
        qualified_benchmark["replay_run_dir"] = _relative_artifact_path(benchmark_result["run_dir"])
        benchmark_frames.append(qualified_benchmark)
        benchmark_detail = benchmark_result["detail_by_origin"].copy()
        if not benchmark_detail.empty:
            benchmark_detail["native_candidate_label"] = benchmark_detail["candidate_label"].astype("string")
            benchmark_detail["candidate_label"] = benchmark_detail["native_candidate_label"].map(
                lambda value: _qualify_control_candidate_label(selection, str(value))
            )
            benchmark_detail["replay_pool_rank"] = int(pool_rank)
            benchmark_detail["replay_pool_source_type"] = str(pool_item["pool_source_type"])
            benchmark_detail["replay_pool_source_run_id"] = str(pool_item["pool_source_run_id"])
            benchmark_detail["replay_resolution"] = str(selection.get("resolution", ""))
            benchmark_detail["replay_feature_set"] = str(selection.get("feature_set", ""))
            benchmark_detail["replay_model_label"] = str(selection.get("model_label", ""))
            benchmark_detail["replay_pool_prior_phase_eval_metric"] = float(
                pool_item.get("pool_prior_phase_eval_metric", float("nan"))
            )
            benchmark_detail["replay_pool_prior_phase_support_runs"] = int(
                pool_item.get("pool_prior_phase_support_runs", 0)
            )
            benchmark_detail["replay_run_dir"] = _relative_artifact_path(benchmark_result["run_dir"])
            benchmark_detail_frames.append(benchmark_detail)

        if str(evaluation_result.get("replay_cache_status", "")) in {"hit", "subset_hit"}:
            evaluation_cache_hits += 1
        elif str(evaluation_result.get("replay_cache_status", "")) == "miss":
            evaluation_cache_misses += 1
        qualified_evaluation = evaluation_result["by_origin"].copy()
        qualified_evaluation["native_candidate_label"] = (
            qualified_evaluation["candidate_label"].astype("string")
        )
        qualified_evaluation["candidate_label"] = qualified_evaluation["native_candidate_label"].map(
            lambda value: _qualify_control_candidate_label(selection, str(value))
        )
        qualified_evaluation["replay_pool_rank"] = int(pool_rank)
        qualified_evaluation["replay_pool_source_type"] = str(pool_item["pool_source_type"])
        qualified_evaluation["replay_pool_source_run_id"] = str(pool_item["pool_source_run_id"])
        qualified_evaluation["replay_resolution"] = str(selection.get("resolution", ""))
        qualified_evaluation["replay_feature_set"] = str(selection.get("feature_set", ""))
        qualified_evaluation["replay_model_label"] = str(selection.get("model_label", ""))
        qualified_evaluation["replay_pool_prior_phase_eval_metric"] = float(
            pool_item.get("pool_prior_phase_eval_metric", float("nan"))
        )
        qualified_evaluation["replay_pool_prior_phase_support_runs"] = int(
            pool_item.get("pool_prior_phase_support_runs", 0)
        )
        qualified_evaluation["replay_run_dir"] = _relative_artifact_path(evaluation_result["run_dir"])
        evaluation_frames.append(qualified_evaluation)
        evaluation_detail = evaluation_result["detail_by_origin"].copy()
        if not evaluation_detail.empty:
            evaluation_detail["native_candidate_label"] = evaluation_detail["candidate_label"].astype("string")
            evaluation_detail["candidate_label"] = evaluation_detail["native_candidate_label"].map(
                lambda value: _qualify_control_candidate_label(selection, str(value))
            )
            evaluation_detail["replay_pool_rank"] = int(pool_rank)
            evaluation_detail["replay_pool_source_type"] = str(pool_item["pool_source_type"])
            evaluation_detail["replay_pool_source_run_id"] = str(pool_item["pool_source_run_id"])
            evaluation_detail["replay_resolution"] = str(selection.get("resolution", ""))
            evaluation_detail["replay_feature_set"] = str(selection.get("feature_set", ""))
            evaluation_detail["replay_model_label"] = str(selection.get("model_label", ""))
            evaluation_detail["replay_pool_prior_phase_eval_metric"] = float(
                pool_item.get("pool_prior_phase_eval_metric", float("nan"))
            )
            evaluation_detail["replay_pool_prior_phase_support_runs"] = int(
                pool_item.get("pool_prior_phase_support_runs", 0)
            )
            evaluation_detail["replay_run_dir"] = _relative_artifact_path(evaluation_result["run_dir"])
            evaluation_detail_frames.append(evaluation_detail)

    combined_by_origin = pd.concat(benchmark_frames, ignore_index=True).reset_index(drop=True)
    combined_detail = (
        pd.concat(benchmark_detail_frames, ignore_index=True).reset_index(drop=True)
        if benchmark_detail_frames
        else pd.DataFrame()
    )
    benchmark, _ = _benchmark_control_layer_candidates(
        combined_by_origin,
        layer_role=str(layer_role),
        detail_by_origin=combined_detail,
    )
    evaluation_by_origin = pd.concat(evaluation_frames, ignore_index=True).reset_index(drop=True)
    evaluation_detail = (
        pd.concat(evaluation_detail_frames, ignore_index=True).reset_index(drop=True)
        if evaluation_detail_frames
        else pd.DataFrame()
    )
    evaluation_benchmark, _ = _benchmark_control_layer_candidates(
        evaluation_by_origin,
        layer_role=str(layer_role),
        detail_by_origin=evaluation_detail,
    )
    candidate_meta = (
        combined_by_origin.sort_values(["replay_pool_rank", "candidate_label"], kind="stable")
        .drop_duplicates(subset=["candidate_label"])
        .loc[
            :,
            [
                "candidate_label",
                "replay_pool_rank",
                "replay_pool_source_type",
                "replay_pool_source_run_id",
                "replay_resolution",
                "replay_feature_set",
                "replay_model_label",
                "replay_pool_prior_phase_eval_metric",
                "replay_pool_prior_phase_support_runs",
                "replay_run_dir",
            ],
        ]
    )
    benchmark = benchmark.merge(candidate_meta, on="candidate_label", how="left")
    evaluation_benchmark = evaluation_benchmark.merge(candidate_meta, on="candidate_label", how="left")
    benchmark = benchmark.merge(
        _rename_control_layer_evaluation_metrics(evaluation_benchmark.copy()),
        on="candidate_label",
        how="left",
    )
    selected_row, selection_mode = _select_control_layer_candidate(
        calibration_benchmark=benchmark,
        evaluation_benchmark=evaluation_benchmark,
        upstream_label=upstream_label,
    )
    selected_selection = next(
        (
            cast(dict[str, Any], pool_item["selection"])
            for pool_item in candidate_pool
            if _control_candidate_signature(cast(dict[str, Any], pool_item["selection"]))
            == (
                str(selected_row.get("replay_resolution", "")),
                str(selected_row.get("replay_feature_set", "")),
                str(selected_row.get("replay_model_label", "")),
                str(
                    cast(dict[str, Any], pool_item["selection"]).get("portfolio_candidate_label", "")
                ),
            )
        ),
        dict(upstream_selection),
    )
    selected_selection = dict(selected_selection)
    selected_selection["requested_candidate_label"] = _native_control_candidate_label(
        selected_selection,
        str(selected_row.get("candidate_label", "")),
    )
    selected_benchmark_origins = list(benchmark_origin_timestamps)
    selected_evaluation_origins = list(evaluation_origin_timestamps)
    if str(layer_role) == "phase":
        phase_cap = int(MULTIRES_FORECAST_CONTROL["phase_control_origin_cap"])
        if phase_cap > 0:
            selected_benchmark_origins = _sample_control_origins(
                benchmark_origin_timestamps,
                cap=phase_cap,
            )
            selected_evaluation_origins = _sample_control_origins(
                evaluation_origin_timestamps,
                cap=phase_cap,
            )
    benchmark_result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_root / "selected_benchmark",
        layer_role=str(layer_role),
        selection=selected_selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=selected_benchmark_origins,
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=False,
    )
    full_result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_root / "selected_full",
        layer_role=str(layer_role),
        selection=selected_selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        origin_timestamps=selected_evaluation_origins,
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=True,
    )
    full_detail_by_origin = full_result["detail_by_origin"].copy()
    if not full_detail_by_origin.empty:
        full_detail_by_origin["native_candidate_label"] = (
            full_detail_by_origin["candidate_label"].astype("string")
        )
        full_detail_by_origin["candidate_label"] = full_detail_by_origin["native_candidate_label"].map(
            lambda value: _qualify_control_candidate_label(selected_selection, str(value))
        )
    benchmark_detail_by_origin = benchmark_result["detail_by_origin"].copy()
    if not benchmark_detail_by_origin.empty:
        benchmark_detail_by_origin["native_candidate_label"] = (
            benchmark_detail_by_origin["candidate_label"].astype("string")
        )
        benchmark_detail_by_origin["candidate_label"] = benchmark_detail_by_origin[
            "native_candidate_label"
        ].map(lambda value: _qualify_control_candidate_label(selected_selection, str(value)))
    selected_run_dir = full_result["run_dir"]
    candidate_selection_by_pool_rank = {
        int(pool_rank): dict(cast(dict[str, Any], pool_item["selection"]))
        for pool_rank, pool_item in enumerate(candidate_pool, start=1)
    }
    return {
        "selection": selected_selection,
        "upstream_selection": upstream_selection,
        "origin_policy": origin_policy,
        "selection_target": selection_target,
        "benchmark_result": {
            "run_dir": benchmark_result["run_dir"],
            "detail_by_origin": benchmark_detail_by_origin,
        },
        "result": {
            "run_dir": selected_run_dir,
            "detail_by_origin": full_detail_by_origin,
        },
        "candidate_label": str(selected_row["candidate_label"]),
        "candidate_type": str(selected_row.get("candidate_type", "")),
        "source_model_label": str(selected_row.get("source_model_label", "")),
        "target_mode": str(selected_row.get("target_mode", "")),
        "control_selection_metric": str(selected_row.get("selection_metric_name", "")),
        "control_selection_metric_value": float(selected_row.get("selection_metric_value", float("nan"))),
        "control_selection_metric_pct": float(selected_row.get("selection_metric_pct", float("nan"))),
        "evaluation_selection_metric_value": float(
            selected_row.get("evaluation_selection_metric_value", selected_row.get("selection_metric_value", float("nan")))
        ),
        "evaluation_selection_metric_pct": float(
            selected_row.get("evaluation_selection_metric_pct", selected_row.get("selection_metric_pct", float("nan")))
        ),
        "control_selection_mode": selection_mode,
        "upstream_candidate_label": upstream_label,
        "candidate_pool_count": int(len(candidate_pool)),
        "benchmark_origin_count": int(len(selected_benchmark_origins)),
        "benchmark_origin_mode": str(benchmark_origin_mode),
        "benchmark_origin_timestamps": [pd.Timestamp(value) for value in selected_benchmark_origins],
        "evaluation_origin_count": int(len(selected_evaluation_origins)),
        "evaluation_origin_timestamps": [pd.Timestamp(value) for value in selected_evaluation_origins],
        "evaluation_benchmark_origin_count": int(len(evaluation_benchmark_origins)),
        "evaluation_benchmark_origin_mode": str(evaluation_benchmark_origin_mode),
        "benchmark_replay_cache_hits": int(benchmark_cache_hits),
        "benchmark_replay_cache_misses": int(benchmark_cache_misses),
        "evaluation_replay_cache_hits": int(evaluation_cache_hits),
        "evaluation_replay_cache_misses": int(evaluation_cache_misses),
        "selected_benchmark_replay_cache_status": str(benchmark_result.get("replay_cache_status", "")),
        "selected_replay_cache_status": str(full_result.get("replay_cache_status", "")),
        "selected_replay_cache_artifact": _relative_artifact_path(Path(full_result["run_dir"])),
        "candidate_benchmarks": benchmark,
        "candidate_benchmark_by_origin": combined_by_origin.copy(),
        "candidate_benchmark_detail_by_origin": (
            pd.concat(benchmark_detail_frames, ignore_index=True).reset_index(drop=True)
            if benchmark_detail_frames
            else pd.DataFrame()
        ),
        "candidate_evaluation_by_origin": evaluation_by_origin.copy(),
        "candidate_evaluation_detail_by_origin": (
            pd.concat(evaluation_detail_frames, ignore_index=True).reset_index(drop=True)
            if evaluation_detail_frames
            else pd.DataFrame()
        ),
        "candidate_selection_by_pool_rank": candidate_selection_by_pool_rank,
    }


def _extract_candidate_path(
    detail_by_origin: pd.DataFrame,
    *,
    origin_timestamp: pd.Timestamp,
    candidate_label: str,
    minute_index: pd.DatetimeIndex,
) -> pd.Series:
    """Extract and align one candidate's detailed path for a single control origin."""
    required_columns = {"origin_timestamp", "candidate_label", "forecast_timestamp", "predicted_load"}
    if not required_columns.issubset(detail_by_origin.columns):
        raise RuntimeError(
            f"Missing detailed replay rows for {candidate_label} at {pd.Timestamp(origin_timestamp).isoformat()}."
        )
    matched = detail_by_origin.loc[
        detail_by_origin["origin_timestamp"].astype("string").eq(pd.Timestamp(origin_timestamp).isoformat())
        & detail_by_origin["candidate_label"].astype("string").eq(str(candidate_label))
    ].copy()
    if matched.empty:
        raise RuntimeError(
            f"Missing detailed replay rows for {candidate_label} at {pd.Timestamp(origin_timestamp).isoformat()}."
        )
    matched["forecast_timestamp"] = pd.to_datetime(matched["forecast_timestamp"], errors="coerce")
    matched = matched.loc[matched["forecast_timestamp"].notna()].sort_values(
        "forecast_timestamp",
        kind="stable",
    )
    matched = matched.drop_duplicates(subset=["forecast_timestamp"], keep="last")
    series = matched.set_index("forecast_timestamp")["predicted_load"].astype(float)
    aligned = series.reindex(minute_index).ffill().bfill()
    if aligned.isna().all():
        raise RuntimeError(
            f"Detailed replay rows for {candidate_label} at {pd.Timestamp(origin_timestamp).isoformat()} "
            "did not cover the requested control minute index."
        )
    return aligned.astype(float)


def _apply_rollout_updates(
    base_forecast: pd.Series,
    *,
    detail_by_origin: pd.DataFrame,
    candidate_label: str,
    update_origins: list[pd.Timestamp],
    horizon_minutes: int,
) -> pd.Series:
    """Apply one rollout layer's updates over the affected future forecast windows."""
    updated = base_forecast.copy()
    for origin_timestamp in update_origins:
        segment_index = updated.index[
            (updated.index > pd.Timestamp(origin_timestamp))
            & (updated.index <= pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=horizon_minutes))
        ]
        if segment_index.empty:
            continue
        try:
            candidate_path = _extract_candidate_path(
                detail_by_origin,
                origin_timestamp=origin_timestamp,
                candidate_label=candidate_label,
                minute_index=segment_index,
            )
        except RuntimeError as exc:
            if "Missing detailed replay rows" in str(exc) or "did not cover the requested control minute index" in str(exc):
                continue
            raise
        updated.loc[segment_index] = candidate_path.loc[segment_index].to_numpy(dtype=float)
    return updated


def _apply_bucketed_rollout_updates(
    base_forecast: pd.Series,
    *,
    detail_by_origin: pd.DataFrame,
    candidate_by_bucket: dict[int, str],
    update_origins: list[pd.Timestamp],
    horizon_minutes: int,
    bucket_minutes: int,
    cycle_minutes: int = 60,
) -> pd.Series:
    """Apply rollout updates using a bucket-specific candidate label for each origin timestamp."""
    updated = base_forecast.copy()
    for origin_timestamp in sorted(pd.Timestamp(value) for value in update_origins):
        bucket_key = _timestamp_minute_bucket(
            pd.Timestamp(origin_timestamp),
            bucket_minutes=int(bucket_minutes),
            cycle_minutes=int(cycle_minutes),
        )
        candidate_label = str(candidate_by_bucket.get(bucket_key, "")).strip()
        if not candidate_label:
            continue
        segment_index = updated.index[
            (updated.index > pd.Timestamp(origin_timestamp))
            & (updated.index <= pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=horizon_minutes))
        ]
        if segment_index.empty:
            continue
        try:
            candidate_path = _extract_candidate_path(
                detail_by_origin,
                origin_timestamp=pd.Timestamp(origin_timestamp),
                candidate_label=candidate_label,
                minute_index=segment_index,
            )
        except RuntimeError as exc:
            if "Missing detailed replay rows" in str(exc) or "did not cover the requested control minute index" in str(exc):
                continue
            raise
        updated.loc[segment_index] = candidate_path.loc[segment_index].to_numpy(dtype=float)
    return updated.astype(float)


def _apply_bucketed_series_updates(
    base_forecast: pd.Series,
    *,
    series_by_candidate: dict[str, pd.Series],
    candidate_by_bucket: dict[int, str],
    update_origins: list[pd.Timestamp],
    horizon_minutes: int,
    bucket_minutes: int,
    cycle_minutes: int = 60,
) -> pd.Series:
    """Apply bucket-routed updates from already-materialized candidate series."""
    updated = base_forecast.astype(float).copy()
    for origin_timestamp in sorted(pd.Timestamp(value) for value in update_origins):
        bucket_key = _timestamp_minute_bucket(
            pd.Timestamp(origin_timestamp),
            bucket_minutes=int(bucket_minutes),
            cycle_minutes=int(cycle_minutes),
        )
        candidate_label = str(candidate_by_bucket.get(bucket_key, "")).strip()
        candidate_series = series_by_candidate.get(candidate_label)
        if not candidate_label or candidate_series is None:
            continue
        segment_mask = (
            (updated.index > pd.Timestamp(origin_timestamp))
            & (updated.index <= pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=horizon_minutes))
        )
        if not bool(segment_mask.any()):
            continue
        if len(candidate_series) == len(updated) and candidate_series.index.equals(updated.index):
            updated.iloc[np.flatnonzero(segment_mask)] = candidate_series.loc[segment_mask].to_numpy(dtype=float)
            continue
        segment_index = updated.index[
            (updated.index > pd.Timestamp(origin_timestamp))
            & (updated.index <= pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=horizon_minutes))
        ]
        if segment_index.empty:
            continue
        aligned = candidate_series.reindex(segment_index).ffill().bfill()
        if aligned.isna().all():
            continue
        updated.loc[segment_index] = aligned.to_numpy(dtype=float)
    return updated.astype(float)


def _apply_nowcast_updates(
    base_forecast: pd.Series,
    nowcast_predictions: pd.Series,
) -> pd.Series:
    """Overlay exact timestamp-level 1-minute nowcasts onto the broader phase path."""
    updated = base_forecast.astype(float).copy()
    aligned_nowcast = nowcast_predictions.reindex(updated.index)
    valid_mask = aligned_nowcast.notna()
    if bool(valid_mask.any()):
        updated.loc[valid_mask] = aligned_nowcast.loc[valid_mask].to_numpy(dtype=float)
    return updated


def _build_control_minute_timeline(
    *,
    cycle_origins: list[pd.Timestamp],
    actual_minute_base: pd.DataFrame,
    day_ahead: dict[str, Any],
    hourly: dict[str, Any],
    phase: dict[str, Any],
    result_key: str = "result",
    day_ahead_horizon: int,
    hourly_horizon: int,
    phase_horizon: int,
    hourly_origins: list[pd.Timestamp],
    phase_origins: list[pd.Timestamp],
) -> pd.DataFrame:
    """Materialize the evaluation-ready control minute timeline for one origin set."""
    minute_frames: list[pd.DataFrame] = []
    for cycle_origin in cycle_origins:
        minute_index = _minute_index_for_cycle(actual_minute_base, cycle_origin)
        actual_minute = (
            actual_minute_base.loc[
                actual_minute_base["timestamp"].isin(minute_index),
                ["timestamp", "avg_load"],
            ]
            .drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")
            .reindex(minute_index)
        )
        if actual_minute["avg_load"].isna().any():
            raise RuntimeError(f"Actual minute grid is incomplete for control cycle {cycle_origin.isoformat()}.")
        day_ahead_series = _extract_candidate_path(
            cast(dict[str, Any], day_ahead[str(result_key)])["detail_by_origin"],
            origin_timestamp=cycle_origin,
            candidate_label=str(day_ahead["candidate_label"]),
            minute_index=minute_index,
        )
        hourly_series = _apply_rollout_updates(
            day_ahead_series,
            detail_by_origin=cast(dict[str, Any], hourly[str(result_key)])["detail_by_origin"],
            candidate_label=str(hourly["candidate_label"]),
            update_origins=[
                timestamp
                for timestamp in hourly_origins
                if cycle_origin <= timestamp < cycle_origin + pd.Timedelta(minutes=day_ahead_horizon)
            ],
            horizon_minutes=hourly_horizon,
        )
        phase_series = _apply_rollout_updates(
            hourly_series,
            detail_by_origin=cast(dict[str, Any], phase[str(result_key)])["detail_by_origin"],
            candidate_label=str(phase["candidate_label"]),
            update_origins=[
                timestamp
                for timestamp in phase_origins
                if cycle_origin <= timestamp < cycle_origin + pd.Timedelta(minutes=day_ahead_horizon)
            ],
            horizon_minutes=phase_horizon,
        )
        minute_frames.append(
            pd.DataFrame(
                {
                    "cycle_origin_timestamp": pd.Timestamp(cycle_origin).isoformat(),
                    "timestamp": minute_index,
                    "actual_load": actual_minute["avg_load"].to_numpy(dtype=float),
                    "day_ahead_pred": day_ahead_series.to_numpy(dtype=float),
                    "hourly_pred": hourly_series.to_numpy(dtype=float),
                    "phase_pred": phase_series.to_numpy(dtype=float),
                }
            )
        )
    if not minute_frames:
        return pd.DataFrame(
            columns=[
                "cycle_origin_timestamp",
                "timestamp",
                "actual_load",
                "day_ahead_pred",
                "hourly_pred",
                "phase_pred",
            ]
        )
    return pd.concat(minute_frames, ignore_index=True)


def _replay_selected_scope_result(
    *,
    layer_payload: dict[str, Any],
    cache_root: Path | None,
    temp_output_root: Path,
    layer_role: str,
    horizon_minutes: int,
    origin_timestamps: list[pd.Timestamp],
) -> dict[str, Any]:
    """Replay the selected Stage-10 layer on an arbitrary origin set."""
    selection = dict(cast(dict[str, Any], layer_payload["selection"]))
    result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_output_root,
        layer_role=str(layer_role),
        selection=selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(layer_payload["origin_policy"]),
        selection_target=str(layer_payload["selection_target"]),
        origin_timestamps=[pd.Timestamp(value) for value in origin_timestamps],
        capture_path_details=True,
        candidate_scope="selected_only",
        persist_artifacts=False,
    )
    by_origin = result["by_origin"].copy()
    if not by_origin.empty:
        by_origin["native_candidate_label"] = by_origin["candidate_label"].astype("string")
        by_origin["candidate_label"] = by_origin["native_candidate_label"].map(
            lambda value: _qualify_control_candidate_label(selection, str(value))
        )
        result["by_origin"] = by_origin
    detail_by_origin = result["detail_by_origin"].copy()
    if not detail_by_origin.empty:
        detail_by_origin["native_candidate_label"] = detail_by_origin["candidate_label"].astype("string")
        detail_by_origin["candidate_label"] = detail_by_origin["native_candidate_label"].map(
            lambda value: _qualify_control_candidate_label(selection, str(value))
        )
        result["detail_by_origin"] = detail_by_origin
    return result


def _resolve_phase_stack_replay_metadata(
    *,
    phase_payload: dict[str, Any],
    phase_stack_selected_row: pd.Series,
    phase_stack_guard: dict[str, Any],
) -> dict[str, Any]:
    """Resolve how the selected phase policy should be replayed on another scope."""
    def _selection_for_parent(parent_label: str) -> dict[str, Any]:
        if parent_label == str(phase_payload["candidate_label"]):
            return dict(cast(dict[str, Any], phase_payload["selection"]))
        selected_pool_rank = int(float(phase_stack_selected_row.get("replay_pool_rank", float("nan"))))
        candidate_selection_lookup = cast(dict[int, dict[str, Any]], phase_payload["candidate_selection_by_pool_rank"])
        selection = dict(candidate_selection_lookup[selected_pool_rank])
        selection["requested_candidate_label"] = _native_control_candidate_label(selection, parent_label)
        return selection

    selected_candidate_label = str(phase_stack_guard["applied_candidate_label"])
    if str(phase_stack_guard["recommended_policy"]) != "phase_candidate":
        return {
            "mode": "hourly_passthrough",
            "candidate_label": selected_candidate_label,
            "selection": None,
            "blend_weight": float("nan"),
            "blend_parent_candidate_label": "",
            "reference_candidate_label": "",
            "replay_run_dir": "",
        }
    if str(phase_stack_selected_row.get("stack_candidate_family", "")) == "phase_bucket_portfolio":
        return {
            "mode": "phase_bucket_portfolio",
            "candidate_label": selected_candidate_label,
            "selection": None,
            "blend_weight": float("nan"),
            "blend_parent_candidate_label": "",
            "reference_candidate_label": "",
            "replay_run_dir": str(phase_stack_selected_row.get("replay_run_dir", "")),
            "bucket_policy_json": str(phase_stack_selected_row.get("stack_bucket_policy_json", "")),
            "bucket_granularity_minutes": int(
                float(phase_stack_selected_row.get("stack_bucket_granularity_minutes", float("nan")))
            ),
        }
    if str(phase_stack_selected_row.get("stack_candidate_family", "")) == "hourly_phase_blend":
        parent_label = str(phase_stack_selected_row.get("stack_blend_parent_candidate_label", ""))
        return {
            "mode": "hourly_phase_blend",
            "candidate_label": selected_candidate_label,
            "selection": _selection_for_parent(parent_label),
            "blend_weight": float(phase_stack_selected_row.get("stack_blend_weight", float("nan"))),
            "blend_parent_candidate_label": parent_label,
            "reference_candidate_label": "",
            "bucket_weight_json": "",
            "replay_run_dir": str(phase_stack_selected_row.get("replay_run_dir", "")),
        }
    if str(phase_stack_selected_row.get("stack_candidate_family", "")) == "phase_baseline_control_blend":
        parent_label = str(phase_stack_selected_row.get("stack_blend_parent_candidate_label", ""))
        return {
            "mode": "phase_baseline_control_blend",
            "candidate_label": selected_candidate_label,
            "selection": _selection_for_parent(parent_label),
            "blend_weight": float(phase_stack_selected_row.get("stack_blend_weight", float("nan"))),
            "blend_parent_candidate_label": parent_label,
            "reference_candidate_label": str(phase_stack_selected_row.get("stack_reference_candidate_label", "")),
            "bucket_weight_json": "",
            "replay_run_dir": str(phase_stack_selected_row.get("replay_run_dir", "")),
        }
    if str(phase_stack_selected_row.get("stack_candidate_family", "")) == "phase_baseline_bucket_control_blend":
        parent_label = str(phase_stack_selected_row.get("stack_blend_parent_candidate_label", ""))
        return {
            "mode": "phase_baseline_bucket_control_blend",
            "candidate_label": selected_candidate_label,
            "selection": _selection_for_parent(parent_label),
            "blend_weight": float("nan"),
            "blend_parent_candidate_label": parent_label,
            "reference_candidate_label": str(phase_stack_selected_row.get("stack_reference_candidate_label", "")),
            "bucket_weight_json": str(phase_stack_selected_row.get("stack_bucket_weight_json", "")),
            "bucket_granularity_minutes": int(
                float(phase_stack_selected_row.get("stack_bucket_granularity_minutes", float("nan")))
            ),
            "replay_run_dir": str(phase_stack_selected_row.get("replay_run_dir", "")),
        }
    if selected_candidate_label == str(phase_payload["candidate_label"]):
        return {
            "mode": "native_phase_candidate",
            "candidate_label": selected_candidate_label,
            "selection": dict(cast(dict[str, Any], phase_payload["selection"])),
            "blend_weight": float("nan"),
            "blend_parent_candidate_label": "",
            "reference_candidate_label": "",
            "replay_run_dir": _relative_artifact_path(cast(dict[str, Any], phase_payload["result"])["run_dir"]),
        }
    return {
        "mode": "native_phase_candidate",
        "candidate_label": selected_candidate_label,
        "selection": _selection_for_parent(selected_candidate_label),
        "blend_weight": float("nan"),
        "blend_parent_candidate_label": "",
        "reference_candidate_label": "",
        "replay_run_dir": str(phase_stack_selected_row.get("replay_run_dir", "")),
    }


def _qualify_phase_replay_detail(
    detail_by_origin: pd.DataFrame,
    *,
    replay_selection: dict[str, Any],
) -> pd.DataFrame:
    """Normalize replayed phase-detail labels so reconstructed stack policies can match them."""
    qualified = detail_by_origin.copy()
    if qualified.empty:
        return qualified
    qualified["native_candidate_label"] = qualified["candidate_label"].astype("string")
    qualified["candidate_label"] = qualified["native_candidate_label"].map(
        lambda value: _qualify_control_candidate_label(replay_selection, str(value))
    )
    return qualified


def _phase_replay_detail_strategy(
    *,
    phase_payload: dict[str, Any],
    phase_replay_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Choose how to rematerialize phase-detail paths for another scope.

    Native phase candidates and blend wrappers can usually be replayed from their
    stored selection directly. Stack-composed policies such as bucket portfolios
    may not have a single native selection, so they must replay the parent phase
    family and reconstruct the policy from those qualified candidate paths.
    """
    mode = str(phase_replay_metadata.get("mode", ""))
    if mode == "hourly_passthrough":
        return {
            "enabled": False,
            "selection": None,
            "candidate_scope": "",
            "qualification_selection": None,
            "replay_mode": "hourly_passthrough",
        }
    replay_selection = cast(dict[str, Any] | None, phase_replay_metadata.get("selection"))
    if replay_selection is not None:
        replay_selection = dict(replay_selection)
        return {
            "enabled": True,
            "selection": replay_selection,
            "candidate_scope": "selected_only",
            "qualification_selection": replay_selection,
            "replay_mode": "selection_replay",
        }
    family_selection = dict(cast(dict[str, Any], phase_payload["selection"]))
    return {
        "enabled": True,
        "selection": family_selection,
        "candidate_scope": "full_family",
        "qualification_selection": family_selection,
        "replay_mode": "family_reconstruction",
    }


def _replay_phase_detail_for_scope(
    *,
    phase_payload: dict[str, Any],
    phase_replay_metadata: dict[str, Any],
    cache_root: Path | None,
    temp_output_root: Path,
    origin_timestamps: list[pd.Timestamp],
    horizon_minutes: int,
    persist_artifacts: bool,
) -> pd.DataFrame:
    """Replay the phase layer on one origin set and return qualified detail rows."""
    replay_strategy = _phase_replay_detail_strategy(
        phase_payload=phase_payload,
        phase_replay_metadata=phase_replay_metadata,
    )
    if not bool(replay_strategy["enabled"]) or not origin_timestamps:
        return pd.DataFrame()
    replay_selection = dict(cast(dict[str, Any], replay_strategy["selection"]))
    representable_origins = _representable_selection_origins(
        selection=replay_selection,
        horizon_minutes=int(horizon_minutes),
        origin_timestamps=[pd.Timestamp(value) for value in origin_timestamps],
    )
    if not representable_origins:
        return pd.DataFrame()
    replay_result = _run_cached_rollout_evaluation(
        cache_root=cache_root,
        temp_output_root=temp_output_root,
        layer_role="phase",
        selection=replay_selection,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(phase_payload["origin_policy"]),
        selection_target=str(phase_payload["selection_target"]),
        origin_timestamps=representable_origins,
        capture_path_details=True,
        candidate_scope=str(replay_strategy["candidate_scope"]),
        persist_artifacts=bool(persist_artifacts),
    )
    return _qualify_phase_replay_detail(
        replay_result["detail_by_origin"],
        replay_selection=cast(dict[str, Any], replay_strategy["qualification_selection"]),
    )


def _selected_phase_series_for_scope(
    *,
    minute_timeline: pd.DataFrame,
    hourly_pred_column: str,
    phase_replay_metadata: dict[str, Any],
    phase_detail_by_origin: pd.DataFrame,
    phase_origins: list[pd.Timestamp],
    phase_horizon: int,
) -> pd.Series:
    """Build the selected phase policy on top of one hourly control path."""
    timeline_index = pd.DatetimeIndex(pd.to_datetime(minute_timeline["timestamp"], errors="raise"))
    hourly_series = pd.Series(
        minute_timeline[hourly_pred_column].to_numpy(dtype=float),
        index=timeline_index,
        dtype=float,
    )
    mode = str(phase_replay_metadata["mode"])
    if mode == "hourly_passthrough":
        return hourly_series.astype(float)
    if mode == "phase_bucket_portfolio":
        bucket_policy_json = str(phase_replay_metadata.get("bucket_policy_json", ""))
        if not bucket_policy_json:
            return hourly_series.astype(float)
        candidate_by_bucket = {
            int(key): str(value) for key, value in json.loads(bucket_policy_json).items()
        }
        bucket_minutes = int(phase_replay_metadata.get("bucket_granularity_minutes", 15))
        return _apply_bucketed_rollout_updates(
            hourly_series,
            detail_by_origin=phase_detail_by_origin,
            candidate_by_bucket=candidate_by_bucket,
            update_origins=phase_origins,
            horizon_minutes=int(phase_horizon),
            bucket_minutes=int(bucket_minutes),
            cycle_minutes=60,
        )
    if mode in {"phase_baseline_control_blend", "phase_baseline_bucket_control_blend"}:
        parent_label = str(phase_replay_metadata.get("blend_parent_candidate_label", ""))
        reference_label = str(phase_replay_metadata.get("reference_candidate_label", ""))
        if not parent_label or not reference_label:
            return hourly_series.astype(float)
        parent_series = _apply_rollout_updates(
            hourly_series,
            detail_by_origin=phase_detail_by_origin,
            candidate_label=parent_label,
            update_origins=phase_origins,
            horizon_minutes=int(phase_horizon),
        )
        reference_series = _apply_rollout_updates(
            hourly_series,
            detail_by_origin=phase_detail_by_origin,
            candidate_label=reference_label,
            update_origins=phase_origins,
            horizon_minutes=int(phase_horizon),
        )
        if mode == "phase_baseline_control_blend":
            return _blend_control_prediction_series(
                candidate_series=parent_series,
                reference_series=reference_series,
                candidate_weight=float(phase_replay_metadata.get("blend_weight", float("nan"))),
            )
        bucket_weight_json = str(phase_replay_metadata.get("bucket_weight_json", ""))
        if not bucket_weight_json:
            return reference_series.astype(float)
        bucket_weights = {int(key): float(value) for key, value in json.loads(bucket_weight_json).items()}
        return _blend_control_prediction_series_by_bucket(
            candidate_series=parent_series,
            reference_series=reference_series,
            candidate_weight_by_bucket=bucket_weights,
            bucket_size_minutes=int(phase_replay_metadata.get("bucket_granularity_minutes", 5)),
            lock_interval_minutes=int(phase_horizon),
        )
    candidate_label = str(
        phase_replay_metadata["blend_parent_candidate_label"]
        if mode == "hourly_phase_blend"
        else phase_replay_metadata["candidate_label"]
    )
    phase_series = _apply_rollout_updates(
        hourly_series,
        detail_by_origin=phase_detail_by_origin,
        candidate_label=str(candidate_label),
        update_origins=phase_origins,
        horizon_minutes=int(phase_horizon),
    )
    if mode != "hourly_phase_blend":
        return phase_series.astype(float)
    blend_weight = float(phase_replay_metadata["blend_weight"])
    return (hourly_series + blend_weight * (phase_series - hourly_series)).astype(float)


def _prepare_rolling_phase_support_context(
    *,
    actual_minute_base: pd.DataFrame,
    cache_root: Path | None,
    day_ahead: dict[str, Any],
    hourly: dict[str, Any],
    phase: dict[str, Any],
    day_ahead_horizon: int,
    hourly_horizon: int,
    phase_horizon: int,
    rolling_calibration_cycle_origins: list[pd.Timestamp],
    rolling_evaluation_cycle_origins: list[pd.Timestamp],
    phase_replay_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Prepare broader rolling timelines for phase-support checks and final replay."""
    if not rolling_calibration_cycle_origins or not rolling_evaluation_cycle_origins:
        return {}
    rolling_hourly_origins = _layer_update_origins(
        cycle_origins=rolling_calibration_cycle_origins,
        update_interval_minutes=hourly_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    rolling_hourly_evaluation_origins = _layer_update_origins(
        cycle_origins=rolling_evaluation_cycle_origins,
        update_interval_minutes=hourly_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    rolling_phase_origins = _layer_update_origins(
        cycle_origins=rolling_calibration_cycle_origins,
        update_interval_minutes=phase_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    rolling_phase_evaluation_origins = _layer_update_origins(
        cycle_origins=rolling_evaluation_cycle_origins,
        update_interval_minutes=phase_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    rolling_refresh_origins = _day_ahead_refresh_origins(rolling_calibration_cycle_origins)
    rolling_refresh_evaluation_origins = _day_ahead_refresh_origins(rolling_evaluation_cycle_origins)
    rolling_cycle_union_origins = sorted(
        {
            pd.Timestamp(value)
            for value in [*rolling_calibration_cycle_origins, *rolling_evaluation_cycle_origins]
        }
    )
    rolling_hourly_union_origins = sorted(
        {
            pd.Timestamp(value)
            for value in [*rolling_hourly_origins, *rolling_hourly_evaluation_origins]
        }
    )
    rolling_phase_union_origins = sorted(
        {
            pd.Timestamp(value)
            for value in [*rolling_phase_origins, *rolling_phase_evaluation_origins]
        }
    )
    with TemporaryDirectory(prefix="elf_forecast_control_rolling_phase_") as rolling_temp_dir:
        rolling_temp_root = Path(rolling_temp_dir)
        rolling_day_ahead_union = _replay_selected_scope_result(
            layer_payload=day_ahead,
            cache_root=cache_root,
            temp_output_root=rolling_temp_root / "day_ahead_union",
            layer_role="day_ahead",
            horizon_minutes=day_ahead_horizon,
            origin_timestamps=rolling_cycle_union_origins,
        )
        rolling_day_ahead_calibration = _subset_rollout_result_for_origins(
            rolling_day_ahead_union,
            origin_timestamps=rolling_calibration_cycle_origins,
            require_path_details=True,
        )
        rolling_day_ahead_evaluation = _subset_rollout_result_for_origins(
            rolling_day_ahead_union,
            origin_timestamps=rolling_evaluation_cycle_origins,
            require_path_details=True,
        )
        rolling_hourly_union = _replay_selected_scope_result(
            layer_payload=hourly,
            cache_root=cache_root,
            temp_output_root=rolling_temp_root / "hourly_union",
            layer_role="hourly",
            horizon_minutes=hourly_horizon,
            origin_timestamps=rolling_hourly_union_origins,
        )
        rolling_hourly_calibration = _subset_rollout_result_for_origins(
            rolling_hourly_union,
            origin_timestamps=rolling_hourly_origins,
            require_path_details=True,
        )
        rolling_hourly_evaluation = _subset_rollout_result_for_origins(
            rolling_hourly_union,
            origin_timestamps=rolling_hourly_evaluation_origins,
            require_path_details=True,
        )
        rolling_passthrough_phase = {
            "benchmark_result": {"detail_by_origin": pd.DataFrame()},
            "result": {"detail_by_origin": pd.DataFrame()},
            "candidate_label": str(hourly["candidate_label"]),
        }
        rolling_calibration_hourly_timeline = _build_control_minute_timeline(
            cycle_origins=rolling_calibration_cycle_origins,
            actual_minute_base=actual_minute_base,
            day_ahead={
                **day_ahead,
                "benchmark_result": {
                    "detail_by_origin": rolling_day_ahead_calibration["detail_by_origin"].copy()
                },
            },
            hourly={
                **hourly,
                "benchmark_result": {
                    "detail_by_origin": rolling_hourly_calibration["detail_by_origin"].copy()
                },
            },
            phase=rolling_passthrough_phase,
            result_key="benchmark_result",
            day_ahead_horizon=day_ahead_horizon,
            hourly_horizon=hourly_horizon,
            phase_horizon=phase_horizon,
            hourly_origins=rolling_hourly_origins,
            phase_origins=[],
        )
        rolling_evaluation_hourly_timeline = _build_control_minute_timeline(
            cycle_origins=rolling_evaluation_cycle_origins,
            actual_minute_base=actual_minute_base,
            day_ahead={
                **day_ahead,
                "result": {
                    "detail_by_origin": rolling_day_ahead_evaluation["detail_by_origin"].copy()
                },
            },
            hourly={
                **hourly,
                "result": {
                    "detail_by_origin": rolling_hourly_evaluation["detail_by_origin"].copy()
                },
            },
            phase=rolling_passthrough_phase,
            result_key="result",
            day_ahead_horizon=day_ahead_horizon,
            hourly_horizon=hourly_horizon,
            phase_horizon=phase_horizon,
            hourly_origins=rolling_hourly_evaluation_origins,
            phase_origins=[],
        )
        rolling_phase_union_detail = _replay_phase_detail_for_scope(
            phase_payload=phase,
            phase_replay_metadata=phase_replay_metadata,
            cache_root=cache_root,
            temp_output_root=rolling_temp_root / "phase_union",
            origin_timestamps=rolling_phase_union_origins,
            horizon_minutes=phase_horizon,
            persist_artifacts=False,
        )
        rolling_phase_calibration_detail = _subset_detail_by_origin(
            rolling_phase_union_detail,
            origin_timestamps=rolling_phase_origins,
        )
        rolling_phase_evaluation_detail = _subset_detail_by_origin(
            rolling_phase_union_detail,
            origin_timestamps=rolling_phase_evaluation_origins,
        )
        rolling_calibration_phase_series = _selected_phase_series_for_scope(
            minute_timeline=rolling_calibration_hourly_timeline,
            hourly_pred_column="hourly_pred",
            phase_replay_metadata=phase_replay_metadata,
            phase_detail_by_origin=rolling_phase_calibration_detail,
            phase_origins=rolling_phase_origins,
            phase_horizon=phase_horizon,
        )
        rolling_evaluation_phase_series = _selected_phase_series_for_scope(
            minute_timeline=rolling_evaluation_hourly_timeline,
            hourly_pred_column="hourly_pred",
            phase_replay_metadata=phase_replay_metadata,
            phase_detail_by_origin=rolling_phase_evaluation_detail,
            phase_origins=rolling_phase_evaluation_origins,
            phase_horizon=phase_horizon,
        )
    rolling_calibration_phase_timeline = rolling_calibration_hourly_timeline.copy()
    rolling_calibration_phase_timeline["phase_pred"] = rolling_calibration_phase_series.reindex(
        pd.DatetimeIndex(pd.to_datetime(rolling_calibration_phase_timeline["timestamp"], errors="raise"))
    ).to_numpy(dtype=float)
    rolling_evaluation_phase_timeline = rolling_evaluation_hourly_timeline.copy()
    rolling_evaluation_phase_timeline["phase_pred"] = rolling_evaluation_phase_series.reindex(
        pd.DatetimeIndex(pd.to_datetime(rolling_evaluation_phase_timeline["timestamp"], errors="raise"))
    ).to_numpy(dtype=float)
    return {
        "rolling_hourly_origins": rolling_hourly_origins,
        "rolling_hourly_evaluation_origins": rolling_hourly_evaluation_origins,
        "rolling_phase_origins": rolling_phase_origins,
        "rolling_phase_evaluation_origins": rolling_phase_evaluation_origins,
        "rolling_refresh_origins": rolling_refresh_origins,
        "rolling_refresh_evaluation_origins": rolling_refresh_evaluation_origins,
        "day_ahead_calibration_detail": rolling_day_ahead_calibration["detail_by_origin"].copy(),
        "day_ahead_evaluation_detail": rolling_day_ahead_evaluation["detail_by_origin"].copy(),
        "calibration_hourly_timeline": rolling_calibration_hourly_timeline,
        "evaluation_hourly_timeline": rolling_evaluation_hourly_timeline,
        "calibration_phase_timeline": rolling_calibration_phase_timeline,
        "evaluation_phase_timeline": rolling_evaluation_phase_timeline,
    }


def _phase_bucket_policy_from_origin_metrics(
    detail_by_origin: pd.DataFrame,
    *,
    selection_metric: str,
    bucket_minutes: int,
) -> dict[int, str]:
    """Choose the best phase candidate label for each update bucket from origin-level replay metrics."""
    if detail_by_origin.empty or selection_metric not in detail_by_origin.columns:
        return {}
    working = detail_by_origin.copy()
    working["bucket_key"] = working["origin_timestamp"].map(
        lambda value: _timestamp_minute_bucket(
            value,
            bucket_minutes=int(bucket_minutes),
            cycle_minutes=60,
        )
    )
    grouped = (
        working.groupby(["bucket_key", "candidate_label"], dropna=False)[selection_metric]
        .mean()
        .reset_index()
        .sort_values(["bucket_key", selection_metric, "candidate_label"], kind="stable")
    )
    if grouped.empty:
        return {}
    winners = grouped.groupby("bucket_key", as_index=False).first()
    return {
        int(row["bucket_key"]): str(row["candidate_label"])
        for _, row in winners.iterrows()
        if pd.notna(row["candidate_label"])
    }


def _phase_stack_origin_metric_rows(
    *,
    candidate_label: str,
    minute_timeline: pd.DataFrame,
    predicted_series: pd.Series,
    phase_origins: list[pd.Timestamp],
    horizon_minutes: int,
) -> list[dict[str, Any]]:
    """Score one stacked phase candidate on each update origin so bucket routing can be stack-aware."""
    working = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(minute_timeline["timestamp"], errors="raise"),
            "actual_load": minute_timeline["actual_load"].to_numpy(dtype=float),
            "predicted_load": pd.Series(predicted_series, copy=False).to_numpy(dtype=float),
        }
    )
    rows: list[dict[str, Any]] = []
    for origin_timestamp in sorted(pd.Timestamp(value) for value in phase_origins):
        segment = working.loc[
            (working["timestamp"] > pd.Timestamp(origin_timestamp))
            & (working["timestamp"] <= pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=horizon_minutes))
        ].copy()
        if segment.empty:
            continue
        valid = segment["actual_load"].notna() & segment["predicted_load"].notna()
        if not bool(valid.any()):
            continue
        abs_error = (segment.loc[valid, "predicted_load"] - segment.loc[valid, "actual_load"]).abs()
        rows.append(
            {
                "origin_timestamp": pd.Timestamp(origin_timestamp).isoformat(),
                "candidate_label": str(candidate_label),
                "next_lock_mae": float(abs_error.mean()),
            }
        )
    return rows


def _evaluate_control_scope(
    *,
    minute_timeline: pd.DataFrame,
    nowcast_anchor: dict[str, Any],
    lock_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate one control timeline into interval, per-cycle, and summary artifacts."""
    if minute_timeline.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    interval_frames: list[pd.DataFrame] = []
    cycle_rows: list[dict[str, Any]] = []
    for cycle_origin, cycle_frame in minute_timeline.groupby("cycle_origin_timestamp", sort=True):
        cycle_timestamp = pd.Timestamp(cycle_origin)
        interval_frame = _interval_timeline_for_cycle(
            cycle_origin_timestamp=cycle_timestamp,
            minute_frame=cycle_frame,
            lock_interval_minutes=lock_interval,
        )
        interval_frames.append(interval_frame)
        cycle_rows.append(
            _cycle_summary(
                cycle_origin_timestamp=cycle_timestamp,
                minute_frame=cycle_frame,
                interval_frame=interval_frame,
            )
        )
    interval_timeline = pd.concat(interval_frames, ignore_index=True)
    by_cycle = pd.DataFrame(cycle_rows).sort_values("cycle_origin_timestamp", kind="stable").reset_index(drop=True)
    summary = _summary_frame(by_cycle, nowcast_anchor).reset_index(drop=True)
    return interval_timeline, by_cycle, summary


def _control_layer_specs(minute_frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the ordered prediction columns currently present in one control minute frame."""
    layers = [
        ("day_ahead", "day_ahead_pred"),
        ("hourly", "hourly_pred"),
        ("phase", "phase_pred"),
    ]
    if "nowcast_pred" in minute_frame.columns:
        layers.append(("nowcast", "nowcast_pred"))
    return layers


def _control_layer_label(layer_name: str) -> str:
    """Map an internal layer role to the human-readable Stage-10 summary label."""
    return {
        "day_ahead": "day_ahead_frozen",
        "hourly": "after_hourly_updates",
        "phase": "after_phase_updates",
        "nowcast": "after_nowcast_updates",
    }[str(layer_name)]


@lru_cache(maxsize=1)
def _load_optimizer_delivery_context_by_timestamp_cached() -> pd.DataFrame:
    """Load cached minute-level operating context used by the dynamic nowcast controller."""
    try:
        gold = _load_stage5_gold_with_full_grid(
            str(MULTIRES_FORECAST_CONTROL["actual_resolution"]),
            preferred_output_path(PATHS["gold_dir"]),
        )
    except Exception:
        return pd.DataFrame()
    try:
        gold, _ = _augment_stage5_curated_ramp_features(
            gold,
            ramp_quantile=float(MODELING_PERFORMANCE_RAMP["quantile"]),
        )
    except Exception:
        pass
    context_columns = [
        column_name
        for column_name in (
            "timestamp",
            "workday_transition",
            "profile_active_flag",
            "profile_activity_ratio",
            "ramp_flag",
        )
        if column_name in gold.columns
    ]
    if "timestamp" not in context_columns:
        return pd.DataFrame()
    context = gold.loc[:, context_columns].copy()
    context["timestamp"] = pd.to_datetime(context["timestamp"], errors="coerce")
    context = context.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    for column_name in ("workday_transition", "profile_active_flag", "profile_activity_ratio", "ramp_flag"):
        if column_name in context.columns:
            context[column_name] = pd.to_numeric(context[column_name], errors="coerce").astype(float)
    return context.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _load_optimizer_delivery_context_by_timestamp() -> pd.DataFrame:
    """Return a defensive copy of the cached optimizer-delivery context table."""
    cached = _load_optimizer_delivery_context_by_timestamp_cached()
    return cached.copy() if not cached.empty else pd.DataFrame()


def _derive_interval_operating_regime(
    *,
    transition_active: bool,
    profile_active_fraction: float,
) -> str:
    """Map interval context into the same broad operating regimes used by Stage-5 advisory evidence."""
    profile_threshold = float(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_profile_active_threshold"])
    profile_active = bool(np.isfinite(profile_active_fraction) and profile_active_fraction >= profile_threshold)
    if bool(transition_active) and profile_active:
        return "transition_active"
    if bool(transition_active):
        return "transition_only"
    if profile_active:
        return "active_only"
    return "none_inactive"


def _derive_interval_ramp_band(high_ramp_fraction: float) -> str:
    """Bucket interval-level ramp share into the same labels used by Stage-5 advisory outputs."""
    if not np.isfinite(high_ramp_fraction):
        return "unknown"
    threshold = float(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_high_ramp_fraction_threshold"])
    return "high_ramp" if float(high_ramp_fraction) >= threshold else "stable_ramp"


def _layer_cycle_metrics(
    *,
    minute_frame: pd.DataFrame,
    layer_name: str,
    prediction_column: str,
    lock_interval_minutes: int,
) -> dict[str, float]:
    """Compute one control layer's per-cycle path, lock, shape, and energy metrics."""
    actual_minute = minute_frame.set_index("timestamp")["actual_load"].astype(float)
    predicted_minute = minute_frame.set_index("timestamp")[prediction_column].astype(float)
    metrics = _layer_metrics(actual_minute, predicted_minute)

    working = minute_frame.loc[:, ["timestamp", "actual_load", prediction_column]].copy()
    working["interval_start"] = working["timestamp"].dt.floor(f"{lock_interval_minutes}min")
    interval = (
        working.groupby("interval_start", dropna=False)
        .agg(
            actual_interval_mean=("actual_load", "mean"),
            predicted_interval_mean=(prediction_column, "mean"),
        )
        .reset_index()
        .sort_values("interval_start", kind="stable")
    )
    interval_abs_error = (
        interval["actual_interval_mean"].astype(float) - interval["predicted_interval_mean"].astype(float)
    ).abs()
    interval_actual = interval["actual_interval_mean"].astype(float)
    next_lock_mae = float(interval_abs_error.iloc[0]) if not interval_abs_error.empty else float("nan")
    next_lock_actual = float(abs(interval_actual.iloc[0])) if not interval_actual.empty else float("nan")
    peak_valid = interval.loc[
        interval["actual_interval_mean"].notna() & interval["predicted_interval_mean"].notna()
    ].copy()
    peak_value_mae = float("nan")
    peak_value_mae_pct = float("nan")
    peak_interval_hit = float("nan")
    peak_interval_offset_minutes = float("nan")
    if not peak_valid.empty:
        actual_peak_idx = peak_valid["actual_interval_mean"].astype(float).idxmax()
        predicted_peak_idx = peak_valid["predicted_interval_mean"].astype(float).idxmax()
        actual_peak_start = pd.Timestamp(peak_valid.loc[actual_peak_idx, "interval_start"])
        predicted_peak_start = pd.Timestamp(peak_valid.loc[predicted_peak_idx, "interval_start"])
        actual_peak_value = float(peak_valid.loc[actual_peak_idx, "actual_interval_mean"])
        predicted_peak_value = float(peak_valid.loc[predicted_peak_idx, "predicted_interval_mean"])
        peak_value_mae = float(abs(predicted_peak_value - actual_peak_value))
        peak_value_mae_pct = _safe_pct(peak_value_mae, abs(actual_peak_value))
        peak_interval_hit = float(actual_peak_start == predicted_peak_start)
        peak_interval_offset_minutes = float(
            abs((predicted_peak_start - actual_peak_start) / pd.Timedelta(minutes=1))
        )
    return {
        f"{layer_name}_minute_path_mae": float(metrics["minute_path_mae"]),
        f"{layer_name}_minute_path_mae_pct": float(metrics["minute_path_mae_pct"]),
        f"{layer_name}_lock_mae": float(interval_abs_error.mean()),
        f"{layer_name}_lock_mae_pct": _safe_pct(
            float(interval_abs_error.sum()),
            float(np.sum(np.abs(interval_actual.to_numpy(dtype=float)))),
        ),
        f"{layer_name}_next_lock_mae": next_lock_mae,
        f"{layer_name}_next_lock_mae_pct": _safe_pct(next_lock_mae, next_lock_actual),
        f"{layer_name}_profile_shape_mae": float(metrics["profile_shape_mae"]),
        f"{layer_name}_profile_shape_mae_pct": float(metrics["profile_shape_mae_pct"]),
        f"{layer_name}_energy_mae": float(metrics["energy_mae"]),
        f"{layer_name}_energy_mae_pct": float(metrics["energy_mae_pct"]),
        f"{layer_name}_peak_value_mae": peak_value_mae,
        f"{layer_name}_peak_value_mae_pct": peak_value_mae_pct,
        f"{layer_name}_peak_interval_hit": peak_interval_hit,
        f"{layer_name}_peak_interval_offset_minutes": peak_interval_offset_minutes,
    }


def _interval_timeline_for_cycle(
    *,
    cycle_origin_timestamp: pd.Timestamp,
    minute_frame: pd.DataFrame,
    lock_interval_minutes: int,
) -> pd.DataFrame:
    """Collapse minute predictions into the locked billing intervals used for control."""
    working = minute_frame.copy()
    context_lookup = _load_optimizer_delivery_context_by_timestamp()
    if not context_lookup.empty and "timestamp" in working.columns:
        working = working.merge(context_lookup, on="timestamp", how="left")
    working["interval_start"] = working["timestamp"].dt.floor(f"{lock_interval_minutes}min")
    named_aggregations: dict[str, tuple[str, str]] = {"actual_interval_mean": ("actual_load", "mean")}
    for layer_name, prediction_column in _control_layer_specs(minute_frame):
        named_aggregations[f"{layer_name}_interval_mean"] = (prediction_column, "mean")
    if "workday_transition" in working.columns:
        named_aggregations["workday_transition_fraction"] = ("workday_transition", "mean")
    if "profile_active_flag" in working.columns:
        named_aggregations["profile_active_fraction"] = ("profile_active_flag", "mean")
    elif "profile_activity_ratio" in working.columns:
        named_aggregations["profile_active_fraction"] = ("profile_activity_ratio", "mean")
    if "ramp_flag" in working.columns:
        named_aggregations["high_ramp_fraction"] = ("ramp_flag", "mean")
    grouped = (
        working.groupby("interval_start", dropna=False)
        .agg(**named_aggregations)
        .reset_index()
        .sort_values("interval_start", kind="stable")
    )
    grouped["cycle_origin_timestamp"] = pd.Timestamp(cycle_origin_timestamp).isoformat()
    grouped["interval_end"] = grouped["interval_start"] + pd.Timedelta(minutes=lock_interval_minutes)
    if "workday_transition_fraction" in grouped.columns:
        grouped["workday_transition_active"] = (
            pd.to_numeric(grouped["workday_transition_fraction"], errors="coerce").fillna(0.0).gt(0.0)
        )
    else:
        grouped["workday_transition_active"] = False
    if "profile_active_fraction" not in grouped.columns:
        grouped["profile_active_fraction"] = float("nan")
    grouped["profile_active_majority"] = [
        bool(
            np.isfinite(value)
            and float(value) >= float(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_profile_active_threshold"])
        )
        for value in pd.to_numeric(grouped["profile_active_fraction"], errors="coerce").to_numpy(dtype=float)
    ]
    if "high_ramp_fraction" not in grouped.columns:
        grouped["high_ramp_fraction"] = float("nan")
    grouped["actual_ramp_band"] = [
        _derive_interval_ramp_band(float(value))
        for value in pd.to_numeric(grouped["high_ramp_fraction"], errors="coerce").to_numpy(dtype=float)
    ]
    grouped["operating_regime"] = [
        _derive_interval_operating_regime(
            transition_active=bool(transition_active),
            profile_active_fraction=float(profile_fraction),
        )
        for transition_active, profile_fraction in zip(
            grouped["workday_transition_active"].to_numpy(dtype=bool),
            pd.to_numeric(grouped["profile_active_fraction"], errors="coerce").to_numpy(dtype=float),
            strict=False,
        )
    ]
    for layer, _ in _control_layer_specs(minute_frame):
        grouped[f"{layer}_abs_error"] = (
            grouped[f"{layer}_interval_mean"] - grouped["actual_interval_mean"]
        ).abs()
    return grouped


def _cycle_summary(
    *,
    cycle_origin_timestamp: pd.Timestamp,
    minute_frame: pd.DataFrame,
    interval_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Reduce one control cycle to the reported lock, path, shape, and energy metrics."""
    actual_minute = minute_frame.set_index("timestamp")["actual_load"].astype(float)
    summary: dict[str, Any] = {
        "cycle_origin_timestamp": pd.Timestamp(cycle_origin_timestamp).isoformat(),
        "minute_n": int(len(actual_minute)),
        "interval_n": int(len(interval_frame)),
    }
    for layer_name, prediction_column in _control_layer_specs(minute_frame):
        summary.update(
            _layer_cycle_metrics(
                minute_frame=minute_frame,
                layer_name=str(layer_name),
                prediction_column=str(prediction_column),
                lock_interval_minutes=int(
                    MULTIRES_FORECAST_CONTROL["lock_interval_minutes"]
                ),
            )
        )
    summary["hourly_lock_mae_gain_vs_day_ahead"] = (
        summary["day_ahead_lock_mae"] - summary["hourly_lock_mae"]
    )
    summary["hourly_next_lock_mae_gain_vs_day_ahead"] = (
        summary["day_ahead_next_lock_mae"] - summary["hourly_next_lock_mae"]
    )
    summary["phase_lock_mae_gain_vs_day_ahead"] = (
        summary["day_ahead_lock_mae"] - summary["phase_lock_mae"]
    )
    summary["phase_next_lock_mae_gain_vs_day_ahead"] = (
        summary["day_ahead_next_lock_mae"] - summary["phase_next_lock_mae"]
    )
    summary["phase_profile_shape_mae_gain_vs_day_ahead"] = (
        summary["day_ahead_profile_shape_mae"] - summary["phase_profile_shape_mae"]
    )
    if "nowcast_pred" in minute_frame.columns:
        summary["nowcast_lock_mae_gain_vs_day_ahead"] = (
            summary["day_ahead_lock_mae"] - summary["nowcast_lock_mae"]
        )
        summary["nowcast_next_lock_mae_gain_vs_day_ahead"] = (
            summary["day_ahead_next_lock_mae"] - summary["nowcast_next_lock_mae"]
        )
        summary["nowcast_lock_mae_gain_vs_phase"] = (
            summary["phase_lock_mae"] - summary["nowcast_lock_mae"]
        )
        summary["nowcast_next_lock_mae_gain_vs_phase"] = (
            summary["phase_next_lock_mae"] - summary["nowcast_next_lock_mae"]
        )
        summary["nowcast_profile_shape_mae_gain_vs_day_ahead"] = (
            summary["day_ahead_profile_shape_mae"] - summary["nowcast_profile_shape_mae"]
        )
        summary["nowcast_profile_shape_mae_gain_vs_phase"] = (
            summary["phase_profile_shape_mae"] - summary["nowcast_profile_shape_mae"]
        )
    return summary


def _summary_frame(by_cycle: pd.DataFrame, nowcast_anchor: dict[str, Any]) -> pd.DataFrame:
    """Aggregate per-cycle results into the stage-level backtest summary."""
    def _mean_or_nan(column_name: str) -> float:
        if column_name not in by_cycle.columns:
            return float("nan")
        return float(pd.to_numeric(by_cycle[column_name], errors="coerce").mean())

    rows: list[dict[str, Any]] = []
    layer_names = ["day_ahead", "hourly", "phase"]
    if f"nowcast_lock_mae" in by_cycle.columns:
        layer_names.append("nowcast")
    for layer_name in layer_names:
        rows.append(
            {
                "layer": _control_layer_label(layer_name),
                "role": layer_name,
                "cycle_n": int(len(by_cycle)),
                "minute_path_mae": _mean_or_nan(f"{layer_name}_minute_path_mae"),
                "minute_path_mae_pct": _mean_or_nan(f"{layer_name}_minute_path_mae_pct"),
                "lock_mae": _mean_or_nan(f"{layer_name}_lock_mae"),
                "lock_mae_pct": _mean_or_nan(f"{layer_name}_lock_mae_pct"),
                "next_lock_mae": _mean_or_nan(f"{layer_name}_next_lock_mae"),
                "next_lock_mae_pct": _mean_or_nan(f"{layer_name}_next_lock_mae_pct"),
                "profile_shape_mae": _mean_or_nan(f"{layer_name}_profile_shape_mae"),
                "profile_shape_mae_pct": _mean_or_nan(f"{layer_name}_profile_shape_mae_pct"),
                "energy_mae": _mean_or_nan(f"{layer_name}_energy_mae"),
                "energy_mae_pct": _mean_or_nan(f"{layer_name}_energy_mae_pct"),
                "peak_value_mae": _mean_or_nan(f"{layer_name}_peak_value_mae"),
                "peak_value_mae_pct": _mean_or_nan(f"{layer_name}_peak_value_mae_pct"),
                "peak_interval_hit_rate": _mean_or_nan(f"{layer_name}_peak_interval_hit"),
                "peak_interval_offset_minutes": _mean_or_nan(f"{layer_name}_peak_interval_offset_minutes"),
            }
        )
    summary = pd.DataFrame(rows)
    day_ahead_lock = float(summary.loc[summary["role"].eq("day_ahead"), "lock_mae"].iloc[0])
    day_ahead_next_lock = float(summary.loc[summary["role"].eq("day_ahead"), "next_lock_mae"].iloc[0])
    day_ahead_profile = float(summary.loc[summary["role"].eq("day_ahead"), "profile_shape_mae"].iloc[0])
    summary["lock_mae_gain_vs_day_ahead"] = day_ahead_lock - summary["lock_mae"]
    summary["next_lock_mae_gain_vs_day_ahead"] = day_ahead_next_lock - summary["next_lock_mae"]
    summary["profile_shape_mae_gain_vs_day_ahead"] = day_ahead_profile - summary["profile_shape_mae"]
    summary["nowcast_candidate_label"] = str(nowcast_anchor["candidate_label"])
    summary["nowcast_candidate_type"] = str(nowcast_anchor["candidate_type"])
    summary["nowcast_selection_metric"] = str(nowcast_anchor["control_selection_metric"])
    summary["nowcast_selection_metric_value"] = float(nowcast_anchor["control_selection_metric_value"])
    summary["nowcast_selection_metric_pct"] = float(nowcast_anchor["control_selection_metric_pct"])
    summary["nowcast_anchor_candidate_label"] = str(nowcast_anchor["candidate_label"])
    summary["nowcast_anchor_candidate_type"] = str(nowcast_anchor["candidate_type"])
    summary["nowcast_anchor_mae"] = float(nowcast_anchor["minute_path_mae"])
    summary["nowcast_anchor_mae_pct"] = float(nowcast_anchor["minute_path_mae_pct"])
    return summary


def _annotate_cycle_scope(
    *,
    by_cycle: pd.DataFrame,
    origin_catalog: pd.DataFrame,
    scope_name: str,
) -> pd.DataFrame:
    """Attach split metadata to one per-cycle control frame."""
    if by_cycle.empty:
        return by_cycle.copy()
    if origin_catalog.empty:
        return by_cycle.assign(scope=str(scope_name))
    lookup = origin_catalog.loc[:, ["origin_timestamp", "split_name"]].copy()
    lookup["cycle_origin_timestamp"] = lookup["origin_timestamp"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    annotated = by_cycle.merge(
        lookup.loc[:, ["cycle_origin_timestamp", "split_name"]],
        on="cycle_origin_timestamp",
        how="left",
    )
    annotated["scope"] = str(scope_name)
    return annotated


def _apply_selected_nowcast_to_timeline(
    *,
    minute_timeline: pd.DataFrame,
    prediction_series: pd.Series,
) -> pd.DataFrame:
    """Overlay the selected minute policy onto a control minute timeline."""
    if minute_timeline.empty:
        return minute_timeline.copy()
    updated = minute_timeline.copy()
    updated["nowcast_pred"] = _apply_nowcast_updates(
        pd.Series(
            updated["phase_pred"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(updated["timestamp"], errors="raise")),
            dtype=float,
        ),
        prediction_series,
    ).to_numpy(dtype=float)
    return updated


def _placeholder_nowcast_anchor() -> dict[str, Any]:
    """Return a minimal placeholder anchor for pre-nowcast stack evaluation."""
    return {
        "candidate_label": "",
        "candidate_type": "",
        "control_selection_metric": "",
        "control_selection_metric_value": float("nan"),
        "control_selection_metric_pct": float("nan"),
        "minute_path_mae": float("nan"),
        "minute_path_mae_pct": float("nan"),
    }


def _empty_phase_stack_rolling_support_guard(
    *,
    required: bool,
    reason: str,
) -> dict[str, Any]:
    """Return a no-op rolling support guard that leaves the exact phase decision untouched."""
    recommended_policy = "hourly_passthrough" if required else "phase_candidate"
    return {
        "enabled": False,
        "required": bool(required),
        "decision_scope": str(MULTIRES_FORECAST_CONTROL["phase_stack_guard_rolling_scope"]),
        "recommended_policy": recommended_policy,
        "applied_candidate_label": "",
        "lock_gain_vs_hourly": float("nan"),
        "lock_gain_pct_vs_hourly": float("nan"),
        "next_lock_regress_vs_hourly": float("nan"),
        "next_lock_regress_pct_vs_hourly": float("nan"),
        "profile_degrade_vs_hourly": float("nan"),
        "profile_degrade_pct_vs_hourly": float("nan"),
        "peak_value_regress_vs_hourly": float("nan"),
        "peak_value_regress_pct_vs_hourly": float("nan"),
        "peak_hit_gain_vs_hourly": float("nan"),
        "optimizer_regress_vs_hourly": float("nan"),
        "optimizer_regress_pct_vs_hourly": float("nan"),
        "meets_lock_gain_rule": True,
        "meets_next_lock_rule": True,
        "meets_profile_rule": True,
        "meets_peak_value_rule": True,
        "meets_peak_hit_rule": True,
        "meets_optimizer_rule": True,
        "reason": str(reason),
    }


def _phase_stack_guard_decision(
    *,
    calibration_summary: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    hourly_candidate_label: str,
    phase_candidate_label: str,
) -> dict[str, Any]:
    """Decide whether the phase layer should stay active in the stacked control path.

    Stage-10 phase candidate selection happens on isolated phase-layer replay
    metrics. This guard adds one more question: after the hourly layer is
    already applied, does the phase layer still add enough stack-level value to
    justify itself on the chosen control scope?
    """
    if not bool(MULTIRES_FORECAST_CONTROL["phase_stack_guard_enabled"]):
        return {
            "enabled": False,
            "decision_scope": "disabled",
            "recommended_policy": "phase_candidate",
            "applied_candidate_label": str(phase_candidate_label),
            "lock_gain_vs_hourly": float("nan"),
            "lock_gain_pct_vs_hourly": float("nan"),
            "next_lock_regress_vs_hourly": float("nan"),
            "next_lock_regress_pct_vs_hourly": float("nan"),
            "profile_degrade_vs_hourly": float("nan"),
            "profile_degrade_pct_vs_hourly": float("nan"),
            "peak_value_regress_vs_hourly": float("nan"),
            "peak_value_regress_pct_vs_hourly": float("nan"),
            "peak_hit_gain_vs_hourly": float("nan"),
            "optimizer_regress_vs_hourly": float("nan"),
            "optimizer_regress_pct_vs_hourly": float("nan"),
            "meets_lock_gain_rule": True,
            "meets_next_lock_rule": True,
            "meets_profile_rule": True,
            "meets_peak_value_rule": True,
            "meets_peak_hit_rule": True,
            "meets_optimizer_rule": True,
            "reason": "Phase stack guard disabled; keeping the replayed phase candidate.",
        }

    decision_scope = str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"])
    scope_summary = evaluation_summary if decision_scope == "held_out_evaluation" else calibration_summary
    hourly_row = scope_summary.loc[scope_summary["role"].eq("hourly")].iloc[0]
    phase_row = scope_summary.loc[scope_summary["role"].eq("phase")].iloc[0]
    hourly_lock = float(hourly_row["lock_mae"])
    phase_lock = float(phase_row["lock_mae"])
    hourly_profile = float(hourly_row["profile_shape_mae"])
    phase_profile = float(phase_row["profile_shape_mae"])
    hourly_next_lock = float(hourly_row.get("next_lock_mae", float("nan")))
    phase_next_lock = float(phase_row.get("next_lock_mae", float("nan")))
    hourly_peak_value = float(hourly_row.get("peak_value_mae", float("nan")))
    phase_peak_value = float(phase_row.get("peak_value_mae", float("nan")))
    hourly_peak_hit = float(hourly_row.get("peak_interval_hit_rate", float("nan")))
    phase_peak_hit = float(phase_row.get("peak_interval_hit_rate", float("nan")))
    hourly_optimizer_score = _optimizer_score_from_row(hourly_row)
    phase_optimizer_score = _optimizer_score_from_row(phase_row)
    lock_gain = hourly_lock - phase_lock
    lock_gain_pct = float(lock_gain / hourly_lock) if hourly_lock > 0.0 else float("nan")
    next_lock_regress = phase_next_lock - hourly_next_lock
    next_lock_regress_pct = (
        float(max(next_lock_regress, 0.0) / hourly_next_lock)
        if np.isfinite(hourly_next_lock) and hourly_next_lock > 0.0
        else float("nan")
    )
    profile_degrade = phase_profile - hourly_profile
    profile_degrade_pct = (
        float(max(profile_degrade, 0.0) / hourly_profile) if hourly_profile > 0.0 else float("nan")
    )
    peak_value_regress = phase_peak_value - hourly_peak_value
    peak_value_regress_pct = (
        float(max(peak_value_regress, 0.0) / hourly_peak_value)
        if np.isfinite(hourly_peak_value) and hourly_peak_value > 0.0
        else float("nan")
    )
    peak_hit_gain = (
        float(phase_peak_hit - hourly_peak_hit)
        if np.isfinite(phase_peak_hit) and np.isfinite(hourly_peak_hit)
        else float("nan")
    )
    min_lock_gain_pct = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_lock_gain_pct"])
    max_next_lock_regress_pct = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_next_lock_regress_pct"])
    max_profile_degrade_pct = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_profile_degrade_pct"])
    max_peak_value_regress_pct = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_peak_value_regress_pct"])
    min_peak_hit_gain = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_peak_hit_gain"])
    max_optimizer_regress_pct = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_optimizer_regress_pct"])
    optimizer_regress = phase_optimizer_score - hourly_optimizer_score
    optimizer_regress_pct = (
        float(max(optimizer_regress, 0.0) / hourly_optimizer_score)
        if np.isfinite(hourly_optimizer_score) and hourly_optimizer_score > 0.0
        else float("nan")
    )
    meets_lock_gain_rule = bool(np.isfinite(lock_gain_pct)) and lock_gain_pct >= min_lock_gain_pct
    meets_next_lock_rule = (not np.isfinite(next_lock_regress_pct)) or next_lock_regress_pct <= max_next_lock_regress_pct
    meets_profile_rule = bool(np.isfinite(profile_degrade_pct)) and profile_degrade_pct <= max_profile_degrade_pct
    meets_peak_value_rule = (not np.isfinite(peak_value_regress_pct)) or peak_value_regress_pct <= max_peak_value_regress_pct
    meets_peak_hit_rule = (not np.isfinite(peak_hit_gain)) or peak_hit_gain >= min_peak_hit_gain
    meets_optimizer_rule = (not np.isfinite(optimizer_regress_pct)) or optimizer_regress_pct <= max_optimizer_regress_pct
    recommended_policy = "phase_candidate"
    applied_candidate_label = str(phase_candidate_label)
    if not (
        meets_lock_gain_rule
        and meets_next_lock_rule
        and meets_profile_rule
        and meets_peak_value_rule
        and meets_peak_hit_rule
        and meets_optimizer_rule
    ):
        recommended_policy = "hourly_passthrough"
        applied_candidate_label = str(hourly_candidate_label)
    if recommended_policy == "phase_candidate":
        reason = (
            "Keeping the phase layer because it still adds enough lock-MAE gain over the hourly "
            "stack, stays within the allowed next-lock, peak, and profile regressions, and does not "
            "regress the optimizer-weighted score beyond the configured tolerance."
        )
    else:
        reason = (
            "Falling back to the hourly path at the phase layer because the isolated phase winner "
            "did not add enough stack-level lock-MAE gain, regressed next-lock or peak behavior "
            "beyond the configured tolerance, degraded profile shape too far, or regressed the "
            "optimizer-weighted score too far."
        )
    return {
        "enabled": True,
        "decision_scope": decision_scope,
        "recommended_policy": recommended_policy,
        "applied_candidate_label": applied_candidate_label,
        "lock_gain_vs_hourly": float(lock_gain),
        "lock_gain_pct_vs_hourly": float(lock_gain_pct),
        "next_lock_regress_vs_hourly": float(next_lock_regress),
        "next_lock_regress_pct_vs_hourly": float(next_lock_regress_pct),
        "profile_degrade_vs_hourly": float(profile_degrade),
        "profile_degrade_pct_vs_hourly": float(profile_degrade_pct),
        "peak_value_regress_vs_hourly": float(peak_value_regress),
        "peak_value_regress_pct_vs_hourly": float(peak_value_regress_pct),
        "peak_hit_gain_vs_hourly": float(peak_hit_gain),
        "optimizer_regress_vs_hourly": float(optimizer_regress),
        "optimizer_regress_pct_vs_hourly": float(optimizer_regress_pct),
        "meets_lock_gain_rule": bool(meets_lock_gain_rule),
        "meets_next_lock_rule": bool(meets_next_lock_rule),
        "meets_profile_rule": bool(meets_profile_rule),
        "meets_peak_value_rule": bool(meets_peak_value_rule),
        "meets_peak_hit_rule": bool(meets_peak_hit_rule),
        "meets_optimizer_rule": bool(meets_optimizer_rule),
        "reason": reason,
    }


def _rolling_phase_stack_guard_decision(
    *,
    calibration_summary: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    combined_summary: pd.DataFrame,
    hourly_candidate_label: str,
    phase_candidate_label: str,
) -> dict[str, Any]:
    """Decide whether the exact phase winner also has enough broader rolling support."""
    if not bool(MULTIRES_FORECAST_CONTROL["phase_stack_guard_require_rolling_support"]):
        return _empty_phase_stack_rolling_support_guard(
            required=False,
            reason="Rolling support is optional in config; keeping the exact phase decision.",
        )

    decision_scope = str(MULTIRES_FORECAST_CONTROL["phase_stack_guard_rolling_scope"])
    if decision_scope == "rolling_calibration":
        scope_summary = calibration_summary
    elif decision_scope == "rolling_combined":
        scope_summary = combined_summary
    else:
        scope_summary = evaluation_summary
    if scope_summary.empty:
        return _empty_phase_stack_rolling_support_guard(
            required=True,
            reason="Rolling support was required, but the broader rolling benchmark did not produce a comparable phase scope.",
        )
    hourly_row = scope_summary.loc[scope_summary["role"].eq("hourly")]
    phase_row = scope_summary.loc[scope_summary["role"].eq("phase")]
    if hourly_row.empty or phase_row.empty:
        return _empty_phase_stack_rolling_support_guard(
            required=True,
            reason="Rolling support was required, but the rolling summary was missing the hourly or phase row.",
        )
    hourly_row = hourly_row.iloc[0]
    phase_row = phase_row.iloc[0]
    hourly_lock = float(hourly_row["lock_mae"])
    phase_lock = float(phase_row["lock_mae"])
    hourly_next_lock = float(hourly_row.get("next_lock_mae", float("nan")))
    phase_next_lock = float(phase_row.get("next_lock_mae", float("nan")))
    hourly_profile = float(hourly_row.get("profile_shape_mae", float("nan")))
    phase_profile = float(phase_row.get("profile_shape_mae", float("nan")))
    hourly_peak_value = float(hourly_row.get("peak_value_mae", float("nan")))
    phase_peak_value = float(phase_row.get("peak_value_mae", float("nan")))
    hourly_peak_hit = float(hourly_row.get("peak_interval_hit_rate", float("nan")))
    phase_peak_hit = float(phase_row.get("peak_interval_hit_rate", float("nan")))
    hourly_optimizer_score = _optimizer_score_from_row(hourly_row)
    phase_optimizer_score = _optimizer_score_from_row(phase_row)
    lock_gain = hourly_lock - phase_lock
    lock_gain_pct = float(lock_gain / hourly_lock) if hourly_lock > 0.0 else float("nan")
    next_lock_regress = phase_next_lock - hourly_next_lock
    next_lock_regress_pct = (
        float(max(next_lock_regress, 0.0) / hourly_next_lock)
        if np.isfinite(hourly_next_lock) and hourly_next_lock > 0.0
        else float("nan")
    )
    profile_degrade = phase_profile - hourly_profile
    profile_degrade_pct = (
        float(max(profile_degrade, 0.0) / hourly_profile)
        if np.isfinite(hourly_profile) and hourly_profile > 0.0
        else float("nan")
    )
    peak_value_regress = phase_peak_value - hourly_peak_value
    peak_value_regress_pct = (
        float(max(peak_value_regress, 0.0) / hourly_peak_value)
        if np.isfinite(hourly_peak_value) and hourly_peak_value > 0.0
        else float("nan")
    )
    peak_hit_gain = (
        float(phase_peak_hit - hourly_peak_hit)
        if np.isfinite(phase_peak_hit) and np.isfinite(hourly_peak_hit)
        else float("nan")
    )
    optimizer_regress = phase_optimizer_score - hourly_optimizer_score
    optimizer_regress_pct = (
        float(max(optimizer_regress, 0.0) / hourly_optimizer_score)
        if np.isfinite(hourly_optimizer_score) and hourly_optimizer_score > 0.0
        else float("nan")
    )
    min_lock_gain_pct = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_rolling_lock_gain_pct"])
    max_next_lock_regress_pct = float(
        MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_next_lock_regress_pct"]
    )
    max_profile_degrade_pct = float(
        MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_profile_degrade_pct"]
    )
    max_peak_value_regress_pct = float(
        MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_peak_value_regress_pct"]
    )
    min_peak_hit_gain = float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_rolling_peak_hit_gain"])
    max_optimizer_regress_pct = float(
        MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_optimizer_regress_pct"]
    )
    meets_lock_gain_rule = bool(np.isfinite(lock_gain_pct)) and lock_gain_pct >= min_lock_gain_pct
    meets_next_lock_rule = (not np.isfinite(next_lock_regress_pct)) or next_lock_regress_pct <= max_next_lock_regress_pct
    meets_profile_rule = (not np.isfinite(profile_degrade_pct)) or profile_degrade_pct <= max_profile_degrade_pct
    meets_peak_value_rule = (not np.isfinite(peak_value_regress_pct)) or peak_value_regress_pct <= max_peak_value_regress_pct
    meets_peak_hit_rule = (not np.isfinite(peak_hit_gain)) or peak_hit_gain >= min_peak_hit_gain
    meets_optimizer_rule = (not np.isfinite(optimizer_regress_pct)) or optimizer_regress_pct <= max_optimizer_regress_pct
    recommended_policy = "phase_candidate"
    applied_candidate_label = str(phase_candidate_label)
    if not (
        meets_lock_gain_rule
        and meets_next_lock_rule
        and meets_profile_rule
        and meets_peak_value_rule
        and meets_peak_hit_rule
        and meets_optimizer_rule
    ):
        recommended_policy = "hourly_passthrough"
        applied_candidate_label = str(hourly_candidate_label)
    if recommended_policy == "phase_candidate":
        reason = (
            "Keeping the phase layer because it still shows broader rolling support on the configured "
            "scope without regressing next-lock, peak behavior, profile shape, or optimizer score "
            "beyond tolerance."
        )
    else:
        reason = (
            "Falling back to the hourly path because the exact phase winner did not show enough broader "
            "rolling lock gain, or it regressed rolling next-lock, peak behavior, profile shape, or "
            "optimizer score beyond tolerance."
        )
    return {
        "enabled": True,
        "required": True,
        "decision_scope": decision_scope,
        "recommended_policy": recommended_policy,
        "applied_candidate_label": applied_candidate_label,
        "lock_gain_vs_hourly": float(lock_gain),
        "lock_gain_pct_vs_hourly": float(lock_gain_pct),
        "next_lock_regress_vs_hourly": float(next_lock_regress),
        "next_lock_regress_pct_vs_hourly": float(next_lock_regress_pct),
        "profile_degrade_vs_hourly": float(profile_degrade),
        "profile_degrade_pct_vs_hourly": float(profile_degrade_pct),
        "peak_value_regress_vs_hourly": float(peak_value_regress),
        "peak_value_regress_pct_vs_hourly": float(peak_value_regress_pct),
        "peak_hit_gain_vs_hourly": float(peak_hit_gain),
        "optimizer_regress_vs_hourly": float(optimizer_regress),
        "optimizer_regress_pct_vs_hourly": float(optimizer_regress_pct),
        "meets_lock_gain_rule": bool(meets_lock_gain_rule),
        "meets_next_lock_rule": bool(meets_next_lock_rule),
        "meets_profile_rule": bool(meets_profile_rule),
        "meets_peak_value_rule": bool(meets_peak_value_rule),
        "meets_peak_hit_rule": bool(meets_peak_hit_rule),
        "meets_optimizer_rule": bool(meets_optimizer_rule),
        "reason": reason,
    }


def _combine_phase_stack_guard_with_rolling_support(
    *,
    phase_stack_guard: dict[str, Any],
    rolling_support_guard: dict[str, Any],
    hourly_candidate_label: str,
) -> dict[str, Any]:
    """Apply the rolling support verdict to the exact phase decision surface."""
    combined = dict(phase_stack_guard)
    combined["rolling_support_enabled"] = bool(rolling_support_guard["enabled"])
    combined["rolling_support_required"] = bool(rolling_support_guard["required"])
    combined["rolling_support_scope"] = str(rolling_support_guard["decision_scope"])
    combined["rolling_support_recommended_policy"] = str(rolling_support_guard["recommended_policy"])
    combined["rolling_support_applied_candidate_label"] = str(rolling_support_guard["applied_candidate_label"])
    combined["rolling_support_lock_gain_vs_hourly"] = float(rolling_support_guard["lock_gain_vs_hourly"])
    combined["rolling_support_lock_gain_pct_vs_hourly"] = float(rolling_support_guard["lock_gain_pct_vs_hourly"])
    combined["rolling_support_next_lock_regress_vs_hourly"] = float(
        rolling_support_guard["next_lock_regress_vs_hourly"]
    )
    combined["rolling_support_next_lock_regress_pct_vs_hourly"] = float(
        rolling_support_guard["next_lock_regress_pct_vs_hourly"]
    )
    combined["rolling_support_profile_degrade_vs_hourly"] = float(
        rolling_support_guard["profile_degrade_vs_hourly"]
    )
    combined["rolling_support_profile_degrade_pct_vs_hourly"] = float(
        rolling_support_guard["profile_degrade_pct_vs_hourly"]
    )
    combined["rolling_support_meets_lock_gain_rule"] = bool(rolling_support_guard["meets_lock_gain_rule"])
    combined["rolling_support_meets_next_lock_rule"] = bool(rolling_support_guard["meets_next_lock_rule"])
    combined["rolling_support_meets_profile_rule"] = bool(rolling_support_guard["meets_profile_rule"])
    combined["rolling_support_reason"] = str(rolling_support_guard["reason"])
    combined["rolling_support_applied_veto"] = False
    if (
        str(phase_stack_guard["recommended_policy"]) == "phase_candidate"
        and bool(rolling_support_guard["required"])
        and str(rolling_support_guard["recommended_policy"]) != "phase_candidate"
    ):
        combined["recommended_policy"] = "hourly_passthrough"
        combined["applied_candidate_label"] = str(hourly_candidate_label)
        combined["rolling_support_applied_veto"] = True
        combined["reason"] = (
            "The exact phase stack benchmark selected a phase candidate, but the broader rolling support "
            "guard vetoed it, so the final policy falls back to the hourly path."
        )
    return combined


def _phase_stack_guard_summary_frame(
    *,
    calibration_summary: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    decision: dict[str, Any],
    guard_name: str = "exact_stack_guard",
    calibration_scope_name: str = "calibration",
    evaluation_scope_name: str = "evaluation",
    combined_summary: pd.DataFrame | None = None,
    combined_scope_name: str | None = None,
) -> pd.DataFrame:
    """Serialize the stack-level hourly-versus-phase comparison used by the guard."""
    rows: list[dict[str, Any]] = []
    scope_specs: list[tuple[str, pd.DataFrame]] = [
        (calibration_scope_name, calibration_summary),
        (evaluation_scope_name, evaluation_summary),
    ]
    if combined_summary is not None and combined_scope_name is not None and not combined_summary.empty:
        scope_specs.append((combined_scope_name, combined_summary))
    for scope_name, scope_summary in scope_specs:
        hourly_row = scope_summary.loc[scope_summary["role"].eq("hourly")].iloc[0]
        phase_row = scope_summary.loc[scope_summary["role"].eq("phase")].iloc[0]
        hourly_lock = float(hourly_row["lock_mae"])
        phase_lock = float(phase_row["lock_mae"])
        hourly_profile = float(hourly_row["profile_shape_mae"])
        phase_profile = float(phase_row["profile_shape_mae"])
        hourly_next_lock = float(hourly_row.get("next_lock_mae", float("nan")))
        phase_next_lock = float(phase_row.get("next_lock_mae", float("nan")))
        hourly_peak_value = float(hourly_row.get("peak_value_mae", float("nan")))
        phase_peak_value = float(phase_row.get("peak_value_mae", float("nan")))
        hourly_peak_hit = float(hourly_row.get("peak_interval_hit_rate", float("nan")))
        phase_peak_hit = float(phase_row.get("peak_interval_hit_rate", float("nan")))
        hourly_optimizer_score = _optimizer_score_from_row(hourly_row)
        phase_optimizer_score = _optimizer_score_from_row(phase_row)
        rows.append(
            {
                "scope": scope_name,
                "hourly_lock_mae": hourly_lock,
                "phase_lock_mae": phase_lock,
                "phase_lock_gain_vs_hourly": hourly_lock - phase_lock,
                "phase_lock_gain_pct_vs_hourly": (
                    float((hourly_lock - phase_lock) / hourly_lock) if hourly_lock > 0.0 else float("nan")
                ),
                "hourly_profile_shape_mae": hourly_profile,
                "phase_profile_shape_mae": phase_profile,
                "phase_profile_degrade_vs_hourly": phase_profile - hourly_profile,
                "phase_profile_degrade_pct_vs_hourly": (
                    float(max(phase_profile - hourly_profile, 0.0) / hourly_profile)
                    if hourly_profile > 0.0
                    else float("nan")
                ),
                "hourly_next_lock_mae": hourly_next_lock,
                "phase_next_lock_mae": phase_next_lock,
                "phase_next_lock_regress_vs_hourly": phase_next_lock - hourly_next_lock,
                "phase_next_lock_regress_pct_vs_hourly": (
                    float(max(phase_next_lock - hourly_next_lock, 0.0) / hourly_next_lock)
                    if np.isfinite(hourly_next_lock) and hourly_next_lock > 0.0
                    else float("nan")
                ),
                "hourly_peak_value_mae": hourly_peak_value,
                "phase_peak_value_mae": phase_peak_value,
                "phase_peak_value_regress_vs_hourly": phase_peak_value - hourly_peak_value,
                "phase_peak_value_regress_pct_vs_hourly": (
                    float(max(phase_peak_value - hourly_peak_value, 0.0) / hourly_peak_value)
                    if np.isfinite(hourly_peak_value) and hourly_peak_value > 0.0
                    else float("nan")
                ),
                "hourly_peak_interval_hit_rate": hourly_peak_hit,
                "phase_peak_interval_hit_rate": phase_peak_hit,
                "phase_peak_hit_gain_vs_hourly": (
                    float(phase_peak_hit - hourly_peak_hit)
                    if np.isfinite(phase_peak_hit) and np.isfinite(hourly_peak_hit)
                    else float("nan")
                ),
                "hourly_optimizer_score": hourly_optimizer_score,
                "phase_optimizer_score": phase_optimizer_score,
                "phase_optimizer_regress_vs_hourly": phase_optimizer_score - hourly_optimizer_score,
                "phase_optimizer_regress_pct_vs_hourly": (
                    float(max(phase_optimizer_score - hourly_optimizer_score, 0.0) / hourly_optimizer_score)
                    if np.isfinite(hourly_optimizer_score) and hourly_optimizer_score > 0.0
                    else float("nan")
                ),
                "recommended_policy": str(decision["recommended_policy"]),
                "decision_scope": str(decision["decision_scope"]),
                "guard_name": str(guard_name),
                "meets_lock_gain_rule": bool(decision["meets_lock_gain_rule"]),
                "meets_next_lock_rule": bool(decision["meets_next_lock_rule"]),
                "meets_profile_rule": bool(decision["meets_profile_rule"]),
                "meets_peak_value_rule": bool(decision["meets_peak_value_rule"]),
                "meets_peak_hit_rule": bool(decision["meets_peak_hit_rule"]),
                "meets_optimizer_rule": bool(decision["meets_optimizer_rule"]),
                "reason": str(decision["reason"]),
            }
        )
    return pd.DataFrame(rows)


def _phase_stack_candidate_summary_for_series(
    *,
    minute_timeline: pd.DataFrame,
    candidate_series: pd.Series,
    lock_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one phase-layer prediction series on a control minute timeline."""
    timeline_index = pd.DatetimeIndex(pd.to_datetime(minute_timeline["timestamp"], errors="raise"))
    candidate_timeline = minute_timeline.copy()
    candidate_timeline["phase_pred"] = candidate_series.reindex(timeline_index).to_numpy(dtype=float)
    _, candidate_by_cycle, candidate_summary = _evaluate_control_scope(
        minute_timeline=candidate_timeline,
        nowcast_anchor=_placeholder_nowcast_anchor(),
        lock_interval=lock_interval,
    )
    return candidate_timeline, candidate_by_cycle, candidate_summary


def _phase_stack_row_from_summary(
    *,
    hourly_row: pd.Series,
    candidate_label: str,
    candidate_type: str,
    source_model_label: str,
    target_mode: str,
    replay_pool_rank: float,
    replay_pool_source_type: str,
    replay_pool_source_run_id: str,
    replay_resolution: str,
    replay_feature_set: str,
    replay_model_label: str,
    replay_run_dir: str,
    candidate_by_cycle: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    stack_candidate_policy: str,
    stack_blend_weight: float,
    stack_blend_parent_candidate_label: str,
    stack_candidate_family: str,
    stack_reference_candidate_label: str = "",
    stack_bucket_policy_json: str = "",
    stack_bucket_weight_json: str = "",
    stack_bucket_granularity_minutes: float = float("nan"),
) -> dict[str, Any]:
    """Serialize one stack-aware phase candidate into the benchmark table format."""
    phase_row = candidate_summary.loc[candidate_summary["role"].eq("phase")].iloc[0]
    selection_metric_name = _phase_stack_selection_metric()
    lock_mae = float(phase_row["lock_mae"])
    profile_shape_mae = float(phase_row["profile_shape_mae"])
    next_lock_mae = float(phase_row.get("next_lock_mae", float("nan")))
    peak_value_mae = float(phase_row.get("peak_value_mae", float("nan")))
    peak_interval_hit_rate = float(phase_row.get("peak_interval_hit_rate", float("nan")))
    peak_interval_miss_rate = float(phase_row.get("peak_interval_miss_rate", float("nan")))
    peak_interval_offset_minutes = float(phase_row.get("peak_interval_offset_minutes", float("nan")))
    optimizer_score = _optimizer_score_from_row(phase_row)
    lock_gain_vs_hourly = float(hourly_row["lock_mae"] - lock_mae)
    lock_gain_pct_vs_hourly = (
        float(lock_gain_vs_hourly / hourly_row["lock_mae"]) if float(hourly_row["lock_mae"]) > 0.0 else float("nan")
    )
    profile_degrade_vs_hourly = float(profile_shape_mae - float(hourly_row["profile_shape_mae"]))
    profile_degrade_pct_vs_hourly = (
        float(max(profile_degrade_vs_hourly, 0.0) / float(hourly_row["profile_shape_mae"]))
        if float(hourly_row["profile_shape_mae"]) > 0.0
        else float("nan")
    )
    hourly_next_lock_mae = float(hourly_row.get("next_lock_mae", float("nan")))
    next_lock_regress_vs_hourly = next_lock_mae - hourly_next_lock_mae
    next_lock_regress_pct_vs_hourly = (
        float(max(next_lock_regress_vs_hourly, 0.0) / hourly_next_lock_mae)
        if np.isfinite(hourly_next_lock_mae) and hourly_next_lock_mae > 0.0
        else float("nan")
    )
    hourly_peak_value_mae = float(hourly_row.get("peak_value_mae", float("nan")))
    peak_value_regress_vs_hourly = peak_value_mae - hourly_peak_value_mae
    peak_value_regress_pct_vs_hourly = (
        float(max(peak_value_regress_vs_hourly, 0.0) / hourly_peak_value_mae)
        if np.isfinite(hourly_peak_value_mae) and hourly_peak_value_mae > 0.0
        else float("nan")
    )
    hourly_peak_interval_hit_rate = float(hourly_row.get("peak_interval_hit_rate", float("nan")))
    peak_hit_gain_vs_hourly = (
        float(peak_interval_hit_rate - hourly_peak_interval_hit_rate)
        if np.isfinite(peak_interval_hit_rate) and np.isfinite(hourly_peak_interval_hit_rate)
        else float("nan")
    )
    hourly_optimizer_score = _optimizer_score_from_row(hourly_row)
    optimizer_regress_vs_hourly = float(optimizer_score - hourly_optimizer_score)
    optimizer_regress_pct_vs_hourly = (
        float(max(optimizer_regress_vs_hourly, 0.0) / hourly_optimizer_score)
        if np.isfinite(hourly_optimizer_score) and hourly_optimizer_score > 0.0
        else float("nan")
    )
    meets_lock_gain_rule = (
        stack_candidate_policy == "phase_candidate"
        and np.isfinite(lock_gain_pct_vs_hourly)
        and lock_gain_pct_vs_hourly >= float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_lock_gain_pct"])
    )
    meets_next_lock_rule = (
        stack_candidate_policy == "phase_candidate"
        and (
            not np.isfinite(next_lock_regress_pct_vs_hourly)
            or next_lock_regress_pct_vs_hourly
            <= float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_next_lock_regress_pct"])
        )
    )
    meets_profile_rule = (
        stack_candidate_policy == "phase_candidate"
        and np.isfinite(profile_degrade_pct_vs_hourly)
        and profile_degrade_pct_vs_hourly
        <= float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_profile_degrade_pct"])
    )
    meets_peak_value_rule = (
        stack_candidate_policy == "phase_candidate"
        and (
            not np.isfinite(peak_value_regress_pct_vs_hourly)
            or peak_value_regress_pct_vs_hourly
            <= float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_peak_value_regress_pct"])
        )
    )
    meets_peak_hit_rule = (
        stack_candidate_policy == "phase_candidate"
        and (
            not np.isfinite(peak_hit_gain_vs_hourly)
            or peak_hit_gain_vs_hourly >= float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_peak_hit_gain"])
        )
    )
    meets_optimizer_rule = (
        stack_candidate_policy == "phase_candidate"
        and (
            not np.isfinite(optimizer_regress_pct_vs_hourly)
            or optimizer_regress_pct_vs_hourly
            <= float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_optimizer_regress_pct"])
        )
    )
    selection_metric_value = (
        optimizer_score
        if selection_metric_name == "optimizer_score"
        else float(phase_row.get(selection_metric_name, float("nan")))
    )
    if selection_metric_name == "optimizer_score":
        selection_metric_pct = float(optimizer_score)
    else:
        selection_metric_pct = float(
            phase_row.get(
                f"{selection_metric_name}_pct",
                phase_row.get(selection_metric_name, float("nan")),
            )
        )

    def _cycle_quantile(column: str, quantile: float) -> float:
        if column not in candidate_by_cycle.columns:
            return float("nan")
        values = pd.to_numeric(candidate_by_cycle[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.dropna()
        if values.empty:
            return float("nan")
        return float(values.quantile(quantile))

    return {
        "candidate_label": str(candidate_label),
        "candidate_type": str(candidate_type),
        "source_model_label": str(source_model_label),
        "target_mode": str(target_mode),
        "minute_path_mae": float(phase_row["minute_path_mae"]),
        "minute_path_mae_pct": float(phase_row["minute_path_mae_pct"]),
        "lock_mae": lock_mae,
        "lock_mae_pct": float(phase_row["lock_mae_pct"]),
        "next_lock_mae": next_lock_mae,
        "next_lock_mae_pct": float(phase_row.get("next_lock_mae_pct", float("nan"))),
        "profile_shape_mae": profile_shape_mae,
        "profile_shape_mae_pct": float(phase_row["profile_shape_mae_pct"]),
        "energy_mae": float(phase_row["energy_mae"]),
        "energy_mae_pct": float(phase_row["energy_mae_pct"]),
        "peak_value_mae": peak_value_mae,
        "peak_value_mae_pct": float(phase_row.get("peak_value_mae_pct", float("nan"))),
        "peak_interval_hit_rate": peak_interval_hit_rate,
        "peak_interval_miss_rate": peak_interval_miss_rate,
        "peak_interval_offset_minutes": peak_interval_offset_minutes,
        "optimizer_score": optimizer_score,
        "cycle_n": int(phase_row["cycle_n"]),
        "lock_mae_p50": _cycle_quantile("phase_lock_mae", 0.5),
        "lock_mae_p90": _cycle_quantile("phase_lock_mae", 0.9),
        "next_lock_mae_p50": _cycle_quantile("phase_next_lock_mae", 0.5),
        "next_lock_mae_p90": _cycle_quantile("phase_next_lock_mae", 0.9),
        "profile_shape_mae_p50": _cycle_quantile("phase_profile_shape_mae", 0.5),
        "profile_shape_mae_p90": _cycle_quantile("phase_profile_shape_mae", 0.9),
        "minute_path_mae_p50": _cycle_quantile("phase_minute_path_mae", 0.5),
        "minute_path_mae_p90": _cycle_quantile("phase_minute_path_mae", 0.9),
        "peak_value_mae_p50": _cycle_quantile("phase_peak_value_mae", 0.5),
        "peak_value_mae_p90": _cycle_quantile("phase_peak_value_mae", 0.9),
        "peak_interval_hit_rate_p50": _cycle_quantile("phase_peak_interval_hit_rate", 0.5),
        "peak_interval_hit_rate_p90": _cycle_quantile("phase_peak_interval_hit_rate", 0.9),
        "peak_interval_miss_rate_p50": _cycle_quantile("phase_peak_interval_miss_rate", 0.5),
        "peak_interval_miss_rate_p90": _cycle_quantile("phase_peak_interval_miss_rate", 0.9),
        "peak_interval_offset_minutes_p50": _cycle_quantile("phase_peak_interval_offset_minutes", 0.5),
        "peak_interval_offset_minutes_p90": _cycle_quantile("phase_peak_interval_offset_minutes", 0.9),
        "optimizer_score_p50": _cycle_quantile("phase_optimizer_score", 0.5),
        "optimizer_score_p90": _cycle_quantile("phase_optimizer_score", 0.9),
        "lock_gain_vs_hourly": lock_gain_vs_hourly,
        "lock_gain_pct_vs_hourly": float(lock_gain_pct_vs_hourly),
        "next_lock_regress_vs_hourly": float(next_lock_regress_vs_hourly),
        "next_lock_regress_pct_vs_hourly": float(next_lock_regress_pct_vs_hourly),
        "profile_degrade_vs_hourly": profile_degrade_vs_hourly,
        "profile_degrade_pct_vs_hourly": float(profile_degrade_pct_vs_hourly),
        "peak_value_regress_vs_hourly": float(peak_value_regress_vs_hourly),
        "peak_value_regress_pct_vs_hourly": float(peak_value_regress_pct_vs_hourly),
        "peak_hit_gain_vs_hourly": float(peak_hit_gain_vs_hourly),
        "optimizer_regress_vs_hourly": optimizer_regress_vs_hourly,
        "optimizer_regress_pct_vs_hourly": float(optimizer_regress_pct_vs_hourly),
        "meets_lock_gain_rule": bool(meets_lock_gain_rule),
        "meets_next_lock_rule": bool(meets_next_lock_rule),
        "meets_profile_rule": bool(meets_profile_rule),
        "meets_peak_value_rule": bool(meets_peak_value_rule),
        "meets_peak_hit_rule": bool(meets_peak_hit_rule),
        "meets_optimizer_rule": bool(meets_optimizer_rule),
        "meets_stack_guard": bool(
            meets_lock_gain_rule
            and meets_next_lock_rule
            and meets_profile_rule
            and meets_peak_value_rule
            and meets_peak_hit_rule
            and meets_optimizer_rule
        ),
        "stack_candidate_policy": str(stack_candidate_policy),
        "selection_metric_name": selection_metric_name,
        "selection_metric_value": float(selection_metric_value),
        "selection_metric_pct": float(selection_metric_pct),
        "replay_pool_rank": float(replay_pool_rank),
        "replay_pool_source_type": str(replay_pool_source_type),
        "replay_pool_source_run_id": str(replay_pool_source_run_id),
        "replay_resolution": str(replay_resolution),
        "replay_feature_set": str(replay_feature_set),
        "replay_model_label": str(replay_model_label),
        "replay_run_dir": str(replay_run_dir),
        "stack_blend_weight": float(stack_blend_weight),
        "stack_blend_parent_candidate_label": str(stack_blend_parent_candidate_label),
        "stack_reference_candidate_label": str(stack_reference_candidate_label),
        "stack_candidate_family": str(stack_candidate_family),
        "stack_bucket_policy_json": str(stack_bucket_policy_json),
        "stack_bucket_weight_json": str(stack_bucket_weight_json),
        "stack_bucket_granularity_minutes": float(stack_bucket_granularity_minutes),
    }


def _phase_stack_native_candidate_shortlist(candidate_meta: pd.DataFrame) -> pd.DataFrame:
    """Keep only the strongest native phase candidates per replay context before stack expansion."""
    if candidate_meta.empty:
        return candidate_meta.copy()

    learned_limit = int(MULTIRES_FORECAST_CONTROL["phase_stack_native_learned_top_candidates_per_pool"])
    baseline_limit = int(MULTIRES_FORECAST_CONTROL["phase_stack_native_baseline_top_candidates_per_pool"])
    if learned_limit <= 0 and baseline_limit <= 0:
        return candidate_meta.head(1).reset_index(drop=True)

    working = candidate_meta.copy().drop_duplicates(subset=["candidate_label"], keep="first").reset_index(drop=True)
    decision_scope = str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"])
    prefix = "evaluation_" if decision_scope == "held_out_evaluation" else ""

    def _metric_series(*column_names: str, default: float = float("inf")) -> pd.Series:
        for column in column_names:
            if column in working.columns:
                return pd.to_numeric(working[column], errors="coerce")
        return pd.Series(default, index=working.index, dtype="float64")

    working["_shortlist_selection_metric"] = _metric_series(
        f"{prefix}selection_metric_value",
        "selection_metric_value",
    )
    working["_shortlist_next_lock_mae"] = _metric_series(
        f"{prefix}next_lock_mae",
        "next_lock_mae",
    )
    working["_shortlist_profile_shape_mae"] = _metric_series(
        f"{prefix}profile_shape_mae",
        "profile_shape_mae",
    )
    working["_shortlist_lock_mae"] = _metric_series(
        f"{prefix}lock_mae",
        "lock_mae",
    )
    working["_shortlist_prior_phase_eval_metric"] = _metric_series(
        "replay_pool_prior_phase_eval_metric",
        default=float("inf"),
    )
    working["_shortlist_prior_phase_support_runs"] = pd.to_numeric(
        working.get("replay_pool_prior_phase_support_runs", pd.Series(index=working.index, dtype="float64")),
        errors="coerce",
    ).fillna(0.0)
    working["_shortlist_prior_phase_supported"] = (
        working["_shortlist_prior_phase_support_runs"].gt(0)
        & working["_shortlist_prior_phase_eval_metric"].replace([np.inf, -np.inf], np.nan).notna()
    )
    working = working.sort_values(
        [
            "_shortlist_prior_phase_supported",
            "_shortlist_prior_phase_support_runs",
            "_shortlist_prior_phase_eval_metric",
            "_shortlist_selection_metric",
            "_shortlist_next_lock_mae",
            "_shortlist_profile_shape_mae",
            "_shortlist_lock_mae",
            "candidate_label",
        ],
        ascending=[False, False, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    group_columns = [
        column
        for column in (
            "replay_pool_rank",
            "replay_pool_source_type",
            "replay_pool_source_run_id",
            "replay_resolution",
            "replay_feature_set",
        )
        if column in working.columns
    ]
    if group_columns:
        grouped_frames = [
            group.copy().reset_index(drop=True)
            for _, group in working.groupby(group_columns, dropna=False, sort=False)
        ]
    else:
        grouped_frames = [working]

    selected_chunks: list[pd.DataFrame] = []
    for group in grouped_frames:
        candidate_type = group["candidate_type"].astype("string")
        learned = group.loc[candidate_type.eq("learned")].head(learned_limit) if learned_limit > 0 else group.iloc[0:0]
        baseline = (
            group.loc[candidate_type.eq("baseline")].head(baseline_limit) if baseline_limit > 0 else group.iloc[0:0]
        )
        other = group.loc[~candidate_type.isin(["learned", "baseline"])].copy()
        chosen = pd.concat([baseline, learned, other], ignore_index=False)
        if chosen.empty:
            chosen = group.head(1)
        selected_chunks.append(chosen)

    shortlisted = pd.concat(selected_chunks, ignore_index=False)
    shortlisted = shortlisted.drop_duplicates(subset=["candidate_label"], keep="first").reset_index(drop=True)
    helper_columns = [column for column in shortlisted.columns if column.startswith("_shortlist_")]
    if helper_columns:
        shortlisted = shortlisted.drop(columns=helper_columns)
    return shortlisted


def _phase_stack_candidate_benchmark_scope(
    *,
    minute_timeline: pd.DataFrame,
    candidate_detail_by_origin: pd.DataFrame,
    candidate_metrics_by_origin: pd.DataFrame,
    phase_origins: list[pd.Timestamp],
    phase_horizon: int,
    lock_interval: int,
    hourly_candidate_label: str,
    hourly_candidate_type: str,
    hourly_source_model_label: str,
    candidate_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Benchmark phase candidates on top of the already-selected hourly stack."""
    phase_stack_selection_metric = _phase_stack_selection_metric()
    working = minute_timeline.copy()
    timeline_index = pd.DatetimeIndex(pd.to_datetime(working["timestamp"], errors="raise"))
    hourly_series = pd.Series(working["hourly_pred"].to_numpy(dtype=float), index=timeline_index, dtype=float)
    passthrough_timeline = working.copy()
    passthrough_timeline["phase_pred"] = hourly_series.reindex(timeline_index).to_numpy(dtype=float)
    _, passthrough_by_cycle, passthrough_summary = _evaluate_control_scope(
        minute_timeline=passthrough_timeline,
        nowcast_anchor=_placeholder_nowcast_anchor(),
        lock_interval=lock_interval,
    )
    hourly_row = passthrough_summary.loc[passthrough_summary["role"].eq("phase")].iloc[0]
    candidate_rows: list[dict[str, Any]] = []
    prediction_map: dict[str, pd.Series] = {str(hourly_candidate_label): hourly_series.copy()}
    summary_map: dict[str, pd.DataFrame] = {str(hourly_candidate_label): passthrough_summary.copy()}
    stack_candidate_metric_rows: list[dict[str, Any]] = _phase_stack_origin_metric_rows(
        candidate_label=str(hourly_candidate_label),
        minute_timeline=working,
        predicted_series=hourly_series,
        phase_origins=phase_origins,
        horizon_minutes=phase_horizon,
    )
    shortlisted_meta = _phase_stack_native_candidate_shortlist(candidate_meta)
    meta_lookup_source = shortlisted_meta if not shortlisted_meta.empty else candidate_meta
    meta_lookup = (
        meta_lookup_source.sort_values(["candidate_label"], kind="stable")
        .drop_duplicates(subset=["candidate_label"])
        .set_index("candidate_label", drop=False)
        if not meta_lookup_source.empty
        else pd.DataFrame()
    )

    candidate_rows.append(
        _phase_stack_row_from_summary(
            hourly_row=hourly_row,
            candidate_label=str(hourly_candidate_label),
            candidate_type="passthrough",
            source_model_label=str(hourly_source_model_label),
            target_mode="hourly_passthrough",
            replay_pool_rank=0.0,
            replay_pool_source_type="hourly_passthrough",
            replay_pool_source_run_id="",
            replay_resolution="",
            replay_feature_set="",
            replay_model_label=str(hourly_source_model_label),
            replay_run_dir="",
            candidate_by_cycle=passthrough_by_cycle,
            candidate_summary=passthrough_summary,
            stack_candidate_policy="hourly_passthrough",
            stack_blend_weight=float("nan"),
            stack_blend_parent_candidate_label="",
            stack_candidate_family="hourly_passthrough",
        )
    )

    if candidate_detail_by_origin.empty:
        return pd.DataFrame(candidate_rows), prediction_map, summary_map

    blend_parent_payloads: dict[str, dict[str, Any]] = {}
    if not shortlisted_meta.empty:
        shortlisted_labels = shortlisted_meta["candidate_label"].astype("string").dropna().tolist()
        shortlisted_label_set = set(shortlisted_labels)
        candidate_detail_by_origin = candidate_detail_by_origin.loc[
            candidate_detail_by_origin["candidate_label"].astype("string").isin(shortlisted_label_set)
        ].copy()
        if not candidate_metrics_by_origin.empty and "candidate_label" in candidate_metrics_by_origin.columns:
            candidate_metrics_by_origin = candidate_metrics_by_origin.loc[
                candidate_metrics_by_origin["candidate_label"].astype("string").isin(shortlisted_label_set)
            ].copy()
        available_labels = set(candidate_detail_by_origin["candidate_label"].astype("string").dropna().tolist())
        candidate_labels = [str(candidate_label) for candidate_label in shortlisted_labels if str(candidate_label) in available_labels]
        if not candidate_labels:
            candidate_labels = sorted(available_labels)
    else:
        candidate_labels = sorted(candidate_detail_by_origin["candidate_label"].astype("string").dropna().unique().tolist())
    for candidate_label in candidate_labels:
        candidate_series = _apply_rollout_updates(
            hourly_series,
            detail_by_origin=candidate_detail_by_origin,
            candidate_label=str(candidate_label),
            update_origins=phase_origins,
            horizon_minutes=phase_horizon,
        )
        candidate_timeline = working.copy()
        candidate_timeline["phase_pred"] = candidate_series.reindex(timeline_index).to_numpy(dtype=float)
        _, candidate_by_cycle, candidate_summary = _evaluate_control_scope(
            minute_timeline=candidate_timeline,
            nowcast_anchor=_placeholder_nowcast_anchor(),
            lock_interval=lock_interval,
        )
        if isinstance(meta_lookup, pd.DataFrame) and not meta_lookup.empty and str(candidate_label) in meta_lookup.index:
            meta_row = meta_lookup.loc[str(candidate_label)]
        else:
            meta_row = pd.Series(dtype=object)
        candidate_rows.append(
            _phase_stack_row_from_summary(
                hourly_row=hourly_row,
                candidate_label=str(candidate_label),
                candidate_type=str(meta_row.get("candidate_type", "learned")),
                source_model_label=str(meta_row.get("source_model_label", "")),
                target_mode=str(meta_row.get("target_mode", "")),
                replay_pool_rank=float(meta_row.get("replay_pool_rank", float("nan"))),
                replay_pool_source_type=str(meta_row.get("replay_pool_source_type", "")),
                replay_pool_source_run_id=str(meta_row.get("replay_pool_source_run_id", "")),
                replay_resolution=str(meta_row.get("replay_resolution", "")),
                replay_feature_set=str(meta_row.get("replay_feature_set", "")),
                replay_model_label=str(meta_row.get("replay_model_label", "")),
                replay_run_dir=str(meta_row.get("replay_run_dir", "")),
                candidate_by_cycle=candidate_by_cycle,
                candidate_summary=candidate_summary,
                stack_candidate_policy="phase_candidate",
                stack_blend_weight=float("nan"),
                stack_blend_parent_candidate_label="",
                stack_candidate_family="native_phase_candidate",
            )
        )
        prediction_map[str(candidate_label)] = candidate_series
        summary_map[str(candidate_label)] = candidate_summary.copy()
        stack_candidate_metric_rows.extend(
            _phase_stack_origin_metric_rows(
                candidate_label=str(candidate_label),
                minute_timeline=working,
                predicted_series=candidate_series,
                phase_origins=phase_origins,
                horizon_minutes=phase_horizon,
            )
        )
        candidate_type = str(meta_row.get("candidate_type", "learned"))
        if candidate_type == "learned":
            blend_parent_payloads[str(candidate_label)] = {
                "candidate_series": candidate_series.astype(float),
                "meta_row": meta_row.copy(),
            }
    blend_parent_limit = int(MULTIRES_FORECAST_CONTROL.get("phase_stack_blend_parent_top_candidates", 0))
    blend_parent_labels = _select_phase_stack_blend_parent_labels(
        pd.DataFrame(candidate_rows),
        limit=int(blend_parent_limit),
    )
    for candidate_label in blend_parent_labels:
        payload = blend_parent_payloads.get(str(candidate_label))
        if payload is None:
            continue
        candidate_series = cast(pd.Series, payload["candidate_series"])
        meta_row = cast(pd.Series, payload["meta_row"])
        for blend_weight in MULTIRES_FORECAST_CONTROL["phase_stack_blend_weights"]:
            weight = float(blend_weight)
            blended_label = f"{candidate_label}|stack_blend_w{weight:.2f}"
            blended_series = hourly_series + weight * (candidate_series - hourly_series)
            blended_timeline = working.copy()
            blended_timeline["phase_pred"] = blended_series.reindex(timeline_index).to_numpy(dtype=float)
            _, blended_by_cycle, blended_summary = _evaluate_control_scope(
                minute_timeline=blended_timeline,
                nowcast_anchor=_placeholder_nowcast_anchor(),
                lock_interval=lock_interval,
            )
            candidate_rows.append(
                _phase_stack_row_from_summary(
                    hourly_row=hourly_row,
                    candidate_label=str(blended_label),
                    candidate_type="learned",
                    source_model_label=str(meta_row.get("source_model_label", "")),
                    target_mode=f"{str(meta_row.get('target_mode', ''))}|stack_blend",
                    replay_pool_rank=float(meta_row.get("replay_pool_rank", float("nan"))),
                    replay_pool_source_type=str(meta_row.get("replay_pool_source_type", "")),
                    replay_pool_source_run_id=str(meta_row.get("replay_pool_source_run_id", "")),
                    replay_resolution=str(meta_row.get("replay_resolution", "")),
                    replay_feature_set=str(meta_row.get("replay_feature_set", "")),
                    replay_model_label=str(meta_row.get("replay_model_label", "")),
                    replay_run_dir=str(meta_row.get("replay_run_dir", "")),
                    candidate_by_cycle=blended_by_cycle,
                    candidate_summary=blended_summary,
                    stack_candidate_policy="phase_candidate",
                    stack_blend_weight=float(weight),
                    stack_blend_parent_candidate_label=str(candidate_label),
                    stack_candidate_family="hourly_phase_blend",
                )
            )
            prediction_map[str(blended_label)] = blended_series.astype(float)
            summary_map[str(blended_label)] = blended_summary.copy()
            stack_candidate_metric_rows.extend(
                _phase_stack_origin_metric_rows(
                    candidate_label=str(blended_label),
                    minute_timeline=working,
                    predicted_series=blended_series,
                    phase_origins=phase_origins,
                    horizon_minutes=phase_horizon,
                )
            )
    if bool(MULTIRES_FORECAST_CONTROL["phase_stack_bucket_policy_enabled"]):
        bucket_minutes = int(MULTIRES_FORECAST_CONTROL["phase_stack_bucket_granularity_minutes"])
        bucket_policy = _phase_bucket_policy_from_origin_metrics(
            candidate_metrics_by_origin,
            selection_metric=str(MULTIRES_FORECAST_CONTROL["phase_selection_metric"]),
            bucket_minutes=int(bucket_minutes),
        )
        if bucket_policy:
            portfolio_label = "phase_bucket_portfolio::origin_minute_policy"
            portfolio_series = _apply_bucketed_rollout_updates(
                hourly_series,
                detail_by_origin=candidate_detail_by_origin,
                candidate_by_bucket=bucket_policy,
                update_origins=phase_origins,
                horizon_minutes=phase_horizon,
                bucket_minutes=int(bucket_minutes),
                cycle_minutes=60,
            )
            portfolio_timeline = working.copy()
            portfolio_timeline["phase_pred"] = portfolio_series.reindex(timeline_index).to_numpy(dtype=float)
            _, portfolio_by_cycle, portfolio_summary = _evaluate_control_scope(
                minute_timeline=portfolio_timeline,
                nowcast_anchor=_placeholder_nowcast_anchor(),
                lock_interval=lock_interval,
            )
            candidate_rows.append(
                _phase_stack_row_from_summary(
                    hourly_row=hourly_row,
                    candidate_label=portfolio_label,
                    candidate_type="portfolio",
                    source_model_label="phase_bucket_portfolio",
                    target_mode="origin_minute_policy",
                    replay_pool_rank=float("nan"),
                    replay_pool_source_type="phase_bucket_portfolio",
                    replay_pool_source_run_id="",
                    replay_resolution="mixed",
                    replay_feature_set="portfolio",
                    replay_model_label="phase_bucket_portfolio",
                    replay_run_dir="",
                    candidate_by_cycle=portfolio_by_cycle,
                    candidate_summary=portfolio_summary,
                    stack_candidate_policy="phase_candidate",
                    stack_blend_weight=float("nan"),
                    stack_blend_parent_candidate_label="",
                    stack_candidate_family="phase_bucket_portfolio",
                    stack_bucket_policy_json=json.dumps(bucket_policy, sort_keys=True),
                    stack_bucket_granularity_minutes=float(bucket_minutes),
                )
            )
            prediction_map[portfolio_label] = portfolio_series.astype(float)
            summary_map[portfolio_label] = portfolio_summary.copy()
        stack_metric_frame = pd.DataFrame(stack_candidate_metric_rows)
        stack_bucket_selection_metric = (
            phase_stack_selection_metric
            if phase_stack_selection_metric in stack_metric_frame.columns
            else "next_lock_mae"
        )
        stack_bucket_policy = _phase_bucket_policy_from_origin_metrics(
            stack_metric_frame,
            selection_metric=stack_bucket_selection_metric,
            bucket_minutes=int(bucket_minutes),
        )
        if stack_bucket_policy:
            stack_portfolio_label = "phase_bucket_portfolio::stack_origin_metric_policy"
            stack_portfolio_series = _apply_bucketed_series_updates(
                hourly_series,
                series_by_candidate=prediction_map,
                candidate_by_bucket=stack_bucket_policy,
                update_origins=phase_origins,
                horizon_minutes=phase_horizon,
                bucket_minutes=int(bucket_minutes),
                cycle_minutes=60,
            )
            stack_portfolio_timeline = working.copy()
            stack_portfolio_timeline["phase_pred"] = stack_portfolio_series.reindex(timeline_index).to_numpy(
                dtype=float
            )
            _, stack_portfolio_by_cycle, stack_portfolio_summary = _evaluate_control_scope(
                minute_timeline=stack_portfolio_timeline,
                nowcast_anchor=_placeholder_nowcast_anchor(),
                lock_interval=lock_interval,
            )
            candidate_rows.append(
                _phase_stack_row_from_summary(
                    hourly_row=hourly_row,
                    candidate_label=stack_portfolio_label,
                    candidate_type="portfolio",
                    source_model_label="phase_bucket_portfolio",
                    target_mode="stack_origin_metric_policy",
                    replay_pool_rank=float("nan"),
                    replay_pool_source_type="phase_bucket_portfolio",
                    replay_pool_source_run_id="",
                    replay_resolution="mixed",
                    replay_feature_set="portfolio",
                    replay_model_label="phase_bucket_portfolio",
                    replay_run_dir="",
                    candidate_by_cycle=stack_portfolio_by_cycle,
                    candidate_summary=stack_portfolio_summary,
                    stack_candidate_policy="phase_candidate",
                    stack_blend_weight=float("nan"),
                    stack_blend_parent_candidate_label="",
                    stack_candidate_family="phase_bucket_portfolio",
                    stack_bucket_policy_json=json.dumps(stack_bucket_policy, sort_keys=True),
                    stack_bucket_granularity_minutes=float(bucket_minutes),
                )
            )
            prediction_map[stack_portfolio_label] = stack_portfolio_series.astype(float)
            summary_map[stack_portfolio_label] = stack_portfolio_summary.copy()
    benchmark = _sort_phase_stack_benchmark(pd.DataFrame(candidate_rows))
    return benchmark, prediction_map, summary_map


def _sort_phase_stack_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical ordering for stack-aware phase candidate tables."""
    if frame.empty:
        return frame.copy()
    selection_metric = _phase_stack_selection_metric()
    sort_columns = [column for column in _phase_stack_metric_sort_columns(selection_metric) if column in frame.columns]
    if "candidate_label" not in sort_columns:
        sort_columns.append("candidate_label")
    return frame.sort_values(
        ["stack_candidate_policy", "meets_stack_guard", *sort_columns],
        ascending=[True, False, *([True] * len(sort_columns))],
        kind="stable",
    ).reset_index(drop=True)


def _select_phase_stack_blend_parent_labels(benchmark: pd.DataFrame, *, limit: int) -> list[str]:
    """Choose the strongest native learned phase candidates to expand into stack blends."""
    if benchmark.empty:
        return []
    eligible = benchmark.loc[
        benchmark["stack_candidate_family"].astype("string").eq("native_phase_candidate")
        & benchmark["candidate_type"].astype("string").eq("learned")
    ].copy()
    if eligible.empty:
        return []
    selection_metric = _phase_stack_selection_metric()
    sort_columns = [column for column in _phase_stack_metric_sort_columns(selection_metric) if column in eligible.columns]
    if "candidate_label" not in sort_columns:
        sort_columns.append("candidate_label")
    eligible = eligible.sort_values(sort_columns, ascending=[True] * len(sort_columns), kind="stable")
    eligible = eligible.drop_duplicates(subset=["candidate_label"], keep="first")
    if int(limit) > 0:
        eligible = eligible.head(int(limit))
    return eligible["candidate_label"].astype("string").tolist()


def _phase_stack_reference_baseline_row(
    *,
    calibration_benchmark: pd.DataFrame,
    source_row: pd.Series,
) -> pd.Series:
    """Pick the best reconstructable baseline anchor for one learned phase candidate."""
    if calibration_benchmark.empty:
        return pd.Series(dtype=object)
    baseline_rows = calibration_benchmark.loc[
        calibration_benchmark["candidate_type"].astype("string").eq("baseline")
        & calibration_benchmark["stack_candidate_family"].astype("string").eq("native_phase_candidate")
        & calibration_benchmark["replay_pool_source_run_id"].astype("string").eq(
            str(source_row.get("replay_pool_source_run_id", ""))
        )
        & calibration_benchmark["replay_resolution"].astype("string").eq(
            str(source_row.get("replay_resolution", ""))
        )
    ].copy()
    if baseline_rows.empty:
        return pd.Series(dtype=object)
    baseline_rows = baseline_rows.sort_values(
        ["lock_mae", "profile_shape_mae", "minute_path_mae", "candidate_label"],
        kind="stable",
    )
    return baseline_rows.iloc[0]


def _phase_stack_baseline_control_candidates(
    *,
    calibration_minute_timeline: pd.DataFrame,
    evaluation_minute_timeline: pd.DataFrame,
    lock_interval: int,
    calibration_benchmark: pd.DataFrame,
    evaluation_benchmark: pd.DataFrame,
    calibration_predictions: dict[str, pd.Series],
    evaluation_predictions: dict[str, pd.Series],
    calibration_summaries: dict[str, pd.DataFrame],
    evaluation_summaries: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Calibrate stack-level phase blends toward the best same-family baseline on held-out control data."""
    if not bool(MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_enabled"]):
        return (
            calibration_benchmark,
            evaluation_benchmark,
            calibration_predictions,
            evaluation_predictions,
            calibration_summaries,
            evaluation_summaries,
        )
    if calibration_benchmark.empty or evaluation_benchmark.empty:
        return (
            calibration_benchmark,
            evaluation_benchmark,
            calibration_predictions,
            evaluation_predictions,
            calibration_summaries,
            evaluation_summaries,
        )
    calibration_hourly_row = calibration_benchmark.loc[
        calibration_benchmark["stack_candidate_family"].astype("string").eq("hourly_passthrough")
    ]
    evaluation_hourly_row = evaluation_benchmark.loc[
        evaluation_benchmark["stack_candidate_family"].astype("string").eq("hourly_passthrough")
    ]
    if calibration_hourly_row.empty or evaluation_hourly_row.empty:
        return (
            calibration_benchmark,
            evaluation_benchmark,
            calibration_predictions,
            evaluation_predictions,
            calibration_summaries,
            evaluation_summaries,
        )
    calibration_hourly_row = calibration_hourly_row.iloc[0]
    evaluation_hourly_row = evaluation_hourly_row.iloc[0]
    source_benchmark = calibration_benchmark
    source_predictions = calibration_predictions
    source_minute_timeline = calibration_minute_timeline
    learned_rows = source_benchmark.loc[
        source_benchmark["candidate_type"].astype("string").eq("learned")
        & source_benchmark["stack_candidate_family"].astype("string").eq("native_phase_candidate")
    ].copy()
    fallback_to_evaluation = False
    if learned_rows.empty:
        source_benchmark = evaluation_benchmark
        source_predictions = evaluation_predictions
        source_minute_timeline = evaluation_minute_timeline
        learned_rows = source_benchmark.loc[
            source_benchmark["candidate_type"].astype("string").eq("learned")
            & source_benchmark["stack_candidate_family"].astype("string").eq("native_phase_candidate")
        ].copy()
        fallback_to_evaluation = True
    if learned_rows.empty:
        return (
            calibration_benchmark,
            evaluation_benchmark,
            calibration_predictions,
            evaluation_predictions,
            calibration_summaries,
            evaluation_summaries,
        )
    phase_stack_selection_metric = _phase_stack_selection_metric()
    learned_sort_columns = [
        column for column in _phase_stack_metric_sort_columns(phase_stack_selection_metric) if column in learned_rows.columns
    ]
    if "candidate_label" not in learned_sort_columns:
        learned_sort_columns.append("candidate_label")
    learned_rows = learned_rows.sort_values(learned_sort_columns, kind="stable").head(
        int(MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_top_candidates"])
    )
    blend_weights = [
        float(value) for value in MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_blend_weights"]
    ]
    bucket_size = int(MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_bucket_size_minutes"])
    bucket_enabled = bool(MULTIRES_FORECAST_CONTROL["phase_stack_baseline_control_bucket_blend_enabled"])
    bucket_keys = sorted(
        {
            int(value)
            for value in _index_minute_buckets(
                pd.DatetimeIndex(pd.to_datetime(source_minute_timeline["timestamp"], errors="raise")),
                bucket_minutes=int(bucket_size),
                cycle_minutes=int(lock_interval),
            )
            .dropna()
            .astype(int)
            .tolist()
        }
    )
    calibration_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    updated_calibration_predictions = dict(calibration_predictions)
    updated_evaluation_predictions = dict(evaluation_predictions)
    updated_calibration_summaries = dict(calibration_summaries)
    updated_evaluation_summaries = dict(evaluation_summaries)
    for _, source_row in learned_rows.iterrows():
        source_label = str(source_row["candidate_label"])
        if source_label not in source_predictions or source_label not in evaluation_predictions:
            continue
        baseline_row = _phase_stack_reference_baseline_row(
            calibration_benchmark=source_benchmark,
            source_row=source_row,
        )
        if baseline_row.empty:
            continue
        baseline_label = str(baseline_row["candidate_label"])
        if baseline_label not in source_predictions or baseline_label not in evaluation_predictions:
            continue
        source_prediction_series = source_predictions[source_label]
        evaluation_source_series = evaluation_predictions[source_label]
        source_baseline_series = source_predictions[baseline_label]
        evaluation_baseline_series = evaluation_predictions[baseline_label]
        best_global_weight = float("nan")
        best_metric: tuple[float, ...] | None = None
        best_source_summary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
        for candidate_weight in blend_weights:
            blended_source = _blend_control_prediction_series(
                candidate_series=source_prediction_series,
                reference_series=source_baseline_series,
                candidate_weight=float(candidate_weight),
            )
            source_timeline, source_by_cycle, source_summary = _phase_stack_candidate_summary_for_series(
                minute_timeline=source_minute_timeline,
                candidate_series=blended_source,
                lock_interval=lock_interval,
            )
            calibration_phase_row = source_summary.loc[source_summary["role"].eq("phase")].iloc[0]
            metric_tuple = _phase_stack_metric_tuple_from_row(
                calibration_phase_row,
                selection_metric=phase_stack_selection_metric,
                tie_breaker=float(candidate_weight),
            )
            if best_source_summary is None or best_metric is None:
                best_metric = metric_tuple
                best_global_weight = float(candidate_weight)
                best_source_summary = (source_timeline, source_by_cycle, source_summary)
            elif metric_tuple < best_metric:
                best_metric = metric_tuple
                best_global_weight = float(candidate_weight)
                best_source_summary = (source_timeline, source_by_cycle, source_summary)
        if best_source_summary is None or not np.isfinite(best_global_weight):
            continue
        evaluation_blended = _blend_control_prediction_series(
            candidate_series=evaluation_source_series,
            reference_series=evaluation_baseline_series,
            candidate_weight=float(best_global_weight),
        )
        (
            _,
            evaluation_by_cycle,
            evaluation_summary,
        ) = _phase_stack_candidate_summary_for_series(
            minute_timeline=evaluation_minute_timeline,
            candidate_series=evaluation_blended,
            lock_interval=lock_interval,
        )
        global_label = f"{source_label}|baseline_control_blend_w{best_global_weight:.2f}"
        if not fallback_to_evaluation:
            calibration_rows.append(
                _phase_stack_row_from_summary(
                    hourly_row=calibration_hourly_row,
                    candidate_label=global_label,
                    candidate_type=str(source_row.get("candidate_type", "learned")),
                    source_model_label=str(source_row.get("source_model_label", "")),
                    target_mode=f"{str(source_row.get('target_mode', ''))}|baseline_control_blend",
                    replay_pool_rank=float(source_row.get("replay_pool_rank", float("nan"))),
                    replay_pool_source_type=str(source_row.get("replay_pool_source_type", "")),
                    replay_pool_source_run_id=str(source_row.get("replay_pool_source_run_id", "")),
                    replay_resolution=str(source_row.get("replay_resolution", "")),
                    replay_feature_set=str(source_row.get("replay_feature_set", "")),
                    replay_model_label=str(source_row.get("replay_model_label", "")),
                    replay_run_dir=str(source_row.get("replay_run_dir", "")),
                    candidate_by_cycle=best_source_summary[1],
                    candidate_summary=best_source_summary[2],
                    stack_candidate_policy="phase_candidate",
                    stack_blend_weight=float(best_global_weight),
                    stack_blend_parent_candidate_label=source_label,
                    stack_candidate_family="phase_baseline_control_blend",
                    stack_reference_candidate_label=baseline_label,
                )
            )
        evaluation_rows.append(
            _phase_stack_row_from_summary(
                hourly_row=evaluation_hourly_row,
                candidate_label=global_label,
                candidate_type=str(source_row.get("candidate_type", "learned")),
                source_model_label=str(source_row.get("source_model_label", "")),
                target_mode=f"{str(source_row.get('target_mode', ''))}|baseline_control_blend",
                replay_pool_rank=float(source_row.get("replay_pool_rank", float("nan"))),
                replay_pool_source_type=str(source_row.get("replay_pool_source_type", "")),
                replay_pool_source_run_id=str(source_row.get("replay_pool_source_run_id", "")),
                replay_resolution=str(source_row.get("replay_resolution", "")),
                replay_feature_set=str(source_row.get("replay_feature_set", "")),
                replay_model_label=str(source_row.get("replay_model_label", "")),
                replay_run_dir=str(source_row.get("replay_run_dir", "")),
                candidate_by_cycle=evaluation_by_cycle,
                candidate_summary=evaluation_summary,
                stack_candidate_policy="phase_candidate",
                stack_blend_weight=float(best_global_weight),
                stack_blend_parent_candidate_label=source_label,
                stack_candidate_family="phase_baseline_control_blend",
                stack_reference_candidate_label=baseline_label,
            )
        )
        if not fallback_to_evaluation:
            updated_calibration_predictions[global_label] = _blend_control_prediction_series(
                candidate_series=source_prediction_series,
                reference_series=source_baseline_series,
                candidate_weight=float(best_global_weight),
            )
        updated_evaluation_predictions[global_label] = evaluation_blended.astype(float)
        if not fallback_to_evaluation:
            updated_calibration_summaries[global_label] = best_source_summary[2].copy()
        updated_evaluation_summaries[global_label] = evaluation_summary.copy()
        if not bucket_enabled or not bucket_keys:
            continue
        chosen_weights = {int(bucket): float(best_global_weight) for bucket in bucket_keys}
        for bucket_key in bucket_keys:
            best_bucket_metric: tuple[float, float, float, float] | None = None
            best_bucket_weight = chosen_weights[int(bucket_key)]
            for candidate_weight in blend_weights:
                trial_weights = dict(chosen_weights)
                trial_weights[int(bucket_key)] = float(candidate_weight)
                bucketed_source = _blend_control_prediction_series_by_bucket(
                    candidate_series=source_prediction_series,
                    reference_series=source_baseline_series,
                    candidate_weight_by_bucket=trial_weights,
                    bucket_size_minutes=int(bucket_size),
                    lock_interval_minutes=int(lock_interval),
                )
                _, bucketed_by_cycle, bucketed_summary = _phase_stack_candidate_summary_for_series(
                    minute_timeline=source_minute_timeline,
                    candidate_series=bucketed_source,
                    lock_interval=lock_interval,
                )
                bucketed_phase_row = bucketed_summary.loc[bucketed_summary["role"].eq("phase")].iloc[0]
                bucket_metric = _phase_stack_metric_tuple_from_row(
                    bucketed_phase_row,
                    selection_metric=phase_stack_selection_metric,
                    tie_breaker=float(candidate_weight),
                )
                if best_bucket_metric is None or bucket_metric < best_bucket_metric:
                    best_bucket_metric = bucket_metric
                    best_bucket_weight = float(candidate_weight)
            chosen_weights[int(bucket_key)] = float(best_bucket_weight)
        bucketed_source_series = _blend_control_prediction_series_by_bucket(
            candidate_series=source_prediction_series,
            reference_series=source_baseline_series,
            candidate_weight_by_bucket=chosen_weights,
            bucket_size_minutes=int(bucket_size),
            lock_interval_minutes=int(lock_interval),
        )
        if not fallback_to_evaluation:
            (
                _,
                bucketed_calibration_by_cycle,
                bucketed_calibration_summary,
            ) = _phase_stack_candidate_summary_for_series(
                minute_timeline=calibration_minute_timeline,
                candidate_series=bucketed_source_series,
                lock_interval=lock_interval,
            )
        bucketed_evaluation_series = _blend_control_prediction_series_by_bucket(
            candidate_series=evaluation_source_series,
            reference_series=evaluation_baseline_series,
            candidate_weight_by_bucket=chosen_weights,
            bucket_size_minutes=int(bucket_size),
            lock_interval_minutes=int(lock_interval),
        )
        (
            _,
            bucketed_evaluation_by_cycle,
            bucketed_evaluation_summary,
        ) = _phase_stack_candidate_summary_for_series(
            minute_timeline=evaluation_minute_timeline,
            candidate_series=bucketed_evaluation_series,
            lock_interval=lock_interval,
        )
        bucket_label = f"{source_label}|baseline_control_bucket_blend_b{bucket_size}"
        bucket_weights_json = json.dumps(chosen_weights, sort_keys=True)
        if not fallback_to_evaluation:
            calibration_rows.append(
                _phase_stack_row_from_summary(
                    hourly_row=calibration_hourly_row,
                    candidate_label=bucket_label,
                    candidate_type=str(source_row.get("candidate_type", "learned")),
                    source_model_label=str(source_row.get("source_model_label", "")),
                    target_mode=f"{str(source_row.get('target_mode', ''))}|baseline_control_bucket_blend",
                    replay_pool_rank=float(source_row.get("replay_pool_rank", float("nan"))),
                    replay_pool_source_type=str(source_row.get("replay_pool_source_type", "")),
                    replay_pool_source_run_id=str(source_row.get("replay_pool_source_run_id", "")),
                    replay_resolution=str(source_row.get("replay_resolution", "")),
                    replay_feature_set=str(source_row.get("replay_feature_set", "")),
                    replay_model_label=str(source_row.get("replay_model_label", "")),
                    replay_run_dir=str(source_row.get("replay_run_dir", "")),
                    candidate_by_cycle=bucketed_calibration_by_cycle,
                    candidate_summary=bucketed_calibration_summary,
                    stack_candidate_policy="phase_candidate",
                    stack_blend_weight=float("nan"),
                    stack_blend_parent_candidate_label=source_label,
                    stack_candidate_family="phase_baseline_bucket_control_blend",
                    stack_reference_candidate_label=baseline_label,
                    stack_bucket_weight_json=bucket_weights_json,
                    stack_bucket_granularity_minutes=float(bucket_size),
                )
            )
        evaluation_rows.append(
            _phase_stack_row_from_summary(
                hourly_row=evaluation_hourly_row,
                candidate_label=bucket_label,
                candidate_type=str(source_row.get("candidate_type", "learned")),
                source_model_label=str(source_row.get("source_model_label", "")),
                target_mode=f"{str(source_row.get('target_mode', ''))}|baseline_control_bucket_blend",
                replay_pool_rank=float(source_row.get("replay_pool_rank", float("nan"))),
                replay_pool_source_type=str(source_row.get("replay_pool_source_type", "")),
                replay_pool_source_run_id=str(source_row.get("replay_pool_source_run_id", "")),
                replay_resolution=str(source_row.get("replay_resolution", "")),
                replay_feature_set=str(source_row.get("replay_feature_set", "")),
                replay_model_label=str(source_row.get("replay_model_label", "")),
                replay_run_dir=str(source_row.get("replay_run_dir", "")),
                candidate_by_cycle=bucketed_evaluation_by_cycle,
                candidate_summary=bucketed_evaluation_summary,
                stack_candidate_policy="phase_candidate",
                stack_blend_weight=float("nan"),
                stack_blend_parent_candidate_label=source_label,
                stack_candidate_family="phase_baseline_bucket_control_blend",
                stack_reference_candidate_label=baseline_label,
                stack_bucket_weight_json=bucket_weights_json,
                stack_bucket_granularity_minutes=float(bucket_size),
            )
        )
        if not fallback_to_evaluation:
            updated_calibration_predictions[bucket_label] = bucketed_source_series.astype(float)
        updated_evaluation_predictions[bucket_label] = bucketed_evaluation_series.astype(float)
        if not fallback_to_evaluation:
            updated_calibration_summaries[bucket_label] = bucketed_calibration_summary.copy()
        updated_evaluation_summaries[bucket_label] = bucketed_evaluation_summary.copy()
    if calibration_rows:
        calibration_benchmark = _sort_phase_stack_benchmark(
            pd.concat([calibration_benchmark, pd.DataFrame(calibration_rows)], ignore_index=True)
        )
    if evaluation_rows:
        evaluation_benchmark = _sort_phase_stack_benchmark(
            pd.concat([evaluation_benchmark, pd.DataFrame(evaluation_rows)], ignore_index=True)
        )
    return (
        calibration_benchmark,
        evaluation_benchmark,
        updated_calibration_predictions,
        updated_evaluation_predictions,
        updated_calibration_summaries,
        updated_evaluation_summaries,
    )


def _rename_phase_stack_evaluation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename one phase-stack benchmark so calibration and evaluation tables can be compared side by side."""
    rename_map = {
        "candidate_type": "evaluation_candidate_type",
        "source_model_label": "evaluation_source_model_label",
        "target_mode": "evaluation_target_mode",
        "minute_path_mae": "evaluation_minute_path_mae",
        "minute_path_mae_pct": "evaluation_minute_path_mae_pct",
        "lock_mae": "evaluation_lock_mae",
        "lock_mae_pct": "evaluation_lock_mae_pct",
        "profile_shape_mae": "evaluation_profile_shape_mae",
        "profile_shape_mae_pct": "evaluation_profile_shape_mae_pct",
        "energy_mae": "evaluation_energy_mae",
        "energy_mae_pct": "evaluation_energy_mae_pct",
        "cycle_n": "evaluation_cycle_n",
        "lock_mae_p50": "evaluation_lock_mae_p50",
        "lock_mae_p90": "evaluation_lock_mae_p90",
        "profile_shape_mae_p50": "evaluation_profile_shape_mae_p50",
        "profile_shape_mae_p90": "evaluation_profile_shape_mae_p90",
        "minute_path_mae_p50": "evaluation_minute_path_mae_p50",
        "minute_path_mae_p90": "evaluation_minute_path_mae_p90",
        "lock_gain_vs_hourly": "evaluation_lock_gain_vs_hourly",
        "lock_gain_pct_vs_hourly": "evaluation_lock_gain_pct_vs_hourly",
        "next_lock_regress_vs_hourly": "evaluation_next_lock_regress_vs_hourly",
        "next_lock_regress_pct_vs_hourly": "evaluation_next_lock_regress_pct_vs_hourly",
        "profile_degrade_vs_hourly": "evaluation_profile_degrade_vs_hourly",
        "profile_degrade_pct_vs_hourly": "evaluation_profile_degrade_pct_vs_hourly",
        "peak_value_regress_vs_hourly": "evaluation_peak_value_regress_vs_hourly",
        "peak_value_regress_pct_vs_hourly": "evaluation_peak_value_regress_pct_vs_hourly",
        "peak_hit_gain_vs_hourly": "evaluation_peak_hit_gain_vs_hourly",
        "meets_lock_gain_rule": "evaluation_meets_lock_gain_rule",
        "meets_next_lock_rule": "evaluation_meets_next_lock_rule",
        "meets_profile_rule": "evaluation_meets_profile_rule",
        "meets_peak_value_rule": "evaluation_meets_peak_value_rule",
        "meets_peak_hit_rule": "evaluation_meets_peak_hit_rule",
        "meets_stack_guard": "evaluation_meets_stack_guard",
        "stack_candidate_policy": "evaluation_stack_candidate_policy",
        "selection_metric_name": "evaluation_selection_metric_name",
        "selection_metric_value": "evaluation_selection_metric_value",
        "selection_metric_pct": "evaluation_selection_metric_pct",
    }
    keep_columns = ["candidate_label"] + [column for column in rename_map if column in frame.columns]
    return frame.loc[:, keep_columns].rename(columns=rename_map)


def _rename_phase_stack_calibration_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename one phase-stack benchmark so held-out tables can include calibration diagnostics."""
    rename_map = {
        "candidate_type": "calibration_candidate_type",
        "source_model_label": "calibration_source_model_label",
        "target_mode": "calibration_target_mode",
        "minute_path_mae": "calibration_minute_path_mae",
        "minute_path_mae_pct": "calibration_minute_path_mae_pct",
        "lock_mae": "calibration_lock_mae",
        "lock_mae_pct": "calibration_lock_mae_pct",
        "profile_shape_mae": "calibration_profile_shape_mae",
        "profile_shape_mae_pct": "calibration_profile_shape_mae_pct",
        "energy_mae": "calibration_energy_mae",
        "energy_mae_pct": "calibration_energy_mae_pct",
        "cycle_n": "calibration_cycle_n",
        "lock_mae_p50": "calibration_lock_mae_p50",
        "lock_mae_p90": "calibration_lock_mae_p90",
        "profile_shape_mae_p50": "calibration_profile_shape_mae_p50",
        "profile_shape_mae_p90": "calibration_profile_shape_mae_p90",
        "minute_path_mae_p50": "calibration_minute_path_mae_p50",
        "minute_path_mae_p90": "calibration_minute_path_mae_p90",
        "lock_gain_vs_hourly": "calibration_lock_gain_vs_hourly",
        "lock_gain_pct_vs_hourly": "calibration_lock_gain_pct_vs_hourly",
        "next_lock_regress_vs_hourly": "calibration_next_lock_regress_vs_hourly",
        "next_lock_regress_pct_vs_hourly": "calibration_next_lock_regress_pct_vs_hourly",
        "profile_degrade_vs_hourly": "calibration_profile_degrade_vs_hourly",
        "profile_degrade_pct_vs_hourly": "calibration_profile_degrade_pct_vs_hourly",
        "peak_value_regress_vs_hourly": "calibration_peak_value_regress_vs_hourly",
        "peak_value_regress_pct_vs_hourly": "calibration_peak_value_regress_pct_vs_hourly",
        "peak_hit_gain_vs_hourly": "calibration_peak_hit_gain_vs_hourly",
        "meets_lock_gain_rule": "calibration_meets_lock_gain_rule",
        "meets_next_lock_rule": "calibration_meets_next_lock_rule",
        "meets_profile_rule": "calibration_meets_profile_rule",
        "meets_peak_value_rule": "calibration_meets_peak_value_rule",
        "meets_peak_hit_rule": "calibration_meets_peak_hit_rule",
        "meets_stack_guard": "calibration_meets_stack_guard",
        "stack_candidate_policy": "calibration_stack_candidate_policy",
        "selection_metric_name": "calibration_selection_metric_name",
        "selection_metric_value": "calibration_selection_metric_value",
        "selection_metric_pct": "calibration_selection_metric_pct",
        "stack_bucket_policy_json": "calibration_stack_bucket_policy_json",
        "stack_bucket_granularity_minutes": "calibration_stack_bucket_granularity_minutes",
    }
    keep_columns = ["candidate_label"] + [column for column in rename_map if column in frame.columns]
    return frame.loc[:, keep_columns].rename(columns=rename_map)


def _select_phase_stack_candidate(
    *,
    calibration_benchmark: pd.DataFrame,
    evaluation_benchmark: pd.DataFrame,
    hourly_candidate_label: str,
) -> tuple[pd.Series, str]:
    """Select the stack-applied phase policy from hourly passthrough plus stack-qualified challengers."""
    selection_metric = _phase_stack_selection_metric()
    requested_sort_columns = _phase_stack_metric_sort_columns(selection_metric)
    use_evaluation = str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"]) == "held_out_evaluation"
    if use_evaluation:
        comparison = evaluation_benchmark.copy()
        if comparison.empty:
            raise RuntimeError("Phase stack benchmark did not produce any held-out candidates.")
        policy_column = "stack_candidate_policy"
        meets_guard_column = "meets_stack_guard"
    else:
        comparison = calibration_benchmark.copy()
        if comparison.empty:
            raise RuntimeError("Phase stack benchmark did not produce any calibration candidates.")
        policy_column = "stack_candidate_policy"
        meets_guard_column = "meets_stack_guard"
    eligible = comparison.loc[
        comparison[policy_column].astype("string").eq("phase_candidate")
        & comparison[meets_guard_column].astype(bool)
    ].copy()
    if not eligible.empty:
        sort_columns = [column for column in requested_sort_columns if column in eligible.columns]
        if "candidate_label" not in sort_columns:
            sort_columns.append("candidate_label")
        selected = eligible.sort_values(sort_columns, ascending=[True] * len(sort_columns), kind="stable").iloc[0]
        return selected, ("held_out_phase_stack_candidate_benchmark" if use_evaluation else "calibration_phase_stack_candidate_benchmark")
    hourly_row = comparison.loc[
        comparison["candidate_label"].astype("string").eq(str(hourly_candidate_label))
    ].copy()
    if hourly_row.empty:
        raise RuntimeError("Phase stack benchmark did not retain the hourly passthrough candidate.")
    return hourly_row.iloc[0], ("held_out_phase_stack_hourly_passthrough" if use_evaluation else "calibration_phase_stack_hourly_passthrough")


def _phase_stack_decision_from_selected_row(
    *,
    selected_row: pd.Series,
    selection_mode: str,
    hourly_candidate_label: str,
    isolated_candidate_label: str,
) -> dict[str, Any]:
    """Translate the chosen stack-level phase candidate into the persisted guard/policy decision surface."""
    use_evaluation = str(selection_mode).startswith("held_out_")
    prefix = "evaluation_" if use_evaluation else ""
    candidate_policy = str(selected_row.get(f"{prefix}stack_candidate_policy", selected_row.get("stack_candidate_policy", "")))
    lock_gain = float(selected_row.get(f"{prefix}lock_gain_vs_hourly", selected_row.get("lock_gain_vs_hourly", float("nan"))))
    lock_gain_pct = float(
        selected_row.get(f"{prefix}lock_gain_pct_vs_hourly", selected_row.get("lock_gain_pct_vs_hourly", float("nan")))
    )
    next_lock_regress = float(
        selected_row.get(
            f"{prefix}next_lock_regress_vs_hourly",
            selected_row.get("next_lock_regress_vs_hourly", float("nan")),
        )
    )
    next_lock_regress_pct = float(
        selected_row.get(
            f"{prefix}next_lock_regress_pct_vs_hourly",
            selected_row.get("next_lock_regress_pct_vs_hourly", float("nan")),
        )
    )
    profile_degrade = float(
        selected_row.get(f"{prefix}profile_degrade_vs_hourly", selected_row.get("profile_degrade_vs_hourly", float("nan")))
    )
    profile_degrade_pct = float(
        selected_row.get(
            f"{prefix}profile_degrade_pct_vs_hourly",
            selected_row.get("profile_degrade_pct_vs_hourly", float("nan")),
        )
    )
    peak_value_regress = float(
        selected_row.get(
            f"{prefix}peak_value_regress_vs_hourly",
            selected_row.get("peak_value_regress_vs_hourly", float("nan")),
        )
    )
    peak_value_regress_pct = float(
        selected_row.get(
            f"{prefix}peak_value_regress_pct_vs_hourly",
            selected_row.get("peak_value_regress_pct_vs_hourly", float("nan")),
        )
    )
    peak_hit_gain = float(
        selected_row.get(
            f"{prefix}peak_hit_gain_vs_hourly",
            selected_row.get("peak_hit_gain_vs_hourly", float("nan")),
        )
    )
    optimizer_regress = float(
        selected_row.get(
            f"{prefix}optimizer_regress_vs_hourly",
            selected_row.get("optimizer_regress_vs_hourly", float("nan")),
        )
    )
    optimizer_regress_pct = float(
        selected_row.get(
            f"{prefix}optimizer_regress_pct_vs_hourly",
            selected_row.get("optimizer_regress_pct_vs_hourly", float("nan")),
        )
    )
    meets_lock_gain_rule = bool(
        selected_row.get(f"{prefix}meets_lock_gain_rule", selected_row.get("meets_lock_gain_rule", False))
    )
    meets_next_lock_rule = bool(
        selected_row.get(f"{prefix}meets_next_lock_rule", selected_row.get("meets_next_lock_rule", False))
    )
    meets_profile_rule = bool(
        selected_row.get(f"{prefix}meets_profile_rule", selected_row.get("meets_profile_rule", False))
    )
    meets_peak_value_rule = bool(
        selected_row.get(f"{prefix}meets_peak_value_rule", selected_row.get("meets_peak_value_rule", False))
    )
    meets_peak_hit_rule = bool(
        selected_row.get(f"{prefix}meets_peak_hit_rule", selected_row.get("meets_peak_hit_rule", False))
    )
    meets_optimizer_rule = bool(
        selected_row.get(f"{prefix}meets_optimizer_rule", selected_row.get("meets_optimizer_rule", False))
    )
    recommended_policy = "phase_candidate" if candidate_policy == "phase_candidate" else "hourly_passthrough"
    applied_candidate_label = (
        str(selected_row.get("candidate_label", "")) if recommended_policy == "phase_candidate" else str(hourly_candidate_label)
    )
    if recommended_policy == "phase_candidate":
        reason = (
            "Selected the stack-aware phase candidate because it cleared the phase stack guard on the "
            "chosen control scope and minimized the configured stack-level selection metric without "
            "unacceptable next-lock, peak, optimizer, or profile regression."
        )
    else:
        reason = (
            "Falling back to the hourly path at the phase layer because no replayed phase candidate "
            "cleared the stack-level lock-gain, next-lock, peak, profile-regression, and "
            "optimizer-regression guard on the chosen control scope."
        )
    return {
        "enabled": True,
        "decision_scope": str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"]),
        "recommended_policy": recommended_policy,
        "applied_candidate_label": applied_candidate_label,
        "lock_gain_vs_hourly": float(lock_gain),
        "lock_gain_pct_vs_hourly": float(lock_gain_pct),
        "next_lock_regress_vs_hourly": float(next_lock_regress),
        "next_lock_regress_pct_vs_hourly": float(next_lock_regress_pct),
        "profile_degrade_vs_hourly": float(profile_degrade),
        "profile_degrade_pct_vs_hourly": float(profile_degrade_pct),
        "peak_value_regress_vs_hourly": float(peak_value_regress),
        "peak_value_regress_pct_vs_hourly": float(peak_value_regress_pct),
        "peak_hit_gain_vs_hourly": float(peak_hit_gain),
        "optimizer_regress_vs_hourly": float(optimizer_regress),
        "optimizer_regress_pct_vs_hourly": float(optimizer_regress_pct),
        "meets_lock_gain_rule": bool(meets_lock_gain_rule),
        "meets_next_lock_rule": bool(meets_next_lock_rule),
        "meets_profile_rule": bool(meets_profile_rule),
        "meets_peak_value_rule": bool(meets_peak_value_rule),
        "meets_peak_hit_rule": bool(meets_peak_hit_rule),
        "meets_optimizer_rule": bool(meets_optimizer_rule),
        "reason": reason,
        "selection_mode": str(selection_mode),
        "isolated_candidate_label": str(isolated_candidate_label),
        "stack_selected_candidate_label": str(selected_row.get("candidate_label", "")),
    }


def _day_ahead_refresh_summary_frame(by_cycle: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frozen versus refreshed day-ahead scenarios into one comparison table."""
    rows: list[dict[str, Any]] = []
    for scenario in ("frozen_day_ahead", "unconditional_refresh", "triggered_refresh"):
        update_column = (
            f"{scenario}_update_count"
            if f"{scenario}_update_count" in by_cycle.columns
            else f"{scenario}_refresh_update_count"
        )
        rows.append(
            {
                "scenario": scenario,
                "cycle_n": int(len(by_cycle)),
                "minute_path_mae": float(by_cycle[f"{scenario}_minute_path_mae"].mean()),
                "minute_path_mae_p50": float(by_cycle[f"{scenario}_minute_path_mae"].quantile(0.5)),
                "minute_path_mae_p90": float(by_cycle[f"{scenario}_minute_path_mae"].quantile(0.9)),
                "minute_path_mae_pct": float(by_cycle[f"{scenario}_minute_path_mae_pct"].mean()),
                "lock_mae": float(by_cycle[f"{scenario}_lock_mae"].mean()),
                "lock_mae_p50": float(by_cycle[f"{scenario}_lock_mae"].quantile(0.5)),
                "lock_mae_p90": float(by_cycle[f"{scenario}_lock_mae"].quantile(0.9)),
                "lock_mae_pct": float(by_cycle[f"{scenario}_lock_mae_pct"].mean()),
                "profile_shape_mae": float(by_cycle[f"{scenario}_profile_shape_mae"].mean()),
                "profile_shape_mae_p50": float(by_cycle[f"{scenario}_profile_shape_mae"].quantile(0.5)),
                "profile_shape_mae_p90": float(by_cycle[f"{scenario}_profile_shape_mae"].quantile(0.9)),
                "profile_shape_mae_pct": float(by_cycle[f"{scenario}_profile_shape_mae_pct"].mean()),
                "energy_mae": float(by_cycle[f"{scenario}_energy_mae"].mean()),
                "energy_mae_p50": float(by_cycle[f"{scenario}_energy_mae"].quantile(0.5)),
                "energy_mae_p90": float(by_cycle[f"{scenario}_energy_mae"].quantile(0.9)),
                "energy_mae_pct": float(by_cycle[f"{scenario}_energy_mae_pct"].mean()),
                "refresh_update_count": (
                    float(by_cycle[update_column].mean())
                    if update_column in by_cycle.columns
                    else 0.0
                ),
            }
        )
    summary = pd.DataFrame(rows)
    frozen_lock = float(summary.loc[summary["scenario"].eq("frozen_day_ahead"), "lock_mae"].iloc[0])
    frozen_profile = float(
        summary.loc[summary["scenario"].eq("frozen_day_ahead"), "profile_shape_mae"].iloc[0]
    )
    summary["lock_mae_gain_vs_frozen"] = frozen_lock - summary["lock_mae"]
    summary["profile_shape_mae_gain_vs_frozen"] = frozen_profile - summary["profile_shape_mae"]
    frozen_lock_p50 = float(summary.loc[summary["scenario"].eq("frozen_day_ahead"), "lock_mae_p50"].iloc[0])
    frozen_lock_p90 = float(summary.loc[summary["scenario"].eq("frozen_day_ahead"), "lock_mae_p90"].iloc[0])
    frozen_profile_p50 = float(
        summary.loc[summary["scenario"].eq("frozen_day_ahead"), "profile_shape_mae_p50"].iloc[0]
    )
    frozen_profile_p90 = float(
        summary.loc[summary["scenario"].eq("frozen_day_ahead"), "profile_shape_mae_p90"].iloc[0]
    )
    summary["lock_mae_gain_vs_frozen_p50"] = frozen_lock_p50 - summary["lock_mae_p50"]
    summary["lock_mae_gain_vs_frozen_p90"] = frozen_lock_p90 - summary["lock_mae_p90"]
    summary["profile_shape_mae_gain_vs_frozen_p50"] = frozen_profile_p50 - summary["profile_shape_mae_p50"]
    summary["profile_shape_mae_gain_vs_frozen_p90"] = frozen_profile_p90 - summary["profile_shape_mae_p90"]
    return summary


def _recommend_day_ahead_refresh(
    refresh_summary: pd.DataFrame,
    refresh_decisions: pd.DataFrame,
) -> dict[str, Any]:
    """Promote the triggered refresh policy only when it beats the frozen day-ahead baseline."""
    def _metric_or_fallback(row: pd.Series, metric_name: str) -> float:
        value = row.get(metric_name, row.get(metric_name.replace("_p50", "").replace("_p90", ""), float("nan")))
        return float(value)

    frozen = refresh_summary.loc[refresh_summary["scenario"].eq("frozen_day_ahead")].iloc[0]
    unconditional = refresh_summary.loc[refresh_summary["scenario"].eq("unconditional_refresh")].iloc[0]
    triggered = refresh_summary.loc[refresh_summary["scenario"].eq("triggered_refresh")].iloc[0]
    profile_improves = float(triggered["profile_shape_mae"]) < float(frozen["profile_shape_mae"])
    lock_improves = float(triggered["lock_mae"]) <= float(frozen["lock_mae"])
    profile_improves_p50 = _metric_or_fallback(triggered, "profile_shape_mae_p50") < _metric_or_fallback(
        frozen, "profile_shape_mae_p50"
    )
    profile_improves_p90 = _metric_or_fallback(triggered, "profile_shape_mae_p90") < _metric_or_fallback(
        frozen, "profile_shape_mae_p90"
    )
    lock_improves_p50 = _metric_or_fallback(triggered, "lock_mae_p50") <= _metric_or_fallback(
        frozen, "lock_mae_p50"
    )
    lock_improves_p90 = _metric_or_fallback(triggered, "lock_mae_p90") <= _metric_or_fallback(
        frozen, "lock_mae_p90"
    )
    trigger_rate = (
        float(refresh_decisions["refresh_triggered"].astype(bool).mean())
        if not refresh_decisions.empty
        else 0.0
    )
    min_trigger_rate = float(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_trigger_rate"])
    max_trigger_rate = float(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_max_trigger_rate"])
    trigger_rate_in_band = min_trigger_rate <= trigger_rate <= max_trigger_rate
    unconditional_profile_gain = float(frozen["profile_shape_mae"] - unconditional["profile_shape_mae"])
    triggered_profile_gain = float(frozen["profile_shape_mae"] - triggered["profile_shape_mae"])
    unconditional_lock_gain = float(frozen["lock_mae"] - unconditional["lock_mae"])
    triggered_lock_gain = float(frozen["lock_mae"] - triggered["lock_mae"])
    triggered_profile_gain_fraction_vs_unconditional = (
        float(triggered_profile_gain / unconditional_profile_gain)
        if unconditional_profile_gain > 0.0
        else float("nan")
    )
    triggered_lock_gain_fraction_vs_unconditional = (
        float(triggered_lock_gain / unconditional_lock_gain)
        if unconditional_lock_gain > 0.0
        else float("nan")
    )
    min_profile_fraction = float(
        MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_profile_gain_fraction_vs_unconditional"]
    )
    min_lock_fraction = float(
        MULTIRES_FORECAST_CONTROL["day_ahead_refresh_min_lock_gain_fraction_vs_unconditional"]
    )
    retains_profile_gain = (
        unconditional_profile_gain <= 0.0
        or (
            np.isfinite(triggered_profile_gain_fraction_vs_unconditional)
            and triggered_profile_gain_fraction_vs_unconditional >= min_profile_fraction
        )
    )
    retains_lock_gain = (
        unconditional_lock_gain <= 0.0
        or (
            np.isfinite(triggered_lock_gain_fraction_vs_unconditional)
            and triggered_lock_gain_fraction_vs_unconditional >= min_lock_fraction
        )
    )
    recommended_policy = "frozen_day_ahead"
    if (
        profile_improves
        and lock_improves
        and profile_improves_p50
        and profile_improves_p90
        and lock_improves_p50
        and lock_improves_p90
        and trigger_rate_in_band
        and retains_profile_gain
        and retains_lock_gain
    ):
        recommended_policy = "triggered_refresh"
    elif (
        float(unconditional["profile_shape_mae"]) < float(frozen["profile_shape_mae"])
        and float(unconditional["lock_mae"]) <= float(frozen["lock_mae"])
        and _metric_or_fallback(unconditional, "profile_shape_mae_p50") < _metric_or_fallback(
            frozen, "profile_shape_mae_p50"
        )
        and _metric_or_fallback(unconditional, "profile_shape_mae_p90") < _metric_or_fallback(
            frozen, "profile_shape_mae_p90"
        )
        and _metric_or_fallback(unconditional, "lock_mae_p50") <= _metric_or_fallback(frozen, "lock_mae_p50")
        and _metric_or_fallback(unconditional, "lock_mae_p90") <= _metric_or_fallback(frozen, "lock_mae_p90")
    ):
        recommended_policy = "unconditional_refresh"
    if recommended_policy == "triggered_refresh":
        reason = (
            "Triggered day-ahead refresh beat the frozen 24h path on profile-shape MAE "
            "without giving back locked-interval accuracy on the exact control cycles, and its "
            "refresh rate stayed inside the configured selectivity band."
        )
    elif recommended_policy == "unconditional_refresh":
        if not trigger_rate_in_band:
            reason = (
                "Refreshing the 24h path improved the exact-control profile, but the learned trigger "
                "was still effectively always-on or always-off, so the benchmark recommends treating "
                "this layer as an unconditional refresh until the trigger becomes selective."
            )
        else:
            reason = (
                "Refreshing the 24h path improved the exact-control profile, but the triggered policy "
                "still gave back too much of the unconditional refresh gain, so the benchmark keeps "
                "the refresh layer unconditional until the trigger becomes more selective without "
                "surrendering that benefit."
            )
    elif float(unconditional["profile_shape_mae"]) < float(frozen["profile_shape_mae"]):
        reason = (
            "Unconditional refresh improved the exact-control day-ahead profile, but the repo keeps "
            "the frozen default because the stricter triggered-refresh promotion rule was not met."
        )
    else:
        reason = (
            "Triggered day-ahead refresh did not beat the frozen 24h path on the exact control cycles, "
            "so the current default remains the frozen profile."
        )
    return {
        "recommended_policy": recommended_policy,
        "promotion_primary_metric": "profile_shape_mae",
        "promotion_guardrail_metric": "lock_mae",
        "triggered_beats_frozen_profile_shape": bool(profile_improves),
        "triggered_beats_frozen_lock": bool(lock_improves),
        "triggered_beats_frozen_profile_shape_p50": bool(profile_improves_p50),
        "triggered_beats_frozen_profile_shape_p90": bool(profile_improves_p90),
        "triggered_beats_frozen_lock_p50": bool(lock_improves_p50),
        "triggered_beats_frozen_lock_p90": bool(lock_improves_p90),
        "unconditional_beats_frozen_profile_shape": bool(
            float(unconditional["profile_shape_mae"]) < float(frozen["profile_shape_mae"])
        ),
        "unconditional_beats_frozen_lock": bool(
            float(unconditional["lock_mae"]) <= float(frozen["lock_mae"])
        ),
        "trigger_rate": trigger_rate,
        "trigger_rate_in_band": bool(trigger_rate_in_band),
        "triggered_profile_gain_fraction_vs_unconditional": float(
            triggered_profile_gain_fraction_vs_unconditional
        ),
        "triggered_lock_gain_fraction_vs_unconditional": float(
            triggered_lock_gain_fraction_vs_unconditional
        ),
        "retains_profile_gain_vs_unconditional": bool(retains_profile_gain),
        "retains_lock_gain_vs_unconditional": bool(retains_lock_gain),
        "reason": reason,
    }


def _threshold_quantile(series: pd.Series, quantile: float, fallback: float) -> float:
    """Return a stable quantile threshold with a configured fallback for sparse inputs."""
    valid = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return float(fallback)
    return float(valid.quantile(float(quantile)))


def _day_ahead_refresh_threshold_candidates(signal_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build the threshold grid evaluated on the calibration control cycles."""
    defaults = _default_day_ahead_refresh_thresholds()
    quantiles = [
        float(value) for value in MULTIRES_FORECAST_CONTROL["day_ahead_refresh_threshold_quantiles"]
    ]
    trigger_modes = [
        str(value).strip().lower()
        for value in MULTIRES_FORECAST_CONTROL["day_ahead_refresh_candidate_trigger_modes"]
    ]
    residual_values = signal_frame.get("residual_mae_pct", pd.Series(dtype=float))
    transition_values = (
        signal_frame.loc[
            signal_frame.get("transition_state_mismatch", pd.Series(dtype=bool)).astype(bool),
            "transition_residual_mae_pct",
        ]
        if not signal_frame.empty and "transition_state_mismatch" in signal_frame.columns
        else pd.Series(dtype=float)
    )
    if transition_values.empty:
        transition_values = residual_values
    activity_values = signal_frame.get("activity_ratio_shift", pd.Series(dtype=float))

    candidates: list[dict[str, Any]] = [
        {
            **defaults,
            "threshold_source": "configured_defaults",
            "residual_drift_quantile": float("nan"),
            "transition_quantile": float("nan"),
            "activity_ratio_shift_quantile": float("nan"),
        }
    ]
    for trigger_mode in trigger_modes:
        for residual_quantile in quantiles:
            residual_threshold = _threshold_quantile(
                residual_values,
                residual_quantile,
                float(defaults["residual_drift_mae_pct_threshold"]),
            )
            for transition_quantile in quantiles:
                transition_threshold = _threshold_quantile(
                    transition_values,
                    transition_quantile,
                    float(defaults["transition_mae_pct_threshold"]),
                )
                for activity_quantile in quantiles:
                    activity_threshold = _threshold_quantile(
                        activity_values,
                        activity_quantile,
                        float(defaults["activity_ratio_shift_threshold"]),
                    )
                    candidates.append(
                        {
                            "residual_drift_mae_pct_threshold": residual_threshold,
                            "transition_mae_pct_threshold": transition_threshold,
                            "activity_ratio_shift_threshold": activity_threshold,
                            "trigger_mode": str(trigger_mode),
                            "threshold_source": "calibration_quantile_grid",
                            "residual_drift_quantile": float(residual_quantile),
                            "transition_quantile": float(transition_quantile),
                            "activity_ratio_shift_quantile": float(activity_quantile),
                        }
                    )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, str]] = set()
    for candidate in candidates:
        signature = (
            round(float(candidate["residual_drift_mae_pct_threshold"]), 9),
            round(float(candidate["transition_mae_pct_threshold"]), 9),
            round(float(candidate["activity_ratio_shift_threshold"]), 9),
            str(candidate["trigger_mode"]),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(candidate)
    return deduped


def _build_day_ahead_refresh_scope_inputs(
    *,
    cycle_origins: list[pd.Timestamp],
    refresh_origins: list[pd.Timestamp],
    actual_minute_base: pd.DataFrame,
    minute_feature_frame: pd.DataFrame,
    day_ahead: dict[str, Any],
    day_ahead_refresh: dict[str, Any] | None,
    result_key: str = "result",
    day_ahead_horizon: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Build the frozen and unconditional refresh paths plus raw trigger signals for one scope."""
    if day_ahead_refresh is None or not cycle_origins:
        return [], pd.DataFrame()
    cycle_inputs: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    lock_interval_minutes = int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"])
    for cycle_origin in cycle_origins:
        minute_index = _minute_index_for_cycle(actual_minute_base, cycle_origin)
        actual_minute = (
            actual_minute_base.loc[
                actual_minute_base["timestamp"].isin(minute_index),
                ["timestamp", "avg_load"],
            ]
            .drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")
            .reindex(minute_index)
        )
        if actual_minute["avg_load"].isna().any():
            raise RuntimeError(f"Actual minute grid is incomplete for control cycle {cycle_origin.isoformat()}.")
        day_ahead_series = _extract_candidate_path(
            cast(dict[str, Any], day_ahead[str(result_key)])["detail_by_origin"],
            origin_timestamp=cycle_origin,
            candidate_label=str(day_ahead["candidate_label"]),
            minute_index=minute_index,
        )
        cycle_refresh_origins = [
            timestamp
            for timestamp in refresh_origins
            if cycle_origin < timestamp < cycle_origin + pd.Timedelta(minutes=day_ahead_horizon)
        ]
        unconditional_series = _apply_rollout_updates(
            day_ahead_series,
            detail_by_origin=cast(dict[str, Any], day_ahead_refresh[str(result_key)])["detail_by_origin"],
            candidate_label=str(day_ahead_refresh["candidate_label"]),
            update_origins=cycle_refresh_origins,
            horizon_minutes=day_ahead_horizon,
        )
        minute_frame = pd.DataFrame(
            {
                "cycle_origin_timestamp": pd.Timestamp(cycle_origin).isoformat(),
                "timestamp": minute_index,
                "actual_load": actual_minute["avg_load"].to_numpy(dtype=float),
                "day_ahead_pred": day_ahead_series.to_numpy(dtype=float),
                "unconditional_refresh_pred": unconditional_series.to_numpy(dtype=float),
            }
        )
        frozen_day_ahead_metrics = _scenario_cycle_metrics(
            minute_frame=minute_frame,
            prediction_column="day_ahead_pred",
            lock_interval_minutes=lock_interval_minutes,
        )
        unconditional_refresh_metrics = _scenario_cycle_metrics(
            minute_frame=minute_frame,
            prediction_column="unconditional_refresh_pred",
            lock_interval_minutes=lock_interval_minutes,
        )
        cycle_inputs.append(
            {
                "cycle_origin_timestamp": pd.Timestamp(cycle_origin),
                "cycle_origin_label": pd.Timestamp(cycle_origin).isoformat(),
                "cycle_refresh_origins": list(cycle_refresh_origins),
                "minute_frame": minute_frame,
                "frozen_day_ahead_metrics": dict(frozen_day_ahead_metrics),
                "unconditional_refresh_metrics": dict(unconditional_refresh_metrics),
            }
        )
        signal_rows.extend(
            [
                _day_ahead_refresh_signal_row(
                    cycle_origin_timestamp=pd.Timestamp(cycle_origin),
                    refresh_origin_timestamp=pd.Timestamp(refresh_origin),
                    minute_feature_frame=minute_feature_frame,
                    frozen_forecast=day_ahead_series,
                )
                for refresh_origin in cycle_refresh_origins
            ]
        )
    signal_frame = (
        pd.DataFrame(signal_rows)
        .sort_values(["cycle_origin_timestamp", "refresh_origin_timestamp"], kind="stable")
        .reset_index(drop=True)
        if signal_rows
        else pd.DataFrame()
    )
    return cycle_inputs, signal_frame


def _day_ahead_refresh_decisions(
    signal_frame: pd.DataFrame,
    *,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Apply one threshold tuple to the refresh signal frame and return the decision rows."""
    if signal_frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            _apply_day_ahead_refresh_thresholds(
                signal_row=dict(row._asdict()),
                thresholds=thresholds,
            )
            for row in signal_frame.itertuples(index=False)
        ]
    )


def _day_ahead_refresh_decision_signature(decision_rows: pd.DataFrame) -> tuple[tuple[str, str, int], ...]:
    """Encode one threshold policy by its exact ordered trigger decisions on the calibration frame."""
    if decision_rows.empty:
        return ()
    ordered = decision_rows.sort_values(
        ["cycle_origin_timestamp", "refresh_origin_timestamp"],
        kind="stable",
    )
    return tuple(
        (
            str(cycle_origin),
            str(refresh_origin),
            int(bool(refresh_triggered)),
        )
        for cycle_origin, refresh_origin, refresh_triggered in zip(
            ordered["cycle_origin_timestamp"].astype("string"),
            ordered["refresh_origin_timestamp"].astype("string"),
            ordered["refresh_triggered"].astype(bool),
            strict=False,
        )
    )


def _evaluate_day_ahead_refresh_policy_from_decisions(
    *,
    cycle_inputs: list[dict[str, Any]],
    decision_rows: pd.DataFrame,
    day_ahead_refresh: dict[str, Any],
    result_key: str = "result",
    day_ahead_horizon: int,
    lock_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one trigger policy after the refresh decisions have already been computed."""
    if not cycle_inputs:
        empty = pd.DataFrame()
        return empty, empty
    triggered_origins_by_cycle: dict[str, list[pd.Timestamp]] = {}
    if not decision_rows.empty:
        triggered_rows = decision_rows.loc[
            decision_rows["refresh_triggered"].astype(bool),
            ["cycle_origin_timestamp", "refresh_origin_timestamp"],
        ].copy()
        if not triggered_rows.empty:
            triggered_origins_by_cycle = {
                str(cycle_origin): [pd.Timestamp(value) for value in refresh_values.tolist()]
                for cycle_origin, refresh_values in triggered_rows.groupby(
                    "cycle_origin_timestamp",
                    sort=False,
                )["refresh_origin_timestamp"]
            }
    refresh_cycle_rows: list[dict[str, Any]] = []
    for cycle_input in cycle_inputs:
        cycle_origin = pd.Timestamp(cycle_input["cycle_origin_timestamp"])
        cycle_origin_label = str(cycle_input.get("cycle_origin_label", cycle_origin.isoformat()))
        minute_frame = cast(pd.DataFrame, cycle_input["minute_frame"]).copy()
        triggered_origins = list(triggered_origins_by_cycle.get(cycle_origin_label, []))
        triggered_series = _apply_rollout_updates(
            minute_frame.set_index("timestamp")["day_ahead_pred"].astype(float),
            detail_by_origin=cast(dict[str, Any], day_ahead_refresh[str(result_key)])["detail_by_origin"],
            candidate_label=str(day_ahead_refresh["candidate_label"]),
            update_origins=triggered_origins,
            horizon_minutes=day_ahead_horizon,
        )
        minute_frame["triggered_refresh_pred"] = (
            triggered_series.reindex(minute_frame["timestamp"]).to_numpy(dtype=float)
        )
        frozen_day_ahead_metrics = cast(
            dict[str, float] | None,
            cycle_input.get("frozen_day_ahead_metrics"),
        )
        if not frozen_day_ahead_metrics:
            frozen_day_ahead_metrics = _scenario_cycle_metrics(
                minute_frame=minute_frame,
                prediction_column="day_ahead_pred",
                lock_interval_minutes=lock_interval,
            )
        unconditional_refresh_metrics = cast(
            dict[str, float] | None,
            cycle_input.get("unconditional_refresh_metrics"),
        )
        if not unconditional_refresh_metrics:
            unconditional_refresh_metrics = _scenario_cycle_metrics(
                minute_frame=minute_frame,
                prediction_column="unconditional_refresh_pred",
                lock_interval_minutes=lock_interval,
            )
        triggered_refresh_metrics = _scenario_cycle_metrics(
            minute_frame=minute_frame,
            prediction_column="triggered_refresh_pred",
            lock_interval_minutes=lock_interval,
        )
        refresh_cycle_rows.append(
            {
                "cycle_origin_timestamp": cycle_origin_label,
                **{
                    f"frozen_day_ahead_{key}": value
                    for key, value in frozen_day_ahead_metrics.items()
                },
                **{
                    f"unconditional_refresh_{key}": value
                    for key, value in unconditional_refresh_metrics.items()
                },
                **{
                    f"triggered_refresh_{key}": value
                    for key, value in triggered_refresh_metrics.items()
                },
                "frozen_day_ahead_update_count": 0,
                "unconditional_refresh_update_count": int(len(cycle_input["cycle_refresh_origins"])),
                "triggered_refresh_update_count": int(len(triggered_origins)),
            }
        )
    refresh_by_cycle = pd.DataFrame(refresh_cycle_rows).sort_values(
        "cycle_origin_timestamp",
        kind="stable",
    ).reset_index(drop=True)
    refresh_summary = _day_ahead_refresh_summary_frame(refresh_by_cycle).reset_index(drop=True)
    return refresh_by_cycle, refresh_summary


def _evaluate_day_ahead_refresh_policy(
    *,
    cycle_inputs: list[dict[str, Any]],
    signal_frame: pd.DataFrame,
    thresholds: dict[str, float],
    day_ahead_refresh: dict[str, Any],
    result_key: str = "result",
    day_ahead_horizon: int,
    lock_interval: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one triggered refresh threshold policy on a specific control scope."""
    if not cycle_inputs:
        empty = pd.DataFrame()
        return empty, empty, empty
    decision_rows = _day_ahead_refresh_decisions(
        signal_frame,
        thresholds=thresholds,
    )
    refresh_by_cycle, refresh_summary = _evaluate_day_ahead_refresh_policy_from_decisions(
        cycle_inputs=cycle_inputs,
        decision_rows=decision_rows,
        day_ahead_refresh=day_ahead_refresh,
        result_key=result_key,
        day_ahead_horizon=day_ahead_horizon,
        lock_interval=lock_interval,
    )
    return decision_rows, refresh_by_cycle, refresh_summary


def _select_day_ahead_refresh_thresholds(
    *,
    calibration_cycle_inputs: list[dict[str, Any]],
    calibration_signal_frame: pd.DataFrame,
    day_ahead_refresh: dict[str, Any],
    result_key: str = "benchmark_result",
    day_ahead_horizon: int,
    lock_interval: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Search trigger thresholds on calibration cycles and return the chosen policy plus artifacts."""
    if not calibration_cycle_inputs or calibration_signal_frame.empty:
        defaults = _default_day_ahead_refresh_thresholds()
        empty = pd.DataFrame()
        return defaults, empty, empty, empty, empty
    grid_rows: list[dict[str, Any]] = []
    threshold_outputs: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    threshold_signature_cache: dict[
        tuple[tuple[str, str, int], ...],
        dict[str, Any],
    ] = {}
    threshold_grid = _day_ahead_refresh_threshold_candidates(calibration_signal_frame)
    for threshold_rank, thresholds in enumerate(threshold_grid, start=1):
        decisions = _day_ahead_refresh_decisions(
            calibration_signal_frame,
            thresholds=thresholds,
        )
        decision_signature = _day_ahead_refresh_decision_signature(decisions)
        cached_result = threshold_signature_cache.get(decision_signature)
        if cached_result is None:
            refresh_by_cycle, refresh_summary = _evaluate_day_ahead_refresh_policy_from_decisions(
                cycle_inputs=calibration_cycle_inputs,
                decision_rows=decisions,
                day_ahead_refresh=day_ahead_refresh,
                result_key=str(result_key),
                day_ahead_horizon=day_ahead_horizon,
                lock_interval=lock_interval,
            )
            recommendation = _recommend_day_ahead_refresh(refresh_summary, decisions)
            cached_result = {
                "threshold_rank": int(threshold_rank),
                "decisions": decisions,
                "refresh_by_cycle": refresh_by_cycle,
                "refresh_summary": refresh_summary,
                "recommendation": recommendation,
            }
            threshold_signature_cache[decision_signature] = cached_result
        else:
            refresh_by_cycle = cast(pd.DataFrame, cached_result["refresh_by_cycle"])
            refresh_summary = cast(pd.DataFrame, cached_result["refresh_summary"])
            recommendation = cast(dict[str, Any], cached_result["recommendation"])
        signature_origin_rank = int(cached_result["threshold_rank"])
        frozen = refresh_summary.loc[refresh_summary["scenario"].eq("frozen_day_ahead")].iloc[0]
        triggered = refresh_summary.loc[refresh_summary["scenario"].eq("triggered_refresh")].iloc[0]
        grid_rows.append(
            {
                "threshold_rank": int(threshold_rank),
                "decision_signature_origin_rank": int(signature_origin_rank),
                "decision_signature_duplicate": bool(signature_origin_rank != int(threshold_rank)),
                **thresholds,
                "calibration_cycle_n": int(len(refresh_by_cycle)),
                "frozen_lock_mae": float(frozen["lock_mae"]),
                "triggered_lock_mae": float(triggered["lock_mae"]),
                "frozen_profile_shape_mae": float(frozen["profile_shape_mae"]),
                "triggered_profile_shape_mae": float(triggered["profile_shape_mae"]),
                "triggered_lock_mae_gain_vs_frozen": float(frozen["lock_mae"] - triggered["lock_mae"]),
                "triggered_profile_shape_mae_gain_vs_frozen": float(
                    frozen["profile_shape_mae"] - triggered["profile_shape_mae"]
                ),
                "trigger_rate": float(recommendation["trigger_rate"]),
                "trigger_rate_in_band": bool(recommendation.get("trigger_rate_in_band", False)),
                "calibration_can_promote": bool(
                    recommendation["recommended_policy"] == "triggered_refresh"
                ),
                "calibration_retains_profile_gain_vs_unconditional": bool(
                    recommendation["retains_profile_gain_vs_unconditional"]
                ),
                "calibration_retains_lock_gain_vs_unconditional": bool(
                    recommendation["retains_lock_gain_vs_unconditional"]
                ),
                "calibration_triggered_profile_gain_fraction_vs_unconditional": float(
                    recommendation["triggered_profile_gain_fraction_vs_unconditional"]
                ),
                "calibration_triggered_lock_gain_fraction_vs_unconditional": float(
                    recommendation["triggered_lock_gain_fraction_vs_unconditional"]
                ),
                "calibration_recommended_policy": str(recommendation["recommended_policy"]),
                "calibration_reason": str(recommendation["reason"]),
            }
        )
        threshold_outputs.append(
            (
                cast(pd.DataFrame, cached_result["decisions"]),
                refresh_by_cycle,
                refresh_summary,
            )
        )
    grid = pd.DataFrame(grid_rows).sort_values(
        [
            "calibration_can_promote",
            "trigger_rate_in_band",
            "calibration_retains_profile_gain_vs_unconditional",
            "calibration_retains_lock_gain_vs_unconditional",
            "calibration_triggered_profile_gain_fraction_vs_unconditional",
            "calibration_triggered_lock_gain_fraction_vs_unconditional",
            "trigger_rate",
            "triggered_profile_shape_mae",
            "triggered_lock_mae",
            "trigger_mode",
            "residual_drift_mae_pct_threshold",
            "transition_mae_pct_threshold",
            "activity_ratio_shift_threshold",
            "threshold_rank",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    chosen = grid.iloc[0]
    chosen_thresholds = {
        "residual_drift_mae_pct_threshold": float(chosen["residual_drift_mae_pct_threshold"]),
        "transition_mae_pct_threshold": float(chosen["transition_mae_pct_threshold"]),
        "activity_ratio_shift_threshold": float(chosen["activity_ratio_shift_threshold"]),
        "trigger_mode": str(chosen["trigger_mode"]),
    }
    chosen_rank = int(chosen["threshold_rank"]) - 1
    chosen_decisions, chosen_refresh_by_cycle, chosen_refresh_summary = threshold_outputs[chosen_rank]
    return chosen_thresholds, grid, chosen_decisions, chosen_refresh_by_cycle, chosen_refresh_summary


def _nowcast_sort_columns(metric_name: str) -> list[str]:
    """Return the deterministic candidate ranking order for the exact-control nowcast layer."""
    if metric_name == "optimizer_score":
        return [
            "optimizer_score",
            "next_lock_mae",
            "peak_interval_miss_rate",
            "peak_value_mae",
            "lock_mae",
            "candidate_label",
        ]
    if metric_name == "lock_mae":
        return ["lock_mae", "next_lock_mae", "peak_interval_miss_rate", "peak_value_mae", "candidate_label"]
    if metric_name == "minute_path_mae":
        return ["minute_path_mae", "lock_mae", "profile_shape_mae", "candidate_label"]
    if metric_name == "next_lock_mae":
        return ["next_lock_mae", "peak_interval_miss_rate", "peak_value_mae", "lock_mae", "candidate_label"]
    if metric_name == "peak_value_mae":
        return ["peak_value_mae", "peak_interval_miss_rate", "next_lock_mae", "lock_mae", "candidate_label"]
    if metric_name == "peak_interval_miss_rate":
        return ["peak_interval_miss_rate", "next_lock_mae", "peak_value_mae", "lock_mae", "candidate_label"]
    if metric_name == "profile_shape_mae":
        return ["profile_shape_mae", "lock_mae", "minute_path_mae", "candidate_label"]
    if metric_name == "energy_mae":
        return ["energy_mae", "lock_mae", "minute_path_mae", "candidate_label"]
    raise ValueError(f"Unsupported nowcast benchmark metric: {metric_name}")


def _prefixed_nowcast_sort_columns(metric_name: str, prefix: str) -> list[str]:
    """Return the nowcast ranking order after applying one column-name prefix."""
    return [f"{prefix}{column}" if column != "candidate_label" else column for column in _nowcast_sort_columns(metric_name)]


def _apply_nowcast_advisory_tie_break(
    frame: pd.DataFrame,
    *,
    selection_metric: str,
    metric_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use broader Stage-5 advisory evidence only to break materially negligible nowcast ties."""
    default_meta = {
        "considered": False,
        "applied": False,
        "reason": "no_near_tie_candidates",
        "candidate_count": 0,
        "previous_top_candidate_label": "",
        "selected_candidate_label": "",
    }
    working = _attach_nowcast_advisory_evidence(frame)
    if working.empty or not bool(MULTIRES_FORECAST_CONTROL["nowcast_advisory_evidence_enabled"]):
        return working, default_meta
    tolerance = float(MULTIRES_FORECAST_CONTROL["nowcast_advisory_tie_tolerance"])
    if tolerance <= 0.0:
        return working, default_meta
    metric_column = f"{metric_prefix}selection_metric_value"
    if metric_column not in working.columns:
        return working, {**default_meta, "reason": "missing_metric_column"}
    ordered = working.sort_values(
        _prefixed_nowcast_sort_columns(selection_metric, metric_prefix),
        ascending=[True] * len(_prefixed_nowcast_sort_columns(selection_metric, metric_prefix)),
        kind="stable",
    ).reset_index(drop=True)
    metric_values = pd.to_numeric(ordered[metric_column], errors="coerce")
    if metric_values.empty or not np.isfinite(float(metric_values.iloc[0])):
        return ordered, {**default_meta, "reason": "missing_finite_metric_value"}
    best_metric_value = float(metric_values.iloc[0])
    tie_mask = metric_values.sub(best_metric_value).abs().le(tolerance)
    tie_count = int(tie_mask.sum())
    if tie_count < 2:
        return ordered, {**default_meta, "reason": "no_near_tie_candidates"}
    tied = ordered.loc[tie_mask].copy()
    if not tied["advisory_surface_supported"].astype(bool).any():
        return ordered, {
            **default_meta,
            "considered": True,
            "candidate_count": tie_count,
            "reason": "no_supported_advisory_candidates_in_tie_band",
            "previous_top_candidate_label": str(ordered.iloc[0]["candidate_label"]),
            "selected_candidate_label": str(ordered.iloc[0]["candidate_label"]),
        }
    tied["_advisory_supported_sort"] = (~tied["advisory_surface_supported"].astype(bool)).astype(int)
    tied["_advisory_supported_regime_rank"] = -pd.to_numeric(
        tied["advisory_supported_regime_count"],
        errors="coerce",
    ).fillna(0.0)
    for column_name in (
        "advisory_transition_best_ratio_to_persistence",
        "advisory_surface_candidate_mae_ratio_to_persistence",
        "advisory_high_ramp_ratio_to_persistence",
    ):
        tied[column_name] = pd.to_numeric(tied[column_name], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(float("inf"))
    previous_top_candidate_label = str(ordered.iloc[0]["candidate_label"])
    tied = tied.sort_values(
        [
            "_advisory_supported_sort",
            "_advisory_supported_regime_rank",
            "advisory_transition_best_ratio_to_persistence",
            "advisory_surface_candidate_mae_ratio_to_persistence",
            "advisory_high_ramp_ratio_to_persistence",
            *_prefixed_nowcast_sort_columns(selection_metric, metric_prefix),
        ],
        ascending=[True, True, True, True, True, *([True] * len(_prefixed_nowcast_sort_columns(selection_metric, metric_prefix)))],
        kind="stable",
    ).drop(columns=["_advisory_supported_sort", "_advisory_supported_regime_rank"])
    tied["advisory_tie_break_considered"] = True
    selected_candidate_label = str(tied.iloc[0]["candidate_label"])
    tie_break_applied = selected_candidate_label != previous_top_candidate_label
    tied["advisory_tie_break_applied"] = bool(tie_break_applied)
    tied["advisory_tie_break_reason"] = (
        "broader_stage5_transition_evidence"
        if tie_break_applied
        else "exact_control_winner_retained_after_advisory_review"
    )
    remainder = ordered.loc[~tie_mask].copy()
    remainder["advisory_tie_break_considered"] = False
    remainder["advisory_tie_break_applied"] = False
    remainder["advisory_tie_break_reason"] = "outside_tie_tolerance"
    ordered = pd.concat([tied, remainder], ignore_index=True)
    return ordered, {
        "considered": True,
        "applied": bool(tie_break_applied),
        "reason": str(tied.iloc[0]["advisory_tie_break_reason"]),
        "candidate_count": tie_count,
        "previous_top_candidate_label": previous_top_candidate_label,
        "selected_candidate_label": selected_candidate_label,
    }


def _ensure_nowcast_metric_columns(frame: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    """Backfill sparse nowcast benchmark rows so ranking and reporting stay stable."""
    working = _ensure_selection_metric_column(
        frame,
        selection_metric=selection_metric,
        selection_metric_value_column="selection_metric_value",
        fallback_metrics=[
            "next_lock_mae",
            "lock_mae",
            "minute_path_mae",
            "profile_shape_mae",
            "energy_mae",
        ],
    )
    numeric_defaults = {
        "minute_path_mae": float("nan"),
        "minute_path_mae_pct": float("nan"),
        "lock_mae": float("nan"),
        "lock_mae_pct": float("nan"),
        "next_lock_mae": float("nan"),
        "next_lock_mae_pct": float("nan"),
        "profile_shape_mae": float("nan"),
        "profile_shape_mae_pct": float("nan"),
        "energy_mae": float("nan"),
        "energy_mae_pct": float("nan"),
        "peak_value_mae": float("nan"),
        "peak_value_mae_pct": float("nan"),
        "peak_interval_hit_rate": float("nan"),
        "peak_interval_miss_rate": float("nan"),
        "peak_interval_offset_minutes": float("nan"),
        "optimizer_score": float("nan"),
        "selection_metric_value": float("nan"),
        "selection_metric_pct": float("nan"),
        "mean_coverage": float("nan"),
        "origin_n": 0,
    }
    for column, default_value in numeric_defaults.items():
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            continue
        if column == selection_metric:
            working[column] = pd.to_numeric(working[selection_metric], errors="coerce")
        elif column == "selection_metric_value":
            working[column] = pd.to_numeric(working[selection_metric], errors="coerce")
        elif column == "selection_metric_pct" and f"{selection_metric}_pct" in working.columns:
            working[column] = pd.to_numeric(working[f"{selection_metric}_pct"], errors="coerce")
        else:
            working[column] = default_value
    return working


def _score_nowcast_candidate_predictions(
    *,
    minute_timeline: pd.DataFrame,
    prediction_series: pd.Series,
    candidate_label: str,
    candidate_type: str,
    source_model_label: str,
    target_mode: str,
    selection_metric: str,
) -> dict[str, Any]:
    """Score one minute-level prediction series on a given control timeline."""
    lock_interval = int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"])
    candidate_minute = minute_timeline.copy()
    candidate_minute["nowcast_pred"] = _apply_nowcast_updates(
        pd.Series(
            candidate_minute["phase_pred"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(candidate_minute["timestamp"], errors="raise")),
            dtype=float,
        ),
        prediction_series,
    ).to_numpy(dtype=float)
    cycle_rows: list[dict[str, Any]] = []
    for cycle_origin, cycle_frame in candidate_minute.groupby("cycle_origin_timestamp", sort=True):
        cycle_rows.append(
            {
                "cycle_origin_timestamp": str(cycle_origin),
                **_layer_cycle_metrics(
                    minute_frame=cycle_frame,
                    layer_name="nowcast",
                    prediction_column="nowcast_pred",
                    lock_interval_minutes=lock_interval,
                ),
            }
        )
    cycle_frame = pd.DataFrame(cycle_rows)
    coverage = float(
        candidate_minute["timestamp"].map(prediction_series).notna().mean()
    )
    result = {
        "candidate_label": str(candidate_label),
        "candidate_type": str(candidate_type),
        "source_model_label": str(source_model_label),
        "target_mode": str(target_mode),
        "minute_path_mae": float(cycle_frame["nowcast_minute_path_mae"].mean()),
        "minute_path_mae_pct": float(cycle_frame["nowcast_minute_path_mae_pct"].mean()),
        "lock_mae": float(cycle_frame["nowcast_lock_mae"].mean()),
        "lock_mae_pct": float(cycle_frame["nowcast_lock_mae_pct"].mean()),
        "next_lock_mae": float(cycle_frame["nowcast_next_lock_mae"].mean()),
        "next_lock_mae_pct": float(cycle_frame["nowcast_next_lock_mae_pct"].mean()),
        "profile_shape_mae": float(cycle_frame["nowcast_profile_shape_mae"].mean()),
        "profile_shape_mae_pct": float(cycle_frame["nowcast_profile_shape_mae_pct"].mean()),
        "energy_mae": float(cycle_frame["nowcast_energy_mae"].mean()),
        "energy_mae_pct": float(cycle_frame["nowcast_energy_mae_pct"].mean()),
        "peak_value_mae": float(cycle_frame["nowcast_peak_value_mae"].mean()),
        "peak_value_mae_pct": float(cycle_frame["nowcast_peak_value_mae_pct"].mean()),
        "peak_interval_hit_rate": float(cycle_frame["nowcast_peak_interval_hit"].mean()),
        "peak_interval_miss_rate": float(1.0 - cycle_frame["nowcast_peak_interval_hit"].mean()),
        "peak_interval_offset_minutes": float(cycle_frame["nowcast_peak_interval_offset_minutes"].mean()),
        "optimizer_score": float(
            cycle_frame.apply(
                lambda row: _optimizer_score_from_components(
                    next_lock_mae_pct=float(row.get("nowcast_next_lock_mae_pct", float("nan"))),
                    lock_mae_pct=float(row.get("nowcast_lock_mae_pct", float("nan"))),
                    peak_value_mae_pct=float(row.get("nowcast_peak_value_mae_pct", float("nan"))),
                    peak_interval_miss_rate=(
                        1.0 - float(row.get("nowcast_peak_interval_hit", float("nan")))
                        if np.isfinite(float(row.get("nowcast_peak_interval_hit", float("nan"))))
                        else float("nan")
                    ),
                ),
                axis=1,
            ).mean()
        ),
        "mean_coverage": coverage,
        "origin_n": int(cycle_frame["cycle_origin_timestamp"].nunique()),
        "control_layer": "nowcast",
        "selection_metric_name": selection_metric,
    }
    result["selection_metric_value"] = float(result.get(selection_metric, float("nan")))
    metric_pct_key = f"{selection_metric}_pct"
    result["selection_metric_pct"] = (
        float(result.get(metric_pct_key, float("nan"))) if metric_pct_key in result else float("nan")
    )
    return result


def _blend_control_prediction_series(
    *,
    candidate_series: pd.Series,
    reference_series: pd.Series,
    candidate_weight: float,
) -> pd.Series:
    """Blend one control-layer candidate toward a reference policy on shared timestamps."""
    reference_aligned = reference_series.astype(float).copy()
    candidate_aligned = candidate_series.reindex(reference_aligned.index).astype(float)
    blended = reference_aligned.copy()
    valid_mask = candidate_aligned.notna()
    if bool(valid_mask.any()):
        blended.loc[valid_mask] = (
            reference_aligned.loc[valid_mask]
            + float(candidate_weight) * (candidate_aligned.loc[valid_mask] - reference_aligned.loc[valid_mask])
        )
    return blended.astype(float)


def _blend_control_prediction_series_by_bucket(
    *,
    candidate_series: pd.Series,
    reference_series: pd.Series,
    candidate_weight_by_bucket: dict[int, float],
    bucket_size_minutes: int,
    lock_interval_minutes: int,
) -> pd.Series:
    """Blend one control-layer candidate toward a reference policy by interval bucket."""
    reference_aligned = reference_series.astype(float).copy()
    candidate_aligned = candidate_series.reindex(reference_aligned.index).astype(float)
    blended = reference_aligned.copy()
    bucket_index = _index_minute_buckets(
        reference_aligned.index,
        bucket_minutes=int(bucket_size_minutes),
        cycle_minutes=int(lock_interval_minutes),
    )
    for bucket_key, candidate_weight in sorted(candidate_weight_by_bucket.items()):
        bucket_mask = bucket_index.eq(int(bucket_key))
        valid_mask = bucket_mask & candidate_aligned.notna()
        if not bool(valid_mask.any()):
            continue
        blended.loc[valid_mask] = (
            reference_aligned.loc[valid_mask]
            + float(candidate_weight)
            * (candidate_aligned.loc[valid_mask] - reference_aligned.loc[valid_mask])
        )
    return blended.astype(float)


def _blend_nowcast_prediction_series(
    *,
    candidate_series: pd.Series,
    persistence_series: pd.Series,
    candidate_weight: float,
) -> pd.Series:
    """Blend a learned minute candidate toward persistence on the timestamps it covers."""
    return _blend_control_prediction_series(
        candidate_series=candidate_series,
        reference_series=persistence_series,
        candidate_weight=float(candidate_weight),
    )


def _blend_nowcast_prediction_series_by_bucket(
    *,
    candidate_series: pd.Series,
    persistence_series: pd.Series,
    candidate_weight_by_bucket: dict[int, float],
    bucket_size_minutes: int,
    lock_interval_minutes: int,
) -> pd.Series:
    """Blend a minute candidate toward persistence with weights that vary inside the billing interval."""
    return _blend_control_prediction_series_by_bucket(
        candidate_series=candidate_series,
        reference_series=persistence_series,
        candidate_weight_by_bucket=candidate_weight_by_bucket,
        bucket_size_minutes=int(bucket_size_minutes),
        lock_interval_minutes=int(lock_interval_minutes),
    )


def _calibrate_nowcast_control_blend(
    *,
    candidate: dict[str, Any],
    benchmark_minute_timeline: pd.DataFrame,
    evaluation_minute_timeline: pd.DataFrame,
    benchmark_candidate_series: pd.Series,
    evaluation_candidate_series: pd.Series,
    benchmark_persistence_series: pd.Series,
    evaluation_persistence_series: pd.Series,
    selection_metric: str,
) -> dict[str, Any] | None:
    """Calibrate the best persistence-blend weight for one learned minute candidate."""
    if not bool(MULTIRES_FORECAST_CONTROL["nowcast_control_blend_enabled"]):
        return None
    candidate_label = str(candidate["candidate_label"])
    if candidate_label == "persistence" or str(candidate["candidate_type"]) != "learned":
        return None
    candidate_weights = [
        float(value)
        for value in MULTIRES_FORECAST_CONTROL["nowcast_control_blend_weights"]
        if 0.0 < float(value) < 1.0
    ]
    if not candidate_weights:
        return None
    benchmark_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_series_map: dict[float, pd.Series] = {}
    for candidate_weight in candidate_weights:
        blended_benchmark = _blend_nowcast_prediction_series(
            candidate_series=benchmark_candidate_series,
            persistence_series=benchmark_persistence_series,
            candidate_weight=candidate_weight,
        )
        blended_evaluation = _blend_nowcast_prediction_series(
            candidate_series=evaluation_candidate_series,
            persistence_series=evaluation_persistence_series,
            candidate_weight=candidate_weight,
        )
        label = f"{candidate_label}|control_blend_w{candidate_weight:.2f}"
        benchmark_rows.append(
            {
                **_score_nowcast_candidate_predictions(
                    minute_timeline=benchmark_minute_timeline,
                    prediction_series=blended_benchmark,
                    candidate_label=label,
                    candidate_type=str(candidate["candidate_type"]),
                    source_model_label=str(candidate["model_label"]),
                    target_mode=f"{candidate['target_mode']}|control_blend",
                    selection_metric=selection_metric,
                ),
                "control_blend_weight": float(candidate_weight),
                "blend_base_candidate_label": candidate_label,
            }
        )
        evaluation_rows.append(
            {
                **_score_nowcast_candidate_predictions(
                    minute_timeline=evaluation_minute_timeline,
                    prediction_series=blended_evaluation,
                    candidate_label=label,
                    candidate_type=str(candidate["candidate_type"]),
                    source_model_label=str(candidate["model_label"]),
                    target_mode=f"{candidate['target_mode']}|control_blend",
                    selection_metric=selection_metric,
                ),
                "control_blend_weight": float(candidate_weight),
                "blend_base_candidate_label": candidate_label,
            }
        )
        evaluation_series_map[float(candidate_weight)] = blended_evaluation
    benchmark = _ensure_nowcast_metric_columns(pd.DataFrame(benchmark_rows), selection_metric=selection_metric)
    benchmark = _ensure_sort_columns(
        benchmark,
        sort_columns=_nowcast_sort_columns(selection_metric),
        selection_metric=selection_metric,
        selection_metric_value_column="selection_metric_value",
    ).sort_values(
        _nowcast_sort_columns(selection_metric),
        ascending=[True] * len(_nowcast_sort_columns(selection_metric)),
        kind="stable",
    ).reset_index(drop=True)
    benchmark = _attach_nowcast_advisory_evidence(benchmark)
    evaluation = _ensure_nowcast_metric_columns(pd.DataFrame(evaluation_rows), selection_metric=selection_metric)
    evaluation = _ensure_sort_columns(
        evaluation,
        sort_columns=_nowcast_sort_columns(selection_metric),
        selection_metric=selection_metric,
        selection_metric_value_column="selection_metric_value",
    ).sort_values(
        _nowcast_sort_columns(selection_metric),
        ascending=[True] * len(_nowcast_sort_columns(selection_metric)),
        kind="stable",
    ).reset_index(drop=True)
    evaluation = _attach_nowcast_advisory_evidence(evaluation)
    if benchmark.empty:
        return None
    best = benchmark.iloc[0]
    best_weight = float(best["control_blend_weight"])
    return {
        "candidate_label": str(best["candidate_label"]),
        "candidate_type": str(best["candidate_type"]),
        "source_model_label": str(best["source_model_label"]),
        "target_mode": str(best["target_mode"]),
        "benchmark_row": best.to_dict(),
        "evaluation_row": evaluation.loc[
            evaluation["candidate_label"].astype("string").eq(str(best["candidate_label"]))
        ].iloc[0].to_dict(),
        "benchmark_prediction_series": _blend_nowcast_prediction_series(
            candidate_series=benchmark_candidate_series,
            persistence_series=benchmark_persistence_series,
            candidate_weight=best_weight,
        ),
        "evaluation_prediction_series": evaluation_series_map[best_weight],
        "control_blend_weight": best_weight,
        "blend_base_candidate_label": candidate_label,
    }


def _calibrate_nowcast_control_bucket_blend(
    *,
    candidate: dict[str, Any],
    benchmark_minute_timeline: pd.DataFrame,
    evaluation_minute_timeline: pd.DataFrame,
    benchmark_candidate_series: pd.Series,
    evaluation_candidate_series: pd.Series,
    benchmark_persistence_series: pd.Series,
    evaluation_persistence_series: pd.Series,
    selection_metric: str,
) -> dict[str, Any] | None:
    """Greedily calibrate minute-bucket blend weights for one learned candidate."""
    if not bool(MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_enabled"]):
        return None
    candidate_label = str(candidate["candidate_label"])
    if candidate_label == "persistence" or str(candidate["candidate_type"]) != "learned":
        return None
    bucket_size_minutes = int(MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_size_minutes"])
    lock_interval_minutes = int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"])
    candidate_weights = [
        float(value)
        for value in MULTIRES_FORECAST_CONTROL["nowcast_control_bucket_blend_weights"]
        if 0.0 <= float(value) <= 1.0
    ]
    if not candidate_weights:
        return None
    benchmark_index = pd.DatetimeIndex(pd.to_datetime(benchmark_minute_timeline["timestamp"], errors="raise"))
    candidate_buckets = sorted(
        {
            _timestamp_minute_bucket(
                timestamp=value,
                bucket_minutes=int(bucket_size_minutes),
                cycle_minutes=int(lock_interval_minutes),
            )
            for value in benchmark_index
        }
    )
    if not candidate_buckets:
        return None

    selected_weights: dict[int, float] = {}
    for bucket_key in candidate_buckets:
        best_weight = 0.0
        best_metric = float("inf")
        best_tie_break = float("inf")
        for candidate_weight in candidate_weights:
            working_weights = {**selected_weights, int(bucket_key): float(candidate_weight)}
            blended_benchmark = _blend_nowcast_prediction_series_by_bucket(
                candidate_series=benchmark_candidate_series,
                persistence_series=benchmark_persistence_series,
                candidate_weight_by_bucket=working_weights,
                bucket_size_minutes=int(bucket_size_minutes),
                lock_interval_minutes=int(lock_interval_minutes),
            )
            score_row = _score_nowcast_candidate_predictions(
                minute_timeline=benchmark_minute_timeline,
                prediction_series=blended_benchmark,
                candidate_label=f"{candidate_label}|control_bucket_blend_b{bucket_size_minutes}",
                candidate_type=str(candidate["candidate_type"]),
                source_model_label=str(candidate["model_label"]),
                target_mode=f"{candidate['target_mode']}|control_bucket_blend",
                selection_metric=selection_metric,
            )
            metric_value = float(score_row["selection_metric_value"])
            tie_break = abs(float(candidate_weight))
            if (
                metric_value < best_metric
                or (math.isclose(metric_value, best_metric, rel_tol=0.0, abs_tol=1e-12) and tie_break < best_tie_break)
            ):
                best_metric = metric_value
                best_tie_break = tie_break
                best_weight = float(candidate_weight)
        selected_weights[int(bucket_key)] = float(best_weight)

    label = f"{candidate_label}|control_bucket_blend_b{bucket_size_minutes}"
    benchmark_prediction = _blend_nowcast_prediction_series_by_bucket(
        candidate_series=benchmark_candidate_series,
        persistence_series=benchmark_persistence_series,
        candidate_weight_by_bucket=selected_weights,
        bucket_size_minutes=int(bucket_size_minutes),
        lock_interval_minutes=int(lock_interval_minutes),
    )
    evaluation_prediction = _blend_nowcast_prediction_series_by_bucket(
        candidate_series=evaluation_candidate_series,
        persistence_series=evaluation_persistence_series,
        candidate_weight_by_bucket=selected_weights,
        bucket_size_minutes=int(bucket_size_minutes),
        lock_interval_minutes=int(lock_interval_minutes),
    )
    benchmark_row = _score_nowcast_candidate_predictions(
        minute_timeline=benchmark_minute_timeline,
        prediction_series=benchmark_prediction,
        candidate_label=label,
        candidate_type=str(candidate["candidate_type"]),
        source_model_label=str(candidate["model_label"]),
        target_mode=f"{candidate['target_mode']}|control_bucket_blend",
        selection_metric=selection_metric,
    )
    evaluation_row = _score_nowcast_candidate_predictions(
        minute_timeline=evaluation_minute_timeline,
        prediction_series=evaluation_prediction,
        candidate_label=label,
        candidate_type=str(candidate["candidate_type"]),
        source_model_label=str(candidate["model_label"]),
        target_mode=f"{candidate['target_mode']}|control_bucket_blend",
        selection_metric=selection_metric,
    )
    return {
        "candidate_label": str(label),
        "candidate_type": str(benchmark_row["candidate_type"]),
        "source_model_label": str(benchmark_row["source_model_label"]),
        "target_mode": str(benchmark_row["target_mode"]),
        "benchmark_row": {
            **benchmark_row,
            "control_bucket_size_minutes": int(bucket_size_minutes),
            "control_bucket_weights_json": json.dumps(selected_weights, sort_keys=True),
            "blend_base_candidate_label": candidate_label,
        },
        "evaluation_row": {
            **evaluation_row,
            "control_bucket_size_minutes": int(bucket_size_minutes),
            "control_bucket_weights_json": json.dumps(selected_weights, sort_keys=True),
            "blend_base_candidate_label": candidate_label,
        },
        "evaluation_prediction_series": evaluation_prediction,
        "benchmark_prediction_series": benchmark_prediction,
        "control_bucket_size_minutes": int(bucket_size_minutes),
        "control_bucket_weights_json": json.dumps(selected_weights, sort_keys=True),
        "blend_base_candidate_label": candidate_label,
    }


def _benchmark_nowcast_layer(
    benchmark_minute_timeline: pd.DataFrame,
    evaluation_minute_timeline: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Benchmark and calibrate Stage-5 minute candidates on calibration then held-out control cycles."""
    upstream_anchor = _load_stage5_nowcast_anchor()
    candidate_pool = _load_stage5_nowcast_candidate_pool(upstream_anchor=upstream_anchor)
    same_scope = evaluation_minute_timeline is None
    evaluation_minute_timeline = (
        benchmark_minute_timeline.copy()
        if evaluation_minute_timeline is None
        else evaluation_minute_timeline.copy()
    )
    selection_metric = str(MULTIRES_FORECAST_CONTROL["nowcast_selection_metric"])
    contexts = _stage5_nowcast_contexts()
    benchmark_context = contexts["evaluation"] if same_scope else contexts["calibration"]
    evaluation_context = contexts["evaluation"]

    base_prediction_maps: dict[str, dict[str, pd.Series]] = {"benchmark": {}, "evaluation": {}}
    available_candidates: list[dict[str, Any]] = []
    for candidate in candidate_pool:
        benchmark_frame = _stage5_candidate_predictions(candidate=candidate, context=benchmark_context)
        evaluation_frame = _stage5_candidate_predictions(candidate=candidate, context=evaluation_context)
        if benchmark_frame.empty or evaluation_frame.empty:
            continue
        base_prediction_maps["benchmark"][str(candidate["candidate_label"])] = (
            benchmark_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
        base_prediction_maps["evaluation"][str(candidate["candidate_label"])] = (
            evaluation_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
        available_candidates.append(candidate)

    if not available_candidates:
        raise RuntimeError("No Stage-5 nowcast candidates produced exact-control predictions.")

    benchmark_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    evaluation_prediction_maps: dict[str, pd.Series] = {}
    persistence_benchmark = base_prediction_maps["benchmark"].get("persistence")
    persistence_evaluation = base_prediction_maps["evaluation"].get("persistence")
    for pool_rank, candidate in enumerate(available_candidates, start=1):
        candidate_label = str(candidate["candidate_label"])
        benchmark_row = _score_nowcast_candidate_predictions(
            minute_timeline=benchmark_minute_timeline,
            prediction_series=base_prediction_maps["benchmark"][candidate_label],
            candidate_label=candidate_label,
            candidate_type=str(candidate["candidate_type"]),
            source_model_label=str(candidate["model_label"]),
            target_mode=str(candidate["target_mode"]),
            selection_metric=selection_metric,
        )
        evaluation_row = _score_nowcast_candidate_predictions(
            minute_timeline=evaluation_minute_timeline,
            prediction_series=base_prediction_maps["evaluation"][candidate_label],
            candidate_label=candidate_label,
            candidate_type=str(candidate["candidate_type"]),
            source_model_label=str(candidate["model_label"]),
            target_mode=str(candidate["target_mode"]),
            selection_metric=selection_metric,
        )
        benchmark_rows.append(
            {
                **benchmark_row,
                "replay_pool_rank": int(pool_rank),
                "replay_pool_source_type": str(candidate["pool_source_type"]),
                "replay_pool_source_run_id": str(candidate["pool_source_run_id"]),
                "replay_resolution": str(candidate["resolution"]),
                "replay_feature_set": str(candidate["feature_set"]),
                "replay_model_label": str(candidate["model_label"]),
                "replay_run_dir": str(candidate["artifact_path"]),
                "origin_split_scope": "calibration",
                "control_blend_weight": float("nan"),
                "blend_base_candidate_label": "",
            }
        )
        evaluation_rows.append(
            {
                **evaluation_row,
                "origin_split_scope": "evaluation",
                "control_blend_weight": float("nan"),
                "blend_base_candidate_label": "",
                "control_bucket_size_minutes": float("nan"),
                "control_bucket_weights_json": "",
            }
        )
        evaluation_prediction_maps[candidate_label] = base_prediction_maps["evaluation"][candidate_label]

        if (
            persistence_benchmark is None
            or persistence_evaluation is None
            or candidate_label == "persistence"
            or str(candidate["candidate_type"]) != "learned"
        ):
            continue
        calibrated_blend = _calibrate_nowcast_control_blend(
            candidate=candidate,
            benchmark_minute_timeline=benchmark_minute_timeline,
            evaluation_minute_timeline=evaluation_minute_timeline,
            benchmark_candidate_series=base_prediction_maps["benchmark"][candidate_label],
            evaluation_candidate_series=base_prediction_maps["evaluation"][candidate_label],
            benchmark_persistence_series=persistence_benchmark,
            evaluation_persistence_series=persistence_evaluation,
            selection_metric=selection_metric,
        )
        if calibrated_blend is None:
            continue
        benchmark_rows.append(
            {
                **calibrated_blend["benchmark_row"],
                "replay_pool_rank": int(pool_rank),
                "replay_pool_source_type": str(candidate["pool_source_type"]),
                "replay_pool_source_run_id": str(candidate["pool_source_run_id"]),
                "replay_resolution": str(candidate["resolution"]),
                "replay_feature_set": str(candidate["feature_set"]),
                "replay_model_label": str(candidate["model_label"]),
                "replay_run_dir": str(candidate["artifact_path"]),
                "origin_split_scope": "calibration",
            }
        )
        evaluation_rows.append(
            {
                **calibrated_blend["evaluation_row"],
                "origin_split_scope": "evaluation",
            }
        )
        evaluation_prediction_maps[str(calibrated_blend["candidate_label"])] = cast(
            pd.Series,
            calibrated_blend["evaluation_prediction_series"],
        )
        benchmark_prediction_maps = base_prediction_maps.setdefault("benchmark", {})
        benchmark_prediction_maps[str(calibrated_blend["candidate_label"])] = cast(
            pd.Series,
            calibrated_blend["benchmark_prediction_series"],
        )

        calibrated_bucket_blend = _calibrate_nowcast_control_bucket_blend(
            candidate=candidate,
            benchmark_minute_timeline=benchmark_minute_timeline,
            evaluation_minute_timeline=evaluation_minute_timeline,
            benchmark_candidate_series=base_prediction_maps["benchmark"][candidate_label],
            evaluation_candidate_series=base_prediction_maps["evaluation"][candidate_label],
            benchmark_persistence_series=persistence_benchmark,
            evaluation_persistence_series=persistence_evaluation,
            selection_metric=selection_metric,
        )
        if calibrated_bucket_blend is None:
            continue
        benchmark_rows.append(
            {
                **calibrated_bucket_blend["benchmark_row"],
                "replay_pool_rank": int(pool_rank),
                "replay_pool_source_type": str(candidate["pool_source_type"]),
                "replay_pool_source_run_id": str(candidate["pool_source_run_id"]),
                "replay_resolution": str(candidate["resolution"]),
                "replay_feature_set": str(candidate["feature_set"]),
                "replay_model_label": str(candidate["model_label"]),
                "replay_run_dir": str(candidate["artifact_path"]),
                "origin_split_scope": "calibration",
                "control_blend_weight": float("nan"),
            }
        )
        evaluation_rows.append(
            {
                **calibrated_bucket_blend["evaluation_row"],
                "origin_split_scope": "evaluation",
                "control_blend_weight": float("nan"),
            }
        )
        evaluation_prediction_maps[str(calibrated_bucket_blend["candidate_label"])] = cast(
            pd.Series,
            calibrated_bucket_blend["evaluation_prediction_series"],
        )
        benchmark_prediction_maps[str(calibrated_bucket_blend["candidate_label"])] = cast(
            pd.Series,
            calibrated_bucket_blend["benchmark_prediction_series"],
        )

    benchmark = _ensure_nowcast_metric_columns(pd.DataFrame(benchmark_rows), selection_metric=selection_metric)
    benchmark = _ensure_sort_columns(
        benchmark,
        sort_columns=_nowcast_sort_columns(selection_metric),
        selection_metric=selection_metric,
        selection_metric_value_column="selection_metric_value",
    ).sort_values(
        _nowcast_sort_columns(selection_metric),
        ascending=[True] * len(_nowcast_sort_columns(selection_metric)),
        kind="stable",
    ).reset_index(drop=True)
    evaluation = _ensure_nowcast_metric_columns(pd.DataFrame(evaluation_rows), selection_metric=selection_metric)
    evaluation = _ensure_sort_columns(
        evaluation,
        sort_columns=_nowcast_sort_columns(selection_metric),
        selection_metric=selection_metric,
        selection_metric_value_column="selection_metric_value",
    ).sort_values(
        _nowcast_sort_columns(selection_metric),
        ascending=[True] * len(_nowcast_sort_columns(selection_metric)),
        kind="stable",
    ).reset_index(drop=True)
    evaluation_meta = evaluation.loc[
        :,
        [
            "candidate_label",
            "minute_path_mae",
            "minute_path_mae_pct",
            "lock_mae",
            "lock_mae_pct",
            "next_lock_mae",
            "next_lock_mae_pct",
            "profile_shape_mae",
            "profile_shape_mae_pct",
            "energy_mae",
            "energy_mae_pct",
            "peak_value_mae",
            "peak_value_mae_pct",
            "peak_interval_hit_rate",
            "peak_interval_miss_rate",
            "peak_interval_offset_minutes",
            "optimizer_score",
            "selection_metric_value",
            "selection_metric_pct",
            "origin_n",
            "mean_coverage",
        ],
    ].rename(
        columns={
            "minute_path_mae": "evaluation_minute_path_mae",
            "minute_path_mae_pct": "evaluation_minute_path_mae_pct",
            "lock_mae": "evaluation_lock_mae",
            "lock_mae_pct": "evaluation_lock_mae_pct",
            "next_lock_mae": "evaluation_next_lock_mae",
            "next_lock_mae_pct": "evaluation_next_lock_mae_pct",
            "profile_shape_mae": "evaluation_profile_shape_mae",
            "profile_shape_mae_pct": "evaluation_profile_shape_mae_pct",
            "energy_mae": "evaluation_energy_mae",
            "energy_mae_pct": "evaluation_energy_mae_pct",
            "peak_value_mae": "evaluation_peak_value_mae",
            "peak_value_mae_pct": "evaluation_peak_value_mae_pct",
            "peak_interval_hit_rate": "evaluation_peak_interval_hit_rate",
            "peak_interval_miss_rate": "evaluation_peak_interval_miss_rate",
            "peak_interval_offset_minutes": "evaluation_peak_interval_offset_minutes",
            "optimizer_score": "evaluation_optimizer_score",
            "selection_metric_value": "evaluation_selection_metric_value",
            "selection_metric_pct": "evaluation_selection_metric_pct",
            "origin_n": "evaluation_origin_n",
            "mean_coverage": "evaluation_mean_coverage",
        }
    )
    benchmark = benchmark.merge(evaluation_meta, on="candidate_label", how="left")
    benchmark = _attach_nowcast_advisory_evidence(benchmark)
    selection_scope = str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"])
    selection_mode = "control_layer_candidate_benchmark"
    selected_row = benchmark.iloc[0]
    advisory_tie_break_meta = {
        "considered": False,
        "applied": False,
        "reason": "not_used",
        "candidate_count": 0,
        "previous_top_candidate_label": "",
        "selected_candidate_label": "",
    }
    if (
        selection_scope == "held_out_evaluation"
        and bool(MULTIRES_FORECAST_CONTROL["optimize_replayed_candidates"])
        and not benchmark.empty
    ):
        benchmark_row_n = len(benchmark)
        guarded_benchmark = _apply_optimizer_promotion_guard(
            benchmark,
            upstream_label=str(upstream_anchor["candidate_label"]),
            metric_prefix="evaluation_",
        )
        benchmark = _ensure_sort_columns(
            guarded_benchmark,
            sort_columns=_prefixed_nowcast_sort_columns(selection_metric, "evaluation_"),
            selection_metric=f"evaluation_{selection_metric}",
            selection_metric_value_column="evaluation_selection_metric_value",
        )
        benchmark, advisory_tie_break_meta = _apply_nowcast_advisory_tie_break(
            benchmark,
            selection_metric=selection_metric,
            metric_prefix="evaluation_",
        )
        selected_row = benchmark.iloc[0]
        selection_mode = "held_out_control_layer_candidate_benchmark"
        if len(guarded_benchmark) != benchmark_row_n:
            selection_mode = "held_out_control_layer_candidate_benchmark_guarded"
        if bool(advisory_tie_break_meta["applied"]):
            selection_mode = f"{selection_mode}_advisory_tiebreak"
    elif not bool(MULTIRES_FORECAST_CONTROL["optimize_replayed_candidates"]):
        selection_mode = "stage5_holdout_anchor"
        matched = benchmark.loc[
            benchmark["candidate_label"].astype("string").eq(str(upstream_anchor["candidate_label"]))
        ].copy()
        if not matched.empty:
            selected_row = matched.iloc[0]
    else:
        guarded_benchmark = _apply_optimizer_promotion_guard(
            benchmark,
            upstream_label=str(upstream_anchor["candidate_label"]),
            metric_prefix="evaluation_",
        )
        if len(guarded_benchmark) != len(benchmark):
            selection_mode = "control_layer_candidate_benchmark_guarded"
        selected_row = guarded_benchmark.iloc[0]
    selected_label = str(selected_row["candidate_label"])
    return {
        "candidate_label": selected_label,
        "candidate_type": str(selected_row["candidate_type"]),
        "source_model_label": str(selected_row["source_model_label"]),
        "target_mode": str(selected_row["target_mode"]),
        "control_blend_weight": float(selected_row.get("control_blend_weight", float("nan"))),
        "control_bucket_size_minutes": float(
            selected_row.get("control_bucket_size_minutes", float("nan"))
        ),
        "control_bucket_weights_json": str(selected_row.get("control_bucket_weights_json", "")),
        "blend_base_candidate_label": str(selected_row.get("blend_base_candidate_label", "")),
        "control_selection_metric": str(selected_row["selection_metric_name"]),
        "control_selection_metric_value": float(selected_row["selection_metric_value"]),
        "control_selection_metric_pct": float(selected_row["selection_metric_pct"]),
        "evaluation_selection_metric_value": float(
            selected_row.get("evaluation_selection_metric_value", float("nan"))
        ),
        "evaluation_selection_metric_pct": float(
            selected_row.get("evaluation_selection_metric_pct", float("nan"))
        ),
        "control_selection_mode": selection_mode,
        "upstream_candidate_label": str(upstream_anchor["candidate_label"]),
        "candidate_pool_count": int(len(benchmark)),
        "benchmark_origin_mode": "full_control_scope",
        "benchmark_origin_count": int(benchmark_minute_timeline["cycle_origin_timestamp"].nunique()),
        "evaluation_origin_count": int(evaluation_minute_timeline["cycle_origin_timestamp"].nunique()),
        "selection_artifact": str(selected_row["replay_run_dir"]),
        "minute_path_mae": float(selected_row.get("evaluation_minute_path_mae", selected_row["minute_path_mae"])),
        "minute_path_mae_pct": float(
            selected_row.get("evaluation_minute_path_mae_pct", selected_row["minute_path_mae_pct"])
        ),
        "advisory_base_candidate_label": str(selected_row.get("advisory_base_candidate_label", "")),
        "advisory_surface_supported": bool(selected_row.get("advisory_surface_supported", False)),
        "advisory_supported_regime_count": int(selected_row.get("advisory_supported_regime_count", 0)),
        "advisory_supported_operating_regimes": [
            value for value in str(selected_row.get("advisory_supported_operating_regimes", "")).split(",") if value
        ],
        "advisory_surface_candidate_mae_ratio_to_persistence": float(
            selected_row.get("advisory_surface_candidate_mae_ratio_to_persistence", float("nan"))
        ),
        "advisory_transition_best_ratio_to_persistence": float(
            selected_row.get("advisory_transition_best_ratio_to_persistence", float("nan"))
        ),
        "advisory_high_ramp_ratio_to_persistence": float(
            selected_row.get("advisory_high_ramp_ratio_to_persistence", float("nan"))
        ),
        "advisory_tie_break_considered": bool(advisory_tie_break_meta.get("considered", False)),
        "advisory_tie_break_applied": bool(advisory_tie_break_meta.get("applied", False)),
        "advisory_tie_break_candidate_count": int(advisory_tie_break_meta.get("candidate_count", 0)),
        "advisory_tie_break_previous_top_candidate_label": str(
            advisory_tie_break_meta.get("previous_top_candidate_label", "")
        ),
        "advisory_tie_break_reason": str(advisory_tie_break_meta.get("reason", "")),
        "benchmark_prediction_series": base_prediction_maps["benchmark"].get(selected_label, pd.Series(dtype=float)),
        "prediction_series": evaluation_prediction_maps[selected_label],
        "candidate_benchmarks": benchmark,
        "upstream_anchor": upstream_anchor,
    }


def _selected_nowcast_prediction_series(
    *,
    nowcast_anchor: dict[str, Any],
    scope_name: str,
) -> pd.Series:
    """Replay the selected Stage-10 minute policy on an arbitrary control scope."""
    contexts = _stage5_nowcast_contexts()
    context_key = "calibration" if str(scope_name) == "calibration" else "evaluation"
    context = contexts[context_key]
    upstream_anchor = cast(dict[str, Any], nowcast_anchor["upstream_anchor"])
    candidate_pool = _load_stage5_nowcast_candidate_pool(upstream_anchor=upstream_anchor)
    pool_by_label = {str(candidate["candidate_label"]): dict(candidate) for candidate in candidate_pool}
    selected_label = str(nowcast_anchor["candidate_label"])
    base_label = str(nowcast_anchor.get("blend_base_candidate_label", ""))
    control_blend_weight = float(nowcast_anchor.get("control_blend_weight", float("nan")))
    control_bucket_size_minutes = int(
        nowcast_anchor.get("control_bucket_size_minutes", float("nan"))
    ) if pd.notna(nowcast_anchor.get("control_bucket_size_minutes")) else 0
    control_bucket_weights_json = str(nowcast_anchor.get("control_bucket_weights_json", ""))
    if selected_label in pool_by_label:
        prediction_frame = _stage5_candidate_predictions(candidate=pool_by_label[selected_label], context=context)
        if prediction_frame.empty:
            return pd.Series(dtype=float)
        return (
            prediction_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
    if base_label and base_label in pool_by_label and np.isfinite(control_blend_weight):
        base_frame = _stage5_candidate_predictions(candidate=pool_by_label[base_label], context=context)
        persistence_frame = _stage5_candidate_predictions(candidate=pool_by_label["persistence"], context=context)
        if base_frame.empty or persistence_frame.empty:
            return pd.Series(dtype=float)
        base_series = (
            base_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
        persistence_series = (
            persistence_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
        return _blend_nowcast_prediction_series(
            candidate_series=base_series,
            persistence_series=persistence_series,
            candidate_weight=float(control_blend_weight),
        )
    if base_label and base_label in pool_by_label and control_bucket_size_minutes > 0 and control_bucket_weights_json:
        base_frame = _stage5_candidate_predictions(candidate=pool_by_label[base_label], context=context)
        persistence_frame = _stage5_candidate_predictions(candidate=pool_by_label["persistence"], context=context)
        if base_frame.empty or persistence_frame.empty:
            return pd.Series(dtype=float)
        base_series = (
            base_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
        persistence_series = (
            persistence_frame.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")["predicted_load"]
            .astype(float)
        )
        bucket_weights = {
            int(key): float(value)
            for key, value in json.loads(control_bucket_weights_json).items()
        }
        return _blend_nowcast_prediction_series_by_bucket(
            candidate_series=base_series,
            persistence_series=persistence_series,
            candidate_weight_by_bucket=bucket_weights,
            bucket_size_minutes=int(control_bucket_size_minutes),
            lock_interval_minutes=int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"]),
        )
    return pd.Series(dtype=float)


def _plot_lock_mae(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot lock-interval MAE after each control layer is applied."""
    output_path = output_dir / "fig_control_lock_mae.png"
    working = summary.loc[summary["role"].isin(["day_ahead", "hourly", "phase", "nowcast"])].copy()
    plt.figure(figsize=(8, 5))
    plt.bar(
        working["layer"],
        working["lock_mae"],
        color=["#8c8c8c", "#4c78a8", "#54a24b", "#e45756"][: len(working)],
    )
    plt.ylabel("15-minute lock MAE")
    plt.title("Control-Layer Locked-Interval Error")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_example_cycle(intervals: pd.DataFrame, output_dir: Path) -> None:
    """Plot one 24-hour control cycle on 15-minute interval means."""
    output_path = output_dir / "fig_control_example_cycle.png"
    example_cycle = str(intervals.iloc[0]["cycle_origin_timestamp"])
    example = intervals.loc[intervals["cycle_origin_timestamp"].astype("string").eq(example_cycle)].copy()
    plt.figure(figsize=(12, 6))
    plt.plot(example["interval_start"], example["actual_interval_mean"], label="actual", linewidth=2)
    plt.plot(example["interval_start"], example["day_ahead_interval_mean"], label="day_ahead", alpha=0.8)
    plt.plot(example["interval_start"], example["hourly_interval_mean"], label="after_hourly", alpha=0.8)
    plt.plot(example["interval_start"], example["phase_interval_mean"], label="after_phase", alpha=0.9)
    if "nowcast_interval_mean" in example.columns:
        plt.plot(example["interval_start"], example["nowcast_interval_mean"], label="after_nowcast", alpha=0.9)
    plt.legend()
    plt.title("Example 24h Control Cycle on 15-Minute Means")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_day_ahead_refresh_policy(refresh_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot the frozen versus refreshed day-ahead scenarios on the exact control cycles."""
    output_path = output_dir / "fig_day_ahead_refresh_policy.png"
    working = refresh_summary.copy()
    plt.figure(figsize=(10, 5))
    x = np.arange(len(working))
    width = 0.35
    plt.bar(x - width / 2, working["profile_shape_mae"], width=width, label="profile_shape_mae", color="#4c78a8")
    plt.bar(x + width / 2, working["lock_mae"], width=width, label="lock_mae", color="#f58518")
    plt.xticks(x, working["scenario"], rotation=10, ha="right")
    plt.ylabel("MAE")
    plt.title("Day-Ahead Refresh Policy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_rolling_control_lock_distribution(rolling_by_cycle: pd.DataFrame, output_dir: Path) -> None:
    """Plot the lock-MAE distribution by layer over the broader rolling benchmark."""
    output_path = output_dir / "fig_control_lock_distribution.png"
    if rolling_by_cycle.empty:
        return
    plot_rows: list[pd.DataFrame] = []
    for layer_name in ("day_ahead", "hourly", "phase", "nowcast"):
        column = f"{layer_name}_lock_mae"
        if column not in rolling_by_cycle.columns:
            continue
        plot_rows.append(
            pd.DataFrame(
                {
                    "layer": _control_layer_label(layer_name),
                    "lock_mae": pd.to_numeric(rolling_by_cycle[column], errors="coerce"),
                }
            )
        )
    if not plot_rows:
        return
    plot_df = pd.concat(plot_rows, ignore_index=True).dropna(subset=["lock_mae"])
    ordered_layers = [
        _control_layer_label(layer_name)
        for layer_name in ("day_ahead", "hourly", "phase", "nowcast")
        if _control_layer_label(layer_name) in set(plot_df["layer"])
    ]
    grouped = [
        plot_df.loc[plot_df["layer"].astype("string").eq(layer), "lock_mae"].to_numpy(dtype=float)
        for layer in ordered_layers
    ]
    plt.figure(figsize=(10, 5))
    plt.boxplot(grouped, tick_labels=ordered_layers, showmeans=True)
    plt.ylabel("15-minute lock MAE")
    plt.title("Rolling Control-Cycle Lock-MAE Distribution")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_control_layer_gain_ci(inference: pd.DataFrame, output_dir: Path) -> None:
    """Plot lock-MAE gain confidence intervals for each control-layer comparison."""
    output_path = output_dir / "fig_control_layer_gain_ci.png"
    if inference.empty:
        return
    plot_df = inference.loc[
        inference["scope"].astype("string").eq("rolling_evaluation")
        & inference["metric_name"].astype("string").eq("lock_mae")
    ].copy()
    if plot_df.empty:
        plot_df = inference.loc[
            inference["scope"].astype("string").eq("rolling_combined")
            & inference["metric_name"].astype("string").eq("lock_mae")
        ].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("gain_metric", ascending=True, kind="stable").reset_index(drop=True)
    y = np.arange(len(plot_df))
    lower = plot_df["gain_metric"] - plot_df["gain_metric_ci_low"]
    upper = plot_df["gain_metric_ci_high"] - plot_df["gain_metric"]
    plt.figure(figsize=(9, 4 + 0.6 * len(plot_df)))
    plt.errorbar(
        plot_df["gain_metric"].to_numpy(dtype=float),
        y,
        xerr=np.vstack([lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)]),
        fmt="o",
        color="#4c78a8",
        ecolor="#4c78a8",
        capsize=4,
    )
    plt.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
    plt.yticks(y, plot_df["comparison_label"])
    plt.xlabel("Lock-MAE gain versus previous layer")
    plt.title("Rolling Benchmark Control-Layer Gain CIs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_phase_stack_candidates(phase_stack_candidate_benchmarks: pd.DataFrame, output_dir: Path) -> None:
    """Plot lock gain versus profile regression for the stack-aware phase candidates."""
    output_path = output_dir / "fig_phase_stack_candidates.png"
    if phase_stack_candidate_benchmarks.empty:
        return
    x_column = (
        "evaluation_profile_degrade_pct_vs_hourly"
        if "evaluation_profile_degrade_pct_vs_hourly" in phase_stack_candidate_benchmarks.columns
        else "profile_degrade_pct_vs_hourly"
    )
    y_column = (
        "evaluation_lock_gain_pct_vs_hourly"
        if "evaluation_lock_gain_pct_vs_hourly" in phase_stack_candidate_benchmarks.columns
        else "lock_gain_pct_vs_hourly"
    )
    plot_df = phase_stack_candidate_benchmarks.copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column])
    if plot_df.empty:
        return
    colors = np.where(
        plot_df["stack_selected_candidate"].astype(bool),
        "#e45756",
        np.where(plot_df["candidate_type"].astype("string").eq("learned"), "#4c78a8", "#72b7b2"),
    )
    plt.figure(figsize=(9, 6))
    plt.scatter(plot_df[x_column], plot_df[y_column], c=colors, alpha=0.85)
    plt.axvline(
        float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_profile_degrade_pct"]),
        color="#666666",
        linestyle="--",
        linewidth=1,
    )
    plt.axhline(
        float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_lock_gain_pct"]),
        color="#666666",
        linestyle="--",
        linewidth=1,
    )
    for row in plot_df.itertuples(index=False):
        if bool(getattr(row, "stack_selected_candidate", False)):
            plt.annotate(
                str(getattr(row, "candidate_label")),
                (float(getattr(row, x_column)), float(getattr(row, y_column))),
                textcoords="offset points",
                xytext=(6, 4),
                ha="left",
            )
    plt.xlabel("Profile regression versus hourly")
    plt.ylabel("Lock-MAE gain versus hourly")
    plt.title("Phase Stack Candidate Frontier")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _format_md_float(value: Any, digits: int = 6) -> str:
    """Render one numeric value for markdown while preserving missing-value intent."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _build_current_evidence_index(
    *,
    summary: pd.DataFrame,
    policy: dict[str, Any],
    rolling_scope_summary: pd.DataFrame | None = None,
    rolling_layer_inference: pd.DataFrame | None = None,
    refresh_summary: pd.DataFrame | None = None,
    rolling_refresh_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one compact evidence index for the repo's current layered operating answer."""
    rows: list[dict[str, Any]] = []

    def append_row(
        *,
        section: str,
        scope: str,
        layer: str,
        objective: str,
        candidate_label: str,
        metric_name: str,
        metric_value: Any,
        metric_pct: Any = float("nan"),
        source_artifact: str,
        notes: str = "",
        ci_low: Any = float("nan"),
        ci_high: Any = float("nan"),
        p_value: Any = float("nan"),
    ) -> None:
        rows.append(
            {
                "section": str(section),
                "scope": str(scope),
                "layer": str(layer),
                "objective": str(objective),
                "candidate_label": str(candidate_label),
                "metric_name": str(metric_name),
                "metric_value": float(metric_value) if pd.notna(metric_value) else float("nan"),
                "metric_pct": float(metric_pct) if pd.notna(metric_pct) else float("nan"),
                "ci_low": float(ci_low) if pd.notna(ci_low) else float("nan"),
                "ci_high": float(ci_high) if pd.notna(ci_high) else float("nan"),
                "p_value": float(p_value) if pd.notna(p_value) else float("nan"),
                "source_artifact": str(source_artifact),
                "notes": str(notes),
            }
        )

    layer_policy_map = {
        "day_ahead": cast(dict[str, Any], policy["day_ahead"]),
        "hourly": cast(dict[str, Any], policy["hourly"]),
        "phase": cast(dict[str, Any], policy["phase"]),
        "nowcast": cast(dict[str, Any], policy["nowcast_anchor"]),
    }
    for row in summary.itertuples(index=False):
        role = str(getattr(row, "role"))
        payload = layer_policy_map[role]
        candidate_label = (
            str(payload.get("stack_guard_applied_candidate_label", payload["candidate_label"]))
            if role == "phase"
            else str(payload["candidate_label"])
        )
        for metric_name in (
            "lock_mae",
            "next_lock_mae",
            "profile_shape_mae",
            "minute_path_mae",
            "energy_mae",
            "peak_value_mae",
            "peak_interval_hit_rate",
            "peak_interval_offset_minutes",
        ):
            if not hasattr(row, metric_name):
                continue
            metric_pct = getattr(row, f"{metric_name}_pct") if hasattr(row, f"{metric_name}_pct") else float("nan")
            append_row(
                section="stage10_exact_control",
                scope="exact_control_evaluation",
                layer=str(getattr(row, "layer")),
                objective=metric_name,
                candidate_label=candidate_label,
                metric_name=metric_name,
                metric_value=getattr(row, metric_name),
                metric_pct=metric_pct,
                source_artifact="control_backtest_summary.csv",
                notes="Current selected control-stack layer on the exact shared control cycles.",
            )

    if refresh_summary is not None and not refresh_summary.empty and "day_ahead_refresh" in policy:
        refresh_policy = cast(dict[str, Any], policy["day_ahead_refresh"])
        recommended = str(refresh_policy.get("recommended_policy", ""))
        recommended_row = refresh_summary.loc[
            refresh_summary["scenario"].astype("string").eq(recommended)
        ]
        if not recommended_row.empty:
            row = recommended_row.iloc[0]
            append_row(
                section="stage10_refresh_exact",
                scope="exact_control_evaluation",
                layer="day_ahead_refresh",
                objective="profile_shape_mae",
                candidate_label=str(refresh_policy.get("candidate_label", "")),
                metric_name="profile_shape_mae",
                metric_value=row.get("profile_shape_mae", float("nan")),
                metric_pct=row.get("profile_shape_mae_pct", float("nan")),
                source_artifact="day_ahead_refresh_summary.csv",
                notes=f"Recommended exact-control refresh mode: {recommended}.",
            )
            append_row(
                section="stage10_refresh_exact",
                scope="exact_control_evaluation",
                layer="day_ahead_refresh",
                objective="lock_mae",
                candidate_label=str(refresh_policy.get("candidate_label", "")),
                metric_name="lock_mae",
                metric_value=row.get("lock_mae", float("nan")),
                metric_pct=row.get("lock_mae_pct", float("nan")),
                source_artifact="day_ahead_refresh_summary.csv",
                notes=f"Recommended exact-control refresh mode: {recommended}.",
            )

    if rolling_scope_summary is not None and not rolling_scope_summary.empty:
        for row in rolling_scope_summary.itertuples(index=False):
            for metric_name in ("lock_mae", "next_lock_mae", "profile_shape_mae", "peak_interval_hit_rate"):
                if not hasattr(row, metric_name):
                    continue
                metric_pct = getattr(row, f"{metric_name}_pct") if hasattr(row, f"{metric_name}_pct") else float("nan")
                note_tail = ""
                if hasattr(row, f"{metric_name}_p50") and hasattr(row, f"{metric_name}_p90"):
                    note_tail = (
                        f" Rolling benchmark mean with p50={_format_md_float(getattr(row, f'{metric_name}_p50'))} "
                        f"and p90={_format_md_float(getattr(row, f'{metric_name}_p90'))}."
                    )
                append_row(
                    section="stage10_rolling_benchmark",
                    scope=str(getattr(row, "scope")),
                    layer=str(getattr(row, "layer")),
                    objective=metric_name,
                    candidate_label=(
                        str(
                            layer_policy_map[str(getattr(row, "role"))].get(
                                "stack_guard_applied_candidate_label",
                                layer_policy_map[str(getattr(row, "role"))]["candidate_label"],
                            )
                        )
                        if str(getattr(row, "role")) == "phase"
                        else str(layer_policy_map[str(getattr(row, "role"))]["candidate_label"])
                    ),
                    metric_name=metric_name,
                    metric_value=getattr(row, metric_name),
                    metric_pct=metric_pct,
                    source_artifact="rolling_control_scope_summary.csv",
                    notes=note_tail,
                )

    if rolling_layer_inference is not None and not rolling_layer_inference.empty:
        for row in rolling_layer_inference.loc[
            rolling_layer_inference["metric_name"].astype("string").isin(["lock_mae", "next_lock_mae", "profile_shape_mae"])
        ].itertuples(index=False):
            append_row(
                section="stage10_rolling_inference",
                scope=str(getattr(row, "scope")),
                layer=str(getattr(row, "comparison_label")),
                objective=f"{str(getattr(row, 'metric_name'))}_gain",
                candidate_label=str(getattr(row, "comparison_label")),
                metric_name="gain_metric",
                metric_value=getattr(row, "gain_metric"),
                source_artifact="rolling_control_layer_inference.csv",
                notes="Positive gain means the later layer reduced error versus the previous stacked layer.",
                ci_low=getattr(row, "gain_metric_ci_low"),
                ci_high=getattr(row, "gain_metric_ci_high"),
                p_value=getattr(row, "two_sided_p"),
            )

    performance_holdout_path = preferred_output_path(PATHS["outputs_performance_dir"]) / "latest" / "holdout_evaluation.csv"
    if performance_holdout_path.exists():
        holdout = _read_csv_if_present(performance_holdout_path)
        if not holdout.empty:
            persistence_row = holdout.loc[
                holdout["candidate_label"].astype("string").eq("persistence")
            ]
            learned_rows = holdout.loc[
                holdout["candidate_type"].astype("string").str.contains("learned", na=False)
            ]
            if not persistence_row.empty:
                row = persistence_row.iloc[0]
                append_row(
                    section="stage5_holdout",
                    scope="holdout",
                    layer="1m_anchor",
                    objective="mae",
                    candidate_label="persistence",
                    metric_name="mae",
                    metric_value=row.get("mae", float("nan")),
                    metric_pct=row.get("mae_pct", float("nan")),
                    source_artifact=_relative_artifact_path(performance_holdout_path),
                    notes="Current Stage-5 baseline winner.",
                )
            if not learned_rows.empty:
                row = learned_rows.sort_values("mae", kind="stable").iloc[0]
                append_row(
                    section="stage5_holdout",
                    scope="holdout",
                    layer="1m_anchor",
                    objective="mae",
                    candidate_label=str(row.get("candidate_label", "")),
                    metric_name="mae",
                    metric_value=row.get("mae", float("nan")),
                    metric_pct=row.get("mae_pct", float("nan")),
                    source_artifact=_relative_artifact_path(performance_holdout_path),
                    notes="Best current learned 1-minute challenger on holdout.",
                )

    horizon_curve_path = preferred_output_path(PATHS["outputs_horizon_curve_dir"]) / "latest" / "horizon_curve_summary.csv"
    if horizon_curve_path.exists():
        horizon_curve = _read_csv_if_present(horizon_curve_path)
        if not horizon_curve.empty:
            for horizon in (1, 15, 60, 1440):
                horizon_rows = horizon_curve.loc[pd.to_numeric(horizon_curve["horizon_minutes"], errors="coerce").eq(horizon)]
                if horizon_rows.empty:
                    continue
                row = horizon_rows.iloc[0]
                selection_target = str(row.get("selection_target", ""))
                metric_lookup = {
                    "endpoint_mae": ("learned_endpoint_mae", "learned_endpoint_mae_pct"),
                    "path_mae": ("learned_path_mae", "learned_path_mae_pct"),
                    "phase_mean_mae": ("learned_phase_mean_mae", "learned_phase_mean_mae_pct"),
                    "next_lock_mae": ("learned_next_lock_mae", "learned_next_lock_mae_pct"),
                    "profile_shape_mae": ("learned_profile_shape_mae", "learned_profile_shape_mae_pct"),
                }
                metric_name, metric_pct_name = metric_lookup.get(
                    selection_target,
                    ("learned_path_mae", "learned_path_mae_pct"),
                )
                append_row(
                    section="stage8_horizon_curve",
                    scope=f"{horizon}m",
                    layer=f"{horizon}m_objective_winner",
                    objective=selection_target,
                    candidate_label=str(row.get("candidate_label", "")),
                    metric_name=selection_target,
                    metric_value=row.get(metric_name, float("nan")),
                    metric_pct=row.get(metric_pct_name, float("nan")),
                    source_artifact=_relative_artifact_path(horizon_curve_path),
                    notes=f"Current Stage-8 winner at {horizon} minutes.",
                )

    return pd.DataFrame(rows)


def _write_current_evidence_index_md(index: pd.DataFrame, output_path: Path) -> None:
    """Write a readable markdown index for the current layered operating evidence."""
    lines = [
        "# Current Evidence Index",
        "",
        "This file summarizes the current winners and strongest supporting artifacts across the repo's layered decision stack.",
        "",
    ]
    if index.empty:
        lines.extend(["No evidence rows were available for this run.", ""])
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return
    for section, section_frame in index.groupby("section", sort=False):
        lines.extend([f"## {str(section)}", ""])
        for row in section_frame.itertuples(index=False):
            metric_pct = _format_md_float(getattr(row, "metric_pct"), 3)
            ci_low = _format_md_float(getattr(row, "ci_low"), 6)
            ci_high = _format_md_float(getattr(row, "ci_high"), 6)
            p_value = _format_md_float(getattr(row, "p_value"), 4)
            line = (
                f"- `{row.scope}` / `{row.layer}` / `{row.objective}`: "
                f"`{row.candidate_label}` -> `{row.metric_name}={_format_md_float(row.metric_value)}`"
            )
            if metric_pct != "n/a":
                line += f" (`{metric_pct}%`)"
            if ci_low != "n/a" or ci_high != "n/a":
                line += f", CI [`{ci_low}`, `{ci_high}`]"
            if p_value != "n/a":
                line += f", p=`{p_value}`"
            line += f". Source: `{row.source_artifact}`."
            if str(row.notes):
                line += f" {row.notes}"
            lines.append(line)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_md(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    policy: dict[str, Any],
    refresh_summary: pd.DataFrame | None = None,
    rolling_scope_summary: pd.DataFrame | None = None,
    rolling_layer_inference: pd.DataFrame | None = None,
    rolling_refresh_summary: pd.DataFrame | None = None,
) -> None:
    """Write the narrative markdown summary for the selected control stack."""
    lines = [
        "# Forecast-Control Backtest",
        "",
        f"- Load type: `{DATASET['load_type']}`",
        f"- Control cycles: `{int(summary['cycle_n'].max()) if not summary.empty else 0}`",
        (
            "- Day-ahead policy: "
            f"`{policy['day_ahead']['candidate_label']}` "
            f"({policy['day_ahead']['candidate_type']}; "
            f"{policy['day_ahead']['control_selection_metric']}="
            f"{float(policy['day_ahead']['control_selection_metric_value']):.6f}, "
            f"{float(policy['day_ahead']['control_selection_metric_pct']):.3f}%)"
        ),
        (
            "- Hourly policy: "
            f"`{policy['hourly']['candidate_label']}` "
            f"({policy['hourly']['candidate_type']}; "
            f"{policy['hourly']['control_selection_metric']}="
            f"{float(policy['hourly']['control_selection_metric_value']):.6f}, "
            f"{float(policy['hourly']['control_selection_metric_pct']):.3f}%)"
        ),
        (
            "- Phase policy: "
            f"applied `{policy['phase'].get('stack_guard_applied_candidate_label', policy['phase']['candidate_label'])}`; "
            f"isolated winner `{policy['phase']['candidate_label']}` "
            f"({policy['phase']['candidate_type']}; "
            f"{policy['phase']['control_selection_metric']}="
            f"{float(policy['phase']['control_selection_metric_value']):.6f}, "
            f"{float(policy['phase']['control_selection_metric_pct']):.3f}%)"
        ),
        (
            "- 1-minute nowcast policy: "
            f"`{policy['nowcast_anchor']['candidate_label']}` "
            f"({policy['nowcast_anchor']['candidate_type']}; "
            f"{policy['nowcast_anchor']['control_selection_metric']}="
            f"{float(policy['nowcast_anchor']['control_selection_metric_value']):.6f}, "
            f"{float(policy['nowcast_anchor']['control_selection_metric_pct']):.3f}%)"
        ),
        "",
        "## How To Read This",
        "",
        "- `day_ahead` is the frozen 24-hour profile before any intraday correction.",
        "- `after_hourly` shows what happens after the 60-minute layer overwrites the affected future window.",
        "- `after_phase` shows the final stack after the 15-minute correction layer is applied.",
        "- `after_nowcast` shows the true last-mile stack after 1-minute predictions overwrite the immediate minute path as the cycle unfolds.",
        "- Lower MAE and MAE% are better. The lock metric is the operational one: it measures the next 15-minute interval that will lock in before the customer can react again.",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary.iterrows():
        next_lock_mae = float(row.get("next_lock_mae", float("nan")))
        next_lock_mae_pct = float(row.get("next_lock_mae_pct", float("nan")))
        next_lock_gain = float(row.get("next_lock_mae_gain_vs_day_ahead", float("nan")))
        peak_value_mae = float(row.get("peak_value_mae", float("nan")))
        peak_value_mae_pct = float(row.get("peak_value_mae_pct", float("nan")))
        peak_hit_rate = float(row.get("peak_interval_hit_rate", float("nan")))
        peak_offset_minutes = float(row.get("peak_interval_offset_minutes", float("nan")))
        lines.extend(
            [
                f"### {str(row['layer'])}",
                "",
                f"- Minute path MAE / %: `{float(row['minute_path_mae']):.6f}` / `{float(row['minute_path_mae_pct']):.3f}%`",
                f"- 15-minute lock MAE / %: `{float(row['lock_mae']):.6f}` / `{float(row['lock_mae_pct']):.3f}%`",
                f"- Next-lock MAE / %: `{next_lock_mae:.6f}` / `{next_lock_mae_pct:.3f}%`",
                f"- Profile-shape MAE / %: `{float(row['profile_shape_mae']):.6f}` / `{float(row['profile_shape_mae_pct']):.3f}%`",
                f"- Energy MAE / %: `{float(row['energy_mae']):.6f}` / `{float(row['energy_mae_pct']):.3f}%`",
                f"- Lock-MAE gain vs frozen day-ahead: `{float(row['lock_mae_gain_vs_day_ahead']):.6f}`",
                f"- Next-lock gain vs frozen day-ahead: `{next_lock_gain:.6f}`",
                f"- Peak-value MAE / %: `{peak_value_mae:.6f}` / `{peak_value_mae_pct:.3f}%`",
                f"- Peak-interval hit rate: `{peak_hit_rate * 100.0:.2f}%`",
                f"- Peak-interval mean offset: `{peak_offset_minutes:.3f}` minutes",
                "",
            ]
        )
    lines.extend(
        [
            "## Replay Cache",
            "",
            (
                "- Day-ahead replay cache: "
                f"`{policy['day_ahead']['selected_replay_cache_status']}` "
                f"at `{policy['day_ahead']['selected_replay_cache_artifact']}`"
            ),
            (
                "- Day-ahead benchmark scope: "
                f"`{policy['day_ahead']['benchmark_origin_mode']}` "
                f"with `{policy['day_ahead']['candidate_pool_count']}` pooled candidate contexts"
            ),
            (
                "- Hourly replay cache: "
                f"`{policy['hourly']['selected_replay_cache_status']}` "
                f"at `{policy['hourly']['selected_replay_cache_artifact']}`"
            ),
            (
                "- Hourly benchmark scope: "
                f"`{policy['hourly']['benchmark_origin_mode']}` "
                f"with `{policy['hourly']['candidate_pool_count']}` pooled candidate contexts"
            ),
            (
                "- Phase replay cache: "
                f"`{policy['phase']['selected_replay_cache_status']}` "
                f"at `{policy['phase']['selected_replay_cache_artifact']}`"
            ),
            (
                "- Phase benchmark scope: "
                f"`{policy['phase']['benchmark_origin_mode']}` "
                f"with `{policy['phase']['candidate_pool_count']}` pooled candidate contexts"
            ),
            (
                "- Phase stack guard: "
                f"`{policy['phase'].get('stack_guard_recommended_policy', 'phase_candidate')}` "
                f"applied `{policy['phase'].get('stack_guard_applied_candidate_label', policy['phase']['candidate_label'])}`. "
                f"Lock gain vs hourly: `{float(policy['phase'].get('stack_guard_lock_gain_vs_hourly', float('nan'))):.6f}` "
                f"({float(policy['phase'].get('stack_guard_lock_gain_pct_vs_hourly', float('nan'))) * 100.0:.3f}%). "
                f"Profile regression vs hourly: "
                f"`{float(policy['phase'].get('stack_guard_profile_degrade_vs_hourly', float('nan'))):.6f}` "
                f"({float(policy['phase'].get('stack_guard_profile_degrade_pct_vs_hourly', float('nan'))) * 100.0:.3f}%)."
            ),
                (
                    "- Phase stack selection mode: "
                    f"`{policy['phase'].get('stack_selection_mode', 'legacy')}` "
                    f"with applied replay artifact "
                    f"`{policy['phase'].get('stack_guard_applied_selection_artifact', policy['phase'].get('selection_artifact', ''))}`."
                ),
            (
                "- Phase stack guard rationale: "
                f"{policy['phase'].get('stack_guard_reason', 'Legacy policy surface did not record a stack guard decision.')}"
            ),
            (
                "- Phase rolling support: "
                f"`{policy['phase'].get('rolling_support_recommended_policy', 'phase_candidate')}` on "
                f"`{policy['phase'].get('rolling_support_scope', 'rolling_evaluation')}`. "
                f"Lock gain vs hourly: "
                f"`{float(policy['phase'].get('rolling_support_lock_gain_vs_hourly', float('nan'))):.6f}` "
                f"({float(policy['phase'].get('rolling_support_lock_gain_pct_vs_hourly', float('nan'))) * 100.0:.3f}%). "
                f"Next-lock regress vs hourly: "
                f"`{float(policy['phase'].get('rolling_support_next_lock_regress_vs_hourly', float('nan'))):.6f}`. "
                f"Profile regress vs hourly: "
                f"`{float(policy['phase'].get('rolling_support_profile_degrade_vs_hourly', float('nan'))):.6f}`. "
                f"Veto applied: `{bool(policy['phase'].get('rolling_support_applied_veto', False))}`."
            ),
            (
                "- Phase rolling support rationale: "
                f"{policy['phase'].get('rolling_support_reason', 'Rolling support was not recorded in this policy surface.')}"
            ),
            (
                "- 1-minute nowcast benchmark scope: "
                f"`{policy['nowcast_anchor']['benchmark_origin_mode']}` "
                f"with `{policy['nowcast_anchor']['candidate_pool_count']}` pooled Stage-5 candidate contexts"
            ),
            "",
            "## Visuals",
            "",
            "The figures below answer two questions: does each update layer reduce locked-interval error, and what does one corrected 24-hour cycle actually look like?",
            "",
            "### Locked-Interval MAE Progression",
            "",
            "![Locked-interval MAE progression](fig_control_lock_mae.png)",
            "",
            "Look for a downward move from `day_ahead` to `after_hourly` to `after_phase` to `after_nowcast`. If a layer makes the bar move up, that update policy is not helping on the exact control cycles.",
            "",
            "### Example 24h Control Cycle",
            "",
            "![Example 24h control cycle](fig_control_example_cycle.png)",
            "",
            "Look for whether hourly, phase, and minute-level nowcasts pull the forecast toward actual peaks and troughs before the next costly interval locks in.",
        ]
    )
    if rolling_scope_summary is not None and not rolling_scope_summary.empty:
        lines.extend(
            [
                "",
                "## Rolling Benchmark",
                "",
                "The exact-control backtest remains the operational replay surface, but the rolling benchmark broadens the evidence base across more start times and adds uncertainty estimates for stacked gains.",
                "",
            ]
        )
        for scope_name in ("rolling_calibration", "rolling_evaluation"):
            scope_rows = rolling_scope_summary.loc[
                rolling_scope_summary["scope"].astype("string").eq(scope_name)
            ]
            if scope_rows.empty:
                continue
            cycle_n = int(scope_rows["cycle_n"].max())
            lines.extend([f"### {scope_name}", "", f"- Cycles: `{cycle_n}`", ""])
            for row in scope_rows.itertuples(index=False):
                lines.extend(
                    [
                        f"- `{row.layer}` lock MAE mean / p50 / p90: `{_format_md_float(row.lock_mae)}` / `{_format_md_float(row.lock_mae_p50)}` / `{_format_md_float(row.lock_mae_p90)}`",
                        f"- `{row.layer}` next-lock MAE mean / p50 / p90: `{_format_md_float(row.next_lock_mae)}` / `{_format_md_float(row.next_lock_mae_p50)}` / `{_format_md_float(row.next_lock_mae_p90)}`",
                        f"- `{row.layer}` profile-shape MAE mean / p50 / p90: `{_format_md_float(row.profile_shape_mae)}` / `{_format_md_float(row.profile_shape_mae_p50)}` / `{_format_md_float(row.profile_shape_mae_p90)}`",
                        f"- `{row.layer}` peak-interval hit rate mean / p50 / p90: `{_format_md_float(row.peak_interval_hit_rate, 4)}` / `{_format_md_float(row.peak_interval_hit_rate_p50, 4)}` / `{_format_md_float(row.peak_interval_hit_rate_p90, 4)}`",
                    ]
                )
            lines.append("")
        if rolling_layer_inference is not None and not rolling_layer_inference.empty:
            lines.extend(["### Layer-Gain Inference", ""])
            inference_rows = rolling_layer_inference.loc[
                rolling_layer_inference["scope"].astype("string").eq("rolling_evaluation")
                & rolling_layer_inference["metric_name"].astype("string").isin(["lock_mae", "next_lock_mae"])
            ]
            if inference_rows.empty:
                inference_rows = rolling_layer_inference.loc[
                    rolling_layer_inference["scope"].astype("string").eq("rolling_combined")
                    & rolling_layer_inference["metric_name"].astype("string").isin(["lock_mae", "next_lock_mae"])
                ]
            for row in inference_rows.itertuples(index=False):
                lines.append(
                    f"- `{row.comparison_label}` `{row.metric_name}` gain: `{_format_md_float(row.gain_metric)}` "
                    f"with 95% CI [`{_format_md_float(row.gain_metric_ci_low)}`, `{_format_md_float(row.gain_metric_ci_high)}`] "
                    f"and two-sided p=`{_format_md_float(row.two_sided_p, 4)}`."
                )
            lines.extend(
                [
                    "",
                    "### Rolling Lock-MAE Distribution",
                    "",
                    "![Rolling lock-MAE distribution](fig_control_lock_distribution.png)",
                    "",
                    "Look for the distribution of each layer to shift downward, not just the mean. That is the quickest way to see whether a layer is helping consistently or only on a few easy cycles.",
                    "",
                    "### Rolling Gain Confidence Intervals",
                    "",
                    "![Rolling gain confidence intervals](fig_control_layer_gain_ci.png)",
                    "",
                    "Look for confidence intervals that stay above zero. That indicates the later layer is improving error versus the previous layer with better statistical credibility than a single point estimate.",
                ]
            )
    if Path(output_dir / "fig_phase_stack_candidates.png").exists():
        lines.extend(
            [
                "",
                "## Phase Stack Frontier",
                "",
                "![Phase stack candidate frontier](fig_phase_stack_candidates.png)",
                "",
                "The x-axis shows profile regression versus the hourly stack and the y-axis shows lock-MAE gain versus the hourly stack. The desired region is upper-left: better lock error without giving back too much profile quality.",
            ]
        )
    if refresh_summary is not None and not refresh_summary.empty and "day_ahead_refresh" in policy:
        refresh_policy = cast(dict[str, Any], policy["day_ahead_refresh"])
        lines.extend(
            [
                "",
                "## Day-Ahead Refresh Study",
                "",
                (
                    "- Refresh candidate: "
                    f"`{refresh_policy['candidate_label']}` "
                    f"replayed every `{int(refresh_policy['refresh_interval_minutes'])}` minutes."
                ),
                (
                    "- Trigger policy: refresh when the configured trigger-mode "
                    f"`{refresh_policy.get('trigger_mode', 'any')}` fires across recent residual drift, "
                    "workday-transition mismatch, and activity-profile shift signals."
                ),
                (
                    "- Promotion rule: only recommend `triggered_refresh` when it improves "
                    "`profile_shape_mae` and does not worsen `lock_mae` versus the frozen day-ahead path, "
                    "while still preserving enough of the unconditional refresh gain to stay worth the added complexity."
                ),
                (
                    "- Recommended day-ahead mode: "
                    f"`{refresh_policy['recommended_policy']}` "
                    f"({refresh_policy['reason']})"
                ),
                (
                    "- Threshold source: "
                    f"`{refresh_policy.get('threshold_source', 'exact_control')}`"
                ),
                (
                    "- Trigger calibration result: "
                    f"triggered refresh kept "
                    f"`{float(refresh_policy.get('triggered_profile_gain_fraction_vs_unconditional', float('nan'))) * 100.0:.2f}%` "
                    "of the unconditional profile-shape gain and "
                    f"`{float(refresh_policy.get('triggered_lock_gain_fraction_vs_unconditional', float('nan'))) * 100.0:.2f}%` "
                    "of the unconditional lock-MAE gain."
                ),
                "",
            ]
        )
        for _, row in refresh_summary.iterrows():
            lines.extend(
                [
                    f"### {str(row['scenario'])}",
                    "",
                    f"- Minute path MAE / %: `{float(row['minute_path_mae']):.6f}` / `{float(row['minute_path_mae_pct']):.3f}%`",
                    f"- 15-minute lock MAE / %: `{float(row['lock_mae']):.6f}` / `{float(row['lock_mae_pct']):.3f}%`",
                    f"- Profile-shape MAE / %: `{float(row['profile_shape_mae']):.6f}` / `{float(row['profile_shape_mae_pct']):.3f}%`",
                    f"- Lock-MAE gain vs frozen day-ahead: `{float(row['lock_mae_gain_vs_frozen']):.6f}`",
                    f"- Profile-shape gain vs frozen day-ahead: `{float(row['profile_shape_mae_gain_vs_frozen']):.6f}`",
                    f"- Mean refresh updates applied per cycle: `{float(row['refresh_update_count']):.3f}`",
                    "",
                ]
            )
        lines.extend(
            [
                "### Refresh Policy Figure",
                "",
                "![Day-ahead refresh policy comparison](fig_day_ahead_refresh_policy.png)",
                "",
                "Look for whether `triggered_refresh` preserves or improves the 24-hour profile while avoiding a lock-MAE regression versus the frozen path. `unconditional_refresh` is a stress test; it shows whether the residual model is helpful even before triggers are applied.",
            ]
        )
    if rolling_refresh_summary is not None and not rolling_refresh_summary.empty and "day_ahead_refresh" in policy:
        refresh_policy = cast(dict[str, Any], policy["day_ahead_refresh"])
        rolling_policy = cast(dict[str, Any], refresh_policy.get("rolling_benchmark", {}))
        lines.extend(
            [
                "",
                "## Rolling Refresh Calibration",
                "",
                (
                    "- Rolling benchmark recommendation: "
                    f"`{rolling_policy.get('recommended_policy', 'n/a')}` "
                    f"({rolling_policy.get('reason', 'n/a')})"
                ),
                (
                    "- Rolling trigger rate: "
                    f"`{_format_md_float(rolling_policy.get('trigger_rate', float('nan')), 4)}`"
                ),
                "",
            ]
        )
        for row in rolling_refresh_summary.itertuples(index=False):
            lines.extend(
                [
                    f"- `{row.scenario}` rolling lock MAE / profile-shape MAE: `{_format_md_float(row.lock_mae)}` / `{_format_md_float(row.profile_shape_mae)}`",
                    f"  p50 lock / profile: `{_format_md_float(row.lock_mae_p50)}` / `{_format_md_float(row.profile_shape_mae_p50)}`; p90 lock / profile: `{_format_md_float(row.lock_mae_p90)}` / `{_format_md_float(row.profile_shape_mae_p90)}`",
                ]
            )
    if Path(output_dir / "optimizer_delivery_contract.json").exists():
        lines.extend(
            [
                "",
                "## Optimizer Delivery Contract",
                "",
                "- Delivery contract: `optimizer_delivery_contract.json`",
                "- Operational policy: `optimizer_operational_policy.json`",
                "- Interval forecast preview: `optimizer_delivery_preview.csv`",
                "- Uncertainty calibration table: `optimizer_delivery_uncertainty_calibration.csv`",
                "- Uncertainty summary: `optimizer_delivery_uncertainty_summary.csv`",
                "",
                "These artifacts turn the Stage-10 replay into a pre-optimizer delivery surface: each 15-minute interval now carries as-of timing, freshness fields, confidence hints, candidate provenance, and calibrated interval bands derived from the held-out control calibration windows.",
            ]
        )
    (output_dir / "control_backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _control_interval_layer_specs(interval_frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the ordered interval-level prediction columns currently available."""
    layers = [
        ("day_ahead", "day_ahead_interval_mean"),
        ("hourly", "hourly_interval_mean"),
        ("phase", "phase_interval_mean"),
        ("nowcast", "nowcast_interval_mean"),
    ]
    return [(role, column) for role, column in layers if column in interval_frame.columns]


def _add_control_interval_context(
    interval_frame: pd.DataFrame,
    *,
    lock_interval_minutes: int,
) -> pd.DataFrame:
    """Attach lead-index context needed for optimizer-facing interval delivery artifacts."""
    if interval_frame.empty:
        return interval_frame.copy()
    working = interval_frame.copy()
    working["cycle_origin_timestamp"] = pd.to_datetime(working["cycle_origin_timestamp"], errors="raise")
    working["interval_start"] = pd.to_datetime(working["interval_start"], errors="raise")
    working["interval_end"] = pd.to_datetime(working["interval_end"], errors="raise")
    lead_index = (
        (working["interval_start"] - working["cycle_origin_timestamp"]) / pd.Timedelta(minutes=lock_interval_minutes)
    )
    working["lead_interval_index"] = np.rint(pd.to_numeric(lead_index, errors="coerce")).astype("int64")
    horizon_minutes = (
        (working["interval_end"] - working["cycle_origin_timestamp"]) / pd.Timedelta(minutes=1)
    )
    working["horizon_minutes"] = np.rint(pd.to_numeric(horizon_minutes, errors="coerce")).astype("int64")
    working["is_next_lock_interval"] = working["lead_interval_index"].eq(0)
    return working


def _build_optimizer_delivery_uncertainty_calibration(
    *,
    calibration_interval_timeline: pd.DataFrame,
    lock_interval_minutes: int,
    min_lead_specific_samples: int | None = None,
) -> pd.DataFrame:
    """Calibrate interval bands from held-out residual quantiles on control-calibration cycles."""
    if calibration_interval_timeline.empty:
        return pd.DataFrame()
    if min_lead_specific_samples is None:
        min_lead_specific_samples = int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_min_lead_specific_samples"])
    min_next_lock_samples = int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_min_samples"])
    min_predicted_peak_samples = int(MULTIRES_FORECAST_CONTROL["optimizer_delivery_predicted_peak_min_samples"])
    min_predicted_peak_lead_samples = int(
        MULTIRES_FORECAST_CONTROL["optimizer_delivery_predicted_peak_lead_min_samples"]
    )
    working = _add_control_interval_context(
        calibration_interval_timeline,
        lock_interval_minutes=lock_interval_minutes,
    )
    rows: list[dict[str, Any]] = []
    for layer_role, prediction_column in _control_interval_layer_specs(working):
        valid = working.loc[
            :,
            ["cycle_origin_timestamp", "lead_interval_index", "actual_interval_mean", prediction_column],
        ].copy()
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        predicted_peak_rank = valid.groupby("cycle_origin_timestamp", dropna=False)[prediction_column].rank(
            method="first",
            ascending=False,
        )
        valid["is_predicted_peak_interval"] = predicted_peak_rank.eq(1)
        residuals = valid["actual_interval_mean"].astype(float) - valid[prediction_column].astype(float)
        if residuals.empty:
            continue
        global_quantiles = residuals.quantile([0.025, 0.1, 0.9, 0.975]).to_dict()
        rows.append(
            {
                "layer_role": str(layer_role),
                "lead_interval_index": -1,
                "lead_interval_start_minutes": -1,
                "lead_interval_end_minutes": -1,
                "raw_support_n": int(len(residuals)),
                "calibration_sample_n": int(len(residuals)),
                "peak_context": "all",
                "quantile_source": "layer_global",
                "residual_q025": float(global_quantiles.get(0.025, float("nan"))),
                "residual_q10": float(global_quantiles.get(0.1, float("nan"))),
                "residual_q90": float(global_quantiles.get(0.9, float("nan"))),
                "residual_q975": float(global_quantiles.get(0.975, float("nan"))),
                }
            )
        next_lock_residuals = (
            valid.loc[valid["lead_interval_index"].eq(0), "actual_interval_mean"].astype(float)
            - valid.loc[valid["lead_interval_index"].eq(0), prediction_column].astype(float)
        )
        if len(next_lock_residuals) >= int(min_next_lock_samples):
            next_lock_quantiles = next_lock_residuals.quantile([0.025, 0.1, 0.9, 0.975]).to_dict()
            rows.append(
                {
                    "layer_role": str(layer_role),
                    "lead_interval_index": -1,
                    "lead_interval_start_minutes": 0,
                    "lead_interval_end_minutes": int(lock_interval_minutes),
                    "raw_support_n": int(len(next_lock_residuals)),
                    "calibration_sample_n": int(len(next_lock_residuals)),
                    "peak_context": "next_lock",
                    "quantile_source": "next_lock_global",
                    "residual_q025": float(next_lock_quantiles.get(0.025, float("nan"))),
                    "residual_q10": float(next_lock_quantiles.get(0.1, float("nan"))),
                    "residual_q90": float(next_lock_quantiles.get(0.9, float("nan"))),
                    "residual_q975": float(next_lock_quantiles.get(0.975, float("nan"))),
                }
            )
            if bool(MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scaled_enabled"]):
                next_lock_predictions = (
                    valid.loc[valid["lead_interval_index"].eq(0), prediction_column]
                    .astype(float)
                    .abs()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if not next_lock_predictions.empty:
                    scale_floor_quantile = float(
                        MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scale_floor_quantile"]
                    )
                    scale_floor_min_load = float(
                        MULTIRES_FORECAST_CONTROL["optimizer_delivery_next_lock_scale_floor_min_load"]
                    )
                    scale_floor_value = float(
                        max(
                            scale_floor_min_load,
                            np.nanquantile(next_lock_predictions.to_numpy(dtype=float), scale_floor_quantile),
                        )
                    )
                    scale_series = next_lock_predictions.clip(lower=scale_floor_value)
                    scaled_residuals = next_lock_residuals.reindex(scale_series.index).astype(float).divide(
                        scale_series,
                    )
                    scaled_quantiles = scaled_residuals.quantile([0.025, 0.1, 0.9, 0.975]).to_dict()
                    rows.append(
                        {
                            "layer_role": str(layer_role),
                            "lead_interval_index": -1,
                            "lead_interval_start_minutes": 0,
                            "lead_interval_end_minutes": int(lock_interval_minutes),
                            "raw_support_n": int(len(scaled_residuals)),
                            "calibration_sample_n": int(len(scaled_residuals)),
                            "peak_context": "next_lock",
                            "quantile_source": "next_lock_scaled_global",
                            "residual_q025": float(scaled_quantiles.get(0.025, float("nan"))),
                            "residual_q10": float(scaled_quantiles.get(0.1, float("nan"))),
                            "residual_q90": float(scaled_quantiles.get(0.9, float("nan"))),
                            "residual_q975": float(scaled_quantiles.get(0.975, float("nan"))),
                            "residual_scale_mode": "absolute_prediction_floor",
                            "scale_floor_value": float(scale_floor_value),
                        }
                    )
        predicted_peak_residuals = (
            valid.loc[valid["is_predicted_peak_interval"].astype(bool), "actual_interval_mean"].astype(float)
            - valid.loc[valid["is_predicted_peak_interval"].astype(bool), prediction_column].astype(float)
        )
        if len(predicted_peak_residuals) >= int(min_predicted_peak_samples):
            predicted_peak_quantiles = predicted_peak_residuals.quantile([0.025, 0.1, 0.9, 0.975]).to_dict()
            rows.append(
                {
                    "layer_role": str(layer_role),
                    "lead_interval_index": -1,
                    "lead_interval_start_minutes": -1,
                    "lead_interval_end_minutes": -1,
                    "raw_support_n": int(len(predicted_peak_residuals)),
                    "calibration_sample_n": int(len(predicted_peak_residuals)),
                    "peak_context": "predicted_peak",
                    "quantile_source": "predicted_peak_global",
                    "residual_q025": float(predicted_peak_quantiles.get(0.025, float("nan"))),
                    "residual_q10": float(predicted_peak_quantiles.get(0.1, float("nan"))),
                    "residual_q90": float(predicted_peak_quantiles.get(0.9, float("nan"))),
                    "residual_q975": float(predicted_peak_quantiles.get(0.975, float("nan"))),
                }
            )
        for lead_interval_index, lead_group in valid.groupby("lead_interval_index", sort=True):
            lead_predicted_peak_mask = lead_group["is_predicted_peak_interval"].astype(bool)
            lead_predicted_peak_residuals = (
                lead_group.loc[lead_predicted_peak_mask, "actual_interval_mean"].astype(float)
                - lead_group.loc[lead_predicted_peak_mask, prediction_column].astype(float)
            )
            if len(lead_predicted_peak_residuals) >= int(min_predicted_peak_lead_samples):
                lead_predicted_peak_quantiles = lead_predicted_peak_residuals.quantile(
                    [0.025, 0.1, 0.9, 0.975]
                ).to_dict()
                rows.append(
                    {
                        "layer_role": str(layer_role),
                        "lead_interval_index": int(lead_interval_index),
                        "lead_interval_start_minutes": int(lead_interval_index) * int(lock_interval_minutes),
                        "lead_interval_end_minutes": int(int(lead_interval_index) + 1)
                        * int(lock_interval_minutes),
                        "raw_support_n": int(len(lead_predicted_peak_residuals)),
                        "calibration_sample_n": int(len(lead_predicted_peak_residuals)),
                        "peak_context": "predicted_peak",
                        "quantile_source": "predicted_peak_lead_interval",
                        "residual_q025": float(lead_predicted_peak_quantiles.get(0.025, float("nan"))),
                        "residual_q10": float(lead_predicted_peak_quantiles.get(0.1, float("nan"))),
                        "residual_q90": float(lead_predicted_peak_quantiles.get(0.9, float("nan"))),
                        "residual_q975": float(lead_predicted_peak_quantiles.get(0.975, float("nan"))),
                    }
                )
            lead_residuals = (
                lead_group["actual_interval_mean"].astype(float) - lead_group[prediction_column].astype(float)
            )
            raw_support_n = int(len(lead_residuals))
            use_residuals = lead_residuals if raw_support_n >= int(min_lead_specific_samples) else residuals
            quantiles = use_residuals.quantile([0.025, 0.1, 0.9, 0.975]).to_dict()
            rows.append(
                {
                    "layer_role": str(layer_role),
                    "lead_interval_index": int(lead_interval_index),
                    "lead_interval_start_minutes": int(lead_interval_index) * int(lock_interval_minutes),
                    "lead_interval_end_minutes": int(int(lead_interval_index) + 1) * int(lock_interval_minutes),
                    "raw_support_n": raw_support_n,
                    "calibration_sample_n": int(len(use_residuals)),
                    "peak_context": "all",
                    "quantile_source": (
                        "lead_interval" if raw_support_n >= int(min_lead_specific_samples) else "layer_global_fallback"
                    ),
                    "residual_q025": float(quantiles.get(0.025, float("nan"))),
                    "residual_q10": float(quantiles.get(0.1, float("nan"))),
                    "residual_q90": float(quantiles.get(0.9, float("nan"))),
                    "residual_q975": float(quantiles.get(0.975, float("nan"))),
                }
            )
    calibration = pd.DataFrame(rows)
    if calibration.empty:
        return calibration
    if "residual_scale_mode" not in calibration.columns:
        calibration["residual_scale_mode"] = ""
    if "scale_floor_value" not in calibration.columns:
        calibration["scale_floor_value"] = float("nan")
    return calibration.sort_values(["layer_role", "lead_interval_index"], kind="stable").reset_index(drop=True)


def _selected_candidate_labels(policy: dict[str, Any]) -> dict[str, str]:
    """Map each stacked layer role onto the selected persisted candidate label."""
    return {
        "day_ahead": str(cast(dict[str, Any], policy["day_ahead"]).get("candidate_label", "")),
        "hourly": str(cast(dict[str, Any], policy["hourly"]).get("candidate_label", "")),
        "phase": str(
            cast(dict[str, Any], policy["phase"]).get(
                "stack_guard_applied_candidate_label",
                cast(dict[str, Any], policy["phase"]).get("candidate_label", ""),
            )
        ),
        "nowcast": str(cast(dict[str, Any], policy["nowcast_anchor"]).get("candidate_label", "")),
    }


def _optimizer_fallback_reason_map() -> dict[str, str]:
    """Describe why each layer would appear in the emitted delivery rows."""
    return {
        "nowcast": "full_stack_available",
        "phase": "nowcast_unavailable",
        "hourly": "phase_and_nowcast_unavailable",
        "day_ahead": "intraday_updates_unavailable",
    }


def _optimizer_layer_fallback_target(layer_role: str) -> str | None:
    """Resolve the next older layer to use when the requested layer is unavailable or stale."""
    fallback_targets = {
        "nowcast": "phase",
        "phase": "hourly",
        "hourly": "day_ahead",
        "day_ahead": None,
    }
    return fallback_targets.get(str(layer_role))


def _optimizer_layer_contracts(candidate_labels: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Return the machine-readable cadence and fallback contract for each delivery layer."""
    cadence_by_layer = _optimizer_layer_cadence_minutes_map()
    stale_threshold_by_layer = _optimizer_layer_stale_threshold_minutes_map()
    contracts: dict[str, dict[str, Any]] = {}
    for layer_role in OPTIMIZER_LAYER_PRIORITY:
        contracts[layer_role] = {
            "selected_candidate_label": str(candidate_labels.get(layer_role, "")),
            "expected_update_cadence_minutes": int(cadence_by_layer[layer_role]),
            "stale_threshold_minutes": int(stale_threshold_by_layer[layer_role]),
            "fallback_target_when_unavailable": _optimizer_layer_fallback_target(layer_role),
            "fallback_reason_when_selected": _optimizer_fallback_reason_map().get(layer_role, "unknown"),
        }
    return contracts


def _optimizer_layer_prediction_column_map(frame: pd.DataFrame) -> dict[str, str]:
    """Expose the prediction column attached to each delivery layer role."""
    return {str(role): str(column) for role, column in _control_interval_layer_specs(frame)}


def _policy_reason_token(value: Any) -> str:
    """Normalize a policy/debug reason into a filesystem- and CSV-friendly token."""
    token = str(value).strip().lower()
    if not token:
        return "unspecified"
    for old, new in ((" ", "_"), ("-", "_"), ("/", "_"), (":", "_")):
        token = token.replace(old, new)
    return token


def _attach_nowcast_dynamic_overlay_policy(
    frame: pd.DataFrame,
    *,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Attach per-row nowcast-overlay eligibility derived from broader Stage-5 evidence plus strategic intervals."""
    working = frame.copy()
    if working.empty:
        return working
    nowcast_policy = dict(cast(dict[str, Any], policy.get("nowcast_anchor", {})))
    advisory_surface_supported = bool(nowcast_policy.get("advisory_surface_supported", False))
    supported_regimes = {
        str(value)
        for value in nowcast_policy.get("advisory_supported_operating_regimes", [])
        if str(value).strip()
    }
    advisory_high_ramp_ratio = float(nowcast_policy.get("advisory_high_ramp_ratio_to_persistence", float("nan")))
    if (
        not advisory_surface_supported
        or not supported_regimes
        or not np.isfinite(advisory_high_ramp_ratio)
    ):
        advisory_lookup = _load_stage5_nowcast_advisory_evidence()
        advisory_keys = [
            str(nowcast_policy.get("blend_base_candidate_label", "")),
            str(nowcast_policy.get("candidate_label", "")),
        ]
        for advisory_key in advisory_keys:
            advisory_payload = cast(dict[str, Any] | None, advisory_lookup.get(str(advisory_key)))
            if not advisory_payload:
                continue
            advisory_surface_supported = bool(
                advisory_payload.get(
                    "advisory_surface_supported",
                    advisory_payload.get("learned_beats_persistence", False),
                )
            )
            supported_regimes = {
                str(value)
                for value in advisory_payload.get(
                    "advisory_supported_operating_regimes",
                    advisory_payload.get("learned_supported_operating_regimes", []),
                )
                if str(value).strip()
            }
            advisory_high_ramp_ratio = float(
                advisory_payload.get("advisory_high_ramp_ratio_to_persistence", float("nan"))
            )
            nowcast_policy.setdefault("advisory_surface_supported", advisory_surface_supported)
            nowcast_policy.setdefault(
                "advisory_supported_operating_regimes",
                sorted(supported_regimes),
            )
            nowcast_policy.setdefault(
                "advisory_supported_regime_count",
                int(advisory_payload.get("advisory_supported_regime_count", len(supported_regimes))),
            )
            nowcast_policy.setdefault(
                "advisory_surface_candidate_mae_ratio_to_persistence",
                float(advisory_payload.get("advisory_surface_candidate_mae_ratio_to_persistence", float("nan"))),
            )
            nowcast_policy.setdefault(
                "advisory_transition_best_ratio_to_persistence",
                float(advisory_payload.get("advisory_transition_best_ratio_to_persistence", float("nan"))),
            )
            nowcast_policy.setdefault(
                "advisory_high_ramp_ratio_to_persistence",
                float(advisory_high_ramp_ratio),
            )
            break
    enabled = bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enabled"])
    enforce = bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"])
    high_ramp_supported = bool(np.isfinite(advisory_high_ramp_ratio) and advisory_high_ramp_ratio < 1.0)
    dynamic_gate_enabled = bool(enabled and advisory_surface_supported)
    working["nowcast_dynamic_overlay_enabled"] = dynamic_gate_enabled
    working["nowcast_dynamic_overlay_enforced"] = bool(dynamic_gate_enabled and enforce)
    working["nowcast_dynamic_supported_operating_regimes"] = ",".join(sorted(supported_regimes))
    working["nowcast_dynamic_high_ramp_supported"] = bool(high_ramp_supported)
    working["nowcast_dynamic_surface_supported"] = bool(advisory_surface_supported)
    working["nowcast_dynamic_surface_candidate_mae_ratio_to_persistence"] = float(
        nowcast_policy.get("advisory_surface_candidate_mae_ratio_to_persistence", float("nan"))
    )
    working["nowcast_dynamic_transition_best_ratio_to_persistence"] = float(
        nowcast_policy.get("advisory_transition_best_ratio_to_persistence", float("nan"))
    )
    working["nowcast_dynamic_high_ramp_ratio_to_persistence"] = float(advisory_high_ramp_ratio)
    if not dynamic_gate_enabled:
        working["nowcast_dynamic_overlay_eligible"] = True
        working["nowcast_dynamic_overlay_reason"] = (
            "dynamic_overlay_disabled" if not enabled else "no_advisory_surface_support"
        )
        return working

    operating_regime = working.get("operating_regime", pd.Series("", index=working.index)).astype("string")
    high_ramp_fraction = pd.to_numeric(
        working.get("high_ramp_fraction", pd.Series(float("nan"), index=working.index)),
        errors="coerce",
    )
    next_lock_mask = (
        working.get("is_next_lock_interval", pd.Series(False, index=working.index)).astype(bool)
        if bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_next_lock"])
        else pd.Series(False, index=working.index)
    )
    predicted_peak_mask = (
        working.get("requested_is_predicted_peak_interval", pd.Series(False, index=working.index)).astype(bool)
        if bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_predicted_peak"])
        else pd.Series(False, index=working.index)
    )
    supported_regime_mask = (
        operating_regime.isin(sorted(supported_regimes)) if supported_regimes else pd.Series(False, index=working.index)
    )
    high_ramp_mask = (
        high_ramp_fraction.ge(float(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_high_ramp_fraction_threshold"]))
        if high_ramp_supported
        else pd.Series(False, index=working.index)
    )
    eligible_mask = next_lock_mask | predicted_peak_mask | supported_regime_mask | high_ramp_mask
    working["nowcast_dynamic_overlay_eligible"] = eligible_mask.astype(bool)
    default_reason = pd.Series("background_interval", index=working.index, dtype="string")
    default_reason.loc[supported_regime_mask] = "supported_operating_regime"
    default_reason.loc[high_ramp_mask] = "high_ramp_interval"
    default_reason.loc[predicted_peak_mask] = "predicted_peak_interval"
    default_reason.loc[next_lock_mask] = "next_lock_interval"
    working["nowcast_dynamic_overlay_reason"] = default_reason.astype("string")
    return working


def _optimizer_layer_resolution(
    row: pd.Series | dict[str, Any],
    *,
    layer_column_map: dict[str, str],
    candidate_labels: dict[str, str],
    cadence_by_layer: dict[str, int],
    stale_threshold_by_layer: dict[str, int],
    as_of_timestamp: pd.Timestamp | str | None = None,
) -> dict[str, Any]:
    """Resolve one interval row onto the freshest usable layer under the configured fallback chain."""
    cycle_origin = pd.Timestamp(row.get("cycle_origin_timestamp"))
    as_of_ts = pd.Timestamp(as_of_timestamp) if as_of_timestamp is not None else cycle_origin
    age_minutes = max(float((as_of_ts - cycle_origin) / pd.Timedelta(minutes=1)), 0.0)
    requested_layer_role = ""
    requested_forecast_value = float("nan")
    for layer_role in OPTIMIZER_LAYER_PRIORITY:
        prediction_column = layer_column_map.get(str(layer_role))
        if not prediction_column:
            continue
        forecast_value = pd.to_numeric(pd.Series([row.get(prediction_column)]), errors="coerce").iloc[0]
        if pd.isna(forecast_value):
            continue
        requested_layer_role = str(layer_role)
        requested_forecast_value = float(forecast_value)
        break
    if not requested_layer_role:
        return {
            "as_of_timestamp": as_of_ts,
            "effective_forecast_as_of": cycle_origin,
            "forecast_age_minutes": age_minutes,
            "requested_layer_role": "",
            "requested_candidate_label": "",
            "selected_layer_role": "",
            "selected_layer": "",
            "selected_candidate_label": "",
            "expected_layer_cadence_minutes": float("nan"),
            "stale_threshold_minutes": float("nan"),
            "is_stale_forecast": True,
            "forecast_value": float("nan"),
            "fallback_applied": True,
            "fallback_from_layer_role": "",
            "fallback_to_layer_role": "",
            "fallback_trigger": "unavailable",
            "fallback_reason": "no_available_layer",
            "resolution_path": "",
            "requested_forecast_value": float("nan"),
        }
    selected_layer_role = requested_layer_role
    fallback_from_layer_role = ""
    fallback_trigger = "none"
    resolution_steps: list[str] = []
    last_available_role = requested_layer_role
    last_available_value = requested_forecast_value
    current_role: str | None = requested_layer_role
    while current_role is not None:
        prediction_column = layer_column_map.get(str(current_role))
        forecast_value = pd.to_numeric(pd.Series([row.get(prediction_column)]), errors="coerce").iloc[0]
        if pd.isna(forecast_value):
            resolution_steps.append(f"{current_role}:missing")
            if not fallback_from_layer_role:
                fallback_from_layer_role = str(current_role)
                fallback_trigger = "unavailable"
            current_role = _optimizer_layer_fallback_target(str(current_role))
            continue
        if str(current_role) == "nowcast":
            dynamic_gate_enabled = bool(row.get("nowcast_dynamic_overlay_enforced", False))
            dynamic_gate_eligible = bool(row.get("nowcast_dynamic_overlay_eligible", True))
            if dynamic_gate_enabled and not dynamic_gate_eligible:
                resolution_steps.append("nowcast:dynamic_gate")
                if not fallback_from_layer_role:
                    fallback_from_layer_role = "nowcast"
                    fallback_trigger = "dynamic_gate"
                current_role = _optimizer_layer_fallback_target(str(current_role))
                continue
        last_available_role = str(current_role)
        last_available_value = float(forecast_value)
        stale_threshold_minutes = int(stale_threshold_by_layer[str(current_role)])
        is_stale = bool(age_minutes > stale_threshold_minutes)
        resolution_steps.append(f"{current_role}:{'stale' if is_stale else 'ready'}")
        if not is_stale:
            selected_layer_role = str(current_role)
            break
        if not fallback_from_layer_role:
            fallback_from_layer_role = str(current_role)
            fallback_trigger = "stale"
        current_role = _optimizer_layer_fallback_target(str(current_role))
    else:
        selected_layer_role = str(last_available_role)
    selected_prediction_column = layer_column_map.get(str(selected_layer_role))
    selected_forecast_value = pd.to_numeric(
        pd.Series([row.get(selected_prediction_column)]),
        errors="coerce",
    ).iloc[0]
    selected_stale_threshold = int(stale_threshold_by_layer[str(selected_layer_role)])
    selected_is_stale = bool(age_minutes > selected_stale_threshold)
    fallback_applied = str(selected_layer_role) != str(requested_layer_role)
    fallback_reason = _optimizer_fallback_reason_map().get(str(selected_layer_role), "unknown")
    if fallback_applied:
        fallback_reason = f"{fallback_from_layer_role}_{fallback_trigger}"
        if str(fallback_trigger) == "dynamic_gate":
            fallback_reason = (
                f"{fallback_reason}_{_policy_reason_token(row.get('nowcast_dynamic_overlay_reason', 'background_interval'))}"
            )
    return {
        "as_of_timestamp": as_of_ts,
        "effective_forecast_as_of": cycle_origin,
        "forecast_age_minutes": age_minutes,
        "requested_layer_role": str(requested_layer_role),
        "requested_candidate_label": str(candidate_labels.get(str(requested_layer_role), "")),
        "selected_layer_role": str(selected_layer_role),
        "selected_layer": _control_layer_label(str(selected_layer_role)),
        "selected_candidate_label": str(candidate_labels.get(str(selected_layer_role), "")),
        "expected_layer_cadence_minutes": int(cadence_by_layer[str(selected_layer_role)]),
        "stale_threshold_minutes": int(selected_stale_threshold),
        "is_stale_forecast": bool(selected_is_stale),
        "forecast_value": float(selected_forecast_value) if pd.notna(selected_forecast_value) else float("nan"),
        "fallback_applied": bool(fallback_applied),
        "fallback_from_layer_role": str(fallback_from_layer_role),
        "fallback_to_layer_role": str(selected_layer_role) if fallback_applied else "",
        "fallback_trigger": str(fallback_trigger),
        "fallback_reason": str(fallback_reason),
        "resolution_path": " > ".join(resolution_steps),
        "requested_forecast_value": float(requested_forecast_value),
    }


def _optimizer_confidence_score(
    *,
    band_width_95_pct: float,
    calibration_sample_n: float,
    quantile_source: str,
    layer_role: str,
    is_stale_forecast: bool,
) -> float:
    """Convert uncertainty width plus support into a simple operational trust score."""
    width_value = float(band_width_95_pct)
    width_component = 0.35
    width_scale = float(MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_band_width_scale_pct"])
    if np.isfinite(width_value) and width_value >= 0.0:
        width_component = 1.0 / (1.0 + (width_value / max(width_scale, 1e-9)))
    sample_value = float(calibration_sample_n)
    sample_component = 0.35
    full_support_n = float(MULTIRES_FORECAST_CONTROL["optimizer_delivery_confidence_full_support_n"])
    if np.isfinite(sample_value) and sample_value > 0.0:
        sample_component = min(sample_value / max(full_support_n, 1.0), 1.0)
    source_component = _optimizer_quantile_source_confidence_multiplier_map().get(
        str(quantile_source),
        0.5,
    )
    layer_component = _optimizer_layer_confidence_multiplier_map().get(str(layer_role), 0.6)
    freshness_component = 0.25 if bool(is_stale_forecast) else 1.0
    score = width_component * sample_component * source_component * layer_component * freshness_component
    return float(min(max(score, 0.0), 1.0))


def _optimizer_confidence_tier(score: float) -> str:
    """Bucket the trust score into a small set of digestible operational tiers."""
    score_value = float(score)
    if not np.isfinite(score_value):
        return "low"
    if score_value >= 0.75:
        return "high"
    if score_value >= 0.5:
        return "medium"
    return "low"


def _apply_optimizer_delivery_staleness(
    preview: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp | str,
) -> pd.DataFrame:
    """Apply wall-clock freshness checks to a delivery preview and refresh trust hints."""
    if preview.empty:
        return preview.copy()
    working = preview.copy()
    as_of_ts = pd.Timestamp(as_of_timestamp)
    previous_selected_layer = working.get("selected_layer_role", pd.Series("", index=working.index)).astype("string")
    layer_column_map = _optimizer_layer_prediction_column_map(working)
    candidate_label_columns = {
        "day_ahead": "day_ahead_candidate_label",
        "hourly": "hourly_candidate_label",
        "phase": "phase_candidate_label",
        "nowcast": "nowcast_candidate_label",
    }
    if layer_column_map and "cycle_origin_timestamp" in working.columns:
        resolution_rows = []
        cadence_by_layer = _optimizer_layer_cadence_minutes_map()
        stale_threshold_by_layer = _optimizer_layer_stale_threshold_minutes_map()
        for _, row in working.iterrows():
            candidate_labels = {
                layer_role: str(row.get(column_name, ""))
                for layer_role, column_name in candidate_label_columns.items()
            }
            resolution_rows.append(
                _optimizer_layer_resolution(
                    row,
                    layer_column_map=layer_column_map,
                    candidate_labels=candidate_labels,
                    cadence_by_layer=cadence_by_layer,
                    stale_threshold_by_layer=stale_threshold_by_layer,
                    as_of_timestamp=as_of_ts,
                )
            )
        resolution_frame = pd.DataFrame(resolution_rows, index=working.index)
        for column in resolution_frame.columns:
            working[column] = resolution_frame[column]
        selection_changed = previous_selected_layer.ne(working["selected_layer_role"].astype("string"))
        if "actual_interval_mean" in working.columns:
            working["selected_abs_error"] = (
                pd.to_numeric(working["forecast_value"], errors="coerce")
                - pd.to_numeric(working["actual_interval_mean"], errors="coerce")
            ).abs()
        if selection_changed.any():
            for column in (
                "forecast_lower_80",
                "forecast_upper_80",
                "forecast_lower_95",
                "forecast_upper_95",
                "uncertainty_band_width_80",
                "uncertainty_band_width_95",
                "uncertainty_band_width_95_pct",
                "raw_support_n",
                "calibration_sample_n",
                "residual_q025",
                "residual_q10",
                "residual_q90",
                "residual_q975",
                "scale_floor_value",
            ):
                if column in working.columns:
                    working.loc[selection_changed, column] = float("nan")
            if "quantile_source" in working.columns:
                working.loc[selection_changed, "quantile_source"] = "unavailable"
            if "residual_scale_mode" in working.columns:
                working.loc[selection_changed, "residual_scale_mode"] = ""
            for column in ("within_80_band", "within_95_band"):
                if column in working.columns:
                    working.loc[selection_changed, column] = False
    else:
        working["as_of_timestamp"] = as_of_ts
        effective_as_of = pd.to_datetime(working["effective_forecast_as_of"], errors="coerce")
        age_minutes = (as_of_ts - effective_as_of) / pd.Timedelta(minutes=1)
        working["forecast_age_minutes"] = np.maximum(pd.to_numeric(age_minutes, errors="coerce"), 0.0)
        working["is_stale_forecast"] = pd.to_numeric(
            working["forecast_age_minutes"],
            errors="coerce",
        ).gt(pd.to_numeric(working["stale_threshold_minutes"], errors="coerce"))
    confidence_inputs = working.loc[
        :,
        [
            "uncertainty_band_width_95_pct",
            "calibration_sample_n",
            "quantile_source",
            "selected_layer_role",
            "is_stale_forecast",
        ],
    ].to_dict(orient="records")
    working["confidence_score"] = [
        _optimizer_confidence_score(
            band_width_95_pct=float(row.get("uncertainty_band_width_95_pct", float("nan"))),
            calibration_sample_n=float(row.get("calibration_sample_n", float("nan"))),
            quantile_source=str(row.get("quantile_source", "unavailable")),
            layer_role=str(row.get("selected_layer_role", "")),
            is_stale_forecast=bool(row.get("is_stale_forecast", False)),
        )
        for row in confidence_inputs
    ]
    working["confidence_tier"] = working["confidence_score"].map(_optimizer_confidence_tier)
    return working


def _build_optimizer_delivery_preview(
    *,
    interval_timeline: pd.DataFrame,
    policy: dict[str, Any],
    uncertainty_calibration: pd.DataFrame,
    lock_interval_minutes: int,
    run_id: str,
    config_hash: str,
    delivery_as_of_timestamp: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Materialize a delivery-shaped interval preview from the selected Stage-10 stack."""
    if interval_timeline.empty:
        return pd.DataFrame()
    working = _add_control_interval_context(interval_timeline, lock_interval_minutes=lock_interval_minutes)
    cadence_by_layer = _optimizer_layer_cadence_minutes_map()
    stale_threshold_by_layer = _optimizer_layer_stale_threshold_minutes_map()
    candidate_labels = _selected_candidate_labels(policy)
    layer_specs = _control_interval_layer_specs(working)
    layer_column_map = _optimizer_layer_prediction_column_map(working)
    requested_peak_driver = pd.Series(float("nan"), index=working.index, dtype=float)
    interval_columns_by_role = {str(layer_role): str(column_name) for layer_role, column_name in layer_specs}
    for layer_role in OPTIMIZER_LAYER_PRIORITY:
        prediction_column = interval_columns_by_role.get(str(layer_role))
        if not prediction_column:
            continue
        prediction_values = pd.to_numeric(working[prediction_column], errors="coerce")
        fill_mask = requested_peak_driver.isna() & prediction_values.notna()
        if bool(fill_mask.any()):
            requested_peak_driver.loc[fill_mask] = prediction_values.loc[fill_mask].astype(float)
    working["requested_peak_driver_value"] = requested_peak_driver
    requested_peak_rank = working.groupby("cycle_origin_timestamp", dropna=False)["requested_peak_driver_value"].rank(
        method="first",
        ascending=False,
    )
    working["requested_is_predicted_peak_interval"] = requested_peak_rank.eq(1)
    working = _attach_nowcast_dynamic_overlay_policy(working, policy=policy)
    resolution_rows = [
        _optimizer_layer_resolution(
            row,
            layer_column_map=layer_column_map,
            candidate_labels=candidate_labels,
            cadence_by_layer=cadence_by_layer,
            stale_threshold_by_layer=stale_threshold_by_layer,
            as_of_timestamp=delivery_as_of_timestamp,
        )
        for _, row in working.iterrows()
    ]
    resolution_frame = pd.DataFrame(resolution_rows, index=working.index)
    working["producer_stage"] = "010_forecast_control"
    working["contract_version"] = OPTIMIZER_CONTRACT_VERSION
    working["run_id"] = str(run_id)
    working["config_hash"] = str(config_hash)
    for column in resolution_frame.columns:
        working[column] = resolution_frame[column]
    for layer_role, candidate_label in candidate_labels.items():
        working[f"{layer_role}_candidate_label"] = str(candidate_label)
    working["selected_abs_error"] = (
        pd.to_numeric(working["forecast_value"], errors="coerce")
        - pd.to_numeric(working["actual_interval_mean"], errors="coerce")
    ).abs()
    actual_peak_rank = working.groupby("cycle_origin_timestamp", dropna=False)["actual_interval_mean"].rank(
        method="first",
        ascending=False,
    )
    predicted_peak_rank = working.groupby("cycle_origin_timestamp", dropna=False)["forecast_value"].rank(
        method="first",
        ascending=False,
    )
    working["is_actual_peak_interval"] = actual_peak_rank.eq(1)
    working["is_predicted_peak_interval"] = predicted_peak_rank.eq(1)
    working["residual_scale_mode"] = ""
    working["scale_floor_value"] = float("nan")
    if not uncertainty_calibration.empty:
        calibration_table = uncertainty_calibration.copy()
        if "peak_context" not in calibration_table.columns:
            calibration_table["peak_context"] = "all"
        if "residual_scale_mode" not in calibration_table.columns:
            calibration_table["residual_scale_mode"] = ""
        if "scale_floor_value" not in calibration_table.columns:
            calibration_table["scale_floor_value"] = float("nan")
        lead_table = calibration_table.loc[
            calibration_table["lead_interval_index"].ge(0)
            & calibration_table["peak_context"].astype("string").eq("all")
        ].rename(
            columns={"layer_role": "selected_layer_role"}
        )
        global_table = calibration_table.loc[
            calibration_table["lead_interval_index"].eq(-1)
            & calibration_table["peak_context"].astype("string").eq("all")
        ].rename(
            columns={
                "layer_role": "selected_layer_role",
                "raw_support_n": "global_raw_support_n",
                "calibration_sample_n": "global_calibration_sample_n",
                "quantile_source": "global_quantile_source",
                "residual_q025": "global_residual_q025",
                "residual_q10": "global_residual_q10",
                "residual_q90": "global_residual_q90",
                "residual_q975": "global_residual_q975",
            }
        )
        next_lock_table = calibration_table.loc[
            calibration_table["lead_interval_index"].eq(-1)
            & calibration_table["peak_context"].astype("string").eq("next_lock")
            & ~calibration_table["quantile_source"].astype("string").eq("next_lock_scaled_global")
        ].rename(
            columns={
                "layer_role": "selected_layer_role",
                "raw_support_n": "next_lock_raw_support_n",
                "calibration_sample_n": "next_lock_calibration_sample_n",
                "quantile_source": "next_lock_quantile_source",
                "residual_q025": "next_lock_residual_q025",
                "residual_q10": "next_lock_residual_q10",
                "residual_q90": "next_lock_residual_q90",
                "residual_q975": "next_lock_residual_q975",
            }
        )
        next_lock_scaled_table = calibration_table.loc[
            calibration_table["lead_interval_index"].eq(-1)
            & calibration_table["peak_context"].astype("string").eq("next_lock")
            & calibration_table["quantile_source"].astype("string").eq("next_lock_scaled_global")
        ].rename(
            columns={
                "layer_role": "selected_layer_role",
                "raw_support_n": "next_lock_scaled_raw_support_n",
                "calibration_sample_n": "next_lock_scaled_calibration_sample_n",
                "quantile_source": "next_lock_scaled_quantile_source",
                "residual_q025": "next_lock_scaled_residual_q025",
                "residual_q10": "next_lock_scaled_residual_q10",
                "residual_q90": "next_lock_scaled_residual_q90",
                "residual_q975": "next_lock_scaled_residual_q975",
                "residual_scale_mode": "next_lock_scaled_residual_scale_mode",
                "scale_floor_value": "next_lock_scaled_scale_floor_value",
            }
        )
        predicted_peak_lead_table = calibration_table.loc[
            calibration_table["lead_interval_index"].ge(0)
            & calibration_table["peak_context"].astype("string").eq("predicted_peak")
        ].rename(
            columns={
                "layer_role": "selected_layer_role",
                "raw_support_n": "predicted_peak_lead_raw_support_n",
                "calibration_sample_n": "predicted_peak_lead_calibration_sample_n",
                "quantile_source": "predicted_peak_lead_quantile_source",
                "residual_q025": "predicted_peak_lead_residual_q025",
                "residual_q10": "predicted_peak_lead_residual_q10",
                "residual_q90": "predicted_peak_lead_residual_q90",
                "residual_q975": "predicted_peak_lead_residual_q975",
            }
        )
        predicted_peak_table = calibration_table.loc[
            calibration_table["lead_interval_index"].eq(-1)
            & calibration_table["peak_context"].astype("string").eq("predicted_peak")
        ].rename(
            columns={
                "layer_role": "selected_layer_role",
                "raw_support_n": "predicted_peak_raw_support_n",
                "calibration_sample_n": "predicted_peak_calibration_sample_n",
                "quantile_source": "predicted_peak_quantile_source",
                "residual_q025": "predicted_peak_residual_q025",
                "residual_q10": "predicted_peak_residual_q10",
                "residual_q90": "predicted_peak_residual_q90",
                "residual_q975": "predicted_peak_residual_q975",
            }
        )
        working = working.merge(
            lead_table,
            on=["selected_layer_role", "lead_interval_index"],
            how="left",
        )
        working = working.merge(
            global_table.loc[
                :,
                [
                    "selected_layer_role",
                    "global_raw_support_n",
                    "global_calibration_sample_n",
                    "global_quantile_source",
                    "global_residual_q025",
                    "global_residual_q10",
                    "global_residual_q90",
                    "global_residual_q975",
                ],
            ],
            on="selected_layer_role",
            how="left",
        )
        working = working.merge(
            next_lock_table.loc[
                :,
                [
                    "selected_layer_role",
                    "next_lock_raw_support_n",
                    "next_lock_calibration_sample_n",
                    "next_lock_quantile_source",
                    "next_lock_residual_q025",
                    "next_lock_residual_q10",
                    "next_lock_residual_q90",
                    "next_lock_residual_q975",
                ],
            ],
            on="selected_layer_role",
            how="left",
        )
        working = working.merge(
            next_lock_scaled_table.loc[
                :,
                [
                    "selected_layer_role",
                    "next_lock_scaled_raw_support_n",
                    "next_lock_scaled_calibration_sample_n",
                    "next_lock_scaled_quantile_source",
                    "next_lock_scaled_residual_q025",
                    "next_lock_scaled_residual_q10",
                    "next_lock_scaled_residual_q90",
                    "next_lock_scaled_residual_q975",
                    "next_lock_scaled_residual_scale_mode",
                    "next_lock_scaled_scale_floor_value",
                ],
            ],
            on="selected_layer_role",
            how="left",
        )
        working = working.merge(
            predicted_peak_lead_table.loc[
                :,
                [
                    "selected_layer_role",
                    "lead_interval_index",
                    "predicted_peak_lead_raw_support_n",
                    "predicted_peak_lead_calibration_sample_n",
                    "predicted_peak_lead_quantile_source",
                    "predicted_peak_lead_residual_q025",
                    "predicted_peak_lead_residual_q10",
                    "predicted_peak_lead_residual_q90",
                    "predicted_peak_lead_residual_q975",
                ],
            ],
            on=["selected_layer_role", "lead_interval_index"],
            how="left",
        )
        working = working.merge(
            predicted_peak_table.loc[
                :,
                [
                    "selected_layer_role",
                    "predicted_peak_raw_support_n",
                    "predicted_peak_calibration_sample_n",
                    "predicted_peak_quantile_source",
                    "predicted_peak_residual_q025",
                    "predicted_peak_residual_q10",
                    "predicted_peak_residual_q90",
                    "predicted_peak_residual_q975",
                ],
            ],
            on="selected_layer_role",
            how="left",
        )
        for column_name, global_column_name in (
            ("raw_support_n", "global_raw_support_n"),
            ("calibration_sample_n", "global_calibration_sample_n"),
            ("quantile_source", "global_quantile_source"),
            ("residual_q025", "global_residual_q025"),
            ("residual_q10", "global_residual_q10"),
            ("residual_q90", "global_residual_q90"),
            ("residual_q975", "global_residual_q975"),
        ):
            working[column_name] = working[column_name].where(working[column_name].notna(), working[global_column_name])
        next_lock_scaled_override_mask = (
            working["is_next_lock_interval"].astype(bool)
            & working["next_lock_scaled_calibration_sample_n"].notna()
        )
        for column_name, next_lock_scaled_column_name in (
            ("raw_support_n", "next_lock_scaled_raw_support_n"),
            ("calibration_sample_n", "next_lock_scaled_calibration_sample_n"),
            ("quantile_source", "next_lock_scaled_quantile_source"),
            ("residual_q025", "next_lock_scaled_residual_q025"),
            ("residual_q10", "next_lock_scaled_residual_q10"),
            ("residual_q90", "next_lock_scaled_residual_q90"),
            ("residual_q975", "next_lock_scaled_residual_q975"),
            ("residual_scale_mode", "next_lock_scaled_residual_scale_mode"),
            ("scale_floor_value", "next_lock_scaled_scale_floor_value"),
        ):
            working.loc[next_lock_scaled_override_mask, column_name] = working.loc[
                next_lock_scaled_override_mask,
                next_lock_scaled_column_name,
            ]
        next_lock_override_mask = (
            working["is_next_lock_interval"].astype(bool)
            & ~next_lock_scaled_override_mask
            & working["next_lock_calibration_sample_n"].notna()
            & working["quantile_source"].astype("string").isin({"layer_global", "layer_global_fallback"})
        )
        for column_name, next_lock_column_name in (
            ("raw_support_n", "next_lock_raw_support_n"),
            ("calibration_sample_n", "next_lock_calibration_sample_n"),
            ("quantile_source", "next_lock_quantile_source"),
            ("residual_q025", "next_lock_residual_q025"),
            ("residual_q10", "next_lock_residual_q10"),
            ("residual_q90", "next_lock_residual_q90"),
            ("residual_q975", "next_lock_residual_q975"),
        ):
            working.loc[next_lock_override_mask, column_name] = working.loc[
                next_lock_override_mask,
                next_lock_column_name,
            ]
        working.loc[next_lock_override_mask, "residual_scale_mode"] = ""
        working.loc[next_lock_override_mask, "scale_floor_value"] = float("nan")
        predicted_peak_lead_override_mask = (
            working["is_predicted_peak_interval"].astype(bool)
            & working["predicted_peak_lead_calibration_sample_n"].notna()
        )
        for column_name, peak_column_name in (
            ("raw_support_n", "predicted_peak_lead_raw_support_n"),
            ("calibration_sample_n", "predicted_peak_lead_calibration_sample_n"),
            ("quantile_source", "predicted_peak_lead_quantile_source"),
            ("residual_q025", "predicted_peak_lead_residual_q025"),
            ("residual_q10", "predicted_peak_lead_residual_q10"),
            ("residual_q90", "predicted_peak_lead_residual_q90"),
            ("residual_q975", "predicted_peak_lead_residual_q975"),
        ):
            working.loc[predicted_peak_lead_override_mask, column_name] = working.loc[
                predicted_peak_lead_override_mask,
                peak_column_name,
            ]
        working.loc[predicted_peak_lead_override_mask, "residual_scale_mode"] = ""
        working.loc[predicted_peak_lead_override_mask, "scale_floor_value"] = float("nan")
        predicted_peak_override_mask = (
            working["is_predicted_peak_interval"].astype(bool)
            & ~predicted_peak_lead_override_mask
            & working["predicted_peak_calibration_sample_n"].notna()
        )
        for column_name, peak_column_name in (
            ("raw_support_n", "predicted_peak_raw_support_n"),
            ("calibration_sample_n", "predicted_peak_calibration_sample_n"),
            ("quantile_source", "predicted_peak_quantile_source"),
            ("residual_q025", "predicted_peak_residual_q025"),
            ("residual_q10", "predicted_peak_residual_q10"),
            ("residual_q90", "predicted_peak_residual_q90"),
            ("residual_q975", "predicted_peak_residual_q975"),
        ):
            working.loc[predicted_peak_override_mask, column_name] = working.loc[
                predicted_peak_override_mask,
                peak_column_name,
            ]
        working.loc[predicted_peak_override_mask, "residual_scale_mode"] = ""
        working.loc[predicted_peak_override_mask, "scale_floor_value"] = float("nan")
    else:
        working["raw_support_n"] = float("nan")
        working["calibration_sample_n"] = float("nan")
        working["quantile_source"] = "unavailable"
        working["residual_q025"] = float("nan")
        working["residual_q10"] = float("nan")
        working["residual_q90"] = float("nan")
        working["residual_q975"] = float("nan")
        working["residual_scale_mode"] = ""
        working["scale_floor_value"] = float("nan")
    working["residual_scale_mode"] = working["residual_scale_mode"].fillna("")
    forecast_value = pd.to_numeric(working["forecast_value"], errors="coerce")
    scale_floor_value = pd.to_numeric(working["scale_floor_value"], errors="coerce")
    residual_scale = np.maximum(forecast_value.abs(), scale_floor_value)
    scaled_quantile_mask = working["residual_scale_mode"].fillna("").astype("string").eq(
        "absolute_prediction_floor"
    )
    lower_80 = forecast_value + pd.to_numeric(working["residual_q10"], errors="coerce")
    upper_80 = forecast_value + pd.to_numeric(working["residual_q90"], errors="coerce")
    lower_95 = forecast_value + pd.to_numeric(working["residual_q025"], errors="coerce")
    upper_95 = forecast_value + pd.to_numeric(working["residual_q975"], errors="coerce")
    lower_80 = lower_80.where(
        ~scaled_quantile_mask,
        forecast_value + pd.to_numeric(working["residual_q10"], errors="coerce") * residual_scale,
    )
    upper_80 = upper_80.where(
        ~scaled_quantile_mask,
        forecast_value + pd.to_numeric(working["residual_q90"], errors="coerce") * residual_scale,
    )
    lower_95 = lower_95.where(
        ~scaled_quantile_mask,
        forecast_value + pd.to_numeric(working["residual_q025"], errors="coerce") * residual_scale,
    )
    upper_95 = upper_95.where(
        ~scaled_quantile_mask,
        forecast_value + pd.to_numeric(working["residual_q975"], errors="coerce") * residual_scale,
    )
    working["forecast_lower_80"] = lower_80
    working["forecast_upper_80"] = upper_80
    working["forecast_lower_95"] = lower_95
    working["forecast_upper_95"] = upper_95
    working["uncertainty_band_width_80"] = working["forecast_upper_80"] - working["forecast_lower_80"]
    working["uncertainty_band_width_95"] = working["forecast_upper_95"] - working["forecast_lower_95"]
    forecast_abs = pd.to_numeric(working["forecast_value"], errors="coerce").abs()
    working["uncertainty_band_width_95_pct"] = np.where(
        forecast_abs > 0.0,
        100.0 * pd.to_numeric(working["uncertainty_band_width_95"], errors="coerce") / forecast_abs,
        float("nan"),
    )
    confidence_inputs = working.loc[
        :,
        [
            "uncertainty_band_width_95_pct",
            "calibration_sample_n",
            "quantile_source",
            "selected_layer_role",
            "is_stale_forecast",
        ],
    ].to_dict(orient="records")
    working["confidence_score"] = [
        _optimizer_confidence_score(
            band_width_95_pct=float(row.get("uncertainty_band_width_95_pct", float("nan"))),
            calibration_sample_n=float(row.get("calibration_sample_n", float("nan"))),
            quantile_source=str(row.get("quantile_source", "unavailable")),
            layer_role=str(row.get("selected_layer_role", "")),
            is_stale_forecast=bool(row.get("is_stale_forecast", False)),
        )
        for row in confidence_inputs
    ]
    working["confidence_tier"] = working["confidence_score"].map(_optimizer_confidence_tier)
    actual_interval = pd.to_numeric(working["actual_interval_mean"], errors="coerce")
    working["within_80_band"] = (
        actual_interval.ge(pd.to_numeric(working["forecast_lower_80"], errors="coerce"))
        & actual_interval.le(pd.to_numeric(working["forecast_upper_80"], errors="coerce"))
    )
    working["within_95_band"] = (
        actual_interval.ge(pd.to_numeric(working["forecast_lower_95"], errors="coerce"))
        & actual_interval.le(pd.to_numeric(working["forecast_upper_95"], errors="coerce"))
    )
    ordered_columns = [
        "producer_stage",
        "contract_version",
        "run_id",
        "config_hash",
        "cycle_origin_timestamp",
        "as_of_timestamp",
        "interval_start",
        "interval_end",
        "lead_interval_index",
        "horizon_minutes",
        "is_next_lock_interval",
        "requested_is_predicted_peak_interval",
        "operating_regime",
        "actual_ramp_band",
        "high_ramp_fraction",
        "effective_forecast_as_of",
        "requested_layer_role",
        "requested_candidate_label",
        "nowcast_dynamic_overlay_enabled",
        "nowcast_dynamic_overlay_enforced",
        "nowcast_dynamic_overlay_eligible",
        "nowcast_dynamic_overlay_reason",
        "selected_layer_role",
        "selected_layer",
        "selected_candidate_label",
        "expected_layer_cadence_minutes",
        "forecast_age_minutes",
        "stale_threshold_minutes",
        "is_stale_forecast",
        "fallback_applied",
        "fallback_from_layer_role",
        "fallback_to_layer_role",
        "fallback_trigger",
        "fallback_reason",
        "resolution_path",
        "forecast_value",
        "requested_forecast_value",
        "actual_interval_mean",
        "selected_abs_error",
        "forecast_lower_80",
        "forecast_upper_80",
        "forecast_lower_95",
        "forecast_upper_95",
        "uncertainty_band_width_80",
        "uncertainty_band_width_95",
        "uncertainty_band_width_95_pct",
        "quantile_source",
        "residual_scale_mode",
        "scale_floor_value",
        "calibration_sample_n",
        "raw_support_n",
        "confidence_score",
        "confidence_tier",
        "within_80_band",
        "within_95_band",
        "is_actual_peak_interval",
        "is_predicted_peak_interval",
        "day_ahead_candidate_label",
        "hourly_candidate_label",
        "phase_candidate_label",
        "nowcast_candidate_label",
    ]
    prediction_columns = [column for _, column in layer_specs]
    keep_columns = [column for column in ordered_columns + prediction_columns if column in working.columns]
    preview = working.loc[:, keep_columns].copy()
    return preview


def _summarize_optimizer_delivery_uncertainty(preview: pd.DataFrame) -> pd.DataFrame:
    """Summarize empirical interval-band behavior on the held-out evaluation replay."""
    if preview.empty:
        return pd.DataFrame()

    def _scope_row(scope_name: str, frame: pd.DataFrame) -> dict[str, Any] | None:
        if frame.empty:
            return None
        return {
            "scope": str(scope_name),
            "row_n": int(len(frame)),
            "interval_80_coverage": float(frame["within_80_band"].astype(float).mean()),
            "interval_95_coverage": float(frame["within_95_band"].astype(float).mean()),
            "mean_abs_error": float(pd.to_numeric(frame["selected_abs_error"], errors="coerce").mean()),
            "mean_band_width_80": float(pd.to_numeric(frame["uncertainty_band_width_80"], errors="coerce").mean()),
            "mean_band_width_95": float(pd.to_numeric(frame["uncertainty_band_width_95"], errors="coerce").mean()),
            "mean_band_width_95_pct": float(
                pd.to_numeric(frame["uncertainty_band_width_95_pct"], errors="coerce").mean()
            ),
            "mean_calibration_sample_n": float(pd.to_numeric(frame["calibration_sample_n"], errors="coerce").mean()),
            "lead_specific_quantile_rate": float(
                frame["quantile_source"].astype("string").isin(
                    {"lead_interval", "predicted_peak_lead_interval"}
                ).mean()
            ),
            "context_specific_quantile_rate": float(
                frame["quantile_source"].astype("string").isin(
                    {
                        "lead_interval",
                        "next_lock_scaled_global",
                        "predicted_peak_lead_interval",
                        "predicted_peak_global",
                        "next_lock_global",
                    }
                ).mean()
            ),
        }

    rows: list[dict[str, Any]] = []
    for scope_name, mask in (
        ("all_intervals", pd.Series(True, index=preview.index)),
        ("next_lock_intervals", preview["is_next_lock_interval"].astype(bool)),
        ("actual_peak_intervals", preview["is_actual_peak_interval"].astype(bool)),
        ("predicted_peak_intervals", preview["is_predicted_peak_interval"].astype(bool)),
    ):
        row = _scope_row(scope_name, preview.loc[mask].copy())
        if row is not None:
            rows.append(row)
    for layer_role, layer_frame in preview.groupby("selected_layer_role", sort=True):
        row = _scope_row(f"layer::{layer_role}", layer_frame.copy())
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def _build_optimizer_delivery_serving_preview(preview: pd.DataFrame) -> pd.DataFrame:
    """Strip replay-only truth columns from the delivery preview to expose a serving-shaped payload."""
    if preview.empty:
        return preview.copy()
    serving_columns = [
        "producer_stage",
        "contract_version",
        "run_id",
        "config_hash",
        "cycle_origin_timestamp",
        "as_of_timestamp",
        "interval_start",
        "interval_end",
        "lead_interval_index",
        "horizon_minutes",
        "is_next_lock_interval",
        "requested_is_predicted_peak_interval",
        "operating_regime",
        "actual_ramp_band",
        "high_ramp_fraction",
        "effective_forecast_as_of",
        "requested_layer_role",
        "requested_candidate_label",
        "nowcast_dynamic_overlay_enabled",
        "nowcast_dynamic_overlay_enforced",
        "nowcast_dynamic_overlay_eligible",
        "nowcast_dynamic_overlay_reason",
        "selected_layer_role",
        "selected_layer",
        "selected_candidate_label",
        "expected_layer_cadence_minutes",
        "forecast_age_minutes",
        "stale_threshold_minutes",
        "is_stale_forecast",
        "fallback_applied",
        "fallback_from_layer_role",
        "fallback_to_layer_role",
        "fallback_trigger",
        "fallback_reason",
        "resolution_path",
        "forecast_value",
        "forecast_lower_80",
        "forecast_upper_80",
        "forecast_lower_95",
        "forecast_upper_95",
        "uncertainty_band_width_80",
        "uncertainty_band_width_95",
        "uncertainty_band_width_95_pct",
        "quantile_source",
        "calibration_sample_n",
        "raw_support_n",
        "confidence_score",
        "confidence_tier",
    ]
    keep_columns = [column for column in serving_columns if column in preview.columns]
    return preview.loc[:, keep_columns].copy()


def _reresolve_optimizer_delivery_preview(
    preview: pd.DataFrame,
    *,
    enforce_dynamic_overlay: bool,
) -> pd.DataFrame:
    """Re-resolve delivery rows under a requested dynamic-overlay enforcement mode."""
    if preview.empty:
        return preview.copy()
    working = preview.copy()
    if "nowcast_dynamic_overlay_enabled" not in working.columns:
        working["nowcast_dynamic_overlay_enabled"] = False
    if "nowcast_dynamic_overlay_enforced" not in working.columns:
        working["nowcast_dynamic_overlay_enforced"] = False
    if "nowcast_dynamic_overlay_eligible" not in working.columns:
        working["nowcast_dynamic_overlay_eligible"] = True
    if "nowcast_dynamic_overlay_reason" not in working.columns:
        working["nowcast_dynamic_overlay_reason"] = "background_interval"
    working["nowcast_dynamic_overlay_enforced"] = (
        working["nowcast_dynamic_overlay_enabled"].astype(bool) & bool(enforce_dynamic_overlay)
    )
    layer_column_map = _optimizer_layer_prediction_column_map(working)
    candidate_label_columns = {
        "day_ahead": "day_ahead_candidate_label",
        "hourly": "hourly_candidate_label",
        "phase": "phase_candidate_label",
        "nowcast": "nowcast_candidate_label",
    }
    resolution_rows = []
    cadence_by_layer = _optimizer_layer_cadence_minutes_map()
    stale_threshold_by_layer = _optimizer_layer_stale_threshold_minutes_map()
    for _, row in working.iterrows():
        candidate_labels = {
            layer_role: str(row.get(column_name, ""))
            for layer_role, column_name in candidate_label_columns.items()
        }
        resolution_rows.append(
            _optimizer_layer_resolution(
                row,
                layer_column_map=layer_column_map,
                candidate_labels=candidate_labels,
                cadence_by_layer=cadence_by_layer,
                stale_threshold_by_layer=stale_threshold_by_layer,
                as_of_timestamp=row.get("as_of_timestamp"),
            )
        )
    resolution_frame = pd.DataFrame(resolution_rows, index=working.index)
    for column in resolution_frame.columns:
        working[column] = resolution_frame[column]
    if "actual_interval_mean" in working.columns:
        working["selected_abs_error"] = (
            pd.to_numeric(working["forecast_value"], errors="coerce")
            - pd.to_numeric(working["actual_interval_mean"], errors="coerce")
        ).abs()
    if "forecast_value" in working.columns:
        predicted_peak_rank = working.groupby("cycle_origin_timestamp", dropna=False)["forecast_value"].rank(
            method="first",
            ascending=False,
        )
        working["is_predicted_peak_interval"] = predicted_peak_rank.eq(1)
    return working


def _optimizer_dynamic_soft_overlay_bucket(preview: pd.DataFrame) -> pd.Series:
    """Bucket each interval into the strategic/supported/background soft-overlay policy groups."""
    index = preview.index
    enabled_mask = preview.get("nowcast_dynamic_overlay_enabled", pd.Series(False, index=index)).astype(bool)
    requested_layer = preview.get(
        "requested_layer_role",
        preview.get("selected_layer_role", pd.Series("", index=index)),
    ).astype("string")
    applicable_mask = enabled_mask & requested_layer.eq("nowcast")
    strategic_mask = pd.Series(False, index=index, dtype=bool)
    if bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_next_lock"]):
        strategic_mask = strategic_mask | preview.get(
            "is_next_lock_interval",
            pd.Series(False, index=index),
        ).astype(bool)
    if bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_predicted_peak"]):
        strategic_mask = strategic_mask | preview.get(
            "requested_is_predicted_peak_interval",
            pd.Series(False, index=index),
        ).astype(bool)
    eligible_mask = preview.get("nowcast_dynamic_overlay_eligible", pd.Series(True, index=index)).astype(bool)
    bucket = pd.Series("not_applicable", index=index, dtype="string")
    bucket.loc[requested_layer.eq("nowcast") & ~enabled_mask] = "disabled"
    bucket.loc[applicable_mask & ~eligible_mask] = "background"
    bucket.loc[applicable_mask & eligible_mask] = "supported"
    bucket.loc[applicable_mask & strategic_mask] = "strategic"
    return bucket.astype("string")


def _optimizer_dynamic_soft_overlay_upstream_forecast(preview: pd.DataFrame) -> pd.Series:
    """Resolve the upstream interval forecast that the minute overlay should blend against."""
    upstream = pd.Series(float("nan"), index=preview.index, dtype=float)
    for column_name in ("phase_interval_mean", "hourly_interval_mean", "day_ahead_interval_mean"):
        if column_name not in preview.columns:
            continue
        column = pd.to_numeric(preview[column_name], errors="coerce")
        upstream = upstream.where(upstream.notna(), column)
    if "forecast_value" in preview.columns:
        upstream = upstream.where(
            upstream.notna(),
            pd.to_numeric(preview["forecast_value"], errors="coerce"),
        )
    return upstream.astype(float)


def _optimizer_dynamic_soft_overlay_policy_label(
    *,
    supported_weight: float,
    background_weight: float,
) -> str:
    """Render one soft minute-overlay candidate label for artifacts and auditability."""
    supported_token = int(round(float(supported_weight) * 100.0))
    background_token = int(round(float(background_weight) * 100.0))
    return f"soft_overlay_sw{supported_token:03d}_bw{background_token:03d}"


def _apply_optimizer_dynamic_soft_overlay_candidate(
    preview: pd.DataFrame,
    *,
    supported_weight: float,
    background_weight: float,
) -> pd.DataFrame:
    """Apply one soft minute-overlay counterfactual onto the delivery preview."""
    if preview.empty:
        return preview.copy()
    working = preview.copy()
    bucket = _optimizer_dynamic_soft_overlay_bucket(working)
    nowcast_forecast = pd.to_numeric(
        working.get("nowcast_interval_mean", pd.Series(float("nan"), index=working.index)),
        errors="coerce",
    )
    upstream_forecast = _optimizer_dynamic_soft_overlay_upstream_forecast(working)
    base_forecast = pd.to_numeric(
        working.get("forecast_value", pd.Series(float("nan"), index=working.index)),
        errors="coerce",
    )
    soft_weight = pd.Series(1.0, index=working.index, dtype=float)
    soft_weight.loc[bucket.eq("supported")] = float(supported_weight)
    soft_weight.loc[bucket.eq("background")] = float(background_weight)
    soft_weight.loc[bucket.eq("strategic")] = 1.0
    soft_forecast = base_forecast.copy()
    candidate_mask = bucket.isin(["strategic", "supported", "background"])
    blend_mask = candidate_mask & nowcast_forecast.notna() & upstream_forecast.notna()
    soft_forecast.loc[blend_mask] = (
        (1.0 - soft_weight.loc[blend_mask]) * upstream_forecast.loc[blend_mask]
        + soft_weight.loc[blend_mask] * nowcast_forecast.loc[blend_mask]
    )
    nowcast_only_mask = candidate_mask & nowcast_forecast.notna() & upstream_forecast.isna()
    soft_forecast.loc[nowcast_only_mask] = nowcast_forecast.loc[nowcast_only_mask]
    upstream_only_mask = candidate_mask & nowcast_forecast.isna() & upstream_forecast.notna()
    soft_forecast.loc[upstream_only_mask] = upstream_forecast.loc[upstream_only_mask]
    soft_forecast = soft_forecast.where(soft_forecast.notna(), nowcast_forecast)
    soft_forecast = soft_forecast.where(soft_forecast.notna(), upstream_forecast)
    working["nowcast_dynamic_soft_bucket"] = bucket.astype("string")
    working["nowcast_dynamic_soft_weight"] = soft_weight.astype(float)
    working["nowcast_dynamic_soft_supported_weight"] = float(supported_weight)
    working["nowcast_dynamic_soft_background_weight"] = float(background_weight)
    working["nowcast_dynamic_soft_policy_label"] = _optimizer_dynamic_soft_overlay_policy_label(
        supported_weight=float(supported_weight),
        background_weight=float(background_weight),
    )
    working["forecast_value"] = soft_forecast.astype(float)
    if "actual_interval_mean" in working.columns:
        working["selected_abs_error"] = (
            pd.to_numeric(working["forecast_value"], errors="coerce")
            - pd.to_numeric(working["actual_interval_mean"], errors="coerce")
        ).abs()
    if "forecast_value" in working.columns and "cycle_origin_timestamp" in working.columns:
        predicted_peak_rank = working.groupby("cycle_origin_timestamp", dropna=False)["forecast_value"].rank(
            method="first",
            ascending=False,
        )
        working["is_predicted_peak_interval"] = predicted_peak_rank.eq(1)
    return working


def _evaluate_optimizer_dynamic_soft_overlay_candidates(preview: pd.DataFrame) -> pd.DataFrame:
    """Benchmark soft minute-overlay candidates against the pure-nowcast shadow surface."""
    if preview.empty or not bool(MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_shadow_enabled"]):
        return pd.DataFrame()
    if "nowcast_interval_mean" not in preview.columns:
        return pd.DataFrame()
    bucket = _optimizer_dynamic_soft_overlay_bucket(preview)
    applicable_mask = bucket.isin(["strategic", "supported", "background"])
    if not bool(applicable_mask.any()):
        return pd.DataFrame()
    shadow_metrics = _optimizer_delivery_shadow_metrics(preview)
    shadow_mean_abs_error = float(shadow_metrics.get("mean_selected_abs_error", float("nan")))
    shadow_next_lock_mae = float(shadow_metrics.get("next_lock_mae", float("nan")))
    shadow_peak_hit_rate = float(shadow_metrics.get("peak_hit_rate", float("nan")))
    thresholds = _optimizer_dynamic_soft_overlay_thresholds()
    rows: list[dict[str, Any]] = []
    for supported_weight, background_weight in _optimizer_dynamic_soft_overlay_grid():
        candidate_preview = _apply_optimizer_dynamic_soft_overlay_candidate(
            preview,
            supported_weight=float(supported_weight),
            background_weight=float(background_weight),
        )
        metrics = _optimizer_delivery_shadow_metrics(candidate_preview)
        candidate_mean_abs_error = float(metrics.get("mean_selected_abs_error", float("nan")))
        candidate_next_lock_mae = float(metrics.get("next_lock_mae", float("nan")))
        candidate_peak_hit_rate = float(metrics.get("peak_hit_rate", float("nan")))
        delta_mean_abs_error = candidate_mean_abs_error - shadow_mean_abs_error
        delta_next_lock_mae = candidate_next_lock_mae - shadow_next_lock_mae
        delta_peak_hit_rate = candidate_peak_hit_rate - shadow_peak_hit_rate
        next_lock_regress_pct = float("nan")
        if np.isfinite(candidate_next_lock_mae) and np.isfinite(shadow_next_lock_mae):
            if shadow_next_lock_mae > 0.0:
                next_lock_regress_pct = float(max(delta_next_lock_mae, 0.0) / shadow_next_lock_mae)
            else:
                next_lock_regress_pct = float(max(delta_next_lock_mae, 0.0))
        peak_hit_regress = float("nan")
        if np.isfinite(candidate_peak_hit_rate) and np.isfinite(shadow_peak_hit_rate):
            peak_hit_regress = float(max(shadow_peak_hit_rate - candidate_peak_hit_rate, 0.0))
        meets_next_lock_rule = (
            (not np.isfinite(next_lock_regress_pct))
            or next_lock_regress_pct <= float(thresholds["max_next_lock_regress_pct"])
        )
        meets_peak_hit_rule = (
            (not np.isfinite(peak_hit_regress))
            or peak_hit_regress <= float(thresholds["max_peak_hit_regress"])
        )
        meets_non_regression_rules = bool(meets_next_lock_rule and meets_peak_hit_rule)
        beats_shadow = bool(
            meets_non_regression_rules
            and np.isfinite(delta_mean_abs_error)
            and delta_mean_abs_error < 0.0
        )
        soft_bucket = candidate_preview.get(
            "nowcast_dynamic_soft_bucket",
            pd.Series("not_applicable", index=candidate_preview.index, dtype="string"),
        ).astype("string")
        soft_weight = pd.to_numeric(
            candidate_preview.get("nowcast_dynamic_soft_weight", pd.Series(float("nan"), index=candidate_preview.index)),
            errors="coerce",
        )
        rows.append(
            {
                "soft_policy_label": _optimizer_dynamic_soft_overlay_policy_label(
                    supported_weight=float(supported_weight),
                    background_weight=float(background_weight),
                ),
                "supported_weight": float(supported_weight),
                "background_weight": float(background_weight),
                "row_count": int(len(candidate_preview)),
                "cycle_count": int(
                    candidate_preview.get("cycle_origin_timestamp", pd.Series(dtype="string")).nunique(dropna=True)
                ),
                "applicable_row_count": int(applicable_mask.sum()),
                "strategic_row_count": int(soft_bucket.eq("strategic").sum()),
                "supported_row_count": int(soft_bucket.eq("supported").sum()),
                "background_row_count": int(soft_bucket.eq("background").sum()),
                "mean_soft_weight": float(soft_weight.mean()) if soft_weight.notna().any() else float("nan"),
                "mean_selected_abs_error": float(candidate_mean_abs_error),
                "next_lock_mae": float(candidate_next_lock_mae),
                "peak_hit_rate": float(candidate_peak_hit_rate),
                "delta_mean_selected_abs_error_vs_shadow": float(delta_mean_abs_error),
                "delta_next_lock_mae_vs_shadow": float(delta_next_lock_mae),
                "delta_peak_hit_rate_vs_shadow": float(delta_peak_hit_rate),
                "next_lock_regress_pct_vs_shadow": float(next_lock_regress_pct),
                "peak_hit_regress_vs_shadow": float(peak_hit_regress),
                "meets_next_lock_rule": bool(meets_next_lock_rule),
                "meets_peak_hit_rule": bool(meets_peak_hit_rule),
                "meets_non_regression_rules": bool(meets_non_regression_rules),
                "beats_shadow": bool(beats_shadow),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        [
            "meets_non_regression_rules",
            "beats_shadow",
            "mean_selected_abs_error",
            "next_lock_mae",
            "peak_hit_rate",
            "supported_weight",
            "background_weight",
        ],
        ascending=[False, False, True, True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1, dtype=int)
    return candidates


def _optimizer_dynamic_soft_overlay_candidate_record(row: pd.Series) -> dict[str, Any]:
    """Convert one soft-overlay candidate row into a JSON-safe summary record."""
    payload: dict[str, Any] = {}
    for column_name in (
        "candidate_rank",
        "soft_policy_label",
        "supported_weight",
        "background_weight",
        "mean_soft_weight",
        "mean_selected_abs_error",
        "next_lock_mae",
        "peak_hit_rate",
        "delta_mean_selected_abs_error_vs_shadow",
        "delta_next_lock_mae_vs_shadow",
        "delta_peak_hit_rate_vs_shadow",
        "next_lock_regress_pct_vs_shadow",
        "peak_hit_regress_vs_shadow",
        "meets_next_lock_rule",
        "meets_peak_hit_rule",
        "meets_non_regression_rules",
        "beats_shadow",
        "strategic_row_count",
        "supported_row_count",
        "background_row_count",
    ):
        value = row.get(column_name)
        if pd.isna(value):
            payload[column_name] = None
        elif isinstance(value, (np.bool_, bool)):
            payload[column_name] = bool(value)
        elif isinstance(value, (np.integer, int)):
            payload[column_name] = int(value)
        elif isinstance(value, (np.floating, float)):
            payload[column_name] = float(value)
        else:
            payload[column_name] = str(value)
    return payload


def _build_optimizer_dynamic_overlay_soft_summary(
    preview: pd.DataFrame,
    soft_candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize the soft minute-overlay shadow search on the Stage-10 delivery surface."""
    if preview.empty:
        return {
            "enabled": False,
            "shadow_enabled": bool(MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_shadow_enabled"]),
            "recommendation": "no_delivery_preview",
            "reason": "optimizer delivery preview was empty",
        }
    shadow_metrics = _optimizer_delivery_shadow_metrics(preview)
    bucket = _optimizer_dynamic_soft_overlay_bucket(preview)
    enabled_mask = preview.get("nowcast_dynamic_overlay_enabled", pd.Series(False, index=preview.index)).astype(bool)
    applicable_mask = bucket.isin(["strategic", "supported", "background"])
    summary: dict[str, Any] = {
        "enabled": bool(enabled_mask.any()),
        "shadow_enabled": bool(MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_shadow_enabled"]),
        "row_count": int(len(preview)),
        "cycle_count": int(preview.get("cycle_origin_timestamp", pd.Series(dtype="string")).nunique(dropna=True)),
        "applicable_row_count": int(applicable_mask.sum()),
        "strategic_row_count": int(bucket.eq("strategic").sum()),
        "supported_row_count": int(bucket.eq("supported").sum()),
        "background_row_count": int(bucket.eq("background").sum()),
        "weight_grid": {
            "supported_weights": [
                float(value)
                for value in cast(list[float], MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"])
            ],
            "background_weights": [
                float(value)
                for value in cast(list[float], MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"])
            ],
        },
        "non_regression_thresholds": _optimizer_dynamic_soft_overlay_thresholds(),
        "shadow_mode": shadow_metrics,
        "soft_candidate_count": int(len(soft_candidates)),
    }
    if soft_candidates.empty:
        summary["recommendation"] = "keep_pure_nowcast_shadow"
        summary["reason"] = (
            "no comparable soft minute-overlay candidates were available on this preview, so the repo should keep "
            "the pure-nowcast shadow surface"
        )
        return summary
    admissible = soft_candidates.loc[soft_candidates["meets_non_regression_rules"].astype(bool)].copy()
    improving = admissible.loc[admissible["beats_shadow"].astype(bool)].copy()
    summary["top_candidates"] = [
        _optimizer_dynamic_soft_overlay_candidate_record(row)
        for _, row in soft_candidates.head(5).iterrows()
    ]
    if not admissible.empty:
        summary["best_admissible_candidate"] = _optimizer_dynamic_soft_overlay_candidate_record(admissible.iloc[0])
    if not improving.empty:
        summary["best_improving_candidate"] = _optimizer_dynamic_soft_overlay_candidate_record(improving.iloc[0])
        summary["recommendation"] = "shadow_soft_overlay_candidate_positive"
        summary["reason"] = (
            "at least one soft minute-overlay policy reduced all-interval error without regressing next-lock or "
            "peak-hit behavior on the same replay surface"
        )
    else:
        summary["recommendation"] = "keep_pure_nowcast_shadow"
        summary["reason"] = (
            "no soft minute-overlay policy improved the pure-nowcast shadow surface while clearing the configured "
            "next-lock and peak-hit non-regression rules"
        )
    return summary


def _optimizer_delivery_peak_hit_rate(preview: pd.DataFrame) -> float:
    """Measure how often the selected forecast peak lands on the actual peak interval."""
    if preview.empty:
        return float("nan")
    required_columns = {"cycle_origin_timestamp", "interval_start", "is_actual_peak_interval", "is_predicted_peak_interval"}
    if not required_columns.issubset(preview.columns):
        return float("nan")
    hits: list[float] = []
    for _, cycle_frame in preview.groupby("cycle_origin_timestamp", dropna=False):
        actual_rows = cycle_frame.loc[cycle_frame["is_actual_peak_interval"].astype(bool)]
        predicted_rows = cycle_frame.loc[cycle_frame["is_predicted_peak_interval"].astype(bool)]
        if actual_rows.empty or predicted_rows.empty:
            continue
        actual_interval = pd.Timestamp(actual_rows.iloc[0]["interval_start"])
        predicted_interval = pd.Timestamp(predicted_rows.iloc[0]["interval_start"])
        hits.append(float(actual_interval == predicted_interval))
    if not hits:
        return float("nan")
    return float(np.mean(hits))


def _optimizer_delivery_shadow_metrics(preview: pd.DataFrame) -> dict[str, Any]:
    """Summarize one delivery surface for shadow-policy comparison."""
    if preview.empty:
        return {
            "row_count": 0,
            "cycle_count": 0,
            "selected_layer_counts": {},
            "mean_selected_abs_error": float("nan"),
            "next_lock_mae": float("nan"),
            "predicted_peak_interval_count": 0,
            "peak_hit_rate": float("nan"),
        }
    selected_layer_counts = {
        str(key): int(value)
        for key, value in preview.get("selected_layer_role", pd.Series(dtype="string"))
        .astype("string")
        .value_counts(dropna=False)
        .items()
    }
    abs_error = pd.to_numeric(preview.get("selected_abs_error"), errors="coerce")
    next_lock_mask = preview.get("is_next_lock_interval", pd.Series(False, index=preview.index)).astype(bool)
    predicted_peak_mask = preview.get("requested_is_predicted_peak_interval", pd.Series(False, index=preview.index)).astype(bool)
    return {
        "row_count": int(len(preview)),
        "cycle_count": int(preview.get("cycle_origin_timestamp", pd.Series(dtype="string")).nunique(dropna=True)),
        "selected_layer_counts": selected_layer_counts,
        "mean_selected_abs_error": float(abs_error.mean()) if abs_error.notna().any() else float("nan"),
        "next_lock_mae": (
            float(abs_error.loc[next_lock_mask].mean())
            if bool(next_lock_mask.any()) and abs_error.loc[next_lock_mask].notna().any()
            else float("nan")
        ),
        "predicted_peak_interval_count": int(predicted_peak_mask.sum()),
        "peak_hit_rate": _optimizer_delivery_peak_hit_rate(preview),
    }


def _build_optimizer_dynamic_overlay_shadow_summary(preview: pd.DataFrame) -> dict[str, Any]:
    """Compare shadow-only versus enforced dynamic minute routing on the same Stage-10 surface."""
    if preview.empty:
        return {
            "enabled": False,
            "live_enforcement_enabled": bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"]),
            "recommendation": "no_delivery_preview",
            "reason": "optimizer delivery preview was empty",
        }
    enabled_mask = preview.get("nowcast_dynamic_overlay_enabled", pd.Series(False, index=preview.index)).astype(bool)
    eligible_mask = preview.get("nowcast_dynamic_overlay_eligible", pd.Series(True, index=preview.index)).astype(bool)
    dynamic_enabled = bool(enabled_mask.any())
    live_enforcement_enabled = bool(preview.get("nowcast_dynamic_overlay_enforced", pd.Series(False, index=preview.index)).astype(bool).any())
    reason_counts = {
        str(key): int(value)
        for key, value in preview.get("nowcast_dynamic_overlay_reason", pd.Series(dtype="string"))
        .astype("string")
        .value_counts(dropna=False)
        .items()
    }
    shadow_preview = _reresolve_optimizer_delivery_preview(preview, enforce_dynamic_overlay=False)
    enforced_preview = _reresolve_optimizer_delivery_preview(preview, enforce_dynamic_overlay=True)
    shadow_metrics = _optimizer_delivery_shadow_metrics(shadow_preview)
    enforced_metrics = _optimizer_delivery_shadow_metrics(enforced_preview)
    delta_mean_abs_error = float(enforced_metrics["mean_selected_abs_error"]) - float(shadow_metrics["mean_selected_abs_error"])
    delta_next_lock_mae = float(enforced_metrics["next_lock_mae"]) - float(shadow_metrics["next_lock_mae"])
    delta_peak_hit_rate = float(enforced_metrics["peak_hit_rate"]) - float(shadow_metrics["peak_hit_rate"])

    recommendation = "no_dynamic_signal"
    reason = "no advisory-backed dynamic minute surface was present on this preview"
    if dynamic_enabled:
        harmful = False
        if np.isfinite(delta_mean_abs_error) and delta_mean_abs_error > 0.0:
            harmful = True
        if np.isfinite(delta_next_lock_mae) and delta_next_lock_mae > 0.0:
            harmful = True
        if np.isfinite(delta_peak_hit_rate) and delta_peak_hit_rate < 0.0:
            harmful = True
        if harmful:
            recommendation = "keep_shadow_mode"
            reason = (
                "enforcing the dynamic minute gate worsened at least one live-facing Stage-10 metric, so the "
                "controller should remain diagnostic-only for now"
            )
        elif (
            np.isfinite(delta_mean_abs_error)
            and delta_mean_abs_error < 0.0
            and (not np.isfinite(delta_next_lock_mae) or delta_next_lock_mae <= 0.0)
            and (not np.isfinite(delta_peak_hit_rate) or delta_peak_hit_rate >= 0.0)
        ):
            recommendation = "consider_enforcement"
            reason = (
                "enforcing the dynamic minute gate improved all-interval error without regressing next-lock or "
                "peak-hit behavior on the same replay surface"
            )
        else:
            recommendation = "continue_shadow_monitoring"
            reason = (
                "the enforced counterfactual was mixed or neutral, so the controller should keep gathering evidence "
                "before it can influence live layer resolution"
            )
    return {
        "enabled": dynamic_enabled,
        "live_enforcement_enabled": live_enforcement_enabled,
        "row_count": int(len(preview)),
        "cycle_count": int(preview.get("cycle_origin_timestamp", pd.Series(dtype="string")).nunique(dropna=True)),
        "eligible_row_count": int((enabled_mask & eligible_mask).sum()),
        "ineligible_row_count": int((enabled_mask & ~eligible_mask).sum()),
        "eligible_rate": float((enabled_mask & eligible_mask).mean()) if len(preview) else float("nan"),
        "background_row_count": int((enabled_mask & ~eligible_mask).sum()),
        "reason_counts": reason_counts,
        "shadow_mode": shadow_metrics,
        "enforced_counterfactual": enforced_metrics,
        "delta_enforced_minus_shadow": {
            "mean_selected_abs_error": float(delta_mean_abs_error),
            "next_lock_mae": float(delta_next_lock_mae),
            "peak_hit_rate": float(delta_peak_hit_rate),
        },
        "recommendation": recommendation,
        "reason": reason,
    }


def _build_optimizer_delivery_contract(
    *,
    run_id: str,
    config_hash: str,
    policy: dict[str, Any],
    lock_interval_minutes: int,
    uncertainty_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Describe the optimizer-facing interval contract emitted by the Stage-10 replay."""
    summary_lookup = (
        uncertainty_summary.set_index("scope", drop=False).to_dict(orient="index")
        if not uncertainty_summary.empty
        else {}
    )
    candidate_labels = _selected_candidate_labels(policy)
    return {
        "contract_version": OPTIMIZER_CONTRACT_VERSION,
        "intent": "pre_optimizer_interval_forecast",
        "producer_stage": "010_forecast_control",
        "run_id": str(run_id),
        "config_hash": str(config_hash),
        "load_type": str(DATASET["load_type"]),
        "cadence_minutes": int(lock_interval_minutes),
        "selected_layer_priority": list(OPTIMIZER_LAYER_PRIORITY),
        "selected_candidates": candidate_labels,
        "serving_preview_artifact": "optimizer_delivery_serving_preview.csv",
        "dynamic_overlay_shadow_summary_artifact": "optimizer_dynamic_overlay_shadow_summary.json",
        "dynamic_overlay_soft_summary_artifact": "optimizer_dynamic_overlay_soft_summary.json",
        "dynamic_overlay_soft_candidates_artifact": "optimizer_dynamic_overlay_soft_candidates.csv",
        "day_ahead_refresh_policy": cast(dict[str, Any], policy.get("day_ahead_refresh", {})).get(
            "recommended_policy",
            "disabled",
        ),
        "phase_stack_guard_policy": cast(dict[str, Any], policy.get("phase", {})).get(
            "stack_guard_recommended_policy",
            "phase_candidate",
        ),
        "phase_rolling_support_policy": cast(dict[str, Any], policy.get("phase", {})).get(
            "rolling_support_recommended_policy",
            "phase_candidate",
        ),
        "operational_policy_artifact": "optimizer_operational_policy.json",
        "freshness": {
            "row_fields": [
                "as_of_timestamp",
                "effective_forecast_as_of",
                "expected_layer_cadence_minutes",
                "forecast_age_minutes",
                "stale_threshold_minutes",
                "is_stale_forecast",
            ],
            "notes": (
                "Rows carry an executable freshness surface: when a requested layer is stale or unavailable, the "
                "delivery resolver walks the fallback chain and emits the resolved layer instead of only flagging "
                "the row as stale."
            ),
        },
        "live_resolution": {
            "enabled": True,
            "row_fields": [
                "requested_layer_role",
                "requested_candidate_label",
                "selected_layer_role",
                "selected_candidate_label",
                "fallback_applied",
                "fallback_from_layer_role",
                "fallback_to_layer_role",
                "fallback_trigger",
                "fallback_reason",
                "resolution_path",
            ],
            "notes": (
                "selected_layer_role is the resolved layer after availability and staleness checks, while "
                "requested_layer_role records the latest layer that would have been used before fallback."
            ),
        },
        "uncertainty": {
            "enabled": not uncertainty_summary.empty,
            "method": "contextual_empirical_residual_quantiles",
            "calibration_scope": [
                "exact_control_calibration",
                "rolling_control_calibration",
            ],
            "calibration_contexts": [
                "lead_interval",
                "predicted_peak_lead_interval",
                "predicted_peak_global",
                "next_lock_global",
                "layer_global_fallback",
            ],
            "levels": {"80": [0.1, 0.9], "95": [0.025, 0.975]},
            "evaluation_summary": summary_lookup.get("all_intervals", {}),
            "notes": (
                "Bands are calibrated from held-out residual quantiles on the control calibration windows, using "
                "the most specific available context before falling back to broader layer-level residuals. "
                "They are intended as risk signals for optimizer consumption, not as a claim of fully "
                "probabilistic state-of-the-art forecasting."
            ),
        },
        "confidence_signal": {
            "enabled": True,
            "type": "heuristic_operational_trust_score",
            "range": [0.0, 1.0],
            "row_fields": ["confidence_score", "confidence_tier"],
            "tier_thresholds": {"high": 0.75, "medium": 0.5, "low": 0.0},
            "notes": (
                "confidence_score is not a probability. It is a compact trust hint derived from interval-band width, "
                "calibration support, layer role, and freshness."
            ),
        },
        "fields": [
            {"name": "cycle_origin_timestamp", "type": "datetime", "required": True, "description": "As-of timestamp for the replayed control cycle."},
            {"name": "as_of_timestamp", "type": "datetime", "required": True, "description": "Explicit optimizer-facing alias for the cycle as-of timestamp."},
            {"name": "interval_start", "type": "datetime", "required": True, "description": "Start of the optimizer-facing 15-minute interval target."},
            {"name": "interval_end", "type": "datetime", "required": True, "description": "End of the optimizer-facing 15-minute interval target."},
            {"name": "lead_interval_index", "type": "integer", "required": True, "description": "0-based interval index from the cycle origin."},
            {"name": "horizon_minutes", "type": "integer", "required": True, "description": "Minutes from the cycle origin to the interval end."},
            {"name": "producer_stage", "type": "string", "required": True, "description": "Stage identifier that emitted the delivery row."},
            {"name": "contract_version", "type": "string", "required": True, "description": "Version of the emitted delivery contract."},
            {"name": "run_id", "type": "string", "required": True, "description": "Stage-10 run id that produced the row."},
            {"name": "config_hash", "type": "string", "required": True, "description": "Stable config hash for reproducibility and provenance."},
            {"name": "operating_regime", "type": "string", "required": False, "description": "Interval operating context derived from transition and profile-activity signals."},
            {"name": "actual_ramp_band", "type": "string", "required": False, "description": "Interval ramp bucket derived from minute-level ramp share inside the lock window."},
            {"name": "requested_layer_role", "type": "string", "required": True, "description": "Latest available layer before freshness and fallback checks are applied."},
            {"name": "requested_candidate_label", "type": "string", "required": True, "description": "Candidate label attached to the requested_layer_role."},
            {"name": "nowcast_dynamic_overlay_enforced", "type": "boolean", "required": False, "description": "Whether the dynamic minute-overlay policy is currently allowed to alter live layer selection instead of running in shadow mode."},
            {"name": "nowcast_dynamic_overlay_eligible", "type": "boolean", "required": False, "description": "Whether the learned minute overlay is considered strategically useful for this interval under the dynamic controller."},
            {"name": "nowcast_dynamic_overlay_reason", "type": "string", "required": False, "description": "Reason emitted by the dynamic minute-overlay controller for keeping or demoting the learned nowcast."},
            {"name": "selected_layer_role", "type": "string", "required": True, "description": "Resolved forecast layer after availability and staleness checks."},
            {"name": "selected_candidate_label", "type": "string", "required": True, "description": "Persisted candidate label that produced the selected forecast."},
            {"name": "effective_forecast_as_of", "type": "datetime", "required": True, "description": "Timestamp of the forecast instance currently attached to the row."},
            {"name": "expected_layer_cadence_minutes", "type": "integer", "required": True, "description": "Nominal update cadence for the selected forecast layer."},
            {"name": "forecast_age_minutes", "type": "float", "required": True, "description": "Age of the selected forecast relative to the emitted as-of timestamp."},
            {"name": "stale_threshold_minutes", "type": "integer", "required": True, "description": "Operational threshold beyond which the selected forecast should be treated as stale."},
            {"name": "is_stale_forecast", "type": "boolean", "required": True, "description": "Whether the selected forecast exceeded the configured staleness threshold."},
            {"name": "fallback_applied", "type": "boolean", "required": True, "description": "Whether the requested layer had to fall back to an older layer."},
            {"name": "fallback_from_layer_role", "type": "string", "required": False, "description": "Layer role that triggered fallback because it was stale or unavailable."},
            {"name": "fallback_to_layer_role", "type": "string", "required": False, "description": "Resolved fallback layer role when fallback_applied is true."},
            {"name": "fallback_trigger", "type": "string", "required": True, "description": "Primary fallback trigger, such as unavailable or stale."},
            {"name": "forecast_value", "type": "float", "required": True, "description": "Point forecast for the selected interval."},
            {"name": "forecast_lower_80", "type": "float", "required": False, "description": "Lower bound of the 80% empirical residual band."},
            {"name": "forecast_upper_80", "type": "float", "required": False, "description": "Upper bound of the 80% empirical residual band."},
            {"name": "forecast_lower_95", "type": "float", "required": False, "description": "Lower bound of the 95% empirical residual band."},
            {"name": "forecast_upper_95", "type": "float", "required": False, "description": "Upper bound of the 95% empirical residual band."},
            {"name": "fallback_reason", "type": "string", "required": True, "description": "Why the selected layer was used instead of a later update layer."},
            {"name": "resolution_path", "type": "string", "required": False, "description": "Compact trace of the layer-resolution chain for debugging and operator review."},
            {"name": "quantile_source", "type": "string", "required": False, "description": "Whether the uncertainty band came from lead-specific calibration or a layer-global fallback."},
            {"name": "calibration_sample_n", "type": "integer", "required": False, "description": "Effective calibration support behind the emitted interval band."},
            {"name": "confidence_score", "type": "float", "required": True, "description": "Heuristic 0-1 trust score derived from uncertainty width, support, layer role, and freshness."},
            {"name": "confidence_tier", "type": "string", "required": True, "description": "Digestible operational tier derived from confidence_score."},
        ],
    }


def _build_optimizer_operational_policy(
    *,
    run_id: str,
    config_hash: str,
    policy: dict[str, Any],
    lock_interval_minutes: int,
    uncertainty_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize how the optimizer-facing surface should behave under fallback and host variability."""
    candidate_labels = _selected_candidate_labels(policy)
    refresh_policy = cast(dict[str, Any], policy.get("day_ahead_refresh", {}))
    summary_lookup = (
        uncertainty_summary.set_index("scope", drop=False).to_dict(orient="index")
        if not uncertainty_summary.empty
        else {}
    )
    runtime = runtime_summary(int(MODELING_STAGE_PARALLEL["forecast_control"]["max_workers"])).as_dict()
    return {
        "policy_version": OPTIMIZER_POLICY_VERSION,
        "contract_version": OPTIMIZER_CONTRACT_VERSION,
        "intent": "pre_optimizer_operating_policy",
        "producer_stage": "010_forecast_control",
        "run_id": str(run_id),
        "config_hash": str(config_hash),
        "delivery_cadence_minutes": int(lock_interval_minutes),
        "layer_priority": list(OPTIMIZER_LAYER_PRIORITY),
        "layer_contracts": _optimizer_layer_contracts(candidate_labels),
        "selection_policy": {
            "priority_order": list(OPTIMIZER_LAYER_PRIORITY),
            "fallback_reason_map": _optimizer_fallback_reason_map(),
            "dynamic_overlay_shadow_summary_artifact": "optimizer_dynamic_overlay_shadow_summary.json",
            "dynamic_overlay_soft_summary_artifact": "optimizer_dynamic_overlay_soft_summary.json",
            "dynamic_overlay_soft_candidates_artifact": "optimizer_dynamic_overlay_soft_candidates.csv",
            "nowcast_dynamic_overlay": {
                "enabled": bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enabled"]),
                "enforce": bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_enforce"]),
                "selection_mode": "layered_dynamic_overlay_specialist",
                "soft_shadow_enabled": bool(MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_shadow_enabled"]),
                "soft_shadow_supported_weights": [
                    float(value)
                    for value in cast(
                        list[float],
                        MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_supported_weights"],
                    )
                ],
                "soft_shadow_background_weights": [
                    float(value)
                    for value in cast(
                        list[float],
                        MULTIRES_FORECAST_CONTROL["nowcast_soft_overlay_background_weights"],
                    )
                ],
                "soft_shadow_non_regression_thresholds": _optimizer_dynamic_soft_overlay_thresholds(),
                "supported_operating_regimes": list(
                    cast(dict[str, Any], policy.get("nowcast_anchor", {})).get(
                        "advisory_supported_operating_regimes",
                        [],
                    )
                ),
                "strategic_interval_rules": {
                    "allow_next_lock": bool(MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_next_lock"]),
                    "allow_predicted_peak": bool(
                        MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_allow_predicted_peak"]
                    ),
                },
                "profile_active_threshold": float(
                    MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_profile_active_threshold"]
                ),
                "high_ramp_fraction_threshold": float(
                    MULTIRES_FORECAST_CONTROL["nowcast_dynamic_overlay_high_ramp_fraction_threshold"]
                ),
                "advisory_surface_supported": bool(
                    cast(dict[str, Any], policy.get("nowcast_anchor", {})).get(
                        "advisory_surface_supported",
                        False,
                    )
                ),
                "advisory_transition_best_ratio_to_persistence": float(
                    cast(dict[str, Any], policy.get("nowcast_anchor", {})).get(
                        "advisory_transition_best_ratio_to_persistence",
                        float("nan"),
                    )
                ),
                "advisory_high_ramp_ratio_to_persistence": float(
                    cast(dict[str, Any], policy.get("nowcast_anchor", {})).get(
                        "advisory_high_ramp_ratio_to_persistence",
                        float("nan"),
                    )
                ),
                "notes": (
                    "The minute layer is treated as a dynamic corrective specialist. It remains the global latest "
                    "candidate, but the dynamic overlay policy is shadow-only by default until it proves itself "
                    "on Stage-10 replay. The current strategic rules highlight next-lock intervals, predicted-peak "
                    "intervals, and regimes where Stage-5 supplemental evidence showed learned minute corrections "
                    "beating persistence."
                ),
            },
            "control_promotion_guard": {
                "enabled": bool(MULTIRES_FORECAST_CONTROL["control_promotion_guard_enabled"]),
                "max_next_lock_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_next_lock_regress_pct"]
                ),
                "max_peak_value_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_peak_value_regress_pct"]
                ),
                "max_peak_miss_regress": float(
                    MULTIRES_FORECAST_CONTROL["control_promotion_guard_max_peak_miss_regress"]
                ),
                "notes": (
                    "Hourly and nowcast challengers must clear next-lock and peak non-regression guardrails "
                    "against the upstream choice before they can replace it."
                ),
            },
            "phase_stack_guard_policy": cast(dict[str, Any], policy.get("phase", {})).get(
                "stack_guard_recommended_policy",
                "phase_candidate",
            ),
            "phase_stack_selection_metric": str(MULTIRES_FORECAST_CONTROL["phase_stack_selection_metric"]),
            "phase_stack_guard_thresholds": {
                "min_lock_gain_pct": float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_lock_gain_pct"]),
                "max_next_lock_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_next_lock_regress_pct"]
                ),
                "max_profile_degrade_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_profile_degrade_pct"]
                ),
                "max_peak_value_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_peak_value_regress_pct"]
                ),
                "min_peak_hit_gain": float(MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_peak_hit_gain"]),
                "max_optimizer_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_optimizer_regress_pct"]
                ),
            },
            "phase_stack_rolling_support": {
                "required": bool(MULTIRES_FORECAST_CONTROL["phase_stack_guard_require_rolling_support"]),
                "scope": str(MULTIRES_FORECAST_CONTROL["phase_stack_guard_rolling_scope"]),
                "min_lock_gain_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_rolling_lock_gain_pct"]
                ),
                "max_next_lock_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_next_lock_regress_pct"]
                ),
                "max_profile_degrade_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_profile_degrade_pct"]
                ),
                "max_peak_value_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_peak_value_regress_pct"]
                ),
                "min_peak_hit_gain": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_min_rolling_peak_hit_gain"]
                ),
                "max_optimizer_regress_pct": float(
                    MULTIRES_FORECAST_CONTROL["phase_stack_guard_max_rolling_optimizer_regress_pct"]
                ),
            },
            "minute_layer_policy": {
                "standalone_stage5_position": "baseline_led_until_holdout_changes",
                "stage10_operating_role": "corrective_overlay",
                "deployment_rule": (
                    "Use learned minute candidates inside the stacked Stage-10 control path when they improve the "
                    "held-out control surface. Do not claim standalone 1m learned superiority unless Stage-5 says so."
                ),
            },
        },
        "uncertainty_policy": {
            "method": "contextual_empirical_residual_quantiles",
            "preferred_quantile_sources": [
                "predicted_peak_lead_interval",
                "lead_interval",
                "predicted_peak_global",
                "next_lock_global",
            ],
            "fallback_quantile_source": "layer_global_fallback",
            "quantile_source_multipliers": _optimizer_quantile_source_confidence_multiplier_map(),
            "current_coverage_summary": summary_lookup.get("all_intervals", {}),
            "confidence_score_policy": {
                "type": "heuristic_operational_trust_score",
                "range": [0.0, 1.0],
                "tier_thresholds": {"high": 0.75, "medium": 0.5, "low": 0.0},
                "layer_multipliers": _optimizer_layer_confidence_multiplier_map(),
                "formula": (
                    "(1 / (1 + uncertainty_band_width_95_pct / 100))"
                    " * min(calibration_sample_n / 8, 1)"
                    " * layer_multiplier"
                    " * quantile_source_multiplier"
                    " * freshness_multiplier"
                ),
                "notes": (
                    "The score is a trust hint for operator and optimizer consumption. It is not a calibrated "
                    "probability and should not be marketed as one."
                ),
            },
        },
        "data_integrity_policy": {
            "missing_or_delayed_signal_behavior": (
                "Resolve the freshest usable layer in priority order, degrading to older layers when the requested "
                "layer is stale or unavailable, and emit explicit fallback fields instead of hiding the gap."
            ),
            "row_staleness_fields": [
                "effective_forecast_as_of",
                "forecast_age_minutes",
                "stale_threshold_minutes",
                "is_stale_forecast",
            ],
            "resolver_fields": [
                "requested_layer_role",
                "selected_layer_role",
                "fallback_applied",
                "fallback_from_layer_role",
                "fallback_to_layer_role",
                "fallback_trigger",
                "fallback_reason",
                "resolution_path",
            ],
            "replay_note": (
                "Replay previews still originate from historical control cycles, but the emitted row surface now "
                "uses the same layer-resolution rules that a live-serving adapter should apply."
            ),
        },
        "day_ahead_refresh_policy": {
            "recommended_policy": refresh_policy.get("recommended_policy", "disabled"),
            "trigger_mode": refresh_policy.get("trigger_mode", "disabled"),
            "refresh_interval_minutes": int(refresh_policy.get("refresh_interval_minutes", 60)),
            "lookback_minutes": int(refresh_policy.get("lookback_minutes", 0)),
            "threshold_source": refresh_policy.get("threshold_source", "n/a"),
            "thresholds": {
                "residual_drift_mae_pct_threshold": refresh_policy.get(
                    "residual_drift_mae_pct_threshold",
                    float("nan"),
                ),
                "transition_mae_pct_threshold": refresh_policy.get(
                    "transition_mae_pct_threshold",
                    float("nan"),
                ),
                "activity_ratio_shift_threshold": refresh_policy.get(
                    "activity_ratio_shift_threshold",
                    float("nan"),
                ),
            },
            "evaluation_trigger_rate": refresh_policy.get("evaluation_trigger_rate", float("nan")),
            "rolling_trigger_rate": cast(dict[str, Any], refresh_policy.get("rolling_benchmark", {})).get(
                "trigger_rate",
                float("nan"),
            ),
            "reason": refresh_policy.get("reason", ""),
        },
        "retraining_policy": {
            "mode": "manual_review_gated",
            "recommended_review_triggers": [
                "sustained day-ahead residual drift beyond the selected trigger band",
                "delivery uncertainty coverage degrading below target on the latest held-out replay",
                "new challenger replay overtakes the current selected layer on the held-out control surface",
                "material new labeled data window ready for Stage-5/Stage-10 refresh",
            ],
            "notes": (
                "The repo persists evidence and promotion artifacts, but automatic champion replacement is still out "
                "of scope. Keep retraining and promotion gated by explicit review."
            ),
        },
        "hardware_policy": {
            "runtime_summary": runtime,
            "portable_default": (
                "CPU-safe HGB and baseline paths remain the default-safe contract for non-accelerated and ARM64 hosts."
            ),
            "accelerated_optional_path": (
                "Optional XGBoost/CUDA candidates may participate when the runtime exposes a supported accelerator, "
                "but the evidence bundle remains valid when acceleration is absent."
            ),
        },
    }


def run_forecast_control_backtest(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    replay_cache_enabled: bool | None = None,
) -> dict[str, Any]:
    """Replay the current control stack and measure end-to-end correction value."""
    validate_config()
    actual_resolution = str(MULTIRES_FORECAST_CONTROL["actual_resolution"])
    actual_minute_base = load_base_gold(actual_resolution)
    actual_minute_base["timestamp"] = pd.to_datetime(actual_minute_base["timestamp"], errors="raise")
    actual_minute_feature_frame = build_causal_feature_frame(actual_minute_base.copy(), actual_resolution)
    actual_minute_feature_frame["timestamp"] = pd.to_datetime(
        actual_minute_feature_frame["timestamp"],
        errors="raise",
    )
    actual_minute_feature_frame = actual_minute_feature_frame.set_index("timestamp", drop=False)
    origin_catalog, calibration_cycle_origins, evaluation_cycle_origins = _resolve_control_origin_sets(
        actual_minute_base
    )
    rolling_origin_catalog, rolling_calibration_cycle_origins, rolling_evaluation_cycle_origins = (
        _resolve_rolling_control_origin_sets(actual_minute_base)
    )
    if not evaluation_cycle_origins:
        raise RuntimeError("No eligible forecast-control cycles were available in the configured evaluation window.")
    if not calibration_cycle_origins:
        raise RuntimeError("No eligible forecast-control cycles were available in the calibration window.")

    effective_replay_cache_enabled = (
        bool(MULTIRES_FORECAST_CONTROL["replay_cache_enabled"])
        if replay_cache_enabled is None
        else bool(replay_cache_enabled)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = _replay_cache_root(output_root) if effective_replay_cache_enabled else None
    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started_at_utc, run_started_perf = _start_runtime_step()
    runtime_profile_records: list[dict[str, Any]] = []

    day_ahead_horizon = int(MULTIRES_FORECAST_CONTROL["day_ahead_horizon_minutes"])
    hourly_horizon = int(MULTIRES_FORECAST_CONTROL["hourly_horizon_minutes"])
    phase_horizon = int(MULTIRES_FORECAST_CONTROL["phase_horizon_minutes"])
    lock_interval = int(MULTIRES_FORECAST_CONTROL["lock_interval_minutes"])
    calibration_refresh_origins = _day_ahead_refresh_origins(calibration_cycle_origins)
    evaluation_refresh_origins = _day_ahead_refresh_origins(evaluation_cycle_origins)
    calibration_hourly_origins = _layer_update_origins(
        cycle_origins=calibration_cycle_origins,
        update_interval_minutes=hourly_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    evaluation_hourly_origins = _layer_update_origins(
        cycle_origins=evaluation_cycle_origins,
        update_interval_minutes=hourly_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    calibration_phase_origins = _layer_update_origins(
        cycle_origins=calibration_cycle_origins,
        update_interval_minutes=phase_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    evaluation_phase_origins = _layer_update_origins(
        cycle_origins=evaluation_cycle_origins,
        update_interval_minutes=phase_horizon,
        cycle_horizon_minutes=day_ahead_horizon,
    )
    _append_runtime_step(
        runtime_profile_records,
        step="prepare_control_scope",
        category="setup",
        started_at_utc=run_started_at_utc,
        started_perf=run_started_perf,
        calibration_cycle_count=int(len(calibration_cycle_origins)),
        evaluation_cycle_count=int(len(evaluation_cycle_origins)),
        rolling_calibration_cycle_count=int(len(rolling_calibration_cycle_origins)),
        rolling_evaluation_cycle_count=int(len(rolling_evaluation_cycle_origins)),
    )

    with TemporaryDirectory(prefix="elf_forecast_control_") as temp_dir:
        temp_root = Path(temp_dir)
        day_ahead_started_at_utc, day_ahead_started_perf = _start_runtime_step()
        day_ahead = _replay_rollout_layer(
            temp_root=temp_root / "day_ahead",
            cache_root=cache_root,
            layer_role="day_ahead",
            horizon_minutes=day_ahead_horizon,
            benchmark_origin_timestamps=calibration_cycle_origins,
            evaluation_origin_timestamps=evaluation_cycle_origins,
        )
        _append_runtime_step(
            runtime_profile_records,
            step="replay_day_ahead_layer",
            category="replay",
            started_at_utc=day_ahead_started_at_utc,
            started_perf=day_ahead_started_perf,
            candidate_pool_count=int(day_ahead["candidate_pool_count"]),
            benchmark_origin_count=int(day_ahead["benchmark_origin_count"]),
            evaluation_origin_count=int(day_ahead["evaluation_origin_count"]),
        )
        hourly_started_at_utc, hourly_started_perf = _start_runtime_step()
        hourly = _replay_rollout_layer(
            temp_root=temp_root / "hourly",
            cache_root=cache_root,
            layer_role="hourly",
            horizon_minutes=hourly_horizon,
            benchmark_origin_timestamps=calibration_hourly_origins,
            evaluation_origin_timestamps=evaluation_hourly_origins,
        )
        _append_runtime_step(
            runtime_profile_records,
            step="replay_hourly_layer",
            category="replay",
            started_at_utc=hourly_started_at_utc,
            started_perf=hourly_started_perf,
            candidate_pool_count=int(hourly["candidate_pool_count"]),
            benchmark_origin_count=int(hourly["benchmark_origin_count"]),
            evaluation_origin_count=int(hourly["evaluation_origin_count"]),
        )
        phase_started_at_utc, phase_started_perf = _start_runtime_step()
        phase = _replay_rollout_layer(
            temp_root=temp_root / "phase",
            cache_root=cache_root,
            layer_role="phase",
            horizon_minutes=phase_horizon,
            benchmark_origin_timestamps=calibration_phase_origins,
            evaluation_origin_timestamps=evaluation_phase_origins,
        )
        _append_runtime_step(
            runtime_profile_records,
            step="replay_phase_layer",
            category="replay",
            started_at_utc=phase_started_at_utc,
            started_perf=phase_started_perf,
            candidate_pool_count=int(phase["candidate_pool_count"]),
            benchmark_origin_count=int(phase["benchmark_origin_count"]),
            evaluation_origin_count=int(phase["evaluation_origin_count"]),
        )
        refresh_started_at_utc, refresh_started_perf = _start_runtime_step()
        day_ahead_refresh = _replay_day_ahead_refresh_candidate(
            temp_root=temp_root / "day_ahead_refresh",
            cache_root=cache_root,
            day_ahead=day_ahead,
            benchmark_origin_timestamps=calibration_refresh_origins,
            evaluation_origin_timestamps=evaluation_refresh_origins,
        )
        _append_runtime_step(
            runtime_profile_records,
            step="replay_day_ahead_refresh_candidate",
            category="replay",
            started_at_utc=refresh_started_at_utc,
            started_perf=refresh_started_perf,
            enabled=bool(day_ahead_refresh is not None),
            benchmark_origin_count=int(len(calibration_refresh_origins)),
            evaluation_origin_count=int(len(evaluation_refresh_origins)),
        )

    calibration_selected_cycle_origins = [
        pd.Timestamp(value) for value in cast(list[pd.Timestamp], day_ahead["benchmark_origin_timestamps"])
    ]
    evaluation_selected_cycle_origins = [
        pd.Timestamp(value) for value in cast(list[pd.Timestamp], day_ahead["evaluation_origin_timestamps"])
    ]
    calibration_selected_hourly_origins = [
        pd.Timestamp(value) for value in cast(list[pd.Timestamp], hourly["benchmark_origin_timestamps"])
    ]
    evaluation_selected_hourly_origins = [
        pd.Timestamp(value) for value in cast(list[pd.Timestamp], hourly["evaluation_origin_timestamps"])
    ]
    calibration_selected_phase_origins = [
        pd.Timestamp(value) for value in cast(list[pd.Timestamp], phase["benchmark_origin_timestamps"])
    ]
    evaluation_selected_phase_origins = [
        pd.Timestamp(value) for value in cast(list[pd.Timestamp], phase["evaluation_origin_timestamps"])
    ]

    warnings: list[str] = []
    if len(calibration_cycle_origins) < 3:
        warnings.append(f"low_control_calibration_cycle_count:{len(calibration_cycle_origins)}")
    if len(evaluation_cycle_origins) < 3:
        warnings.append(f"low_control_evaluation_cycle_count:{len(evaluation_cycle_origins)}")
    for layer_name, payload in (("day_ahead", day_ahead), ("hourly", hourly), ("phase", phase)):
        if str(payload["candidate_label"]) != str(payload["upstream_candidate_label"]):
            warnings.append(
                f"{layer_name}_control_candidate_swapped:{payload['upstream_candidate_label']}->{payload['candidate_label']}"
            )

    exact_timeline_started_at_utc, exact_timeline_started_perf = _start_runtime_step()
    calibration_minute_timeline = _build_control_minute_timeline(
        cycle_origins=calibration_selected_cycle_origins,
        actual_minute_base=actual_minute_base,
        day_ahead=day_ahead,
        hourly=hourly,
        phase=phase,
        result_key="benchmark_result",
        day_ahead_horizon=day_ahead_horizon,
        hourly_horizon=hourly_horizon,
        phase_horizon=phase_horizon,
        hourly_origins=calibration_selected_hourly_origins,
        phase_origins=calibration_selected_phase_origins,
    )
    minute_timeline = _build_control_minute_timeline(
        cycle_origins=evaluation_selected_cycle_origins,
        actual_minute_base=actual_minute_base,
        day_ahead=day_ahead,
        hourly=hourly,
        phase=phase,
        result_key="result",
        day_ahead_horizon=day_ahead_horizon,
        hourly_horizon=hourly_horizon,
        phase_horizon=phase_horizon,
        hourly_origins=evaluation_selected_hourly_origins,
        phase_origins=evaluation_selected_phase_origins,
    )
    _append_runtime_step(
        runtime_profile_records,
        step="build_exact_control_timelines",
        category="evaluation",
        started_at_utc=exact_timeline_started_at_utc,
        started_perf=exact_timeline_started_perf,
        calibration_row_count=int(len(calibration_minute_timeline)),
        evaluation_row_count=int(len(minute_timeline)),
    )

    phase_stack_started_at_utc, phase_stack_started_perf = _start_runtime_step()
    phase_candidate_meta = cast(pd.DataFrame, phase["candidate_benchmarks"]).copy()
    calibration_phase_stack_benchmark, calibration_phase_stack_predictions, calibration_phase_stack_summaries = (
        _phase_stack_candidate_benchmark_scope(
            minute_timeline=calibration_minute_timeline,
            candidate_detail_by_origin=cast(pd.DataFrame, phase["candidate_benchmark_detail_by_origin"]),
            candidate_metrics_by_origin=cast(pd.DataFrame, phase["candidate_benchmark_by_origin"]),
            phase_origins=calibration_selected_phase_origins,
            phase_horizon=phase_horizon,
            lock_interval=lock_interval,
            hourly_candidate_label=str(hourly["candidate_label"]),
            hourly_candidate_type=str(hourly["candidate_type"]),
            hourly_source_model_label=str(hourly["source_model_label"]),
            candidate_meta=phase_candidate_meta,
        )
    )
    evaluation_phase_stack_benchmark, evaluation_phase_stack_predictions, evaluation_phase_stack_summaries = (
        _phase_stack_candidate_benchmark_scope(
            minute_timeline=minute_timeline,
            candidate_detail_by_origin=cast(pd.DataFrame, phase["candidate_evaluation_detail_by_origin"]),
            candidate_metrics_by_origin=cast(pd.DataFrame, phase["candidate_evaluation_by_origin"]),
            phase_origins=evaluation_selected_phase_origins,
            phase_horizon=phase_horizon,
            lock_interval=lock_interval,
            hourly_candidate_label=str(hourly["candidate_label"]),
            hourly_candidate_type=str(hourly["candidate_type"]),
            hourly_source_model_label=str(hourly["source_model_label"]),
            candidate_meta=phase_candidate_meta,
        )
    )
    (
        calibration_phase_stack_benchmark,
        evaluation_phase_stack_benchmark,
        calibration_phase_stack_predictions,
        evaluation_phase_stack_predictions,
        calibration_phase_stack_summaries,
        evaluation_phase_stack_summaries,
    ) = _phase_stack_baseline_control_candidates(
        calibration_minute_timeline=calibration_minute_timeline,
        evaluation_minute_timeline=minute_timeline,
        lock_interval=lock_interval,
        calibration_benchmark=calibration_phase_stack_benchmark,
        evaluation_benchmark=evaluation_phase_stack_benchmark,
        calibration_predictions=calibration_phase_stack_predictions,
        evaluation_predictions=evaluation_phase_stack_predictions,
        calibration_summaries=calibration_phase_stack_summaries,
        evaluation_summaries=evaluation_phase_stack_summaries,
    )
    phase_stack_selected_row, phase_stack_selection_mode = _select_phase_stack_candidate(
        calibration_benchmark=calibration_phase_stack_benchmark,
        evaluation_benchmark=evaluation_phase_stack_benchmark,
        hourly_candidate_label=str(hourly["candidate_label"]),
    )
    phase_stack_guard = _phase_stack_decision_from_selected_row(
        selected_row=phase_stack_selected_row,
        selection_mode=phase_stack_selection_mode,
        hourly_candidate_label=str(hourly["candidate_label"]),
        isolated_candidate_label=str(phase["candidate_label"]),
    )
    rolling_phase_support_context: dict[str, Any] = {}
    rolling_phase_guard_calibration_summary = pd.DataFrame()
    rolling_phase_guard_evaluation_summary = pd.DataFrame()
    rolling_phase_guard_combined_summary = pd.DataFrame()
    if str(phase_stack_guard["recommended_policy"]) == "phase_candidate":
        exact_phase_replay_metadata = _resolve_phase_stack_replay_metadata(
            phase_payload=phase,
            phase_stack_selected_row=phase_stack_selected_row,
            phase_stack_guard=phase_stack_guard,
        )
        if (
            bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"])
            and rolling_calibration_cycle_origins
            and rolling_evaluation_cycle_origins
        ):
            rolling_phase_support_context = _prepare_rolling_phase_support_context(
                actual_minute_base=actual_minute_base,
                cache_root=cache_root,
                day_ahead=day_ahead,
                hourly=hourly,
                phase=phase,
                day_ahead_horizon=day_ahead_horizon,
                hourly_horizon=hourly_horizon,
                phase_horizon=phase_horizon,
                rolling_calibration_cycle_origins=rolling_calibration_cycle_origins,
                rolling_evaluation_cycle_origins=rolling_evaluation_cycle_origins,
                phase_replay_metadata=exact_phase_replay_metadata,
            )
            if rolling_phase_support_context:
                _, rolling_phase_guard_calibration_by_cycle, rolling_phase_guard_calibration_summary = (
                    _evaluate_control_scope(
                        minute_timeline=cast(
                            pd.DataFrame,
                            rolling_phase_support_context["calibration_phase_timeline"],
                        ),
                        nowcast_anchor=_placeholder_nowcast_anchor(),
                        lock_interval=lock_interval,
                    )
                )
                _, rolling_phase_guard_evaluation_by_cycle, rolling_phase_guard_evaluation_summary = (
                    _evaluate_control_scope(
                        minute_timeline=cast(
                            pd.DataFrame,
                            rolling_phase_support_context["evaluation_phase_timeline"],
                        ),
                        nowcast_anchor=_placeholder_nowcast_anchor(),
                        lock_interval=lock_interval,
                    )
                )
                rolling_phase_guard_combined_by_cycle = pd.concat(
                    [rolling_phase_guard_calibration_by_cycle, rolling_phase_guard_evaluation_by_cycle],
                    ignore_index=True,
                )
                if not rolling_phase_guard_combined_by_cycle.empty:
                    rolling_phase_guard_combined_summary = _summary_frame(
                        rolling_phase_guard_combined_by_cycle,
                        _placeholder_nowcast_anchor(),
                    ).reset_index(drop=True)
                phase_stack_rolling_support_guard = _rolling_phase_stack_guard_decision(
                    calibration_summary=rolling_phase_guard_calibration_summary,
                    evaluation_summary=rolling_phase_guard_evaluation_summary,
                    combined_summary=rolling_phase_guard_combined_summary,
                    hourly_candidate_label=str(hourly["candidate_label"]),
                    phase_candidate_label=str(phase_stack_guard["applied_candidate_label"]),
                )
            else:
                phase_stack_rolling_support_guard = _empty_phase_stack_rolling_support_guard(
                    required=bool(MULTIRES_FORECAST_CONTROL["phase_stack_guard_require_rolling_support"]),
                    reason="Rolling support was required, but the broader rolling context could not be prepared.",
                )
        else:
            phase_stack_rolling_support_guard = _empty_phase_stack_rolling_support_guard(
                required=bool(MULTIRES_FORECAST_CONTROL["phase_stack_guard_require_rolling_support"]),
                reason="Rolling support was skipped because the broader rolling benchmark is disabled or has no eligible cycles.",
            )
    else:
        phase_stack_rolling_support_guard = _empty_phase_stack_rolling_support_guard(
            required=False,
            reason="Exact stack guard already chose hourly passthrough, so no rolling phase support check was needed.",
        )
    _append_runtime_step(
        runtime_profile_records,
        step="select_phase_stack_policy",
        category="evaluation",
        started_at_utc=phase_stack_started_at_utc,
        started_perf=phase_stack_started_perf,
        exact_recommended_policy=str(phase_stack_guard["recommended_policy"]),
        rolling_support_required=bool(phase_stack_rolling_support_guard["required"]),
        rolling_support_used=bool(rolling_phase_support_context),
        calibration_candidate_count=int(len(calibration_phase_stack_benchmark)),
        evaluation_candidate_count=int(len(evaluation_phase_stack_benchmark)),
        evaluation_native_candidate_count=int(
            evaluation_phase_stack_benchmark["stack_candidate_family"].astype("string").eq("native_phase_candidate").sum()
        )
        if not evaluation_phase_stack_benchmark.empty
        else 0,
        evaluation_blend_candidate_count=int(
            evaluation_phase_stack_benchmark["stack_candidate_family"].astype("string").eq("hourly_phase_blend").sum()
        )
        if not evaluation_phase_stack_benchmark.empty
        else 0,
    )
    phase_stack_guard = _combine_phase_stack_guard_with_rolling_support(
        phase_stack_guard=phase_stack_guard,
        rolling_support_guard=phase_stack_rolling_support_guard,
        hourly_candidate_label=str(hourly["candidate_label"]),
    )
    if str(MULTIRES_FORECAST_CONTROL["control_promotion_scope"]) == "held_out_evaluation":
        phase_stack_candidate_benchmarks = evaluation_phase_stack_benchmark.merge(
            _rename_phase_stack_calibration_metrics(calibration_phase_stack_benchmark.copy()),
            on="candidate_label",
            how="left",
        )
    else:
        phase_stack_candidate_benchmarks = calibration_phase_stack_benchmark.merge(
            _rename_phase_stack_evaluation_metrics(evaluation_phase_stack_benchmark.copy()),
            on="candidate_label",
            how="left",
        )
    phase_stack_candidate_benchmarks["stack_selection_mode"] = str(phase_stack_selection_mode)
    phase_stack_candidate_benchmarks["stack_recommended_policy"] = str(phase_stack_guard["recommended_policy"])
    phase_stack_candidate_benchmarks["stack_selected_candidate"] = (
        phase_stack_candidate_benchmarks["candidate_label"].astype("string").eq(
            str(phase_stack_guard["stack_selected_candidate_label"])
        )
    )
    if (
        str(phase_stack_guard["recommended_policy"]) == "phase_candidate"
        and str(phase_stack_guard["stack_selected_candidate_label"]) != str(phase["candidate_label"])
    ):
        warnings.append(
            "phase_stack_candidate_swapped:"
            f"{phase['candidate_label']}->{phase_stack_guard['stack_selected_candidate_label']}"
        )
    if str(phase_stack_guard["recommended_policy"]) != "phase_candidate":
        warnings.append(
            "phase_stack_guard_passthrough:"
            f"{phase['candidate_label']}->{phase_stack_guard['applied_candidate_label']}"
        )
    if bool(phase_stack_guard.get("rolling_support_applied_veto", False)):
        warnings.append(
            "phase_stack_rolling_support_passthrough:"
            f"{phase_stack_guard['stack_selected_candidate_label']}->{phase_stack_guard['applied_candidate_label']}"
        )
    selected_phase_candidate_label = str(phase_stack_guard["applied_candidate_label"])
    selected_phase_replay_metadata = _resolve_phase_stack_replay_metadata(
        phase_payload=phase,
        phase_stack_selected_row=phase_stack_selected_row,
        phase_stack_guard=phase_stack_guard,
    )
    calibration_hourly_series = pd.Series(
        calibration_minute_timeline["hourly_pred"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(calibration_minute_timeline["timestamp"], errors="raise")),
        dtype=float,
    )
    evaluation_hourly_series = pd.Series(
        minute_timeline["hourly_pred"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(minute_timeline["timestamp"], errors="raise")),
        dtype=float,
    )
    if (
        str(phase_stack_guard["recommended_policy"]) == "phase_candidate"
        and selected_phase_candidate_label in calibration_phase_stack_predictions
        and selected_phase_candidate_label in evaluation_phase_stack_predictions
    ):
        calibration_selected_phase_series = calibration_phase_stack_predictions[selected_phase_candidate_label]
        evaluation_selected_phase_series = evaluation_phase_stack_predictions[selected_phase_candidate_label]
        calibration_selected_phase_summary = calibration_phase_stack_summaries[selected_phase_candidate_label]
        evaluation_selected_phase_summary = evaluation_phase_stack_summaries[selected_phase_candidate_label]
        phase_stack_selected_row["replay_run_dir"] = str(phase_stack_selected_row.get("replay_run_dir", ""))
    elif str(phase_stack_guard["recommended_policy"]) == "phase_candidate":
        replay_selection = cast(dict[str, Any] | None, selected_phase_replay_metadata.get("selection"))
        if (
            selected_phase_candidate_label == str(phase["candidate_label"])
            and str(selected_phase_replay_metadata.get("mode", "")) == "native_phase_candidate"
        ):
            selected_phase_benchmark_detail = cast(dict[str, Any], phase["benchmark_result"])[
                "detail_by_origin"
            ].copy()
            selected_phase_evaluation_detail = cast(dict[str, Any], phase["result"])[
                "detail_by_origin"
            ].copy()
            selected_phase_run_dir = _relative_artifact_path(cast(dict[str, Any], phase["result"])["run_dir"])
        elif replay_selection is not None:
            selected_phase_benchmark_result = _run_cached_rollout_evaluation(
                cache_root=cache_root,
                temp_output_root=temp_root / "phase_stack_selected_benchmark",
                layer_role="phase",
                selection=replay_selection,
                horizon_minutes=int(phase_horizon),
                origin_policy=str(phase["origin_policy"]),
                selection_target=str(phase["selection_target"]),
                origin_timestamps=calibration_selected_phase_origins,
                capture_path_details=True,
                candidate_scope="selected_only",
                persist_artifacts=False,
            )
            selected_phase_full_result = _run_cached_rollout_evaluation(
                cache_root=cache_root,
                temp_output_root=temp_root / "phase_stack_selected_full",
                layer_role="phase",
                selection=replay_selection,
                horizon_minutes=int(phase_horizon),
                origin_policy=str(phase["origin_policy"]),
                selection_target=str(phase["selection_target"]),
                origin_timestamps=evaluation_selected_phase_origins,
                capture_path_details=True,
                candidate_scope="selected_only",
                persist_artifacts=True,
            )
            selected_phase_benchmark_detail = selected_phase_benchmark_result["detail_by_origin"].copy()
            selected_phase_evaluation_detail = selected_phase_full_result["detail_by_origin"].copy()
            selected_phase_benchmark_detail = _qualify_phase_replay_detail(
                selected_phase_benchmark_detail,
                replay_selection=replay_selection,
            )
            selected_phase_evaluation_detail = _qualify_phase_replay_detail(
                selected_phase_evaluation_detail,
                replay_selection=replay_selection,
            )
            selected_phase_run_dir = _relative_artifact_path(Path(selected_phase_full_result["run_dir"]))
        else:
            selected_phase_benchmark_detail = cast(pd.DataFrame, phase["candidate_benchmark_detail_by_origin"]).copy()
            selected_phase_evaluation_detail = cast(pd.DataFrame, phase["candidate_evaluation_detail_by_origin"]).copy()
            selected_phase_run_dir = str(selected_phase_replay_metadata.get("replay_run_dir", ""))
        calibration_selected_phase_series = _selected_phase_series_for_scope(
            minute_timeline=calibration_minute_timeline,
            hourly_pred_column="hourly_pred",
            phase_replay_metadata=selected_phase_replay_metadata,
            phase_detail_by_origin=selected_phase_benchmark_detail,
            phase_origins=calibration_selected_phase_origins,
            phase_horizon=phase_horizon,
        )
        evaluation_selected_phase_series = _selected_phase_series_for_scope(
            minute_timeline=minute_timeline,
            hourly_pred_column="hourly_pred",
            phase_replay_metadata=selected_phase_replay_metadata,
            phase_detail_by_origin=selected_phase_evaluation_detail,
            phase_origins=evaluation_selected_phase_origins,
            phase_horizon=phase_horizon,
        )
        calibration_phase_selected_timeline = calibration_minute_timeline.copy()
        calibration_phase_selected_timeline["phase_pred"] = calibration_selected_phase_series.reindex(
            pd.DatetimeIndex(pd.to_datetime(calibration_phase_selected_timeline["timestamp"], errors="raise"))
        ).to_numpy(dtype=float)
        minute_phase_selected_timeline = minute_timeline.copy()
        minute_phase_selected_timeline["phase_pred"] = evaluation_selected_phase_series.reindex(
            pd.DatetimeIndex(pd.to_datetime(minute_phase_selected_timeline["timestamp"], errors="raise"))
        ).to_numpy(dtype=float)
        _, _, calibration_selected_phase_summary = _evaluate_control_scope(
            minute_timeline=calibration_phase_selected_timeline,
            nowcast_anchor=_placeholder_nowcast_anchor(),
            lock_interval=lock_interval,
        )
        _, _, evaluation_selected_phase_summary = _evaluate_control_scope(
            minute_timeline=minute_phase_selected_timeline,
            nowcast_anchor=_placeholder_nowcast_anchor(),
            lock_interval=lock_interval,
        )
        phase_stack_selected_row["replay_run_dir"] = str(selected_phase_run_dir)
    else:
        calibration_selected_phase_series = calibration_phase_stack_predictions[selected_phase_candidate_label]
        evaluation_selected_phase_series = evaluation_phase_stack_predictions[selected_phase_candidate_label]
        calibration_selected_phase_summary = calibration_phase_stack_summaries[selected_phase_candidate_label]
        evaluation_selected_phase_summary = evaluation_phase_stack_summaries[selected_phase_candidate_label]
    calibration_minute_timeline["phase_pred"] = calibration_selected_phase_series.reindex(
        pd.DatetimeIndex(pd.to_datetime(calibration_minute_timeline["timestamp"], errors="raise"))
    ).to_numpy(dtype=float)
    minute_timeline["phase_pred"] = evaluation_selected_phase_series.reindex(
        pd.DatetimeIndex(pd.to_datetime(minute_timeline["timestamp"], errors="raise"))
    ).to_numpy(dtype=float)
    phase_stack_guard_summary = _phase_stack_guard_summary_frame(
        calibration_summary=calibration_selected_phase_summary,
        evaluation_summary=evaluation_selected_phase_summary,
        decision=phase_stack_guard,
    )
    if not rolling_phase_guard_calibration_summary.empty and not rolling_phase_guard_evaluation_summary.empty:
        phase_stack_guard_summary = pd.concat(
            [
                phase_stack_guard_summary,
                _phase_stack_guard_summary_frame(
                    calibration_summary=rolling_phase_guard_calibration_summary,
                    evaluation_summary=rolling_phase_guard_evaluation_summary,
                    combined_summary=rolling_phase_guard_combined_summary,
                    combined_scope_name="rolling_combined",
                    decision=phase_stack_rolling_support_guard,
                    guard_name="rolling_support_guard",
                    calibration_scope_name="rolling_calibration",
                    evaluation_scope_name="rolling_evaluation",
                ),
            ],
            ignore_index=True,
        )

    nowcast_exact_started_at_utc, nowcast_exact_started_perf = _start_runtime_step()
    nowcast_anchor = _benchmark_nowcast_layer(calibration_minute_timeline, minute_timeline)
    if not bool(nowcast_anchor["upstream_anchor"]["beats_persistence"]):
        warnings.append("stage5_nowcast_anchor_fell_back_to_persistence")
    if str(nowcast_anchor["candidate_label"]) != str(nowcast_anchor["upstream_candidate_label"]):
        warnings.append(
            "nowcast_control_candidate_swapped:"
            f"{nowcast_anchor['upstream_candidate_label']}->{nowcast_anchor['candidate_label']}"
        )
    calibration_minute_timeline["nowcast_pred"] = _apply_nowcast_updates(
        pd.Series(
            calibration_minute_timeline["phase_pred"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(calibration_minute_timeline["timestamp"], errors="raise")),
            dtype=float,
        ),
        cast(pd.Series, nowcast_anchor["benchmark_prediction_series"]),
    ).to_numpy(dtype=float)
    minute_timeline["nowcast_pred"] = _apply_nowcast_updates(
        pd.Series(
            minute_timeline["phase_pred"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(minute_timeline["timestamp"], errors="raise")),
            dtype=float,
        ),
        cast(pd.Series, nowcast_anchor["prediction_series"]),
    ).to_numpy(dtype=float)
    calibration_interval_timeline, calibration_by_cycle, calibration_summary = _evaluate_control_scope(
        minute_timeline=calibration_minute_timeline,
        nowcast_anchor=nowcast_anchor,
        lock_interval=lock_interval,
    )
    interval_timeline, by_cycle, summary = _evaluate_control_scope(
        minute_timeline=minute_timeline,
        nowcast_anchor=nowcast_anchor,
        lock_interval=lock_interval,
    )
    scope_summary = pd.concat(
        [
            calibration_summary.assign(scope="calibration"),
            summary.assign(scope="evaluation"),
        ],
        ignore_index=True,
    )
    _append_runtime_step(
        runtime_profile_records,
        step="benchmark_nowcast_and_evaluate_exact_control",
        category="evaluation",
        started_at_utc=nowcast_exact_started_at_utc,
        started_perf=nowcast_exact_started_perf,
        nowcast_candidate_label=str(nowcast_anchor["candidate_label"]),
        exact_lock_mae=float(summary["lock_mae"].min()) if not summary.empty else float("nan"),
        evaluation_cycle_count=int(len(by_cycle)),
    )

    refresh_eval_started_at_utc, refresh_eval_started_perf = _start_runtime_step()
    calibration_refresh_inputs, calibration_refresh_signal = _build_day_ahead_refresh_scope_inputs(
        cycle_origins=calibration_selected_cycle_origins,
        refresh_origins=[
            timestamp for timestamp in calibration_refresh_origins if timestamp in calibration_selected_hourly_origins
        ],
        actual_minute_base=actual_minute_base,
        minute_feature_frame=actual_minute_feature_frame,
        day_ahead=day_ahead,
        day_ahead_refresh=day_ahead_refresh,
        result_key="benchmark_result",
        day_ahead_horizon=day_ahead_horizon,
    )
    evaluation_refresh_inputs, evaluation_refresh_signal = _build_day_ahead_refresh_scope_inputs(
        cycle_origins=evaluation_selected_cycle_origins,
        refresh_origins=[
            timestamp for timestamp in evaluation_refresh_origins if timestamp in evaluation_selected_hourly_origins
        ],
        actual_minute_base=actual_minute_base,
        minute_feature_frame=actual_minute_feature_frame,
        day_ahead=day_ahead,
        day_ahead_refresh=day_ahead_refresh,
        result_key="result",
        day_ahead_horizon=day_ahead_horizon,
    )
    if day_ahead_refresh is not None and calibration_refresh_inputs:
        (
            refresh_thresholds,
            refresh_threshold_grid,
            calibration_refresh_decisions,
            calibration_refresh_by_cycle,
            calibration_refresh_summary,
        ) = _select_day_ahead_refresh_thresholds(
            calibration_cycle_inputs=calibration_refresh_inputs,
            calibration_signal_frame=calibration_refresh_signal,
            day_ahead_refresh=day_ahead_refresh,
            result_key="benchmark_result",
            day_ahead_horizon=day_ahead_horizon,
            lock_interval=lock_interval,
        )
        refresh_decisions, refresh_by_cycle, refresh_summary = _evaluate_day_ahead_refresh_policy(
            cycle_inputs=evaluation_refresh_inputs,
            signal_frame=evaluation_refresh_signal,
            thresholds=refresh_thresholds,
            day_ahead_refresh=day_ahead_refresh,
            result_key="result",
            day_ahead_horizon=day_ahead_horizon,
            lock_interval=lock_interval,
        )
        refresh_policy = _recommend_day_ahead_refresh(refresh_summary, refresh_decisions)
        if (
            not bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"])
            or not rolling_calibration_cycle_origins
            or not rolling_evaluation_cycle_origins
        ) and str(refresh_policy["recommended_policy"]) != "triggered_refresh":
            warnings.append("day_ahead_refresh_not_promoted")
        if (
            not bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"])
            or not rolling_calibration_cycle_origins
            or not rolling_evaluation_cycle_origins
        ) and float(refresh_policy["trigger_rate"]) > float(
            MULTIRES_FORECAST_CONTROL["day_ahead_refresh_max_trigger_rate"]
        ):
            warnings.append("day_ahead_refresh_trigger_near_always_on")
    else:
        refresh_thresholds = _default_day_ahead_refresh_thresholds()
        refresh_threshold_grid = pd.DataFrame()
        calibration_refresh_decisions = pd.DataFrame()
        calibration_refresh_by_cycle = pd.DataFrame()
        calibration_refresh_summary = pd.DataFrame()
        refresh_decisions = pd.DataFrame()
        refresh_by_cycle = pd.DataFrame()
        refresh_summary = pd.DataFrame()
        refresh_policy = None
    _append_runtime_step(
        runtime_profile_records,
        step="evaluate_day_ahead_refresh_policy",
        category="evaluation",
        started_at_utc=refresh_eval_started_at_utc,
        started_perf=refresh_eval_started_perf,
        refresh_enabled=bool(day_ahead_refresh is not None),
        calibration_input_count=int(len(calibration_refresh_inputs)),
        evaluation_input_count=int(len(evaluation_refresh_inputs)),
    )

    rolling_calibration_interval_timeline = pd.DataFrame()
    rolling_interval_timeline = pd.DataFrame()
    rolling_calibration_by_cycle = pd.DataFrame()
    rolling_by_cycle = pd.DataFrame()
    rolling_calibration_summary = pd.DataFrame()
    rolling_summary = pd.DataFrame()
    rolling_scope_summary = pd.DataFrame()
    rolling_layer_inference = pd.DataFrame()
    rolling_refresh_thresholds: dict[str, float] = _default_day_ahead_refresh_thresholds()
    rolling_refresh_threshold_grid = pd.DataFrame()
    rolling_refresh_decisions = pd.DataFrame()
    rolling_refresh_by_cycle = pd.DataFrame()
    rolling_refresh_summary = pd.DataFrame()
    rolling_refresh_policy = None
    rolling_calibration_refresh_decisions = pd.DataFrame()
    rolling_calibration_refresh_by_cycle = pd.DataFrame()
    rolling_calibration_refresh_summary = pd.DataFrame()
    rolling_eval_started_at_utc, rolling_eval_started_perf = _start_runtime_step()
    if (
        bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"])
        and rolling_calibration_cycle_origins
        and rolling_evaluation_cycle_origins
    ):
        if rolling_phase_support_context:
            rolling_hourly_origins = cast(list[pd.Timestamp], rolling_phase_support_context["rolling_hourly_origins"])
            rolling_hourly_evaluation_origins = cast(
                list[pd.Timestamp],
                rolling_phase_support_context["rolling_hourly_evaluation_origins"],
            )
            rolling_refresh_origins = cast(list[pd.Timestamp], rolling_phase_support_context["rolling_refresh_origins"])
            rolling_refresh_evaluation_origins = cast(
                list[pd.Timestamp],
                rolling_phase_support_context["rolling_refresh_evaluation_origins"],
            )
            rolling_day_ahead_calibration_detail = cast(
                pd.DataFrame,
                rolling_phase_support_context["day_ahead_calibration_detail"],
            ).copy()
            rolling_day_ahead_evaluation_detail = cast(
                pd.DataFrame,
                rolling_phase_support_context["day_ahead_evaluation_detail"],
            ).copy()
            if str(phase_stack_guard["recommended_policy"]) == "phase_candidate":
                rolling_calibration_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["calibration_phase_timeline"],
                ).copy()
                rolling_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["evaluation_phase_timeline"],
                ).copy()
            else:
                rolling_calibration_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["calibration_hourly_timeline"],
                ).copy()
                rolling_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["evaluation_hourly_timeline"],
                ).copy()
        else:
            rolling_hourly_origins = _layer_update_origins(
                cycle_origins=rolling_calibration_cycle_origins,
                update_interval_minutes=hourly_horizon,
                cycle_horizon_minutes=day_ahead_horizon,
            )
            rolling_hourly_evaluation_origins = _layer_update_origins(
                cycle_origins=rolling_evaluation_cycle_origins,
                update_interval_minutes=hourly_horizon,
                cycle_horizon_minutes=day_ahead_horizon,
            )
            rolling_refresh_origins = _day_ahead_refresh_origins(rolling_calibration_cycle_origins)
            rolling_refresh_evaluation_origins = _day_ahead_refresh_origins(rolling_evaluation_cycle_origins)
            phase_replay_metadata = _resolve_phase_stack_replay_metadata(
                phase_payload=phase,
                phase_stack_selected_row=phase_stack_selected_row,
                phase_stack_guard=phase_stack_guard,
            )
            rolling_phase_support_context = _prepare_rolling_phase_support_context(
                actual_minute_base=actual_minute_base,
                cache_root=cache_root,
                day_ahead=day_ahead,
                hourly=hourly,
                phase=phase,
                day_ahead_horizon=day_ahead_horizon,
                hourly_horizon=hourly_horizon,
                phase_horizon=phase_horizon,
                rolling_calibration_cycle_origins=rolling_calibration_cycle_origins,
                rolling_evaluation_cycle_origins=rolling_evaluation_cycle_origins,
                phase_replay_metadata=phase_replay_metadata,
            )
            if str(phase_stack_guard["recommended_policy"]) == "phase_candidate":
                rolling_calibration_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["calibration_phase_timeline"],
                ).copy()
                rolling_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["evaluation_phase_timeline"],
                ).copy()
            else:
                rolling_calibration_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["calibration_hourly_timeline"],
                ).copy()
                rolling_minute_timeline = cast(
                    pd.DataFrame,
                    rolling_phase_support_context["evaluation_hourly_timeline"],
                ).copy()
            rolling_day_ahead_calibration_detail = cast(
                pd.DataFrame,
                rolling_phase_support_context["day_ahead_calibration_detail"],
            ).copy()
            rolling_day_ahead_evaluation_detail = cast(
                pd.DataFrame,
                rolling_phase_support_context["day_ahead_evaluation_detail"],
            ).copy()
        if day_ahead_refresh is not None:
            refresh_selection = dict(cast(dict[str, Any], day_ahead_refresh["selection"]))
            rolling_refresh_origins = _representable_selection_origins(
                selection=refresh_selection,
                horizon_minutes=int(day_ahead_horizon),
                origin_timestamps=rolling_refresh_origins,
            )
            rolling_refresh_evaluation_origins = _representable_selection_origins(
                selection=refresh_selection,
                horizon_minutes=int(day_ahead_horizon),
                origin_timestamps=rolling_refresh_evaluation_origins,
            )
        rolling_calibration_nowcast_series = _selected_nowcast_prediction_series(
            nowcast_anchor=nowcast_anchor,
            scope_name="calibration",
        )
        rolling_evaluation_nowcast_series = _selected_nowcast_prediction_series(
            nowcast_anchor=nowcast_anchor,
            scope_name="evaluation",
        )
        rolling_calibration_minute_timeline = _apply_selected_nowcast_to_timeline(
            minute_timeline=rolling_calibration_minute_timeline,
            prediction_series=rolling_calibration_nowcast_series,
        )
        rolling_minute_timeline = _apply_selected_nowcast_to_timeline(
            minute_timeline=rolling_minute_timeline,
            prediction_series=rolling_evaluation_nowcast_series,
        )
        (
            rolling_calibration_interval_timeline,
            rolling_calibration_by_cycle,
            rolling_calibration_summary,
        ) = _evaluate_control_scope(
            minute_timeline=rolling_calibration_minute_timeline,
            nowcast_anchor=nowcast_anchor,
            lock_interval=lock_interval,
        )
        rolling_interval_timeline, rolling_by_cycle, rolling_summary = _evaluate_control_scope(
            minute_timeline=rolling_minute_timeline,
            nowcast_anchor=nowcast_anchor,
            lock_interval=lock_interval,
        )
        if day_ahead_refresh is not None:
            with TemporaryDirectory(prefix="elf_forecast_control_rolling_refresh_") as rolling_temp_dir:
                rolling_temp_root = Path(rolling_temp_dir)
                rolling_refresh_union_origins = sorted(
                    {
                        pd.Timestamp(value)
                        for value in [*rolling_refresh_origins, *rolling_refresh_evaluation_origins]
                    }
                )
                rolling_day_ahead_refresh_union = _run_cached_rollout_evaluation(
                    cache_root=cache_root,
                    temp_output_root=rolling_temp_root / "day_ahead_refresh_union",
                    layer_role="day_ahead_refresh",
                    selection=dict(cast(dict[str, Any], day_ahead_refresh["selection"])),
                    horizon_minutes=int(day_ahead_horizon),
                    origin_policy=str(day_ahead["origin_policy"]),
                    selection_target=str(day_ahead["selection_target"]),
                    origin_timestamps=rolling_refresh_union_origins,
                    capture_path_details=True,
                    candidate_scope="selected_only",
                    persist_artifacts=False,
                )
                rolling_day_ahead_refresh_calibration = _subset_rollout_result_for_origins(
                    rolling_day_ahead_refresh_union,
                    origin_timestamps=rolling_refresh_origins,
                    require_path_details=True,
                )
                rolling_day_ahead_refresh_evaluation = _subset_rollout_result_for_origins(
                    rolling_day_ahead_refresh_union,
                    origin_timestamps=rolling_refresh_evaluation_origins,
                    require_path_details=True,
                )
                rolling_day_ahead_refresh_payload = {
                    "candidate_label": str(day_ahead_refresh["candidate_label"]),
                    "benchmark_result": {
                        "detail_by_origin": rolling_day_ahead_refresh_calibration["detail_by_origin"].copy()
                    },
                    "result": {
                        "detail_by_origin": rolling_day_ahead_refresh_evaluation["detail_by_origin"].copy()
                    },
                }
                rolling_calibration_refresh_inputs, rolling_calibration_refresh_signal = (
                    _build_day_ahead_refresh_scope_inputs(
                        cycle_origins=rolling_calibration_cycle_origins,
                        refresh_origins=[
                            timestamp for timestamp in rolling_refresh_origins if timestamp in rolling_hourly_origins
                        ],
                        actual_minute_base=actual_minute_base,
                        minute_feature_frame=actual_minute_feature_frame,
                        day_ahead={
                            **day_ahead,
                            "benchmark_result": {
                                "detail_by_origin": rolling_day_ahead_calibration_detail.copy()
                            },
                        },
                        day_ahead_refresh=rolling_day_ahead_refresh_payload,
                        result_key="benchmark_result",
                        day_ahead_horizon=day_ahead_horizon,
                    )
                )
                rolling_evaluation_refresh_inputs, rolling_evaluation_refresh_signal = (
                    _build_day_ahead_refresh_scope_inputs(
                        cycle_origins=rolling_evaluation_cycle_origins,
                        refresh_origins=[
                            timestamp
                            for timestamp in rolling_refresh_evaluation_origins
                            if timestamp in rolling_hourly_evaluation_origins
                        ],
                        actual_minute_base=actual_minute_base,
                        minute_feature_frame=actual_minute_feature_frame,
                        day_ahead={
                            **day_ahead,
                            "result": {
                                "detail_by_origin": rolling_day_ahead_evaluation_detail.copy()
                            },
                        },
                        day_ahead_refresh=rolling_day_ahead_refresh_payload,
                        result_key="result",
                        day_ahead_horizon=day_ahead_horizon,
                    )
                )
                if rolling_calibration_refresh_inputs:
                    (
                        rolling_refresh_thresholds,
                        rolling_refresh_threshold_grid,
                        rolling_calibration_refresh_decisions,
                        rolling_calibration_refresh_by_cycle,
                        rolling_calibration_refresh_summary,
                    ) = _select_day_ahead_refresh_thresholds(
                        calibration_cycle_inputs=rolling_calibration_refresh_inputs,
                        calibration_signal_frame=rolling_calibration_refresh_signal,
                        day_ahead_refresh=rolling_day_ahead_refresh_payload,
                        result_key="benchmark_result",
                        day_ahead_horizon=day_ahead_horizon,
                        lock_interval=lock_interval,
                    )
                    (
                        rolling_refresh_decisions,
                        rolling_refresh_by_cycle,
                        rolling_refresh_summary,
                    ) = _evaluate_day_ahead_refresh_policy(
                        cycle_inputs=rolling_evaluation_refresh_inputs,
                        signal_frame=rolling_evaluation_refresh_signal,
                        thresholds=rolling_refresh_thresholds,
                        day_ahead_refresh=rolling_day_ahead_refresh_payload,
                        result_key="result",
                        day_ahead_horizon=day_ahead_horizon,
                        lock_interval=lock_interval,
                    )
                rolling_refresh_policy = _recommend_day_ahead_refresh(
                    rolling_refresh_summary,
                    rolling_refresh_decisions,
                )
        rolling_calibration_by_cycle = _annotate_cycle_scope(
            by_cycle=rolling_calibration_by_cycle,
            origin_catalog=rolling_origin_catalog,
            scope_name="rolling_calibration",
        )
        rolling_by_cycle = _annotate_cycle_scope(
            by_cycle=rolling_by_cycle,
            origin_catalog=rolling_origin_catalog,
            scope_name="rolling_evaluation",
        )
        rolling_scope_summary = pd.concat(
            [
                _rolling_control_summary_frame(by_cycle=rolling_calibration_by_cycle, scope_name="rolling_calibration"),
                _rolling_control_summary_frame(by_cycle=rolling_by_cycle, scope_name="rolling_evaluation"),
            ],
            ignore_index=True,
        )
        rolling_layer_inference = pd.concat(
            [
                _control_layer_inference_frame(by_cycle=rolling_calibration_by_cycle, scope_name="rolling_calibration"),
                _control_layer_inference_frame(by_cycle=rolling_by_cycle, scope_name="rolling_evaluation"),
                _control_layer_inference_frame(
                    by_cycle=pd.concat([rolling_calibration_by_cycle, rolling_by_cycle], ignore_index=True),
                    scope_name="rolling_combined",
                ),
            ],
            ignore_index=True,
        )
        if rolling_refresh_policy is not None and str(rolling_refresh_policy["recommended_policy"]) != "triggered_refresh":
            warnings.append("rolling_day_ahead_refresh_not_promoted")
    _append_runtime_step(
        runtime_profile_records,
        step="evaluate_rolling_control_scope",
        category="evaluation",
        started_at_utc=rolling_eval_started_at_utc,
        started_perf=rolling_eval_started_perf,
        rolling_enabled=bool(
            bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"])
            and rolling_calibration_cycle_origins
            and rolling_evaluation_cycle_origins
        ),
        rolling_evaluation_cycle_count=int(len(rolling_by_cycle)),
    )
    effective_refresh_threshold_source = "exact_control"
    effective_refresh_thresholds = dict(refresh_thresholds)
    effective_refresh_threshold_grid = refresh_threshold_grid.copy()
    effective_refresh_decisions = refresh_decisions.copy()
    effective_refresh_by_cycle = refresh_by_cycle.copy()
    effective_refresh_summary = refresh_summary.copy()
    effective_refresh_policy = refresh_policy
    effective_calibration_refresh_decisions = calibration_refresh_decisions.copy()
    effective_calibration_refresh_by_cycle = calibration_refresh_by_cycle.copy()
    effective_calibration_refresh_summary = calibration_refresh_summary.copy()
    if (
        day_ahead_refresh is not None
        and rolling_refresh_policy is not None
        and evaluation_refresh_inputs
    ):
        effective_refresh_threshold_source = "rolling_benchmark"
        effective_refresh_thresholds = dict(rolling_refresh_thresholds)
        effective_refresh_threshold_grid = rolling_refresh_threshold_grid.copy()
        if calibration_refresh_inputs:
            (
                effective_calibration_refresh_decisions,
                effective_calibration_refresh_by_cycle,
                effective_calibration_refresh_summary,
            ) = _evaluate_day_ahead_refresh_policy(
                cycle_inputs=calibration_refresh_inputs,
                signal_frame=calibration_refresh_signal,
                thresholds=effective_refresh_thresholds,
                day_ahead_refresh=day_ahead_refresh,
                result_key="benchmark_result",
                day_ahead_horizon=day_ahead_horizon,
                lock_interval=lock_interval,
            )
        (
            effective_refresh_decisions,
            effective_refresh_by_cycle,
            effective_refresh_summary,
        ) = _evaluate_day_ahead_refresh_policy(
            cycle_inputs=evaluation_refresh_inputs,
            signal_frame=evaluation_refresh_signal,
            thresholds=effective_refresh_thresholds,
            day_ahead_refresh=day_ahead_refresh,
            result_key="result",
            day_ahead_horizon=day_ahead_horizon,
            lock_interval=lock_interval,
        )
        effective_refresh_policy = _recommend_day_ahead_refresh(
            effective_refresh_summary,
            effective_refresh_decisions,
        )
    benchmark_frames = [
        day_ahead["candidate_benchmarks"].copy().assign(origin_split_scope="calibration"),
        hourly["candidate_benchmarks"].copy().assign(origin_split_scope="calibration"),
        phase["candidate_benchmarks"].copy().assign(origin_split_scope="calibration"),
        cast(pd.DataFrame, nowcast_anchor["candidate_benchmarks"]).copy(),
    ]
    candidate_benchmarks = pd.concat(benchmark_frames, ignore_index=True).reset_index(drop=True)
    candidate_benchmarks.to_csv(
        run_dir / "control_layer_candidate_benchmarks.csv",
        index=False,
        float_format="%.6f",
    )
    phase_stack_candidate_benchmarks.to_csv(
        run_dir / "phase_stack_candidate_benchmarks.csv",
        index=False,
        float_format="%.6f",
    )
    phase_stack_guard_summary.to_csv(
        run_dir / "phase_stack_guard_summary.csv",
        index=False,
        float_format="%.6f",
    )

    _plot_lock_mae(summary, run_dir)
    _plot_example_cycle(interval_timeline, run_dir)
    if not effective_refresh_summary.empty:
        _plot_day_ahead_refresh_policy(effective_refresh_summary, run_dir)
    if not rolling_by_cycle.empty:
        _plot_rolling_control_lock_distribution(rolling_by_cycle, run_dir)
    if not rolling_layer_inference.empty:
        _plot_control_layer_gain_ci(rolling_layer_inference, run_dir)
    if not phase_stack_candidate_benchmarks.empty:
        _plot_phase_stack_candidates(phase_stack_candidate_benchmarks, run_dir)
    figure_entries = [
        FigureGuideEntry(
            filename="fig_control_lock_mae.png",
            title="Locked-interval MAE progression",
            intent="Show how much each control layer reduces the next locked 15-minute interval error.",
            how_to_read="Each bar is the lock MAE after applying one more layer of updates to the frozen day-ahead forecast.",
            look_for="A clear downward progression from day-ahead to hourly to phase to nowcast updates; if that pattern breaks, the control stack is not adding value.",
        ),
        FigureGuideEntry(
            filename="fig_control_example_cycle.png",
            title="Example control cycle",
            intent="Show how the full 24-hour profile changes as hourly, phase, and minute-level updates are applied.",
            how_to_read="Compare the actual line with the frozen day-ahead path, then with the hourly-updated, phase-updated, and nowcast-updated paths over the same cycle.",
            look_for="Whether intraday updates pull the forecast toward actual peaks and troughs before the next costly interval locks in, especially in the last-minute correction layer.",
        ),
    ]
    if not effective_refresh_summary.empty:
        figure_entries.append(
            FigureGuideEntry(
                filename="fig_day_ahead_refresh_policy.png",
                title="Day-ahead refresh policy comparison",
                intent="Compare the frozen 24-hour profile with unconditional and triggered residual-refresh policies on the same exact control cycles.",
                how_to_read="Compare the profile-shape and lock MAE bars for the frozen, unconditional, and triggered scenarios. Lower is better for both metrics.",
                look_for="Triggered refresh should improve profile-shape error without increasing lock MAE versus the frozen path. If unconditional refresh wins but triggered does not, the trigger logic still needs work.",
            )
        )
    if not rolling_by_cycle.empty:
        figure_entries.append(
            FigureGuideEntry(
                filename="fig_control_lock_distribution.png",
                title="Rolling control lock-MAE distribution",
                intent="Show whether the selected stack improves locked-interval error across a broader set of control cycles, not just the exact evaluation slice.",
                how_to_read="Each box summarizes the rolling benchmark distribution for one layer's 15-minute lock MAE. Lower boxes and medians are better.",
                look_for="A lower hourly box than day-ahead, and a lower nowcast box than phase. If distributions overlap heavily or move upward, the stacked gain is not robust.",
            )
        )
    if not rolling_layer_inference.empty:
        figure_entries.append(
            FigureGuideEntry(
                filename="fig_control_layer_gain_ci.png",
                title="Rolling layer-gain confidence intervals",
                intent="Show the uncertainty around stacked layer gains on the rolling benchmark.",
                how_to_read="Each point is the mean gain versus the previous layer, with a bootstrap confidence interval. Positive values mean the later layer reduced error.",
                look_for="Intervals entirely above zero indicate stronger evidence that the layer helps beyond noise in a small sample of control cycles.",
            )
        )
    if not phase_stack_candidate_benchmarks.empty:
        figure_entries.append(
            FigureGuideEntry(
                filename="fig_phase_stack_candidates.png",
                title="Phase stack candidate frontier",
                intent="Show which phase candidates improve lock error after the hourly stack without giving back too much profile quality.",
                how_to_read="Candidates in the upper-left improve lock MAE while keeping profile regression low. The selected candidate is annotated directly on the plot.",
                look_for="A learned candidate that clears the guard thresholds and beats the persistence passthrough baseline on the stacked surface.",
            )
        )
    write_figure_guide(
        output_path=run_dir / "figure_guide.md",
        stage_title="Stage-10 Forecast-Control Figures",
        stage_purpose=(
            "These figures explain whether the stacked day-ahead plus intraday "
            "update policy actually reduces operational error on shared control cycles."
        ),
        figures=figure_entries,
    )

    selected_refresh_threshold_row = (
        effective_refresh_threshold_grid.iloc[0].to_dict() if not effective_refresh_threshold_grid.empty else {}
    )
    phase_applied_selection_artifact = ""
    if str(phase_stack_guard["applied_candidate_label"]) == str(hourly["candidate_label"]):
        phase_applied_selection_artifact = _relative_artifact_path(hourly["result"]["run_dir"])
    elif str(phase_stack_guard["applied_candidate_label"]) == str(phase["candidate_label"]):
        phase_applied_selection_artifact = _relative_artifact_path(phase["result"]["run_dir"])
    else:
        phase_applied_selection_artifact = str(phase_stack_selected_row.get("replay_run_dir", ""))

    policy = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "control_cycle_scope": {
            "calibration_splits": [str(value) for value in MULTIRES_FORECAST_CONTROL["calibration_splits"]],
            "evaluation_splits": [str(value) for value in MULTIRES_FORECAST_CONTROL["evaluation_splits"]],
            "calibration_cycle_count": int(len(calibration_cycle_origins)),
            "evaluation_cycle_count": int(len(evaluation_cycle_origins)),
            "cycle_origin_stride_minutes": int(MULTIRES_FORECAST_CONTROL["cycle_origin_stride_minutes"]),
            "rolling_benchmark_enabled": bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"]),
            "rolling_origin_stride_minutes": int(MULTIRES_FORECAST_CONTROL["rolling_benchmark_origin_stride_minutes"]),
            "rolling_calibration_cycle_count": int(len(rolling_calibration_cycle_origins)),
            "rolling_evaluation_cycle_count": int(len(rolling_evaluation_cycle_origins)),
        },
        "day_ahead": {
            "horizon_minutes": day_ahead_horizon,
            "candidate_label": str(day_ahead["candidate_label"]),
            "candidate_type": str(day_ahead["candidate_type"]),
            "source_model_label": str(day_ahead["source_model_label"]),
            "target_mode": str(day_ahead["target_mode"]),
            "selection_target": str(day_ahead["selection_target"]),
            "control_selection_metric": str(day_ahead["control_selection_metric"]),
            "control_selection_metric_value": float(day_ahead["control_selection_metric_value"]),
            "control_selection_metric_pct": float(day_ahead["control_selection_metric_pct"]),
            "evaluation_selection_metric_value": float(day_ahead["evaluation_selection_metric_value"]),
            "evaluation_selection_metric_pct": float(day_ahead["evaluation_selection_metric_pct"]),
            "control_selection_mode": str(day_ahead["control_selection_mode"]),
            "upstream_candidate_label": str(day_ahead["upstream_candidate_label"]),
            "candidate_pool_count": int(day_ahead["candidate_pool_count"]),
            "benchmark_origin_mode": str(day_ahead["benchmark_origin_mode"]),
            "benchmark_origin_count": int(day_ahead["benchmark_origin_count"]),
            "evaluation_origin_count": int(day_ahead["evaluation_origin_count"]),
            "evaluation_benchmark_origin_mode": str(day_ahead["evaluation_benchmark_origin_mode"]),
            "evaluation_benchmark_origin_count": int(day_ahead["evaluation_benchmark_origin_count"]),
            "origin_policy": str(day_ahead["origin_policy"]),
            "selection_context": day_ahead["selection"],
            "upstream_selection_context": day_ahead["upstream_selection"],
            "selection_artifact": _relative_artifact_path(day_ahead["result"]["run_dir"]),
            "benchmark_replay_cache_hits": int(day_ahead["benchmark_replay_cache_hits"]),
            "benchmark_replay_cache_misses": int(day_ahead["benchmark_replay_cache_misses"]),
            "evaluation_replay_cache_hits": int(day_ahead["evaluation_replay_cache_hits"]),
            "evaluation_replay_cache_misses": int(day_ahead["evaluation_replay_cache_misses"]),
            "selected_benchmark_replay_cache_status": str(day_ahead["selected_benchmark_replay_cache_status"]),
            "selected_replay_cache_status": str(day_ahead["selected_replay_cache_status"]),
            "selected_replay_cache_artifact": str(day_ahead["selected_replay_cache_artifact"]),
        },
        "hourly": {
            "horizon_minutes": hourly_horizon,
            "candidate_label": str(hourly["candidate_label"]),
            "candidate_type": str(hourly["candidate_type"]),
            "source_model_label": str(hourly["source_model_label"]),
            "target_mode": str(hourly["target_mode"]),
            "selection_target": str(hourly["selection_target"]),
            "control_selection_metric": str(hourly["control_selection_metric"]),
            "control_selection_metric_value": float(hourly["control_selection_metric_value"]),
            "control_selection_metric_pct": float(hourly["control_selection_metric_pct"]),
            "evaluation_selection_metric_value": float(hourly["evaluation_selection_metric_value"]),
            "evaluation_selection_metric_pct": float(hourly["evaluation_selection_metric_pct"]),
            "control_selection_mode": str(hourly["control_selection_mode"]),
            "upstream_candidate_label": str(hourly["upstream_candidate_label"]),
            "candidate_pool_count": int(hourly["candidate_pool_count"]),
            "benchmark_origin_mode": str(hourly["benchmark_origin_mode"]),
            "benchmark_origin_count": int(hourly["benchmark_origin_count"]),
            "evaluation_origin_count": int(hourly["evaluation_origin_count"]),
            "evaluation_benchmark_origin_mode": str(hourly["evaluation_benchmark_origin_mode"]),
            "evaluation_benchmark_origin_count": int(hourly["evaluation_benchmark_origin_count"]),
            "origin_policy": str(hourly["origin_policy"]),
            "selection_context": hourly["selection"],
            "upstream_selection_context": hourly["upstream_selection"],
            "selection_artifact": _relative_artifact_path(hourly["result"]["run_dir"]),
            "benchmark_replay_cache_hits": int(hourly["benchmark_replay_cache_hits"]),
            "benchmark_replay_cache_misses": int(hourly["benchmark_replay_cache_misses"]),
            "evaluation_replay_cache_hits": int(hourly["evaluation_replay_cache_hits"]),
            "evaluation_replay_cache_misses": int(hourly["evaluation_replay_cache_misses"]),
            "selected_benchmark_replay_cache_status": str(hourly["selected_benchmark_replay_cache_status"]),
            "selected_replay_cache_status": str(hourly["selected_replay_cache_status"]),
            "selected_replay_cache_artifact": str(hourly["selected_replay_cache_artifact"]),
        },
        "phase": {
            "horizon_minutes": phase_horizon,
            "candidate_label": str(phase["candidate_label"]),
            "candidate_type": str(phase["candidate_type"]),
            "source_model_label": str(phase["source_model_label"]),
            "target_mode": str(phase["target_mode"]),
            "selection_target": str(phase["selection_target"]),
            "control_selection_metric": str(phase["control_selection_metric"]),
            "control_selection_metric_value": float(phase["control_selection_metric_value"]),
            "control_selection_metric_pct": float(phase["control_selection_metric_pct"]),
            "evaluation_selection_metric_value": float(phase["evaluation_selection_metric_value"]),
            "evaluation_selection_metric_pct": float(phase["evaluation_selection_metric_pct"]),
            "control_selection_mode": str(phase["control_selection_mode"]),
            "upstream_candidate_label": str(phase["upstream_candidate_label"]),
            "candidate_pool_count": int(phase["candidate_pool_count"]),
            "benchmark_origin_mode": str(phase["benchmark_origin_mode"]),
            "benchmark_origin_count": int(phase["benchmark_origin_count"]),
            "evaluation_origin_count": int(phase["evaluation_origin_count"]),
            "evaluation_benchmark_origin_mode": str(phase["evaluation_benchmark_origin_mode"]),
            "evaluation_benchmark_origin_count": int(phase["evaluation_benchmark_origin_count"]),
            "origin_policy": str(phase["origin_policy"]),
            "selection_context": phase["selection"],
            "upstream_selection_context": phase["upstream_selection"],
            "selection_artifact": _relative_artifact_path(phase["result"]["run_dir"]),
            "benchmark_replay_cache_hits": int(phase["benchmark_replay_cache_hits"]),
            "benchmark_replay_cache_misses": int(phase["benchmark_replay_cache_misses"]),
            "evaluation_replay_cache_hits": int(phase["evaluation_replay_cache_hits"]),
            "evaluation_replay_cache_misses": int(phase["evaluation_replay_cache_misses"]),
            "selected_benchmark_replay_cache_status": str(phase["selected_benchmark_replay_cache_status"]),
            "selected_replay_cache_status": str(phase["selected_replay_cache_status"]),
            "selected_replay_cache_artifact": str(phase["selected_replay_cache_artifact"]),
            "isolated_candidate_label": str(phase_stack_guard["isolated_candidate_label"]),
            "stack_selection_mode": str(phase_stack_guard["selection_mode"]),
            "stack_selected_candidate_label": str(phase_stack_guard["stack_selected_candidate_label"]),
            "stack_candidate_benchmark_metric": str(MULTIRES_FORECAST_CONTROL["phase_stack_selection_metric"]),
            "stack_selected_replay_run_dir": str(phase_stack_selected_row.get("replay_run_dir", "")),
            "stack_guard_enabled": bool(phase_stack_guard["enabled"]),
            "stack_guard_decision_scope": str(phase_stack_guard["decision_scope"]),
            "stack_guard_recommended_policy": str(phase_stack_guard["recommended_policy"]),
            "stack_guard_applied_candidate_label": str(phase_stack_guard["applied_candidate_label"]),
            "stack_guard_applied_selection_artifact": str(phase_applied_selection_artifact),
            "stack_guard_lock_gain_vs_hourly": float(phase_stack_guard["lock_gain_vs_hourly"]),
            "stack_guard_lock_gain_pct_vs_hourly": float(phase_stack_guard["lock_gain_pct_vs_hourly"]),
            "stack_guard_next_lock_regress_vs_hourly": float(phase_stack_guard["next_lock_regress_vs_hourly"]),
            "stack_guard_next_lock_regress_pct_vs_hourly": float(
                phase_stack_guard["next_lock_regress_pct_vs_hourly"]
            ),
            "stack_guard_profile_degrade_vs_hourly": float(phase_stack_guard["profile_degrade_vs_hourly"]),
            "stack_guard_profile_degrade_pct_vs_hourly": float(
                phase_stack_guard["profile_degrade_pct_vs_hourly"]
            ),
            "stack_guard_peak_value_regress_vs_hourly": float(
                phase_stack_guard["peak_value_regress_vs_hourly"]
            ),
            "stack_guard_peak_value_regress_pct_vs_hourly": float(
                phase_stack_guard["peak_value_regress_pct_vs_hourly"]
            ),
            "stack_guard_peak_hit_gain_vs_hourly": float(phase_stack_guard["peak_hit_gain_vs_hourly"]),
            "stack_guard_optimizer_regress_vs_hourly": float(
                phase_stack_guard["optimizer_regress_vs_hourly"]
            ),
            "stack_guard_optimizer_regress_pct_vs_hourly": float(
                phase_stack_guard["optimizer_regress_pct_vs_hourly"]
            ),
            "stack_guard_meets_lock_gain_rule": bool(phase_stack_guard["meets_lock_gain_rule"]),
            "stack_guard_meets_next_lock_rule": bool(phase_stack_guard["meets_next_lock_rule"]),
            "stack_guard_meets_profile_rule": bool(phase_stack_guard["meets_profile_rule"]),
            "stack_guard_meets_peak_value_rule": bool(phase_stack_guard["meets_peak_value_rule"]),
            "stack_guard_meets_peak_hit_rule": bool(phase_stack_guard["meets_peak_hit_rule"]),
            "stack_guard_meets_optimizer_rule": bool(phase_stack_guard["meets_optimizer_rule"]),
            "stack_guard_reason": str(phase_stack_guard["reason"]),
            "rolling_support_enabled": bool(phase_stack_guard.get("rolling_support_enabled", False)),
            "rolling_support_required": bool(phase_stack_guard.get("rolling_support_required", False)),
            "rolling_support_scope": str(phase_stack_guard.get("rolling_support_scope", "")),
            "rolling_support_recommended_policy": str(
                phase_stack_guard.get("rolling_support_recommended_policy", "phase_candidate")
            ),
            "rolling_support_applied_candidate_label": str(
                phase_stack_guard.get("rolling_support_applied_candidate_label", "")
            ),
            "rolling_support_applied_veto": bool(phase_stack_guard.get("rolling_support_applied_veto", False)),
            "rolling_support_lock_gain_vs_hourly": float(
                phase_stack_guard.get("rolling_support_lock_gain_vs_hourly", float("nan"))
            ),
            "rolling_support_lock_gain_pct_vs_hourly": float(
                phase_stack_guard.get("rolling_support_lock_gain_pct_vs_hourly", float("nan"))
            ),
            "rolling_support_next_lock_regress_vs_hourly": float(
                phase_stack_guard.get("rolling_support_next_lock_regress_vs_hourly", float("nan"))
            ),
            "rolling_support_next_lock_regress_pct_vs_hourly": float(
                phase_stack_guard.get("rolling_support_next_lock_regress_pct_vs_hourly", float("nan"))
            ),
            "rolling_support_profile_degrade_vs_hourly": float(
                phase_stack_guard.get("rolling_support_profile_degrade_vs_hourly", float("nan"))
            ),
            "rolling_support_profile_degrade_pct_vs_hourly": float(
                phase_stack_guard.get("rolling_support_profile_degrade_pct_vs_hourly", float("nan"))
            ),
            "rolling_support_meets_lock_gain_rule": bool(
                phase_stack_guard.get("rolling_support_meets_lock_gain_rule", True)
            ),
            "rolling_support_meets_next_lock_rule": bool(
                phase_stack_guard.get("rolling_support_meets_next_lock_rule", True)
            ),
            "rolling_support_meets_profile_rule": bool(
                phase_stack_guard.get("rolling_support_meets_profile_rule", True)
            ),
            "rolling_support_reason": str(phase_stack_guard.get("rolling_support_reason", "")),
        },
        "nowcast_anchor": {
            "horizon_minutes": int(MULTIRES_FORECAST_CONTROL["nowcast_horizon_minutes"]),
            "candidate_label": str(nowcast_anchor["candidate_label"]),
            "candidate_type": str(nowcast_anchor["candidate_type"]),
            "source_model_label": str(nowcast_anchor["source_model_label"]),
            "target_mode": str(nowcast_anchor["target_mode"]),
            "selection_target": str(MULTIRES_FORECAST_CONTROL["nowcast_selection_metric"]),
            "control_selection_metric": str(nowcast_anchor["control_selection_metric"]),
            "control_selection_metric_value": float(nowcast_anchor["control_selection_metric_value"]),
            "control_selection_metric_pct": float(nowcast_anchor["control_selection_metric_pct"]),
            "evaluation_selection_metric_value": float(nowcast_anchor["evaluation_selection_metric_value"]),
            "evaluation_selection_metric_pct": float(nowcast_anchor["evaluation_selection_metric_pct"]),
            "control_selection_mode": str(nowcast_anchor["control_selection_mode"]),
            "upstream_candidate_label": str(nowcast_anchor["upstream_candidate_label"]),
            "candidate_pool_count": int(nowcast_anchor["candidate_pool_count"]),
            "benchmark_origin_mode": str(nowcast_anchor["benchmark_origin_mode"]),
            "benchmark_origin_count": int(nowcast_anchor["benchmark_origin_count"]),
            "evaluation_origin_count": int(nowcast_anchor["evaluation_origin_count"]),
            "selection_artifact": str(nowcast_anchor["selection_artifact"]),
            "control_blend_weight": float(nowcast_anchor.get("control_blend_weight", float("nan"))),
            "blend_base_candidate_label": str(nowcast_anchor.get("blend_base_candidate_label", "")),
            "control_bucket_size_minutes": float(
                nowcast_anchor.get("control_bucket_size_minutes", float("nan"))
            ),
            "control_bucket_weights_json": str(nowcast_anchor.get("control_bucket_weights_json", "")),
            "minute_path_mae": float(nowcast_anchor["minute_path_mae"]),
            "minute_path_mae_pct": float(nowcast_anchor["minute_path_mae_pct"]),
            "advisory_surface_supported": bool(nowcast_anchor.get("advisory_surface_supported", False)),
            "advisory_supported_regime_count": int(nowcast_anchor.get("advisory_supported_regime_count", 0)),
            "advisory_supported_operating_regimes": [
                value
                for value in str(nowcast_anchor.get("advisory_supported_operating_regimes", "")).split(",")
                if value
            ],
            "advisory_surface_candidate_mae_ratio_to_persistence": float(
                nowcast_anchor.get("advisory_surface_candidate_mae_ratio_to_persistence", float("nan"))
            ),
            "advisory_transition_best_ratio_to_persistence": float(
                nowcast_anchor.get("advisory_transition_best_ratio_to_persistence", float("nan"))
            ),
            "advisory_high_ramp_ratio_to_persistence": float(
                nowcast_anchor.get("advisory_high_ramp_ratio_to_persistence", float("nan"))
            ),
            "upstream_selection_context": nowcast_anchor["upstream_anchor"],
        },
        "rolling_benchmark": {
            "enabled": bool(MULTIRES_FORECAST_CONTROL["rolling_benchmark_enabled"]),
            "origin_stride_minutes": int(MULTIRES_FORECAST_CONTROL["rolling_benchmark_origin_stride_minutes"]),
            "calibration_cycle_count": int(len(rolling_calibration_cycle_origins)),
            "evaluation_cycle_count": int(len(rolling_evaluation_cycle_origins)),
            "scope_summary_artifact": (
                "rolling_control_scope_summary.csv" if not rolling_scope_summary.empty else ""
            ),
            "layer_inference_artifact": (
                "rolling_control_layer_inference.csv" if not rolling_layer_inference.empty else ""
            ),
        },
    }
    if effective_refresh_policy is not None and day_ahead_refresh is not None:
        trigger_reason_counts = (
            effective_refresh_decisions.loc[
                effective_refresh_decisions["refresh_triggered"].astype(bool), "trigger_reasons"
            ]
            .astype("string")
            .value_counts(dropna=False)
            .to_dict()
            if not effective_refresh_decisions.empty
            else {}
        )
        policy["day_ahead_refresh"] = {
            "enabled": True,
            "candidate_label": str(day_ahead_refresh["candidate_label"]),
            "refresh_interval_minutes": int(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_interval_minutes"]),
            "lookback_minutes": int(MULTIRES_FORECAST_CONTROL["day_ahead_refresh_lookback_minutes"]),
            "residual_drift_mae_pct_threshold": float(
                effective_refresh_thresholds["residual_drift_mae_pct_threshold"]
            ),
            "transition_mae_pct_threshold": float(effective_refresh_thresholds["transition_mae_pct_threshold"]),
            "activity_ratio_shift_threshold": float(
                effective_refresh_thresholds["activity_ratio_shift_threshold"]
            ),
            "threshold_source": str(effective_refresh_threshold_source),
            "residual_drift_quantile": float(selected_refresh_threshold_row.get("residual_drift_quantile", float("nan"))),
            "transition_quantile": float(selected_refresh_threshold_row.get("transition_quantile", float("nan"))),
            "activity_ratio_shift_quantile": float(
                selected_refresh_threshold_row.get("activity_ratio_shift_quantile", float("nan"))
            ),
            "trigger_mode": str(effective_refresh_thresholds.get("trigger_mode", "any")),
            "recommended_policy": str(effective_refresh_policy["recommended_policy"]),
            "promotion_primary_metric": str(effective_refresh_policy["promotion_primary_metric"]),
            "promotion_guardrail_metric": str(effective_refresh_policy["promotion_guardrail_metric"]),
            "triggered_beats_frozen_profile_shape": bool(
                effective_refresh_policy["triggered_beats_frozen_profile_shape"]
            ),
            "triggered_beats_frozen_lock": bool(effective_refresh_policy["triggered_beats_frozen_lock"]),
            "unconditional_beats_frozen_profile_shape": bool(
                effective_refresh_policy["unconditional_beats_frozen_profile_shape"]
            ),
            "unconditional_beats_frozen_lock": bool(effective_refresh_policy["unconditional_beats_frozen_lock"]),
            "trigger_rate_in_band": bool(effective_refresh_policy.get("trigger_rate_in_band", False)),
            "triggered_profile_gain_fraction_vs_unconditional": float(
                effective_refresh_policy["triggered_profile_gain_fraction_vs_unconditional"]
            ),
            "triggered_lock_gain_fraction_vs_unconditional": float(
                effective_refresh_policy["triggered_lock_gain_fraction_vs_unconditional"]
            ),
            "retains_profile_gain_vs_unconditional": bool(
                effective_refresh_policy["retains_profile_gain_vs_unconditional"]
            ),
            "retains_lock_gain_vs_unconditional": bool(
                effective_refresh_policy["retains_lock_gain_vs_unconditional"]
            ),
            "calibration_trigger_rate": float(selected_refresh_threshold_row.get("trigger_rate", float("nan"))),
            "evaluation_trigger_rate": float(effective_refresh_policy["trigger_rate"]),
            "reason": str(effective_refresh_policy["reason"]),
            "trigger_reason_counts": {str(key): int(value) for key, value in trigger_reason_counts.items()},
            "selection_artifact": str(day_ahead_refresh["selected_replay_cache_artifact"]),
            "benchmark_replay_cache_status": str(day_ahead_refresh["benchmark_replay_cache_status"]),
            "selected_replay_cache_status": str(day_ahead_refresh["selected_replay_cache_status"]),
            "selection_context": day_ahead_refresh["selection"],
            "exact_control_recommended_policy": (
                str(refresh_policy["recommended_policy"]) if refresh_policy is not None else ""
            ),
            "rolling_benchmark": (
                {
                    "recommended_policy": str(rolling_refresh_policy["recommended_policy"]),
                    "trigger_mode": str(rolling_refresh_thresholds.get("trigger_mode", "any")),
                    "threshold_source": "rolling_benchmark",
                    "trigger_rate": float(rolling_refresh_policy["trigger_rate"]),
                    "reason": str(rolling_refresh_policy["reason"]),
                    "profile_gain_fraction_vs_unconditional": float(
                        rolling_refresh_policy["triggered_profile_gain_fraction_vs_unconditional"]
                    ),
                    "lock_gain_fraction_vs_unconditional": float(
                        rolling_refresh_policy["triggered_lock_gain_fraction_vs_unconditional"]
                    ),
                }
                if rolling_refresh_policy is not None
                else {}
            ),
        }
    else:
        policy["day_ahead_refresh"] = {"enabled": False}
    artifact_write_started_at_utc, artifact_write_started_perf = _start_runtime_step()
    (run_dir / "control_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    origin_catalog.to_csv(run_dir / "control_origin_catalog.csv", index=False)
    scope_summary.to_csv(run_dir / "control_scope_summary.csv", index=False, float_format="%.6f")
    summary.to_csv(run_dir / "control_backtest_summary.csv", index=False, float_format="%.6f")
    by_cycle.to_csv(run_dir / "control_backtest_by_cycle.csv", index=False, float_format="%.6f")
    minute_timeline.to_csv(run_dir / "control_minute_timeline.csv", index=False, float_format="%.6f")
    interval_timeline.to_csv(run_dir / "control_interval_timeline.csv", index=False, float_format="%.6f")
    if not calibration_summary.empty:
        calibration_summary.to_csv(
            run_dir / "control_backtest_calibration_summary.csv",
            index=False,
            float_format="%.6f",
        )
    if not calibration_by_cycle.empty:
        calibration_by_cycle.to_csv(
            run_dir / "control_backtest_calibration_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
    if not effective_refresh_threshold_grid.empty:
        effective_refresh_threshold_grid.to_csv(
            run_dir / "day_ahead_refresh_threshold_grid.csv",
            index=False,
            float_format="%.6f",
        )
    if not effective_calibration_refresh_summary.empty:
        effective_calibration_refresh_summary.to_csv(
            run_dir / "day_ahead_refresh_calibration_summary.csv",
            index=False,
            float_format="%.6f",
        )
        effective_calibration_refresh_by_cycle.to_csv(
            run_dir / "day_ahead_refresh_calibration_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
        effective_calibration_refresh_decisions.to_csv(
            run_dir / "day_ahead_refresh_calibration_decisions.csv",
            index=False,
            float_format="%.6f",
        )
    if not effective_refresh_summary.empty:
        effective_refresh_summary.to_csv(
            run_dir / "day_ahead_refresh_summary.csv",
            index=False,
            float_format="%.6f",
        )
        effective_refresh_by_cycle.to_csv(
            run_dir / "day_ahead_refresh_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
        effective_refresh_decisions.to_csv(
            run_dir / "day_ahead_refresh_decisions.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_origin_catalog.empty:
        rolling_origin_catalog.to_csv(run_dir / "rolling_control_origin_catalog.csv", index=False)
    if not rolling_calibration_summary.empty:
        rolling_calibration_summary.to_csv(
            run_dir / "rolling_control_backtest_calibration_summary.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_calibration_by_cycle.empty:
        rolling_calibration_by_cycle.to_csv(
            run_dir / "rolling_control_backtest_calibration_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_summary.empty:
        rolling_summary.to_csv(
            run_dir / "rolling_control_backtest_summary.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_by_cycle.empty:
        rolling_by_cycle.to_csv(
            run_dir / "rolling_control_backtest_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_scope_summary.empty:
        rolling_scope_summary.to_csv(
            run_dir / "rolling_control_scope_summary.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_layer_inference.empty:
        rolling_layer_inference.to_csv(
            run_dir / "rolling_control_layer_inference.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_refresh_threshold_grid.empty:
        rolling_refresh_threshold_grid.to_csv(
            run_dir / "rolling_day_ahead_refresh_threshold_grid.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_calibration_refresh_summary.empty:
        rolling_calibration_refresh_summary.to_csv(
            run_dir / "rolling_day_ahead_refresh_calibration_summary.csv",
            index=False,
            float_format="%.6f",
        )
        rolling_calibration_refresh_by_cycle.to_csv(
            run_dir / "rolling_day_ahead_refresh_calibration_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
        rolling_calibration_refresh_decisions.to_csv(
            run_dir / "rolling_day_ahead_refresh_calibration_decisions.csv",
            index=False,
            float_format="%.6f",
        )
    if not rolling_refresh_summary.empty:
        rolling_refresh_summary.to_csv(
            run_dir / "rolling_day_ahead_refresh_summary.csv",
            index=False,
            float_format="%.6f",
        )
        rolling_refresh_by_cycle.to_csv(
            run_dir / "rolling_day_ahead_refresh_by_cycle.csv",
            index=False,
            float_format="%.6f",
        )
        rolling_refresh_decisions.to_csv(
            run_dir / "rolling_day_ahead_refresh_decisions.csv",
            index=False,
            float_format="%.6f",
        )
    current_evidence_index = _build_current_evidence_index(
        summary=summary,
        policy=policy,
        rolling_scope_summary=rolling_scope_summary,
        rolling_layer_inference=rolling_layer_inference,
        refresh_summary=effective_refresh_summary,
        rolling_refresh_summary=rolling_refresh_summary,
    )
    current_evidence_index.to_csv(
        run_dir / "current_evidence_index.csv",
        index=False,
        float_format="%.6f",
    )
    _write_current_evidence_index_md(current_evidence_index, run_dir / "current_evidence_index.md")
    config_hash = stable_config_hash(
        {
            "forecast_control": MULTIRES_FORECAST_CONTROL,
            "load_type": DATASET["load_type"],
            "calibration_cycle_origins": [timestamp.isoformat() for timestamp in calibration_cycle_origins],
            "evaluation_cycle_origins": [timestamp.isoformat() for timestamp in evaluation_cycle_origins],
            "rolling_calibration_cycle_origins": [
                timestamp.isoformat() for timestamp in rolling_calibration_cycle_origins
            ],
            "rolling_evaluation_cycle_origins": [
                timestamp.isoformat() for timestamp in rolling_evaluation_cycle_origins
            ],
            "day_ahead_candidate": day_ahead["candidate_label"],
            "hourly_candidate": hourly["candidate_label"],
            "phase_candidate": phase["candidate_label"],
            "nowcast_anchor": nowcast_anchor["candidate_label"],
            "day_ahead_refresh_candidate": (
                day_ahead_refresh["candidate_label"] if day_ahead_refresh is not None else ""
            ),
            "day_ahead_refresh_thresholds": effective_refresh_thresholds,
        }
    )
    optimizer_uncertainty_inputs = [frame for frame in (calibration_interval_timeline, rolling_calibration_interval_timeline) if not frame.empty]
    optimizer_uncertainty_source = (
        pd.concat(optimizer_uncertainty_inputs, ignore_index=True)
        if optimizer_uncertainty_inputs
        else pd.DataFrame()
    )
    optimizer_delivery_uncertainty_calibration = _build_optimizer_delivery_uncertainty_calibration(
        calibration_interval_timeline=optimizer_uncertainty_source,
        lock_interval_minutes=lock_interval,
    )
    optimizer_delivery_preview = _build_optimizer_delivery_preview(
        interval_timeline=interval_timeline,
        policy=policy,
        uncertainty_calibration=optimizer_delivery_uncertainty_calibration,
        lock_interval_minutes=lock_interval,
        run_id=run_dir.name,
        config_hash=config_hash,
    )
    optimizer_delivery_uncertainty_summary = _summarize_optimizer_delivery_uncertainty(
        optimizer_delivery_preview
    )
    optimizer_delivery_serving_preview = _build_optimizer_delivery_serving_preview(
        optimizer_delivery_preview
    )
    optimizer_dynamic_overlay_shadow_summary = _build_optimizer_dynamic_overlay_shadow_summary(
        optimizer_delivery_preview
    )
    optimizer_dynamic_overlay_soft_candidates = _evaluate_optimizer_dynamic_soft_overlay_candidates(
        optimizer_delivery_preview
    )
    optimizer_dynamic_overlay_soft_summary = _build_optimizer_dynamic_overlay_soft_summary(
        optimizer_delivery_preview,
        optimizer_dynamic_overlay_soft_candidates,
    )
    optimizer_delivery_contract = _build_optimizer_delivery_contract(
        run_id=run_dir.name,
        config_hash=config_hash,
        policy=policy,
        lock_interval_minutes=lock_interval,
        uncertainty_summary=optimizer_delivery_uncertainty_summary,
    )
    optimizer_operational_policy = _build_optimizer_operational_policy(
        run_id=run_dir.name,
        config_hash=config_hash,
        policy=policy,
        lock_interval_minutes=lock_interval,
        uncertainty_summary=optimizer_delivery_uncertainty_summary,
    )
    (run_dir / "optimizer_delivery_contract.json").write_text(
        json.dumps(optimizer_delivery_contract, indent=2),
        encoding="utf-8",
    )
    (run_dir / "optimizer_operational_policy.json").write_text(
        json.dumps(optimizer_operational_policy, indent=2),
        encoding="utf-8",
    )
    if not optimizer_delivery_preview.empty:
        optimizer_delivery_preview.to_csv(
            run_dir / "optimizer_delivery_preview.csv",
            index=False,
            float_format="%.6f",
        )
    if not optimizer_delivery_serving_preview.empty:
        optimizer_delivery_serving_preview.to_csv(
            run_dir / "optimizer_delivery_serving_preview.csv",
            index=False,
            float_format="%.6f",
        )
    (run_dir / "optimizer_dynamic_overlay_shadow_summary.json").write_text(
        json.dumps(optimizer_dynamic_overlay_shadow_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "optimizer_dynamic_overlay_soft_summary.json").write_text(
        json.dumps(optimizer_dynamic_overlay_soft_summary, indent=2),
        encoding="utf-8",
    )
    if not optimizer_dynamic_overlay_soft_candidates.empty:
        optimizer_dynamic_overlay_soft_candidates.to_csv(
            run_dir / "optimizer_dynamic_overlay_soft_candidates.csv",
            index=False,
            float_format="%.6f",
        )
    if not optimizer_delivery_uncertainty_calibration.empty:
        optimizer_delivery_uncertainty_calibration.to_csv(
            run_dir / "optimizer_delivery_uncertainty_calibration.csv",
            index=False,
            float_format="%.6f",
        )
    if not optimizer_delivery_uncertainty_summary.empty:
        optimizer_delivery_uncertainty_summary.to_csv(
            run_dir / "optimizer_delivery_uncertainty_summary.csv",
            index=False,
            float_format="%.6f",
        )
    _write_summary_md(
        output_dir=run_dir,
        summary=summary,
        policy=policy,
        refresh_summary=effective_refresh_summary,
        rolling_scope_summary=rolling_scope_summary,
        rolling_layer_inference=rolling_layer_inference,
        rolling_refresh_summary=rolling_refresh_summary,
    )
    artifacts = {
        "control_policy": "control_policy.json",
        "control_origin_catalog": "control_origin_catalog.csv",
        "control_scope_summary": "control_scope_summary.csv",
        "control_backtest_summary_csv": "control_backtest_summary.csv",
        "control_backtest_summary_md": "control_backtest_summary.md",
        "control_backtest_by_cycle": "control_backtest_by_cycle.csv",
        "control_layer_candidate_benchmarks": "control_layer_candidate_benchmarks.csv",
        "control_minute_timeline": "control_minute_timeline.csv",
        "control_interval_timeline": "control_interval_timeline.csv",
        "current_evidence_index_csv": "current_evidence_index.csv",
        "current_evidence_index_md": "current_evidence_index.md",
        "optimizer_delivery_contract": "optimizer_delivery_contract.json",
        "optimizer_operational_policy": "optimizer_operational_policy.json",
        "optimizer_delivery_preview": "optimizer_delivery_preview.csv",
        "optimizer_delivery_serving_preview": "optimizer_delivery_serving_preview.csv",
        "optimizer_dynamic_overlay_shadow_summary": "optimizer_dynamic_overlay_shadow_summary.json",
        "optimizer_dynamic_overlay_soft_summary": "optimizer_dynamic_overlay_soft_summary.json",
        "optimizer_delivery_uncertainty_calibration": "optimizer_delivery_uncertainty_calibration.csv",
        "optimizer_delivery_uncertainty_summary": "optimizer_delivery_uncertainty_summary.csv",
        "figure_guide_md": "figure_guide.md",
        "fig_control_lock_mae": "fig_control_lock_mae.png",
        "fig_control_example_cycle": "fig_control_example_cycle.png",
    }
    if not calibration_summary.empty:
        artifacts["control_backtest_calibration_summary"] = "control_backtest_calibration_summary.csv"
        artifacts["control_backtest_calibration_by_cycle"] = "control_backtest_calibration_by_cycle.csv"
    if not effective_refresh_threshold_grid.empty:
        artifacts["day_ahead_refresh_threshold_grid"] = "day_ahead_refresh_threshold_grid.csv"
    if not effective_calibration_refresh_summary.empty:
        artifacts["day_ahead_refresh_calibration_summary"] = "day_ahead_refresh_calibration_summary.csv"
        artifacts["day_ahead_refresh_calibration_by_cycle"] = "day_ahead_refresh_calibration_by_cycle.csv"
        artifacts["day_ahead_refresh_calibration_decisions"] = "day_ahead_refresh_calibration_decisions.csv"
    if not effective_refresh_summary.empty:
        artifacts.update(
            {
                "day_ahead_refresh_summary": "day_ahead_refresh_summary.csv",
                "day_ahead_refresh_by_cycle": "day_ahead_refresh_by_cycle.csv",
                "day_ahead_refresh_decisions": "day_ahead_refresh_decisions.csv",
                "fig_day_ahead_refresh_policy": "fig_day_ahead_refresh_policy.png",
            }
        )
    if not rolling_origin_catalog.empty:
        artifacts["rolling_control_origin_catalog"] = "rolling_control_origin_catalog.csv"
    if not rolling_calibration_summary.empty:
        artifacts["rolling_control_backtest_calibration_summary"] = "rolling_control_backtest_calibration_summary.csv"
    if not rolling_calibration_by_cycle.empty:
        artifacts["rolling_control_backtest_calibration_by_cycle"] = "rolling_control_backtest_calibration_by_cycle.csv"
    if not rolling_summary.empty:
        artifacts["rolling_control_backtest_summary"] = "rolling_control_backtest_summary.csv"
    if not rolling_by_cycle.empty:
        artifacts["rolling_control_backtest_by_cycle"] = "rolling_control_backtest_by_cycle.csv"
        artifacts["fig_control_lock_distribution"] = "fig_control_lock_distribution.png"
    if not rolling_scope_summary.empty:
        artifacts["rolling_control_scope_summary"] = "rolling_control_scope_summary.csv"
    if not rolling_layer_inference.empty:
        artifacts["rolling_control_layer_inference"] = "rolling_control_layer_inference.csv"
        artifacts["fig_control_layer_gain_ci"] = "fig_control_layer_gain_ci.png"
    if not rolling_refresh_threshold_grid.empty:
        artifacts["rolling_day_ahead_refresh_threshold_grid"] = "rolling_day_ahead_refresh_threshold_grid.csv"
    if not rolling_calibration_refresh_summary.empty:
        artifacts["rolling_day_ahead_refresh_calibration_summary"] = "rolling_day_ahead_refresh_calibration_summary.csv"
        artifacts["rolling_day_ahead_refresh_calibration_by_cycle"] = "rolling_day_ahead_refresh_calibration_by_cycle.csv"
        artifacts["rolling_day_ahead_refresh_calibration_decisions"] = "rolling_day_ahead_refresh_calibration_decisions.csv"
    if not rolling_refresh_summary.empty:
        artifacts["rolling_day_ahead_refresh_summary"] = "rolling_day_ahead_refresh_summary.csv"
        artifacts["rolling_day_ahead_refresh_by_cycle"] = "rolling_day_ahead_refresh_by_cycle.csv"
        artifacts["rolling_day_ahead_refresh_decisions"] = "rolling_day_ahead_refresh_decisions.csv"
    if not optimizer_dynamic_overlay_soft_candidates.empty:
        artifacts["optimizer_dynamic_overlay_soft_candidates"] = "optimizer_dynamic_overlay_soft_candidates.csv"
    if not phase_stack_guard_summary.empty:
        artifacts["phase_stack_guard_summary"] = "phase_stack_guard_summary.csv"
    if not phase_stack_candidate_benchmarks.empty:
        artifacts["phase_stack_candidate_benchmarks"] = "phase_stack_candidate_benchmarks.csv"
        artifacts["fig_phase_stack_candidates"] = "fig_phase_stack_candidates.png"
    _append_runtime_step(
        runtime_profile_records,
        step="write_stage_outputs",
        category="artifacts",
        started_at_utc=artifact_write_started_at_utc,
        started_perf=artifact_write_started_perf,
        artifact_count_planned=int(len(artifacts) + 2),
    )
    runtime_profile = pd.DataFrame(runtime_profile_records)
    runtime_summary = _build_runtime_profile_summary(
        runtime_profile,
        wall_clock_seconds=perf_counter() - run_started_perf,
    )
    if not runtime_profile.empty:
        runtime_profile.to_csv(
            run_dir / "runtime_profile.csv",
            index=False,
            float_format="%.6f",
        )
    (run_dir / "runtime_summary.json").write_text(
        json.dumps(runtime_summary, indent=2),
        encoding="utf-8",
    )
    artifacts["runtime_profile"] = "runtime_profile.csv"
    artifacts["runtime_summary"] = "runtime_summary.json"
    manifest = {
        "run_id": run_dir.name,
        "stage": "010_forecast_control",
        "config_hash": config_hash,
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "actual_resolution": actual_resolution,
        "day_ahead_horizon_minutes": day_ahead_horizon,
        "hourly_horizon_minutes": hourly_horizon,
        "phase_horizon_minutes": phase_horizon,
        "lock_interval_minutes": lock_interval,
        "calibration_cycle_count": len(calibration_cycle_origins),
        "evaluation_cycle_count": len(evaluation_cycle_origins),
        "warnings": warnings,
        "artifacts": artifacts,
        "replay_cache_enabled": bool(effective_replay_cache_enabled),
        "replay_cache_registry": (
            _relative_artifact_path(_replay_cache_root(output_root) / "replay_cache_registry.csv")
            if bool(effective_replay_cache_enabled)
            else ""
        ),
        "runtime_summary": runtime_summary,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "success",
    }
    if bool(effective_replay_cache_enabled):
        manifest["artifacts"]["replay_cache_registry"] = _relative_artifact_path(
            _replay_cache_root(output_root) / "replay_cache_registry.csv"
        )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if bool(MULTIRES_FORECAST_CONTROL["write_latest"]):
        update_latest_alias(run_dir, output_root / "latest", enabled=True)
    emit_quality_gate(
        "FORECAST CONTROL",
        True,
        details={
            "evaluation_cycles": len(evaluation_cycle_origins),
            "best_lock_mae": round(float(summary["lock_mae"].min()), 3),
        },
        logger_instance=logger,
    )
    logger.info("Forecast-control artifacts written to %s", run_dir)
    return {"run_dir": run_dir, "summary": summary, "by_cycle": by_cycle, "policy": policy, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Stage-10 forecast-control backtest."""
    parser = argparse.ArgumentParser(description="Run the Stage-10 forecast-control backtest.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Optional alternate output root for isolated or cold validation runs.",
    )
    parser.add_argument(
        "--disable-replay-cache",
        action="store_true",
        help="Force Stage-10 to recompute exact-origin replays instead of reusing cached snapshots.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the Stage-10 backtest and write one forecast-control run directory."""
    _configure_logging()
    args = parse_args()
    run_forecast_control_backtest(
        output_root=Path(args.output_root),
        replay_cache_enabled=not bool(args.disable_replay_cache),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
