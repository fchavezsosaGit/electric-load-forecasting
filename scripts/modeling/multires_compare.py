"""Stage-6 multiresolution comparison with leakage-safe matched-horizon evaluation.

This stage compares candidate resolutions on shared real-time horizons so the repo
can answer not just "which model is lowest error?" but also "which cadence is
worth running once runtime, coverage, and stability are considered together?"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from functools import partial
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

import sys

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    DATASET,
    FEATURE_SETS,
    MULTIRES_BASELINES,
    MULTIRES_CONFIG,
    MULTIRES_HYBRID,
    MULTIRES_PROFILES,
    MULTIRES_ROLLOUT,
    MULTIRES_RUNTIME,
    MULTIRES_SELECTION,
    MODELING_PARALLEL,
    PATHS,
    RESOLUTION_TO_SUFFIX,
    SPLIT_DAY_RANGES,
    resolve_horizon_policy,
    scoped_output_path,
    validate_config,
)
from utils import emit_quality_gate

from modeling.common import (
    FigureGuideEntry,
    build_model_catalog,
    canonical_resolution,
    lead_steps_for_horizon,
    predict_model,
    resolution_total_minutes,
    stable_config_hash,
    train_model,
    update_latest_alias,
    validate_png_artifact,
    write_figure_guide,
)
from modeling.metrics import aggregate_absolute_error_percentage
from modeling.multires import (
    FoldSpec,
    actual_path,
    anchored_workday_path,
    avg_workday_path,
    blend_candidate_paths,
    build_causal_feature_frame,
    build_walkforward_folds,
    build_workday_profile,
    compare_recursive_paths,
    evaluate_predictions,
    filter_day_range,
    lead_target_series,
    load_base_gold,
    mae_ratio,
    native_step_baselines,
    persistence_path,
    previous_day_path,
    recursive_predict_path,
    rmse_ratio,
    select_origin_positions,
)
from modeling.parallel import run_stage_jobs
from modeling.runtime import runtime_summary

logger = logging.getLogger(__name__)
PROJECT_ROOT = SCRIPTS_DIR.parent


@dataclass(frozen=True)
class NativeMetricTask:
    """One Stage-6 native-step evaluation payload."""

    fold: FoldSpec
    feature_set: str
    model_label: str
    emit_baselines: bool


@dataclass(frozen=True)
class MatchedMetricTask:
    """One Stage-6 matched-horizon evaluation payload."""

    fold: FoldSpec
    feature_set: str
    model_label: str
    horizon_minutes: int
    emit_baselines: bool


def _configure_logging() -> None:
    """Initialize a simple logger for standalone Stage-6 execution."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _resolve_mode_profile(mode: str) -> dict[str, Any]:
    """Resolve the configured Stage-6 mode profile into concrete run settings."""
    mode_key = mode.lower().strip()
    if mode_key not in MULTIRES_PROFILES:
        raise ValueError(f"Unsupported mode: {mode}")
    profile = MULTIRES_PROFILES[mode_key]
    return {
        "n_folds": int(profile["n_folds"]),
        "val_window_days": int(profile["val_window_days"]),
        "origins_per_fold": int(profile["origins_per_fold"]),
        "resolutions": list(profile["resolutions"]),
        "horizons_minutes": list(profile["horizons_minutes"]),
        "feature_sets": list(profile["feature_sets"]),
        "model_labels": list(profile["model_labels"]),
    }


def _resolve_horizon_scope(
    *,
    horizon_minutes: int,
    feature_sets: list[str],
    model_labels: list[str],
) -> tuple[list[str], list[str]]:
    """Filter the selected feature/model scope through the centralized horizon policy."""
    policy = resolve_horizon_policy(int(horizon_minutes))
    resolved_feature_sets = [name for name in feature_sets if name in set(policy["feature_sets"])]
    resolved_model_labels = [name for name in model_labels if name in set(policy["model_labels"])]
    return (
        resolved_feature_sets or list(feature_sets),
        resolved_model_labels or list(model_labels),
    )


def _partition_representable_horizons(
    resolution: str, horizons_minutes: list[int]
) -> tuple[list[int], list[int]]:
    """Split requested horizons into representable and skipped values for one resolution."""
    valid: list[int] = []
    skipped: list[int] = []
    for horizon_minutes in horizons_minutes:
        try:
            lead_steps_for_horizon(resolution, horizon_minutes)
        except ValueError:
            skipped.append(int(horizon_minutes))
        else:
            valid.append(int(horizon_minutes))
    return valid, skipped


def _gold_input_path(resolution: str) -> Path:
    """Return the expected gold parquet path for a resolution."""
    suffix = RESOLUTION_TO_SUFFIX[canonical_resolution(resolution)]
    return PATHS["gold_dir"] / f"power_load_{suffix}_all_features.parquet"


def _partition_available_resolutions(resolutions: list[str]) -> tuple[list[str], list[str]]:
    """Split requested resolutions into available and missing gold inputs."""
    available: list[str] = []
    missing: list[str] = []
    for resolution in resolutions:
        input_path = _gold_input_path(resolution)
        if input_path.exists():
            available.append(resolution)
        else:
            missing.append(f"{resolution}:{input_path}")
    return available, missing


def _deduplicate_metric_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate metric rows while preserving stable order."""
    if frame.empty:
        return frame, 0
    deduped = frame.drop_duplicates(keep="first").reset_index(drop=True)
    removed = int(len(frame) - len(deduped))
    return deduped, removed


def _day_class_lookup(base: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """Build a normalized-date lookup used by recursive baseline paths."""
    lookup = (
        base.loc[:, ["timestamp", "day_class"]]
        .assign(date=lambda df: df["timestamp"].dt.normalize())
        .drop_duplicates(subset=["date"])
        .set_index("date")["day_class"]
    )
    return {pd.Timestamp(index): str(value) for index, value in lookup.items()}


def _enabled_native_baseline_labels() -> list[str]:
    """Return the one-step baselines enabled for native-step comparisons."""
    labels = ["persistence"]
    if MULTIRES_BASELINES["include_previous_day"]:
        labels.append("previous_day")
    if MULTIRES_BASELINES["include_avg_workday"]:
        labels.append("avg_workday")
    return labels


def _enabled_path_baseline_labels() -> list[str]:
    """Return the path baselines enabled for matched-horizon comparisons."""
    labels = _enabled_native_baseline_labels()
    if MULTIRES_BASELINES["include_anchored_workday"]:
        labels.append("anchored_workday")
    if MULTIRES_BASELINES["include_hybrid_workday"]:
        labels.append("hybrid_workday")
    return labels


def _aggregate_error_rows(errors: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate endpoint/path errors while preserving NaN-only strategies."""
    if not errors:
        return {
            "endpoint_mae": float("nan"),
            "endpoint_rmse": float("nan"),
            "endpoint_mae_pct": float("nan"),
            "path_mae": float("nan"),
            "path_mae_pct": float("nan"),
            "coverage": 0.0,
            "n_eval": 0,
        }
    error_df = pd.DataFrame(errors)
    endpoint_abs = pd.to_numeric(error_df["endpoint_abs_error"], errors="coerce")
    endpoint_sq = pd.to_numeric(error_df["endpoint_sq_error"], errors="coerce")
    endpoint_actual_abs = pd.to_numeric(error_df.get("endpoint_actual_abs"), errors="coerce")
    path_mae = pd.to_numeric(error_df["path_mae"], errors="coerce")
    path_abs_error_sum = pd.to_numeric(error_df.get("path_abs_error_sum"), errors="coerce")
    path_actual_abs_sum = pd.to_numeric(error_df.get("path_actual_abs_sum"), errors="coerce")
    coverage = pd.to_numeric(error_df["coverage"], errors="coerce")
    n_eval = pd.to_numeric(error_df["n_eval"], errors="coerce")
    endpoint_mae = float(endpoint_abs.mean(skipna=True)) if endpoint_abs.notna().any() else float("nan")
    endpoint_rmse = (
        float(np.sqrt(endpoint_sq.mean(skipna=True))) if endpoint_sq.notna().any() else float("nan")
    )
    mean_path_mae = float(path_mae.mean(skipna=True)) if path_mae.notna().any() else float("nan")
    endpoint_mae_pct = (
        aggregate_absolute_error_percentage(
            error_sum=float(endpoint_abs.sum(skipna=True)),
            actual_abs_sum=float(endpoint_actual_abs.sum(skipna=True)),
        )
        if endpoint_abs.notna().any() and endpoint_actual_abs.notna().any()
        else float("nan")
    )
    path_mae_pct = (
        aggregate_absolute_error_percentage(
            error_sum=float(path_abs_error_sum.sum(skipna=True)),
            actual_abs_sum=float(path_actual_abs_sum.sum(skipna=True)),
        )
        if path_abs_error_sum.notna().any() and path_actual_abs_sum.notna().any()
        else float("nan")
    )
    mean_coverage = float(coverage.mean(skipna=True)) if coverage.notna().any() else 0.0
    total_eval = int(n_eval.fillna(0.0).sum())
    return {
        "endpoint_mae": endpoint_mae,
        "endpoint_rmse": endpoint_rmse,
        "endpoint_mae_pct": endpoint_mae_pct,
        "path_mae": mean_path_mae,
        "path_mae_pct": path_mae_pct,
        "coverage": mean_coverage,
        "n_eval": total_eval,
    }


def _native_step_metrics_for_fold(
    *,
    frame: pd.DataFrame,
    fold: FoldSpec,
    feature_set: str,
    model_label: str,
    emit_baselines: bool,
    resolution: str,
) -> list[dict[str, Any]]:
    """Evaluate one native-step fold for the selected learned model and baselines."""
    catalog = build_model_catalog()
    feature_columns = FEATURE_SETS[feature_set]
    train_df = filter_day_range(frame, fold.train_start_day, fold.train_end_day)
    val_df = filter_day_range(frame, fold.val_start_day, fold.val_end_day)
    baselines = native_step_baselines(train_df, val_df)
    n_total = int(len(val_df))
    persistence_metrics = evaluate_predictions(val_df["avg_load"], baselines["persistence"], n_total=n_total)
    rows: list[dict[str, Any]] = []
    horizon_minutes = resolution_total_minutes(resolution)
    if emit_baselines:
        for baseline_label in _enabled_native_baseline_labels():
            preds = baselines[baseline_label]
            metrics = evaluate_predictions(val_df["avg_load"], preds, n_total=n_total)
            rows.append(
                {
                    "comparison_mode": "native_step",
                    "resolution": resolution,
                    "horizon_minutes": horizon_minutes,
                    "fold": fold.fold,
                    "feature_set": "baseline",
                    "model_label": baseline_label,
                    "baseline_label": "persistence",
                    "candidate_type": "baseline",
                    "forecast_strategy": "one_step",
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mae_pct": metrics["mae_pct"],
                    "rmse_pct": metrics["rmse_pct"],
                    "path_mae": metrics["mae"],
                    "path_mae_pct": metrics["mae_pct"],
                    "mae_ratio_to_persistence": 1.0
                    if baseline_label == "persistence"
                    else mae_ratio(float(metrics["mae"]), float(persistence_metrics["mae"])),
                    "rmse_ratio_to_persistence": 1.0
                    if baseline_label == "persistence"
                    else rmse_ratio(float(metrics["rmse"]), float(persistence_metrics["rmse"])),
                    "n_eval": metrics["n_eval"],
                    "eval_coverage": metrics["coverage"],
                    "runtime_seconds": 0.0,
                    "train_start_day": fold.train_start_day,
                    "train_end_day": fold.train_end_day,
                    "eval_start_day": fold.val_start_day,
                    "eval_end_day": fold.val_end_day,
                    "source_mode": "bronze_direct",
                }
            )

    model_spec = catalog[model_label]
    start = time.perf_counter()
    trained = train_model(train_df, feature_columns, model_spec)
    preds = predict_model(trained, val_df)
    runtime_seconds = time.perf_counter() - start
    metrics = evaluate_predictions(val_df["avg_load"], preds, n_total=n_total)
    rows.append(
        {
            "comparison_mode": "native_step",
            "resolution": resolution,
            "horizon_minutes": horizon_minutes,
            "fold": fold.fold,
            "feature_set": feature_set,
            "model_label": model_label,
            "baseline_label": "persistence",
            "candidate_type": "learned",
            "forecast_strategy": "one_step",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mae_pct": metrics["mae_pct"],
            "rmse_pct": metrics["rmse_pct"],
            "path_mae": metrics["mae"],
            "path_mae_pct": metrics["mae_pct"],
            "mae_ratio_to_persistence": mae_ratio(float(metrics["mae"]), float(persistence_metrics["mae"])),
            "rmse_ratio_to_persistence": rmse_ratio(float(metrics["rmse"]), float(persistence_metrics["rmse"])),
            "n_eval": metrics["n_eval"],
            "eval_coverage": metrics["coverage"],
            "runtime_seconds": runtime_seconds,
            "train_start_day": fold.train_start_day,
            "train_end_day": fold.train_end_day,
            "eval_start_day": fold.val_start_day,
            "eval_end_day": fold.val_end_day,
            "source_mode": "bronze_direct",
        }
    )
    return rows


def _recursive_metrics_for_fold(
    *,
    base: pd.DataFrame,
    frame: pd.DataFrame,
    day_class_lookup: dict[pd.Timestamp, str],
    fold: FoldSpec,
    feature_set: str,
    model_label: str,
    emit_baselines: bool,
    resolution: str,
    horizon_minutes: int,
    origins_per_fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one matched horizon fold across baseline and learned recursive paths."""
    catalog = build_model_catalog()
    feature_columns = FEATURE_SETS[feature_set]
    train_df = filter_day_range(frame, fold.train_start_day, fold.train_end_day)
    horizon_steps = lead_steps_for_horizon(resolution, horizon_minutes)
    origins = select_origin_positions(
        base,
        start_day=fold.val_start_day,
        end_day=fold.val_end_day,
        lead_steps=horizon_steps,
        max_origins=origins_per_fold,
    )
    profile = build_workday_profile(train_df)
    metric_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    matched_strategies = tuple(MULTIRES_CONFIG["matched_strategies"])
    trained_recursive = (
        train_model(train_df, feature_columns, catalog[model_label])
        if "recursive" in matched_strategies
        else None
    )
    trained_direct = None
    if "direct_endpoint" in matched_strategies:
        direct_frame = frame.copy()
        direct_frame["lead_target"] = lead_target_series(
            direct_frame,
            resolution=resolution,
            horizon_minutes=horizon_minutes,
        )
        direct_train = filter_day_range(direct_frame, fold.train_start_day, fold.train_end_day).copy()
        direct_train["avg_load"] = direct_train["lead_target"]
        direct_train = direct_train.drop(columns=["lead_target"])
        trained_direct = train_model(direct_train, feature_columns, catalog[model_label])
    learned_runtime: dict[str, float] = {
        "recursive": 0.0,
        "direct_endpoint": 0.0,
    }
    enabled_baselines = ["persistence"]
    if emit_baselines:
        enabled_baselines = _enabled_path_baseline_labels()
    baseline_errors: dict[str, list[dict[str, float]]] = {label: [] for label in enabled_baselines}
    learned_errors: dict[str, list[dict[str, float]]] = {strategy: [] for strategy in matched_strategies}

    for origin_position in origins:
        origin_timestamp = pd.Timestamp(base.iloc[origin_position]["timestamp"])
        history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
        actual = actual_path(base, origin_position=origin_position, horizon_steps=horizon_steps)
        candidate_paths: dict[str, pd.DataFrame] = {
            "persistence": persistence_path(
                history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
            )
        }
        if emit_baselines and MULTIRES_BASELINES["include_previous_day"]:
            candidate_paths["previous_day"] = previous_day_path(
                history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
            )
        if emit_baselines and MULTIRES_BASELINES["include_avg_workday"]:
            candidate_paths["avg_workday"] = avg_workday_path(
                profile,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_class_lookup,
            )
        if emit_baselines and (
            MULTIRES_BASELINES["include_anchored_workday"]
            or MULTIRES_BASELINES["include_hybrid_workday"]
        ):
            anchored_path = anchored_workday_path(
                profile,
                history=history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_class_lookup,
            )
            if MULTIRES_BASELINES["include_anchored_workday"]:
                candidate_paths["anchored_workday"] = anchored_path
            if MULTIRES_BASELINES["include_hybrid_workday"]:
                candidate_paths["hybrid_workday"] = blend_candidate_paths(
                    candidate_paths["persistence"],
                    anchored_path,
                    primary_weight_start=MULTIRES_HYBRID["persistence_weight_start"],
                    primary_weight_end=MULTIRES_HYBRID["persistence_weight_end"],
                    curve=MULTIRES_HYBRID["curve"],
                )
        baseline_compared = compare_recursive_paths(
            actual,
            {label: path for label, path in candidate_paths.items() if label in enabled_baselines},
        )
        for _, row in baseline_compared.iterrows():
            candidate_label = str(row["candidate_label"])
            baseline_errors[candidate_label].append(
                {
                    "endpoint_abs_error": float(row["endpoint_abs_error"]),
                    "endpoint_sq_error": float(row["endpoint_sq_error"]),
                    "endpoint_actual_abs": float(row["endpoint_actual_abs"]),
                    "path_mae": float(row["path_mae"]),
                    "path_abs_error_sum": float(row["path_abs_error_sum"]),
                    "path_actual_abs_sum": float(row["path_actual_abs_sum"]),
                    "coverage": float(row["coverage"]),
                    "n_eval": float(row["n_eval"]),
                }
            )
            if emit_baselines:
                origin_rows.append(
                    {
                        "comparison_mode": "matched_horizon",
                        "resolution": resolution,
                        "horizon_minutes": horizon_minutes,
                        "fold": fold.fold,
                        "origin_timestamp": origin_timestamp.isoformat(),
                        "candidate_label": candidate_label,
                        "feature_set": "baseline",
                        "forecast_strategy": "path_baseline",
                        "endpoint_abs_error": float(row["endpoint_abs_error"]),
                        "endpoint_actual_abs": float(row["endpoint_actual_abs"]),
                        "path_mae": float(row["path_mae"]),
                        "path_abs_error_sum": float(row["path_abs_error_sum"]),
                        "path_actual_abs_sum": float(row["path_actual_abs_sum"]),
                        "coverage": float(row["coverage"]),
                    }
                )
        if trained_recursive is not None:
            start = time.perf_counter()
            learned_path = recursive_predict_path(
                trained=trained_recursive,
                history=history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_class_lookup,
                profile=profile,
            )
            learned_runtime["recursive"] += time.perf_counter() - start
            learned_row = compare_recursive_paths(actual, {model_label: learned_path}).iloc[0]
            learned_errors["recursive"].append(
                {
                    "endpoint_abs_error": float(learned_row["endpoint_abs_error"]),
                    "endpoint_sq_error": float(learned_row["endpoint_sq_error"]),
                    "endpoint_actual_abs": float(learned_row["endpoint_actual_abs"]),
                    "path_mae": float(learned_row["path_mae"]),
                    "path_abs_error_sum": float(learned_row["path_abs_error_sum"]),
                    "path_actual_abs_sum": float(learned_row["path_actual_abs_sum"]),
                    "coverage": float(learned_row["coverage"]),
                    "n_eval": float(learned_row["n_eval"]),
                }
            )
            origin_rows.append(
                {
                    "comparison_mode": "matched_horizon",
                    "resolution": resolution,
                    "horizon_minutes": horizon_minutes,
                    "fold": fold.fold,
                    "origin_timestamp": origin_timestamp.isoformat(),
                    "candidate_label": model_label,
                    "feature_set": feature_set,
                    "forecast_strategy": "recursive",
                    "endpoint_abs_error": float(learned_row["endpoint_abs_error"]),
                    "endpoint_actual_abs": float(learned_row["endpoint_actual_abs"]),
                    "path_mae": float(learned_row["path_mae"]),
                    "path_abs_error_sum": float(learned_row["path_abs_error_sum"]),
                    "path_actual_abs_sum": float(learned_row["path_actual_abs_sum"]),
                    "coverage": float(learned_row["coverage"]),
                }
            )
        if trained_direct is not None:
            actual_endpoint = float(actual.iloc[-1]["avg_load"]) if not actual.empty else float("nan")
            start = time.perf_counter()
            direct_prediction = float(predict_model(trained_direct, frame.iloc[[origin_position]]).iloc[0])
            learned_runtime["direct_endpoint"] += time.perf_counter() - start
            if np.isfinite(actual_endpoint) and np.isfinite(direct_prediction):
                endpoint_error = actual_endpoint - direct_prediction
                learned_errors["direct_endpoint"].append(
                    {
                        "endpoint_abs_error": float(abs(endpoint_error)),
                        "endpoint_sq_error": float(endpoint_error * endpoint_error),
                        "endpoint_actual_abs": float(abs(actual_endpoint)),
                        "path_mae": float("nan"),
                        "path_abs_error_sum": float("nan"),
                        "path_actual_abs_sum": float("nan"),
                        "coverage": 1.0,
                        "n_eval": 1.0,
                    }
                )
                origin_rows.append(
                    {
                        "comparison_mode": "matched_horizon",
                        "resolution": resolution,
                        "horizon_minutes": horizon_minutes,
                        "fold": fold.fold,
                        "origin_timestamp": origin_timestamp.isoformat(),
                        "candidate_label": model_label,
                        "feature_set": feature_set,
                        "forecast_strategy": "direct_endpoint",
                        "endpoint_abs_error": float(abs(endpoint_error)),
                        "endpoint_actual_abs": float(abs(actual_endpoint)),
                        "path_mae": float("nan"),
                        "path_abs_error_sum": float("nan"),
                        "path_actual_abs_sum": float("nan"),
                        "coverage": 1.0,
                    }
                )

    persistence_metrics = _aggregate_error_rows(baseline_errors["persistence"])
    persistence_mae = persistence_metrics["endpoint_mae"]
    persistence_rmse = persistence_metrics["endpoint_rmse"]

    if emit_baselines:
        for baseline_label, errors in baseline_errors.items():
            aggregated = _aggregate_error_rows(errors)
            metric_rows.append(
                {
                    "comparison_mode": "matched_horizon",
                    "resolution": resolution,
                    "horizon_minutes": horizon_minutes,
                    "fold": fold.fold,
                    "feature_set": "baseline",
                    "model_label": baseline_label,
                    "baseline_label": "persistence",
                    "candidate_type": "baseline",
                    "forecast_strategy": "path_baseline",
                    "mae": aggregated["endpoint_mae"],
                    "rmse": aggregated["endpoint_rmse"],
                    "mae_pct": aggregated["endpoint_mae_pct"],
                    "path_mae": aggregated["path_mae"],
                    "path_mae_pct": aggregated["path_mae_pct"],
                    "mae_ratio_to_persistence": 1.0
                    if baseline_label == "persistence"
                    else mae_ratio(aggregated["endpoint_mae"], persistence_mae),
                    "rmse_ratio_to_persistence": 1.0
                    if baseline_label == "persistence"
                    else rmse_ratio(aggregated["endpoint_rmse"], persistence_rmse),
                    "n_eval": aggregated["n_eval"],
                    "eval_coverage": aggregated["coverage"],
                    "runtime_seconds": 0.0,
                    "train_start_day": fold.train_start_day,
                    "train_end_day": fold.train_end_day,
                    "eval_start_day": fold.val_start_day,
                    "eval_end_day": fold.val_end_day,
                    "source_mode": "bronze_direct",
                }
            )

    for strategy, errors in learned_errors.items():
        aggregated = _aggregate_error_rows(errors)
        metric_rows.append(
            {
                "comparison_mode": "matched_horizon",
                "resolution": resolution,
                "horizon_minutes": horizon_minutes,
                "fold": fold.fold,
                "feature_set": feature_set,
                "model_label": model_label,
                "baseline_label": "persistence",
                "candidate_type": "learned",
                "forecast_strategy": strategy,
                "mae": aggregated["endpoint_mae"],
                "rmse": aggregated["endpoint_rmse"],
                "mae_pct": aggregated["endpoint_mae_pct"],
                "path_mae": aggregated["path_mae"],
                "path_mae_pct": aggregated["path_mae_pct"],
                "mae_ratio_to_persistence": mae_ratio(aggregated["endpoint_mae"], persistence_mae),
                "rmse_ratio_to_persistence": rmse_ratio(aggregated["endpoint_rmse"], persistence_rmse),
                "n_eval": aggregated["n_eval"],
                "eval_coverage": aggregated["coverage"],
                "runtime_seconds": learned_runtime.get(strategy, 0.0),
                "train_start_day": fold.train_start_day,
                "train_end_day": fold.train_end_day,
                "eval_start_day": fold.val_start_day,
                "eval_end_day": fold.val_end_day,
                "source_mode": "bronze_direct",
            }
        )
    return metric_rows, origin_rows


def _run_native_step_task(
    task: NativeMetricTask,
    *,
    frame: pd.DataFrame,
    resolution: str,
) -> list[dict[str, Any]]:
    """Execute one native-step fold/feature/model comparison unit."""
    return _native_step_metrics_for_fold(
        frame=frame,
        fold=task.fold,
        feature_set=task.feature_set,
        model_label=task.model_label,
        emit_baselines=task.emit_baselines,
        resolution=resolution,
    )


def _run_matched_horizon_task(
    task: MatchedMetricTask,
    *,
    base: pd.DataFrame,
    frame: pd.DataFrame,
    day_class_lookup: dict[pd.Timestamp, str],
    resolution: str,
    origins_per_fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one matched-horizon fold/feature/model/horizon comparison unit."""
    return _recursive_metrics_for_fold(
        base=base,
        frame=frame,
        day_class_lookup=day_class_lookup,
        fold=task.fold,
        feature_set=task.feature_set,
        model_label=task.model_label,
        emit_baselines=task.emit_baselines,
        resolution=resolution,
        horizon_minutes=task.horizon_minutes,
        origins_per_fold=origins_per_fold,
    )


def _aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics into the Stage-6 summary schema."""
    group_cols = [
        "comparison_mode",
        "resolution",
        "horizon_minutes",
        "feature_set",
        "model_label",
        "baseline_label",
        "candidate_type",
        "forecast_strategy",
    ]
    if fold_metrics.empty:
        return pd.DataFrame(columns=group_cols)
    work = fold_metrics.copy()
    for column in ("mae_pct", "path_mae_pct"):
        if column not in work.columns:
            work[column] = float("nan")
    aggregated = (
        work.groupby(group_cols, dropna=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mae_pct=("mae_pct", "mean"),
            path_mae=("path_mae", "mean"),
            path_mae_pct=("path_mae_pct", "mean"),
            mae_ratio_to_persistence=("mae_ratio_to_persistence", "mean"),
            rmse_ratio_to_persistence=("rmse_ratio_to_persistence", "mean"),
            fold_std_mae_ratio=("mae_ratio_to_persistence", "std"),
            n_eval=("n_eval", "sum"),
            eval_coverage=("eval_coverage", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            fold_n=("fold", "nunique"),
            source_mode=("source_mode", "first"),
        )
        .reset_index()
    )
    aggregated["fold_std_mae_ratio"] = aggregated["fold_std_mae_ratio"].fillna(0.0)
    return aggregated.sort_values(
        ["comparison_mode", "horizon_minutes", "mae_ratio_to_persistence", "runtime_seconds"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def _build_resolution_health(summary: pd.DataFrame) -> pd.DataFrame:
    """Project Stage-6 summary rows into an operator-facing health table."""
    if summary.empty:
        return pd.DataFrame()
    health = summary.copy()
    health["n_train"] = np.nan
    health["n_validate"] = np.nan
    health["n_test"] = np.nan
    health["warmup_loss_pct"] = 1.0 - health["eval_coverage"]
    health["status"] = np.where(
        health["eval_coverage"] >= MULTIRES_SELECTION["min_eval_coverage"], "pass", "fail"
    )
    health["failure_reason"] = np.where(
        health["status"] == "fail",
        "coverage_below_threshold",
        "",
    )
    ordered = [
        "resolution",
        "horizon_minutes",
        "feature_set",
        "model_label",
        "forecast_strategy",
        "source_mode",
        "n_train",
        "n_validate",
        "n_test",
        "n_eval",
        "eval_coverage",
        "warmup_loss_pct",
        "runtime_seconds",
        "status",
        "failure_reason",
        "comparison_mode",
        "candidate_type",
    ]
    return health.loc[:, ordered]


def _compute_pareto_frontier(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the non-dominated candidate slice on error, stability, and runtime."""
    if candidates.empty:
        return candidates.copy()
    rows: list[int] = []
    for idx, row in candidates.iterrows():
        dominated = False
        for other_idx, other in candidates.iterrows():
            if idx == other_idx:
                continue
            no_worse = (
                float(other["mae_ratio_to_persistence"]) <= float(row["mae_ratio_to_persistence"])
                and float(other["fold_std_mae_ratio"]) <= float(row["fold_std_mae_ratio"])
                and float(other["runtime_seconds"]) <= float(row["runtime_seconds"])
            )
            strictly_better = (
                float(other["mae_ratio_to_persistence"]) < float(row["mae_ratio_to_persistence"])
                or float(other["fold_std_mae_ratio"]) < float(row["fold_std_mae_ratio"])
                or float(other["runtime_seconds"]) < float(row["runtime_seconds"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(idx)
    return candidates.loc[rows].copy()


def _annotate_selection_flags(summary: pd.DataFrame) -> pd.DataFrame:
    """Mark which Stage-6 candidates clear coverage, gain, and Pareto gates."""
    annotated = summary.copy()
    annotated["eligible"] = False
    annotated["practical_gain_passed"] = False
    annotated["pareto_passed"] = False

    matched = annotated.loc[annotated["comparison_mode"] == "matched_horizon"].copy()
    candidate_mask = ~(
        (matched["candidate_type"] == "baseline") & matched["model_label"].eq("persistence")
    )
    if candidate_mask.any():
        candidates = matched.loc[candidate_mask].copy()
        candidates["eligible"] = (
            (candidates["eval_coverage"] >= MULTIRES_SELECTION["min_eval_coverage"])
            & (candidates["fold_std_mae_ratio"] <= MULTIRES_SELECTION["max_fold_std_mae_ratio"])
            & ((candidates["runtime_seconds"] / 60.0) <= MULTIRES_SELECTION["max_candidate_runtime_minutes"])
        )
        candidates["practical_gain_passed"] = (
            ((1.0 - candidates["mae_ratio_to_persistence"]) >= MULTIRES_SELECTION["min_practical_mae_gain_pct"])
            | ((1.0 - candidates["rmse_ratio_to_persistence"]) >= MULTIRES_SELECTION["min_practical_rmse_gain_pct"])
        )
        eligible = candidates.loc[candidates["eligible"] & candidates["practical_gain_passed"]].copy()
        if not eligible.empty:
            pareto = _compute_pareto_frontier(eligible) if MULTIRES_SELECTION["pareto_enabled"] else eligible
            if not pareto.empty:
                candidates.loc[pareto.index, "pareto_passed"] = True
        annotated.loc[candidates.index, ["eligible", "practical_gain_passed", "pareto_passed"]] = candidates[
            ["eligible", "practical_gain_passed", "pareto_passed"]
        ]
    return annotated


def _select_winners(summary: pd.DataFrame) -> pd.DataFrame:
    """Select one operational winner per matched-horizon use case."""
    rows: list[dict[str, Any]] = []
    matched = _annotate_selection_flags(summary)
    matched = matched.loc[matched["comparison_mode"] == "matched_horizon"].copy()
    for horizon in sorted(matched["horizon_minutes"].dropna().unique()):
        horizon_df = matched.loc[matched["horizon_minutes"] == horizon].copy()
        persistence_row = horizon_df.loc[horizon_df["model_label"] == "persistence"].head(1)
        candidates = horizon_df.loc[horizon_df["pareto_passed"]].copy()
        if not candidates.empty:
            candidates = candidates.loc[
                ~((candidates["candidate_type"] == "baseline") & candidates["model_label"].eq("persistence"))
            ].copy()
        pareto = candidates.copy()
        if pareto.empty:
            persistence_metrics = persistence_row.iloc[0] if not persistence_row.empty else None
            rows.append(
                {
                    "use_case": f"matched_horizon_{int(horizon)}m",
                    "winner_type": "baseline_model",
                    "winner_resolution": str(persistence_row.iloc[0]["resolution"]) if not persistence_row.empty else "unknown",
                    "winner_feature_set": "baseline",
                    "winner_model_label": "persistence",
                    "winner_forecast_strategy": (
                        str(persistence_row.iloc[0]["forecast_strategy"]) if not persistence_row.empty else "path_baseline"
                    ),
                    "winner_horizon_minutes": int(horizon),
                    "decision_reason": "No non-persistence candidate cleared eligibility and practical-gain gates.",
                    "practical_gain_passed": False,
                    "pareto_passed": False,
                    "winner_endpoint_mae": (
                        float(persistence_metrics.get("mae", float("nan"))) if persistence_metrics is not None else float("nan")
                    ),
                    "winner_endpoint_mae_pct": (
                        float(persistence_metrics.get("mae_pct", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                    "winner_path_mae": (
                        float(persistence_metrics.get("path_mae", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                    "winner_path_mae_pct": (
                        float(persistence_metrics.get("path_mae_pct", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                    "winner_mae_ratio_to_persistence": (
                        float(persistence_metrics.get("mae_ratio_to_persistence", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                    "winner_rmse_ratio_to_persistence": (
                        float(persistence_metrics.get("rmse_ratio_to_persistence", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                    "winner_eval_coverage": (
                        float(persistence_metrics.get("eval_coverage", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                    "winner_runtime_seconds": (
                        float(persistence_metrics.get("runtime_seconds", float("nan")))
                        if persistence_metrics is not None
                        else float("nan")
                    ),
                }
            )
            continue
        winner = pareto.sort_values(
            ["mae_ratio_to_persistence", "fold_std_mae_ratio", "runtime_seconds"],
            ascending=[True, True, True],
        ).iloc[0]
        rows.append(
                {
                    "use_case": f"matched_horizon_{int(horizon)}m",
                    "winner_type": f"{winner['candidate_type']}_model",
                    "winner_resolution": str(winner["resolution"]),
                    "winner_feature_set": str(winner["feature_set"]),
                    "winner_model_label": str(winner["model_label"]),
                    "winner_forecast_strategy": str(winner["forecast_strategy"]),
                    "winner_horizon_minutes": int(horizon),
                    "decision_reason": (
                        "Selected from Pareto-eligible non-persistence candidates after coverage, "
                        "stability, runtime, practical-gain, and forecast-strategy gates."
                    ),
                    "practical_gain_passed": bool(winner["practical_gain_passed"]),
                    "pareto_passed": True,
                    "winner_endpoint_mae": float(winner.get("mae", float("nan"))),
                    "winner_endpoint_mae_pct": float(winner.get("mae_pct", float("nan"))),
                    "winner_path_mae": float(winner.get("path_mae", float("nan"))),
                    "winner_path_mae_pct": float(winner.get("path_mae_pct", float("nan"))),
                    "winner_mae_ratio_to_persistence": float(winner.get("mae_ratio_to_persistence", float("nan"))),
                    "winner_rmse_ratio_to_persistence": float(winner.get("rmse_ratio_to_persistence", float("nan"))),
                    "winner_eval_coverage": float(winner.get("eval_coverage", float("nan"))),
                    "winner_runtime_seconds": float(winner.get("runtime_seconds", float("nan"))),
                }
        )
    return pd.DataFrame(rows)


def _write_selection_summary_md(selection_summary: pd.DataFrame, output_dir: Path) -> None:
    """Write the human-readable markdown companion for the Stage-6 winners."""
    lines = ["# Multi-Resolution Selection Summary", ""]
    if selection_summary.empty:
        lines.extend(["No matched-horizon winner rows were produced.", ""])
    else:
        for _, row in selection_summary.iterrows():
            lines.extend(
                [
                    f"## {row['use_case']}",
                    "",
                    f"- Winner type: `{row['winner_type']}`",
                    f"- Resolution: `{row['winner_resolution']}`",
                    f"- Feature set: `{row['winner_feature_set']}`",
                    f"- Model: `{row['winner_model_label']}`",
                    f"- Forecast strategy: `{row['winner_forecast_strategy']}`",
                    f"- Horizon: `{int(row['winner_horizon_minutes'])}m`",
                    f"- Endpoint MAE: `{float(row['winner_endpoint_mae']):.6f}` ({float(row['winner_endpoint_mae_pct']):.3f}%)",
                    (
                        f"- Path MAE: `{float(row['winner_path_mae']):.6f}` ({float(row['winner_path_mae_pct']):.3f}%)"
                        if pd.notna(row.get("winner_path_mae")) and pd.notna(row.get("winner_path_mae_pct"))
                        else "- Path MAE: `n/a`"
                    ),
                    f"- Coverage: `{float(row['winner_eval_coverage']):.3f}`",
                    f"- Runtime seconds: `{float(row['winner_runtime_seconds']):.3f}`",
                    f"- Reason: {row['decision_reason']}",
                    "",
                ]
            )
    (output_dir / "selection_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_runtime_vs_gain(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot persistence-relative gain against runtime for learned candidates."""
    learned = summary.loc[
        (summary["comparison_mode"] == "matched_horizon") & (summary["candidate_type"] == "learned")
    ].copy()
    output_path = output_dir / "fig_runtime_vs_gain.png"
    plt.figure(figsize=(10, 6))
    if not learned.empty:
        x = learned["runtime_seconds"] / 60.0
        y = 1.0 - learned["mae_ratio_to_persistence"]
        labels = (
            learned["resolution"]
            + " @ "
            + learned["horizon_minutes"].astype(int).astype(str)
            + "m"
        )
        plt.scatter(x, y)
        for x_value, y_value, label in zip(x, y, labels, strict=True):
            plt.annotate(label, (x_value, y_value), fontsize=8)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("MAE gain vs persistence")
    plt.title("Matched-horizon runtime vs gain")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_pareto(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot the stability-vs-error Pareto slice for the first matched horizon."""
    matched = summary.loc[
        (summary["comparison_mode"] == "matched_horizon") & (summary["candidate_type"] == "learned")
    ].copy()
    output_path = output_dir / "fig_resolution_pareto.png"
    if matched.empty:
        plt.figure(figsize=(10, 6))
        plt.title("Resolution Pareto frontier (no learned candidates)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        validate_png_artifact(output_path)
        return
    target_horizon = int(sorted(matched["horizon_minutes"].unique())[0])
    slice_df = matched.loc[matched["horizon_minutes"] == target_horizon].copy()
    pareto = _compute_pareto_frontier(slice_df)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        slice_df["fold_std_mae_ratio"],
        slice_df["mae_ratio_to_persistence"],
        alpha=0.6,
        label="eligible",
    )
    if not pareto.empty:
        plt.scatter(
            pareto["fold_std_mae_ratio"],
            pareto["mae_ratio_to_persistence"],
            color="red",
            label="pareto",
        )
    for _, row in slice_df.iterrows():
        plt.annotate(
            f"{row['resolution']}:{row['model_label']}",
            (float(row["fold_std_mae_ratio"]), float(row["mae_ratio_to_persistence"])),
            fontsize=8,
        )
    plt.xlabel("Fold std of MAE ratio")
    plt.ylabel("MAE ratio to persistence")
    plt.title(f"Pareto frontier at {target_horizon}m matched horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _build_artifact_manifest(
    *,
    mode: str,
    comparison_mode: str,
    selected_resolutions: list[str],
    selected_horizons: list[int],
    selected_feature_sets: list[str],
    selected_model_labels: list[str],
    runtime_seconds: float,
    config_hash: str,
    selection_summary: pd.DataFrame,
    output_dir: Path,
    warnings: list[str],
    parallel_plans: list[dict[str, Any]],
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build the Stage-6 manifest that records settings, warnings, and artifact names."""
    artifacts = {
        name: str(path.name)
        for name, path in {
            "native_step_metrics": output_dir / "native_step_metrics.csv",
            "matched_horizon_metrics": output_dir / "matched_horizon_metrics.csv",
            "resolution_health": output_dir / "resolution_health.csv",
            "fold_metrics": output_dir / "fold_metrics.csv",
            "origin_metrics": output_dir / "origin_metrics.csv",
            "selection_summary_csv": output_dir / "selection_summary.csv",
            "selection_summary_md": output_dir / "selection_summary.md",
            "figure_guide_md": output_dir / "figure_guide.md",
            "winner_registry": output_dir / "winner_registry.csv",
            "fig_runtime_vs_gain": output_dir / "fig_runtime_vs_gain.png",
            "fig_resolution_pareto": output_dir / "fig_resolution_pareto.png",
        }.items()
    }
    return {
        "run_id": output_dir.name,
        "stage": "006_multires",
        "mode": mode,
        "comparison_mode": comparison_mode,
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "config_hash": config_hash,
        "source_mode": "bronze_direct",
        "resolutions": selected_resolutions,
        "horizons_minutes": selected_horizons,
        "matched_strategies": list(MULTIRES_CONFIG["matched_strategies"]),
        "feature_sets": selected_feature_sets,
        "model_labels": [*_enabled_path_baseline_labels(), *selected_model_labels],
        "selection_policy_version": "v2",
        "runtime_seconds": runtime_seconds,
        "parallel_runtime": {
            "config": dict(MODELING_PARALLEL),
            "resolved_plans": parallel_plans,
        },
        "runtime_environment": runtime_summary(int(MODELING_PARALLEL["max_workers"])).as_dict(),
        "status": "success" if not selection_summary.empty else "partial_failure",
        "warnings": [
            *warnings,
            *(["selection_summary_empty"] if selection_summary.empty else []),
        ],
        "artifacts": artifacts,
        "generated_at_utc": generated_at_utc,
    }


def _resolve_winner_metric_fields(selection_row: pd.Series, matched_metrics: pd.DataFrame) -> dict[str, float]:
    """Return winner metric fields from selection summary or backfill from matched metrics."""
    metric_columns = {
        "winner_endpoint_mae": "mae",
        "winner_endpoint_mae_pct": "mae_pct",
        "winner_path_mae": "path_mae",
        "winner_path_mae_pct": "path_mae_pct",
        "winner_mae_ratio_to_persistence": "mae_ratio_to_persistence",
        "winner_rmse_ratio_to_persistence": "rmse_ratio_to_persistence",
        "winner_eval_coverage": "eval_coverage",
        "winner_runtime_seconds": "runtime_seconds",
    }
    resolved = {
        field: pd.to_numeric(pd.Series([selection_row.get(field)]), errors="coerce").iloc[0]
        for field in metric_columns
    }
    if any(pd.notna(value) for value in resolved.values()):
        return resolved
    if matched_metrics.empty:
        return resolved

    candidates = matched_metrics.loc[
        matched_metrics["comparison_mode"].astype("string") == "matched_horizon"
    ].copy()
    for source_column, winner_field in (
        ("resolution", "winner_resolution"),
        ("feature_set", "winner_feature_set"),
        ("model_label", "winner_model_label"),
        ("forecast_strategy", "winner_forecast_strategy"),
    ):
        candidates = candidates.loc[
            candidates[source_column].astype("string") == str(selection_row.get(winner_field))
        ]
    horizon = pd.to_numeric(pd.Series([selection_row.get("winner_horizon_minutes")]), errors="coerce").iloc[0]
    if pd.notna(horizon):
        candidates = candidates.loc[
            pd.to_numeric(candidates["horizon_minutes"], errors="coerce") == float(horizon)
        ]
    if candidates.empty:
        return resolved
    winner_metrics = candidates.sort_values(
        ["mae_ratio_to_persistence", "runtime_seconds"],
        ascending=[True, True],
        kind="stable",
    ).iloc[0]
    return {
        field: pd.to_numeric(pd.Series([winner_metrics.get(source_column)]), errors="coerce").iloc[0]
        for field, source_column in metric_columns.items()
    }


def _build_winner_registry_snapshot(output_root: Path) -> pd.DataFrame:
    """Scan completed Stage-6 run directories into a registry of winner rows."""
    columns = [
        "run_id",
        "generated_at_utc",
        "mode",
        "comparison_mode",
        "selection_summary_path",
        "use_case",
        "winner_type",
        "winner_resolution",
        "winner_feature_set",
        "winner_model_label",
        "winner_forecast_strategy",
        "winner_horizon_minutes",
        "decision_reason",
        "practical_gain_passed",
        "pareto_passed",
        "winner_endpoint_mae",
        "winner_endpoint_mae_pct",
        "winner_path_mae",
        "winner_path_mae_pct",
        "winner_mae_ratio_to_persistence",
        "winner_rmse_ratio_to_persistence",
        "winner_eval_coverage",
        "winner_runtime_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(output_root.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir() or run_dir.name.startswith("latest"):
            continue
        manifest_path = run_dir / "run_manifest.json"
        selection_path = run_dir / "selection_summary.csv"
        if not manifest_path.exists() or not selection_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selection = pd.read_csv(selection_path)
        except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
            continue
        if selection.empty:
            continue
        matched_metrics_path = run_dir / "matched_horizon_metrics.csv"
        try:
            matched_metrics = pd.read_csv(matched_metrics_path) if matched_metrics_path.exists() else pd.DataFrame()
        except (OSError, pd.errors.EmptyDataError):
            matched_metrics = pd.DataFrame()
        try:
            selection_summary_path = str(selection_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            selection_summary_path = str(selection_path)
        for _, row in selection.iterrows():
            winner_metrics = _resolve_winner_metric_fields(row, matched_metrics)
            rows.append(
                {
                    "run_id": run_dir.name,
                    "generated_at_utc": manifest.get("generated_at_utc"),
                    "mode": manifest.get("mode"),
                    "comparison_mode": manifest.get("comparison_mode"),
                    "selection_summary_path": selection_summary_path,
                    "use_case": row.get("use_case"),
                    "winner_type": row.get("winner_type"),
                    "winner_resolution": row.get("winner_resolution"),
                    "winner_feature_set": row.get("winner_feature_set"),
                    "winner_model_label": row.get("winner_model_label"),
                    "winner_forecast_strategy": row.get("winner_forecast_strategy"),
                    "winner_horizon_minutes": row.get("winner_horizon_minutes"),
                    "decision_reason": row.get("decision_reason"),
                    "practical_gain_passed": row.get("practical_gain_passed"),
                    "pareto_passed": row.get("pareto_passed"),
                    "winner_endpoint_mae": winner_metrics["winner_endpoint_mae"],
                    "winner_endpoint_mae_pct": winner_metrics["winner_endpoint_mae_pct"],
                    "winner_path_mae": winner_metrics["winner_path_mae"],
                    "winner_path_mae_pct": winner_metrics["winner_path_mae_pct"],
                    "winner_mae_ratio_to_persistence": winner_metrics["winner_mae_ratio_to_persistence"],
                    "winner_rmse_ratio_to_persistence": winner_metrics["winner_rmse_ratio_to_persistence"],
                    "winner_eval_coverage": winner_metrics["winner_eval_coverage"],
                    "winner_runtime_seconds": winner_metrics["winner_runtime_seconds"],
                }
            )
    registry = pd.DataFrame(rows, columns=columns)
    if registry.empty:
        return registry
    registry["winner_horizon_minutes"] = pd.to_numeric(
        registry["winner_horizon_minutes"], errors="coerce"
    )
    registry["generated_at_utc"] = pd.to_datetime(registry["generated_at_utc"], errors="coerce", utc=True)
    for column in ("practical_gain_passed", "pareto_passed"):
        registry[column] = registry[column].astype("string").str.lower().map(
            {"true": True, "false": False}
        ).fillna(False)
    for column in (
        "winner_endpoint_mae",
        "winner_endpoint_mae_pct",
        "winner_path_mae",
        "winner_path_mae_pct",
        "winner_mae_ratio_to_persistence",
        "winner_rmse_ratio_to_persistence",
        "winner_eval_coverage",
        "winner_runtime_seconds",
    ):
        registry[column] = pd.to_numeric(registry[column], errors="coerce")
    registry = registry.sort_values(
        ["winner_horizon_minutes", "generated_at_utc", "run_id"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return registry


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Stage-6 multiresolution comparison runner."""
    parser = argparse.ArgumentParser(description="Run multiresolution matched-horizon comparison.")
    parser.add_argument("--mode", choices=sorted(MULTIRES_PROFILES), default=MULTIRES_CONFIG["mode"])
    parser.add_argument(
        "--comparison-mode",
        choices=["native_step", "matched_horizon"],
        default=MULTIRES_CONFIG["comparison_mode"],
    )
    parser.add_argument("--resolution", action="append", dest="resolutions")
    parser.add_argument("--horizon", type=int, action="append", dest="horizons_minutes")
    parser.add_argument("--feature-set", action="append", dest="feature_sets")
    parser.add_argument("--model-label", action="append", dest="model_labels")
    parser.add_argument("--output-dir", default=str(scoped_output_path(PATHS["outputs_multires_dir"])))
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--val-window-days", type=int, default=None)
    parser.add_argument("--origins-per-fold", type=int, default=None)
    parser.add_argument("--disable-native-step", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute Stage-6 and persist one timestamped multiresolution run directory."""
    _configure_logging()
    validate_config()
    args = parse_args()
    profile = _resolve_mode_profile(args.mode)
    selected_resolutions = [
        canonical_resolution(item) for item in (args.resolutions or profile["resolutions"])
    ]
    selected_horizons = [int(item) for item in (args.horizons_minutes or profile["horizons_minutes"])]
    selected_feature_sets = list(args.feature_sets or profile["feature_sets"])
    selected_model_labels = list(args.model_labels or profile["model_labels"])
    n_folds = int(args.n_folds or profile["n_folds"])
    val_window_days = int(args.val_window_days or profile["val_window_days"])
    origins_per_fold = int(args.origins_per_fold or profile["origins_per_fold"])
    include_native_step = MULTIRES_CONFIG["native_step_enabled"] and not bool(args.disable_native_step)

    for feature_set in selected_feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
    catalog = build_model_catalog()
    unavailable_model_labels = [
        model_label for model_label in selected_model_labels if model_label not in catalog
    ]
    if unavailable_model_labels:
        logger.warning(
            "Skipping unavailable model labels for this environment: %s",
            unavailable_model_labels,
        )
    selected_model_labels = [
        model_label for model_label in selected_model_labels if model_label in catalog
    ]
    if not selected_model_labels:
        raise ValueError(
            "None of the requested model labels are available. "
            f"Requested: {sorted(set(args.model_labels or profile['model_labels']))}; "
            f"available: {sorted(catalog)}"
        )

    output_root = Path(args.output_dir).resolve()
    manifest_warnings: list[str] = []
    selected_resolutions = list(dict.fromkeys(selected_resolutions))
    available_resolutions, missing_resolutions = _partition_available_resolutions(selected_resolutions)
    if missing_resolutions:
        for missing in missing_resolutions:
            logger.warning("Skipping missing gold input for Stage-6 resolution=%s", missing)
        manifest_warnings.extend(f"skipped_missing_resolution:{item}" for item in missing_resolutions)
    if not available_resolutions:
        logger.error(
            "No Stage-6 resolutions have available gold inputs. Requested=%s",
            ", ".join(selected_resolutions),
        )
        return 1
    selected_resolutions = available_resolutions

    run_timestamp = datetime.now(UTC)
    generated_at_utc = run_timestamp.isoformat()
    run_dir = output_root / run_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_config = {
        "mode": args.mode,
        "comparison_mode": args.comparison_mode,
        "resolutions": selected_resolutions,
        "horizons_minutes": selected_horizons,
        "matched_strategies": list(MULTIRES_CONFIG["matched_strategies"]),
        "feature_sets": selected_feature_sets,
        "model_labels": selected_model_labels,
        "n_folds": n_folds,
        "val_window_days": val_window_days,
        "origins_per_fold": origins_per_fold,
        "selection": dict(MULTIRES_SELECTION),
        "runtime": dict(MULTIRES_RUNTIME),
        "parallel_runtime": dict(MODELING_PARALLEL),
        "rollout_defaults": dict(MULTIRES_ROLLOUT),
    }
    config_hash = stable_config_hash(effective_config)

    all_metric_rows: list[dict[str, Any]] = []
    all_origin_rows: list[dict[str, Any]] = []
    resolved_parallel_plans: list[dict[str, Any]] = []
    started = time.perf_counter()
    for resolution in selected_resolutions:
        logger.info("Running multires comparison for resolution=%s", resolution)
        base = load_base_gold(resolution)
        frame = build_causal_feature_frame(base, resolution)
        day_lookup = _day_class_lookup(base)
        valid_horizons, skipped_horizons = _partition_representable_horizons(
            resolution,
            selected_horizons,
        )
        if skipped_horizons:
            warning = (
                f"skipped_non_representable:{resolution}:"
                + ",".join(str(value) for value in skipped_horizons)
            )
            manifest_warnings.append(warning)
            logger.warning(
                "Skipping non-representable matched horizons for %s: %s",
                resolution,
                ", ".join(f"{value}m" for value in skipped_horizons),
            )
        folds = build_walkforward_folds(
            holdout_start_day=int(SPLIT_DAY_RANGES["test"][0]),
            n_folds=n_folds,
            val_window_days=val_window_days,
            train_start_day=int(SPLIT_DAY_RANGES["train"][0]),
        )
        if include_native_step:
            native_feature_sets, native_model_labels = _resolve_horizon_scope(
                horizon_minutes=max(1, int(np.ceil(resolution_total_minutes(resolution)))),
                feature_sets=selected_feature_sets,
                model_labels=selected_model_labels,
            )
            native_tasks = [
                NativeMetricTask(
                    fold=fold,
                    feature_set=feature_set,
                    model_label=model_label,
                    emit_baselines=(
                        feature_set == native_feature_sets[0]
                        and model_label == native_model_labels[0]
                    ),
                )
                for fold in folds
                for feature_set in native_feature_sets
                for model_label in native_model_labels
            ]
            native_results, native_plan = run_stage_jobs(
                "multires",
                native_tasks,
                worker=partial(
                    _run_native_step_task,
                    frame=frame,
                    resolution=resolution,
                ),
                logger_instance=logger,
            )
            resolved_parallel_plans.append(
                {
                    **native_plan.as_dict(),
                    "resolution": resolution,
                    "job_type": "native_step",
                }
            )
            for rows in native_results:
                all_metric_rows.extend(rows)
        if args.comparison_mode == "matched_horizon":
            if not valid_horizons:
                logger.warning(
                    "No representable matched horizons remain for resolution=%s; skipping matched-horizon evaluation.",
                    resolution,
                )
                continue
            matched_tasks: list[MatchedMetricTask] = []
            for horizon_minutes in valid_horizons:
                horizon_feature_sets, horizon_model_labels = _resolve_horizon_scope(
                    horizon_minutes=int(horizon_minutes),
                    feature_sets=selected_feature_sets,
                    model_labels=selected_model_labels,
                )
                matched_tasks.extend(
                    [
                        MatchedMetricTask(
                            fold=fold,
                            feature_set=feature_set,
                            model_label=model_label,
                            horizon_minutes=horizon_minutes,
                            emit_baselines=(
                                feature_set == horizon_feature_sets[0]
                                and model_label == horizon_model_labels[0]
                            ),
                        )
                        for fold in folds
                        for feature_set in horizon_feature_sets
                        for model_label in horizon_model_labels
                    ]
                )
            matched_results, matched_plan = run_stage_jobs(
                "multires",
                matched_tasks,
                worker=partial(
                    _run_matched_horizon_task,
                    base=base,
                    frame=frame,
                    day_class_lookup=day_lookup,
                    resolution=resolution,
                    origins_per_fold=origins_per_fold,
                ),
                logger_instance=logger,
            )
            resolved_parallel_plans.append(
                {
                    **matched_plan.as_dict(),
                    "resolution": resolution,
                    "job_type": "matched_horizon",
                }
            )
            for metric_rows, origin_rows in matched_results:
                all_metric_rows.extend(metric_rows)
                all_origin_rows.extend(origin_rows)

    fold_metrics = pd.DataFrame(all_metric_rows)
    if fold_metrics.empty:
        logger.error("No multiresolution metrics were generated.")
        return 1
    fold_metrics, removed_fold_duplicates = _deduplicate_metric_frame(fold_metrics)
    if removed_fold_duplicates:
        warning = f"deduplicated_fold_metrics:{removed_fold_duplicates}"
        manifest_warnings.append(warning)
        logger.warning("Removed %d exact duplicate Stage-6 fold metric rows.", removed_fold_duplicates)
    fold_metrics = fold_metrics.sort_values(
        [
            "comparison_mode",
            "resolution",
            "horizon_minutes",
            "fold",
            "feature_set",
            "model_label",
        ],
        kind="stable",
    ).reset_index(drop=True)
    summary = _aggregate_metrics(fold_metrics)
    summary = _annotate_selection_flags(summary)
    native_summary = summary.loc[summary["comparison_mode"] == "native_step"].copy()
    matched_summary = summary.loc[summary["comparison_mode"] == "matched_horizon"].copy()
    health = _build_resolution_health(summary)
    selection_summary = _select_winners(summary)
    runtime_seconds = time.perf_counter() - started

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False, float_format="%.6f")
    native_summary.to_csv(run_dir / "native_step_metrics.csv", index=False, float_format="%.6f")
    matched_summary.to_csv(
        run_dir / "matched_horizon_metrics.csv", index=False, float_format="%.6f"
    )
    health.to_csv(run_dir / "resolution_health.csv", index=False, float_format="%.6f")
    selection_summary.to_csv(run_dir / "selection_summary.csv", index=False)
    origin_metrics = pd.DataFrame(all_origin_rows)
    origin_metrics, removed_origin_duplicates = _deduplicate_metric_frame(origin_metrics)
    if removed_origin_duplicates:
        warning = f"deduplicated_origin_metrics:{removed_origin_duplicates}"
        manifest_warnings.append(warning)
        logger.warning("Removed %d exact duplicate Stage-6 origin metric rows.", removed_origin_duplicates)
    if not origin_metrics.empty:
        origin_metrics = origin_metrics.sort_values(
            [
                "comparison_mode",
                "resolution",
                "horizon_minutes",
                "fold",
                "origin_timestamp",
                "candidate_label",
            ],
            kind="stable",
        ).reset_index(drop=True)
    origin_metrics.to_csv(run_dir / "origin_metrics.csv", index=False, float_format="%.6f")
    _write_selection_summary_md(selection_summary, run_dir)
    _plot_runtime_vs_gain(summary, run_dir)
    _plot_pareto(summary, run_dir)
    write_figure_guide(
        output_path=run_dir / "figure_guide.md",
        stage_title="Stage-6 Multiresolution Figures",
        stage_purpose=(
            "These figures explain how Stage-6 balances runtime, stability, and "
            "persistence-relative gain when choosing a resolution-horizon winner."
        ),
        figures=[
            FigureGuideEntry(
                filename="fig_runtime_vs_gain.png",
                title="Runtime vs gain",
                intent="Show whether a slower candidate earns enough persistence-relative improvement to justify its runtime.",
                how_to_read="The x-axis is runtime in minutes and the y-axis is MAE gain over persistence. Better candidates sit higher and further left.",
                look_for="Candidates above zero gain that do not drift into disproportionately high runtime.",
            ),
            FigureGuideEntry(
                filename="fig_resolution_pareto.png",
                title="Resolution Pareto frontier",
                intent="Show which eligible learned candidates are on the best observed stability-error frontier.",
                how_to_read="The x-axis is fold variability and the y-axis is MAE ratio to persistence. Better candidates sit toward the lower-left frontier.",
                look_for="Resolution choices that are both low-error and stable across folds, not just one-off low-error outliers.",
            ),
        ],
    )

    manifest = _build_artifact_manifest(
        mode=args.mode,
        comparison_mode=args.comparison_mode,
        selected_resolutions=selected_resolutions,
        selected_horizons=selected_horizons,
        selected_feature_sets=selected_feature_sets,
        selected_model_labels=selected_model_labels,
        runtime_seconds=runtime_seconds,
        config_hash=config_hash,
        selection_summary=selection_summary,
        output_dir=run_dir,
        warnings=manifest_warnings,
        parallel_plans=resolved_parallel_plans,
        generated_at_utc=generated_at_utc,
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    winner_registry = _build_winner_registry_snapshot(output_root)
    winner_registry.to_csv(output_root / "winner_registry.csv", index=False)
    winner_registry.to_csv(run_dir / "winner_registry.csv", index=False)
    update_latest_alias(run_dir, output_root / "latest", enabled=bool(MULTIRES_CONFIG["write_latest"]))
    update_latest_alias(
        run_dir,
        output_root / f"latest_{args.mode}",
        enabled=bool(MULTIRES_CONFIG["write_latest"]),
    )
    emit_quality_gate(
        "MULTIRES HEALTH",
        not selection_summary.empty,
        details={
            "resolutions": len(selected_resolutions),
            "horizons": len(selected_horizons),
            "runtime_seconds": round(runtime_seconds, 2),
        },
        logger_instance=logger,
    )
    logger.info("Multiresolution artifacts written to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
