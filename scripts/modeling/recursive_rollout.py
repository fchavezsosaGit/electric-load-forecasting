"""Stage-7 recursive rollout evaluation at the selected multiresolution candidate.

This stage takes the learned or baseline candidate selected by Stage-6 or the
challenger-sweep registry and measures how that policy behaves over a recursive
forecast horizon. Its artifacts are later reused by the challenger sweep,
horizon curve, and forecast-control backtest.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
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

from config import (
    DATASET,
    FEATURE_SETS,
    MODELING_PERFORMANCE_EVALUATION,
    MULTIRES_BASELINES,
    MULTIRES_CONFIG,
    MULTIRES_HYBRID,
    MULTIRES_ROLLOUT,
    MULTIRES_ROLLOUT_LEARNED_BLENDS,
    MULTIRES_ROLLOUT_POLICY_CANDIDATES,
    output_path_candidates,
    PATHS,
    PROJECT_ROOT,
    SPLIT_DAY_RANGES,
    resolve_horizon_policy,
    resolve_rollout_origin_policy,
    resolve_rollout_selection_target,
    scoped_output_path,
    validate_config,
)
from utils import emit_quality_gate

from modeling.common import (
    build_model_catalog,
    canonical_resolution,
    FigureGuideEntry,
    stable_config_hash,
    train_model,
    update_latest_alias,
    validate_png_artifact,
    write_figure_guide,
)
from modeling.multires import (
    actual_path,
    anchored_workday_path,
    avg_workday_path,
    blend_candidate_paths,
    build_causal_feature_frame,
    build_workday_profile,
    compare_recursive_paths,
    lead_steps_for_horizon,
    load_base_gold,
    persistence_path,
    previous_day_path,
    recursive_predict_path,
    recursive_predict_residual_path,
)

logger = logging.getLogger(__name__)
_BLEND_SUFFIX_RE = re.compile(r"_e(?P<weight>\d{2,3})$")
_ROLLOUT_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
_ROLLOUT_RUNTIME_CACHE_LOCK = threading.Lock()


def _configure_logging() -> None:
    """Initialize a simple logger for standalone Stage-7 execution."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _day_class_lookup(base: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """Build a normalized-date lookup used by recursive baseline paths."""
    lookup = (
        base.loc[:, ["timestamp", "day_class"]]
        .assign(date=lambda df: df["timestamp"].dt.normalize())
        .drop_duplicates(subset=["date"])
        .set_index("date")["day_class"]
    )
    return {pd.Timestamp(index): str(value) for index, value in lookup.items()}


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    """Read a CSV artifact when present, otherwise return an empty frame."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Normalize loose string and null boolean values into strict booleans."""
    return (
        series.astype("string")
        .str.lower()
        .map({"true": True, "false": False})
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )


def _relative_artifact_path(path: Path) -> str:
    """Render an artifact path relative to the repository root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _rollout_runtime_cache_key(
    *,
    resolution: str,
    feature_set: str,
    model_label: str,
    horizon_minutes: int,
) -> str:
    """Hash the stable inputs that define one reusable Stage-7 training context.

    The expensive part of Stage-7 exact-origin replay is not origin sampling,
    but rebuilding the causal frame and refitting the same one-step models.
    This key isolates that shared setup so Stage-10 can reuse it across many
    exact-origin control replays inside one Python process.
    """
    horizon_policy = resolve_horizon_policy(int(horizon_minutes))
    payload = {
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "resolution": canonical_resolution(str(resolution)),
        "feature_set": str(feature_set),
        "model_label": str(model_label),
        "horizon_minutes": int(horizon_minutes),
        "feature_columns": _resolve_rollout_feature_columns(str(feature_set)),
        "residual_baseline": str(horizon_policy["rollout_residual_baseline"]),
        "residual_candidates": [
            str(value) for value in horizon_policy.get("rollout_residual_candidates", [])
        ],
        "allow_residual": bool(horizon_policy["allow_residual"]),
        "train_start_day": int(SPLIT_DAY_RANGES["train"][0]),
        "train_end_day": int(SPLIT_DAY_RANGES["validate"][1]),
    }
    return stable_config_hash(payload).removeprefix("sha256:")


def _resolve_artifact_path(path_value: str | None) -> Path | None:
    """Resolve an optional stored artifact path into an absolute filesystem path."""
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _prepare_rollout_runtime_context(
    *,
    resolution: str,
    feature_set: str,
    model_label: str,
    horizon_minutes: int,
) -> dict[str, Any]:
    """Build and memoize the shared Stage-7 runtime state for one candidate family.

    The returned context is read-only from the caller's perspective. Per-origin
    histories and candidate pruning still happen inside `run_rollout_evaluation`,
    but the loaded gold data, causal feature frame, trained raw model, and any
    residual models are reused when Stage-10 replays the same selection across
    different exact-origin subsets.
    """
    cache_key = _rollout_runtime_cache_key(
        resolution=str(resolution),
        feature_set=str(feature_set),
        model_label=str(model_label),
        horizon_minutes=int(horizon_minutes),
    )
    with _ROLLOUT_RUNTIME_CACHE_LOCK:
        cached = _ROLLOUT_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    catalog = build_model_catalog()
    if model_label not in catalog:
        raise ValueError(f"Unknown model label: {model_label}")

    canonical_resolution_value = canonical_resolution(str(resolution))
    base = load_base_gold(canonical_resolution_value)
    frame = build_causal_feature_frame(base, canonical_resolution_value)
    day_lookup = _day_class_lookup(base)
    horizon_steps = lead_steps_for_horizon(canonical_resolution_value, int(horizon_minutes))
    train_start_day, train_end_day = SPLIT_DAY_RANGES["train"][0], SPLIT_DAY_RANGES["validate"][1]
    train_df = frame.loc[frame["day_idx"].between(train_start_day, train_end_day)].copy()
    horizon_policy = resolve_horizon_policy(int(horizon_minutes))
    raw_feature_columns = _resolve_rollout_feature_columns(str(feature_set))
    trained = train_model(train_df, raw_feature_columns, catalog[str(model_label)])
    profile = build_workday_profile(train_df)
    residual_baseline = str(horizon_policy["rollout_residual_baseline"])
    residual_models: dict[str, Any] = {}
    if bool(horizon_policy["allow_residual"]):
        residual_candidates = [
            str(value)
            for value in horizon_policy.get("rollout_residual_candidates", [residual_baseline])
        ]
        if residual_baseline not in residual_candidates:
            residual_candidates = [residual_baseline, *residual_candidates]
        for candidate_baseline in dict.fromkeys(residual_candidates):
            residual_feature_columns = _resolve_rollout_feature_columns(
                str(feature_set),
                residual_baseline=str(candidate_baseline),
            )
            residual_models[str(candidate_baseline)] = _train_rollout_residual_model(
                train_df=train_df,
                feature_columns=residual_feature_columns,
                model_label=str(model_label),
                residual_baseline=str(candidate_baseline),
                resolution=canonical_resolution_value,
                horizon_minutes=int(horizon_minutes),
            )

    runtime = {
        "base": base,
        "frame": frame,
        "day_lookup": day_lookup,
        "horizon_steps": int(horizon_steps),
        "train_df": train_df,
        "horizon_policy": horizon_policy,
        "raw_feature_columns": raw_feature_columns,
        "trained": trained,
        "profile": profile,
        "residual_baseline": residual_baseline,
        "residual_models": residual_models,
    }
    with _ROLLOUT_RUNTIME_CACHE_LOCK:
        existing = _ROLLOUT_RUNTIME_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _ROLLOUT_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _phase_bucket_seconds(timestamp: pd.Timestamp) -> int:
    """Bucket a timestamp by its offset within a 15-minute billing phase."""
    return int((int(timestamp.minute) * 60 + int(timestamp.second)) % (15 * 60))


def _hybrid_phase_gate_weight(origin_timestamp: pd.Timestamp) -> float:
    """Choose the hybrid-workday fallback weight from the origin's quarter-hour phase."""
    bucket = _phase_bucket_seconds(pd.Timestamp(origin_timestamp))
    bucket_weights = MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_bucket_weights"]
    if bucket in bucket_weights:
        return float(bucket_weights[bucket])
    if bucket == 0:
        return float(MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_aligned_weight"])
    return float(MULTIRES_ROLLOUT_LEARNED_BLENDS["hybrid_phase_gate_non_aligned_weight"])


def _evenly_sample_positions(positions: list[int], count: int) -> list[int]:
    """Return deterministic evenly spaced positions without duplicates."""
    if count <= 0 or not positions:
        return []
    if count >= len(positions):
        return list(positions)
    index_values = np.linspace(0, len(positions) - 1, num=count, dtype=int)
    selected: list[int] = []
    seen: set[int] = set()
    for index_value in index_values.tolist():
        position = positions[int(index_value)]
        if position not in seen:
            seen.add(position)
            selected.append(position)
    for position in positions:
        if len(selected) >= count:
            break
        if position not in seen:
            seen.add(position)
            selected.append(position)
    return selected


def _select_phase_balanced_origins(
    base: pd.DataFrame,
    *,
    candidate_positions: list[int],
    max_origins: int,
) -> list[int]:
    """Spread rollout origins across the 15-minute phase cycle before subsampling time."""
    if not candidate_positions or len(candidate_positions) <= max_origins:
        return list(candidate_positions)
    buckets: dict[int, list[int]] = {}
    for position in candidate_positions:
        timestamp = pd.Timestamp(base.iloc[position]["timestamp"])
        bucket = _phase_bucket_seconds(timestamp)
        buckets.setdefault(bucket, []).append(position)
    if not buckets:
        return _evenly_sample_positions(candidate_positions, max_origins)

    target = min(int(max_origins), len(candidate_positions))
    selected_bucket_keys = _evenly_sample_positions(sorted(buckets), min(target, len(buckets)))
    if not selected_bucket_keys:
        return _evenly_sample_positions(candidate_positions, max_origins)
    max_per_bucket = max(1, math.ceil(target / len(selected_bucket_keys)))
    bucket_samples = {
        bucket: _evenly_sample_positions(positions, max_per_bucket)
        for bucket, positions in buckets.items()
        if bucket in selected_bucket_keys
    }
    selected: list[int] = []
    seen: set[int] = set()
    round_count = max((len(samples) for samples in bucket_samples.values()), default=0)
    for round_idx in range(round_count):
        for bucket in selected_bucket_keys:
            samples = bucket_samples[bucket]
            if round_idx >= len(samples):
                continue
            position = samples[round_idx]
            if position in seen:
                continue
            seen.add(position)
            selected.append(position)
            if len(selected) >= target:
                return selected

    leftovers = [position for position in candidate_positions if position not in seen]
    if leftovers and len(selected) < target:
        selected.extend(_evenly_sample_positions(leftovers, target - len(selected)))
    return selected[:target]


def _selection_metric_fields(selection_target: str) -> dict[str, str]:
    """Return the metric columns and labels tied to one rollout selection target."""
    if selection_target == "path_mae":
        return {
            "metric": "learned_path_mae",
            "secondary": "learned_endpoint_mae",
            "beat_baseline": "beats_best_baseline_path",
            "beat_persistence": "beats_persistence_path",
            "label": "path MAE",
        }
    if selection_target == "endpoint_mae":
        return {
            "metric": "learned_endpoint_mae",
            "secondary": "learned_path_mae",
            "beat_baseline": "beats_best_baseline_endpoint",
            "beat_persistence": "beats_persistence_endpoint",
            "label": "endpoint MAE",
        }
    if selection_target == "phase_mean_mae":
        return {
            "metric": "learned_phase_mean_mae",
            "secondary": "learned_path_mae",
            "beat_baseline": "beats_best_baseline_phase",
            "beat_persistence": "beats_persistence_phase",
            "label": "15-minute phase-average MAE",
        }
    if selection_target == "next_lock_mae":
        return {
            "metric": "learned_next_lock_mae",
            "secondary": "learned_path_mae",
            "beat_baseline": "beats_best_baseline_next_lock",
            "beat_persistence": "beats_persistence_next_lock",
            "label": "next 15-minute MAE",
        }
    if selection_target == "profile_shape_mae":
        return {
            "metric": "learned_profile_shape_mae",
            "secondary": "learned_path_mae",
            "beat_baseline": "beats_best_baseline_profile_shape",
            "beat_persistence": "beats_persistence_profile_shape",
            "label": "profile-shape MAE",
        }
    raise ValueError(f"Unsupported rollout selection target: {selection_target}")


def _rollout_policy_metric_fields(selection_target: str) -> dict[str, str]:
    """Return the by-origin metric fields used for derived rollout policies."""
    if selection_target == "path_mae":
        return {
            "metric": "path_mae",
            "secondary": "endpoint_abs_error",
            "policy_suffix": "path",
        }
    if selection_target == "next_lock_mae":
        return {
            "metric": "next_lock_mae",
            "secondary": "path_mae",
            "policy_suffix": "next_lock",
        }
    if selection_target == "endpoint_mae":
        return {
            "metric": "endpoint_abs_error",
            "secondary": "path_mae",
            "policy_suffix": "endpoint",
        }
    if selection_target == "phase_mean_mae":
        return {
            "metric": "phase_mean_abs_error",
            "secondary": "path_mae",
            "policy_suffix": "phase",
        }
    raise ValueError(f"Unsupported rollout policy selection target: {selection_target}")


def _build_phase_bucket_policy_candidates(
    by_origin: pd.DataFrame,
    *,
    model_label: str,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Synthesize short-horizon policy candidates from measured phase-bucket champions."""
    if (
        by_origin.empty
        or not bool(MULTIRES_ROLLOUT_POLICY_CANDIDATES["enabled"])
        or int(horizon_minutes) > int(MULTIRES_ROLLOUT_POLICY_CANDIDATES["max_horizon_minutes"])
    ):
        return pd.DataFrame(), []
    if "origin_timestamp" not in by_origin.columns:
        return pd.DataFrame(), []

    working = by_origin.copy()
    working["origin_timestamp"] = pd.to_datetime(working["origin_timestamp"], errors="coerce")
    working = working.loc[working["origin_timestamp"].notna()].copy()
    if working.empty:
        return pd.DataFrame(), []
    working["phase_bucket_seconds"] = working["origin_timestamp"].map(_phase_bucket_seconds)
    unique_buckets = sorted({int(bucket) for bucket in working["phase_bucket_seconds"].dropna().tolist()})
    if len(unique_buckets) < 2:
        return pd.DataFrame(), []

    selector_rows: list[dict[str, Any]] = []
    selector_metadata: list[dict[str, Any]] = []
    unique_origins = (
        working.loc[:, ["origin_timestamp", "phase_bucket_seconds"]]
        .drop_duplicates()
        .sort_values("origin_timestamp", kind="stable")
        .reset_index(drop=True)
    )
    for selection_target in MULTIRES_ROLLOUT_POLICY_CANDIDATES["selection_targets"]:
        fields = _rollout_policy_metric_fields(str(selection_target))
        metric_column = fields["metric"]
        secondary_column = fields["secondary"]
        if metric_column not in working.columns or secondary_column not in working.columns:
            continue

        bucket_mapping: dict[int, str] = {}
        bucket_details: list[dict[str, Any]] = []
        for bucket in unique_buckets:
            bucket_rows = working.loc[working["phase_bucket_seconds"].eq(int(bucket))].copy()
            if bucket_rows.empty:
                continue
            ranked = (
                bucket_rows.groupby("candidate_label", dropna=False)
                .agg(
                    metric_mean=(metric_column, "mean"),
                    secondary_mean=(secondary_column, "mean"),
                    candidate_type=("candidate_type", "first"),
                    target_mode=("target_mode", "first"),
                    origin_n=("origin_timestamp", "nunique"),
                )
                .reset_index()
                .sort_values(
                    ["metric_mean", "secondary_mean", "candidate_label"],
                    ascending=[True, True, True],
                    kind="stable",
                )
                .reset_index(drop=True)
            )
            winner = ranked.iloc[0]
            winner_label = str(winner["candidate_label"])
            bucket_mapping[int(bucket)] = winner_label
            bucket_details.append(
                {
                    "phase_bucket_seconds": int(bucket),
                    "selected_candidate_label": winner_label,
                    "selected_candidate_type": str(winner.get("candidate_type", "")),
                    "selected_target_mode": str(winner.get("target_mode", "")),
                    "mean_metric_value": float(winner["metric_mean"]),
                    "mean_secondary_value": float(winner["secondary_mean"]),
                    "origin_n": int(winner["origin_n"]),
                }
            )
        if len(bucket_mapping) != len(unique_buckets):
            continue

        selector_label = f"{model_label}::phase_bucket_{fields['policy_suffix']}_policy"
        selector_target_mode = f"phase_bucket_{fields['policy_suffix']}_policy"
        for _, origin_row in unique_origins.iterrows():
            origin_timestamp = pd.Timestamp(origin_row["origin_timestamp"])
            bucket = int(origin_row["phase_bucket_seconds"])
            chosen_label = bucket_mapping[bucket]
            matched = working.loc[
                working["origin_timestamp"].eq(origin_timestamp)
                & working["candidate_label"].astype("string").eq(chosen_label)
            ].copy()
            if matched.empty:
                continue
            row_payload = matched.iloc[0].to_dict()
            row_payload["candidate_label"] = selector_label
            row_payload["candidate_type"] = "learned"
            row_payload["source_model_label"] = str(model_label)
            row_payload["target_mode"] = selector_target_mode
            row_payload["policy_selection_target"] = str(selection_target)
            row_payload["policy_source_candidate"] = chosen_label
            row_payload["policy_phase_bucket_seconds"] = int(bucket)
            selector_rows.append(row_payload)

        selector_metadata.append(
            {
                "candidate_label": selector_label,
                "target_mode": selector_target_mode,
                "selection_target": str(selection_target),
                "metric_column": metric_column,
                "secondary_column": secondary_column,
                "phase_bucket_mapping": bucket_mapping,
                "phase_bucket_details": bucket_details,
            }
        )

    if not selector_rows:
        return pd.DataFrame(), []
    return pd.DataFrame(selector_rows), selector_metadata


def _select_stage6_summary_candidate(summary: pd.DataFrame, *, requested_horizon_minutes: int) -> pd.Series | None:
    """Pick the best recursive learned Stage-6 winner for an exact rollout horizon."""
    if summary.empty:
        return None
    learned = summary.loc[summary["winner_type"].astype("string").eq("learned_model")].copy()
    if learned.empty:
        return None
    if "winner_forecast_strategy" in learned.columns:
        learned = learned.loc[
            learned["winner_forecast_strategy"].astype("string").fillna("recursive").eq("recursive")
        ].copy()
    if learned.empty:
        return None
    learned["winner_horizon_minutes"] = pd.to_numeric(learned["winner_horizon_minutes"], errors="coerce")
    learned = learned.loc[learned["winner_horizon_minutes"].eq(int(requested_horizon_minutes))].copy()
    if learned.empty:
        return None
    learned["pareto_passed"] = _coerce_bool_series(
        learned.get("pareto_passed", pd.Series(False, index=learned.index))
    )
    learned["practical_gain_passed"] = _coerce_bool_series(
        learned.get("practical_gain_passed", pd.Series(False, index=learned.index))
    )
    learned = learned.sort_values(
        ["pareto_passed", "practical_gain_passed", "winner_resolution", "winner_model_label"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    return learned.iloc[0]


def _build_rollout_registry_snapshot(output_root: Path) -> pd.DataFrame:
    """Scan completed Stage-7 runs into a registry of learned-candidate outcomes."""
    columns = [
        "run_id",
        "generated_at_utc",
        "mode",
        "strategy",
        "resolution",
        "feature_set",
        "model_label",
        "learned_candidate_label",
        "learned_target_mode",
        "horizon_minutes",
        "origin_policy",
        "origins_per_run",
        "selection_source",
        "selection_policy",
        "selection_run_id",
        "selection_run_stage",
        "selection_target",
        "selection_context_path",
        "metrics_path",
        "selection_summary_path",
        "learned_endpoint_mae",
        "learned_endpoint_mae_pct",
        "learned_path_mae",
        "learned_path_mae_pct",
        "learned_phase_mean_mae",
        "learned_phase_mean_mae_pct",
        "learned_next_lock_mae",
        "learned_next_lock_mae_pct",
        "learned_profile_shape_mae",
        "learned_profile_shape_mae_pct",
        "learned_energy_mae",
        "learned_energy_mae_pct",
        "learned_mean_coverage",
        "learned_origin_n",
        "persistence_endpoint_mae",
        "persistence_endpoint_mae_pct",
        "persistence_path_mae",
        "persistence_path_mae_pct",
        "persistence_phase_mean_mae",
        "persistence_phase_mean_mae_pct",
        "persistence_next_lock_mae",
        "persistence_next_lock_mae_pct",
        "persistence_profile_shape_mae",
        "persistence_profile_shape_mae_pct",
        "persistence_energy_mae",
        "persistence_energy_mae_pct",
        "best_baseline_endpoint_label",
        "best_baseline_endpoint_mae",
        "best_baseline_endpoint_mae_pct",
        "best_baseline_path_label",
        "best_baseline_path_mae",
        "best_baseline_path_mae_pct",
        "best_baseline_phase_label",
        "best_baseline_phase_mae",
        "best_baseline_phase_mae_pct",
        "best_baseline_next_lock_label",
        "best_baseline_next_lock_mae",
        "best_baseline_next_lock_mae_pct",
        "best_baseline_profile_shape_label",
        "best_baseline_profile_shape_mae",
        "best_baseline_profile_shape_mae_pct",
        "overall_endpoint_winner",
        "overall_path_winner",
        "overall_phase_winner",
        "overall_next_lock_winner",
        "overall_profile_shape_winner",
        "beats_persistence_endpoint",
        "beats_persistence_path",
        "beats_persistence_phase",
        "beats_persistence_next_lock",
        "beats_persistence_profile_shape",
        "beats_best_baseline_endpoint",
        "beats_best_baseline_path",
        "beats_best_baseline_phase",
        "beats_best_baseline_next_lock",
        "beats_best_baseline_profile_shape",
    ]
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return pd.DataFrame(columns=columns)
    for run_dir in sorted(output_root.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir() or run_dir.name.startswith("latest"):
            continue
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "recursive_rollout_metrics.csv"
        context_path = run_dir / "selection_context.json"
        summary_path = run_dir / "rollout_selection_summary.csv"
        if not manifest_path.exists() or not metrics_path.exists() or not context_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = pd.read_csv(metrics_path)
            selection_context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
            continue
        if "phase_mean_mae" not in metrics.columns:
            metrics["phase_mean_mae"] = metrics.get("path_mae", np.nan)
        if "phase_mean_mae_pct" not in metrics.columns:
            metrics["phase_mean_mae_pct"] = metrics.get("path_mae_pct", np.nan)
        if "next_lock_mae" not in metrics.columns:
            metrics["next_lock_mae"] = np.where(
                pd.to_numeric(manifest.get("horizon_minutes", selection_context.get("requested_horizon_minutes")), errors="coerce") <= 15,
                metrics.get("path_mae", np.nan),
                np.nan,
            )
        if "next_lock_mae_pct" not in metrics.columns:
            metrics["next_lock_mae_pct"] = np.where(
                pd.to_numeric(manifest.get("horizon_minutes", selection_context.get("requested_horizon_minutes")), errors="coerce") <= 15,
                metrics.get("path_mae_pct", np.nan),
                np.nan,
            )
        if "profile_shape_mae" not in metrics.columns:
            metrics["profile_shape_mae"] = np.nan
        if "profile_shape_mae_pct" not in metrics.columns:
            metrics["profile_shape_mae_pct"] = np.nan
        if "energy_mae" not in metrics.columns:
            metrics["energy_mae"] = np.nan
        if "energy_mae_pct" not in metrics.columns:
            metrics["energy_mae_pct"] = np.nan
        selection_summary = _read_csv_if_present(summary_path)
        if metrics.empty:
            continue
        model_label = str(selection_context.get("model_label", "")).strip()
        learned = metrics.loc[
            metrics.get("candidate_type", pd.Series(index=metrics.index, dtype="string"))
            .astype("string")
            .fillna("baseline")
            .eq("learned")
        ].copy()
        if learned.empty and model_label:
            learned = metrics.loc[
                metrics["candidate_label"].astype("string").str.startswith(model_label)
            ].copy()
        if learned.empty:
            continue
        selection_target = str(selection_context.get("selection_target", "path_mae"))
        if selection_target == "path_mae":
            learned_metric = "path_mae"
            learned_secondary = "endpoint_mae"
        elif selection_target == "endpoint_mae":
            learned_metric = "endpoint_mae"
            learned_secondary = "path_mae"
        elif selection_target == "phase_mean_mae":
            learned_metric = "phase_mean_mae"
            learned_secondary = "path_mae"
        elif selection_target == "next_lock_mae":
            learned_metric = "next_lock_mae"
            learned_secondary = "path_mae"
        elif selection_target == "profile_shape_mae":
            learned_metric = "profile_shape_mae"
            learned_secondary = "path_mae"
        else:
            raise ValueError(f"Unsupported rollout selection target: {selection_target}")
        learned_row = learned.sort_values(
            [learned_metric, learned_secondary, "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        baselines = metrics.loc[~metrics["candidate_label"].astype("string").eq(model_label)].copy()
        if "candidate_type" in metrics.columns:
            baselines = metrics.loc[
                metrics["candidate_type"].astype("string").fillna("baseline").eq("baseline")
            ].copy()
        persistence = baselines.loc[baselines["candidate_label"].astype("string").eq("persistence")].copy()
        best_baseline_endpoint = pd.Series(dtype=object)
        best_baseline_path = pd.Series(dtype=object)
        best_baseline_phase = pd.Series(dtype=object)
        best_baseline_next_lock = pd.Series(dtype=object)
        best_baseline_profile_shape = pd.Series(dtype=object)
        if not baselines.empty:
            best_baseline_endpoint = baselines.sort_values(
                ["endpoint_mae", "path_mae", "candidate_label"],
                ascending=[True, True, True],
                kind="stable",
            ).iloc[0]
            best_baseline_path = baselines.sort_values(
                ["path_mae", "endpoint_mae", "candidate_label"],
                ascending=[True, True, True],
                kind="stable",
            ).iloc[0]
            best_baseline_phase = baselines.sort_values(
                ["phase_mean_mae", "path_mae", "candidate_label"],
                ascending=[True, True, True],
                kind="stable",
            ).iloc[0]
            best_baseline_next_lock = baselines.sort_values(
                ["next_lock_mae", "path_mae", "candidate_label"],
                ascending=[True, True, True],
                kind="stable",
            ).iloc[0]
            best_baseline_profile_shape = baselines.sort_values(
                ["profile_shape_mae", "path_mae", "candidate_label"],
                ascending=[True, True, True],
                kind="stable",
            ).iloc[0]

        if "selection_target" in selection_summary.columns:
            selection_targets = selection_summary["selection_target"].astype("string")
            endpoint_summary = selection_summary.loc[selection_targets.eq("endpoint_mae")].copy()
            path_summary = selection_summary.loc[selection_targets.eq("path_mae")].copy()
            phase_summary = selection_summary.loc[selection_targets.eq("phase_mean_mae")].copy()
            next_lock_summary = selection_summary.loc[selection_targets.eq("next_lock_mae")].copy()
            profile_shape_summary = selection_summary.loc[selection_targets.eq("profile_shape_mae")].copy()
        else:
            endpoint_summary = pd.DataFrame()
            path_summary = pd.DataFrame()
            phase_summary = pd.DataFrame()
            next_lock_summary = pd.DataFrame()
            profile_shape_summary = pd.DataFrame()
        rows.append(
            {
                "run_id": run_dir.name,
                "generated_at_utc": manifest.get("generated_at_utc"),
                "mode": manifest.get("mode"),
                "strategy": manifest.get("strategy"),
                "resolution": selection_context.get("resolution"),
                "feature_set": selection_context.get("feature_set"),
                "model_label": model_label,
                "learned_candidate_label": learned_row.get("candidate_label"),
                "learned_target_mode": learned_row.get("target_mode"),
                "horizon_minutes": manifest.get("horizon_minutes", selection_context.get("requested_horizon_minutes")),
                "origin_policy": manifest.get(
                    "origin_policy", selection_context.get("requested_origin_policy")
                ),
                "origins_per_run": manifest.get("origins_per_run"),
                "selection_source": selection_context.get("selection_source"),
                "selection_policy": selection_context.get("selection_policy"),
                "selection_run_id": selection_context.get("selection_run_id"),
                "selection_run_stage": selection_context.get("selection_run_stage"),
                "selection_target": selection_context.get("selection_target"),
                "selection_context_path": _relative_artifact_path(context_path),
                "metrics_path": _relative_artifact_path(metrics_path),
                "selection_summary_path": _relative_artifact_path(summary_path),
                "learned_endpoint_mae": learned_row.get("endpoint_mae"),
                "learned_endpoint_mae_pct": learned_row.get("endpoint_mae_pct"),
                "learned_path_mae": learned_row.get("path_mae"),
                "learned_path_mae_pct": learned_row.get("path_mae_pct"),
                "learned_phase_mean_mae": learned_row.get("phase_mean_mae"),
                "learned_phase_mean_mae_pct": learned_row.get("phase_mean_mae_pct"),
                "learned_next_lock_mae": learned_row.get("next_lock_mae"),
                "learned_next_lock_mae_pct": learned_row.get("next_lock_mae_pct"),
                "learned_profile_shape_mae": learned_row.get("profile_shape_mae"),
                "learned_profile_shape_mae_pct": learned_row.get("profile_shape_mae_pct"),
                "learned_energy_mae": learned_row.get("energy_mae"),
                "learned_energy_mae_pct": learned_row.get("energy_mae_pct"),
                "learned_mean_coverage": learned_row.get("mean_coverage"),
                "learned_origin_n": learned_row.get("origin_n"),
                "persistence_endpoint_mae": (
                    persistence.iloc[0]["endpoint_mae"] if not persistence.empty else np.nan
                ),
                "persistence_endpoint_mae_pct": (
                    persistence.iloc[0].get("endpoint_mae_pct", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_path_mae": persistence.iloc[0]["path_mae"] if not persistence.empty else np.nan,
                "persistence_path_mae_pct": (
                    persistence.iloc[0].get("path_mae_pct", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_phase_mean_mae": (
                    persistence.iloc[0].get("phase_mean_mae", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_phase_mean_mae_pct": (
                    persistence.iloc[0].get("phase_mean_mae_pct", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_next_lock_mae": (
                    persistence.iloc[0].get("next_lock_mae", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_next_lock_mae_pct": (
                    persistence.iloc[0].get("next_lock_mae_pct", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_profile_shape_mae": (
                    persistence.iloc[0].get("profile_shape_mae", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_profile_shape_mae_pct": (
                    persistence.iloc[0].get("profile_shape_mae_pct", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_energy_mae": (
                    persistence.iloc[0].get("energy_mae", np.nan) if not persistence.empty else np.nan
                ),
                "persistence_energy_mae_pct": (
                    persistence.iloc[0].get("energy_mae_pct", np.nan) if not persistence.empty else np.nan
                ),
                "best_baseline_endpoint_label": (
                    best_baseline_endpoint.get("candidate_label") if not best_baseline_endpoint.empty else pd.NA
                ),
                "best_baseline_endpoint_mae": (
                    best_baseline_endpoint.get("endpoint_mae") if not best_baseline_endpoint.empty else np.nan
                ),
                "best_baseline_endpoint_mae_pct": (
                    best_baseline_endpoint.get("endpoint_mae_pct") if not best_baseline_endpoint.empty else np.nan
                ),
                "best_baseline_path_label": (
                    best_baseline_path.get("candidate_label") if not best_baseline_path.empty else pd.NA
                ),
                "best_baseline_path_mae": (
                    best_baseline_path.get("path_mae") if not best_baseline_path.empty else np.nan
                ),
                "best_baseline_path_mae_pct": (
                    best_baseline_path.get("path_mae_pct") if not best_baseline_path.empty else np.nan
                ),
                "best_baseline_phase_label": (
                    best_baseline_phase.get("candidate_label") if not best_baseline_phase.empty else pd.NA
                ),
                "best_baseline_phase_mae": (
                    best_baseline_phase.get("phase_mean_mae") if not best_baseline_phase.empty else np.nan
                ),
                "best_baseline_phase_mae_pct": (
                    best_baseline_phase.get("phase_mean_mae_pct") if not best_baseline_phase.empty else np.nan
                ),
                "best_baseline_next_lock_label": (
                    best_baseline_next_lock.get("candidate_label") if not best_baseline_next_lock.empty else pd.NA
                ),
                "best_baseline_next_lock_mae": (
                    best_baseline_next_lock.get("next_lock_mae") if not best_baseline_next_lock.empty else np.nan
                ),
                "best_baseline_next_lock_mae_pct": (
                    best_baseline_next_lock.get("next_lock_mae_pct") if not best_baseline_next_lock.empty else np.nan
                ),
                "best_baseline_profile_shape_label": (
                    best_baseline_profile_shape.get("candidate_label")
                    if not best_baseline_profile_shape.empty
                    else pd.NA
                ),
                "best_baseline_profile_shape_mae": (
                    best_baseline_profile_shape.get("profile_shape_mae")
                    if not best_baseline_profile_shape.empty
                    else np.nan
                ),
                "best_baseline_profile_shape_mae_pct": (
                    best_baseline_profile_shape.get("profile_shape_mae_pct")
                    if not best_baseline_profile_shape.empty
                    else np.nan
                ),
                "overall_endpoint_winner": (
                    endpoint_summary.iloc[0]["winner_candidate_label"] if not endpoint_summary.empty else pd.NA
                ),
                "overall_path_winner": (
                    path_summary.iloc[0]["winner_candidate_label"] if not path_summary.empty else pd.NA
                ),
                "overall_phase_winner": (
                    phase_summary.iloc[0]["winner_candidate_label"] if not phase_summary.empty else pd.NA
                ),
                "overall_next_lock_winner": (
                    next_lock_summary.iloc[0]["winner_candidate_label"] if not next_lock_summary.empty else pd.NA
                ),
                "overall_profile_shape_winner": (
                    profile_shape_summary.iloc[0]["winner_candidate_label"]
                    if not profile_shape_summary.empty
                    else pd.NA
                ),
                "beats_persistence_endpoint": (
                    float(learned_row["endpoint_mae"]) < float(persistence.iloc[0]["endpoint_mae"])
                    if not persistence.empty
                    else False
                ),
                "beats_persistence_path": (
                    float(learned_row["path_mae"]) < float(persistence.iloc[0]["path_mae"])
                    if not persistence.empty
                    else False
                ),
                "beats_persistence_phase": (
                    float(learned_row["phase_mean_mae"]) < float(persistence.iloc[0]["phase_mean_mae"])
                    if not persistence.empty
                    else False
                ),
                "beats_persistence_next_lock": (
                    float(learned_row["next_lock_mae"]) < float(persistence.iloc[0]["next_lock_mae"])
                    if not persistence.empty
                    else False
                ),
                "beats_persistence_profile_shape": (
                    float(learned_row["profile_shape_mae"]) < float(persistence.iloc[0]["profile_shape_mae"])
                    if not persistence.empty
                    else False
                ),
                "beats_best_baseline_endpoint": (
                    float(learned_row["endpoint_mae"]) < float(best_baseline_endpoint["endpoint_mae"])
                    if not best_baseline_endpoint.empty
                    else False
                ),
                "beats_best_baseline_path": (
                    float(learned_row["path_mae"]) < float(best_baseline_path["path_mae"])
                    if not best_baseline_path.empty
                    else False
                ),
                "beats_best_baseline_phase": (
                    float(learned_row["phase_mean_mae"]) < float(best_baseline_phase["phase_mean_mae"])
                    if not best_baseline_phase.empty
                    else False
                ),
                "beats_best_baseline_next_lock": (
                    float(learned_row["next_lock_mae"]) < float(best_baseline_next_lock["next_lock_mae"])
                    if not best_baseline_next_lock.empty
                    else False
                ),
                "beats_best_baseline_profile_shape": (
                    float(learned_row["profile_shape_mae"])
                    < float(best_baseline_profile_shape["profile_shape_mae"])
                    if not best_baseline_profile_shape.empty
                    else False
                ),
            }
        )
    registry = pd.DataFrame(rows, columns=columns)
    if registry.empty:
        return registry
    registry["generated_at_utc"] = pd.to_datetime(registry["generated_at_utc"], errors="coerce", utc=True)
    for column in (
        "horizon_minutes",
        "origins_per_run",
        "learned_origin_n",
    ):
        registry[column] = pd.to_numeric(registry[column], errors="coerce")
    for column in (
        "learned_endpoint_mae",
        "learned_endpoint_mae_pct",
        "learned_path_mae",
        "learned_path_mae_pct",
        "learned_phase_mean_mae",
        "learned_phase_mean_mae_pct",
        "learned_next_lock_mae",
        "learned_next_lock_mae_pct",
        "learned_profile_shape_mae",
        "learned_profile_shape_mae_pct",
        "learned_energy_mae",
        "learned_energy_mae_pct",
        "learned_mean_coverage",
        "persistence_endpoint_mae",
        "persistence_endpoint_mae_pct",
        "persistence_path_mae",
        "persistence_path_mae_pct",
        "persistence_phase_mean_mae",
        "persistence_phase_mean_mae_pct",
        "persistence_next_lock_mae",
        "persistence_next_lock_mae_pct",
        "persistence_profile_shape_mae",
        "persistence_profile_shape_mae_pct",
        "persistence_energy_mae",
        "persistence_energy_mae_pct",
        "best_baseline_endpoint_mae",
        "best_baseline_endpoint_mae_pct",
        "best_baseline_path_mae",
        "best_baseline_path_mae_pct",
        "best_baseline_phase_mae",
        "best_baseline_phase_mae_pct",
        "best_baseline_next_lock_mae",
        "best_baseline_next_lock_mae_pct",
        "best_baseline_profile_shape_mae",
        "best_baseline_profile_shape_mae_pct",
    ):
        registry[column] = pd.to_numeric(registry[column], errors="coerce")
    for column in (
        "beats_persistence_endpoint",
        "beats_persistence_path",
        "beats_persistence_phase",
        "beats_persistence_next_lock",
        "beats_persistence_profile_shape",
        "beats_best_baseline_endpoint",
        "beats_best_baseline_path",
        "beats_best_baseline_phase",
        "beats_best_baseline_next_lock",
        "beats_best_baseline_profile_shape",
    ):
        registry[column] = _coerce_bool_series(registry[column])
    registry["origin_policy"] = registry["origin_policy"].astype("string").fillna("")
    registry = registry.sort_values(
        ["horizon_minutes", "generated_at_utc", "run_id"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return registry


def _select_rollout_registry_candidate(
    *,
    output_root: Path,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
) -> pd.Series | None:
    """Pick the best prior Stage-7 run comparable to the requested rollout objective."""
    registry = _build_rollout_registry_snapshot(output_root)
    if registry.empty:
        return None
    registry = registry.loc[registry["horizon_minutes"].eq(int(requested_horizon_minutes))].copy()
    if registry.empty:
        return None
    fields = _selection_metric_fields(str(selection_target))
    metric_column = fields["metric"]
    secondary_metric = fields["secondary"]
    beat_baseline_column = fields["beat_baseline"]
    beat_persistence_column = fields["beat_persistence"]
    registry["origin_policy_match"] = registry["origin_policy"].astype("string").eq(str(requested_origin_policy))
    registry["selection_target_match"] = registry["selection_target"].astype("string").eq(str(selection_target))
    registry = registry.sort_values(
        [
            "origin_policy_match",
            beat_baseline_column,
            beat_persistence_column,
            "selection_target_match",
            "learned_origin_n",
            "learned_mean_coverage",
            metric_column,
            secondary_metric,
            "generated_at_utc",
            "run_id",
        ],
        ascending=[False, False, False, False, False, False, True, True, False, False],
        kind="stable",
    )
    return registry.iloc[0]


def _read_challenger_sweep_registry(output_root: Path) -> pd.DataFrame:
    """Load and backfill the Stage-7 challenger sweep registry."""
    registry_path = Path(output_root).resolve() / "challenger_sweep_registry.csv"
    registry = _read_csv_if_present(registry_path)
    if registry.empty:
        return registry
    if "origin_selection_scope" not in registry.columns:
        registry["origin_selection_scope"] = ""
    if "shared_origin_count" not in registry.columns:
        registry["shared_origin_count"] = pd.NA
    if "recommended_source_type" not in registry.columns:
        registry["recommended_source_type"] = ""
    if "sweep_path" not in registry.columns:
        registry["sweep_path"] = ""
    if "recommended_candidate_path" not in registry.columns:
        registry["recommended_candidate_path"] = ""

    for idx, row in registry.iterrows():
        sweep_dir = _resolve_artifact_path(row.get("sweep_path"))
        manifest_path = sweep_dir / "run_manifest.json" if sweep_dir is not None else None
        origin_scope_value = row.get("origin_selection_scope")
        if (
            pd.isna(origin_scope_value)
            or not str(origin_scope_value).strip()
            or str(origin_scope_value).strip().lower() == "nan"
            or pd.isna(row.get("shared_origin_count"))
        ) and manifest_path is not None and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if (
                pd.isna(origin_scope_value)
                or not str(origin_scope_value).strip()
                or str(origin_scope_value).strip().lower() == "nan"
            ):
                registry.at[idx, "origin_selection_scope"] = str(manifest.get("origin_selection_scope", ""))
            if pd.isna(row.get("shared_origin_count")):
                registry.at[idx, "shared_origin_count"] = manifest.get("shared_origin_count")
        recommended_candidate_path = _resolve_artifact_path(row.get("recommended_candidate_path"))
        source_type_value = row.get("recommended_source_type")
        if (
            (
                pd.isna(source_type_value)
                or not str(source_type_value).strip()
                or str(source_type_value).strip().lower() == "nan"
            )
            and recommended_candidate_path is not None
            and recommended_candidate_path.exists()
        ):
            try:
                payload = json.loads(recommended_candidate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            registry.at[idx, "recommended_source_type"] = str(payload.get("recommended_source_type", ""))
    for column in (
        "requested_horizon_minutes",
        "shared_origin_count",
        "origin_n",
    ):
        if column in registry.columns:
            registry[column] = pd.to_numeric(registry[column], errors="coerce")
    for column in (
        "recommended_metric_value",
        "recommended_metric_pct",
        "endpoint_mae",
        "endpoint_mae_pct",
        "path_mae",
        "path_mae_pct",
        "phase_mean_mae",
        "phase_mean_mae_pct",
        "next_lock_mae",
        "next_lock_mae_pct",
        "profile_shape_mae",
        "profile_shape_mae_pct",
        "energy_mae",
        "energy_mae_pct",
        "mean_coverage",
        "persistence_endpoint_mae",
        "persistence_endpoint_mae_pct",
        "persistence_path_mae",
        "persistence_path_mae_pct",
        "persistence_phase_mean_mae",
        "persistence_phase_mean_mae_pct",
        "persistence_next_lock_mae",
        "persistence_next_lock_mae_pct",
        "persistence_profile_shape_mae",
        "persistence_profile_shape_mae_pct",
        "best_baseline_endpoint_mae",
        "best_baseline_endpoint_mae_pct",
        "best_baseline_path_mae",
        "best_baseline_path_mae_pct",
        "best_baseline_phase_mae",
        "best_baseline_phase_mae_pct",
        "best_baseline_next_lock_mae",
        "best_baseline_next_lock_mae_pct",
        "best_baseline_profile_shape_mae",
        "best_baseline_profile_shape_mae_pct",
    ):
        if column in registry.columns:
            registry[column] = pd.to_numeric(registry[column], errors="coerce")
    for column in (
        "beats_persistence_endpoint",
        "beats_persistence_path",
        "beats_persistence_phase",
        "beats_persistence_next_lock",
        "beats_persistence_profile_shape",
        "beats_best_baseline_endpoint",
        "beats_best_baseline_path",
        "beats_best_baseline_phase",
        "beats_best_baseline_next_lock",
        "beats_best_baseline_profile_shape",
    ):
        if column in registry.columns:
            registry[column] = _coerce_bool_series(registry[column])
    if "generated_at_utc" in registry.columns:
        registry["generated_at_utc"] = pd.to_datetime(registry["generated_at_utc"], errors="coerce", utc=True)
    return registry


def _select_challenger_sweep_registry_candidate(
    *,
    output_root: Path,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
) -> pd.Series | None:
    """Select the strongest comparable challenger-sweep recommendation for reuse."""
    registry = _read_challenger_sweep_registry(output_root)
    if registry.empty:
        return None
    registry = registry.loc[
        registry["requested_horizon_minutes"].eq(int(requested_horizon_minutes))
        & registry["selection_target"].astype("string").eq(str(selection_target))
    ].copy()
    if registry.empty:
        return None
    fields = _selection_metric_fields(str(selection_target))
    secondary_metric = fields["secondary"].replace("learned_", "")
    if secondary_metric not in registry.columns:
        registry[secondary_metric] = float("nan")
    if "origin_selection_scope" not in registry.columns:
        registry["origin_selection_scope"] = ""
    if "shared_origin_count" not in registry.columns:
        registry["shared_origin_count"] = registry.get("origin_n", 0)
    if "recommended_metric_value" not in registry.columns:
        registry["recommended_metric_value"] = registry.get(
            fields["metric"].replace("learned_", ""),
            float("nan"),
        )
    registry["origin_policy_match"] = registry["recommended_origin_policy"].astype("string").eq(
        str(requested_origin_policy)
    )
    registry["shared_origin_scope_match"] = registry["origin_selection_scope"].astype("string").eq(
        "shared_timestamp_intersection"
    )
    registry = registry.sort_values(
        [
            "origin_policy_match",
            "shared_origin_scope_match",
            fields["beat_baseline"],
            fields["beat_persistence"],
            "shared_origin_count",
            "origin_n",
            "mean_coverage",
            "recommended_metric_value",
            secondary_metric,
            "generated_at_utc",
            "sweep_run_id",
        ],
        ascending=[False, False, False, False, False, False, False, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return registry.iloc[0]


def _resolve_selection_context(
    *,
    resolution: str | None,
    feature_set: str | None,
    model_label: str | None,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
    selection_run_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the rollout candidate from explicit overrides, registries, or config defaults."""
    has_explicit_candidate = any(value is not None for value in (resolution, feature_set, model_label))
    context: dict[str, Any] = {
        "resolution": resolution or MULTIRES_ROLLOUT["selected_resolution"],
        "feature_set": feature_set or MULTIRES_ROLLOUT["feature_set"],
        "model_label": model_label or MULTIRES_ROLLOUT["model_label"],
        "forecast_strategy": "recursive",
        "requested_horizon_minutes": int(requested_horizon_minutes),
        "requested_origin_policy": str(requested_origin_policy),
        "selection_target": str(selection_target),
        "matched_stage6_horizon_minutes": None,
        "matched_rollout_registry_horizon_minutes": None,
        "selection_source": "multires.toml",
        "selection_policy": "multires.toml",
        "selection_reason": "Using multires.toml rollout defaults.",
        "selection_run_id": None,
        "selection_run_stage": None,
        "explicit_candidate_override": has_explicit_candidate,
    }
    if has_explicit_candidate:
        context["selection_source"] = "cli_or_multires.toml"
        context["selection_policy"] = "explicit_candidate_override"
        context["selection_reason"] = (
            "Using explicit rollout candidate overrides with multires.toml defaults for unspecified fields."
        )
        context["resolution"] = canonical_resolution(context["resolution"])
        return context

    def _apply_winner(top: pd.Series, *, source: str, reason: str, run_id: str | None) -> dict[str, Any]:
        context["resolution"] = str(top.get("winner_resolution", context["resolution"]))
        context["feature_set"] = str(top.get("winner_feature_set", context["feature_set"]))
        context["model_label"] = str(top.get("winner_model_label", context["model_label"]))
        context["forecast_strategy"] = str(top.get("winner_forecast_strategy", context["forecast_strategy"]))
        context["matched_stage6_horizon_minutes"] = int(top["winner_horizon_minutes"])
        context["selection_source"] = source
        context["selection_policy"] = "stage6_exact_horizon"
        context["selection_reason"] = reason
        context["selection_run_id"] = run_id
        context["selection_run_stage"] = "006_multires"
        context["matched_rollout_registry_horizon_minutes"] = None
        return context

    def _apply_rollout_registry_winner(
        top: pd.Series,
        *,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        context["resolution"] = canonical_resolution(str(top.get("resolution", context["resolution"])))
        context["feature_set"] = str(top.get("feature_set", context["feature_set"]))
        context["model_label"] = str(top.get("model_label", context["model_label"]))
        context["forecast_strategy"] = "recursive"
        context["matched_rollout_registry_horizon_minutes"] = int(top["horizon_minutes"])
        context["selection_source"] = source
        context["selection_policy"] = "stage7_rollout_registry"
        context["selection_reason"] = reason
        context["selection_run_id"] = str(top.get("run_id")) if pd.notna(top.get("run_id")) else None
        context["selection_run_stage"] = "007_rollout"
        return context

    def _apply_challenger_sweep_winner(
        top: pd.Series,
        *,
        source: str,
        reason: str,
    ) -> dict[str, Any]:
        resolution_value = str(top.get("recommended_resolution", context["resolution"]))
        context["resolution"] = (
            canonical_resolution(resolution_value)
            if resolution_value not in {"", "mixed"}
            else resolution_value or context["resolution"]
        )
        context["feature_set"] = str(top.get("recommended_feature_set", context["feature_set"]))
        context["model_label"] = str(top.get("recommended_model_label", context["model_label"]))
        context["forecast_strategy"] = "recursive"
        context["matched_stage6_horizon_minutes"] = None
        context["matched_rollout_registry_horizon_minutes"] = int(top["requested_horizon_minutes"])
        context["selection_source"] = source
        context["selection_policy"] = "stage7_challenger_sweep_registry"
        context["selection_reason"] = reason
        context["selection_run_id"] = str(top.get("sweep_run_id")) if pd.notna(top.get("sweep_run_id")) else None
        context["selection_run_stage"] = "007_rollout_challenger_sweep"
        context["sweep_run_id"] = context["selection_run_id"]
        context["sweep_candidate_label"] = str(top.get("recommended_candidate_label", ""))
        context["sweep_recommended_run_id"] = (
            str(top.get("recommended_run_id")) if pd.notna(top.get("recommended_run_id")) else None
        )
        context["sweep_recommended_run_path"] = str(top.get("recommended_run_path", ""))
        context["sweep_recommended_origin_policy"] = str(
            top.get("recommended_origin_policy", requested_origin_policy)
        )
        context["sweep_recommended_target_mode"] = str(top.get("recommended_target_mode", ""))
        context["sweep_recommended_source_type"] = str(top.get("recommended_source_type", ""))
        recommended_candidate_path = str(top.get("recommended_candidate_path", ""))
        if recommended_candidate_path:
            context["sweep_recommended_candidate_path"] = recommended_candidate_path
        if (
            context["model_label"] == "cross_candidate_portfolio"
            or str(top.get("recommended_source_type", "")).strip() == "cross_candidate_phase_bucket_portfolio"
        ):
            context["portfolio_candidate_label"] = str(top.get("recommended_candidate_label", ""))
            recommended_candidate_file = _resolve_artifact_path(recommended_candidate_path)
            if recommended_candidate_file is not None:
                context["portfolio_policy_candidates_path"] = _relative_artifact_path(
                    recommended_candidate_file.parent / "portfolio_policy_candidates.json"
                )
        return context

    attempted_sources: list[str] = []
    if selection_run_id:
        attempted_sources.append(f"run:{selection_run_id}")
        for outputs_multires_dir in output_path_candidates(PATHS["outputs_multires_dir"]):
            run_summary = outputs_multires_dir / selection_run_id / "selection_summary.csv"
            top = (
                _select_stage6_summary_candidate(
                    _read_csv_if_present(run_summary),
                    requested_horizon_minutes=requested_horizon_minutes,
                )
                if run_summary.exists()
                else None
            )
            if top is not None:
                return _apply_winner(
                    top,
                    source=str(run_summary),
                    reason="Using the explicitly requested Stage-6 run winner at the requested rollout horizon.",
                    run_id=selection_run_id,
                )

    attempted_sources.append("challenger_sweep_registry")
    for outputs_rollout_dir in output_path_candidates(PATHS["outputs_rollout_dir"]):
        challenger_registry_path = outputs_rollout_dir / "challenger_sweep_registry.csv"
        sweep_top = _select_challenger_sweep_registry_candidate(
            output_root=outputs_rollout_dir,
            requested_horizon_minutes=requested_horizon_minutes,
            requested_origin_policy=requested_origin_policy,
            selection_target=selection_target,
        )
        if sweep_top is not None:
            metric_name = _selection_metric_fields(str(selection_target))["label"]
            return _apply_challenger_sweep_winner(
                sweep_top,
                source=str(challenger_registry_path),
                reason=(
                    "Using the Stage-7 challenger sweep registry to recover the strongest shared-origin "
                    f"rollout candidate for {selection_target} ({metric_name}) at the requested horizon."
                ),
            )

    attempted_sources.append("winner_registry")
    for outputs_multires_dir in output_path_candidates(PATHS["outputs_multires_dir"]):
        registry_path = outputs_multires_dir / "winner_registry.csv"
        if not registry_path.exists():
            continue
        registry = _read_csv_if_present(registry_path)
        if registry.empty:
            continue
        registry = registry.loc[registry["winner_type"].astype("string").eq("learned_model")].copy()
        if "winner_forecast_strategy" in registry.columns:
            registry = registry.loc[
                registry["winner_forecast_strategy"].astype("string").fillna("recursive").eq("recursive")
            ].copy()
        if not registry.empty:
            registry["winner_horizon_minutes"] = pd.to_numeric(
                registry["winner_horizon_minutes"], errors="coerce"
            )
            registry = registry.loc[registry["winner_horizon_minutes"].eq(int(requested_horizon_minutes))].copy()
        if registry.empty:
            continue
        registry["pareto_passed"] = _coerce_bool_series(
            registry.get("pareto_passed", pd.Series(False, index=registry.index))
        )
        registry["practical_gain_passed"] = _coerce_bool_series(
            registry.get("practical_gain_passed", pd.Series(False, index=registry.index))
        )
        registry["generated_at_utc"] = pd.to_datetime(
            registry.get("generated_at_utc"), errors="coerce", utc=True
        )
        registry = registry.sort_values(
            ["pareto_passed", "practical_gain_passed", "generated_at_utc", "run_id"],
            ascending=[False, False, False, False],
            kind="stable",
        )
        top = registry.iloc[0]
        return _apply_winner(
            top,
            source=str(registry_path),
            reason="Using the Stage-6 winner registry to recover the best known recursive learned winner.",
            run_id=str(top.get("run_id")) if pd.notna(top.get("run_id")) else None,
        )

    attempted_sources.append("latest")
    for outputs_multires_dir in output_path_candidates(PATHS["outputs_multires_dir"]):
        latest_summary = outputs_multires_dir / "latest" / "selection_summary.csv"
        if not latest_summary.exists():
            continue
        top = _select_stage6_summary_candidate(
            _read_csv_if_present(latest_summary),
            requested_horizon_minutes=requested_horizon_minutes,
        )
        if top is not None:
            return _apply_winner(
                top,
                source=str(latest_summary),
                reason="Using the latest Stage-6 recursive learned winner at the requested rollout horizon.",
                run_id=None,
            )

    attempted_sources.append("rollout_registry")
    for outputs_rollout_dir in output_path_candidates(PATHS["outputs_rollout_dir"]):
        rollout_registry_path = outputs_rollout_dir / "rollout_registry.csv"
        rollout_top = _select_rollout_registry_candidate(
            output_root=outputs_rollout_dir,
            requested_horizon_minutes=requested_horizon_minutes,
            requested_origin_policy=requested_origin_policy,
            selection_target=selection_target,
        )
        if rollout_top is not None:
            metric_name = _selection_metric_fields(str(selection_target))["label"]
            return _apply_rollout_registry_winner(
                rollout_top,
                source=str(rollout_registry_path),
                reason=(
                    "Using the Stage-7 rollout registry to recover the best known learned "
                    f"candidate for {selection_target} ({metric_name}) at the requested horizon."
                ),
            )

    attempted = ", ".join(attempted_sources)
    context["selection_reason"] = (
        "No Stage-6 exact-horizon recursive learned winner or Stage-7 rollout registry "
        f"candidate matched the requested rollout horizon; using multires.toml rollout defaults after checking {attempted}."
    )
    context["resolution"] = canonical_resolution(context["resolution"])
    return context


def resolve_rollout_selection_context(
    *,
    resolution: str | None,
    feature_set: str | None,
    model_label: str | None,
    requested_horizon_minutes: int,
    requested_origin_policy: str,
    selection_target: str,
    selection_run_id: str | None = None,
) -> dict[str, Any]:
    """Public wrapper for Stage-7 selection resolution used by downstream stages."""
    return _resolve_selection_context(
        resolution=resolution,
        feature_set=feature_set,
        model_label=model_label,
        requested_horizon_minutes=int(requested_horizon_minutes),
        requested_origin_policy=str(requested_origin_policy),
        selection_target=str(selection_target),
        selection_run_id=selection_run_id,
    )


def _select_rollout_origins(
    base: pd.DataFrame,
    *,
    horizon_steps: int,
    max_origins: int,
    origin_policy: str,
) -> list[int]:
    """Select rollout origin positions under the requested origin-sampling policy."""
    start_day, end_day = SPLIT_DAY_RANGES["test"]
    day_idx = base["day_idx"].to_numpy(dtype=int)
    candidate_positions = [
        idx
        for idx in range(len(base))
        if idx + horizon_steps < len(base)
        and start_day <= int(day_idx[idx]) <= end_day - 1
        and int(day_idx[idx + horizon_steps]) <= end_day
    ]
    if origin_policy == "midnight":
        times = base["timestamp"].dt.time
        candidate_positions = [
            idx
            for idx in candidate_positions
            if times.iloc[idx].hour == 0 and times.iloc[idx].minute == 0
        ]
    elif origin_policy == "billing_aligned":
        timestamps = base["timestamp"]
        candidate_positions = [
            idx
            for idx in candidate_positions
            if timestamps.iloc[idx].minute % 15 == 0 and timestamps.iloc[idx].second == 0
        ]
    elif origin_policy == "phase_balanced":
        candidate_positions = _select_phase_balanced_origins(
            base,
            candidate_positions=candidate_positions,
            max_origins=int(max_origins),
        )
    elif origin_policy != "uniform":
        raise ValueError(f"Unsupported rollout origin_policy: {origin_policy}")
    candidates = candidate_positions
    if not candidates:
        return []
    if len(candidates) <= max_origins:
        return candidates
    selection = np.linspace(0, len(candidates) - 1, num=max_origins, dtype=int)
    return [candidates[idx] for idx in selection]


def _sample_rollout_origin_timestamps(
    origin_timestamps: list[pd.Timestamp],
    *,
    max_origins: int,
    origin_policy: str,
) -> list[pd.Timestamp]:
    """Subsample already-eligible origin timestamps without changing their policy semantics."""
    if not origin_timestamps:
        return []
    ordered = sorted(pd.Timestamp(timestamp) for timestamp in origin_timestamps)
    if len(ordered) <= int(max_origins):
        return ordered
    if origin_policy == "phase_balanced":
        buckets: dict[int, list[pd.Timestamp]] = {}
        for timestamp in ordered:
            buckets.setdefault(_phase_bucket_seconds(timestamp), []).append(timestamp)
        selected_bucket_keys = _evenly_sample_positions(
            sorted(buckets),
            min(int(max_origins), len(buckets)),
        )
        max_per_bucket = max(1, math.ceil(int(max_origins) / len(selected_bucket_keys)))
        bucket_samples = {
            bucket: _evenly_sample_positions(list(range(len(buckets[bucket]))), max_per_bucket)
            for bucket in selected_bucket_keys
        }
        selected: list[pd.Timestamp] = []
        seen: set[pd.Timestamp] = set()
        round_count = max((len(samples) for samples in bucket_samples.values()), default=0)
        for round_idx in range(round_count):
            for bucket in selected_bucket_keys:
                samples = bucket_samples[bucket]
                if round_idx >= len(samples):
                    continue
                timestamp = buckets[bucket][samples[round_idx]]
                if timestamp in seen:
                    continue
                seen.add(timestamp)
                selected.append(timestamp)
                if len(selected) >= int(max_origins):
                    return selected
        leftovers = [timestamp for timestamp in ordered if timestamp not in seen]
        if leftovers and len(selected) < int(max_origins):
            leftover_positions = _evenly_sample_positions(
                list(range(len(leftovers))),
                int(max_origins) - len(selected),
            )
            selected.extend(leftovers[position] for position in leftover_positions)
        return selected[: int(max_origins)]
    positions = _evenly_sample_positions(list(range(len(ordered))), int(max_origins))
    return [ordered[position] for position in positions]


def _resolve_explicit_rollout_origins(
    base: pd.DataFrame,
    *,
    horizon_steps: int,
    origin_policy: str,
    origin_timestamps: list[pd.Timestamp],
) -> list[int]:
    """Map an explicit shared-origin plan onto one resolution's base frame.

    Explicit origins are already the caller's shared-origin contract, so they
    should only be checked for timestamp/horizon representability on the target
    frame. They must not be filtered back through the candidate's nominal
    origin-policy preset such as `midnight` or `phase_balanced`.
    """
    if not origin_timestamps:
        return []
    eligible_positions = [
        idx
        for idx in range(len(base))
        if idx + int(horizon_steps) < len(base)
    ]
    if not eligible_positions:
        return []
    eligible_lookup = {
        pd.Timestamp(base.iloc[position]["timestamp"]): int(position)
        for position in eligible_positions
    }
    selected_positions: list[int] = []
    missing: list[str] = []
    seen: set[int] = set()
    for timestamp in sorted(pd.Timestamp(value) for value in origin_timestamps):
        position = eligible_lookup.get(timestamp)
        if position is None:
            missing.append(timestamp.isoformat())
            continue
        if position in seen:
            continue
        seen.add(position)
        selected_positions.append(position)
    if missing:
        raise ValueError(
            "Explicit rollout origins are not representable for this candidate: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    return selected_positions


def _aggregate_rollout_metrics(by_origin: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by-origin rollout rows into the Stage-7 metric summary table."""
    group_columns = ["candidate_label"]
    for column in ("candidate_type", "source_model_label", "target_mode"):
        if column in by_origin.columns:
            group_columns.append(column)
    metrics = (
        by_origin.groupby(group_columns, dropna=False)
        .agg(
            endpoint_mae=("endpoint_abs_error", "mean"),
            endpoint_rmse=("endpoint_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            path_mae=("path_mae", "mean"),
            phase_mean_mae=("phase_mean_abs_error", "mean"),
            phase_mean_rmse=("phase_mean_sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            next_lock_mae=("next_lock_mae", "mean"),
            profile_shape_mae=("profile_shape_mae", "mean"),
            energy_mae=("energy_abs_error", "mean"),
            endpoint_abs_error_sum=("endpoint_abs_error", "sum"),
            endpoint_actual_abs_sum=("endpoint_actual_abs", "sum"),
            path_abs_error_sum=("path_abs_error_sum", "sum"),
            path_actual_abs_sum=("path_actual_abs_sum", "sum"),
            phase_mean_abs_error_sum=("phase_mean_abs_error", "sum"),
            phase_mean_actual_abs_sum=("phase_mean_actual_abs", "sum"),
            next_lock_abs_error_sum=("next_lock_abs_error_sum", "sum"),
            next_lock_actual_abs_sum=("next_lock_actual_abs_sum", "sum"),
            profile_shape_abs_error_sum=("profile_shape_abs_error_sum", "sum"),
            profile_shape_actual_abs_sum=("profile_shape_actual_abs_sum", "sum"),
            energy_abs_error_sum=("energy_abs_error", "sum"),
            energy_actual_abs_sum=("energy_actual_abs", "sum"),
            mean_coverage=("coverage", "mean"),
            origin_n=("origin_timestamp", "nunique"),
        )
        .reset_index()
        .sort_values("endpoint_mae")
        .reset_index(drop=True)
    )
    metrics["endpoint_mae_pct"] = (
        100.0 * metrics["endpoint_abs_error_sum"] / metrics["endpoint_actual_abs_sum"]
    )
    metrics["path_mae_pct"] = 100.0 * metrics["path_abs_error_sum"] / metrics["path_actual_abs_sum"]
    metrics["phase_mean_mae_pct"] = (
        100.0 * metrics["phase_mean_abs_error_sum"] / metrics["phase_mean_actual_abs_sum"]
    )
    metrics["next_lock_mae_pct"] = (
        100.0 * metrics["next_lock_abs_error_sum"] / metrics["next_lock_actual_abs_sum"]
    )
    metrics["profile_shape_mae_pct"] = (
        100.0 * metrics["profile_shape_abs_error_sum"] / metrics["profile_shape_actual_abs_sum"]
    )
    metrics["energy_mae_pct"] = (
        100.0 * metrics["energy_abs_error_sum"] / metrics["energy_actual_abs_sum"]
    )
    metrics.loc[metrics["endpoint_actual_abs_sum"] <= 0.0, "endpoint_mae_pct"] = float("nan")
    metrics.loc[metrics["path_actual_abs_sum"] <= 0.0, "path_mae_pct"] = float("nan")
    metrics.loc[metrics["phase_mean_actual_abs_sum"] <= 0.0, "phase_mean_mae_pct"] = float("nan")
    metrics.loc[metrics["next_lock_actual_abs_sum"] <= 0.0, "next_lock_mae_pct"] = float("nan")
    metrics.loc[metrics["profile_shape_actual_abs_sum"] <= 0.0, "profile_shape_mae_pct"] = float("nan")
    metrics.loc[metrics["energy_actual_abs_sum"] <= 0.0, "energy_mae_pct"] = float("nan")
    return metrics.drop(
        columns=[
            "endpoint_abs_error_sum",
            "endpoint_actual_abs_sum",
            "path_abs_error_sum",
            "path_actual_abs_sum",
            "phase_mean_abs_error_sum",
            "phase_mean_actual_abs_sum",
            "next_lock_abs_error_sum",
            "next_lock_actual_abs_sum",
            "profile_shape_abs_error_sum",
            "profile_shape_actual_abs_sum",
            "energy_abs_error_sum",
            "energy_actual_abs_sum",
        ]
    )


def _build_rollout_selection_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best endpoint, path, and phase-average candidates from one rollout run."""
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "selection_target",
                "winner_candidate_label",
                "winner_metric_value",
                "winner_metric_pct",
                "supporting_endpoint_mae",
                "supporting_endpoint_mae_pct",
                "supporting_path_mae",
                "supporting_path_mae_pct",
                "supporting_phase_mean_mae",
                "supporting_phase_mean_mae_pct",
                "origin_n",
                "decision_reason",
            ]
        )
    endpoint_winner = metrics.sort_values(
        ["endpoint_mae", "path_mae", "candidate_label"],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]
    path_winner = metrics.sort_values(
        ["path_mae", "endpoint_mae", "candidate_label"],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]
    phase_winner = metrics.sort_values(
        ["phase_mean_mae", "path_mae", "candidate_label"],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]
    rows = [
        {
            "selection_target": "endpoint_mae",
            "winner_candidate_label": str(endpoint_winner["candidate_label"]),
            "winner_metric_value": float(endpoint_winner["endpoint_mae"]),
            "winner_metric_pct": float(endpoint_winner["endpoint_mae_pct"]),
            "supporting_endpoint_mae": float(endpoint_winner["endpoint_mae"]),
            "supporting_endpoint_mae_pct": float(endpoint_winner["endpoint_mae_pct"]),
            "supporting_path_mae": float(endpoint_winner["path_mae"]),
            "supporting_path_mae_pct": float(endpoint_winner["path_mae_pct"]),
            "supporting_phase_mean_mae": float(endpoint_winner["phase_mean_mae"]),
            "supporting_phase_mean_mae_pct": float(endpoint_winner["phase_mean_mae_pct"]),
            "origin_n": int(endpoint_winner["origin_n"]),
            "decision_reason": "Lowest endpoint MAE across rollout candidates.",
        },
        {
            "selection_target": "path_mae",
            "winner_candidate_label": str(path_winner["candidate_label"]),
            "winner_metric_value": float(path_winner["path_mae"]),
            "winner_metric_pct": float(path_winner["path_mae_pct"]),
            "supporting_endpoint_mae": float(path_winner["endpoint_mae"]),
            "supporting_endpoint_mae_pct": float(path_winner["endpoint_mae_pct"]),
            "supporting_path_mae": float(path_winner["path_mae"]),
            "supporting_path_mae_pct": float(path_winner["path_mae_pct"]),
            "supporting_phase_mean_mae": float(path_winner["phase_mean_mae"]),
            "supporting_phase_mean_mae_pct": float(path_winner["phase_mean_mae_pct"]),
            "origin_n": int(path_winner["origin_n"]),
            "decision_reason": "Lowest path MAE across rollout candidates.",
        },
        {
            "selection_target": "phase_mean_mae",
            "winner_candidate_label": str(phase_winner["candidate_label"]),
            "winner_metric_value": float(phase_winner["phase_mean_mae"]),
            "winner_metric_pct": float(phase_winner["phase_mean_mae_pct"]),
            "supporting_endpoint_mae": float(phase_winner["endpoint_mae"]),
            "supporting_endpoint_mae_pct": float(phase_winner["endpoint_mae_pct"]),
            "supporting_path_mae": float(phase_winner["path_mae"]),
            "supporting_path_mae_pct": float(phase_winner["path_mae_pct"]),
            "supporting_phase_mean_mae": float(phase_winner["phase_mean_mae"]),
            "supporting_phase_mean_mae_pct": float(phase_winner["phase_mean_mae_pct"]),
            "origin_n": int(phase_winner["origin_n"]),
            "decision_reason": "Lowest 15-minute phase-average MAE across rollout candidates.",
        },
    ]
    if {"next_lock_mae", "next_lock_mae_pct"}.issubset(metrics.columns):
        next_lock_winner = metrics.sort_values(
            ["next_lock_mae", "path_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                "selection_target": "next_lock_mae",
                "winner_candidate_label": str(next_lock_winner["candidate_label"]),
                "winner_metric_value": float(next_lock_winner["next_lock_mae"]),
                "winner_metric_pct": float(next_lock_winner["next_lock_mae_pct"]),
                "supporting_endpoint_mae": float(next_lock_winner["endpoint_mae"]),
                "supporting_endpoint_mae_pct": float(next_lock_winner["endpoint_mae_pct"]),
                "supporting_path_mae": float(next_lock_winner["path_mae"]),
                "supporting_path_mae_pct": float(next_lock_winner["path_mae_pct"]),
                "supporting_phase_mean_mae": float(next_lock_winner["phase_mean_mae"]),
                "supporting_phase_mean_mae_pct": float(next_lock_winner["phase_mean_mae_pct"]),
                "origin_n": int(next_lock_winner["origin_n"]),
                "decision_reason": "Lowest next 15-minute MAE across rollout candidates.",
            }
        )
    if {"profile_shape_mae", "profile_shape_mae_pct"}.issubset(metrics.columns):
        profile_shape_winner = metrics.sort_values(
            ["profile_shape_mae", "path_mae", "candidate_label"],
            ascending=[True, True, True],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                "selection_target": "profile_shape_mae",
                "winner_candidate_label": str(profile_shape_winner["candidate_label"]),
                "winner_metric_value": float(profile_shape_winner["profile_shape_mae"]),
                "winner_metric_pct": float(profile_shape_winner["profile_shape_mae_pct"]),
                "supporting_endpoint_mae": float(profile_shape_winner["endpoint_mae"]),
                "supporting_endpoint_mae_pct": float(profile_shape_winner["endpoint_mae_pct"]),
                "supporting_path_mae": float(profile_shape_winner["path_mae"]),
                "supporting_path_mae_pct": float(profile_shape_winner["path_mae_pct"]),
                "supporting_phase_mean_mae": float(profile_shape_winner["phase_mean_mae"]),
                "supporting_phase_mean_mae_pct": float(profile_shape_winner["phase_mean_mae_pct"]),
                "origin_n": int(profile_shape_winner["origin_n"]),
                "decision_reason": "Lowest profile-shape MAE across rollout candidates.",
            }
        )
    return pd.DataFrame(rows)


def _write_rollout_selection_summary_md(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a readable summary of rollout endpoint/path winners."""
    lines = ["# Rollout Selection Summary", ""]
    if summary.empty:
        lines.extend(["No rollout summary rows were produced.", ""])
    else:
        for _, row in summary.iterrows():
            lines.extend(
                [
                    f"## {row['selection_target']}",
                    "",
                    f"- Winner: `{row['winner_candidate_label']}`",
                    f"- Metric value: `{float(row['winner_metric_value']):.6f}`",
                    f"- Metric percent: `{float(row['winner_metric_pct']):.3f}%`",
                    f"- Supporting endpoint MAE: `{float(row['supporting_endpoint_mae']):.6f}`",
                    f"- Supporting endpoint MAE %: `{float(row['supporting_endpoint_mae_pct']):.3f}%`",
                    f"- Supporting path MAE: `{float(row['supporting_path_mae']):.6f}`",
                    f"- Supporting path MAE %: `{float(row['supporting_path_mae_pct']):.3f}%`",
                    f"- Supporting phase-average MAE: `{float(row['supporting_phase_mean_mae']):.6f}`",
                    f"- Supporting phase-average MAE %: `{float(row['supporting_phase_mean_mae_pct']):.3f}%`",
                    f"- Origins: `{int(row['origin_n'])}`",
                    f"- Reason: {row['decision_reason']}",
                    "",
                ]
            )
    (output_dir / "rollout_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _enabled_rollout_baselines() -> list[str]:
    """Return the rollout baselines enabled for the current Stage-7 run."""
    labels = ["persistence"]
    if MULTIRES_BASELINES["include_previous_day"]:
        labels.append("previous_day")
    if MULTIRES_BASELINES["include_avg_workday"]:
        labels.append("avg_workday")
    if MULTIRES_BASELINES["include_anchored_workday"]:
        labels.append("anchored_workday")
    if MULTIRES_BASELINES["include_hybrid_workday"]:
        labels.append("hybrid_workday")
    return labels


def _resolve_rollout_feature_columns(feature_set: str, *, residual_baseline: str | None = None) -> list[str]:
    """Resolve the feature columns required by one rollout candidate family."""
    columns = list(FEATURE_SETS[feature_set])
    if residual_baseline in {"avg_workday", "anchored_workday", "hybrid_workday"}:
        for feature in (
            "avg_workday_baseline",
            "anchored_workday_baseline",
            "profile_residual_lag_1",
            "previous_day_residual",
            "profile_activity_ratio",
            "profile_active_flag",
        ):
            if feature in FEATURE_SETS.get("full", FEATURE_SETS[feature_set]) and feature not in columns:
                columns.append(feature)
        if residual_baseline == "hybrid_workday" and "lag_1" not in columns:
            columns.append("lag_1")
    elif residual_baseline == "persistence" and "lag_1" not in columns:
        columns.append("lag_1")
    return list(dict.fromkeys(columns))


def _train_rollout_residual_model(
    *,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_label: str,
    residual_baseline: str,
    resolution: str,
    horizon_minutes: int,
) -> Any:
    """Train a rollout model on residuals to the requested baseline path."""
    residual_train = train_df.copy()
    if residual_baseline == "avg_workday":
        residual_train["avg_load"] = residual_train["avg_load"] - residual_train["avg_workday_baseline"]
    elif residual_baseline == "anchored_workday":
        residual_train["avg_load"] = residual_train["avg_load"] - residual_train["anchored_workday_baseline"]
    elif residual_baseline == "hybrid_workday":
        horizon_steps = max(1, lead_steps_for_horizon(resolution, horizon_minutes))
        if horizon_steps == 1:
            persistence_weight = float(MULTIRES_HYBRID["persistence_weight_end"])
        else:
            persistence_weight = float(
                np.linspace(
                    float(MULTIRES_HYBRID["persistence_weight_start"]),
                    float(MULTIRES_HYBRID["persistence_weight_end"]),
                    num=horizon_steps,
                    dtype=float,
                )[0]
            )
        residual_train["avg_load"] = residual_train["avg_load"] - (
            persistence_weight * residual_train["lag_1"]
            + (1.0 - persistence_weight) * residual_train["anchored_workday_baseline"]
        )
    elif residual_baseline == "persistence":
        residual_train["avg_load"] = residual_train["avg_load"] - residual_train["lag_1"]
    else:
        raise ValueError(f"Unsupported residual baseline: {residual_baseline}")
    return train_model(
        residual_train,
        feature_columns,
        build_model_catalog()[model_label],
    )


def _residual_target_mode(residual_baseline: str) -> str:
    """Map a rollout residual-baseline label to its target-mode identifier."""
    if residual_baseline == "avg_workday":
        return "avg_workday_residual"
    if residual_baseline == "anchored_workday":
        return "anchored_workday_residual"
    if residual_baseline == "hybrid_workday":
        return "hybrid_workday_residual"
    if residual_baseline == "persistence":
        return "persistence_residual"
    raise ValueError(f"Unsupported residual baseline: {residual_baseline}")


def _parse_blend_end_weight(
    candidate_label: str,
    *,
    blend_family: str,
    configured_end: float,
) -> float | None:
    """Extract an explicit blend end-weight from a learned blend label when present."""
    marker = f"::{blend_family}"
    if marker not in str(candidate_label):
        return None
    suffix = str(candidate_label).split(marker, 1)[1]
    if suffix == "":
        return round(float(configured_end), 3)
    matched = _BLEND_SUFFIX_RE.fullmatch(suffix)
    if matched is None:
        return None
    return round(float(int(matched.group("weight")) / 100.0), 3)


def _historical_rollout_blend_center(
    *,
    output_root: Path,
    resolution: str,
    feature_set: str,
    model_label: str,
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    blend_family: str,
    configured_end: float,
) -> float | None:
    """Recover the historically best rollout blend end-weight from prior artifacts."""
    registry = _build_rollout_registry_snapshot(output_root)
    if registry.empty:
        return None
    fields = _selection_metric_fields(str(selection_target))
    metric_column = fields["metric"]
    secondary_metric = fields["secondary"]
    beat_baseline = fields["beat_baseline"]
    beat_persistence = fields["beat_persistence"]
    family_mask = (
        registry["learned_candidate_label"]
        .astype("string")
        .fillna("")
        .str.contains(rf"::{re.escape(blend_family)}(?:_e\d{{2,3}})?$", regex=True)
    )
    filtered = registry.loc[
        registry["horizon_minutes"].eq(int(horizon_minutes))
        & registry["resolution"].astype("string").eq(str(resolution))
        & registry["feature_set"].astype("string").eq(str(feature_set))
        & registry["model_label"].astype("string").eq(str(model_label))
        & registry["origin_policy"].astype("string").eq(str(origin_policy))
        & family_mask
    ].copy()
    if filtered.empty:
        return None
    exact_target = filtered.loc[
        filtered["selection_target"].astype("string").eq(str(selection_target))
    ].copy()
    ranked = exact_target if not exact_target.empty else filtered
    ranked = ranked.sort_values(
        [
            beat_baseline,
            beat_persistence,
            metric_column,
            secondary_metric,
            "generated_at_utc",
            "run_id",
        ],
        ascending=[False, False, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return _parse_blend_end_weight(
        str(ranked.iloc[0]["learned_candidate_label"]),
        blend_family=blend_family,
        configured_end=float(configured_end),
    )


def _rollout_blend_end_weights(
    configured_end: float,
    *,
    blend_family: str,
    output_root: Path,
    resolution: str,
    feature_set: str,
    model_label: str,
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    alternate_end: float | None = None,
) -> list[float]:
    """Return a small deterministic end-weight grid anchored to config and prior evidence."""
    configured_end = round(float(configured_end), 3)
    anchor_weights = {configured_end, 0.10}
    if alternate_end is not None:
        anchor_weights.add(round(float(alternate_end), 3))

    pivot_weight = _historical_rollout_blend_center(
        output_root=output_root,
        resolution=str(resolution),
        feature_set=str(feature_set),
        model_label=str(model_label),
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        selection_target=str(selection_target),
        blend_family=str(blend_family),
        configured_end=float(configured_end),
    )
    if pivot_weight is None:
        pivot_weight = configured_end

    weights = {weight for weight in anchor_weights if 0.0 <= weight <= 1.0}
    if bool(MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_enabled"]):
        step = float(MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_step"])
        neighbors = int(MULTIRES_ROLLOUT_LEARNED_BLENDS["refinement_neighbors"])
        for offset in range(-neighbors, neighbors + 1):
            candidate_weight = round(float(pivot_weight) + offset * step, 3)
            if 0.0 <= candidate_weight <= 1.0:
                weights.add(candidate_weight)

    max_weights = int(MULTIRES_ROLLOUT_LEARNED_BLENDS["max_weights_per_family"])
    if len(weights) <= max_weights:
        return sorted(weights)

    required = {weight for weight in anchor_weights if 0.0 <= weight <= 1.0}
    required.add(round(float(pivot_weight), 3))
    prioritized = sorted(
        weights - required,
        key=lambda weight: (
            abs(float(weight) - float(pivot_weight)),
            abs(float(weight) - float(configured_end)),
            float(weight),
        ),
    )
    selected = list(sorted(required))
    for weight in prioritized:
        if len(selected) >= max_weights:
            break
        selected.append(weight)
    return sorted(selected[:max_weights])


def _plot_rollout_paths(detail: dict[str, pd.DataFrame], actual: pd.DataFrame, output_dir: Path) -> None:
    """Plot one representative recursive rollout path against actual load."""
    output_path = output_dir / "fig_rollout_paths.png"
    plt.figure(figsize=(12, 6))
    plt.plot(actual["timestamp"], actual["avg_load"], label="actual", linewidth=2)
    for label, df in detail.items():
        plt.plot(df["timestamp"], df["y_pred"], label=label, alpha=0.85)
    plt.legend()
    plt.title("Recursive rollout paths")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _plot_error_by_origin(by_origin: pd.DataFrame, output_dir: Path) -> None:
    """Plot endpoint absolute error by origin timestamp across rollout candidates."""
    output_path = output_dir / "fig_rollout_error_by_origin.png"
    pivot = by_origin.pivot(index="origin_timestamp", columns="candidate_label", values="endpoint_abs_error")
    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Endpoint absolute error")
    plt.title("Recursive rollout endpoint error by origin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    validate_png_artifact(output_path)


def _is_cross_candidate_portfolio_selection(selection: dict[str, Any]) -> bool:
    """Return whether the selection points to a sweep-derived portfolio policy."""
    model_label = str(selection.get("model_label", "")).strip()
    candidate_label = str(selection.get("portfolio_candidate_label", "")).strip()
    source_type = str(selection.get("sweep_recommended_source_type", "")).strip()
    return (
        model_label == "cross_candidate_portfolio"
        or candidate_label.startswith("cross_candidate_portfolio::")
        or source_type == "cross_candidate_phase_bucket_portfolio"
    )


def _load_portfolio_policy_candidate(selection: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Load the JSON payload that defines a sweep-derived portfolio candidate."""
    policy_path = _resolve_artifact_path(selection.get("portfolio_policy_candidates_path"))
    if policy_path is None or not policy_path.exists():
        raise RuntimeError("Missing portfolio policy candidate artifact for cross-candidate rollout selection.")
    try:
        payloads = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Malformed portfolio policy candidate artifact: {policy_path}") from exc
    if not isinstance(payloads, list):
        raise RuntimeError(f"Portfolio policy candidate artifact must contain a list: {policy_path}")
    candidate_label = str(selection.get("portfolio_candidate_label", "")).strip()
    for payload in payloads:
        if str(payload.get("candidate_label", "")).strip() == candidate_label:
            return payload, policy_path
    raise RuntimeError(
        f"Portfolio candidate {candidate_label!r} was not found in {policy_path}."
    )


def _selected_rollout_candidate_label(selection: dict[str, Any]) -> str | None:
    """Return the exact sweep-backed learned candidate label requested for replay."""
    requested_candidate_label = str(selection.get("requested_candidate_label", "")).strip()
    if requested_candidate_label:
        return requested_candidate_label
    candidate_label = str(selection.get("sweep_candidate_label", "")).strip()
    if candidate_label:
        return candidate_label
    return None


def _requested_rollout_source_labels(
    selection: dict[str, Any],
    *,
    requested_candidate_label: str,
) -> tuple[set[str], dict[str, Any] | None]:
    """Resolve the minimum learned labels needed to replay one selected rollout candidate."""
    if "::phase_bucket_" in requested_candidate_label:
        policy_payload = _load_sweep_phase_policy_candidate(selection)
        if policy_payload is not None:
            mapping_raw = policy_payload.get("phase_bucket_mapping", {})
            if isinstance(mapping_raw, dict):
                labels = {
                    str(source_label).strip()
                    for source_label in mapping_raw.values()
                    if str(source_label).strip()
                }
                if labels:
                    return labels, policy_payload
        return {str(requested_candidate_label)}, policy_payload
    return {str(requested_candidate_label)}, None


def _rollout_candidate_scope_flags(
    *,
    required_source_labels: set[str],
    required_baselines: set[str],
    prune_to_selected_candidate: bool,
    residual_baseline: str,
) -> dict[str, Any]:
    """Resolve which rollout candidate families must be generated for the current replay scope."""
    need_raw_candidate = (not prune_to_selected_candidate) or any(
        label.endswith("::raw")
        or "::persistence_raw_blend" in label
        or "::raw_residual_blend" in label
        for label in required_source_labels
    )
    residual_candidates_needed: set[str] = set()
    if prune_to_selected_candidate:
        for label in required_source_labels:
            if "::persistence_residual" in label:
                residual_candidates_needed.add("persistence")
            if "::avg_workday_residual" in label:
                residual_candidates_needed.add("avg_workday")
            if "::anchored_workday_residual" in label:
                residual_candidates_needed.add("anchored_workday")
            if "::hybrid_workday_residual" in label:
                residual_candidates_needed.add("hybrid_workday")
            if label.endswith("::hybrid_phase_gate"):
                residual_candidates_needed.update({"anchored_workday", residual_baseline})
    need_persistence_baseline = (not prune_to_selected_candidate) or (
        "persistence" in required_baselines
        or any(
            label == "hybrid_workday"
            or "::hybrid_workday_residual" in label
            or "::persistence_" in label
            or label.endswith("::hybrid_phase_gate")
            for label in required_source_labels
        )
    )
    need_previous_day_baseline = (not prune_to_selected_candidate) or (
        "previous_day" in required_baselines or "previous_day" in required_source_labels
    )
    need_avg_workday_baseline = (not prune_to_selected_candidate) or (
        "avg_workday" in required_baselines
        or any(label == "avg_workday" or "::avg_workday_residual" in label for label in required_source_labels)
    )
    need_anchored_workday_baseline = (not prune_to_selected_candidate) or (
        "anchored_workday" in required_baselines
        or any(
            label in {"anchored_workday", "hybrid_workday"}
            or "::anchored_workday_residual" in label
            or "::hybrid_workday_residual" in label
            or label.endswith("::hybrid_phase_gate")
            for label in required_source_labels
        )
    )
    need_hybrid_workday_baseline = (not prune_to_selected_candidate) or (
        "hybrid_workday" in required_baselines
        or any(
            label == "hybrid_workday"
            or "::hybrid_workday_residual" in label
            or label.endswith("::hybrid_phase_gate")
            for label in required_source_labels
        )
    )
    return {
        "need_raw_candidate": need_raw_candidate,
        "residual_candidates_needed": residual_candidates_needed,
        "need_persistence_baseline": need_persistence_baseline,
        "need_previous_day_baseline": need_previous_day_baseline,
        "need_avg_workday_baseline": need_avg_workday_baseline,
        "need_anchored_workday_baseline": need_anchored_workday_baseline,
        "need_hybrid_workday_baseline": need_hybrid_workday_baseline,
        "include_persistence_to_raw": (
            bool(MULTIRES_ROLLOUT_LEARNED_BLENDS["include_persistence_to_raw"])
            and (
                not prune_to_selected_candidate
                or any("::persistence_raw_blend" in label for label in required_source_labels)
            )
        ),
        "include_persistence_to_residual": (
            bool(MULTIRES_ROLLOUT_LEARNED_BLENDS["include_persistence_to_residual"])
            and (
                not prune_to_selected_candidate
                or any("::persistence_residual_blend" in label for label in required_source_labels)
            )
        ),
        "include_raw_to_residual": (
            bool(MULTIRES_ROLLOUT_LEARNED_BLENDS["include_raw_to_residual"])
            and (
                not prune_to_selected_candidate
                or any("::raw_residual_blend" in label for label in required_source_labels)
            )
        ),
        "include_hybrid_phase_gate": (
            bool(MULTIRES_ROLLOUT_LEARNED_BLENDS["include_hybrid_phase_gate"])
            and (
                not prune_to_selected_candidate
                or any(label.endswith("::hybrid_phase_gate") for label in required_source_labels)
            )
        ),
    }


def _append_candidate_path_details(
    detail_rows: list[dict[str, Any]],
    *,
    origin_timestamp: pd.Timestamp,
    actual: pd.DataFrame,
    candidate_paths: dict[str, pd.DataFrame],
    candidate_meta: dict[str, dict[str, Any]],
) -> None:
    """Capture predicted vs actual path rows for one rollout origin."""
    actual_frame = (
        actual.loc[:, ["timestamp", "avg_load"]]
        .copy()
        .rename(columns={"avg_load": "actual_load"})
        .assign(
            timestamp=lambda df: pd.to_datetime(df["timestamp"], errors="coerce"),
            step_index=np.arange(len(actual), dtype=int),
        )
    )
    actual_frame = actual_frame.loc[actual_frame["timestamp"].notna()].copy()
    for candidate_label, predicted in candidate_paths.items():
        meta = candidate_meta[candidate_label]
        predicted_frame = (
            predicted.loc[:, ["timestamp", "y_pred"]]
            .copy()
            .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"], errors="coerce"))
        )
        merged = actual_frame.merge(predicted_frame, on="timestamp", how="left").sort_values(
            "timestamp",
            kind="stable",
        )
        for _, row in merged.iterrows():
            detail_rows.append(
                {
                    "origin_timestamp": pd.Timestamp(origin_timestamp).isoformat(),
                    "candidate_label": str(candidate_label),
                    "candidate_type": str(meta.get("candidate_type", "")),
                    "source_model_label": str(meta.get("source_model_label", "")),
                    "target_mode": str(meta.get("target_mode", "")),
                    "forecast_timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "step_index": int(row["step_index"]),
                    "actual_load": float(row["actual_load"]),
                    "predicted_load": float(row["y_pred"]) if pd.notna(row["y_pred"]) else float("nan"),
                }
            )


def _build_derived_policy_detail_rows(
    detail_by_origin: pd.DataFrame,
    *,
    derived_policy_metadata: list[dict[str, Any]],
    model_label: str,
) -> pd.DataFrame:
    """Project phase-bucket selector candidates onto captured source path rows."""
    if detail_by_origin.empty or not derived_policy_metadata:
        return pd.DataFrame()
    working = detail_by_origin.copy()
    working["origin_timestamp"] = pd.to_datetime(working["origin_timestamp"], errors="coerce")
    working = working.loc[working["origin_timestamp"].notna()].copy()
    if working.empty:
        return pd.DataFrame()
    derived_rows: list[dict[str, Any]] = []
    for payload in derived_policy_metadata:
        candidate_label = str(payload.get("candidate_label", "")).strip()
        target_mode = str(payload.get("target_mode", "")).strip()
        mapping_raw = payload.get("phase_bucket_mapping", {})
        if not candidate_label or not isinstance(mapping_raw, dict) or not mapping_raw:
            continue
        bucket_mapping = {int(bucket): str(source_label) for bucket, source_label in mapping_raw.items()}
        for origin_value in working["origin_timestamp"].drop_duplicates().tolist():
            origin_timestamp = pd.Timestamp(origin_value)
            source_label = bucket_mapping.get(_phase_bucket_seconds(origin_timestamp))
            if source_label is None:
                continue
            matched = working.loc[
                working["origin_timestamp"].eq(origin_timestamp)
                & working["candidate_label"].astype("string").eq(source_label)
            ].copy()
            if matched.empty:
                continue
            matched["candidate_label"] = candidate_label
            matched["candidate_type"] = "learned"
            matched["source_model_label"] = str(model_label)
            matched["target_mode"] = target_mode
            derived_rows.extend(matched.to_dict(orient="records"))
    if not derived_rows:
        return pd.DataFrame()
    return pd.DataFrame(derived_rows)


def _filter_rollout_rows_to_selected_candidate(
    frame: pd.DataFrame,
    *,
    candidate_label: str | None,
) -> pd.DataFrame:
    """Keep baselines plus the requested learned candidate when replaying sweep winners."""
    if frame.empty or not candidate_label:
        return frame
    if "candidate_label" not in frame.columns:
        return frame
    labels = frame["candidate_label"].astype("string")
    if not labels.eq(str(candidate_label)).any():
        raise RuntimeError(
            f"Requested rollout candidate {candidate_label!r} was not produced by this replay."
        )
    if "candidate_type" not in frame.columns:
        return frame.loc[labels.eq(str(candidate_label))].copy().reset_index(drop=True)
    learned_mask = frame["candidate_type"].astype("string").eq("learned")
    return frame.loc[(~learned_mask) | labels.eq(str(candidate_label))].copy().reset_index(drop=True)


def _filter_first_origin_detail(
    detail: dict[str, pd.DataFrame] | None,
    *,
    candidate_label: str | None,
    candidate_meta: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame] | None:
    """Limit first-origin figures to the selected learned candidate plus baselines."""
    if detail is None or not candidate_label:
        return detail
    filtered: dict[str, pd.DataFrame] = {}
    for label, frame in detail.items():
        meta = candidate_meta.get(label, {})
        if str(meta.get("candidate_type", "baseline")) != "learned" or str(label) == str(candidate_label):
            filtered[str(label)] = frame
    if str(candidate_label) not in filtered and str(candidate_label) in detail:
        filtered[str(candidate_label)] = detail[str(candidate_label)]
    if not filtered:
        raise RuntimeError(
            f"Requested first-origin detail candidate {candidate_label!r} was not available."
        )
    return filtered


def _load_sweep_phase_policy_candidate(selection: dict[str, Any]) -> dict[str, Any] | None:
    """Load a saved same-model phase-bucket policy candidate from a sweep-backed run."""
    candidate_label = _selected_rollout_candidate_label(selection)
    if candidate_label is None or "::phase_bucket_" not in candidate_label:
        return None
    run_path = _resolve_artifact_path(selection.get("sweep_recommended_run_path"))
    if run_path is None:
        candidate_path = _resolve_artifact_path(selection.get("sweep_recommended_candidate_path"))
        run_path = candidate_path.parent if candidate_path is not None else None
    if run_path is None:
        return None
    policy_path = run_path / "rollout_policy_candidates.json"
    if not policy_path.exists():
        return None
    try:
        payloads = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payloads, list):
        return None
    for payload in payloads:
        if str(payload.get("candidate_label", "")).strip() == str(candidate_label):
            return dict(payload)
    return None


def _materialize_external_phase_policy_candidate(
    *,
    by_origin: pd.DataFrame,
    detail_by_origin: pd.DataFrame,
    requested_candidate_label: str,
    policy_payload: dict[str, Any],
    model_label: str,
    first_origin_detail: dict[str, pd.DataFrame] | None,
    first_origin_timestamp: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame] | None]:
    """Reconstruct a measured phase-bucket selector from saved policy metadata."""
    if by_origin.empty:
        return by_origin, detail_by_origin, first_origin_detail
    mapping_raw = policy_payload.get("phase_bucket_mapping", {})
    if not isinstance(mapping_raw, dict) or not mapping_raw:
        return by_origin, detail_by_origin, first_origin_detail
    target_mode = str(policy_payload.get("target_mode", "")).strip() or "phase_bucket_policy"
    bucket_mapping = {int(bucket): str(source_label) for bucket, source_label in mapping_raw.items()}
    working = by_origin.copy()
    working["origin_timestamp"] = pd.to_datetime(working["origin_timestamp"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for origin_value in working["origin_timestamp"].dropna().drop_duplicates().tolist():
        origin_timestamp = pd.Timestamp(origin_value)
        source_label = bucket_mapping.get(_phase_bucket_seconds(origin_timestamp))
        if source_label is None:
            continue
        matched = working.loc[
            working["origin_timestamp"].eq(origin_timestamp)
            & working["candidate_label"].astype("string").eq(source_label)
        ].copy()
        if matched.empty:
            continue
        row_payload = matched.iloc[0].to_dict()
        row_payload["origin_timestamp"] = origin_timestamp.isoformat()
        row_payload["candidate_label"] = str(requested_candidate_label)
        row_payload["candidate_type"] = "learned"
        row_payload["source_model_label"] = str(model_label)
        row_payload["target_mode"] = target_mode
        row_payload["policy_selection_target"] = str(policy_payload.get("selection_target", ""))
        row_payload["policy_source_candidate"] = str(source_label)
        row_payload["policy_phase_bucket_seconds"] = int(_phase_bucket_seconds(origin_timestamp))
        rows.append(row_payload)
    if rows:
        by_origin = pd.concat([by_origin, pd.DataFrame(rows)], ignore_index=True)
    if not detail_by_origin.empty:
        detail_working = detail_by_origin.copy()
        detail_working["origin_timestamp"] = pd.to_datetime(detail_working["origin_timestamp"], errors="coerce")
        detail_rows: list[dict[str, Any]] = []
        for origin_value in detail_working["origin_timestamp"].dropna().drop_duplicates().tolist():
            origin_timestamp = pd.Timestamp(origin_value)
            source_label = bucket_mapping.get(_phase_bucket_seconds(origin_timestamp))
            if source_label is None:
                continue
            matched = detail_working.loc[
                detail_working["origin_timestamp"].eq(origin_timestamp)
                & detail_working["candidate_label"].astype("string").eq(source_label)
            ].copy()
            if matched.empty:
                continue
            matched["origin_timestamp"] = origin_timestamp.isoformat()
            matched["candidate_label"] = str(requested_candidate_label)
            matched["candidate_type"] = "learned"
            matched["source_model_label"] = str(model_label)
            matched["target_mode"] = target_mode
            detail_rows.extend(matched.to_dict(orient="records"))
        if detail_rows:
            detail_by_origin = pd.concat([detail_by_origin, pd.DataFrame(detail_rows)], ignore_index=True)
    if first_origin_detail is not None and first_origin_timestamp is not None:
        source_label = bucket_mapping.get(_phase_bucket_seconds(pd.Timestamp(first_origin_timestamp)))
        if source_label and source_label in first_origin_detail:
            first_origin_detail[str(requested_candidate_label)] = first_origin_detail[str(source_label)].copy()
    return by_origin, detail_by_origin, first_origin_detail


def _shared_origin_timestamps_for_portfolio(
    *,
    source_specs: list[dict[str, Any]],
    horizon_minutes: int,
    origins: int,
    origin_policy: str,
) -> tuple[list[pd.Timestamp], list[str]]:
    """Build the shared timestamp set required to replay a portfolio candidate fairly."""
    shared: set[pd.Timestamp] | None = None
    warnings: list[str] = []
    seen_resolutions: set[str] = set()
    for spec in source_specs:
        resolution = canonical_resolution(str(spec["resolution"]))
        if resolution in seen_resolutions:
            continue
        seen_resolutions.add(resolution)
        base = load_base_gold(resolution)
        horizon_steps = lead_steps_for_horizon(resolution, int(horizon_minutes))
        positions = _select_rollout_origins(
            base,
            horizon_steps=horizon_steps,
            max_origins=len(base),
            origin_policy=str(origin_policy),
        )
        timestamps = {
            pd.Timestamp(base.iloc[position]["timestamp"])
            for position in positions
        }
        shared = timestamps if shared is None else shared & timestamps
    if not shared:
        raise RuntimeError(
            "No shared rollout origin timestamps were available across the selected portfolio source candidates."
        )
    sampled = _sample_rollout_origin_timestamps(
        list(shared),
        max_origins=int(origins),
        origin_policy=str(origin_policy),
    )
    if len(sampled) < int(origins):
        warnings.append(f"shared_origin_count_limited:{len(sampled)}:{int(origins)}")
    return sampled, warnings


def _portfolio_source_selection(
    source_spec: dict[str, Any],
    *,
    horizon_minutes: int,
    origin_policy: str,
    selection_target: str,
    policy_path: Path,
    portfolio_candidate_label: str,
) -> dict[str, Any]:
    """Build a normal Stage-7 selection payload for one portfolio source model."""
    return {
        "resolution": canonical_resolution(str(source_spec["resolution"])),
        "feature_set": str(source_spec["feature_set"]),
        "model_label": str(source_spec["model_label"]),
        "forecast_strategy": "recursive",
        "requested_horizon_minutes": int(horizon_minutes),
        "requested_origin_policy": str(origin_policy),
        "selection_target": str(selection_target),
        "matched_stage6_horizon_minutes": None,
        "matched_rollout_registry_horizon_minutes": int(horizon_minutes),
        "selection_source": _relative_artifact_path(policy_path),
        "selection_policy": "cross_candidate_portfolio_source_replay",
        "selection_reason": (
            "Replaying a source candidate for measured cross-candidate portfolio "
            f"{portfolio_candidate_label}."
        ),
        "selection_run_id": str(source_spec.get("run_id", "")) or None,
        "selection_run_stage": "007_rollout_challenger_sweep",
        "explicit_candidate_override": True,
        "requested_candidate_label": str(source_spec.get("candidate_label", "")).strip(),
    }


def _persist_rollout_run(
    *,
    run_dir: Path,
    output_root: Path,
    selection: dict[str, Any],
    resolution: str,
    feature_set: str,
    model_label: str,
    horizon_minutes: int,
    origin_policy: str,
    origin_selection_scope: str,
    origin_timestamps_provided: bool,
    selection_target: str,
    config_hash: str,
    by_origin: pd.DataFrame,
    selected_origin_rows: pd.DataFrame,
    runtime_seconds: float,
    first_origin_detail: dict[str, pd.DataFrame] | None,
    first_origin_actual: pd.DataFrame | None,
    first_origin_timestamp: pd.Timestamp | None,
    detail_by_origin: pd.DataFrame | None = None,
    derived_policy_metadata: list[dict[str, Any]] | None = None,
    additional_artifacts: dict[str, str] | None = None,
    additional_manifest_fields: dict[str, Any] | None = None,
    manifest_warnings: list[str] | None = None,
    persist_artifacts: bool = True,
    refresh_root_registry: bool = True,
    refresh_latest_alias: bool = True,
) -> dict[str, Any]:
    """Write the full Stage-7 artifact set for a completed rollout evaluation."""
    derived_policy_metadata = list(derived_policy_metadata or [])
    additional_artifacts = dict(additional_artifacts or {})
    additional_manifest_fields = dict(additional_manifest_fields or {})
    manifest_warnings = list(manifest_warnings or [])

    metrics = _aggregate_rollout_metrics(by_origin)
    selection_summary = _build_rollout_selection_summary(metrics)

    rollout_health = pd.DataFrame(
        [
            {
                "resolution": resolution,
                "feature_set": feature_set,
                "model_label": model_label,
                "horizon_minutes": int(horizon_minutes),
                "origin_count": len(selected_origin_rows),
                "origin_policy": str(origin_policy),
                "runtime_seconds": runtime_seconds,
                "status": "pass",
                "failure_reason": "",
            }
        ]
    )

    generated_at_utc = datetime.now(UTC).isoformat()
    manifest = {
        "run_id": run_dir.name,
        "stage": "007_rollout",
        "mode": "candidate",
        "config_hash": config_hash,
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "resolution": resolution,
        "strategy": "recursive",
        "horizon_minutes": int(horizon_minutes),
        "origins_per_run": len(selected_origin_rows),
        "origin_policy": str(origin_policy),
        "origin_selection_scope": origin_selection_scope,
        "origin_timestamps_provided": bool(origin_timestamps_provided),
        "selection_target": str(selection_target),
        "selection_policy_version": "v2",
        "selection_policy": selection["selection_policy"],
        "selection_source": selection["selection_source"],
        "selection_run_id": selection.get("selection_run_id"),
        "selection_run_stage": selection.get("selection_run_stage"),
        "baseline_labels": _enabled_rollout_baselines(),
        "learned_candidate_labels": sorted(
            pd.Series(by_origin.get("candidate_label", pd.Series(dtype="string")))
            .astype("string")
            .loc[
                pd.Series(by_origin.get("candidate_type", pd.Series(dtype="string")))
                .astype("string")
                .eq("learned")
            ]
            .dropna()
            .unique()
            .tolist()
        ),
        "runtime_seconds": runtime_seconds,
        "status": "success",
        "warnings": manifest_warnings,
        "artifacts": {
            "recursive_rollout_metrics": "recursive_rollout_metrics.csv",
            "recursive_rollout_by_origin": "recursive_rollout_by_origin.csv",
            "selected_origins": "selected_origins.csv",
            "rollout_selection_summary_csv": "rollout_selection_summary.csv",
            "rollout_selection_summary_md": "rollout_selection_summary.md",
            "figure_guide_md": "figure_guide.md",
            "rollout_health": "rollout_health.csv",
            "rollout_registry": "rollout_registry.csv",
            "selection_context": "selection_context.json",
            "fig_rollout_paths": "fig_rollout_paths.png",
            "fig_rollout_error_by_origin": "fig_rollout_error_by_origin.png",
        },
        "derived_policy_candidates": [
            {
                "candidate_label": payload.get("candidate_label"),
                "selection_target": payload.get("selection_target"),
            }
            for payload in derived_policy_metadata
        ],
        "generated_at_utc": generated_at_utc,
    }
    if derived_policy_metadata:
        manifest["artifacts"]["rollout_policy_candidates"] = "rollout_policy_candidates.json"
    if detail_by_origin is not None and not detail_by_origin.empty:
        manifest["artifacts"]["recursive_rollout_detail_by_origin"] = (
            "recursive_rollout_detail_by_origin.csv"
        )
    manifest["artifacts"].update(additional_artifacts)
    manifest.update(additional_manifest_fields)
    if persist_artifacts:
        metrics.to_csv(run_dir / "recursive_rollout_metrics.csv", index=False, float_format="%.6f")
        by_origin.to_csv(run_dir / "recursive_rollout_by_origin.csv", index=False, float_format="%.6f")
        selected_origin_rows.to_csv(run_dir / "selected_origins.csv", index=False)
        if detail_by_origin is not None and not detail_by_origin.empty:
            detail_by_origin.to_csv(
                run_dir / "recursive_rollout_detail_by_origin.csv",
                index=False,
                float_format="%.6f",
            )
        selection_summary.to_csv(run_dir / "rollout_selection_summary.csv", index=False, float_format="%.6f")
        _write_rollout_selection_summary_md(selection_summary, run_dir)
        if derived_policy_metadata:
            (run_dir / "rollout_policy_candidates.json").write_text(
                json.dumps(derived_policy_metadata, indent=2),
                encoding="utf-8",
            )
        rollout_health.to_csv(run_dir / "rollout_health.csv", index=False, float_format="%.6f")
        (run_dir / "selection_context.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        if first_origin_detail is not None and first_origin_actual is not None:
            _plot_rollout_paths(first_origin_detail, first_origin_actual, run_dir)
        _plot_error_by_origin(by_origin, run_dir)
        figure_guide_entries = [
            FigureGuideEntry(
                filename="fig_rollout_error_by_origin.png",
                title="Endpoint error by origin",
                intent="Show how sensitive each rollout candidate is to the starting timestamp.",
                how_to_read="Each grouped bar compares endpoint absolute error across origin timestamps for the evaluated candidates.",
                look_for="Candidates whose error remains stable across origins rather than winning only on a few convenient start times.",
            )
        ]
        if first_origin_detail is not None and first_origin_actual is not None:
            figure_guide_entries.insert(
                0,
                FigureGuideEntry(
                    filename="fig_rollout_paths.png",
                    title="Representative rollout paths",
                    intent="Show how the first selected origin evolves over the full recursive horizon for the learned candidate and baselines.",
                    how_to_read="Compare each forecast path against the actual load line over the same timestamps.",
                    look_for="Divergence after the origin, missed peaks, and whether the learned candidate corrects or amplifies baseline drift.",
                ),
            )
        write_figure_guide(
            output_path=run_dir / "figure_guide.md",
            stage_title="Stage-7 Rollout Figures",
            stage_purpose=(
                "These figures explain not only average rollout error, but also how "
                "candidate quality changes by origin and over the recursive forecast path."
            ),
            figures=figure_guide_entries,
        )
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rollout_registry = _build_rollout_registry_snapshot(output_root)
        if refresh_root_registry:
            rollout_registry.to_csv(output_root / "rollout_registry.csv", index=False)
        rollout_registry.to_csv(run_dir / "rollout_registry.csv", index=False)
        if refresh_latest_alias:
            update_latest_alias(run_dir, output_root / "latest", enabled=bool(MULTIRES_CONFIG["write_latest"]))
    emit_quality_gate(
        "ROLLOUT HEALTH",
        True,
        details={"origins": len(selected_origin_rows), "runtime_seconds": round(runtime_seconds, 2)},
        logger_instance=logger,
    )
    logger.info("Recursive rollout artifacts written to %s", run_dir)
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "by_origin": by_origin,
        "selected_origins": selected_origin_rows,
        "selection_summary": selection_summary,
        "rollout_health": rollout_health,
        "manifest": manifest,
        "runtime_seconds": runtime_seconds,
        "selection": selection,
        "first_origin_detail": first_origin_detail,
        "first_origin_actual": first_origin_actual,
        "first_origin_timestamp": first_origin_timestamp,
        "detail_by_origin": detail_by_origin if detail_by_origin is not None else pd.DataFrame(),
    }


def _run_cross_candidate_portfolio_evaluation(
    *,
    run_dir: Path,
    output_root: Path,
    selection: dict[str, Any],
    horizon_minutes: int,
    origins: int,
    origin_policy: str,
    selection_target: str,
    origin_timestamps: list[pd.Timestamp] | None,
    config_hash: str,
    capture_path_details: bool,
    candidate_scope: str,
    persist_artifacts: bool,
    refresh_root_registry: bool,
    refresh_latest_alias: bool,
) -> dict[str, Any]:
    """Replay and persist a sweep-derived cross-candidate portfolio as a Stage-7 run."""
    policy_payload, policy_path = _load_portfolio_policy_candidate(selection)
    phase_bucket_mapping_raw = policy_payload.get("phase_bucket_mapping", {})
    if not isinstance(phase_bucket_mapping_raw, dict) or not phase_bucket_mapping_raw:
        raise RuntimeError(
            f"Portfolio candidate {selection.get('portfolio_candidate_label')} has no phase bucket mapping."
        )
    phase_bucket_mapping = {
        int(bucket): dict(payload)
        for bucket, payload in phase_bucket_mapping_raw.items()
    }
    source_specs: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str, str, str]] = set()
    for payload in phase_bucket_mapping.values():
        key = (
            canonical_resolution(str(payload["resolution"])),
            str(payload["feature_set"]),
            str(payload["model_label"]),
            str(payload["candidate_label"]),
        )
        if key in seen_sources:
            continue
        seen_sources.add(key)
        source_specs.append(
            {
                "run_id": str(payload.get("run_id", "")),
                "resolution": key[0],
                "feature_set": key[1],
                "model_label": key[2],
                "candidate_label": key[3],
            }
        )
    if origin_timestamps is not None:
        shared_origin_timestamps = sorted({pd.Timestamp(value) for value in origin_timestamps})
        shared_origin_warnings: list[str] = []
    else:
        shared_origin_timestamps, shared_origin_warnings = _shared_origin_timestamps_for_portfolio(
            source_specs=source_specs,
            horizon_minutes=int(horizon_minutes),
            origins=int(origins),
            origin_policy=str(origin_policy),
        )
    if not shared_origin_timestamps:
        raise RuntimeError("No shared rollout origins remained after portfolio replay sampling.")

    combined_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    selected_origin_rows: list[dict[str, Any]] = []
    first_origin_detail: dict[str, pd.DataFrame] | None = None
    first_origin_actual: pd.DataFrame | None = None
    first_origin_timestamp: pd.Timestamp | None = None
    started = time.perf_counter()
    with TemporaryDirectory(prefix="elf_rollout_portfolio_") as temp_dir:
        temp_root = Path(temp_dir)
        source_results: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for source_spec in source_specs:
            source_selection = _portfolio_source_selection(
                source_spec,
                horizon_minutes=int(horizon_minutes),
                origin_policy=str(origin_policy),
                selection_target=str(selection_target),
                policy_path=policy_path,
                portfolio_candidate_label=str(policy_payload.get("candidate_label", "")),
            )
            source_results[
                (
                    str(source_spec["resolution"]),
                    str(source_spec["feature_set"]),
                    str(source_spec["model_label"]),
                    str(source_spec["candidate_label"]),
                )
            ] = run_rollout_evaluation(
                output_root=temp_root,
                selection=source_selection,
                horizon_minutes=int(horizon_minutes),
                origins=len(shared_origin_timestamps),
                origin_policy=str(origin_policy),
                selection_target=str(selection_target),
                origin_timestamps=shared_origin_timestamps,
                capture_path_details=bool(capture_path_details),
                candidate_scope=str(candidate_scope),
                persist_artifacts=bool(persist_artifacts),
                refresh_root_registry=False,
                refresh_latest_alias=False,
            )

        portfolio_candidate_label = str(policy_payload.get("candidate_label", "cross_candidate_portfolio::policy"))
        portfolio_target_mode = str(policy_payload.get("target_mode", "portfolio_policy"))
        for timestamp in shared_origin_timestamps:
            bucket = _phase_bucket_seconds(pd.Timestamp(timestamp))
            selected_source = phase_bucket_mapping.get(int(bucket))
            if selected_source is None:
                raise RuntimeError(
                    f"Portfolio candidate {portfolio_candidate_label} is missing a mapping for phase bucket {bucket}."
                )
            source_key = (
                canonical_resolution(str(selected_source["resolution"])),
                str(selected_source["feature_set"]),
                str(selected_source["model_label"]),
                str(selected_source["candidate_label"]),
            )
            source_result = source_results[source_key]
            source_by_origin = source_result["by_origin"]
            origin_label = pd.Timestamp(timestamp).isoformat()
            matched_candidate = source_by_origin.loc[
                source_by_origin["origin_timestamp"].astype("string").eq(origin_label)
                & source_by_origin["candidate_label"].astype("string").eq(str(selected_source["candidate_label"]))
            ].copy()
            if matched_candidate.empty:
                raise RuntimeError(
                    "Portfolio source replay did not produce the requested candidate row for "
                    f"{origin_label}: {selected_source['candidate_label']}"
                )
            candidate_row = matched_candidate.iloc[0].to_dict()
            candidate_row["candidate_label"] = portfolio_candidate_label
            candidate_row["candidate_type"] = "learned"
            candidate_row["source_model_label"] = "cross_candidate_portfolio"
            candidate_row["target_mode"] = portfolio_target_mode
            candidate_row["policy_selection_target"] = str(policy_payload.get("selection_target", selection_target))
            candidate_row["policy_phase_bucket_seconds"] = int(bucket)
            candidate_row["policy_source_candidate"] = str(selected_source["candidate_label"])
            candidate_row["policy_source_resolution"] = str(selected_source["resolution"])
            candidate_row["policy_source_feature_set"] = str(selected_source["feature_set"])
            candidate_row["policy_source_model_label"] = str(selected_source["model_label"])
            candidate_row["policy_source_run_id"] = str(selected_source.get("run_id", ""))
            combined_rows.append(candidate_row)

            matched_baselines = source_by_origin.loc[
                source_by_origin["origin_timestamp"].astype("string").eq(origin_label)
                & source_by_origin.get(
                    "candidate_type",
                    pd.Series(index=source_by_origin.index, dtype="string"),
                )
                .astype("string")
                .fillna("baseline")
                .eq("baseline")
            ].copy()
            if not matched_baselines.empty:
                combined_rows.extend(matched_baselines.to_dict(orient="records"))
            if capture_path_details:
                source_detail = source_result.get("detail_by_origin", pd.DataFrame())
                if isinstance(source_detail, pd.DataFrame) and not source_detail.empty:
                    matched_detail = source_detail.loc[
                        source_detail["origin_timestamp"].astype("string").eq(origin_label)
                        & source_detail["candidate_label"].astype("string").eq(str(selected_source["candidate_label"]))
                    ].copy()
                    if not matched_detail.empty:
                        matched_detail["candidate_label"] = portfolio_candidate_label
                        matched_detail["candidate_type"] = "learned"
                        matched_detail["source_model_label"] = "cross_candidate_portfolio"
                        matched_detail["target_mode"] = portfolio_target_mode
                        detail_rows.extend(matched_detail.to_dict(orient="records"))
                    baseline_detail = source_detail.loc[
                        source_detail["origin_timestamp"].astype("string").eq(origin_label)
                        & source_detail.get(
                            "candidate_type",
                            pd.Series(index=source_detail.index, dtype="string"),
                        )
                        .astype("string")
                        .fillna("baseline")
                        .eq("baseline")
                    ].copy()
                    if not baseline_detail.empty:
                        detail_rows.extend(baseline_detail.to_dict(orient="records"))
            selected_origin_rows.append(
                {
                    "origin_position": pd.NA,
                    "origin_timestamp": origin_label,
                    "phase_bucket_seconds": int(bucket),
                    "policy_source_candidate": str(selected_source["candidate_label"]),
                    "policy_source_resolution": str(selected_source["resolution"]),
                    "policy_source_feature_set": str(selected_source["feature_set"]),
                    "policy_source_model_label": str(selected_source["model_label"]),
                }
            )
            if first_origin_detail is None:
                detail = source_result.get("first_origin_detail")
                actual = source_result.get("first_origin_actual")
                if detail is not None and actual is not None:
                    first_origin_detail = {
                        str(label): frame.copy()
                        for label, frame in detail.items()
                    }
                    source_label = str(selected_source["candidate_label"])
                    if source_label in first_origin_detail:
                        first_origin_detail[portfolio_candidate_label] = first_origin_detail[source_label].copy()
                    first_origin_actual = actual.copy()
                    first_origin_timestamp = pd.Timestamp(timestamp)
    runtime_seconds = time.perf_counter() - started
    combined_by_origin = pd.DataFrame(combined_rows)
    if combined_by_origin.empty:
        raise RuntimeError("Cross-candidate portfolio replay produced no rollout rows.")
    selected_origin_frame = pd.DataFrame(selected_origin_rows)
    portfolio_policy_output = run_dir / "portfolio_policy_candidate.json"
    shared_origins_output = run_dir / "shared_origins.csv"
    if persist_artifacts:
        portfolio_policy_output.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")
        selected_origin_frame.to_csv(shared_origins_output, index=False)
    detail_by_origin = (
        pd.DataFrame(detail_rows)
        .drop_duplicates(
            subset=["origin_timestamp", "candidate_label", "forecast_timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
        if capture_path_details and detail_rows
        else pd.DataFrame()
    )
    return _persist_rollout_run(
        run_dir=run_dir,
        output_root=output_root,
        selection=selection,
        resolution="mixed",
        feature_set="portfolio",
        model_label="cross_candidate_portfolio",
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        origin_selection_scope=(
            "explicit_timestamps" if origin_timestamps is not None else "shared_timestamp_intersection"
        ),
        origin_timestamps_provided=True,
        selection_target=str(selection_target),
        config_hash=config_hash,
        by_origin=combined_by_origin,
        selected_origin_rows=selected_origin_frame,
        runtime_seconds=runtime_seconds,
        first_origin_detail=first_origin_detail,
        first_origin_actual=first_origin_actual,
        first_origin_timestamp=first_origin_timestamp,
        detail_by_origin=detail_by_origin,
        additional_artifacts={
            "portfolio_policy_candidate": "portfolio_policy_candidate.json",
            "shared_origins": "shared_origins.csv",
        },
        additional_manifest_fields={
            "portfolio_policy_candidate_label": str(policy_payload.get("candidate_label", "")),
            "portfolio_source_candidates": source_specs,
        },
        manifest_warnings=shared_origin_warnings,
        persist_artifacts=persist_artifacts,
        refresh_root_registry=refresh_root_registry,
        refresh_latest_alias=refresh_latest_alias,
    )


def run_rollout_evaluation(
    *,
    output_root: Path,
    selection: dict[str, Any],
    horizon_minutes: int,
    origins: int,
    origin_policy: str,
    selection_target: str,
    origin_timestamps: list[pd.Timestamp] | None = None,
    capture_path_details: bool = False,
    candidate_scope: str = "full_family",
    persist_artifacts: bool = True,
    refresh_root_registry: bool = True,
    refresh_latest_alias: bool = True,
) -> dict[str, Any]:
    """Execute one Stage-7 rollout candidate evaluation and optionally persist artifacts."""
    resolution = str(selection["resolution"])
    feature_set = str(selection["feature_set"])
    model_label = str(selection["model_label"])

    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_config = {
        "resolution": resolution,
        "feature_set": feature_set,
        "model_label": model_label,
        "horizon_minutes": int(horizon_minutes),
        "origins": int(origins),
        "origin_policy": str(origin_policy),
        "selection_target": str(selection_target),
        "origin_selection_scope": "explicit_timestamps" if origin_timestamps is not None else "policy_sampled",
        "origin_timestamps_provided": origin_timestamps is not None,
        "selection_policy": selection["selection_policy"],
        "selection_source": selection["selection_source"],
        "selection_run_id": selection.get("selection_run_id"),
        "selection_run_stage": selection.get("selection_run_stage"),
        "candidate_scope": str(candidate_scope),
    }
    config_hash = stable_config_hash(effective_config)

    if _is_cross_candidate_portfolio_selection(selection):
        return _run_cross_candidate_portfolio_evaluation(
            run_dir=run_dir,
            output_root=output_root,
            selection=selection,
            horizon_minutes=int(horizon_minutes),
            origins=int(origins),
            origin_policy=str(origin_policy),
            selection_target=str(selection_target),
            origin_timestamps=origin_timestamps,
            config_hash=config_hash,
            capture_path_details=bool(capture_path_details),
            candidate_scope=str(candidate_scope),
            persist_artifacts=bool(persist_artifacts),
            refresh_root_registry=refresh_root_registry,
            refresh_latest_alias=refresh_latest_alias,
        )

    runtime_context = _prepare_rollout_runtime_context(
        resolution=resolution,
        feature_set=feature_set,
        model_label=model_label,
        horizon_minutes=int(horizon_minutes),
    )
    base = cast(pd.DataFrame, runtime_context["base"])
    day_lookup = cast(dict[pd.Timestamp, str], runtime_context["day_lookup"])
    horizon_steps = int(runtime_context["horizon_steps"])
    if origin_timestamps is not None:
        selected_origins = _resolve_explicit_rollout_origins(
            base,
            horizon_steps=horizon_steps,
            origin_policy=str(origin_policy),
            origin_timestamps=[pd.Timestamp(value) for value in origin_timestamps],
        )
        origin_selection_scope = "explicit_timestamps"
    else:
        selected_origins = _select_rollout_origins(
            base,
            horizon_steps=horizon_steps,
            max_origins=int(origins),
            origin_policy=str(origin_policy),
        )
        origin_selection_scope = "policy_sampled"
    if not selected_origins:
        raise RuntimeError("No rollout origins available for the requested horizon.")
    selected_origin_rows = pd.DataFrame(
        [
            {
                "origin_position": int(position),
                "origin_timestamp": pd.Timestamp(base.iloc[position]["timestamp"]).isoformat(),
            }
            for position in selected_origins
        ]
    )

    horizon_policy = cast(dict[str, Any], runtime_context["horizon_policy"])
    trained = runtime_context["trained"]
    profile = runtime_context["profile"]
    residual_baseline = str(runtime_context["residual_baseline"])
    residual_models = cast(dict[str, Any], runtime_context["residual_models"])
    requested_candidate_label = (
        None
        if _is_cross_candidate_portfolio_selection(selection)
        else _selected_rollout_candidate_label(selection)
    )
    if str(candidate_scope) not in {"full_family", "selected_plus_baselines", "selected_only"}:
        raise ValueError(f"Unsupported rollout candidate scope: {candidate_scope}")
    prune_to_selected_candidate = (
        str(candidate_scope) != "full_family" and requested_candidate_label is not None
    )
    required_source_labels: set[str] = set()
    phase_policy_payload: dict[str, Any] | None = None
    if requested_candidate_label is not None:
        required_source_labels, phase_policy_payload = _requested_rollout_source_labels(
            selection,
            requested_candidate_label=str(requested_candidate_label),
        )
    enabled_baselines = set(_enabled_rollout_baselines())
    required_baselines = set()
    if prune_to_selected_candidate and str(candidate_scope) == "selected_plus_baselines":
        required_baselines = set(enabled_baselines)
    elif prune_to_selected_candidate:
        required_baselines.update(
            label for label in required_source_labels if label in enabled_baselines
        )
    scope_flags = _rollout_candidate_scope_flags(
        required_source_labels=required_source_labels,
        required_baselines=required_baselines,
        prune_to_selected_candidate=prune_to_selected_candidate,
        residual_baseline=residual_baseline,
    )
    phase_policy_bucket_mapping = (
        {
            int(bucket): str(source_label).strip()
            for bucket, source_label in cast(dict[str, Any], phase_policy_payload.get("phase_bucket_mapping", {})).items()
            if str(source_label).strip()
        }
        if isinstance(phase_policy_payload, dict)
        and isinstance(phase_policy_payload.get("phase_bucket_mapping", {}), dict)
        else {}
    )
    by_origin_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    first_origin_detail: dict[str, pd.DataFrame] | None = None
    first_origin_actual: pd.DataFrame | None = None
    first_origin_timestamp: pd.Timestamp | None = None
    started = time.perf_counter()
    for origin_position in selected_origins:
        origin_timestamp = pd.Timestamp(base.iloc[origin_position]["timestamp"])
        history = base.iloc[: origin_position + 1].set_index("timestamp")["avg_load"].astype(float)
        actual = actual_path(base, origin_position=origin_position, horizon_steps=horizon_steps)
        raw_candidate_label = f"{model_label}::raw"
        origin_required_source_labels = set(required_source_labels)
        if (
            prune_to_selected_candidate
            and str(candidate_scope) == "selected_only"
            and phase_policy_bucket_mapping
        ):
            mapped_label = phase_policy_bucket_mapping.get(_phase_bucket_seconds(origin_timestamp))
            origin_required_source_labels = {mapped_label} if mapped_label else set()
        origin_scope_flags = (
            scope_flags
            if origin_required_source_labels == required_source_labels
            else _rollout_candidate_scope_flags(
                required_source_labels=origin_required_source_labels,
                required_baselines=required_baselines,
                prune_to_selected_candidate=prune_to_selected_candidate,
                residual_baseline=residual_baseline,
            )
        )
        candidate_paths: dict[str, pd.DataFrame] = {}
        candidate_meta: dict[str, dict[str, Any]] = {}
        if bool(origin_scope_flags["need_raw_candidate"]):
            learned_path = recursive_predict_path(
                trained=trained,
                history=history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_lookup,
                profile=profile,
            )
            candidate_paths[raw_candidate_label] = learned_path
            candidate_meta[raw_candidate_label] = {
                "candidate_type": "learned",
                "source_model_label": model_label,
                "target_mode": "raw",
            }
        if bool(origin_scope_flags["need_persistence_baseline"]):
            candidate_paths["persistence"] = persistence_path(
                history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
            )
            candidate_meta["persistence"] = {
                "candidate_type": "baseline",
                "source_model_label": "persistence",
                "target_mode": "baseline",
            }
        residual_labels: dict[str, str] = {}
        for candidate_baseline, trained_residual in residual_models.items():
            if prune_to_selected_candidate and (
                str(candidate_baseline) not in set(origin_scope_flags["residual_candidates_needed"])
            ):
                continue
            residual_target_mode = _residual_target_mode(str(candidate_baseline))
            residual_label = f"{model_label}::{residual_target_mode}"
            candidate_paths[residual_label] = recursive_predict_residual_path(
                trained=trained_residual,
                history=history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_lookup,
                residual_baseline=str(candidate_baseline),
                profile=profile,
            )
            candidate_meta[residual_label] = {
                "candidate_type": "learned",
                "source_model_label": model_label,
                "target_mode": residual_target_mode,
            }
            residual_labels[str(candidate_baseline)] = residual_label
        default_residual_label = residual_labels.get(residual_baseline)
        preferred_hybrid_gate_label = (
            residual_labels.get("anchored_workday")
            or default_residual_label
        )
        if bool(horizon_policy["allow_blend"]) and MULTIRES_ROLLOUT_LEARNED_BLENDS["enabled"]:
            if (
                bool(origin_scope_flags["include_persistence_to_raw"])
                and "persistence" in candidate_paths
                and raw_candidate_label in candidate_paths
            ):
                for end_weight in _rollout_blend_end_weights(
                    MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_end"],
                    blend_family="persistence_raw_blend",
                    output_root=output_root,
                    resolution=resolution,
                    feature_set=feature_set,
                    model_label=model_label,
                    horizon_minutes=int(horizon_minutes),
                    origin_policy=str(origin_policy),
                    selection_target=str(selection_target),
                    alternate_end=MULTIRES_HYBRID["persistence_weight_end"],
                ):
                    label_suffix = (
                        ""
                        if abs(end_weight - float(MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_end"])) < 1e-9
                        else f"_e{int(round(end_weight * 100)):02d}"
                    )
                    persistence_raw_label = f"{model_label}::persistence_raw_blend{label_suffix}"
                    candidate_paths[persistence_raw_label] = blend_candidate_paths(
                        candidate_paths["persistence"],
                        candidate_paths[raw_candidate_label],
                        primary_weight_start=MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_start"],
                        primary_weight_end=end_weight,
                        curve=MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"],
                    )
                    candidate_meta[persistence_raw_label] = {
                        "candidate_type": "learned",
                        "source_model_label": model_label,
                        "target_mode": "persistence_raw_blend",
                    }
            if (
                default_residual_label is not None
                and bool(origin_scope_flags["include_persistence_to_residual"])
                and "persistence" in candidate_paths
            ):
                for end_weight in _rollout_blend_end_weights(
                    MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_end"],
                    blend_family="persistence_residual_blend",
                    output_root=output_root,
                    resolution=resolution,
                    feature_set=feature_set,
                    model_label=model_label,
                    horizon_minutes=int(horizon_minutes),
                    origin_policy=str(origin_policy),
                    selection_target=str(selection_target),
                    alternate_end=MULTIRES_HYBRID["persistence_weight_end"],
                ):
                    label_suffix = (
                        ""
                        if abs(end_weight - float(MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_end"])) < 1e-9
                        else f"_e{int(round(end_weight * 100)):02d}"
                    )
                    persistence_residual_label = f"{model_label}::persistence_residual_blend{label_suffix}"
                    candidate_paths[persistence_residual_label] = blend_candidate_paths(
                        candidate_paths["persistence"],
                        candidate_paths[default_residual_label],
                        primary_weight_start=MULTIRES_ROLLOUT_LEARNED_BLENDS["persistence_weight_start"],
                        primary_weight_end=end_weight,
                        curve=MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"],
                    )
                    candidate_meta[persistence_residual_label] = {
                        "candidate_type": "learned",
                        "source_model_label": model_label,
                        "target_mode": "persistence_residual_blend",
                    }
            if (
                default_residual_label is not None
                and bool(origin_scope_flags["include_raw_to_residual"])
                and raw_candidate_label in candidate_paths
            ):
                for end_weight in _rollout_blend_end_weights(
                    MULTIRES_ROLLOUT_LEARNED_BLENDS["raw_weight_end"],
                    blend_family="raw_residual_blend",
                    output_root=output_root,
                    resolution=resolution,
                    feature_set=feature_set,
                    model_label=model_label,
                    horizon_minutes=int(horizon_minutes),
                    origin_policy=str(origin_policy),
                    selection_target=str(selection_target),
                ):
                    label_suffix = (
                        ""
                        if abs(end_weight - float(MULTIRES_ROLLOUT_LEARNED_BLENDS["raw_weight_end"])) < 1e-9
                        else f"_e{int(round(end_weight * 100)):02d}"
                    )
                    raw_residual_label = f"{model_label}::raw_residual_blend{label_suffix}"
                    candidate_paths[raw_residual_label] = blend_candidate_paths(
                        candidate_paths[raw_candidate_label],
                        candidate_paths[default_residual_label],
                        primary_weight_start=MULTIRES_ROLLOUT_LEARNED_BLENDS["raw_weight_start"],
                        primary_weight_end=end_weight,
                        curve=MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"],
                    )
                    candidate_meta[raw_residual_label] = {
                        "candidate_type": "learned",
                        "source_model_label": model_label,
                        "target_mode": "raw_residual_blend",
                    }
        if MULTIRES_BASELINES["include_previous_day"] and bool(origin_scope_flags["need_previous_day_baseline"]):
            candidate_paths["previous_day"] = previous_day_path(
                history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
            )
            candidate_meta["previous_day"] = {
                "candidate_type": "baseline",
                "source_model_label": "previous_day",
                "target_mode": "baseline",
            }
        if MULTIRES_BASELINES["include_avg_workday"] and bool(origin_scope_flags["need_avg_workday_baseline"]):
            candidate_paths["avg_workday"] = avg_workday_path(
                profile,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_lookup,
            )
            candidate_meta["avg_workday"] = {
                "candidate_type": "baseline",
                "source_model_label": "avg_workday",
                "target_mode": "baseline",
            }
        if (
            (MULTIRES_BASELINES["include_anchored_workday"] and bool(origin_scope_flags["need_anchored_workday_baseline"]))
            or (MULTIRES_BASELINES["include_hybrid_workday"] and bool(origin_scope_flags["need_hybrid_workday_baseline"]))
        ):
            anchored_path = anchored_workday_path(
                profile,
                history=history,
                origin_timestamp=origin_timestamp,
                horizon_steps=horizon_steps,
                resolution=resolution,
                day_class_lookup=day_lookup,
            )
            if MULTIRES_BASELINES["include_anchored_workday"] and bool(origin_scope_flags["need_anchored_workday_baseline"]):
                candidate_paths["anchored_workday"] = anchored_path
                candidate_meta["anchored_workday"] = {
                    "candidate_type": "baseline",
                    "source_model_label": "anchored_workday",
                    "target_mode": "baseline",
                }
            if (
                MULTIRES_BASELINES["include_hybrid_workday"]
                and bool(origin_scope_flags["need_hybrid_workday_baseline"])
                and "persistence" in candidate_paths
            ):
                candidate_paths["hybrid_workday"] = blend_candidate_paths(
                    candidate_paths["persistence"],
                    anchored_path,
                    primary_weight_start=MULTIRES_HYBRID["persistence_weight_start"],
                    primary_weight_end=MULTIRES_HYBRID["persistence_weight_end"],
                    curve=MULTIRES_HYBRID["curve"],
                )
                candidate_meta["hybrid_workday"] = {
                    "candidate_type": "baseline",
                    "source_model_label": "hybrid_workday",
                    "target_mode": "baseline",
                }
        if (
            preferred_hybrid_gate_label is not None
            and "hybrid_workday" in candidate_paths
            and bool(origin_scope_flags["include_hybrid_phase_gate"])
        ):
            hybrid_weight = _hybrid_phase_gate_weight(origin_timestamp)
            hybrid_phase_gate_label = f"{model_label}::hybrid_phase_gate"
            candidate_paths[hybrid_phase_gate_label] = blend_candidate_paths(
                candidate_paths["hybrid_workday"],
                candidate_paths[preferred_hybrid_gate_label],
                primary_weight_start=hybrid_weight,
                primary_weight_end=hybrid_weight,
                curve=MULTIRES_ROLLOUT_LEARNED_BLENDS["curve"],
            )
            candidate_meta[hybrid_phase_gate_label] = {
                "candidate_type": "learned",
                "source_model_label": model_label,
                "target_mode": "hybrid_phase_gate",
            }
        if prune_to_selected_candidate:
            allowed_labels = set(required_baselines) | set(origin_required_source_labels)
            candidate_paths = {
                label: path for label, path in candidate_paths.items() if label in allowed_labels
            }
            candidate_meta = {
                label: meta for label, meta in candidate_meta.items() if label in allowed_labels
            }
            if not candidate_paths:
                raise RuntimeError(
                    f"Selected rollout replay did not generate any paths for {requested_candidate_label!r}."
                )
        compared = compare_recursive_paths(actual, candidate_paths)
        for _, row in compared.iterrows():
            label = str(row["candidate_label"])
            meta = candidate_meta[label]
            by_origin_rows.append(
                {
                    "origin_timestamp": origin_timestamp.isoformat(),
                    "candidate_label": label,
                    **meta,
                    "endpoint_abs_error": float(row["endpoint_abs_error"]),
                    "endpoint_sq_error": float(row["endpoint_sq_error"]),
                    "endpoint_actual_abs": float(row["endpoint_actual_abs"]),
                    "path_mae": float(row["path_mae"]),
                    "path_rmse": float(row["path_rmse"]),
                    "path_abs_error_sum": float(row["path_abs_error_sum"]),
                    "path_actual_abs_sum": float(row["path_actual_abs_sum"]),
                    "phase_mean_abs_error": float(row["phase_mean_abs_error"]),
                    "phase_mean_sq_error": float(row["phase_mean_sq_error"]),
                    "phase_mean_actual_abs": float(row["phase_mean_actual_abs"]),
                    "next_lock_mae": float(row["next_lock_mae"]),
                    "next_lock_abs_error_sum": float(row["next_lock_abs_error_sum"]),
                    "next_lock_actual_abs_sum": float(row["next_lock_actual_abs_sum"]),
                    "profile_shape_mae": float(row["profile_shape_mae"]),
                    "profile_shape_abs_error_sum": float(row["profile_shape_abs_error_sum"]),
                    "profile_shape_actual_abs_sum": float(row["profile_shape_actual_abs_sum"]),
                    "energy_abs_error": float(row["energy_abs_error"]),
                    "energy_actual_abs": float(row["energy_actual_abs"]),
                    "coverage": float(row["coverage"]),
                    "n_eval": float(row["n_eval"]),
                }
            )
        if capture_path_details:
            _append_candidate_path_details(
                detail_rows,
                origin_timestamp=origin_timestamp,
                actual=actual,
                candidate_paths=candidate_paths,
                candidate_meta=candidate_meta,
            )
        if first_origin_detail is None:
            first_origin_detail = candidate_paths
            first_origin_actual = actual
            first_origin_timestamp = origin_timestamp

    runtime_seconds = time.perf_counter() - started
    by_origin = pd.DataFrame(by_origin_rows)
    derived_policy_metadata: list[dict[str, Any]] = []
    derived_policy_rows, derived_policy_metadata = _build_phase_bucket_policy_candidates(
        by_origin,
        model_label=model_label,
        horizon_minutes=int(horizon_minutes),
    )
    if not derived_policy_rows.empty:
        by_origin = pd.concat([by_origin, derived_policy_rows], ignore_index=True)
        if first_origin_detail is not None and first_origin_timestamp is not None:
            first_origin_bucket = _phase_bucket_seconds(pd.Timestamp(first_origin_timestamp))
            for payload in derived_policy_metadata:
                source_mapping = {
                    int(bucket): str(label)
                    for bucket, label in payload.get("phase_bucket_mapping", {}).items()
                }
                source_label = source_mapping.get(int(first_origin_bucket))
                selector_label = str(payload.get("candidate_label", ""))
                if source_label and source_label in first_origin_detail:
                    first_origin_detail[selector_label] = first_origin_detail[source_label].copy()
    detail_by_origin = pd.DataFrame(detail_rows)
    if capture_path_details and not detail_by_origin.empty and derived_policy_metadata:
        derived_detail_rows = _build_derived_policy_detail_rows(
            detail_by_origin,
            derived_policy_metadata=derived_policy_metadata,
            model_label=model_label,
        )
        if not derived_detail_rows.empty:
            detail_by_origin = pd.concat([detail_by_origin, derived_detail_rows], ignore_index=True)
    if requested_candidate_label and "::phase_bucket_" in str(requested_candidate_label):
        if phase_policy_payload is None:
            phase_policy_payload = _load_sweep_phase_policy_candidate(selection)
        if phase_policy_payload is not None:
            by_origin, detail_by_origin, first_origin_detail = _materialize_external_phase_policy_candidate(
                by_origin=by_origin,
                detail_by_origin=detail_by_origin,
                requested_candidate_label=str(requested_candidate_label),
                policy_payload=phase_policy_payload,
                model_label=model_label,
                first_origin_detail=first_origin_detail,
                first_origin_timestamp=first_origin_timestamp,
            )
    by_origin = _filter_rollout_rows_to_selected_candidate(
        by_origin,
        candidate_label=requested_candidate_label,
    )
    detail_by_origin = _filter_rollout_rows_to_selected_candidate(
        detail_by_origin,
        candidate_label=requested_candidate_label,
    )
    first_origin_detail = _filter_first_origin_detail(
        first_origin_detail,
        candidate_label=requested_candidate_label,
        candidate_meta=candidate_meta,
    )
    return _persist_rollout_run(
        run_dir=run_dir,
        output_root=output_root,
        selection=selection,
        resolution=resolution,
        feature_set=feature_set,
        model_label=model_label,
        horizon_minutes=int(horizon_minutes),
        origin_policy=str(origin_policy),
        origin_selection_scope=origin_selection_scope,
        origin_timestamps_provided=origin_timestamps is not None,
        selection_target=str(selection_target),
        config_hash=config_hash,
        by_origin=by_origin,
        selected_origin_rows=selected_origin_rows,
        runtime_seconds=runtime_seconds,
        first_origin_detail=first_origin_detail,
        first_origin_actual=first_origin_actual,
        first_origin_timestamp=first_origin_timestamp,
        detail_by_origin=detail_by_origin,
        derived_policy_metadata=derived_policy_metadata,
        persist_artifacts=bool(persist_artifacts),
        refresh_root_registry=refresh_root_registry,
        refresh_latest_alias=refresh_latest_alias,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Stage-7 recursive rollout runner."""
    parser = argparse.ArgumentParser(description="Run recursive rollout evaluation.")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--selection-run-id", default=None)
    parser.add_argument(
        "--selection-target",
        choices=[
            "endpoint_mae",
            "path_mae",
            "phase_mean_mae",
            "next_lock_mae",
            "profile_shape_mae",
            "auto",
        ],
        default=MULTIRES_ROLLOUT["selection_target"],
    )
    parser.add_argument("--horizon-minutes", type=int, default=MULTIRES_ROLLOUT["horizon_minutes"])
    parser.add_argument("--origins", type=int, default=MULTIRES_ROLLOUT["origins_per_run"])
    parser.add_argument(
        "--origin-policy",
        choices=["uniform", "midnight", "billing_aligned", "phase_balanced", "auto"],
        default=MULTIRES_ROLLOUT["origin_policy"],
    )
    parser.add_argument("--output-dir", default=str(scoped_output_path(PATHS["outputs_rollout_dir"])))
    return parser.parse_args()


def main() -> int:
    """Execute Stage-7 rollout evaluation and persist one run directory."""
    _configure_logging()
    validate_config()
    args = parse_args()
    resolved_origin_policy = resolve_rollout_origin_policy(
        int(args.horizon_minutes),
        str(args.origin_policy),
    )
    resolved_selection_target = resolve_rollout_selection_target(
        int(args.horizon_minutes),
        str(args.selection_target),
    )
    selection = _resolve_selection_context(
        resolution=args.resolution,
        feature_set=args.feature_set,
        model_label=args.model_label,
        requested_horizon_minutes=int(args.horizon_minutes),
        requested_origin_policy=resolved_origin_policy,
        selection_target=resolved_selection_target,
        selection_run_id=args.selection_run_id,
    )
    output_root = Path(args.output_dir).resolve()
    try:
        run_rollout_evaluation(
            output_root=output_root,
            selection=selection,
            horizon_minutes=int(args.horizon_minutes),
            origins=int(args.origins),
            origin_policy=resolved_origin_policy,
            selection_target=resolved_selection_target,
            refresh_root_registry=True,
            refresh_latest_alias=True,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
