"""Stage-5 model-performance workflow for short-horizon challenger selection.

This module owns the post-notebook performance surface that decides whether a
learned short-horizon candidate deserves promotion over the persistence anchor.
It is executed by `run_pipeline.py --stage performance` and produces the artifacts
later consumed by the horizon curve and forecast-control stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (  # noqa: E402
    DATASET,
    FEATURE_SETS,
    FULL_STABLE_EXCLUDED_COLUMNS,
    FULL_STABLE_LEGACY_COLUMNS,
    FULL_STABLE_LEGACY_FEATURE_SET_NAME,
    FULL_STABLE_FEATURE_SET_NAME,
    MODELING_HORIZON_POLICIES,
    MODELING_PARALLEL,
    MODELING_PERFORMANCE_BLEND_SEARCH,
    MODELING_PERFORMANCE_EVALUATION,
    MODELING_PERFORMANCE_HGB_SEARCH,
    MODELING_PERFORMANCE_RAMP,
    PATHS,
    REGIME_PROFILE_FEATURE_SET_NAME,
    RESOLUTION_ALIASES,
    RESOLUTION_TO_SUFFIX,
    SPLIT_DAY_RANGES,
    SUPPORTED_RESOLUTIONS,
    TARGET_COLUMN,
    preferred_output_path,
    resolve_horizon_policy,
    resolve_performance_quick_profile,
    scoped_output_path,
    validate_config,
)
from modeling.common import (  # noqa: E402
    FigureGuideEntry,
    ModelSpec,
    build_model_catalog,
    update_latest_alias,
    validate_png_artifact,
    write_figure_guide,
)
from modeling.metrics import compute_regression_metrics
from modeling.parallel import ParallelPlan, run_stage_jobs
from modeling.runtime import runtime_summary
from utils import optimal_acf_depth

logger = logging.getLogger(__name__)
PROJECT_ROOT = SCRIPTS_DIR.parent
STEP4_ARTIFACT_DIR = preferred_output_path(PATHS["outputs_modeling_dir"])
DEFAULT_OUTPUT_DIR = scoped_output_path(PATHS["outputs_performance_dir"])
LATEST_ALIAS_NAME = "latest"
OHE_COLUMNS = ("workday", "hour", "day_of_week", "season", "time_of_day")
RAMP_FEATURE_SET_NAME = "curated_ramp"
PROMOTION_MIN_COVERAGE = 0.95
OPERATING_POLICY_MIN_SEGMENT_ROWS = 4
RAMP_ADDITIONAL_FEATURES = (
    "rolling_mean_3",
    "rolling_std_3",
    "ramp_flag",
    "hour_x_delta_5",
)
RESIDUAL_SUPPORT_FEATURES = (
    "avg_workday_baseline",
    "profile_residual_lag_1",
    "previous_day_residual",
    "profile_activity_ratio",
    "profile_active_flag",
)


@dataclass(frozen=True)
class BlendConfig:
    """Centralized blend-shape parameters for Stage-5 causal blending."""

    window: int
    sharpness: float
    min_weight: float
    max_weight: float


@dataclass(frozen=True)
class BucketBlendConfig:
    """Fixed minute-bucket blend weights for one Stage-5 promoted candidate."""

    bucket_size_minutes: int
    cycle_minutes: int
    bucket_weights: tuple[tuple[int, float], ...]

    def weight_map(self) -> dict[int, float]:
        """Return the persisted bucket-weight mapping in plain-dict form."""
        return {int(bucket): float(weight) for bucket, weight in self.bucket_weights}


@dataclass(frozen=True)
class SigmoidBucketBlendConfig:
    """Bucketed wrapper applied on top of one saved Stage-5 sigmoid blend policy."""

    blend_config: BlendConfig
    bucket_config: BucketBlendConfig

    def weight_map(self) -> dict[int, float]:
        """Return the persisted bucket-weight mapping in plain-dict form."""
        return self.bucket_config.weight_map()


@dataclass(frozen=True)
class FoldMetricTask:
    """One walk-forward evaluation payload for the Stage-5 fold grid."""

    fold: dict[str, int]
    feature_set: str
    model_label: str
    target_mode: str


@dataclass
class HoldoutDiagnostics:
    """Expanded Stage-5 holdout artifacts used for rigor, interpretation, and docs."""

    holdout_summary: pd.DataFrame
    holdout_blend_decisions: pd.DataFrame | None
    deployment_recommendation: dict[str, Any]
    holdout_segment_evaluation: pd.DataFrame
    holdout_operating_regime_evaluation: pd.DataFrame
    operating_policy: dict[str, Any]
    holdout_coverage_segments: pd.DataFrame
    holdout_coverage_summary: dict[str, Any]
    holdout_predictions: pd.DataFrame
    holdout_inference: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_importance_summary: dict[str, Any] | None
    shap_importance: pd.DataFrame | None = None
    shap_importance_summary: dict[str, Any] | None = None
    supplemental_surface_summary: pd.DataFrame | None = None
    supplemental_surface_source_evaluation: pd.DataFrame | None = None
    supplemental_surface_segment_evaluation: pd.DataFrame | None = None
    supplemental_surface_operating_regime_evaluation: pd.DataFrame | None = None
    supplemental_surface_coverage_segments: pd.DataFrame | None = None
    supplemental_surface_coverage_summary: dict[str, Any] | None = None
    supplemental_surface_predictions: pd.DataFrame | None = None
    supplemental_surface_advisory: dict[str, Any] | None = None


def _hgb_coordinate_specs() -> dict[str, ModelSpec]:
    """Expose the coordinate-search HGB variants used by Stage-5."""
    return {
        label: spec
        for label, spec in build_model_catalog(
            include_hgb_coordinate_search=True,
            include_hgb_frontier=False,
        ).items()
        if label.startswith("hgb-coordinate-")
    }


def _hgb_frontier_specs() -> dict[str, ModelSpec]:
    """Expose the targeted frontier HGB variants used by Stage-5/Stage-6."""
    return {
        label: spec
        for label, spec in build_model_catalog(
            include_hgb_coordinate_search=False,
            include_hgb_frontier=True,
        ).items()
        if label.startswith("hgb-frontier-")
    }


def _configure_logging() -> None:
    """Initialize a simple logger for standalone Stage-5 execution."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _canonical_resolution(resolution: str) -> str:
    """Resolve aliases and reject unsupported Stage-5 resolutions."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    if canonical not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
            f"(aliases: {sorted(RESOLUTION_ALIASES)})"
        )
    return canonical


def _resolution_to_pandas_freq(resolution: str) -> str:
    """Map a supported resolution label to the matching pandas frequency string."""
    return {
        "1s": "1s",
        "5s": "5s",
        "10s": "10s",
        "30s": "30s",
        "1min": "1min",
        "5min": "5min",
        "10min": "10min",
        "15min": "15min",
    }[resolution]


def _gold_input_path(resolution: str, gold_dir: Path) -> Path:
    """Build the expected Stage-5 gold parquet path for one resolution."""
    suffix = RESOLUTION_TO_SUFFIX[resolution]
    return gold_dir / f"power_load_{suffix}_all_features.parquet"


def _model_split_path(resolution: str, feature_set: str, split: str, model_dir: Path) -> Path:
    """Build the Stage-4 model-split artifact path reused by Stage-5 checks."""
    suffix = RESOLUTION_TO_SUFFIX[resolution]
    return model_dir / f"{suffix}_{feature_set}_{split}.parquet"


def _git_commit() -> str:
    """Return the current git commit hash for manifest traceability."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_safe(value: Any) -> Any:
    """Convert pandas and numpy values into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def _relative_artifact_path(path: Path) -> str:
    """Render an artifact path relative to the repository root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _stage5_blend_policy_kind(target_mode: str) -> str:
    """Return which Stage-5 blend wrapper, if any, is encoded in one target mode."""
    target_mode_str = str(target_mode)
    nested_bucket_marker = "+blend_bucket_blend_b"
    if nested_bucket_marker in target_mode_str:
        base, suffix = target_mode_str.rsplit(nested_bucket_marker, 1)
        if base and suffix.isdigit():
            return "bucket"
    bucket_marker = "+bucket_blend_b"
    if bucket_marker in target_mode_str:
        base, suffix = target_mode_str.rsplit(bucket_marker, 1)
        if base and suffix.isdigit():
            return "bucket"
    if target_mode_str.endswith("+blend"):
        return "sigmoid"
    return ""


def _stage5_base_target_mode(target_mode: str) -> str:
    """Strip any Stage-5 blend wrapper and return the underlying raw/residual mode."""
    target_mode_str = str(target_mode)
    policy_kind = _stage5_blend_policy_kind(target_mode_str)
    if policy_kind == "bucket":
        if "+blend_bucket_blend_b" in target_mode_str:
            return target_mode_str.rsplit("+blend_bucket_blend_b", 1)[0]
        return target_mode_str.rsplit("+bucket_blend_b", 1)[0]
    if policy_kind == "sigmoid":
        return target_mode_str.removesuffix("+blend")
    return target_mode_str


def _stage5_blend_base_policy_kind(target_mode: str) -> str:
    """Return whether a Stage-5 wrapper is anchored on raw/residual or on a prior sigmoid blend."""
    target_mode_str = str(target_mode)
    if "+blend_bucket_blend_b" in target_mode_str:
        return "sigmoid"
    if _stage5_blend_policy_kind(target_mode_str):
        return "raw"
    return ""


def _stage5_bucket_blend_target_mode(
    base_target_mode: str,
    bucket_size_minutes: int,
    *,
    base_policy_kind: str = "raw",
) -> str:
    """Encode one fixed-size minute-bucket blend target mode."""
    if str(base_policy_kind) == "sigmoid":
        return f"{str(base_target_mode)}+blend_bucket_blend_b{int(bucket_size_minutes)}"
    return f"{str(base_target_mode)}+bucket_blend_b{int(bucket_size_minutes)}"


def _blend_config_from_manifest(
    manifest: dict[str, Any],
    *,
    feature_set: str,
    model_label: str,
    target_mode: str,
) -> dict[str, Any]:
    """Extract the saved Stage-5 blend config for one promoted blend candidate."""
    policy_kind = _stage5_blend_policy_kind(str(target_mode))
    if not policy_kind:
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
        payload = {
            "blend_policy_kind": "bucket",
            "blend_base_policy_kind": str(blend_policy.get("base_policy_kind", "raw")),
            "blend_bucket_size_minutes": int(blend_policy["bucket_size_minutes"]),
            "blend_bucket_cycle_minutes": int(blend_policy.get("cycle_minutes", 15)),
            "blend_bucket_weights_json": json.dumps(bucket_weights, sort_keys=True),
        }
        pre_bucket_blend = blend_policy.get("pre_bucket_blend")
        if isinstance(pre_bucket_blend, dict):
            payload.update(
                {
                    "blend_window": int(pre_bucket_blend["window"]),
                    "blend_sharpness": float(pre_bucket_blend["sharpness"]),
                    "blend_min_weight": float(pre_bucket_blend["min_weight"]),
                    "blend_max_weight": float(pre_bucket_blend["max_weight"]),
                }
            )
        return payload
    return {
        "blend_policy_kind": "sigmoid",
        "blend_base_policy_kind": "raw",
        "blend_window": int(blend_policy["window"]),
        "blend_sharpness": float(blend_policy["sharpness"]),
        "blend_min_weight": float(blend_policy["min_weight"]),
        "blend_max_weight": float(blend_policy["max_weight"]),
    }


def _empty_holdout_registry() -> pd.DataFrame:
    """Return the canonical empty Stage-5 holdout registry frame."""
    return pd.DataFrame(
        columns=[
            "run_id",
            "generated_at_utc",
            "mode",
            "resolution",
            "learned_candidate_label",
            "learned_feature_set",
            "learned_model_label",
            "learned_target_mode",
            "learned_mae",
            "learned_mae_pct",
            "learned_mae_ratio_to_persistence",
            "persistence_mae",
            "persistence_mae_pct",
            "learned_beats_persistence",
            "best_baseline_label",
            "best_baseline_mae",
            "best_baseline_mae_pct",
            "learned_beats_best_baseline",
            "recommended_candidate_label",
            "recommended_candidate_type",
            "decision_reason",
            "holdout_evaluation_artifact",
            "deployment_recommendation_artifact",
            "run_manifest_artifact",
            "blend_policy_kind",
            "blend_base_policy_kind",
            "blend_window",
            "blend_sharpness",
            "blend_min_weight",
            "blend_max_weight",
            "blend_bucket_size_minutes",
            "blend_bucket_cycle_minutes",
            "blend_bucket_weights_json",
        ]
    )


def _holdout_registry_row(run_dir: Path) -> dict[str, Any] | None:
    """Summarize one Stage-5 run into the cross-run holdout registry."""
    holdout_path = run_dir / "holdout_evaluation.csv"
    manifest_path = run_dir / "run_manifest.json"
    if not holdout_path.exists() or not manifest_path.exists():
        return None
    try:
        holdout = pd.read_csv(holdout_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
        return None
    learned = holdout.loc[holdout["candidate_type"].astype("string").ne("baseline")].copy()
    persistence = holdout.loc[holdout["candidate_label"].astype("string").eq("persistence")].copy()
    if learned.empty or persistence.empty:
        return None
    learned = learned.sort_values(["mae", "candidate_label"], ascending=[True, True], kind="stable")
    learned_row = learned.iloc[0]
    persistence_row = persistence.iloc[0]
    deployment_path = run_dir / "deployment_recommendation.json"
    deployment = {}
    if deployment_path.exists():
        try:
            deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            deployment = {}
    blend_config = _blend_config_from_manifest(
        manifest,
        feature_set=str(learned_row["feature_set"]),
        model_label=str(learned_row["model_label"]),
        target_mode=str(learned_row["target_mode"]),
    )
    return {
        "run_id": str(manifest.get("run_id", run_dir.name)),
        "generated_at_utc": str(manifest.get("generated_at_utc", "")),
        "mode": str(manifest.get("mode", "")),
        "resolution": str(learned_row["resolution"]),
        "learned_candidate_label": str(learned_row["candidate_label"]),
        "learned_feature_set": str(learned_row["feature_set"]),
        "learned_model_label": str(learned_row["model_label"]),
        "learned_target_mode": str(learned_row["target_mode"]),
        "learned_mae": float(learned_row["mae"]),
        "learned_mae_pct": float(learned_row["mae_pct"]),
        "learned_mae_ratio_to_persistence": float(learned_row["mae_ratio_to_persistence"]),
        "persistence_mae": float(persistence_row["mae"]),
        "persistence_mae_pct": float(persistence_row["mae_pct"]),
        "learned_beats_persistence": bool(float(learned_row["mae"]) < float(persistence_row["mae"])),
        "best_baseline_label": str(deployment.get("best_baseline_label", "persistence")),
        "best_baseline_mae": float(deployment.get("best_baseline_mae", float(persistence_row["mae"]))),
        "best_baseline_mae_pct": float(
            deployment.get("best_baseline_mae_pct", float(persistence_row["mae_pct"]))
        ),
        "learned_beats_best_baseline": bool(
            deployment.get("learned_beats_best_baseline", float(learned_row["mae"]) < float(persistence_row["mae"]))
        ),
        "recommended_candidate_label": str(deployment.get("recommended_candidate_label", "persistence")),
        "recommended_candidate_type": str(deployment.get("recommended_candidate_type", "baseline")),
        "decision_reason": str(deployment.get("decision_reason", "")),
        "holdout_evaluation_artifact": _relative_artifact_path(holdout_path),
        "deployment_recommendation_artifact": _relative_artifact_path(deployment_path),
        "run_manifest_artifact": _relative_artifact_path(manifest_path),
        **blend_config,
    }


def build_stage5_holdout_registry(output_root: Path) -> pd.DataFrame:
    """Backfill the Stage-5 cross-run holdout registry from historical artifacts."""
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return _empty_holdout_registry()
    for run_dir in output_root.iterdir():
        if not run_dir.is_dir() or run_dir.name == LATEST_ALIAS_NAME:
            continue
        row = _holdout_registry_row(run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        return _empty_holdout_registry()
    registry = pd.DataFrame(rows)
    registry["generated_at_utc"] = registry["generated_at_utc"].astype("string")
    registry = registry.sort_values(
        [
            "learned_beats_best_baseline",
            "learned_beats_persistence",
            "learned_mae",
            "generated_at_utc",
            "run_id",
        ],
        ascending=[False, False, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return registry


def refresh_stage5_holdout_registry(output_root: Path) -> pd.DataFrame:
    """Refresh the Stage-5 cross-run holdout registry at the stage root."""
    registry = build_stage5_holdout_registry(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_root / "holdout_registry.csv", index=False, float_format="%.6f")
    return registry


def read_stage5_holdout_registry(output_root: Path) -> pd.DataFrame:
    """Read the Stage-5 holdout registry, backfilling it when missing."""
    registry_path = output_root / "holdout_registry.csv"
    if registry_path.exists():
        try:
            registry = pd.read_csv(registry_path)
        except (OSError, pd.errors.EmptyDataError):
            registry = pd.DataFrame()
        if not registry.empty:
            return registry
    return refresh_stage5_holdout_registry(output_root)


def mae_ratio(model_mae: float, persistence_mae: float) -> float:
    """Return the model-to-persistence MAE ratio used in Stage-5 ranking tables."""
    if persistence_mae <= 0 or math.isnan(model_mae) or math.isnan(persistence_mae):
        return float("nan")
    return float(model_mae / persistence_mae)


def _load_gold_with_full_grid(resolution: str, gold_dir: Path) -> pd.DataFrame:
    """Load gold data and reindex it to the full expected timestamp grid."""
    input_path = _gold_input_path(resolution, gold_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing gold parquet: {input_path}")
    gold = pd.read_parquet(input_path)
    gold["timestamp"] = pd.to_datetime(gold["timestamp"], errors="raise")
    gold = gold.sort_values("timestamp").set_index("timestamp")
    freq = _resolution_to_pandas_freq(resolution)
    full_index = pd.date_range(start=gold.index.min(), end=gold.index.max(), freq=freq)
    gold = gold.reindex(full_index).reset_index().rename(columns={"index": "timestamp"})
    day_map = {
        day: idx + 1 for idx, day in enumerate(sorted(gold["timestamp"].dt.normalize().unique()))
    }
    gold["day_idx"] = gold["timestamp"].dt.normalize().map(day_map).astype(int)
    return gold


def build_walkforward_folds(
    *, holdout_start_day: int, n_folds: int, val_window_days: int, train_start_day: int = 1
) -> list[dict[str, int]]:
    """Build deterministic leakage-safe walk-forward folds for Stage-5."""
    if n_folds < 1 or val_window_days < 1:
        raise ValueError("n_folds and val_window_days must be >= 1")
    val_end_max = holdout_start_day - 1
    first_val_start = val_end_max - (n_folds * val_window_days) + 1
    if first_val_start <= train_start_day:
        raise ValueError("Not enough history for requested fold layout.")
    folds: list[dict[str, int]] = []
    for fold_idx in range(n_folds):
        val_start_day = first_val_start + fold_idx * val_window_days
        val_end_day = val_start_day + val_window_days - 1
        folds.append(
            {
                "fold": fold_idx + 1,
                "train_start_day": train_start_day,
                "train_end_day": val_start_day - 1,
                "val_start_day": val_start_day,
                "val_end_day": val_end_day,
            }
        )
    return folds


def _expected_steps_for_day_range(start_day: int, end_day: int, steps_per_day: int) -> int:
    """Return the expected row count for an inclusive day range at one resolution."""
    return max(end_day - start_day + 1, 0) * steps_per_day


def _dedupe_feature_columns(columns: list[str]) -> list[str]:
    """Preserve feature order while removing duplicate column names."""
    return list(dict.fromkeys(columns))


def _augment_with_curated_ramp_features(gold: pd.DataFrame, *, ramp_quantile: float) -> tuple[pd.DataFrame, float]:
    """Add short-horizon ramp flags used by the curated-ramp challenger set."""
    if "curated" not in FEATURE_SETS:
        raise ValueError("Feature set 'curated' is required to build curated_ramp.")
    if not 0.0 < ramp_quantile < 1.0:
        raise ValueError(f"ramp_quantile must be in (0,1), got {ramp_quantile}.")

    work = gold.copy()
    shifted_target = work[TARGET_COLUMN].shift(1)
    work["rolling_mean_3"] = shifted_target.rolling(3, min_periods=3).mean()
    work["rolling_std_3"] = shifted_target.rolling(3, min_periods=3).std()

    delta_series = work["delta_5"] if "delta_5" in work.columns else (work["lag_5"] - work["lag_1"])
    delta_series = pd.to_numeric(delta_series, errors="coerce")
    abs_delta = delta_series.abs()
    threshold = float(np.nanquantile(abs_delta.to_numpy(dtype=float), ramp_quantile))
    work["ramp_flag"] = np.where(
        delta_series.notna(),
        (abs_delta > threshold).astype(float),
        np.nan,
    )
    work["hour_x_delta_5"] = work["hour"] * delta_series
    return work, threshold


def _build_feature_sets(*, include_curated_ramp: bool, include_full_stable: bool = True) -> dict[str, list[str]]:
    """Build the effective Stage-5 feature-set catalog for this run."""
    feature_sets = {name: list(columns) for name, columns in FEATURE_SETS.items()}
    if include_full_stable:
        full = feature_sets.get("full")
        if full is None:
            raise ValueError("Feature set 'full' not found in configuration.")
        feature_sets[FULL_STABLE_FEATURE_SET_NAME] = [
            column for column in full if column not in FULL_STABLE_EXCLUDED_COLUMNS
        ]
    else:
        feature_sets.pop(FULL_STABLE_FEATURE_SET_NAME, None)
    if include_curated_ramp:
        curated = feature_sets.get("curated")
        if curated is None:
            raise ValueError("Feature set 'curated' not found in configuration.")
        feature_sets[RAMP_FEATURE_SET_NAME] = _dedupe_feature_columns(curated + list(RAMP_ADDITIONAL_FEATURES))
    return feature_sets


def _resolve_feature_set_columns(
    feature_set: str,
    *,
    feature_sets: dict[str, list[str]],
    residual_baseline: str | None = None,
) -> list[str]:
    """Return the concrete feature columns for one candidate and target mode."""
    columns = list(feature_sets[feature_set])
    if residual_baseline == "avg_workday":
        columns = _dedupe_feature_columns(
            columns + [feature for feature in RESIDUAL_SUPPORT_FEATURES if feature not in columns]
        )
    return columns


def _build_auto_hgb_specs() -> dict[str, ModelSpec]:
    """Build centrally configured HGB search candidates with deterministic labels."""
    specs: dict[str, ModelSpec] = {}
    for learning_rate in MODELING_PERFORMANCE_HGB_SEARCH["learning_rates"]:
        for max_depth in MODELING_PERFORMANCE_HGB_SEARCH["max_depths"]:
            for min_samples_leaf in MODELING_PERFORMANCE_HGB_SEARCH["min_samples_leaf"]:
                for l2_regularization in MODELING_PERFORMANCE_HGB_SEARCH["l2_regularization"]:
                    for max_iter in MODELING_PERFORMANCE_HGB_SEARCH["max_iters"]:
                        label = (
                            "hgb-auto-"
                            f"lr{int(round(learning_rate * 1000)):03d}-"
                            f"d{int(max_depth)}-"
                            f"leaf{int(min_samples_leaf)}-"
                            f"l2{int(round(l2_regularization * 1000)):04d}-"
                            f"it{int(max_iter)}"
                        )
                        specs[label] = ModelSpec(
                            model_label=label,
                            family="hgb",
                            params={
                                "max_depth": int(max_depth),
                                "max_iter": int(max_iter),
                                "learning_rate": float(learning_rate),
                                "min_samples_leaf": int(min_samples_leaf),
                                "l2_regularization": float(l2_regularization),
                                "early_stopping": False,
                                "random_state": 42,
                            },
                            factory=lambda params={
                                "max_depth": int(max_depth),
                                "max_iter": int(max_iter),
                                "learning_rate": float(learning_rate),
                                "min_samples_leaf": int(min_samples_leaf),
                                "l2_regularization": float(l2_regularization),
                                "early_stopping": False,
                                "random_state": 42,
                            }: HistGradientBoostingRegressor(**params),
                        )
    return specs


def _screen_adaptive_hgb_candidates(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]],
    feature_sets: dict[str, list[str]],
    selected_feature_sets: list[str],
    resolution: str,
    steps_per_day: int,
    policy: dict[str, Any],
) -> tuple[list[ModelSpec], pd.DataFrame]:
    """Screen the configured HGB search space on early folds and keep the strongest survivors."""
    if not MODELING_PERFORMANCE_HGB_SEARCH["enabled"]:
        return [], pd.DataFrame()
    auto_specs = _build_auto_hgb_specs()
    if not auto_specs:
        return [], pd.DataFrame()
    screen_folds = folds[: min(len(folds), int(MODELING_PERFORMANCE_HGB_SEARCH["screen_folds"]))]
    if not screen_folds:
        return [], pd.DataFrame()
    candidate_feature_sets = [
        name
        for name in selected_feature_sets
        if name in set(policy["feature_sets"]) and name in feature_sets
    ]
    if not candidate_feature_sets:
        candidate_feature_sets = list(selected_feature_sets)
    preferred_order = [
        FULL_STABLE_FEATURE_SET_NAME,
        FULL_STABLE_LEGACY_FEATURE_SET_NAME,
        REGIME_PROFILE_FEATURE_SET_NAME,
        RAMP_FEATURE_SET_NAME,
        "curated",
        "minimal",
    ]
    candidate_feature_sets = [
        name for name in preferred_order if name in candidate_feature_sets
    ] or candidate_feature_sets
    candidate_feature_sets = candidate_feature_sets[:2]
    rows: list[dict[str, Any]] = []
    for model_spec in auto_specs.values():
        for feature_set in candidate_feature_sets:
            feature_cols = _resolve_feature_set_columns(feature_set, feature_sets=feature_sets)
            for fold in screen_folds:
                train_df = gold.loc[
                    gold["day_idx"].between(fold["train_start_day"], fold["train_end_day"])
                ].copy()
                val_df = gold.loc[
                    gold["day_idx"].between(fold["val_start_day"], fold["val_end_day"])
                ].copy()
                n_eval_total = _expected_steps_for_day_range(
                    fold["val_start_day"],
                    fold["val_end_day"],
                    steps_per_day,
                )
                metrics = _fit_and_evaluate(
                    train_df=train_df,
                    eval_df=val_df,
                    feature_cols=feature_cols,
                    model_spec=model_spec,
                    target_mode="raw",
                    n_eval_total=n_eval_total,
                )
                if metrics is None:
                    continue
                rows.append(
                    {
                        "model_label": model_spec.model_label,
                        "feature_set": feature_set,
                        "fold": int(fold["fold"]),
                        "mae_ratio": float(metrics["mae_ratio"]),
                        "mae_pct": float(metrics["mae_pct"]),
                        "coverage": float(metrics["coverage"]),
                    }
                )
    if not rows:
        return [], pd.DataFrame()
    screen = pd.DataFrame(rows)
    ranked = (
        screen.groupby(["model_label", "feature_set"], dropna=False)
        .agg(
            fold_mean_mae_ratio=("mae_ratio", "mean"),
            fold_std_mae_ratio=("mae_ratio", "std"),
            mean_mae_pct=("mae_pct", "mean"),
            mean_coverage=("coverage", "mean"),
            fold_n=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["mean_coverage", "fold_mean_mae_ratio", "fold_std_mae_ratio", "mean_mae_pct"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    ranked["fold_std_mae_ratio"] = ranked["fold_std_mae_ratio"].fillna(0.0)
    keep = min(
        int(MODELING_PERFORMANCE_HGB_SEARCH["max_candidates"]),
        max(int(MODELING_PERFORMANCE_HGB_SEARCH["min_candidates"]), ranked["model_label"].nunique()),
    )
    selected_labels = ranked["model_label"].drop_duplicates().head(keep).tolist()
    selected_specs = [auto_specs[label] for label in selected_labels]
    return selected_specs, ranked


def _classify_feature_causality(feature: str) -> tuple[str, str]:
    """Classify whether one feature is causal, non-causal, or still needs review."""
    if feature in {"rolling_mean_3", "rolling_std_3"}:
        return ("causal", "Derived from avg_load.shift(1) short-history windows.")
    if feature in {"ramp_flag", "hour_x_delta_5"}:
        return ("causal", "Derived from prior lag and calendar context.")
    if feature in {
        "avg_workday_baseline",
        "profile_residual_lag_1",
        "previous_day_residual",
        "previous_day_load",
        "prev_day_workday",
        "next_day_workday",
        "workday_transition",
        "profile_activity_ratio",
        "profile_active_flag",
    }:
        return ("causal", "Derived from historical profile state and known calendar schedule.")
    if feature.startswith("rolling_"):
        return ("needs_review", "Rolling features need shifted-history verification.")
    if feature.startswith(("lag_min_", "rolling_mean_min_", "rolling_std_min_", "rolling_max_min_", "rolling_min_min_", "slope_min_")):
        return ("causal", "Derived from resolution-normalized causal history windows.")
    if feature.startswith(("lag_", "delta_", "slope_")):
        return ("causal", "History-based derived feature.")
    if feature in {
        "workday",
        "year",
        "quarter",
        "month",
        "day",
        "day_of_week",
        "hour",
        "season",
        "time_of_day",
    }:
        return ("causal", "Calendar/business context available at inference time.")
    if feature == TARGET_COLUMN:
        return ("non_causal", "Target in feature set is direct leakage.")
    return ("unknown", "Feature requires manual causality review.")


def _feature_causality_audit(
    selected_feature_sets: list[str], *, feature_sets: dict[str, list[str]]
) -> pd.DataFrame:
    """Write one row per selected feature documenting its causality status."""
    rows: list[dict[str, str]] = []
    for feature_set in selected_feature_sets:
        for feature in feature_sets[feature_set]:
            status, rationale = _classify_feature_causality(feature)
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "status": status,
                    "rationale": rationale,
                }
            )
    return pd.DataFrame(rows)


def _minute_integrity_audit(
    gold: pd.DataFrame, folds: list[dict[str, int]], *, steps_per_day: int
) -> pd.DataFrame:
    """Audit expected versus observed target rows across splits and validation folds."""
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validate", "test"):
        start_day, end_day = SPLIT_DAY_RANGES[split_name]
        mask = gold["day_idx"].between(start_day, end_day)
        expected = _expected_steps_for_day_range(start_day, end_day, steps_per_day)
        actual = int(gold.loc[mask, TARGET_COLUMN].notna().sum())
        rows.append(
            {
                "scope": "split",
                "name": split_name,
                "start_day": start_day,
                "end_day": end_day,
                "expected_steps": expected,
                "actual_target_rows": actual,
                "missing_steps": expected - actual,
            }
        )
    for fold in folds:
        start_day = fold["val_start_day"]
        end_day = fold["val_end_day"]
        mask = gold["day_idx"].between(start_day, end_day)
        expected = _expected_steps_for_day_range(start_day, end_day, steps_per_day)
        actual = int(gold.loc[mask, TARGET_COLUMN].notna().sum())
        rows.append(
            {
                "scope": "fold_validate",
                "name": f"fold_{fold['fold']}",
                "start_day": start_day,
                "end_day": end_day,
                "expected_steps": expected,
                "actual_target_rows": actual,
                "missing_steps": expected - actual,
            }
        )
    return pd.DataFrame(rows)


def _compute_persistence_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute the persistence baseline metrics for one evaluation frame."""
    metrics = compute_regression_metrics(
        df[TARGET_COLUMN],
        df["lag_1"],
        n_total=int(len(df)),
    )
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mae_pct": float(metrics["mae_pct"]),
        "rmse_pct": float(metrics["rmse_pct"]),
        "n_eval": int(metrics["n_eval"]),
    }


def _reproduce_baseline(*, model_dir: Path, resolution: str, tolerance_mae: float) -> dict[str, Any]:
    """Verify that current persistence metrics still reproduce the Stage-4 reference."""
    reference_path = STEP4_ARTIFACT_DIR / "metrics_overall.csv"
    if not reference_path.exists():
        return {"status": "fail", "reason": f"Missing reference metrics: {reference_path}", "checks": []}
    reference = pd.read_csv(reference_path)
    checks: list[dict[str, Any]] = []
    overall_pass = True
    for split in ("validate", "test"):
        split_path = _model_split_path(resolution, "minimal", split, model_dir)
        if not split_path.exists():
            checks.append({"split": split, "status": "fail", "reason": f"Missing split: {split_path}"})
            overall_pass = False
            continue
        current = _compute_persistence_metrics(pd.read_parquet(split_path))
        ref_row = reference[(reference["model"] == "persistence") & (reference["split"] == split)]
        if ref_row.empty:
            checks.append({"split": split, "status": "fail", "reason": "Missing persistence reference row."})
            overall_pass = False
            continue
        ref_mae = float(ref_row.iloc[0]["mae"])
        delta = float(abs(float(current["mae"]) - ref_mae))
        passed = bool(delta <= tolerance_mae)
        checks.append(
            {
                "split": split,
                "status": "pass" if passed else "fail",
                "reference_mae": ref_mae,
                "current_mae": float(current["mae"]),
                "delta_mae": delta,
                "tolerance_mae": tolerance_mae,
                "n_eval": int(current["n_eval"]),
            }
        )
        overall_pass = overall_pass and passed
    return {"status": "pass" if overall_pass else "fail", "checks": checks}


def _step4_prediction_mode() -> dict[str, Any]:
    """Confirm that Stage-4 artifacts were generated in online single-step mode."""
    manifest_path = STEP4_ARTIFACT_DIR / "run_manifest.json"
    if not manifest_path.exists():
        return {"status": "fail", "reason": f"Missing run manifest: {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = manifest.get("prediction_mode")
    if mode != "online_single_step":
        return {"status": "fail", "reason": f"Expected 'online_single_step', found {mode!r}"}
    return {"status": "pass", "prediction_mode": mode}


def run_preflight_audit(
    *,
    gold: pd.DataFrame,
    selected_feature_sets: list[str],
    feature_sets: dict[str, list[str]],
    folds: list[dict[str, int]],
    output_dir: Path,
    resolution: str,
    tolerance_mae: float,
    steps_per_day: int,
) -> dict[str, Any]:
    """Run the Stage-5 protocol checks before fold evaluation starts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_check = _step4_prediction_mode()
    causality_df = _feature_causality_audit(selected_feature_sets, feature_sets=feature_sets)
    causality_df.to_csv(output_dir / "feature_causality_audit.csv", index=False)
    integrity_df = _minute_integrity_audit(gold, folds, steps_per_day=steps_per_day)
    integrity_df.to_csv(output_dir / "minute_integrity_audit.csv", index=False)
    baseline_check = _reproduce_baseline(
        model_dir=PATHS["model_dir"], resolution=resolution, tolerance_mae=tolerance_mae
    )

    holdout_start, holdout_end = SPLIT_DAY_RANGES["test"]
    holdout_dates = (
        gold.loc[gold["day_idx"].between(holdout_start, holdout_end), "timestamp"]
        .dropna()
        .sort_values()
    )
    holdout_lock = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "resolution": resolution,
        "test_day_range": [holdout_start, holdout_end],
        "expected_steps": _expected_steps_for_day_range(holdout_start, holdout_end, steps_per_day),
        "first_timestamp": holdout_dates.iloc[0].isoformat() if not holdout_dates.empty else None,
        "last_timestamp": holdout_dates.iloc[-1].isoformat() if not holdout_dates.empty else None,
    }
    (output_dir / "holdout_lock.json").write_text(json.dumps(holdout_lock, indent=2), encoding="utf-8")

    non_causal_count = int(causality_df["status"].eq("non_causal").sum())
    checks = {
        "prediction_semantics": mode_check,
        "baseline_reproduction": baseline_check,
        "feature_causality": {
            "status": "pass" if non_causal_count == 0 else "fail",
            "non_causal_count": non_causal_count,
            "needs_review_count": int(causality_df["status"].eq("needs_review").sum()),
            "unknown_count": int(causality_df["status"].eq("unknown").sum()),
        },
    }
    overall_pass = (
        checks["prediction_semantics"]["status"] == "pass"
        and checks["baseline_reproduction"]["status"] == "pass"
        and checks["feature_causality"]["status"] == "pass"
    )

    lines = [
        "# Step 5 Preflight Audit",
        "",
        f"- Generated: `{datetime.now(UTC).isoformat()}`",
        f"- Resolution: `{resolution}`",
        f"- Overall status: `{'pass' if overall_pass else 'fail'}`",
        "",
        "## Checks",
        "",
        f"- Prediction semantics: `{checks['prediction_semantics']['status']}`",
        f"- Baseline reproduction: `{checks['baseline_reproduction']['status']}`",
        f"- Feature causality: `{checks['feature_causality']['status']}`",
    ]
    (output_dir / "preflight_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"overall_status": "pass" if overall_pass else "fail", "checks": checks}


def _encode_for_ridge(train_x: pd.DataFrame, eval_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply aligned one-hot encoding for ridge candidates only."""
    ohe_cols = [col for col in OHE_COLUMNS if col in train_x.columns]
    train_e = train_x.copy()
    eval_e = eval_x.copy()
    if ohe_cols:
        train_e[ohe_cols] = train_e[ohe_cols].astype("category")
        eval_e[ohe_cols] = eval_e[ohe_cols].astype("category")
        train_e = pd.get_dummies(train_e, columns=ohe_cols, drop_first=True)
        eval_e = pd.get_dummies(eval_e, columns=ohe_cols, drop_first=True)
        train_e, eval_e = train_e.align(eval_e, join="left", axis=1, fill_value=0.0)
        eval_e = eval_e.reindex(columns=train_e.columns, fill_value=0.0)
    train_e.columns = [str(col).replace(".0", "") for col in train_e.columns]
    eval_e.columns = [str(col).replace(".0", "") for col in eval_e.columns]
    return train_e, eval_e


def _fit_and_evaluate(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    model_spec: ModelSpec,
    target_mode: str,
    n_eval_total: int,
) -> dict[str, Any] | None:
    """Train one candidate and return aligned evaluation metrics against persistence."""
    aligned_result = _fit_and_align(
        train_df=train_df,
        eval_df=eval_df,
        feature_cols=feature_cols,
        model_spec=model_spec,
        target_mode=target_mode,
    )
    if aligned_result is None:
        return None

    aligned, train_mae = aligned_result
    candidate_metrics = compute_regression_metrics(
        aligned["y_true"],
        aligned["y_pred"],
        n_total=int(n_eval_total),
    )
    persistence_metrics = compute_regression_metrics(
        aligned["y_true"],
        aligned["y_persist"],
        n_total=int(n_eval_total),
    )
    model_mae = float(candidate_metrics["mae"])
    persistence_mae = float(persistence_metrics["mae"])
    n_eval = int(candidate_metrics["n_eval"])
    return {
        "mae": model_mae,
        "rmse": float(candidate_metrics["rmse"]),
        "mae_pct": float(candidate_metrics["mae_pct"]),
        "rmse_pct": float(candidate_metrics["rmse_pct"]),
        "mae_ratio": mae_ratio(model_mae, persistence_mae),
        "persistence_mae": persistence_mae,
        "persistence_mae_pct": float(persistence_metrics["mae_pct"]),
        "train_mae": train_mae,
        "train_val_mae_ratio": float(train_mae / model_mae) if model_mae > 0 else float("nan"),
        "n_eval": n_eval,
        "n_eval_total": int(n_eval_total),
        "coverage": float(candidate_metrics["coverage"]),
    }


def _fit_model_bundle(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    model_spec: ModelSpec,
    target_mode: str,
) -> dict[str, Any] | None:
    """Fit one candidate and return the aligned bundle used by Stage-5 diagnostics."""
    required_cols = list(dict.fromkeys(list(feature_cols) + [TARGET_COLUMN, "lag_1"]))
    train_work = train_df[required_cols].copy()
    eval_work = eval_df[required_cols].copy()

    if target_mode == "residual":
        train_work = train_work.dropna(subset=[TARGET_COLUMN, "lag_1"])
        eval_work = eval_work.dropna(subset=[TARGET_COLUMN, "lag_1"])
        y_train = train_work[TARGET_COLUMN] - train_work["lag_1"]
        y_eval = eval_work[TARGET_COLUMN]
    else:
        train_work = train_work.dropna(subset=[TARGET_COLUMN])
        eval_work = eval_work.dropna(subset=[TARGET_COLUMN, "lag_1"])
        y_train = train_work[TARGET_COLUMN]
        y_eval = eval_work[TARGET_COLUMN]

    if model_spec.family == "ridge":
        train_work = train_work.dropna(subset=feature_cols)
        eval_work = eval_work.dropna(subset=feature_cols)
        y_train = y_train.loc[train_work.index]
        y_eval = y_eval.loc[eval_work.index]

    if train_work.empty or eval_work.empty:
        return None

    x_train = train_work[feature_cols]
    x_eval = eval_work[feature_cols]
    if model_spec.family == "ridge":
        x_train, x_eval = _encode_for_ridge(x_train, x_eval)

    model = model_spec.factory()
    model.fit(x_train, y_train)

    train_pred_base = pd.Series(model.predict(x_train), index=train_work.index, dtype=float)
    train_pred = train_work["lag_1"] + train_pred_base if target_mode == "residual" else train_pred_base
    train_truth = train_work[TARGET_COLUMN]
    train_aligned = pd.DataFrame({"y_true": train_truth, "y_pred": train_pred}).dropna()
    if train_aligned.empty:
        return None
    train_mae = float(np.mean(np.abs(train_aligned["y_true"] - train_aligned["y_pred"])))

    pred = pd.Series(model.predict(x_eval), index=eval_work.index, dtype=float)
    y_pred = eval_work["lag_1"] + pred if target_mode == "residual" else pred

    aligned = pd.DataFrame(
        {"y_true": y_eval, "y_pred": y_pred, "y_persist": eval_work["lag_1"]}
    ).dropna()
    if aligned.empty:
        return None
    aligned = aligned.sort_index()
    x_eval_aligned = x_eval.loc[aligned.index].copy()
    eval_work_aligned = eval_work.loc[aligned.index].copy()
    return {
        "model": model,
        "aligned": aligned,
        "train_mae": train_mae,
        "x_eval": x_eval_aligned,
        "eval_work": eval_work_aligned,
        "feature_names": list(x_eval_aligned.columns.astype(str)),
        "target_mode": target_mode,
    }


def _build_stage5_eval_payloads(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]],
    feature_cols: list[str],
    model_spec: ModelSpec,
    target_mode: str,
    steps_per_day: int,
) -> list[dict[str, Any]]:
    """Build leakage-safe train/eval payloads for one Stage-5 candidate across folds."""
    payloads: list[dict[str, Any]] = []
    for fold in folds:
        train_df = gold.loc[
            gold["day_idx"].between(fold["train_start_day"], fold["train_end_day"])
        ].copy()
        eval_df = gold.loc[
            gold["day_idx"].between(fold["val_start_day"], fold["val_end_day"])
        ].copy()
        n_eval_total = _expected_steps_for_day_range(
            int(fold["val_start_day"]),
            int(fold["val_end_day"]),
            steps_per_day,
        )
        aligned_result = _fit_and_align(
            train_df=train_df,
            eval_df=eval_df,
            feature_cols=feature_cols,
            model_spec=model_spec,
            target_mode=target_mode,
        )
        if aligned_result is None:
            continue
        aligned, _ = aligned_result
        timestamps = (
            pd.to_datetime(eval_df.loc[aligned.index, "timestamp"], errors="raise")
            if "timestamp" in eval_df.columns
            else None
        )
        payloads.append(
            {
                "fold_meta": dict(fold),
                "train_df": train_df,
                "eval_df": eval_df,
                "aligned": aligned,
                "timestamps": timestamps,
                "n_eval_total": int(n_eval_total),
            }
        )
    return payloads


def _fit_and_align(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    model_spec: ModelSpec,
    target_mode: str,
) -> tuple[pd.DataFrame, float] | None:
    """Fit one candidate and return aligned truth, predictions, and train MAE."""
    bundle = _fit_model_bundle(
        train_df=train_df,
        eval_df=eval_df,
        feature_cols=feature_cols,
        model_spec=model_spec,
        target_mode=target_mode,
    )
    if bundle is None:
        return None
    return cast(pd.DataFrame, bundle["aligned"]), float(bundle["train_mae"])


def _sigmoid(x: float) -> float:
    """Evaluate a numerically stable sigmoid used by the blend guardrail."""
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _apply_blend_policy(
    *,
    aligned: pd.DataFrame,
    blend_config: BlendConfig,
    n_eval_total: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Blend learned and persistence predictions using recent realized error."""
    model_error_history: list[float] = []
    persist_error_history: list[float] = []
    decisions: list[dict[str, Any]] = []
    for idx, row in aligned.iterrows():
        if model_error_history and persist_error_history:
            model_recent = float(np.mean(model_error_history))
            persist_recent = float(np.mean(persist_error_history))
            skill = (persist_recent - model_recent) / (persist_recent + 1e-9)
            weight = _sigmoid(blend_config.sharpness * skill)
        else:
            weight = 0.5
        weight = float(np.clip(weight, blend_config.min_weight, blend_config.max_weight))
        blend_pred = weight * float(row["y_pred"]) + (1.0 - weight) * float(row["y_persist"])
        model_abs_error = abs(float(row["y_true"]) - float(row["y_pred"]))
        persist_abs_error = abs(float(row["y_true"]) - float(row["y_persist"]))
        blend_abs_error = abs(float(row["y_true"]) - blend_pred)
        decisions.append(
            {
                "row_index": int(idx),  # type: ignore[arg-type]
                "blend_weight": weight,
                "model_pred": float(row["y_pred"]),
                "persistence_pred": float(row["y_persist"]),
                "blend_pred": blend_pred,
                "y_true": float(row["y_true"]),
                "model_abs_error": model_abs_error,
                "persistence_abs_error": persist_abs_error,
                "blend_abs_error": blend_abs_error,
                "blend_policy_kind": "sigmoid",
            }
        )
        model_error_history.append(model_abs_error)
        persist_error_history.append(persist_abs_error)
        if len(model_error_history) > blend_config.window:
            model_error_history.pop(0)
            persist_error_history.pop(0)

    decision_df = pd.DataFrame(decisions)
    if decision_df.empty:
        return {}, decision_df
    blend_eval = compute_regression_metrics(
        decision_df["y_true"],
        decision_df["blend_pred"],
        n_total=int(n_eval_total),
    )
    persistence_eval = compute_regression_metrics(
        decision_df["y_true"],
        decision_df["persistence_pred"],
        n_total=int(n_eval_total),
    )
    blend_mae = float(blend_eval["mae"])
    persist_mae = float(persistence_eval["mae"])
    metrics = {
        "mae": blend_mae,
        "rmse": float(blend_eval["rmse"]),
        "mae_pct": float(blend_eval["mae_pct"]),
        "rmse_pct": float(blend_eval["rmse_pct"]),
        "mae_ratio": mae_ratio(blend_mae, persist_mae),
        "persistence_mae": persist_mae,
        "persistence_mae_pct": float(persistence_eval["mae_pct"]),
        "train_mae": float("nan"),
        "train_val_mae_ratio": float("nan"),
        "n_eval": int(len(decision_df)),
        "n_eval_total": int(n_eval_total),
        "coverage": float(blend_eval["coverage"]),
        "mean_blend_weight": float(decision_df["blend_weight"].mean()),
        "model_dominated_frac": float((decision_df["blend_weight"] >= 0.5).mean()),
    }
    return metrics, decision_df


def _calibrate_bucket_blend_config(
    *,
    aligned: pd.DataFrame,
    timestamps: pd.Index | pd.Series | pd.DatetimeIndex,
    bucket_size_minutes: int,
    cycle_minutes: int,
    candidate_weights: list[float],
) -> BucketBlendConfig | None:
    """Greedily select one fixed minute-bucket blend map on a calibration surface."""
    if aligned.empty:
        return None
    valid_weights = sorted(
        {
            float(value)
            for value in candidate_weights
            if np.isfinite(value) and 0.0 <= float(value) <= 1.0
        }
    )
    if not valid_weights:
        return None
    timestamp_index = pd.DatetimeIndex(pd.to_datetime(pd.Index(timestamps), errors="raise"))
    if len(timestamp_index) != len(aligned):
        raise ValueError("Bucket-blend calibration timestamps must align with the candidate rows.")
    bucket_index = _index_minute_buckets(
        timestamp_index,
        bucket_minutes=int(bucket_size_minutes),
        cycle_minutes=int(cycle_minutes),
    )
    candidate_buckets = sorted({int(value) for value in bucket_index.tolist()})
    if not candidate_buckets:
        return None

    selected_weights: dict[int, float] = {}
    for bucket_key in candidate_buckets:
        best_weight = 0.0
        best_mae = float("inf")
        best_tie_break = float("inf")
        for candidate_weight in valid_weights:
            working_weights = {**selected_weights, int(bucket_key): float(candidate_weight)}
            blended, _ = _blend_prediction_series_by_bucket(
                candidate_series=aligned["y_pred"],
                persistence_series=aligned["y_persist"],
                timestamps=timestamp_index,
                bucket_weights=working_weights,
                bucket_size_minutes=int(bucket_size_minutes),
                cycle_minutes=int(cycle_minutes),
            )
            mae_value = float(
                compute_regression_metrics(
                    aligned["y_true"],
                    blended,
                    n_total=int(len(aligned)),
                )["mae"]
            )
            tie_break = abs(float(candidate_weight))
            if (
                mae_value < best_mae
                or (math.isclose(mae_value, best_mae, rel_tol=0.0, abs_tol=1e-12) and tie_break < best_tie_break)
            ):
                best_mae = mae_value
                best_tie_break = tie_break
                best_weight = float(candidate_weight)
        selected_weights[int(bucket_key)] = float(best_weight)
    return BucketBlendConfig(
        bucket_size_minutes=int(bucket_size_minutes),
        cycle_minutes=int(cycle_minutes),
        bucket_weights=tuple(sorted(selected_weights.items())),
    )


def _apply_bucket_blend_policy(
    *,
    aligned: pd.DataFrame,
    timestamps: pd.Index | pd.Series | pd.DatetimeIndex,
    bucket_config: BucketBlendConfig,
    n_eval_total: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply one fixed minute-bucket blend policy to aligned candidate predictions."""
    blended, bucket_index = _blend_prediction_series_by_bucket(
        candidate_series=aligned["y_pred"],
        persistence_series=aligned["y_persist"],
        timestamps=timestamps,
        bucket_weights=bucket_config.weight_map(),
        bucket_size_minutes=int(bucket_config.bucket_size_minutes),
        cycle_minutes=int(bucket_config.cycle_minutes),
    )
    decision_df = pd.DataFrame(
        {
            "row_index": aligned.index.astype(int),
            "blend_weight": bucket_index.map(bucket_config.weight_map()).astype(float).to_numpy(),
            "blend_bucket": bucket_index.astype(int).to_numpy(),
            "model_pred": aligned["y_pred"].astype(float).to_numpy(),
            "persistence_pred": aligned["y_persist"].astype(float).to_numpy(),
            "blend_pred": blended.astype(float).to_numpy(),
            "y_true": aligned["y_true"].astype(float).to_numpy(),
            "blend_policy_kind": "bucket",
        }
    )
    decision_df["model_abs_error"] = (decision_df["y_true"] - decision_df["model_pred"]).abs()
    decision_df["persistence_abs_error"] = (
        decision_df["y_true"] - decision_df["persistence_pred"]
    ).abs()
    decision_df["blend_abs_error"] = (decision_df["y_true"] - decision_df["blend_pred"]).abs()
    if decision_df.empty:
        return {}, decision_df
    blend_eval = compute_regression_metrics(
        decision_df["y_true"],
        decision_df["blend_pred"],
        n_total=int(n_eval_total),
    )
    persistence_eval = compute_regression_metrics(
        decision_df["y_true"],
        decision_df["persistence_pred"],
        n_total=int(n_eval_total),
    )
    blend_mae = float(blend_eval["mae"])
    persist_mae = float(persistence_eval["mae"])
    metrics = {
        "mae": blend_mae,
        "rmse": float(blend_eval["rmse"]),
        "mae_pct": float(blend_eval["mae_pct"]),
        "rmse_pct": float(blend_eval["rmse_pct"]),
        "mae_ratio": mae_ratio(blend_mae, persist_mae),
        "persistence_mae": persist_mae,
        "persistence_mae_pct": float(persistence_eval["mae_pct"]),
        "train_mae": float("nan"),
        "train_val_mae_ratio": float("nan"),
        "n_eval": int(len(decision_df)),
        "n_eval_total": int(n_eval_total),
        "coverage": float(blend_eval["coverage"]),
        "mean_blend_weight": float(decision_df["blend_weight"].mean()),
        "model_dominated_frac": float((decision_df["blend_weight"] >= 0.5).mean()),
    }
    return metrics, decision_df


def _prediction_column_name(label: str) -> str:
    """Normalize a prediction label into a stable wide-column artifact name."""
    safe = "".join(char if char.isalnum() else "_" for char in str(label).lower()).strip("_")
    return f"{safe}_pred"


def _resolution_step_seconds(resolution: str) -> int:
    """Return the atomic step size for one supported resolution in seconds."""
    return max(1, int(pd.Timedelta(_resolution_to_pandas_freq(resolution)).total_seconds()))


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


def _blend_prediction_series_by_bucket(
    *,
    candidate_series: pd.Series,
    persistence_series: pd.Series,
    timestamps: pd.Index | pd.Series | pd.DatetimeIndex,
    bucket_weights: dict[int, float],
    bucket_size_minutes: int,
    cycle_minutes: int,
) -> tuple[pd.Series, pd.Series]:
    """Blend one learned prediction path toward persistence using minute buckets."""
    persistence_aligned = persistence_series.astype(float).copy()
    candidate_aligned = candidate_series.reindex(persistence_aligned.index).astype(float)
    timestamp_index = pd.DatetimeIndex(pd.to_datetime(pd.Index(timestamps), errors="raise"))
    if len(timestamp_index) != len(persistence_aligned):
        raise ValueError("Bucket blend timestamps must align one-to-one with the prediction rows.")
    bucket_index = _index_minute_buckets(
        timestamp_index,
        bucket_minutes=int(bucket_size_minutes),
        cycle_minutes=int(cycle_minutes),
    )
    bucket_index.index = persistence_aligned.index
    blended = persistence_aligned.copy()
    for bucket_key, candidate_weight in sorted(bucket_weights.items()):
        valid_mask = bucket_index.eq(int(bucket_key)) & candidate_aligned.notna()
        if not bool(valid_mask.any()):
            continue
        blended.loc[valid_mask] = (
            persistence_aligned.loc[valid_mask]
            + float(candidate_weight)
            * (candidate_aligned.loc[valid_mask] - persistence_aligned.loc[valid_mask])
        )
    return blended.astype(float), bucket_index.astype(int)


def _prepare_classical_benchmark_series(train_series: pd.Series) -> pd.Series:
    """Normalize benchmark series into a statsmodels-safe dense index."""
    clean = pd.Series(train_series, copy=False).dropna().astype(float)
    return clean.reset_index(drop=True)


def _holt_damped_forecast(train_series: pd.Series, horizon_steps: int) -> pd.Series | None:
    """Forecast the holdout horizon with a damped Holt exponential-smoothing baseline."""
    clean = _prepare_classical_benchmark_series(train_series)
    if horizon_steps <= 0 or len(clean) < 10:
        return None
    try:
        fit = ExponentialSmoothing(
            clean,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)
        forecast = pd.Series(fit.forecast(horizon_steps), dtype=float)
    except Exception as exc:  # pragma: no cover - defensive on statsmodels convergence quirks.
        logger.warning("Stage-5 Holt-damped benchmark failed: %s", exc)
        return None
    return forecast.clip(lower=0.0)


def _arima_forecast(
    train_series: pd.Series, horizon_steps: int, *, order: tuple[int, int, int] = (1, 1, 1)
) -> pd.Series | None:
    """Forecast the holdout horizon with an ARIMA baseline."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    clean = _prepare_classical_benchmark_series(train_series)
    if horizon_steps <= 0 or len(clean) < 30:
        return None
    try:
        fit = _ARIMA(
            clean,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecast = pd.Series(fit.forecast(horizon_steps), dtype=float)
    except Exception as exc:  # pragma: no cover - defensive on statsmodels convergence quirks.
        logger.warning("Stage-5 ARIMA(%s) benchmark failed: %s", order, exc)
        return None
    return forecast.clip(lower=0.0)


def _block_bootstrap_sample_indices(
    *,
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one moving-block bootstrap index vector for autocorrelated errors."""
    if n_obs <= 0:
        return np.empty(0, dtype=int)
    if block_length >= n_obs:
        return np.arange(n_obs, dtype=int)
    max_start = n_obs - block_length
    block_count = int(math.ceil(n_obs / block_length))
    starts = rng.integers(0, max_start + 1, size=block_count)
    sampled = np.concatenate(
        [np.arange(int(start), int(start) + block_length, dtype=int) for start in starts]
    )
    return sampled[:n_obs]


def _bootstrap_block_length_steps(
    *,
    error_delta: pd.Series,
    resolution: str,
) -> int:
    """Infer a bootstrap block length from the observed paired-error autocorrelation."""
    step_seconds = _resolution_step_seconds(resolution)
    min_depth = max(
        1,
        int(
            math.ceil(
                int(MODELING_PERFORMANCE_EVALUATION["bootstrap_min_block_minutes"]) * 60 / step_seconds
            )
        ),
    )
    max_depth = max(
        min_depth,
        int(
            math.floor(
                int(MODELING_PERFORMANCE_EVALUATION["bootstrap_max_block_minutes"]) * 60 / step_seconds
            )
        ),
    )
    max_depth = min(max_depth, max(1, len(error_delta) - 1))
    min_depth = min(min_depth, max_depth)
    if len(error_delta) <= max(
        12,
        min_depth + int(MODELING_PERFORMANCE_EVALUATION["bootstrap_consecutive_insignificant"]),
    ):
        return int(max(1, min(max_depth, min_depth)))
    confidence_level = float(MODELING_PERFORMANCE_EVALUATION["bootstrap_confidence_level"])
    significance_level = max(1e-6, 1.0 - confidence_level)
    return int(
        optimal_acf_depth(
            error_delta,
            significance_level=significance_level,
            min_depth=min_depth,
            max_depth=max_depth,
            consecutive_insignificant=int(
                MODELING_PERFORMANCE_EVALUATION["bootstrap_consecutive_insignificant"]
            ),
        )
    )


def _build_holdout_baseline_predictions(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    aligned: pd.DataFrame,
    resolution: str,
) -> dict[str, pd.Series]:
    """Collect the holdout baseline prediction families scored alongside the promoted model."""
    aligned_index = aligned.index
    baseline_predictions: dict[str, pd.Series] = {
        "persistence": aligned["y_persist"].astype(float).copy()
    }
    aligned_eval = eval_df.loc[aligned_index].copy()
    direct_baselines = {
        "previous_day": "previous_day_load",
        "avg_workday": "avg_workday_baseline",
        "anchored_workday": "anchored_workday_baseline",
    }
    for baseline_label, column_name in direct_baselines.items():
        if column_name not in aligned_eval.columns:
            continue
        prediction = pd.to_numeric(aligned_eval[column_name], errors="coerce")
        if prediction.notna().sum() <= 0:
            continue
        baseline_predictions[baseline_label] = prediction.astype(float)

    if "holt_damped" in MODELING_PERFORMANCE_EVALUATION["classical_benchmarks"]:
        holt_forecast = _holt_damped_forecast(
            train_df[TARGET_COLUMN],
            horizon_steps=int(len(aligned_index)),
        )
        if holt_forecast is not None and len(holt_forecast) == len(aligned_index):
            holt_forecast.index = aligned_index
            baseline_predictions["holt_damped"] = holt_forecast.astype(float)
    if "arima" in MODELING_PERFORMANCE_EVALUATION["classical_benchmarks"]:
        arima_fc = _arima_forecast(
            train_df[TARGET_COLUMN],
            horizon_steps=int(len(aligned_index)),
        )
        if arima_fc is not None and len(arima_fc) == len(aligned_index):
            arima_fc.index = aligned_index
            baseline_predictions["arima"] = arima_fc.astype(float)
    return baseline_predictions


def _holdout_summary_row(
    *,
    candidate_label: str,
    candidate_type: str,
    resolution: str,
    feature_set: str,
    model_label: str,
    target_mode: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    n_eval_total: int,
    persistence_mae: float,
) -> dict[str, Any]:
    """Build one holdout summary row with raw and percentage error metrics."""
    metrics = compute_regression_metrics(y_true, y_pred, n_total=int(n_eval_total))
    return {
        "candidate_label": candidate_label,
        "candidate_type": candidate_type,
        "resolution": resolution,
        "feature_set": feature_set,
        "model_label": model_label,
        "target_mode": target_mode,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mae_pct": float(metrics["mae_pct"]),
        "rmse_pct": float(metrics["rmse_pct"]),
        "mae_ratio_to_persistence": mae_ratio(float(metrics["mae"]), persistence_mae),
        "coverage": float(metrics["coverage"]),
        "n_eval": int(metrics["n_eval"]),
        "n_eval_total": int(n_eval_total),
    }


def _build_holdout_predictions_frame(
    *,
    eval_df: pd.DataFrame,
    aligned: pd.DataFrame,
    candidate_label: str,
    candidate_predictions: pd.Series,
    baseline_predictions: dict[str, pd.Series],
    extra_columns: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Persist the exact holdout path used by inference, plots, and docs."""
    aligned_index = aligned.index
    frame = pd.DataFrame(index=aligned_index)
    if "timestamp" in eval_df.columns:
        frame["timestamp"] = pd.to_datetime(eval_df.loc[aligned_index, "timestamp"], errors="coerce")
    frame["y_true"] = aligned["y_true"].astype(float).to_numpy()
    prediction_columns = {
        candidate_label: _prediction_column_name(candidate_label),
    }
    frame[prediction_columns[candidate_label]] = candidate_predictions.reindex(aligned_index).astype(float).to_numpy()
    for baseline_label, series in baseline_predictions.items():
        prediction_columns[baseline_label] = _prediction_column_name(baseline_label)
        frame[prediction_columns[baseline_label]] = series.reindex(aligned_index).astype(float).to_numpy()

    for column in MODELING_PERFORMANCE_EVALUATION["segment_columns"]:
        if column in eval_df.columns:
            frame[column] = eval_df.loc[aligned_index, column].to_numpy()
    if extra_columns:
        for column, value in extra_columns.items():
            if isinstance(value, pd.Series):
                frame[column] = value.reindex(aligned_index).to_numpy()
            else:
                frame[column] = value
    return frame.reset_index(drop=True), prediction_columns


def _stage5_surface_candidate_predictions(
    *,
    aligned: pd.DataFrame,
    eval_df: pd.DataFrame,
    resolution: str,
    feature_set: str,
    model_label: str,
    target_mode: str,
    blend_config: BlendConfig | BucketBlendConfig | SigmoidBucketBlendConfig | None,
    n_eval_total: int,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame | None]:
    """Materialize the promoted candidate predictions for one Stage-5 evaluation surface."""
    candidate_predictions = aligned["y_pred"].copy()
    decisions: pd.DataFrame | None = None
    blend_metrics: dict[str, Any] | None = None
    blend_policy_kind = _stage5_blend_policy_kind(target_mode)
    if blend_policy_kind == "sigmoid":
        if blend_config is None:
            raise ValueError("Blend promotion selected but no blend configuration is available.")
        if not isinstance(blend_config, BlendConfig):
            raise ValueError("Sigmoid blend promotion selected but the saved config is not a BlendConfig.")
        blend_metrics, decisions = _apply_blend_policy(
            aligned=aligned,
            blend_config=blend_config,
            n_eval_total=n_eval_total,
        )
        if decisions is not None and not decisions.empty:
            decisions = decisions.copy()
            decisions["resolution"] = resolution
            decisions["feature_set"] = feature_set
            decisions["model_label"] = model_label
            decisions["target_mode"] = target_mode
            candidate_predictions = (
                decisions.set_index("row_index")["blend_pred"].reindex(aligned.index).astype(float)
            )
    elif blend_policy_kind == "bucket":
        if blend_config is None:
            raise ValueError("Bucket blend promotion selected but no bucket configuration is available.")
        if not isinstance(blend_config, (BucketBlendConfig, SigmoidBucketBlendConfig)):
            raise ValueError(
                "Bucket blend promotion selected but the saved config is not a recognized bucket config."
            )
        bucket_aligned = aligned.loc[:, ["y_true", "y_persist", "y_pred"]].copy()
        bucket_runtime_config: BucketBlendConfig
        if isinstance(blend_config, SigmoidBucketBlendConfig):
            _, inner_decisions = _apply_blend_policy(
                aligned=aligned,
                blend_config=blend_config.blend_config,
                n_eval_total=n_eval_total,
            )
            bucket_aligned["y_pred"] = (
                inner_decisions.set_index("row_index")["blend_pred"].reindex(aligned.index).astype(float).to_numpy()
            )
            bucket_runtime_config = blend_config.bucket_config
        else:
            bucket_runtime_config = blend_config
        blend_metrics, decisions = _apply_bucket_blend_policy(
            aligned=bucket_aligned,
            timestamps=pd.to_datetime(eval_df.loc[aligned.index, "timestamp"], errors="raise"),
            bucket_config=bucket_runtime_config,
            n_eval_total=n_eval_total,
        )
        if decisions is not None and not decisions.empty:
            decisions = decisions.copy()
            decisions["resolution"] = resolution
            decisions["feature_set"] = feature_set
            decisions["model_label"] = model_label
            decisions["target_mode"] = target_mode
            decisions["blend_base_policy_kind"] = (
                "sigmoid" if isinstance(blend_config, SigmoidBucketBlendConfig) else "raw"
            )
            if isinstance(blend_config, SigmoidBucketBlendConfig):
                decisions["blend_window"] = int(blend_config.blend_config.window)
                decisions["blend_sharpness"] = float(blend_config.blend_config.sharpness)
                decisions["blend_min_weight"] = float(blend_config.blend_config.min_weight)
                decisions["blend_max_weight"] = float(blend_config.blend_config.max_weight)
            decisions["blend_bucket_size_minutes"] = int(bucket_runtime_config.bucket_size_minutes)
            decisions["blend_bucket_cycle_minutes"] = int(bucket_runtime_config.cycle_minutes)
            decisions["blend_bucket_weights_json"] = json.dumps(
                bucket_runtime_config.weight_map(),
                sort_keys=True,
            )
            candidate_predictions = (
                decisions.set_index("row_index")["blend_pred"].reindex(aligned.index).astype(float)
            )

    metrics = (
        blend_metrics
        if blend_metrics is not None
        else compute_regression_metrics(
            aligned["y_true"],
            aligned["y_pred"],
            n_total=int(n_eval_total),
        )
    )
    return candidate_predictions, metrics, decisions


def _prediction_surface_summary(
    *,
    prediction_frame: pd.DataFrame,
    prediction_columns: dict[str, str],
    candidate_label: str,
    resolution: str,
    feature_set: str,
    model_label: str,
    target_mode: str,
) -> tuple[pd.DataFrame, str, float, float, float]:
    """Summarize one stitched Stage-5 prediction surface against the baseline family."""
    if prediction_frame.empty:
        return pd.DataFrame(), "", float("nan"), float("nan"), float("nan")
    if candidate_label not in prediction_columns or "persistence" not in prediction_columns:
        raise ValueError("Prediction surface summary requires candidate and persistence prediction columns.")
    y_true = prediction_frame["y_true"].astype(float)
    persistence_label = "persistence"
    persistence_metrics = compute_regression_metrics(
        y_true,
        prediction_frame[prediction_columns[persistence_label]].astype(float),
        n_total=len(prediction_frame),
    )
    persistence_mae = float(persistence_metrics["mae"])
    holdout_rows: list[dict[str, Any]] = [
        _holdout_summary_row(
            candidate_label=candidate_label,
            candidate_type="promoted_learned",
            resolution=resolution,
            feature_set=feature_set,
            model_label=model_label,
            target_mode=target_mode,
            y_true=y_true,
            y_pred=prediction_frame[prediction_columns[candidate_label]].astype(float),
            n_eval_total=len(prediction_frame),
            persistence_mae=persistence_mae,
        )
    ]
    baseline_meta = {
        "persistence": ("baseline", "persistence", "raw"),
        "previous_day": ("baseline", "previous_day", "raw"),
        "avg_workday": ("baseline", "avg_workday", "raw"),
        "anchored_workday": ("baseline", "anchored_workday", "raw"),
        "holt_damped": ("baseline", "holt_damped", "forecast"),
        "arima": ("baseline", "arima", "forecast"),
    }
    for baseline_label, column_name in prediction_columns.items():
        if baseline_label == candidate_label:
            continue
        feature_set_label, model_label_baseline, target_mode_baseline = baseline_meta.get(
            baseline_label,
            ("baseline", baseline_label, "raw"),
        )
        holdout_rows.append(
            _holdout_summary_row(
                candidate_label=baseline_label,
                candidate_type="baseline",
                resolution=resolution,
                feature_set=feature_set_label,
                model_label=model_label_baseline,
                target_mode=target_mode_baseline,
                y_true=y_true,
                y_pred=prediction_frame[column_name].astype(float),
                n_eval_total=len(prediction_frame),
                persistence_mae=persistence_mae,
            )
        )
    summary = pd.DataFrame(holdout_rows)
    baseline_summary = summary.loc[summary["candidate_type"].astype("string").eq("baseline")].copy()
    best_baseline_row = baseline_summary.sort_values(["mae", "rmse", "candidate_label"], kind="stable").iloc[0]
    best_baseline_label = str(best_baseline_row["candidate_label"])
    best_baseline_mae = float(best_baseline_row["mae"])
    best_baseline_mae_pct = float(best_baseline_row["mae_pct"])
    summary["mae_ratio_to_best_baseline"] = summary["mae"].astype(float).map(
        lambda value: mae_ratio(float(value), best_baseline_mae)
    )
    summary["is_best_baseline"] = summary["candidate_label"].astype("string").eq(best_baseline_label)
    return summary, best_baseline_label, best_baseline_mae, best_baseline_mae_pct, persistence_mae


def _derive_supplemental_actual_load_band(
    prediction_frame: pd.DataFrame,
) -> pd.Series:
    """Bucket the Stage-5 supplemental surface into high-load vs typical-load windows."""
    actual = pd.to_numeric(prediction_frame.get("y_true"), errors="coerce").astype(float)
    quantile = float(MODELING_PERFORMANCE_EVALUATION["supplemental_load_band_quantile"])
    threshold = float(actual.quantile(quantile)) if actual.notna().any() else float("nan")
    band = pd.Series("typical_load", index=prediction_frame.index, dtype="string")
    if np.isfinite(threshold):
        band.loc[actual >= threshold] = "high_load"
    band.loc[actual.isna()] = "unknown"
    return band


def _derive_supplemental_actual_ramp_band(
    prediction_frame: pd.DataFrame,
) -> pd.Series:
    """Bucket the Stage-5 supplemental surface into high-ramp vs stable windows."""
    actual = pd.to_numeric(prediction_frame.get("y_true"), errors="coerce").astype(float)
    abs_delta = actual.diff().abs()
    quantile = float(MODELING_PERFORMANCE_EVALUATION["supplemental_ramp_band_quantile"])
    threshold = float(abs_delta.quantile(quantile)) if abs_delta.notna().any() else float("nan")
    band = pd.Series("stable_ramp", index=prediction_frame.index, dtype="string")
    if np.isfinite(threshold):
        band.loc[abs_delta >= threshold] = "high_ramp"
    band.loc[abs_delta.isna()] = "unknown"
    return band


def _prediction_surface_segment_evaluation(
    *,
    prediction_frame: pd.DataFrame,
    prediction_columns: dict[str, str],
    candidate_label: str,
    best_baseline_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate one prediction surface by operating segments and diagnostic buckets."""
    if prediction_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    working = prediction_frame.copy()
    if "operating_regime" not in working.columns:
        operating_regime = _derive_operating_regime(working)
        if operating_regime is not None:
            working["operating_regime"] = operating_regime.astype("string")
    if "actual_load_band" not in working.columns:
        working["actual_load_band"] = _derive_supplemental_actual_load_band(working)
    if "actual_ramp_band" not in working.columns:
        working["actual_ramp_band"] = _derive_supplemental_actual_ramp_band(working)

    segment_columns = [
        *[column for column in MODELING_PERFORMANCE_EVALUATION["segment_columns"] if column in working.columns],
        *[
            column
            for column in ("operating_regime", "actual_load_band", "actual_ramp_band", "evaluation_surface")
            if column in working.columns
        ],
    ]
    y_true = working["y_true"].astype(float)
    candidate_pred = working[prediction_columns[candidate_label]].astype(float)
    persistence_pred = working[prediction_columns["persistence"]].astype(float)
    best_baseline_pred = working[prediction_columns[best_baseline_label]].astype(float)
    segment_rows: list[dict[str, Any]] = []
    for segment_column in segment_columns:
        grouped = working.groupby(segment_column, dropna=False)
        for segment_value, group in grouped:
            candidate_metrics = compute_regression_metrics(
                y_true.loc[group.index],
                candidate_pred.loc[group.index],
                n_total=int(len(group)),
            )
            persistence_metrics = compute_regression_metrics(
                y_true.loc[group.index],
                persistence_pred.loc[group.index],
                n_total=int(len(group)),
            )
            best_baseline_metrics = compute_regression_metrics(
                y_true.loc[group.index],
                best_baseline_pred.loc[group.index],
                n_total=int(len(group)),
            )
            segment_rows.append(
                {
                    "segment_column": segment_column,
                    "segment_value": segment_value,
                    "candidate_label": candidate_label,
                    "candidate_mae": float(candidate_metrics["mae"]),
                    "candidate_mae_pct": float(candidate_metrics["mae_pct"]),
                    "persistence_mae": float(persistence_metrics["mae"]),
                    "persistence_mae_pct": float(persistence_metrics["mae_pct"]),
                    "best_baseline_label": best_baseline_label,
                    "best_baseline_mae": float(best_baseline_metrics["mae"]),
                    "best_baseline_mae_pct": float(best_baseline_metrics["mae_pct"]),
                    "candidate_mae_ratio_to_persistence": mae_ratio(
                        float(candidate_metrics["mae"]),
                        float(persistence_metrics["mae"]),
                    ),
                    "candidate_mae_ratio_to_best_baseline": mae_ratio(
                        float(candidate_metrics["mae"]),
                        float(best_baseline_metrics["mae"]),
                    ),
                    "rows": int(len(group)),
                }
            )
    segment_evaluation = pd.DataFrame(segment_rows)
    operating_regime_evaluation = (
        segment_evaluation.loc[
            segment_evaluation["segment_column"].astype("string").eq("operating_regime")
        ].reset_index(drop=True)
        if not segment_evaluation.empty
        else pd.DataFrame()
    )
    coverage_segments, coverage_summary = _build_holdout_coverage_summary(
        holdout_frame=working,
        segment_columns=segment_columns,
    )
    return segment_evaluation, operating_regime_evaluation, coverage_segments, coverage_summary


def _bootstrap_comparison_rows(
    *,
    y_true: pd.Series,
    candidate_pred: pd.Series,
    baseline_pred: pd.Series,
    candidate_label: str,
    baseline_label: str,
    comparison_type: str,
    resolution: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Return bootstrap CI and paired-significance rows for one holdout comparison."""
    valid = pd.DataFrame(
        {
            "y_true": pd.to_numeric(y_true, errors="coerce"),
            "candidate_pred": pd.to_numeric(candidate_pred, errors="coerce"),
            "baseline_pred": pd.to_numeric(baseline_pred, errors="coerce"),
        }
    ).dropna()
    if valid.empty:
        return []

    actual = valid["y_true"].to_numpy(dtype=float)
    candidate = valid["candidate_pred"].to_numpy(dtype=float)
    baseline = valid["baseline_pred"].to_numpy(dtype=float)
    abs_error_delta = np.abs(actual - candidate) - np.abs(actual - baseline)
    block_length_steps = _bootstrap_block_length_steps(
        error_delta=pd.Series(abs_error_delta),
        resolution=resolution,
    )
    block_length_minutes = float(block_length_steps * _resolution_step_seconds(resolution) / 60.0)
    bootstrap_samples = int(MODELING_PERFORMANCE_EVALUATION["bootstrap_samples"])
    confidence_level = float(MODELING_PERFORMANCE_EVALUATION["bootstrap_confidence_level"])
    alpha = 1.0 - confidence_level

    observed_candidate = compute_regression_metrics(actual, candidate, n_total=len(actual))
    observed_baseline = compute_regression_metrics(actual, baseline, n_total=len(actual))

    distributions: dict[str, list[float]] = {
        "candidate_mae": [],
        "baseline_mae": [],
        "delta_mae": [],
        "candidate_mae_pct": [],
        "baseline_mae_pct": [],
        "delta_mae_pct": [],
        "candidate_rmse": [],
        "baseline_rmse": [],
        "delta_rmse": [],
        "candidate_rmse_pct": [],
        "baseline_rmse_pct": [],
        "delta_rmse_pct": [],
    }

    rng = np.random.default_rng(int(seed))
    for _ in range(bootstrap_samples):
        sample_idx = _block_bootstrap_sample_indices(
            n_obs=len(actual),
            block_length=block_length_steps,
            rng=rng,
        )
        candidate_sample = compute_regression_metrics(
            actual[sample_idx],
            candidate[sample_idx],
            n_total=len(sample_idx),
        )
        baseline_sample = compute_regression_metrics(
            actual[sample_idx],
            baseline[sample_idx],
            n_total=len(sample_idx),
        )
        for metric_name in ("mae", "rmse"):
            distributions[f"candidate_{metric_name}"].append(float(candidate_sample[metric_name]))
            distributions[f"baseline_{metric_name}"].append(float(baseline_sample[metric_name]))
            distributions[f"delta_{metric_name}"].append(
                float(candidate_sample[metric_name]) - float(baseline_sample[metric_name])
            )
            pct_key = f"{metric_name}_pct"
            distributions[f"candidate_{pct_key}"].append(float(candidate_sample[pct_key]))
            distributions[f"baseline_{pct_key}"].append(float(baseline_sample[pct_key]))
            distributions[f"delta_{pct_key}"].append(
                float(candidate_sample[pct_key]) - float(baseline_sample[pct_key])
            )

    rows: list[dict[str, Any]] = []
    metric_labels = {
        "mae": "Mean absolute error",
        "rmse": "Root mean squared error",
    }
    for metric_name in ("mae", "rmse"):
        candidate_metric = float(observed_candidate[metric_name])
        baseline_metric = float(observed_baseline[metric_name])
        candidate_metric_pct = float(observed_candidate[f"{metric_name}_pct"])
        baseline_metric_pct = float(observed_baseline[f"{metric_name}_pct"])
        delta_distribution = np.asarray(distributions[f"delta_{metric_name}"], dtype=float)
        delta_pct_distribution = np.asarray(distributions[f"delta_{metric_name}_pct"], dtype=float)
        one_sided_p = float(np.mean(delta_distribution >= 0.0))
        two_sided_p = float(min(1.0, 2.0 * min(np.mean(delta_distribution >= 0.0), np.mean(delta_distribution <= 0.0))))
        rows.append(
            {
                "comparison_type": comparison_type,
                "candidate_label": candidate_label,
                "baseline_label": baseline_label,
                "metric_name": metric_name,
                "metric_label": metric_labels[metric_name],
                "candidate_metric": candidate_metric,
                "candidate_metric_pct": candidate_metric_pct,
                "candidate_metric_ci_low": float(
                    np.quantile(np.asarray(distributions[f"candidate_{metric_name}"], dtype=float), alpha / 2.0)
                ),
                "candidate_metric_ci_high": float(
                    np.quantile(
                        np.asarray(distributions[f"candidate_{metric_name}"], dtype=float),
                        1.0 - alpha / 2.0,
                    )
                ),
                "candidate_metric_pct_ci_low": float(
                    np.quantile(np.asarray(distributions[f"candidate_{metric_name}_pct"], dtype=float), alpha / 2.0)
                ),
                "candidate_metric_pct_ci_high": float(
                    np.quantile(
                        np.asarray(distributions[f"candidate_{metric_name}_pct"], dtype=float),
                        1.0 - alpha / 2.0,
                    )
                ),
                "baseline_metric": baseline_metric,
                "baseline_metric_pct": baseline_metric_pct,
                "baseline_metric_ci_low": float(
                    np.quantile(np.asarray(distributions[f"baseline_{metric_name}"], dtype=float), alpha / 2.0)
                ),
                "baseline_metric_ci_high": float(
                    np.quantile(
                        np.asarray(distributions[f"baseline_{metric_name}"], dtype=float),
                        1.0 - alpha / 2.0,
                    )
                ),
                "baseline_metric_pct_ci_low": float(
                    np.quantile(np.asarray(distributions[f"baseline_{metric_name}_pct"], dtype=float), alpha / 2.0)
                ),
                "baseline_metric_pct_ci_high": float(
                    np.quantile(
                        np.asarray(distributions[f"baseline_{metric_name}_pct"], dtype=float),
                        1.0 - alpha / 2.0,
                    )
                ),
                "delta_metric": candidate_metric - baseline_metric,
                "delta_metric_pct": candidate_metric_pct - baseline_metric_pct,
                "delta_metric_ci_low": float(np.quantile(delta_distribution, alpha / 2.0)),
                "delta_metric_ci_high": float(np.quantile(delta_distribution, 1.0 - alpha / 2.0)),
                "delta_metric_pct_ci_low": float(np.quantile(delta_pct_distribution, alpha / 2.0)),
                "delta_metric_pct_ci_high": float(
                    np.quantile(delta_pct_distribution, 1.0 - alpha / 2.0)
                ),
                "candidate_metric_ratio_to_baseline": mae_ratio(candidate_metric, baseline_metric),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_confidence_level": confidence_level,
                "bootstrap_block_length_steps": int(block_length_steps),
                "bootstrap_block_length_minutes": block_length_minutes,
                "bootstrap_method": "moving_block_bootstrap",
                "one_sided_p_candidate_lt_baseline": one_sided_p,
                "two_sided_p": two_sided_p,
                "candidate_better_than_baseline": bool(candidate_metric < baseline_metric),
                "delta_ci_excludes_zero": bool(
                    float(np.quantile(delta_distribution, 1.0 - alpha / 2.0)) < 0.0
                    or float(np.quantile(delta_distribution, alpha / 2.0)) > 0.0
                ),
                "n_eval": int(len(actual)),
            }
        )
    return rows


def _build_holdout_inference(
    *,
    prediction_frame: pd.DataFrame,
    prediction_columns: dict[str, str],
    candidate_label: str,
    best_baseline_label: str,
    resolution: str,
) -> pd.DataFrame:
    """Compute bootstrap confidence intervals and paired significance tests for holdout comparisons."""
    comparisons: list[tuple[str, str, str]] = [
        (candidate_label, "persistence", "candidate_vs_persistence")
    ]
    for baseline_label in ("previous_day", "avg_workday", "anchored_workday", "holt_damped", "arima"):
        if baseline_label in prediction_columns:
            comparisons.append((baseline_label, "persistence", "baseline_vs_persistence"))
    if best_baseline_label != "persistence":
        comparisons.append((candidate_label, best_baseline_label, "candidate_vs_best_baseline"))

    y_true = prediction_frame["y_true"].astype(float)
    rows: list[dict[str, Any]] = []
    for seed_offset, (candidate_cmp_label, baseline_cmp_label, comparison_type) in enumerate(comparisons):
        candidate_column = prediction_columns.get(candidate_cmp_label)
        baseline_column = prediction_columns.get(baseline_cmp_label)
        if candidate_column is None or baseline_column is None:
            continue
        rows.extend(
            _bootstrap_comparison_rows(
                y_true=y_true,
                candidate_pred=prediction_frame[candidate_column].astype(float),
                baseline_pred=prediction_frame[baseline_column].astype(float),
                candidate_label=candidate_cmp_label,
                baseline_label=baseline_cmp_label,
                comparison_type=comparison_type,
                resolution=resolution,
                seed=int(MODELING_PERFORMANCE_EVALUATION["importance_random_state"]) + seed_offset,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["comparison_type", "metric_name", "candidate_label", "baseline_label"],
        kind="stable",
    ).reset_index(drop=True)


def _make_absolute_mae_scorer(
    *,
    target_mode: str,
    feature_names: list[str],
) -> Any:
    """Build a scorer that evaluates permutation importance on absolute-load MAE."""
    lag_index = feature_names.index("lag_1") if "lag_1" in feature_names else None

    def _score(estimator: Any, x_eval: Any, y_true: Any) -> float:
        pred = np.asarray(estimator.predict(x_eval), dtype=float)
        if target_mode == "residual":
            if isinstance(x_eval, pd.DataFrame):
                lag_values = pd.to_numeric(x_eval["lag_1"], errors="coerce").to_numpy(dtype=float)
            elif lag_index is not None:
                lag_values = np.asarray(x_eval, dtype=float)[:, lag_index]
            else:
                raise ValueError("Residual permutation importance requires lag_1 in the evaluation frame.")
            pred = lag_values + pred
        y_true_arr = np.asarray(y_true, dtype=float)
        return -float(np.mean(np.abs(y_true_arr - pred)))

    return _score


def _compute_holdout_feature_importance(
    *,
    model: Any,
    x_eval: pd.DataFrame,
    y_true: pd.Series,
    target_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Compute permutation importance for the learned holdout challenger."""
    if x_eval.empty:
        return pd.DataFrame(), None
    scoring = _make_absolute_mae_scorer(
        target_mode=target_mode,
        feature_names=list(x_eval.columns.astype(str)),
    )
    result = permutation_importance(
        model,
        x_eval,
        y_true.astype(float),
        n_repeats=int(MODELING_PERFORMANCE_EVALUATION["importance_repeats"]),
        random_state=int(MODELING_PERFORMANCE_EVALUATION["importance_random_state"]),
        scoring=scoring,
    )
    importance_df = pd.DataFrame(
        {
            "feature": list(x_eval.columns.astype(str)),
            "importance_mean": result.importances_mean.astype(float),
            "importance_std": result.importances_std.astype(float),
        }
    ).sort_values(["importance_mean", "importance_std", "feature"], ascending=[False, True, True], kind="stable")
    positive_total = float(importance_df["importance_mean"].clip(lower=0.0).sum())
    if positive_total > 0:
        importance_df["importance_share"] = importance_df["importance_mean"].clip(lower=0.0) / positive_total
        importance_df["cumulative_share"] = importance_df["importance_share"].cumsum()
    else:
        importance_df["importance_share"] = 0.0
        importance_df["cumulative_share"] = 0.0
    top_n = int(MODELING_PERFORMANCE_EVALUATION["importance_max_features"])
    top_df = importance_df.head(top_n).reset_index(drop=True)
    summary = {
        "method": "permutation_importance_on_holdout_mae",
        "target_mode": target_mode,
        "feature_count": int(len(importance_df)),
        "top_feature": str(top_df.iloc[0]["feature"]) if not top_df.empty else None,
        "top_feature_importance": float(top_df.iloc[0]["importance_mean"]) if not top_df.empty else None,
        "top_5_cumulative_share": float(top_df.head(5)["importance_share"].sum()) if not top_df.empty else 0.0,
        "top_10_cumulative_share": float(top_df.head(10)["importance_share"].sum()) if not top_df.empty else 0.0,
    }
    return top_df, summary


def _compute_holdout_shap_importance(
    *,
    model: Any,
    x_eval: pd.DataFrame,
    candidate_label: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Compute SHAP values for the promoted holdout candidate."""
    try:
        import shap  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("shap package not installed; skipping SHAP analysis.")
        return pd.DataFrame(), None

    if x_eval.empty:
        return pd.DataFrame(), None

    try:
        if hasattr(model, "n_iter_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, x_eval)
        shap_values = explainer.shap_values(x_eval)
    except Exception as exc:  # pragma: no cover - defensive on SHAP internals.
        logger.warning("SHAP computation failed: %s", exc)
        return pd.DataFrame(), None

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": list(x_eval.columns.astype(str)),
            "mean_abs_shap": mean_abs.astype(float),
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)

    total = float(shap_df["mean_abs_shap"].sum())
    if total > 0:
        shap_df["shap_share"] = shap_df["mean_abs_shap"] / total
        shap_df["cumulative_share"] = shap_df["shap_share"].cumsum()
    else:
        shap_df["shap_share"] = 0.0
        shap_df["cumulative_share"] = 0.0

    summary = {
        "method": "shap_mean_absolute",
        "explainer_type": "tree" if hasattr(model, "n_iter_") else "linear",
        "candidate_label": candidate_label,
        "feature_count": int(len(shap_df)),
        "top_feature": str(shap_df.iloc[0]["feature"]) if not shap_df.empty else None,
        "top_feature_shap": float(shap_df.iloc[0]["mean_abs_shap"]) if not shap_df.empty else None,
        "top_5_cumulative_share": float(shap_df.head(5)["shap_share"].sum()) if not shap_df.empty else 0.0,
    }
    return shap_df, summary


def _write_shap_importance_figure(
    shap_df: pd.DataFrame, candidate_label: str, output_path: Path, *, top_n: int = 15
) -> None:
    """Write a horizontal bar chart of mean |SHAP| values."""
    plot_df = shap_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(plot_df))))
    ax.barh(plot_df["feature"].astype(str), plot_df["mean_abs_shap"].astype(float))
    ax.set_xlabel("Mean |SHAP value| (W)")
    ax.set_title(f"SHAP Feature Importance: {candidate_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    validate_png_artifact(output_path)


def _run_fold_metrics(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]],
    selected_feature_sets: list[str],
    feature_sets: dict[str, list[str]],
    selected_models: list[ModelSpec],
    resolution: str,
    include_residual: bool,
    steps_per_day: int,
) -> tuple[pd.DataFrame, ParallelPlan]:
    """Execute the Stage-5 fold grid and return the collected metric table."""
    target_modes = ["raw", "residual"] if include_residual else ["raw"]
    selected_model_catalog = {spec.model_label: spec for spec in selected_models}
    include_hgb_coordinate_search = any(
        model.model_label.startswith("hgb-coordinate") for model in selected_models
    )
    tasks = [
        FoldMetricTask(
            fold=fold,
            feature_set=feature_set,
            model_label=model_spec.model_label,
            target_mode=target_mode,
        )
        for fold in folds
        for feature_set in selected_feature_sets
        for model_spec in selected_models
        for target_mode in target_modes
    ]
    worker = partial(
        _run_fold_metric_task,
        gold=gold,
        feature_sets=feature_sets,
        resolution=resolution,
        steps_per_day=steps_per_day,
        model_catalog=selected_model_catalog,
        include_hgb_coordinate_search=include_hgb_coordinate_search,
    )
    rows, parallel_plan = run_stage_jobs(
        "performance",
        tasks,
        worker=worker,
        logger_instance=logger,
    )
    materialized_rows = [row for row in rows if row is not None]
    metrics_fold = pd.DataFrame(materialized_rows)
    if metrics_fold.empty:
        return metrics_fold, parallel_plan
    metrics_fold = metrics_fold.sort_values(
        ["fold", "feature_set", "model_label", "target_mode"],
        kind="stable",
    ).reset_index(drop=True)
    return metrics_fold, parallel_plan


def _run_fold_metric_task(
    task: FoldMetricTask,
    *,
    gold: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    resolution: str,
    steps_per_day: int,
    model_catalog: dict[str, ModelSpec],
    include_hgb_coordinate_search: bool,
) -> dict[str, Any] | None:
    """Run one fold/feature/model/target evaluation unit."""
    fold = task.fold
    model_spec = model_catalog.get(task.model_label)
    if model_spec is None:
        catalog = build_model_catalog(
            include_hgb_coordinate_search=include_hgb_coordinate_search,
            include_hgb_frontier=True,
        )
        model_spec = catalog[task.model_label]
    train_df = gold.loc[
        gold["day_idx"].between(fold["train_start_day"], fold["train_end_day"])
    ].copy()
    val_df = gold.loc[
        gold["day_idx"].between(fold["val_start_day"], fold["val_end_day"])
    ].copy()
    n_eval_total = _expected_steps_for_day_range(
        fold["val_start_day"],
        fold["val_end_day"],
        steps_per_day,
    )
    metrics = _fit_and_evaluate(
        train_df=train_df,
        eval_df=val_df,
        feature_cols=feature_sets[task.feature_set],
        model_spec=model_spec,
        target_mode=task.target_mode,
        n_eval_total=n_eval_total,
    )
    if metrics is None:
        return None
    return {
        "fold": int(fold["fold"]),
        "resolution": resolution,
        "feature_set": task.feature_set,
        "model": model_spec.family,
        "model_label": model_spec.model_label,
        "params": json.dumps(model_spec.params, sort_keys=True),
        "target_mode": task.target_mode,
        **metrics,
        "train_start_day": fold["train_start_day"],
        "train_end_day": fold["train_end_day"],
        "val_start_day": fold["val_start_day"],
        "val_end_day": fold["val_end_day"],
    }


def _evaluate_blend_candidate(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]],
    selection_scoreboard: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    model_catalog: dict[str, ModelSpec],
    resolution: str,
    base_blend_config: BlendConfig,
    steps_per_day: int,
    preferred_candidate: dict[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any] | None,
    BlendConfig | BucketBlendConfig | SigmoidBucketBlendConfig | None,
    pd.DataFrame,
]:
    """Search guarded blend configs across eligible learned candidates."""
    blend_candidates = selection_scoreboard[
        selection_scoreboard["model_label"].astype("string").isin(pd.Index(model_catalog.keys()).astype("string"))
    ].copy()
    blend_candidates = blend_candidates.sort_values(
        ["fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae"],
        ascending=[True, True, True],
    )
    if blend_candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()

    candidate_pool = blend_candidates.head(3).copy()
    diversified_rows: list[pd.DataFrame] = [candidate_pool]
    for feature_set in blend_candidates["feature_set"].astype("string").dropna().unique().tolist():
        for target_mode in ("raw", "residual"):
            feature_mode_rows = blend_candidates.loc[
                blend_candidates["feature_set"].astype("string").eq(str(feature_set))
                & blend_candidates["target_mode"].astype("string").eq(target_mode)
            ].head(1)
            if not feature_mode_rows.empty:
                diversified_rows.append(feature_mode_rows)
    candidate_pool = pd.concat(diversified_rows, ignore_index=True)
    candidate_pool = candidate_pool.drop_duplicates(
        subset=["feature_set", "model_label", "target_mode"],
        keep="first",
    )
    if preferred_candidate is not None:
        preferred_model = str(preferred_candidate.get("model_label", ""))
        preferred_feature_set = str(preferred_candidate.get("feature_set", ""))
        preferred_target_mode = str(preferred_candidate.get("target_mode", ""))
        preferred_rows = blend_candidates[
            (blend_candidates["model_label"] == preferred_model)
            & (blend_candidates["feature_set"] == preferred_feature_set)
            & (blend_candidates["target_mode"] == preferred_target_mode)
        ]
        if not preferred_rows.empty:
            candidate_pool = pd.concat([preferred_rows, candidate_pool], ignore_index=True)
            candidate_pool = candidate_pool.drop_duplicates(
                subset=["feature_set", "model_label", "target_mode"],
                keep="first",
            )
    blend_candidates = candidate_pool.sort_values(
        ["fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    window_values = sorted(
        {
            max(10, int(round(base_blend_config.window * multiplier)))
            for multiplier in MODELING_PERFORMANCE_BLEND_SEARCH["window_multipliers"]
        }
        | {int(base_blend_config.window)}
    )
    sharpness_values = sorted(
        {
            max(1.0, float(base_blend_config.sharpness * multiplier))
            for multiplier in MODELING_PERFORMANCE_BLEND_SEARCH["sharpness_multipliers"]
        }
        | {float(base_blend_config.sharpness)}
    )
    weight_pairs = [(0.0, 1.0), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8)]
    candidate_configs: list[BlendConfig] = []
    for window in window_values:
        for sharpness in sharpness_values:
            for min_weight, max_weight in weight_pairs:
                candidate_configs.append(
                    BlendConfig(
                        window=int(window),
                        sharpness=float(sharpness),
                        min_weight=float(min_weight),
                        max_weight=float(max_weight),
                    )
                )
    unique_configs: list[BlendConfig] = []
    seen = set()
    for cfg in candidate_configs:
        key = (cfg.window, cfg.sharpness, cfg.min_weight, cfg.max_weight)
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    config_results: list[dict[str, Any]] = []
    for _, candidate in blend_candidates.iterrows():
        model_label = str(candidate["model_label"])
        feature_set = str(candidate["feature_set"])
        target_mode = str(candidate["target_mode"])
        model_spec = model_catalog.get(model_label)
        if model_spec is None or feature_set not in feature_sets:
            continue
        fold_payload = _build_stage5_eval_payloads(
            gold=gold,
            folds=folds,
            feature_cols=feature_sets[feature_set],
            model_spec=model_spec,
            target_mode=target_mode,
            steps_per_day=steps_per_day,
        )
        for payload in fold_payload:
            aligned = cast(pd.DataFrame, payload["aligned"])
            raw_error = aligned["y_true"] - aligned["y_pred"]
            persist_error = aligned["y_true"] - aligned["y_persist"]
            payload["raw_ratio"] = mae_ratio(
                float(np.mean(np.abs(raw_error))),
                float(np.mean(np.abs(persist_error))),
            )
        if not fold_payload:
            continue

        candidate_result_entries: list[dict[str, Any]] = []
        for cfg in unique_configs:
            fold_metrics: list[dict[str, Any]] = []
            fold_decisions: list[pd.DataFrame] = []
            for payload in fold_payload:
                metrics, decisions = _apply_blend_policy(
                    aligned=payload["aligned"],
                    blend_config=cfg,
                    n_eval_total=int(payload["n_eval_total"]),
                )
                if not metrics:
                    continue
                fold = payload["fold_meta"]
                raw_ratio = float(payload["raw_ratio"])
                degrade_pct = ((float(metrics["mae_ratio"]) - raw_ratio) / raw_ratio * 100.0) if raw_ratio > 0 else 0.0
                fold_metrics.append(
                    {
                        "fold": int(fold["fold"]),
                        "metrics": metrics,
                        "degrade_pct": degrade_pct,
                        "fold_meta": fold,
                    }
                )
                decisions = decisions.copy()
                decisions["fold"] = int(fold["fold"])
                decisions["resolution"] = resolution
                decisions["feature_set"] = feature_set
                decisions["model_label"] = model_label
                decisions["source_target_mode"] = target_mode
                fold_decisions.append(decisions)
            if not fold_metrics:
                continue
            mean_ratio = float(np.mean([f["metrics"]["mae_ratio"] for f in fold_metrics]))
            std_ratio = float(np.std([f["metrics"]["mae_ratio"] for f in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0
            max_degrade = float(np.max([f["degrade_pct"] for f in fold_metrics]))
            candidate_result_entries.append(
                {
                    "candidate_meta": {
                        "model_label": model_label,
                        "feature_set": feature_set,
                        "target_mode": target_mode,
                    },
                    "policy_kind": "sigmoid",
                    "base_policy_kind": "raw",
                    "config": cfg,
                    "mean_ratio": mean_ratio,
                    "std_ratio": std_ratio,
                    "max_degrade_pct": max_degrade,
                    "fold_metrics": fold_metrics,
                    "decisions": fold_decisions,
                }
            )

        if (
            bool(MODELING_PERFORMANCE_BLEND_SEARCH["bucket_enabled"])
            and all(payload.get("timestamps") is not None for payload in fold_payload)
        ):
            calibration_aligned = pd.concat(
                [cast(pd.DataFrame, payload["aligned"]) for payload in fold_payload],
                axis=0,
            )
            calibration_timestamps = pd.DatetimeIndex(
                np.concatenate(
                    [
                        pd.DatetimeIndex(cast(pd.Index, payload["timestamps"])).to_numpy()
                        for payload in fold_payload
                    ]
                )
            )
            bucket_config = _calibrate_bucket_blend_config(
                aligned=calibration_aligned,
                timestamps=calibration_timestamps,
                bucket_size_minutes=int(MODELING_PERFORMANCE_BLEND_SEARCH["bucket_size_minutes"]),
                cycle_minutes=int(MODELING_PERFORMANCE_BLEND_SEARCH["bucket_cycle_minutes"]),
                candidate_weights=[
                    float(value) for value in MODELING_PERFORMANCE_BLEND_SEARCH["bucket_candidate_weights"]
                ],
            )
            if bucket_config is not None:
                fold_metrics = []
                fold_decisions = []
                for payload in fold_payload:
                    metrics, decisions = _apply_bucket_blend_policy(
                        aligned=cast(pd.DataFrame, payload["aligned"]),
                        timestamps=cast(pd.Index, payload["timestamps"]),
                        bucket_config=bucket_config,
                        n_eval_total=int(payload["n_eval_total"]),
                    )
                    if not metrics:
                        continue
                    fold = cast(dict[str, int], payload["fold_meta"])
                    raw_ratio = float(payload["raw_ratio"])
                    degrade_pct = (
                        (float(metrics["mae_ratio"]) - raw_ratio) / raw_ratio * 100.0
                    ) if raw_ratio > 0 else 0.0
                    fold_metrics.append(
                        {
                            "fold": int(fold["fold"]),
                            "metrics": metrics,
                            "degrade_pct": degrade_pct,
                            "fold_meta": fold,
                        }
                    )
                    decisions = decisions.copy()
                    decisions["fold"] = int(fold["fold"])
                    decisions["resolution"] = resolution
                    decisions["feature_set"] = feature_set
                    decisions["model_label"] = model_label
                    decisions["source_target_mode"] = target_mode
                    decisions["blend_bucket_size_minutes"] = int(bucket_config.bucket_size_minutes)
                    decisions["blend_bucket_cycle_minutes"] = int(bucket_config.cycle_minutes)
                    decisions["blend_bucket_weights_json"] = json.dumps(
                        bucket_config.weight_map(),
                        sort_keys=True,
                    )
                    fold_decisions.append(decisions)
                if fold_metrics:
                    mean_ratio = float(np.mean([f["metrics"]["mae_ratio"] for f in fold_metrics]))
                    std_ratio = (
                        float(np.std([f["metrics"]["mae_ratio"] for f in fold_metrics], ddof=1))
                        if len(fold_metrics) > 1
                        else 0.0
                    )
                    max_degrade = float(np.max([f["degrade_pct"] for f in fold_metrics]))
                    config_results.append(
                        {
                            "candidate_meta": {
                                "model_label": model_label,
                                "feature_set": feature_set,
                                "target_mode": target_mode,
                            },
                            "policy_kind": "bucket",
                            "base_policy_kind": "raw",
                            "config": bucket_config,
                            "mean_ratio": mean_ratio,
                            "std_ratio": std_ratio,
                            "max_degrade_pct": max_degrade,
                            "fold_metrics": fold_metrics,
                            "decisions": fold_decisions,
                        }
                    )
            sigmoid_results = [
                result for result in candidate_result_entries if str(result.get("policy_kind", "")) == "sigmoid"
            ]
            if sigmoid_results:
                accepted_sigmoid = [r for r in sigmoid_results if r["max_degrade_pct"] <= 2.0]
                ranking_pool = accepted_sigmoid if accepted_sigmoid else sigmoid_results
                ranking_pool.sort(key=lambda r: (r["mean_ratio"], r["std_ratio"], r["max_degrade_pct"]))
                best_sigmoid_result = ranking_pool[0]
                sigmoid_cfg = cast(BlendConfig, best_sigmoid_result["config"])
                sigmoid_fold_payload: list[dict[str, Any]] = []
                sigmoid_decisions = cast(list[pd.DataFrame], best_sigmoid_result["decisions"])
                for payload, decisions in zip(fold_payload, sigmoid_decisions):
                    base_aligned = cast(pd.DataFrame, payload["aligned"])
                    blended_prediction = (
                        decisions.set_index("row_index")["blend_pred"].reindex(base_aligned.index).astype(float)
                    )
                    sigmoid_aligned = base_aligned.loc[:, ["y_true", "y_persist"]].copy()
                    sigmoid_aligned["y_pred"] = blended_prediction.to_numpy(dtype=float)
                    sigmoid_fold_payload.append(
                        {
                            **payload,
                            "aligned": sigmoid_aligned,
                        }
                    )
                calibration_aligned = pd.concat(
                    [cast(pd.DataFrame, payload["aligned"]) for payload in sigmoid_fold_payload],
                    axis=0,
                )
                calibration_timestamps = pd.DatetimeIndex(
                    np.concatenate(
                        [
                            pd.DatetimeIndex(cast(pd.Index, payload["timestamps"])).to_numpy()
                            for payload in sigmoid_fold_payload
                        ]
                    )
                )
                sigmoid_bucket_config = _calibrate_bucket_blend_config(
                    aligned=calibration_aligned,
                    timestamps=calibration_timestamps,
                    bucket_size_minutes=int(MODELING_PERFORMANCE_BLEND_SEARCH["bucket_size_minutes"]),
                    cycle_minutes=int(MODELING_PERFORMANCE_BLEND_SEARCH["bucket_cycle_minutes"]),
                    candidate_weights=[
                        float(value) for value in MODELING_PERFORMANCE_BLEND_SEARCH["bucket_candidate_weights"]
                    ],
                )
                if sigmoid_bucket_config is not None:
                    fold_metrics = []
                    fold_decisions = []
                    for payload in sigmoid_fold_payload:
                        metrics, decisions = _apply_bucket_blend_policy(
                            aligned=cast(pd.DataFrame, payload["aligned"]),
                            timestamps=cast(pd.Index, payload["timestamps"]),
                            bucket_config=sigmoid_bucket_config,
                            n_eval_total=int(payload["n_eval_total"]),
                        )
                        if not metrics:
                            continue
                        fold = cast(dict[str, int], payload["fold_meta"])
                        raw_ratio = float(payload["raw_ratio"])
                        degrade_pct = (
                            (float(metrics["mae_ratio"]) - raw_ratio) / raw_ratio * 100.0
                        ) if raw_ratio > 0 else 0.0
                        fold_metrics.append(
                            {
                                "fold": int(fold["fold"]),
                                "metrics": metrics,
                                "degrade_pct": degrade_pct,
                                "fold_meta": fold,
                            }
                        )
                        decisions = decisions.copy()
                        decisions["fold"] = int(fold["fold"])
                        decisions["resolution"] = resolution
                        decisions["feature_set"] = feature_set
                        decisions["model_label"] = model_label
                        decisions["source_target_mode"] = target_mode
                        decisions["blend_base_policy_kind"] = "sigmoid"
                        decisions["blend_window"] = int(sigmoid_cfg.window)
                        decisions["blend_sharpness"] = float(sigmoid_cfg.sharpness)
                        decisions["blend_min_weight"] = float(sigmoid_cfg.min_weight)
                        decisions["blend_max_weight"] = float(sigmoid_cfg.max_weight)
                        decisions["blend_bucket_size_minutes"] = int(sigmoid_bucket_config.bucket_size_minutes)
                        decisions["blend_bucket_cycle_minutes"] = int(sigmoid_bucket_config.cycle_minutes)
                        decisions["blend_bucket_weights_json"] = json.dumps(
                            sigmoid_bucket_config.weight_map(),
                            sort_keys=True,
                        )
                        fold_decisions.append(decisions)
                    if fold_metrics:
                        mean_ratio = float(np.mean([f["metrics"]["mae_ratio"] for f in fold_metrics]))
                        std_ratio = (
                            float(np.std([f["metrics"]["mae_ratio"] for f in fold_metrics], ddof=1))
                            if len(fold_metrics) > 1
                            else 0.0
                        )
                        max_degrade = float(np.max([f["degrade_pct"] for f in fold_metrics]))
                        candidate_result_entries.append(
                            {
                                "candidate_meta": {
                                    "model_label": model_label,
                                    "feature_set": feature_set,
                                    "target_mode": target_mode,
                                },
                                "policy_kind": "bucket",
                                "base_policy_kind": "sigmoid",
                                "config": SigmoidBucketBlendConfig(
                                    blend_config=sigmoid_cfg,
                                    bucket_config=sigmoid_bucket_config,
                                ),
                                "mean_ratio": mean_ratio,
                                "std_ratio": std_ratio,
                                "max_degrade_pct": max_degrade,
                                "fold_metrics": fold_metrics,
                                "decisions": fold_decisions,
                            }
                        )
        config_results.extend(candidate_result_entries)

    if not config_results:
        return pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame()

    candidate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in config_results:
        candidate_meta = cast(dict[str, Any], result["candidate_meta"])
        candidate_key = (
            str(candidate_meta["feature_set"]),
            str(candidate_meta["model_label"]),
            str(candidate_meta["target_mode"]),
        )
        candidate_groups.setdefault(candidate_key, []).append(result)

    candidate_best_results: list[dict[str, Any]] = []
    blend_finalists_rows: list[dict[str, Any]] = []
    for candidate_key, candidate_results in candidate_groups.items():
        accepted = [r for r in candidate_results if r["max_degrade_pct"] <= 2.0]
        ranking_pool = accepted if accepted else candidate_results
        ranking_pool.sort(key=lambda r: (r["mean_ratio"], r["std_ratio"], r["max_degrade_pct"]))
        best_result = ranking_pool[0]
        candidate_best_results.append(best_result)
        candidate_meta = cast(dict[str, Any], best_result["candidate_meta"])
        fold_metrics = cast(list[dict[str, Any]], best_result["fold_metrics"])
        policy_kind = str(best_result.get("policy_kind", "sigmoid"))
        base_policy_kind = str(best_result.get("base_policy_kind", "raw"))
        config = best_result["config"]
        target_mode_label = (
            _stage5_bucket_blend_target_mode(
                candidate_key[2],
                (
                    cast(SigmoidBucketBlendConfig, config).bucket_config.bucket_size_minutes
                    if isinstance(config, SigmoidBucketBlendConfig)
                    else cast(BucketBlendConfig, config).bucket_size_minutes
                ),
                base_policy_kind=base_policy_kind,
            )
            if policy_kind == "bucket"
            else f"{candidate_key[2]}+blend"
        )
        candidate_label = f"{candidate_key[0]}/{candidate_key[1]}/{target_mode_label}"
        blend_row = {
            "resolution": resolution,
            "feature_set": candidate_key[0],
            "model_label": candidate_key[1],
            "source_target_mode": candidate_key[2],
            "target_mode": target_mode_label,
            "candidate_label": candidate_label,
            "fold_mean_mae_ratio": float(best_result["mean_ratio"]),
            "fold_std_mae_ratio": float(best_result["std_ratio"]),
            "blend_validate_mae": float(np.mean([float(item["metrics"]["mae"]) for item in fold_metrics])),
            "blend_validate_mae_pct": float(
                np.mean([float(item["metrics"]["mae_pct"]) for item in fold_metrics])
            ),
            "blend_validate_rmse_pct": float(
                np.mean([float(item["metrics"]["rmse_pct"]) for item in fold_metrics])
            ),
            "mean_coverage": float(
                np.mean([float(item["metrics"]["coverage"]) for item in fold_metrics])
            ),
            "max_fold_degrade_pct": float(best_result["max_degrade_pct"]),
            "selected_blend_policy_kind": policy_kind,
            "selected_blend_base_policy_kind": base_policy_kind,
            "selected_blend_window": float("nan"),
            "selected_blend_sharpness": float("nan"),
            "selected_blend_min_weight": float("nan"),
            "selected_blend_max_weight": float("nan"),
            "selected_blend_bucket_size_minutes": float("nan"),
            "selected_blend_bucket_cycle_minutes": float("nan"),
            "selected_blend_bucket_weights_json": "",
            "meets_p2_fold_degrade_cap": bool(best_result["max_degrade_pct"] <= 2.0),
            "preferred_candidate_match": bool(
                preferred_candidate is not None
                and str(candidate_meta["feature_set"]) == str(preferred_candidate.get("feature_set", ""))
                and str(candidate_meta["model_label"]) == str(preferred_candidate.get("model_label", ""))
                and str(candidate_meta["target_mode"]) == str(preferred_candidate.get("target_mode", ""))
            ),
        }
        if policy_kind == "bucket":
            bucket_cfg = (
                cast(SigmoidBucketBlendConfig, config).bucket_config
                if isinstance(config, SigmoidBucketBlendConfig)
                else cast(BucketBlendConfig, config)
            )
            blend_row["selected_blend_bucket_size_minutes"] = int(bucket_cfg.bucket_size_minutes)
            blend_row["selected_blend_bucket_cycle_minutes"] = int(bucket_cfg.cycle_minutes)
            blend_row["selected_blend_bucket_weights_json"] = json.dumps(
                bucket_cfg.weight_map(),
                sort_keys=True,
            )
            if isinstance(config, SigmoidBucketBlendConfig):
                sigmoid_cfg = config.blend_config
                blend_row["selected_blend_window"] = int(sigmoid_cfg.window)
                blend_row["selected_blend_sharpness"] = float(sigmoid_cfg.sharpness)
                blend_row["selected_blend_min_weight"] = float(sigmoid_cfg.min_weight)
                blend_row["selected_blend_max_weight"] = float(sigmoid_cfg.max_weight)
        else:
            cfg = cast(BlendConfig, config)
            blend_row["selected_blend_window"] = int(cfg.window)
            blend_row["selected_blend_sharpness"] = float(cfg.sharpness)
            blend_row["selected_blend_min_weight"] = float(cfg.min_weight)
            blend_row["selected_blend_max_weight"] = float(cfg.max_weight)
        blend_finalists_rows.append(
            blend_row
        )

    blend_finalists = pd.DataFrame(blend_finalists_rows)
    if not blend_finalists.empty:
        blend_finalists = blend_finalists.sort_values(
            [
                "meets_p2_fold_degrade_cap",
                "fold_mean_mae_ratio",
                "fold_std_mae_ratio",
                "max_fold_degrade_pct",
                "feature_set",
                "model_label",
                "source_target_mode",
            ],
            ascending=[False, True, True, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        blend_finalists["blend_rank"] = np.arange(1, len(blend_finalists) + 1, dtype=int)

    candidate_best_results.sort(key=lambda r: (r["mean_ratio"], r["std_ratio"], r["max_degrade_pct"]))
    selected = candidate_best_results[0]
    selected_config = selected["config"]
    selected_meta = selected["candidate_meta"]
    selected_policy_kind = str(selected.get("policy_kind", "sigmoid"))
    selected_base_policy_kind = str(selected.get("base_policy_kind", "raw"))
    selected_target_mode = (
        _stage5_bucket_blend_target_mode(
            str(selected_meta["target_mode"]),
            (
                cast(SigmoidBucketBlendConfig, selected_config).bucket_config.bucket_size_minutes
                if isinstance(selected_config, SigmoidBucketBlendConfig)
                else cast(BucketBlendConfig, selected_config).bucket_size_minutes
            ),
            base_policy_kind=selected_base_policy_kind,
        )
        if selected_policy_kind == "bucket"
        else f"{selected_meta['target_mode']}+blend"
    )

    blend_rows: list[dict[str, Any]] = []
    for fold_item in selected["fold_metrics"]:
        fold = fold_item["fold_meta"]
        blend_rows.append(
            {
                "fold": int(fold["fold"]),
                "resolution": resolution,
                "feature_set": selected_meta["feature_set"],
                "model": model_catalog[selected_meta["model_label"]].family,
                "model_label": selected_meta["model_label"],
                "params": json.dumps(model_catalog[selected_meta["model_label"]].params, sort_keys=True),
                "target_mode": selected_target_mode,
                **fold_item["metrics"],
                "train_start_day": fold["train_start_day"],
                "train_end_day": fold["train_end_day"],
                "val_start_day": fold["val_start_day"],
                "val_end_day": fold["val_end_day"],
            }
        )
    decisions_df = pd.concat(selected["decisions"], ignore_index=True) if selected["decisions"] else pd.DataFrame()
    blend_df = pd.DataFrame(blend_rows)
    candidate_meta = {
        "model_label": selected_meta["model_label"],
        "feature_set": selected_meta["feature_set"],
        "target_mode": selected_meta["target_mode"],
        "selected_blend_policy_kind": selected_policy_kind,
        "selected_blend_base_policy_kind": selected_base_policy_kind,
        "selected_blend_mean_ratio": selected["mean_ratio"],
        "selected_blend_std_ratio": selected["std_ratio"],
        "selected_blend_max_degrade_pct": selected["max_degrade_pct"],
        "meets_p2_fold_degrade_cap": bool(selected["max_degrade_pct"] <= 2.0),
    }
    if selected_policy_kind == "bucket":
        bucket_cfg = (
            cast(SigmoidBucketBlendConfig, selected_config).bucket_config
            if isinstance(selected_config, SigmoidBucketBlendConfig)
            else cast(BucketBlendConfig, selected_config)
        )
        candidate_meta.update(
            {
                "selected_blend_bucket_size_minutes": bucket_cfg.bucket_size_minutes,
                "selected_blend_bucket_cycle_minutes": bucket_cfg.cycle_minutes,
                "selected_blend_bucket_weights_json": json.dumps(
                    bucket_cfg.weight_map(),
                    sort_keys=True,
                ),
            }
        )
        if isinstance(selected_config, SigmoidBucketBlendConfig):
            candidate_meta.update(
                {
                    "selected_blend_window": selected_config.blend_config.window,
                    "selected_blend_sharpness": selected_config.blend_config.sharpness,
                    "selected_blend_min_weight": selected_config.blend_config.min_weight,
                    "selected_blend_max_weight": selected_config.blend_config.max_weight,
                }
            )
    else:
        sigmoid_cfg = cast(BlendConfig, selected_config)
        candidate_meta.update(
            {
                "selected_blend_window": sigmoid_cfg.window,
                "selected_blend_sharpness": sigmoid_cfg.sharpness,
                "selected_blend_min_weight": sigmoid_cfg.min_weight,
                "selected_blend_max_weight": sigmoid_cfg.max_weight,
            }
        )
    return blend_df, decisions_df, candidate_meta, selected_config, blend_finalists


def build_selection_scoreboard(metrics_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold results into the canonical Stage-5 promotion scoreboard."""
    work = metrics_fold.copy()
    for column in ("mae_pct", "rmse_pct"):
        if column not in work.columns:
            work[column] = float("nan")
    group_cols = ["resolution", "feature_set", "model_label", "target_mode"]
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            fold_mean_mae_ratio=("mae_ratio", "mean"),
            fold_std_mae_ratio=("mae_ratio", "std"),
            fold_n=("fold", "nunique"),
            raw_validate_mae=("mae", "mean"),
            raw_validate_mae_pct=("mae_pct", "mean"),
            raw_validate_rmse_pct=("rmse_pct", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .reset_index()
    )
    grouped["fold_std_mae_ratio"] = grouped["fold_std_mae_ratio"].fillna(0.0)
    return grouped.sort_values(
        ["fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae", "raw_validate_mae_pct"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_hgb_coordinate_summary(metrics_fold: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Summarize the bounded HGB search and return the recommended survivor."""
    subset = metrics_fold[
        (metrics_fold["feature_set"] == "full")
        & (metrics_fold["target_mode"] == "raw")
        & (metrics_fold["model_label"].astype(str).str.startswith("hgb"))
    ].copy()
    if subset.empty:
        return pd.DataFrame(), None
    grouped = (
        subset.groupby("model_label", dropna=False)
        .agg(
            fold_mean_mae_ratio=("mae_ratio", "mean"),
            fold_std_mae_ratio=("mae_ratio", "std"),
            fold_n=("fold", "nunique"),
            mean_train_val_mae_ratio=("train_val_mae_ratio", "mean"),
        )
        .reset_index()
    )
    grouped["fold_std_mae_ratio"] = grouped["fold_std_mae_ratio"].fillna(0.0)
    grouped["train_val_gap_to_one"] = (grouped["mean_train_val_mae_ratio"] - 1.0).abs()
    baseline_row = grouped[grouped["model_label"] == "hgb-aggressive"]
    if baseline_row.empty:
        grouped["delta_mean_mae_ratio"] = float("nan")
        grouped["delta_std_mae_ratio"] = float("nan")
        grouped["delta_train_val_gap"] = float("nan")
        grouped["meets_p1b_acceptance"] = False
        return grouped.sort_values("fold_mean_mae_ratio").reset_index(drop=True), None

    baseline = baseline_row.iloc[0]
    grouped["delta_mean_mae_ratio"] = grouped["fold_mean_mae_ratio"] - float(baseline["fold_mean_mae_ratio"])
    grouped["delta_std_mae_ratio"] = grouped["fold_std_mae_ratio"] - float(baseline["fold_std_mae_ratio"])
    grouped["delta_train_val_gap"] = grouped["train_val_gap_to_one"] - float(baseline["train_val_gap_to_one"])
    grouped["meets_p1b_acceptance"] = (
        (grouped["delta_mean_mae_ratio"] <= -0.005)
        & (grouped["delta_std_mae_ratio"] <= 0.0)
        & (grouped["delta_train_val_gap"] <= 0.0)
    )
    grouped = grouped.sort_values(
        ["meets_p1b_acceptance", "fold_mean_mae_ratio", "fold_std_mae_ratio"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    recommended = None
    if not grouped.empty:
        top = grouped.iloc[0]
        recommended = {
            "model_label": str(top["model_label"]),
            "feature_set": "full",
            "target_mode": "raw",
            "meets_p1b_acceptance": bool(top["meets_p1b_acceptance"]),
        }
    return grouped, recommended


def build_residual_ablation(selection_scoreboard: pd.DataFrame) -> pd.DataFrame:
    """Compare raw, residual, and blend target modes for the same candidates."""
    keys = ["resolution", "feature_set", "model_label"]
    raw = selection_scoreboard[selection_scoreboard["target_mode"] == "raw"][
        keys + ["fold_mean_mae_ratio", "raw_validate_mae"]
    ].rename(
        columns={"fold_mean_mae_ratio": "raw_fold_mean_mae_ratio", "raw_validate_mae": "raw_mean_mae"}
    )
    residual = selection_scoreboard[selection_scoreboard["target_mode"] == "residual"][
        keys + ["fold_mean_mae_ratio", "raw_validate_mae"]
    ].rename(
        columns={
            "fold_mean_mae_ratio": "residual_fold_mean_mae_ratio",
            "raw_validate_mae": "residual_mean_mae",
        }
    )
    joined = raw.merge(residual, on=keys, how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined["delta_fold_mean_mae_ratio"] = (
        joined["residual_fold_mean_mae_ratio"] - joined["raw_fold_mean_mae_ratio"]
    )
    joined["delta_mean_mae"] = joined["residual_mean_mae"] - joined["raw_mean_mae"]
    return joined.sort_values(keys).reset_index(drop=True)


def _build_coverage_audit(
    *,
    gold: pd.DataFrame,
    resolution: str,
    feature_sets: dict[str, list[str]],
    selected_feature_sets: list[str],
    steps_per_day: int,
) -> pd.DataFrame:
    """Summarize coverage health by feature set and split."""
    rows: list[dict[str, Any]] = []
    for feature_set in selected_feature_sets:
        required_columns = list(dict.fromkeys([*feature_sets[feature_set], TARGET_COLUMN, "lag_1"]))
        for split_name in ("train", "validate", "test"):
            start_day, end_day = SPLIT_DAY_RANGES[split_name]
            split_df = gold.loc[gold["day_idx"].between(start_day, end_day)].copy()
            expected_rows = _expected_steps_for_day_range(start_day, end_day, steps_per_day)
            usable_rows = int(split_df[required_columns].notna().all(axis=1).sum()) if expected_rows > 0 else 0
            coverage = float(usable_rows / expected_rows) if expected_rows > 0 else float("nan")
            rows.append(
                {
                    "resolution": resolution,
                    "feature_set": feature_set,
                    "split": split_name,
                    "expected_rows": expected_rows,
                    "usable_rows": usable_rows,
                    "coverage": coverage,
                    "status": "pass" if coverage >= PROMOTION_MIN_COVERAGE else "fail",
                }
            )
    return pd.DataFrame(rows)


def _select_promotion_candidate(selection_scoreboard: pd.DataFrame) -> dict[str, Any] | None:
    """Choose the Stage-5 candidate eligible for holdout promotion."""
    if selection_scoreboard.empty:
        return None
    eligible = selection_scoreboard.loc[
        pd.to_numeric(selection_scoreboard["mean_coverage"], errors="coerce").fillna(0.0)
        >= PROMOTION_MIN_COVERAGE
    ].copy()
    if eligible.empty:
        ranked = selection_scoreboard.sort_values(
            ["mean_coverage", "fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae"],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        top = ranked.iloc[0].to_dict()
        return {
            **top,
            "selection_reason": (
                f"fallback_highest_coverage_then_mae_ratio (threshold={PROMOTION_MIN_COVERAGE:.2f})"
            ),
            "coverage_gate_passed": False,
            "promotion_threshold": PROMOTION_MIN_COVERAGE,
        }
    ranked = eligible.sort_values(
        ["fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    top = ranked.iloc[0].to_dict()
    return {
        **top,
        "selection_reason": f"coverage_guarded_mae_ratio (threshold={PROMOTION_MIN_COVERAGE:.2f})",
        "coverage_gate_passed": True,
        "promotion_threshold": PROMOTION_MIN_COVERAGE,
    }


def _derive_operating_regime(eval_df: pd.DataFrame) -> pd.Series | None:
    """Create one deployment-oriented regime label from existing causal feature columns."""
    available = {"day_class", "profile_active_flag", "workday_transition"} & set(eval_df.columns)
    if not available:
        return None
    index = eval_df.index
    day_class = (
        eval_df["day_class"].astype("string").fillna("unknown")
        if "day_class" in eval_df.columns
        else pd.Series("unknown", index=index, dtype="string")
    )
    active = (
        pd.to_numeric(eval_df["profile_active_flag"], errors="coerce").fillna(0.0).astype(float) > 0.0
        if "profile_active_flag" in eval_df.columns
        else pd.Series(False, index=index, dtype=bool)
    )
    transition = (
        pd.to_numeric(eval_df["workday_transition"], errors="coerce").fillna(0.0).astype(float) > 0.0
        if "workday_transition" in eval_df.columns
        else pd.Series(False, index=index, dtype=bool)
    )
    operating_regime = pd.Series("inactive", index=index, dtype="string")
    known_day_class = day_class.ne("unknown")
    operating_regime.loc[known_day_class] = day_class.loc[known_day_class] + "_inactive"
    operating_regime.loc[active] = "active_profile"
    operating_regime.loc[transition] = "transition_only"
    operating_regime.loc[transition & active] = "transition_active"
    return operating_regime


def _build_stage5_operating_policy(
    *,
    deployment_recommendation: dict[str, Any],
    candidate_label: str,
    best_baseline_label: str,
    operating_regime_evaluation: pd.DataFrame,
) -> dict[str, Any]:
    """Convert Stage-5 holdout evidence into an honest standalone-vs-overlay policy surface."""
    default_candidate_label = str(deployment_recommendation["recommended_candidate_label"])
    default_candidate_type = str(deployment_recommendation["recommended_candidate_type"])
    learned_beats_best_baseline = bool(deployment_recommendation["learned_beats_best_baseline"])
    standalone_role = "learned_anchor" if learned_beats_best_baseline else "baseline_anchor"
    stage10_role = "standalone_and_overlay" if learned_beats_best_baseline else "corrective_overlay_specialist"
    regime_overrides: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    if not operating_regime_evaluation.empty:
        working = operating_regime_evaluation.copy()
        working = working.sort_values(
            ["candidate_mae_ratio_to_best_baseline", "candidate_mae_ratio_to_persistence", "rows", "segment_value"],
            ascending=[True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        for row in working.itertuples(index=False):
            rows = int(getattr(row, "rows", 0))
            candidate_mae = float(getattr(row, "candidate_mae"))
            persistence_mae = float(getattr(row, "persistence_mae"))
            best_baseline_mae = float(getattr(row, "best_baseline_mae"))
            learned_supported = (
                rows >= OPERATING_POLICY_MIN_SEGMENT_ROWS
                and np.isfinite(candidate_mae)
                and np.isfinite(persistence_mae)
                and np.isfinite(best_baseline_mae)
                and candidate_mae < persistence_mae
                and candidate_mae < best_baseline_mae
            )
            recommended_candidate_label = candidate_label if learned_supported else best_baseline_label
            recommended_candidate_type = "promoted_learned" if learned_supported else "baseline"
            if learned_supported:
                evidence_status = "learned_supported"
                reason = (
                    "The learned minute candidate beat both persistence and the strongest baseline "
                    "within this operating regime on the Stage-5 holdout slice."
                )
            elif rows < OPERATING_POLICY_MIN_SEGMENT_ROWS:
                evidence_status = "insufficient_rows"
                reason = (
                    "This regime does not have enough holdout support to override the global default, "
                    "so the policy stays on the safer baseline recommendation."
                )
            else:
                evidence_status = "baseline_preferred"
                reason = (
                    "The learned minute candidate did not beat the strongest baseline in this regime, "
                    "so the operating policy keeps the safer baseline recommendation."
                )
            payload = {
                "operating_regime": str(getattr(row, "segment_value")),
                "rows": rows,
                "recommended_candidate_label": recommended_candidate_label,
                "recommended_candidate_type": recommended_candidate_type,
                "candidate_mae": candidate_mae,
                "candidate_mae_ratio_to_persistence": float(getattr(row, "candidate_mae_ratio_to_persistence")),
                "candidate_mae_ratio_to_best_baseline": float(
                    getattr(row, "candidate_mae_ratio_to_best_baseline")
                ),
                "best_baseline_label": str(getattr(row, "best_baseline_label")),
                "best_baseline_mae": best_baseline_mae,
                "persistence_mae": persistence_mae,
                "evidence_status": evidence_status,
                "reason": reason,
            }
            regime_rows.append(payload)
            if recommended_candidate_label != default_candidate_label:
                regime_overrides.append(payload)
    return {
        "policy_version": 1,
        "policy_mode": "global_default_with_regime_overrides",
        "default_candidate_label": default_candidate_label,
        "default_candidate_type": default_candidate_type,
        "best_baseline_label": best_baseline_label,
        "standalone_operating_role": standalone_role,
        "stage10_operating_role": stage10_role,
        "regime_column": "operating_regime",
        "min_segment_rows": int(OPERATING_POLICY_MIN_SEGMENT_ROWS),
        "regime_overrides": regime_overrides,
        "regime_evidence": regime_rows,
        "summary": {
            "override_count": int(len(regime_overrides)),
            "learned_supported_regime_count": int(
                sum(1 for row in regime_rows if str(row["evidence_status"]) == "learned_supported")
            ),
        },
        "reason": (
            "The standalone Stage-5 minute policy stays on the learned candidate globally."
            if learned_beats_best_baseline
            else "The standalone Stage-5 minute policy stays baseline-led globally, while Stage-10 may still use learned minute overlays where broader control evidence supports them."
        ),
    }


def _build_holdout_coverage_summary(
    *,
    holdout_frame: pd.DataFrame,
    segment_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize how broad or narrow the promoted Stage-5 holdout surface really is."""
    if holdout_frame.empty:
        return pd.DataFrame(), {}
    working = holdout_frame.copy()
    coverage_rows: list[dict[str, Any]] = []
    single_value_segment_columns: list[str] = []
    for column in segment_columns:
        if column not in working.columns:
            continue
        value_counts = (
            working[column]
            .astype("string")
            .fillna("<NA>")
            .value_counts(dropna=False, sort=True)
        )
        unique_value_count = int(len(value_counts))
        if unique_value_count == 1:
            single_value_segment_columns.append(str(column))
        dominant_value = str(value_counts.index[0]) if unique_value_count else ""
        dominant_rows = int(value_counts.iloc[0]) if unique_value_count else 0
        for segment_value, rows in value_counts.items():
            coverage_rows.append(
                {
                    "segment_column": str(column),
                    "segment_value": str(segment_value),
                    "rows": int(rows),
                    "row_fraction": float(rows / len(working)),
                    "unique_value_count": unique_value_count,
                    "is_only_value": bool(unique_value_count == 1),
                    "is_dominant_value": bool(str(segment_value) == dominant_value),
                    "dominant_value": dominant_value,
                    "dominant_rows": dominant_rows,
                }
            )
    coverage_segments = pd.DataFrame(coverage_rows)
    timestamp_series = pd.to_datetime(
        working.get("timestamp", pd.Series(index=working.index, dtype="datetime64[ns]")),
        errors="coerce",
    )
    unique_day_count = int(timestamp_series.dt.normalize().nunique()) if not timestamp_series.empty else 0
    operating_regime_counts = (
        working["operating_regime"].astype("string").value_counts(dropna=False)
        if "operating_regime" in working.columns
        else pd.Series(dtype="int64")
    )
    dominant_regime = str(operating_regime_counts.index[0]) if not operating_regime_counts.empty else ""
    dominant_regime_rows = int(operating_regime_counts.iloc[0]) if not operating_regime_counts.empty else 0
    summary = {
        "row_n": int(len(working)),
        "unique_day_n": unique_day_count,
        "start_timestamp": (
            timestamp_series.min().isoformat() if not timestamp_series.empty and pd.notna(timestamp_series.min()) else ""
        ),
        "end_timestamp": (
            timestamp_series.max().isoformat() if not timestamp_series.empty and pd.notna(timestamp_series.max()) else ""
        ),
        "segment_columns": [str(column) for column in segment_columns if column in working.columns],
        "single_value_segment_columns": single_value_segment_columns,
        "single_value_segment_count": int(len(single_value_segment_columns)),
        "operating_regime_unique_n": int(len(operating_regime_counts)),
        "dominant_operating_regime": dominant_regime,
        "dominant_operating_regime_rows": dominant_regime_rows,
        "dominant_operating_regime_fraction": (
            float(dominant_regime_rows / len(working)) if len(working) else float("nan")
        ),
        "narrow_regime_support": bool(len(single_value_segment_columns) > 0),
        "reason": (
            "The promoted Stage-5 holdout surface covers only one observed value for at least one key operating segment, so standalone 1-minute claims should be read as narrow-regime evidence."
            if single_value_segment_columns
            else "The promoted Stage-5 holdout surface spans more than one observed value across the configured operating segments."
        ),
    }
    return coverage_segments, summary


def _evaluate_promoted_holdout_artifacts(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]] | None,
    feature_sets: dict[str, list[str]],
    model_catalog: dict[str, ModelSpec],
    resolution: str,
    promoted_candidate: dict[str, Any],
    steps_per_day: int,
    blend_config: BlendConfig | BucketBlendConfig | SigmoidBucketBlendConfig | None,
) -> HoldoutDiagnostics:
    """Run the promoted candidate against holdout and return the full diagnostic bundle."""
    feature_set = str(promoted_candidate["feature_set"])
    model_label = str(promoted_candidate["model_label"])
    target_mode = str(promoted_candidate["target_mode"])
    base_target_mode = _stage5_base_target_mode(target_mode)
    blend_policy_kind = _stage5_blend_policy_kind(target_mode)
    model_spec = model_catalog[model_label]

    train_start_day = int(SPLIT_DAY_RANGES["train"][0])
    train_end_day = int(SPLIT_DAY_RANGES["validate"][1])
    holdout_start_day = int(SPLIT_DAY_RANGES["test"][0])
    holdout_end_day = int(SPLIT_DAY_RANGES["test"][1])
    train_df = gold.loc[gold["day_idx"].between(train_start_day, train_end_day)].copy()
    eval_df = gold.loc[gold["day_idx"].between(holdout_start_day, holdout_end_day)].copy()
    n_eval_total = _expected_steps_for_day_range(holdout_start_day, holdout_end_day, steps_per_day)
    feature_columns = _resolve_feature_set_columns(feature_set, feature_sets=feature_sets)
    bundle = _fit_model_bundle(
        train_df=train_df,
        eval_df=eval_df,
        feature_cols=feature_columns,
        model_spec=model_spec,
        target_mode=base_target_mode,
    )
    if bundle is None:
        raise ValueError("Promoted Stage-5 candidate produced no holdout-aligned rows.")
    aligned = cast(pd.DataFrame, bundle["aligned"])
    train_mae = float(bundle["train_mae"])

    candidate_name = f"{feature_set}/{model_label}/{target_mode}"
    candidate_predictions, candidate_eval, holdout_decisions = _stage5_surface_candidate_predictions(
        aligned=aligned,
        eval_df=eval_df,
        resolution=resolution,
        feature_set=feature_set,
        model_label=model_label,
        target_mode=target_mode,
        blend_config=blend_config,
        n_eval_total=n_eval_total,
    )
    candidate_mae = float(candidate_eval["mae"])
    candidate_rmse = float(candidate_eval["rmse"])
    candidate_coverage = float(candidate_eval["coverage"])
    candidate_n_eval = int(candidate_eval["n_eval"])
    candidate_mae_pct = float(candidate_eval["mae_pct"])
    candidate_rmse_pct = float(candidate_eval["rmse_pct"])

    persistence_eval = compute_regression_metrics(
        aligned["y_true"],
        aligned["y_persist"],
        n_total=int(n_eval_total),
    )
    persistence_mae = float(persistence_eval["mae"])
    persistence_rmse = float(persistence_eval["rmse"])
    persistence_n_eval = int(persistence_eval["n_eval"])
    persistence_coverage = float(persistence_eval["coverage"])
    persistence_mae_pct = float(persistence_eval["mae_pct"])
    persistence_rmse_pct = float(persistence_eval["rmse_pct"])

    baseline_predictions = _build_holdout_baseline_predictions(
        train_df=train_df,
        eval_df=eval_df,
        aligned=aligned,
        resolution=resolution,
    )
    holdout_rows: list[dict[str, Any]] = [
        {
            "candidate_label": candidate_name,
            "candidate_type": "promoted_learned",
            "resolution": resolution,
            "feature_set": feature_set,
            "model_label": model_label,
            "target_mode": target_mode,
            "mae": candidate_mae,
            "rmse": candidate_rmse,
            "mae_pct": candidate_mae_pct,
            "rmse_pct": candidate_rmse_pct,
            "mae_ratio_to_persistence": mae_ratio(candidate_mae, persistence_mae),
            "coverage": candidate_coverage,
            "n_eval": candidate_n_eval,
            "n_eval_total": n_eval_total,
        }
    ]
    baseline_meta = {
        "persistence": ("baseline", "persistence", "raw"),
        "previous_day": ("baseline", "previous_day", "raw"),
        "avg_workday": ("baseline", "avg_workday", "raw"),
        "anchored_workday": ("baseline", "anchored_workday", "raw"),
        "holt_damped": ("baseline", "holt_damped", "forecast"),
    }
    for baseline_label, prediction in baseline_predictions.items():
        feature_set_label, model_label_baseline, target_mode_baseline = baseline_meta.get(
            baseline_label,
            ("baseline", baseline_label, "raw"),
        )
        holdout_rows.append(
            _holdout_summary_row(
                candidate_label=baseline_label,
                candidate_type="baseline",
                resolution=resolution,
                feature_set=feature_set_label,
                model_label=model_label_baseline,
                target_mode=target_mode_baseline,
                y_true=aligned["y_true"],
                y_pred=prediction.reindex(aligned.index).astype(float),
                n_eval_total=n_eval_total,
                persistence_mae=persistence_mae,
            )
        )
    holdout_summary = pd.DataFrame(holdout_rows)
    baseline_summary = holdout_summary.loc[holdout_summary["candidate_type"].astype("string").eq("baseline")].copy()
    best_baseline_row = baseline_summary.sort_values(["mae", "rmse", "candidate_label"], kind="stable").iloc[0]
    best_baseline_label = str(best_baseline_row["candidate_label"])
    best_baseline_mae = float(best_baseline_row["mae"])
    best_baseline_mae_pct = float(best_baseline_row["mae_pct"])
    holdout_summary["mae_ratio_to_best_baseline"] = holdout_summary["mae"].astype(float).map(
        lambda value: mae_ratio(float(value), best_baseline_mae)
    )
    holdout_summary["is_best_baseline"] = holdout_summary["candidate_label"].astype("string").eq(best_baseline_label)

    deployment_recommendation = {
        "recommended_candidate_label": candidate_name if candidate_mae < best_baseline_mae else best_baseline_label,
        "recommended_candidate_type": "promoted_learned" if candidate_mae < best_baseline_mae else "baseline",
        "decision_reason": (
            f"Promoted Stage-5 candidate beat the best baseline ({best_baseline_label}) on holdout MAE."
            if candidate_mae < best_baseline_mae
            else (
                f"{best_baseline_label} remains the operational winner because the promoted "
                "Stage-5 candidate did not beat the strongest baseline on holdout MAE."
            )
        ),
        "train_day_range": [train_start_day, train_end_day],
        "holdout_day_range": [holdout_start_day, holdout_end_day],
        "train_mae": train_mae,
        "recommended_mae_pct": candidate_mae_pct if candidate_mae < best_baseline_mae else best_baseline_mae_pct,
        "best_baseline_label": best_baseline_label,
        "best_baseline_mae": best_baseline_mae,
        "best_baseline_mae_pct": best_baseline_mae_pct,
        "learned_beats_persistence": bool(candidate_mae < persistence_mae),
        "learned_beats_best_baseline": bool(candidate_mae < best_baseline_mae),
    }
    segment_columns = [
        column for column in MODELING_PERFORMANCE_EVALUATION["segment_columns"] if column in eval_df.columns
    ]
    operating_regime = _derive_operating_regime(eval_df.loc[aligned.index].copy())
    if operating_regime is not None:
        segment_columns = [*segment_columns, "operating_regime"]
    segment_rows: list[dict[str, Any]] = []
    segment_base = (
        eval_df.loc[aligned.index, [column for column in segment_columns if column in eval_df.columns]].copy()
        if segment_columns
        else pd.DataFrame(index=aligned.index)
    )
    if operating_regime is not None:
        segment_base["operating_regime"] = operating_regime.astype("string")
    segment_base["y_true"] = aligned["y_true"].to_numpy(dtype=float)
    segment_base["candidate_pred"] = candidate_predictions.to_numpy(dtype=float)
    segment_base["persistence_pred"] = aligned["y_persist"].to_numpy(dtype=float)
    segment_base["best_baseline_pred"] = (
        baseline_predictions[best_baseline_label].reindex(aligned.index).astype(float).to_numpy()
    )
    for segment_column in segment_columns:
        grouped = segment_base.groupby(segment_column, dropna=False)
        for segment_value, group in grouped:
            candidate_metrics = compute_regression_metrics(
                group["y_true"],
                group["candidate_pred"],
                n_total=int(len(group)),
            )
            persistence_metrics = compute_regression_metrics(
                group["y_true"],
                group["persistence_pred"],
                n_total=int(len(group)),
            )
            best_baseline_metrics = compute_regression_metrics(
                group["y_true"],
                group["best_baseline_pred"],
                n_total=int(len(group)),
            )
            segment_rows.append(
                {
                    "segment_column": segment_column,
                    "segment_value": segment_value,
                    "candidate_label": candidate_name,
                    "candidate_mae": float(candidate_metrics["mae"]),
                    "candidate_mae_pct": float(candidate_metrics["mae_pct"]),
                    "persistence_mae": float(persistence_metrics["mae"]),
                    "persistence_mae_pct": float(persistence_metrics["mae_pct"]),
                    "best_baseline_label": best_baseline_label,
                    "best_baseline_mae": float(best_baseline_metrics["mae"]),
                    "best_baseline_mae_pct": float(best_baseline_metrics["mae_pct"]),
                    "candidate_mae_ratio_to_persistence": mae_ratio(
                        float(candidate_metrics["mae"]),
                        float(persistence_metrics["mae"]),
                    ),
                    "candidate_mae_ratio_to_best_baseline": mae_ratio(
                        float(candidate_metrics["mae"]),
                        float(best_baseline_metrics["mae"]),
                    ),
                    "rows": int(len(group)),
                }
            )
    operating_regime_evaluation = (
        pd.DataFrame(segment_rows)
        .loc[lambda frame: frame["segment_column"].astype("string").eq("operating_regime")]
        .reset_index(drop=True)
        if segment_rows
        else pd.DataFrame()
    )
    operating_policy = _build_stage5_operating_policy(
        deployment_recommendation=deployment_recommendation,
        candidate_label=candidate_name,
        best_baseline_label=best_baseline_label,
        operating_regime_evaluation=operating_regime_evaluation,
    )
    holdout_coverage_frame = eval_df.loc[aligned.index].copy()
    if operating_regime is not None:
        holdout_coverage_frame["operating_regime"] = operating_regime.astype("string")
    holdout_coverage_segments, holdout_coverage_summary = _build_holdout_coverage_summary(
        holdout_frame=holdout_coverage_frame,
        segment_columns=segment_columns,
    )
    deployment_recommendation.update(
        {
            "operating_policy_mode": str(operating_policy["policy_mode"]),
            "standalone_operating_role": str(operating_policy["standalone_operating_role"]),
            "stage10_operating_role": str(operating_policy["stage10_operating_role"]),
            "operating_policy_default_candidate_label": str(operating_policy["default_candidate_label"]),
            "operating_policy_override_count": int(operating_policy["summary"]["override_count"]),
            "operating_policy_learned_supported_regime_count": int(
                operating_policy["summary"]["learned_supported_regime_count"]
            ),
        }
    )
    holdout_predictions, prediction_columns = _build_holdout_predictions_frame(
        eval_df=eval_df,
        aligned=aligned,
        candidate_label=candidate_name,
        candidate_predictions=candidate_predictions,
        baseline_predictions=baseline_predictions,
    )
    holdout_inference = _build_holdout_inference(
        prediction_frame=holdout_predictions,
        prediction_columns=prediction_columns,
        candidate_label=candidate_name,
        best_baseline_label=best_baseline_label,
        resolution=resolution,
    )
    supplemental_surface = _build_supplemental_surface_artifacts(
        gold=gold,
        folds=folds,
        feature_sets=feature_sets,
        model_catalog=model_catalog,
        resolution=resolution,
        promoted_candidate=promoted_candidate,
        steps_per_day=steps_per_day,
        blend_config=blend_config,
        holdout_predictions=holdout_predictions,
        holdout_prediction_columns=prediction_columns,
    )
    feature_importance, feature_importance_summary = _compute_holdout_feature_importance(
        model=bundle["model"],
        x_eval=cast(pd.DataFrame, bundle["x_eval"]),
        y_true=aligned["y_true"].astype(float),
        target_mode=base_target_mode,
    )
    if feature_importance_summary is not None:
            feature_importance_summary = {
                **feature_importance_summary,
                "candidate_label": candidate_name,
                "base_target_mode": base_target_mode,
                "blend_wrapper_applied": bool(blend_policy_kind),
                "best_baseline_label": best_baseline_label,
            }
    shap_importance: pd.DataFrame | None = None
    shap_importance_summary: dict[str, Any] | None = None
    if MODELING_PERFORMANCE_EVALUATION.get("compute_shap", False):
        shap_importance, shap_importance_summary = _compute_holdout_shap_importance(
            model=bundle["model"],
            x_eval=cast(pd.DataFrame, bundle["x_eval"]),
            candidate_label=candidate_name,
        )
    return HoldoutDiagnostics(
        holdout_summary=holdout_summary.sort_values(["candidate_type", "mae"], ascending=[True, True], kind="stable").reset_index(drop=True),
        holdout_blend_decisions=holdout_decisions,
        deployment_recommendation=deployment_recommendation,
        holdout_segment_evaluation=pd.DataFrame(segment_rows),
        holdout_operating_regime_evaluation=operating_regime_evaluation,
        operating_policy=operating_policy,
        holdout_coverage_segments=holdout_coverage_segments,
        holdout_coverage_summary=holdout_coverage_summary,
        holdout_predictions=holdout_predictions,
        holdout_inference=holdout_inference,
        feature_importance=feature_importance,
        feature_importance_summary=feature_importance_summary,
        shap_importance=shap_importance,
        shap_importance_summary=shap_importance_summary,
        supplemental_surface_summary=cast(pd.DataFrame | None, supplemental_surface.get("summary")),
        supplemental_surface_source_evaluation=cast(pd.DataFrame | None, supplemental_surface.get("source_evaluation")),
        supplemental_surface_segment_evaluation=cast(pd.DataFrame | None, supplemental_surface.get("segment_evaluation")),
        supplemental_surface_operating_regime_evaluation=cast(pd.DataFrame | None, supplemental_surface.get("operating_regime_evaluation")),
        supplemental_surface_coverage_segments=cast(pd.DataFrame | None, supplemental_surface.get("coverage_segments")),
        supplemental_surface_coverage_summary=cast(dict[str, Any] | None, supplemental_surface.get("coverage_summary")),
        supplemental_surface_predictions=cast(pd.DataFrame | None, supplemental_surface.get("predictions")),
        supplemental_surface_advisory=cast(dict[str, Any] | None, supplemental_surface.get("advisory")),
    )


def _build_supplemental_surface_artifacts(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]] | None,
    feature_sets: dict[str, list[str]],
    model_catalog: dict[str, ModelSpec],
    resolution: str,
    promoted_candidate: dict[str, Any],
    steps_per_day: int,
    blend_config: BlendConfig | BucketBlendConfig | SigmoidBucketBlendConfig | None,
    holdout_predictions: pd.DataFrame,
    holdout_prediction_columns: dict[str, str],
) -> dict[str, Any]:
    """Build advisory-only Stage-5 evidence across validate walk-forward plus holdout rows."""
    if not folds:
        return {}
    configured_splits = {
        str(split_name) for split_name in MODELING_PERFORMANCE_EVALUATION["supplemental_surface_splits"]
    }
    if not configured_splits:
        return {}

    feature_set = str(promoted_candidate["feature_set"])
    model_label = str(promoted_candidate["model_label"])
    target_mode = str(promoted_candidate["target_mode"])
    base_target_mode = _stage5_base_target_mode(target_mode)
    candidate_name = f"{feature_set}/{model_label}/{target_mode}"
    feature_columns = _resolve_feature_set_columns(feature_set, feature_sets=feature_sets)
    model_spec = model_catalog[model_label]

    supplemental_frames: list[pd.DataFrame] = []
    if "validate" in configured_splits:
        validate_payloads = _build_stage5_eval_payloads(
            gold=gold,
            folds=folds,
            feature_cols=feature_columns,
            model_spec=model_spec,
            target_mode=base_target_mode,
            steps_per_day=steps_per_day,
        )
        validate_range = SPLIT_DAY_RANGES["validate"]
        for payload in validate_payloads:
            fold_meta = cast(dict[str, int], payload["fold_meta"])
            if not (
                int(fold_meta["val_start_day"]) >= int(validate_range[0])
                and int(fold_meta["val_end_day"]) <= int(validate_range[1])
            ):
                continue
            aligned = cast(pd.DataFrame, payload["aligned"])
            eval_df = cast(pd.DataFrame, payload["eval_df"])
            train_df = cast(pd.DataFrame, payload["train_df"])
            candidate_predictions, _, _ = _stage5_surface_candidate_predictions(
                aligned=aligned,
                eval_df=eval_df,
                resolution=resolution,
                feature_set=feature_set,
                model_label=model_label,
                target_mode=target_mode,
                blend_config=blend_config,
                n_eval_total=int(payload["n_eval_total"]),
            )
            baseline_predictions = _build_holdout_baseline_predictions(
                train_df=train_df,
                eval_df=eval_df,
                aligned=aligned,
                resolution=resolution,
            )
            fold_predictions, prediction_columns = _build_holdout_predictions_frame(
                eval_df=eval_df,
                aligned=aligned,
                candidate_label=candidate_name,
                candidate_predictions=candidate_predictions,
                baseline_predictions=baseline_predictions,
                extra_columns={
                    "evaluation_surface": "validate_walkforward",
                    "fold": int(fold_meta["fold"]),
                    "surface_train_start_day": int(fold_meta["train_start_day"]),
                    "surface_train_end_day": int(fold_meta["train_end_day"]),
                    "surface_eval_start_day": int(fold_meta["val_start_day"]),
                    "surface_eval_end_day": int(fold_meta["val_end_day"]),
                },
            )
            if prediction_columns != holdout_prediction_columns:
                missing_columns = sorted(
                    set(holdout_prediction_columns.values()) - set(fold_predictions.columns)
                )
                for column in missing_columns:
                    fold_predictions[column] = float("nan")
                fold_predictions = fold_predictions.reindex(
                    columns=list(dict.fromkeys([*holdout_predictions.columns, *fold_predictions.columns]))
                )
            supplemental_frames.append(fold_predictions)

    if "test" in configured_splits and not holdout_predictions.empty:
        holdout_surface = holdout_predictions.copy()
        holdout_surface["evaluation_surface"] = "test_holdout"
        holdout_surface["fold"] = pd.NA
        holdout_surface["surface_train_start_day"] = int(SPLIT_DAY_RANGES["train"][0])
        holdout_surface["surface_train_end_day"] = int(SPLIT_DAY_RANGES["validate"][1])
        holdout_surface["surface_eval_start_day"] = int(SPLIT_DAY_RANGES["test"][0])
        holdout_surface["surface_eval_end_day"] = int(SPLIT_DAY_RANGES["test"][1])
        supplemental_frames.append(holdout_surface)

    if not supplemental_frames:
        return {}

    supplemental_predictions = pd.concat(supplemental_frames, ignore_index=True, sort=False)
    operating_regime = _derive_operating_regime(supplemental_predictions)
    supplemental_predictions["operating_regime"] = (
        operating_regime.astype("string")
        if operating_regime is not None
        else pd.Series("unknown", index=supplemental_predictions.index, dtype="string")
    )
    supplemental_predictions["actual_load_band"] = _derive_supplemental_actual_load_band(
        supplemental_predictions
    )
    supplemental_predictions["actual_ramp_band"] = _derive_supplemental_actual_ramp_band(
        supplemental_predictions
    )
    supplemental_summary, best_baseline_label, best_baseline_mae, _, persistence_mae = _prediction_surface_summary(
        prediction_frame=supplemental_predictions,
        prediction_columns=holdout_prediction_columns,
        candidate_label=candidate_name,
        resolution=resolution,
        feature_set=feature_set,
        model_label=model_label,
        target_mode=target_mode,
    )
    if supplemental_summary.empty:
        return {}
    source_rows: list[pd.DataFrame] = []
    for surface_name, group in supplemental_predictions.groupby("evaluation_surface", dropna=False):
        surface_summary, _, _, _, _ = _prediction_surface_summary(
            prediction_frame=group.reset_index(drop=True),
            prediction_columns=holdout_prediction_columns,
            candidate_label=candidate_name,
            resolution=resolution,
            feature_set=feature_set,
            model_label=model_label,
            target_mode=target_mode,
        )
        if surface_summary.empty:
            continue
        surface_summary["evaluation_surface"] = str(surface_name)
        source_rows.append(surface_summary)
    source_evaluation = (
        pd.concat(source_rows, ignore_index=True).reset_index(drop=True)
        if source_rows
        else pd.DataFrame()
    )
    segment_evaluation, operating_regime_evaluation, coverage_segments, coverage_summary = (
        _prediction_surface_segment_evaluation(
            prediction_frame=supplemental_predictions,
            prediction_columns=holdout_prediction_columns,
            candidate_label=candidate_name,
            best_baseline_label=best_baseline_label,
        )
    )
    summary_candidate_row = supplemental_summary.loc[
        supplemental_summary["candidate_label"].astype("string").eq(candidate_name)
    ].iloc[0]
    learned_supported_regimes = sorted(
        {
            str(getattr(row, "segment_value"))
            for row in operating_regime_evaluation.itertuples(index=False)
            if int(getattr(row, "rows", 0)) >= OPERATING_POLICY_MIN_SEGMENT_ROWS
            and float(getattr(row, "candidate_mae")) < float(getattr(row, "persistence_mae"))
            and float(getattr(row, "candidate_mae")) < float(getattr(row, "best_baseline_mae"))
        }
    )
    evaluation_surface_counts = (
        supplemental_predictions["evaluation_surface"].astype("string").value_counts(dropna=False).to_dict()
    )
    operating_regime_counts = (
        supplemental_predictions["operating_regime"].astype("string").value_counts(dropna=False).to_dict()
    )
    advisory = {
        "surface_name": "supplemental_validate_walkforward_plus_holdout",
        "canonical_holdout_preserved": True,
        "supports_deployment_override": False,
        "validate_day_range": list(SPLIT_DAY_RANGES["validate"]),
        "holdout_day_range": list(SPLIT_DAY_RANGES["test"]),
        "row_n": int(len(supplemental_predictions)),
        "unique_day_n": int(
            pd.to_datetime(supplemental_predictions["timestamp"], errors="coerce").dt.normalize().nunique()
        ),
        "evaluation_surface_counts": {str(key): int(value) for key, value in evaluation_surface_counts.items()},
        "operating_regime_counts": {str(key): int(value) for key, value in operating_regime_counts.items()},
        "candidate_label": candidate_name,
        "best_baseline_label": best_baseline_label,
        "best_baseline_mae": float(best_baseline_mae),
        "persistence_mae": float(persistence_mae),
        "candidate_mae": float(summary_candidate_row["mae"]),
        "candidate_mae_ratio_to_persistence": float(summary_candidate_row["mae_ratio_to_persistence"]),
        "candidate_mae_ratio_to_best_baseline": float(summary_candidate_row["mae_ratio_to_best_baseline"]),
        "learned_beats_persistence": bool(float(summary_candidate_row["mae"]) < float(persistence_mae)),
        "learned_beats_best_baseline": bool(float(summary_candidate_row["mae"]) < float(best_baseline_mae)),
        "learned_supported_operating_regimes": learned_supported_regimes,
        "learned_supported_operating_regime_count": int(len(learned_supported_regimes)),
        "reason": (
            "This supplemental Stage-5 surface stitches leakage-safe validate walk-forward rows together with the "
            "canonical test holdout to broaden the minute evidence surface. It is advisory only; "
            "deployment_recommendation.json remains the only deployment gate."
        ),
    }
    return {
        "summary": supplemental_summary.sort_values(
            ["candidate_type", "mae"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True),
        "source_evaluation": source_evaluation,
        "segment_evaluation": segment_evaluation,
        "operating_regime_evaluation": operating_regime_evaluation,
        "coverage_segments": coverage_segments,
        "coverage_summary": coverage_summary,
        "predictions": supplemental_predictions.reset_index(drop=True),
        "advisory": advisory,
    }


def _evaluate_promoted_holdout_candidate(
    *,
    gold: pd.DataFrame,
    folds: list[dict[str, int]] | None = None,
    feature_sets: dict[str, list[str]],
    model_catalog: dict[str, ModelSpec],
    resolution: str,
    promoted_candidate: dict[str, Any],
    steps_per_day: int,
    blend_config: BlendConfig | BucketBlendConfig | SigmoidBucketBlendConfig | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any], pd.DataFrame]:
    """Return the legacy Stage-5 holdout tuple while richer diagnostics live beside it."""
    diagnostics = _evaluate_promoted_holdout_artifacts(
        gold=gold,
        folds=folds,
        feature_sets=feature_sets,
        model_catalog=model_catalog,
        resolution=resolution,
        promoted_candidate=promoted_candidate,
        steps_per_day=steps_per_day,
        blend_config=blend_config,
    )
    return (
        diagnostics.holdout_summary,
        diagnostics.holdout_blend_decisions,
        diagnostics.deployment_recommendation,
        diagnostics.holdout_segment_evaluation,
    )


def _write_selection_frontier_figure(selection_scoreboard: pd.DataFrame, output_path: Path) -> None:
    """Write a compact Stage-5 frontier plot for quick visual review."""
    if selection_scoreboard.empty:
        return
    plot_df = selection_scoreboard.nsmallest(24, "fold_mean_mae_ratio").copy()
    if plot_df.empty:
        return
    plot_df["label"] = (
        plot_df["feature_set"].astype(str)
        + "/"
        + plot_df["model_label"].astype(str)
        + "/"
        + plot_df["target_mode"].astype(str)
    )
    def _target_mode_color(target_mode: str) -> str:
        if _stage5_blend_policy_kind(target_mode) == "bucket":
            return "#2ca02c"
        if _stage5_blend_policy_kind(target_mode) == "sigmoid":
            return "#54a24b"
        if str(target_mode) == "residual":
            return "#f58518"
        return "#4c78a8"

    fig, ax = plt.subplots(figsize=(11, 6))
    for target_mode, subset in plot_df.groupby("target_mode", dropna=False):
        ax.scatter(
            subset["mean_coverage"],
            subset["fold_mean_mae_ratio"],
            label=str(target_mode),
            color=_target_mode_color(str(target_mode)),
            alpha=0.85,
            s=48,
        )
    for _, row in plot_df.nsmallest(8, "fold_mean_mae_ratio").iterrows():
        ax.annotate(
            str(row["label"]),
            (float(row["mean_coverage"]), float(row["fold_mean_mae_ratio"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_title("Stage-5 Selection Frontier")
    ax.set_xlabel("Mean validation coverage")
    ax.set_ylabel("Fold mean MAE ratio vs persistence")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(title="Target mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    validate_png_artifact(output_path)
    plt.close(fig)


def _write_hgb_tradeoff_figure(hgb_coordinate_summary: pd.DataFrame, output_path: Path) -> None:
    """Write the HGB error-vs-generalization tradeoff figure."""
    if hgb_coordinate_summary is None or hgb_coordinate_summary.empty:
        return
    plot_df = hgb_coordinate_summary.copy()
    colors = np.where(plot_df["meets_p1b_acceptance"], "#54a24b", "#e45756")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        plot_df["train_val_gap_to_one"],
        plot_df["fold_mean_mae_ratio"],
        c=colors,
        alpha=0.9,
        s=58,
    )
    for _, row in plot_df.head(8).iterrows():
        ax.annotate(
            str(row["model_label"]),
            (float(row["train_val_gap_to_one"]), float(row["fold_mean_mae_ratio"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_title("HGB Coordinate and Frontier Tradeoff")
    ax.set_xlabel("|Train/validation MAE ratio - 1|")
    ax.set_ylabel("Fold mean MAE ratio vs persistence")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    validate_png_artifact(output_path)
    plt.close(fig)


def _write_holdout_benchmark_ci_figure(
    holdout_summary: pd.DataFrame,
    holdout_inference: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize holdout MAE point estimates with moving-block bootstrap confidence intervals."""
    if holdout_summary.empty or holdout_inference.empty:
        return
    mae_rows = holdout_inference.loc[holdout_inference["metric_name"].astype("string").eq("mae")].copy()
    if mae_rows.empty:
        return
    interval_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for _, row in mae_rows.iterrows():
        candidate_label = str(row["candidate_label"])
        if candidate_label not in seen_labels:
            seen_labels.add(candidate_label)
            interval_rows.append(
                {
                    "candidate_label": candidate_label,
                    "candidate_metric": float(row["candidate_metric"]),
                    "candidate_metric_ci_low": float(row["candidate_metric_ci_low"]),
                    "candidate_metric_ci_high": float(row["candidate_metric_ci_high"]),
                }
            )
        baseline_label = str(row["baseline_label"])
        if baseline_label not in seen_labels:
            seen_labels.add(baseline_label)
            interval_rows.append(
                {
                    "candidate_label": baseline_label,
                    "candidate_metric": float(row["baseline_metric"]),
                    "candidate_metric_ci_low": float(row["baseline_metric_ci_low"]),
                    "candidate_metric_ci_high": float(row["baseline_metric_ci_high"]),
                }
            )
    interval_df = pd.DataFrame(interval_rows)
    plot_df = holdout_summary.merge(interval_df, on="candidate_label", how="left")
    plot_df = plot_df.sort_values(["mae", "candidate_label"], ascending=[True, True], kind="stable")
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.9 * len(plot_df))))
    colors = np.where(
        plot_df["candidate_type"].astype("string").eq("promoted_learned"),
        "#4c78a8",
        "#9c755f",
    )
    y_positions = np.arange(len(plot_df))
    mae_values = plot_df["mae"].astype(float).to_numpy()
    lower = mae_values - plot_df["candidate_metric_ci_low"].astype(float).to_numpy()
    upper = plot_df["candidate_metric_ci_high"].astype(float).to_numpy() - mae_values
    ax.errorbar(
        mae_values,
        y_positions,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#666666",
        elinewidth=1.2,
        capsize=3,
        alpha=0.85,
    )
    ax.scatter(mae_values, y_positions, c=colors, s=62, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"{label} ({float(mae_pct):.2f}%)"
            for label, mae_pct in zip(
                plot_df["candidate_label"].astype(str),
                plot_df["mae_pct"].astype(float),
                strict=False,
            )
        ]
    )
    ax.set_xlabel("Holdout MAE (watts)")
    ax.set_title("Stage-5 Holdout MAE with 95% Block-Bootstrap Intervals")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    validate_png_artifact(output_path)
    plt.close(fig)


def _write_feature_importance_figure(
    feature_importance: pd.DataFrame,
    *,
    candidate_label: str,
    output_path: Path,
) -> None:
    """Write the top permutation-importance features for the learned holdout challenger."""
    if feature_importance.empty:
        return
    plot_df = feature_importance.copy()
    plot_df = plot_df.sort_values(["importance_mean", "feature"], ascending=[True, True], kind="stable")
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(plot_df))))
    ax.barh(
        plot_df["feature"].astype(str),
        plot_df["importance_mean"].astype(float),
        xerr=plot_df["importance_std"].astype(float),
        color="#54a24b",
        alpha=0.9,
        error_kw={"elinewidth": 1.0, "ecolor": "#2f4b1a", "capsize": 2},
    )
    ax.set_xlabel("Permutation importance on holdout MAE")
    ax.set_title(f"Stage-5 Learned Challenger Feature Importance\n{candidate_label}")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    validate_png_artifact(output_path)
    plt.close(fig)


def _write_outputs(
    *,
    output_dir: Path,
    output_root: Path,
    run_mode: str,
    metrics_fold: pd.DataFrame,
    selection_scoreboard: pd.DataFrame,
    residual_ablation: pd.DataFrame,
    folds: list[dict[str, int]],
    resolution: str,
    selected_feature_sets: list[str],
    feature_sets: dict[str, list[str]],
    selected_models: list[ModelSpec],
    include_residual: bool,
    ramp_quantile: float | None,
    ramp_threshold: float | None,
    coverage_audit: pd.DataFrame,
    promoted_candidate: dict[str, Any] | None,
    blend_finalists: pd.DataFrame | None,
    holdout_summary: pd.DataFrame | None,
    holdout_segment_evaluation: pd.DataFrame | None,
    holdout_operating_regime_evaluation: pd.DataFrame | None,
    holdout_coverage_segments: pd.DataFrame | None,
    holdout_coverage_summary: dict[str, Any] | None,
    holdout_blend_decisions: pd.DataFrame | None,
    deployment_recommendation: dict[str, Any] | None,
    operating_policy: dict[str, Any] | None,
    holdout_predictions: pd.DataFrame | None,
    holdout_inference: pd.DataFrame | None,
    feature_importance: pd.DataFrame | None,
    feature_importance_summary: dict[str, Any] | None,
    shap_importance: pd.DataFrame | None,
    shap_importance_summary: dict[str, Any] | None,
    supplemental_surface_summary: pd.DataFrame | None,
    supplemental_surface_source_evaluation: pd.DataFrame | None,
    supplemental_surface_segment_evaluation: pd.DataFrame | None,
    supplemental_surface_operating_regime_evaluation: pd.DataFrame | None,
    supplemental_surface_coverage_segments: pd.DataFrame | None,
    supplemental_surface_coverage_summary: dict[str, Any] | None,
    supplemental_surface_predictions: pd.DataFrame | None,
    supplemental_surface_advisory: dict[str, Any] | None,
    blend_config: BlendConfig | BucketBlendConfig | SigmoidBucketBlendConfig | None,
    blend_candidate: dict[str, Any] | None,
    guardrail_decisions: pd.DataFrame | None,
    hgb_coordinate_summary: pd.DataFrame | None,
    hgb_coordinate_recommended: dict[str, Any] | None,
    adaptive_hgb_screen: pd.DataFrame | None,
    horizon_policy: dict[str, Any],
    parallel_plan: ParallelPlan,
) -> None:
    """Persist the full Stage-5 artifact set, figures, and run manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_fold.csv"
    metrics_fold.to_csv(metrics_path, index=False, float_format="%.6f")
    summary_records = selection_scoreboard.to_dict(orient="records")
    (output_dir / "metrics_fold_summary.json").write_text(
        json.dumps(summary_records, indent=2), encoding="utf-8"
    )
    selection_scoreboard.to_csv(output_dir / "selection_scoreboard.csv", index=False, float_format="%.6f")
    residual_ablation.to_csv(output_dir / "residual_ablation.csv", index=False, float_format="%.6f")
    coverage_audit.to_csv(output_dir / "coverage_audit.csv", index=False, float_format="%.6f")
    _write_selection_frontier_figure(selection_scoreboard, output_dir / "fig_selection_frontier.png")
    if promoted_candidate is not None:
        (output_dir / "promotion_candidate.json").write_text(
            json.dumps(_json_safe(promoted_candidate), indent=2),
            encoding="utf-8",
        )
    if blend_finalists is not None and not blend_finalists.empty:
        blend_finalists.to_csv(output_dir / "blend_finalists.csv", index=False, float_format="%.6f")
    if holdout_summary is not None and not holdout_summary.empty:
        holdout_summary.to_csv(output_dir / "holdout_evaluation.csv", index=False, float_format="%.6f")
    if holdout_segment_evaluation is not None and not holdout_segment_evaluation.empty:
        holdout_segment_evaluation.to_csv(
            output_dir / "holdout_segment_evaluation.csv",
            index=False,
            float_format="%.6f",
        )
    if holdout_operating_regime_evaluation is not None and not holdout_operating_regime_evaluation.empty:
        holdout_operating_regime_evaluation.to_csv(
            output_dir / "holdout_operating_regime_evaluation.csv",
            index=False,
            float_format="%.6f",
        )
    if holdout_coverage_segments is not None and not holdout_coverage_segments.empty:
        holdout_coverage_segments.to_csv(
            output_dir / "holdout_coverage_segments.csv",
            index=False,
            float_format="%.6f",
        )
    if holdout_coverage_summary is not None:
        (output_dir / "holdout_coverage_summary.json").write_text(
            json.dumps(_json_safe(holdout_coverage_summary), indent=2),
            encoding="utf-8",
        )
    if holdout_blend_decisions is not None and not holdout_blend_decisions.empty:
        holdout_blend_decisions.to_csv(
            output_dir / "holdout_blend_decisions.csv",
            index=False,
            float_format="%.6f",
        )
    if deployment_recommendation is not None:
        (output_dir / "deployment_recommendation.json").write_text(
            json.dumps(_json_safe(deployment_recommendation), indent=2),
            encoding="utf-8",
        )
    if operating_policy is not None:
        (output_dir / "operating_policy.json").write_text(
            json.dumps(_json_safe(operating_policy), indent=2),
            encoding="utf-8",
        )
    if holdout_predictions is not None and not holdout_predictions.empty:
        holdout_predictions.to_csv(
            output_dir / "holdout_predictions.csv",
            index=False,
            float_format="%.6f",
        )
    if holdout_inference is not None and not holdout_inference.empty:
        holdout_inference.to_csv(
            output_dir / "holdout_inference.csv",
            index=False,
            float_format="%.6f",
        )
        _write_holdout_benchmark_ci_figure(
            holdout_summary if holdout_summary is not None else pd.DataFrame(),
            holdout_inference,
            output_dir / "fig_holdout_benchmark_ci.png",
        )
    if supplemental_surface_summary is not None and not supplemental_surface_summary.empty:
        supplemental_surface_summary.to_csv(
            output_dir / "supplemental_surface_summary.csv",
            index=False,
            float_format="%.6f",
        )
    if supplemental_surface_source_evaluation is not None and not supplemental_surface_source_evaluation.empty:
        supplemental_surface_source_evaluation.to_csv(
            output_dir / "supplemental_surface_source_evaluation.csv",
            index=False,
            float_format="%.6f",
        )
    if supplemental_surface_segment_evaluation is not None and not supplemental_surface_segment_evaluation.empty:
        supplemental_surface_segment_evaluation.to_csv(
            output_dir / "supplemental_surface_segment_evaluation.csv",
            index=False,
            float_format="%.6f",
        )
    if (
        supplemental_surface_operating_regime_evaluation is not None
        and not supplemental_surface_operating_regime_evaluation.empty
    ):
        supplemental_surface_operating_regime_evaluation.to_csv(
            output_dir / "supplemental_surface_operating_regime_evaluation.csv",
            index=False,
            float_format="%.6f",
        )
    if supplemental_surface_coverage_segments is not None and not supplemental_surface_coverage_segments.empty:
        supplemental_surface_coverage_segments.to_csv(
            output_dir / "supplemental_surface_coverage_segments.csv",
            index=False,
            float_format="%.6f",
        )
    if supplemental_surface_coverage_summary is not None:
        (output_dir / "supplemental_surface_coverage_summary.json").write_text(
            json.dumps(_json_safe(supplemental_surface_coverage_summary), indent=2),
            encoding="utf-8",
        )
    if supplemental_surface_predictions is not None and not supplemental_surface_predictions.empty:
        supplemental_surface_predictions.to_csv(
            output_dir / "supplemental_surface_predictions.csv",
            index=False,
            float_format="%.6f",
        )
    if supplemental_surface_advisory is not None:
        (output_dir / "supplemental_surface_advisory.json").write_text(
            json.dumps(_json_safe(supplemental_surface_advisory), indent=2),
            encoding="utf-8",
        )
    if feature_importance is not None and not feature_importance.empty:
        feature_importance.to_csv(
            output_dir / "feature_importance_permutation.csv",
            index=False,
            float_format="%.6f",
        )
        candidate_label = (
            str(feature_importance_summary.get("candidate_label"))
            if isinstance(feature_importance_summary, dict)
            else "learned_candidate"
        )
        _write_feature_importance_figure(
            feature_importance,
            candidate_label=candidate_label,
            output_path=output_dir / "fig_feature_importance.png",
        )
    if feature_importance_summary is not None:
        (output_dir / "feature_importance_summary.json").write_text(
            json.dumps(_json_safe(feature_importance_summary), indent=2),
            encoding="utf-8",
        )
    if shap_importance is not None and not shap_importance.empty:
        shap_importance.to_csv(
            output_dir / "shap_importance.csv",
            index=False,
            float_format="%.6f",
        )
        shap_candidate_label = (
            str(shap_importance_summary.get("candidate_label"))
            if isinstance(shap_importance_summary, dict)
            else "learned_candidate"
        )
        _write_shap_importance_figure(
            shap_importance,
            candidate_label=shap_candidate_label,
            output_path=output_dir / "fig_shap_importance.png",
        )
    if shap_importance_summary is not None:
        (output_dir / "shap_importance_summary.json").write_text(
            json.dumps(_json_safe(shap_importance_summary), indent=2),
            encoding="utf-8",
        )
    if hgb_coordinate_summary is not None and not hgb_coordinate_summary.empty:
        hgb_coordinate_summary.to_csv(
            output_dir / "hgb_coordinate_summary.csv",
            index=False,
            float_format="%.6f",
        )
        _write_hgb_tradeoff_figure(hgb_coordinate_summary, output_dir / "fig_hgb_tradeoff.png")
    if adaptive_hgb_screen is not None and not adaptive_hgb_screen.empty:
        adaptive_hgb_screen.to_csv(
            output_dir / "adaptive_hgb_screen.csv",
            index=False,
            float_format="%.6f",
        )
    if guardrail_decisions is not None and not guardrail_decisions.empty:
        guardrail_decisions.to_csv(output_dir / "guardrail_decisions.csv", index=False, float_format="%.6f")
        guardrail_summary = (
            guardrail_decisions.groupby(["resolution", "feature_set", "model_label"], dropna=False)
            .agg(
                mean_blend_weight=("blend_weight", "mean"),
                model_dominated_frac=("blend_weight", lambda x: float((x >= 0.5).mean())),
                mean_model_abs_error=("model_abs_error", "mean"),
                mean_persistence_abs_error=("persistence_abs_error", "mean"),
                mean_blend_abs_error=("blend_abs_error", "mean"),
                rows=("blend_weight", "size"),
            )
            .reset_index()
        )
        guardrail_summary.to_csv(output_dir / "guardrail_summary.csv", index=False, float_format="%.6f")
    figure_guide_entries = [
        FigureGuideEntry(
            filename="fig_selection_frontier.png",
            title="Selection frontier",
            intent="Show which Stage-5 candidates balance low error against strong validation coverage.",
            how_to_read="Read left-to-right as coverage and bottom-to-top as MAE ratio versus persistence. Better candidates sit low and to the right.",
            look_for="Candidates below 1.0 MAE ratio that also stay near full coverage; low-coverage wins should not be promoted.",
        )
    ]
    if holdout_inference is not None and not holdout_inference.empty:
        figure_guide_entries.append(
            FigureGuideEntry(
                filename="fig_holdout_benchmark_ci.png",
                title="Holdout benchmark intervals",
                intent="Show the promoted learned challenger and the short-horizon baselines with autocorrelation-aware holdout uncertainty, not just point estimates.",
                how_to_read="Each point is holdout MAE and the horizontal bar is the 95% moving-block bootstrap interval. Compare overlap, but rely on `holdout_inference.csv` for the exact paired delta tests.",
                look_for="Whether the learned challenger clearly separates from persistence or whether the intervals mostly overlap, which is common on strongly autocorrelated 1-minute load.",
            )
        )
    if feature_importance is not None and not feature_importance.empty:
        figure_guide_entries.append(
            FigureGuideEntry(
                filename="fig_feature_importance.png",
                title="Learned challenger feature importance",
                intent="Show which predictors matter most for the current best learned short-horizon challenger.",
                how_to_read="Longer bars mean permuting that feature hurts holdout MAE more. Importance is measured on the honest holdout window, so small values indicate the model depends mostly on autocorrelation and only secondarily on added features.",
                look_for="Whether a small set of lag, phase, or profile features dominates the top of the chart and how quickly the cumulative importance concentrates.",
            )
        )
    if shap_importance is not None and not shap_importance.empty:
        figure_guide_entries.append(
            FigureGuideEntry(
                filename="fig_shap_importance.png",
                title="SHAP feature importance",
                intent="Show which predictors contribute most to each individual prediction via Shapley additive explanations.",
                how_to_read="Longer bars indicate higher mean absolute SHAP values in watts. Unlike permutation importance, SHAP decomposes each prediction, revealing per-sample feature contributions.",
                look_for="Whether lag_1 dominates (expected for 1-minute load) and whether ramp or profile features have meaningfully different SHAP rankings compared to permutation importance.",
            )
        )
    if hgb_coordinate_summary is not None and not hgb_coordinate_summary.empty:
        figure_guide_entries.append(
            FigureGuideEntry(
                filename="fig_hgb_tradeoff.png",
                title="HGB tradeoff surface",
                intent="Show whether lower fold MAE comes from a stable HGB variant or from a fragile overfit candidate.",
                how_to_read="The x-axis is train-validation gap and the y-axis is MAE ratio versus persistence. Better points stay lower and closer to zero gap.",
                look_for="Variants that reduce error without increasing the train-validation gap; rejected points usually buy error with instability.",
            )
        )
    write_figure_guide(
        output_path=output_dir / "figure_guide.md",
        stage_title="Stage-5 Performance Figures",
        stage_purpose=(
            "These figures explain how Stage-5 measures candidate quality, why "
            "coverage matters, where the promoted challenger sits relative to "
            "short-horizon baselines, and which features actually drive the learned "
            "improvement that remains after the persistence anchor."
        ),
        figures=figure_guide_entries,
    )
    manifest = {
        "run_id": output_dir.name,
        "stage": "005_performance",
        "mode": run_mode,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "git_commit": _git_commit(),
        "resolution": resolution,
        "feature_sets": selected_feature_sets,
        "feature_set_columns": {name: feature_sets[name] for name in selected_feature_sets},
        "models": [
            {"model_label": model.model_label, "family": model.family, "params": model.params}
            for model in selected_models
        ],
        "include_residual": include_residual,
        "horizon_policy": _json_safe(horizon_policy),
        "include_curated_ramp": RAMP_FEATURE_SET_NAME in selected_feature_sets,
        "ramp_quantile": ramp_quantile,
        "ramp_threshold": ramp_threshold,
        "blend_policy": (
            (
                {
                    "enabled": True,
                    "policy_kind": "bucket",
                    "base_policy_kind": "sigmoid",
                    "bucket_size_minutes": blend_config.bucket_config.bucket_size_minutes,
                    "cycle_minutes": blend_config.bucket_config.cycle_minutes,
                    "bucket_weights": blend_config.bucket_config.weight_map(),
                    "pre_bucket_blend": {
                        "window": blend_config.blend_config.window,
                        "sharpness": blend_config.blend_config.sharpness,
                        "min_weight": blend_config.blend_config.min_weight,
                        "max_weight": blend_config.blend_config.max_weight,
                    },
                    "candidate": blend_candidate,
                }
                if isinstance(blend_config, SigmoidBucketBlendConfig)
                else {
                    "enabled": True,
                    "policy_kind": "bucket",
                    "base_policy_kind": "raw",
                    "bucket_size_minutes": blend_config.bucket_size_minutes,
                    "cycle_minutes": blend_config.cycle_minutes,
                    "bucket_weights": blend_config.weight_map(),
                    "candidate": blend_candidate,
                }
                if isinstance(blend_config, BucketBlendConfig)
                else {
                    "enabled": True,
                    "policy_kind": "sigmoid",
                    "window": blend_config.window,
                    "sharpness": blend_config.sharpness,
                    "min_weight": blend_config.min_weight,
                    "max_weight": blend_config.max_weight,
                    "candidate": blend_candidate,
                }
            )
            if blend_config is not None
            else {"enabled": False}
        ),
        "hgb_coordinate_search": (
            {
                "enabled": hgb_coordinate_summary is not None and not hgb_coordinate_summary.empty,
                "recommended": hgb_coordinate_recommended,
            }
            if hgb_coordinate_summary is not None
            else {"enabled": False}
        ),
        "parallel_runtime": {
            "config": dict(MODELING_PARALLEL),
            "resolved_plan": parallel_plan.as_dict(),
        },
        "runtime_environment": runtime_summary(int(MODELING_PARALLEL["max_workers"])).as_dict(),
        "promotion_policy": {
            "min_coverage": PROMOTION_MIN_COVERAGE,
            "selected_candidate": _json_safe(promoted_candidate),
            "deployment_recommendation": _json_safe(deployment_recommendation),
            "operating_policy": _json_safe(operating_policy),
            "supplemental_surface_advisory": _json_safe(supplemental_surface_advisory),
        },
        "folds": folds,
        "artifacts": [
            "metrics_fold.csv",
            "metrics_fold_summary.json",
            "selection_scoreboard.csv",
            "residual_ablation.csv",
            "coverage_audit.csv",
            "fig_selection_frontier.png",
            "figure_guide.md",
            *(
                ["blend_finalists.csv"]
                if blend_finalists is not None and not blend_finalists.empty
                else []
            ),
            *(
                [
                    "promotion_candidate.json",
                    "holdout_evaluation.csv",
                    "holdout_segment_evaluation.csv",
                    "deployment_recommendation.json",
                    "operating_policy.json",
                ]
                if promoted_candidate is not None and holdout_summary is not None and not holdout_summary.empty
                else []
            ),
            *(
                ["holdout_operating_regime_evaluation.csv"]
                if holdout_operating_regime_evaluation is not None and not holdout_operating_regime_evaluation.empty
                else []
            ),
            *(
                ["holdout_coverage_segments.csv", "holdout_coverage_summary.json"]
                if holdout_coverage_segments is not None and not holdout_coverage_segments.empty
                else []
            ),
            *(
                ["holdout_predictions.csv"]
                if holdout_predictions is not None and not holdout_predictions.empty
                else []
            ),
            *(
                [
                    "supplemental_surface_summary.csv",
                    "supplemental_surface_source_evaluation.csv",
                    "supplemental_surface_segment_evaluation.csv",
                    "supplemental_surface_operating_regime_evaluation.csv",
                    "supplemental_surface_coverage_segments.csv",
                    "supplemental_surface_coverage_summary.json",
                    "supplemental_surface_predictions.csv",
                    "supplemental_surface_advisory.json",
                ]
                if supplemental_surface_summary is not None and not supplemental_surface_summary.empty
                else []
            ),
            *(
                ["holdout_inference.csv", "fig_holdout_benchmark_ci.png"]
                if holdout_inference is not None and not holdout_inference.empty
                else []
            ),
            *(
                ["feature_importance_permutation.csv", "feature_importance_summary.json", "fig_feature_importance.png"]
                if feature_importance is not None and not feature_importance.empty
                else []
            ),
            *(
                ["shap_importance.csv", "shap_importance_summary.json", "fig_shap_importance.png"]
                if shap_importance is not None and not shap_importance.empty
                else []
            ),
            *(
                ["holdout_blend_decisions.csv"]
                if holdout_blend_decisions is not None and not holdout_blend_decisions.empty
                else []
            ),
            *(
                ["hgb_coordinate_summary.csv", "fig_hgb_tradeoff.png"]
                if hgb_coordinate_summary is not None and not hgb_coordinate_summary.empty
                else []
            ),
            *(
                ["adaptive_hgb_screen.csv"]
                if adaptive_hgb_screen is not None and not adaptive_hgb_screen.empty
                else []
            ),
            *(
                ["guardrail_decisions.csv", "guardrail_summary.csv"]
                if guardrail_decisions is not None and not guardrail_decisions.empty
                else []
            ),
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    refresh_stage5_holdout_registry(output_root)
    update_latest_alias(output_dir, output_root / LATEST_ALIAS_NAME, enabled=True)


def _prepare_output_run_dir(output_root: Path) -> Path:
    """Create one timestamped Stage-5 output directory."""
    run_dir = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_preflight_manifest(
    *,
    output_dir: Path,
    output_root: Path,
    resolution: str,
    selected_feature_sets: list[str],
    selected_models: list[ModelSpec],
    preflight: dict[str, Any],
) -> None:
    """Persist the preflight-only manifest when Stage-5 exits before model execution."""
    manifest = {
        "run_id": output_dir.name,
        "stage": "005_performance",
        "mode": "preflight",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "load_type": DATASET["load_type"],
        "artifact_namespace": DATASET["artifact_namespace"],
        "git_commit": _git_commit(),
        "resolution": resolution,
        "feature_sets": selected_feature_sets,
        "models": [
            {"model_label": model.model_label, "family": model.family, "params": model.params}
            for model in selected_models
        ],
        "preflight_status": preflight["overall_status"],
        "status": "success" if preflight["overall_status"] == "pass" else "failed",
        "runtime_environment": runtime_summary(int(MODELING_PARALLEL["max_workers"])).as_dict(),
        "artifacts": [
            "feature_causality_audit.csv",
            "minute_integrity_audit.csv",
            "holdout_lock.json",
            "preflight_audit.md",
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_latest_alias(output_dir, output_root / LATEST_ALIAS_NAME, enabled=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Stage-5 performance runner."""
    parser = argparse.ArgumentParser(
        description="Run model performance preflight + walk-forward evaluation."
    )
    parser.add_argument("--resolution", default="1min")
    parser.add_argument("--feature-set", action="append", dest="feature_sets")
    parser.add_argument("--model-label", action="append", dest="model_labels")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-window-days", type=int, default=2)
    parser.add_argument("--steps-per-day", type=int, default=1440)
    parser.add_argument("--horizon-minutes", type=int, default=1)
    parser.add_argument("--holdout-start-day", type=int, default=SPLIT_DAY_RANGES["test"][0])
    parser.add_argument("--tolerance-mae", type=float, default=0.1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--disable-curated-ramp", action="store_true")
    parser.add_argument("--disable-hgb-coordinate-search", action="store_true")
    parser.add_argument("--disable-adaptive-hgb-search", action="store_true")
    parser.add_argument("--disable-blend-policy", action="store_true")
    parser.add_argument(
        "--blend-window",
        type=int,
        default=MODELING_PERFORMANCE_BLEND_SEARCH["base_window"],
    )
    parser.add_argument(
        "--blend-sharpness",
        type=float,
        default=MODELING_PERFORMANCE_BLEND_SEARCH["base_sharpness"],
    )
    parser.add_argument(
        "--blend-min-weight",
        type=float,
        default=MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"],
    )
    parser.add_argument(
        "--blend-max-weight",
        type=float,
        default=MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"],
    )
    parser.add_argument(
        "--ramp-quantile",
        type=float,
        default=MODELING_PERFORMANCE_RAMP["quantile"],
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute Stage-5 using the parsed configuration and write one run directory."""
    _configure_logging()
    validate_config()
    args = parse_args()
    resolution = _canonical_resolution(args.resolution)
    output_root = Path(args.output_dir).resolve()
    output_dir = _prepare_output_run_dir(output_root)
    if args.preflight_only and args.skip_preflight:
        raise ValueError("--preflight-only cannot be combined with --skip-preflight.")
    feature_sets = _build_feature_sets(include_curated_ramp=not bool(args.disable_curated_ramp))
    horizon_policy = resolve_horizon_policy(int(args.horizon_minutes))
    if args.blend_window < 1:
        raise ValueError("--blend-window must be >= 1.")
    if not 0.0 <= float(args.blend_min_weight) <= 1.0:
        raise ValueError("--blend-min-weight must be within [0,1].")
    if not 0.0 <= float(args.blend_max_weight) <= 1.0:
        raise ValueError("--blend-max-weight must be within [0,1].")
    if float(args.blend_min_weight) > float(args.blend_max_weight):
        raise ValueError("--blend-min-weight must be <= --blend-max-weight.")

    selected_feature_sets = (
        list(args.feature_sets)
        if args.feature_sets
        else [name for name in horizon_policy["feature_sets"] if name in feature_sets]
    )
    for feature_set in selected_feature_sets:
        if feature_set not in feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}. Available: {sorted(feature_sets)}")

    include_hgb_coordinate_search = not bool(args.disable_hgb_coordinate_search)
    catalog = build_model_catalog(
        include_hgb_coordinate_search=include_hgb_coordinate_search,
        include_hgb_frontier=True,
    )
    model_labels = (
        list(args.model_labels)
        if args.model_labels
        else [label for label in horizon_policy["model_labels"] if label in catalog]
    )
    if not model_labels:
        model_labels = list(catalog.keys())
    for label in model_labels:
        if label not in catalog:
            raise ValueError(f"Unknown model label: {label}. Available: {sorted(catalog)}")
    selected_models = [catalog[label] for label in model_labels]
    n_folds = int(args.n_folds)
    val_window_days = int(args.val_window_days)
    include_residual = bool(horizon_policy["allow_residual"]) and not bool(args.disable_residual)
    blend_config = (
        BlendConfig(
            window=int(args.blend_window),
            sharpness=float(args.blend_sharpness),
            min_weight=float(args.blend_min_weight),
            max_weight=float(args.blend_max_weight),
        )
        if (
            MODELING_PERFORMANCE_BLEND_SEARCH["enabled"]
            and bool(horizon_policy["allow_blend"])
            and not bool(args.disable_blend_policy)
        )
        else None
    )

    if args.quick:
        quick_profile = resolve_performance_quick_profile(int(args.horizon_minutes))
        selected_feature_sets = [
            name for name in quick_profile["feature_sets"] if name in feature_sets
        ]
        selected_models = [
            catalog[label]
            for label in quick_profile["model_labels"]
            if label in catalog
        ]
        n_folds = int(quick_profile["n_folds"])
        val_window_days = int(quick_profile["val_window_days"])
        include_residual = True
        blend_config = (
            BlendConfig(
                window=int(MODELING_PERFORMANCE_BLEND_SEARCH["base_window"]),
                sharpness=float(MODELING_PERFORMANCE_BLEND_SEARCH["base_sharpness"]),
                min_weight=float(MODELING_PERFORMANCE_BLEND_SEARCH["min_weight"]),
                max_weight=float(MODELING_PERFORMANCE_BLEND_SEARCH["max_weight"]),
            )
            if (
                MODELING_PERFORMANCE_BLEND_SEARCH["enabled"]
                and bool(horizon_policy["allow_blend"])
                and not bool(args.disable_blend_policy)
            )
            else None
        )

    gold = _load_gold_with_full_grid(resolution, PATHS["gold_dir"])
    ramp_threshold: float | None = None
    if RAMP_FEATURE_SET_NAME in selected_feature_sets:
        gold, ramp_threshold = _augment_with_curated_ramp_features(
            gold,
            ramp_quantile=float(args.ramp_quantile),
        )
    folds = build_walkforward_folds(
        holdout_start_day=int(args.holdout_start_day),
        n_folds=n_folds,
        val_window_days=val_window_days,
        train_start_day=int(SPLIT_DAY_RANGES["train"][0]),
    )
    adaptive_hgb_screen = pd.DataFrame()
    if (
        not args.quick
        and not bool(args.disable_adaptive_hgb_search)
        and not args.model_labels
    ):
        adaptive_specs, adaptive_hgb_screen = _screen_adaptive_hgb_candidates(
            gold=gold,
            folds=folds,
            feature_sets=feature_sets,
            selected_feature_sets=selected_feature_sets,
            resolution=resolution,
            steps_per_day=int(args.steps_per_day),
            policy=horizon_policy,
        )
        if adaptive_specs:
            for spec in adaptive_specs:
                catalog[spec.model_label] = spec
            selected_models = list({
                spec.model_label: spec for spec in [*selected_models, *adaptive_specs]
            }.values())

    if not args.skip_preflight:
        preflight = run_preflight_audit(
            gold=gold,
            selected_feature_sets=selected_feature_sets,
            feature_sets=feature_sets,
            folds=folds,
            output_dir=output_dir,
            resolution=resolution,
            tolerance_mae=float(args.tolerance_mae),
            steps_per_day=int(args.steps_per_day),
        )
        logger.info("Preflight status: %s", preflight["overall_status"])
        if preflight["overall_status"] != "pass":
            logger.error("Preflight failed. Resolve findings before tuning.")
            _write_preflight_manifest(
                output_dir=output_dir,
                output_root=output_root,
                resolution=resolution,
                selected_feature_sets=selected_feature_sets,
                selected_models=selected_models,
                preflight=preflight,
            )
            return 1

    if args.preflight_only:
        _write_preflight_manifest(
            output_dir=output_dir,
            output_root=output_root,
            resolution=resolution,
            selected_feature_sets=selected_feature_sets,
            selected_models=selected_models,
            preflight=preflight,
        )
        logger.info("Preflight-only run complete.")
        return 0

    logger.info(
        "Running folds: resolution=%s feature_sets=%s models=%s folds=%d residual=%s",
        resolution,
        selected_feature_sets,
        [m.model_label for m in selected_models],
        len(folds),
        include_residual,
    )
    metrics_fold, parallel_plan = _run_fold_metrics(
        gold=gold,
        folds=folds,
        selected_feature_sets=selected_feature_sets,
        feature_sets=feature_sets,
        selected_models=selected_models,
        resolution=resolution,
        include_residual=include_residual,
        steps_per_day=int(args.steps_per_day),
    )
    if metrics_fold.empty:
        logger.error("No metrics rows produced.")
        return 1

    selection_scoreboard = build_selection_scoreboard(metrics_fold)
    coverage_audit = _build_coverage_audit(
        gold=gold,
        resolution=resolution,
        feature_sets=feature_sets,
        selected_feature_sets=selected_feature_sets,
        steps_per_day=int(args.steps_per_day),
    )
    hgb_coordinate_summary, hgb_coordinate_recommended = build_hgb_coordinate_summary(metrics_fold)
    guardrail_decisions: pd.DataFrame | None = None
    blend_candidate: dict[str, Any] | None = None
    blend_finalists = pd.DataFrame()
    if blend_config is not None:
        blend_metrics, guardrail_decisions, blend_candidate, selected_blend_config, blend_finalists = _evaluate_blend_candidate(
            gold=gold,
            folds=folds,
            selection_scoreboard=selection_scoreboard,
            feature_sets=feature_sets,
            model_catalog={spec.model_label: spec for spec in selected_models},
            resolution=resolution,
            base_blend_config=blend_config,
            steps_per_day=int(args.steps_per_day),
            preferred_candidate=hgb_coordinate_recommended,
        )
        if selected_blend_config is not None:
            blend_config = selected_blend_config
        if not blend_metrics.empty:
            metrics_fold = pd.concat([metrics_fold, blend_metrics], ignore_index=True)
            selection_scoreboard = build_selection_scoreboard(metrics_fold)
    residual_ablation = build_residual_ablation(selection_scoreboard)
    promoted_candidate = _select_promotion_candidate(selection_scoreboard)
    holdout_summary: pd.DataFrame | None = None
    holdout_segment_evaluation: pd.DataFrame | None = None
    holdout_operating_regime_evaluation: pd.DataFrame | None = None
    holdout_coverage_segments: pd.DataFrame | None = None
    holdout_coverage_summary: dict[str, Any] | None = None
    holdout_blend_decisions: pd.DataFrame | None = None
    deployment_recommendation: dict[str, Any] | None = None
    operating_policy: dict[str, Any] | None = None
    holdout_predictions: pd.DataFrame | None = None
    holdout_inference: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None
    feature_importance_summary: dict[str, Any] | None = None
    shap_importance: pd.DataFrame | None = None
    shap_importance_summary: dict[str, Any] | None = None
    supplemental_surface_summary: pd.DataFrame | None = None
    supplemental_surface_source_evaluation: pd.DataFrame | None = None
    supplemental_surface_segment_evaluation: pd.DataFrame | None = None
    supplemental_surface_operating_regime_evaluation: pd.DataFrame | None = None
    supplemental_surface_coverage_segments: pd.DataFrame | None = None
    supplemental_surface_coverage_summary: dict[str, Any] | None = None
    supplemental_surface_predictions: pd.DataFrame | None = None
    supplemental_surface_advisory: dict[str, Any] | None = None
    if promoted_candidate is not None:
        diagnostics = _evaluate_promoted_holdout_artifacts(
            gold=gold,
            folds=folds,
            feature_sets=feature_sets,
            model_catalog={spec.model_label: spec for spec in selected_models},
            resolution=resolution,
            promoted_candidate=promoted_candidate,
            steps_per_day=int(args.steps_per_day),
            blend_config=blend_config,
        )
        holdout_summary = diagnostics.holdout_summary
        holdout_blend_decisions = diagnostics.holdout_blend_decisions
        deployment_recommendation = diagnostics.deployment_recommendation
        holdout_segment_evaluation = diagnostics.holdout_segment_evaluation
        holdout_operating_regime_evaluation = diagnostics.holdout_operating_regime_evaluation
        operating_policy = diagnostics.operating_policy
        holdout_coverage_segments = diagnostics.holdout_coverage_segments
        holdout_coverage_summary = diagnostics.holdout_coverage_summary
        holdout_predictions = diagnostics.holdout_predictions
        holdout_inference = diagnostics.holdout_inference
        feature_importance = diagnostics.feature_importance
        feature_importance_summary = diagnostics.feature_importance_summary
        shap_importance = diagnostics.shap_importance
        shap_importance_summary = diagnostics.shap_importance_summary
        supplemental_surface_summary = diagnostics.supplemental_surface_summary
        supplemental_surface_source_evaluation = diagnostics.supplemental_surface_source_evaluation
        supplemental_surface_segment_evaluation = diagnostics.supplemental_surface_segment_evaluation
        supplemental_surface_operating_regime_evaluation = diagnostics.supplemental_surface_operating_regime_evaluation
        supplemental_surface_coverage_segments = diagnostics.supplemental_surface_coverage_segments
        supplemental_surface_coverage_summary = diagnostics.supplemental_surface_coverage_summary
        supplemental_surface_predictions = diagnostics.supplemental_surface_predictions
        supplemental_surface_advisory = diagnostics.supplemental_surface_advisory
    _write_outputs(
        output_dir=output_dir,
        output_root=output_root,
        run_mode="quick" if args.quick else "full",
        metrics_fold=metrics_fold,
        selection_scoreboard=selection_scoreboard,
        residual_ablation=residual_ablation,
        folds=folds,
        resolution=resolution,
        selected_feature_sets=selected_feature_sets,
        feature_sets=feature_sets,
        selected_models=selected_models,
        include_residual=include_residual,
        ramp_quantile=float(args.ramp_quantile) if RAMP_FEATURE_SET_NAME in selected_feature_sets else None,
        ramp_threshold=ramp_threshold,
        coverage_audit=coverage_audit,
        promoted_candidate=promoted_candidate,
        blend_finalists=blend_finalists,
        holdout_summary=holdout_summary,
        holdout_segment_evaluation=holdout_segment_evaluation,
        holdout_operating_regime_evaluation=holdout_operating_regime_evaluation,
        holdout_coverage_segments=holdout_coverage_segments,
        holdout_coverage_summary=holdout_coverage_summary,
        holdout_blend_decisions=holdout_blend_decisions,
        deployment_recommendation=deployment_recommendation,
        operating_policy=operating_policy,
        holdout_predictions=holdout_predictions,
        holdout_inference=holdout_inference,
        feature_importance=feature_importance,
        feature_importance_summary=feature_importance_summary,
        shap_importance=shap_importance,
        shap_importance_summary=shap_importance_summary,
        supplemental_surface_summary=supplemental_surface_summary,
        supplemental_surface_source_evaluation=supplemental_surface_source_evaluation,
        supplemental_surface_segment_evaluation=supplemental_surface_segment_evaluation,
        supplemental_surface_operating_regime_evaluation=supplemental_surface_operating_regime_evaluation,
        supplemental_surface_coverage_segments=supplemental_surface_coverage_segments,
        supplemental_surface_coverage_summary=supplemental_surface_coverage_summary,
        supplemental_surface_predictions=supplemental_surface_predictions,
        supplemental_surface_advisory=supplemental_surface_advisory,
        blend_config=blend_config,
        blend_candidate=blend_candidate,
        guardrail_decisions=guardrail_decisions,
        hgb_coordinate_summary=hgb_coordinate_summary,
        hgb_coordinate_recommended=hgb_coordinate_recommended,
        adaptive_hgb_screen=adaptive_hgb_screen,
        horizon_policy=horizon_policy,
        parallel_plan=parallel_plan,
    )
    logger.info("Model performance artifacts written: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
