"""Shared modeling helpers used across Stage-5 through Stage-10 runners.

This module centralizes the small pieces of infrastructure that multiple modeling
stages need to agree on:

- canonical model-catalog construction
- resolution and horizon conversions
- deterministic config hashing and latest-alias handling
- artifact validation helpers
- human-readable figure guides written next to persisted PNG outputs

The goal is to keep the stage-specific modules focused on modeling logic while
preserving one consistent contract for common outputs and runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from config import RESOLUTION_ALIASES, SUPPORTED_RESOLUTIONS
from modeling.runtime import XGBRegressor, resolve_xgboost_runtime

OHE_COLUMNS = ("workday", "hour", "day_of_week", "season", "time_of_day")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Factory metadata for a learned model candidate."""

    model_label: str
    family: str
    params: dict[str, Any]
    factory: Callable[[], Any]


@dataclass
class TrainedModel:
    """Fitted model plus encoding metadata."""

    spec: ModelSpec
    feature_columns: tuple[str, ...]
    model: Any
    encoded_columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FigureGuideEntry:
    """Human-readable explanation for one persisted figure artifact.

    Stage runners use these entries to emit a markdown sidecar next to generated
    figures so reviewers can understand:

    - why the figure exists
    - how to read it
    - what signals should drive interpretation
    """

    filename: str
    title: str
    intent: str
    how_to_read: str
    look_for: str


def _build_hgb_spec(model_label: str, params: dict[str, Any]) -> ModelSpec:
    """Create a HistGradientBoosting model spec with stable params."""
    frozen_params = dict(params)
    return ModelSpec(
        model_label=model_label,
        family="hgb",
        params=frozen_params,
        factory=lambda params=frozen_params: HistGradientBoostingRegressor(**params),
    )


def _build_xgb_spec(model_label: str, params: dict[str, Any]) -> ModelSpec:
    """Create an optional XGBoost model spec with stable params."""
    if XGBRegressor is None:
        raise RuntimeError("XGBoost is unavailable in this environment.")
    frozen_params = dict(params)
    return ModelSpec(
        model_label=model_label,
        family="xgb",
        params=frozen_params,
        factory=lambda params=frozen_params: XGBRegressor(**params),
    )


def _hgb_base_params() -> dict[str, Any]:
    """Return the default HGB parameter anchor used across search variants."""
    return {
        "max_depth": 7,
        "max_iter": 300,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_leaf_nodes": None,
        "early_stopping": False,
        "random_state": 42,
    }


def _xgb_base_params(device: str) -> dict[str, Any]:
    """Return the default XGBoost parameter anchor used across search variants."""
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "n_jobs": 1,
        "random_state": 42,
        "verbosity": 0,
    }


def _hgb_coordinate_specs() -> dict[str, ModelSpec]:
    """Return one-factor HGB coordinate-search variants."""
    base = _hgb_base_params()
    coordinate_variants = {
        "hgb-coordinate-depth3": {"max_depth": 3},
        "hgb-coordinate-depth5": {"max_depth": 5},
        "hgb-coordinate-iter100": {"max_iter": 100},
        "hgb-coordinate-iter200": {"max_iter": 200},
        "hgb-coordinate-lr010": {"learning_rate": 0.1},
        "hgb-coordinate-leaf50": {"min_samples_leaf": 50},
        "hgb-coordinate-leaf100": {"min_samples_leaf": 100},
        "hgb-coordinate-l2001": {"l2_regularization": 0.1},
        "hgb-coordinate-l2100": {"l2_regularization": 1.0},
        "hgb-coordinate-leafnodes31": {"max_leaf_nodes": 31},
        "hgb-coordinate-leafnodes63": {"max_leaf_nodes": 63},
        "hgb-coordinate-mixed-reg": {"min_samples_leaf": 50, "l2_regularization": 0.1},
    }
    return {
        model_label: _build_hgb_spec(model_label, {**base, **overrides})
        for model_label, overrides in coordinate_variants.items()
    }


def _hgb_frontier_specs() -> dict[str, ModelSpec]:
    """Return targeted second-order HGB variants built from the Stage-5 frontier."""
    base = _hgb_base_params()
    frontier_variants = {
        "hgb-frontier-lr010-l2001": {
            "learning_rate": 0.1,
            "l2_regularization": 0.1,
        },
        "hgb-frontier-lr010-leaf100": {
            "learning_rate": 0.1,
            "min_samples_leaf": 100,
        },
        "hgb-frontier-lr010-depth5": {
            "learning_rate": 0.1,
            "max_depth": 5,
        },
        "hgb-frontier-lr010-depth5-leaf100-l2001": {
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_samples_leaf": 100,
            "l2_regularization": 0.1,
        },
    }
    return {
        model_label: _build_hgb_spec(model_label, {**base, **overrides})
        for model_label, overrides in frontier_variants.items()
    }


def _xgb_specs(device: str) -> dict[str, ModelSpec]:
    """Return optional XGBoost variants when the dependency is available."""
    base = _xgb_base_params(device)
    variants = {
        "xgb-balanced": {},
        "xgb-frontier-lr010": {
            "learning_rate": 0.1,
            "n_estimators": 200,
        },
        "xgb-frontier-depth8": {
            "max_depth": 8,
            "n_estimators": 400,
        },
    }
    return {
        model_label: _build_xgb_spec(model_label, {**base, **overrides})
        for model_label, overrides in variants.items()
    }


def canonical_resolution(resolution: str) -> str:
    """Resolve aliases and validate a configured resolution."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    if canonical not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
            f"(aliases: {sorted(RESOLUTION_ALIASES)})"
        )
    return canonical


def resolution_timedelta(resolution: str) -> pd.Timedelta:
    """Return a pandas timedelta for a supported resolution."""
    return pd.to_timedelta(canonical_resolution(resolution))


def resolution_seconds(resolution: str) -> int:
    """Return integer seconds for a supported resolution."""
    return int(resolution_timedelta(resolution).total_seconds())


def resolution_total_minutes(resolution: str) -> float:
    """Return exact minutes for a supported resolution, including sub-minute cadences."""
    return float(resolution_timedelta(resolution).total_seconds() / 60.0)


def resolution_minutes(resolution: str) -> int:
    """Return integer minutes for a supported resolution."""
    seconds = resolution_seconds(resolution)
    if seconds % 60 != 0:
        raise ValueError(
            f"Resolution '{resolution}' is not minute-aligned. Got {seconds} seconds."
        )
    return seconds // 60


def steps_per_day(resolution: str) -> int:
    """Return the number of steps per day at a supported resolution."""
    seconds = resolution_seconds(resolution)
    if 86400 % seconds != 0:
        raise ValueError(
            f"Resolution '{resolution}' does not divide evenly into one day."
        )
    return 86400 // seconds


def lead_steps_for_horizon(resolution: str, horizon_minutes: int) -> int:
    """Convert a matched real-time horizon into native steps."""
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive. Got: {horizon_minutes}")
    horizon_seconds = int(horizon_minutes) * 60
    step_seconds = resolution_seconds(resolution)
    if horizon_seconds % step_seconds != 0:
        raise ValueError(
            f"Horizon {horizon_minutes}m is not representable at resolution {resolution}."
        )
    return horizon_seconds // step_seconds


def mode_origins_per_fold(mode: str, runtime_config: dict[str, int | bool | str]) -> int:
    """Resolve per-mode recursive origin count from runtime config."""
    mode_key = mode.lower().strip()
    if mode_key == "smoke":
        return int(runtime_config["smoke_origins_per_fold"])
    if mode_key == "candidate":
        return int(runtime_config["candidate_origins_per_fold"])
    if mode_key == "full":
        return int(runtime_config["full_origins_per_fold"])
    if mode_key.startswith("focus_") or mode_key.startswith("tune_"):
        return int(runtime_config["candidate_origins_per_fold"])
    raise ValueError(f"Unsupported mode: {mode}")


def stable_config_hash(payload: dict[str, Any]) -> str:
    """Return a deterministic hash for an effective config payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def update_latest_alias(run_dir: Path, latest_dir: Path, *, enabled: bool) -> None:
    """Refresh a directory copy used as a Windows-safe latest alias."""
    if not enabled:
        return
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def validate_png_artifact(path: Path, *, min_size_bytes: int = 1000) -> tuple[int, int]:
    """Validate a written PNG artifact and return its dimensions."""
    if not path.exists():
        raise FileNotFoundError(f"Missing PNG artifact: {path}")
    size_bytes = int(path.stat().st_size)
    if size_bytes < int(min_size_bytes):
        raise ValueError(f"PNG artifact is unexpectedly small ({size_bytes} bytes): {path}")
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG signature: {path}")
        _ = struct.unpack(">I", handle.read(4))[0]
        if handle.read(4) != b"IHDR":
            raise ValueError(f"Missing PNG IHDR chunk: {path}")
        width, height = struct.unpack(">II", handle.read(8))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions ({width}x{height}): {path}")
    image = mpimg.imread(path)
    if image.size == 0:
        raise ValueError(f"Decoded PNG artifact is empty: {path}")
    return (int(width), int(height))


def write_figure_guide(
    *,
    output_path: Path,
    stage_title: str,
    stage_purpose: str,
    figures: list[FigureGuideEntry],
) -> None:
    """Write a markdown guide that explains persisted visualization artifacts.

    The guide is intentionally lightweight and artifact-local. It gives readers a
    stable explanation surface even when they open a timestamped output directory
    without the notebook or stage source code beside it.
    """

    lines = [
        "# Visualization Guide",
        "",
        f"## {stage_title}",
        "",
        stage_purpose.strip(),
        "",
    ]
    for entry in figures:
        lines.extend(
            [
                f"### `{entry.filename}`: {entry.title}",
                "",
                f"- Intent: {entry.intent}",
                f"- How to read it: {entry.how_to_read}",
                f"- What to look for: {entry.look_for}",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_model_catalog(
    *,
    include_hgb_coordinate_search: bool = False,
    include_hgb_frontier: bool = True,
    include_optional_xgb: bool = True,
) -> dict[str, ModelSpec]:
    """Return the supported learned model catalog for modeling stages."""
    catalog = {
        "ridge-light": ModelSpec("ridge-light", "ridge", {"alpha": 0.1}, lambda: Ridge(alpha=0.1)),
        "ridge-medium": ModelSpec("ridge-medium", "ridge", {"alpha": 1.0}, lambda: Ridge(alpha=1.0)),
        "ridge-strong": ModelSpec("ridge-strong", "ridge", {"alpha": 10.0}, lambda: Ridge(alpha=10.0)),
    }
    catalog.update(
        {
            "hgb-conservative": _build_hgb_spec(
                "hgb-conservative",
                {"max_depth": 3, "max_iter": 100, "learning_rate": 0.1, "early_stopping": False, "random_state": 42},
            ),
            "hgb-balanced": _build_hgb_spec(
                "hgb-balanced",
                {"max_depth": 5, "max_iter": 200, "learning_rate": 0.1, "early_stopping": False, "random_state": 42},
            ),
            "hgb-aggressive": _build_hgb_spec(
                "hgb-aggressive",
                {"max_depth": 7, "max_iter": 300, "learning_rate": 0.05, "early_stopping": False, "random_state": 42},
            ),
        }
    )
    if include_hgb_frontier:
        catalog.update(_hgb_frontier_specs())
    if include_hgb_coordinate_search:
        catalog.update(_hgb_coordinate_specs())
    if include_optional_xgb:
        xgb_runtime = resolve_xgboost_runtime()
        if xgb_runtime.available and xgb_runtime.device is not None:
            catalog.update(_xgb_specs(xgb_runtime.device))
        else:
            logger.debug("Skipping optional XGBoost catalog: %s", xgb_runtime.reason)
    return catalog


def _encode_for_ridge(train_x: pd.DataFrame, eval_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical columns consistently for ridge models."""
    ohe_cols = [col for col in OHE_COLUMNS if col in train_x.columns]
    train_encoded = train_x.copy()
    eval_encoded = eval_x.copy()
    if ohe_cols:
        train_encoded[ohe_cols] = train_encoded[ohe_cols].astype("category")
        eval_encoded[ohe_cols] = eval_encoded[ohe_cols].astype("category")
        train_encoded = pd.get_dummies(train_encoded, columns=ohe_cols, drop_first=True)
        eval_encoded = pd.get_dummies(eval_encoded, columns=ohe_cols, drop_first=True)
        eval_encoded = eval_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
    return train_encoded.astype(float), eval_encoded.astype(float)


def train_model(train_df: pd.DataFrame, feature_columns: list[str], model_spec: ModelSpec) -> TrainedModel:
    """Fit a learned model candidate on a feature frame."""
    train_rows = train_df.dropna(subset=[*feature_columns, "avg_load"]).copy()
    if train_rows.empty:
        raise ValueError("No training rows available after dropping NaNs.")
    train_x = train_rows[feature_columns].copy()
    train_y = train_rows["avg_load"].to_numpy(dtype=float)
    model = model_spec.factory()
    if model_spec.family == "ridge":
        train_x_encoded, _ = _encode_for_ridge(train_x, train_x)
        model.fit(train_x_encoded, train_y)
        return TrainedModel(
            spec=model_spec,
            feature_columns=tuple(feature_columns),
            model=model,
            encoded_columns=tuple(train_x_encoded.columns),
        )
    model.fit(train_x.astype(float), train_y)
    return TrainedModel(
        spec=model_spec,
        feature_columns=tuple(feature_columns),
        model=model,
        encoded_columns=None,
    )


def predict_model(trained: TrainedModel, eval_df: pd.DataFrame) -> pd.Series:
    """Generate predictions for a fitted learned model candidate."""
    feature_columns = list(trained.feature_columns)
    eval_x = eval_df[feature_columns].copy()
    if trained.spec.family == "ridge":
        preds = pd.Series(np.nan, index=eval_df.index, dtype=float)
        valid_mask = eval_x.notna().all(axis=1)
        if not bool(valid_mask.any()):
            return preds
        eval_valid = eval_x.loc[valid_mask].copy()
        _, eval_encoded = _encode_for_ridge(eval_valid, eval_valid)
        if trained.encoded_columns is not None:
            eval_encoded = eval_encoded.reindex(columns=list(trained.encoded_columns), fill_value=0.0)
        preds.loc[valid_mask] = trained.model.predict(eval_encoded)
        return preds
    preds = trained.model.predict(eval_x.astype(float))
    return pd.Series(preds, index=eval_df.index, dtype=float)
