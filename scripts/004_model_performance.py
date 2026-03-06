"""Model performance workflow: preflight + walk-forward + residual ablation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    FEATURE_SETS,
    PATHS,
    RESOLUTION_ALIASES,
    RESOLUTION_TO_SUFFIX,
    SPLIT_DAY_RANGES,
    SUPPORTED_RESOLUTIONS,
    TARGET_COLUMN,
    validate_config,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = SCRIPT_DIR.parent
STEP4_ARTIFACT_DIR = PATHS["outputs_modeling_dir"]
DEFAULT_OUTPUT_DIR = PATHS["outputs_performance_dir"]
OHE_COLUMNS = ("workday", "hour", "day_of_week", "season", "time_of_day")
RAMP_FEATURE_SET_NAME = "curated_ramp"
RAMP_ADDITIONAL_FEATURES = (
    "rolling_mean_3",
    "rolling_std_3",
    "ramp_flag",
    "hour_x_delta_5",
)


@dataclass(frozen=True)
class ModelSpec:
    model_label: str
    family: str
    params: dict[str, Any]
    factory: Callable[[], Any]


@dataclass(frozen=True)
class BlendConfig:
    window: int
    sharpness: float
    min_weight: float
    max_weight: float


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _canonical_resolution(resolution: str) -> str:
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    if canonical not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
            f"(aliases: {sorted(RESOLUTION_ALIASES)})"
        )
    return canonical


def _resolution_to_pandas_freq(resolution: str) -> str:
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
    suffix = RESOLUTION_TO_SUFFIX[resolution]
    return gold_dir / f"power_load_{suffix}_all_features.parquet"


def _model_split_path(resolution: str, feature_set: str, split: str, model_dir: Path) -> Path:
    suffix = RESOLUTION_TO_SUFFIX[resolution]
    return model_dir / f"{suffix}_{feature_set}_{split}.parquet"


def _git_commit() -> str:
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


def _hgb_coordinate_specs() -> dict[str, ModelSpec]:
    base = {
        "max_depth": 7,
        "max_iter": 300,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_leaf_nodes": None,
        "early_stopping": False,
        "random_state": 42,
    }
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

    specs: dict[str, ModelSpec] = {}
    for label, overrides in coordinate_variants.items():
        params = {**base, **overrides}
        specs[label] = ModelSpec(
            label,
            "hgb",
            params,
            lambda params=params: HistGradientBoostingRegressor(**params),
        )
    return specs


def _model_catalog(*, include_hgb_coordinate_search: bool) -> dict[str, ModelSpec]:
    catalog = {
        "ridge-light": ModelSpec("ridge-light", "ridge", {"alpha": 0.1}, lambda: Ridge(alpha=0.1)),
        "ridge-medium": ModelSpec("ridge-medium", "ridge", {"alpha": 1.0}, lambda: Ridge(alpha=1.0)),
        "ridge-strong": ModelSpec("ridge-strong", "ridge", {"alpha": 10.0}, lambda: Ridge(alpha=10.0)),
        "hgb-conservative": ModelSpec(
            "hgb-conservative",
            "hgb",
            {"max_depth": 3, "max_iter": 100, "learning_rate": 0.1, "early_stopping": False, "random_state": 42},
            lambda: HistGradientBoostingRegressor(
                max_depth=3, max_iter=100, learning_rate=0.1, early_stopping=False, random_state=42
            ),
        ),
        "hgb-balanced": ModelSpec(
            "hgb-balanced",
            "hgb",
            {"max_depth": 5, "max_iter": 200, "learning_rate": 0.1, "early_stopping": False, "random_state": 42},
            lambda: HistGradientBoostingRegressor(
                max_depth=5, max_iter=200, learning_rate=0.1, early_stopping=False, random_state=42
            ),
        ),
        "hgb-aggressive": ModelSpec(
            "hgb-aggressive",
            "hgb",
            {"max_depth": 7, "max_iter": 300, "learning_rate": 0.05, "early_stopping": False, "random_state": 42},
            lambda: HistGradientBoostingRegressor(
                max_depth=7, max_iter=300, learning_rate=0.05, early_stopping=False, random_state=42
            ),
        ),
    }
    if include_hgb_coordinate_search:
        catalog.update(_hgb_coordinate_specs())
    return catalog


def mae_ratio(model_mae: float, persistence_mae: float) -> float:
    if persistence_mae <= 0 or math.isnan(model_mae) or math.isnan(persistence_mae):
        return float("nan")
    return float(model_mae / persistence_mae)


def _load_gold_with_full_grid(resolution: str, gold_dir: Path) -> pd.DataFrame:
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
    return max(end_day - start_day + 1, 0) * steps_per_day


def _dedupe_feature_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _augment_with_curated_ramp_features(gold: pd.DataFrame, *, ramp_quantile: float) -> tuple[pd.DataFrame, float]:
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


def _build_feature_sets(*, include_curated_ramp: bool) -> dict[str, list[str]]:
    feature_sets = {name: list(columns) for name, columns in FEATURE_SETS.items()}
    if include_curated_ramp:
        curated = feature_sets.get("curated")
        if curated is None:
            raise ValueError("Feature set 'curated' not found in configuration.")
        feature_sets[RAMP_FEATURE_SET_NAME] = _dedupe_feature_columns(curated + list(RAMP_ADDITIONAL_FEATURES))
    return feature_sets


def _classify_feature_causality(feature: str) -> tuple[str, str]:
    if feature in {"rolling_mean_3", "rolling_std_3"}:
        return ("causal", "Derived from avg_load.shift(1) short-history windows.")
    if feature in {"ramp_flag", "hour_x_delta_5"}:
        return ("causal", "Derived from prior lag and calendar context.")
    if feature.startswith("rolling_"):
        return ("needs_review", "Rolling features need shifted-history verification.")
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
    mask = df[[TARGET_COLUMN, "lag_1"]].notna().all(axis=1)
    if int(mask.sum()) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_eval": 0}
    yt = df.loc[mask, TARGET_COLUMN].to_numpy(dtype=float)
    yp = df.loc[mask, "lag_1"].to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean(np.square(yt - yp)))),
        "n_eval": int(mask.sum()),
    }


def _reproduce_baseline(*, model_dir: Path, resolution: str, tolerance_mae: float) -> dict[str, Any]:
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
    error = aligned["y_true"] - aligned["y_pred"]
    persist_error = aligned["y_true"] - aligned["y_persist"]
    model_mae = float(np.mean(np.abs(error)))
    persistence_mae = float(np.mean(np.abs(persist_error)))
    n_eval = int(len(aligned))
    return {
        "mae": model_mae,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae_ratio": mae_ratio(model_mae, persistence_mae),
        "persistence_mae": persistence_mae,
        "train_mae": train_mae,
        "train_val_mae_ratio": float(train_mae / model_mae) if model_mae > 0 else float("nan"),
        "n_eval": n_eval,
        "n_eval_total": int(n_eval_total),
        "coverage": float(n_eval / n_eval_total) if n_eval_total > 0 else float("nan"),
    }


def _fit_and_align(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    model_spec: ModelSpec,
    target_mode: str,
) -> tuple[pd.DataFrame, float] | None:
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
    return aligned.sort_index(), train_mae


def _sigmoid(x: float) -> float:
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
    blend_mae = float(decision_df["blend_abs_error"].mean())
    blend_rmse = float(np.sqrt(np.mean(np.square(decision_df["y_true"] - decision_df["blend_pred"]))))
    persist_mae = float(decision_df["persistence_abs_error"].mean())
    metrics = {
        "mae": blend_mae,
        "rmse": blend_rmse,
        "mae_ratio": mae_ratio(blend_mae, persist_mae),
        "persistence_mae": persist_mae,
        "train_mae": float("nan"),
        "train_val_mae_ratio": float("nan"),
        "n_eval": int(len(decision_df)),
        "n_eval_total": int(n_eval_total),
        "coverage": float(len(decision_df) / n_eval_total) if n_eval_total > 0 else float("nan"),
        "mean_blend_weight": float(decision_df["blend_weight"].mean()),
        "model_dominated_frac": float((decision_df["blend_weight"] >= 0.5).mean()),
    }
    return metrics, decision_df


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
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_modes = ["raw", "residual"] if include_residual else ["raw"]
    for fold in folds:
        train_df = gold.loc[
            gold["day_idx"].between(fold["train_start_day"], fold["train_end_day"])
        ].copy()
        val_df = gold.loc[gold["day_idx"].between(fold["val_start_day"], fold["val_end_day"])].copy()
        n_eval_total = _expected_steps_for_day_range(fold["val_start_day"], fold["val_end_day"], steps_per_day)
        for feature_set in selected_feature_sets:
            feature_cols = feature_sets[feature_set]
            for model_spec in selected_models:
                for target_mode in target_modes:
                    metrics = _fit_and_evaluate(
                        train_df=train_df,
                        eval_df=val_df,
                        feature_cols=feature_cols,
                        model_spec=model_spec,
                        target_mode=target_mode,
                        n_eval_total=n_eval_total,
                    )
                    if metrics is None:
                        continue
                    rows.append(
                        {
                            "fold": int(fold["fold"]),
                            "resolution": resolution,
                            "feature_set": feature_set,
                            "model": model_spec.family,
                            "model_label": model_spec.model_label,
                            "params": json.dumps(model_spec.params, sort_keys=True),
                            "target_mode": target_mode,
                            **metrics,
                            "train_start_day": fold["train_start_day"],
                            "train_end_day": fold["train_end_day"],
                            "val_start_day": fold["val_start_day"],
                            "val_end_day": fold["val_end_day"],
                        }
                    )
    return pd.DataFrame(rows)


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None, BlendConfig | None]:
    hgb_candidates = selection_scoreboard[
        selection_scoreboard["model_label"].astype(str).str.startswith("hgb")
    ].copy()
    if preferred_candidate is not None:
        preferred_model = str(preferred_candidate.get("model_label", ""))
        preferred_feature_set = str(preferred_candidate.get("feature_set", ""))
        preferred_target_mode = str(preferred_candidate.get("target_mode", ""))
        preferred_rows = hgb_candidates[
            (hgb_candidates["model_label"] == preferred_model)
            & (hgb_candidates["feature_set"] == preferred_feature_set)
            & (hgb_candidates["target_mode"] == preferred_target_mode)
        ]
        if not preferred_rows.empty:
            hgb_candidates = preferred_rows
    hgb_candidates = hgb_candidates.sort_values(
        ["fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae"],
        ascending=[True, True, True],
    )
    if hgb_candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    candidate = hgb_candidates.iloc[0]
    model_label = str(candidate["model_label"])
    feature_set = str(candidate["feature_set"])
    target_mode = str(candidate["target_mode"])
    model_spec = model_catalog.get(model_label)
    if model_spec is None or feature_set not in feature_sets:
        return pd.DataFrame(), pd.DataFrame(), None, None

    fold_payload: list[dict[str, Any]] = []
    for fold in folds:
        train_df = gold.loc[
            gold["day_idx"].between(fold["train_start_day"], fold["train_end_day"])
        ].copy()
        val_df = gold.loc[gold["day_idx"].between(fold["val_start_day"], fold["val_end_day"])].copy()
        n_eval_total = _expected_steps_for_day_range(fold["val_start_day"], fold["val_end_day"], steps_per_day)
        aligned_result = _fit_and_align(
            train_df=train_df,
            eval_df=val_df,
            feature_cols=feature_sets[feature_set],
            model_spec=model_spec,
            target_mode=target_mode,
        )
        if aligned_result is None:
            continue
        aligned, _ = aligned_result
        raw_error = aligned["y_true"] - aligned["y_pred"]
        persist_error = aligned["y_true"] - aligned["y_persist"]
        raw_ratio = mae_ratio(float(np.mean(np.abs(raw_error))), float(np.mean(np.abs(persist_error))))
        fold_payload.append(
            {
                "fold_meta": fold,
                "aligned": aligned,
                "n_eval_total": n_eval_total,
                "raw_ratio": raw_ratio,
            }
        )

    if not fold_payload:
        return pd.DataFrame(), pd.DataFrame(), None, None

    window_values = sorted({max(10, base_blend_config.window // 2), base_blend_config.window, base_blend_config.window * 2})
    sharpness_values = sorted({max(1.0, base_blend_config.sharpness * 0.5), base_blend_config.sharpness, base_blend_config.sharpness * 1.5})
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
        config_results.append(
            {
                "config": cfg,
                "mean_ratio": mean_ratio,
                "std_ratio": std_ratio,
                "max_degrade_pct": max_degrade,
                "fold_metrics": fold_metrics,
                "decisions": fold_decisions,
            }
        )

    if not config_results:
        return pd.DataFrame(), pd.DataFrame(), None, None

    accepted = [r for r in config_results if r["max_degrade_pct"] <= 2.0]
    ranking_pool = accepted if accepted else config_results
    ranking_pool.sort(key=lambda r: (r["mean_ratio"], r["std_ratio"], r["max_degrade_pct"]))
    selected = ranking_pool[0]
    selected_config = selected["config"]

    blend_rows: list[dict[str, Any]] = []
    for fold_item in selected["fold_metrics"]:
        fold = fold_item["fold_meta"]
        blend_rows.append(
            {
                "fold": int(fold["fold"]),
                "resolution": resolution,
                "feature_set": feature_set,
                "model": model_spec.family,
                "model_label": model_spec.model_label,
                "params": json.dumps(model_spec.params, sort_keys=True),
                "target_mode": f"{target_mode}+blend",
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
        "model_label": model_label,
        "feature_set": feature_set,
        "target_mode": target_mode,
        "selected_blend_window": selected_config.window,
        "selected_blend_sharpness": selected_config.sharpness,
        "selected_blend_min_weight": selected_config.min_weight,
        "selected_blend_max_weight": selected_config.max_weight,
        "selected_blend_mean_ratio": selected["mean_ratio"],
        "selected_blend_std_ratio": selected["std_ratio"],
        "selected_blend_max_degrade_pct": selected["max_degrade_pct"],
        "meets_p2_fold_degrade_cap": bool(selected["max_degrade_pct"] <= 2.0),
    }
    return blend_df, decisions_df, candidate_meta, selected_config


def build_selection_scoreboard(metrics_fold: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["resolution", "feature_set", "model_label", "target_mode"]
    grouped = (
        metrics_fold.groupby(group_cols, dropna=False)
        .agg(
            fold_mean_mae_ratio=("mae_ratio", "mean"),
            fold_std_mae_ratio=("mae_ratio", "std"),
            fold_n=("fold", "nunique"),
            raw_validate_mae=("mae", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .reset_index()
    )
    grouped["fold_std_mae_ratio"] = grouped["fold_std_mae_ratio"].fillna(0.0)
    return grouped.sort_values(
        ["fold_mean_mae_ratio", "fold_std_mae_ratio", "raw_validate_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_hgb_coordinate_summary(metrics_fold: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
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


def _write_outputs(
    *,
    output_dir: Path,
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
    blend_config: BlendConfig | None,
    blend_candidate: dict[str, Any] | None,
    guardrail_decisions: pd.DataFrame | None,
    hgb_coordinate_summary: pd.DataFrame | None,
    hgb_coordinate_recommended: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_fold.csv"
    metrics_fold.to_csv(metrics_path, index=False, float_format="%.6f")
    summary_records = selection_scoreboard.to_dict(orient="records")
    (output_dir / "metrics_fold_summary.json").write_text(
        json.dumps(summary_records, indent=2), encoding="utf-8"
    )
    selection_scoreboard.to_csv(output_dir / "selection_scoreboard.csv", index=False, float_format="%.6f")
    residual_ablation.to_csv(output_dir / "residual_ablation.csv", index=False, float_format="%.6f")
    if hgb_coordinate_summary is not None and not hgb_coordinate_summary.empty:
        hgb_coordinate_summary.to_csv(
            output_dir / "hgb_coordinate_summary.csv",
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
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "resolution": resolution,
        "feature_sets": selected_feature_sets,
        "feature_set_columns": {name: feature_sets[name] for name in selected_feature_sets},
        "models": [
            {"model_label": model.model_label, "family": model.family, "params": model.params}
            for model in selected_models
        ],
        "include_residual": include_residual,
        "include_curated_ramp": RAMP_FEATURE_SET_NAME in selected_feature_sets,
        "ramp_quantile": ramp_quantile,
        "ramp_threshold": ramp_threshold,
        "blend_policy": (
            {
                "enabled": True,
                "window": blend_config.window,
                "sharpness": blend_config.sharpness,
                "min_weight": blend_config.min_weight,
                "max_weight": blend_config.max_weight,
                "candidate": blend_candidate,
            }
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
        "folds": folds,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--holdout-start-day", type=int, default=SPLIT_DAY_RANGES["test"][0])
    parser.add_argument("--tolerance-mae", type=float, default=0.1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--disable-curated-ramp", action="store_true")
    parser.add_argument("--disable-hgb-coordinate-search", action="store_true")
    parser.add_argument("--disable-blend-policy", action="store_true")
    parser.add_argument("--blend-window", type=int, default=120)
    parser.add_argument("--blend-sharpness", type=float, default=6.0)
    parser.add_argument("--blend-min-weight", type=float, default=0.10)
    parser.add_argument("--blend-max-weight", type=float, default=0.90)
    parser.add_argument("--ramp-quantile", type=float, default=0.85)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    validate_config()
    args = parse_args()
    resolution = _canonical_resolution(args.resolution)
    output_dir = Path(args.output_dir).resolve()
    feature_sets = _build_feature_sets(include_curated_ramp=not bool(args.disable_curated_ramp))
    if args.blend_window < 1:
        raise ValueError("--blend-window must be >= 1.")
    if not 0.0 <= float(args.blend_min_weight) <= 1.0:
        raise ValueError("--blend-min-weight must be within [0,1].")
    if not 0.0 <= float(args.blend_max_weight) <= 1.0:
        raise ValueError("--blend-max-weight must be within [0,1].")
    if float(args.blend_min_weight) > float(args.blend_max_weight):
        raise ValueError("--blend-min-weight must be <= --blend-max-weight.")

    selected_feature_sets = list(args.feature_sets) if args.feature_sets else list(feature_sets.keys())
    for feature_set in selected_feature_sets:
        if feature_set not in feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}. Available: {sorted(feature_sets)}")

    catalog = _model_catalog(include_hgb_coordinate_search=not bool(args.disable_hgb_coordinate_search))
    model_labels = list(args.model_labels) if args.model_labels else list(catalog.keys())
    for label in model_labels:
        if label not in catalog:
            raise ValueError(f"Unknown model label: {label}. Available: {sorted(catalog)}")
    selected_models = [catalog[label] for label in model_labels]
    n_folds = int(args.n_folds)
    val_window_days = int(args.val_window_days)
    include_residual = not bool(args.disable_residual)
    blend_config = (
        BlendConfig(
            window=int(args.blend_window),
            sharpness=float(args.blend_sharpness),
            min_weight=float(args.blend_min_weight),
            max_weight=float(args.blend_max_weight),
        )
        if not bool(args.disable_blend_policy)
        else None
    )

    if args.quick:
        selected_feature_sets = ["curated"]
        if RAMP_FEATURE_SET_NAME in feature_sets:
            selected_feature_sets.append(RAMP_FEATURE_SET_NAME)
        selected_models = [catalog["ridge-medium"], catalog["hgb-balanced"]]
        n_folds = 2
        val_window_days = 2
        include_residual = True
        blend_config = BlendConfig(window=60, sharpness=6.0, min_weight=0.10, max_weight=0.90)

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
            return 1

    if args.preflight_only:
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
    metrics_fold = _run_fold_metrics(
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
    hgb_coordinate_summary, hgb_coordinate_recommended = build_hgb_coordinate_summary(metrics_fold)
    guardrail_decisions: pd.DataFrame | None = None
    blend_candidate: dict[str, Any] | None = None
    if blend_config is not None:
        blend_metrics, guardrail_decisions, blend_candidate, selected_blend_config = _evaluate_blend_candidate(
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
    _write_outputs(
        output_dir=output_dir,
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
        blend_config=blend_config,
        blend_candidate=blend_candidate,
        guardrail_decisions=guardrail_decisions,
        hgb_coordinate_summary=hgb_coordinate_summary,
        hgb_coordinate_recommended=hgb_coordinate_recommended,
    )
    logger.info("Model performance artifacts written: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
