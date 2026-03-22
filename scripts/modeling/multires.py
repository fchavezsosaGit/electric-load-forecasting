"""Leakage-safe multiresolution dataset, selection, and rollout helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DAY_CLASS_MAP,
    FEATURE_CONFIG,
    MULTIRES_HYBRID,
    PATHS,
    RESOLUTION_TO_SUFFIX,
    SPLIT_DAY_RANGES,
)
from modeling.feature_engineering import (
    add_calendar_context,
    add_phase_context,
    add_period_history_features,
    add_profile_regime_features,
    add_time_normalized_features,
    normalized_window_steps,
)
from utils import (
    build_fourier_feature_frame,
    hour_to_time_of_day,
    rolling_slope,
)

from modeling.common import (
    TrainedModel,
    canonical_resolution,
    lead_steps_for_horizon,
    predict_model,
    resolution_seconds,
    resolution_timedelta,
    steps_per_day,
)
from modeling.metrics import compute_regression_metrics, safe_percent


@dataclass(frozen=True)
class FoldSpec:
    """Chronological walk-forward fold definition."""

    fold: int
    train_start_day: int
    train_end_day: int
    val_start_day: int
    val_end_day: int


@dataclass(frozen=True)
class WorkdayProfile:
    """Time-of-day profile baseline built from historical data only."""

    by_workday_slot: dict[tuple[int, str], float]
    by_slot: dict[str, float]
    peak_by_workday: dict[int, float]
    global_mean: float


def build_walkforward_folds(
    *, holdout_start_day: int, n_folds: int, val_window_days: int, train_start_day: int = 1
) -> list[FoldSpec]:
    """Build expanding train / rolling validation folds before the holdout window."""
    if n_folds < 1 or val_window_days < 1:
        raise ValueError("n_folds and val_window_days must be >= 1")
    val_end_max = holdout_start_day - 1
    first_val_start = val_end_max - (n_folds * val_window_days) + 1
    if first_val_start <= train_start_day:
        raise ValueError("Not enough history for requested fold layout.")
    folds: list[FoldSpec] = []
    for fold_idx in range(n_folds):
        val_start_day = first_val_start + fold_idx * val_window_days
        val_end_day = val_start_day + val_window_days - 1
        folds.append(
            FoldSpec(
                fold=fold_idx + 1,
                train_start_day=train_start_day,
                train_end_day=val_start_day - 1,
                val_start_day=val_start_day,
                val_end_day=val_end_day,
            )
        )
    return folds


def build_day_index(timestamps: pd.Series) -> pd.Series:
    """Map normalized dates to contiguous 1-based day indices."""
    unique_dates = sorted(pd.to_datetime(timestamps, errors="raise").dt.normalize().unique())
    day_map = {date: idx + 1 for idx, date in enumerate(unique_dates)}
    return pd.to_datetime(timestamps, errors="raise").dt.normalize().map(day_map).astype(int)


def load_base_gold(resolution: str, gold_dir: Path | None = None) -> pd.DataFrame:
    """Load gold data for a resolution and keep only base columns needed for causal rebuild."""
    canonical = canonical_resolution(resolution)
    suffix = RESOLUTION_TO_SUFFIX[canonical]
    root = gold_dir or PATHS["gold_dir"]
    path = root / f"power_load_{suffix}_all_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing gold parquet for resolution {canonical}: {path}")
    gold = pd.read_parquet(path)
    required_columns = {"timestamp", "avg_load", "day_class"}
    missing = sorted(required_columns - set(gold.columns))
    if missing:
        raise ValueError(f"Gold parquet missing required columns {missing}: {path}")
    base = gold.loc[:, ["timestamp", "avg_load", "day_class"]].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="raise")
    base = base.sort_values("timestamp").reset_index(drop=True)
    base["day_idx"] = build_day_index(base["timestamp"])
    return base


def _slot_label(timestamps: pd.Series) -> pd.Series:
    """Return a stable time-of-day slot label for profile baselines."""
    return pd.to_datetime(timestamps, errors="raise").dt.strftime("%H:%M:%S")


def build_causal_feature_frame(base: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Rebuild leakage-safe one-step features from gold targets only."""
    frame = base.copy()
    frame = frame.set_index("timestamp", drop=False)
    frame = add_calendar_context(frame)
    frame = add_phase_context(frame, resolution=resolution)
    frame = add_period_history_features(frame, resolution=resolution)
    frame = add_time_normalized_features(frame, resolution=resolution)
    frame = add_profile_regime_features(frame, resolution=resolution)
    frame = frame.reset_index(drop=True)
    return frame


def filter_day_range(frame: pd.DataFrame, start_day: int, end_day: int) -> pd.DataFrame:
    """Filter a feature frame to an inclusive day range."""
    return frame.loc[frame["day_idx"].between(start_day, end_day)].copy()


def evaluate_predictions(
    y_true: pd.Series, y_pred: pd.Series, *, n_total: int
) -> dict[str, float | int]:
    """Compute MAE/RMSE plus percentage-normalized metrics on aligned predictions."""
    metrics = compute_regression_metrics(y_true, y_pred, n_total=n_total)
    if math.isnan(float(metrics["coverage"])):
        metrics["coverage"] = 0.0
    return metrics


def mae_ratio(model_mae: float, persistence_mae: float) -> float:
    """Return model/baseline MAE ratio."""
    if persistence_mae <= 0 or math.isnan(model_mae) or math.isnan(persistence_mae):
        return float("nan")
    return float(model_mae / persistence_mae)


def rmse_ratio(model_rmse: float, persistence_rmse: float) -> float:
    """Return model/baseline RMSE ratio."""
    if persistence_rmse <= 0 or math.isnan(model_rmse) or math.isnan(persistence_rmse):
        return float("nan")
    return float(model_rmse / persistence_rmse)


def build_workday_profile(train_df: pd.DataFrame) -> WorkdayProfile:
    """Build an average workday baseline profile from historical data only."""
    if train_df.empty:
        return WorkdayProfile(
            by_workday_slot={},
            by_slot={},
            peak_by_workday={},
            global_mean=float("nan"),
        )
    grouped = (
        train_df.groupby(["workday", "slot_label"], dropna=False)["avg_load"].mean().reset_index()
    )
    by_workday_slot = {
        (int(row["workday"]), str(row["slot_label"])): float(row["avg_load"])
        for _, row in grouped.iterrows()
    }
    by_slot = {
        str(slot): float(value)
        for slot, value in train_df.groupby("slot_label", dropna=False)["avg_load"].mean().items()
    }
    peak_by_workday = {
        int(workday): float(values.max())
        for workday, values in grouped.groupby("workday", dropna=False)["avg_load"]
    }
    return WorkdayProfile(
        by_workday_slot=by_workday_slot,
        by_slot=by_slot,
        peak_by_workday=peak_by_workday,
        global_mean=float(train_df["avg_load"].mean()),
    )


def profile_predict(
    timestamps: pd.Series, workday_values: pd.Series, profile: WorkdayProfile
) -> pd.Series:
    """Predict from a workday/time-of-day profile with graceful fallbacks."""
    slots = _slot_label(timestamps)
    values: list[float] = []
    for slot, workday in zip(slots, workday_values, strict=True):
        key = (int(workday), str(slot))
        if key in profile.by_workday_slot:
            values.append(profile.by_workday_slot[key])
        elif str(slot) in profile.by_slot:
            values.append(profile.by_slot[str(slot)])
        else:
            values.append(profile.global_mean)
    return pd.Series(values, index=timestamps.index, dtype=float)


def native_step_baselines(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build one-step baseline predictions for a validation slice."""
    profile = build_workday_profile(train_df)
    return {
        "persistence": eval_df["lag_1"].astype(float),
        "previous_day": eval_df["previous_day_load"].astype(float),
        "avg_workday": profile_predict(eval_df["timestamp"], eval_df["workday"], profile),
    }


def select_origin_positions(
    base: pd.DataFrame,
    *,
    start_day: int,
    end_day: int,
    lead_steps: int,
    max_origins: int,
) -> list[int]:
    """Choose evenly spaced recursive origins within a day window."""
    if max_origins <= 0:
        raise ValueError("max_origins must be positive.")
    day_idx = base["day_idx"].to_numpy(dtype=int)
    limit = len(base) - lead_steps - 1
    candidates = [
        idx
        for idx in range(len(base))
        if idx <= limit and start_day <= int(day_idx[idx]) <= end_day and int(day_idx[idx + lead_steps]) <= end_day
    ]
    if not candidates:
        return []
    if len(candidates) <= max_origins:
        return candidates
    selection = np.linspace(0, len(candidates) - 1, num=max_origins, dtype=int)
    return [candidates[idx] for idx in selection]


def _window_values(
    history: pd.Series, target_timestamp: pd.Timestamp, *, resolution: str, window: int
) -> np.ndarray:
    """Return the exact historical window used by recursive feature builders.

    The window ends one step before `target_timestamp` so any rolling statistic
    derived from it remains causal during multistep rollout simulation.
    """
    delta = resolution_timedelta(resolution)
    end = target_timestamp - delta
    start = end - (window - 1) * delta
    index = pd.date_range(start=start, end=end, freq=delta)
    return history.reindex(index).to_numpy(dtype=float)


def _profile_value(profile: WorkdayProfile, *, timestamp: pd.Timestamp, day_class: str) -> float:
    """Look up one profile-baseline value with slot and global fallbacks.

    This helper is shared by recursive baseline paths that need one-step profile
    values at future timestamps when only the historical workday profile object
    is available.
    """
    workday = int(DAY_CLASS_MAP[day_class])
    slot = timestamp.strftime("%H:%M:%S")
    key = (workday, slot)
    if key in profile.by_workday_slot:
        return float(profile.by_workday_slot[key])
    if slot in profile.by_slot:
        return float(profile.by_slot[slot])
    return float(profile.global_mean)


def build_recursive_feature_row(
    history: pd.Series,
    *,
    target_timestamp: pd.Timestamp,
    resolution: str,
    day_class: str,
    profile: WorkdayProfile | None = None,
    day_class_lookup: dict[pd.Timestamp, str] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build one causal feature row for a future target timestamp."""
    requested_columns = set(feature_columns or [])

    def need(column_name: str) -> bool:
        return not requested_columns or column_name in requested_columns

    prev_day = (target_timestamp - pd.Timedelta(days=1)).normalize()
    next_day = (target_timestamp + pd.Timedelta(days=1)).normalize()
    prev_day_class = (
        str(day_class_lookup.get(prev_day, day_class))
        if day_class_lookup is not None
        else str(day_class)
    )
    next_day_class = (
        str(day_class_lookup.get(next_day, day_class))
        if day_class_lookup is not None
        else str(day_class)
    )
    workday_value = int(DAY_CLASS_MAP[day_class])
    prev_day_workday_value = int(DAY_CLASS_MAP[prev_day_class])
    next_day_workday_value = int(DAY_CLASS_MAP[next_day_class])
    slot_label = target_timestamp.strftime("%H:%M:%S")
    row: dict[str, Any] = {"timestamp": target_timestamp, "day_class": day_class}
    if need("workday"):
        row["workday"] = workday_value
    if need("year"):
        row["year"] = int(target_timestamp.year)
    if need("quarter"):
        row["quarter"] = int(target_timestamp.quarter)
    if need("month"):
        row["month"] = int(target_timestamp.month)
    if need("day"):
        row["day"] = int(target_timestamp.day)
    if need("day_of_week"):
        row["day_of_week"] = int((target_timestamp.day_of_week + 1) % 7)
    if need("hour"):
        row["hour"] = int(target_timestamp.hour)
    if need("season"):
        row["season"] = int(((target_timestamp.month % 12) // 3) + 1)
    if need("time_of_day"):
        row["time_of_day"] = int(hour_to_time_of_day(int(target_timestamp.hour)))
    if need("slot_label"):
        row["slot_label"] = slot_label
    if need("prev_day_workday") or need("workday_transition"):
        row["prev_day_workday"] = prev_day_workday_value
    if need("next_day_workday") or need("workday_transition"):
        row["next_day_workday"] = next_day_workday_value
    if need("workday_transition"):
        row["workday_transition"] = float(
            (prev_day_workday_value != workday_value) or (next_day_workday_value != workday_value)
        )
    delta = resolution_timedelta(resolution)

    if any(need(column) for column in ("hour_sin", "hour_cos", "dow_sin", "dow_cos")):
        fourier_frame = build_fourier_feature_frame(
            pd.DatetimeIndex([target_timestamp]),
            cycles=FEATURE_CONFIG["fourier_cycles"],
        )
        for column_name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            if need(column_name):
                row[column_name] = float(fourier_frame.iloc[0][column_name])

    if any(
        need(column)
        for column in (
            "phase_minute_15m",
            "phase_progress_15m",
            "phase_boundary_dist_15m",
            "phase_boundary_flag_15m",
            "phase_sin_15m",
            "phase_cos_15m",
        )
    ):
        phase_seconds = int(((target_timestamp.minute * 60) + target_timestamp.second) % (15 * 60))
        distance = float(min(phase_seconds, (15 * 60) - phase_seconds))
        boundary_band_seconds = min(max(int(resolution_seconds(resolution)), 60), 300)
        if need("phase_minute_15m"):
            row["phase_minute_15m"] = int(phase_seconds // 60)
        if need("phase_progress_15m"):
            row["phase_progress_15m"] = float(phase_seconds) / float(15 * 60)
        if need("phase_boundary_dist_15m"):
            row["phase_boundary_dist_15m"] = distance / 60.0
        if need("phase_boundary_flag_15m"):
            row["phase_boundary_flag_15m"] = int(distance <= float(boundary_band_seconds))
        if need("phase_sin_15m") or need("phase_cos_15m"):
            angle = 2.0 * math.pi * (float(phase_seconds) / float(15 * 60))
            if need("phase_sin_15m"):
                row["phase_sin_15m"] = float(math.sin(angle))
            if need("phase_cos_15m"):
                row["phase_cos_15m"] = float(math.cos(angle))

    lag_periods_needed = [
        int(lag)
        for lag in FEATURE_CONFIG["lag_periods"]
        if need(f"lag_{lag}")
        or (int(lag) == 1 and any(need(f"delta_{other_lag}") for other_lag in FEATURE_CONFIG["lag_periods"] if int(other_lag) != 1))
        or any(
            need(column_name)
            for column_name in (f"delta_{lag}", "anchored_workday_baseline", "profile_residual_lag_1")
        )
    ]
    for lag in lag_periods_needed:
        row[f"lag_{lag}"] = history.get(target_timestamp - lag * delta, np.nan)

    rolling_periods_needed = [
        int(window)
        for window in FEATURE_CONFIG["rolling_periods"]
        if any(
            need(column_name)
            for column_name in (
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_max_{window}",
                f"rolling_min_{window}",
            )
        )
    ]
    for window in rolling_periods_needed:
        values = _window_values(history, target_timestamp, resolution=resolution, window=window)
        finite_values = np.isfinite(values).all()
        if need(f"rolling_mean_{window}"):
            row[f"rolling_mean_{window}"] = float(np.mean(values)) if finite_values else np.nan
        if need(f"rolling_std_{window}"):
            row[f"rolling_std_{window}"] = (
                float(np.std(values, ddof=1)) if finite_values and len(values) > 1 else np.nan
            )
        if need(f"rolling_max_{window}"):
            row[f"rolling_max_{window}"] = float(np.max(values)) if finite_values else np.nan
        if need(f"rolling_min_{window}"):
            row[f"rolling_min_{window}"] = float(np.min(values)) if finite_values else np.nan

    for lag in FEATURE_CONFIG["lag_periods"]:
        if lag == 1 or not need(f"delta_{lag}"):
            continue
        lag_value = row[f"lag_{lag}"]
        lag_one = row["lag_1"]
        row[f"delta_{lag}"] = float(lag_value - lag_one) if pd.notna(lag_value) and pd.notna(lag_one) else np.nan

    slope_periods_needed = [
        int(window) for window in FEATURE_CONFIG["slope_periods"] if need(f"slope_{window}")
    ]
    for window in slope_periods_needed:
        values = _window_values(history, target_timestamp, resolution=resolution, window=window)
        row[f"slope_{window}"] = rolling_slope(values)

    if need("previous_day_load") or need("previous_day_residual"):
        row["previous_day_load"] = history.get(target_timestamp - pd.Timedelta(days=1), np.nan)
    for minutes in FEATURE_CONFIG["lag_minutes"]:
        if not need(f"lag_min_{minutes}"):
            continue
        steps = normalized_window_steps(resolution, int(minutes))
        row[f"lag_min_{minutes}"] = history.get(target_timestamp - steps * delta, np.nan)
    for minutes in FEATURE_CONFIG["rolling_minutes"]:
        if not any(
            need(column_name)
            for column_name in (
                f"rolling_mean_min_{minutes}",
                f"rolling_std_min_{minutes}",
                f"rolling_max_min_{minutes}",
                f"rolling_min_min_{minutes}",
            )
        ):
            continue
        steps = normalized_window_steps(resolution, int(minutes))
        values = _window_values(history, target_timestamp, resolution=resolution, window=steps)
        finite = np.isfinite(values).all()
        if need(f"rolling_mean_min_{minutes}"):
            row[f"rolling_mean_min_{minutes}"] = float(np.mean(values)) if finite else np.nan
        if need(f"rolling_std_min_{minutes}"):
            row[f"rolling_std_min_{minutes}"] = (
                float(np.std(values, ddof=1)) if finite and len(values) > 1 else np.nan
            )
        if need(f"rolling_max_min_{minutes}"):
            row[f"rolling_max_min_{minutes}"] = float(np.max(values)) if finite else np.nan
        if need(f"rolling_min_min_{minutes}"):
            row[f"rolling_min_min_{minutes}"] = float(np.min(values)) if finite else np.nan
    for minutes in FEATURE_CONFIG["slope_minutes"]:
        if not need(f"slope_min_{minutes}"):
            continue
        steps = normalized_window_steps(resolution, int(minutes))
        values = _window_values(history, target_timestamp, resolution=resolution, window=steps)
        row[f"slope_min_{minutes}"] = rolling_slope(values) if steps >= 2 else np.nan
    if profile is not None:
        need_profile_block = any(
            need(column_name)
            for column_name in (
                "avg_workday_baseline",
                "anchored_workday_baseline",
                "previous_day_residual",
                "profile_residual_lag_1",
                "profile_activity_ratio",
                "profile_active_flag",
            )
        )
        if need_profile_block:
            baseline = _profile_value(profile, timestamp=target_timestamp, day_class=day_class)
            prior_timestamp = target_timestamp - delta
            prior_day = (
                str(day_class_lookup.get(prior_timestamp.normalize(), day_class))
                if day_class_lookup is not None
                else str(day_class)
            )
            prior_baseline = _profile_value(profile, timestamp=prior_timestamp, day_class=prior_day)
            lag_one = row.get("lag_1", history.get(target_timestamp - delta, np.nan))
            previous_day_load = row.get(
                "previous_day_load",
                history.get(target_timestamp - pd.Timedelta(days=1), np.nan),
            )
            if need("avg_workday_baseline"):
                row["avg_workday_baseline"] = baseline
            if need("anchored_workday_baseline"):
                row["anchored_workday_baseline"] = (
                    float(baseline + (lag_one - prior_baseline))
                    if pd.notna(lag_one)
                    else np.nan
                )
            if need("previous_day_residual"):
                row["previous_day_residual"] = (
                    float(previous_day_load - baseline)
                    if pd.notna(previous_day_load)
                    else np.nan
                )
            if need("profile_residual_lag_1"):
                row["profile_residual_lag_1"] = (
                    float(lag_one - prior_baseline) if pd.notna(lag_one) else np.nan
                )
            if need("profile_activity_ratio") or need("profile_active_flag"):
                peak = float(profile.peak_by_workday.get(workday_value, np.nan))
                ratio = float(baseline / peak) if pd.notna(peak) and peak != 0.0 else np.nan
                if need("profile_activity_ratio"):
                    row["profile_activity_ratio"] = ratio
                if need("profile_active_flag"):
                    row["profile_active_flag"] = (
                        float(ratio >= float(FEATURE_CONFIG["profile_activity_threshold"]))
                        if pd.notna(ratio)
                        else np.nan
                    )
    if need("avg_load"):
        row["avg_load"] = np.nan
    return pd.DataFrame([row])


def recursive_predict_path(
    *,
    trained: TrainedModel,
    history: pd.Series,
    origin_timestamp: pd.Timestamp,
    horizon_steps: int,
    resolution: str,
    day_class_lookup: dict[pd.Timestamp, str],
    profile: WorkdayProfile | None = None,
) -> pd.DataFrame:
    """Generate a recursive forecast path from a trained one-step model."""
    delta = resolution_timedelta(resolution)
    history_work = history.copy()
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon_steps + 1):
        target_timestamp = origin_timestamp + step * delta
        day_key = target_timestamp.normalize()
        if day_key not in day_class_lookup:
            raise KeyError(f"Missing day_class lookup for forecast date {day_key.date()}.")
        feature_row = build_recursive_feature_row(
            history_work,
            target_timestamp=target_timestamp,
            resolution=resolution,
            day_class=day_class_lookup[day_key],
            profile=profile,
            day_class_lookup=day_class_lookup,
            feature_columns=list(trained.feature_columns),
        )
        prediction = float(predict_model(trained, feature_row).iloc[0])
        history_work.loc[target_timestamp] = prediction
        rows.append({"timestamp": target_timestamp, "y_pred": prediction})
    return pd.DataFrame(rows)


def recursive_predict_residual_path(
    *,
    trained: TrainedModel,
    history: pd.Series,
    origin_timestamp: pd.Timestamp,
    horizon_steps: int,
    resolution: str,
    day_class_lookup: dict[pd.Timestamp, str],
    residual_baseline: str,
    profile: WorkdayProfile | None = None,
) -> pd.DataFrame:
    """Generate a recursive forecast path on residuals to a supported baseline."""
    delta = resolution_timedelta(resolution)
    history_work = history.copy()
    rows: list[dict[str, Any]] = []
    hybrid_weights = (
        np.array([float(MULTIRES_HYBRID["persistence_weight_end"])], dtype=float)
        if int(horizon_steps) <= 1
        else np.linspace(
            float(MULTIRES_HYBRID["persistence_weight_start"]),
            float(MULTIRES_HYBRID["persistence_weight_end"]),
            num=int(horizon_steps),
            dtype=float,
        )
    )
    for step in range(1, horizon_steps + 1):
        target_timestamp = origin_timestamp + step * delta
        day_key = target_timestamp.normalize()
        if day_key not in day_class_lookup:
            raise KeyError(f"Missing day_class lookup for forecast date {day_key.date()}.")
        day_class = day_class_lookup[day_key]
        feature_row = build_recursive_feature_row(
            history_work,
            target_timestamp=target_timestamp,
            resolution=resolution,
            day_class=day_class,
            profile=profile,
            day_class_lookup=day_class_lookup,
            feature_columns=list(trained.feature_columns),
        )
        if residual_baseline == "avg_workday":
            baseline = float(feature_row.iloc[0]["avg_workday_baseline"])
        elif residual_baseline == "anchored_workday":
            baseline = float(feature_row.iloc[0]["anchored_workday_baseline"])
        elif residual_baseline == "hybrid_workday":
            anchored = float(feature_row.iloc[0]["anchored_workday_baseline"])
            persistence = float(history_work.iloc[-1])
            persistence_weight = float(hybrid_weights[step - 1])
            baseline = persistence_weight * persistence + (1.0 - persistence_weight) * anchored
        elif residual_baseline == "persistence":
            baseline = float(history_work.iloc[-1])
        else:
            raise ValueError(f"Unsupported residual baseline: {residual_baseline}")
        residual = float(predict_model(trained, feature_row).iloc[0])
        prediction = baseline + residual
        history_work.loc[target_timestamp] = prediction
        rows.append({"timestamp": target_timestamp, "y_pred": prediction})
    return pd.DataFrame(rows)


def persistence_path(history: pd.Series, *, origin_timestamp: pd.Timestamp, horizon_steps: int, resolution: str) -> pd.DataFrame:
    """Generate a constant persistence path from the last observed value."""
    delta = resolution_timedelta(resolution)
    last_value = float(history.iloc[-1])
    timestamps = [origin_timestamp + step * delta for step in range(1, horizon_steps + 1)]
    return pd.DataFrame({"timestamp": timestamps, "y_pred": [last_value] * len(timestamps)})


def previous_day_path(history: pd.Series, *, origin_timestamp: pd.Timestamp, horizon_steps: int, resolution: str) -> pd.DataFrame:
    """Generate a previous-day same-slot baseline path when historical values exist."""
    delta = resolution_timedelta(resolution)
    timestamps = [origin_timestamp + step * delta for step in range(1, horizon_steps + 1)]
    preds = [history.get(timestamp - pd.Timedelta(days=1), np.nan) for timestamp in timestamps]
    return pd.DataFrame({"timestamp": timestamps, "y_pred": preds})


def avg_workday_path(
    profile: WorkdayProfile,
    *,
    origin_timestamp: pd.Timestamp,
    horizon_steps: int,
    resolution: str,
    day_class_lookup: dict[pd.Timestamp, str],
) -> pd.DataFrame:
    """Generate an average workday profile path for future timestamps."""
    delta = resolution_timedelta(resolution)
    timestamps = pd.Series(
        [origin_timestamp + step * delta for step in range(1, horizon_steps + 1)],
        dtype="datetime64[ns]",
    )
    workday_values = timestamps.dt.normalize().map(lambda date: DAY_CLASS_MAP[day_class_lookup[date]])
    preds = profile_predict(timestamps, workday_values, profile)
    return pd.DataFrame({"timestamp": timestamps, "y_pred": preds})


def anchored_workday_path(
    profile: WorkdayProfile,
    *,
    history: pd.Series,
    origin_timestamp: pd.Timestamp,
    horizon_steps: int,
    resolution: str,
    day_class_lookup: dict[pd.Timestamp, str],
) -> pd.DataFrame:
    """Anchor the average-workday shape to the latest observed level.

    This keeps the workday-profile trajectory but removes level bias by forcing the
    origin slot to match the most recent observed load. It behaves like a hybrid of
    persistence (level anchor) and workday profile (future shape).
    """
    raw_profile = avg_workday_path(
        profile,
        origin_timestamp=origin_timestamp,
        horizon_steps=horizon_steps,
        resolution=resolution,
        day_class_lookup=day_class_lookup,
    )
    origin_day = origin_timestamp.normalize()
    if origin_day not in day_class_lookup:
        raise KeyError(f"Missing day_class lookup for origin date {origin_day.date()}.")
    origin_workday = pd.Series([int(DAY_CLASS_MAP[day_class_lookup[origin_day]])], dtype=int)
    origin_slot = pd.Series([origin_timestamp], dtype="datetime64[ns]")
    origin_profile_level = float(profile_predict(origin_slot, origin_workday, profile).iloc[0])
    anchored = raw_profile.copy()
    anchored["y_pred"] = float(history.iloc[-1]) + (anchored["y_pred"] - origin_profile_level)
    return anchored

def blend_candidate_paths(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    primary_weight_start: float,
    primary_weight_end: float,
    curve: str = "linear",
) -> pd.DataFrame:
    """Blend two path candidates across the horizon using a deterministic weight curve."""
    if curve != "linear":
        raise ValueError(f"Unsupported blend curve: {curve}")
    merged = primary.merge(
        secondary,
        on="timestamp",
        how="inner",
        suffixes=("_primary", "_secondary"),
    )
    if merged.empty or len(merged) != len(primary) or len(merged) != len(secondary):
        raise ValueError("Blend paths must share the same timestamp grid.")
    if len(merged) == 1:
        weights = np.array([primary_weight_end], dtype=float)
    else:
        weights = np.linspace(primary_weight_start, primary_weight_end, num=len(merged), dtype=float)
    merged["y_pred"] = (
        weights * merged["y_pred_primary"].to_numpy(dtype=float)
        + (1.0 - weights) * merged["y_pred_secondary"].to_numpy(dtype=float)
    )
    return merged.loc[:, ["timestamp", "y_pred"]].copy()


def actual_path(base: pd.DataFrame, *, origin_position: int, horizon_steps: int) -> pd.DataFrame:
    """Return the realized future path after an origin index."""
    path = base.iloc[origin_position + 1 : origin_position + 1 + horizon_steps][["timestamp", "avg_load"]].copy()
    return path.reset_index(drop=True)


def horizon_endpoint_metrics(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float]:
    """Return endpoint, path, and phase-average metrics for one recursive origin."""
    merged = actual.merge(predicted, on="timestamp", how="left")
    valid = merged.dropna(subset=["avg_load", "y_pred"])
    if valid.empty:
        return {
            "endpoint_abs_error": float("nan"),
            "endpoint_sq_error": float("nan"),
            "endpoint_actual_abs": float("nan"),
            "endpoint_ae_pct": float("nan"),
            "path_mae": float("nan"),
            "path_rmse": float("nan"),
            "path_abs_error_sum": float("nan"),
            "path_actual_abs_sum": float("nan"),
            "path_mae_pct": float("nan"),
            "phase_mean_abs_error": float("nan"),
            "phase_mean_sq_error": float("nan"),
            "phase_mean_actual_abs": float("nan"),
            "phase_mean_ae_pct": float("nan"),
            "next_lock_mae": float("nan"),
            "next_lock_abs_error_sum": float("nan"),
            "next_lock_actual_abs_sum": float("nan"),
            "next_lock_mae_pct": float("nan"),
            "profile_shape_mae": float("nan"),
            "profile_shape_abs_error_sum": float("nan"),
            "profile_shape_actual_abs_sum": float("nan"),
            "profile_shape_mae_pct": float("nan"),
            "energy_abs_error": float("nan"),
            "energy_actual_abs": float("nan"),
            "energy_abs_error_pct": float("nan"),
            "coverage": 0.0,
            "n_eval": 0.0,
        }
    endpoint_row = valid.iloc[-1]
    actual_values = valid["avg_load"].to_numpy(dtype=float)
    pred_values = valid["y_pred"].to_numpy(dtype=float)
    errors = actual_values - pred_values
    endpoint_error = float(endpoint_row["avg_load"] - endpoint_row["y_pred"])
    endpoint_actual_abs = float(abs(float(endpoint_row["avg_load"])))
    path_abs_error_sum = float(np.sum(np.abs(errors)))
    path_actual_abs_sum = float(np.sum(np.abs(actual_values)))
    path_mae = float(np.mean(np.abs(errors)))
    path_actual_abs_mean = float(np.mean(np.abs(actual_values)))
    phase_actual_mean = float(np.mean(actual_values))
    phase_pred_mean = float(np.mean(pred_values))
    phase_mean_abs_error = float(abs(phase_actual_mean - phase_pred_mean))
    phase_mean_actual_abs = float(abs(phase_actual_mean))
    if len(valid) > 1:
        lock_delta = pd.Timestamp(valid.iloc[1]["timestamp"]) - pd.Timestamp(valid.iloc[0]["timestamp"])
        next_lock_steps = max(1, int(pd.Timedelta(minutes=15) // lock_delta))
    else:
        next_lock_steps = 1
    next_lock_values = actual_values[:next_lock_steps]
    next_lock_preds = pred_values[:next_lock_steps]
    next_lock_errors = next_lock_values - next_lock_preds
    next_lock_abs_error_sum = float(np.sum(np.abs(next_lock_errors)))
    next_lock_actual_abs_sum = float(np.sum(np.abs(next_lock_values)))
    next_lock_mae = float(np.mean(np.abs(next_lock_errors)))

    actual_total = float(np.sum(actual_values))
    pred_total = float(np.sum(pred_values))
    energy_abs_error = float(abs(actual_total - pred_total))
    energy_actual_abs = float(abs(actual_total))
    if abs(pred_total) > 1e-9:
        scaled_pred_values = pred_values * (actual_total / pred_total)
        profile_shape_abs_errors = np.abs(actual_values - scaled_pred_values)
        profile_shape_abs_error_sum = float(np.sum(profile_shape_abs_errors))
        profile_shape_actual_abs_sum = float(np.sum(np.abs(actual_values)))
        profile_shape_mae = float(np.mean(profile_shape_abs_errors))
    else:
        profile_shape_abs_error_sum = float("nan")
        profile_shape_actual_abs_sum = float(np.sum(np.abs(actual_values)))
        profile_shape_mae = float("nan")
    return {
        "endpoint_abs_error": float(abs(endpoint_error)),
        "endpoint_sq_error": float(endpoint_error * endpoint_error),
        "endpoint_actual_abs": endpoint_actual_abs,
        "endpoint_ae_pct": safe_percent(float(abs(endpoint_error)), endpoint_actual_abs),
        "path_mae": path_mae,
        "path_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "path_abs_error_sum": path_abs_error_sum,
        "path_actual_abs_sum": path_actual_abs_sum,
        "path_mae_pct": safe_percent(path_mae, path_actual_abs_mean),
        "phase_mean_abs_error": phase_mean_abs_error,
        "phase_mean_sq_error": float((phase_actual_mean - phase_pred_mean) ** 2),
        "phase_mean_actual_abs": phase_mean_actual_abs,
        "phase_mean_ae_pct": safe_percent(phase_mean_abs_error, phase_mean_actual_abs),
        "next_lock_mae": next_lock_mae,
        "next_lock_abs_error_sum": next_lock_abs_error_sum,
        "next_lock_actual_abs_sum": next_lock_actual_abs_sum,
        "next_lock_mae_pct": safe_percent(
            next_lock_mae,
            float(np.mean(np.abs(next_lock_values))),
        ),
        "profile_shape_mae": profile_shape_mae,
        "profile_shape_abs_error_sum": profile_shape_abs_error_sum,
        "profile_shape_actual_abs_sum": profile_shape_actual_abs_sum,
        "profile_shape_mae_pct": safe_percent(profile_shape_mae, path_actual_abs_mean),
        "energy_abs_error": energy_abs_error,
        "energy_actual_abs": energy_actual_abs,
        "energy_abs_error_pct": safe_percent(energy_abs_error, energy_actual_abs),
        "coverage": float(len(valid) / len(actual)) if len(actual) > 0 else 0.0,
        "n_eval": float(len(valid)),
    }


def compare_recursive_paths(actual: pd.DataFrame, candidate_paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate multiple candidate paths against one actual rollout."""
    rows: list[dict[str, Any]] = []
    for label, predicted in candidate_paths.items():
        metrics = horizon_endpoint_metrics(actual, predicted)
        rows.append({"candidate_label": label, **metrics})
    return pd.DataFrame(rows)


def split_day_bounds(split_name: str) -> tuple[int, int]:
    """Return configured day bounds for a named split."""
    return tuple(int(value) for value in SPLIT_DAY_RANGES[split_name])


def lead_target_series(frame: pd.DataFrame, *, resolution: str, horizon_minutes: int) -> pd.Series:
    """Build a direct lead target series on a causal one-step frame."""
    steps = lead_steps_for_horizon(resolution, horizon_minutes)
    return frame["avg_load"].shift(-steps)
