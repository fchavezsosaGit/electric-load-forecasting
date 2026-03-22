"""Shared causal feature-engineering helpers used by silver and multires stages."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import DAY_CLASS_MAP, FEATURE_CONFIG
from modeling.common import resolution_seconds, steps_per_day
from utils import build_fourier_feature_frame, hour_to_time_of_day, month_to_season, rolling_slope_series


def normalized_window_steps(resolution: str, minutes: int) -> int:
    """Return the minimum whole-step window that covers the requested clock duration."""
    seconds = resolution_seconds(resolution)
    return max(1, int(math.ceil((int(minutes) * 60) / seconds)))


def slot_labels(timestamps: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Return stable same-slot labels for workday-profile features."""
    return pd.to_datetime(timestamps, errors="raise").strftime("%H:%M:%S").astype("string")


def add_calendar_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar, business, and Fourier features to a timestamp-indexed frame."""
    work = frame.copy()
    ts = pd.DatetimeIndex(work.index)
    if work["day_class"].isna().any():
        raise ValueError("day_class must be non-null before feature engineering.")
    work["workday"] = work["day_class"].map(DAY_CLASS_MAP).astype("Int64")
    if work["workday"].isna().any():
        unknown = set(work.loc[work["workday"].isna(), "day_class"].dropna().unique())
        raise ValueError(f"Unable to map day_class values to workday codes: {sorted(unknown)}")
    work["year"] = ts.year.astype(int)
    work["quarter"] = ts.quarter.astype(int)
    work["month"] = ts.month.astype(int)
    work["day"] = ts.day.astype(int)
    work["day_of_week"] = ((ts.dayofweek + 1) % 7).astype(int)
    work["hour"] = ts.hour.astype(int)
    work["season"] = ts.month.map(month_to_season).astype(int)
    work["time_of_day"] = ts.hour.map(hour_to_time_of_day).astype(int)
    fourier = build_fourier_feature_frame(ts, cycles=FEATURE_CONFIG["fourier_cycles"])
    for column_name in fourier.columns:
        work[column_name] = fourier[column_name].to_numpy(dtype=float, copy=False)
    work["slot_label"] = slot_labels(ts)
    return work


def add_phase_context(frame: pd.DataFrame, *, resolution: str) -> pd.DataFrame:
    """Add quarter-hour phase features for short-horizon gating around settlement boundaries."""
    work = frame.copy()
    ts = pd.DatetimeIndex(work.index)
    phase_seconds = ((ts.minute * 60) + ts.second) % (15 * 60)
    phase_seconds = phase_seconds.astype(int)
    work["phase_minute_15m"] = (phase_seconds // 60).astype(int)
    work["phase_progress_15m"] = phase_seconds.astype(float) / float(15 * 60)
    distance = np.minimum(phase_seconds, (15 * 60) - phase_seconds).astype(float)
    work["phase_boundary_dist_15m"] = distance / 60.0
    boundary_band_seconds = min(max(int(resolution_seconds(resolution)), 60), 300)
    work["phase_boundary_flag_15m"] = (distance <= float(boundary_band_seconds)).astype(int)
    angle = 2.0 * math.pi * (phase_seconds.astype(float) / float(15 * 60))
    work["phase_sin_15m"] = np.sin(angle)
    work["phase_cos_15m"] = np.cos(angle)
    return work


def _groupwise_causal_mean(values: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """Return a per-group historical mean that excludes the current observation.

    This helper is used by the profile/regime feature builder to create
    same-slot and workday-slot baselines without leaking the current target into
    its own feature row.
    """
    grouped = values.groupby(keys, sort=False)
    prior_sum = grouped.cumsum() - values
    prior_count = grouped.cumcount()
    prior_count = prior_count.where(prior_count > 0, np.nan)
    return (prior_sum / prior_count).astype(float)


def add_period_history_features(frame: pd.DataFrame, *, resolution: str) -> pd.DataFrame:
    """Add existing period-based lag, rolling, delta, slope, and prior-day features."""
    work = frame.copy()
    row_count = work.shape[0]
    series = pd.to_numeric(work["avg_load"], errors="coerce")
    shifted = series.shift(1)

    for lag in FEATURE_CONFIG["lag_periods"]:
        work[f"lag_{lag}"] = series.shift(lag)

    for window in FEATURE_CONFIG["rolling_periods"]:
        rolling = shifted.rolling(window=window, min_periods=window)
        work[f"rolling_mean_{window}"] = rolling.mean()
        work[f"rolling_std_{window}"] = rolling.std()
        work[f"rolling_max_{window}"] = rolling.max()
        work[f"rolling_min_{window}"] = rolling.min()

    for lag in FEATURE_CONFIG["lag_periods"]:
        if lag == 1:
            continue
        work[f"delta_{lag}"] = work[f"lag_{lag}"] - work["lag_1"]

    for window in FEATURE_CONFIG["slope_periods"]:
        if window <= row_count:
            work[f"slope_{window}"] = rolling_slope_series(series, window).shift(1)
        else:
            work[f"slope_{window}"] = np.nan

    work["previous_day_load"] = series.shift(steps_per_day(resolution))
    return work


def add_time_normalized_features(frame: pd.DataFrame, *, resolution: str) -> pd.DataFrame:
    """Add resolution-adjusted clock-window lag, rolling, and slope features."""
    work = frame.copy()
    series = pd.to_numeric(work["avg_load"], errors="coerce")
    shifted = series.shift(1)

    for minutes in FEATURE_CONFIG["lag_minutes"]:
        steps = normalized_window_steps(resolution, int(minutes))
        work[f"lag_min_{minutes}"] = series.shift(steps)

    for minutes in FEATURE_CONFIG["rolling_minutes"]:
        steps = normalized_window_steps(resolution, int(minutes))
        rolling = shifted.rolling(window=steps, min_periods=steps)
        work[f"rolling_mean_min_{minutes}"] = rolling.mean()
        if steps >= 2:
            work[f"rolling_std_min_{minutes}"] = rolling.std()
        else:
            # A one-step clock window has zero spread once causal history exists.
            work[f"rolling_std_min_{minutes}"] = pd.Series(
                np.where(shifted.notna(), 0.0, np.nan),
                index=work.index,
                dtype=float,
            )
        work[f"rolling_max_min_{minutes}"] = rolling.max()
        work[f"rolling_min_min_{minutes}"] = rolling.min()

    for minutes in FEATURE_CONFIG["slope_minutes"]:
        steps = normalized_window_steps(resolution, int(minutes))
        if steps >= 2:
            work[f"slope_min_{minutes}"] = rolling_slope_series(series, steps).shift(1)
        else:
            # A one-step clock window has no resolvable trend beyond the prior point.
            work[f"slope_min_{minutes}"] = pd.Series(
                np.where(shifted.notna(), 0.0, np.nan),
                index=work.index,
                dtype=float,
            )
    return work


def add_profile_regime_features(frame: pd.DataFrame, *, resolution: str) -> pd.DataFrame:
    """Add causal profile baselines plus regime/context helper features."""
    work = frame.copy()
    series = pd.to_numeric(work["avg_load"], errors="coerce")
    if "slot_label" not in work.columns:
        work["slot_label"] = slot_labels(pd.DatetimeIndex(work.index))

    slot_label = work["slot_label"].astype("string")
    workday = pd.to_numeric(work["workday"], errors="coerce")
    global_prior = series.expanding(min_periods=1).mean().shift(1)
    slot_prior = _groupwise_causal_mean(series, [slot_label])
    workday_slot_prior = _groupwise_causal_mean(series, [workday, slot_label])
    work["avg_workday_baseline"] = workday_slot_prior.fillna(slot_prior).fillna(global_prior)

    prior_step_baseline = work["avg_workday_baseline"].shift(1)
    work["anchored_workday_baseline"] = work["avg_workday_baseline"] + (
        work["lag_1"] - prior_step_baseline
    )
    work["profile_residual_lag_1"] = work["lag_1"] - prior_step_baseline
    work["previous_day_residual"] = work["previous_day_load"] - work["avg_workday_baseline"]

    daily_steps = steps_per_day(resolution)
    current_workday = pd.to_numeric(work["workday"], errors="coerce").astype(float)
    work["prev_day_workday"] = current_workday.shift(daily_steps).fillna(current_workday)
    work["next_day_workday"] = current_workday.shift(-daily_steps).fillna(current_workday)
    work["workday_transition"] = np.where(
        (work["prev_day_workday"] != current_workday) | (work["next_day_workday"] != current_workday),
        1.0,
        0.0,
    )

    workday_peak = (
        work["avg_workday_baseline"]
        .groupby(current_workday, sort=False)
        .cummax()
        .replace(0.0, np.nan)
    )
    work["profile_activity_ratio"] = work["avg_workday_baseline"] / workday_peak
    work["profile_active_flag"] = np.where(
        work["profile_activity_ratio"].notna(),
        (
            work["profile_activity_ratio"]
            >= float(FEATURE_CONFIG["profile_activity_threshold"])
        ).astype(float),
        np.nan,
    )
    return work
