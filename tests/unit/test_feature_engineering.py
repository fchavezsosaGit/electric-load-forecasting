"""Feature engineering and EDA utility tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from modeling.feature_engineering import add_time_normalized_features
from scripts.config import DAY_CLASS_MAP, EDA_CONFIG
from scripts.utils import (
    adaptive_outlier_threshold,
    hour_to_time_of_day,
    month_to_season,
    optimal_acf_depth,
    optimal_bin_count,
    rolling_slope,
    rolling_slope_series,
)


def test_month_to_season_values():
    """Ensure month-to-season mapping returns expected season codes."""
    assert month_to_season(12) == 1
    assert month_to_season(3) == 2
    assert month_to_season(6) == 3
    assert month_to_season(9) == 4
    assert month_to_season(11) == 4
    assert month_to_season(2) == 1


def test_hour_to_time_of_day_boundaries():
    """Ensure hour bucket boundaries map to expected time-of-day codes."""
    assert hour_to_time_of_day(6) == 0
    assert hour_to_time_of_day(11) == 0
    assert hour_to_time_of_day(12) == 1
    assert hour_to_time_of_day(16) == 1
    assert hour_to_time_of_day(17) == 2
    assert hour_to_time_of_day(21) == 2
    assert hour_to_time_of_day(22) == 3
    assert hour_to_time_of_day(5) == 3


def test_rolling_slope_cases():
    """Ensure rolling_slope handles increasing, constant, and NaN inputs."""
    increasing = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    constant = np.array([7.0, 7.0, 7.0, 7.0], dtype=float)
    has_nan = np.array([1.0, np.nan, 3.0], dtype=float)

    assert rolling_slope(increasing) > 0
    assert rolling_slope(constant) == 0.0
    assert np.isnan(rolling_slope(has_nan))


def test_workday_mapping():
    """Ensure configured day-class mappings remain stable."""
    assert DAY_CLASS_MAP["none"] == 0
    assert DAY_CLASS_MAP["half"] == 1
    assert DAY_CLASS_MAP["full"] == 2


def test_rolling_slope_series_linear_positive():
    """Ensure positive linear trends produce positive rolling slopes."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    result = rolling_slope_series(series, window=3)
    assert result.iloc[2] > 0
    assert result.iloc[3] > 0
    assert result.iloc[4] > 0


def test_rolling_slope_series_constant_zero():
    """Ensure constant series produce zero non-null rolling slopes."""
    series = pd.Series([7.0, 7.0, 7.0, 7.0, 7.0], dtype=float)
    result = rolling_slope_series(series, window=3)
    assert np.allclose(result.dropna().to_numpy(), 0.0)


def test_rolling_slope_series_nan_propagation():
    """Ensure windows containing NaN values produce NaN slopes."""
    series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], dtype=float)
    result = rolling_slope_series(series, window=3)
    assert np.isnan(result.iloc[2])
    assert np.isnan(result.iloc[3])


def test_rolling_slope_series_window_larger_than_series_returns_all_nan():
    """Ensure oversized windows return all-NaN output."""
    series = pd.Series([1.0, 2.0, 3.0], dtype=float)
    result = rolling_slope_series(series, window=5)
    assert result.isna().all()


def test_rolling_slope_series_window_two():
    """Ensure two-point windows compute first-order slope correctly."""
    series = pd.Series([10.0, 12.0, 14.0], dtype=float)
    result = rolling_slope_series(series, window=2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[2] == pytest.approx(2.0)


def test_time_normalized_features_define_collapsed_single_step_windows():
    """Single-step clock windows should stay causal instead of producing all-null coarse features."""
    index = pd.date_range("2025-01-01 00:00:00", periods=4, freq="15min")
    frame = pd.DataFrame({"avg_load": [10.0, 20.0, 30.0, 40.0]}, index=index)

    engineered = add_time_normalized_features(frame, resolution="15min")

    assert np.isnan(engineered.loc[index[0], "rolling_std_min_15"])
    assert np.isnan(engineered.loc[index[0], "slope_min_15"])
    assert engineered.loc[index[1]:, "rolling_std_min_15"].eq(0.0).all()
    assert engineered.loc[index[1]:, "slope_min_15"].eq(0.0).all()
    assert engineered["rolling_std_min_15"].notna().sum() == len(index) - 1
    assert engineered["slope_min_15"].notna().sum() == len(index) - 1


def test_rolling_slope_series_large_window_remains_linear():
    """Ensure very large windows stay numerically correct without huge allocations."""
    series = pd.Series(np.arange(20_000, dtype=float))
    result = rolling_slope_series(series, window=14_400)
    assert result.iloc[:14_399].isna().all()
    assert result.iloc[-1] == pytest.approx(1.0, abs=1e-12)
    assert result.iloc[-10:].notna().all()


def test_month_to_season_invalid_input_raises():
    """Ensure invalid months raise ValueError."""
    with pytest.raises(ValueError):
        month_to_season(0)
    with pytest.raises(ValueError):
        month_to_season(13)


def test_hour_to_time_of_day_invalid_input_raises():
    """Ensure invalid hours raise ValueError."""
    with pytest.raises(ValueError):
        hour_to_time_of_day(-1)
    with pytest.raises(ValueError):
        hour_to_time_of_day(24)


def test_rolling_slope_invalid_shape_raises():
    """Ensure non-1D rolling_slope inputs raise ValueError."""
    with pytest.raises(ValueError):
        rolling_slope(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))


def test_optimal_bin_count_methods_and_clamping():
    """Ensure bin count works across methods and clamps correctly."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=2000)

    fd_bins = optimal_bin_count(data, method="fd", min_bins=10, max_bins=120)
    sturges_bins = optimal_bin_count(data, method="sturges", min_bins=10, max_bins=120)
    sqrt_bins = optimal_bin_count(data, method="sqrt", min_bins=10, max_bins=120)

    assert 10 <= fd_bins <= 120
    assert 10 <= sturges_bins <= 120
    assert 10 <= sqrt_bins <= 120


def test_optimal_bin_count_edge_cases_return_min_bins():
    """Ensure constant, empty, and all-NaN data return min_bins."""
    assert optimal_bin_count([5.0, 5.0, 5.0], min_bins=11, max_bins=100) == 11
    assert optimal_bin_count([], min_bins=9, max_bins=100) == 9
    assert optimal_bin_count([np.nan, np.nan], min_bins=13, max_bins=100) == 13


def test_optimal_bin_count_invalid_method_raises():
    """Ensure unknown binning methods raise ValueError."""
    with pytest.raises(ValueError, match="Unknown binning method"):
        optimal_bin_count([1.0, 2.0, 3.0], method="bad")


def test_adaptive_outlier_threshold_iqr_bounds():
    """Ensure IQR method returns standard Tukey bounds."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=float)
    bounds = adaptive_outlier_threshold(data, method="iqr")

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    assert bounds["lower"] == pytest.approx(q1 - 1.5 * iqr)
    assert bounds["upper"] == pytest.approx(q3 + 1.5 * iqr)
    assert bounds["method"] == "iqr"


def test_adaptive_outlier_threshold_mad_for_skewed_data():
    """Ensure MAD method returns finite symmetric bounds around median."""
    data = np.array([1.0, 1.1, 1.2, 1.3, 20.0], dtype=float)
    bounds = adaptive_outlier_threshold(data, method="mad")

    median = float(np.median(data))
    assert math.isfinite(float(bounds["lower"]))
    assert math.isfinite(float(bounds["upper"]))
    assert float(bounds["lower"]) < median < float(bounds["upper"])
    assert bounds["method"] == "mad"


def test_adaptive_outlier_threshold_zscore_uses_config_default():
    """Ensure zscore method uses centralized config threshold."""
    data = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
    bounds = adaptive_outlier_threshold(data, method="zscore")

    z = EDA_CONFIG["zscore_threshold"]
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=0))
    assert bounds["lower"] == pytest.approx(mean - z * std)
    assert bounds["upper"] == pytest.approx(mean + z * std)
    assert bounds["method"] == "zscore"


def test_adaptive_outlier_threshold_all_nan_returns_nan_bounds():
    """Ensure all-NaN input returns NaN bounds without raising."""
    bounds = adaptive_outlier_threshold([np.nan, np.nan], method="iqr")
    assert np.isnan(float(bounds["lower"]))
    assert np.isnan(float(bounds["upper"]))


def test_optimal_acf_depth_autocorrelated_series_returns_larger_depth():
    """Ensure highly autocorrelated series yields depth above min_depth."""
    rng = np.random.default_rng(42)
    n = 1200
    eps = rng.normal(scale=0.1, size=n)
    series = np.zeros(n, dtype=float)
    for idx in range(1, n):
        series[idx] = 0.95 * series[idx - 1] + eps[idx]

    depth = optimal_acf_depth(series, min_depth=10, max_depth=500, consecutive_insignificant=5)
    assert depth > 10


def test_optimal_acf_depth_white_noise_returns_min_depth():
    """Ensure white-noise series tends to return the minimum depth."""
    rng = np.random.default_rng(0)
    noise = rng.normal(size=500)
    depth = optimal_acf_depth(noise, min_depth=10, max_depth=200, consecutive_insignificant=5)
    assert depth == 10


def test_optimal_acf_depth_all_nan_returns_min_depth():
    """Ensure all-NaN series returns min_depth without raising."""
    depth = optimal_acf_depth([np.nan, np.nan, np.nan], min_depth=7, max_depth=100)
    assert depth == 7
