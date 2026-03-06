"""Shared utility helpers for feature engineering, EDA validation, and logging."""

from __future__ import annotations

import logging
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def emit_quality_gate(
    gate_name: str,
    passed: bool,
    *,
    details: dict[str, object] | None = None,
    logger_instance: logging.Logger | None = None,
    failure_level: int = logging.WARNING,
) -> None:
    """Emit a standardized PASS/FAIL gate summary to logs.

    This keeps stage-end health messages machine-readable and consistent across
    pipeline scripts while leaving hard structural failures to the calling code.
    """
    target_logger = logger_instance or logger
    status = "PASS" if passed else "FAIL"
    message = f"{gate_name}: {status}"
    if details:
        rendered = " | ".join(f"{key}={value}" for key, value in details.items())
        message = f"{message} | {rendered}"
    if passed:
        target_logger.info(message)
    else:
        target_logger.log(failure_level, message)


def month_to_season(month: int) -> int:
    """Map month to season code.

    1 = Winter (Dec-Feb)
    2 = Spring (Mar-May)
    3 = Summer (Jun-Aug)
    4 = Fall (Sep-Nov)
    """
    if not isinstance(month, (int, np.integer)):
        raise ValueError(f"month must be an integer in [1, 12], got {type(month).__name__}")
    if month < 1 or month > 12:
        raise ValueError(f"month must be in [1, 12], got {month}")

    if month in (12, 1, 2):
        return 1
    if month in (3, 4, 5):
        return 2
    if month in (6, 7, 8):
        return 3
    return 4


def hour_to_time_of_day(hour: int) -> int:
    """Map hour to time-of-day bucket.

    0 = morning (6-11)
    1 = afternoon (12-16)
    2 = evening (17-21)
    3 = night (22-5)
    """
    if not isinstance(hour, (int, np.integer)):
        raise ValueError(f"hour must be an integer in [0, 23], got {type(hour).__name__}")
    if hour < 0 or hour > 23:
        raise ValueError(f"hour must be in [0, 23], got {hour}")

    if 6 <= hour <= 11:
        return 0
    if 12 <= hour <= 16:
        return 1
    if 17 <= hour <= 21:
        return 2
    return 3


def rolling_slope(values: np.ndarray) -> float:
    """Compute a least-squares slope over a 1D numeric array.

    Formula:
    slope = sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2)

    Behavior:
    - Constant arrays return 0.0.
    - Empty arrays, arrays with length < 2, or arrays containing NaN return NaN.
    - Non-1D or non-numeric input raises ValueError.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"rolling_slope expects a 1D array. Got ndim={array.ndim}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"rolling_slope expects numeric input. Got dtype={array.dtype}")
    if np.isinf(array).any():
        raise ValueError("rolling_slope does not accept +/-inf values.")

    if array.size < 2 or np.any(np.isnan(array)):
        return float("nan")
    x = np.arange(array.size, dtype=float)
    x_centered = x - x.mean()
    y_centered = array.astype(float) - array.mean()
    denom = np.sum(x_centered * x_centered)
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def rolling_slope_series(series: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling slope for a pandas Series.

    Returns a series aligned with `series` where the first `window - 1`
    observations are NaN. Any window containing NaN returns NaN.
    """
    if not isinstance(series, pd.Series):
        raise ValueError(f"rolling_slope_series expects a pandas Series, got {type(series).__name__}")
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")

    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, copy=False)
    if np.isinf(values).any():
        raise ValueError("rolling_slope_series does not accept +/-inf values in series.")
    slopes = np.full(values.shape[0], np.nan, dtype=float)

    if window == 1:
        raise ValueError("window must be >= 2 for slope calculation.")
    if values.shape[0] < window:
        logger.warning(
            "rolling_slope_series window (%d) exceeds series length (%d); returning all-NaN output",
            window,
            values.shape[0],
        )
        return pd.Series(slopes, index=series.index, dtype=float)

    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered * x_centered)

    has_nan = np.isnan(windows).any(axis=1)
    means = windows.mean(axis=1)
    centered = windows - means[:, None]
    numerators = centered @ x_centered
    slopes_window = numerators / denom
    slopes_window[has_nan] = np.nan

    slopes[window - 1 :] = slopes_window
    return pd.Series(slopes, index=series.index, dtype=float)


def _to_numeric_array(data: Any) -> np.ndarray:
    """Normalize list/array/series input to a finite float numpy array."""
    if isinstance(data, pd.Series):
        values = pd.to_numeric(data, errors="coerce").to_numpy(dtype=float, copy=False)
    else:
        values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        values = values.ravel()
    values = values[np.isfinite(values)]
    return values


def optimal_bin_count(
    data: Any,
    method: str = "fd",
    min_bins: int = 10,
    max_bins: int = 300,
) -> int:
    """Compute a data-driven histogram bin count.

    Supported methods:
    - "fd": Freedman-Diaconis rule.
    - "sturges": Sturges' rule.
    - "sqrt": square-root rule.

    Edge-case behavior:
    - Empty, all-NaN, or constant data returns `min_bins`.
    - Output is clamped to [min_bins, max_bins].
    """
    if min_bins <= 0 or max_bins <= 0:
        raise ValueError("min_bins and max_bins must be positive integers.")
    if min_bins > max_bins:
        raise ValueError("min_bins must be <= max_bins.")

    clean = _to_numeric_array(data)
    n = clean.size
    if n == 0:
        return int(min_bins)

    data_range = float(np.nanmax(clean) - np.nanmin(clean))
    if data_range == 0.0:
        return int(min_bins)

    method_key = method.lower().strip()
    if method_key == "fd":
        q1, q3 = np.percentile(clean, [25, 75])
        iqr = float(q3 - q1)
        if iqr == 0.0:
            bins = int(math.ceil(math.sqrt(n)))
        else:
            bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
            if bin_width <= 0.0:
                bins = int(min_bins)
            else:
                bins = int(math.ceil(data_range / bin_width))
    elif method_key == "sturges":
        bins = int(math.ceil(math.log2(n)) + 1)
    elif method_key == "sqrt":
        bins = int(math.ceil(math.sqrt(n)))
    else:
        raise ValueError(f"Unknown binning method '{method}'. Use one of: fd, sturges, sqrt.")

    bins = max(min_bins, bins)
    bins = min(max_bins, bins)
    return int(bins)


def adaptive_outlier_threshold(data: Any, method: str = "iqr") -> dict[str, float | str]:
    """Return lower/upper outlier bounds using a robust method.

    Methods:
    - "iqr": [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    - "zscore": [mean - z*std, mean + z*std] where z comes from EDA config
    - "mad": [median - k*MAD, median + k*MAD] with k=3

    Returns: {"lower": float, "upper": float, "method": str}
    """
    clean = _to_numeric_array(data)
    if clean.size == 0:
        return {"lower": float("nan"), "upper": float("nan"), "method": method}

    method_key = method.lower().strip()
    if method_key == "iqr":
        q1, q3 = np.percentile(clean, [25, 75])
        iqr = float(q3 - q1)
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
    elif method_key == "zscore":
        # Lazy import avoids unnecessary coupling for consumers that do not use this method.
        from config import EDA_CONFIG

        z = float(EDA_CONFIG["zscore_threshold"])
        mean = float(np.mean(clean))
        std = float(np.std(clean, ddof=0))
        if std == 0.0:
            lower = mean
            upper = mean
        else:
            lower = mean - z * std
            upper = mean + z * std
    elif method_key == "mad":
        median = float(np.median(clean))
        mad = float(np.median(np.abs(clean - median)))
        k = 3.0
        lower = median - k * mad
        upper = median + k * mad
    else:
        raise ValueError(f"Unknown outlier method '{method}'. Use one of: iqr, zscore, mad.")

    return {"lower": float(lower), "upper": float(upper), "method": method_key}


def optimal_acf_depth(
    series: Any,
    significance_level: float = 0.05,
    min_depth: int = 10,
    max_depth: int = 2000,
    consecutive_insignificant: int = 5,
) -> int:
    """Estimate an informative maximum lag depth from autocorrelation behavior.

    The function scans lag autocorrelation values and returns the lag where
    sustained insignificance begins. Significance bound is:
        z(alpha/2) / sqrt(n)
    with z(0.05/2) ~= 1.96.

    Edge-case behavior:
    - Empty/all-NaN/constant data returns `min_depth`.
    - Output is clamped to [min_depth, max_depth].
    """
    if min_depth <= 0 or max_depth <= 0:
        raise ValueError("min_depth and max_depth must be positive integers.")
    if min_depth > max_depth:
        raise ValueError("min_depth must be <= max_depth.")
    if not (0 < significance_level < 1):
        raise ValueError("significance_level must be in (0, 1).")
    if consecutive_insignificant <= 0:
        raise ValueError("consecutive_insignificant must be a positive integer.")

    clean = _to_numeric_array(series)
    n = clean.size
    if n < 3:
        return int(min_depth)
    if float(np.nanstd(clean, ddof=0)) == 0.0:
        return int(min_depth)

    max_scan = min(max_depth, n - 1)
    if max_scan < 1:
        return int(min_depth)

    z_value = NormalDist().inv_cdf(1.0 - significance_level / 2.0)
    significance_bound = float(z_value / math.sqrt(n))

    s = pd.Series(clean)
    consecutive = 0
    start_lag = 1

    for lag in range(1, max_scan + 1):
        acf_val = s.autocorr(lag=lag)
        if pd.isna(acf_val):
            acf_val = 0.0

        if abs(float(acf_val)) < significance_bound:
            if consecutive == 0:
                start_lag = lag
            consecutive += 1
            if consecutive >= consecutive_insignificant:
                return int(max(min_depth, min(max_depth, start_lag)))
        else:
            consecutive = 0

    return int(max(min_depth, min(max_depth, max_scan)))


def validate_schema_columns(
    df: pd.DataFrame, expected_columns: list[str], layer_name: str
) -> None:
    """Validate exact schema ordering and membership for a dataframe."""
    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        missing = [col for col in expected_columns if col not in actual_columns]
        unexpected = [col for col in actual_columns if col not in expected_columns]
        raise ValueError(
            f"{layer_name} schema mismatch. Missing={missing}, unexpected={unexpected}"
        )
