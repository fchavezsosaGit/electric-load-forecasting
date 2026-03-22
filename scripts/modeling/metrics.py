"""Shared scale-aware regression metric helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def safe_percent(numerator: float, denominator: float) -> float:
    """Return a percentage when the denominator is positive and finite."""
    if denominator <= 0 or math.isnan(numerator) or math.isnan(denominator):
        return float("nan")
    return float(100.0 * numerator / denominator)


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    n_total: int | None = None,
) -> dict[str, float | int]:
    """Compute error metrics plus scale-aware percentage variants.

    `mae_pct` and `rmse_pct` are normalized by the mean absolute actual load over
    the valid evaluation rows, so they remain interpretable when the raw load
    scale changes across sites or load types.
    """
    valid = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if valid.empty:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mae_pct": float("nan"),
            "rmse_pct": float("nan"),
            "mean_abs_actual": float("nan"),
            "total_abs_actual": float("nan"),
            "n_eval": 0,
            "coverage": 0.0 if n_total is not None else float("nan"),
        }
    actual = valid["y_true"].to_numpy(dtype=float)
    predicted = valid["y_pred"].to_numpy(dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mean_abs_actual = float(np.mean(np.abs(actual)))
    total_abs_actual = float(np.sum(np.abs(actual)))
    coverage = float(len(valid) / n_total) if n_total is not None and n_total > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_pct": safe_percent(mae, mean_abs_actual),
        "rmse_pct": safe_percent(rmse, mean_abs_actual),
        "mean_abs_actual": mean_abs_actual,
        "total_abs_actual": total_abs_actual,
        "n_eval": int(len(valid)),
        "coverage": coverage,
    }


def aggregate_absolute_error_percentage(
    *,
    error_sum: float,
    actual_abs_sum: float,
) -> float:
    """Return an aggregated absolute-error percentage from summed numerators/denominators."""
    if math.isnan(error_sum) or math.isnan(actual_abs_sum):
        return float("nan")
    return safe_percent(float(error_sum), float(actual_abs_sum))


def json_safe_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy/pandas scalars inside metric dicts into plain Python values."""
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            normalized[key] = value.item()
        elif value is pd.NA:
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized
