"""Unit tests for multiresolution causal feature and horizon alignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.common import (
    build_model_catalog,
    lead_steps_for_horizon,
    predict_model,
    resolution_total_minutes,
    stable_config_hash,
    train_model,
)
from modeling.multires import build_causal_feature_frame, select_origin_positions


def _base_frame(periods: int = 12) -> pd.DataFrame:
    """Build a minimal synthetic base frame for causal-feature alignment tests."""
    timestamps = pd.date_range("2025-12-01 00:00:00", periods=periods, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "avg_load": np.arange(1, periods + 1, dtype=float),
            "day_class": ["full"] * periods,
            "day_idx": [1] * periods,
        }
    )


def test_lead_steps_for_horizon_converts_and_rejects_invalid_pairs():
    """Convert valid resolution-horizon pairs and reject non-representable ones."""
    assert lead_steps_for_horizon("1min", 60) == 60
    assert lead_steps_for_horizon("5min", 60) == 12
    assert lead_steps_for_horizon("30s", 15) == 30
    assert lead_steps_for_horizon("10s", 15) == 90
    assert lead_steps_for_horizon("1s", 15) == 900
    with pytest.raises(ValueError, match="not representable"):
        lead_steps_for_horizon("10min", 15)


def test_resolution_total_minutes_preserves_subminute_precision():
    """Preserve subminute precision when normalizing supported resolutions."""
    assert resolution_total_minutes("1min") == 1.0
    assert resolution_total_minutes("30s") == 0.5
    assert resolution_total_minutes("10s") == pytest.approx(1.0 / 6.0)


def test_stable_config_hash_is_deterministic_for_equivalent_payloads():
    """Hash equivalent config payloads identically regardless of key order."""
    left = {"mode": "smoke", "resolutions": ["1min", "5min"], "horizons": [15, 60]}
    right = {"horizons": [15, 60], "resolutions": ["1min", "5min"], "mode": "smoke"}
    assert stable_config_hash(left) == stable_config_hash(right)


def test_build_causal_feature_frame_uses_shifted_history_not_current_target():
    """Ensure causal features use prior history rather than the current target row."""
    frame = build_causal_feature_frame(_base_frame(), "1min")

    assert pd.isna(frame.loc[0, "lag_1"])
    assert frame.loc[1, "lag_1"] == 1.0
    assert round(float(frame.loc[5, "rolling_mean_5"]), 6) == 3.0
    assert frame.loc[5, "rolling_max_5"] == 5.0
    assert frame.loc[5, "delta_5"] == frame.loc[5, "lag_5"] - frame.loc[5, "lag_1"]


def test_select_origin_positions_respects_future_horizon_and_caps_count():
    """Respect future-horizon availability and maximum-origin limits when sampling."""
    timestamps = pd.date_range("2025-12-01 00:00:00", periods=20, freq="5min")
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "avg_load": np.linspace(10.0, 29.0, len(timestamps)),
            "day_class": ["full"] * len(timestamps),
            "day_idx": [1] * 10 + [2] * 10,
        }
    )

    origins = select_origin_positions(base, start_day=1, end_day=2, lead_steps=4, max_origins=3)

    assert len(origins) == 3
    assert all(index + 4 < len(base) for index in origins)


def test_predict_model_returns_nan_for_ridge_rows_with_missing_features():
    """Emit NaN predictions when ridge inference rows are missing required features."""
    frame = build_causal_feature_frame(_base_frame(periods=24), "1min")
    train_df = frame.iloc[6:18].copy()
    trained = train_model(train_df, ["workday", "hour", "lag_1"], build_model_catalog()["ridge-medium"])

    eval_df = frame.iloc[18:20].copy()
    eval_df.loc[eval_df.index[1], "lag_1"] = np.nan
    preds = predict_model(trained, eval_df)

    assert preds.notna().sum() == 1
    assert pd.isna(preds.iloc[1])
