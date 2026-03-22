"""Bronze-to-silver transformation tests across resolutions and edge cases."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd
import pytest

from scripts.config import SCHEMAS


def test_bronze_to_silver_multi_resolution_outputs(silver_module, synthetic_bronze_df, tmp_path):
    """Ensure expected silver files are produced for multiple resolutions."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    outputs = silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1min", "5min", "10min"],
    )

    assert len(outputs) == 3
    for path in outputs:
        assert path.exists()

    silver_1m = pd.read_parquet(silver_dir / "power_load_1m.parquet")
    silver_5m = pd.read_parquet(silver_dir / "power_load_5m.parquet")
    silver_10m = pd.read_parquet(silver_dir / "power_load_10m.parquet")

    assert silver_1m.shape[0] == 2880
    assert silver_5m.shape[0] == 576
    assert silver_10m.shape[0] == 288
    assert list(silver_1m.columns) == SCHEMAS["silver"]["columns"]


def test_bronze_to_silver_avg_load_and_nan_minute(silver_module, synthetic_bronze_df, tmp_path):
    """Ensure minute averages and fully missing minutes are handled correctly."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1min"],
    )
    silver_1m = pd.read_parquet(silver_dir / "power_load_1m.parquet")

    first_minute = cast(float, silver_1m.loc[0, "avg_load"])
    assert abs(first_minute - 29.5) < 1e-9

    nan_minute = silver_1m.loc[silver_1m["timestamp"] == pd.Timestamp("2025-01-01 01:00:00"), "avg_load"]
    assert nan_minute.shape[0] == 1
    assert nan_minute.isna().iloc[0]


def test_bronze_to_silver_lag_and_rolling_warmup(silver_module, synthetic_bronze_df, tmp_path):
    """Ensure lag, rolling, and delta features follow warm-up expectations."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1min"],
    )
    silver_1m = pd.read_parquet(silver_dir / "power_load_1m.parquet")

    # A row away from the injected NaN minute.
    idx = 100
    assert silver_1m.loc[idx, "lag_1"] == silver_1m.loc[idx - 1, "avg_load"]
    assert silver_1m["rolling_mean_5"].iloc[:4].isna().all()

    # delta_5 should be lag_5 - lag_1 whenever both are present.
    valid = silver_1m[["delta_5", "lag_5", "lag_1"]].dropna().iloc[0]
    assert valid["delta_5"] == valid["lag_5"] - valid["lag_1"]
    assert {"lag_min_15", "rolling_mean_min_60", "avg_workday_baseline", "profile_activity_ratio"}.issubset(
        set(silver_1m.columns)
    )


def test_bronze_to_silver_builds_regime_features(silver_module, synthetic_bronze_df, tmp_path):
    """Ensure derived workday-profile regime features are populated causally."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1min"],
    )
    silver_1m = pd.read_parquet(silver_dir / "power_load_1m.parquet")

    assert silver_1m["prev_day_workday"].notna().all()
    assert silver_1m["next_day_workday"].notna().all()
    assert set(silver_1m["workday_transition"].dropna().unique()).issubset({0.0, 1.0})
    assert silver_1m["previous_day_load"].notna().any()
    assert silver_1m["avg_workday_baseline"].notna().any()
    assert silver_1m["anchored_workday_baseline"].notna().any()
    assert set(
        [
            "phase_minute_15m",
            "phase_progress_15m",
            "phase_boundary_dist_15m",
            "phase_boundary_flag_15m",
            "phase_sin_15m",
            "phase_cos_15m",
        ]
    ).issubset(silver_1m.columns)
    midnight = silver_1m.loc[silver_1m["timestamp"].eq(pd.Timestamp("2025-01-01 00:00:00"))].iloc[0]
    assert int(midnight["phase_minute_15m"]) == 0
    assert float(midnight["phase_progress_15m"]) == pytest.approx(0.0)
    assert float(midnight["phase_boundary_dist_15m"]) == pytest.approx(0.0)
    assert int(midnight["phase_boundary_flag_15m"]) == 1


def test_bronze_to_silver_supports_seconds_and_15min_and_alias(
    silver_module, synthetic_bronze_df, tmp_path
):
    """Ensure second-level, 15-minute, and alias resolutions are supported."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    outputs = silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1s", "15min", "60s"],
    )

    assert len(outputs) == 3
    assert (silver_dir / "power_load_1s.parquet").exists()
    assert (silver_dir / "power_load_15m.parquet").exists()
    assert (silver_dir / "power_load_1m.parquet").exists()

    silver_1s = pd.read_parquet(silver_dir / "power_load_1s.parquet")
    silver_15m = pd.read_parquet(silver_dir / "power_load_15m.parquet")
    silver_1m = pd.read_parquet(silver_dir / "power_load_1m.parquet")

    assert silver_1s.shape[0] == 2 * 24 * 60 * 60
    assert silver_15m.shape[0] == 2 * 24 * 4
    assert silver_1m.shape[0] == 2 * 24 * 60


def test_bronze_to_silver_invalid_resolution_raises(silver_module, synthetic_bronze_df, tmp_path):
    """Ensure unsupported resolutions raise a validation error."""
    bronze_path = tmp_path / "bronze.parquet"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)
    with pytest.raises(ValueError, match="Unsupported resolution"):
        silver_module.bronze_to_silver(
            bronze_path=bronze_path,
            silver_dir=tmp_path / "silver",
            resolutions=["7min"],
        )


def test_bronze_to_silver_all_nan_bronze_raises(silver_module, all_nan_bronze_df, tmp_path):
    """Ensure all-NaN bronze load input raises an error."""
    bronze_path = tmp_path / "bronze.parquet"
    all_nan_bronze_df.to_parquet(bronze_path, index=False)

    with pytest.raises(ValueError, match="no non-NaN load rows"):
        silver_module.bronze_to_silver(
            bronze_path=bronze_path,
            silver_dir=tmp_path / "silver",
            resolutions=["1min"],
        )


def test_bronze_to_silver_empty_bronze_raises(silver_module, empty_bronze_df, tmp_path):
    """Ensure empty bronze inputs are rejected."""
    bronze_path = tmp_path / "bronze.parquet"
    empty_bronze_df.to_parquet(bronze_path, index=False)

    with pytest.raises(ValueError, match="Bronze input is empty"):
        silver_module.bronze_to_silver(
            bronze_path=bronze_path,
            silver_dir=tmp_path / "silver",
            resolutions=["1min"],
        )


def test_bronze_to_silver_single_day_15min_rows(silver_module, single_day_bronze_df, tmp_path):
    """Ensure 15-minute aggregation yields 96 rows for one day."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    single_day_bronze_df.to_parquet(bronze_path, index=False)

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["15min"],
    )
    silver_15m = pd.read_parquet(silver_dir / "power_load_15m.parquet")
    assert silver_15m.shape[0] == 96


def test_bronze_to_silver_required_non_lag_columns_have_no_nulls(
    silver_module, synthetic_bronze_df, tmp_path
):
    """Ensure required non-lag silver columns remain non-null."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1min"],
    )
    silver_df = pd.read_parquet(silver_dir / "power_load_1m.parquet")
    required = SCHEMAS["silver"]["required_not_null"]
    assert int(silver_df[required].isna().sum().sum()) == 0


def test_bronze_to_silver_fourier_features_are_continuous_and_valid(
    silver_module, synthetic_bronze_df, tmp_path
):
    """Ensure Fourier features are non-null, continuous, and lie on the unit circle."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1s"],
    )
    silver_df = pd.read_parquet(silver_dir / "power_load_1s.parquet")

    for column_name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert int(silver_df[column_name].isna().sum()) == 0

    midnight = silver_df.loc[silver_df["timestamp"] == pd.Timestamp("2025-01-01 00:00:00")].iloc[0]
    assert abs(float(midnight["hour_sin"])) < 1e-12
    assert abs(float(midnight["hour_cos"]) - 1.0) < 1e-12

    quarter_day = silver_df.loc[
        silver_df["timestamp"] == pd.Timestamp("2025-01-01 06:00:00")
    ].iloc[0]
    assert abs(float(quarter_day["hour_sin"]) - 1.0) < 1e-12
    assert abs(float(quarter_day["hour_cos"])) < 1e-9

    within_hour_a = silver_df.loc[
        silver_df["timestamp"] == pd.Timestamp("2025-01-01 00:00:00")
    ].iloc[0]
    within_hour_b = silver_df.loc[
        silver_df["timestamp"] == pd.Timestamp("2025-01-01 00:00:30")
    ].iloc[0]
    assert float(within_hour_b["hour_sin"]) > float(within_hour_a["hour_sin"])
    assert float(within_hour_b["hour_cos"]) < float(within_hour_a["hour_cos"])

    hour_identity = silver_df["hour_sin"] ** 2 + silver_df["hour_cos"] ** 2
    dow_identity = silver_df["dow_sin"] ** 2 + silver_df["dow_cos"] ** 2
    assert ((hour_identity - 1.0).abs() < 1e-10).all()
    assert ((dow_identity - 1.0).abs() < 1e-10).all()


def test_bronze_to_silver_logs_quality_gate(
    silver_module, synthetic_bronze_df, tmp_path, caplog
):
    """Ensure silver stage emits a standardized quality-gate summary per resolution."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)

    with caplog.at_level(logging.INFO):
        silver_module.bronze_to_silver(
            bronze_path=bronze_path,
            silver_dir=silver_dir,
            resolutions=["1min"],
        )
    assert "SILVER QUALITY GATE: PASS" in caplog.text
    assert "resolution=1min" in caplog.text
