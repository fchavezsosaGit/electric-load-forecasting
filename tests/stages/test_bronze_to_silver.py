"""Bronze-to-silver transformation tests across resolutions and edge cases."""

from __future__ import annotations

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
