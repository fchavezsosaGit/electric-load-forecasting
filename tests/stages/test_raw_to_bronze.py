"""Raw-to-bronze ingestion tests for schema checks and failure paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_raw_to_bronze_schema_and_row_count(raw_module, synthetic_raw_dict, tmp_path, monkeypatch):
    """Ensure raw ingestion produces expected schema, rows, and ordering."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    monkeypatch.setattr(raw_module, "loadmat", lambda _: synthetic_raw_dict)
    result = raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)

    assert result == output_path
    assert output_path.exists()

    df = pd.read_parquet(output_path)
    assert list(df.columns) == ["timestamp", "day_class", "load"]
    assert df.shape[0] == 86400 * 2
    assert df["timestamp"].is_monotonic_increasing
    assert set(df["day_class"].unique()) <= {"full", "half", "none"}
    assert int(df["load"].isna().sum()) == 1


def test_raw_to_bronze_missing_p_data_raises(raw_module, synthetic_raw_dict, tmp_path, monkeypatch):
    """Ensure missing P_data key fails fast."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data.pop("P_data")
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    with pytest.raises(ValueError, match="Key 'P_data' not found"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_wrong_shape_raises(raw_module, synthetic_raw_dict, tmp_path, monkeypatch):
    """Ensure invalid P_data row count is rejected."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data["P_data"] = bad_data["P_data"][:-1, :]
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    with pytest.raises(ValueError, match="Expected 86400 rows"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_missing_input_path_raises(raw_module, tmp_path):
    """Ensure missing raw file path raises a clear error."""
    missing_path = tmp_path / "does_not_exist.mat"
    output_path = tmp_path / "bronze.parquet"

    with pytest.raises(ValueError, match="Raw data file not found"):
        raw_module.raw_to_bronze(raw_path=missing_path, output_path=output_path)


def test_raw_to_bronze_unexpected_day_class_raises(
    raw_module, synthetic_raw_dict, tmp_path, monkeypatch
):
    """Ensure unexpected day_class values are rejected."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data["day_class"] = bad_data["day_class"].copy()
    bad_data["day_class"][0, 1] = "holiday"
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    with pytest.raises(ValueError, match="Unexpected day_class values"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_loadmat_failure_raises_runtime(raw_module, tmp_path, monkeypatch):
    """Ensure MATLAB load failures are wrapped as runtime errors."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    def _raise(_path):
        raise ValueError("corrupt file")

    monkeypatch.setattr(raw_module, "loadmat", _raise)
    with pytest.raises(RuntimeError, match="Could not load MATLAB file"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_replaces_infinity(raw_module, synthetic_raw_dict, tmp_path, monkeypatch):
    """Ensure infinity load values are replaced before writing bronze."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data["P_data"] = bad_data["P_data"].copy()
    bad_data["P_data"][1, 0] = np.inf
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)
    df = pd.read_parquet(output_path)
    assert np.isinf(df["load"]).sum() == 0
    assert df["load"].isna().sum() >= 2


def test_raw_to_bronze_duplicate_day_dates_raises(
    raw_module, synthetic_raw_dict, tmp_path, monkeypatch
):
    """Ensure duplicate day_data dates are rejected."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data["day_data"] = np.array([["2025-01-01", "2025-01-01"]], dtype=object)
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    with pytest.raises(ValueError, match="Duplicate dates found in day_data"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_non_numeric_p_data_raises(raw_module, synthetic_raw_dict, tmp_path, monkeypatch):
    """Ensure non-numeric P_data values are rejected."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data["P_data"] = np.array([["bad", "data"]] * 86400, dtype=object)
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    with pytest.raises(ValueError, match="P_data must be numeric"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_zero_day_columns_raises(raw_module, tmp_path, monkeypatch):
    """Ensure zero-day-column P_data inputs are rejected."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = {
        "P_data": np.empty((86400, 0), dtype=float),
        "day_data": np.array([[]], dtype=object),
        "day_class": np.array([[]], dtype=object),
    }
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    with pytest.raises(ValueError, match="zero columns"):
        raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)


def test_raw_to_bronze_all_nan_day_logs_warning(raw_module, synthetic_raw_dict, tmp_path, monkeypatch, caplog):
    """Ensure all-NaN day columns emit a warning."""
    input_path = tmp_path / "mock.mat"
    output_path = tmp_path / "bronze.parquet"
    input_path.touch()

    bad_data = dict(synthetic_raw_dict)
    bad_data["P_data"] = bad_data["P_data"].copy()
    bad_data["P_data"][:, 1] = np.nan
    monkeypatch.setattr(raw_module, "loadmat", lambda _: bad_data)

    raw_module.raw_to_bronze(raw_path=input_path, output_path=output_path)
    assert "all-NaN load values" in caplog.text
