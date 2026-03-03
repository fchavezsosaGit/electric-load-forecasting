"""Shared pytest fixtures for loading modules and synthetic datasets."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename: str, module_name: str):
    """Load a script module from the scripts directory."""
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def raw_module():
    """Provide the raw-to-bronze script module."""
    return _load_module("000_raw_to_bronze.py", "test_raw_to_bronze")


@pytest.fixture(scope="session")
def silver_module():
    """Provide the bronze-to-silver script module."""
    return _load_module("001_bronze_to_silver.py", "test_bronze_to_silver")


@pytest.fixture(scope="session")
def config_module():
    """Provide the shared configuration module."""
    return _load_module("config.py", "test_config")


@pytest.fixture(scope="session")
def gold_module():
    """Provide the silver-to-gold script module."""
    return _load_module("002_silver_to_gold.py", "test_silver_to_gold")


@pytest.fixture(scope="session")
def model_dataset_module():
    """Provide the model dataset creation script module."""
    return _load_module("003_create_model_datasets.py", "test_model_datasets")


@pytest.fixture(scope="session")
def pipeline_module():
    """Provide a reloaded pipeline orchestrator module."""
    module = importlib.import_module("run_pipeline")
    return importlib.reload(module)


@pytest.fixture()
def synthetic_raw_dict():
    """Create minimal raw MATLAB-like structures for ingestion tests."""
    day_count = 2
    p_data = np.tile(np.arange(86400, dtype=float).reshape(-1, 1), (1, day_count))
    p_data[0, 0] = np.nan
    day_data = np.array([["2025-01-01", "2025-01-02"]], dtype=object)
    day_class = np.array([["full", "none"]], dtype=object)
    return {"P_data": p_data, "day_data": day_data, "day_class": day_class}


@pytest.fixture()
def synthetic_bronze_df():
    """Create a two-day second-level bronze dataframe with one NaN minute."""
    start = pd.Timestamp("2025-01-01 00:00:00")
    periods = 2 * 24 * 60 * 60
    timestamps = pd.date_range(start=start, periods=periods, freq="s")

    load = np.arange(periods, dtype=float)
    nan_start = 60 * 60  # 01:00:00 of day 1
    load[nan_start : nan_start + 60] = np.nan

    day_class = np.where(timestamps.normalize() == pd.Timestamp("2025-01-01"), "full", "none")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "day_class": day_class,
            "load": load,
        }
    )


@pytest.fixture()
def all_nan_bronze_df():
    """Create bronze data where load is entirely NaN."""
    timestamps = pd.date_range("2025-01-01", periods=120, freq="s")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "day_class": ["full"] * len(timestamps),
            "load": [np.nan] * len(timestamps),
        }
    )


@pytest.fixture()
def single_row_bronze_df():
    """Create a one-row bronze dataframe for edge-case tests."""
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01 00:00:00")],
            "day_class": ["full"],
            "load": [123.0],
        }
    )


@pytest.fixture()
def single_day_bronze_df():
    """Create one full day of second-level bronze observations."""
    timestamps = pd.date_range("2025-01-01", periods=86400, freq="s")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "day_class": ["full"] * len(timestamps),
            "load": np.linspace(1000.0, 2000.0, len(timestamps)),
        }
    )


@pytest.fixture()
def empty_bronze_df():
    """Create an empty bronze dataframe with expected dtypes."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "day_class": pd.Series(dtype="object"),
            "load": pd.Series(dtype="float64"),
        }
    )

