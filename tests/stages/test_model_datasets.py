"""Model dataset split tests for chronological integrity and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.config import FEATURE_SETS, SCHEMAS, SPLIT_DAY_RANGES, TARGET_COLUMN


def _build_full_gold_df() -> pd.DataFrame:
    """Build a complete synthetic gold dataframe for split testing."""
    timestamps = pd.date_range("2025-11-28", periods=31, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "avg_load": np.linspace(100.0, 130.0, len(timestamps)),
            "day_class": ["full"] * len(timestamps),
            "workday": [2] * len(timestamps),
            "year": timestamps.year,
            "quarter": timestamps.quarter,
            "month": timestamps.month,
            "day": timestamps.day,
            "day_of_week": ((timestamps.dayofweek + 1) % 7).astype(int),
            "hour": [12] * len(timestamps),
            "season": [1] * len(timestamps),
            "time_of_day": [1] * len(timestamps),
        }
    )

    for column in SCHEMAS["gold"]["columns"]:
        if column in df.columns:
            continue
        df[column] = np.linspace(1.0, float(len(timestamps)), len(timestamps))

    return df[SCHEMAS["gold"]["columns"]]


def _write_gold(gold_dir: Path, suffix: str, df: pd.DataFrame) -> Path:
    """Write a synthetic gold parquet file and return its path."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / f"power_load_{suffix}_all_features.parquet"
    df.to_parquet(path, index=False)
    return path


def test_create_model_datasets_chronological_split_and_target(
    model_dataset_module, tmp_path
):
    """Ensure train/validate/test splits are chronological and include target/features."""
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "model"
    _write_gold(gold_dir, "1m", _build_full_gold_df())

    outputs = model_dataset_module.create_model_datasets(
        gold_dir=gold_dir,
        model_dir=model_dir,
        resolutions=["1min"],
        feature_sets=["minimal"],
    )

    assert len(outputs) == 3
    train = pd.read_parquet(model_dir / "1m_minimal_train.parquet")
    validate = pd.read_parquet(model_dir / "1m_minimal_validate.parquet")
    test = pd.read_parquet(model_dir / "1m_minimal_test.parquet")

    for split_df in (train, validate, test):
        assert TARGET_COLUMN in split_df.columns
        assert "workday" in split_df.columns
        assert "hour" in split_df.columns
        assert "lag_1" in split_df.columns

    assert train["timestamp"].max() < validate["timestamp"].min()
    assert validate["timestamp"].max() < test["timestamp"].min()
    assert set(train["timestamp"].dt.date) & set(validate["timestamp"].dt.date) == set()
    assert set(train["timestamp"].dt.date) & set(test["timestamp"].dt.date) == set()
    assert set(validate["timestamp"].dt.date) & set(test["timestamp"].dt.date) == set()


def test_create_model_datasets_missing_gold_input_raises(model_dataset_module, tmp_path):
    """Ensure missing gold inputs raise an error."""
    with pytest.raises(ValueError, match="Gold input file not found"):
        model_dataset_module.create_model_datasets(
            gold_dir=tmp_path / "missing",
            model_dir=tmp_path / "model",
            resolutions=["1min"],
            feature_sets=["minimal"],
        )


def test_create_model_datasets_unknown_feature_set_raises(model_dataset_module, tmp_path):
    """Ensure unknown feature set names are rejected."""
    gold_dir = tmp_path / "gold"
    _write_gold(gold_dir, "1m", _build_full_gold_df())

    with pytest.raises(ValueError, match="Unknown feature set"):
        model_dataset_module.create_model_datasets(
            gold_dir=gold_dir,
            model_dir=tmp_path / "model",
            resolutions=["1min"],
            feature_sets=["does_not_exist"],
        )


def test_create_model_datasets_missing_feature_column_raises(model_dataset_module, tmp_path):
    """Ensure missing feature columns in gold raise an error."""
    gold_dir = tmp_path / "gold"
    bad_df = _build_full_gold_df().drop(columns=["lag_1"])
    _write_gold(gold_dir, "1m", bad_df)

    with pytest.raises(ValueError, match="missing columns in gold"):
        model_dataset_module.create_model_datasets(
            gold_dir=gold_dir,
            model_dir=tmp_path / "model",
            resolutions=["1min"],
            feature_sets=["minimal"],
        )


def test_create_model_datasets_feature_set_with_target_raises(model_dataset_module, tmp_path):
    """Ensure feature sets containing the target column are rejected."""
    gold_dir = tmp_path / "gold"
    _write_gold(gold_dir, "1m", _build_full_gold_df())

    model_dataset_module.FEATURE_SETS["bad_set"] = ["hour", TARGET_COLUMN]
    try:
        with pytest.raises(ValueError, match="must not include target column"):
            model_dataset_module.create_model_datasets(
                gold_dir=gold_dir,
                model_dir=tmp_path / "model",
                resolutions=["1min"],
                feature_sets=["bad_set"],
            )
    finally:
        model_dataset_module.FEATURE_SETS.pop("bad_set", None)


def test_create_model_datasets_all_feature_sets(model_dataset_module, tmp_path):
    """Ensure datasets can be produced for every configured feature set."""
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "model"
    _write_gold(gold_dir, "1m", _build_full_gold_df())

    outputs = model_dataset_module.create_model_datasets(
        gold_dir=gold_dir,
        model_dir=model_dir,
        resolutions=["1min"],
        feature_sets=list(FEATURE_SETS),
    )

    assert len(outputs) == 3 * len(FEATURE_SETS)
    for feature_set_name, columns in FEATURE_SETS.items():
        train_path = model_dir / f"1m_{feature_set_name}_train.parquet"
        assert train_path.exists()
        train_df = pd.read_parquet(train_path)
        expected_columns = {"timestamp", "day_class", TARGET_COLUMN, *columns}
        assert set(train_df.columns) == expected_columns


def test_create_model_datasets_multi_resolution_outputs(model_dataset_module, tmp_path):
    """Ensure model datasets are generated for multiple resolutions."""
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "model"
    gold_df = _build_full_gold_df()
    _write_gold(gold_dir, "1m", gold_df)
    _write_gold(gold_dir, "5m", gold_df)

    outputs = model_dataset_module.create_model_datasets(
        gold_dir=gold_dir,
        model_dir=model_dir,
        resolutions=["1min", "5min"],
        feature_sets=["minimal"],
    )
    assert len(outputs) == 6
    assert (model_dir / "1m_minimal_train.parquet").exists()
    assert (model_dir / "5m_minimal_train.parquet").exists()


def test_create_model_datasets_split_ranges_match_config(model_dataset_module, tmp_path):
    """Ensure split day counts match configured split ranges."""
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "model"
    _write_gold(gold_dir, "1m", _build_full_gold_df())

    model_dataset_module.create_model_datasets(
        gold_dir=gold_dir,
        model_dir=model_dir,
        resolutions=["1min"],
        feature_sets=["minimal"],
    )

    split_to_days = {}
    for split_name in SPLIT_DAY_RANGES:
        split_df = pd.read_parquet(model_dir / f"1m_minimal_{split_name}.parquet")
        unique_dates = sorted(pd.to_datetime(split_df["timestamp"]).dt.date.unique())
        split_to_days[split_name] = len(unique_dates)

    assert split_to_days["train"] == 25
    assert split_to_days["validate"] == 3
    assert split_to_days["test"] == 3


def test_create_model_datasets_logs_quality_gate(model_dataset_module, tmp_path, caplog):
    """Ensure model dataset generation emits the standardized completion gate."""
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "model"
    _write_gold(gold_dir, "1m", _build_full_gold_df())

    with caplog.at_level(logging.INFO):
        model_dataset_module.create_model_datasets(
            gold_dir=gold_dir,
            model_dir=model_dir,
            resolutions=["1min"],
            feature_sets=["minimal"],
        )

    assert "MODEL DATASETS GATE: PASS" in caplog.text
    assert "written_files=3" in caplog.text
