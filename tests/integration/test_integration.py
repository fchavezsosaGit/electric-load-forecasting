"""End-to-end integration tests for deterministic pipeline outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import SCHEMAS


def _synthetic_raw_dict(day_count: int = 3) -> dict[str, object]:
    """Build synthetic raw MATLAB-like inputs for integration testing."""
    p_data = np.tile(np.arange(86400, dtype=float).reshape(-1, 1), (1, day_count))
    p_data[120:180, 0] = np.nan
    day_data = np.array(
        [[f"2025-01-{day:02d}" for day in range(1, day_count + 1)]],
        dtype=object,
    )
    classes = ["full", "half", "none"][:day_count]
    day_class = np.array([classes], dtype=object)
    return {"P_data": p_data, "day_data": day_data, "day_class": day_class}


def test_end_to_end_pipeline_deterministic_and_ordered(
    raw_module, silver_module, gold_module, model_dataset_module, tmp_path, monkeypatch
):
    """Validate pipeline determinism, schema integrity, and split ordering end-to-end."""
    input_path = tmp_path / "mock.mat"
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"
    model_dir = tmp_path / "model"
    input_path.touch()

    monkeypatch.setattr(raw_module, "loadmat", lambda _: _synthetic_raw_dict(day_count=3))

    raw_module.raw_to_bronze(raw_path=input_path, output_path=bronze_path)
    bronze_df = pd.read_parquet(bronze_path)
    assert bronze_df.shape[0] == 86400 * 3
    assert list(bronze_df.columns) == SCHEMAS["bronze"]["columns"]

    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=["1min", "5min"],
    )
    silver_1m = pd.read_parquet(silver_dir / "power_load_1m.parquet")
    silver_5m = pd.read_parquet(silver_dir / "power_load_5m.parquet")
    assert list(silver_1m.columns) == SCHEMAS["silver"]["columns"]
    assert silver_1m.shape[0] == 3 * 24 * 60
    assert silver_5m.shape[0] == 3 * 24 * 12

    gold_module.silver_to_gold(
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        resolutions=["1min", "5min"],
    )
    gold_1m_path = gold_dir / "power_load_1m_all_features.parquet"
    gold_5m_path = gold_dir / "power_load_5m_all_features.parquet"
    gold_1m = pd.read_parquet(gold_1m_path)
    gold_5m = pd.read_parquet(gold_5m_path)
    assert list(gold_1m.columns) == SCHEMAS["gold"]["columns"]
    assert gold_1m.shape[0] <= silver_1m.shape[0]
    assert gold_5m.shape[0] <= silver_5m.shape[0]

    first_gold_bytes = gold_1m_path.read_bytes()
    gold_module.silver_to_gold(
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        resolutions=["1min"],
    )
    second_gold_bytes = gold_1m_path.read_bytes()
    assert first_gold_bytes == second_gold_bytes

    monkeypatch.setattr(
        model_dataset_module,
        "SPLIT_DAY_RANGES",
        {"train": (1, 1), "validate": (2, 2), "test": (3, 3)},
    )
    model_dataset_module.create_model_datasets(
        gold_dir=gold_dir,
        model_dir=model_dir,
        resolutions=["1min"],
        feature_sets=["minimal"],
    )

    train = pd.read_parquet(model_dir / "1m_minimal_train.parquet")
    validate = pd.read_parquet(model_dir / "1m_minimal_validate.parquet")
    test = pd.read_parquet(model_dir / "1m_minimal_test.parquet")
    assert train["timestamp"].max() < validate["timestamp"].min()
    assert validate["timestamp"].max() < test["timestamp"].min()

    first_model_bytes = (model_dir / "1m_minimal_train.parquet").read_bytes()
    model_dataset_module.create_model_datasets(
        gold_dir=gold_dir,
        model_dir=model_dir,
        resolutions=["1min"],
        feature_sets=["minimal"],
    )
    second_model_bytes = (model_dir / "1m_minimal_train.parquet").read_bytes()
    assert first_model_bytes == second_model_bytes
