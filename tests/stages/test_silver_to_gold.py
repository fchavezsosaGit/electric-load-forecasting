"""Silver-to-gold transformation tests for validation and determinism."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.config import SCHEMAS


def _build_silver_inputs(
    silver_module, synthetic_bronze_df: pd.DataFrame, tmp_path: Path, resolutions: list[str]
) -> tuple[Path, Path]:
    """Generate temporary silver inputs for downstream gold tests."""
    bronze_path = tmp_path / "bronze.parquet"
    silver_dir = tmp_path / "silver"
    synthetic_bronze_df.to_parquet(bronze_path, index=False)
    silver_module.bronze_to_silver(
        bronze_path=bronze_path,
        silver_dir=silver_dir,
        resolutions=resolutions,
    )
    return bronze_path, silver_dir


def test_silver_to_gold_generates_output(gold_module, silver_module, synthetic_bronze_df, tmp_path):
    """Ensure gold output is generated with required schema and non-null columns."""
    _, silver_dir = _build_silver_inputs(silver_module, synthetic_bronze_df, tmp_path, ["1min"])
    gold_dir = tmp_path / "gold"
    outputs = gold_module.silver_to_gold(
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        resolutions=["1min"],
    )

    assert len(outputs) == 1
    output_path = gold_dir / "power_load_1m_all_features.parquet"
    assert output_path.exists()

    silver_df = pd.read_parquet(silver_dir / "power_load_1m.parquet")
    gold_df = pd.read_parquet(output_path)

    assert list(gold_df.columns) == SCHEMAS["gold"]["columns"]
    assert gold_df.shape[0] < silver_df.shape[0]
    required = SCHEMAS["gold"]["required_not_null"]
    assert int(gold_df[required].isna().sum().sum()) == 0


def test_silver_to_gold_missing_input_raises(gold_module, tmp_path):
    """Ensure missing silver input files raise an error."""
    with pytest.raises(ValueError, match="Missing silver input"):
        gold_module.silver_to_gold(
            silver_dir=tmp_path / "missing_silver",
            gold_dir=tmp_path / "gold",
            resolutions=["1min"],
        )


def test_silver_to_gold_invalid_resolution_raises(gold_module, tmp_path):
    """Ensure unsupported resolutions are rejected."""
    with pytest.raises(ValueError, match="Unsupported resolution"):
        gold_module.silver_to_gold(
            silver_dir=tmp_path / "silver",
            gold_dir=tmp_path / "gold",
            resolutions=["2min"],
        )


def test_silver_to_gold_schema_mismatch_raises(gold_module, tmp_path):
    """Ensure schema mismatches in silver input fail validation."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    bad_df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01")], "avg_load": [1.0]})
    bad_df.to_parquet(silver_dir / "power_load_1m.parquet", index=False)

    with pytest.raises(ValueError, match="Silver schema mismatch"):
        gold_module.silver_to_gold(
            silver_dir=silver_dir,
            gold_dir=tmp_path / "gold",
            resolutions=["1min"],
        )


def test_silver_to_gold_unexpected_day_class_raises(
    gold_module, silver_module, synthetic_bronze_df, tmp_path
):
    """Ensure unexpected day_class values in silver are rejected."""
    _, silver_dir = _build_silver_inputs(silver_module, synthetic_bronze_df, tmp_path, ["1min"])
    silver_path = silver_dir / "power_load_1m.parquet"
    silver_df = pd.read_parquet(silver_path)
    silver_df.loc[0, "day_class"] = "holiday"
    silver_df.to_parquet(silver_path, index=False)

    with pytest.raises(ValueError, match="Unexpected day_class values"):
        gold_module.silver_to_gold(
            silver_dir=silver_dir,
            gold_dir=tmp_path / "gold",
            resolutions=["1min"],
        )


def test_silver_to_gold_preserves_lag_nan(gold_module, silver_module, synthetic_bronze_df, tmp_path):
    """Ensure warm-up NaNs remain present in lag and rolling features."""
    _, silver_dir = _build_silver_inputs(silver_module, synthetic_bronze_df, tmp_path, ["1min"])
    gold_dir = tmp_path / "gold"
    gold_module.silver_to_gold(silver_dir=silver_dir, gold_dir=gold_dir, resolutions=["1min"])

    gold_df = pd.read_parquet(gold_dir / "power_load_1m_all_features.parquet")
    assert gold_df["lag_1440"].isna().any()
    assert gold_df["rolling_mean_1440"].isna().any()


def test_silver_to_gold_empty_silver_raises(gold_module, tmp_path):
    """Ensure empty silver inputs raise an error."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)
    empty_df = pd.DataFrame(columns=SCHEMAS["silver"]["columns"])
    empty_df.to_parquet(silver_dir / "power_load_1m.parquet", index=False)

    with pytest.raises(ValueError, match="zero rows"):
        gold_module.silver_to_gold(
            silver_dir=silver_dir,
            gold_dir=tmp_path / "gold",
            resolutions=["1min"],
        )


def test_silver_to_gold_all_rows_dropped_logs_critical(
    gold_module, silver_module, synthetic_bronze_df, tmp_path, caplog
):
    """Ensure all-row drops are logged as critical and output remains valid."""
    _, silver_dir = _build_silver_inputs(silver_module, synthetic_bronze_df, tmp_path, ["1min"])
    silver_path = silver_dir / "power_load_1m.parquet"
    silver_df = pd.read_parquet(silver_path)
    silver_df["avg_load"] = pd.NA
    silver_df.to_parquet(silver_path, index=False)

    gold_dir = tmp_path / "gold"
    gold_module.silver_to_gold(silver_dir=silver_dir, gold_dir=gold_dir, resolutions=["1min"])
    gold_df = pd.read_parquet(gold_dir / "power_load_1m_all_features.parquet")

    assert gold_df.empty
    assert "Gold output is empty" in caplog.text


def test_silver_to_gold_deterministic_output(gold_module, silver_module, synthetic_bronze_df, tmp_path):
    """Ensure repeated gold generation is byte-for-byte deterministic."""
    _, silver_dir = _build_silver_inputs(silver_module, synthetic_bronze_df, tmp_path, ["1min"])
    gold_dir = tmp_path / "gold"

    gold_module.silver_to_gold(silver_dir=silver_dir, gold_dir=gold_dir, resolutions=["1min"])
    first_bytes = (gold_dir / "power_load_1m_all_features.parquet").read_bytes()

    gold_module.silver_to_gold(silver_dir=silver_dir, gold_dir=gold_dir, resolutions=["1min"])
    second_bytes = (gold_dir / "power_load_1m_all_features.parquet").read_bytes()

    assert first_bytes == second_bytes


def test_silver_to_gold_multi_resolution_outputs(
    gold_module, silver_module, synthetic_bronze_df, tmp_path
):
    """Ensure gold outputs are generated for multiple resolutions."""
    _, silver_dir = _build_silver_inputs(
        silver_module,
        synthetic_bronze_df,
        tmp_path,
        ["1min", "5min"],
    )
    gold_dir = tmp_path / "gold"
    outputs = gold_module.silver_to_gold(
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        resolutions=["1min", "5min"],
    )
    assert len(outputs) == 2
    assert (gold_dir / "power_load_1m_all_features.parquet").exists()
    assert (gold_dir / "power_load_5m_all_features.parquet").exists()
