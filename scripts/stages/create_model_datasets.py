"""Create model-ready datasets from gold data using chronological splits."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    DEFAULT_RESOLUTIONS,
    FEATURE_SETS,
    MODEL_MIN_SPLIT_ROWS,
    PATHS,
    RESOLUTION_ALIASES,
    RESOLUTION_TO_SUFFIX,
    SPLIT_DAY_RANGES,
    SUPPORTED_RESOLUTIONS,
    TARGET_COLUMN,
    validate_config,
)
from utils import emit_quality_gate

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure default logging when no handlers are present."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _gold_input_path(resolution: str, gold_dir: Path) -> Path:
    """Build the gold input parquet path for a resolution."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    suffix = RESOLUTION_TO_SUFFIX[canonical]
    return gold_dir / f"power_load_{suffix}_all_features.parquet"


def _model_output_path(resolution: str, feature_set: str, split: str, model_dir: Path) -> Path:
    """Build the model dataset output path for one split."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    suffix = RESOLUTION_TO_SUFFIX[canonical]
    return model_dir / f"{suffix}_{feature_set}_{split}.parquet"


def _day_order_mapping(timestamps: pd.Series) -> dict[pd.Timestamp, int]:
    """Map unique normalized dates to contiguous 1-based day indices."""
    unique_dates = sorted(timestamps.dt.normalize().unique())
    return {date: idx + 1 for idx, date in enumerate(unique_dates)}


def _split_dates(day_map: dict[pd.Timestamp, int], split_name: str) -> set[pd.Timestamp]:
    """Return the set of dates assigned to a named split range."""
    start, end = SPLIT_DAY_RANGES[split_name]
    return {date for date, day_num in day_map.items() if start <= day_num <= end}


def create_model_datasets(
    gold_dir: Path | None = None,
    model_dir: Path | None = None,
    resolutions: Iterable[str] | None = None,
    feature_sets: Iterable[str] | None = None,
) -> list[Path]:
    """Create split datasets for each configured resolution and feature set."""
    _configure_logging()
    validate_config()

    input_dir = gold_dir or PATHS["gold_dir"]
    output_dir = model_dir or PATHS["model_dir"]
    target_resolutions = (
        list(resolutions) if resolutions is not None else list(DEFAULT_RESOLUTIONS)
    )
    target_feature_sets = list(feature_sets) if feature_sets is not None else list(FEATURE_SETS)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Creating model datasets. resolutions=%s feature_sets=%s",
        target_resolutions,
        target_feature_sets,
    )
    logger.info(
        "Chronological split uses day-order ranges: train=%s validate=%s test=%s",
        SPLIT_DAY_RANGES["train"],
        SPLIT_DAY_RANGES["validate"],
        SPLIT_DAY_RANGES["test"],
    )

    outputs: list[Path] = []
    split_row_counts: list[int] = []

    for resolution in target_resolutions:
        canonical = RESOLUTION_ALIASES.get(resolution, resolution)
        if canonical not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
                f"(aliases: {sorted(RESOLUTION_ALIASES)})"
            )

        input_path = _gold_input_path(resolution, input_dir)
        if not input_path.exists():
            raise ValueError(f"Gold input file not found: {input_path}")

        try:
            gold = pd.read_parquet(input_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ValueError(f"Failed to read gold input at {input_path}: {exc}") from exc
        gold = gold.sort_values("timestamp").reset_index(drop=True)
        gold["timestamp"] = pd.to_datetime(gold["timestamp"], errors="raise")

        day_map = _day_order_mapping(gold["timestamp"])
        date_sets = {name: _split_dates(day_map, name) for name in SPLIT_DAY_RANGES}

        if date_sets["train"] & date_sets["validate"]:
            raise ValueError("Train and validate date overlap detected.")
        if date_sets["train"] & date_sets["test"]:
            raise ValueError("Train and test date overlap detected.")
        if date_sets["validate"] & date_sets["test"]:
            raise ValueError("Validate and test date overlap detected.")
        if date_sets["train"] and date_sets["validate"]:
            if max(date_sets["train"]) >= min(date_sets["validate"]):
                raise ValueError("Chronological order violation: train dates overlap or exceed validate.")
        if date_sets["validate"] and date_sets["test"]:
            if max(date_sets["validate"]) >= min(date_sets["test"]):
                raise ValueError("Chronological order violation: validate dates overlap or exceed test.")

        logger.info(
            "Resolution=%s unique_days=%d train_days=%d validate_days=%d test_days=%d",
            resolution,
            len(day_map),
            len(date_sets["train"]),
            len(date_sets["validate"]),
            len(date_sets["test"]),
        )

        for feature_set_name in target_feature_sets:
            if feature_set_name not in FEATURE_SETS:
                raise ValueError(f"Unknown feature set: {feature_set_name}")
            feature_columns = FEATURE_SETS[feature_set_name]

            missing_columns = [col for col in feature_columns if col not in gold.columns]
            if missing_columns:
                raise ValueError(
                    f"Feature set '{feature_set_name}' has missing columns in gold ({resolution}): "
                    f"{missing_columns}"
                )
            if TARGET_COLUMN in feature_columns:
                raise ValueError(
                    f"Feature set '{feature_set_name}' must not include target column '{TARGET_COLUMN}'."
                )

            select_columns = ["timestamp", "day_class", TARGET_COLUMN] + feature_columns
            select_columns = list(dict.fromkeys(select_columns))

            for split_name, split_dates in date_sets.items():
                split_df = gold[gold["timestamp"].dt.normalize().isin(split_dates)][select_columns].copy()
                split_df = split_df.sort_values("timestamp").reset_index(drop=True)

                rows_before_dropna = split_df.shape[0]
                split_df = split_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
                rows_after_dropna = split_df.shape[0]

                output_path = _model_output_path(
                    resolution=resolution,
                    feature_set=feature_set_name,
                    split=split_name,
                    model_dir=output_dir,
                )
                try:
                    split_df.to_parquet(output_path, index=False)
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to write model split parquet at {output_path}: {exc}"
                    ) from exc

                date_min = split_df["timestamp"].min() if not split_df.empty else None
                date_max = split_df["timestamp"].max() if not split_df.empty else None
                null_rate = (
                    split_df.isna().sum().sum() / (split_df.shape[0] * split_df.shape[1])
                    if split_df.shape[0] > 0 and split_df.shape[1] > 0
                    else 0.0
                )
                feature_null_rates = {
                    col: float(split_df[col].isna().mean()) for col in feature_columns if split_df.shape[0] > 0
                }

                if TARGET_COLUMN in split_df.columns and not split_df.empty:
                    stats = split_df[TARGET_COLUMN].describe()
                    target_summary = {
                        "mean": float(stats["mean"]),
                        "std": float(stats["std"]),
                        "min": float(stats["min"]),
                        "max": float(stats["max"]),
                    }
                else:
                    target_summary = {}

                logger.info(
                    "Wrote model split: res=%s set=%s split=%s rows=%d rows_before_dropna=%d "
                    "date_min=%s date_max=%s null_rate=%.6f feature_null_rates=%s "
                    "target_stats=%s path=%s",
                    resolution,
                    feature_set_name,
                    split_name,
                    rows_after_dropna,
                    rows_before_dropna,
                    date_min,
                    date_max,
                    null_rate,
                    feature_null_rates,
                    target_summary,
                    output_path,
                )
                split_row_counts.append(rows_after_dropna)
                outputs.append(output_path)

    expected_output_count = (
        len(target_resolutions) * len(target_feature_sets) * len(SPLIT_DAY_RANGES)
    )
    empty_split_count = sum(1 for row_count in split_row_counts if row_count < MODEL_MIN_SPLIT_ROWS)
    emit_quality_gate(
        "MODEL DATASETS GATE",
        len(outputs) == expected_output_count and empty_split_count == 0,
        details={
            "expected_files": expected_output_count,
            "written_files": len(outputs),
            "empty_splits": empty_split_count,
            "min_split_rows": MODEL_MIN_SPLIT_ROWS,
        },
        logger_instance=logger,
    )
    return outputs


if __name__ == "__main__":
    create_model_datasets()
