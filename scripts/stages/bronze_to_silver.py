"""Bronze-to-silver ingestion and feature engineering.

Input:
- Bronze parquet with columns: timestamp, day_class, load

Outputs:
- Silver parquet files for each configured resolution (defaults: 1min, 5min, 10min, 15min)
- Includes temporal, business, Fourier, lag, rolling, delta, and slope features
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    DEFAULT_RESOLUTIONS,
    PATHS,
    RESOLUTION_ALIASES,
    RESOLUTION_TO_SUFFIX,
    SCHEMAS,
    SILVER_NAN_DROP_FAIL_PCT,
    SILVER_NAN_DROP_WARN_PCT,
    SUPPORTED_RESOLUTIONS,
    VALID_DAY_CLASSES,
)
from modeling.feature_engineering import (
    add_calendar_context,
    add_phase_context,
    add_period_history_features,
    add_profile_regime_features,
    add_time_normalized_features,
)
from utils import (
    emit_quality_gate,
    validate_schema_columns,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure default logging when no handlers are present."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _validate_resolution(resolution: str) -> None:
    """Validate a resolution string, including supported aliases."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    if canonical not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. Supported: {SUPPORTED_RESOLUTIONS} "
            f"(aliases: {sorted(RESOLUTION_ALIASES)})"
        )


def _silver_output_path(resolution: str, silver_dir: Path) -> Path:
    """Build the silver output parquet path for a resolution."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    suffix = RESOLUTION_TO_SUFFIX[canonical]
    return silver_dir / f"power_load_{suffix}.parquet"


def _validate_silver_not_null(df: pd.DataFrame) -> None:
    """Enforce non-null constraints for required silver columns."""
    required = cast(list[str], SCHEMAS["silver"]["required_not_null"])
    null_counts = {col: int(df[col].isna().sum()) for col in required if df[col].isna().any()}
    if null_counts:
        raise ValueError(
            "Unexpected null values in required silver columns: "
            f"{null_counts}. Investigate resample/day_class mapping."
        )


def _feature_columns_for_resolution(resampled: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Create temporal, business, Fourier, lag, rolling, delta, and slope features."""
    silver = resampled.copy()

    if silver["day_class"].isna().any():
        raise ValueError(
            f"Resolution={resolution} contains NaN day_class values after resampling. "
            "Investigate bronze input completeness."
        )
    silver = add_calendar_context(silver)
    silver = add_phase_context(silver, resolution=resolution)
    silver = add_period_history_features(silver, resolution=resolution)
    silver = add_time_normalized_features(silver, resolution=resolution)
    silver = add_profile_regime_features(silver, resolution=resolution)
    silver = silver.drop(columns=["slot_label"], errors="ignore")
    return silver


def _build_silver_for_resolution(
    bronze: pd.DataFrame, resolution: str, source_path: Path
) -> pd.DataFrame:
    """Build and validate one silver dataset from bronze input for a resolution."""
    _validate_resolution(resolution)
    canonical_resolution = RESOLUTION_ALIASES.get(resolution, resolution)

    if bronze.empty:
        raise ValueError(f"Bronze input is empty: {source_path}")

    base = bronze.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="raise")
    base = base.sort_values("timestamp").set_index("timestamp")
    if not base.index.is_monotonic_increasing:
        raise ValueError(
            f"Bronze timestamps are not monotonic after sorting for resolution={resolution}."
        )

    total_rows = base.shape[0]
    if base["load"].notna().sum() == 0:
        raise ValueError(
            f"Bronze input has no non-NaN load rows for resolution={resolution}: {source_path}"
        )

    clean = base[base["load"].notna()]
    if clean.empty:
        raise ValueError(
            f"All bronze load rows became empty after NaN filtering for resolution={resolution}: "
            f"{source_path}"
        )

    dropped_rows = total_rows - clean.shape[0]
    dropped_pct = (dropped_rows / total_rows) * 100.0 if total_rows else 0.0
    logger.info(
        "Resolution=%s dropped NaN load rows: count=%d, pct=%.2f%%",
        resolution,
        dropped_rows,
        dropped_pct,
    )
    if dropped_pct > SILVER_NAN_DROP_WARN_PCT:
        logger.warning(
            "Resolution=%s dropped NaN percentage (%.2f%%) exceeds configured %.2f%% threshold",
            resolution,
            dropped_pct,
            SILVER_NAN_DROP_WARN_PCT,
        )

    resampled = pd.concat(
        [
            clean["load"]
            .resample(canonical_resolution, closed="left", label="left")
            .mean()
            .rename("avg_load"),
            base["day_class"]
            .resample(canonical_resolution, closed="left", label="left")
            .first()
            .rename("day_class"),
        ],
        axis=1,
    )
    if not resampled.index.is_monotonic_increasing:
        raise ValueError(f"Resampled index is not monotonic for resolution={resolution}")

    if resampled.empty:
        raise ValueError(f"Resampling produced zero rows for resolution={resolution}: {source_path}")

    inf_count = int(np.isinf(resampled["avg_load"]).sum())
    if inf_count:
        logger.warning(
            "Resolution=%s replacing %d +/-inf values in avg_load with NaN after resampling",
            resolution,
            inf_count,
        )
        resampled["avg_load"] = resampled["avg_load"].replace([np.inf, -np.inf], np.nan)

    logger.info(
        "Resolution=%s post-resample null counts: %s",
        resolution,
        {
            "avg_load": int(resampled["avg_load"].isna().sum()),
            "day_class": int(resampled["day_class"].isna().sum()),
        },
    )

    day_class_values = set(resampled["day_class"].dropna().unique())
    unexpected = day_class_values - VALID_DAY_CLASSES
    if unexpected:
        raise ValueError(
            f"Resolution={resolution} has unexpected day_class values: {sorted(unexpected)}"
        )

    silver = _feature_columns_for_resolution(resampled, resolution)
    silver = silver.reset_index()

    expected_columns = cast(list[str], SCHEMAS["silver"]["columns"])
    silver = silver[expected_columns]
    validate_schema_columns(silver, expected_columns, "Silver")
    _validate_silver_not_null(silver)

    non_lag_nulls = {
        col: int(silver[col].isna().sum())
        for col in ["avg_load", "day_class", "workday", "year", "quarter", "month", "day_of_week"]
    }
    logger.info("Resolution=%s non-lag null counts: %s", resolution, non_lag_nulls)

    return silver


def bronze_to_silver(
    bronze_path: Path | None = None,
    silver_dir: Path | None = None,
    resolutions: Iterable[str] | None = None,
) -> list[Path]:
    """Generate silver outputs for one or more resolutions."""
    _configure_logging()

    source_path = bronze_path or PATHS["bronze_file"]
    target_dir = silver_dir or PATHS["silver_dir"]
    target_resolutions = (
        list(resolutions) if resolutions is not None else list(DEFAULT_RESOLUTIONS)
    )

    for resolution in target_resolutions:
        _validate_resolution(resolution)

    logger.info("Loading bronze data from %s", source_path)
    if not source_path.exists():
        raise ValueError(f"Bronze input file not found: {source_path}")

    bronze = pd.read_parquet(source_path)
    required_cols = {"timestamp", "load", "day_class"}
    missing = required_cols - set(bronze.columns)
    if missing:
        raise ValueError(f"Bronze file missing required columns: {sorted(missing)}")

    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for resolution in target_resolutions:
        logger.info("Processing silver output for resolution=%s", resolution)
        silver = _build_silver_for_resolution(bronze, resolution, source_path)
        output_path = _silver_output_path(resolution, target_dir)
        try:
            silver.to_parquet(output_path, index=False)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write silver parquet for resolution={resolution} at {output_path}: {exc}"
            ) from exc
        file_size_bytes = output_path.stat().st_size

        logger.info(
            "Silver write complete: resolution=%s rows=%d cols=%d file=%s size_bytes=%d",
            resolution,
            silver.shape[0],
            silver.shape[1],
            output_path,
            file_size_bytes,
        )
        logger.info(
            "Silver timestamp bounds (%s): %s -> %s",
            resolution,
            silver["timestamp"].min(),
            silver["timestamp"].max(),
        )
        dropped_pct = (
            float(bronze["load"].isna().mean() * 100.0)
            if not bronze.empty
            else 0.0
        )
        required_null_total = int(
            silver[cast(list[str], SCHEMAS["silver"]["required_not_null"])].isna().sum().sum()
        )
        emit_quality_gate(
            "SILVER QUALITY GATE",
            dropped_pct <= SILVER_NAN_DROP_FAIL_PCT and required_null_total == 0,
            details={
                "resolution": resolution,
                "rows": silver.shape[0],
                "required_nulls": required_null_total,
                "nan_drop_pct": f"{dropped_pct:.2f}",
                "warn_threshold_pct": f"{SILVER_NAN_DROP_WARN_PCT:.2f}",
                "fail_threshold_pct": f"{SILVER_NAN_DROP_FAIL_PCT:.2f}",
            },
            logger_instance=logger,
        )
        outputs.append(output_path)

    return outputs


if __name__ == "__main__":
    bronze_to_silver()
