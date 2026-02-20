"""Silver-to-gold transformation.

Gold definition used here:
- Gold is a model-ready, validated view of silver.
- Rows with null values in required modeling columns are dropped.
- Schema is validated and outputs are deterministic.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, cast

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import DEFAULT_RESOLUTIONS, RESOLUTION_ALIASES, SUPPORTED_RESOLUTIONS
from config import PATHS, RESOLUTION_TO_SUFFIX, SCHEMAS, VALID_DAY_CLASSES
from utils import validate_schema_columns

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


def _silver_input_path(resolution: str, silver_dir: Path) -> Path:
    """Build the silver input parquet path for a resolution."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    suffix = RESOLUTION_TO_SUFFIX[canonical]
    return silver_dir / f"power_load_{suffix}.parquet"


def _gold_output_path(resolution: str, gold_dir: Path) -> Path:
    """Build the gold output parquet path for a resolution."""
    canonical = RESOLUTION_ALIASES.get(resolution, resolution)
    suffix = RESOLUTION_TO_SUFFIX[canonical]
    return gold_dir / f"power_load_{suffix}_all_features.parquet"


def silver_to_gold(
    silver_dir: Path | None = None,
    gold_dir: Path | None = None,
    resolutions: Iterable[str] | None = None,
) -> list[Path]:
    """Generate gold outputs for one or more resolutions."""
    _configure_logging()

    input_dir = silver_dir or PATHS["silver_dir"]
    output_dir = gold_dir or PATHS["gold_dir"]
    target_resolutions = (
        list(resolutions) if resolutions is not None else list(DEFAULT_RESOLUTIONS)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    required_not_null = cast(list[str], SCHEMAS["gold"]["required_not_null"])

    for resolution in target_resolutions:
        _validate_resolution(resolution)
        input_path = _silver_input_path(resolution, input_dir)
        output_path = _gold_output_path(resolution, output_dir)

        if not input_path.exists():
            raise ValueError(f"Missing silver input for resolution={resolution}: {input_path}")

        logger.info("Loading silver data: resolution=%s path=%s", resolution, input_path)
        try:
            silver = pd.read_parquet(input_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ValueError(
                f"Failed to read silver input for resolution={resolution} at {input_path}: {exc}"
            ) from exc

        if silver.empty:
            raise ValueError(f"Silver input has zero rows for resolution={resolution}: {input_path}")

        validate_schema_columns(silver, cast(list[str], SCHEMAS["silver"]["columns"]), "Silver")
        silver = silver.sort_values("timestamp").reset_index(drop=True)

        day_class_values = set(silver["day_class"].dropna().unique())
        unexpected_day_class = day_class_values - VALID_DAY_CLASSES
        if unexpected_day_class:
            raise ValueError(
                f"Unexpected day_class values in silver ({resolution}): {sorted(unexpected_day_class)}"
            )

        missing_required_columns = [col for col in required_not_null if col not in silver.columns]
        if missing_required_columns:
            raise ValueError(
                f"Silver input missing required_not_null columns for gold ({resolution}): "
                f"{missing_required_columns}"
            )

        required_null_counts = {
            col: int(silver[col].isna().sum())
            for col in required_not_null
            if silver[col].isna().any()
        }

        input_rows = silver.shape[0]
        gold = silver.dropna(subset=required_not_null).reset_index(drop=True)
        dropped_rows = input_rows - gold.shape[0]
        dropped_pct = (dropped_rows / input_rows) * 100.0 if input_rows else 0.0

        validate_schema_columns(gold, cast(list[str], SCHEMAS["gold"]["columns"]), "Gold")
        post_null_counts = gold.isna().sum().to_dict()

        filtered_day_class_values = set(gold["day_class"].dropna().unique())
        unexpected_filtered_day_class = filtered_day_class_values - VALID_DAY_CLASSES
        if unexpected_filtered_day_class:
            raise ValueError(
                f"Unexpected day_class values in gold ({resolution}): "
                f"{sorted(unexpected_filtered_day_class)}"
            )

        if gold.empty:
            logger.critical(
                "Gold output is empty after required-not-null filtering for resolution=%s. "
                "Required-column null counts: %s",
                resolution,
                required_null_counts,
            )

        try:
            gold.to_parquet(output_path, index=False)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write gold parquet for resolution={resolution} at {output_path}: {exc}"
            ) from exc

        file_size_bytes = output_path.stat().st_size
        logger.info(
            "Gold write complete: resolution=%s input_rows=%d output_rows=%d dropped_rows=%d "
            "dropped_pct=%.2f file=%s size_bytes=%d",
            resolution,
            input_rows,
            gold.shape[0],
            dropped_rows,
            dropped_pct,
            output_path,
            file_size_bytes,
        )
        logger.info("Gold required-not-null drop breakdown (%s): %s", resolution, required_null_counts)
        logger.info(
            "Gold timestamp bounds (%s): %s -> %s",
            resolution,
            gold["timestamp"].min() if not gold.empty else None,
            gold["timestamp"].max() if not gold.empty else None,
        )
        logger.info("Gold null counts (%s): %s", resolution, post_null_counts)
        outputs.append(output_path)

    return outputs


if __name__ == "__main__":
    silver_to_gold()
