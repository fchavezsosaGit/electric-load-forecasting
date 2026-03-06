"""Raw-to-bronze ingestion.

Input:
- MATLAB file with keys: P_data, day_data, day_class
- P_data shape: (seconds_per_day, d), where each column is a day of second-level load

Output:
- Parquet file with schema: timestamp, day_class, load
- One row per second across all days
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.io import loadmat

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    EDA_CONFIG,
    MATLAB_REQUIRED_KEYS,
    PATHS,
    RAW_MAX_NAN_PCT,
    RAW_MAX_OUT_OF_RANGE_PCT,
    SCHEMAS,
    SECONDS_PER_DAY,
    VALID_DAY_CLASSES,
)
from utils import emit_quality_gate

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure default logging when the caller has not configured handlers."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _extract_cell_values(values: np.ndarray) -> list[Any]:
    """Extract scalar values from MATLAB cell-like numpy arrays."""
    extracted: list[Any] = []
    flattened = np.asarray(values).squeeze()
    for item in flattened:
        value: Any = item
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
            value = "".join(value.tolist())
        extracted.append(value)
    return extracted


def raw_to_bronze(raw_path: Path | None = None, output_path: Path | None = None) -> Path:
    """Execute raw-to-bronze ingestion and return the output path."""
    _configure_logging()

    source_path = raw_path or PATHS["raw_mat"]
    target_path = output_path or PATHS["bronze_file"]

    logger.info("Executing raw-to-bronze pipeline")
    logger.info("Raw input path: %s", source_path)
    logger.info("Bronze output path: %s", target_path)

    if not source_path.exists():
        raise ValueError(f"Raw data file not found: {source_path}")

    try:
        raw_data = loadmat(source_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise RuntimeError(f"Could not load MATLAB file at {source_path}: {exc}") from exc

    missing_keys = [key for key in MATLAB_REQUIRED_KEYS if key not in raw_data]
    if missing_keys:
        missing_key = missing_keys[0]
        raise ValueError(f"Key '{missing_key}' not found in .mat file: {source_path}")

    p_data = raw_data["P_data"]
    day_data = raw_data["day_data"]
    day_class = raw_data["day_class"]

    p_data_array = np.asarray(p_data)
    if p_data_array.ndim != 2:
        raise ValueError(f"Expected 2D P_data array, got ndim={p_data_array.ndim}")
    if p_data_array.shape[0] != SECONDS_PER_DAY:
        raise ValueError(
            f"Expected {SECONDS_PER_DAY} rows (seconds/day), got {p_data_array.shape[0]}"
        )
    if p_data_array.shape[1] < 1:
        raise ValueError("P_data has zero columns (no days). Expected at least one day column.")

    if np.issubdtype(p_data_array.dtype, np.number):
        p_data_array = p_data_array.astype(float, copy=False)
    else:
        try:
            p_data_array = p_data_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"P_data must be numeric. Received dtype={p_data_array.dtype} with non-numeric values."
            ) from exc

    inf_mask = np.isinf(p_data_array)
    inf_count = int(inf_mask.sum())
    if inf_count:
        logger.warning("Replacing %d infinity load values with NaN before bronze conversion", inf_count)
        p_data_array = p_data_array.copy()
        p_data_array[inf_mask] = np.nan

    day_values = _extract_cell_values(day_data)
    class_values = [str(v).strip().lower() for v in _extract_cell_values(day_class)]

    if p_data_array.shape[1] != len(day_values) or p_data_array.shape[1] != len(class_values):
        raise ValueError(
            "Day dimensions mismatch: "
            f"P_data days={p_data_array.shape[1]}, day_data={len(day_values)}, day_class={len(class_values)}"
        )

    dates = pd.to_datetime(day_values, errors="raise")
    if pd.Index(dates).has_duplicates:
        duplicate_dates = sorted({str(value.date()) for value in dates[pd.Index(dates).duplicated()]})
        raise ValueError(f"Duplicate dates found in day_data: {duplicate_dates}")

    all_nan_days = [
        str(pd.Timestamp(dates[idx]).date())
        for idx in range(p_data_array.shape[1])
        if np.isnan(p_data_array[:, idx]).all()
    ]
    if all_nan_days:
        logger.warning("Detected day(s) with all-NaN load values: %s", all_nan_days)

    min_load_allowed = cast(float, EDA_CONFIG["physical_load_min_watts"])
    max_load_allowed = cast(float, EDA_CONFIG["physical_load_max_watts"])
    out_of_range_mask = ~np.isnan(p_data_array) & (
        (p_data_array < min_load_allowed) | (p_data_array > max_load_allowed)
    )
    out_of_range_count = int(out_of_range_mask.sum())
    if out_of_range_count:
        logger.warning(
            "Detected %d load values outside plausible range [%.1f, %.1f] watts",
            out_of_range_count,
            min_load_allowed,
            max_load_allowed,
        )

    # Melt converts wide day-columns into a long per-second time series.
    df = (
        pd.DataFrame(p_data_array, columns=dates)
        .assign(second_of_day=np.arange(SECONDS_PER_DAY))
        .melt(id_vars="second_of_day", var_name="date", value_name="load")
        .assign(
            date=lambda x: pd.to_datetime(x["date"]),
            timestamp=lambda x: x["date"] + pd.to_timedelta(x["second_of_day"], unit="s"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    day_class_df = pd.DataFrame(
        {
            "date": pd.to_datetime(day_values),
            "day_class": class_values,
        }
    )

    df = df.merge(day_class_df, on="date", how="left")
    df = df[["timestamp", "day_class", "load"]]

    if df["day_class"].isna().any():
        missing_dates = sorted(
            {
                str(ts.date())
                for ts in pd.to_datetime(
                    df.loc[df["day_class"].isna(), "timestamp"], errors="coerce"
                ).dt.normalize().dropna().unique()
            }
        )
        raise ValueError(
            "Post-merge day_class contains NaN values. Unmatched dates: "
            f"{missing_dates[:10]}"
        )

    unexpected = set(df["day_class"].dropna().unique()) - VALID_DAY_CLASSES
    if unexpected:
        raise ValueError(f"Unexpected day_class values found: {sorted(unexpected)}")

    if not df["timestamp"].is_monotonic_increasing:
        negative_diffs = df["timestamp"].diff() < pd.Timedelta(0)
        bad_index = int(negative_diffs.idxmax())
        prev_ts = df.loc[bad_index - 1, "timestamp"]
        curr_ts = df.loc[bad_index, "timestamp"]
        raise ValueError(
            f"Timestamps not monotonic at row {bad_index}: previous={prev_ts}, current={curr_ts}"
        )

    expected_rows = int(p_data_array.size)
    if df.shape[0] != expected_rows:
        raise ValueError(f"Row count mismatch: expected={expected_rows}, actual={df.shape[0]}")

    expected_columns = SCHEMAS["bronze"]["columns"]
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Bronze schema mismatch. expected={expected_columns}, actual={list(df.columns)}"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(target_path, index=False)
    except OSError as exc:
        raise RuntimeError(
            f"Could not write bronze parquet to {target_path}. Check permissions or disk space. {exc}"
        ) from exc

    date_values = df["timestamp"].dt.normalize()
    load_min = float(df["load"].min(skipna=True)) if df["load"].notna().any() else float("nan")
    load_max = float(df["load"].max(skipna=True)) if df["load"].notna().any() else float("nan")
    load_mean = float(df["load"].mean(skipna=True)) if df["load"].notna().any() else float("nan")
    file_size_bytes = target_path.stat().st_size
    nan_pct = float(df["load"].isna().mean() * 100.0)
    out_of_range_pct = (out_of_range_count / expected_rows) * 100.0 if expected_rows else 0.0
    bronze_gate_passed = (
        nan_pct <= RAW_MAX_NAN_PCT
        and out_of_range_pct <= RAW_MAX_OUT_OF_RANGE_PCT
        and int(df["timestamp"].duplicated().sum()) == 0
    )

    logger.info(
        "Bronze write complete: rows=%d, timestamp_min=%s, timestamp_max=%s",
        df.shape[0],
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    logger.info(
        "Bronze date coverage: unique_dates=%d, first_date=%s, last_date=%s",
        int(date_values.nunique()),
        date_values.min(),
        date_values.max(),
    )
    logger.info(
        "Bronze load summary: min=%.3f max=%.3f mean=%.3f",
        load_min,
        load_max,
        load_mean,
    )
    logger.info("Bronze null counts: %s", df.isna().sum().to_dict())
    logger.info("Bronze day_class distribution: %s", df["day_class"].value_counts().to_dict())
    logger.info("Bronze output saved: %s (%d bytes)", target_path, file_size_bytes)
    emit_quality_gate(
        "BRONZE QUALITY GATE",
        bronze_gate_passed,
        details={
            "rows": df.shape[0],
            "nan_pct": f"{nan_pct:.2f}",
            "nan_threshold_pct": f"{RAW_MAX_NAN_PCT:.2f}",
            "out_of_range_pct": f"{out_of_range_pct:.4f}",
            "out_of_range_threshold_pct": f"{RAW_MAX_OUT_OF_RANGE_PCT:.4f}",
            "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        },
        logger_instance=logger,
    )

    return target_path


if __name__ == "__main__":
    raw_to_bronze()
