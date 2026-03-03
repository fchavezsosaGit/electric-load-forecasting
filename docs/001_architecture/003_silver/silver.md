# Silver Layer

The silver layer resamples the second-level bronze time series to configurable temporal
resolutions and engineers the full set of modeling features. This is where raw load data
becomes analytically useful: temporal context, business classification, lagged values,
rolling statistics, deltas, and slopes are all computed here.

## Related Documents

| Document | Location |
|----------|----------|
| Architecture overview | [../000_overview/architecture.md](../000_overview/architecture.md) |
| Previous layer (Bronze) | [../002_bronze/bronze.md](../002_bronze/bronze.md) |
| Next layer (Gold) | [../004_gold/gold.md](../004_gold/gold.md) |
| Feature set definitions | [feature_sets.md](../../003_modeling/feature_sets.md) |
| Pipeline operations | [pipeline.md](../../002_pipeline/pipeline.md) |
| Glossary | [glossary.md](../../004_reference/glossary.md) |

## Script

| Field | Value |
|-------|-------|
| Script | `scripts/001_bronze_to_silver.py` |
| Entry function | `bronze_to_silver(bronze_path=None, silver_dir=None, resolutions=None)` |
| Config source | `scripts/config.py` (`PATHS`, `SCHEMAS`, `FEATURE_CONFIG`, `DAY_CLASS_MAP`, resolutions) |
| Utility source | `scripts/utils.py` (`month_to_season`, `hour_to_time_of_day`, `rolling_slope_series`) |

## Input and Output

| Direction | Path | Format |
|-----------|------|--------|
| Input | `data/001_bronze/power_load_1s.parquet` | Apache Parquet |
| Output (per resolution) | `data/002_silver/power_load_{suffix}.parquet` | Apache Parquet |

Output files for default resolutions:

| Resolution | Suffix | File | Approximate Row Count |
|------------|--------|------|-----------------------|
| `1min` | `1m` | `power_load_1m.parquet` | 44,640 |
| `5min` | `5m` | `power_load_5m.parquet` | 8,928 |
| `10min` | `10m` | `power_load_10m.parquet` | 4,464 |
| `15min` | `15m` | `power_load_15m.parquet` | 2,976 |

Additional supported resolutions (`1s`, `5s`, `10s`, `30s`) produce outputs when
explicitly requested via the orchestrator or function parameter.

## Schema (44 Columns)

### Core Columns

| Column | Dtype | Nullable | Description |
|--------|-------|----------|-------------|
| `timestamp` | `datetime64[ns]` | No | Start of the resampled interval |
| `avg_load` | `float64` | Yes | Mean load over the interval (watts). NaN if all source seconds were NaN. |
| `day_class` | `string` | No | Business-day classification (first value within interval) |

### Business Feature

| Column | Dtype | Nullable | Description |
|--------|-------|----------|-------------|
| `workday` | `Int64` | No | Ternary encoding: `none=0`, `half=1`, `full=2` |

### Temporal Features

| Column | Dtype | Nullable | Description |
|--------|-------|----------|-------------|
| `year` | `int` | No | Calendar year (e.g., 2025) |
| `quarter` | `int` | No | Quarter of year (1-4) |
| `month` | `int` | No | Month (1-12) |
| `day` | `int` | No | Day of month (1-31) |
| `day_of_week` | `int` | No | Day of week: 0=Sunday through 6=Saturday |
| `hour` | `int` | No | Hour of day (0-23) |
| `season` | `int` | No | Season: 1=Winter (Dec-Feb), 2=Spring (Mar-May), 3=Summer (Jun-Aug), 4=Fall (Sep-Nov) |
| `time_of_day` | `int` | No | Time bucket: 0=morning (6-11), 1=afternoon (12-16), 2=evening (17-21), 3=night (22-5) |

### Lag Features

Lag features represent prior-period values of `avg_load`. The period unit matches the
resolution (e.g., at 5-minute resolution, `lag_1` means one 5-minute period ago).

| Column | Description |
|--------|-------------|
| `lag_1` | `avg_load` shifted by 1 period |
| `lag_5` | `avg_load` shifted by 5 periods |
| `lag_15` | `avg_load` shifted by 15 periods |
| `lag_60` | `avg_load` shifted by 60 periods |
| `lag_1440` | `avg_load` shifted by 1440 periods |

These columns contain expected warm-up NaN values at the start of the time series.
For example, `lag_1440` requires 1440 prior periods before producing a non-NaN value.

### Rolling Features

Rolling features compute aggregates over trailing windows. Window size is in periods,
not fixed time. Each window produces four statistics.

| Window | Mean | Std | Max | Min |
|--------|------|-----|-----|-----|
| 5 | `rolling_mean_5` | `rolling_std_5` | `rolling_max_5` | `rolling_min_5` |
| 15 | `rolling_mean_15` | `rolling_std_15` | `rolling_max_15` | `rolling_min_15` |
| 60 | `rolling_mean_60` | `rolling_std_60` | `rolling_max_60` | `rolling_min_60` |
| 240 | `rolling_mean_240` | `rolling_std_240` | `rolling_max_240` | `rolling_min_240` |
| 1440 | `rolling_mean_1440` | `rolling_std_1440` | `rolling_max_1440` | `rolling_min_1440` |

Rolling features use `min_periods=window` so the first `window - 1` rows are NaN.

### Delta Features

Delta features capture the difference between a longer lag and the immediate lag.
They indicate how much the load has changed between two lookback horizons.

| Column | Formula |
|--------|---------|
| `delta_5` | `lag_5 - lag_1` |
| `delta_15` | `lag_15 - lag_1` |
| `delta_60` | `lag_60 - lag_1` |
| `delta_1440` | `lag_1440 - lag_1` |

### Slope Features

Slope features measure the local linear trend over a trailing window. The slope is
computed using a vectorized least-squares fit (`scripts/utils.py:rolling_slope_series`)
and shifted by 1 period so it uses only past data.

| Column | Window | Description |
|--------|--------|-------------|
| `slope_5` | 5 periods | Short-term trend |
| `slope_15` | 15 periods | Medium-term trend |
| `slope_60` | 60 periods | Longer-term trend |

## Transformation Logic

For each configured resolution, the script performs:

1. Copy bronze data and parse timestamps.
2. Filter out rows where `load` is NaN. Log the count and percentage of dropped rows.
3. Resample `load` to the target resolution using `.resample().mean()` to produce `avg_load`.
4. Resample `day_class` using `.resample().first()` (constant within a day).
5. Validate that all `day_class` values are in the expected set.
6. Engineer features in order:
   a. Business: map `day_class` to `workday` via `DAY_CLASS_MAP`.
   b. Temporal: extract from timestamp index (`year`, `quarter`, `month`, `day`,
      `day_of_week`, `hour`, `season`, `time_of_day`).
   c. Lag: shift `avg_load` by each configured lag period.
   d. Rolling: compute mean/std/max/min over each configured window.
   e. Delta: subtract `lag_1` from each longer lag.
   f. Slope: compute vectorized rolling slope and shift by 1.
7. Reset index to make `timestamp` a column.
8. Reorder columns to match `SCHEMAS["silver"]["columns"]`.
9. Validate schema and non-lag null safety.
10. Write parquet.
11. Log: resolution, row count, column count, file path, timestamp bounds,
    non-lag null counts.

## NaN Handling Strategy

| Source of NaN | Handling |
|---------------|----------|
| Raw sensor NaN (in bronze `load`) | Dropped before resampling. Count and percentage logged. |
| Resampled interval where all source seconds were NaN | Produces NaN in `avg_load`. |
| Lag warm-up | Expected NaN in first N rows of each lag column. |
| Rolling warm-up | Expected NaN in first `window - 1` rows of each rolling column. |
| Slope warm-up + shift | Expected NaN in first `window` rows of each slope column. |
| Non-lag core columns (`timestamp`, `day_class`, temporal features) | Validated as non-null. Raises `ValueError` if any nulls found. |

## Performance

The slope calculation uses `numpy.lib.stride_tricks.sliding_window_view` for vectorized
computation instead of `pandas.Series.apply()` with `np.polyfit`. This provides
approximately 5-10x speedup on the slope features compared to the row-by-row approach.

See `scripts/utils.py:rolling_slope_series` for the implementation.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Bronze file missing | `ValueError` with path |
| Bronze missing required columns | `ValueError` listing missing columns |
| Unsupported resolution requested | `ValueError` listing supported resolutions |
| Unexpected `day_class` values | `ValueError` listing unexpected values |
| `day_class` fails to map to workday | `ValueError` listing unmapped values |
| Schema mismatch after feature engineering | `ValueError` with missing/unexpected column diff |
| Null in required non-lag columns | `ValueError` with null counts |
| Output directory missing | Auto-created via `Path.mkdir(parents=True)` |

## EDA Coverage

Exploratory analysis of the silver layer is documented in `notebooks/002_silver_eda.ipynb`:
- Feature correlation heatmap (identify pairs with |r| > 0.95)
- NaN cascade table (column, NaN count, NaN percentage, reason)
- Feature distribution histograms
- Load autocorrelation (ACF/PACF)
- Multi-resolution comparison (1m vs 5m vs 10m statistics)
- Workday profile analysis (hourly and daily breakdowns)



