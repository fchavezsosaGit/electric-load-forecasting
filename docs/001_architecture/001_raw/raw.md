# Raw Layer

The raw layer contains the original, untouched data provided by the customer. No
transformations, filtering, or restructuring are applied at this stage. The raw file
is treated as read-only throughout the pipeline.

## Related Documents

| Document | Location |
|----------|----------|
| Architecture overview | [../000_overview/architecture.md](../000_overview/architecture.md) |
| Next layer (Bronze) | [../002_bronze/bronze.md](../002_bronze/bronze.md) |
| Pipeline operations | [pipeline.md](../../002_pipeline/pipeline.md) |
| Glossary | [glossary.md](../../004_reference/glossary.md) |

## Input

| Field | Value |
|-------|-------|
| File | `data/000_raw/P_data.mat` |
| Format | MATLAB `.mat` (v5 or v7) |
| Provider | Customer |
| Handling | Read-only; never modified by pipeline scripts |

## Contents

The `.mat` file contains three arrays:

### `P_data`

| Property | Value |
|----------|-------|
| Shape | `(86400, d)` where `d` = number of days |
| Current dataset | `(86400, 31)` -- 31 days |
| Dtype | `float64` |
| Row semantics | One row per second of the day (86,400 seconds = 24 hours) |
| Column semantics | One column per calendar day |
| Value semantics | Instantaneous power load in watts at that second |
| Valid zero | A value of `0` is treated as a valid power load of 0 watts |
| Missing data | Represented as `NaN` -- indicates sensor failure or invalid reading |

### `day_data`

| Property | Value |
|----------|-------|
| Shape | `(1, d)` |
| Current dataset | `(1, 31)` |
| Value semantics | Calendar date corresponding to each column in `P_data` |
| Date range (current) | 2025-11-28 (Friday) through 2025-12-28 (Sunday) |
| Format | String date values stored in MATLAB cell arrays |

### `day_class`

| Property | Value |
|----------|-------|
| Shape | `(1, d)` |
| Current dataset | `(1, 31)` |
| Value semantics | Business-day classification assigned by the customer |
| Valid values | `full`, `half`, `none` |

Day-class distribution in the current dataset:

| Class | Count | Description |
|-------|-------|-------------|
| `full` | 13 | Full working day |
| `half` | 8 | Half working day |
| `none` | 10 | Non-working day |

## Data Characteristics

Date coverage:
- Start: Friday, November 28, 2025
- End: Sunday, December 28, 2025
- Total days: 31
- Total second-level records: 2,678,400 (86,400 x 31)

Missing data:
- 14,576 NaN values in `P_data` (0.54% of total)
- NaN values are scattered across multiple days and hours (not concentrated in a
  single block)

Load range:
- Non-NaN values range from 0 watts to approximately 20,000+ watts
- Typical commercial facility pattern with clear daytime peaks and nighttime baselines

## Assumptions

1. One `.mat` file per customer containing all available data.
2. `P_data` always has exactly 86,400 rows (seconds per day).
3. Column count in `P_data`, `day_data`, and `day_class` must match.
4. `day_class` values are limited to `full`, `half`, and `none`.
5. No incremental data loading is implemented -- the full dataset is processed in each run.
6. The pipeline does not modify the raw file under any circumstances.

## Validation Performed by Downstream (Bronze)

The bronze ingestion script (`scripts/000_raw_to_bronze.py`) validates the following
before processing:

- File exists at the expected path
- Keys `P_data`, `day_data`, `day_class` are present
- `P_data` is 2-dimensional with exactly 86,400 rows
- Column counts match across all three arrays
- `day_class` values are within the valid set
- Dates parse correctly

If any validation fails, the bronze script raises a descriptive error and halts.
See [../002_bronze/bronze.md](../002_bronze/bronze.md) for details.

## EDA Coverage

Exploratory analysis of the raw layer is documented in `notebooks/000_raw_eda.ipynb`:
- Per-day summary statistics (min, max, mean, median, std)
- NaN heatmap (day x hour density)
- Load distribution histogram with outlier detection
- All-day load profile overlay
- Day-class distribution table with dates



