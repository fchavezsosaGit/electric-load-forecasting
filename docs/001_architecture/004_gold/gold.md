# Gold Layer

The gold layer produces a validated, model-ready view of silver data. It applies null
filtering on required core columns, preserves the silver schema exactly, and enforces
deterministic ordering. Gold is the last transformation before data enters the modeling
stage.

## Related Documents

| Document | Location |
|----------|----------|
| Architecture overview | [../000_overview/architecture.md](../000_overview/architecture.md) |
| Previous layer (Silver) | [../003_silver/silver.md](../003_silver/silver.md) |
| Next layer (Model) | [../005_model/model.md](../005_model/model.md) |
| Pipeline operations | [pipeline.md](../../002_pipeline/pipeline.md) |
| Glossary | [glossary.md](../../004_reference/glossary.md) |

## Script

| Field | Value |
|-------|-------|
| Entry script | `scripts/002_silver_to_gold.py` |
| Canonical implementation | `scripts/stages/silver_to_gold.py` |
| Entry function | `silver_to_gold(silver_dir=None, gold_dir=None, resolutions=None)` |
| Config source | `scripts/config.py` (`PATHS`, `SCHEMAS`, `DEFAULT_RESOLUTIONS`, resolution mappings) |

## Input and Output

| Direction | Path Pattern | Format |
|-----------|-------------|--------|
| Input | `data/002_silver/power_load_{suffix}.parquet` | Apache Parquet |
| Output | `data/003_gold/power_load_{suffix}_all_features.parquet` | Apache Parquet |

Output files for default resolutions:

| Resolution | Input File | Output File |
|------------|-----------|-------------|
| `1min` | `power_load_1m.parquet` | `power_load_1m_all_features.parquet` |
| `5min` | `power_load_5m.parquet` | `power_load_5m_all_features.parquet` |
| `10min` | `power_load_10m.parquet` | `power_load_10m_all_features.parquet` |
| `15min` | `power_load_15m.parquet` | `power_load_15m_all_features.parquet` |

## Schema

Gold uses the same 82-column schema as silver. No columns are added or removed.
The difference is in data completeness: rows with null values in required core columns
are dropped.

See [../003_silver/silver.md](../003_silver/silver.md) for the full column listing.

## Gold Definition

Gold is defined as:
- Same schema as silver (82 columns, same dtypes).
- Rows where any required core modeling column is null are dropped.
- Deterministic sort by `timestamp` is preserved.
- No additional features or transformations are applied.

Required non-null columns for gold (from `SCHEMAS["gold"]["required_not_null"]`):

| Column | Why required |
|--------|-------------|
| `timestamp` | Row identity and split assignment |
| `day_class` | Business context for analysis |
| `workday` | Predictor in all feature sets |
| `year` | Temporal predictor |
| `quarter` | Temporal predictor |
| `month` | Temporal predictor |
| `day` | Temporal predictor |
| `day_of_week` | Temporal predictor |
| `hour` | Temporal predictor |
| `season` | Temporal predictor |
| `time_of_day` | Temporal predictor |
| `hour_sin` | Daily cyclical predictor |
| `hour_cos` | Daily cyclical predictor |
| `dow_sin` | Weekly cyclical predictor |
| `dow_cos` | Weekly cyclical predictor |
| `avg_load` | Modeling target -- cannot train/evaluate without it |

Lag, rolling, delta, and slope columns are allowed to contain NaN in gold. These
represent expected warm-up periods at the start of the time series and are handled
during model dataset creation or by the modeling framework.

## Transformation Logic

For each configured resolution:

1. Read the corresponding silver parquet file.
2. Sort by `timestamp` and reset index.
3. Validate that the silver schema matches `SCHEMAS["silver"]["columns"]`.
4. Validate that `day_class` values are within `VALID_DAY_CLASSES`.
5. Drop rows where any column in `SCHEMAS["gold"]["required_not_null"]` is null.
6. Validate that the resulting DataFrame matches `SCHEMAS["gold"]["columns"]`.
7. Write to gold output path.
8. Log: resolution, input row count, output row count, dropped row count, timestamp
   bounds, and remaining null counts across all columns.

## Row Impact

The primary cause of row drops at the gold stage is NaN in `avg_load`. This occurs
when an entire resampled interval had no valid load readings in bronze (all source
seconds were NaN for that interval).

At 1-minute resolution in the current dataset, approximately 40 rows have NaN
`avg_load` (0.09% of 44,640 rows). The temporal and business columns should never
be null after silver processing.

## Determinism

Gold outputs are deterministic: given the same silver inputs, re-running the gold
script produces byte-for-byte identical parquet files. This is guaranteed by:
- Deterministic sort by `timestamp`.
- No random operations.
- Consistent column ordering from schema enforcement.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Silver file missing for a resolution | `ValueError` with path |
| Silver schema does not match expected | `ValueError` with column diff |
| Unexpected `day_class` values | `ValueError` listing unexpected values |
| Output directory missing | Auto-created via `Path.mkdir(parents=True)` |

## Design Rationale

Why a separate gold layer instead of filtering in silver:
- Silver preserves all rows including those with NaN in core columns, which is important
  for EDA and understanding data completeness.
- Gold applies the modeling contract: only rows that can be used for training or
  evaluation survive.
- Separating these concerns makes it possible to run EDA on silver without losing
  context, while guaranteeing that any dataset derived from gold is safe for modeling.

Why gold does not add features:
- All feature engineering is concentrated in the silver layer to keep the transformation
  graph simple. Gold adds quality assurance, not information.
- If future work requires additional derived features (e.g., interaction terms), the
  decision should be documented and the silver schema updated accordingly rather than
  introducing gold-only columns.



