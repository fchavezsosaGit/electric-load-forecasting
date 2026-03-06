# Architecture Overview

This document describes the high-level data architecture for the Daily Electric Load
Forecasting pipeline. The system follows a medallion architecture pattern, progressively
transforming raw sensor data into model-ready datasets through well-defined layers.

## Related Documents

| Document | Location |
|----------|----------|
| Pipeline operations | [pipeline.md](../../002_pipeline/pipeline.md) |
| Step-by-step plan | [plan.md](../../002_pipeline/plan.md) |
| Feature set definitions | [feature_sets.md](../../003_modeling/feature_sets.md) |
| Hypotheses | [hypothesis.md](../../003_modeling/hypothesis.md) |
| MVMP scope | [mvmp.md](../../003_modeling/mvmp.md) |
| Glossary | [glossary.md](../../004_reference/glossary.md) |
| Execution specification | [000_spec.md](../../000_governance/000_spec.md) |
| Notebook configuration specification | [001_spec.md](../../000_governance/001_spec.md) |
| Changelog index | [changelog.md](../../../changelog.md) |

## Medallion Architecture

The pipeline implements a four-layer medallion architecture with an additional model
dataset generation stage. Each layer has a single responsibility and a defined contract
with its upstream and downstream neighbors.

```text
+------------------+     +------------------+     +------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |     |                  |     |                  |
|    Raw Layer     | --> |  Bronze Layer    | --> |  Silver Layer    | --> |   Gold Layer     | --> |  Model Datasets  |
|                  |     |                  |     |                  |     |                  |     |                  |
| Source MATLAB    |     | Long-format      |     | Resampled +      |     | Validated +      |     | Split + feature- |
| payload          |     | second-level     |     | feature-         |     | null-filtered    |     | selected train/  |
|                  |     | time series      |     | engineered       |     | model-ready view |     | validate/test    |
+------------------+     +------------------+     +------------------+     +------------------+     +------------------+
  P_data.mat               power_load_1s      power_load_{suffix}     power_load_{suffix}      {suffix}_{feature_set}_{split}
                           .parquet            .parquet                _all_features.parquet    .parquet
```

## Layer Summary

| Layer | Input | Output | Script | Detail Doc |
|-------|-------|--------|--------|------------|
| Raw | Customer-provided `.mat` file | N/A (read-only) | N/A | [raw.md](../001_raw/raw.md) |
| Bronze | `data/000_raw/P_data.mat` | `data/001_bronze/power_load_1s.parquet` | `scripts/000_raw_to_bronze.py` | [bronze.md](../002_bronze/bronze.md) |
| Silver | `data/001_bronze/power_load_1s.parquet` | `data/002_silver/power_load_{suffix}.parquet` | `scripts/001_bronze_to_silver.py` | [silver.md](../003_silver/silver.md) |
| Gold | `data/002_silver/power_load_{suffix}.parquet` | `data/003_gold/power_load_{suffix}_all_features.parquet` | `scripts/002_silver_to_gold.py` | [gold.md](../004_gold/gold.md) |
| Model | `data/003_gold/power_load_{suffix}_all_features.parquet` | `data/004_model/{suffix}_{feature_set}_{split}.parquet` | `scripts/003_create_model_datasets.py` | [model.md](../005_model/model.md) |

## Data Flow

```text
                    +-------------------+
                    |   P_data.mat      |
                    | (86400 x 31)      |
                    | second-level load |
                    +--------+----------+
                             |
                     000_raw_to_bronze.py
                             |
                             v
                    +-------------------+
                    | power_load_1s     |
                    | .parquet          |
                    | 2,678,400 rows    |
                    | 3 columns         |
                    +--------+----------+
                             |
                   001_bronze_to_silver.py
                             |
              +--------------+--------------+--------------+
              |              |              |              |
              v              v              v              v
     +------------+  +------------+  +------------+  +------------+
     | 1m silver  |  | 5m silver  |  | 10m silver |  | 15m silver |
     | 44,640 rows|  | 8,928 rows |  | 4,464 rows |  | 2,976 rows |
     | 44 columns |  | 44 columns |  | 44 columns |  | 44 columns |
     +------+-----+  +------+-----+  +------+-----+  +------+-----+
            |              |              |              |
     (+ additional resolutions 1s, 5s, 10s, 30s when explicitly requested)
                        002_silver_to_gold.py
            |              |              |              |
            v              v              v              v
     +------------+  +------------+  +------------+  +------------+
     | 1m gold    |  | 5m gold    |  | 10m gold   |  | 15m gold   |
     | null-safe  |  | null-safe  |  | null-safe  |  | null-safe  |
     | 44 columns |  | 44 columns |  | 44 columns |  | 44 columns |
     +------+-----+  +------+-----+  +------+-----+  +------+-----+
            |              |              |              |
                     003_create_model_datasets.py
            |              |              |              |
            v              v              v              v
     +--------------+--------------+--------------+------+
     | train        | validate     | test                |
     | (days 1-25)  | (days 26-28) | (days 29-31)       |
     | per resolution, per feature set                   |
     +--------------+--------------+--------------+------+
```

## Report IV Modeling and Evaluation Flow (1min MVP)

The repository now includes an explicit modeling/evaluation path layered on top of the
medallion pipeline. This path keeps validation model selection and holdout testing
separate, with a coverage guard to prevent low-coverage rows from biasing holdout
selection, and adds a Stage-5 performance layer for walk-forward robustness checks
without introducing a second report notebook.

```text
gold/model datasets (1min)
        |
        v
notebooks/003_modeling.ipynb
  - baselines: persistence / previous_day / avg_workday
  - 24-grid: 4 feature sets x (Ridge x3, HistGradientBoostingRegressor (HGB) x3)
  - H1 control + day-ahead extension
        |
        v
validation metrics (outputs/004_modeling/metrics_overall.csv)
        |
        +--> raw-best by MAE (for audit visibility)
        |
        +--> coverage guard (MIN_VALIDATE_COVERAGE=0.95)
                 |
                 v
            selected holdout model
                 |
                 v
            one-shot test evaluation
                 |
                 v
run_manifest.json + report figures/CSVs
        |
        v
scripts/004_model_performance.py
  - preflight protocol checks
  - walk-forward fold evaluation
  - residual-target ablation
  - HGB coordinate search
  - causal blend guardrail
        |
        +--> outputs/005_performance/*
        |
        v
notebooks/003_modeling.ipynb
  - optional Stage-5 artifact summary
```

Key artifacts:
- `outputs/004_modeling/metrics_overall.csv`
- `outputs/004_modeling/metrics_by_day_class.csv`
- `outputs/004_modeling/metrics_by_hour.csv`
- `outputs/004_modeling/run_manifest.json`
- `outputs/004_modeling/fig_*.png`
- `outputs/005_performance/preflight_audit.md`
- `outputs/005_performance/metrics_fold.csv`
- `outputs/005_performance/selection_scoreboard.csv`
- `outputs/005_performance/residual_ablation.csv`
- `outputs/005_performance/hgb_coordinate_summary.csv`
- `outputs/005_performance/guardrail_decisions.csv`
- `outputs/005_performance/guardrail_summary.csv`

## Resolution Policy

The pipeline supports multiple temporal resolutions from second-level to 15-minute
intervals. Each resolution produces independent silver, gold, and model outputs.

Supported resolutions:
- `1s`, `5s`, `10s`, `30s`, `1min`, `5min`, `10min`, `15min`
- Alias: `60s` resolves to `1min`

Default pipeline resolutions (run when no override is provided):
- `1min`, `5min`, `10min`, `15min`

Billing safeguard: if downstream billing settles in 15-minute intervals, the `15min`
resolution should be generated from the start of processing. Post-hoc re-aggregation
from finer resolutions can introduce reconciliation risk.

See the [Glossary](../../004_reference/glossary.md) for resolution terminology.

## Shared Infrastructure

### Configuration (`scripts/config.py`)

`scripts/config.py` is the import surface used by scripts, notebooks, and tests. Declarative
values are loaded from TOML and normalized into a stable Python API at import time.

TOML sources:
- `config/pipeline.toml`: paths, resolutions, aliases/suffixes, feature windows, day-class
  mappings, split ranges, target column, and named feature sets.
- `config/eda.toml`: notebook visualization defaults, analysis thresholds, and resolution
  mode defaults.

This keeps declarative settings editable without Python changes while preserving a stable
configuration API for scripts, notebooks, and tests. Runtime entrypoints
(`run_pipeline.py`, stage scripts, and pytest setup) bootstrap `scripts/` on `sys.path`
for `config`/`utils` imports, and notebooks import `scripts.config` and `scripts.utils`
directly from project root.

Key exports:
- `PROJECT_ROOT`, `PATHS` -- canonical file and directory locations
- `SECONDS_PER_DAY`, `MATLAB_REQUIRED_KEYS` -- raw ingestion contract
- `RAW_MAX_NAN_PCT`, `RAW_MAX_OUT_OF_RANGE_PCT` -- raw/bronze gate thresholds
- `SILVER_NAN_DROP_WARN_PCT`, `SILVER_NAN_DROP_FAIL_PCT` -- silver gate thresholds
- `GOLD_MIN_RETENTION_PCT`, `MODEL_MIN_SPLIT_ROWS` -- downstream gate thresholds
- `SUPPORTED_RESOLUTIONS`, `DEFAULT_RESOLUTIONS`, `RESOLUTION_ALIASES` -- resolution handling
- `EDA_CONFIG` -- centralized notebook visualization and analysis defaults
- `EDA_RESOLUTION_MODES`, `EDA_DEFAULT_RESOLUTION_MODE` -- notebook resolution mode controls
- `resolve_eda_resolutions()`, `resolve_resolution_suffix()` -- canonical mode and suffix resolution
- `get_silver_path()`, `get_gold_path()` -- resolution-aware parquet path builders
- `FEATURE_CONFIG` -- lag, rolling, and slope window definitions
- `DAY_CLASS_MAP`, `VALID_DAY_CLASSES` -- business-day encoding
- `SCHEMAS` -- expected column lists and null-safety rules per layer
- `FEATURE_SETS` -- named predictor column groups for modeling
- `SPLIT_DAY_RANGES` -- chronological train/validate/test boundaries

Notebook resolution modes:
- `all`: every supported resolution (`1s`, `5s`, `10s`, `30s`, `1min`, `5min`, `10min`, `15min`)
- `default`: pipeline defaults (`1min`, `5min`, `10min`, `15min`)
- `custom`: explicit user list with alias normalization (for example `60s -> 1min`)

### Utilities (`scripts/utils.py`)

Shared feature engineering functions used by the silver layer and tested independently:
- `month_to_season(month)` -- maps month to seasonal code (1=Winter through 4=Fall)
- `hour_to_time_of_day(hour)` -- maps hour to time bucket (0=morning through 3=night)
- `rolling_slope_series(series, window)` -- vectorized rolling linear slope using stride tricks

### Orchestration (`run_pipeline.py`)

Single entry point that runs bronze, silver, and gold stages in sequence. Supports:
- `--stage` to run individual stages
- `--resolution` to limit to a single resolution
- `--dry-run` to validate inputs and config without running transformations
- `--verbose` for DEBUG-level logging
- Console and file logging (`logs/pipeline.log`)

See [pipeline.md](../../002_pipeline/pipeline.md) for command examples.

## Conventions

- `workday` is ternary: `none=0`, `half=1`, `full=2`
- `day_of_week`: `0=Sunday` through `6=Saturday`
- `avg_load` is the modeling target; it is excluded from all feature sets
- All derived data (bronze, silver, gold, model, logs) is treated as generated output
  and excluded from version control via `.gitignore`
- Schema validation is enforced at every layer boundary
- Structured logging (Python `logging` module) is used throughout; no `print()` statements

## Orchestration Sequence Diagram

```text
run_pipeline.py
  |
  +-- Stage: bronze
  |     000_raw_to_bronze.py
  |       - Validate .mat file exists and keys present
  |       - Validate shapes: P_data (86400, d), day_data (1, d), day_class (1, d)
  |       - Melt wide format to long (one row per second)
  |       - Merge day_class labels
  |       - Validate: monotonic timestamps, expected row count, schema
  |       - Write parquet, log profile
  |
  +-- Stage: silver
  |     001_bronze_to_silver.py
  |       - For each resolution in config:
  |           - Filter NaN load rows (log count and percentage)
  |           - Resample to target resolution
  |           - Engineer features: temporal, business, lag, rolling, delta, slope
  |           - Validate schema and non-lag null safety
  |           - Write parquet, log row counts and timestamp bounds
  |
  +-- Stage: gold
        002_silver_to_gold.py
          - For each resolution in config:
              - Read silver parquet
              - Validate schema
              - Drop rows with null in required core columns
              - Write parquet, log input/output row counts and null summary
```



