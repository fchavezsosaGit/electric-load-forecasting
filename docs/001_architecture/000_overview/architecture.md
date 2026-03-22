# Architecture Overview

This document describes the high-level data architecture for the Daily Electric Load
Forecasting pipeline. The pipeline exists to deliver optimizer-ready load predictions across nowcast and
forecast horizons, and to characterize the accuracy envelope a downstream optimizer
can rely on.
The five project hypotheses (H1-H5) defined in
[hypothesis.md](../../003_modeling/hypothesis.md) drive every architectural decision below.

The system follows a medallion architecture pattern, progressively transforming raw sensor
data into model-ready datasets through well-defined layers.

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

| Layer | Input | Output | Entry Script | Canonical Implementation | Detail Doc |
|-------|-------|--------|--------------|--------------------------|------------|
| Raw | Customer-provided `.mat` file | N/A (read-only) | N/A | N/A | [raw.md](../001_raw/raw.md) |
| Bronze | `data/000_raw/P_data.mat` | `data/001_bronze/power_load_1s.parquet` | `scripts/000_raw_to_bronze.py` | `scripts/stages/raw_to_bronze.py` | [bronze.md](../002_bronze/bronze.md) |
| Silver | `data/001_bronze/power_load_1s.parquet` | `data/002_silver/power_load_{suffix}.parquet` | `scripts/001_bronze_to_silver.py` | `scripts/stages/bronze_to_silver.py` | [silver.md](../003_silver/silver.md) |
| Gold | `data/002_silver/power_load_{suffix}.parquet` | `data/003_gold/power_load_{suffix}_all_features.parquet` | `scripts/002_silver_to_gold.py` | `scripts/stages/silver_to_gold.py` | [gold.md](../004_gold/gold.md) |
| Model | `data/003_gold/power_load_{suffix}_all_features.parquet` | `data/004_model/{suffix}_{feature_set}_{split}.parquet` | `scripts/003_create_model_datasets.py` | `scripts/stages/create_model_datasets.py` | [model.md](../005_model/model.md) |

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
     | 82 columns |  | 82 columns |  | 82 columns |  | 82 columns |
     +------+-----+  +------+-----+  +------+-----+  +------+-----+
            |              |              |              |
     (+ additional resolutions 1s, 5s, 10s, 30s when explicitly requested)
                        002_silver_to_gold.py
            |              |              |              |
            v              v              v              v
     +------------+  +------------+  +------------+  +------------+
     | 1m gold    |  | 5m gold    |  | 10m gold   |  | 15m gold   |
     | null-safe  |  | null-safe  |  | null-safe  |  | null-safe  |
     | 82 columns |  | 82 columns |  | 82 columns |  | 82 columns |
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
without introducing a second report notebook. Post-MVP work is now implemented as
Stage-6 matched-horizon multiresolution comparison plus Stage-7 recursive rollout,
both kept outside notebooks.

```text
gold/model datasets (1min)
        |
        v
notebooks/003_modeling.ipynb
  - baselines: persistence / previous_day / avg_workday
  - 30-grid: 5 feature sets x (Ridge x3, HistGradientBoostingRegressor (HGB) x3)
  - H1 control + day-ahead extension
        |
        v
validation metrics (outputs/004_modeling/<artifact_namespace>/metrics_overall.csv)
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
  -> scripts/modeling/model_performance.py
  - preflight protocol checks
  - walk-forward fold evaluation
  - residual-target ablation
  - adaptive HGB coordinate search
  - horizon-policy feature/model selection
  - causal blend guardrail
  - coverage audit + promotion candidate selection
  - one-shot promoted holdout comparison vs persistence
  - segmented holdout evaluation by regime columns
        |
        +--> outputs/005_performance/<artifact_namespace>/<run_id>/*
        +--> outputs/005_performance/<artifact_namespace>/latest/*
        |
        v
notebooks/003_modeling.ipynb
  - optional Stage-5 artifact summary
```

Key artifacts (latest successful Stage-5 alias shown; timestamped directories remain the
source of truth):
- `outputs/004_modeling/<artifact_namespace>/metrics_overall.csv`
- `outputs/004_modeling/<artifact_namespace>/metrics_by_day_class.csv`
- `outputs/004_modeling/<artifact_namespace>/metrics_by_hour.csv`
- `outputs/004_modeling/<artifact_namespace>/run_manifest.json`
- `outputs/004_modeling/<artifact_namespace>/fig_*.png`
- `outputs/005_performance/<artifact_namespace>/latest/preflight_audit.md`
- `outputs/005_performance/<artifact_namespace>/latest/feature_causality_audit.csv`
- `outputs/005_performance/<artifact_namespace>/latest/metrics_fold.csv`
- `outputs/005_performance/<artifact_namespace>/latest/selection_scoreboard.csv`
- `outputs/005_performance/<artifact_namespace>/latest/residual_ablation.csv`
- `outputs/005_performance/<artifact_namespace>/latest/adaptive_hgb_screen.csv`
- `outputs/005_performance/<artifact_namespace>/latest/coverage_audit.csv`
- `outputs/005_performance/<artifact_namespace>/latest/promotion_candidate.json`
- `outputs/005_performance/<artifact_namespace>/latest/holdout_evaluation.csv`
- `outputs/005_performance/<artifact_namespace>/latest/holdout_segment_evaluation.csv`
- `outputs/005_performance/<artifact_namespace>/latest/deployment_recommendation.json`
- `outputs/005_performance/<artifact_namespace>/latest/hgb_coordinate_summary.csv`
- `outputs/005_performance/<artifact_namespace>/latest/guardrail_decisions.csv`
- `outputs/005_performance/<artifact_namespace>/latest/guardrail_summary.csv`
- `outputs/005_performance/<artifact_namespace>/latest/run_manifest.json`

The Stage-4 modeling CSVs include both raw and normalized errors. `mae_pct` and
`rmse_pct` mean `100 * error / mean(abs(actual_load))` over valid rows.

Notebook execution evidence is archived separately from tracked notebook files:
- `outputs/008_notebook_runs/<run_id>/run_manifest.json`
- `outputs/008_notebook_runs/<run_id>/notebooks/*.ipynb`
- `outputs/008_notebook_runs/latest/*`

## Post-MVP Multi-Resolution, Rollout, and Horizon-Curve Flow

The repository also includes a post-MVP extension path that compares resolutions fairly
and evaluates multi-step rollout without changing the canonical `1min` Report-IV
notebook narrative.

```text
gold/model datasets (1min, 5min, 10min, 15min)
        |
        v
scripts/005_multires_compare.py
  -> scripts/modeling/multires_compare.py
  - native-step summary
  - matched-horizon recursive comparison
  - coverage / stability / runtime / practical-gain gates
  - missing-resolution preflight skip
  - raw artifact deduplication
  - cross-run winner registry emission
  - persistence fallback when no learned candidate clears gates
        |
        +--> outputs/006_multires/<artifact_namespace>/*
        |
        v
selection_summary.csv / selection_summary.md / winner_registry.csv
        |
        v
scripts/006_recursive_rollout.py
  -> scripts/modeling/recursive_rollout.py
  - exact-horizon Stage-6 learned winner from explicit run, registry, latest alias, or config fallback
  - recursive persistence / previous_day / avg_workday / anchored_workday / hybrid_workday baselines
  - learned raw and `avg_workday`-residual rollout candidates
  - multi-origin path evaluation
  - endpoint-aware and path-aware winner selection
        |
        +--> outputs/007_rollout/<artifact_namespace>/*
        |
        v
scripts/007_rollout_challenger_sweep.py
  -> scripts/modeling/rollout_challenger_sweep.py
  - registry-backed long-horizon challenger ranking
  - scoped latest promotion for the current best measured rollout candidate
        |
        +--> outputs/007_rollout/<artifact_namespace>/challenger_sweeps/*
        |
        v
scripts/008_horizon_curve.py
  -> scripts/modeling/horizon_curve.py
  - Stage-5 holdout anchor at 1m
  - Stage-7 challenger-sweep evidence at 15m through 1440m
  - H5 capability-envelope summary, crossover summary, and figures
        |
        +--> outputs/009_horizon_curve/<artifact_namespace>/*
```

Key Stage-6 artifacts:
- `outputs/006_multires/<artifact_namespace>/fold_metrics.csv`
- `outputs/006_multires/<artifact_namespace>/matched_horizon_metrics.csv`
- `outputs/006_multires/<artifact_namespace>/resolution_health.csv`
- `outputs/006_multires/<artifact_namespace>/selection_summary.csv`
- `outputs/006_multires/<artifact_namespace>/selection_summary.md`
- `outputs/006_multires/<artifact_namespace>/winner_registry.csv`

Key Stage-7 artifacts:
- `outputs/007_rollout/<artifact_namespace>/recursive_rollout_metrics.csv`
- `outputs/007_rollout/<artifact_namespace>/recursive_rollout_by_origin.csv`
- `outputs/007_rollout/<artifact_namespace>/rollout_selection_summary.csv`
- `outputs/007_rollout/<artifact_namespace>/rollout_selection_summary.md`
- `outputs/007_rollout/<artifact_namespace>/rollout_registry.csv`
- `outputs/007_rollout/<artifact_namespace>/rollout_health.csv`
- `outputs/007_rollout/<artifact_namespace>/selection_context.json`
- `outputs/007_rollout/<artifact_namespace>/challenger_sweeps/latest/candidate_results.csv`
- `outputs/007_rollout/<artifact_namespace>/challenger_sweeps/latest/candidate_plan.csv`
- `outputs/007_rollout/<artifact_namespace>/challenger_sweeps/latest/recommended_candidate.json`

Key Stage-8 artifacts:
- `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_summary.csv`
- `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_candidates.csv`
- `outputs/009_horizon_curve/<artifact_namespace>/horizon_curve_summary.md`
- `outputs/009_horizon_curve/<artifact_namespace>/crossover_summary.json`
- `outputs/009_horizon_curve/<artifact_namespace>/fig_horizon_ratio_curve.png`
- `outputs/009_horizon_curve/<artifact_namespace>/fig_horizon_absolute_mae.png`

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
- `config/modeling.toml`: shared Stage-5/Stage-6 joblib runtime controls (backend,
  worker caps, batching, dispatch policy, and per-stage toggles).
- `config/multires.toml`: Stage-6/Stage-8 output paths, comparison modes, horizons,
  baseline toggles, selection gates, rollout defaults, horizon-curve defaults, and
  mode-specific profile scopes
  for second-level plus minute-level cadences.

This keeps declarative settings editable without Python changes while preserving a stable
configuration API for scripts, notebooks, and tests. Runtime entrypoints
(`run_pipeline.py`, numbered stage scripts, and pytest setup) bootstrap `scripts/` on
`sys.path` for `config`/`utils` imports. The numbered root scripts are stable
compatibility entrypoints; the canonical implementations live under `scripts/stages/`,
`scripts/modeling/`, and `scripts/tooling/`. Notebooks import `scripts.config` and
`scripts.utils` directly from project root.

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
- `FEATURE_CONFIG` -- legacy period-based and time-normalized lag, rolling, slope,
  activity-threshold, and Fourier cycle definitions
- `DAY_CLASS_MAP`, `VALID_DAY_CLASSES` -- business-day encoding
- `SCHEMAS` -- expected column lists and null-safety rules per layer
- `FEATURE_SETS` -- named predictor column groups for modeling
- `SPLIT_DAY_RANGES` -- chronological train/validate/test boundaries
- `MODELING_PARALLEL`, `MODELING_STAGE_PARALLEL` -- shared Stage-5/Stage-6 job execution policy
- `MODELING_PERFORMANCE_HGB_SEARCH`, `MODELING_HORIZON_POLICIES`,
  `MODELING_PERFORMANCE_EVALUATION` -- centralized Stage-5 tuning and evaluation policy
- `MULTIRES_CONFIG`, `MULTIRES_BASELINES`, `MULTIRES_SELECTION` -- Stage-6 comparison contract
- `MULTIRES_PROFILES`, `MULTIRES_RUNTIME`, `MULTIRES_HYBRID`, `MULTIRES_ROLLOUT`,
  `MULTIRES_HORIZON_CURVE` -- Stage-6/Stage-8 profile scopes, runtime limits,
  rollout defaults, and H5 horizon-curve controls

Notebook resolution modes:
- `all`: every supported resolution (`1s`, `5s`, `10s`, `30s`, `1min`, `5min`, `10min`, `15min`)
- `default`: pipeline defaults (`1min`, `5min`, `10min`, `15min`)
- `custom`: explicit user list with alias normalization (for example `60s -> 1min`)

### Utilities (`scripts/utils.py`)

Shared feature engineering functions used by the silver layer and tested independently:
- `month_to_season(month)` -- maps month to seasonal code (1=Winter through 4=Fall)
- `hour_to_time_of_day(hour)` -- maps hour to time bucket (0=morning through 3=night)
- `build_fourier_feature_frame(timestamps, cycles)` -- generates continuous daily and
  weekly sin/cos features from timestamp phase
- `rolling_slope_series(series, window)` -- vectorized rolling linear slope using stride tricks

Shared modeling feature builders now also live in `scripts/modeling/feature_engineering.py`:
- `add_time_normalized_features()` -- causal minute-based lag/rolling/slope families
- `add_profile_regime_features()` -- previous-day, avg-workday baseline, residual, and
  regime-transition context
- `add_period_history_features()` -- legacy period-based lag/rolling/delta/slope families

### Orchestration (`run_pipeline.py`)

Single entry point that runs bronze, silver, and gold stages in sequence. Supports:
- `--stage` to run individual stages
- `--resolution` to limit to a single resolution
- `--dry-run` to validate inputs and config without running transformations
- `--verbose` for DEBUG-level logging
- `--include-performance`, `--include-multires`, and `--include-rollout` when `--stage all`
- direct stages `performance`, `multires`, and `rollout`
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
  |           - Engineer features: temporal, business, Fourier, lag, rolling, delta, slope
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



