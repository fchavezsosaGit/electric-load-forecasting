# Data Pipeline

This document defines the repository pipeline from raw ingestion to model-ready datasets.
For detailed layer-by-layer documentation, see the architecture docs listed below.

Related references:
- Execution specification: [000_spec.md](../000_governance/000_spec.md)
- Notebook configurability specification: [001_spec.md](../000_governance/001_spec.md)
- Architecture overview: [architecture.md](../001_architecture/000_overview/architecture.md)
- Feature set definitions: [feature_sets.md](../003_modeling/feature_sets.md)
- Report IV run summary: [report_iv_run_summary.md](../003_modeling/report_iv_run_summary.md)
- Report IV success scorecard: [report_iv_success_scorecard.md](../003_modeling/report_iv_success_scorecard.md)
- Glossary: [glossary.md](../004_reference/glossary.md)

## Layers

| Layer | Script | Detail Doc |
|-------|--------|------------|
| Raw | N/A (read-only) | [raw.md](../001_architecture/001_raw/raw.md) |
| Bronze | `scripts/000_raw_to_bronze.py` | [bronze.md](../001_architecture/002_bronze/bronze.md) |
| Silver | `scripts/001_bronze_to_silver.py` | [silver.md](../001_architecture/003_silver/silver.md) |
| Gold | `scripts/002_silver_to_gold.py` | [gold.md](../001_architecture/004_gold/gold.md) |
| Model | `scripts/003_create_model_datasets.py` | [model.md](../001_architecture/005_model/model.md) |

## Resolution Policy

Supported resolutions:
- `1s`, `5s`, `10s`, `30s`, `60s` (alias of `1min`), `1min`, `5min`, `10min`, `15min`

Default pipeline resolutions:
- `1min`, `5min`, `10min`, `15min`

Billing safeguard:
- If billing uses 15-minute settlement, treat `15min` as required from the start
  of ingestion and validation. Post-hoc re-aggregation can create reconciliation risk.

Notebook resolution modes:
- `all`: analyze all supported resolutions.
- `default`: analyze pipeline defaults (`1min`, `5min`, `10min`, `15min`).
- `custom`: analyze an explicit list after alias normalization.

## Notebook Configuration

All EDA notebooks follow a shared two-cell pattern at the top:
- Import cell: project root discovery, `from scripts.config import ...`, utility imports.
- Configuration cell: resolution mode, visualization defaults, analysis thresholds, and
  `AUTO_*` controls sourced from `EDA_CONFIG`.

Central behavior:
- Figure sizes, color palette, threshold defaults, and distribution feature lists come from
  `EDA_CONFIG` (loaded from `config/eda.toml`).
- Self-optimizing helpers (`optimal_bin_count`, `adaptive_outlier_threshold`,
  `optimal_acf_depth`) are opt-in via notebook `AUTO_*` flags.
- The silver notebook resolves input paths using `get_silver_path()` and mode selection via
  `resolve_eda_resolutions()`, with no hardcoded resolution file paths.

Validation overrides:
- `ELF_NB_RESOLUTION_MODE`
- `ELF_NB_CUSTOM_RESOLUTIONS`
- `ELF_NB_AUTO_BINS`
- `ELF_NB_AUTO_OUTLIER`
- `ELF_NB_AUTO_ACF_DEPTH`

`scripts/validate_notebooks.py` executes core notebooks (`000`-`003`) with default settings and runs a silver
validation matrix covering `default`, `all`, and `custom` resolution modes plus both
automatic and fixed-parameter behaviors.

## TOML Configuration Layout

Declarative configuration is stored under `config/`:
- `config/pipeline.toml`: pipeline paths, resolution policy, feature windows, day-class
  mapping, split ranges, target, feature sets, raw ingestion contract, and stage quality thresholds.
- `config/eda.toml`: notebook visualization and analysis defaults, physical range bounds,
  and default notebook resolution mode.

`scripts/config.py` is the stable runtime API. It loads both TOML files with `tomllib`,
normalizes types (for example path strings to `Path` objects and split lists to tuples),
builds computed values (`SILVER_COLUMNS`, `SCHEMAS`, `full` feature set), and enforces
runtime validation through `validate_config()`.

Stage-end gate logging now uses those centralized thresholds to emit:
- `BRONZE QUALITY GATE`
- `SILVER QUALITY GATE`
- `GOLD QUALITY GATE`
- `MODEL DATASETS GATE`
- `PIPELINE HEALTH`

## Raw Layer

Input file:
- `data/000_raw/P_data.mat`

Expected fields:
- `P_data`: shape `(seconds_per_day, d)` where `seconds_per_day` is sourced from
  `config/pipeline.toml [raw_contract]`
- `day_data`: one value per day
- `day_class`: one value per day (`full`, `half`, `none`)

## Bronze Layer

Script:
- `scripts/000_raw_to_bronze.py`

Output:
- `data/001_bronze/power_load_1s.parquet`

Schema:
- `timestamp` (`datetime64[ns]`)
- `day_class` (`string`)
- `load` (`float64`, NaN allowed)

## Silver Layer

Script:
- `scripts/001_bronze_to_silver.py`

Outputs:
- Default outputs (generated unless overridden):
  - `data/002_silver/power_load_1m.parquet`
  - `data/002_silver/power_load_5m.parquet`
  - `data/002_silver/power_load_10m.parquet`
  - `data/002_silver/power_load_15m.parquet`
- Additional optional outputs (generated when explicitly requested):
  - `data/002_silver/power_load_1s.parquet`
  - `data/002_silver/power_load_5s.parquet`
  - `data/002_silver/power_load_10s.parquet`
  - `data/002_silver/power_load_30s.parquet`

Schema (44 columns):
- Core: `timestamp`, `avg_load`, `day_class`
- Business: `workday`
- Temporal: `year`, `quarter`, `month`, `day`, `day_of_week`, `hour`, `season`, `time_of_day`
- Lag: `lag_1`, `lag_5`, `lag_15`, `lag_60`, `lag_1440`
- Rolling: `rolling_mean_*`, `rolling_std_*`, `rolling_max_*`, `rolling_min_*`
- Delta: `delta_5`, `delta_15`, `delta_60`, `delta_1440`
- Slope: `slope_5`, `slope_15`, `slope_60`

Notes:
- Lag/rolling/slope windows are period-based and therefore scale with resolution.
- Non-lag core columns are validated for null safety.

## Gold Layer

Script:
- `scripts/002_silver_to_gold.py`

Outputs:
- Default outputs (generated unless overridden):
  - `data/003_gold/power_load_1m_all_features.parquet`
  - `data/003_gold/power_load_5m_all_features.parquet`
  - `data/003_gold/power_load_10m_all_features.parquet`
  - `data/003_gold/power_load_15m_all_features.parquet`
- Additional optional outputs (generated when explicitly requested):
  - `data/003_gold/power_load_1s_all_features.parquet`
  - `data/003_gold/power_load_5s_all_features.parquet`
  - `data/003_gold/power_load_10s_all_features.parquet`
  - `data/003_gold/power_load_30s_all_features.parquet`

Gold definition:
- Same schema as silver.
- Drop rows where required core modeling columns are null.
- Preserve deterministic sort by timestamp.

## Model Dataset Layer

Script:
- `scripts/003_create_model_datasets.py`

Output pattern:
- `data/004_model/{suffix}_{feature_set}_{split}.parquet`

Rules:
- `avg_load` is target and included separately from features.
- Chronological split (train/validate/test), no random split.

## Orchestration

Main entrypoint:
- `run_pipeline.py`

Logging override:
- `ELF_PIPELINE_LOG_FILE=<path>` writes file logs to a custom location.
- `ELF_PIPELINE_LOG_FILE=off` disables file logs while keeping console logs.

Common commands:

```bash
./setup.sh
python run_pipeline.py
python run_pipeline.py --stage silver --resolution 15min
python run_pipeline.py --stage gold --resolution 60s
python run_pipeline.py --stage performance --performance-mode quick
python run_pipeline.py --stage performance --performance-mode full
python scripts/run_e2e.py --mode quick
python scripts/run_e2e.py --mode full
python run_pipeline.py --dry-run
```

Run all data stages and include Stage-5 performance evaluation:

```bash
python run_pipeline.py --stage all --include-performance --performance-mode quick
```

Performance mode details:
- `quick`: smoke path (curated + curated_ramp, 2 folds, residual + blend guardrail).
- `full`: full fold grid with HGB coordinate regularization search and blend guardrail output.
- `preflight`: protocol checks only, no fold training.

Root wrappers are available for both shells:
- `./run_e2e.sh`
- `.\run_e2e.ps1`
- `PYTHON_BIN=python3.12 ./run_e2e.sh --mode quick`
- `.\run_e2e.ps1 -PythonExe py -- --mode quick`

Notebook smoke validation:

```bash
python scripts/validate_notebooks.py
```

Validation behavior notes:
- Notebook execution uses a Python-managed nbconvert runner in
  `scripts/validate_notebooks.py` so Windows runs apply selector event-loop policy
  automatically and avoid prior `zmq` runtime warnings.
- Notebook validation now clears transient cell outputs after successful execution by
  default so tracked notebooks do not retain machine-specific warning paths or local
  runtime noise. Use `--keep-output` only when notebook output retention is intentional.
- Default smoke scope includes `000_raw_eda.ipynb`, `001_bronze_eda.ipynb`,
  `002_silver_eda.ipynb`, and `003_modeling.ipynb`.
- Silver notebook validation includes baseline plus three profile runs:
  `default`, `all`, and `custom`.
- Stage-5 performance is executed through `scripts/004_model_performance.py`
  (or `run_pipeline.py --stage performance`) and summarized back into
  `003_modeling.ipynb` when artifacts exist.
- When Stage-5 is invoked through `run_pipeline.py` and step-4 artifacts are
  missing, the orchestrator now bootstraps model dataset generation plus the
  `003_modeling.ipynb` artifact export before running performance.

Latest verification snapshot (2026-03-06):
- `python run_pipeline.py --stage all` -> success (bronze/silver/gold rebuilt for all configured resolutions)
- `python run_pipeline.py --stage all --include-performance --performance-mode full` -> success (full integrated bronze/silver/gold/performance pass)
- `python scripts/run_e2e.py --mode full` -> success (`2813.70s` total; pipeline `2611.81s`, notebooks `174.87s`, pytest `27.01s`)
- `python scripts/validate_notebooks.py` -> success (default smoke scope: `000_raw_eda.ipynb` through `003_modeling.ipynb` + silver profile matrix; transient cell outputs cleared after execution)
- `pytest -q tests/notebooks/test_validate_notebooks.py tests/performance/test_model_performance.py` -> success
- `pytest -q` -> success
