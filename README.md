# Daily Electric Load Forecasting

University of California, San Diego (UCSD)
Master of Advanced Studies in Data Science and Engineering (MAS DSE)
DSE 260: Capstone Project

**Advisor:** Prof. Raymond de Callafon
**Team:** Spencer Hoyle, Sean He, Frank Chavezsosa

---

## Project Overview

This repository implements a reproducible, end-to-end data pipeline and modeling
framework for short-term electric load forecasting. The system ingests raw
second-level power consumption data from a commercial facility, transforms it
through a medallion architecture (raw, bronze, silver, gold), engineers a
comprehensive feature set, and produces model-ready train/validate/test datasets
for supervised forecasting experiments.

The pipeline is designed for auditability, determinism, and professional rigor:
every transformation is logged, every schema is validated at layer boundaries,
and every modeling decision is traceable to documentation.

## Data Summary

The source data consists of one month of second-level power load measurements
from a single commercial customer, delivered as a MATLAB `.mat` file.

| Property | Value |
|----------|-------|
| Source file | `data/000_raw/P_data.mat` |
| Date range | November 28, 2025 through December 28, 2025 |
| Duration | 31 calendar days |
| Granularity | 1 measurement per second (86,400 per day) |
| Total records | 2,678,400 second-level readings |
| Missing data | 14,576 NaN values (0.54% of total) |
| Business-day classes | `full` (13 days), `half` (8 days), `none` (10 days) |
| Load range | 0 to approximately 20,000+ watts |
| Load pattern | Commercial facility with clear daytime peaks and nighttime baselines |

Business-day classification (`day_class`) is provided by the customer and
reflects operational schedules: `full` indicates a full working day, `half`
a partial working day, and `none` a non-working day (weekends, holidays).

## Architecture

The pipeline follows a **medallion architecture** pattern, progressively
transforming raw sensor data into validated, model-ready datasets through
five well-defined layers. Each layer has a single responsibility and a
documented contract with its neighbors.

```text
Raw (.mat)  -->  Bronze (1s parquet)  -->  Silver (multi-resolution, 44 features)
                                                |
                                                v
                                     Gold (null-filtered, model-ready)
                                                |
                                                v
                                     Model Datasets (train/validate/test splits)
```

### Layer Descriptions

| Layer | Purpose | Script | Output | Detail Doc |
|-------|---------|--------|--------|------------|
| Raw | Untouched customer data; read-only | N/A | `data/000_raw/P_data.mat` | [Raw Layer](docs/001_architecture/001_raw/raw.md) |
| Bronze | Format conversion to long-format parquet; no filtering or enrichment | `scripts/000_raw_to_bronze.py` | `data/001_bronze/power_load_1s.parquet` | [Bronze Layer](docs/001_architecture/002_bronze/bronze.md) |
| Silver | Resampling to configurable resolutions + full feature engineering (44 columns) | `scripts/001_bronze_to_silver.py` | `data/002_silver/power_load_{suffix}.parquet` | [Silver Layer](docs/001_architecture/003_silver/silver.md) |
| Gold | Null filtering on required core columns; validated model-ready view | `scripts/002_silver_to_gold.py` | `data/003_gold/power_load_{suffix}_all_features.parquet` | [Gold Layer](docs/001_architecture/004_gold/gold.md) |
| Model | Chronological train/validate/test splits filtered to named feature sets | `scripts/003_create_model_datasets.py` | `data/004_model/{suffix}_{feature_set}_{split}.parquet` | [Model Layer](docs/001_architecture/005_model/model.md) |

For the full architecture overview including data flow diagrams, resolution
policy, shared infrastructure, and orchestration sequence, see the
[Architecture Overview](docs/001_architecture/000_overview/architecture.md).

## Feature Engineering

The silver layer engineers 44 columns from the raw load signal, organized into
six categories:

| Category | Columns | Examples |
|----------|---------|---------|
| Core | 3 | `timestamp`, `avg_load`, `day_class` |
| Business | 1 | `workday` (ternary: none=0, half=1, full=2) |
| Temporal | 8 | `year`, `quarter`, `month`, `day`, `day_of_week`, `hour`, `season`, `time_of_day` |
| Lag | 5 | `lag_1`, `lag_5`, `lag_15`, `lag_60`, `lag_1440` |
| Rolling | 20 | `rolling_mean_*`, `rolling_std_*`, `rolling_max_*`, `rolling_min_*` (windows: 5, 15, 60, 240, 1440) |
| Delta and Slope | 7 | `delta_5`, `delta_15`, `delta_60`, `delta_1440`, `slope_5`, `slope_15`, `slope_60` |

Lag and rolling window sizes are specified in **periods** (not fixed time), so
their real-world duration scales with resolution. For example, `lag_60` at
`5min` resolution looks back 300 minutes (5 hours), while at `1min` resolution
it looks back 60 minutes (1 hour).

For complete column definitions, see the
[Silver Layer](docs/001_architecture/003_silver/silver.md) documentation.

## Modeling Approach

Four canonical feature sets are defined for structured experimentation:

| Feature Set | Columns | Purpose |
|-------------|---------|---------|
| `minimal` | 3 (`workday`, `hour`, `lag_1`) | Fast baseline with minimal context |
| `temporal` | 10 (calendar features + `workday` + `lag_1`) | Calendar structure + immediate lag |
| `curated` | 11 (selected lags, rolling stats, slope) | Balanced signal with reduced collinearity |
| `full` | 41 (all non-metadata columns) | Maximum information content benchmark |

Report IV modeling is currently executed as a **1-minute MVP** with fixed,
reproducible experiment design:

- 24-grid comparison: 4 feature sets x 6 model configurations
- Baselines: persistence, previous-day, avg-workday
- Model families: Ridge (`alpha` in `{0.1, 1.0, 10.0}`) and
  HistGradientBoostingRegressor (3 fixed configs, `random_state=42`)
- Primary hypothesis evaluation split: validation
- Final holdout policy: one-shot test evaluation only after model selection

Current hypothesis posture:

| ID | Hypothesis | Metric | Target | MVP Status |
|----|-----------|--------|--------|------------|
| H1 | Workday signal adds measurable value beyond calendar structure | MAE | >=10% improvement | Evaluated at `1min` |
| H2 | Lag/rolling context reduces large transition errors | RMSE | >=8% improvement | Evaluated at `1min` |
| H3 | Resolution tradeoff (`1min` vs `5min`) | MAE | <=5% degradation | Deferred (multi-resolution phase) |
| H4 | Nonlinear model behavior vs regularized linear baseline | MAE/RMSE | Exploratory | Evaluated at `1min` |

The current MVMP anchor is `1min` + `minimal` for initialization, with all
experiments executed in online single-step forecasting mode.

For full definitions, see:
- [Feature Sets](docs/003_modeling/feature_sets.md)
- [Hypotheses](docs/003_modeling/hypothesis.md)
- [MVMP Scope](docs/003_modeling/mvmp.md)
- [Report IV Run Summary](docs/003_modeling/report_iv_run_summary.md)

## Resolution Support

The pipeline supports multiple temporal resolutions from second-level to
15-minute intervals:

| Resolution | Supported | Default | Approximate Rows (31 days) |
|------------|-----------|---------|---------------------------|
| `1s` | Yes | No | 2,678,400 |
| `5s` | Yes | No | 535,680 |
| `10s` | Yes | No | 267,840 |
| `30s` | Yes | No | 89,280 |
| `1min` | Yes | Yes | 44,640 |
| `5min` | Yes | Yes | 8,928 |
| `10min` | Yes | Yes | 4,464 |
| `15min` | Yes | Yes | 2,976 |

The alias `60s` resolves to `1min`. If downstream billing settles in 15-minute
intervals, generate and validate `15min` outputs from the start of processing
to avoid post-hoc reconciliation risk.

## Setup

**Requirements:** Python 3.11 or later (tested on Python 3.12).

1. Clone this repository.

2. Install dependencies:

```bash
pip install -e ".[dev]"
```

Or use the bootstrap scripts at repo root:

```bash
./setup.sh
```

```powershell
.\setup.ps1
```

3. Review configuration files (TOML-backed runtime config):

```text
config/pipeline.toml
config/eda.toml
```

`config/pipeline.toml` includes operational contracts such as:
- raw ingestion contract (`seconds_per_day`, required MATLAB keys)
- silver quality warning threshold (`silver_nan_drop_warn_pct`)

4. Place raw data at the expected path:

```text
data/000_raw/P_data.mat
```

## Quick Start

```bash
./setup.sh
python run_pipeline.py --dry-run
python run_pipeline.py
```

PowerShell:

```powershell
.\setup.ps1
python run_pipeline.py --dry-run
python run_pipeline.py
```

## Running the Pipeline

Run all stages (bronze, silver, gold):

```bash
python run_pipeline.py
```

Run a single stage:

```bash
python run_pipeline.py --stage bronze
python run_pipeline.py --stage silver
python run_pipeline.py --stage gold
```

Run a single resolution:

```bash
python run_pipeline.py --stage silver --resolution 15min
python run_pipeline.py --stage gold --resolution 60s
```

Dry-run validation (checks inputs and config without running transformations):

```bash
python run_pipeline.py --dry-run
```

Verbose logging:

```bash
python run_pipeline.py --verbose
```

Logs are written to `logs/pipeline.log`.

Optional logging overrides:
- `ELF_PIPELINE_LOG_FILE=<path>` writes pipeline logs to a custom file.
- `ELF_PIPELINE_LOG_FILE=off` disables file logging (console logging remains enabled).

## Report IV Modeling (MVP)

Run the complete modeling chain used for Report IV artifacts:

```bash
python run_pipeline.py --stage all
python scripts/003_create_model_datasets.py
python scripts/validate_notebooks.py --notebook notebooks/003_modeling.ipynb
```

Modeling outputs are written to:

```text
outputs/step4_artifacts/
```

Expected artifact files:
- `metrics_overall.csv` (validation + one-shot holdout test metrics)
- `metrics_by_day_class.csv`
- `metrics_by_hour.csv`
- `run_manifest.json`
- `fig_actual_vs_predicted.png`
- `fig_error_by_hour.png`
- `fig_model_comparison.png`
- `fig_day_ahead.png`

## Tooling Files

- `pyproject.toml` is intentionally at repository root because Python packaging and tool
  discovery (PEP 518/621, pytest, coverage, pip editable install) resolve from root.
- `pyrightconfig.json` is intentionally at repository root because pyright uses root
  config auto-discovery and workspace-relative import resolution from that location.
- The `config/` folder is reserved for runtime pipeline/EDA TOML settings:
  `config/pipeline.toml` and `config/eda.toml`.

## Testing and Validation

Run the full test suite:

```bash
pytest
```

Run with coverage reporting:

```bash
pytest --cov=scripts --cov=run_pipeline --cov-report=term
```

Run notebook smoke validation (executes all core notebooks end-to-end, including
silver resolution/profile matrix checks):

```bash
python scripts/validate_notebooks.py
```

Latest verification snapshot (2026-03-04):
- `python run_pipeline.py --stage all` -> success (bronze/silver/gold rebuilt for all configured resolutions)
- `python scripts/validate_notebooks.py` -> success (default smoke scope: `000_raw_eda.ipynb` through `003_modeling.ipynb` + silver profile matrix)
- `pytest -q tests/notebooks/test_validate_notebooks.py` -> `4 passed`

Latest full-suite reference snapshot (2026-02-20):
- `pytest -q` -> `98 passed`
- `pyright run_pipeline.py scripts tests` -> `0 errors, 0 warnings`

## Documentation Index

| Document | Description | Location |
|----------|-------------|----------|
| Execution Specification | Canonical implementation spec with phase definitions and acceptance criteria | [000_spec.md](docs/000_governance/000_spec.md) |
| Notebook Configurability Spec | Notebook configuration cells, resolution selection, self-optimizing parameters | [001_spec.md](docs/000_governance/001_spec.md) |
| Architecture Overview | High-level data architecture, data flow diagrams, resolution policy, shared infrastructure | [architecture.md](docs/001_architecture/000_overview/architecture.md) |
| Raw Layer | Source data format, MATLAB payload structure, validation assumptions | [raw.md](docs/001_architecture/001_raw/raw.md) |
| Bronze Layer | Format conversion logic, schema, error handling, logging | [bronze.md](docs/001_architecture/002_bronze/bronze.md) |
| Silver Layer | Resampling, feature engineering, 44-column schema, NaN handling, performance | [silver.md](docs/001_architecture/003_silver/silver.md) |
| Gold Layer | Null filtering, model-readiness validation, determinism guarantees | [gold.md](docs/001_architecture/004_gold/gold.md) |
| Model Layer | Chronological splitting, feature-set selection, target separation, leakage prevention | [model.md](docs/001_architecture/005_model/model.md) |
| Pipeline Operations | Layer-by-layer pipeline reference, resolution policy, orchestration commands | [pipeline.md](docs/002_pipeline/pipeline.md) |
| Step-by-Step Plan | Detailed processing steps from raw through evaluation | [plan.md](docs/002_pipeline/plan.md) |
| Feature Sets | Canonical feature set definitions with rationale, risks, and hypothesis connections | [feature_sets.md](docs/003_modeling/feature_sets.md) |
| Hypotheses | Testable hypotheses connecting EDA observations to modeling experiments | [hypothesis.md](docs/003_modeling/hypothesis.md) |
| MVMP Scope | First-pass modeling target with success criteria and persistence baseline | [mvmp.md](docs/003_modeling/mvmp.md) |
| Report IV Run Summary | Measured outcomes from latest executed 1min MVP run and hypothesis snapshot | [report_iv_run_summary.md](docs/003_modeling/report_iv_run_summary.md) |
| Report IV Success Scorecard | Stage-by-stage success readout against `personal/success.md` criteria | [report_iv_success_scorecard.md](docs/003_modeling/report_iv_success_scorecard.md) |
| Glossary | Shared terminology for architecture, features, modeling, and data quality concepts | [glossary.md](docs/004_reference/glossary.md) |
| Changelog Index | Index pointing to spec-specific implementation changelogs | [changelog.md](changelog.md) |

### Documentation Conventions

- Folder prefixes are ordered from `000` upward for predictable navigation.
- Markdown filenames remain semantic (e.g., `architecture.md`, `pipeline.md`).
- Data layer folders and core pipeline scripts use three-digit prefixes
  (`000`, `001`, `002`, `003`) to encode pipeline order.
- The `personal/` directory is non-authoritative scratch space and is excluded
  from version control.

## Project Structure

```text
electric-load-forecasting/
|-- config/
|   |-- pipeline.toml                    Pipeline paths/resolutions/features/splits
|   `-- eda.toml                         Notebook visualization and analysis defaults
|-- data/
|   |-- 000_raw/                         Raw MATLAB source data (read-only)
|   |-- 001_bronze/                      Long-format second-level parquet
|   |-- 002_silver/                      Multi-resolution feature-engineered parquet
|   |-- 003_gold/                        Null-filtered model-ready parquet
|   `-- 004_model/                       Train/validate/test splits by feature set
|-- docs/
|   |-- 000_governance/
|   |   |-- 000_spec.md                   Canonical implementation specification
|   |   `-- CLAUDE.md                     AI assistant configuration (not tracked)
|   |-- 001_architecture/
|   |   |-- 000_overview/architecture.md High-level architecture and data flow
|   |   |-- 001_raw/raw.md               Raw layer documentation
|   |   |-- 002_bronze/bronze.md         Bronze layer documentation
|   |   |-- 003_silver/silver.md         Silver layer documentation
|   |   |-- 004_gold/gold.md             Gold layer documentation
|   |   `-- 005_model/model.md           Model dataset layer documentation
|   |-- 002_pipeline/
|   |   |-- pipeline.md                   Pipeline operations reference
|   |   `-- plan.md                       Step-by-step processing plan
|   |-- 003_modeling/
|   |   |-- feature_sets.md               Feature set definitions
|   |   |-- hypothesis.md                 Testable hypotheses
|   |   |-- mvmp.md                       Minimum Viable Modeling Product scope
|   |   `-- report_iv_run_summary.md      Latest Report IV run outcomes and interpretation
|   |-- 004_reference/
|   |   `-- glossary.md                   Shared terminology
|   `-- change logs/
|       |-- 000spec/changelog.md          SPEC-000 implementation history
|       `-- 001spec/changelog.md          SPEC-001 implementation history
|-- notebooks/
|   |-- 000_raw_eda.ipynb                  Raw data exploratory analysis
|   |-- 001_bronze_eda.ipynb               Bronze data exploratory analysis
|   |-- 002_silver_eda.ipynb               Silver data exploratory analysis
|   `-- 003_modeling.ipynb                 Report IV modeling experiments and artifact export
|-- outputs/
|   `-- step4_artifacts/                   Modeling metrics, figures, and run manifest
|-- scripts/
|   |-- config.py                         Centralized paths, schemas, feature config
|   |-- utils.py                          Shared feature engineering utilities
|   |-- 000_raw_to_bronze.py              Raw-to-bronze ingestion
|   |-- 001_bronze_to_silver.py           Bronze-to-silver transformation
|   |-- 002_silver_to_gold.py             Silver-to-gold validation
|   |-- 003_create_model_datasets.py      Model dataset generation
|   `-- validate_notebooks.py             Notebook smoke-run tool
|-- tests/
|   |-- conftest.py                       Pytest fixtures (synthetic data)
|   |-- unit/
|   |   |-- test_config.py                Configuration validation tests
|   |   `-- test_feature_engineering.py   Shared utility/feature tests
|   |-- stages/
|   |   |-- test_raw_to_bronze.py         Bronze ingestion tests
|   |   |-- test_bronze_to_silver.py      Silver transformation tests
|   |   |-- test_silver_to_gold.py        Gold validation tests
|   |   `-- test_model_datasets.py        Split and leakage tests
|   |-- orchestration/
|   |   `-- test_run_pipeline.py          Orchestrator CLI and error-path tests
|   |-- integration/
|   |   `-- test_integration.py           End-to-end synthetic pipeline tests
|   `-- notebooks/
|       |-- test_validate_notebooks.py    Notebook runner behavior tests
|       `-- test_notebook_structure.py    Notebook quality guard tests
|-- run_pipeline.py                       Pipeline orchestrator
|-- pyproject.toml                        Single source of truth for dependencies/tools
|-- pyrightconfig.json                    Static type-check import path configuration
|-- setup.sh                              Unix-like environment bootstrap script
|-- setup.ps1                             PowerShell environment bootstrap script
`-- changelog.md                          Change history with rationale
```

## Conventions

| Convention | Value | Notes |
|------------|-------|-------|
| `workday` encoding | Ternary: `none=0`, `half=1`, `full=2` | Derived from customer-provided `day_class` |
| `day_of_week` encoding | `0=Sunday` through `6=Saturday` | Matches Python `datetime` adjusted convention |
| Modeling target | `avg_load` (watts) | Explicitly excluded from all predictor feature sets |
| Season encoding | `1=Winter` (Dec-Feb), `2=Spring` (Mar-May), `3=Summer` (Jun-Aug), `4=Fall` (Sep-Nov) | Meteorological seasons |
| Time-of-day encoding | `0=morning` (6-11), `1=afternoon` (12-16), `2=evening` (17-21), `3=night` (22-5) | Coarse time buckets |
| Chronological split | Train: days 1-25, Validate: days 26-28, Test: days 29-31 | No random splitting; strict time order |
| Schema source of truth | `scripts/config.py` | Runtime API backed by `config/pipeline.toml` and `config/eda.toml` |
| Logging | Python `logging` module | No `print()` statements in pipeline scripts |
| Derived data | Not version-controlled | Bronze, silver, gold, model, and log outputs are in `.gitignore` |

For detailed definitions of all terms used in this project, see the
[Glossary](docs/004_reference/glossary.md).

## Source of Truth

The canonical implementation specifications for this project are
[000_spec.md](docs/000_governance/000_spec.md) (pipeline hardening) and
[001_spec.md](docs/000_governance/001_spec.md) (notebook development and
configuration migration). If any planning text conflicts with implementation
details in code or documentation, the governing specification and the
corresponding entries in the spec-specific changelogs linked from the
[Changelog Index](changelog.md) take precedence.
