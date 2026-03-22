# Daily Electric Load Forecasting

University of California, San Diego (UCSD)
Master of Advanced Studies in Data Science and Engineering (MAS DSE)
DSE 260: Capstone Project

**Advisor:** Prof. Raymond de Callafon
**Team:** Spencer Hoyle, Sean He, Frank Chavezsosa

---

## Goal

Build a reproducible forecasting system that delivers optimizer-ready load
predictions across nowcast and forecast horizons for a single commercial
facility, using 31 days of second-level power consumption data. The end
product is an accuracy envelope that a downstream optimizer can rely on;
the optimizer itself is out of scope.

## Hypotheses

The repo now keeps two hypothesis layers on purpose:
- historical notebook-era hypotheses (`H1`-`H5`) that explain how the project
  started and why `1min` was the original focus
- current operational hypotheses (`O1`-`O5`) that govern the optimizer-facing
  stacked forecast/nowcast surface

Current operating emphasis:

| ID | Question | Primary readout | Current status |
|----|----------|------------------|----------------|
| O1 | Does the layered stack beat the frozen day-ahead path where the optimizer cares most? | `next_lock_mae`, `lock_mae`, peak metrics | Supported |
| O2 | Does the minute layer help as a corrective overlay even if standalone `1m` stays baseline-led? | stacked gain vs phase layer | Supported |
| O3 | Do delivered forecasts include honest uncertainty, not just point values? | interval-band coverage and width | Implemented, still calibrating |
| O4 | Can the repo emit a stable pre-optimizer contract with provenance and fallback semantics? | contract completeness | Implemented |
| O5 | Is the pipeline explicit and safe across host capabilities and fallback paths? | policy/contract/runtime behavior | Implemented, still hardening |

Docs:
- historical hypotheses: [hypothesis.md](docs/003_modeling/hypothesis.md)
- current operational hypotheses: [operational_hypotheses.md](docs/003_modeling/operational_hypotheses.md)
- optimizer-facing delivery contract: [optimizer_delivery_contract.md](docs/003_modeling/optimizer_delivery_contract.md)
- optimizer-facing operating summary: [current_operating_approach.md](docs/003_modeling/current_operating_approach.md)
- active operating-direction spec: [002_operating_direction_spec.md](docs/000_governance/002_operating_direction_spec.md)
- active implementation plan: [003_operating_direction_implementation_plan.md](docs/000_governance/003_operating_direction_implementation_plan.md)

Current plain-English operating summary:
- [current_validation_snapshot.md](docs/003_modeling/current_validation_snapshot.md)
- [current_operating_approach.md](docs/003_modeling/current_operating_approach.md)
- [model_and_blend_guide.md](docs/003_modeling/model_and_blend_guide.md)
- [optimizer_delivery_contract.md](docs/003_modeling/optimizer_delivery_contract.md)
- [002_operating_direction_spec.md](docs/000_governance/002_operating_direction_spec.md)
- [stage_map.md](docs/002_pipeline/stage_map.md)

## Direction Change

The repo no longer treats "find one learned `1m` winner" as the main goal.

The current evidence changed the direction:
- Stage-5 canonical holdout still keeps `persistence` as the honest standalone
  `1m` anchor on a narrow `3`-day `none_inactive` slice.
- The broader leakage-safe Stage-5 advisory surface is learned-positive, but
  mainly in transition and high-ramp regimes.
- Stage-10 exact and rolling replay show the real value is the layered stack:
  learned day-ahead anchor, useful hourly correction, structural `15m` slot,
  and a strong minute overlay.
- The current dynamic minute gate is not live-ready because hard enforcement
  would worsen all-interval error sharply, so it remains shadow-only.

The active direction, retired bets, and source credit now live in
[002_operating_direction_spec.md](docs/000_governance/002_operating_direction_spec.md).
The phased execution plan and acceptance criteria live in
[003_operating_direction_implementation_plan.md](docs/000_governance/003_operating_direction_implementation_plan.md).

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

## Operating Stack

The repository now works as a layered decision stack rather than as a single
"best model" search:

1. Data preparation: raw -> bronze -> silver -> gold -> model datasets.
2. Stage-4 notebook modeling: establish the canonical `1min` benchmark surface and
   required figures.
3. Stage-5 performance: promote the strongest short-horizon learned challenger,
   then test it directly against persistence on holdout and persist an explicit
   minute operating policy that distinguishes standalone deployment from
   Stage-10 overlay use.
4. Stage-6 and Stage-7: compare matched horizons and recursive rollout candidates
   so each horizon can keep the policy that actually wins for that objective.
5. Stage-8 horizon curve: consolidate the best validated candidate at each horizon
   into one objective-aware capability envelope.
6. Stage-10 forecast control: replay the current day-ahead plus intraday stack on
   shared control cycles to verify that updates reduce the next locked-interval
   error and the overall day profile error.

That layering is intentional. Day-ahead profile quality, hourly correction, and
sub-15-minute correction quality are related, but they are not the same decision.
The repo therefore keeps separate evidence surfaces and only promotes a policy
when it wins on the metric that matters for that layer.

If you only need the current answer instead of the full research history, read
[current_validation_snapshot.md](docs/003_modeling/current_validation_snapshot.md)
first, then
[current_operating_approach.md](docs/003_modeling/current_operating_approach.md)
before diving into the deeper stage reports. If the candidate labels still look
opaque, [model_and_blend_guide.md](docs/003_modeling/model_and_blend_guide.md)
is the fastest way to understand what the wrappers and winners mean.

## Plain-English Stage Map

The numeric stage folders are kept stable because they are part of the repo's
artifact and documentation contract. Renaming them aggressively would make old
run paths and report references harder to trust. The easier fix is to pair each
stable stage id with a plain-English label:

- Stage-0 to Stage-3: data preparation (`raw` -> `bronze` -> `silver` -> `gold`)
- Stage-4 (`outputs/004_modeling`): notebook benchmark surface
- Stage-5 (`outputs/005_performance`): short-horizon holdout gate
- Stage-6 (`outputs/006_multires`): matched-horizon comparison
- Stage-7 (`outputs/007_rollout`): recursive rollout selection and challenger sweeps
- Stage-8 (`outputs/009_horizon_curve`): horizon capability curve
- Stage-10 (`outputs/010_forecast_control`): end-to-end control backtest

The repo therefore uses stable numeric folders for provenance and friendlier
labels in the docs when the question is "what does this stage actually do?"
For the fuller version with "what question does this stage answer?" guidance,
see [stage_map.md](docs/002_pipeline/stage_map.md).
`outputs/008_notebook_runs/` is a notebook evidence archive surface, not a
separate modeling decision stage.

Canonical current-state note:
- `docs/003_modeling/current_validation_snapshot.md` is now generated from the
  latest artifacts and is the repo's one-page current-state answer.

## How Quality Is Measured

The repository measures quality at the decision layer where the forecast is used,
not only at one generic MAE leaderboard:

- Stage-4 and Stage-5 benchmark the `1min` surface with `MAE`, `RMSE`, `MAE%`,
  `RMSE%`, split coverage, and holdout comparisons against persistence and other
  baselines.
- Stage-6 compares matched horizons across resolutions with `endpoint_mae`,
  `path_mae`, and their normalized percentages so cross-resolution winners remain
  interpretable on different load scales.
- Stage-7 rollout selection is objective-aware. Short corrective horizons can use
  `next_lock_mae` or `phase_mean_mae`, while broader horizons can use
  `path_mae` or `profile_shape_mae`.
- Stage-8 consolidates those objective-aware winners into one horizon curve so
  the repo can answer where learned models help, where baselines still win, and
  where the operating policy must stay layered.
- Stage-10 backtests the stacked control policy directly and asks the business
  question: does freezing the day-ahead forecast perform worse than updating it
  before the next costly interval locks?

Interpretation rule:
- raw MAE is reported in the native load unit and remains operationally useful
  for one facility
- MAE% and related normalized metrics are the scale-aware companion and should
  be used whenever results are compared across horizons, runs, or future load
  types
- promotion logic should be trusted only when the candidate also clears its
  baseline gate on the objective that matters for that stage

## How Results Are Visualized

The visualization layer follows the same top-down structure as the measurement
layer:

- Stage-4 notebook visuals explain the `1min` benchmark surface: fit quality,
  hour-of-day error structure, model ranking, and the day-ahead extension.
- Stage-8 horizon-curve visuals explain how performance changes as horizon grows
  and where the learned stack crosses above or below persistence/baseline parity.
- Stage-10 control visuals explain whether hourly and phase updates actually pull
  locked-interval error and whole-day profile error down versus a frozen
  day-ahead forecast.

Every decision-facing PNG now has a sibling `figure_guide.md` in the same output
folder. Those guides state:
- why the figure exists
- how to read it
- what pattern should count as a win, warning, or failure

The repo also maintains one cross-stage reporting surface for the latest state:
- `docs/003_modeling/current_visualization_guide.md`
- `outputs/reports/commercial_facility/latest/validation_dashboard.html`

Primary guide locations:
- `outputs/004_modeling/commercial_facility/figure_guide.md`
- `outputs/005_performance/commercial_facility/latest/figure_guide.md`
- `outputs/006_multires/commercial_facility/latest/figure_guide.md`
- `outputs/007_rollout/commercial_facility/latest/figure_guide.md`
- `outputs/009_horizon_curve/commercial_facility/latest/figure_guide.md`
- `outputs/010_forecast_control/commercial_facility/latest/figure_guide.md`

Executed notebooks are also archived before outputs are cleared. That archive
keeps the exact notebook snapshot, artifact checks, and figure metadata under
`outputs/008_notebook_runs/<artifact_namespace>/<run_id>/`.

The same validation flow now also refreshes
`docs/003_modeling/current_validation_snapshot.md`,
`docs/003_modeling/current_visualization_guide.md`, and
`outputs/reports/commercial_facility/latest/validation_dashboard.html`, so the
high-level repo status and visual story are sourced from artifacts instead of
manual copy edits.

## Artifact Retention

Timestamped output folders are preserved for provenance, but they can become
hard to navigate after repeated tuning cycles. The repository now includes a
conservative cleanup tool:

- `python scripts/tooling/cleanup_outputs.py`

That tool:

- keeps `latest/` aliases and support folders intact
- keeps a small recent buffer per stage
- keeps any dated run that is still referenced by the current docs or latest
  artifact surface
- deletes only superseded dated runs that are no longer part of the active
  evidence chain

The latest cleanup report is written to:

- `personal/output_cleanup_report.md`
- `personal/output_cleanup_report.json`

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
Raw (.mat)  -->  Bronze (1s parquet)  -->  Silver (multi-resolution, 89 columns)
                                                |
                                                v
                                     Gold (null-filtered, model-ready)
                                                |
                                                v
                                     Model Datasets (train/validate/test splits)
```

### Layer Descriptions

| Layer | Purpose | Entry Script | Canonical Implementation | Output | Detail Doc |
|-------|---------|--------------|--------------------------|--------|------------|
| Raw | Untouched customer data; read-only | N/A | N/A | `data/000_raw/P_data.mat` | [Raw Layer](docs/001_architecture/001_raw/raw.md) |
| Bronze | Format conversion to long-format parquet; no filtering or enrichment | `scripts/000_raw_to_bronze.py` | `scripts/stages/raw_to_bronze.py` | `data/001_bronze/power_load_1s.parquet` | [Bronze Layer](docs/001_architecture/002_bronze/bronze.md) |
| Silver | Resampling to configurable resolutions + full feature engineering (89 columns) | `scripts/001_bronze_to_silver.py` | `scripts/stages/bronze_to_silver.py` | `data/002_silver/power_load_{suffix}.parquet` | [Silver Layer](docs/001_architecture/003_silver/silver.md) |
| Gold | Null filtering on required core columns; validated model-ready view | `scripts/002_silver_to_gold.py` | `scripts/stages/silver_to_gold.py` | `data/003_gold/power_load_{suffix}_all_features.parquet` | [Gold Layer](docs/001_architecture/004_gold/gold.md) |
| Model | Chronological train/validate/test splits filtered to named feature sets | `scripts/003_create_model_datasets.py` | `scripts/stages/create_model_datasets.py` | `data/004_model/{suffix}_{feature_set}_{split}.parquet` | [Model Layer](docs/001_architecture/005_model/model.md) |

For the full architecture overview including data flow diagrams, resolution
policy, shared infrastructure, and orchestration sequence, see the
[Architecture Overview](docs/001_architecture/000_overview/architecture.md).

## Feature Engineering

The silver layer now engineers 89 columns from the raw load signal. It retains
the original period-based features and adds time-normalized, regime-aware, and
baseline-relative features so Stage-5 through Stage-8 can compare horizons more
fairly.

| Category | Columns | Examples |
|----------|---------|---------|
| Core | 3 | `timestamp`, `avg_load`, `day_class` |
| Business | 1 | `workday` (ternary: none=0, half=1, full=2) |
| Temporal | 8 | `year`, `quarter`, `month`, `day`, `day_of_week`, `hour`, `season`, `time_of_day` |
| Fourier | 4 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| Phase context | 6 | `phase_progress_15m`, `phase_boundary_dist_15m`, `phase_sin_15m` |
| Lag | 5 | `lag_1`, `lag_5`, `lag_15`, `lag_60`, `lag_1440` |
| Rolling | 20 | `rolling_mean_*`, `rolling_std_*`, `rolling_max_*`, `rolling_min_*` (windows: 5, 15, 60, 240, 1440) |
| Delta and Slope | 7 | `delta_5`, `delta_15`, `delta_60`, `delta_1440`, `slope_5`, `slope_15`, `slope_60` |
| Time-normalized lags/rolling/slope | 25 | `lag_min_15`, `rolling_mean_min_60`, `slope_min_240` |
| Baseline and regime context | 10 | `avg_workday_baseline`, `anchored_workday_baseline`, `previous_day_residual`, `profile_active_flag` |

The pipeline now carries two windowing families:
- Legacy period-based windows such as `lag_60` and `rolling_mean_60`, which scale
  with resolution.
- Time-normalized windows such as `lag_min_60` and `rolling_mean_min_240`, which
  represent the same real-world lookback across resolutions.

For complete column definitions, see the
[Silver Layer](docs/001_architecture/003_silver/silver.md) documentation.

## Modeling Approach

Eight canonical feature sets are defined for structured experimentation:

| Feature Set | Columns | Purpose |
|-------------|---------|---------|
| `minimal` | 3 (`workday`, `hour`, `lag_1`) | Fast baseline with minimal context |
| `minimal_phase` | 9 (`minimal` + quarter-hour phase features) | Lean short-horizon phase-awareness set |
| `minimal_phase_anchor` | 15 (`minimal_phase` + workday-profile anchor context) | Lean 15-minute billing-oriented corrective set |
| `temporal` | 14 (calendar features + continuous cyclical encoding + `lag_1`) | Calendar structure + immediate lag |
| `curated` | 15 (selected lags, rolling stats, slope, cyclical encoding) | Balanced signal with reduced collinearity |
| `full` | 86 (all non-metadata predictors) | Maximum information content benchmark |
| `full_stable` | 78 (`full` minus the longest legacy rolling windows) | Coverage-safe high-capacity benchmark |
| `regime_profile` | 32 (calendar + short memory + profile residual context) | Horizon-aware baseline-correction set |

Stage-5 also uses:
- `curated_ramp`: a non-canonical helper set that extends `curated` with short-ramp features.

Report IV modeling is still anchored by a **1-minute Minimum Viable Product (MVP)**,
but the scripted modeling stack is now more adaptive and centralized:

- canonical notebook grid: 5 notebook-facing feature sets x 6 model configurations
- Baselines: persistence, previous-day, avg-workday
- Model families: Ridge (`alpha` in `{0.1, 1.0, 10.0}`) and
  HistGradientBoostingRegressor (3 fixed configs, `random_state=42`), with
  optional XGBoost candidates available in the scripted Stage-5/Stage-6 stack
  when the acceleration extra is installed
- Primary hypothesis evaluation split: validation
- Final holdout policy: one-shot test evaluation only after model selection
- Stage-5 horizon policies now centralize which feature sets, model families, and
  residual/blend options are eligible for short, hourly, and day-ahead comparisons
- Stage-5 also writes segmented holdout diagnostics, adaptive HGB screening output,
  and deployment recommendations based on both MAE and MAE%
- Stage-7 now evaluates raw learned rollouts plus `persistence`, `avg_workday`, and
  `anchored_workday` residual learned rollouts, then selects endpoint, path, or
  15-minute phase winners explicitly by objective

Hypothesis targets and current status are summarized in the
[Hypotheses](#hypotheses) section at the top of this document.

The current Minimum Viable Modeling Product (MVMP) anchor is `1min` + `minimal` for initialization, with all
experiments executed in online single-step forecasting mode.

For full definitions, see:
- [Feature Sets](docs/003_modeling/feature_sets.md)
- [Hypotheses](docs/003_modeling/hypothesis.md)
- [MVMP Scope](docs/003_modeling/mvmp.md)
- [Model and Blend Guide](docs/003_modeling/model_and_blend_guide.md)
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

Current operating policy:
- the pipeline supports sub-minute experimentation, but the current validated
  optimizer-facing control surface still runs at `1min`
- `30s` is the only sub-minute cadence with meaningful current matched-horizon
  promise, but it remains exploratory because the latest Stage-6 evidence does
  not clear the repo's full stability/selection gates for promotion
- `1s`, `5s`, and `10s` remain supported for ingestion and comparison, not the
  current production-facing default path
- a dedicated `subminute_focus` multires profile now exists for repeatable
  `30s` vs `1min` investigation without changing the current operating contract

## Scripts Layout

The `scripts/` directory now separates stable entrypoints from implementation code:

- `scripts/000_raw_to_bronze.py` through `scripts/009_forecast_control_backtest.py` remain
  the numbered entrypoints used by commands, tests, and notebooks.
- `scripts/stages/` contains the canonical bronze/silver/gold/model dataset
  implementations.
- `scripts/modeling/` contains stage-5 through stage-10 modeling code and shared helpers.
- `scripts/tooling/` contains environment bootstrap, notebook validation, and E2E tooling.
- `scripts/config.py` and `scripts/utils.py` remain top-level because they are the
  shared runtime API imported throughout the repo.

This keeps the CLI stable while reducing top-level script sprawl.

## Setup

**Requirements:** Python 3.11 or later (tested on Python 3.12 and 3.13).

1. Clone this repository.

2. Bootstrap a local environment from the repo-owned scripts:

```bash
./setup.sh
```

```powershell
.\setup.ps1
```

Optional variants:

```bash
./setup.sh --venv-dir .venv311
./setup.sh --no-venv
./setup.sh --with-acceleration
python scripts/bootstrap_env.py --install --with-dev --check
python scripts/bootstrap_env.py --install --with-dev --with-acceleration --check
```

```powershell
.\setup.ps1 -VenvDir .venv311
.\setup.ps1 -NoVenv
.\setup.ps1 -WithAcceleration
python scripts/bootstrap_env.py --install --with-dev --check
python scripts/bootstrap_env.py --install --with-dev --with-acceleration --check
```

Setup notes:
- `setup.ps1` prefers `py -3.13` or `py -3.12` on Windows when available to
  reduce wheel-compatibility surprises on newer interpreters.
- Both setup scripts now recreate an existing virtual environment automatically
  if it is no longer runnable on the current machine (for example, after moving
  from an ARM64 Python install to an x64/AMD64 host).
- `--with-acceleration` installs the optional XGBoost stack on supported x64
  Windows/Linux machines so Stage-5 and Stage-6 can evaluate GPU-backed tree
  models when CUDA is available.

Runtime acceleration knobs:
- `ELF_ACCELERATION=auto` (default): use GPU-backed XGBoost when the optional
  dependency is installed and a CUDA-capable runtime probe succeeds, otherwise
  fall back to CPU.
- `ELF_ACCELERATION=cpu`: keep optional XGBoost enabled but force CPU execution.
- `ELF_ACCELERATION=off`: disable optional XGBoost candidates entirely.
- `ELF_MAX_WORKERS` and `ELF_RESERVED_CORES` override the repo's adaptive
  worker sizing if you need to clamp or expand local parallelism manually.

3. Review configuration files (TOML-backed runtime config):

```text
config/pipeline.toml
config/eda.toml
config/modeling.toml
config/multires.toml
```

`config/pipeline.toml` includes operational contracts such as:
- raw ingestion contract (`seconds_per_day`, required MATLAB keys)
- quality thresholds for raw NaN tolerance, silver drop-rate gates, gold retention,
  and model split completeness
- centralized generated-artifact paths for modeling and Stage-5 performance outputs

`config/multires.toml` defines the Stage-6 through Stage-10 execution surface:
- full enabled resolution set (`1s`, `5s`, `10s`, `30s`, `1min`, `5min`, `10min`, `15min`)
- profile-scoped runtime defaults so smoke/candidate runs stay bounded while full mode can
  still include second-level cadences
- supported multiresolution comparison modes (`smoke`, `candidate`, `full`)
- matched-horizon list and baseline toggles
- selection gates (coverage, stability, runtime, practical gain)
- rollout defaults (selected resolution, model label, horizon, origin count, selection target)
- horizon-curve defaults (evaluated horizons, origin policy, selected objective, Stage-5 anchor usage)
- second-level matched-horizon support based on exact second divisibility, so
  `1s`, `5s`, `10s`, and `30s` can participate in the same real-time horizon
  comparisons as minute-level cadences

`config/modeling.toml` defines the shared Stage-5 and Stage-6 execution runtime:
- joblib backend selection (`threading`, `loky`, or `sequential`)
- worker caps, adaptive headroom-aware parallelism, task batching, and dispatch policy
- per-stage toggles for performance and multires batches
- nested thread caps to prevent oversubscription during parallel model evaluation
- the default outer-worker cap remains conservative; use `ELF_MAX_WORKERS` only
  after timing your own workload, because heavier HGB sweeps can slow down when
  too many outer jobs compete for memory and native compute

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

Run the full reproducibility workflow from one entrypoint:

```bash
./run_e2e.sh --mode full
PYTHON_BIN=python3.12 ./run_e2e.sh --mode quick
```

```powershell
.\run_e2e.ps1 --mode full
.\run_e2e.ps1 -PythonExe py -- --mode quick
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
python run_pipeline.py --stage performance --performance-mode quick
python run_pipeline.py --stage performance --performance-mode full
python run_pipeline.py --stage multires --multires-mode smoke
python run_pipeline.py --stage rollout_sweep
python run_pipeline.py --stage rollout
python run_pipeline.py --stage horizon_curve
python run_pipeline.py --stage forecast_control
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

Run all data stages and include Stage-5 performance:

```bash
python run_pipeline.py --stage all --include-performance --performance-mode quick
```

Run the integrated post-MVP comparison stack:

```bash
python run_pipeline.py --stage all --include-multires --multires-mode smoke
python run_pipeline.py --stage all --include-performance --performance-mode quick --include-multires --include-rollout
```

Root E2E runner:

```bash
python scripts/run_e2e.py --mode full
```

Quick smoke path:

```bash
python scripts/run_e2e.py --mode quick
python scripts/run_e2e.py --mode quick --with-multires --with-rollout
python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep
python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control
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
python run_pipeline.py --stage performance --performance-mode full
```

Modeling outputs are written to:

```text
outputs/004_modeling/<artifact_namespace>/
```

Current reviewed namespace:
- `outputs/004_modeling/commercial_facility/`

Expected artifact files:
- `metrics_overall.csv` (validation + one-shot holdout test metrics with `mae_pct` and `rmse_pct`)
- `metrics_by_day_class.csv`
- `metrics_by_hour.csv`
- `run_manifest.json`
- `fig_actual_vs_predicted.png`
- `fig_error_by_hour.png`
- `fig_model_comparison.png`
- `fig_day_ahead.png`

Stage-5 performance artifacts are written to timestamped run directories under:

```text
outputs/005_performance/<artifact_namespace>/
```

Current reviewed namespace:
- `outputs/005_performance/commercial_facility/latest/`

Percentage-metric basis used throughout modeling and rollout artifacts:
- `mae_pct` / `rmse_pct` mean `100 * error / mean(abs(actual_load))` over valid rows

`outputs/005_performance/<artifact_namespace>/latest/` is updated to mirror the most recent successful run.

Expected Stage-5 artifact files (per timestamped run directory or under `latest/`):
- `preflight_audit.md`
- `feature_causality_audit.csv`
- `minute_integrity_audit.csv`
- `holdout_lock.json`
- `metrics_fold.csv`
- `metrics_fold_summary.json`
- `selection_scoreboard.csv`
- `blend_finalists.csv`
- `residual_ablation.csv`
- `coverage_audit.csv`
- `promotion_candidate.json`
- `holdout_evaluation.csv`
- `holdout_predictions.csv`
- `holdout_inference.csv`
- `deployment_recommendation.json`
- `feature_importance_permutation.csv`
- `feature_importance_summary.json`
- `hgb_coordinate_summary.csv`
- `guardrail_decisions.csv`
- `guardrail_summary.csv`
- `fig_holdout_benchmark_ci.png`
- `fig_feature_importance.png`
- `holdout_blend_decisions.csv` (when a promoted blend candidate is selected)
- `run_manifest.json`

Stage-5 uses the canonical `full_stable` feature set and adds one derived helper set:
- `curated_ramp`: causal short-horizon ramp indicators layered onto `curated`
- `full_stable`: the high-capacity `full` set with the unstable `rolling_*_240` and
  `rolling_*_1440` windows removed so validation coverage stays comparable across
  Stage-5 and downstream multiresolution runs
Stage-5 full mode also runs HGB coordinate-search regularization and a causal
blend guardrail policy on the best HGB candidate. The resulting Stage-5 artifacts
are summarized inside `notebooks/003_modeling.ipynb` when they are present; a
standalone `004_performance.ipynb` is intentionally no longer used.
If `outputs/004_modeling/<artifact_namespace>/run_manifest.json` is missing, `run_pipeline.py --stage performance`
now bootstraps `scripts/003_create_model_datasets.py` plus `notebooks/003_modeling.ipynb`
before running Stage-5 so the dependency chain stays intact.
Stage-5 fold evaluation is parallelized through the shared runtime in
`config/modeling.toml`, and `outputs/005_performance/<artifact_namespace>/latest/run_manifest.json`
records both the configured runtime and the resolved worker plan used for the run.
Stage-5 now also writes a coverage audit plus a promotion-and-holdout surface:
- `coverage_audit.csv` records split-level feature-set coverage against the Stage-5 promotion threshold.
- `promotion_candidate.json` records the exact promoted candidate chosen from the fold scoreboard.
- `holdout_evaluation.csv` ranks the promoted learned challenger against the strongest short-horizon baselines (`persistence`, `anchored_workday`, `avg_workday`, `previous_day`, and the classical `holt_damped` exponential-smoothing benchmark when it fits successfully).
- `holdout_predictions.csv` preserves the exact one-shot holdout path used by inference, figures, and downstream review.
- `holdout_inference.csv` adds moving-block bootstrap confidence intervals and paired significance tests so the `1m` claim is not based on point estimates alone.
- `holdout_coverage_summary.json` and `holdout_coverage_segments.csv` record whether the latest Stage-5 holdout spans more than one meaningful operating regime before we make standalone `1m` claims from it.
- `supplemental_surface_summary.csv`, `supplemental_surface_operating_regime_evaluation.csv`, and `supplemental_surface_advisory.json` widen that evidence to stitched validate-walkforward plus holdout rows, so we can see where learned `1m` behavior helps without overwriting the canonical deployment gate.
- `deployment_recommendation.json` records whether the learned candidate or the strongest baseline remains the operational winner.
- `feature_importance_permutation.csv` and `feature_importance_summary.json` explain which predictors actually drive the strongest learned challenger on the honest holdout slice.
- `holdout_registry.csv` records the best learned holdout challenger per Stage-5 run so
  downstream nowcast replay can use the full historical evidence surface rather than
  only the mutable `latest/` alias.
- `blend_finalists.csv` records the best validation-selected guarded blend config for
  each shortlisted Stage-5 learned family so Stage-10 can benchmark more than one
  blended `1m` candidate on the exact control window.

Latest reviewed Stage-5 full rerun (`outputs/005_performance/commercial_facility/latest/`):
- full short-horizon surface now includes `minimal_phase_anchor`, `full_stable`,
  `full_stable_legacy`, `curated_ramp`, and `minimal_phase`, with optional
  XGBoost candidates on accelerated hosts alongside the frontier HGB variants
- promoted candidate: `1min/curated_ramp/xgb-balanced/residual+blend`
- fold mean MAE ratio vs persistence: `0.939531`
- mean validation coverage: `0.998032`
- holdout benchmark leaderboard:
  - persistence: `173.724099` (`8.380502%`)
  - learned challenger: `175.055450` (`8.444727%`)
  - anchored_workday: `257.954664` (`12.443809%`)
  - arima: `863.113597` (`41.636855%`)
  - holt_damped: `864.439892` (`41.700836%`)
- paired MAE inference versus persistence:
  - learned minus persistence MAE: `+1.331351 W`
  - 95% moving-block bootstrap CI: `[+0.401274, +2.225375] W`
  - inferred block length: `15` minutes
  - one-sided p-value for the claim "learned MAE < persistence MAE": `1.0000`
- paired RMSE inference versus persistence:
  - learned minus persistence RMSE: `-2.060269 W`
  - 95% moving-block bootstrap CI: `[-3.150256, -0.942326] W`
- exact Stage-5 conclusion: `1m` learned superiority is **not** supported by the
  current holdout evidence; persistence still wins the latest reviewed holdout
- supplemental advisory conclusion: the broader leakage-safe `validate` walk-forward
  plus holdout surface is learned-positive overall (`369.991626` vs persistence
  `383.585219`) and specifically supports learned `1m` behavior in
  `transition_only` and `transition_active`; the strongest supportive slice is
  `actual_ramp_band=high_ramp` at a `0.878199` MAE ratio to persistence, while
  the canonical `none_inactive` holdout regime remains baseline-led, so that
  advisory evidence does not override the standalone deployment gate
- feature-importance summary for the strongest learned challenger:
  - `lag_1` remains the dominant feature
  - the top 5 features account for `94.19%` of positive permutation importance
  - interpretation: the learned gain is still dominated by autocorrelation plus a
    compact ramp/profile feature set rather than broad feature diversity
- best per-family blend finalists are now persisted for downstream benchmarking;
  the latest exact-control replay currently promotes
  `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
  as the strongest operational minute overlay even though the standalone Stage-5
  gate still keeps `persistence` as the deployable `1m` anchor
- best historical Stage-5 learned holdout winner remains preserved in
  `outputs/005_performance/commercial_facility/holdout_registry.csv`:
  `full_stable/hgb-frontier-lr010-l2001/raw+blend` at `163.003809` (`7.863352%`)

## Threats To Validity

- Validation and holdout windows are still short. The canonical modeling split uses 3 validation days and 3 test days, which is enough to surface failure modes but not enough to claim stable seasonal generalization.
- Holiday contamination is real. Christmas and the immediate post-Christmas period materially change both the learned model and the baselines, so validate-to-test shifts should be read as regime shifts, not as clean generalization gains.
- The repo is still single-facility evidence. All current conclusions are for `commercial_facility`; other load types need their own reruns, registries, and control backtests.
- Minute-level autocorrelation is the core reason `1m` is hard. The new moving-block bootstrap improves statistical rigor, but it does not change the underlying fact that persistence is an unusually strong short-horizon baseline on this data.

## Stage-6 Multi-Resolution Comparison

Stage-6 compares resolutions with two distinct views:
- `native_step`: one-step metrics at each resolution's own cadence
- `matched_horizon`: recursive endpoint/path evaluation on a shared real-time horizon

Enabled Stage-6 resolutions now include:
- second-level: `1s`, `5s`, `10s`, `30s`
- minute-level: `1min`, `5min`, `10min`, `15min`

To keep routine iteration practical, Stage-6 profiles remain scoped:
- `smoke` defaults to `30s` + `1min`
- `candidate` defaults to `10s`, `30s`, `1min`, `5min`, `10min`, `15min`
- `focus_60m` narrows to the strongest current 60-minute contenders: `30s`, `1min`, `5min`, `10min`
- `full` enables the entire configured set, including `1s` and `5s`

CLI `--resolution` overrides can still narrow or explicitly target any enabled cadence.

The selection surface is intentionally conservative:
- persistence is always present as the anchor baseline
- non-persistence candidates (learned or baseline) must clear coverage, stability, runtime, and practical-gain gates
- the selection summary can explicitly keep persistence when no alternative candidate is good enough
Stage-6 uses the same shared joblib runtime from `config/modeling.toml` for
native-step and matched-horizon task grids, and each manifest records the
resolved plan per resolution/job type.
Matched-horizon learned candidates now evaluate both `recursive` and
`direct_endpoint` strategies, and `winner_forecast_strategy` records which
strategy actually cleared the gates.

Stage-6 artifacts are written to:

```text
outputs/006_multires/<artifact_namespace>/
```

Expected Stage-6 artifact files:
- `run_manifest.json`
- `fold_metrics.csv`
- `native_step_metrics.csv`
- `matched_horizon_metrics.csv`
- `resolution_health.csv`
- `origin_metrics.csv`
- `selection_summary.csv`
- `selection_summary.md`
- `winner_registry.csv`
- `fig_runtime_vs_gain.png`
- `fig_resolution_pareto.png`

Stage-6 runtime behavior is now more defensive:
- missing configured gold inputs are skipped with explicit `skipped_missing_resolution:*` manifest warnings instead of crashing the whole run
- raw `fold_metrics.csv` and `origin_metrics.csv` are deduplicated before write so baseline rows have one-row-per-candidate semantics
- `selection_summary.csv` and `winner_registry.csv` now include the selected winner's raw endpoint/path MAE plus normalized MAE percentages
- `winner_registry.csv` records cross-run Stage-6 winners so `latest/` remains a convenience alias rather than the only winner source

Key current Stage-6 evidence:
- Latest smoke run (`outputs/006_multires/commercial_facility/latest_smoke/`, timestamped source `20260309T214557091913Z/`):
  - `15m`: learned winner at `30s/minimal/ridge-medium/direct_endpoint`
  - `60m`: learned winner at `1min/minimal/hgb-balanced/recursive`
- Candidate sweep (`outputs/006_multires/20260307T133220706885Z/`):
  - `30m`: learned winner at `5min/curated/hgb-balanced/direct_endpoint`
  - `120m`: learned winner at `5min/curated/hgb-balanced/recursive`
- Focused 60-minute run (`outputs/006_multires/commercial_facility/20260310T005916684602Z/`):
  - `60m`: learned winner `5min/minimal/hgb-balanced/recursive`
  - endpoint/path MAE: `1148.166851` (`42.691558%`) / `1151.446627` (`36.611950%`)
  - MAE ratio to persistence: `0.654046`
  - the earlier baseline-led result was stale; current focused evidence is learned-positive

That is the current evidence-backed H3 position: learned candidates now justify
promotion at `15m`, `30m`, `60m`, and `120m`, but `60m` is still profile-sensitive
because candidate mode can still fall back to persistence. Keep the gating logic explicit.

## Stage-7 Recursive Rollout

Stage-7 evaluates a selected or explicitly requested learned candidate as a true
multi-step forecast path. It is intentionally separate from the `1min` MVP notebook so
single-step nowcasting metrics are not conflated with rollout metrics.

Stage-7 artifacts are written to:

```text
outputs/007_rollout/<load_type>/
```

Expected Stage-7 artifact files:
- `run_manifest.json`
- `recursive_rollout_metrics.csv`
- `recursive_rollout_by_origin.csv`
- `rollout_selection_summary.csv`
- `rollout_selection_summary.md`
- `rollout_policy_candidates.json` for short-horizon runs that derive phase-bucket policies
- `rollout_health.csv`
- `rollout_registry.csv`
- `selection_context.json`
- `fig_rollout_paths.png`
- `fig_rollout_error_by_origin.png`

Short-horizon Stage-7 outputs now include explicit correction-window and profile metrics:
- `recursive_rollout_metrics.csv` includes `phase_mean_mae`, `next_lock_mae`,
  `profile_shape_mae`, and their percentage counterparts
- `rollout_selection_summary.csv` now includes objective rows for
  `phase_mean_mae`, `next_lock_mae`, and `profile_shape_mae`
- Stage-7 and Stage-8 now support `origin_policy=billing_aligned` and
  `origin_policy=phase_balanced`; `phase_balanced` spreads origins across the full
  15-minute phase cycle so short-horizon robustness can be measured outside only
  quarter-hour starts
- The default rollout and horizon-curve configs now use `origin_policy=auto` and
  `selection_target=auto`, so `<=15m` requests resolve to the centralized short-horizon
  policy in `config/modeling.toml`: `phase_balanced` origins plus `next_lock_mae`,
  while day-ahead requests resolve to `profile_shape_mae`
- Stage-7 can now evaluate multiple residual baselines from the centralized horizon
  policy surface. Short-horizon runs now keep `persistence_residual` and
  `avg_workday_residual` available without one-off code edits.
- Short-horizon learned blends now use a small history-guided local refinement around
  the best previously measured end weight instead of only the fixed `0.10/default/hybrid`
  endpoints. This is how the current `e40` billing winner replaced the earlier `e35`.

Challenger-sweep artifacts are written to:

```text
outputs/007_rollout/<load_type>/challenger_sweeps/
```

Expected challenger-sweep files:
- `candidate_plan.csv`
- `candidate_results.csv`
- `challenger_sweep_registry.csv`
- `challenger_summary.md`
- `recommended_candidate.json`
- `run_manifest.json`

Stage-7 now reuses a Stage-6 learned winner only when the requested rollout horizon
exactly matches the learned winner horizon. If there is no exact-horizon learned winner,
Stage-7 resolves candidates in this order:
- explicit rollout candidate overrides (`--resolution`, `--feature-set`, `--model-label`)
- explicit Stage-6 run pin (`--selection-run-id`)
- objective-aware `outputs/007_rollout/<artifact_namespace>/challenger_sweep_registry.csv`
- `outputs/006_multires/<artifact_namespace>/winner_registry.csv`
- legacy `outputs/006_multires/<artifact_namespace>/latest/selection_summary.csv`
- objective-aware `outputs/007_rollout/<artifact_namespace>/rollout_registry.csv`
- configured rollout fallback in `config/multires.toml`

`selection_context.json` records both the source and the reason, plus the requested
`selection_target` (`endpoint_mae`, `path_mae`, `phase_mean_mae`, `next_lock_mae`,
or `profile_shape_mae`). Endpoint-only
Stage-6 winners (`winner_forecast_strategy=direct_endpoint`) are intentionally excluded
from Stage-7 reuse because rollout still requires a recursive path generator.
The E2E wrapper mirrors that separation: `--with-rollout` no longer implies
`--with-multires`, so rollout-only verification does not rewrite Stage-6 artifacts
unless multires is explicitly requested.
Stage-7 challenger sweeps now also persist `shared_origins.csv` and evaluate every
candidate on the same origin timestamps across resolutions, so cross-candidate
recommendations are no longer biased by different start-time samples.
For hourly horizons, the sweep can also synthesize cross-candidate
`portfolio_policy_candidates.json` and `portfolio_policy_by_origin.csv` artifacts when
different learned candidates win different objectives on the shared-origin surface.
When a sweep-derived portfolio candidate wins, Stage-7 can now replay that policy as a
standalone rollout run with `resolution=mixed`, `feature_set=portfolio`, and
`model_label=cross_candidate_portfolio`, while preserving the source sweep metadata in
`selection_context.json`, `portfolio_policy_candidate.json`, and `shared_origins.csv`.

Current validated `1440m` rollout result
(`outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T231235398730Z/`):
- rollout manifest records `origin_policy=uniform` with
  `origin_selection_scope=shared_timestamp_intersection` across `8` shared origins
- Stage-7 selection target: `profile_shape_mae`
- challenger recommendation:
  `outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T231235398730Z/recommended_candidate.json`
- learned winner: `10min/minimal/hgb-balanced::raw`
- learned profile-shape / path / endpoint MAE:
  `717.7776` (`36.2451%`) / `783.0771` (`39.5425%`) / `968.9096` (`44.1622%`)
- persistence profile-shape / path / endpoint MAE:
  `746.5271` (`37.6968%`) / `1010.6207` (`51.0326%`) / `1119.1373` (`51.0094%`)
- best baseline on profile shape: `persistence` at `746.5271` (`37.6968%`)
- best baseline on path / endpoint: `avg_workday` at `850.1457` (`42.9292%`) /
  `986.6763` (`44.9720%`)
- interpretation: the current day-ahead autoselection is no longer a generic path-first
  fallback. It explicitly targets profile shape, and the same `10min/minimal`
  `hgb-balanced::raw` rollout now beats persistence on profile-shape MAE while also
  remaining better than the strongest baseline on path and endpoint MAE.

Current validated `15m` short-horizon correction readout
(`outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T234800852734Z/`):
- current auto objective: `origin_policy=phase_balanced`, `selection_target=next_lock_mae`
- sweep methodology: `origin_selection_scope=shared_timestamp_intersection`
- challenger recommendation:
  `outputs/007_rollout/commercial_facility/challenger_sweeps/20260310T234800852734Z/recommended_candidate.json`
- learned winner: `5min/minimal/hgb-balanced::phase_bucket_next_lock_policy`
- learned next-lock / path / phase-average MAE:
  `293.8907` (`13.3804%`) / `293.8907` (`13.3804%`) / `231.8214` (`10.5541%`)
- learned profile-shape / endpoint MAE:
  `151.0226` (`6.8760%`) / `308.9010` (`13.6885%`)
- persistence next-lock MAE: `389.4395` (`17.7314%`)
- best baseline next-lock MAE: `persistence` at `389.4395` (`17.7314%`)
- interpretation: the current short-horizon default is no longer the billing-aligned
  anchored residual family. Under the broader phase-balanced correction objective and
  shared-origin comparison, the best measured result is now an auditable
  `phase_bucket_next_lock_policy` synthesized from the evaluated 15-minute candidate
  surface.

Current validated `60m` operational rollout readout
(`outputs/007_rollout/commercial_facility/20260311T015133422915Z/`):
- auto objective: `origin_policy=phase_balanced`, `selection_target=next_lock_mae`
- selection source: `outputs/007_rollout/commercial_facility/challenger_sweep_registry.csv`
- standalone replay of sweep-derived portfolio:
  `cross_candidate_portfolio::phase_bucket_next_lock_policy`
- learned next-lock / path / profile-shape MAE:
  `253.1043` (`15.9698%`) / `496.8937` (`24.2527%`) / `256.4466` (`12.5625%`)
- persistence next-lock / path / profile-shape MAE:
  `379.1165` (`16.7331%`) / `305.4445` (`14.5383%`) / `214.9170` (`10.8416%`)
- best baseline next-lock MAE: `hybrid_workday` at `375.9297` (`16.4835%`)
- interpretation: Stage-7 no longer leaves the best measured `60m` correction policy
  stranded in sweep outputs. The repo now replays the shared-origin portfolio winner as
  a normal rollout artifact while keeping its sweep provenance attached.

The older billing-aligned `15m` result is still useful as a secondary readout:
- run:
  `outputs/007_rollout/commercial_facility/20260310T085909851743Z/rollout_selection_summary.csv`
- billing-aligned phase winner:
  `5min/full/hgb-frontier-lr010-leaf100::anchored_workday_residual`
- phase-average MAE: `315.1285` (`14.4775%`) vs persistence `556.0363` (`25.5452%`)
- interpretation: billing-aligned quarter-hour behavior still favors the anchored
  residual family, but the repo default now prioritizes the broader phase-balanced
  correction objective unless an explicit billing-aligned audit is requested.

## Stage-8 Horizon Degradation Curve

Stage-8 converts the separate nowcast, matched-horizon, and long-horizon rollout
results into the H5 degradation curve. It does not introduce a new model family; it
characterizes where the current learned stack is actually better or worse than
baseline.

Stage-8 artifacts are written to:

```text
outputs/009_horizon_curve/<artifact_namespace>/
```

Expected Stage-8 artifact files:
- `run_manifest.json`
- `horizon_curve_summary.csv`
- `horizon_curve_candidates.csv`
- `horizon_curve_summary.md`
- `crossover_summary.json`
- `fig_horizon_ratio_curve.png`
- `fig_horizon_absolute_mae.png`

Current validated H5 readout
(`outputs/009_horizon_curve/commercial_facility/20260312T065037307284Z/`):
- methodology:
  - `1m` uses the Stage-5 holdout anchor
  - `15m` and `60m` reuse Stage-7 sweeps under `origin_policy=phase_balanced`
    with `selection_target=next_lock_mae`
  - `1440m` reuses the Stage-7 sweep under `origin_policy=uniform`
    with `selection_target=profile_shape_mae`
  - reused Stage-7 sweeps are now ranked with a preference for
    `origin_selection_scope=shared_timestamp_intersection`, so older pre-fix sweep
    artifacts do not outrank comparable post-fix runs
- interpretation rule: this is a capability envelope, not a single-model monotonic
  decay trace, because the best verified candidate can change by horizon and by
  objective
- current objective-aware outcome versus persistence:
  - `1m`: learned superiority is not supported on the current holdout slice
    (`174.8918` vs `173.7241`)
  - `15m`: learned next-lock MAE `293.8907` (`13.3804%`) beats persistence
    `389.4395` (`17.7314%`) and best baseline `389.4395` (`17.7314%`)
  - `60m`: learned next-lock MAE `253.1043` (`15.9698%`) from
    `cross_candidate_portfolio::phase_bucket_next_lock_policy` beats persistence
    `379.1165` (`16.7331%`) and best baseline `375.9297` (`16.4835%`), but path and
    profile-shape MAE still trail the best baselines
  - `1440m`: learned profile-shape MAE `717.7776` (`36.2451%`) beats persistence
    `746.5271` (`37.6968%`) while path MAE `783.0771` also stays ahead of
    `avg_workday` `850.1457`

That is the current evidence-backed H5 position: there is no single crossover horizon.
The repo now exposes a non-monotonic, objective-aware capability envelope where the
near-term lock window and the day-ahead profile can both be improved, while the `1m`
anchor remains the main outstanding weak point.

## Stage-10 Forecast Control

Stage-10 turns the separate day-ahead, hourly, and `15m` winners into one
forecast-control backtest. It replays the current measured policies on shared 24-hour
control cycles, then measures whether intraday corrections actually reduce locked
15-minute error and profile-shape error versus a frozen day-ahead forecast.
Stage-10 now also benchmarks the replayed layer candidates on those exact control
cycles and may promote a stronger baseline inside the backtest when that is what the
evidence supports.

Methodology note:
- the current `latest/` bundle already includes:
  - held-out layer promotion
  - real transition-mismatch refresh triggers
  - denser exact-control `1m` blend search
  - exact-origin replay reuse plus in-process Stage-7 context reuse
- the latest implementation notes and longer findings log still live in
  [`personal/improvement.md`](personal/improvement.md).

Current validated Stage-10 readout
(`outputs/010_forecast_control/commercial_facility/latest/`):
- control policy after exact-origin replay benchmarking across all eligible
  out-of-sample validate/test control cycles on the configured schedules:
  - day-ahead: `10min/minimal/hgb-balanced::raw`
  - hourly: `10min/minimal/hybrid_workday` selected over the upstream
    `mixed/portfolio/cross_candidate_portfolio::phase_bucket_next_lock_policy`
  - isolated phase benchmark winner:
    `5min/full/hybrid_workday`
    selected over the upstream
    `1min/minimal_phase/hgb-balanced::phase_bucket_next_lock_policy`
  - exact stack-aware phase benchmarking still explores candidates such as
    `phase_bucket_portfolio::stack_origin_metric_policy`, and the latest
    exact stack guard now keeps that stack-aware phase candidate live because it
    cleared the next-lock / peak / profile / optimizer guardrail bundle on the
    held-out and rolling control surfaces
  - `1m` nowcast:
    `curated_ramp/xgb-balanced/residual+blend|control_blend_w0.02`
    selected from the exact-control Stage-5 candidate pool
  - the `1m` exact-control pool now mixes historical holdout winners, latest
    `blend_finalists.csv`, and the remaining raw/residual scoreboard challengers
  - the Stage-10 code path now also uses the broader Stage-5 supplemental
    advisory surface as a near-tie breaker when exact-control minute-overlay
    candidates are effectively indistinguishable; that logic is intentionally a
    tie-break, not a hard override
  - the repo now also persists a dynamic minute-overlay shadow analysis; on the
    latest cold Stage-10 bundle, enforcing that gate would have worsened
    all-interval selected absolute error from `47.5035` to `417.1639`, so the
    dynamic controller remains shadow-only for now
  - the repo now also persists a soft minute-overlay shadow search; on the
    latest cold bundle it evaluated `29` soft policies and still selected pure
    nowcast (`soft_overlay_sw100_bw100`) as the best admissible policy, so the
    live minute layer remains unchanged
  - Stage-5 holdout still does not support a blanket learned-superiority claim
    at `1m`, but the current exact-control Stage-10 nowcast surface now selects a
    learned XGBoost control blend over persistence by a large operational margin
- exact-control scope:
  - calibration cycles: `8`
  - evaluation cycles: `8`
- current evidence index artifact:
  `current_evidence_index.md`
- optimizer-serving preview artifact:
  `optimizer_delivery_serving_preview.csv`
- dynamic minute-overlay shadow artifact:
  `optimizer_dynamic_overlay_shadow_summary.json`
- Stage-10 candidate benchmark artifact:
  `control_layer_candidate_benchmarks.csv`
- Stage-10 phase guard artifact:
  `phase_stack_guard_summary.csv`
- Stage-10 runtime telemetry artifacts:
  `runtime_profile.csv` and `runtime_summary.json`
- Stage-10 exact-origin replay cache artifact:
  `outputs/010_forecast_control/commercial_facility/replay_cache/replay_cache_registry.csv`
- frozen day-ahead 15-minute lock MAE: `767.4113` (`40.5182%`)
- after hourly updates 15-minute lock MAE: `490.4285` (`25.1607%`)
- after phase updates 15-minute lock MAE: `417.2299` (`21.4431%`)
- after nowcast updates 15-minute lock MAE: `47.5035` (`2.4591%`)
- frozen day-ahead profile-shape MAE: `788.5337` (`41.1853%`)
- after hourly updates profile-shape MAE: `626.6816` (`32.1760%`)
- after phase updates profile-shape MAE: `570.4450` (`29.2754%`)
- after nowcast updates profile-shape MAE: `174.9563` (`9.0387%`)
- rolling benchmark scope:
  - calibration cycles: `16`
  - evaluation cycles: `16`
  - hourly layer remains statistically useful on the broader rolling surface:
    lock-MAE gain `271.7613` with 95% CI `[146.5097, 379.6016]`, and
    profile-shape gain `159.4673` with 95% CI `[80.1659, 228.3271]`
  - the current applied phase slot is back to a distinct stack-aware correction
    on the latest cold bundle: rolling `phase_vs_hourly` lock gain is
    `221.9810` with 95% CI `[187.0417, 254.1896]`, while next-lock remains flat
  - the nowcast layer remains the strongest stack improvement:
    rolling lock-MAE gain `222.7204` with 95% CI `[217.6082, 228.3587]`
- day-ahead refresh study:
  - refresh candidate: `10min/minimal/hgb-balanced::hybrid_workday_residual`
  - frozen day-ahead lock MAE: `767.4113` (`40.5182%`)
  - unconditional refresh lock MAE: `606.6037` (`31.6781%`)
  - triggered refresh lock MAE: `655.3852` (`33.8058%`)
  - frozen day-ahead profile-shape MAE: `788.5337` (`41.1853%`)
  - unconditional refresh profile-shape MAE: `701.8624` (`36.4205%`)
  - triggered refresh profile-shape MAE: `732.5164` (`37.7821%`)
  - mean triggered refresh updates applied per cycle: `8.75`
  - trigger rate on the exact evaluation cycles: `0.3804`
  - rolling trigger rate: `0.3838`
  - selected trigger mode: `residual_or_activity_active_or_transition`
  - triggered refresh preserved `64.63%` of the unconditional profile-shape gain and
    `69.66%` of the unconditional lock-MAE gain
  - on the rolling evaluation surface, triggered refresh also remains competitive:
    lock/profile-shape MAE `649.4860` / `730.1816`
  - current trigger reason mix is now split across residual drift and activity-profile shift
  - current operating recommendation is `triggered_refresh`
- latest focused `1440m` challenger sweep
  (`outputs/007_rollout/commercial_facility/challenger_sweeps/20260320T090013545419Z/`):
  - standalone winner remains `10min/minimal/hgb-balanced::raw`
  - profile-shape MAE `717.7776` (`36.2451%`)
  - `10min/minimal/hgb-balanced::hybrid_workday_residual` is worse as a standalone
    full-day rollout on the same objective
  - interpretation: `hybrid_workday_residual` is useful as a refresh path layered
    onto a frozen day-ahead profile, not as the current best standalone `24h` rollout
- interpretation: the current exact-control evidence is materially stronger than the
  earlier throughput-limited readouts. The true control stack is now best
  described as learned day-ahead, learned hourly correction, a distinct
  stack-aware `15m` phase correction, and a learned exact-control `1m`
  XGBoost control-blend nowcast. Stage-10 now
  replays all eligible out-of-sample validate/test control cycles on the
  configured schedules; the hourly, phase, and nowcast layers are all doing
  meaningful operational work on the latest cold bundle. The main remaining
  questions are therefore narrower: whether uncertainty can be tightened
  without losing honesty, whether a
  learned `1m` anchor can finally beat persistence on the honest Stage-5
  holdout surface. The code now also carries a broader-evidence minute-overlay
  tie-break for near-equal Stage-10 candidates plus a shadow-only dynamic
  minute controller plus the first soft minute-overlay shadow search. The
  latest cold rerun validated the tie-break path, showed that hard dynamic
  gating is not ready yet, and also showed that background softening does not
  beat pure nowcast on the same replay surface. The latest runtime profile is
  still far better than the old multi-thousand-second baseline; the current
  cold bundle landed at about `722.54s` wall clock, with
  `select_phase_stack_policy` now the dominant hotspot at `174.92s`.

Stage-10 visuals from the latest run:

> **Note:** The figures below are generated by running the pipeline stages and
> are not stored in version control. Run the relevant stage scripts to produce
> them locally, or view the latest archived notebook snapshots in
> `outputs/008_notebook_runs/`.

![Stage-10 locked-interval MAE progression](outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png)

![Stage-10 example control cycle](outputs/010_forecast_control/commercial_facility/latest/fig_control_example_cycle.png)

![Stage-10 day-ahead refresh policy comparison](outputs/010_forecast_control/commercial_facility/latest/fig_day_ahead_refresh_policy.png)

![Stage-10 rolling lock-MAE distribution](outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_distribution.png)

![Stage-10 rolling gain confidence intervals](outputs/010_forecast_control/commercial_facility/latest/fig_control_layer_gain_ci.png)

![Stage-10 phase stack frontier](outputs/010_forecast_control/commercial_facility/latest/fig_phase_stack_candidates.png)

## Tooling Files

- `pyproject.toml` is intentionally at repository root because Python packaging and tool
  discovery (PEP 518/621, pytest, coverage, pip editable install) resolve from root.
- `pyrightconfig.json` is intentionally at repository root because pyright uses root
  config auto-discovery and workspace-relative import resolution from that location.
- The `config/` folder is reserved for runtime pipeline/EDA TOML settings:
  `config/pipeline.toml`, `config/eda.toml`, `config/modeling.toml`, and
  `config/multires.toml`.

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
python scripts/write_validation_snapshot.py
```

Retain notebook cell outputs only when explicitly needed:

```bash
python scripts/validate_notebooks.py --keep-output
```

Notebook validation now archives each executed notebook snapshot under
`outputs/008_notebook_runs/<run_id>/` before tracked outputs are cleared.
`outputs/008_notebook_runs/latest/` mirrors the newest successful archive run and
its manifest records notebook path, execution timestamp, profile index, env overrides,
and whether the tracked notebook was later cleaned.
Before `notebooks/003_modeling.ipynb` is executed, the validator now refreshes
silver, gold, and model datasets so notebook validation does not depend on stale
intermediate artifacts.

Latest validated notebook archive
(`outputs/008_notebook_runs/commercial_facility/20260322T042828995952Z/`):
- all core notebooks (`000` through `003`) executed successfully with `clear_outputs=true`
- `003_modeling.ipynb` artifact validation confirmed:
  - `metrics_overall.csv`: `57` rows
  - `metrics_by_day_class.csv`: `98` rows
  - `metrics_by_hour.csv`: `1176` rows
  - `fig_actual_vs_predicted.png`: `1990x772`
  - `fig_error_by_hour.png`: `1792x814`
  - `fig_model_comparison.png`: `1966x1134`
  - `fig_day_ahead.png`: `1990x772`

## Latest Validated Visuals

Decision-facing figures now have current validated sources:
- Stage-4 modeling notebook:
  `outputs/004_modeling/commercial_facility/fig_actual_vs_predicted.png`,
  `fig_error_by_hour.png`, `fig_model_comparison.png`, and `fig_day_ahead.png`
- Stage-8 horizon characterization:
  `outputs/009_horizon_curve/commercial_facility/latest/fig_horizon_ratio_curve.png`
  and `fig_horizon_absolute_mae.png`
- Stage-10 control backtest:
  `outputs/010_forecast_control/commercial_facility/latest/fig_control_lock_mae.png`
  and `fig_control_example_cycle.png`
  and `fig_day_ahead_refresh_policy.png`

Latest verification snapshot:
- `python scripts/validate_notebooks.py` -> success; latest archive
  `outputs/008_notebook_runs/commercial_facility/20260322T042828995952Z/`
- `python scripts/write_validation_snapshot.py` -> success; canonical current-state page
  `docs/003_modeling/current_validation_snapshot.md`
- `pytest -q` -> success (`248.47s`)
- `python scripts/run_e2e.py --mode quick --with-multires --with-rollout --with-rollout-sweep --with-horizon-curve --with-forecast-control`
  -> success (`2655.07s` total; pipeline `1951.19s`, notebooks `455.41s`,
  pytest `248.47s`)
- `python scripts/modeling/forecast_control_backtest.py`
  -> success with fresh fully persisted Stage-10 bundle under the held-out
  promotion logic, rolling benchmark inference, and current evidence index;
  latest bundle `outputs/010_forecast_control/commercial_facility/20260322T030301040853Z/`

## Documentation Index

| Document | Description | Location |
|----------|-------------|----------|
| Execution Specification | Canonical infrastructure hardening spec with phase definitions and acceptance criteria | [000_spec.md](docs/000_governance/000_spec.md) |
| Notebook Configurability Spec | Notebook configuration cells, resolution selection, self-optimizing parameters | [001_spec.md](docs/000_governance/001_spec.md) |
| Operating Direction Spec | Current optimizer-facing direction, retired bets, source credit, and active phases | [002_operating_direction_spec.md](docs/000_governance/002_operating_direction_spec.md) |
| Architecture Overview | High-level data architecture, data flow diagrams, resolution policy, shared infrastructure | [architecture.md](docs/001_architecture/000_overview/architecture.md) |
| Raw Layer | Source data format, MATLAB payload structure, validation assumptions | [raw.md](docs/001_architecture/001_raw/raw.md) |
| Bronze Layer | Format conversion logic, schema, error handling, logging | [bronze.md](docs/001_architecture/002_bronze/bronze.md) |
| Silver Layer | Resampling, feature engineering, 48-column schema, NaN handling, performance | [silver.md](docs/001_architecture/003_silver/silver.md) |
| Gold Layer | Null filtering, model-readiness validation, determinism guarantees | [gold.md](docs/001_architecture/004_gold/gold.md) |
| Model Layer | Chronological splitting, feature-set selection, target separation, leakage prevention | [model.md](docs/001_architecture/005_model/model.md) |
| Pipeline Operations | Layer-by-layer pipeline reference, resolution policy, orchestration commands | [pipeline.md](docs/002_pipeline/pipeline.md) |
| Stage Map | Plain-English map from stable stage ids to what each stage actually answers | [stage_map.md](docs/002_pipeline/stage_map.md) |
| Step-by-Step Plan | Detailed processing steps from raw through evaluation | [plan.md](docs/002_pipeline/plan.md) |
| Feature Sets | Canonical feature set definitions with rationale, risks, and hypothesis connections | [feature_sets.md](docs/003_modeling/feature_sets.md) |
| Hypotheses | Historical notebook-era hypotheses retained for provenance and retirement context | [hypothesis.md](docs/003_modeling/hypothesis.md) |
| Operational Hypotheses | Current optimizer-facing hypotheses that govern selection and delivery | [operational_hypotheses.md](docs/003_modeling/operational_hypotheses.md) |
| MVMP Scope | First-pass modeling target with success criteria and persistence baseline | [mvmp.md](docs/003_modeling/mvmp.md) |
| Current Validation Snapshot | Generated one-page latest validated state sourced from current artifacts | [current_validation_snapshot.md](docs/003_modeling/current_validation_snapshot.md) |
| Current Operating Approach | Plain-English summary of the decision stack and control surface | [current_operating_approach.md](docs/003_modeling/current_operating_approach.md) |
| Model and Blend Guide | Plain-English decoder for model labels, blend wrappers, stage winners, and current compute policy | [model_and_blend_guide.md](docs/003_modeling/model_and_blend_guide.md) |
| Optimizer Delivery Contract | Current interval-feed contract, freshness fields, fallback behavior, and uncertainty surface | [optimizer_delivery_contract.md](docs/003_modeling/optimizer_delivery_contract.md) |
| Report IV Run Summary | Measured outcomes from latest executed 1min MVP run and hypothesis snapshot | [report_iv_run_summary.md](docs/003_modeling/report_iv_run_summary.md) |
| Report IV Success Scorecard | Stage-by-stage success readout against the repository success criteria | [report_iv_success_scorecard.md](docs/003_modeling/report_iv_success_scorecard.md) |
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
|   |-- eda.toml                         Notebook visualization and analysis defaults
|   |-- modeling.toml                    Shared Stage-5/Stage-6 execution runtime
|   `-- multires.toml                    Stage-6/Stage-8 multiresolution, rollout, and horizon-curve config
|-- data/
|   |-- 000_raw/                         Raw MATLAB source data (read-only)
|   |-- 001_bronze/                      Long-format second-level parquet
|   |-- 002_silver/                      Multi-resolution feature-engineered parquet
|   |-- 003_gold/                        Null-filtered model-ready parquet
|   `-- 004_model/                       Train/validate/test splits by feature set
|-- docs/
|   |-- 000_governance/
|   |   |-- 000_spec.md                   Canonical infrastructure hardening specification
|   |   |-- 001_spec.md                   Notebook configurability specification
|   |   `-- 002_operating_direction_spec.md Active optimizer-facing direction spec
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
|   |   |-- hypothesis.md                 Historical notebook-era hypotheses
|   |   |-- operational_hypotheses.md     Current optimizer-facing hypotheses
|   |   |-- mvmp.md                       Minimum Viable Modeling Product scope
|   |   |-- current_validation_snapshot.md Generated current-state validation summary
|   |   |-- current_operating_approach.md Plain-English operating approach summary
|   |   |-- model_and_blend_guide.md      Candidate-label, blend, and winner decoder
|   |   |-- optimizer_delivery_contract.md Optimizer-facing interval delivery contract
|   |   |-- report_iv_run_summary.md      Latest Report IV run outcomes and interpretation
|   |   `-- report_iv_success_scorecard.md Stage-by-stage success readout
|   |-- 004_reference/
|   |   `-- glossary.md                   Shared terminology
|   `-- change logs/
|       |-- 000spec/changelog.md          SPEC-000 implementation history
|       `-- 001spec/changelog.md          SPEC-001 implementation history
|-- notebooks/
|   |-- 000_raw_eda.ipynb                  Raw data exploratory analysis
|   |-- 001_bronze_eda.ipynb               Bronze data exploratory analysis
|   |-- 002_silver_eda.ipynb               Silver data exploratory analysis
|   `-- 003_modeling.ipynb                 Report IV modeling plus optional Stage-5 artifact summary
|-- outputs/
|   |-- 004_modeling/                     Generated modeling metrics, figures, and run manifest
|   |-- 005_performance/                  Generated preflight/walk-forward/residual performance artifacts
|   |-- 006_multires/                     Generated matched-horizon multiresolution comparison artifacts
|   |-- 007_rollout/                      Generated recursive rollout artifacts
|   |-- 008_notebook_runs/                Archived executed notebook snapshots and manifests
|   |-- 009_horizon_curve/                Generated H5 horizon-degradation curve artifacts
|   `-- 010_forecast_control/             Generated Stage-10 forecast-control backtest artifacts
|-- scripts/
|   |-- _compat.py                        Wrapper loader used by stable numbered entrypoints
|   |-- bootstrap_env.py                  Compatibility wrapper for tooling/bootstrap_env.py
|   |-- config.py                         Centralized paths, schemas, feature config
|   |-- modeling/                         Canonical Stage-5 through Stage-10 implementations
|   |   |-- common.py                    Shared modeling constants and helpers
|   |   |-- feature_engineering.py       Extended feature engineering for advanced stages
|   |   |-- metrics.py                   Shared metric computations
|   |   |-- parallel.py                  Shared joblib runtime planner/executor
|   |   |-- model_performance.py         Stage-5 walk-forward validation
|   |   |-- multires.py                  Multi-resolution utilities
|   |   |-- multires_compare.py          Stage-6 multi-resolution comparison
|   |   |-- recursive_rollout.py         Stage-7 recursive rollout
|   |   |-- rollout_challenger_sweep.py  Stage-7 challenger sweep
|   |   |-- horizon_curve.py             Stage-8 horizon degradation curve
|   |   `-- forecast_control_backtest.py Stage-10 forecast-control backtest
|   |-- stages/                           Canonical bronze/silver/gold/model stage implementations
|   |-- tooling/                          Canonical environment, notebook, and E2E tooling
|   |-- utils.py                          Shared feature engineering utilities
|   |-- 000_raw_to_bronze.py              Compatibility wrapper for stages/raw_to_bronze.py
|   |-- 001_bronze_to_silver.py           Compatibility wrapper for stages/bronze_to_silver.py
|   |-- 002_silver_to_gold.py             Compatibility wrapper for stages/silver_to_gold.py
|   |-- 003_create_model_datasets.py      Compatibility wrapper for stages/create_model_datasets.py
|   |-- 004_model_performance.py          Compatibility wrapper for modeling/model_performance.py
|   |-- 005_multires_compare.py           Compatibility wrapper for modeling/multires_compare.py
|   |-- 006_recursive_rollout.py          Compatibility wrapper for modeling/recursive_rollout.py
|   |-- 007_rollout_challenger_sweep.py   Compatibility wrapper for modeling/rollout_challenger_sweep.py
|   |-- 008_horizon_curve.py              Compatibility wrapper for modeling/horizon_curve.py
|   |-- 009_forecast_control_backtest.py  Compatibility wrapper for modeling/forecast_control_backtest.py
|   |-- run_e2e.py                        Compatibility wrapper for tooling/run_e2e.py
|   `-- validate_notebooks.py             Compatibility wrapper for tooling/validate_notebooks.py
|-- run_e2e.sh                            Unix-like E2E wrapper
|-- run_e2e.ps1                           PowerShell E2E wrapper
|-- tests/
|   |-- conftest.py                       Pytest fixtures (synthetic data)
|   |-- unit/
|   |   |-- test_config.py                Configuration validation tests
|   |   |-- test_bootstrap_env.py         Bootstrap dependency-helper tests
|   |   `-- test_feature_engineering.py   Shared utility/feature tests
|   |-- stages/
|   |   |-- test_raw_to_bronze.py         Bronze ingestion tests
|   |   |-- test_bronze_to_silver.py      Silver transformation tests
|   |   |-- test_silver_to_gold.py        Gold validation tests
|   |   `-- test_model_datasets.py        Split and leakage tests
|   |-- orchestration/
|   |   |-- test_run_pipeline.py          Orchestrator CLI and error-path tests
|   |   |-- test_run_e2e.py              Repository E2E runner tests
|   |   `-- test_cleanup_outputs.py      Output cleanup behavior tests
|   |-- integration/
|   |   |-- test_integration.py           End-to-end synthetic pipeline tests
|   |   `-- test_multires_scripts.py      Stage-6/Stage-8 CLI integration tests
|   |-- modeling/
|   |   |-- test_multires_time_alignment.py     Multires time-alignment and NaN-handling tests
|   |   |-- test_recursive_rollout_logic.py     Recursive rollout helper tests
|   |   |-- test_parallel_runtime.py            Parallel runtime planner tests
|   |   |-- test_multires_mode_profiles.py      Resolution mode profile tests
|   |   |-- test_multires_selection_logic.py    Selection gate and winner tests
|   |   |-- test_rollout_challenger_sweep.py    Challenger sweep tests
|   |   |-- test_horizon_curve.py               Horizon degradation curve tests
|   |   `-- test_forecast_control_backtest.py   Forecast-control backtest tests
|   |-- governance/
|   |   `-- test_no_hardcoded_paths.py   Path hygiene enforcement tests
|   |-- notebooks/
|   |   |-- test_validate_notebooks.py    Notebook runner behavior tests
|   |   `-- test_notebook_structure.py    Notebook quality guard tests
|   `-- performance/
|       `-- test_model_performance.py     Stage-5 performance workflow helper tests
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
| Schema/config source of truth | `scripts/config.py` | Runtime API backed by `config/pipeline.toml`, `config/eda.toml`, and `config/multires.toml` |
| Logging | Python `logging` module | No `print()` statements in pipeline scripts |
| Derived data | Not version-controlled | Bronze, silver, gold, model, and log outputs are in `.gitignore` |

For detailed definitions of all terms used in this project, see the
[Glossary](docs/004_reference/glossary.md).

## Source of Truth

The canonical implementation specifications for this project are
[000_spec.md](docs/000_governance/000_spec.md) (pipeline hardening),
[001_spec.md](docs/000_governance/001_spec.md) (notebook development and
configuration migration), and
[002_operating_direction_spec.md](docs/000_governance/002_operating_direction_spec.md)
(current optimizer-facing operating direction). If any planning text conflicts
with implementation details in code or documentation, the governing
specification for that scope and the corresponding entries in the spec-specific
changelogs linked from the [Changelog Index](changelog.md) take precedence.
