# Glossary

Related references:
- [000_spec.md](../000_governance/000_spec.md)
- [architecture.md](../001_architecture/000_overview/architecture.md)
- [feature_sets.md](../003_modeling/feature_sets.md)

## Architecture and Pipeline

- Medallion architecture: A data engineering pattern that organizes processing into
  progressive layers (raw, bronze, silver, gold). Each layer has a single responsibility
  and a defined contract with its neighbors. Data quality and analytical value increase
  at each stage.
- Raw layer: The untouched customer-provided data. In this project, a MATLAB `.mat` file
  containing second-level power load measurements, date labels, and business-day
  classifications. The pipeline never modifies this file.
- Bronze layer: The first structured representation of the data. Converts the wide MATLAB
  matrix into a long-format time series stored as Parquet with one row per second.
  No filtering, aggregation, or feature engineering is performed.
- Silver layer: The feature-engineered analytical layer. Resamples bronze data to
  configurable resolutions and computes all derived features (temporal, business, lag,
  rolling, delta, slope). Produces 44 columns per resolution.
- Gold layer: The validated, model-ready view of silver. Drops rows where required core
  columns contain null values. Schema is identical to silver; the difference is data
  completeness. Gold is the last transformation before modeling.
- Model dataset layer: Produces train/validate/test splits from gold data, filtered to
  specific feature sets. Enforces chronological splitting and target separation.
- Parquet: A columnar storage format (Apache Parquet) optimized for analytical workloads.
  Used throughout the pipeline for efficient storage and fast reads of large datasets.
- Schema: The expected set of column names, data types, and null-safety rules for a
  given layer. Validated at every layer boundary to catch drift early.
- Orchestrator: The pipeline entry point (`run_pipeline.py`) that runs all stages in
  sequence. Supports stage selection, resolution filtering, dry-run validation, and
  structured logging.

## Time and Resolution

- Resolution: The fixed interval size used when resampling the second-level bronze data.
  For example, `5min` means each row in the output represents five minutes of aggregated
  load. Supported values range from `1s` to `15min`.
- Default resolutions: The resolutions produced when no override is provided: `1min`,
  `5min`, `10min`, `15min`.
- Alias resolution: An alternate label that maps to a canonical resolution. Currently
  `60s` is an alias for `1min`.
- Interval billing: Settlement of energy or demand charges over fixed time windows,
  commonly 15 minutes. If downstream billing uses 15-minute settlement, generate
  `15min` outputs from the start of processing to avoid reconciliation risk.
- Resampling: The process of aggregating finer-grained data into coarser intervals.
  The pipeline uses `.resample().mean()` to produce `avg_load` at each resolution.
- Period: One row at the current resolution. At `5min` resolution, "1 period" equals
  5 minutes. Lag and rolling window sizes are specified in periods, not fixed time.

## Feature Engineering

- `avg_load`: The mean power load (watts) over one resampled interval. This is the
  primary modeling target. It is excluded from all predictor feature sets to prevent
  target leakage.
- `day_class`: A customer-provided label classifying each calendar day as `full`
  (full working day), `half` (half working day), or `none` (non-working day).
- `workday`: A ternary numeric encoding of `day_class` for use as a model predictor.
  Values: `none=0`, `half=1`, `full=2`.
- Lag feature: The value of `avg_load` shifted backward by a specified number of periods.
  For example, `lag_1` at row t equals `avg_load` at row t-1. Lag features capture
  recent load history without leaking future information.
- Rolling feature: A summary statistic (mean, standard deviation, maximum, or minimum)
  computed over a trailing window of recent periods. For example, `rolling_mean_60`
  is the average of the most recent 60 periods of `avg_load`.
- Delta feature: The difference between a longer lag and the immediate lag. For example,
  `delta_5 = lag_5 - lag_1`. Deltas indicate how much the load has changed between two
  lookback horizons.
- Slope feature: The local linear trend of `avg_load` over a trailing window, computed
  using vectorized least-squares regression. Shifted by one period so it uses only
  past data. Positive slope indicates increasing load; negative indicates decreasing.
- Warm-up NaN: Expected null values at the beginning of lag, rolling, and slope columns.
  For example, `lag_1440` requires 1440 prior periods before it can produce a non-null
  value. These nulls are structurally unavoidable and are handled downstream during
  model dataset creation.
- Vectorized slope: A performance optimization for slope calculation that uses
  `numpy.lib.stride_tricks.sliding_window_view` instead of row-by-row `np.polyfit`.
  Provides approximately 5-10x speedup.

## Temporal Features

- `year`, `quarter`, `month`, `day`: Standard calendar components extracted from the
  timestamp.
- `day_of_week`: Day of the week encoded as an integer. Convention: `0=Sunday` through
  `6=Saturday`.
- `hour`: Hour of the day (0-23).
- `season`: Meteorological season encoded as an integer. `1=Winter` (Dec-Feb),
  `2=Spring` (Mar-May), `3=Summer` (Jun-Aug), `4=Fall` (Sep-Nov).
- `time_of_day`: Coarse time bucket encoded as an integer. `0=morning` (6-11),
  `1=afternoon` (12-16), `2=evening` (17-21), `3=night` (22-5).

## Notebook EDA and Statistics

- Configuration cell: The dedicated top-of-notebook cell where adjustable analysis
  parameters are declared. This cell is the only place where notebook users should change
  thresholds, figure sizes, resolution mode, and `AUTO_*` settings.
- `EDA_CONFIG`: Centralized notebook defaults exported by `scripts/config.py` and sourced
  from `config/eda.toml`. Includes visualization sizes, color mapping, outlier defaults,
  and analysis thresholds shared by all EDA notebooks.
- Resolution mode: Notebook control that determines which temporal resolutions to load.
  Supported values are `all`, `default`, and `custom`.
- Self-optimizing parameter: A parameter computed from the input data at runtime instead
  of fixed as a constant. Examples include adaptive histogram bin counts and adaptive ACF
  depth selection.
- Freedman-Diaconis rule: A data-driven histogram binning method using IQR and sample
  size (`2 * IQR * n^(-1/3)`) to choose bin width. Produces wider bins for sparse/noisy
  data and narrower bins for dense data.
- IQR (Interquartile Range): The difference between the 75th and 25th percentiles. Common
  robust spread metric used for outlier thresholds (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`).
- Mutual information: A non-linear dependency metric between a feature and target. Unlike
  Pearson correlation, it can capture non-monotonic and non-linear predictive signal.
- VIF (Variance Inflation Factor): A multicollinearity metric for predictor variables.
  Values above 5 indicate moderate collinearity; values above 10 indicate severe
  collinearity risk for linear-model interpretability.
- Partial autocorrelation (PACF): Correlation of a series with a lag after removing
  intermediate lag effects. Used to identify direct lag structure for autoregressive
  feature selection.
- Power spectral density (PSD): Frequency-domain decomposition of a time series that shows
  how signal power is distributed across frequencies. Useful for identifying dominant
  periodicities (for example daily cycles).
- STL decomposition: Seasonal-Trend decomposition using Loess. Splits a time series into
  trend, seasonal, and residual components for interpretability and stationarity checks.
- Data quality scorecard: Compact table of pass/fail quality checks (row counts, null
  rates, schema rules, physical bounds, duplicates, and correlation ceilings) used before
  downstream modeling.
- TOML: Human-readable configuration format used by this project for declarative settings.
  Loaded via Python stdlib `tomllib` with no runtime code execution side effects.
- `pyproject.toml`: Standard Python project metadata/tooling file. In this repository it
  is the dependency and tooling source of truth (runtime deps + `dev` extras +
  pytest/coverage config).

## Modeling

- Feature set: A named, versioned subset of predictor columns used for model training.
  Defined in `scripts/config.py` and documented in [feature_sets.md](../003_modeling/feature_sets.md).
  Current sets: `minimal` (3 columns), `temporal` (10 columns), `full` (41 columns),
  `curated` (11 columns).
- Target leakage: A modeling error where information about the prediction target
  (`avg_load`) is accidentally included as an input predictor. The pipeline enforces
  runtime checks to prevent this.
- Chronological split: A train/validate/test partition that respects time order. All
  training data precedes all validation data, which precedes all test data. Required for
  time-series forecasting to avoid leaking future information into training.
  Current split: train (days 1-25), validate (days 26-28), test (days 29-31).
- MVMP (Minimum Viable Modeling Product): The first constrained modeling scope, designed
  to verify the end-to-end modeling path before scaling to more complex approaches.
  Current MVMP: `5min` resolution, `minimal` feature set, Linear Regression, MAE as
  primary metric.
- MAE (Mean Absolute Error): The average of the absolute differences between predicted
  and actual values. Primary evaluation metric. Lower is better.
- RMSE (Root Mean Squared Error): The square root of the average of squared differences
  between predicted and actual values. Penalizes large errors more heavily than MAE.
  Secondary evaluation metric.
- Persistence baseline: A naive forecasting model that predicts the next value will equal
  the most recent observed value. Used as a minimum bar that any useful model must beat.
- Hypothesis: A testable statement connecting an EDA observation to a modeling approach,
  with a specific metric and improvement target. Format documented in
  [hypothesis.md](../003_modeling/hypothesis.md).

## Data Quality

- Data contract: The explicit set of schema definitions, null-safety rules, and
  behavioral assumptions that are shared across code, documentation, and tests. Violations
  raise errors rather than producing silent failures.
- Null safety: A validation rule that certain columns (timestamp, day_class, temporal
  features, avg_load in gold) must never contain null values. Enforced at layer
  boundaries.
- Determinism: The property that re-running a pipeline stage with the same inputs produces
  identical outputs. Guaranteed by deterministic sorting, no random operations, and
  consistent column ordering.
- Smoke validation: A fast automated run that confirms core scripts and notebooks execute
  end-to-end without errors. Does not verify correctness of results, only successful
  execution.
- `silver_nan_drop_warn_pct`: Configurable warning threshold (percent) for rows dropped
  after silver-stage NaN filtering. Exceeding the threshold triggers an observability
  warning but does not stop processing.

## Infrastructure

- `config.py`: The centralized configuration module (`scripts/config.py`) that defines
  all paths, resolutions, feature windows, schemas, day-class mappings, split ranges,
  and feature set definitions. Single source of truth for pipeline behavior.
- `SECONDS_PER_DAY`: Raw ingestion contract value sourced from `config/pipeline.toml`
  (`raw_contract.seconds_per_day`). Used by raw-to-bronze validation and reshape logic.
- `MATLAB_REQUIRED_KEYS`: Required `.mat` keys sourced from `config/pipeline.toml`
  (`raw_contract.required_keys`). Enforced before raw ingestion begins.
- Script path bootstrap: Runtime entrypoints prepend `scripts/` to `sys.path` so stage-script
  imports resolve consistently to `scripts/config.py` and `scripts/utils.py`.
  Notebooks import `scripts.config` and `scripts.utils` after adding project root to `sys.path`.
- `utils.py`: Shared utility module (`scripts/utils.py`) containing reusable feature
  engineering functions: `month_to_season`, `hour_to_time_of_day`, and
  `rolling_slope_series`.
- Structured logging: Use of Python's `logging` module instead of `print()` statements.
  All pipeline scripts write structured log output with timestamps, log levels, and
  contextual information.
