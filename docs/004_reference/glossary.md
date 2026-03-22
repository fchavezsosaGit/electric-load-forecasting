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
  Fourier, rolling, delta, slope, time-normalized windows, and regime/profile context).
  Produces 82 columns per resolution.
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
- Time-normalized window: A lag, rolling, or slope feature defined in real minutes
  rather than in periods. Examples include `lag_min_60` and `rolling_mean_min_240`.
  These preserve the same physical lookback across resolutions.
- Native-step comparison: One-step-ahead evaluation at each resolution's own cadence.
  Useful for diagnostics, but not valid as the final cross-resolution winner rule
  because the real-time horizon differs by resolution.
- Matched-horizon comparison: Cross-resolution evaluation where every candidate is
  scored on the same real-world horizon (for example 15 or 60 minutes) even if the
  number of lead steps differs by resolution.
- Lead-step conversion: Translating a real-world horizon into the correct number of
  future periods for a given resolution.
- Origin: The forecast start timestamp used for a matched-horizon or recursive rollout
  evaluation window.
- Execution profile: A named Stage-6 scope (`smoke`, `candidate`, `focus_60m`, `full`) that fixes the
  default resolutions, horizons, feature sets, and model labels for a bounded run.
- Second-level resolution support: Stage-6 ability to compare `1s`, `5s`, `10s`, and
  `30s` cadences in addition to the minute-level stack, while still keeping smoke and
  candidate profiles bounded for routine iteration.
- Temporal hierarchy: A forecasting view where multiple cadences or horizons are
  treated as related layers of the same problem rather than as isolated model
  searches. In this repo, it motivates keeping day-ahead, hourly, structural
  `15m`, and minute layers coherent instead of forcing one universal winner.

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
- Time-normalized lag feature: The value of `avg_load` shifted backward by the number of
  periods required to represent a fixed minute lookback at the current resolution.
- Rolling feature: A summary statistic (mean, standard deviation, maximum, or minimum)
  computed over a trailing window of recent periods. For example, `rolling_mean_60`
  is the average of the most recent 60 periods of `avg_load`.
- Baseline-relative feature: A predictor that encodes deviation from a stable reference
  path such as previous-day load or average-workday profile rather than raw load alone.
- `avg_workday_baseline`: The historical average profile value for the same slot and
  workday status. Used as a stable day-shape baseline in later modeling stages.
- `profile_residual_lag_1`: One-step lag of the residual between recent load and the
  average-workday baseline.
- `previous_day_residual`: Difference between previous-day same-slot load and the
  average-workday baseline at that slot.
- `profile_active_flag`: Binary indicator that the average-workday baseline is in an
  active operating regime, derived from a configurable activity threshold.
- `workday_transition`: Indicator that adjacent days cross workday/non-workday regimes.
- Regime profile: A compact feature family that combines short-memory, calendar, and
  baseline-relative context so the model learns corrections to a stable daily shape.
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
- Fourier feature: A sine/cosine encoding of a cyclical variable on the unit circle.
  Used here so hour-of-day and day-of-week remain continuous at wraparound boundaries.
- `hour_sin`, `hour_cos`: Continuous daily Fourier encoding derived from timestamp phase.
  Unlike raw `hour`, these columns change smoothly within the hour and keep 23:59 close
  to 00:00 numerically.
- `dow_sin`, `dow_cos`: Continuous weekly Fourier encoding derived from day-of-week plus
  within-day phase. Keeps the Saturday/Sunday boundary smooth for nonlinear models.

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
- Config hash: Stable SHA-256 identifier of the effective Stage-6/Stage-7 config after
  CLI overrides. Used for deterministic run identity.
- Joblib runtime: Shared Stage-5/Stage-6 execution layer configured in
  `config/modeling.toml`. Controls backend choice, worker caps, batching, dispatch
  policy, and stage toggles without changing Python code.
- Horizon policy: Centralized Stage-5/Stage-6 rule set that maps a forecast horizon
  bucket to allowed feature sets, allowed model labels, and residual/blend behavior.
- Adaptive HGB screen: Stage-5 pre-selection pass that tests a bounded set of
  HistGradientBoosting configurations and records which settings are worth carrying into
  the more expensive fold evaluation.
- Parallel plan: The resolved runtime decision for one batch execution (backend, worker
  count, task count, batching, and inner-thread limit). Recorded in Stage-5/Stage-6
  run manifests for reproducibility.
- Inner thread cap: Per-worker limit applied to nested BLAS/OpenMP thread pools during
  parallel model evaluation to prevent oversubscription.
- `latest/` alias: Convenience directory under an output root that mirrors the most
  recent successful timestamped run. Timestamped directories remain the source of truth.
- Notebook run archive: Timestamped output directory under `outputs/008_notebook_runs/`
  that stores executed notebook snapshots and a manifest before tracked notebook cell
  outputs are cleared.
- Eligibility gate: Stage-6 rule set that removes a non-persistence candidate from final
  consideration when it fails minimum coverage, completeness, or comparable-evaluation
  requirements.
- Practical-gain gate: Stage-6 rule set that requires a non-persistence candidate to beat the
  baseline by a configured margin before it can replace persistence as the selected
  winner.
- `coverage_below_threshold`: Stage-6 per-row health outcome for a non-persistence
  candidate that ran successfully but did not reach the configured minimum evaluation
  coverage. This is an ineligibility result, not a runtime crash.
- Selection context: Small Stage-7 metadata record that states which resolution/feature
  set/model was actually rolled out, whether it came from Stage-6 selection or from
  explicit config fallback, which optimization target was requested, and why that choice
  was made.
- Winner registry: Stage-6 cross-run artifact (`winner_registry.csv`) that records
  matched-horizon winners by run, horizon, strategy, timestamp, and winner MAE / MAE %
  so downstream selection does not depend on the mutable `latest/` alias alone.
- Rollout registry: Stage-7 cross-run artifact (`rollout_registry.csv`) that records one
  learned-candidate outcome row per rollout run, including endpoint/path MAE and
  baseline comparisons, so long-horizon fallback selection can be driven by measured
  rollout evidence.
- Segmented holdout evaluation: Stage-5 artifact (`holdout_segment_evaluation.csv`) that
  breaks the promoted-candidate vs baseline comparison by regime columns instead of only
  reporting an aggregate holdout metric.
- Rollout challenger sweep: Stage-7 batch process that reruns a bounded set of learned
  long-horizon challengers, ranks them by the configured rollout objective, and promotes
  the current best measured candidate under `challenger_sweeps/latest/`.
- Selection run ID: Explicit Stage-7 CLI reference to one Stage-6 run directory
  (`--selection-run-id`) used when rollout should be anchored to a specific historical
  selection artifact.
- Forecast strategy: The way a Stage-6 learned candidate is evaluated at a matched
  horizon. Current values are `recursive` (one-step model rolled forward through the
  whole path) and `direct_endpoint` (model trained directly against the horizon-end
  target only).
- Rollout target mode: The learned objective used in Stage-7. `raw` predicts load
  directly, while `avg_workday_residual` predicts corrections to the avg-workday
  baseline path.
- Exact-horizon reuse: Stage-7 rule that allows a Stage-6 learned winner to be reused
  only when the requested rollout horizon exactly matches the learned winner horizon
  and the winner itself is recursive. Prevents a short-horizon or endpoint-only winner
  from being promoted into a much longer recursive rollout by accident.
- Anchored workday baseline (`anchored_workday`): A long-horizon baseline that keeps the
  average-workday trajectory shape but anchors the path to the latest observed load
  level, reducing the level bias of a raw average-workday forecast.
- Hybrid workday baseline (`hybrid_workday`): A long-horizon baseline that blends
  persistence with the anchored-workday path across the forecast horizon so near-term
  steps stay persistence-heavy while later steps follow the profile shape.
- Rollout origin policy: Stage-7 rule for choosing which forecast start timestamps are
  evaluated in a rollout run. `uniform` spreads origins across the eligible test window,
  while `midnight` restricts evaluation to day-boundary starts for explicit diagnostics.
- Rollout selection summary: Stage-7 artifact pair (`rollout_selection_summary.csv` and
  `rollout_selection_summary.md`) that records the best endpoint candidate and best path
  candidate from one rollout run.
- Challenger recommendation: Stage-7 sweep output (`recommended_candidate.json`) that
  records the best learned rollout candidate for the requested horizon after ranking
  registry-backed evidence against the configured selection target.
- Horizon curve: Stage-8 H5 artifact set that consolidates the Stage-5 holdout anchor
  (`1m`) and Stage-7 challenger-sweep evidence (`15m` through `1440m`) into one
  cross-horizon summary.
- Capability envelope: The interpretation rule for the Stage-8 horizon curve. Because
  the strongest verified learned candidate can change by horizon, the curve is not a
  single-model monotonic decay trace; it is the best measured candidate at each horizon.
- Crossover horizon: A lead time where the learned model switches from beating a
  baseline to losing to it, or vice versa. The current H5 result is non-monotonic, so
  there is no single global crossover horizon.
- Endpoint MAE: Error measured only at the final forecast point of a horizon window.
- Path MAE: Error averaged across the entire recursive rollout path for that horizon.
- Horizon curve summary: Stage-8 artifact pair (`horizon_curve_summary.csv` and
  `horizon_curve_summary.md`) that records the selected candidate, baselines, and
  relative-performance ratios at each evaluated horizon.
- Crossover summary: Stage-8 JSON artifact (`crossover_summary.json`) that lists which
  horizons beat persistence and which horizons beat the strongest configured baseline.

## Modeling

- Feature set: A named, versioned subset of predictor columns used for model training.
  Defined in `scripts/config.py` and documented in [feature_sets.md](../003_modeling/feature_sets.md).
  Current sets: `minimal` (3 columns), `temporal` (14 columns), `full` (86 columns),
  `curated` (15 columns), `full_stable` (78 columns), and `regime_profile` (32 columns).
- `full_stable`: A canonical high-capacity feature set built from `full` with the
  `rolling_*_240` and `rolling_*_1440` columns removed so high-capacity candidates keep
  comparable validation coverage across Stage-5 and downstream multiresolution runs.
- Target leakage: A modeling error where information about the prediction target
  (`avg_load`) is accidentally included as an input predictor. The pipeline enforces
  runtime checks to prevent this.
- Chronological split: A train/validate/test partition that respects time order. All
  training data precedes all validation data, which precedes all test data. Required for
  time-series forecasting to avoid leaking future information into training.
  Current split: train (days 1-25), validate (days 26-28), test (days 29-31).
- MVMP (Minimum Viable Modeling Product): The first constrained modeling scope, designed
  to verify the end-to-end modeling path before scaling to more complex approaches.
  Current MVMP: `1min` resolution anchor with `minimal` feature set initialization,
  evaluated within a fixed Ridge + HistGradientBoostingRegressor experiment grid.
- Ridge regression: Linear regression with L2 regularization (`alpha`) used as the
  regularized linear baseline in Report IV modeling.
- HistGradientBoostingRegressor (HGB): Tree-based gradient boosting model that supports
  nonlinear structure and native handling of feature NaN values; used as a nonlinear
  comparator against Ridge.
- MAE (Mean Absolute Error): The average of the absolute differences between predicted
  and actual values. Primary evaluation metric. Lower is better.
- MAE % (`mae_pct`): A scale-normalized MAE defined as
  `100 * MAE / mean(abs(actual_load))` over valid rows. This makes error comparable
  across sites and load types instead of relying on watts alone.
- RMSE (Root Mean Squared Error): The square root of the average of squared differences
  between predicted and actual values. Penalizes large errors more heavily than MAE.
  Secondary evaluation metric.
- RMSE % (`rmse_pct`): A scale-normalized RMSE defined as
  `100 * RMSE / mean(abs(actual_load))` over valid rows.
- Persistence baseline: A naive forecasting model that predicts the next value will equal
  the most recent observed value. Used as a minimum bar that any useful model must beat.
- Hypothesis: A testable statement connecting an EDA observation to a modeling approach,
  with a specific metric and improvement target. Format documented in
  [hypothesis.md](../003_modeling/hypothesis.md).
- Evaluation coverage (`eval_coverage`): The fraction of target-available evaluation
  rows that were actually scored for a given experiment (`n_eval / n_eval_total`).
  Lower coverage usually indicates feature-NaN row drop (for example, Ridge on
  long-window rolling features).
- Coverage guard selection policy: Holdout-model selection rule that restricts candidate
  validation rows to experiments meeting a minimum evaluation coverage threshold
  (`MIN_VALIDATE_COVERAGE`, currently 0.95). Prevents low-coverage "easy subset" wins.
- Coverage audit: Stage-5 artifact (`coverage_audit.csv`) that records split-level
  feature-set coverage against the promotion threshold before holdout selection.
- Promotion candidate: Stage-5 artifact (`promotion_candidate.json`) that records the
  exact scoreboard winner promoted into one-shot holdout evaluation.
- Deployment recommendation: Stage-5 artifact (`deployment_recommendation.json`) that
  records whether the promoted learned candidate or persistence is the current
  operational recommendation after holdout evaluation.
- Raw-best vs selected-best: Two recorded holdout-selection views in
  `run_manifest.json`: (1) raw-best by validation MAE regardless of coverage, and
  (2) selected-best after coverage guard policy.
- Recursive rollout: Multi-step forecasting where each newly predicted point is fed back
  into the model as history for the next step.
- Practical gain threshold: Minimum improvement required over persistence before a
  learned candidate can be promoted as a winner.
- Pareto frontier: The subset of candidates that are not strictly dominated on error,
  stability, and runtime simultaneously.
- Selection summary: Plain-language artifact (`selection_summary.csv` and
  `selection_summary.md`) that records whether a learned model, a non-persistence
  baseline, or persistence itself won each matched horizon.
- `next_lock_mae`: Stage-7/Stage-10 metric measuring error on the next locked
  billing/control interval after an update. This is more optimizer-relevant than
  generic path MAE when near-term control actions matter most.
- Optimizer-ready feed: The repo's pre-optimizer delivery surface: interval rows
  that include timestamps, forecast value, selected layer, fallback reason,
  freshness, uncertainty, and provenance metadata.
- Dynamic overlay: A Stage-10 minute-layer controller that decides whether the
  minute overlay should stay active for a given interval context. It is currently
  diagnostic and shadow-only.
- Shadow mode: Operating state where a policy is evaluated and persisted as a
  counterfactual but is not allowed to change the live-selected output.
- Predict-then-optimize: Modeling stance that evaluates forecasts by their
  downstream decision value, not only by generic prediction error. In this repo,
  that means next-lock and peak-aware promotion logic matters more than a single
  global MAE leaderboard.

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
  engineering functions: `month_to_season`, `hour_to_time_of_day`,
  `build_fourier_feature_frame`, and `rolling_slope_series`.
- Structured logging: Use of Python's `logging` module instead of `print()` statements.
  All pipeline scripts write structured log output with timestamps, log levels, and
  contextual information.
