# SPEC-00: Hardening Existing Code and Data Pipeline

| Field            | Value                                                        |
|------------------|--------------------------------------------------------------|
| Project          | Daily Electric Load Forecasting                              |
| Specification ID | SPEC-00                                                      |
| Status           | Implemented (validated 2026-02-20)                           |
| Created          | 2026-02-20                                                   |
| Last Updated     | 2026-02-20                                                   |
| Authors          | Spencer Hoyle, Sean He, Frank Chavezsosa                     |
| Advisor          | Prof. Raymond de Callafon                                    |

## Source of truth declaration

For this repository run, this file is the canonical implementation specification.
If any planning text conflicts with implementation details in code/docs, this file
and the matching entries in the spec-specific changelog at
[docs/change logs/000spec/changelog.md](../change%20logs/000spec/changelog.md) take
precedence. The root [changelog.md](../../changelog.md) serves as an index pointing
to spec-specific changelogs.

## Revision R2 (2026-02-20): Resolution and documentation governance update

This revision formalizes two operational requirements:

1. Expanded resolution menu:
   - `1s`, `5s`, `10s`, `30s`, `60s` (alias of `1min`), `1min`, `5min`, `10min`, `15min`
2. Documentation governance:
   - All repository-level documents live under `docs/` and are cross-referenced.
   - `personal/` is non-authoritative scratch space.
   - [glossary.md](../004_reference/glossary.md) is the shared terminology anchor.

Billing-critical requirement:
- If interval billing is settled at 15-minute granularity, `15min` outputs must be
  generated and validated from the beginning of the pipeline run. Do not rely on
  post-hoc aggregation for financial settlement workflows.

## Revision R3 (2026-02-20): Documentation structure normalization

This revision formalizes documentation organization rules:

1. Documentation folders under `docs/` are numbered from `000` upward for ordered
   navigation.
2. Numbering is applied to folders, not markdown filenames.
3. Canonical document locations are:
   - `docs/000_governance/000_spec.md`
   - `docs/001_architecture/000_overview/architecture.md`
   - `docs/002_pipeline/pipeline.md`
   - `docs/002_pipeline/plan.md`
   - `docs/003_modeling/feature_sets.md`
   - `docs/003_modeling/hypothesis.md`
   - `docs/003_modeling/mvmp.md`
   - `docs/004_reference/glossary.md`

## Revision R4 (2026-02-20): Numbering normalization for operational paths

This revision formalizes three-digit numbering on operational layers:

1. Data layer folders are prefixed to encode pipeline order:
   - `data/000_raw`
   - `data/001_bronze`
   - `data/002_silver`
   - `data/003_gold`
   - `data/004_model`
2. Core pipeline scripts use three-digit prefixes:
   - `scripts/000_raw_to_bronze.py`
   - `scripts/001_bronze_to_silver.py`
   - `scripts/002_silver_to_gold.py`
   - `scripts/003_create_model_datasets.py`
3. Numbering is reserved for operational ordering; helper modules remain semantic:
   - `scripts/config.py`
   - `scripts/utils.py`
   - `scripts/validate_notebooks.py`

## Revision R5 (2026-02-20): Execution progress record

This revision records implementation completion status for this spec execution pass.

Completed in this pass:
1. Phase 1, Step 1 (Config hardening):
   - Added runtime config validation via `validate_config()` in `scripts/config.py`.
   - Added duplicate silver-column detection and stricter feature/split/leakage guards.
2. Phase 1, Step 2 (Raw -> Bronze hardening):
   - Added numeric/infinity/date-uniqueness validations and post-merge null checks.
   - Added edge-case guards (zero-day input, all-NaN day warning, physical-range warning).
   - Expanded logging with load stats, date coverage, and output file size.
3. Phase 1, Step 3 (Bronze -> Silver hardening):
   - Added defensive empty/all-NaN handling and explicit resample boundary parameters.
   - Added monotonic index checks, warm-up warnings, and threshold warnings.
   - Added per-resolution file-size logging and stricter schema enforcement.
4. Phase 1, Step 4 (Silver -> Gold reproducibility hardening):
   - Added required-column existence checks before filtering.
   - Added robust parquet read/write error handling and empty-output critical warnings.
   - Added per-column required-null drop breakdown and file-size logging.
5. Phase 1, Step 5 (Orchestration hardening):
   - Added config validation on startup and `--dry-run` in `run_pipeline.py`.
   - Added stage failure context with elapsed-time reporting.
6. Phase 1, Step 6 (Test expansion):
   - Expanded/added tests for config, raw/bronze/silver/gold/model scripts, orchestrator,
     notebook validator, and integration flow.
   - Added edge fixtures for all-NaN, single-row, single-day, and empty bronze inputs.
7. Phase 2, Step 1 (Notebook execution validation):
   - Re-executed core notebooks end-to-end with `scripts/validate_notebooks.py`.
   - Updated notebook validator to structured logging and explicit failure propagation.

Initial R5 verification snapshot executed on 2026-02-20:
- `pytest tests -q` -> `74 passed`.
- `pytest tests --cov=scripts --cov=run_pipeline --cov-report=term -q` -> `74 passed`, total coverage `80%`.
- `python run_pipeline.py --dry-run` -> success.
- `python run_pipeline.py --stage all` -> success (bronze/silver/gold across default resolutions).
- `python scripts/003_create_model_datasets.py` -> success.
- `python scripts/validate_notebooks.py` -> success.

## Revision R6 (2026-02-20): Verification and documentation refresh

This revision records the latest post-hardening verification pass and updates
documentation references to the current repository health state.

Verification refresh executed on 2026-02-20:
- `python run_pipeline.py --stage all` -> success.
- `python scripts/validate_notebooks.py` -> success (baseline notebooks plus silver
  validation matrix: `default`, `all`, `custom`) with no warning output.
- `pytest -q` -> `98 passed`.
- `pyright run_pipeline.py scripts tests` -> `0 errors, 0 warnings`.

Documentation alignment updates:
- Refreshed spec and changelog verification counts to current values.
- Added explicit note of notebook validator runtime hardening for Windows event-loop
  compatibility and warning-free execution.

## Implementation status snapshot (R5)

Phase 1 status:
- Completed: config hardening, ingestion hardening, multi-stage orchestration, tests,
  documentation baseline.

Phase 2 status:
- Completed: EDA notebook expansion, feature-set formalization, model dataset generation,
  hypothesis and MVMP documentation.

Verification commands:
- `python run_pipeline.py --dry-run`
- `python run_pipeline.py --stage all`
- `pytest tests`
- `python scripts/validate_notebooks.py`

## Workflow and tooling

This specification was authored with the assistance of Claude (Anthropic) through an iterative
review of the full codebase, documentation, notebooks, and project requirements. The team reviewed
and approved the final specification. Execution will be carried out using OpenAI Codex 5.3, selected
for its high efficiency in following structured plans and producing high-quality code updates from
detailed specifications. All changes produced by Codex will be reviewed by the team before merge.

Tooling summary:
- Specification authoring: Claude (Anthropic) -- codebase audit, line-level issue identification, acceptance criteria
- Specification review: Team (manual review and approval)
- Execution: OpenAI Codex 5.3 -- code generation, script creation, documentation updates
- Final review: Team (manual review, testing, and merge)

## Why this specification exists

The previous roadmap was useful as an audit, but it mixed critical foundation work with advanced
modeling tasks. This version is intentionally constrained to two phases so the team can build a
reliable base before scaling.

Primary goal: make the pipeline reproducible, testable, and trustworthy from raw ingestion through
model-ready datasets.

## Initial state snapshot (baseline as of 2026-02-20)

What exists today:
- `scripts/000_raw_to_bronze.py` converts `.mat` data to bronze parquet.
- `scripts/001_bronze_to_silver.py` creates a 1-minute silver dataset with engineered features.
- `data/003_gold/*.parquet` files exist, but there is no script that reproducibly generates them.
- `docs/002_pipeline/pipeline.md` and `docs/002_pipeline/plan.md` exist but are partially incomplete or inconsistent with code.

Observed gaps that block reliable progress:
- Paths and script behavior are hardcoded and fragile.
- Error handling and logging are minimal, so failures are hard to debug.
- Silver outputs are not consistently produced for 1m, 5m, and 10m through one controlled flow.
- There is no single orchestrator to run ingestion end to end.
- There are no automated tests for core ingestion transformations.

## Working principles

1. Reliability over novelty: deterministic ingestion and schema stability come first.
2. Small validated steps: each step must have an explicit done check.
3. Traceability: every material change gets documented in the spec-specific changelog.
4. Value-first decisions: every task below exists because it reduces risk, improves speed, or improves data trust.

---

## Phase 1: Foundation and Ingestion Hardening

Goal:
Make raw to bronze to silver to gold reproducible with clear logging, validation, and tests.

Exit criteria:
- One command can rebuild all derived datasets from raw input.
- Scripts fail with actionable errors.
- Output schemas are validated and documented.
- Core ingestion tests pass locally.

---

### Step 1. Create shared configuration

Rationale: Five hardcoded paths exist across two scripts and three notebooks. Feature settings (`LAG_MINUTES`, `ROLLING_MINUTES`, `SLOPE_MINUTES`) are defined only in `scripts/001_bronze_to_silver.py` (lines 8-10) and repeated nowhere, making future changes error-prone. Supported resolutions (`1m`, `5m`, `10m`) exist only in `docs/002_pipeline/plan.md` lines 44-88 but are not enforced by any code.

Tasks:
- Create `scripts/config.py` with the following sections:
  - `PATHS`: canonical paths for raw, bronze, silver, gold directories and filenames. Replace:
    - `scripts/000_raw_to_bronze.py` lines 5-6 (`RAW_DATA_PATH`, `BRONZE_LOAD_DATA_PATH`)
    - `scripts/001_bronze_to_silver.py` lines 4-5 (`BRONZE_PATH`, `SILVER_PATH`)
  - `RESOLUTIONS`: list of supported output resolutions (`["1min", "5min", "10min"]`)
  - `FEATURE_CONFIG`: move `LAG_MINUTES`, `ROLLING_MINUTES`, `SLOPE_MINUTES` from `001_bronze_to_silver.py` lines 8-10
  - `DAY_CLASS_MAP`: move `DAY_CLASS_TO_WORKDAY` from `001_bronze_to_silver.py` line 12
  - `SCHEMAS`: expected column names and types for bronze, silver, and gold outputs
  - Use `pathlib.Path` throughout; resolve relative to project root via `Path(__file__).resolve().parent.parent`
- Update `scripts/000_raw_to_bronze.py` to import paths from `config.py` and remove lines 5-6.
- Update `scripts/001_bronze_to_silver.py` to import all config from `config.py` and remove lines 4-5, 8-10, 12.
- Add a `PROJECT_ROOT` constant that anchors all paths, so scripts work correctly regardless of working directory.

Config validation (runtime safety):
- Validate that `FEATURE_CONFIG` lag, rolling, and slope window values are positive integers at module load time.
- Validate that `DAY_CLASS_MAP` keys match `VALID_DAY_CLASSES` exactly and that values are unique integers.
- Validate that `DEFAULT_RESOLUTIONS` is a subset of `SUPPORTED_RESOLUTIONS`.
- Validate that `RESOLUTION_ALIASES` values map to resolutions in `SUPPORTED_RESOLUTIONS`.
- Validate that `FEATURE_SETS` column names are subsets of `SCHEMAS["gold"]["columns"]` -- catch column name drift at import time, not at runtime.
- Validate that no feature set includes the target column (`avg_load`) -- prevent target leakage at the config level.
- Validate that `SPLIT_DAY_RANGES` covers a contiguous range with no gaps or overlaps.
- Validate that `_build_silver_columns()` produces no duplicate column names.
- Add a docstring to `_build_silver_columns()` explaining the column construction logic and ordering convention.
- Add a `validate_config()` function that runs all validation checks and can be called by tests and by the pipeline orchestrator during `--dry-run`.

Acceptance criteria:
- No path string literals remain in `000_raw_to_bronze.py` or `001_bronze_to_silver.py`.
- No feature configuration constants remain in `001_bronze_to_silver.py`.
- `config.py` is the single source of truth for all paths, resolutions, and feature settings.
- All scripts run correctly from any working directory (project root, scripts dir, etc.).
- Running `grep -r "data/" scripts/000_raw_to_bronze.py scripts/001_bronze_to_silver.py` returns zero path literals.
- Config validation catches column name drift, target leakage, and resolution inconsistencies at import time.
- A `validate_config()` function exists and is callable by tests and the orchestrator.

---

### Step 2. Harden raw to bronze ingestion (`scripts/000_raw_to_bronze.py`)

Rationale: The script has 8 bare `assert` statements (lines 22-26, 60-62) that provide zero diagnostic info on failure and are silently skipped under `python -O`. File I/O has no try/except, so a missing `.mat` file or a corrupt read produces an opaque stack trace. Output uses `print()` (lines 12, 67-69) instead of structured logging. No output directory is created automatically -- the script silently fails if `data/001_bronze/` does not exist.

Tasks:

Validation and error handling:
- Replace `assert p_data is not None` (line 22) with `if p_data is None: raise ValueError("Key 'P_data' not found in .mat file: {path}")`. Apply same pattern to lines 23-24 for `day_data` and `day_class`.
- Replace `assert p_data.shape[0] == 86400` (line 25) with a `ValueError` that reports the actual shape: `f"Expected 86400 rows, got {p_data.shape[0]}"`.
- Replace `assert p_data.shape[1] == day_data.shape[1] == day_class.shape[1]` (line 26) with a `ValueError` that reports all three shapes.
- Replace `assert df["day_class"].isin(["full","half","none"]).all()` (line 60) with a check that reports unexpected values: `unexpected = set(df["day_class"].unique()) - {"full","half","none"}`.
- Replace `assert df["timestamp"].is_monotonic_increasing` (line 61) with a check that identifies where monotonicity breaks.
- Replace `assert df.shape[0] == p_data.size` (line 62) with a `ValueError` reporting expected vs actual row count.
- Wrap `loadmat()` call (line 15) in try/except to catch `FileNotFoundError` and `ValueError` (corrupt file), re-raising with actionable message including the path.
- Wrap `to_parquet()` call (line 65) in try/except to catch `OSError`, re-raising with message about permissions or disk space.

File I/O:
- Before writing parquet (line 65), add `Path(output_path).parent.mkdir(parents=True, exist_ok=True)` to create output directory automatically.
- Validate that the `.mat` file exists before attempting `loadmat()` -- fail fast with clear error.

Logging:
- Add `import logging` and configure logger at module level: `logger = logging.getLogger(__name__)`.
- Replace `print("Executing raw to bronze pipeline...")` (line 12) with `logger.info(...)`.
- Replace `print("Saved bronze data to:", ...)` (line 67) with `logger.info(...)`.
- Remove `print(df.head())` and `print(df.shape)` (lines 68-69) -- these are debug artifacts.
- After successful write, log a dataset profile: row count, timestamp range (min/max), null count per column, `day_class` value distribution (full/half/none counts).

Code quality:
- Add module-level docstring explaining: input format, output format, schema, and usage.
- Add type hints to `raw_to_bronze()` function signature: `def raw_to_bronze() -> None:`.
- Add inline comments explaining the `melt` operation (lines 35-39) for teammates unfamiliar with pandas reshape.

Data integrity:
- Validate that `P_data` values are numeric dtype -- reject object arrays with mixed types that would silently corrupt downstream computations.
- Check for infinity values (`np.inf`, `-np.inf`) in `P_data` and replace with NaN or raise, since infinity would propagate through all downstream aggregations.
- Validate that `day_data` dates are unique -- duplicate dates would silently corrupt the day_class merge and produce incorrect time series.
- Validate that `day_data` dates parse to valid datetime objects before proceeding -- malformed dates should fail fast, not during the merge step.
- After the day_class merge, validate that no `day_class` values are NaN -- NaN indicates a date that failed to join, which would corrupt downstream business features.

Edge cases:
- Handle `P_data` with zero columns (no days) -- raise `ValueError` with descriptive message rather than producing an empty DataFrame.
- Validate that `P_data.shape[1]` is at least 1 before proceeding.
- Handle the case where all 86,400 values for a single day are NaN -- log a warning identifying the affected date, but allow processing to continue.
- Handle `day_class` values that are byte strings (MATLAB `.mat` files sometimes return `bytes` instead of `str`) -- normalize to Python `str` before validation.
- Validate that `load` values fall within a physically plausible range (e.g., 0 to 100,000 watts). Log warnings for out-of-range values rather than failing, to support exploratory analysis.

Logging enhancements:
- After successful write, log min, max, and mean load values in the output profile alongside existing row count, timestamp bounds, null counts, and day_class distribution.
- Log the unique date count and date range (first date to last date) for verification against expected coverage.
- Log file size of the output parquet file.

Acceptance criteria:
- Script raises `ValueError` or `RuntimeError` with descriptive messages for every failure mode (missing file, corrupt data, unexpected shape, unexpected values).
- Zero `assert` statements remain in the file.
- Zero `print()` statements remain in the file.
- Running the script with a missing `data/000_raw/` directory produces a clear error message naming the expected path.
- Running the script with a missing `data/001_bronze/` directory succeeds (auto-creates the directory).
- Log output includes: row count, timestamp bounds, null counts, day_class distribution, and load value summary statistics.
- Infinity values in raw data are handled (replaced with NaN or rejected).
- Duplicate dates in `day_data` are detected and rejected.
- Post-merge day_class NaN values are detected and rejected.

---

### Step 3. Harden bronze to silver ingestion (`scripts/001_bronze_to_silver.py`)

Rationale: The script silently drops 14,576 NaN rows at line 55 with no logging. It only produces 1-minute output despite the plan requiring 1m/5m/10m (see `docs/002_pipeline/plan.md` lines 44-88). The `workday` encoding (line 12: `{"none": 0, "half": 1, "full": 2}`) is ternary, but `docs/002_pipeline/plan.md` line 134 specifies binary `is_workday`. Column naming uses `weekday` (line 89) but the plan says `day_of_week` (plan line 117). There is a stale comment at line 109 (`"Your text had 'la_1m'"`) left from development. The `rolling_slope` function (lines 37-47) uses `np.polyfit` inside `.apply()` which is slow for 44,640 rows x 3 windows.

Tasks:

Logging and validation:
- Add `import logging` and configure logger: `logger = logging.getLogger(__name__)`.
- Replace `print("Loading bronze data...")` (line 51) with `logger.info(...)`.
- Replace `print("Resampling to 1-minute...")` (line 68) with `logger.info(...)`.
- Replace `print("Silver dataset created successfully:", ...)` (line 131) with `logger.info(...)`.
- Remove `print(silver.head())` and `print("Rows:", ...)` (lines 132-133) -- debug artifacts.
- After filtering NaN load (line 55), log the count and percentage of dropped rows: `logger.info(f"Dropped {count} NaN load rows ({pct:.2f}%)")`.
- After resampling, log remaining NaN count in `avg_load` and other columns.
- Before writing, validate output schema against expected column list from config.
- Before writing, validate no unexpected NaN in non-lag columns (timestamp, day_class, temporal features should be complete).

Multi-resolution output:
- Parameterize the `bronze_to_silver()` function to accept a resolution parameter or iterate over `config.RESOLUTIONS`.
- For each resolution (`1min`, `5min`, `10min`):
  - Resample bronze data at that resolution using `.resample(resolution)`.
  - Recalculate all lag/rolling/slope features appropriately for the resolution.
  - Important: lag values should be in units of the resolution (e.g., at 5min resolution, `lag_1` means 1 period = 5 minutes ago).
  - Write to `data/002_silver/power_load_{res}.parquet` (e.g., `power_load_5m.parquet`, `power_load_10m.parquet`).
- Log row counts for each resolution output. Expected approximate counts: 1m=44,640, 5m=8,928, 10m=4,464.
- Add output directory auto-creation: `Path(output_path).parent.mkdir(parents=True, exist_ok=True)`.

Fix naming and encoding inconsistencies:
- Decide on `weekday` vs `day_of_week` (line 89). The plan uses `day_of_week` -- rename the column to match or update the plan. Document the decision.
- Decide on ternary `workday` (0/1/2) vs binary `is_workday` (0/1) encoding (line 12 vs plan line 134). Document the decision. If keeping ternary, update the plan. If switching to binary, `half` must be assigned to either 0 or 1 with documented rationale.
- Remove stale comment `"Your text had 'la_1m'"` at line 109.
- Clean up the commented-out `dropna` at line 75 -- either enable it with documented rationale or remove it entirely.

Performance optimization:
- Replace the `rolling_slope` function (lines 37-47, called via `.apply(rolling_slope, raw=True)` at lines 121-126) with a vectorized implementation. Options:
  - Use numpy vectorized slope: `slope = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)` via rolling sums.
  - Or use `scipy.stats.linregress` on rolling windows via `stride_tricks`.
  - The `.apply()` with `np.polyfit` is O(n*w) per window and creates Python overhead per row.
- Benchmark before/after to confirm improvement (expect 5-10x speedup on slope calculation).

Code quality:
- Add module-level docstring explaining: input format, all output resolutions, feature engineering logic, and usage.
- Add type hints to all function signatures.
- Add docstring to `bronze_to_silver()` explaining parameters, side effects, and output files.
- Move `month_to_season` and `hour_to_time_of_day` helper functions to a shared `scripts/utils.py` if they will be reused in gold or test scripts. Otherwise keep in-file but ensure they are tested.

Utility function hardening (`scripts/utils.py`):
- Add input validation to `month_to_season()`: reject values outside 1-12 with `ValueError` instead of silently returning a default season.
- Add input validation to `hour_to_time_of_day()`: reject values outside 0-23 with `ValueError` instead of silently returning a default time bucket.
- Add input validation to `rolling_slope()`: verify input is a 1D numeric array of sufficient length.
- Add input validation to `rolling_slope_series()`: verify window is a positive integer and does not exceed Series length (log warning if it does).
- Document the mathematical formula used in `rolling_slope()` (vectorized least-squares) in the docstring for maintainability.
- Ensure `rolling_slope()` behavior for constant data (returns 0.0) and empty/all-NaN data (returns NaN) is explicitly documented and tested.

Defensive resampling:
- Validate that the bronze DataFrame has at least one non-NaN load row before processing -- an empty or all-NaN bronze input should raise `ValueError` rather than producing empty silver files.
- After NaN load filtering, validate that the cleaned DataFrame is not empty. If all rows were NaN, raise `ValueError` with the resolution and input path.
- Explicitly set `closed` and `label` parameters on `.resample()` calls to prevent behavior changes across pandas versions (default behavior changed between pandas 1.x and 2.x for some offset aliases).
- Validate that the resample index is monotonic and sorted before feature engineering begins -- unsorted input silently produces incorrect lag and rolling features.

Edge cases:
- Handle lag periods that exceed the resampled dataset length gracefully -- log a warning noting that the entire lag column will be NaN warm-up, which may affect downstream modeling.
- Validate that rolling window sizes do not exceed the number of rows in the resampled DataFrame. If they do, log a warning.
- Validate that `day_class` does not contain NaN values before the `workday` mapping -- NaN day_class would produce NaN workday, corrupting a required-not-null column.
- Check for infinity values in `avg_load` after resampling and replace with NaN or raise.
- Handle resolutions that produce very few rows (e.g., `15min` on a 1-day dataset = 96 rows) where long lag/rolling windows (e.g., `lag_1440`) would be entirely NaN -- log this as a known limitation.

Threshold monitoring:
- Log a warning if the percentage of NaN rows dropped exceeds 5% of the bronze input, indicating a potential upstream data quality issue.
- Log the effective warm-up period for each lag and rolling feature so the team can verify that expected NaN counts match actual counts.
- Log per-resolution output file size for tracking data growth across resolutions.

Acceptance criteria:
- Silver outputs exist for all configured default resolutions: `data/002_silver/power_load_1m.parquet`, `power_load_5m.parquet`, `power_load_10m.parquet`, `power_load_15m.parquet`.
- Zero `print()` statements remain in the file.
- NaN filtering is logged with count and percentage.
- Column names match the documented schema (consistent between code and plan).
- Workday encoding is consistent between code and documentation.
- No stale comments or commented-out code remain.
- Slope calculation is vectorized (no `.apply()` with Python-level loop).
- Each output file passes schema validation: expected columns, expected dtypes, no unexpected NaN in non-lag columns.
- Running the script with missing output directory succeeds (auto-creates).
- Empty or all-NaN bronze input raises `ValueError` with descriptive message.
- Lag and rolling warm-up NaN counts are logged for each resolution.

---

### Step 4. Add reproducible silver to gold step (`scripts/002_silver_to_gold.py`)

Rationale: Gold parquet files exist in `data/003_gold/` (1m: 5.7 MB, 5m: 1.0 MB, 10m: 0.4 MB) but no script generates them. This means gold data cannot be regenerated after any upstream change, breaking reproducibility. The distinction between silver and gold must be explicitly defined.

Tasks:

Gold definition:
- Create `scripts/002_silver_to_gold.py` that reads silver outputs and produces gold outputs.
- Define and document what gold adds beyond silver:
  - Gold = silver with rows dropped where required core modeling columns are null.
  - Schema is identical to silver (44 columns, same dtypes).
  - No additional features or transformations are applied.
  - Document the chosen definition in a docstring and in `docs/002_pipeline/pipeline.md`.
- Define `required_not_null` columns in `SCHEMAS["gold"]` in config: `timestamp`, `day_class`, `workday`, `year`, `quarter`, `month`, `day`, `day_of_week`, `hour`, `season`, `time_of_day`, `avg_load`.
- Lag, rolling, delta, and slope columns are allowed to contain NaN in gold (expected warm-up periods).

Core transformation:
- For each resolution in `config.RESOLUTIONS`:
  - Read `data/002_silver/power_load_{res}.parquet`.
  - Sort by `timestamp` and reset index.
  - Validate silver schema against `SCHEMAS["silver"]["columns"]`.
  - Validate that `day_class` values are within `VALID_DAY_CLASSES`.
  - Drop rows where any column in `SCHEMAS["gold"]["required_not_null"]` is null.
  - Validate gold schema against `SCHEMAS["gold"]["columns"]`.
  - Write to `data/003_gold/power_load_{res}_all_features.parquet`.

Validation and error handling:
- Validate that all `required_not_null` columns exist in the silver DataFrame before calling `dropna()` -- raise `ValueError` listing any missing columns.
- Wrap `pd.read_parquet()` in try-except to catch corrupted or missing parquet files with a descriptive error including the file path.
- Wrap `.to_parquet()` in try-except to catch write failures (disk full, permission denied) with an actionable error message.
- After `dropna()`, validate that the resulting gold DataFrame is not empty. If all rows were dropped, log a critical warning identifying which `required_not_null` columns had the most NaN values.
- Validate that `day_class` values remain within the expected set after filtering (filtering should never introduce new values, but confirm defensively).
- Validate gold output schema matches `SCHEMAS["gold"]["columns"]` exactly -- raise `ValueError` on any mismatch.

Logging:
- Add `import logging` and configure logger: `logger = logging.getLogger(__name__)`.
- Log per-resolution: input row count, output row count, rows dropped count and percentage, timestamp bounds (min/max).
- Log a per-column breakdown of which `required_not_null` columns had the most NaN values -- this identifies upstream data quality issues (e.g., if `avg_load` NaN is the dominant cause of row drops).
- Log remaining null counts across all gold columns (lag/rolling NaN is expected but should be visible).
- If gold output is empty after filtering, log the null counts that caused all rows to be dropped.
- Log output file size for each resolution.

Edge cases:
- Handle the case where silver input has zero rows -- raise `ValueError` rather than writing an empty gold file.
- Handle resolutions where the warm-up period consumes a large fraction of data (e.g., `lag_1440` at `1min` resolution drops ~1440 rows from avg_load, but these rows are not dropped at gold since only `required_not_null` columns trigger drops).
- Handle the case where silver schema has changed since last run (e.g., new columns added) -- schema validation should catch this and produce a clear diff.

Code quality:
- Extract schema validation logic into a shared utility function in `scripts/utils.py` to eliminate duplication between silver and gold scripts.
- Add module-level docstring explaining: input format, output format, gold definition, required_not_null semantics, and usage.
- Add type hints to all function signatures.
- Import all paths and config from `scripts/config.py` -- no hardcoded paths or constants.
- Add output directory auto-creation: `Path(output_path).parent.mkdir(parents=True, exist_ok=True)`.

Determinism:
- Gold outputs must be deterministic: given the same silver inputs, re-running produces byte-for-byte identical parquet files.
- Guarantee determinism via: deterministic sort by `timestamp`, no random operations, consistent column ordering from schema enforcement.

Acceptance criteria:
- Running `python scripts/002_silver_to_gold.py` regenerates gold files for all configured default resolutions from silver inputs.
- Gold output schema is identical to silver (44 columns, same dtypes).
- Rows with null in any `required_not_null` column are dropped.
- Lag, rolling, delta, and slope NaN values are preserved (not used as drop criteria).
- Log output includes: row counts (input and output), drop count and percentage, per-column NaN breakdown, and timestamp bounds per resolution.
- Gold files are byte-for-byte reproducible given the same silver inputs.
- The script imports all config from `config.py` -- no hardcoded paths or constants.
- Missing or corrupted silver input produces a descriptive `ValueError`.
- Empty gold output (all rows dropped) produces a critical log warning with per-column NaN analysis.
- Schema validation detects any silver schema drift and reports the diff.

---

### Step 5. Add pipeline orchestration (`run_pipeline.py`)

Rationale: Currently, running the full pipeline requires manually executing three scripts in order and checking for errors at each stage. An orchestrator makes this a single command, enables CI/CD, and simplifies onboarding.

Tasks:
- Create `run_pipeline.py` at the project root with:
  - Import and call: `raw_to_bronze()`, `bronze_to_silver()`, `silver_to_gold()` in sequence.
  - Wrap each stage in try/except. On failure: log the stage name, error message, and elapsed time, then exit with non-zero code.
  - Log a summary after each stage: stage name, elapsed time, output file sizes.
  - Log a final summary: total elapsed time, all output files and their sizes.
- Configure logging at the orchestrator level: format with timestamp, level, stage name. Write to both console and a log file (`logs/pipeline.log`).
- Add optional CLI arguments via `argparse`:
  - `--stage`: run a single stage (`bronze`, `silver`, `gold`) instead of all.
  - `--resolution`: limit to a single resolution (useful for development).
  - `--verbose`: set log level to DEBUG.
- Create `logs/` directory automatically if it does not exist.
- Add a `--dry-run` option that validates inputs exist and config is correct without running transformations.

Acceptance criteria:
- `python run_pipeline.py` rebuilds all layers from raw input to gold in one command.
- If any stage fails, the pipeline stops with a clear error naming the failed stage and the exception.
- Log output shows: stage progression, timing per stage, and output file summary.
- `python run_pipeline.py --stage silver` runs only the silver stage.
- Exit code is 0 on success, non-zero on failure.

---

### Step 6. Add ingestion tests (`tests/`)

Rationale: There are zero tests in the project. Any change to feature engineering, resampling logic, or data cleaning could silently break outputs. Tests catch regressions immediately and document expected behavior.

Tasks:

Test infrastructure:
- Create `tests/` directory with `tests/__init__.py` (empty) and `tests/conftest.py`.
- In `conftest.py`, create pytest fixtures that generate small deterministic synthetic datasets:
  - `synthetic_raw`: A small `.mat`-like dict with `P_data` (86400x2 or smaller for speed), `day_data`, `day_class`.
  - `synthetic_bronze`: A DataFrame matching bronze schema (timestamp, load, day_class) with known values including some NaN.
  - `synthetic_silver_1m`: A DataFrame matching silver schema with known feature values.
- Add `pytest` and `pytest-cov` to `pyproject.toml` optional `dev` dependencies.

`tests/stages/test_raw_to_bronze.py`:
- Test that output has exactly 3 columns: `timestamp`, `day_class`, `load`.
- Test that row count equals `86400 * number_of_days` for valid input.
- Test that `timestamp` is monotonically increasing.
- Test that `day_class` contains only `{"full", "half", "none"}`.
- Test that `load` column preserves NaN from raw input (not silently filled or dropped).
- Test that `ValueError` is raised when `P_data` key is missing from `.mat` file.
- Test that `ValueError` is raised when `P_data` has wrong number of rows (not 86400).

`tests/stages/test_bronze_to_silver.py`:
- Test that output columns match expected silver schema from config.
- Test that resampling a 2-day bronze input at 1-minute resolution produces `2 * 1440 = 2880` rows.
- Test that `avg_load` equals the mean of underlying 1-second values for a known minute.
- Test that NaN minutes (where all 60 seconds are NaN) produce NaN `avg_load`.
- Test that lag features are correctly shifted (e.g., `lag_1m` at row t equals `avg_load` at row t-1).
- Test that rolling features have correct NaN warm-up period (first `window-1` rows should be NaN).
- Test that multi-resolution output produces correct row counts: 1m, 5m, 10m.

`tests/unit/test_feature_engineering.py`:
- Test `month_to_season`: December=1 (Winter), March=2 (Spring), June=3 (Summer), September=4 (Fall).
- Test `month_to_season` boundary: November=4 (Fall), February=1 (Winter).
- Test `hour_to_time_of_day`: hour 6=0 (morning), 12=1 (afternoon), 17=2 (evening), 22=3 (night).
- Test `hour_to_time_of_day` boundaries: hour 5=3 (night), hour 11=0 (morning), hour 16=1 (afternoon), hour 21=2 (evening).
- Test `rolling_slope` returns positive slope for linearly increasing data.
- Test `rolling_slope` returns 0.0 for constant data.
- Test `rolling_slope` returns NaN when input contains NaN.
- Test workday mapping: verify correct mapping for each `day_class` value.
- Test delta features: `delta_5m = lag_5m - lag_1m` on known data.

`tests/unit/test_config.py`:
- Test that all paths in config resolve to valid directory structures (parent dirs exist or can be created).
- Test that `RESOLUTIONS` contains only valid pandas resample strings.
- Test that `SCHEMAS` define non-empty column lists for each layer.
- Test that `FEATURE_SETS` do not include the target column (`avg_load`).
- Test that resolution aliases resolve to supported resolutions.
- Test that `SPLIT_DAY_RANGES` covers all 31 days without gaps or overlaps.
- Test that `FEATURE_SETS` column names are a subset of `SCHEMAS["gold"]["columns"]`.
- Test that `FEATURE_CONFIG` lag, rolling, and slope values are positive integers.
- Test that `DAY_CLASS_MAP` contains exactly `{"full", "half", "none"}` as keys.

`tests/stages/test_silver_to_gold.py`:
- Test that gold output has the same 44-column schema as silver.
- Test that gold has fewer rows than silver (NaN rows in required core columns are dropped).
- Test that NaN values in lag/rolling/delta/slope columns are preserved (not used as drop criteria).
- Test that `ValueError` is raised when silver input file is missing.
- Test that `ValueError` is raised for an invalid resolution.
- Test that `ValueError` is raised when silver schema does not match expected columns.
- Test that `ValueError` is raised when silver contains unexpected `day_class` values.
- Test that all `required_not_null` columns have zero NaN in gold output.
- Test that gold output is empty (with warning) when all silver rows have NaN in a required column.
- Test that gold output is deterministic: running twice on the same silver input produces identical parquet.
- Test multiple resolutions produce independent gold files.

`tests/stages/test_model_datasets.py`:
- Test that all three splits (train, validate, test) are produced for a given resolution and feature set.
- Test that `avg_load` (target) is present in every split file.
- Test that splits are strictly chronological: max(train timestamp) < min(validate timestamp) < min(test timestamp).
- Test that no dates overlap between train, validate, and test.
- Test that `ValueError` is raised when gold input file is missing.
- Test that `ValueError` is raised for an unknown feature set name.
- Test that `ValueError` is raised when a feature set column is missing from gold.
- Test that `ValueError` is raised when a feature set includes the target column.
- Test all four feature sets (`minimal`, `temporal`, `full`, `curated`) produce outputs with correct column counts.
- Test that multiple resolutions produce independent model dataset files.
- Test that split day ranges match `SPLIT_DAY_RANGES` in config (train=days 1-25, validate=26-28, test=29-31).

`tests/orchestration/test_run_pipeline.py` (new):
- Test that `--dry-run` validates inputs and config without running transformations.
- Test that `--stage bronze` runs only the bronze stage.
- Test that `--stage silver` runs only the silver stage.
- Test that `--stage gold` runs only the gold stage.
- Test that `--resolution 5min` limits processing to a single resolution.
- Test that an invalid stage name produces a clear error.
- Test that an invalid resolution produces a clear error.
- Test that the orchestrator exits with code 0 on success and non-zero on failure.
- Test that resolution alias `60s` resolves to `1min` during orchestration.
- Test that logging is configured to write to both console and `logs/pipeline.log`.

`tests/integration/test_integration.py` (new):
- Create a synthetic end-to-end dataset (3+ days of second-level data) and run all stages: raw to bronze to silver to gold to model datasets.
- Verify row counts at each layer match expectations.
- Verify schema at each layer matches `SCHEMAS` in config.
- Verify determinism: running the full pipeline twice on the same input produces identical outputs at every layer.
- Verify no data loss through the pipeline: all expected rows from bronze appear in silver (modulo NaN filtering), and gold contains a subset of silver rows.
- Verify that model dataset splits are chronologically ordered and non-overlapping.

`tests/notebooks/test_validate_notebooks.py` (new):
- Test that `validate_notebooks.py` successfully executes a trivial notebook.
- Test that `validate_notebooks.py` reports failure for a notebook with a runtime error.
- Test that `--notebook` argument limits execution to a single named notebook.

Edge case fixtures (add to `conftest.py`):
- Add fixture: `all_nan_bronze_df` -- bronze DataFrame where all `load` values are NaN.
- Add fixture: `single_row_bronze_df` -- bronze DataFrame with exactly one row.
- Add fixture: `single_day_bronze_df` -- bronze DataFrame with exactly one day (86,400 rows).
- Add fixture: `empty_bronze_df` -- bronze DataFrame with correct schema but zero rows.

`tests/unit/test_feature_engineering.py` (expand):
- Test `rolling_slope_series()` with a known linearly increasing Series -- verify positive slope values.
- Test `rolling_slope_series()` with a constant Series -- verify slope values are zero.
- Test `rolling_slope_series()` with NaN values in the Series -- verify NaN propagation.
- Test `rolling_slope_series()` with window size larger than Series length -- verify all-NaN output.
- Test `rolling_slope_series()` with window=2 (minimum valid window) -- verify correct slope.
- Test `month_to_season()` with invalid input (month=0, month=13) -- verify behavior is defined.
- Test `hour_to_time_of_day()` with invalid input (hour=-1, hour=24) -- verify behavior is defined.

Acceptance criteria:
- `pytest tests/` passes with zero failures.
- Tests cover: schema validation, row count invariants, feature correctness, error handling, edge cases.
- Tests use synthetic data (no dependency on actual `.mat` file for unit tests).
- Test runtime is under 60 seconds.
- Coverage report shows at least 80% of `scripts/000_raw_to_bronze.py`, `scripts/001_bronze_to_silver.py`, `scripts/002_silver_to_gold.py`, `scripts/003_create_model_datasets.py`, and helper functions.
- Every pipeline stage has at least 5 dedicated test functions.
- The orchestrator (`run_pipeline.py`) has dedicated tests for all CLI arguments and error paths.
- At least one integration test exercises the full pipeline end-to-end on synthetic data.
- Edge case fixtures exist for all-NaN, single-row, single-day, and empty input scenarios.

---

### Step 7. Tighten repository and documentation baseline

Rationale: Documentation does not match code. `docs/002_pipeline/pipeline.md` has empty Silver and Gold sections. `docs/002_pipeline/plan.md` has 6+ factual errors. `README.md` is 7 lines with no setup or run instructions. The dependency manifest lacks version pinning and key dependencies. `.gitignore` has a minor typo.

Tasks:

`.gitignore` fixes:
- Fix `*.py[codz]` (line 3) to `*.py[cod]` -- the `z` is non-standard and matches nothing useful.
- Add `data/` exclusion rule if data should not be committed (it is currently tracked). Or explicitly document that data is intentionally tracked.
- Add `logs/` exclusion for pipeline log files.
- Ensure `personal/` is excluded (currently line 7 -- confirmed).

`docs/002_pipeline/pipeline.md` updates:
- Fix line 19: says "two attributes" but lists three (`P_data`, `day_data`, `day_class`). Change to "three attributes".
- Add Bronze layer section: describe schema (`timestamp`, `day_class`, `load`), row count (2,678,400), storage format (parquet), and generation command.
- Add Silver layer section: describe resolutions (1m/5m/10m), schema (44 columns for 1m), feature groups (temporal, business, load history), NaN handling strategy, and generation command.
- Add Gold layer section: describe what gold adds beyond silver, schema, validation rules, and generation command.
- Add a schema table for each layer listing column name, dtype, and valid range/values.

`docs/002_pipeline/plan.md` corrections:
- Line 21: output file says `bronze/power_load.parquet` -- actual is `bronze/power_load_1s.parquet`. Fix.
- Line 24: schema shows `(days*86400, 2)` -- actual bronze has 3 columns (`timestamp`, `day_class`, `load`). Fix to `(days*86400, 3)`.
- Line 28: date `2025-11-31` does not exist (November has 30 days). Fix to `2025-11-28` (the actual start date).
- Line 109: season mapping says `Spring (mar-apr)` -- should be `Spring (mar-may)`. Fix.
- Line 121: `6 = Sunday` is duplicated from line 118 (`0 = Sunday`). Should be `6 = Saturday`. Fix.
- Line 134: defines `is_workday` as binary, but code uses ternary `workday`. Align with whatever the team decides in Step 3.
- Lines 139-158: lag names use `power_Xm` / `power_Xd` but code uses `lag_Xm`. Align naming.
- Line 212: `"I'm not really sure what validate is for?"` -- replace with actual validation set purpose: "Used for hyperparameter tuning and model selection without touching the test set."
- Lines 220-221: duplicate "Step 7" numbering. Renumber to Step 8.

`README.md` expansion:
- Add project overview (1-2 sentences): what this project does, for whom.
- Add team and advisor info: include Prof. Raymond de Callafon.
- Add setup instructions: Python version, `pip install -e ".[dev]"`, expected data placement.
- Add pipeline run instructions: `python run_pipeline.py` with expected output.
- Add project structure tree showing key directories and files.
- Add test instructions: `pytest tests/`.

Dependency manifest updates (`pyproject.toml`):
- Pin all dependencies to at least minor version:
  - `scipy>=1.11`
  - `numpy>=1.24`
  - `pandas>=2.0`
  - `matplotlib>=3.7`
  - `seaborn>=0.12`
  - `plotly>=5.15`
- Add parquet runtime dependencies with platform-safe fallback:
  - `pyarrow>=14.0` where wheels are available
  - `fastparquet>=2025.12` as fallback parquet backend on platforms lacking `pyarrow` wheels
- Add missing test dependencies: `pytest>=7.0`, `pytest-cov>=4.0`.
- Add `jupyter>=1.0` for notebook execution.

Acceptance criteria:
- `docs/002_pipeline/pipeline.md` documents all four layers (raw, bronze, silver, gold) with schemas.
- `docs/002_pipeline/plan.md` has zero factual errors and all column names match actual code output.
- `README.md` contains setup instructions, pipeline run command, and project structure.
- `pyproject.toml` includes all dependencies needed to run scripts, notebooks, and tests from a clean environment.
- `.gitignore` has no non-standard patterns.
- A new teammate can clone the repo, install requirements, and run the pipeline using only `README.md` instructions.

---

## Phase 2: Data Trust and Model-Ready Preparation

Goal:
Use the hardened pipeline outputs to create trusted EDA artifacts and reproducible train/validate/test datasets.

Exit criteria:
- EDA clearly explains data quality, missingness, and feature behavior.
- Feature sets are explicit and versioned.
- Model-ready datasets are generated by script with chronological split.

---

### Step 1. Expand EDA with quality-first focus

Rationale: The current notebooks are shallow -- they show basic shapes and a few plots but lack the diagnostics needed to trust the data for modeling. No notebook has summary statistics, distribution analysis, or systematic NaN investigation. Without these, any modeling results built on this data are indefensible.

Tasks:

`notebooks/000_raw_eda.ipynb` (currently 5 cells -- shape/type inspection only):
- Add cell: import from `config.py` instead of hardcoded path (currently `RAW_PATH = '../data/000_raw/P_data.mat'`).
- Add cell: summary statistics for `P_data` -- min, max, mean, median, std per day (31 columns).
- Add cell: NaN analysis -- count NaN per day, visualize as heatmap (day x hour showing NaN density).
- Add cell: distribution of load values -- histogram of all non-NaN values, check for outliers (values > 3 sigma from mean).
- Add cell: per-day load profiles overlaid on single plot (all 31 days) to visually identify anomalous days.
- Add cell: `day_class` distribution table -- counts of full/half/none with corresponding dates.
- Add markdown narrative cells between each analysis cell explaining what the output means and any concerns.
- Remove any unused imports or dead code.

`notebooks/001_bronze_eda.ipynb` (currently 5 cells -- head, shape, NaN count, one time-series plot):
- Add cell: import from `config.py` instead of hardcoded path (currently `bronze_path = "../data/001_bronze/power_load_1s.parquet"`).
- Add cell: day-class breakdown -- table showing date, day_class, row count, NaN count, NaN percentage per day.
- Add cell: NaN pattern analysis -- are NaN values clustered in time (consecutive seconds) or scattered? Plot NaN locations on timeline.
- Add cell: identify and flag outlier days -- days where load pattern deviates significantly from others in same class.
- Add cell: all-days overlay plot colored by `day_class` (full=green, half=blue, none=orange) to see business-day effect on load shape.
- Add cell: basic statistics -- min, max, mean, median, std of load overall and per day_class.
- Add cell: check for zero-load periods -- are there sustained periods of 0W? Are they valid or sensor failures?
- Ensure the existing time-series plot (cell 5) shows all 31 days, not just first 6.
- Add markdown narrative cells interpreting each analysis.

`notebooks/002_silver_eda.ipynb` (currently 8 cells -- NaN count, time-series plot, 3 workday bar charts, daily bar chart):
- Add cell: import from `config.py` instead of assuming paths.
- Add cell: feature correlation heatmap -- show pairwise correlation of all 44 columns. Identify highly correlated pairs (|r| > 0.95) that may cause multicollinearity.
- Add cell: NaN cascade analysis -- for each feature column, count NaN rows and explain why (lag warm-up, rolling warm-up, slope warm-up). Create table: column name, NaN count, NaN percentage, reason.
- Add cell: feature distributions -- histogram grid for all non-lag numeric features (avg_load, temporal features, workday).
- Add cell: load autocorrelation plot -- ACF/PACF of `avg_load` at 1-minute resolution to show temporal dependency structure.
- Add cell: 5-minute and 10-minute resolution analysis -- load the 5m and 10m silver files (once Step 3 of Phase 1 produces them) and show comparative statistics.
- Fix duplicate `hour` computation: cell 8 re-computes `.assign(hour=lambda x: x["timestamp"].dt.hour)` even though `hour` already exists as a column from the silver pipeline.
- Fix misleading `resolution = 1` variable in cell 5 -- it implies 1-second but the data is 1-minute. Rename or remove.
- Add markdown narrative cells summarizing key findings: Which features are most informative? What data quality issues remain? What should the team watch for during modeling?
- Add cell: feature engineering validation -- spot-check that `lag_1` at row t equals `avg_load` at row t-1, that `delta_5 = lag_5 - lag_1`, and that rolling windows match manual calculation for a known row. This confirms pipeline correctness.
- Add cell: cross-resolution consistency check -- verify that the 5-minute `avg_load` equals the mean of the corresponding five 1-minute `avg_load` values for at least 10 sample intervals.
- Add cell: Variance Inflation Factor (VIF) analysis -- compute VIF for all features in the `curated` feature set to quantify multicollinearity beyond the correlation heatmap.
- Add cell: univariate feature-target correlation -- rank features by absolute Pearson correlation with `avg_load` to preview feature importance before modeling.
- Add cell: stratified target analysis -- break down `avg_load` distribution by `day_class`, `hour`, and `season` using box plots or violin plots.
- Document the impact of NaN interpolation used for autocorrelation analysis -- any cell that calls `.interpolate()` must have an adjacent markdown cell explaining why interpolation is acceptable and what bias it may introduce.

Cross-notebook standards:
- Extract all hardcoded threshold values (z-score thresholds, figure sizes, histogram bin counts, percentile lists, max lag values) to a shared `EDA_CONFIG` section in `scripts/config.py` or to clearly labeled constants declared at the top of each notebook.
- Add a data schema validation cell at the start of each notebook: confirm expected columns exist, dtypes are correct, and row counts match expectations from the pipeline documentation.
- Record reproducibility metadata at the top of each notebook: log notebook execution start time, Python version, pandas version, numpy version, and key data shape summaries.
- Ensure all data transformations within notebooks (e.g., interpolation, fillna, resampling) are explicitly documented with rationale in adjacent markdown cells.
- Standardize figure sizes and color schemes across all three notebooks for consistent presentation.

Additional raw EDA (`000_raw_eda.ipynb`):
- Add cell: formal stationarity test -- Augmented Dickey-Fuller (ADF) test on daily mean load series to characterize trend behavior.
- Add cell: quantile analysis -- report p1, p5, p25, p50, p75, p95, p99 percentiles for raw load values.
- Add cell: per-day coefficient of variation to identify days with unusually stable or volatile load.
- Add cell: stratified load analysis by `day_class` -- compare distribution statistics (mean, std, range) across full/half/none day types.

Additional bronze EDA (`001_bronze_eda.ipynb`):
- Add cell: duplicate timestamp check -- validate that no second-level timestamps are repeated.
- Add cell: load value range validation -- check for negative values or values exceeding physical limits (e.g., >50,000 watts for this facility).
- Add cell: load ramp rate analysis -- compute max `dLoad/dt` per day to identify sudden transitions that may challenge model predictions.
- Add cell: NaN correlation with `day_class` -- contingency table or chi-squared test to determine if NaN rates differ significantly by business-day type.

Acceptance criteria:
- Each notebook runs end-to-end without errors on the latest pipeline output.
- Each notebook has at least 3 markdown narrative cells explaining findings and concerns.
- `000_raw_eda.ipynb` includes: summary stats, NaN analysis, distribution, day-class breakdown, stationarity test, quantile analysis.
- `001_bronze_eda.ipynb` includes: day-class analysis, NaN patterns, outlier detection, all-days overlay, duplicate timestamp check, load ramp rate analysis.
- `002_silver_eda.ipynb` includes: correlation heatmap, NaN cascade table, autocorrelation, multi-resolution comparison, feature engineering validation, VIF analysis, feature-target correlation ranking.
- No hardcoded paths remain in any notebook -- all use `config.py` or relative paths anchored to a config.
- No duplicate computations (e.g., recomputing `hour` when it already exists).
- No stale variables or misleading labels.
- Hardcoded thresholds are centralized or declared as named constants at the top of each notebook.
- All data transformations within notebooks are documented with rationale.
- Cross-resolution consistency is verified programmatically.

---

### Step 2. Formalize feature set definitions

Rationale: The silver/gold pipeline produces 44 columns, but there is no documented definition of which features should be used for modeling. `docs/002_pipeline/plan.md` lines 167-191 sketch feature sets A/B/C but uses column names that do not match the actual pipeline output. Without explicit feature set definitions, teammates will use ad hoc column selections, making experiment comparison impossible.

Tasks:
- Create `docs/003_modeling/feature_sets.md` defining at least three named feature sets:
  - Minimal: `avg_load` + `workday` + `hour` + `lag_1m` (smallest set to establish baseline).
  - Temporal: all temporal features (`year`, `quarter`, `month`, `day`, `weekday`, `hour`, `season`, `time_of_day`) + `workday` + `avg_load`.
  - Full: all 44 columns minus `timestamp` and `day_class` (the kitchen sink for comparison).
  - Curated: a reduced set removing highly correlated features identified in EDA Step 1.
- For each feature set:
  - List exact column names matching the silver/gold output.
  - State the rationale: why include these features, what hypothesis does it test.
  - Note expected strengths and risks (e.g., full set has multicollinearity risk, minimal set may underfit).
- Cross-reference with `docs/002_pipeline/plan.md` Step 5 (lines 167-191) and update plan to use consistent naming.
- Include a table mapping plan names (A, B, C) to the named sets (minimal, temporal, full, curated).

Acceptance criteria:
- `docs/003_modeling/feature_sets.md` exists with at least 3 defined feature sets.
- Every column name in every feature set exactly matches a column in the gold output.
- Running `set(feature_set_columns) - set(gold_df.columns)` returns empty for every defined set.
- Each feature set has documented rationale.
- `docs/002_pipeline/plan.md` feature set references are consistent with `feature_sets.md`.

---

### Step 3. Build model dataset generation script

Rationale: No script currently creates train/validate/test splits. Without this, data leakage is likely if teammates manually slice data. A chronological split is required for time-series data -- random splitting would leak future information into training.

Tasks:
- Create `scripts/003_create_model_datasets.py`.
- Implement chronological split based on day-of-month:
  - Train: days 1-25 (Nov 28 through Dec 22 = 25 days).
  - Validate: days 26-28 (Dec 23 through Dec 25 = 3 days). Note: Dec 25 is Christmas -- document this as a known concern for validation representativeness.
  - Test: days 29-31 (Dec 26 through Dec 28 = 3 days).
- For each resolution in `config.RESOLUTIONS`:
  - For each feature set defined in `docs/003_modeling/feature_sets.md`:
    - Read gold parquet, select feature columns, apply chronological split.
    - Write three files: `data/004_model/{resolution}_{feature_set}_train.parquet`, `_validate.parquet`, `_test.parquet`.
- Log for each split: row count, date range, null rate, label (`avg_load`) statistics (mean, std, min, max).
- Validate: no date overlap between train/validate/test.
- Validate: train dates < validate dates < test dates (strict chronological order).
- Import all config and feature set definitions from shared sources.
- Add structured logging (no `print()`).
- Add output directory auto-creation.

Acceptance criteria:
- Running `python scripts/003_create_model_datasets.py` produces train/validate/test parquet files for all resolutions and feature sets.
- No date overlap exists between any split.
- All splits are strictly chronologically ordered.
- Log output includes row counts, date ranges, and label statistics per split.
- Christmas (Dec 25) in the validation set is documented as a known consideration.
- Files are deterministic -- running twice on the same gold input produces identical outputs.

---

### Step 4. Define focused hypothesis and MVMP scope

Rationale: Without a clear hypothesis, the team will build models without a measurable target. The MVMP (Minimum Viable Modeling Product) constrains the first modeling pass to the simplest version that answers a real question.

Tasks:
- Create `docs/003_modeling/hypothesis.md` with the Report IV hypothesis set:
  - H1: workday signal effect (MAE target >=10%) at `1min`.
  - H2: lag/rolling transition value (RMSE target >=8%) at `1min`.
  - H3: resolution tradeoff (`1min` vs `5min`) documented as deferred for the MVP.
  - H4: nonlinear behavior vs regularized linear baseline as exploratory analysis.
- Create `docs/003_modeling/mvmp.md` defining the first modeling scope:
  - Resolution anchor: `1min`.
  - Feature-set anchor: `minimal` for control, with full fixed grid across all feature sets.
  - Models: Ridge + HistGradientBoostingRegressor fixed configs.
  - Metric protocol: use validation split for hypothesis evaluation; reserve test for one-shot final holdout only.
  - Success criteria: reproducibility, protocol integrity, artifact completeness, and baseline transparency.
- Map each hypothesis to which resolution, feature set, and model it requires.
- Map research questions from Report I Section 4 to the hypotheses.

Acceptance criteria:
- `docs/003_modeling/hypothesis.md` defines H1-H3 with H4 exploratory support in the Slide 14 format.
- Each hypothesis names a specific metric and a measurable target.
- `docs/003_modeling/mvmp.md` specifies the `1min` MVP anchor, fixed model family, and validate/test protocol.
- The team can read MVMP and know exactly what is executed now vs deferred.
- Hypotheses are traceable to Report I research questions.

---

## Out of scope until Phase 1 and Phase 2 are complete

- Neural network architecture exploration.
- Advanced ensembling or hyperparameter sweeps.
- Rolling/recursive forecasting framework.
- Robustness stress testing beyond basic missingness diagnostics.

Reason:
These are high-effort items with low value if ingestion, data trust, and split discipline are not already stable.

---

## Recommended execution order

1. Complete Phase 1 Steps 1-3 (config + script hardening).
2. Add Step 4 and Step 5 (gold reproducibility + orchestrator).
3. Add Step 6 and Step 7 (tests + docs/requirements cleanup).
4. Run full pipeline and tests to lock Phase 1.
5. Start Phase 2 notebook updates and feature/split documentation.
6. Finish with model dataset generation script and MVMP definition.

## Definition of done

This two-phase specification is complete when:
- Pipeline is reproducible end to end from raw input.
- Ingestion behavior is tested and documented.
- EDA and feature sets are explicit and trustworthy.
- Model-ready datasets are generated by script with no manual intervention.

## Document map

- Notebook configurability spec: [001_spec.md](001_spec.md)
- Pipeline: [pipeline.md](../002_pipeline/pipeline.md)
- Plan: [plan.md](../002_pipeline/plan.md)
- Feature sets: [feature_sets.md](../003_modeling/feature_sets.md)
- Hypotheses: [hypothesis.md](../003_modeling/hypothesis.md)
- MVMP: [mvmp.md](../003_modeling/mvmp.md)
- Glossary: [glossary.md](../004_reference/glossary.md)
- Spec changelog: [docs/change logs/000spec/changelog.md](../change%20logs/000spec/changelog.md)
- Changelog index: [changelog.md](../../changelog.md)

